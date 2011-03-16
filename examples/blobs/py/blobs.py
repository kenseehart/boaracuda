from ctypes import *
from pyglet.gl import *
import pyglet
import os, sys
from math import pi, sin, cos, sqrt
from random import random, randint

here = os.path.split(__file__)[0] # directory of this module
sys.path.append(os.path.join(here, '..', 'bin')) # access compute kernels

def load_image(f):
    return pyglet.image.load(os.path.join(here, '..','images',f))

def repr_ct(obj):
    """print a ctypes structure"""
    if isinstance(obj, Structure):
        slist = []
        for f,_ in obj._fields_:
            slist.append('%s=%s'%(f, repr_ct(getattr(obj, f))))
        return type(obj).__name__+'('+', '.join(slist)+')'
    elif isinstance(obj, Array):
        if len(obj)<5:
            slist = [repr_ct(x) for x in obj[:]]
        else:
            slist = [repr_ct(x) for x in obj[:2]] + ['...'] + [repr_ct(x) for x in obj[-1:]]
        return '['+', '.join(slist)+']'
    else:
        return repr(obj)

                                                
class V3(Structure):
    "3D Vector"
    _fields_ = [
        ('x', c_float),
        ('y', c_float),
        ('z', c_float),
        ]
    
    def __init__(self, x,y,z):
        self.x = x
        self.y = y
        self.z = z
        
    def __sub__(self, other):
        return V3(self.x-other.x, self.y-other.y, self.z-other.z)
    
    def __add__(self, other):
        return V3(self.x+other.x, self.y+other.y, self.z+other.z)
    
    def __repr__(self):
        return '(%0.2f, %0.2f, %0.2f)' % (self.x, self.y, self.z)
    
    def magnitude(self):
        return sqrt(self.x*self.x + self.y*self.y + self.z*self.z)
    
    def __rmul__(self, k):
        return V3(k*self.x, k*self.y, k*self.z)

def dot(u,v):
    return u.x*v.x+u.y+v.y+u.z+v.z

def cross(u,v):
    return V3(u.y*v.z-u.z*v.y, u.z*v.x-u.x*v.z, u.x*v.y-u.y-v.z)


class State(Structure):
    "Physical state of a node"
    _fields_ = [
        ('p', V3), # position
        ('v', V3), # velocity
        ]

class Attraction(Structure):
    "Unilateral attractive relationship: V*(d-r)**e, where V=normalize difference vector, d=distance"
    _fields_ = [
    ('r', c_float), # stable distance (distance at which force is 0.0)
    ('g', c_float), # gravitational magnitude
    ('e', c_float), # gravitational exponent (e.g. for gravity, use 2.0)
    ('h', c_float), # harmonizer: causes velocity matching with target
    ]
    
    def set(self, r, g, e,h):
        self.r = r
        self.g = g
        self.e = e
        self.h = h
        

class Connection(Structure):
    "Relationship to Node i, specified by Attraction a"
    _fields_ = [
        ('i', c_int), # index to target node
        ('a', c_int), # index to Attraction (level of indirection makes parameters globally tunable)
    ]
    
class Node(Structure):
    """A point mass entity with physical state and relationships to other nodes"""
    _fields_ = [
        ('i', c_int32), # index to this node
        ('state', State), # double buffered physics state
        ('df', c_float), # damping factor
        ('connections', Connection*8), # set of node attraction relationships
        ('nc', c_int32), # number of connections
        ]
    
    def randomize(self, m):
        if self.i>0:
            self.state.p.x = random()-0.5
            self.state.p.y = random()-0.5
            self.state.p.z = random()-0.5
            self.state.v.x = random()*m-m*0.5
            self.state.v.y = random()*m-m*0.5
            self.state.v.z = random()*m-m*0.5

    def connect_to(self, target, a):
        self.connections[self.nc].i = target.i
        self.connections[self.nc].a = a
        self.nc += 1
        
        
    def clear_connections(self):
        self.nc = 0

    def update(self, scene):
        """only called when testing without compute kernel
        buffer_id alternates between 0 and 1"""
        
        s0 = self.state
        
        for i in range(self.nc):
            c = self.connections[i]
            s1 = scene.nodes[c.i].state
            dx = s1.p.x-s0.p.x
            dy = s1.p.y-s0.p.y
            dz = s1.p.z-s0.p.z
            d = sqrt(dx*dx+dy*dy+dz*dz)
            a = scene.attractions[c.a]
            if d>a.r:
                f = a.g*(d-a.r)**a.e
            else:
                f = -a.g*(a.r-d)**a.e
            #assert(f<1000)
            #assert(f>-1000)
            s0.v.x += f*dx
            s0.v.y += f*dy
            s0.v.z += f*dz
            
            s0.v = (1.0-a.h)*s0.v + a.h*s1.v
            
        # magic damping effect increases with speed
        speed = s0.v.magnitude()*10
        s0.v.x *= friction**speed;
        s0.v.y *= friction**speed;
        s0.v.z *= friction**speed;
        
        s0.p.x += s0.v.x
        s0.p.y += s0.v.y
        s0.p.z += s0.v.z
        
#def ptr_add(ptr, i):
#    """performs pointer offset addition: note that offset is multiplied by size, as in C"""
#    return pointer(type(ptr.contents).from_address(addressof(ptr.contents) + i*sizeof(ptr.contents)))

def array_ptr(a, i=0):
    """returns a pointer to an element in an array"""
    T = a._type_
    return pointer(T.from_address(addressof(a) + i*sizeof(T)))


A_CENTER = 0
A_SURFACE = 1
A_OPPOSE = 2
A_ENEMY = 3
A_FRIEND = 4
friction = 0.99

class Blob():
    """An icosohedral blob: blob.nodes[0] = center, blob.nodes[1:] = surface vertices"""
    
    ituples = [(1, 5, 2), (1, 10, 5), (10, 6, 5), (5, 6, 9), (5, 9, 2),
               (9, 11, 2), (9, 4, 11), (6, 4, 9), (6, 3, 4), (3, 8, 4),
               (8, 11, 4), (8, 7, 11), (8, 12, 7), (12, 1, 7), (1, 2, 7),
               (7, 2, 11), (10, 1, 12), (10, 12, 3), (10, 3, 6), (8, 3, 12)]
    
    opposites = [(1,4), (2, 3), (5, 8), (6, 7), (9, 12), (10, 11)]

    def __init__(self, scene, i):
        self.scene = scene
        self.i = i
        self.offset = i*13
        self.nodes = array_ptr(scene.nodes, self.offset)
        self.vertices = array_ptr(scene.vertices, self.offset)
        self.normals = array_ptr(scene.normals, self.offset)
        self.indices = []
        
        for t in self.ituples:
            self.indices.extend(t)
           
        self.nodes[0].i = self.offset
        
        for j in range(1,13):
            self.nodes[j].i = j+self.offset
            self.nodes[j].connect_to(self.nodes[0], A_CENTER)
            
        for (a,b,c) in self.ituples:
            self.nodes[a].connect_to(self.nodes[b], A_SURFACE)
            self.nodes[b].connect_to(self.nodes[c], A_SURFACE)
            self.nodes[c].connect_to(self.nodes[a], A_SURFACE)
        
        for (a,b) in self.opposites:
            self.nodes[a].connect_to(self.nodes[b], A_OPPOSE)
            self.nodes[b].connect_to(self.nodes[a], A_OPPOSE)
            
    def connect_to(self, target, a):
        self.nodes[0].connect_to(target.nodes[0], a)
        
        
    def update(self, scene):
        """only called when testing without compute kernel
        buffer_id alternates between 0 and 1"""
        b = scene.buffer_id
        
        for i in range(13):
            self.nodes[i].update(scene)
            self.vertices[i] = self.nodes[i].state.p
            
        for i in range(20):
            m = self.nodes[0].state.p
            a = self.nodes[self.indices[3*i]].state.p
            b = self.nodes[self.indices[3*i+1]].state.p
            c = self.nodes[self.indices[3*i+2]].state.p
            #self.normals[i] = cross(b-a,c-a)
            self.normals[i] = 3*m-(a+b+c)

    def draw(self):
        glDrawElements(GL_TRIANGLES, 60, GL_UNSIGNED_INT, (c_int*60)(*[self.offset + i for i in self.indices]))


class BlobScene():
    """A whole lot of blobs"""
    def __init__(self, nblobs):
        #self.noise = 0.2
        #self.noise_decay = 0.99
        self.nblobs = nblobs
        self.vertices = (V3*(nblobs*13))()
        self.normals = (V3*(nblobs*20))()
        self.nodes = (Node*(nblobs*13))()
        self.blobs = [Blob(self, i) for i in range(nblobs)]
        self.buffer_id = 0
        self.attractions = (Attraction*5)()
        
        self.attractions[A_OPPOSE].set(0.6, 0.2, 2.0, 0.0)
        self.attractions[A_CENTER].set(0.3, 0.05, 2.0, 0.04)
        self.attractions[A_SURFACE].set(0.3, 0.04, 2.0, 0.0)
        self.attractions[A_ENEMY].set(3.0, 0.0004, 2.0, 0.0)
        self.attractions[A_FRIEND].set(0.5, 0.0008, 2.0, 0.0)
                
        for node in self.nodes:
            node.randomize(0.05)
            
        for i,blob in enumerate(self.blobs[1:]):
            blob.connect_to(self.blobs[0], A_FRIEND)
            j = randint(1,nblobs-1)
            a = randint(A_ENEMY, A_FRIEND)
            if 1!=j:
                blob.connect_to(self.blobs[j], a)

        
    def update(self):
        """only called when testing without compute kernel
        buffer_id alternates between 0 and 1"""
        self.buffer_id = 1 - self.buffer_id
        for blob in self.blobs:
            blob.update(self)

    def draw(self):
        self.update()
        glPushClientAttrib(GL_CLIENT_VERTEX_ARRAY_BIT)
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_NORMAL_ARRAY)
        
        glVertexPointer(3, GL_FLOAT, 0, self.vertices)
        glNormalPointer(GL_FLOAT, 0, self.normals)
        
        for blob in self.blobs:
            blob.draw()
            
#        for i in range(nblobs):
#            j = randint(0, nblobs-1)
#            if j!=i:
                    
        glPopClientAttrib()



try:
    # Try and create a window with multisampling (antialiasing)
    config = Config(sample_buffers=1, samples=4, 
                    depth_size=16, double_buffer=True,)
    window = pyglet.window.Window(resizable=True, config=config)
except pyglet.window.NoSuchConfigException:
    # Fall back to no multisampling for old hardware
    window = pyglet.window.Window(resizable=True)

@window.event
def on_resize(width, height):
    # Override the default on_resize handler to create a 3D projection
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(60., width / float(height), .1, 1000.)
    glMatrixMode(GL_MODELVIEW)
    return pyglet.event.EVENT_HANDLED

def update(dt):
    blob_scene.update()

pyglet.clock.schedule(update)

@window.event
def on_draw():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    glTranslatef(0, 0, -4)
    blob_scene.draw()

def setup():
    # One-time GL setup
    glClearColor(0, 0, 0, 1)
    glColor3f(1, 0, 0)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_CULL_FACE)

    # Uncomment this line for a wireframe view
    #glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

    # Simple light setup.  On Windows GL_LIGHT0 is enabled by default,
    # but this is not the case on Linux or Mac, so remember to always 
    # include it.
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_LIGHT1)

    # Define a simple function to create ctypes arrays of floats:
    def vec(*args):
        return (GLfloat * len(args))(*args)

    glLightfv(GL_LIGHT0, GL_POSITION, vec(.5, .5, 1, 0))
    glLightfv(GL_LIGHT0, GL_SPECULAR, vec(.5, .5, 1, 1))
    glLightfv(GL_LIGHT0, GL_DIFFUSE, vec(1, 1, 1, 1))
    glLightfv(GL_LIGHT1, GL_POSITION, vec(1, 0, .5, 0))
    glLightfv(GL_LIGHT1, GL_DIFFUSE, vec(.5, .5, .5, 1))
    glLightfv(GL_LIGHT1, GL_SPECULAR, vec(1, 1, 1, 1))

    glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, vec(0.5, 0, 0.3, 1))
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, vec(1, 1, 1, 1))
    glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 50)


def main():
    global blob_scene, torus
    setup()
    blob_scene = BlobScene(6)
    pyglet.app.run()


if __name__ == '__main__':
    main()
    