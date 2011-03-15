# cuda wrappers using ctypes

import re
from ctypes import *
from ctypes.util import find_library
import os
import sys


if os.name == "nt":
    # CUDA run time DLL (TODO: think of a better way to select the right DLL)
    for i in range(7,30):
        try:
            cu = getattr(windll, "cudart32_32_%d"%i)
            break
        except WindowsError:
            continue
elif os.name == "posix":
    # assumes DYLD_LIBRARY_PATH or LD_LIBRARY_PATH is set up
    lib_name = "cudart"
    lib_path = find_library(lib_name)
    if lib_path is None: raise Exception("Unable to find Cuda RT library")
    cu = CDLL(lib_path)
else:
    raise Exception("Platform not supported")


c_size_t = c_uint32
c_cuda_p = c_void_p
c_cuda_stream = c_uint32

cudaGetErrorString = cu.cudaGetErrorString
cudaGetErrorString.argtypes = [c_int32]
cudaGetErrorString.restype = c_char_p

class CudaError(StandardError):
    pass

    
    
class CUDAAPI:
    """Wrapper for typical CUDA API call
    Assumes that the wrapped API function returns an error code, which raises CudaError if nonzero.
    reti specifies an optional tuple 
    """
    def __init__(self, f, argtypes=[]):
        self.f = f
        self.reti = [] # list of indexes to output pointer parameters.

        for i,argt in enumerate(argtypes):
            try:
                argtypes[i] = argt.modify(self, i)
            except AttributeError:
                pass
        
        self.f.argtypes = argtypes
        self.__name__ = f.__name__
        
    def __call__(self, *args):
        args = list(args)
        ret = []
        for i in self.reti:
            x = self.f.argtypes[i]._type_()
            args.insert(i, pointer(x))
            ret.append(x)
        
        err = self.f(*args)
        if err:
            raise CudaError(cudaGetErrorString(err))

        for i,x in enumerate(ret):
            if hasattr(x, 'value'):
                ret[i] = x.value
        
        if len(ret)==0:
            return None
        elif len(ret)==1:
            return ret[0]
        else:
            return tuple(ret)
        

    def __repr__(self):
        return '%s(%s) => %s'%(self.__name__,
                               ', '.join([a.__name__ for i,a in enumerate(self.f.argtypes) if i not in self.reti]),
                               ', '.join([a._type_.__name__ for i,a in enumerate(self.f.argtypes) if i in self.reti]))

class RETURNS:
    """Transforms a pointer parameter into a return value
    In other words, in the raw API, the argument is a pointer to the specified type,
    but the corresponding argument is omitted in the wrapper, and the value is returned.
    If multiple RETURNS are used, a tuple is returned.
    """
    def __init__(self, t):
        self.t = t
        
    def modify(self, cudaapi, i):
        cudaapi.reti.append(i)
        return POINTER(self.t)
    
    

def parse_struct_to_ctypes(st):
    """Utility to generate ctypes declarations"""
    rx = re.compile('(\w+)\s+(\w+)(?:\[(\d+)\])?;.*')
    typed = {'char':'c_char', 'size_t':'c_size_t', 'int': 'c_int'}
    sout=[]
    for s in st.split('\n'):
        s = s.strip()
        if s:
            m = rx.match(s)
            if m:
                t,name,n = m.groups()
                tp = typed[t]
                if n:
                    tp = tp+'*'+n
                sout.append("        ('%s', %s)," % (name, tp))
            else:
                print "'%s' doesn't scan" % s
    return '\n'.join(sout)
    
    
class cudaDeviceProp(Structure):
    """
  char   name[256];                 ///< ASCII string identifying device
  size_t totalGlobalMem;            ///< Global memory available on device in bytes
  size_t sharedMemPerBlock;         ///< Shared memory available per block in bytes
  int    regsPerBlock;              ///< 32-bit registers available per block
  int    warpSize;                  ///< Warp size in threads
  size_t memPitch;                  ///< Maximum pitch in bytes allowed by memory copies
  int    maxThreadsPerBlock;        ///< Maximum number of threads per block
  int    maxThreadsDim[3];          ///< Maximum size of each dimension of a block
  int    maxGridSize[3];            ///< Maximum size of each dimension of a grid
  int    clockRate;                 ///< Clock frequency in kilohertz
  size_t totalConstMem;             ///< Constant memory available on device in bytes
  int    major;                     ///< Major compute capability
  int    minor;                     ///< Minor compute capability
  size_t textureAlignment;          ///< Alignment requirement for textures
  int    deviceOverlap;             ///< Device can concurrently copy memory and execute a kernel
  int    multiProcessorCount;       ///< Number of multiprocessors on device
  int    kernelExecTimeoutEnabled;  ///< Specified whether there is a run time limit on kernels
  int    integrated;                ///< Device is integrated as opposed to discrete
  int    canMapHostMemory;          ///< Device can map host memory with cudaHostAlloc/cudaHostGetDevicePointer
  int    computeMode;               ///< Compute mode (See ::cudaComputeMode)
  int    maxTexture1D;              ///< Maximum 1D texture size
  int    maxTexture2D[2];           ///< Maximum 2D texture dimensions
  int    maxTexture3D[3];           ///< Maximum 3D texture dimensions
  int    maxTexture2DArray[3];      ///< Maximum 2D texture array dimensions
  size_t surfaceAlignment;          ///< Alignment requirements for surfaces
  int    concurrentKernels;         ///< Device can possibly execute multiple kernels concurrently
  int    ECCEnabled;                ///< Device has ECC support enabled
  int    pciBusID;                  ///< PCI bus ID of the device
  int    pciDeviceID;               ///< PCI device ID of the device
  int    __cudaReserved[22];
    """
    
    _fields_ = [
        ('name', c_char*256),
        ('totalGlobalMem', c_size_t),
        ('sharedMemPerBlock', c_size_t),
        ('regsPerBlock', c_int),
        ('warpSize', c_int),
        ('memPitch', c_size_t),
        ('maxThreadsPerBlock', c_int),
        ('maxThreadsDim', c_int*3),
        ('maxGridSize', c_int*3),
        ('clockRate', c_int),
        ('totalConstMem', c_size_t),
        ('major', c_int),
        ('minor', c_int),
        ('textureAlignment', c_size_t),
        ('deviceOverlap', c_int),
        ('multiProcessorCount', c_int),
        ('kernelExecTimeoutEnabled', c_int),
        ('integrated', c_int),
        ('canMapHostMemory', c_int),
        ('computeMode', c_int),
        ('maxTexture1D', c_int),
        ('maxTexture2D', c_int*2),
        ('maxTexture3D', c_int*3),
        ('maxTexture2DArray', c_int*3),
        ('surfaceAlignment', c_size_t),
        ('concurrentKernels', c_int),
        ('ECCEnabled', c_int),
        ('pciBusID', c_int),
        ('pciDeviceID', c_int),
        ('__cudaReserved', c_int*22),
        ]

    def dump(self):
        return '\n'.join(['%s = %s' % (k, repr(getattr(self, k))) for k,t in self._fields_])

# print parse_struct(cudaDeviceProp.__doc__)

# enum cudaMemcpyKind:
cudaMemcpyHostToHost          =   0
cudaMemcpyHostToDevice        =   1
cudaMemcpyDeviceToHost        =   2
cudaMemcpyDeviceToDevice      =   3

cudaGetDeviceCount = CUDAAPI(cu.cudaGetDeviceCount, [RETURNS(c_int32)])
cudaGetDeviceProperties = CUDAAPI(cu.cudaGetDeviceProperties, [RETURNS(cudaDeviceProp), c_int32])

cudaMalloc = CUDAAPI(cu.cudaMalloc, [RETURNS(c_cuda_p), c_size_t])
cudaMallocHost = CUDAAPI(cu.cudaMallocHost, [RETURNS(c_cuda_p), c_size_t])
cudaFree = CUDAAPI(cu.cudaFree, [c_cuda_p])
cudaFreeHost = CUDAAPI(cu.cudaFreeHost, [c_cuda_p])

cudaMemcpy = CUDAAPI(cu.cudaMemcpy, [c_cuda_p, c_cuda_p, c_size_t, c_int])
cudaMemcpyAsync = CUDAAPI(cu.cudaMemcpyAsync, [c_cuda_p, c_cuda_p, c_size_t, c_int, c_int])

cudaMemset = CUDAAPI(cu.cudaMemset, [c_cuda_p, c_int, c_size_t])

class MappedBuffer(Structure):
    _fields_ = [('p', c_cuda_p),     # host pointer
                ('pcu', c_cuda_p),   # device pointer
                ('nsize', c_size_t), # number of bytes
                ]

    @property
    def data(self):
        return cast(self.p, POINTER(self.T))[0]
    
    def __init__(self, T, p=None, pcu=None):
        self.nsize = sizeof(T)
        self.T = T
        
        if p:
            self._own_p = False
            self.p = cast(p, c_cuda_p) # c_cuda_p)
        else:
            self._own_p = True
            self.p = cast(cudaMallocHost(self.nsize), c_cuda_p)
            #self.p = cast(pointer(T()), c_cuda_p) 
        
        if pcu:
            self._own_pcu = False
            self.pcu = pcu
        else:
            self._own_pcu = True
            self.pcu = cudaMalloc(self.nsize)

    def __del__(self):
        """free the device memory if owned by this object"""
        if self._own_pcu:
            cudaFree(self.pcu)
        if self._own_p:
            cudaFreeHost(self.p)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        return self.data[i]
    
    def __setitem__(self, i, v):
        self.data[i] = v
    
    def update_device(self):
        cudaMemcpy(self.pcu, self.p, self.nsize, cudaMemcpyHostToDevice)
        
    def update_host(self):
        cudaMemcpy(self.p, self.pcu, self.nsize, cudaMemcpyDeviceToHost)
    
    def update_device_async(self, stream=0):
        cudaMemcpyAsync(self.pcu, self.p, cudaMemcpyHostToDevice, stream)
        
    def update_host_async(self, stream=0):
        cudaMemcpyAsync(self.p, self.pcu, cudaMemcpyDeviceToHost, stream)

    def memset(self, v):
        cudaMemset(self.pcu, v, self.nsize)
        
