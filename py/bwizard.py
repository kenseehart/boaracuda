import os, sys, uuid, re

def template_replace(s, d):
    for k,v in d.items():
        s = s.replace('{{%s}}'%k, v)
    return s

def name_replace(s, d):
    for k,v in d.items():
        s = s.replace(k, v)
    return s

try:
    project_name = sys.argv[1]
except:
    project_name = raw_input('project name:')
    
exclude = re.compile(r'(?:.*\.tmp)|(?:__doc\.txt)|(?:.*\.svn)')

d = {
    'project_name':project_name,
    'sln_guid':str(uuid.uuid1()).upper(),
    'project_guid':str(uuid.uuid1()).upper(),
    }

tdir = os.path.join(os.path.split(__file__)[0], 'templates')

os.mkdir(project_name)

for root, dirs, files in os.walk(tdir):
    print root, dirs, files
    assert (root.startswith(tdir))
    root_tail = root[len(tdir)+1:]
    
    if exclude.match(root_tail):
        continue
    
    root2 = os.path.join(project_name, root_tail)
    
    
    for dd in dirs:
        if exclude.match(dd):
            continue
        dd = name_replace(dd,d)
        os.mkdir(os.path.join(root2,dd))

    for f in files:
        if exclude.match(f):
            continue
    
        f2 = os.path.join(root2, name_replace(f,d))
        s = open(os.path.join(root, f)).read()
        s = template_replace(s,d)
        open(f2, 'w').write(s)

