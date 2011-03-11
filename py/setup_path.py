from distutils.sysconfig import get_python_lib
from os.path import join, split

pylib = get_python_lib()
here = split(__file__)[0]
package = split(here)[1]
if package=='py':
    package = split(split(here)[0])[1]
    
fname = join(pylib, package)+'.pth'

s = """# Created by %s

%s
""" % (__file__, here)
print 'creating', fname
f = open(fname, 'w')
f.write(s)
f.close()

f = open(fname, 'r')
print "new contents of %s: \n\n%s\n" % (fname, f.read())
f.close()

incdir = join(split(here)[0],'include')
bindir = raw_input('path of project bin directory to which dlls will be copied (must be in exe path): ')

reg_env = ("""Windows Registry Editor Version 5.00

[HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\Session Manager\Environment]
"CUDA_CT_INC_PATH"=%s
"PROJECT_BIN_PATH"=%s
""" % (repr(incdir), repr(bindir))).replace("'",'"')

print 'Environment updates:'
print reg_env

open('setup_env.reg', 'w').write(reg_env)

print
print "Next, you need to run setup_env.reg"

raw_input('done')
