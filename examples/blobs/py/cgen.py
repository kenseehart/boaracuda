from ctypes import *
import re

"""Custom C struct code generator
Prints C header code for several ctypes struct classes
"""

ctre = re.compile('(c_)?(.*?)(?:_Array_(\d+))?$')

def cgen(*types):
    """Generate C struct code from a ctypes type"""
    slines = []
    for T in types:
        slines.append('typedef struct %s_struct {' % T.__name__)
        for (field,ft) in T._fields_:
            ts = ft.__name__
            m = ctre.match(ts)
            assert(m)
            ts = m.group(2)
            dim = m.group(3)
            if dim:
                dim = '['+dim+']'
            
            if ts.startswith('c_'):
                ts = ts[2:]
            slines.append('    %s %s%s;' % (ts, field, dim))
            
        slines.append('} %s;\n' % T.__name__)
        
    return '\n'.join(slines)

from blobs import *

print cgen(V3, State, Attraction, Node, Blob)
