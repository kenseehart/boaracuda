from ctypes import *
import ctypes
import re
import os

inc = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v3.2\include'
f_cudart = os.path.join(inc, 'cuda_runtime_api.h')

re_cudart = re.compile(r"extern __host__ cudaError_t CUDARTAPI (\w+)\((.*)\);")
re_arg = re.compile(r"\s*(.*?)\s*(\w+)\s*(?:__dv\((.*?)\))?$")

types = [
    # integer types and ptrs
    (r'size_t', 'c_size_t'),
    (r'size_t \*', 'RETURNS(c_size_t)'),
    (r'int', 'c_int'),
    (r'int \*', 'RETURNS(c_int)'),
    (r'float', 'c_float'),
    (r'float \*', 'RETURNS(c_float)'),
    (r'double', 'c_double'),
    (r'double \*', 'RETURNS(c_double)'),
    (r'unsigned int', 'c_uint'),
    (r'unsigned int \*', 'RETURNS(c_uint)'),
    
    # pointers
    (r'const char \*', 'c_char_p'),
    (r'const void \*', 'c_void_p'),
    (r'const void \*\*', 'RETURNS(c_cuda_p)'),
    (r'void \*', 'c_void_p'),
    (r'void \*\*', 'RETURNS(c_cuda_p)'),
    
    # enum and special int types
    (r'enum (?:\w+)', 'c_int'), #it might be handy to use a custom typesafe int
    (r'enum (?:\w+) \*', 'RETURNS(c_int)'), #it might be handy to use a custom typesafe int
    (r'(\w+)_t', 'c_uint'), # like enums, it might be handy to use a custom typesafe int
    (r'(\w+)_t \*', 'RETURNS(c_uint)'),
    (r'const (\w+)_t \*', 'POINTER(c_uint)'),

    # specific types that differ from generic case
    (r'struct (cudaArray) *', 'POINTER({0})'), # special cases: call with pointer
    (r'dim3', 'c_uint*3'),

    # structs
    (r'struct (\w+)', '{0}'),
    (r'const struct (\w+)\s*\*', 'POINTER({0})'),
    (r'const struct (\w+)\s*\*\*', 'RETURNS({0})'),
    (r'struct (\w+)\s*\*\*', 'RETURNS({0})'),
    (r'struct (\w+)\s*\*', 'RETURNS({0})'),
    ]

fname = f_cudart
api_list = []

with open(fname) as f:
    for line, s in enumerate(f):
        m = re_cudart.match(s)
        if m:
            name, args = m.groups()
            ct_args = []
            for arg in args.split(', '):
                if arg=='void':
                    break
                m = re_arg.match(arg)
                t, a, dv = m.groups()
                ct = None
                for tr, ts in types:
                    m = re.match(tr+'$',t)
                    if m:
                        ct = ts.format(*m.groups())
                        break

                if ct:
                    ct_args.append(ct)
                else:
                    err = '{0}[{1}] Unknown type: {2}'.format(fname, line, t)
                    print err
                    raise TypeError(err)
                    
            api_template = '{0} = CUDAAPI(cu.{0}, [{1}])'
            api_list.append(api_template.format(name, ', '.join(ct_args)))

print '\n'.join(api_list)            
        
        
