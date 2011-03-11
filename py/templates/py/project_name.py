from cuda_ct import *

DEBUG = True

if DEBUG:
    # debug version ends with _d
    {{project_name}}_lib = cdll.{{project_name}}_d
else:
    {{project_name}}_lib = cdll.{{project_name}}

# wrap kernel in python
add = {{project_name}}_lib.add
add.argtypes = [
    c_cuda_p,
    c_cuda_p,
    c_cuda_p,
    c_uint32,
    ]

if __name__ == '__main__':
    # allocate host/device buffers
    a = MappedBuffer(c_int32*10)
    b = MappedBuffer(c_int32*10)
    c = MappedBuffer(c_int32*10)

    # initialize input data
    a.data[:] = range(10)
    b.data[:] = range(10)
    a.update_device()
    b.update_device()

    # invoke the kernel
    add(a.pcu, b.pcu, c.pcu, 10)
    
    # copy output data to the host
    c.update_host()
    
    print c.data[:]
