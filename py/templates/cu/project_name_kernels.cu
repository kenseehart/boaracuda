
#include <cuda.h>
#include "{{project_name}}.h"


///////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void cu_add(int *a, int *b, int *c, int n)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i<n) // prevent overflow
    {
        c[i]=a[i]+b[i];
    }
}

DLL void add(int *a, int *b, int *c, int n)
{
    dim3 dimGrid((n+63)/64, 1, 1);
    dim3 dimBlock(64, 1, 1);
    cu_add<<<dimGrid, dimBlock>>>(a,b,c,n);
}
