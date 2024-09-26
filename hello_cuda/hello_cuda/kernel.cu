
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void hello_cuda()
{
    printf("Hello Cuda World\n");
}

int main()
{
    dim3 grid(2);     // equivalent to 2,1,1 i.e. grid.x=2, grid.y=1, grid.z=1
    dim3 block(10);

    hello_cuda << <grid, block >> > ();   // creates 2 grid of 10 blocks of thread i.e. total 20 thread
    //hello_cuda << <2, 10 >> > ();   // creates 2 grid of 10 blocks of thread i.e. total 20 thread
    cudaDeviceSynchronize();   //to make all thread synchronize
    return 0;
}

