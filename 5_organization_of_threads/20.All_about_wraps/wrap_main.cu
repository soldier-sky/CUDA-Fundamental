/*******************************************************************************
* Author: Sunil Kumar Yadav
* Date: 25 July 2021
* Problem: Priniting details of wrap and understanding how block and wrap work 
*           w.r.t. active and inactive threads as per block size
*
********************************************************************************/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>


// printing wrap details of 2,2 grid with block size as 32. Note wrap size is 32
__global__ void print_wrap_details()
{
    // global thread index
    int gid = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * gridDim.x * blockDim.x;   

    // global block index
    int gbid = blockIdx.y * gridDim.x + blockIdx.x;
    
    // wrap id
    int wid = gid / 32;

    printf("gid: %d, block id: %d, wrap id: %d\n", gid, gbid, wid);
}

int main()
{
    dim3 block_size(42);
    dim3 grid_size(2, 2);

    print_wrap_details << <grid_size, block_size >> > ();
    cudaDeviceSynchronize();
    cudaDeviceReset();

    return 0;
}
