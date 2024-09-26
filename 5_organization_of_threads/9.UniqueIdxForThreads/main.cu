
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>


// works for single grid
__global__ void unique_idx_calc_thread(int* input)
{
    int tid = threadIdx.x;
    printf("threadIdx.x: %d and value: %d\n", tid, input[tid]);
}

__global__ void unique_gid_calc_thread(int* input)
{
    int tid = threadIdx.x;
    int offset = blockIdx.x * blockDim.x;

    int gid = tid + offset;
    printf("threadIdx.x: %d, blockIdx.x: %d, gid: %d and value : % d\n", tid, blockIdx.x, gid, input[gid]);
}



int main()
{
    const int array_size = 8;
    int array_byte_size = sizeof(int) * array_size; 
    int h_data[array_size] = { 11, 44, 66, 22, 55,95, 73, 99 };  //host array

    for (int i = 0; i < array_size; ++i)
        printf("%d ", h_data[i]);

    printf("\n\n");

    // code to copy data from host to device
    int* d_data;        
    cudaMalloc((void**)&d_data, array_byte_size);
    cudaMemcpy(d_data, h_data, array_byte_size, cudaMemcpyHostToDevice);


    dim3 block(4);
    dim3 grid(2);
    
    unique_gid_calc_thread << <grid, block >> > (d_data);
   // unique_idx_calc_thread << <grid, block >> > (d_data);

    cudaDeviceSynchronize();
    cudaDeviceReset();

    return 0;
}


