
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>


// works for single grid
__global__ void unique_idx_calc_thread(int* input)
{
    int tid = threadIdx.x;
    printf("threadIdx.x: %d and value: %d\n", tid, input[tid]);
}

// gid calculation of 2 dimensional block with threads in x dimension(block) only 
// and have 2 such grid in y dimenstion
__global__ void unique_gid_calc_thread_2d(int* input)
{
    int tid = threadIdx.x + blockDim.x * threadIdx.y;

    int num_threads_in_a_block = blockDim.x * blockDim.y;
    int block_offset = blockIdx.x * num_threads_in_a_block;

    int num_threads_in_a_row = num_threads_in_a_block * gridDim.x;
    int row_offset = num_threads_in_a_row * blockIdx.y;

    int gid = tid + block_offset + row_offset;

    printf("threadIdx.x: %d, blockIdx.x: %d, blockDim.y: %d gid: %d and value : % d\n", tid, blockIdx.x, blockDim.y, gid, input[gid]);
}

__global__ void unique_gid_calc_thread_2d_2d(int* input)
{
    int tid = threadIdx.x + threadIdx.y * blockDim.x* blockDim.y;
    int block_offset = blockIdx.x * blockDim.x;
    int row_offset = blockDim.x * gridDim.x * blockIdx.y;

    int gid = tid + block_offset + row_offset;

    printf("threadIdx.x: %d, blockIdx.x: %d, blockIdx.y: %d gid: %d and value : % d\n", tid, blockIdx.x, blockIdx.y, gid, input[gid]);
}


int main()
{
    const int array_size = 16;
    int array_byte_size = sizeof(int) * array_size;
    int h_data[array_size] = { 11, 44, 66, 22, 55, 95, 73, 99 ,211 ,434, 575, 111, 554, 564, 656, 999};  //host array

    for (int i = 0; i < array_size; ++i)
        printf("%d ", h_data[i]);

    printf("\n\n");

    // code to copy data from host to device
    int* d_data;
    cudaMalloc((void**)&d_data, array_byte_size);
    cudaMemcpy(d_data, h_data, array_byte_size, cudaMemcpyHostToDevice);

    /*
    dim3 block(4);
    dim3 grid(2,2);

    // unique_idx_calc_thread << <grid, block >> > (d_data);
    */

    dim3 block(2,2);
    dim3 grid(2, 2);
    printf("\n**********unique_gid_calc_thread_2d_2d*********\n");
    unique_gid_calc_thread_2d << <grid, block >> > (d_data);

    cudaDeviceSynchronize();
    cudaDeviceReset();

    return 0;
}


