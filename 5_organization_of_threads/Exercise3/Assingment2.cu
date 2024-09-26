/**************************************************************************************
* Author: Sunil Kumar Yadav
* Date: 24 July 2021
* Problem: In the assignment you have to implement array summation in GPU which can sum 3 arrays.
*   You have to use error handling mechanisms, timing measuring mechanisms as well.
*   Then you have to measure the execution time of you GPU implementations.
*
* Note: GPU and CPU time are very small number and end up zero for array size 10000. Since clock() resultation was not satisfactory, using chrono lib from C++
* Assignment 2:
* 1. Imagine you have 3 randomly initialized arrays with 2 to the power 22 elements (4194304). You have to write a CUDA program to sum up these three arrays in your device.  
* 2. First write the c function to sum up these 3 arrays in CPU.
* 3. Then write kernel and launch that kernel to sum up these three arrays in GPU.
* 4. You have to use the CPU timer we discussed in the first section to measure the timing of your CPU and GPU implementations.
* 5. You have to add CUDA error checking mechanism we discussed as well.
* 6. Your grid should be 1Dimensional.
* 7. Use 64, 128, 256, 512 as block size in X dimension and run your GPU implementations with each of these block configurations and measure the execution time.
***************************************************************************************/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include<stdlib.h>
#include<time.h>
#include <chrono>   // for accurate time measurement

//---------------------------------------------------------------------------------------
/* helper function to log cuda error code*/
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

//---------------------------------------------------------------------------------------
/* helper function to compare device and host result*/
bool compare_results(int* d_result, int* h_result, int aray_size)
{
    for (int i = 0; i < aray_size; ++i)
    {
        if (d_result[i] != h_result[i])
        {
            return false;
        }
    }

    return true;
}

// kernal definition to calculate summation of 3 array
__global__ void summation_3_array_kernel(int *array1, int* array2, int* array3, int* result, int array_size)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // since calculating parallel on gpu, no need to have loops like traditional CPU calculation
    if (tid < array_size)
    {
         result[tid] = array1[tid] + array2[tid] + array3[tid];
    }  
}

// CUP side summation function 
void summation_3_array_host(int* array1, int* array2, int* array3, int* result, int array_size)
{
    for (int i = 0; i < array_size; ++i)
    {
        result[i] = array1[i] + array2[i] + array3[i];
    }
}
//--------------------------------------------------------------------------------------------------------------------------

int main()
{
    int array_size = 1000000;
    int block_size = 512;
    int grid_size = (array_size / block_size) + 1;

    size_t NO_BYTES = sizeof(int) * array_size;
    int* h_array_a = (int*)malloc(NO_BYTES);
    int* h_array_b = (int*)malloc(NO_BYTES);
    int* h_array_c = (int*)malloc(NO_BYTES);
    int* h_sum_result = (int*)malloc(NO_BYTES);
    int* gpu_calculated_result = (int*)malloc(NO_BYTES);

    memset(gpu_calculated_result, 0, NO_BYTES);
    memset(h_sum_result, 0, NO_BYTES);

    time_t t;
    srand(time(&t));                   // randon no. generator seed

    for (int i = 0; i < array_size; ++i)                   // assigning random value from 0 - 4194304 to array
    {
        h_array_a[i] = rand() & 4194304;
        h_array_b[i] = rand() & 4194304;
        h_array_c[i] = rand() & 4194304;
    }
    
   
    /*Host side summation to validate calcuated result*/
    std::chrono::time_point<std::chrono::system_clock> cpu_start = std::chrono::system_clock::now();
    summation_3_array_host(h_array_a, h_array_b, h_array_c, h_sum_result, array_size);
    std::chrono::time_point<std::chrono::system_clock> cpu_end = std::chrono::system_clock::now();


 //--------------------------------------------------------------------------------------------------------------------------
    /*Device side summation calculation*/
    int* d_array_a, * d_array_b, * d_array_c, *d_sum_result;

    gpuErrchk(cudaMalloc((int**)&d_array_a, NO_BYTES));
    gpuErrchk(cudaMalloc((int**)&d_array_b, NO_BYTES));
    gpuErrchk(cudaMalloc((int**)&d_array_c, NO_BYTES));
    gpuErrchk(cudaMalloc((int**)&d_sum_result, NO_BYTES));

    std::chrono::time_point<std::chrono::system_clock> gpu_copy_start = std::chrono::system_clock::now();
    gpuErrchk(cudaMemcpy(d_array_a, h_array_a, NO_BYTES, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_array_b, h_array_b, NO_BYTES, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_array_c, h_array_c, NO_BYTES, cudaMemcpyHostToDevice));
    std::chrono::time_point<std::chrono::system_clock> gpu_copy_end = std::chrono::system_clock::now();

    // Change block and grid size as per array_size
    dim3 block(block_size);
    dim3 grid(grid_size);

    std::chrono::time_point<std::chrono::system_clock> gpu_execution_start = std::chrono::system_clock::now(); 
    summation_3_array_kernel << < grid, block >> > (d_array_a, d_array_b, d_array_c, d_sum_result, array_size);
    cudaDeviceSynchronize();
    std::chrono::time_point<std::chrono::system_clock> gpu_execution_end = std::chrono::system_clock::now();


    std::chrono::time_point<std::chrono::system_clock> gpu_copy_back_start = std::chrono::system_clock::now();
    gpuErrchk(cudaMemcpy(gpu_calculated_result, d_sum_result, NO_BYTES, cudaMemcpyDeviceToHost));
    std::chrono::time_point<std::chrono::system_clock> gpu_copy_back_end = std::chrono::system_clock::now();

    cudaDeviceReset();

//--------------------------------------------------------------------------------------------------------------------------
    /*compare results cpu summation and gpu summation */
    bool res=compare_results(gpu_calculated_result, h_sum_result, array_size);
    if(res)
        printf("Result match: Success\n");
    else
        printf("Result do not match: Failure\n");

//--------------------------------------------------------------------------------------------------------------------------  
    //auto durationCpu = std::chrono::duration_cast<std::chrono::nanoseconds>(endCpuCalc - startCpuCalc).count();  //+
    //printf("Time to sum the arrays: %d nanoseconds\n", durationCpu);  //+
    /*Printing CPU and GPU execution time*/
    /*
    double cpu_total_time = (double)((double)(cpu_end - cpu_start) / CLOCKS_PER_SEC);

    double gpu_host_to_device_transfer_time = (double)((double)(gpu_copy_end - gpu_copy_start) / CLOCKS_PER_SEC);
    double gpu_execution_time = (double)((double)(gpu_execution_end - gpu_execution_start) / CLOCKS_PER_SEC);
    double gpu_device_to_host_transfer_time = (double)((double)(gpu_copy_back_end - gpu_copy_back_start) / CLOCKS_PER_SEC);

    double gpu_total_time = gpu_host_to_device_transfer_time + gpu_execution_time + gpu_device_to_host_transfer_time;
    */

    auto cpu_total_time = std::chrono::duration_cast<std::chrono::nanoseconds>(cpu_end - cpu_start).count();

    auto gpu_host_to_device_transfer_time = std::chrono::duration_cast<std::chrono::nanoseconds>(gpu_copy_end - gpu_copy_start).count();
    auto gpu_execution_time = std::chrono::duration_cast<std::chrono::nanoseconds>(gpu_execution_end - gpu_execution_start).count();
    auto gpu_device_to_host_transfer_time = std::chrono::duration_cast<std::chrono::nanoseconds>(gpu_copy_back_end - gpu_copy_back_start).count();
    auto gpu_total_time = gpu_host_to_device_transfer_time + gpu_execution_time + gpu_device_to_host_transfer_time;

    printf("Block size: %d and grid size: %d\n", block_size, grid_size);

    printf("CPU total execution time :  %d nanoseconds \n", cpu_total_time);
    printf("Host to Device transfer time :  %d nanoseconds \n", gpu_host_to_device_transfer_time);
    printf("GPU execution time :  %d nanoseconds\n", gpu_execution_time);
    printf("Device to Host transfer time :  %d nanoseconds \n", gpu_device_to_host_transfer_time);
    printf("GPU total execution time :  %d nanoseconds \n", gpu_total_time);

//--------------------------------------------------------------------------------------------------------------------------
    free(h_array_a);
    free(h_array_b);
    free(h_array_c);
    free(h_sum_result);

    cudaFree(d_array_a);
    cudaFree(d_array_b);
    cudaFree(d_array_c);
    cudaFree(d_sum_result);
    
    return 0;
}

