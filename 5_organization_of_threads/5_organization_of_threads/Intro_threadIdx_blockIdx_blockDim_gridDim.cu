#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void print_threadIdx(void)
{
	printf("threadIdx.x: %d, threadIdx.y:%d, threadIdx.z:%d \n", 
		threadIdx.x, threadIdx.y, threadIdx.z);
}

__global__ void print_thread_details(void)
{
	printf("blockIdx.x: %d, blockIdx.y:%d, blockIdx.z:%d, blockDim.x:%d, blockDim.y:%d, blockDim.z:%d, gridDim.x:%d, gridDim.y:%d, gridDim.z:%d  \n",
		blockIdx.x, blockIdx.y, blockIdx.z, blockDim.x, blockDim.y, blockDim.z, gridDim.x, gridDim.y, gridDim.z);
}

int main()
{
	int nx, ny;
	nx = 16;
	ny = 16;

	dim3 block(8, 8);
	dim3 grid(nx / block.x, ny / block.y);
	
	//print_threadIdx << <grid, block >> > ();
	print_thread_details <<<grid,block>>>();
	cudaDeviceSynchronize();

	cudaDeviceReset();

	return 0;
}