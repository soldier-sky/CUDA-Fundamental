/*******************************************************************************
* Author: Sunil Kumar Yadav
* Date: 1 Aug 2021
* Notes: Understanding dynamic parallelism. In this example we will launch cuda 
*		kernel recuersivly via already lauched kernel from host side.
* Compile command : nvcc -gencode=arch=compute_52,code=\"sm_52,compute_52\" -rdc=true DynamicParallism.cu
********************************************************************************/
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void dynamic_parrlalism_check(int size, int depth)
{

	int tid = threadIdx.x;

	printf("Depth: %d, tid: %d \n ", depth, tid);

	if (size == 1)
		return;
		
	if(threadIdx.x==0)					// on every 0th index launch kernel dynamically
		dynamic_parrlalism_check << <1,size/2 >> > (size / 2, depth + 1);		// launch 1 grid with block size half as parent

}

// print depth and tid for thread block with size 8 and grid size as 2
__global__ void dynamic_parrlalism_check_assignment(int size, int depth)
{

	int tid = threadIdx.x;
	int bid = blockIdx.x;
	int gid = tid + blockIdx.x * blockDim.x;


	printf("gid: %d, blockIdx.x: %d, tid:%d, Depth : % d \n ",gid, bid , tid, depth);

	if (size == 1)
		return;

	if (threadIdx.x == 0 && blockIdx.x==0)					// on every 0th thread index and blockIdx.x 0, launch kernel dynamically
		dynamic_parrlalism_check_assignment << <2, size/2 >> > (size / 2, depth + 1);		// launch 1 grid with block size half as parent

}

int main()
{
	int block_size = 8;
	int grid_size = 1;
	int size = 8;

	dim3 block(block_size);
	dim3 grid(grid_size);

	//dynamic_parrlalism_check << <grid, block >>> (size, 0);
	dynamic_parrlalism_check_assignment << <2, 8 >>> (8, 0);
	cudaDeviceSynchronize();
	cudaDeviceReset();
	return 0;
}
