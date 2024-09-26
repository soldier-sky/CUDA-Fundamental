/*******************************************************************************
* Author: Sunil Kumar Yadav
* Date: 25 July 2021
* Problem: Priniting details of wrap and understanding how wrap divergence can 
*           impact performer. 
*
* Note: nvprof --metrics branch_efficiency executable does not work for CC 7.5. 
*       use https://developer.nvidia.com/blog/migrating-nvidia-nsight-tools-nvvp-nvprof/
********************************************************************************/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void code_without_divergence()
{
	int gid = threadIdx.x + blockDim.x * blockIdx.x;

	int wid=gid/32;      // wrap id
	float a, b;
	
	if (wid % 2 == 0)
	{
		a = 100;
		b = 50;
	}
	else
	{
		a = 200;
		b = 75;
	}

}

__global__ void code_with_divergence()
{
	int gid = threadIdx.x + blockDim.x * blockIdx.x;

	float a, b;

	if (gid % 2 == 0)
	{
		a = 100;
		b = 50;
	}
	else
	{
		a = 200;
		b = 75;
	}

}


int main()
{
	int size = 1 << 20;

	dim3 block_size(128);
	dim3 grid_size((size+ block_size.x -1)/block_size.x);

	code_without_divergence << <grid_size, block_size >> > ();
	cudaDeviceSynchronize();

	code_with_divergence << <grid_size, block_size >> > ();
	cudaDeviceSynchronize();

	cudaDeviceReset();
	return 0;
}