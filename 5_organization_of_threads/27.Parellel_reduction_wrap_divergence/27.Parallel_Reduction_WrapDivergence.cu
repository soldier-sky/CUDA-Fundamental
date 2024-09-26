/*******************************************************************************
* Author: Sunil Kumar Yadav
* Date: 1 Aug 2021
* Problem: Parallel reductionnchronization example to calculate summation of array
*          In this examplae using interleaved pair method we can reduce overhead in half
*			in thread block and reduce divergence
********************************************************************************/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include<string.h>
#include<stdlib.h>
#include"common.h"

// Reducing wrap diveragence using parrallel interleaved pair approch
__global__ void parallel_reduction_interleaved_pair_summation(int* input, int* temp, const int size)
{
	int tid = threadIdx.x;
	int gid = threadIdx.x + blockIdx.x * blockDim.x;

	if (gid > size)
		return;

	for (int offset = blockDim.x / 2; offset > 0; offset /= 2)
	{
		if (tid < offset)
			input[gid] += input[gid + offset];

		__syncthreads();					// wait for all thread in a block to calculate summation
	}
	

	if (tid == 0)							// summation result of each block will be at 0th index
		temp[blockIdx.x] = input[gid];

}


// improving neighbored pair kernel
__global__ void parallel_reduction_neighbored_pair_improved(int* input, int* temp, const int size)
{
	int tid = threadIdx.x;
	int gid = threadIdx.x + blockIdx.x * blockDim.x;

	int* i_data = input + blockIdx.x * blockDim.x;       //local pointer to input with block offset

	if (gid > size)
		return;

	
	for (int offset = 1; offset <= blockDim.x / 2; offset *= 2)
	{
		int index = 2 * offset * tid;

		if (index < blockDim.x)
		{
			i_data[index] += i_data[index + offset];
		}

		__syncthreads();
	}					

	if (tid == 0)							// summation result of each block will be at 0th index
		temp[blockIdx.x] = input[gid];

}


int main()
{
	int size = 1 << 27;     // 128Mb of data
	int block_size = 128;
	int grid_size = (size / block_size);

	dim3 block(block_size);
	dim3 grid(grid_size);

	int mem_size = size * sizeof(int);
	int reduction_array_size = grid.x * sizeof(int);

	int* h_input, * h_temp;
	h_input = (int*)malloc(mem_size);
	h_temp = (int*)malloc(reduction_array_size);

	memset(h_input, 0, mem_size);
	memset(h_temp, 0, reduction_array_size);

	//initialize array with random no.
	initialize_1d_array(h_input, size);

	// calculate result on cpu and validate with GPU
	int cpu_result = 0;
	cpu_result = sum_1d_array(h_input, size);

	int* d_input, * d_temp;
	gpuErrchk(cudaMalloc((void**)&d_input, mem_size));
	gpuErrchk(cudaMalloc((void**)&d_temp, reduction_array_size));

	gpuErrchk(cudaMemset(d_temp, 0, reduction_array_size));

	gpuErrchk(cudaMemcpy(d_input, h_input, mem_size, cudaMemcpyHostToDevice));


	printf("Kernal launch configuration: block size: %d and grid size: %d\n\n", block.x, grid.x);
	// lets spin the gup
	parallel_reduction_interleaved_pair_summation << <grid, block >> > (d_input, d_temp, size);
	//parallel_reduction_neighbored_pair_improved << <grid, block >> > (d_input, d_temp, size);   // improvement over 26.parallel_reduction
	gpuErrchk(cudaDeviceSynchronize());

	gpuErrchk(cudaMemcpy(h_temp, d_temp, reduction_array_size, cudaMemcpyDeviceToHost));



	// calculate final result from GPU parallel reduction
	int gpu_result = 0;
	for (int i = 0; i < grid.x; i++)
		gpu_result += h_temp[i];



	printf("Result of CPU calculation: %d and GPU calculation: %d \n", cpu_result, gpu_result);
	if (compare_results(cpu_result, gpu_result))
	{
		printf("Result matches\n");
	}
	else
		printf("Result does not matches\n");


	free(h_input);
	free(h_temp);

	gpuErrchk(cudaFree(d_input));
	gpuErrchk(cudaFree(d_temp));

	gpuErrchk(cudaDeviceReset());
	return 0;
}