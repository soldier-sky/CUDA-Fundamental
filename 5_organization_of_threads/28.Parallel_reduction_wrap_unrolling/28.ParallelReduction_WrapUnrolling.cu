/*******************************************************************************
* Author: Sunil Kumar Yadav
* Date: 1 Aug 2021
* Problem: Parallel reductionnchronization example with wrap unrolling
*          In this examplae we will manually unroll the multiple blocks and allowing us 
*		   to reduce block size requirment. We can perfrom summation of multimple block using 
*		   single thread block at a time
********************************************************************************/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include<string.h>
#include<stdlib.h>
#include<time.h>
#include"common.h"

// Reducing wrap diveragence using manual wrap unrolling 2 block
__global__ void parallel_reduction_wrap_unrolling_2block(int* input, int* temp, const int size)
{
	int tid = threadIdx.x;
	int block_offset = blockDim.x * blockIdx.x * 2;    //offset for  wrap unrolling
	int* i_input = input + block_offset;              //local pointer to input data as per block offset 
	int index = tid + block_offset;

	if ((index+blockDim.x) <size)
	{
		input[index] += input[index + blockDim.x];        // using input as we are unrolling blocks
	}
	__syncthreads();

	for (int offset = blockDim.x / 2; offset > 0; offset /= 2)
	{
		if (tid < offset)
			i_input[tid] += i_input[tid + offset];

		__syncthreads();					// wait for all thread in a block to calculate summation
	}

	if (tid == 0)							// summation result of each block will be at 0th index
		temp[blockIdx.x] = i_input[0];

}

// Reducing wrap diveragence using manual wrap unrolling 4 block
__global__ void parallel_reduction_wrap_unrolling_4block(int* input, int* temp, const int size)
{
	int tid = threadIdx.x;
	int block_offset = blockDim.x * blockIdx.x * 4;    //offset for 4 wrap unrolling
	int* i_input = input + block_offset;              //local pointer to input data as per block offset 
	int index = tid + block_offset;

	if ((index + 3*blockDim.x) < size)
	{
		int a1=  input[index + 3*blockDim.x];
		int a2= input[index + 2*blockDim.x];
		int a3= input[index + blockDim.x];
		input[index] += a1+a2+a3;
	}
	__syncthreads();

	for (int offset = blockDim.x / 2; offset > 0; offset /= 2)
	{
		if (tid < offset)
			i_input[tid] += i_input[tid + offset];

		__syncthreads();					// wait for all thread in a block to calculate summation
	}

	if (tid == 0)							// summation result of each block will be at 0th index
		temp[blockIdx.x] = i_input[0];		


}


// 29. reduction warp unrolling with 8 blocks unrolling. 
// make sure grid size is 1/8th of original grid size
__global__ void reduction_kernel_interleaved_warp_unrolling8_1(int* input,
	int* temp_array, int size)
{
	int tid = threadIdx.x;

	//element index for this thread
	int index = blockDim.x * blockIdx.x * 8 + threadIdx.x;

	//local data pointer
	int* i_data = input + blockDim.x * blockIdx.x * 8;

	if ((index + 7 * blockDim.x) < size)
	{
		int a1 = input[index];
		int a2 = input[index + blockDim.x];
		int a3 = input[index + 2 * blockDim.x];
		int a4 = input[index + 3 * blockDim.x];
		int a5 = input[index + 4 * blockDim.x];
		int a6 = input[index + 5 * blockDim.x];
		int a7 = input[index + 6 * blockDim.x];
		int a8 = input[index + 7 * blockDim.x];

		input[index] = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8;
	}

	__syncthreads();

	for (int offset = blockDim.x / 2; offset >= 64;
		offset = offset / 2)
	{
		if (tid < offset)
		{
			i_data[tid] += i_data[tid + offset];
		}
		__syncthreads();
	}

	if (tid < 32)					// complete unrolling afte tid 32 to reduce divergence
	{
		volatile int* vsmem = i_data;			// volatile to ensure no cached valued used
		vsmem[tid] += vsmem[tid + 32];
		vsmem[tid] += vsmem[tid + 16];
		vsmem[tid] += vsmem[tid + 8];
		vsmem[tid] += vsmem[tid + 4];
		vsmem[tid] += vsmem[tid + 2];
		vsmem[tid] += vsmem[tid + 1];
	}

	if (tid == 0)
	{
		temp_array[blockIdx.x] = i_data[0];
	}
}


// 29. reduction complete unrolling
__global__ void reduction_kernel_complete_unrolling8_1(int* input,
	int* temp, int size)
{
	int tid = threadIdx.x;
	int index = blockDim.x * blockIdx.x * 8 + threadIdx.x;

	int* i_data = input + blockDim.x * blockIdx.x * 8;

	if ((index + 7 * blockDim.x) < size)
	{
		int a1 = input[index];
		int a2 = input[index + blockDim.x];
		int a3 = input[index + 2 * blockDim.x];
		int a4 = input[index + 3 * blockDim.x];
		int a5 = input[index + 4 * blockDim.x];
		int a6 = input[index + 5 * blockDim.x];
		int a7 = input[index + 6 * blockDim.x];
		int a8 = input[index + 7 * blockDim.x];

		input[index] = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8;
	}

	__syncthreads();

	//complete unrolling manually

	//if the block dim == 1024 
	if (blockDim.x == 1024 && tid < 512)
		i_data[tid] += i_data[tid + 512];
	__syncthreads();

	if (blockDim.x >= 512 && tid < 256)
		i_data[tid] += i_data[tid + 256];
	__syncthreads();

	if (blockDim.x >= 256 && tid < 128)
		i_data[tid] += i_data[tid + 128];
	__syncthreads();

	if (blockDim.x >= 128 && tid < 64)
		i_data[tid] += i_data[tid + 64];
	__syncthreads();


	// warp unrolling
	if (tid < 32)
	{
		volatile int* vsmem = i_data;
		vsmem[tid] += vsmem[tid + 32];
		vsmem[tid] += vsmem[tid + 16];
		vsmem[tid] += vsmem[tid + 8];
		vsmem[tid] += vsmem[tid + 4];
		vsmem[tid] += vsmem[tid + 2];
		vsmem[tid] += vsmem[tid + 1];
	}

	if (tid == 0)
	{
		temp[blockIdx.x] = i_data[0];
	}
}


int main()
{
	int size = 1 << 27;     // 128Mb of data
	int block_size = 128;
	int grid_size = (size / block_size);             
													 

	dim3 block(block_size);
	dim3 grid(grid_size/2);						// For wrap unrolling, ensure right no. of blocks. 
											    // For 2 block unrolling we need only half block as without unrolling.

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
	parallel_reduction_wrap_unrolling_2block << <grid, block >> > (d_input, d_temp, size);
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