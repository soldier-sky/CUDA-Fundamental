/************************************************************************************
* Author: Sunil Kumar Yadav
* Date: 15 Aug 2021
* Description: Understanding how to delare static and dynamic shared memory.
*			  Shared memory allows faster memory access than main memory. 
* Note: memory bank access can have conlicts which adversily impact the perfromance 
*************************************************************************************/
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include<stdlib.h>
#include <time.h>
#include<string.h>

#define SMEM_SIZE 128

/* helper function to initialize array with random no.*/
void initialize_1d_array(int* input, const unsigned int size);

/* helper function to compare device and host result*/
bool compare_results(int* d_result, int* h_result, unsigned int size);

/* function to copy input buffer content to output buffer via static shared memory */
__global__ void smem_static_test(int* in, int* out, int size)
{
	__shared__ int smem[SMEM_SIZE];
	int tid = threadIdx.x;
	int gid = tid + blockIdx.x * blockDim.x;

	if(gid<size)
	{
		smem[tid] = in[gid];
		out[gid] = smem[tid];
	}
}

/* function to copy input buffer content to output buffer via dynamic shared memory */
__global__ void smem_dynamic_test(int* in, int* out, int size)
{
	extern __shared__ int smem[];                     // note: dynamic shared memory is only one dimensional
	int tid = threadIdx.x;
	int gid = tid + blockIdx.x * blockDim.x;

	if (gid < size)
	{
		smem[tid] = in[gid];
		out[gid] = smem[tid];
	}
}


int main()
{
	int array_size = 1 << 20; // 1MB data
	int array_size_byte = sizeof(int) * array_size;

	int* h_in, * h_out, *h_ref;
	h_in = (int*)malloc(array_size_byte);
	h_out = (int*)malloc(array_size_byte);
	h_ref = (int*)malloc(array_size_byte);
	

	initialize_1d_array(h_in, array_size);     
	memset(h_out, 0, array_size_byte);
	memset(h_ref, 0, array_size_byte);

	/* host side memory transfer withing two buffers*/
	for (int i = 0; i < array_size; ++i)
		h_out[i] = h_in[i];

	int block_size = 128;
	int grid_size = array_size / block_size;

	dim3 block(block_size);
	dim3 grid(grid_size);

	printf("Launch parameters:\nSize:%d \t Block:%d \t Grid:%d\n", array_size, block.x, grid.x);

	int* d_in, *d_out;
	cudaMalloc((void**)&d_in, array_size_byte);
	cudaMalloc((void**)&d_out, array_size_byte);

	cudaMemcpy(d_in, h_in, array_size_byte, cudaMemcpyHostToDevice);

	bool is_dynamic = true;    // change to launch one of the kernel
	
	if (!is_dynamic)
	{
		// run gpu
		smem_static_test << <grid, block >> > (d_in, d_out, array_size);
	}
	else
	{
		smem_dynamic_test << <grid, block, sizeof(int)*SMEM_SIZE >> > (d_in, d_out, array_size);    // passing size of dynamic shared memory
	}

	cudaDeviceSynchronize();
	cudaMemcpy(h_ref, d_out, array_size_byte, cudaMemcpyDeviceToHost);

	bool status = compare_results(h_ref,h_out, array_size);       // comparing device and host calculation

	free(h_in);
	free(h_out);
	free(h_ref);
	cudaFree(d_in);
	cudaFree(d_out);

	cudaDeviceReset();

	return 0;
}

//--------------------------------------------------------------------------------------------------------------------------------------------------
/* helper function to initialize array with random no.*/
void initialize_1d_array(int* input, const unsigned int size)
{
	time_t t;
	srand(time(&t));                   // randon no. generator seed

	for (unsigned int i = 0; i < size; ++i)                   // assigning random value from 0 - 0xFFFF to array
	{
		input[i] = rand() & 0xFFFF;
	}
}

/* helper function to compare device and host result*/
bool compare_results(int* d_result, int* h_result, unsigned int size)
{

	for (unsigned int i = 0; i < size; ++i)
	{
		if (d_result[i] != h_result[i])
		{
			printf("index:%d, first:%d, second:%d\n", i, d_result[i], h_result[i]);
			return false;
		}
	}
	printf("Result matched\n");
	return true;
}