/*******************************************************************************
* Author: Sunil Kumar Yadav
* Date: 2 Aug 2021
* Notes:  Understanding memory model with the help of Array of Structure vs Structure of Array example.
*		  Example demonstarte 
* 
********************************************************************************/
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include<stdlib.h>
#include<string.h>

#define ARRAY_SIZE 1<<27

// used for AOS
struct test_struct1
{
	int x;
	int y;
};

// used for SOA
struct test_struct2
{
	int x[ARRAY_SIZE];
	int y[ARRAY_SIZE];
};

// Function added offset to input array of structure  and place inside output structure
__global__ void kernel_AOS(struct test_struct1 *input, struct test_struct1* result, int size)
{

	int tid = threadIdx.x;
	int gid = tid + blockDim.x * blockIdx.x;

	struct test_struct1 temp;
	if (gid < size)
	{
		temp = input[gid];
		temp.x += 10;
		temp.y += 50;

		result[gid] = temp;
	}	

}


__global__ void kernel_SOA(struct test_struct2* input, struct test_struct2* result, int size)
{

	int tid = threadIdx.x;
	int gid = tid + blockDim.x * blockIdx.x;

	if (gid < size)
	{
		result->x[gid] = input->x[gid]+10;
		result->y[gid] = input->x[gid] + 50;
	}

}

void  test_AOS()
{
	printf("Running Test AOS block\n");
	int block_size = 128;
	int grid_size = ARRAY_SIZE/ block_size;

	unsigned long long AOS_SIZE = sizeof(struct test_struct1) * ARRAY_SIZE;

	dim3 block(block_size);
	dim3 grid(grid_size);

	struct test_struct1* h_input_aos;
	struct test_struct1* h_result_aos;
	h_input_aos = (struct test_struct1*)malloc(AOS_SIZE);
	h_result_aos = (struct test_struct1*)malloc(AOS_SIZE);

	memset(h_result_aos, 0, AOS_SIZE);
	// initializing two structre arrays
	for (unsigned int i = 0; i < ARRAY_SIZE; ++i)
	{
		h_input_aos[i].x = 10;
		h_input_aos[i].y = 20;
	}


	struct test_struct1* d_input_aos;
	struct test_struct1* d_result;

	cudaMalloc((void**)&d_input_aos, AOS_SIZE);
	cudaMalloc((void**)&d_result, AOS_SIZE);

	cudaMemcpy(d_input_aos, h_input_aos, AOS_SIZE, cudaMemcpyHostToDevice);

	kernel_AOS << <grid, block >> > (d_input_aos, d_result, ARRAY_SIZE);
	cudaDeviceSynchronize();

	cudaMemcpy(h_result_aos, d_result, AOS_SIZE,cudaMemcpyDeviceToHost);
	cudaDeviceReset();

}


int test_SOA()
{

	printf("Running Test SOA block\n");
	int block_size = 128;
	int grid_size = ARRAY_SIZE / block_size;

	unsigned long long SOA_SIZE = sizeof(struct test_struct2) * ARRAY_SIZE;

	dim3 block(block_size);
	dim3 grid(grid_size);

	struct test_struct2* h_input_soa;
	struct test_struct2* h_result_soa;
	h_input_soa = (struct test_struct2*)malloc(SOA_SIZE);
	h_result_soa = (struct test_struct2*)malloc(SOA_SIZE);

	memset(h_result_soa, 0, SOA_SIZE);

	// initializing two structre arrays
	for (unsigned int i = 0; i < ARRAY_SIZE; ++i)
	{
		h_input_soa->x[i] = 10;
		h_input_soa->y[i] = 20;

	}


	struct test_struct2* d_input_soa;
	struct test_struct2* d_result;

	cudaMalloc((void**)&d_input_soa, SOA_SIZE);
	cudaMalloc((void**)&d_result, SOA_SIZE);

	cudaMemcpy(d_input_soa, h_input_soa, SOA_SIZE, cudaMemcpyHostToDevice);

	kernel_SOA << <grid, block >> > (d_input_soa, d_result, ARRAY_SIZE);
	cudaDeviceSynchronize();

	cudaMemcpy(h_result_soa, d_result, SOA_SIZE,cudaMemcpyDeviceToHost);

	cudaDeviceReset();
	return 0;
}

int main(int argc, char** argv)
{
	int kernel_ind = 0;

	if (argc > 1)
	{
		kernel_ind = atoi(argv[1]);
	}

	if (kernel_ind == 0)
	{
		test_AOS();
		printf("Done execution\n");
	}
	else
	{
		test_SOA();
		printf("Done execution\n");
	}

	return EXIT_SUCCESS;
}
