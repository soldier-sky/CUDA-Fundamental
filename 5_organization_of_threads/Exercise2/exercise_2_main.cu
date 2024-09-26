/****************************************************************************
* Author: Sunil Kumar Yadav
* Date: 21 July 2021
* Description: Exercise 2. Randomly initialize 64 element array and pass this
*	array to device. Launch 3D grid where each dimension have 4 threads and 
*   thread blocks will have 2 threads in all dimension. Print the array element
*   value using this grid.
* Note: Use row major format
****************************************************************************/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include<time.h>       // to generate seed for random no. generator
#include<stdlib.h>    // for random no. 


__global__ void unique_gid_calc_thread_3d(int* input)
{
	int tid = threadIdx.x + blockDim.x * threadIdx.y + blockDim.x  *blockDim.y * threadIdx.z; // blockDim.x * threadIdx.y => offset for y plan and blockDim.x  *blockDim.y * threadIdx.z => offset for z plan

	// calculating offset required for block travarsal in x direction
	int num_thread_in_block = blockDim.x * blockDim.y * blockDim.z;
	int block_offset = num_thread_in_block* blockIdx.x;
	
	// calculating offset required for row travarsal in grid in y direction
	int num_thread_in_row = num_thread_in_block * gridDim.x;
	int row_offset = num_thread_in_row * blockIdx.y;

	// calculating offset required for plan travarsal in z direction
	int num_thread_in_xy_plan = num_thread_in_row * gridDim.y;
	int xy_plane_offset = num_thread_in_xy_plan * blockIdx.z;

	int gid = tid + block_offset + row_offset+ xy_plane_offset;
	
	printf("gid:%d and value: %d\n", gid, input[gid]);
}

int main()
{

	int array_size = 64;
	int array_byte_size = sizeof(int) * array_size;
	int *h_array;

	h_array = (int*) malloc(array_byte_size);
	
	time_t t;
	srand(time(&t));

	
	for (int i = 0; i < array_size; ++i)
	{
		h_array[i] = int(rand() & 0xFF);   // random no. from 0 to 255
		//h_array[i] = i;   // random no. from 0 to 255
	}


	for (int i = 0; i < array_size; ++i)
	{
		printf("%d ", h_array[i]);
	}
	printf("\n\n");


	int* d_array;
	cudaMalloc((void**)&d_array, array_byte_size);
	cudaMemcpy(d_array, h_array, array_byte_size, cudaMemcpyHostToDevice);

	dim3 block(2,2,2);
	dim3 grid(2,2,2);

	unique_gid_calc_thread_3d << <grid, block>> > (d_array);

	cudaDeviceSynchronize();
	cudaDeviceReset();

	free(h_array);
	cudaFree(d_array);

	return 0;
}