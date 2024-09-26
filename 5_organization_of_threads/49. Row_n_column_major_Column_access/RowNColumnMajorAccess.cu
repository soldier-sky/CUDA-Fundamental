/************************************************************************************
* Author: Sunil Kumar Yadav
* Date: 15 Aug 2021
* Description: This program deals with differnt ways of accessing matrixs via shared memory.
*              Once program is exucuted, run nvprof of equivalent to learn memory fetchs etc
* 
* Note: memory bank access can have conlicts which adversily impact the perfromance
*************************************************************************************/
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<cuda.h>

#include <stdio.h>
#include<stdlib.h>
#include <time.h>
#include<string.h>

#define MDIMX 32
#define MDIMY 32

/* helper functions*/
void initialize_1d_array(int* input, const unsigned int size);
bool compare_results(int* d_result, int* h_result, unsigned int size);
void printData(char* msg, int* in, const int size);


//--------------------------------------------------------------------------------------------------------------------------------------------------
/* function sets shared mem in row major format and read into out via column major */
__global__ void setRowReadColumn(int* out)
{
	__shared__ int tile[MDIMX][MDIMY];
	int tid = threadIdx.x;
	int idx = tid + threadIdx.y * blockDim.x;

	// store to shared memory
	tile[threadIdx.y][threadIdx.x] = idx;		// [threadyIdx.y][threadyIdx.x] is row major

	__syncthreads();                          // wait till all threads in wrap have populated tile

	// read into out from shared memory in column major format
	out[idx] = tile[threadIdx.x][threadIdx.y];

}

/* function sets shared mem in column major format and read into out via column major */
__global__ void setColumnReadRow(int* out)
{
	__shared__ int tile[MDIMX][MDIMY];
	int tid = threadIdx.x;
	int idx = tid + threadIdx.y * blockDim.x;

	// store to shared memory
	tile[threadIdx.x][threadIdx.y] = idx;		// [threadyIdx.y][threadyIdx.x] is row major format

	__syncthreads();                          // wait till all threads in wrap have populated tile

	// read into out from shared memory in row major format
	out[idx] = tile[threadIdx.y][threadIdx.x];

}


/* function sets shared mem in row major format and read into out via row major */
__global__ void setRowReadRow(int* out)
{
	__shared__ int tile[MDIMX][MDIMY];
	int tid = threadIdx.x;
	int idx = tid + threadIdx.y * blockDim.x;

	// store to shared memory
	tile[threadIdx.y][threadIdx.x] = idx;		// [threadyIdx.y][threadyIdx.x] is row major

	__syncthreads();                          // wait till all threads in wrap have populated tile

	// read into out from shared memory in row major format
	out[idx] = tile[threadIdx.y][threadIdx.x];

}
//--------------------------------------------------------------------------------------------------------------------------------------------------


int main(int argc, char** argv)
{
	int memconfig = 0;
	if (argc > 1)
	{
		memconfig = atoi(argv[1]);
	}


	if (memconfig == 1)
	{
		cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
	}
	else
	{
		cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte);
	}


	cudaSharedMemConfig pConfig;
	cudaDeviceGetSharedMemConfig(&pConfig);
	printf("with Bank Mode:%s ", pConfig == 1 ? "4-Byte" : "8-Byte");


	// set up array size 2048
	int nx = MDIMX;
	int ny = MDIMY;

	bool iprintf = false;

	if (argc > 2) iprintf = atoi(argv[1]);

	size_t nBytes = nx * ny * sizeof(int);

	// execution configuration
	dim3 block(MDIMX, MDIMY);
	dim3 grid(1, 1);
	printf("<<< grid (%d,%d) block (%d,%d)>>>\n", grid.x, grid.y, block.x, block.y);

	// allocate device memory
	int* d_C;
	cudaMalloc((int**)&d_C, nBytes);
	int* gpuRef = (int*)malloc(nBytes);

	cudaMemset(d_C, 0, nBytes);
	setColumnReadRow << <grid, block >> > (d_C);
	cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);

	if (iprintf)  printData("set col read col   ", gpuRef, nx * ny);

	
	cudaMemset(d_C, 0, nBytes);
	setRowReadRow << <grid, block >> > (d_C);
	cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);

	if (iprintf)  printData("set row read row   ", gpuRef, nx * ny);
	
	cudaMemset(d_C, 0, nBytes);
	setColumnReadRow << <grid, block >> > (d_C);
	cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);

	if (iprintf)  printData("set row read col   ", gpuRef, nx * ny);

	// free host and device memory
	cudaFree(d_C);
	free(gpuRef);

	// reset device
	cudaDeviceReset();
	return EXIT_SUCCESS;
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

/* helper function to print the content of buffer*/
void printData(char* msg, int* in, const int size)
{
	printf("%s: ", msg);

	for (int i = 0; i < size; i++)
	{
		printf("%5d", in[i]);
		fflush(stdout);
	}

	printf("\n");
	return;
}
//--------------------------------------------------------------------------------------------------------------------------------------------------