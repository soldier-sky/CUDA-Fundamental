#pragma once

#include<time.h>
#include<string.h>
#include<stdlib.h>	// for random no. generator

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

//--------------------------------------------------------------------------------------------------------------------------------------------------
/* helper function to initialize array with random no.*/
void initialize_1d_array(int* input, const unsigned int size)
{
	time_t t;
	srand(time(&t));                   // randon no. generator seed

	for (unsigned int i = 0; i < size; ++i)                   // assigning random value from 0 - 10 to array
	{
		input[i] = rand()&10;
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

	return true;
}

// transpose calculation from host side
void host_matrix_transpose(int* matrix, int* transpose, int nx, int ny)
{
	for (int ix = 0; ix < nx; ++ix)
		for (int iy = 0; iy < ny; ++iy)
			transpose[ix * ny + iy] = matrix[iy * nx + ix];

}


//--------------------------------------------------------------------------------------------------------------------------------------------------