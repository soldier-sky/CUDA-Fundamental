/*******************************************************************************
* Author: Sunil Kumar Yadav
* Date: 4 Aug 2021
* Problem: Sample program to understand performance of kernel using matrix transpose.
*			Here we will use multiple kernel to gauge the perfomance while solving
*			transpose issue.
* Note: Matrix size can span across multiple grids in x and y direction
* 
* Import Note: Buggy and need to work on unrolled version launch parameter
********************************************************************************/
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"

#include <stdio.h>
#include"common.h"


//---------------------------------------------------------------------------------------
// copy matrix 1 to matrix 2 row wise 
__global__ void copy_matrix_row(int* matrix1, int* matrix2, int nx, int ny)
{
	int ix = threadIdx.x + blockDim.x * blockIdx.x;
	int iy = threadIdx.y + blockDim.y * blockIdx.y;

	if (ix < nx && iy < ny)
	{
		matrix2[iy * nx + ix] = matrix1[iy * nx + ix];
	}
}

// copy matrix 1 to matrix 2 column wise 
__global__ void copy_matrix_column(int* matrix1, int* matrix2, int nx, int ny)
{
	int ix = threadIdx.x + blockDim.x * blockIdx.x;
	int iy = threadIdx.y + blockDim.y * blockIdx.y;

	if (ix < nx && iy < ny)
	{
		matrix2[ix * ny + iy] = matrix1[ix * ny + iy];
	}
}

//---------------------------------------------------------------------------------------

__global__ void transpose_read_row_write_column(int* matrix, int* transpose, int nx, int ny)
{
	int ix = threadIdx.x + blockDim.x * blockIdx.x;
	int iy = threadIdx.y + blockDim.y * blockIdx.y;

	if (ix < nx && iy < ny)
	{
		transpose[ix * ny + iy] = matrix[iy * nx + ix];		// reading matrix in row major format and 
															// writing to transpose in column major format
	}
}


__global__ void transpose_read_column_write_row(int* matrix, int* transpose, int nx, int ny)
{
	int ix = threadIdx.x + blockDim.x * blockIdx.x;
	int iy = threadIdx.y + blockDim.y * blockIdx.y;

	if (ix < nx && iy < ny)
	{
		transpose[iy * nx + ix] = matrix[ix * ny + iy];		// reading matrix in row major format and 
															// writing to transpose in column major format
	}
}

//---------------------------------------------------------------------------------------
// depending upon unroll factor, reduce grid by that much factor
__global__ void transpose_unroll4_row(int* mat, int* transpose, int nx, int ny)
{
	int ix = blockIdx.x * blockDim.x * 4 + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

	int ti = iy * nx + ix;		// row major
	int to = ix * ny + iy;      // column major

	if (ix + 3 * blockDim.x < nx && iy < ny)
	{
		transpose[to] = mat[ti];
		transpose[to + ny * blockDim.x] = mat[ti + blockDim.x];
		transpose[to + ny * 2 * blockDim.x] = mat[ti + 2 * blockDim.x];
		transpose[to + ny * 3 * blockDim.x] = mat[ti + 3 * blockDim.x];
	}
}

__global__ void transpose_unroll4_col(int* mat, int* transpose, int nx, int ny)
{
	int ix = blockIdx.x * blockDim.x * 4 + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

	int ti = iy * nx + ix;
	int to = ix * ny + iy;

	if (ix + 3 * blockDim.x < nx && iy < ny)
	{
		transpose[ti] = mat[to];
		transpose[ti + blockDim.x] = mat[to + blockDim.x * ny];
		transpose[ti + 2 * blockDim.x] = mat[to + 2 * blockDim.x * ny];
		transpose[ti + 3 * blockDim.x] = mat[to + 3 * blockDim.x * ny];
	}
}

//---------------------------------------------------------------------------------------
__global__ void transpose_diagonal_row(int* mat, int* transpose, int nx, int ny)
{
	int blk_x = blockIdx.x;
	int blk_y = (blockIdx.x + blockIdx.y) % gridDim.x;

	int ix = blockIdx.x * blk_x + threadIdx.x;
	int iy = blockIdx.y * blk_y + threadIdx.y;

	if (ix < nx && iy < ny)
	{
		transpose[ix * ny + iy] = mat[iy * nx + ix];
	}
}

//---------------------------------------------------------------------------------------
void (*kernel)(int* matrix1, int* matrix2, int nx, int ny);

int main(int argc, char** argv)
{
	int nx = 1024;
	int ny = 1024;
	unsigned int len = nx * ny;
	int kernel_num = 0;
	char* kernel_name;

	if (argc > 1)
		kernel_num = atoi(argv[1]);

	int block_x = 128;
	int block_y = 8;

	

	// Be careful at launch parameter. Do not copy paster values
	dim3 block(block_x, block_y);
	dim3 grid(nx / block_x, ny/ block_y);
	

	if (kernel_num == 4 || kernel_num == 5)   // unrolled kernel need less no. of thread block
	{
		grid.x = grid.x / 4;
	}
	printf("Kernel launch parameters\n Grid:(%d,%d) and Block:(%d,%d)\n Matrix size:(%d,%d)\n", grid.x, grid.y, block.x, block.y,nx,ny);

	unsigned int matrix_size = len*sizeof(int*);
	int* h_matrix1 = (int*)malloc(matrix_size);
	int* h_matrix2 = (int*)malloc(matrix_size);
	int* h_ref = (int*)malloc(matrix_size);

	memset(h_matrix2, 0, matrix_size);
	memset(h_ref, 0, matrix_size);
	
	initialize_1d_array(h_matrix1, len);

	host_matrix_transpose(h_matrix1, h_matrix2, nx, ny);

	int* d_matrix1, * d_matrix2;
	gpuErrchk(cudaMalloc((void**)&d_matrix1, matrix_size));
	gpuErrchk(cudaMalloc((void**)&d_matrix2, matrix_size));

	gpuErrchk(cudaMemcpy(d_matrix1, h_matrix1, matrix_size, cudaMemcpyHostToDevice));

	// kernal launch
	switch (kernel_num)
	{
	case 0:
		kernel = &copy_matrix_row;
		kernel_name = "copy_matrix_row";
		break;
	case 1:
		kernel = &copy_matrix_column; 
		kernel_name = "copy_matrix_column";
		break;
	case 2:
		kernel = &transpose_read_row_write_column;
		kernel_name = "transpose_read_row_write_column";
		break;
	case 3:
		kernel = &transpose_read_column_write_row;
		kernel_name = "transpose_read_column_write_row";
		break;
	case 4:
		kernel = &transpose_unroll4_row;
		kernel_name = "transpose_unroll4_row";
		break;
	case 5:
		kernel = &transpose_unroll4_col;
		kernel_name = "transpose_read_column_write_row";
		break;
	case 6:
		kernel = &transpose_diagonal_row;
		kernel_name = "transpose_diagonal_row";
		break;
	default:
		printf("Inside default block: Please enter valid integer to select appropriate kernel");
		break;
	}

	printf("Launching Kernel: %s\n", kernel_name);

	kernel << <grid, block >> > (d_matrix1, d_matrix2, nx, ny);
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaMemcpy(h_ref, d_matrix2, matrix_size, cudaMemcpyDeviceToHost));


	printf("Done with device execution\n");
	
	bool status = false;
	if(kernel_num==0 || kernel_num==1)                            // row or column copy
		status = compare_results(h_ref, h_matrix1, len);
	else if(kernel_num >= 2 || kernel_num <=5)					// GPU and CPU transpose result comparsion 
		status = compare_results(h_ref, h_matrix2, len);


	if (status)
		printf("Host and device calculation match\n");
	else
		printf("Host and device calculation does not match\n");

	free(h_matrix1);
	free(h_matrix2);
	gpuErrchk(cudaFree(d_matrix1));
	gpuErrchk(cudaFree(d_matrix2));

	gpuErrchk(cudaDeviceReset());
	return 0;
}