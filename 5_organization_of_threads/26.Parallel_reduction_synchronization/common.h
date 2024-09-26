#pragma once
#include<stdio.h>
#include<stdlib.h>
#include<time.h>

/* helper function to log cuda error code*/
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

//---------------------------------------------------------------------------------------
/* helper function to compare device and host result*/
bool compare_results(int d_result, int h_result)
{
    
     if (d_result == h_result)
     {
         return true;
     }
   
    return false;
}

//---------------------------------------------------------------------------------------
/* helper function to sum array element on host side*/
int sum_1d_array(int* input, const int size)
{
    int result=0;
    for (int i = 0; i < size; ++i)
        result += input[i];

    return result;
}


//---------------------------------------------------------------------------------------
/* helper function to initialize array with random no.*/
void initialize_1d_array(int* input, const int size)
{
    time_t t;
    srand(time(&t));                   // randon no. generator seed

    for (int i = 0; i < size; ++i)                   // assigning random value from 0 - 10 to array
    {
        input[i] = 1;
    }
}