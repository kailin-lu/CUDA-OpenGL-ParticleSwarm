#pragma once 

#include <cuda.h> 
#include <cuda_runtime.h>  
#include <curand_kernel.h>
#include <ctime>

void getCUDAError(char const *msg); 
void calcCUDA(float *devPtr, int NParticles);

__global__ void init_kernel(curandState *state, long seed); 
__global__ void createVertices(float *devPtr, int N); 
__device__ __host__ float func(float x, float y);