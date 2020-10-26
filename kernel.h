#pragma once 

#include <cuda.h> 
#include <cuda_runtime.h>  
#include <curand_kernel.h>
#include <ctime>

void getCUDAError(char const *msg); 
void calcCUDA(cudaGraphicsResource *VBOparticles_CUDA, int NParticles); 

__global__ void init_kernel(curandState *state, long seed); 
__global__ void createVertices(float *positions, curandState *state, int N); 