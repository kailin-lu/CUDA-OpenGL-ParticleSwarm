#include "kernel.h"

__global__ void initPositions(float *positions, curandState *state, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x; 
    if (idx < N) {
        positions[idx + 0] = curand_uniform(state); 
        positions[idx + 1] = curand_uniform(state); 
        positions[idx + 2] = curand_uniform(state); 
    }
}

// Initialize state for random numbers 
__global__ void init_kernel(curandState *state, long seed) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x; 
    curand_init(seed, idx, 0, state);
}


void Kernel::kernelInit(float *positions, int N) {
    // Random seed for gpu 
    curandState *state; 
    cudaMalloc(&state, sizeof(curandState)); 
    init_kernel<<<1,1>>>(state, clock()); 

    int blockSize = 32; 
    int gridSize = (blockSize + N - 1) / N; 

    initPositions<<<gridSize, blockSize>>>(positions, state, N); 
}