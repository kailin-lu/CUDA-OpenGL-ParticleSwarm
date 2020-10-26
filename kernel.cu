#include <iostream>
#include "kernel.h"


// Error handing 
void getCUDAError(char const*msg) {
    if (cudaGetLastError() != cudaSuccess) {
        std::cout << "CUDA ERROR:: " << msg << " " << cudaGetErrorString(cudaGetLastError()) << std::endl;
    }
}


// Calculate positions of particles and map to buffer 
void calcCUDA(cudaGraphicsResource *VBOparticles_CUDA, int NParticles) {
    float *positions;
    cudaGraphicsMapResources(1, &VBOparticles_CUDA, 0); 
    size_t num_bytes;

    // Map buffer to write from CUDA  
    cudaGraphicsResourceGetMappedPointer((void**)&positions, &num_bytes, VBOparticles_CUDA);  
    
    // Initialize random state  
    curandState *state; 
    cudaMalloc(&state, sizeof(curandState)); 
    init_kernel<<<1,1>>>(state, clock()); 

    // Execute kernel
    // dim3 blockSize(16,16,1); 
    // dim3 gridSize; 
    int blockSize = 32; 
    int gridSize = (NParticles + blockSize - 1) / blockSize; 
    createVertices<<<gridSize, blockSize>>>(positions, state, NParticles); 

    // Unmap buffer 
    cudaGraphicsUnmapResources(1, &VBOparticles_CUDA, 0); 
}


// Initialize state for random numbers 
__global__ void init_kernel(curandState *state, long seed) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x; 
    curand_init(seed, idx, 0, state);
}


// Random positions
__global__ void createVertices(float *positions, curandState *state, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x; 
    if (idx < N) {
        positions[3 * idx + 0] = curand_uniform(state); 
        positions[3 * idx + 1] = curand_uniform(state); 
        positions[3 * idx + 2] = curand_uniform(state); 
    }
}

