#include <iostream>
#include "kernel.h"


// Error handing 
void getCUDAError(char const*msg) {
    cudaError_t err = cudaGetLastError(); 
    if (err != cudaSuccess) {
        std::cout << "CUDA ERROR:: " << msg << " ";
        std::cout << cudaGetErrorName(err) << " "; 
        std::cout << cudaGetErrorString(err) << std::endl;
    }
}


// Calculate positions of particles and map to buffer 
void calcCUDA(float *devPtr, int NParticles) {
    int blockSize = 16; 
    int gridSize = (NParticles + blockSize - 1) / blockSize; 
    createVertices<<<gridSize, blockSize>>>(devPtr, NParticles); 
    cudaDeviceSynchronize(); 
}


// Random positions
__global__ void createVertices(float *devPtr, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x; 
    if (idx < N) {
        devPtr[3 * idx + 0] = devPtr[3 * idx + 0] + .01f; 
        devPtr[3 * idx + 1] = devPtr[3 * idx + 0] - .01f; 
        devPtr[3 * idx + 2] = func(devPtr[3 * idx + 0], devPtr[3 * idx + 1]); 
    }
}

__device__ __host__ float func(float x, float y) {
    return pow(x, 2) + pow(y, 2); 
}


// // rastrigin 
// float func(float x, float y) {
//     int A = 10; 
//     int n = 2; 
//     return A * 2 + (pow(x, 2) - A * cos(2 * M_PI * x)) + (pow(y, 2) - A * cos(2 * M_PI * y));
// }
