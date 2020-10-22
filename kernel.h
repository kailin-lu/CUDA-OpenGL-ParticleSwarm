#pragma once 

#include <cuda.h> 
#include <cuda_runtime.h>  
#include <curand_kernel.h>
#include <ctime>

namespace Kernel {
  void kernelInit(float *positions, int N);
}
