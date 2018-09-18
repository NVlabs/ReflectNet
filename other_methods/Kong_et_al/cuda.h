/*
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
Authors: Patrick Wieschollek, Orazio Gallo, Jinwei Gu, and Jan Kautz
*/

#ifndef CUDA_H
#define CUDA_H

#include <cuda_runtime.h>

namespace cuda {

// just to remove lengthly idx computations
__device__ inline unsigned int globalThreadIndex() {return threadIdx.x + blockIdx.x * blockDim.x;}
__device__ inline unsigned int globalThreadCount() {return blockDim.x * gridDim.x;}
__device__ inline unsigned int globalBlockCount() {return gridDim.x;}
__device__ inline unsigned int localThreadIndex() {return threadIdx.x;}
__device__ inline unsigned int localThreadCount() {return blockDim.x;}
__device__ inline unsigned int globalBlockIndex() {return blockIdx.x;}
__device__ inline void synchronize() {__syncthreads(); }

template<typename A, typename B>
A div_floor(A len, B threads) {
  return (len + threads - 1) / threads;
}

template<typename Kernel>
__global__ void launch(const Kernel k) {
  k.cuda();
}

class kernel {
 public:
  virtual __device__ void cuda() const = 0;
  virtual void operator()() = 0;

  cudaError_t device_synchronize() {
    return cudaDeviceSynchronize();
  }
};
};

#endif