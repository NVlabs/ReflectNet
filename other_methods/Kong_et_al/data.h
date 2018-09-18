/*
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
Authors: Patrick Wieschollek, Orazio Gallo, Jinwei Gu, and Jan Kautz
*/

#ifndef CONTAINER_H
#define CONTAINER_H

#include <cuda_runtime.h>

class IDX {
  const int a, b, c, d;
 public:
  IDX(int a_, int b_, int c_, int d_ = 1) : a(a_), b(b_), c(c_), d(d_) {}
  inline int operator()(int a_, int b_, int c_, int d_ = 0) const {
    return a_ * (b * c * d) + b_ * (c * d) + c_ * d + d_;
  }
};

/*
This hides the C-api for CUDa and allows us doing stuff like:

  Data<Dtype> tensor(100);
  tensor.host.malloc();
  tensor.device.malloc();


*/


template<typename Dtype>
class Data;

template<typename Dtype>
class memory_manager {
 public:
  Data<Dtype>* parent;
  Dtype *data;
  virtual void malloc() = 0;
  virtual void free() = 0;
};

template<typename Dtype>
class cpu_memory : public memory_manager<Dtype> {
 public:
  Dtype *data;
  virtual void malloc() {data = new Dtype[memory_manager<Dtype>::parent->len]; }
  virtual void free() {delete[] data; }
};

template<typename Dtype>
class gpu_memory : public memory_manager<Dtype> {
 public:
  Dtype *data;
  virtual void malloc() { cudaMalloc((void**)&data, memory_manager<Dtype>::parent->len * sizeof(Dtype));}
  virtual void free() {cudaFree(data); }
};


template<typename Dtype>
class Data {
 public:
  cpu_memory<Dtype> host;
  gpu_memory<Dtype> device;
  int len;

  Data(int len_) : len(len_) {
    host.parent = this;
    device.parent = this;
  }
  Data() : len(0) {
    host.parent = this;
    device.parent = this;
  }

  void to_host() {
    cudaMemcpy(host.data, device.data, len * sizeof(Dtype), cudaMemcpyDefault);
  }

  void to_device() {
    cudaMemcpy(device.data, host.data, len * sizeof(Dtype), cudaMemcpyDefault);
  }

  Dtype &operator[](int off) {
    return host.data[off];
  }

  const Dtype operator[](int off) const {
    return host.data[off];
  }



};

#endif