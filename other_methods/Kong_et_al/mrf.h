/*
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
Authors: Patrick Wieschollek, Orazio Gallo, Jinwei Gu, and Jan Kautz
*/

#ifndef MRF_H
#define MRF_H

#include "grid.h"
#include "cuda.h"

enum DIRECTION {LEFT, RIGHT, UP, DOWN, END};


namespace CUDA_MRF {

/**
 * @brief gather all incoming data
 */
template<typename Dtype>
class collect_msg : public ::cuda::kernel {
 public:

  unsigned int height;
  unsigned int width;
  unsigned int labels;
  DIRECTION dir;

  Dtype max_value;

  Dtype *smoothness_cost;
  Dtype *data_cost;
  Dtype *msg_buf;
  Dtype *msg_updates;

  virtual __device__ void cuda() const {

    extern __shared__ float s_shm[];
    Dtype* s_smoothness_cost = (Dtype*) &s_shm[0];

    if (threadIdx.x < labels)
      if (threadIdx.y < labels)
        s_smoothness_cost[threadIdx.x * labels + threadIdx.y] = smoothness_cost[threadIdx.x * labels + threadIdx.y];
    __syncthreads();

    for (int h = blockIdx.x * blockDim.x + threadIdx.x; h < height; h += blockDim.x * gridDim.x) {
      for (int w = blockIdx.y * blockDim.y + threadIdx.y; w < width; w += blockDim.y * gridDim.y) {
        // aggregate and find min
        Dtype s = 0;
        for (int l = 0; l < labels; ++l) {
          // Dtype sum_val = 0;
          Dtype min_val = max_value;

          for (int l_in = 0; l_in < labels; ++l_in) {
            Dtype prob = 0;
            prob += s_smoothness_cost[l * labels + l_in];
            prob += data_cost[h * width * labels + w * labels + l_in];

            if (dir != LEFT)  prob += msg_buf[LEFT * height * width * labels  + h * width * labels + w * labels + l_in];
            if (dir != RIGHT) prob += msg_buf[RIGHT * height * width * labels + h * width * labels + w * labels + l_in];
            if (dir != UP)    prob += msg_buf[UP * height * width * labels    + h * width * labels + w * labels + l_in];
            if (dir != DOWN)  prob += msg_buf[DOWN * height * width * labels  + h * width * labels + w * labels + l_in];


            min_val = (min_val < prob) ? min_val : prob;
          }
          msg_updates[h * width * labels + w * labels + l] = min_val;
          s += min_val;
        }
        s /= labels;
        for (int l = 0; l < labels; ++l) {
          msg_updates[h * width * labels + w * labels + l] -= s;
        }

      }
    }
  }

  enum { THREADS = 32 };

  virtual void operator()() {
    dim3 threads(THREADS, THREADS);
    dim3 grid(cuda::div_floor(height, THREADS), cuda::div_floor(width, THREADS));
    int shm = (labels * labels) * sizeof(Dtype);

    cuda::launch <<< grid, threads, shm>>>(*this);
  }

};


/**
 * @brief distribute all gather data to neighboring nodes
 */
template<typename Dtype>
class send_msg : public ::cuda::kernel {
 public:
  unsigned int height;
  unsigned int width;
  unsigned int labels;
  DIRECTION dir;



  Dtype *msg_buf;
  Dtype *msg_updates;

  virtual __device__ void cuda() const {

    for (int h = blockIdx.x * blockDim.x + threadIdx.x; h < height; h += blockDim.x * gridDim.x) {
      for (int w = blockIdx.y * blockDim.y + threadIdx.y; w < width; w += blockDim.y * gridDim.y) {
        for (int l = 0; l < labels; ++l) {

          Dtype new_val = msg_updates[h * width * labels + w * labels + l];

          if ((dir == LEFT) )   msg_buf[RIGHT * height * width * labels  + h * width * labels + (w - 1)*labels + l] = new_val;
          if ((dir == RIGHT) )  msg_buf[LEFT * height * width * labels   + h * width * labels + (w + 1)*labels + l] = new_val;
          if ((dir == UP) )     msg_buf[DOWN * height * width * labels   + (h - 1)*width * labels + w * labels + l] = new_val;
          if ((dir == DOWN) )   msg_buf[UP * height * width * labels     + (h + 1)*width * labels + w * labels + l] = new_val;
        }
      }
    }
  }

  enum { THREADS = 32 };

  virtual void operator()() {
    dim3 threads(THREADS, THREADS);
    dim3 grid(cuda::div_floor(height, THREADS), cuda::div_floor(width, THREADS));

    cuda::launch <<< grid, threads>>>(*this);
  }

};


};


template<typename Dtype>
class MRF {
  const int DIRS = 4;
 protected:
  Data<Dtype> msg_buf;
  Data<Dtype> msg_updates;

 public:
  Data<unsigned int> best_label;
  Grid<Dtype> *grid;

  MRF(Grid<Dtype> *g) : grid(g) {
    const int height = g->height();
    const int width = g->width();
    const int labels = g->labels();

    msg_buf.len = DIRS * height * width * labels;
    best_label.len = height * width;
    msg_updates.len = height * width * labels;

    msg_buf.host.malloc();
    best_label.host.malloc();
    msg_updates.host.malloc();

    msg_buf.device.malloc();
    best_label.device.malloc();
    msg_updates.device.malloc();

    std::fill_n(msg_buf.host.data, 0, DIRS * height * width * labels );
    std::fill_n(best_label.host.data, 0, height * width);
  }

  unsigned int label(int h, int w) const {
    return best_label[h * grid->width() + w];
  }

  bool valid(int h, int w) {
    const int height = grid->height();
    const int width = grid->width();
    return (h >= 0) && (w >= 0) && (h < height) && (w < width);
  }

  void propagate() {

    const int height = grid->height();
    const int width = grid->width();
    const int labels = grid->labels();

    grid->labelCosts.to_device();
    grid->smoothnessCosts.to_device();
    msg_buf.to_device();
    best_label.to_device();
    msg_updates.to_device();

    CUDA_MRF::collect_msg<Dtype> collect_func;
    collect_func.height = height;
    collect_func.width = width;
    collect_func.labels = grid->labels();
    collect_func.max_value = std::numeric_limits<Dtype>::max();
    collect_func.msg_buf = msg_buf.device.data;
    collect_func.msg_updates = msg_updates.device.data;
    collect_func.data_cost = grid->labelCosts.device.data;
    collect_func.smoothness_cost = grid->smoothnessCosts.device.data;

    CUDA_MRF::send_msg<Dtype> send_func;
    send_func.height = height;
    send_func.width = width;
    send_func.labels = grid->labels();
    send_func.msg_buf = msg_buf.device.data;
    send_func.msg_updates = msg_updates.device.data;

    // belief propagation all directions LEF, RIGHT, UP, DOWN
    for ( int cur_dir = LEFT; cur_dir != END; cur_dir++ ) {

      collect_func.dir = static_cast<DIRECTION>(cur_dir);
      send_func.dir = static_cast<DIRECTION>(cur_dir);

      collect_func();
      collect_func.device_synchronize();

      send_func();
      send_func.device_synchronize();
    }

    grid->labelCosts.to_host();
    grid->smoothnessCosts.to_host();
    msg_buf.to_host();
    best_label.to_host();
    msg_updates.to_host();
  }

  /**
   * @brief kind of loss function (maximum a posteriori)
   */
  Dtype map() {
    const int height = grid->height();
    const int width = grid->width();
    const int labels = grid->labels();


    IDX id(DIRS, height, width, labels);

    for (int i = 0; i < height * width; ++i) {
      Dtype min_val = std::numeric_limits<Dtype>::max();
      const int w = i % width;
      const int h = (i - w) / width;

      for (int l = 0; l < labels; ++l) {

        Dtype cost = 0;

        cost += msg_buf[id(LEFT, h, w, l)];
        cost += msg_buf[id(RIGHT, h, w, l)];
        cost += msg_buf[id(UP, h, w, l)];
        cost += msg_buf[id(DOWN, h, w, l)];

        cost += grid->data_cost(h, w, l);

        if (cost < min_val) {
          min_val = cost;
          best_label.host.data[i] = l;
        }
      }
    }

    Dtype energy = 0;
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        const int cur_label = best_label.host.data[h * width + w];

        energy += grid->data_cost(h, w, cur_label);

        if (w - 1 >= 0)     energy += grid->smoothness_cost(cur_label, best_label.host.data[h * width + w - 1]);
        if (w + 1 < width)  energy += grid->smoothness_cost(cur_label, best_label.host.data[h * width + w + 1]);
        if (h - 1 >= 0)     energy += grid->smoothness_cost(cur_label, best_label.host.data[(h - 1) * width + w]);
        if (h + 1 < height) energy += grid->smoothness_cost(cur_label, best_label.host.data[(h + 1) * width + w]);
      }
    }


    return energy;

  }

};

#endif