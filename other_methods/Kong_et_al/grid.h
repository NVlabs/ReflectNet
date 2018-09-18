/*
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
Authors: Patrick Wieschollek, Orazio Gallo, Jinwei Gu, and Jan Kautz
*/

#ifndef GRID_H
#define GRID_H

#include "data.h"

/**
 * @brief Represent grid for belief-propagation
 */
template<typename Dtype>
class Grid {
  int width_, height_, labels_;

 public:
  Data<Dtype> labelCosts;      // size: height * width * labels
  Data<Dtype> smoothnessCosts; // size: labels * labels


  /**
   * @brief create grid width height h width w and l labels
   */
  Grid(int h, int w, int l)
    : height_(h), width_(w), labels_(l) {
    labelCosts.len = h * w * l;
    smoothnessCosts.len = l * l;

    // allocate memory on host and device
    labelCosts.host.malloc();
    smoothnessCosts.host.malloc();

    labelCosts.device.malloc();
    smoothnessCosts.device.malloc();

    // init all values on grid by 0
    std::fill_n(labelCosts.host.data, labelCosts.len, 0);
    std::fill_n(smoothnessCosts.host.data, smoothnessCosts.len, 0);
  }

  inline int width() const { return width_;}
  inline int height() const { return height_;}
  inline int labels() const { return labels_;}

  // hide linear access pattern
  const Dtype data_cost(int h, int w, int l) const {
    return labelCosts[h * width_ * labels_ + w * labels_ + l];
  }

  const Dtype data_cost() const {
    return labelCosts;
  }

  Dtype &data_cost(int h, int w, int l) {
    return labelCosts[h * width_ * labels_ + w * labels_ + l];
  }

  const Dtype smoothness_cost(int a, int b) const {
    return smoothnessCosts[a * labels_ + b];
  }
  Dtype &smoothness_cost(int a, int b) {
    return smoothnessCosts[a * labels_ + b];
  }

};

#endif