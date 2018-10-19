#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
Authors: Patrick Wieschollek, Orazio Gallo, Jinwei Gu, and Jan Kautz
"""

import numpy as np
import cv2


class SampleType:
  Linear = 1
  Blobs = 2


class MapSample(object):
  """docstring for MapSample"""

  def __init__(self, size=(128, 128), blobs=(5, 5), domain=(30, 90),
               rng=np.random, method=SampleType.Blobs, noise=10, exponent=1):
    super(MapSample, self).__init__()
    self.size = size
    self.noise = noise
    self.blobs = blobs
    self.domain = domain
    self.rng = rng
    self.exponent = exponent
    self.method = self._sample_blobs if method == SampleType.Blobs else self._sample_linear

    xx, yy = np.linspace(0., 1., size[1]), np.linspace(0., 1., size[0])
    self.xx, self.yy = np.meshgrid(xx, yy)

  def _sample_blobs(self):
    # ans = self.rng.uniform(low=self.domain[0], high=self.domain[1],
    #                        size=self.blobs).astype(np.float32)
    ans = self.rng.uniform(low=0, high=1,
                           size=self.blobs).astype(np.float32)

    def suni(x):
      return self.rng.uniform(low=0, high=x)
    ans = ans ** self.exponent
    ans -= ans.min()
    ans /= ans.max()
    ans = cv2.resize(ans, (self.size[1], self.size[0]))
    ans = cv2.blur(ans, (self.size[1] // 2, self.size[0] // 2))

    ans -= ans.min()
    ans /= ans.max()
    ans = self.domain[0] + ans * (self.domain[1] - self.domain[0])
    return ans

  def _sample_linear(self):
    """Sample a plane
    """
    def gen_noise():
      return self.rng.uniform(low=0, high=self.noise, size=1)

    p1, p2, p3 = self.rng.uniform(
        low=self.domain[0], high=self.domain[1], size=3)

    p1 = np.array([1, 0, p1])
    p2 = np.array([0, 1, p2])
    p3 = np.array([1, 1, p3])

    v1 = p3 - p1
    v2 = p2 - p1

    cp = np.cross(v1, v2)
    a, b, c = cp
    d = np.dot(cp, p3)

    def height(x, y):
      return (a * x + b * y - d) / -c

    plane = height(self.xx, self.yy)
    plane /= plane.max() / (self.domain[1] -
                            self.rng.uniform(low=0, high=3, size=1))

    return plane

  def sample(self):
    ans = self.method()

    return ans


class LinearSampler(object):
  """docstring for LinearSampler"""

  def __init__(self, shape=(128, 128), domain=(30, 90)):
    super(LinearSampler, self).__init__()
    self.shape = shape
    self.domain = domain

    xx, yy = np.linspace(0., 1., shape[1]), np.linspace(0., 1., shape[0])
    self.xx, self.yy = np.meshgrid(xx, yy)

  def _sample_linear(self):
    """Sample a plane
    """

    p1, p2, p3 = self.rng.uniform(
        low=self.domain[0], high=self.domain[1], size=3)

    p1 = np.array([1, 0, p1])
    p2 = np.array([0, 1, p2])
    p3 = np.array([1, 1, p3])

    v1 = p3 - p1
    v2 = p2 - p1

    cp = np.cross(v1, v2)
    a, b, c = cp
    d = np.dot(cp, p3)

    def height(x, y):
      return (a * x + b * y - d) / -c

    plane = height(self.xx, self.yy)
    plane /= plane.max() / (self.domain[1] -
                            np.random.uniform(low=0, high=3, size=1))

    return plane


class ParabolaSampler(object):
  """docstring for ParabolaSampler"""

  def __init__(self, shape=[128, 128], domain=(30, 90),
               rng=np.random, noise=10, exponent=1):
    super(ParabolaSampler, self).__init__()
    self.shape = (shape[0] * 2, shape[1] * 2)
    self.noise = noise
    self.domain = domain
    self.rng = rng
    self.exponent = exponent

  def sample(self):
    # sample camera position
    xc = self.rng.uniform(low=-80, high=80, size=1)[0]
    yc = self.rng.uniform(low=-50000, high=-5000, size=1)[0]

    xc = -1

    def tt(xc, yc, xp):
      # parabola
      yp = xp**2

      # tangent in (xp, xp**2) is t(x) = f'(xp)x - xp**2 = 2*xp*x - xp**2
      # cf. t(xp) = 2*xp*xp - xp**2 = x**p
      def t(x):
        return 2 * xp * x - yp

      # # find intersecting point (xd, yd) on tangent
      # xd = (-2 * xp**3 + 2 * xp * yc + xc) / (4 * xp**2 + 1)
      theta = np.arctan2(2 * xc * xp + xp**2 - yc, 4 *
                         xc * xp**2 - 4 * xp**3 + xc - xp)
      # flip AOI if too large
      if theta > np.deg2rad(90):
        theta = np.deg2rad(180) - theta
      return theta

    fallback_option = 0

    x0 = self.rng.uniform(low=-800, high=xc + 20, size=1)[0]
    x1 = self.rng.uniform(low=x0 + 200, high=x0 + 301, size=1)[0]

    valid = False
    if np.rad2deg(tt(xc, yc, x0)) > 1:
      valid = True
    if np.rad2deg(tt(xc, yc, (x1 + x0) / 2.)) > 1:
      valid = True
    if np.rad2deg(tt(xc, yc, x1)) > 1:
      valid = True

    if valid is not True:
      fallback_option = 1
    else:
      yc = self.rng.uniform(low=-50000, high=-500000, size=1)[0]

    if fallback_option == 1:
      xc = -1
      # almost linear
      x0 = self.rng.uniform(low=-50, high=-20, size=1)[0]
      x1 = self.rng.uniform(low=-30, high=-19, size=1)[0]
      if x1 < x0:
        x1, x0 = x0, x1

    # print self.shape
    canvas = np.zeros(self.shape, dtype=np.float32)
    for k, xp in enumerate(np.linspace(x0, x1, self.shape[0])):
      aoi = tt(xc, yc, xp)
      canvas[:, k] = aoi

    deg = self.rng.uniform(low=0, high=180, size=1)[0]
    M = cv2.getRotationMatrix2D((self.shape[0] / 2, self.shape[0] / 2), deg, 1)
    canvas = cv2.warpAffine(canvas, M, (self.shape[0], self.shape[1]))

    def center_crop(x, r=128):
      h, w = x.shape[:2]
      return x[h//2-r//2:h//2+r//2, w//2-r//2:w//2+r//2]  # noqa

    canvas = center_crop(canvas)

    return canvas
