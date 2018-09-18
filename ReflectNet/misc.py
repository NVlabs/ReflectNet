"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
Authors: Patrick Wieschollek, Orazio Gallo, Jinwei Gu, and Jan Kautz
"""

import time
import numpy as np
import tensorflow as tf
from contextlib import contextmanager


@contextmanager
def benchmark(name="unnamed context"):
    """timing
    Args:
        name (str): output-name for timing
    Example:
        with meta.benchmark('doing heavy stuff right now tooks'):
            sleep(1)
    """
    elapsed = time.time()
    yield
    elapsed = time.time() - elapsed
    print('[{}] finished in {} ms'.format(name, int(elapsed * 1000)))


def deg2rad(deg):
    pi_on_180 = 0.017453292519943295
    return deg * pi_on_180


def rad2deg(rad):
    pi_on_180 = 0.017453292519943295
    return rad / pi_on_180


def tf_scale(x, rgb=True):
    # for re-shaping final result
    shape = x.get_shape().as_list()
    # batch flatten
    dim = np.prod(shape[1:])
    y = tf.reshape(x, [-1, dim])
    # scaling
    y = y - tf.expand_dims(tf.reduce_min(y, axis=1), axis=-1)
    y = y / (tf.expand_dims(tf.reduce_max(y, axis=1), axis=-1) + 1e-8)
    # btach un-flatten
    y = tf.reshape(y, [-1] + shape[1:])

    if rgb is False:
        y = tf.reduce_mean(y, axis=-1, keep_dims=True)
    return y


def mse(x, y, name=None):
    return tf.reduce_mean(tf.squared_difference(x, y), name=name)


def scale(x, upper=True, lower=True):
    shape = x.get_shape().as_list()
    dim = np.prod(shape[1:])
    y = tf.reshape(x, [-1, dim])
    if lower:
        y = y - tf.expand_dims(tf.reduce_min(y, axis=1), axis=-1)
    if upper:
        y = y / (tf.expand_dims(tf.reduce_max(y, axis=1), axis=-1) + 1e-8)
    y = tf.reshape(y, [-1] + shape[1:])
    return y


class VisualizationTile(object):
    """docstring for VisualizationTile"""

    def __init__(self, batch, rows):
        super(VisualizationTile, self).__init__()
        self.batch = batch
        self.rows = []
        for r in range(rows):
            self.rows.append([])

    def add(self, row, tensors, scale=1):
        for tensor in tensors:
            # tensor = tf.clip_by_value(tensor, 0, 1)
            self.add_single(row, tensor * scale)

    def add_single(self, row, tensor):
        _, h, w, c = tensor.get_shape().as_list()
        assert c in [1, 3], 'tensor should be grayscale or rgb'

        if len(self.rows[row]) > 0:
            _, rh, rw, _ = self.rows[row][0].get_shape().as_list()
            assert (rh == h), 'new image must match height of first image in row'

        if c == 1:
            tensor = tf.image.grayscale_to_rgb(tensor)

        self.rows[row].append(tensor)
        self.rows[row].append(self.separator(h, 5))

    def separator(self, h, w):
        return tf.convert_to_tensor(np.ones((self.batch, h, w, 3), dtype=np.float32) * 255.)

    def visualize(self):
        # get max columns
        row_widths = np.zeros([len(self.rows)]).astype(np.int32)

        for k in range(len(self.rows)):
            row_widths[k] = np.sum([r.get_shape().as_list()[2] for r in self.rows[k]])

        max_width = np.max(row_widths)

        # pad each row
        for k in range(len(self.rows)):
            _, h, w, c = self.rows[k][0].get_shape().as_list()
            self.rows[k].append(self.separator(h, max_width - row_widths[k]))

        row_images = [tf.concat(r, 2) for r in self.rows]
        return tf.cast(tf.clip_by_value(tf.concat(row_images, 1), 0, 255), tf.uint8, name='viz')


class Polarization(object):
    """docstring for Polarization"""

    def __init__(self, kappa=1.474, aoi=None):
        """Init

        Args:
            kappa (float, optional): refraction coefficient for Snell's law
            aoi (None, optional): angle of incident in degree
        """
        super(Polarization, self).__init__()
        self.kappa = kappa
        if aoi is not None:
            self.set_AOI(aoi)

    def set_AOI(self, aoi):
        self.aoi = np.deg2rad(aoi)
        self.aoi_ = self._snells_Law(self.aoi)
        self.R_parallel, self.R_perpendicular = self._reflection_components()

    def _snells_Law(self, theta):
        return np.arcsin(1. / self.kappa * np.sin(theta))

    def _reflection_components(self):

        a, b = np.sin(self.aoi - self.aoi_), np.sin(self.aoi + self.aoi_)
        R_perpendicular = a**2 / b**2
        R_perpendicular = 2. / (1 + R_perpendicular) * R_perpendicular

        a, b = np.tan(self.aoi - self.aoi_), np.tan(self.aoi + self.aoi_)
        R_parallel = a**2 / b**2
        R_parallel = 2. / (1 + R_parallel) * R_parallel

        return R_parallel, R_perpendicular

    def get_alpha(self, phi, phi_perp=0):
        """Compute valid alpha_mask

        Args:
            phi (float): angle of polarization in degree
        """
        assert phi is not None

        # phi_perp = np.deg2rad(phi + 90)
        phi = np.deg2rad(phi)
        phi_perp = np.deg2rad(phi_perp)

        alpha = self.R_perpendicular * np.cos(phi - phi_perp)**2 +\
            self.R_parallel * np.sin(phi - phi_perp)**2

        return alpha

    def mix(self, reflection, transmission, alpha=None, phi=None, phi_perp=None):
        if alpha is None:
            assert phi is not None
            assert phi_perp is not None
            alpha = self.get_alpha(phi, phi_perp)

        Im = alpha[:, :, None] * reflection / 2. +\
            (1 - alpha[:, :, None]) * transmission / 2.
        return Im
