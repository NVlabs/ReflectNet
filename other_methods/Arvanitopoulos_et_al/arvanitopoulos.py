#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
Authors: Patrick Wieschollek, Orazio Gallo, Jinwei Gu, and Jan Kautz

This code is our implementation of the paper:
Arvanitopoulos Darginis, N., Achanta, R., Süsstrunk, S., “Single image reflection suppression,” IEEE CVPR, 2017

IMPORTANT! We didn't use this script for the results in the paper.
We used the original MATLAB implementation provided by the authors.
"""

import tensorflow as tf
import cv2
import numpy as np


def make_kernel(a):
    """Transform a 2D array into a convolution kernel"""
    a = np.asarray(a)
    a = a.reshape(list(a.shape) + [1, 1])
    return tf.constant(a, dtype=1)


def simple_conv(x, k):
    """A simplified 2D convolution operation"""
    x = tf.expand_dims(tf.expand_dims(x, 0), -1)
    y = tf.nn.depthwise_conv2d(x, k, [1, 1, 1, 1], padding='SAME')
    return y[0, :, :, 0]


def laplace(x):
    """Compute the 2D laplacian of an array"""
    # laplace_k = make_kernel([[0.5, 1.0, 0.5],
    #                          [1.0, -6., 1.0],
    #                          [0.5, 1.0, 0.5]])
    # TODO: they use a slightly different kernel
    laplace_k = make_kernel([[0.0, 1.0, 0.0],
                             [1.0, -4., 1.0],
                             [0.0, 1.0, 0.0]])

    r, g, b = tf.unstack(x, axis=3)
    r = tf.nn.depthwise_conv2d(tf.expand_dims(r, axis=-1), laplace_k, [1, 1, 1, 1], padding='SAME')
    g = tf.nn.depthwise_conv2d(tf.expand_dims(g, axis=-1), laplace_k, [1, 1, 1, 1], padding='SAME')
    b = tf.nn.depthwise_conv2d(tf.expand_dims(b, axis=-1), laplace_k, [1, 1, 1, 1], padding='SAME')

    return tf.concat([r, g, b], axis=3)


def sobel(x):

    r, g, b = tf.unstack(x, axis=3)

    r = tf.expand_dims(r, axis=-1)
    g = tf.expand_dims(g, axis=-1)
    b = tf.expand_dims(b, axis=-1)

    sobel_x = tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], tf.float32)
    sobel_x_filter = tf.reshape(sobel_x, [3, 3, 1, 1])
    sobel_y_filter = tf.transpose(sobel_x_filter, [1, 0, 2, 3])

    dx = [tf.nn.conv2d(c, sobel_x_filter,
          strides=[1, 1, 1, 1], padding='SAME') for c in [r, g, b]]
    dy = [tf.nn.conv2d(c, sobel_y_filter,
          strides=[1, 1, 1, 1], padding='SAME') for c in [r, g, b]]
    return tf.concat(dx, axis=3), tf.concat(dy, axis=3)


# img = cv2.imread('cvpr.png').astype(np.float32) / 255.
img = cv2.imread('/data/project/supplemental_material/results/barrol/DSC01887.JPG').astype(np.float32) / 255.
img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
img = img[None, :, :, :]


_, h, w, _ = img.shape

var_init = tf.contrib.layers.variance_scaling_initializer()
Y = tf.placeholder(dtype=tf.float32, shape=img.shape)
B = tf.placeholder(dtype=tf.float32, shape=[])

T_op = tf.get_variable('T_op', [1, h, w, 3], initializer=var_init, trainable=True)
H_op = tf.get_variable('H_op', [1, h, w, 3], initializer=var_init, trainable=False)
V_op = tf.get_variable('V_op', [1, h, w, 3], initializer=var_init, trainable=False)


def mse(x, y):
    return tf.reduce_mean(tf.squared_difference(x, y))


def C(H_op, V_op):
    z = H_op * 0
    o = H_op * 0 + 1
    sel = tf.where(tf.greater(tf.abs(H_op) + tf.abs(V_op), z), o, z)
    return tf.reduce_mean(sel)


Tdx, Tdy = sobel(T_op)

# loss = mse(laplace(T_op), laplace(Y)) + l * C(H_op, V_op) + B * (mse(Tdx, H_op) + mse(Tdy, V_op))
loss = mse(laplace(T_op), laplace(Y)) + B * (mse(Tdx, H_op) + mse(Tdy, V_op))

lamb = 0.002
beta0 = 2 * lamb
beta = beta0
beta_max = 10**5

step1 = tf.train.AdamOptimizer(0.01).minimize(loss)


assign_H_op = H_op.assign(Y)
assign_V_op = V_op.assign(Y)
assign_T_op = T_op.assign(Y)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(assign_T_op, {Y: img})

    ii = 0

    # half-quadratic splitting
    while beta <= beta_max:

        # UPDATE G
        Tdx_, Tdy_ = sess.run([Tdx, Tdy], {Y: img, B: beta})
        crit = Tdx_**2 + Tdy_**2

        current_H = sess.run(H_op)
        current_V = sess.run(V_op)

        current_H[crit <= lamb / beta] = 0
        current_V[crit <= lamb / beta] = 0

        current_H[current_H > 0] = Tdx_[current_H > 0]
        current_V[current_V > 0] = Tdy_[current_V > 0]

        sess.run(assign_H_op, {Y: current_H})
        sess.run(assign_V_op, {Y: current_V})

        # update T_op
        _, lamb = sess.run([step1, loss], {Y: img, B: beta})
        print(lamb)

        beta = 2 * beta

        st = sess.run(T_op)[0]
        cv2.imwrite('setp%03i.jpg' % ii, st * 255)
        ii += 1

    st = sess.run(T_op)[0]
    cv2.imwrite('final.jpg', st * 255)
