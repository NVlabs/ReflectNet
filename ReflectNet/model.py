#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
Authors: Patrick Wieschollek, Orazio Gallo, Jinwei Gu, and Jan Kautz
"""

from tensorpack import *
import tensorflow as tf
from misc import *


TOTAL_BATCH_SIZE = 32
BATCH_SIZE = 32
HSHAPE = 128
HSHAPE = 1000
WSHAPE = 128
WSHAPE = 1500
CHANNELS = 3
DEBUG = True


@layer_register(use_scope=None, log_shape=False)
def INReLU(x, name=None):
    x = InstanceNorm('in', x)
    x = tf.nn.relu(x, name=name)
    return x


def phi_perp_func(I1, I2, I3):
    _a = I1 + I3 - 2 * I2
    _b = I1 - I3

    phi_1_minus_phi_perp = 0.5 * tf.atan2(_a, _b)

    # we assume is within [-45, 45]
    disp = rad2deg(phi_1_minus_phi_perp)
    disp = tf.where(disp < -45, x=disp + 90, y=disp)
    disp = tf.where(disp > 45, x=disp - 90, y=disp)
    phi_1_minus_phi_perp = deg2rad(disp)

    return tf.expand_dims(tf.reduce_mean(phi_1_minus_phi_perp, axis=3), axis=-1)


def img_otho_extraction(I1, I2, I3):
    phi_1_minus_phi_perp = phi_perp_func(I1, I2, I3)
    phi_1_minus_phi_perp = tf.identity(phi_1_minus_phi_perp, name='phi_1_minus_phi_perp')

    # reconstruct parallel and perpendicular intensities
    I_s = (I1 + I3) / 2. + (I1 - I3) / \
        (2. * (tf.cos(2 * phi_1_minus_phi_perp)))
    I_p = (I1 + I3) / 2. - (I1 - I3) / \
        (2. * (tf.cos(2 * phi_1_minus_phi_perp)))

    return I_s, I_p


class Model(ModelDesc):

    def __init__(self, h=HSHAPE, w=WSHAPE):
        self.HSHAPE = h
        self.WSHAPE = w

    def _get_inputs(self):
        return [InputDesc(tf.float32, (None, self.HSHAPE, self.WSHAPE, CHANNELS), 'I1'),
                InputDesc(tf.float32, (None, self.HSHAPE, self.WSHAPE, CHANNELS), 'I2'),
                InputDesc(tf.float32, (None, self.HSHAPE, self.WSHAPE, CHANNELS), 'I3'),
                InputDesc(tf.float32, (None, self.HSHAPE, self.WSHAPE, 1), 'a1'),
                InputDesc(tf.float32, (None, self.HSHAPE, self.WSHAPE, 1), 'a2'),
                InputDesc(tf.float32, (None, self.HSHAPE, self.WSHAPE, 1), 'a3'),
                InputDesc(tf.float32, (None, self.HSHAPE, self.WSHAPE, CHANNELS), 'lb'),
                InputDesc(tf.float32, (None, self.HSHAPE, self.WSHAPE, CHANNELS), 'lr'),
                InputDesc(tf.float32, (None, self.HSHAPE, self.WSHAPE, 1), 'aoi')]

    def _network(self, I_s, I_p, I1, I2, I3):

        observations = tf.concat([I_s, I_p, I1, I2, I3], axis=3)
        observations = observations * 2. - 1.

        def block_down(net, nf, name, stride=2):
            with tf.variable_scope(name):
                skip = Conv2D('conv1', net, nf, stride=stride, nl=tf.identity)
                net = INReLU('inrelu', skip)
                net = Conv2D('conv2', net, nf)
                net = Conv2D('conv3', net, nf)
                net = tf.concat([net, skip], axis=3)
                net = Conv2D('conv4', net, nf, kernel_shape=1)
                return net

        def block_up(net, uskip, nf, name, stride=2):
            with tf.variable_scope(name):
                skip = Deconv2D('conv1', net, nf, kernel_shape=4, stride=stride, nl=tf.identity)
                net = INReLU('inrelu', skip)
                net = tf.concat([net, uskip], axis=3)
                net = Conv2D('conv2', net, nf)
                net = Conv2D('conv3', net, nf)
                net = tf.concat([net, skip], axis=3)
                net = Conv2D('conv4', net, nf, kernel_shape=1)

                return net

        with argscope([Conv2D, Deconv2D], nl=INReLU, kernel_shape=3, stride=1):
            net = observations
            out0 = Conv2D('conv0', net, 15)
            out1 = block_down(out0, 32, 'block1')
            out1b = block_down(out1, 32, 'block1b', stride=1)
            out2 = block_down(out1b, 64, 'block2')
            out2b = block_down(out2, 64, 'block2b', stride=1)
            net = block_down(out2b, 128, 'block3')
            net = block_down(net, 128, 'block3b', stride=1)

            net = block_up(net, out2b, 64, 'block4')
            net = block_up(net, out2, 64, 'block4b', stride=1)
            net = block_up(net, out1b, 32, 'block5')
            net = block_up(net, out1, 32, 'block5b', stride=1)
            net = block_up(net, out0, 16, 'block6')

            tf.identity(net, name='feature001')
            with tf.variable_scope('epilog'):
                net = Conv2D('deconv_1', net, 16)
                net = tf.concat([net, observations], axis=3)
                net = Conv2D('deconv_2', net, 16)
                net = Conv2D('deconv_0', net, 8, kernel_shape=3,
                             stride=1, nl=tf.identity)

            mask_t = tf.expand_dims(tf.sigmoid(net[:, :, :, -1]), axis=-1)
            mask_r = tf.expand_dims(tf.sigmoid(net[:, :, :, -2]), axis=-1)

            pre_t = (tf.tanh(net[:, :, :, :3]) + 1.) / 2.
            pre_r = (tf.tanh(net[:, :, :, 3:6]) + 1.) / 2.

            tf.identity(mask_r, name='mask_r')
            tf.identity(mask_t, name='mask_t')
            tf.identity(pre_r, name='pre_r')
            tf.identity(pre_t, name='pre_t')

            est_lr = mask_r * pre_r + (1 - mask_r) * I_s
            est_lt = mask_t * pre_t + (1 - mask_t) * I_p

        return est_lt, est_lr

    def _visualize(self, I_s, I_p, I1, I2, I3, est_lt, est_lr, gt_lt, gt_lr):
        # gamma correction
        p = 1. / 2.2

        # visualize
        with tf.name_scope('viz'):

            VT = VisualizationTile(BATCH_SIZE, 3)
            VT.add(0, [I1**p, I2**p, I3**p], scale=255)
            VT.add(1, [I_s**p, I_p**p], scale=255)
            VT.add(2, [est_lr**p, est_lt**p, gt_lr**p, gt_lt**p], scale=255)

            tf.summary.image('v_all', VT.visualize(),
                             max_outputs=max(30, BATCH_SIZE))

    def _build_graph(self, inputs):
        # inputs are given within [0, 1]
        I1, I2, I3, alpha1, alpha2, alpha3,\
            gt_lr, gt_lt,\
            AOI_sample = inputs

        AOI_sample = deg2rad(AOI_sample)

        # project onto canonical view
        I_s, I_p = img_otho_extraction(I1, I2, I3)
        I_s = tf.identity(I_s, name='I_s')
        I_p = tf.identity(I_p, name='I_p')

        # estimate layers
        est_lt, est_lr = self._network(I_s, I_p, I1, I2, I3)

        # cost and loss
        cost_lr = mse(est_lr, gt_lr, name='cost_lr')
        cost_lt = mse(est_lt, gt_lt, name='cost_lt')
        self.cost = tf.add(cost_lr, cost_lt, name='total_costs')
        summary.add_moving_summary(cost_lr, cost_lt, self.cost)

        est_lt = tf.identity(est_lt, name='est_lt')
        est_lr = tf.identity(est_lr, name='est_lr')

        if get_current_tower_context().is_training:
            self._visualize(I_s, I_p, I1, I2, I3, est_lt, est_lr, gt_lt, gt_lr)

    def _get_optimizer(self):
        lr = symbolic_functions.get_scalar_var(
            'learning_rate', 5e-3, summary=True)
        return tf.train.AdamOptimizer(lr)


if __name__ == '__main__':
    pass
