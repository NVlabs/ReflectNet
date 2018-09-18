#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
Authors: Patrick Wieschollek, Orazio Gallo, Jinwei Gu, and Jan Kautz
"""

"""
evaluate model on given image

prepare data:
---------------------
cd example
dcraw -v -W -g 1 1 -6 *.ARW
cd ..
i0=example/DSC01908.ppm
i45=example/DSC01909.ppm
i90=example/DSC01910.ppm
prefix=bar
scale=0.25
python eval.py --scale ${scale} --i0 ${i0} --i45 ${i45} --i90 ${i90} --out example/ --prefix ${prefix}

"""

import argparse
import numpy as np
import cv2
import os
from tensorpack import *
import time
import glob
from contextlib import contextmanager
import model


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


def main(args):

    assert os.path.isdir(args.out)

    if not os.path.isfile(args.i0):
        ppms = glob.glob(args.out + "/*.ppm")
        ppms = sorted(ppms)
        I0 = cv2.imread(ppms[0]).astype(np.float32) / 255.
        I45 = cv2.imread(ppms[1]).astype(np.float32) / 255.
        I90 = cv2.imread(ppms[2]).astype(np.float32) / 255.
    else:
        assert os.path.isfile(args.i0)
        assert os.path.isfile(args.i45)
        assert os.path.isfile(args.i90)
        I0 = cv2.imread(args.i0).astype(np.float32) / 255.
        I45 = cv2.imread(args.i45).astype(np.float32) / 255.
        I90 = cv2.imread(args.i90).astype(np.float32) / 255.

    if args.scale is not 1.0:
        I0 = cv2.resize(I0, (0, 0), fx=args.scale, fy=args.scale)
        I45 = cv2.resize(I45, (0, 0), fx=args.scale, fy=args.scale)
        I90 = cv2.resize(I90, (0, 0), fx=args.scale, fy=args.scale)

    h, w, _ = I0.shape
    h = (h // 8) * 8
    w = (w // 8) * 8

    I0 = I0[:h, :w, :]
    I45 = I45[:h, :w, :]
    I90 = I90[:h, :w, :]

    global HSHAPE, WSHAPE
    model.HSHAPE, model.WSHAPE = I0.shape[:2]

    I0 = I0[None, :, :, :]
    I45 = I45[None, :, :, :]
    I90 = I90[None, :, :, :]

    pred = OfflinePredictor(PredictConfig(
        model=model.Model(model.HSHAPE, model.WSHAPE),
        session_init=get_model_loader('data/checkpoint'),
        input_names=['I1', 'I2', 'I3'],
        output_names=['est_lt', 'est_lr', 'I_s', 'I_p']))

    with benchmark('reconstruct'):
        estT, estR, i_s, i_p = pred(I0, I45, I90)

    def clamp(x):
        x[x < 0] = 0
        return x

    estT = clamp(estT[0])
    estR = clamp(estR[0])
    i_s = clamp(i_s[0])
    i_p = clamp(i_p[0])

    cv2.imwrite(os.path.join(args.out, 'ours_%s_s%.2f_T.png' %
                             (args.prefix, args.scale)), (estT * 255.).clip(0, 255))
    cv2.imwrite(os.path.join(args.out, 'ours_%s_s%.2f_R.png' %
                             (args.prefix, args.scale)), (estR * 255.).clip(0, 255))

    print(os.path.join(args.out, 'ours_%s_s%.2f_T.png' % (args.prefix, args.scale)))
    print(os.path.join(args.out, 'ours_%s_s%.2f_R.png' % (args.prefix, args.scale)))

    cv2.imwrite(os.path.join(args.out, 'ours_%s_s%.2f_degamma_T.png' % (
        args.prefix, args.scale)), (estT**(1. / 2.2) * 255.).clip(0, 255))
    cv2.imwrite(os.path.join(args.out, 'ours_%s_s%.2f_degamma_R.png' % (
        args.prefix, args.scale)), (estR**(1. / 2.2) * 255.).clip(0, 255))

    cv2.imwrite(os.path.join(args.out, 'input_%s_s%.2f_0.png' %
                             (args.prefix, args.scale)), (I0[0] * 255.))
    cv2.imwrite(os.path.join(args.out, 'input_%s_s%.2f_1.png' %
                             (args.prefix, args.scale)), (I45[0] * 255.))
    cv2.imwrite(os.path.join(args.out, 'input_%s_s%.2f_2.png' %
                             (args.prefix, args.scale)), (I90[0] * 255.))

    print(estT.shape)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scale', help='scaling of inputs',
                        type=float, default=1.0)
    parser.add_argument('--out', help='output dir', type=str, default='/tmp')
    parser.add_argument('--prefix', help='prefix', type=str, required=True)
    parser.add_argument(
        '--i0', help='image at 0 polarization', type=str, default='')
    parser.add_argument(
        '--i45', help='image at 45 polarization', type=str, default='')
    parser.add_argument(
        '--i90', help='image at 90 polarization', type=str, default='')
    args = parser.parse_args()
    main(args)
