#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
Authors: Patrick Wieschollek, Orazio Gallo, Jinwei Gu, and Jan Kautz

This code is our implementation of the paper:
Farid, H., Adelson, E.H., “Separating reflections and lighting using independent components analysis,” IEEE CVPR, 1999

IMPORTANT!
    This is the best-case scenario with synthetic data outside
    of normal value ranges. This never happens in real-world scenarios.

    Model:

        [y1, y2] = mixing  * [x1, x2]
            Y    = R1*S*R2 * X
                 =    M    * X

    Assumption:
        - X1, X2 are independent, i.e. P(X1X2) = P(X1)P(X2)
        - rank(M) = 2
"""

import cv2
import numpy as np
import argparse


def rgb2gray(x):
    if len(x.shape) == 2:
        return x
    if x.shape[2] == 1:
        return x
    return cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)


def rotmat(theta):
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])


def jointHist(A, B):
    A = A.reshape(1, -1)
    B = B.reshape(1, -1)

    mi = min(A.min(), B.min())
    A, B = A - mi, B - mi

    ma = max(A.max(), B.max())
    A, B = A * 255 / ma, B * 255 / ma

    A = A.astype(np.uint8)
    B = B.astype(np.uint8)

    hist = cv2.calcHist([A, B], [0, 1], None, [256, 256], [0, 256, 0, 256])
    hist = hist / hist.sum()
    hist = np.log(hist + 1)  # increase contrast
    hist -= hist.min()
    hist /= hist.max() / 255.
    return hist


def main(args):

    print(args.a)
    print(args.b)

    Y1 = cv2.imread(args.a).astype(np.float32)
    Y2 = cv2.imread(args.b).astype(np.float32)

    if args.scale is not 1.:
        Y1 = cv2.resize(Y1, (0, 0), fx=args.scale, fy=args.scale)
        Y2 = cv2.resize(Y2, (0, 0), fx=args.scale, fy=args.scale)

    def decompose(Y1, Y2):
        # assumption zero-mean for observations
        Y1 -= Y1.mean()
        Y2 -= Y2.mean()

        # get polar coordinates
        R = Y1**2 + Y2**2
        PHI = np.arctan2(Y2, Y1)

        # step 1: estimate R1
        # -----------------------------------------------
        a = R**2 * np.sin(2 * PHI)
        b = R**2 * np.cos(2 * PHI)

        theta1 = 1. / 2 * np.arctan2(a.sum(), b.sum())                    # eq (6)
        print('theta1', theta1, np.rad2deg(theta1))
        theta1s = theta1 - np.pi / .2

        R1inv = rotmat(theta1).transpose()                                # eq (7)

        # step 2: estimate S
        # -----------------------------------------------
        s1 = ((Y1 * np.cos(theta1) + Y2 * np.sin(theta1))**2).sum()       # eq (8)
        s2 = ((Y1 * np.cos(theta1s) + Y2 * np.sin(theta1s))**2).sum()     # eq (9)
        print('s1', s1, 's2', s2)
        Sinv = np.diag([1. / s1, 1. / s2])                                # eq (10)

        # step 3: estimate R2
        # -----------------------------------------------
        a = R**2 * np.sin(4 * PHI)
        b = R**2 * np.cos(4 * PHI)

        theta2 = 1. / 4 * np.arctan2(a.sum(), b.sum())                    # eq (13)
        print('theta2', theta2, np.rad2deg(theta2))
        R2inv = rotmat(theta2).transpose()                                # eq (14)

        Minv = np.matmul(R2inv, np.matmul(Sinv, R1inv))                   # eq (15)

        return R1inv, Sinv, R2inv, Minv

    def normalize(h):
        h -= float(h.min())
        h *= 255 / h.max()    # eq 17
        return h

    def apply_transform(I1, I2, T):
        # convert each image to a row vector
        Im = np.concatenate([I1.reshape(1, -1),
                            I2.reshape(1, -1)], axis=0)
        Im = np.matmul(T, Im)

        i1 = Im[0, :]
        i2 = Im[1, :]

        i1, i2 = i1 - i1.min(), i2 - i2.min()
        i1, i2 = i1 * 255. / i1.max(), i2 * 255. / i2.max()

        # recover an image and scale to maximal intensity
        I1 = normalize(i1).reshape(I1.shape).clip(0, 255).astype(np.uint8)
        I2 = normalize(i2).reshape(I2.shape).clip(0, 255).astype(np.uint8)

        return I1, I2

    R1inv, Sinv, R2inv, Minv = decompose(Y1, Y2)

    cv2.imwrite('output/decomposed_step0_input_1.png', normalize(Y1))
    cv2.imwrite('output/decomposed_step0_input_2.png', normalize(Y2))
    cv2.imwrite('output/decomposed_step0_hist.png', jointHist(Y1, Y2))

    X1, X2 = apply_transform(Y1, Y2, R1inv)
    cv2.imwrite('output/decomposed_step1_rot1_1.png', X1)
    cv2.imwrite('output/decomposed_step1_rot1_2.png', X2)
    cv2.imwrite('output/decomposed_step1_hist.png', jointHist(X1, X2))

    X1, X2 = apply_transform(Y1, Y2, np.matmul(Sinv, R1inv))
    cv2.imwrite('output/decomposed_step2_scale_1.png', X1)
    cv2.imwrite('output/decomposed_step2_scale_2.png', X2)
    cv2.imwrite('output/decomposed_step2_hist.png', jointHist(X1, X2))

    X1, X2 = apply_transform(Y1, Y2, Minv)
    cv2.imwrite('output/decomposed_step3_1.png', X1)
    cv2.imwrite('output/decomposed_step3_2.png', X2)
    cv2.imwrite('output/decomposed_step3_hist.png', jointHist(X1, X2))

    cv2.imwrite('output/farid99-A.png', X1)
    cv2.imwrite('output/farid99-B.png', X2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--a', help='image 0', type=str)
    parser.add_argument('--b', help='image 1', type=str)
    parser.add_argument('--scale', help='scaling', type=float)
    args = parser.parse_args()
    main(args)
