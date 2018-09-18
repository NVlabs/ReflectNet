#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
Authors: Patrick Wieschollek, Orazio Gallo, Jinwei Gu, and Jan Kautz
"""

import cv2
import numpy as np
import argparse


def phi_(phi, n1=1.5, n2=1):
    return np.arcsin(n2 / n1 * np.sin(phi))


def R_parl2(phi, phi_):
    return np.tan(phi - phi_)**2 / np.tan(phi + phi_)**2


def R_perp2(phi, phi_):
    return np.sin(phi - phi_)**2 / np.sin(phi + phi_)**2


def T_perp2(phi, phi_):
    return 1 - R_perp2(phi, phi_)


def T_parl2(phi, phi_):
    return 1 - R_parl2(phi, phi_)


def R_parl(phi, phi_):
    return 2. / (1 + R_parl2(phi, phi_)) * R_parl2(phi, phi_)


def R_perp(phi, phi_):
    return 2. / (1 + R_perp2(phi, phi_)) * R_perp2(phi, phi_)


def T_perp(phi, phi_):
    return 1 - R_perp(phi, phi_)


def T_parl(phi, phi_):
    return 1 - R_parl(phi, phi_)


def PE_R2(phi, phi_):
    return np.abs(R_perp2(phi, phi_) - R_parl2(phi, phi_)) / np.abs(R_perp2(phi, phi_) + R_parl2(phi, phi_))


def PE_T2(phi, phi_):
    return np.abs(T_perp2(phi, phi_) - T_parl2(phi, phi_)) / np.abs(T_perp2(phi, phi_) + T_parl2(phi, phi_))


def PE_R(phi, phi_):
    return np.abs(R_perp(phi, phi_) - R_parl(phi, phi_)) / np.abs(R_perp(phi, phi_) + R_parl(phi, phi_))


def PE_T(phi, phi_):
    return np.abs(T_perp(phi, phi_) - T_parl(phi, phi_)) / np.abs(T_perp(phi, phi_) + T_parl(phi, phi_))


def project_interval(phi):
    disp = np.rad2deg(phi)
    disp[disp < -45] = disp[disp < -45] + 90
    disp[disp > 45] = disp[disp > 45] - 90
    phi = np.deg2rad(disp)
    return phi


def phi_perp_func(I1, I2, I3):
    _a = I1 + I3 - 2 * I2
    _b = I1 - I3

    phi_1_minus_phi_perp = 0.5 * np.arctan2(_a, _b)
    phi_1_minus_phi_perp = project_interval(phi_1_minus_phi_perp)

    print(phi_1_minus_phi_perp.shape)

    return np.expand_dims(np.mean(phi_1_minus_phi_perp, axis=2), axis=-1)


def image_prob(Im, bins=256):
    Im = Im.reshape(-1)
    ans = np.histogram(Im, bins=range(bins + 1), density=True)[0]
    return ans / float(ans.sum())


def joint_hist(a, b, bins=256):
    a = a.reshape([-1])
    b = b.reshape([-1])
    ans = np.histogram2d(a, b, range(bins + 1))[0]
    return ans / float(ans.sum())


def MI(a, b, bins=256):
    ab = joint_hist(a, b, bins)
    a = image_prob(a, bins)
    b = image_prob(b, bins)

    a = a.reshape((-1, 1))
    b = b.reshape((1, -1))
    ab_prod = np.matmul(a, b)

    P = ab[ab > 0] * np.log(ab[ab > 0] / ab_prod[ab > 0].astype(np.float32))

    if P.shape == [0, 0]:
        return 0
    return P.sum()


def entropy(a):
    a = image_prob(a)
    return -(a[a > 0] * np.log(a[a > 0])).sum()


def ratio(a, b):
    return MI(a, b) / ((entropy(a) + entropy(b)) / 2.)


def find_AOI(fp, fs, xticks=np.arange(1, 90, 0.5)):
    mis = np.zeros((len(xticks)))
    off = 0
    for aoi in xticks:
        print(aoi, xticks[-1])
        IT, IR = estimate(fp, fs, aoi)
        IT, IR = IT.astype(np.uint8), IR.astype(np.uint8)
        mis[off] = ratio(IT, IR)
        off += 1
    return mis[:off]


def img_otho_extraction(I1, I2, I3):
    phi_1_minus_phi_perp = phi_perp_func(I1, I2, I3)

    # reconstruct parallel and perpendicular intensities
    I_s = (I1 + I3) / 2. + (I1 - I3) / \
        (2. * (np.cos(2 * phi_1_minus_phi_perp)))
    I_p = (I1 + I3) / 2. - (I1 - I3) / \
        (2. * (np.cos(2 * phi_1_minus_phi_perp)))

    return I_s, I_p


def estimate(fp, fs, AOI):
    a = np.deg2rad(AOI)
    b = phi_(a)

    rs = R_perp(a, b)
    rp = R_parl(a, b)

    IT = 2 * rs / (rs - rp) * fp - (2 * rp / (rs - rp)) * fs
    IR = (2 - 2 * rp) / (rs - rp) * fs - ((2 - 2 * rs) / (rs - rp)) * fp

    return IT, IR


def main(img1, img2, img3, T, R):

    xticks = np.arange(10, 80, 1)

    I0 = cv2.imread(img1).astype(np.float32)
    I1 = cv2.imread(img2).astype(np.float32)
    I2 = cv2.imread(img3).astype(np.float32)

    print(I0.shape, I0.min(), I0.max())

    fs, fp = img_otho_extraction(I0, I1, I2)

    rsl = find_AOI(fp, fs, xticks=xticks)

    sol = np.argmin(rsl)

    angle = xticks[sol]

    I0 = cv2.imread(img1).astype(np.float32)
    I1 = cv2.imread(img2).astype(np.float32)
    I2 = cv2.imread(img3).astype(np.float32)

    IT, IR = estimate(fp, fs, angle)
    fs, fp = img_otho_extraction(I0, I1, I2)

    cv2.imwrite(T,
                (IT / 255.).clip(0, 1) * 255)
    cv2.imwrite(R,
                (IR / 255.).clip(0, 1) * 255)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img1', help='image', type=str)
    parser.add_argument('--img2', help='image', type=str)
    parser.add_argument('--img3', help='image', type=str)
    parser.add_argument('--T', help='output T', type=str)
    parser.add_argument('--R', help='output R', type=str)
    args = parser.parse_args()
    main(args.img1, args.img2, args.img3, args.T, args.R)
