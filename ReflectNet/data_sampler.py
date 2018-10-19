#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
Authors: Patrick Wieschollek, Orazio Gallo, Jinwei Gu, and Jan Kautz
"""

from tensorpack import *
import argparse
import cv2
import numpy as np
from simulator import MapSample, SampleType
from simulator import ParabolaSample


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


class ImageDecode(MapDataComponent):
    """Decode JPEG buffer to uint8 image array
    """

    def __init__(self, ds, mode='.jpg', dtype=np.uint8, index=0):
        def func(im_data):
            img = cv2.imdecode(np.asarray(bytearray(im_data), dtype=dtype), cv2.IMREAD_COLOR)
            return img[:, :, ::-1]
        super(ImageDecode, self).__init__(ds, func, index=index)


class RejectImageGradient(MapDataComponent):
    """Decode JPEG buffer to uint8 image array
    """

    def __init__(self, ds, mode='.jpg', dtype=np.uint8, index=0):
        def func(img):
            dx, dy, _ = np.gradient(img)
            dx = np.sum((np.sign(np.abs(dx) - 0.05) + 1.) / 2.)
            dy = np.sum((np.sign(np.abs(dy) - 0.05) + 1.) / 2.)
            ps = img.shape[0] * img.shape[1]
            thresh = 1.9
            if (dx / ps < thresh) or (dy / ps < thresh):
                # cv2.imshow('reject', img)
                return None
            else:
                return img

        super(RejectImageGradient, self).__init__(ds, func, index=index)


class GenerateReflection(RNGDataFlow):
    """Simulate observations with reflection given image_stream

    Attributes:
        ds_imagestream (TYPE): stream of image pairs
        ds_aoistream (TYPE): stream of random aoi
        kappa (float): refraction index for Snell's law
    """

    def __init__(self, ds_imagestream):
        self.ds_imagestream = ds_imagestream
        self.kappa = 1.474
        self.jiggle = 2

        self.deformer1 = imgaug.GaussianDeform([(0.5, 0.5), (0.2, 0.8), (0.8, 0.8), (0.8, 0.2)],
                                               (128, 128), 0.1, 3)
        self.deformer2 = imgaug.GaussianDeform([(0.5, 0.5), (0.2, 0.8), (0.8, 0.8), (0.8, 0.2)],
                                               (128, 128), 0.05, 3)
        self.deformer3 = imgaug.GaussianDeform([(0.5, 0.5), (0.2, 0.8), (0.8, 0.8), (0.8, 0.2)],
                                               (128, 128), 0.025, 3)

    def reset_state(self):
        self.ds_imagestream.reset_state()
        super(GenerateReflection, self).reset_state()

    def size(self):
        return self.ds_imagestream.size()

    def get_data(self):

        def STEP_random_illumination(T, R):
            beta_sample = self.rng.uniform(1, 2.8)
            T /= beta_sample
            R *= beta_sample
            return T, R

        def STEP_edgeaware(T, R):
            RR = R.copy()
            TT = T.copy()
            beta_sample = self.rng.uniform(1, 2.8)
            RR = RR * beta_sample
            I0 = TT + RR
            try:
                if I0[I0 > max(1, RR.max())].shape[0] == 0:
                    return TT, RR
                m = I0[I0 > max(1, RR.max())].mean()
            except Exception:
                return TT, RR

            RR = RR - 1.3 * (m - 1)
            RR = RR.clip(0, 1)

            I0 = RR + TT
            I0 = I0.clip(0, 1)

            RR = I0 - TT

            return TT, RR

        def STEP_nonrigid(R):
            choice = self.rng.randint(0, 4, 1, int)[0]

            if choice == 1:
                r1 = self.deformer1._augment(reflection.copy(), self.deformer1._get_augment_params(reflection))
                r2 = self.deformer1._augment(reflection.copy(), self.deformer1._get_augment_params(reflection))
                r3 = self.deformer1._augment(reflection.copy(), self.deformer1._get_augment_params(reflection))
            elif choice == 2:
                r1 = self.deformer2._augment(reflection.copy(), self.deformer2._get_augment_params(reflection))
                r2 = self.deformer2._augment(reflection.copy(), self.deformer2._get_augment_params(reflection))
                r3 = self.deformer2._augment(reflection.copy(), self.deformer2._get_augment_params(reflection))
            elif choice == 3:
                r1 = self.deformer3._augment(reflection.copy(), self.deformer3._get_augment_params(reflection))
                r2 = self.deformer3._augment(reflection.copy(), self.deformer3._get_augment_params(reflection))
                r3 = self.deformer3._augment(reflection.copy(), self.deformer3._get_augment_params(reflection))
            else:
                r1 = reflection.copy()
                r2 = reflection.copy()
                r3 = reflection.copy()

            return r1, r2, r3

        AOI_sampler_linear = None
        AOI_sampler_parabola = None
        PHIPERP_sampler_linear = None
        PHIPERP_sampler_parabola = None
        P = Polarization()

        for img in self.ds_imagestream.get_data():

            # get random patches from PLACE2 dataset
            reflection = img[0].astype(np.float32) / 255.
            transmission = img[1].astype(np.float32) / 255.

            reflection = reflection ** 2.2
            transmission = transmission ** 2.2

            h, w = reflection.shape[:2]

            # sample random values
            if AOI_sampler_linear is None:
                AOI_sampler_linear = MapSample(size=[h, w], domain=[10, 70],
                                               method=SampleType.Linear)
                AOI_sampler_parabola = ParabolaSample(domain=[-45, 45])
                PHIPERP_sampler_linear = MapSample(size=[h, w], domain=[10, 70],
                                                   method=SampleType.Linear)
                PHIPERP_sampler_parabola = ParabolaSample(domain=[-45, 45])

            if self.rng.randint(0, 2, 1, int)[0] == 1:
                AOI_sample = AOI_sampler_linear.sample()
            else:
                AOI_sample = AOI_sampler_parabola.sample()

            if self.rng.randint(0, 2, 1, int)[0] == 1:
                PHIPERP_sample = PHIPERP_sampler_linear.sample()
            else:
                PHIPERP_sample = PHIPERP_sampler_parabola.sample()

            # setup polarization model and get alpha masks
            P.set_AOI(AOI_sample)
            alpha1 = P.get_alpha(0 + self.rng.uniform(-4, 4), PHIPERP_sample)
            alpha2 = P.get_alpha(45 + self.rng.uniform(-4, 4), PHIPERP_sample)
            alpha3 = P.get_alpha(90 + self.rng.uniform(-4, 4), PHIPERP_sample)

            T = transmission.copy()
            R = reflection.copy()

            # rejection sampling, too dark
            if transmission.sum() < 1000 or reflection.sum() < 1000:
                continue

            if self.rng.randint(0, 2, 1, int)[0] == 1:
                T, R = STEP_edgeaware(T, R)
            else:
                T, R = STEP_random_illumination(T, R)

            r1, r2, r3 = STEP_nonrigid(R)

            # rejection sampling, too dark
            if R.sum() < 300 or T.sum() < 300:
                continue

            # create observations by mixing transmission and reflection
            I1 = P.mix(r1, T, alpha=alpha1)
            I2 = P.mix(r2, T, alpha=alpha2)
            I3 = P.mix(r3, T, alpha=alpha3)

            # rescale back avoid to dark
            mm = max([I1.max(), I2.max(), I3.max()])
            I1 /= mm
            I2 /= mm
            I3 /= mm

            I1 *= min([T.max(), R.max()])
            I2 *= min([T.max(), R.max()])
            I3 *= min([T.max(), R.max()])

            AOI_sample = AOI_sample[:, :, None]
            PHIPERP_sample = PHIPERP_sample[:, :, None]

            alpha1 = alpha1[:, :, None]
            alpha2 = alpha2[:, :, None]
            alpha3 = alpha3[:, :, None]

            yield [I1, I2, I3,
                   alpha1, alpha2, alpha3,
                   transmission, reflection,
                   AOI_sample]


def get_data(lmdb, batch_size=None, prefetch=True, shuffle=True):
    augmentors = [imgaug.RandomCrop(128),
                  imgaug.Flip(horiz=True)]

    ds_0 = LMDBDataPoint(lmdb, shuffle=shuffle)
    ds_0 = PrefetchData(ds_0, 100, 1)
    ds_0 = ImageDecode(ds_0, index=0)
    ds_0 = AugmentImageComponent(ds_0, augmentors, index=0, copy=True)
    ds_0 = RejectImageGradient(ds_0, index=0)

    ds_1 = LMDBDataPoint(lmdb, shuffle=shuffle)
    ds_1 = PrefetchData(ds_1, 100, 1)
    ds_1 = ImageDecode(ds_1, index=0)
    ds_1 = AugmentImageComponent(ds_1, augmentors, index=0, copy=True)
    ds_1 = RejectImageGradient(ds_1, index=0)

    ds_img = JoinData([ds_0, ds_1])

    ds = GenerateReflection(ds_img)
    if batch_size is not None:
        ds = BatchData(ds, batch_size)
    if prefetch:
        ds = PrefetchDataZMQ(ds, nr_proc=6)
    return ds


def phi_perp_func(I1, I2, I3):
    _a = I1 + I3 - 2 * I2
    _b = I1 - I3

    phi_1_minus_phi_perp = 0.5 * np.arctan2(_a, _b)

    # # we assume is within [-45, 45]
    disp = np.rad2deg(phi_1_minus_phi_perp)
    disp[disp <= -45] = disp[disp <= -45] + 90
    disp[disp > 45] = disp[disp > 45] - 90
    phi_1_minus_phi_perp = np.deg2rad(disp)

    return np.expand_dims(np.mean(phi_1_minus_phi_perp, axis=2), axis=-1)


def img_otho_extraction(I1, I2, I3):
    phi_1_minus_phi_perp = phi_perp_func(I1, I2, I3) * 0

    # reconstruct parallel and perpendicular intensities
    I_s = (I1 + I3) / 2. + (I1 - I3) / (2. * (np.cos(2 * phi_1_minus_phi_perp)))
    I_p = (I1 + I3) / 2. - (I1 - I3) / (2. * (np.cos(2 * phi_1_minus_phi_perp)))

    return I_s, I_p


def get_alphas(AOI, AOIp_, phi_perp):
    """Given AngleOfIncidence and orientation of phi_perp return alpha masks

    Args:
        AOI (TYPE): Description
        AOIp_ (TYPE): Description
        phi_perp (TYPE): Description

    Returns:
        TYPE: Description
    """
    # compute R_perpendicular
    a = np.sin(AOI - AOIp_)
    b = np.sin(AOI + AOIp_)
    R_perpendicular = a**2 / (b**2 + 1e-8)

    # compute R_parallel
    a = np.tan(AOI - AOIp_)
    b = np.tan(AOI + AOIp_)
    R_parallel = a**2 / (b**2 + 1e-8)

    phi = np.deg2rad(0)
    alpha1 = R_perpendicular * np.cos(phi - phi_perp)**2 + R_parallel * np.sin(phi - phi_perp)**2
    phi = np.deg2rad(45)
    alpha2 = R_perpendicular * np.cos(phi - phi_perp)**2 + R_parallel * np.sin(phi - phi_perp)**2
    phi = np.deg2rad(90)
    alpha3 = R_perpendicular * np.cos(phi - phi_perp)**2 + R_parallel * np.sin(phi - phi_perp)**2

    return alpha1, alpha2, alpha3


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lmdb', help='path to database (to be written)',
                        default='/scratch_shared/wieschol/train_large_places365standard.lmdb')
    parser.add_argument('--debug', action='store_true',
                        help='just show the images')
    args = parser.parse_args()

    ds = get_data(args.lmdb, batch_size=None, prefetch=False)
    ds.reset_state()

    import matplotlib.pyplot as plt
    Blues = plt.get_cmap('viridis')
    sep = np.ones((128, 20, 3)) * 255

    ii = 0

    for I1, I2, I3, a1, a2, a3, transmission, reflection, aoi in ds.get_data():

        a1 = np.dstack([a1] * 3)
        a3 = np.dstack([a3] * 3)
        aoi = np.dstack([aoi] * 3)
        aoi -= aoi.min()
        aoi /= aoi.max()

        I_s, I_p = img_otho_extraction(I1, I2, I3)

        def reconst_lb(Ii, ai, Ij, aj):
            return -2 * (Ii * aj - Ij * ai) / (ai - aj + 1e-10)

        def reconst_lr(Ii, ai, Ij, aj):
            return -2 * (Ii * aj - Ij * ai - Ii + Ij) / (ai - aj + 1e-10)

        rlb = reconst_lb(I1, a1, I3, a3)
        rlr = reconst_lr(I1, a1, I3, a3)

        # img = np.concatenate([I1, I2, I3, sep,
        #                       I_s, I_p, sep,
        #                       transmission, reflection, sep,
        #                       ],
        #                      axis=1)[:, :, ::-1]

        img = np.concatenate([I1, I2, I3, sep, transmission, reflection, aoi],
                             axis=1)[:, :, ::-1]

        # print 'reflection', reflection.sum()
        # print 'transmission', transmission.sum()

        # img = cv2.resize(img, (0, 0), fx=2, fy=2)
        ii += 1

        # print reflection.min(), reflection.max()
        # cv2.imwrite("/tmp/syn_%03i.jpg" % ii, img * 255)
        cv2.imshow("img", img ** (1 / 2.2))
        cv2.waitKey(0)
