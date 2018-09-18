/*
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
Authors: Patrick Wieschollek, Orazio Gallo, Jinwei Gu, and Jan Kautz

This code is our implementation of the paper:
Kong, N., Tai, Y.W., Shin, J.S., “A physically-based approach to reflection separation: From physical modeling to constrained optimization,” IEEE TPAMI, 2014
*/
#include <iostream>
#include <cmath>
#include <functional>
#include <Eigen/Dense>
#include <fstream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/eigen.hpp>

#include "misc.h"
#include "grid.h"
#include "mrf.h"

typedef Eigen::MatrixXd Mat;
typedef Eigen::VectorXd Vec;

/**
 * @brief recover intensity of light from reflected
 * @details intensity of light from reflected at glas surface
 * @details Kong, et al., eq. (18)
 *
 * @param theta [description]
 * @param I_s [description]
 * @param I_p [description]
 * @return [description]
 */
Mat LR_func(double theta, const Mat &I_s, const Mat &I_p) {
  const double rp = RR_p(theta);
  const double rs = RR_s(theta);

  Mat nom = 2 * ((1 - rp) * I_s.array() - (1 - rs) * I_p.array());
  return nom.array() / (rs - rp);
}

/**
 * @brief recover intensity of light from transmitted
 * @details intensity of light from background scene
 * @details Kong, et al., eq. (19)
 *
 * @param theta [description]
 * @param I_s [description]
 * @param I_p [description]
 * @return [description]
 */
Mat LB_func(double theta, const Mat &I_s, const Mat &I_p) {
  const double rp = RR_p(theta);
  const double rs = RR_s(theta);
  Mat nom =  2 * (rp * I_s.array() - rs * I_p.array());
  return nom.array() / (rs - rp);
}


/**
 * @brief synthesize I given all information
 * @details Kong, et al., eq. (3)
 *
 * @param LR intensity before reflections
 * @param LB intensity before transmission
 * @param phi polizer angle
 * @param theta angle of incidence (AOI)
 * @param kappa refraction coefficient (Snell's Law)
 * @return intensity of light after polarizer
 */
Mat I(const Mat &LR, const Mat &LB,
      double phi, double theta, double kappa = 1.474) {
  // angle for orientation of the intersection between polarizer and POI
  const double phi_s = 0;

  const double rs = RR_s(theta, kappa);
  const double rp = RR_p(theta, kappa);

  const double alpha = rs * cos(phi - phi_s) * cos(phi - phi_s) + rp * sin(phi - phi_s);

  // TODO: with /.2 or without ?
  return alpha / 2. * LR.array() + (1 - alpha) / 2. * LB.array();
}




int main(int argc, char const *argv[]) {
  // load ground-truth alpha mattes
  Mat alpha1 = imread("/data/project/literature/images/physical_alpha1_crop.png");
  Mat alpha2 = imread("/data/project/literature/images/physical_alpha2_crop.png");
  Mat alpha3 = imread("/data/project/literature/images/physical_alpha3_crop.png");

  // load ground-truth reflectance and transmission
  Mat LR = imread("/data/project/literature/images/physical_lr_crop.png");
  Mat LB = imread("/data/project/literature/images/physical_lb_crop.png");

  // generate observations
  Mat I1 = (alpha1.array() * LR.array() / 2. + (1 - alpha1.array()) * LB.array() / 2.);
  Mat I2 = (alpha2.array() * LR.array() / 2. + (1 - alpha2.array()) * LB.array() / 2.);
  Mat I3 = (alpha3.array() * LR.array() / 2. + (1 - alpha3.array()) * LB.array() / 2.);

  // NOW, we solve the inverse problem
  std::cout << I1.minCoeff() << " " << I1.maxCoeff() << std::endl;
  std::cout << I2.minCoeff() << " " << I2.maxCoeff() << std::endl;
  std::cout << I3.minCoeff() << " " << I3.maxCoeff() << std::endl;


  const int HEIGHT = alpha1.rows();
  const int WIDTH = alpha1.cols();
  const int LABELS = 85 - 5;
  const int window = 5;


  // 4.1.1 orthogonal image extraction
  // ===============================================
  // eq. (15)
  Mat _a = I1 + I3 - 2 * I2;
  Mat _b = I1 - I3;
  Mat phi_1_minus_phi_perp = 0.5 * atan2(_a, _b);

  // we assume within [-45, 45]
  phi_1_minus_phi_perp = phi_1_minus_phi_perp.unaryExpr(&rad2deg<double>);
  phi_1_minus_phi_perp = phi_1_minus_phi_perp.unaryExpr(&flip_tan);
  phi_1_minus_phi_perp = phi_1_minus_phi_perp.unaryExpr(&deg2rad<double>);


  // eq. (16)
  Mat I_s = (I1 + I3).array() / 2. +
            (I1 - I3).array() / (2 * phi_1_minus_phi_perp).array().cos();

  // eq. (17)
  Mat I_p = (I1 + I3).array() / 2. -
            (I1 - I3).array() / (2 * phi_1_minus_phi_perp).array().cos();


  // 4.1.2 image separation
  // ===============================================
  /*
  - try to estimate AOI theta and to separate LR and LB.
  - LR and LB are functions in theta (unknown)
  - Schechner et al. use "self_information" (normalized mutial information)
    - in best theta
  - for each theta in [0, 85] compute LR and LB to solve eq. (23)
  */

  // PSI - costs (binary)
  // eq. (25)
  float *label_cost = new float[LABELS * LABELS];
  for (int i = 0; i < LABELS; ++i)
    for (int j = 0; j < LABELS; ++j)
      label_cost[i * LABELS + j] = 0.5 * abs(i - j);

  // phi costs (unary)
  float *phi = new float[LABELS * HEIGHT * WIDTH];

  // precompute cube [theta, T/R, 255, 255]
  float *IT_versions = new float[LABELS * HEIGHT * WIDTH];
  float *IR_versions = new float[LABELS * HEIGHT * WIDTH];


  // "We prepare all candidate pairs LR, LB evaluated at a
  //  sequence of regularly sampled values of theta" (p. 7)
  #pragma omp parallel for
  for (int theta_off = 0; theta_off < LABELS; ++theta_off) {
    const int theta = theta_off + 5;

    Mat guess_LB, guess_LR;

    Eigen::Map<Eigen::MatrixXf> guess_ITf(IT_versions + theta_off * HEIGHT * WIDTH, HEIGHT, WIDTH);
    Eigen::Map<Eigen::MatrixXf> guess_IRf(IR_versions + theta_off * HEIGHT * WIDTH, HEIGHT, WIDTH);

    // eq. 18, eq. 19
    estimate(I_p, I_s, theta, &guess_LB, &guess_LR);

    guess_ITf = guess_LB.cast<float>();
    guess_IRf = guess_LR.cast<float>();
  }

  write<float>("IT_versions.fbin", IT_versions, LABELS * HEIGHT * WIDTH);
  write<float>("IR_versions.fbin", IR_versions, LABELS * HEIGHT * WIDTH);


  Eigen::Map<Eigen::MatrixXf> guess_ITf(IT_versions, HEIGHT, WIDTH * LABELS);
  Eigen::Map<Eigen::MatrixXf> guess_IRf(IR_versions, HEIGHT, WIDTH * LABELS);
  std::cout << guess_ITf.minCoeff() << "\t" << guess_ITf.maxCoeff() << std::endl;
  std::cout << guess_IRf.minCoeff() << "\t" << guess_IRf.maxCoeff() << std::endl;
  return 0;


  /*
  "Instead, we assume that theta is static locally, i.e.,
  spatially invariant within a small patch around each
  pixel, based on its smoothness condition in a spatial
  domain.""
  */
  // this takes really a while (reason to not use python), go and have some coffee ...
  #pragma omp parallel for
  for (int theta_off = 0; theta_off < LABELS; ++theta_off) {
    const int theta = theta_off + 5;

    Eigen::Map<Eigen::MatrixXf> guess_ITf(IT_versions + theta_off * HEIGHT * WIDTH, HEIGHT, WIDTH);
    Eigen::Map<Eigen::MatrixXf> guess_IRf(IR_versions + theta_off * HEIGHT * WIDTH, HEIGHT, WIDTH);

    Mat guess_IR = guess_ITf.cast<double>();
    Mat guess_IT = guess_IRf.cast<double>();

    // std::cout << LR.minCoeff() << "\t" << LR.maxCoeff() << std::endl;
    // std::cout << LB.minCoeff() << "\t" << LB.maxCoeff() << std::endl;


    for (int r = 0; r < LR.rows(); ++r) {
      for (int c = 0; c < LR.cols(); ++c) {
        // extract patch around (r, c)
        const int left = mymax(0., r - window);
        const int right = mymin(LR.cols() - 1., r + window);
        const int pwidth = right - left;

        const int top = mymax(0., c - window);
        const int bottom = mymin(LR.rows() - 1., c + window);
        const int pheight = bottom - top;

        float cur_self_info = 0;

        Mat patch_IT = LR.block(top, left, pheight, pwidth);
        Mat patch_IR = LB.block(top, left, pheight, pwidth);


        /*
        "Then, our data cost function can be defined
        in terms of the patch-wise mutual information:"
        eq. (24)
        */
        cur_self_info = self_information(patch_IT, patch_IR);
        if (cur_self_info != cur_self_info)
          cur_self_info = 10000;
        phi[r * WIDTH * LABELS + c * LABELS + theta_off] = cur_self_info;

        // std::cout << cur_self_info
        //           << "\t" << patch_IT.rows() << "\t" << patch_IT.cols()
        //           << "\t" << patch_IR.rows() << "\t" << patch_IR.cols()
        //           << "\t" << patch_IT.minCoeff() << "\t" << patch_IT.maxCoeff()
        //           << "\t" << patch_IR.minCoeff() << "\t" << patch_IR.maxCoeff() << std::endl;

      }
    }
  }

  // write<float>("phi.fbin", phi, LABELS * HEIGHT * WIDTH);
  // read<float>("phi.fbin", phi);

  // some numerical instabilities
  // for (int i = 0; i < HEIGHT * WIDTH * LABELS; ++i) {
  //   if (phi[i] != phi[i])
  //     phi[i] + 10000;
  // }

  Grid<float> G(HEIGHT, WIDTH, LABELS);
  G.labelCosts.host.data = phi;
  G.smoothnessCosts.host.data = label_cost;

  MRF<float> mrf(&G);
  std::cout << "start energy " << mrf.map() << std::endl;

  for (int i = 0; i < 5; ++i) {
    mrf.propagate();
    printf("iter: %03i energy %f\n", i, mrf.map());
  }


  /*
    We also compute alpha(x) for each input image
    based on its definition in Eq. (4), by setting theta(x) to the
    value obtained from belief propagation
  */


  // std::cout << phi[0] << std::endl;
  // Mat debug(phi_1_minus_phi_perp);
  // debug = debug.array() - debug.minCoeff();
  // debug = debug.array() / debug.maxCoeff();
  // imwrite("phi_1_minus_phi_perp.png", debug);
  // imwrite("I_s.png", I_s);
  // imwrite("I_p.png", I_p);
  // imwrite("I1.png", I1);
  // imwrite("I2.png", I2);
  // imwrite("I3.png", I3);
  // // imwrite("label_cost.png", label_cost);

  return 0;
}