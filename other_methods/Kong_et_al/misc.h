/*
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
Authors: Patrick Wieschollek, Orazio Gallo, Jinwei Gu, and Jan Kautz
*/

#ifndef MISC_H
#define MISC_H

#include <cmath>
#include <Eigen/Dense>
#include <string>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/eigen.hpp>

/*
SEE verif.cu for tests of these functions
*/


/************* CV helper ******************/
typedef Eigen::MatrixXd Mat;
typedef Eigen::VectorXd Vec;


/**
 * @brief write plain data for file
 *
 * @param fn path to file
 * @param data raw c pointer to c-array
 * @param len length of array
 */
template<typename Dtype>
void write(std::string fn, Dtype* data, size_t len) {
  std::ofstream out(fn.c_str(), std::ios_base::binary);
  if (out.good()) {
    out.write((char *)data, len * sizeof(Dtype));
    out.close();
  }
}

/**
 * @brief read plain data from file
 *
 * @param fn path to file
 * @param data raw c pointer to c-array
 */
template<typename Dtype>
void read(std::string fn, Dtype* data) {
  std::ifstream in(fn.c_str(), std::ios_base::binary);

  // find size of file
  in.seekg( 0, std::ios::beg );
  std::streampos fsize = in.tellg();
  in.seekg( 0, std::ios::end );
  fsize = in.tellg() - fsize;
  // get number of elements (assuming only one type of data)
  size_t len = fsize / sizeof(Dtype);
  in.seekg( 0, std::ios::beg );

  if (in.good()) {
    in.read((char *)data, len * sizeof(Dtype));
    in.close();
  }
}

// more general min and max <double, float> with implicit cast
template<typename a, typename b>
a mymin(a i, b j) {
  return (i < j) ? i : j;
}


template<typename a, typename b>
a mymax(a i, b j) {
  return (i > j) ? i : j;
}

/**
 * @brief numerical stable way to sum up things
 * @details omit all negative values
 * @return sum
 */
double kahan_sum(const Vec &x) {
  const size_t n = x.size();
  if (n == 0)
    return 0;
  int off = 0;
  double s = 0;
  for (; off < x.size(); ++off) {
    if ((x(off) >= 0) && !std::isnan(x(off))) {
      double s = x(off);
      break;
    }
  }
  off++;

  double c = 0;
  for (int i = off; i < x.size(); ++i) {
    if ((x(i) >= 0)  && !std::isnan(x(off))  ) {
      double y = x(i) - c;
      double t = s + y;
      c = (t - s) - y;
      s = t;
    }
  }
  return s;
}

/**
 * @brief According to PAMI14 paper by Kng et al.
 */
double flip_tan(double deg) {
  deg = (deg < -45.0) ? deg + 90 : deg;
  deg = (deg > 45.0) ? deg - 90 : deg;
  return deg;
}


/**
 * @brief Eigen unary function for atan2
 */
Mat atan2(const Mat &A, const Mat &B) {
  Mat ans = Mat::Zero(A.rows(), A.cols());
  for (int r = 0; r < A.rows(); ++r)
    for (int c = 0; c < A.cols(); ++c)
      ans(r, c) = atan2(A(r, c), B(r, c));
  // return A.binaryExpr(B, std::ptr_fun(atan2));
  return ans;
}

/**
 * @brief replace all NaN by a given value
 */
void replace_nan(Vec *A, double val = 0) {
  for (int i = 0; i < A->size(); ++i) {
    if ((*A)(i) != (*A)(i)) {
      (*A)(i) = val;
    }
  }
}

/**
 * @brief replace all NaN by a given value
 */
void replace_nan(Mat *A, double val = 0) {
  for (int i = 0; i < A->rows(); ++i)
    for (int j = 0; j < A->cols(); ++j) {
      if ((*A)(i, j) != (*A)(i, j)) {
        (*A)(i, j) = val;
      }
    }
}

// read grayscale image from file and return Eigen-Matrix
Mat imread(std::string fn, bool scale = true) {
  Mat eig_mat;
  cv::Mat cv_mat = cv::imread(fn, 0);

  // cv::Mat planes[3];
  // cv::split(cv_mat,planes);
  // cv::cv2eigen(planes[2], eig_mat);
  cv::cv2eigen(cv_mat, eig_mat);
  if (scale)
    eig_mat = eig_mat / 255.;
  return eig_mat;
}

// write grayscale image to file from an eigen matrix
void imwrite(std::string fn, const Mat &eig_mat) {
  cv::Mat cv_mat;
  Mat cp(eig_mat);
  cp = cp * 255;
  cv::eigen2cv(cp, cv_mat);
  cv::imwrite(fn, cv_mat);
}

/************* REFLECTANCE FUNCTION ******************/

/**
 * @brief Compute radians from degree
 */
template<typename T>
T deg2rad (T degrees) {
  // const double pi_on_180 = 4.0 * atan (1.0) / 180.0;
  const double pi_on_180 = 0.017453292519943295;
  return degrees * pi_on_180;
}

/**
 * @brief convert radians to degree
 */
template<typename T>
T rad2deg (T degrees) {
  // const double pi_on_180 = 4.0 * atan (1.0) / 180.0;
  const double pi_on_180 = 0.017453292519943295;
  return degrees / pi_on_180;
}

/**
 * @brief compute refraction according Snell's law
 * @details n1*sin(alpha1) = n2*sin(alpha2)
 *
 * @param theta angle of incidence
 * @param kappa refraction coefficient (n2/n1) default for glas
 * @return refraction angle
 */
template<typename T>
T snell(T theta, T kappa = 1.474) {
  return asin(1. / kappa * sin(theta));
}

/**
 * @brief Compute perpendicular reflectance intensity
 * @details assume single plane (window)
 *          Kong, et al., eq. (5)
 */
double R_s(double theta, double kappa = 1.474) {
  const double theta_t = snell(theta, kappa);
  const double a = sin(theta - theta_t);
  const double b = sin(theta + theta_t);
  return (a * a) / (b * b);
}

/**
 * @brief Compute parallel reflectance intensity
 * @details assume single plane (window)
 *          Kong, et al., eq. (5)
 */
double R_p(double theta, double kappa = 1.474) {
  const double theta_t = snell(theta, kappa);
  const double a = tan(theta - theta_t);
  const double b = tan(theta + theta_t);
  return a * a / b / b;
}

/**
 * @brief Compute perpendicular transmission intensity
 * @details assume single plane (window)
 *          Kong, et al., eq. (6)
 */
double T_s(double theta, double kappa = 1.474) {
  return 1. - R_s(theta, kappa);
}

/**
 * @brief Compute parallel transmission intensity
 * @details assume single plane (window)
 *          Kong, et al., eq. (6)
 */
double T_p(double theta, double kappa = 1.474) {
  return 1. - R_p(theta, kappa);
}

/**
 * @brief Compute perpendicular reflectance intensity
 * @details assume double plane (window)
 *          Kong, et al., eq. (7)
 */
double RR_s(double theta, double kappa = 1.474) {
  const double r = R_s(theta, kappa);
  return 2. * r / (1. + r);
}

/**
 * @brief Compute parallel reflectance intensity
 * @details assume double plane (window)
 *          Kong, et al., eq. (7)
 */
double RR_p(double theta, double kappa = 1.474) {
  const double r = R_p(theta, kappa);
  return 2. * r / (1. + r);
}

/**
 * @brief Compute perpendicular transmission intensity
 * @details assume double plane (window)
 *          Kong, et al., eq. (7)
 */
double TT_s(double theta, double kappa = 1.474) {
  return 1. - RR_s(theta, kappa);
}

/**
 * @brief Compute parallel transmission intensity
 * @details assume double plane (window)
 *          Kong, et al., eq. (7)
 */
double TT_p(double theta, double kappa = 1.474) {
  return 1. - RR_p(theta, kappa);
}


/******** PROBABILITY FUNCTIONS ******/

/**
 * @brief Compute image probability
 * @details [long description]
 *
 * @param I image with value between [0, bins)
 * @param bins number of different bins
 *
 * @return vector of length bins with probabilities
 */
Vec image_prob(const Mat&I, const int bins = 256) {
  Vec prob = Vec::Zero(bins);
  for (int r = 0; r < I.rows(); ++r)
    for (int c = 0; c < I.cols(); ++c) {
      int pixel = I(r, c);

      if (pixel <= bins) {
        pixel = (pixel == bins) ? (bins - 1) : pixel;
        prob(pixel) = prob(pixel) + 1;
      }
    }

  return prob / prob.sum();
}

/**
 * @brief Compute joint histogram (np.histogram2d)
 * @details bi-dimensional histogram
 *
 * @param I image with values between [0, bins)
 * @param J image with values between [0, bins)
 * @param bins number of different bins
 * @return matrix [bins, bins] with joint probabity
 */
Mat join_hist(const Mat&I, const Mat&J, const int bins = 256) {
  Mat prob = Mat::Zero(bins, bins);

  for (int r = 0; r < I.rows(); ++r) {
    for (int c = 0; c < I.cols(); ++c) {
      int Ipixel = (int) I(r, c);
      int Jpixel = (int) J(r, c);
      Ipixel = (Ipixel > (bins - 1)) ? (bins - 1) : Ipixel;
      Jpixel = (Jpixel > (bins - 1)) ? (bins - 1) : Jpixel;

      prob(Ipixel, Jpixel) = prob(Ipixel, Jpixel) + 1;
    }
  }
  return prob / prob.sum();
}

/**
 * @brief Compute mutual information between two images
 * @details [long description]
 *
 * @param A [description]
 * @param B [description]
 * @param AB [description]
 * @return [description]
 */
double MI(const Mat&A, const Mat&B, const int bins = 256) {
  const Vec prob_a = image_prob(A, bins);
  const Vec prob_b = image_prob(B, bins);
  const Mat ab = join_hist(A, B, bins);

  const Mat ab_prod = prob_a * prob_b.transpose();

  Mat P = ab.array() * (ab.array() / ab_prod.array()).array().log();
  replace_nan(&P);

  return P.sum();
}

double entropy(const Mat&I) {
  Vec a = image_prob(I);
  a = a.array() * a.array().log();
  replace_nan(&a);
  return -a.sum();
}

double self_information(const Mat &A, const Mat &B) {
  return MI(A, B) * 2. / (entropy(A) + entropy(B));
}


/******** PROBABILITY FUNCTIONS ******/
// REMOVE ME LATER

void estimate(const Mat &I_p, const Mat &I_s, double AOI,
              Mat *guess_LB, Mat *guess_LR) {
  const double theta = deg2rad(AOI);
  const double rs = R_s(theta);
  const double rp = R_p(theta);

  // *guess_LR = (2-2*rp) / (rs-rp) * I_s.array() - ((2-2*rs) / (rs - rp)) *I_p.array();
  // eq. 18
  *guess_LR = 2 * ( (1 - rp) * I_s.array() - (1 - rs) * I_p.array()  ) / (rs - rp);
  // eq. 19
  *guess_LB = 2 * (rp * I_s.array() - rs * I_p.array()) / (rp - rs);

  std::cout << guess_LR->minCoeff() << " : " << guess_LR->maxCoeff() << std::endl;
  std::cout << guess_LB->minCoeff() << " : " << guess_LB->maxCoeff() << std::endl;

  // *guess_LB = 2 * rs / (rs-rp) * I_p.array()  - (2*rp / (rs - rp)) * I_s.array();
}


#endif