/*
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
Authors: Patrick Wieschollek, Orazio Gallo, Jinwei Gu, and Jan Kautz

This code is our implementation of the paper:
Kong, N., Tai, Y.W., Shin, J.S., “A physically-based approach to reflection separation: From physical modeling to constrained optimization,” IEEE TPAMI, 2014
*/

// just to make sure, the parts of the implementation for the PAMI14 paper are correct
#include "gtest/gtest.h"
#include <Eigen/Dense>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>

#include "misc.h"

#define PI_1   3.14159265358979323846      // 180
#define PI_2   3.14159265358979323846 / 2. // 90
#define PI_4   3.14159265358979323846 / 4. // 45
#define thresh   1e-6

TEST(Basics, atan) {
  // >>> n.arctan2(0.1, 0.5)
  // 0.19739555984988078
  EXPECT_NEAR(atan2(0.1, 0.5), 0.19739555984988078, thresh);

}

TEST(Basics, deg2rad) {
  EXPECT_FLOAT_EQ(deg2rad((float) 0), 0.0);
  EXPECT_FLOAT_EQ(deg2rad((float) 45), PI_4);
  EXPECT_FLOAT_EQ(deg2rad((float) 90), PI_2);
  EXPECT_FLOAT_EQ(deg2rad((float) 180), PI_1);
  EXPECT_FLOAT_EQ(deg2rad((float) 270), 4.71238898038);
  EXPECT_FLOAT_EQ(deg2rad((float) 80), 1.3962634016);
}
TEST(Basics, rad2deg) {
  EXPECT_FLOAT_EQ(rad2deg((float) 0.0), 0.);
  EXPECT_FLOAT_EQ(rad2deg((float) PI_4), 45.);
  EXPECT_FLOAT_EQ(rad2deg((float) PI_2), 90.);
  EXPECT_FLOAT_EQ(rad2deg((float) PI_1), 180.);
  EXPECT_FLOAT_EQ(rad2deg((float) 4.71238898038), 270.);
  EXPECT_FLOAT_EQ(rad2deg((float) 1.3962634016), 80.);
}
TEST(Basics, snell) {
  EXPECT_NEAR(snell((float) 0.0), 0., thresh);
  EXPECT_NEAR(snell((float) PI_4), 0.50033518205243543, thresh);
  EXPECT_NEAR(snell((float) PI_2), 0.74561811743627437, thresh);
  EXPECT_NEAR(snell((float) PI_1), 8.3083229250159652e-17, thresh);
}
TEST(Basics, rs) {
  EXPECT_NEAR(R_s(PI_1), 1., thresh);
  EXPECT_NEAR(R_s(PI_2), 1., thresh);
  EXPECT_NEAR(R_s(PI_4), 0.085874780966717162, thresh);
}
TEST(Basics, rp) {
  EXPECT_NEAR(R_p(PI_1), 1., thresh);
  EXPECT_NEAR(R_p(PI_2), 0.99999999999999922, thresh);
  EXPECT_NEAR(R_p(PI_4), 0.0073744780060816424, thresh);
}
TEST(Basics, rrs) {
  EXPECT_FLOAT_EQ(RR_s(PI_1), 1.);
  EXPECT_FLOAT_EQ(RR_s(PI_2), 1.);
  EXPECT_FLOAT_EQ(RR_s(PI_4), 0.15816700502108685);
}
TEST(Basics, rrp) {
  EXPECT_FLOAT_EQ(RR_p(PI_1), 1.);
  EXPECT_FLOAT_EQ(RR_p(PI_2), 0.99999999999999967);
  EXPECT_FLOAT_EQ(RR_p(PI_4), 0.014640986380115781);
}

TEST(Probability, image_prob_basic) {
  // [0,1,2,3,4,5,6,7,8,9,0,0,0,0,0,....]
  const int len = 10;

  Mat tmp(len, 1);
  for (int i = 0; i < len; ++i)
    tmp(i, 0) = (double) i;

  Vec prob = image_prob(tmp);

  EXPECT_EQ(256, prob.size());
  for (int i = 0; i < 10; ++i)
    EXPECT_NEAR(prob(i), 0.1, 1e-10);
  for (int i = 10; i < 256; ++i)
    EXPECT_NEAR(prob(i), 0, 1e-10);

}

TEST(Probability, image_prob_basic2) {
  // np.arange(256) _> all should be 1/256
  const int len = 256;

  Mat tmp(len, 1);
  for (int i = 0; i < len; ++i)
    tmp(i, 0) = (double) i;

  Vec prob = image_prob(tmp);

  EXPECT_EQ(256, prob.size());
  for (int i = 0; i < 256; ++i)
    EXPECT_NEAR(prob(i), 1. / 256, 1e-10);

}

TEST(Probability, joint_hist_basic) {
  const int len = 256;

  Mat tmp(len, 1);
  for (int i = 0; i < len; ++i)
    tmp(i, 0) = (double) i;

  Mat prob = join_hist(tmp, tmp);

  EXPECT_EQ(256 * 256, prob.size());
  for (int i = 0; i < 256; ++i)
    EXPECT_NEAR(prob(i, i), 1. / 256, 1e-10);
  for (int i = 0; i < 256; ++i)
    for (int j = i + 1; j < 256; ++j)
      EXPECT_NEAR(prob(i, j), 0, 1e-10);

}

TEST(Basics, sum) {
  Vec tmp(10);
  for (int i = 0; i < 10; ++i) {
    tmp(i) = i;
  }
  EXPECT_EQ(tmp.sum(), kahan_sum(tmp));
}

TEST(Probability, entropy) {
  const Mat IT = imread("/data/project/literature/real_layer.png", false).block(0, 0, 156, 156);
  EXPECT_NEAR(4.99342910252, entropy(IT), 1e-5);
}


TEST(Probability, mutual_information) {

  const Mat IT = imread("/data/project/literature/real_layer.png", false).block(0, 0, 156, 156);
  const Mat IR = imread("/data/project/literature/virtual_layer.png", false).block(0, 0, 156, 156);

  EXPECT_NEAR(0.872980173351, MI(IT, IR), 1e-5);

}

TEST(Probability, self_information) {

  const Mat IT = imread("/data/project/literature/real_layer.png", false).block(0, 0, 156, 156);
  const Mat IR = imread("/data/project/literature/virtual_layer.png", false).block(0, 0, 156, 156);

  EXPECT_NEAR(0.17022607913, self_information(IT, IR), 1e-5);

}

int main (int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
