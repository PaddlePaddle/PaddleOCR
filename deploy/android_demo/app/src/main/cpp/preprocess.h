#pragma once

#include <jni.h>
#include <opencv2/opencv.hpp>
#include "common.h"
cv::Mat bitmap_to_cv_mat(JNIEnv *env, jobject bitmap);

cv::Mat resize_img(const cv::Mat& img, int height, int width);

void neon_mean_scale(const float* din,
                     float* dout,
                     int size,
                     const std::vector<float>& mean,
                     const std::vector<float>& scale);
