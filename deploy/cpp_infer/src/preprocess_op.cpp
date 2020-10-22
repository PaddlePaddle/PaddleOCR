// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "paddle_api.h"
#include "paddle_inference_api.h"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <vector>

#include <cstring>
#include <fstream>
#include <numeric>

#include <include/preprocess_op.h>

namespace PaddleOCR {

void Permute::Run(const cv::Mat *im, float *data) {
  int rh = im->rows;
  int rw = im->cols;
  int rc = im->channels();
  for (int i = 0; i < rc; ++i) {
    cv::extractChannel(*im, cv::Mat(rh, rw, CV_32FC1, data + i * rh * rw), i);
  }
}

void Normalize::Run(cv::Mat *im, const std::vector<float> &mean,
                    const std::vector<float> &scale, const bool is_scale) {
  double e = 1.0;
  if (is_scale) {
    e /= 255.0;
  }
  (*im).convertTo(*im, CV_32FC3, e);
  for (int h = 0; h < im->rows; h++) {
    for (int w = 0; w < im->cols; w++) {
      im->at<cv::Vec3f>(h, w)[0] =
          (im->at<cv::Vec3f>(h, w)[0] - mean[0]) * scale[0];
      im->at<cv::Vec3f>(h, w)[1] =
          (im->at<cv::Vec3f>(h, w)[1] - mean[1]) * scale[1];
      im->at<cv::Vec3f>(h, w)[2] =
          (im->at<cv::Vec3f>(h, w)[2] - mean[2]) * scale[2];
    }
  }
}

void ResizeImgType0::Run(const cv::Mat &img, cv::Mat &resize_img,
                         int max_size_len, float &ratio_h, float &ratio_w) {
  int w = img.cols;
  int h = img.rows;

  float ratio = 1.f;
  int max_wh = w >= h ? w : h;
  if (max_wh > max_size_len) {
    if (h > w) {
      ratio = float(max_size_len) / float(h);
    } else {
      ratio = float(max_size_len) / float(w);
    }
  }

  int resize_h = int(float(h) * ratio);
  int resize_w = int(float(w) * ratio);
  if (resize_h % 32 == 0)
    resize_h = resize_h;
  else if (resize_h / 32 < 1 + 1e-5)
    resize_h = 32;
  else
    resize_h = (resize_h / 32 - 1) * 32;

  if (resize_w % 32 == 0)
    resize_w = resize_w;
  else if (resize_w / 32 < 1 + 1e-5)
    resize_w = 32;
  else
    resize_w = (resize_w / 32 - 1) * 32;

  cv::resize(img, resize_img, cv::Size(resize_w, resize_h));

  ratio_h = float(resize_h) / float(h);
  ratio_w = float(resize_w) / float(w);
}

void CrnnResizeImg::Run(const cv::Mat &img, cv::Mat &resize_img, float wh_ratio,
                        const std::vector<int> &rec_image_shape) {
  int imgC, imgH, imgW;
  imgC = rec_image_shape[0];
  imgH = rec_image_shape[1];
  imgW = rec_image_shape[2];

  imgW = int(32 * wh_ratio);

  float ratio = float(img.cols) / float(img.rows);
  int resize_w, resize_h;
  if (ceilf(imgH * ratio) > imgW)
    resize_w = imgW;
  else
    resize_w = int(ceilf(imgH * ratio));

  cv::resize(img, resize_img, cv::Size(resize_w, imgH), 0.f, 0.f,
             cv::INTER_LINEAR);
}

void ClsResizeImg::Run(const cv::Mat &img, cv::Mat &resize_img,
                       const std::vector<int> &rec_image_shape) {
  int imgC, imgH, imgW;
  imgC = rec_image_shape[0];
  imgH = rec_image_shape[1];
  imgW = rec_image_shape[2];

  float ratio = float(img.cols) / float(img.rows);
  int resize_w, resize_h;
  if (ceilf(imgH * ratio) > imgW)
    resize_w = imgW;
  else
    resize_w = int(ceilf(imgH * ratio));

  cv::resize(img, resize_img, cv::Size(resize_w, imgH), 0.f, 0.f,
             cv::INTER_LINEAR);
  if (resize_w < imgW) {
    cv::copyMakeBorder(resize_img, resize_img, 0, 0, 0, imgW - resize_w,
                       cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
  }
}

} // namespace PaddleOCR
