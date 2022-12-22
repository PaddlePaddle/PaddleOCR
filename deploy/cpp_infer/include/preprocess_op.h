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

#pragma once

#include <iostream>
#include <vector>

#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

namespace PaddleOCR {

class Normalize {
public:
  virtual void Run(cv::Mat *im, const std::vector<float> &mean,
                   const std::vector<float> &scale, const bool is_scale = true);
};

// RGB -> CHW
class Permute {
public:
  virtual void Run(const cv::Mat *im, float *data);
};

class PermuteBatch {
public:
  virtual void Run(const std::vector<cv::Mat> imgs, float *data);
};

class ResizeImgType0 {
public:
  virtual void Run(const cv::Mat &img, cv::Mat &resize_img,
                   std::string limit_type, int limit_side_len, float &ratio_h,
                   float &ratio_w, bool use_tensorrt);
};

class CrnnResizeImg {
public:
  virtual void Run(const cv::Mat &img, cv::Mat &resize_img, float wh_ratio,
                   bool use_tensorrt = false,
                   const std::vector<int> &rec_image_shape = {3, 32, 320});
};

class ClsResizeImg {
public:
  virtual void Run(const cv::Mat &img, cv::Mat &resize_img,
                   bool use_tensorrt = false,
                   const std::vector<int> &rec_image_shape = {3, 48, 192});
};

class TableResizeImg {
public:
  virtual void Run(const cv::Mat &img, cv::Mat &resize_img,
                   const int max_len = 488);
};

class TablePadImg {
public:
  virtual void Run(const cv::Mat &img, cv::Mat &resize_img,
                   const int max_len = 488);
};

class Resize {
public:
  virtual void Run(const cv::Mat &img, cv::Mat &resize_img, const int h,
                   const int w);
};

} // namespace PaddleOCR