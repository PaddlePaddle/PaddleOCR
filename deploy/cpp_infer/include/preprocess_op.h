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

#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <vector>

#include <cstring>
#include <fstream>
#include <numeric>

using namespace std;
using namespace paddle;

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

// RGB -> CHW
class ResizeImgType0 {
public:
  virtual void Run(const cv::Mat &img, cv::Mat &resize_img, int max_size_len,
                   float &ratio_h, float &ratio_w);
};

class CrnnResizeImg {
public:
  virtual void Run(const cv::Mat &img, cv::Mat &resize_img, float wh_ratio,
                   const std::vector<int> rec_image_shape = {3, 32, 320});
};

} // namespace PaddleOCR