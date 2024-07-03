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

#include "ocr_cls_process.h"
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

const std::vector<int> CLS_IMAGE_SHAPE = {3, 48, 192};

cv::Mat cls_resize_img(const cv::Mat &img) {
  int imgC = CLS_IMAGE_SHAPE[0];
  int imgW = CLS_IMAGE_SHAPE[2];
  int imgH = CLS_IMAGE_SHAPE[1];

  float ratio = float(img.cols) / float(img.rows);
  int resize_w = 0;
  if (ceilf(imgH * ratio) > imgW)
    resize_w = imgW;
  else
    resize_w = int(ceilf(imgH * ratio));

  cv::Mat resize_img;
  cv::resize(img, resize_img, cv::Size(resize_w, imgH), 0.f, 0.f,
             cv::INTER_CUBIC);

  if (resize_w < imgW) {
    cv::copyMakeBorder(resize_img, resize_img, 0, 0, 0, int(imgW - resize_w),
                       cv::BORDER_CONSTANT, {0, 0, 0});
  }
  return resize_img;
}
