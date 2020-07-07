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

#include <math.h>

#include <iostream>
#include <map>
#include <vector>

#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "utils/clipper.hpp"

template <class T>
T clamp(T x, T min, T max) {
  if (x > max) return max;
  if (x < min) return min;
  return x;
}

std::vector<std::vector<float>> Mat2Vector(cv::Mat mat);

void GetContourArea(std::vector<std::vector<float>> box,
                    float unclip_ratio,
                    float &distance);

cv::RotatedRect Unclip(std::vector<std::vector<float>> box, float unclip_ratio);

std::vector<std::vector<float>> Mat2Vector(cv::Mat mat);

bool XsortFp32(std::vector<float> a, std::vector<float> b);

bool XsortInt(std::vector<int> a, std::vector<int> b);

std::vector<std::vector<int>> OrderPointsClockwise(
    std::vector<std::vector<int>> pts);

std::vector<std::vector<float>> GetMiniBoxes(cv::RotatedRect box, float &ssid);

float BoxScoreFast(std::vector<std::vector<float>> box_array, cv::Mat pred);

std::vector<std::vector<std::vector<int>>> BoxesFromBitmap(
    const cv::Mat pred,
    const cv::Mat bitmap,
    std::map<std::string, double> Config);

std::vector<std::vector<std::vector<int>>> FilterTagDetRes(
    std::vector<std::vector<std::vector<int>>> boxes,
    float ratio_h,
    float ratio_w,
    cv::Mat srcimg);
