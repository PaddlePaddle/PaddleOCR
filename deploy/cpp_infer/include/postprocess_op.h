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

#include "include/clipper.h"

using namespace std;

namespace PaddleOCR {

inline std::vector<std::string> ReadDict(std::string path) {
  std::ifstream in(path);
  std::string filename;
  std::string line;
  std::vector<std::string> m_vec;
  if (in) {
    while (getline(in, line)) {
      m_vec.push_back(line);
    }
  } else {
    std::cout << "no such file" << std::endl;
  }
  return m_vec;
}

template <class ForwardIterator>
inline size_t Argmax(ForwardIterator first, ForwardIterator last) {
  return std::distance(first, std::max_element(first, last));
}

class PostProcessor {
public:
  void GetContourArea(float **box, float unclip_ratio, float &distance);

  cv::RotatedRect unclip(float **box);

  float **Mat2Vec(cv::Mat mat);

  void quickSort_vector(std::vector<std::vector<int>> &box, int l, int r,
                        int axis);

  std::vector<std::vector<int>>
  order_points_clockwise(std::vector<std::vector<int>> pts);

  float **get_mini_boxes(cv::RotatedRect box, float &ssid);

  float box_score_fast(float **box_array, cv::Mat pred);

  std::vector<std::vector<std::vector<int>>>
  boxes_from_bitmap(const cv::Mat pred, const cv::Mat bitmap);

  std::vector<std::vector<std::vector<int>>>
  filter_tag_det_res(std::vector<std::vector<std::vector<int>>> boxes,
                     float ratio_h, float ratio_w, cv::Mat srcimg);

  template <class ForwardIterator>
  inline size_t argmax(ForwardIterator first, ForwardIterator last) {
    return std::distance(first, std::max_element(first, last));
  }

  // CRNN

private:
  void quickSort(float **s, int l, int r);

  inline int _max(int a, int b) { return a >= b ? a : b; }

  inline int _min(int a, int b) { return a >= b ? b : a; }

  template <class T> inline T clamp(T x, T min, T max) {
    if (x > max)
      return max;
    if (x < min)
      return min;
    return x;
  }
  inline float clampf(float x, float min, float max) {
    if (x > max)
      return max;
    if (x < min)
      return min;
    return x;
  }
};

} // namespace PaddleOCR