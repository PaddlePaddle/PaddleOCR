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

#include <include/postprocess_op.h>
#include <include/preprocess_op.h>

namespace PaddleOCR {

class DBDetector {
public:
  explicit DBDetector(const std::string &model_dir, bool use_gpu = false,
                      const int gpu_id = 0, const int max_side_len = 960) {
    LoadModel(model_dir, use_gpu);
    this->max_side_len_ = max_side_len;
  }

  // Load Paddle inference model
  void LoadModel(const std::string &model_dir, bool use_gpu,
                 const int min_subgraph_size = 3, const int batch_size = 1,
                 const int gpu_id = 0);

  // Run predictor
  void Run(cv::Mat &img, std::vector<std::vector<std::vector<int>>> &boxes);

private:
  std::shared_ptr<PaddlePredictor> predictor_;

  int max_side_len_ = 960;

  std::vector<float> mean_ = {0.485f, 0.456f, 0.406f};
  std::vector<float> scale_ = {1 / 0.229f, 1 / 0.224f, 1 / 0.225f};
  bool is_scale_ = true;

  // pre-process
  ResizeImgType0 resize_op_;
  Normalize normalize_op_;
  Permute permute_op_;

  // post-process
  PostProcessor post_processor_;
};

} // namespace PaddleOCR