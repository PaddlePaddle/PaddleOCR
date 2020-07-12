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

#include <include/postprocess_op.h>
#include <include/preprocess_op.h>

namespace PaddleOCR {

class CRNNRecognizer {
public:
  explicit CRNNRecognizer(const std::string &model_dir,
                          const string label_path = "./tools/ppocr_keys_v1.txt",
                          bool use_gpu = false, const int gpu_id = 0) {
    LoadModel(model_dir, use_gpu);

    this->label_list_ = ReadDict(label_path);
  }

  // Load Paddle inference model
  void LoadModel(const std::string &model_dir, bool use_gpu,
                 const int gpu_id = 0, const int min_subgraph_size = 3,
                 const int batch_size = 1);

  void Run(std::vector<std::vector<std::vector<int>>> boxes, cv::Mat &img);

private:
  std::shared_ptr<PaddlePredictor> predictor_;

  std::vector<std::string> label_list_;

  std::vector<float> mean_ = {0.5f, 0.5f, 0.5f};
  std::vector<float> scale_ = {1 / 0.5f, 1 / 0.5f, 1 / 0.5f};
  bool is_scale_ = true;

  // pre-process
  CrnnResizeImg resize_op_;
  Normalize normalize_op_;
  Permute permute_op_;

  // post-process
  PostProcessor post_processor_;

  cv::Mat get_rotate_crop_image(const cv::Mat &srcimage,
                                std::vector<std::vector<int>> box);

  std::vector<std::string> ReadDict(const std::string &path);

  template <class ForwardIterator>
  inline size_t argmax(ForwardIterator first, ForwardIterator last) {
    return std::distance(first, std::max_element(first, last));
  }

}; // class CrnnRecognizer

} // namespace PaddleOCR