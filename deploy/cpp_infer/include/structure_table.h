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

#include "paddle_api.h"
#include "paddle_inference_api.h"

#include <include/postprocess_op.h>
#include <include/preprocess_op.h>

namespace PaddleOCR {

class StructureTableRecognizer {
public:
  explicit StructureTableRecognizer(
      const std::string &model_dir, const bool &use_gpu, const int &gpu_id,
      const int &gpu_mem, const int &cpu_math_library_num_threads,
      const bool &use_mkldnn, const std::string &label_path,
      const bool &use_tensorrt, const std::string &precision,
      const int &table_batch_num, const int &table_max_len,
      const bool &merge_no_span_structure) {
    this->use_gpu_ = use_gpu;
    this->gpu_id_ = gpu_id;
    this->gpu_mem_ = gpu_mem;
    this->cpu_math_library_num_threads_ = cpu_math_library_num_threads;
    this->use_mkldnn_ = use_mkldnn;
    this->use_tensorrt_ = use_tensorrt;
    this->precision_ = precision;
    this->table_batch_num_ = table_batch_num;
    this->table_max_len_ = table_max_len;

    this->post_processor_.init(label_path, merge_no_span_structure);
    LoadModel(model_dir);
  }

  // Load Paddle inference model
  void LoadModel(const std::string &model_dir);

  void Run(std::vector<cv::Mat> img_list,
           std::vector<std::vector<std::string>> &rec_html_tags,
           std::vector<float> &rec_scores,
           std::vector<std::vector<std::vector<int>>> &rec_boxes,
           std::vector<double> &times);

private:
  std::shared_ptr<paddle_infer::Predictor> predictor_;

  bool use_gpu_ = false;
  int gpu_id_ = 0;
  int gpu_mem_ = 4000;
  int cpu_math_library_num_threads_ = 4;
  bool use_mkldnn_ = false;
  int table_max_len_ = 488;

  std::vector<float> mean_ = {0.485f, 0.456f, 0.406f};
  std::vector<float> scale_ = {1 / 0.229f, 1 / 0.224f, 1 / 0.225f};
  bool is_scale_ = true;

  bool use_tensorrt_ = false;
  std::string precision_ = "fp32";
  int table_batch_num_ = 1;

  // pre-process
  TableResizeImg resize_op_;
  Normalize normalize_op_;
  PermuteBatch permute_op_;
  TablePadImg pad_op_;

  // post-process
  TablePostProcessor post_processor_;

}; // class StructureTableRecognizer

} // namespace PaddleOCR
