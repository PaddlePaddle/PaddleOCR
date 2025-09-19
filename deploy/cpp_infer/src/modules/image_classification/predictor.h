// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "processors.h"
#include "src/base/base_batch_sampler.h"
#include "src/base/base_cv_result.h"
#include "src/base/base_predictor.h"
#include "src/common/processors.h"

struct ClasPredictorParams {
  absl::optional<std::string> model_name = absl::nullopt;
  absl::optional<std::string> model_dir = absl::nullopt;
  absl::optional<std::string> device = absl::nullopt;
  std::string precision = "fp32";
  bool enable_mkldnn = true;
  int mkldnn_cache_capacity = 10;
  int cpu_threads = 8;
  int batch_size = 1;
};

struct ClasPredictorResult {
  std::string input_path = "";
  cv::Mat input_image;
  std::vector<int> class_ids;
  std::vector<float> scores;
  std::vector<std::string> label_names;
};

class ClasPredictor : public BasePredictor {
public:
  explicit ClasPredictor(const ClasPredictorParams &params);

  ClasPredictor() = delete;

  absl::Status Build();

  std::vector<std::unique_ptr<BaseCVResult>>
  Process(std::vector<cv::Mat> &batch_data) override;

  std::vector<ClasPredictorResult> PredictorResult() const {
    return predictor_result_vec_;
  };

  void ResetResult() override { predictor_result_vec_.clear(); };

  absl::Status BuildResize();

private:
  ClasPredictorParams params_;
  std::unordered_map<std::string, std::unique_ptr<Topk>> post_op_;
  std::vector<ClasPredictorResult> predictor_result_vec_;
  std::unique_ptr<PaddleInfer> infer_ptr_;
  int input_index_ = 0;
};
