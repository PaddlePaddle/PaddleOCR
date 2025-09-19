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

struct WarpPredictorParams {
  absl::optional<std::string> model_name = absl::nullopt;
  absl::optional<std::string> model_dir = absl::nullopt;
  absl::optional<std::string> device = absl::nullopt;
  bool enable_mkldnn = true;
  std::string precision = "fp32";
  int mkldnn_cache_capacity = 10;
  int cpu_threads = 8;
  int batch_size = 1;
};

struct WarpPredictorResult {
  std::string input_path = "";
  cv::Mat input_image;
  cv::Mat doctr_img;
};

class WarpPredictor : public BasePredictor {
public:
  explicit WarpPredictor(const WarpPredictorParams &params);

  absl::Status Build();

  std::vector<std::unique_ptr<BaseCVResult>>
  Process(std::vector<cv::Mat> &batch_data) override;

  std::vector<WarpPredictorResult> PredictorResult() const {
    return predictor_result_vec_;
  };

  void ResetResult() override { predictor_result_vec_.clear(); };

private:
  std::unordered_map<std::string, std::unique_ptr<DocTrPostProcess>> post_op_;
  std::vector<WarpPredictorResult> predictor_result_vec_;
  std::unique_ptr<PaddleInfer> infer_ptr_;
  WarpPredictorParams params_;
  int input_index_ = 0;
};
