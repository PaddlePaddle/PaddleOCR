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

#include "src/modules/text_detection/predictor.h"

struct TextDetectionParams {
  absl::optional<std::string> model_name = absl::nullopt;
  absl::optional<std::string> model_dir = absl::nullopt;
  absl::optional<std::string> device = absl::nullopt;
  std::string precision = "fp32";
  bool enable_mkldnn = true;
  int mkldnn_cache_capacity = 10;
  int cpu_threads = 8;
  int batch_size = 1;
  absl::optional<int> limit_side_len = absl::nullopt;
  absl::optional<std::string> limit_type = absl::nullopt;
  absl::optional<int> max_side_limit = absl::nullopt;
  absl::optional<float> thresh = absl::nullopt;
  absl::optional<float> box_thresh = absl::nullopt;
  absl::optional<float> unclip_ratio = absl::nullopt;
  absl::optional<std::vector<int>> input_shape = absl::nullopt;
};

class TextDetection {
public:
  TextDetection(const TextDetectionParams &params = TextDetectionParams());

  std::vector<std::unique_ptr<BaseCVResult>> Predict(const std::string &input) {
    std::vector<std::string> inputs = {input};
    return Predict(inputs);
  };
  std::vector<std::unique_ptr<BaseCVResult>>
  Predict(const std::vector<std::string> &input);

  void CreateModel();
  absl::Status CheckParams();
  static TextDetPredictorParams
  ToTextDetectionModelParams(const TextDetectionParams &from);

private:
  TextDetectionParams params_;
  std::unique_ptr<BasePredictor> model_infer_;
};
