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

#include "src/modules/image_unwarping/predictor.h"

struct TextImageUnwarpingParams {
  absl::optional<std::string> model_name = absl::nullopt;
  absl::optional<std::string> model_dir = absl::nullopt;
  absl::optional<std::string> device = absl::nullopt;
  bool enable_mkldnn = true;
  std::string precision = "fp32";
  int mkldnn_cache_capacity = 10;
  int cpu_threads = 8;

  int batch_size = 1;
};

class TextImageUnwarping {
public:
  TextImageUnwarping(
      const TextImageUnwarpingParams &params = TextImageUnwarpingParams());

  std::vector<std::unique_ptr<BaseCVResult>> Predict(const std::string &input) {
    std::vector<std::string> inputs = {input};
    return Predict(inputs);
  };
  std::vector<std::unique_ptr<BaseCVResult>>
  Predict(const std::vector<std::string> &input);

  void CreateModel();
  absl::Status CheckParams();
  static WarpPredictorParams
  ToTextImageUnwarpingModelParams(const TextImageUnwarpingParams &from);

private:
  TextImageUnwarpingParams params_;
  std::unique_ptr<BasePredictor> model_infer_;
};
