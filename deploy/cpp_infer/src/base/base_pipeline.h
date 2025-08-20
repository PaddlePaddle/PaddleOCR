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

#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "base_cv_result.h"
#include "base_predictor.h"

class BasePipeline {
public:
  BasePipeline() = default;
  virtual ~BasePipeline() = default;

  std::vector<std::unique_ptr<BaseCVResult>> Predict(const std::string &input) {
    std::vector<std::string> inputs = {input};
    return Predict(inputs);
  }

  virtual std::vector<std::unique_ptr<BaseCVResult>>
  Predict(const std::vector<std::string> &input) = 0;

  template <typename T, typename... Args>
  std::unique_ptr<BasePredictor> CreateModule(Args &&...args);

  template <typename T, typename... Args>
  std::unique_ptr<BasePipeline> CreatePipeline(Args &&...args);
};

template <typename T, typename... Args>
std::unique_ptr<BasePredictor> BasePipeline::CreateModule(Args &&...args) {
  std::unique_ptr<BasePredictor> base_predictor =
      std::unique_ptr<T>(new T(std::forward<Args>(args)...));
  return base_predictor;
}

template <typename T, typename... Args>
std::unique_ptr<BasePipeline> BasePipeline::CreatePipeline(Args &&...args) {
  std::unique_ptr<BasePipeline> base_pipeline =
      std::unique_ptr<T>(new T(std::forward<Args>(args)...));
  return base_pipeline;
}
