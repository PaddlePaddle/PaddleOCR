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

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "paddle_inference_api.h"
#include "src/utils/ilogger.h"
#include "src/utils/pp_option.h"
class PaddleInfer {
public:
  explicit PaddleInfer(const std::string &model_name,
                       const std::string &model_dir,
                       const std::string &model_file_prefix,
                       const PaddlePredictorOption &option);
  ~PaddleInfer() = default;
  absl::StatusOr<std::vector<cv::Mat>>
  Apply(const std::vector<cv::Mat> &x); //***********

private:
  std::string model_dir_;
  std::string model_file_prefix_;
  std::string model_name_;
  PaddlePredictorOption option_;
  std::shared_ptr<paddle_infer::Predictor> predictor_;

  std::vector<std::unique_ptr<paddle_infer::Tensor>> input_handles_;
  std::vector<std::unique_ptr<paddle_infer::Tensor>> output_handles_;

  absl::StatusOr<std::shared_ptr<paddle_infer::Predictor>> Create();

  absl::Status CheckRunMode();
};
