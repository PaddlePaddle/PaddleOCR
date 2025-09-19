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
#include "src/utils/func_register.h"

class Crop : public BaseProcessor {
public:
  explicit Crop(const std::vector<int> crop_size,
                const std::string &mode = "Center");
  explicit Crop(const int crop_size, const std::string &mode = "Center");
  absl::StatusOr<cv::Mat> CropImage(const cv::Mat &img) const;
  absl::StatusOr<std::vector<cv::Mat>>
  Apply(std::vector<cv::Mat> &input,
        const void *param = nullptr) const override;

private:
  std::vector<int> crop_size_;
  std::string mode_;
};

class Topk {
public:
  struct TopkOutput {
    TopkOutput(int batch) {
      class_ids.reserve(batch);
      scores.reserve(batch);
      label_names.reserve(batch);
    }
    std::vector<int> class_ids;
    std::vector<float> scores;
    std::vector<std::string> label_names;
  };
  explicit Topk(
      const std::vector<std::string> &class_names = std::vector<std::string>(),
      int topk = 1);

  absl::StatusOr<TopkOutput> Process(const cv::Mat &pred_data) const;
  absl::StatusOr<std::vector<TopkOutput>> Apply(const cv::Mat &preds,
                                                const int topk = 1);

private:
  std::vector<std::string> class_names_;
  int topk_;
};
