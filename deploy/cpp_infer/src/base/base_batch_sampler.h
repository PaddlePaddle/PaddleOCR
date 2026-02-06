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
#include <type_traits>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

class BaseBatchSampler {
public:
  explicit BaseBatchSampler(int batch_size) : batch_size_(batch_size) {}
  virtual ~BaseBatchSampler() = default;

  int BatchSize() const;
  absl::Status SetBatchSize(int batch_size);

  template <typename T>
  absl::StatusOr<std::vector<std::vector<cv::Mat>>> Apply(const T &input);

  template <typename T>
  absl::StatusOr<std::vector<std::vector<cv::Mat>>> Sample(const T &input) {
    return absl::InvalidArgumentError(
        "Sample failed! Unsupported type for Sample");
  }

  virtual absl::StatusOr<std::vector<std::vector<cv::Mat>>>
  SampleFromString(const std::string &input) = 0;

  virtual absl::StatusOr<std::vector<std::vector<cv::Mat>>>
  SampleFromVector(const std::vector<std::string> &inputs) = 0;
  std::vector<std::string> InputPath() { return input_path_; };

  virtual absl::StatusOr<std::vector<std::vector<cv::Mat>>>
  SampleFromMatVector(const std::vector<cv::Mat> &inputs) = 0;

  absl::StatusOr<std::vector<std::vector<std::string>>>
  SampleFromStringToStringVector(const std::string &input);
  absl::StatusOr<std::vector<std::vector<std::string>>>
  SampleFromVectorToStringVector(const std::vector<std::string> &input);

  absl::StatusOr<std::vector<std::string>>
  GetFilesList(const std::string &path);

protected:
  int batch_size_ = 1;
  std::vector<std::string> input_path_;
};

template <typename T>
absl::StatusOr<std::vector<std::vector<cv::Mat>>>
BaseBatchSampler::Apply(const T &input) {
  return Sample(input);
}

template <>
inline absl::StatusOr<std::vector<std::vector<cv::Mat>>>
BaseBatchSampler::Sample<std::string>(const std::string &input) {
  return SampleFromString(input);
}

template <>
inline absl::StatusOr<std::vector<std::vector<cv::Mat>>>
BaseBatchSampler::Sample<std::vector<std::string>>(
    const std::vector<std::string> &input) {
  return SampleFromVector(input);
}

template <>
inline absl::StatusOr<std::vector<std::vector<cv::Mat>>>
BaseBatchSampler::Sample<std::vector<cv::Mat>>(
    const std::vector<cv::Mat> &input) {
  return SampleFromMatVector(input);
}
