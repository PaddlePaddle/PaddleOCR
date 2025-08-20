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

#include <chrono>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <string>
#include <unordered_map>

#include "absl/status/statusor.h"

class ImageWriter {};

class BaseCVResult {
public:
  BaseCVResult(const std::string &backend);
  BaseCVResult() = default;
  virtual ~BaseCVResult() = default;
  std::string Str() const;
  std::unordered_map<std::string, cv::Mat> Img() const;
  // absl::Status Print() const;
  absl::Status SaveToImg() const;

  virtual void SaveToImg(const std::string &save_path) = 0;
  virtual void Print() const = 0;
  virtual void SaveToJson(const std::string &save_path) const = 0;

protected:
  std::unordered_map<std::string, std::string> res_;
  ImageWriter img_writer_;
  std::string ToStr() const;
  // virtual std::unordered_map<std::string, cv::Mat> ToImg() const = 0;
};
