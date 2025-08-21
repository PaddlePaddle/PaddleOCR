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

#include "processors.h"

#include <algorithm>
#include <stdexcept>

#include "processors.h"
#include "src/utils/utility.h"

Crop::Crop(const std::vector<int> crop_size, const std::string &mode)
    : mode_(mode) {
  if (crop_size.size() == 1) {
    crop_size_ = std::vector<int>(2, crop_size[0]);
  } else {
    crop_size_ = crop_size;
  }
  assert(mode_ == "Center" || mode_ == "TopLeft");
  assert(crop_size_.size() == 2 && crop_size_[0] > 0 && crop_size_[1] > 0);
}

Crop::Crop(const int crop_size, const std::string &mode)
    : crop_size_(2, crop_size), mode_(mode) {
  assert(mode_ == "Center" || mode_ == "TopLeft");
  assert(crop_size_.size() == 2 && crop_size_[0] > 0 && crop_size_[1] > 0);
}

absl::StatusOr<cv::Mat> Crop::CropImage(const cv::Mat &img) const {
  int h = img.rows;
  int w = img.cols;
  int crop_width = crop_size_[0];
  int crop_height = crop_size_[1];

  if (w < crop_width || h < crop_height) {
    return absl::InvalidArgumentError(
        "Input image (" + std::to_string(w) + ", " + std::to_string(h) +
        ") smaller than target size (" + std::to_string(crop_width) + ", " +
        std::to_string(crop_height) + ").");
  }
  int x1 = 0, y1 = 0;
  if (mode_ == "Center") {
    x1 = std::max(0, (w - crop_width) / 2);
    y1 = std::max(0, (h - crop_height) / 2);
  } else if (mode_ == "TopLeft") {
    x1 = 0;
    y1 = 0;
  } else {
    return absl::InvalidArgumentError("Unsupported crop mode.");
  }

  cv::Rect roi(x1, y1, crop_width, crop_height);
  return img(roi).clone();
}

absl::StatusOr<std::vector<cv::Mat>> Crop::Apply(std::vector<cv::Mat> &imgs,
                                                 const void *param) const {
  std::vector<cv::Mat> result;
  result.reserve(imgs.size());
  for (const auto &img : imgs) {
    auto cropped = CropImage(img);
    if (!cropped.ok())
      return cropped.status();
    result.push_back(cropped.value());
  }
  return result;
}

Topk::Topk(const std::vector<std::string> &class_names, const int topk)
    : class_names_(class_names), topk_(topk) {}

absl::StatusOr<std::vector<Topk::TopkOutput>> Topk::Apply(const cv::Mat &preds,
                                                          const int topk) {
  topk_ = topk;
  auto preds_batch = Utility::SplitBatch(preds);
  if (!preds_batch.ok()) {
    return preds_batch.status();
  }
  std::vector<TopkOutput> topk_results = {};
  topk_results.reserve(preds_batch.value().size());
  for (const auto &pred : preds_batch.value()) {
    auto topk_result = Process(pred);
    if (!topk_result.ok()) {
      return topk_result.status();
    }
    topk_results.push_back(topk_result.value());
  }
  return topk_results;
}

absl::StatusOr<Topk::TopkOutput> Topk::Process(const cv::Mat &pred) const {
  if (pred.dims != 2 || pred.type() != CV_32F) {
    return absl::InvalidArgumentError("Input scores must be 2-D float matrix.");
  }

  TopkOutput topk_result(pred.size[0]);
  int num_classes = pred.size[1];
  const float *row = pred.ptr<float>();

  std::vector<std::pair<float, int>> score_idx;
  for (int j = 0; j < num_classes; ++j) {
    score_idx.emplace_back(row[j], j);
  }
  std::partial_sort(
      score_idx.begin(), score_idx.begin() + topk_, score_idx.end(),
      [](const std::pair<float, int> &a, const std::pair<float, int> &b) {
        return a.first > b.first;
      });

  for (int t = 0; t < topk_; ++t) {
    topk_result.class_ids.push_back(score_idx[t].second);
    topk_result.scores.push_back(score_idx[t].first);
    if (!class_names_.empty() &&
        score_idx[t].second < (int)class_names_.size()) {
      topk_result.label_names.push_back(class_names_[score_idx[t].second]);
    } else {
      topk_result.label_names.push_back(std::to_string(score_idx[t].second));
    }
  }

  return topk_result;
}
