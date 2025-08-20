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

#include <sstream>
#include <stdexcept>

#include "src/utils/utility.h"

DocTrPostProcess::DocTrPostProcess(double scale) : scale_(scale) {}

absl::StatusOr<std::vector<cv::Mat>>
DocTrPostProcess::Apply(const cv::Mat &preds) const {
  auto preds_batch = Utility::SplitBatch(preds);
  if (!preds_batch.ok()) {
    return preds_batch.status();
  }
  std::vector<cv::Mat> doc_out;
  doc_out.reserve(preds_batch.value().size());
  for (auto &pred_data : preds_batch.value()) {
    auto result = Process(pred_data);
    if (!result.ok()) {
      return result.status();
    }
    doc_out.push_back(result.value());
  }
  return doc_out;
}

absl::StatusOr<cv::Mat> DocTrPostProcess::Process(cv::Mat &pred_data) const {
  if (pred_data.dims != 4) {
    return absl::InvalidArgumentError("must have 4D"); //********
  }
  std::vector<int> shape = {};
  for (int i = 1; i < pred_data.dims; i++) {
    shape.push_back(pred_data.size[i]);
  }
  pred_data = pred_data.reshape(1, shape);
  std::vector<cv::Range> ranges(pred_data.size[0]);
  std::vector<cv::Mat> mat_split(pred_data.size[0]);

  for (int i = 0; i < pred_data.size[0]; i++) {
    ranges[0] = cv::Range(i, i + 1);
    for (int j = 1; j < pred_data.dims; j++) {
      ranges[j] = cv::Range::all();
    }
    mat_split[i] = pred_data(&ranges[0]);
  }
  for (auto &item : mat_split) {
    std::vector<int> shape_item = {};
    for (int i = 1; i < item.dims; i++) {
      shape_item.push_back(item.size[i]);
    }
    item = item.reshape(1, shape_item);
    item = item * scale_;
  }
  cv::Mat out_hwc;
  cv::merge(mat_split, out_hwc);
  out_hwc.convertTo(out_hwc, CV_8U);
  return out_hwc;
}
