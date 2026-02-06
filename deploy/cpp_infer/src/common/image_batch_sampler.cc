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

#include "image_batch_sampler.h"

#include <dirent.h>
#include <sys/stat.h>

#include <algorithm>
#include <cctype>
#include <iostream>

#include "src/utils/ilogger.h"
#include "src/utils/utility.h"

const std::set<std::string> ImageBatchSampler::kImgSuffixes = {"jpg", "png",
                                                               "jpeg", "bmp"};

ImageBatchSampler::ImageBatchSampler(int batch_size)
    : BaseBatchSampler(batch_size) {}

absl::StatusOr<std::vector<std::vector<cv::Mat>>>
ImageBatchSampler::SampleFromString(const std::string &input) {
  std::vector<std::string> inputs = {input};
  return SampleFromVector(inputs);
}

absl::StatusOr<std::vector<std::vector<cv::Mat>>>
ImageBatchSampler::SampleFromVector(const std::vector<std::string> &inputs) {
  std::vector<std::vector<cv::Mat>> results;
  std::vector<cv::Mat> current_batch;
  input_path_.clear();
  for (size_t i = 0; i < inputs.size(); ++i) {
    const std::string &input = inputs[i];

    if (Utility::IsDirectory(input)) {
      absl::StatusOr<std::vector<std::string>> files_result =
          GetFilesList(input);
      if (!files_result.ok()) {
        return files_result.status();
      }
      input_path_.insert(input_path_.end(), files_result.value().begin(),
                         files_result.value().end());
      absl::StatusOr<std::vector<std::vector<cv::Mat>>> sub_result =
          SampleFromVector(files_result.value());
      if (!sub_result.ok()) {
        return sub_result.status();
      }

      const std::vector<std::vector<cv::Mat>> &sub_batches = sub_result.value();
      for (size_t j = 0; j < sub_batches.size(); ++j) {
        results.push_back(sub_batches[j]);
      }
    } else if (Utility::IsImageFile(input)) {
      if (!Utility::FileExists(input).ok()) {
        return absl::NotFoundError("File not found: " + input);
      }
      input_path_.push_back(input);
      absl::StatusOr<cv::Mat> image_result = Utility::MyLoadImage(input);
      if (!image_result.ok()) {
        return image_result.status();
      }

      current_batch.push_back(image_result.value());

      if (static_cast<int>(current_batch.size()) == batch_size_) {
        results.push_back(current_batch);
        current_batch.clear();
      }
    } else {
      return absl::InvalidArgumentError("Unsupported file type: " + input);
    }
  }

  if (!current_batch.empty()) {
    results.push_back(current_batch); // last batch
  }
  return results;
}

absl::StatusOr<std::vector<std::vector<cv::Mat>>>
ImageBatchSampler::SampleFromMatVector(const std::vector<cv::Mat> &inputs) {
  std::vector<std::vector<cv::Mat>> results;
  std::vector<cv::Mat> current_batch;

  for (size_t i = 0; i < inputs.size(); ++i) {
    const cv::Mat &image = inputs[i];

    if (image.empty()) {
      return absl::InvalidArgumentError("Input image at index " +
                                        std::to_string(i) + " is empty.");
    }

    current_batch.push_back(image);

    if (static_cast<int>(current_batch.size()) == batch_size_) {
      results.push_back(current_batch);
      current_batch.clear();
    }
  }

  if (!current_batch.empty()) {
    results.push_back(current_batch);
  }

  return results;
}
