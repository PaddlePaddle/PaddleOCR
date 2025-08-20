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

#include "base_batch_sampler.h"

#include "src/utils/ilogger.h"
#include "src/utils/utility.h"
int BaseBatchSampler::BatchSize() const { return batch_size_; }

absl::Status BaseBatchSampler::SetBatchSize(int batch_size) {
  if (batch_size <= 0) {
    return absl::InvalidArgumentError("Batch size must be greater than 0");
  }
  batch_size_ = batch_size;
  return absl::OkStatus();
}

absl::StatusOr<std::vector<std::vector<std::string>>>
BaseBatchSampler::SampleFromVectorToStringVector(
    const std::vector<std::string> &inputs) {
  std::vector<std::vector<std::string>> result;
  std::vector<std::string> current_batch;

  for (size_t i = 0; i < inputs.size(); ++i) {
    const std::string &input = inputs[i];

    if (Utility::IsDirectory(input)) {
      absl::StatusOr<std::vector<std::string>> files_result =
          GetFilesList(input);
      if (!files_result.ok()) {
        return files_result.status();
      }

      absl::StatusOr<std::vector<std::vector<std::string>>> sub_result =
          SampleFromVectorToStringVector(files_result.value());
      if (!sub_result.ok()) {
        return sub_result.status();
      }

      const std::vector<std::vector<std::string>> &sub_batches =
          sub_result.value();
      for (size_t j = 0; j < sub_batches.size(); ++j) {
        result.push_back(sub_batches[j]);
      }
    } else if (Utility::IsImageFile(input)) {
      if (!Utility::FileExists(input).ok()) {
        return absl::NotFoundError("File not found: " + input);
      }
      current_batch.push_back(input);

      if (static_cast<int>(current_batch.size()) == batch_size_) {
        result.push_back(current_batch);
        current_batch.clear();
      }
    } else {
      return absl::InvalidArgumentError("Unsupported file type: " + input);
    }
  }

  if (!current_batch.empty()) {
    result.push_back(current_batch); // last batch
  }

  return result;
}

absl::StatusOr<std::vector<std::vector<std::string>>>
BaseBatchSampler::SampleFromStringToStringVector(const std::string &input) {
  std::vector<std::string> inputs = {input};
  return SampleFromVectorToStringVector(inputs);
}

absl::StatusOr<std::vector<std::string>>
BaseBatchSampler::GetFilesList(const std::string &path) {
  if (!Utility::FileExists(path).ok()) {
    return absl::NotFoundError("Path not found: " + path);
  }

  std::vector<std::string> file_list;

  if (!Utility::IsDirectory(path)) {
    if (Utility::IsImageFile(path)) {
      file_list.push_back(path);
    }
  } else {
    Utility::GetFilesRecursive(path, file_list);
  }

  if (file_list.empty()) {
    return absl::NotFoundError("No image files found in path: " + path);
  }

  std::sort(file_list.begin(), file_list.end());
  return file_list;
}
