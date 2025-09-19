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

#include "src/pipelines/doc_preprocessor/pipeline.h"

struct DocPreprocessorParams {
  absl::optional<std::string> doc_orientation_classify_model_name =
      absl::nullopt;
  absl::optional<std::string> doc_orientation_classify_model_dir =
      absl::nullopt;
  absl::optional<std::string> doc_unwarping_model_name = absl::nullopt;
  absl::optional<std::string> doc_unwarping_model_dir = absl::nullopt;
  absl::optional<bool> use_doc_orientation_classify = absl::nullopt;
  absl::optional<bool> use_doc_unwarping = absl::nullopt;
  absl::optional<std::string> device = absl::nullopt;
  bool enable_mkldnn = true;
  int mkldnn_cache_capacity = 10;
  std::string precision = "fp32";
  int cpu_threads = 8;
  int thread_num = 1;
  absl::optional<Utility::PaddleXConfigVariant> paddlex_config = absl::nullopt;
};

class DocPreprocessor {
public:
  DocPreprocessor(
      const DocPreprocessorParams &params = DocPreprocessorParams());

  std::vector<std::unique_ptr<BaseCVResult>> Predict(const std::string &input) {
    std::vector<std::string> inputs = {input};
    return Predict(inputs);
  };
  std::vector<std::unique_ptr<BaseCVResult>>
  Predict(const std::vector<std::string> &input);

  void CreatePipeline();
  absl::Status CheckParams();
  static DocPreprocessorPipelineParams
  ToDocPreprocessorPipelineParams(const DocPreprocessorParams &from);

private:
  DocPreprocessorParams params_;
  std::unique_ptr<BasePipeline> pipeline_infer_;
};
