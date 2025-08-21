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
#include <string>
#include <unordered_map>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/optional.h"
#include "src/base/base_pipeline.h"
#include "src/common/image_batch_sampler.h"
#include "src/common/parallel.h"
#include "src/common/processors.h"
#include "src/utils/ilogger.h"
#include "src/utils/utility.h"

struct DocPreprocessorPipelineResult {
  std::string input_path = "";
  cv::Mat input_image;
  std::unordered_map<std::string, bool> model_settings;
  int angle = 0;
  cv::Mat rotate_image;
  cv::Mat output_image;
  cv::Mat image_all;
};

struct DocPreprocessorPipelineParams {
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

class _DocPreprocessorPipeline : public BasePipeline {
public:
  explicit _DocPreprocessorPipeline(
      const DocPreprocessorPipelineParams &params);
  virtual ~_DocPreprocessorPipeline() = default;

  _DocPreprocessorPipeline() = delete;

  std::vector<std::unique_ptr<BaseCVResult>>
  Predict(const std::vector<std::string> &input) override;

  std::unordered_map<std::string, bool> GetModelSettings(
      absl::optional<bool> use_doc_orientation_classify = absl::nullopt,
      absl::optional<bool> use_doc_unwarping = absl::nullopt) const;
  absl::Status CheckModelSettingsVaild(
      std::unordered_map<std::string, bool> model_settings) const;

  std::vector<DocPreprocessorPipelineResult> PipelineResult() const {
    return pipeline_result_vec_;
  };

  void OverrideConfig();

private:
  bool use_doc_orientation_classify_;
  bool use_doc_unwarping_;
  std::unique_ptr<BasePredictor> doc_ori_classify_model_;
  std::unique_ptr<BasePredictor> doc_unwarping_model_;
  DocPreprocessorPipelineParams params_;
  YamlConfig config_;
  std::unique_ptr<BaseBatchSampler> batch_sampler_ptr_;
  std::vector<DocPreprocessorPipelineResult> pipeline_result_vec_;
};

class DocPreprocessorPipeline
    : public AutoParallelSimpleInferencePipeline<
          _DocPreprocessorPipeline, DocPreprocessorPipelineParams,
          std::vector<std::string>,
          std::vector<std::unique_ptr<BaseCVResult>>> {
public:
  DocPreprocessorPipeline(const DocPreprocessorPipelineParams &params)
      : AutoParallelSimpleInferencePipeline(params),
        thread_num_(params.thread_num) {
    if (thread_num_ == 1) {
      infer_ =
          std::unique_ptr<BasePipeline>(new _DocPreprocessorPipeline(params));
    }
  };

  std::vector<std::unique_ptr<BaseCVResult>>
  Predict(const std::vector<std::string> &input) override;

private:
  int thread_num_;
  std::unique_ptr<BasePipeline> infer_;
  std::unique_ptr<BaseBatchSampler> batch_sampler_ptr_;
};
