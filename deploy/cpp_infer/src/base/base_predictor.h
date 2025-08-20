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

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/optional.h"
#include "base_batch_sampler.h"
#include "base_cv_result.h"
#include "src/common/static_infer.h"
#include "src/utils/func_register.h"
#include "src/utils/pp_option.h"
#include "src/utils/yaml_config.h"

class BasePredictor {
public:
  BasePredictor(const absl::optional<std::string> &model_dir = absl::nullopt,
                const absl::optional<std::string> &model_name = absl::nullopt,
                const absl::optional<std::string> &device = absl::nullopt,
                const std::string &precision = "fp32",
                const bool enable_mkldnn = true,
                int mkldnn_cache_capacityint = 10, int cpu_threads = 8,
                int batch_size = 1, const std::string sample_type = "");
  virtual ~BasePredictor() = default;
  std::vector<std::unique_ptr<BaseCVResult>> Predict(const std::string &input);

  template <typename T>
  std::vector<std::unique_ptr<BaseCVResult>> Predict(const T &input);

  std::unique_ptr<PaddleInfer> CreateStaticInfer();

  const PaddlePredictorOption &PPOption();
  absl::StatusOr<std::string> ModelName() { return model_name_; };
  std::string ConfigPath() { return config_.ConfigYamlPath(); };

  void SetBatchSize(int batch_size);

  virtual std::vector<std::unique_ptr<BaseCVResult>>
  Process(std::vector<cv::Mat> &batch_data) = 0;
  virtual void ResetResult() = 0;
  absl::Status BuildBatchSampler();

  void SetInputPath(const std::vector<std::string> &input_path) {
    input_path_ = input_path;
  };

  template <typename T, typename... Args>
  void Register(const std::string &key, Args &&...args);

  static constexpr const char *MODEL_FILE_PREFIX = "inference";
  static const std::unordered_set<std::string> SAMPLER_TYPE;
  static bool print_flag;

protected:
  absl::optional<std::string> model_dir_;
  YamlConfig config_;
  int batch_size_;
  std::unique_ptr<BaseBatchSampler> batch_sampler_ptr_;
  std::unique_ptr<PaddlePredictorOption> pp_option_ptr_;
  std::vector<std::string> input_path_;
  std::string model_name_;
  std::string sampler_type_;
  std::unordered_map<std::string, std::unique_ptr<BaseProcessor>> pre_op_;
};

template <typename T, typename... Args>
void BasePredictor::Register(const std::string &key, Args &&...args) {
  auto instance = std::unique_ptr<T>(new T(std::forward<Args>(args)...));
  pre_op_[key] = std::move(instance);
};

template <typename T>
std::vector<std::unique_ptr<BaseCVResult>>
BasePredictor::Predict(const T &input) {
  std::vector<std::unique_ptr<BaseCVResult>> result;
  ResetResult();
  auto batches = batch_sampler_ptr_->Apply(input);
  if (!batches.ok()) {
    INFOE("Get sample fail : %s", batches.status().ToString().c_str());
    exit(-1);
  }
  input_path_ = batch_sampler_ptr_->InputPath();
  for (auto &batch_data : batches.value()) {
    auto predictions = Process(batch_data);
    for (auto &prediction : predictions) {
      result.emplace_back(std::move(prediction));
    }
  }
  return result;
}
