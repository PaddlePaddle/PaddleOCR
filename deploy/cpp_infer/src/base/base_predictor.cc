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

#include "base_predictor.h"

#include <yaml-cpp/yaml.h>

#include <iostream>

#include "base_batch_sampler.h"
#include "src/common/image_batch_sampler.h"
#include "src/utils/ilogger.h"
#include "src/utils/pp_option.h"
#include "src/utils/utility.h"

BasePredictor::BasePredictor(const absl::optional<std::string> &model_dir,
                             const absl::optional<std::string> &model_name,
                             const absl::optional<std::string> &device,
                             const std::string &precision,
                             const bool enable_mkldnn,
                             int mkldnn_cache_capacityint, int cpu_threads,
                             int batch_size, const std::string sampler_type)
    : model_dir_(model_dir), batch_size_(batch_size),
      sampler_type_(sampler_type) {
  if (model_dir_.has_value()) {
    config_ = YamlConfig(model_dir_.value());
  } else {
    INFOE("Model dir is empty.");
    exit(-1);
  }
  auto status_build = BuildBatchSampler();
  if (!status_build.ok()) {
    INFOE("Build sampler fail: %s", status_build.ToString().c_str());
    exit(-1);
  }
  auto model_name_config = config_.GetString(std::string("Global.model_name"));
  if (!model_name_config.ok()) {
    INFOE(model_name_config.status().ToString().c_str());
    exit(-1);
  }
  model_name_ = model_name_config.value();
  if (model_name.has_value()) {
    if (model_name_ != model_name.value()) {
      INFOE(
          "Model name mismatch, please input the correct model dir. model dir "
          "is %s, but model name is %s",
          model_dir_.value().c_str(), model_name.value().c_str());
      exit(-1);
    }
  }
  model_name_ = model_name.value_or(model_name_);
  pp_option_ptr_.reset(new PaddlePredictorOption());
  auto device_result = device.value_or(DEVICE);

  size_t pos = device_result.find(':');
  std::string device_type = "";
  int device_id = 0;
  if (pos != std::string::npos) {
    device_type = device_result.substr(0, pos);
    device_id = std::stoi(device_result.substr(pos + 1));
  } else {
    device_type = device_result;
    device_id = 0;
  }
  auto status_device_type = pp_option_ptr_->SetDeviceType(device_type);
  if (!status_device_type.ok()) {
    INFOE("Failed to set device : %s", status_device_type.ToString().c_str());
    exit(-1);
    ;
  }
  auto status_device_id = pp_option_ptr_->SetDeviceId(device_id);
  if (!status_device_id.ok()) {
    INFOE("Failed to set device id: %s", status_device_id.ToString().c_str());
    exit(-1);
    ;
  }

  if (enable_mkldnn && device_type == "cpu") {
    if (precision == "fp16") {
      INFOW("When MKLDNN is enabled, FP16 precision is not supported.The "
            "computation will proceed with FP32 instead.");
    }
    if (Utility::IsMkldnnAvailable()) {
      auto status_mkldnn = pp_option_ptr_->SetRunMode("mkldnn");
      if (!status_mkldnn.ok()) {
        INFOE("Failed to set run mode: %s", status_mkldnn.ToString().c_str());
        exit(-1);
        ;
      }
    } else {
      INFOW("Mkldnn is not available, using paddle instead!");
      auto status_paddle = pp_option_ptr_->SetRunMode("paddle");
      if (!status_paddle.ok()) {
        INFOE("Failed to set run mode: %s", status_paddle.ToString().c_str());
        exit(-1);
      }
    }
  } else if (precision == "fp16") {
    if (precision == "fp16") {
      auto status_paddle_fp16 = pp_option_ptr_->SetRunMode("paddle_fp16");
      if (!status_paddle_fp16.ok()) {
        INFOE("Failed to set run mode: %s",
              status_paddle_fp16.ToString().c_str());
        exit(-1);
        ;
      }
    }
  } else {
    auto status_paddle = pp_option_ptr_->SetRunMode("paddle");
    if (!status_paddle.ok()) {
      INFOE("Failed to set run mode: %s", status_paddle.ToString().c_str());
      exit(-1);
    }
  }
  auto status_mkldnn_cache_capacityint =
      pp_option_ptr_->SetMkldnnCacheCapacity(mkldnn_cache_capacityint);
  if (!status_mkldnn_cache_capacityint.ok()) {
    INFOE("Set status_mkldnn_cache_capacityint fail : %s",
          status_mkldnn_cache_capacityint.ToString().c_str());
    exit(-1);
  }
  auto status_cpu_threads = pp_option_ptr_->SetCpuThreads(cpu_threads);
  if (!status_cpu_threads.ok()) {
    INFOE("Set cpu threads fail : %s", status_cpu_threads.ToString().c_str());
    exit(-1);
  }
  if (print_flag) {
    INFO(pp_option_ptr_->DebugString().c_str());
    print_flag = false;
  }
  INFO("Create model: %s.", model_name_.c_str());
}

std::vector<std::unique_ptr<BaseCVResult>>
BasePredictor::Predict(const std::string &input) {
  std::vector<std::string> inputs = {input};
  return Predict(inputs);
}

const PaddlePredictorOption &BasePredictor::PPOption() {
  return *pp_option_ptr_;
}

void BasePredictor::SetBatchSize(int batch_size) { batch_size_ = batch_size; }

std::unique_ptr<PaddleInfer> BasePredictor::CreateStaticInfer() {
  return std::unique_ptr<PaddleInfer>(new PaddleInfer(
      model_name_, model_dir_.value(), MODEL_FILE_PREFIX, PPOption()));
}

absl::Status BasePredictor::BuildBatchSampler() {
  if (SAMPLER_TYPE.count(sampler_type_) == 0) {
    return absl::InvalidArgumentError("Unsupported sampler type !");
  } else if (sampler_type_ == "image") {
    batch_sampler_ptr_ =
        std::unique_ptr<BaseBatchSampler>(new ImageBatchSampler(batch_size_));
  }
  return absl::OkStatus();
}

const std::unordered_set<std::string> BasePredictor::SAMPLER_TYPE = {
    "image",
};

bool BasePredictor::print_flag = true;
