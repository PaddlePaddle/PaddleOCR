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

#include "static_infer.h"

#include <fstream>

#include "src/utils/ilogger.h"
#include "src/utils/mkldnn_blocklist.h"
#include "src/utils/utility.h"

PaddleInfer::PaddleInfer(const std::string &model_name,
                         const std::string &model_dir,
                         const std::string &model_file_prefix,
                         const PaddlePredictorOption &option)
    : model_name_(model_name), model_dir_(model_dir),
      model_file_prefix_(model_file_prefix), option_(option) {
  auto result = Create();
  if (!result.ok()) {
    INFOE("Create predictor failed: %s", result.status().ToString().c_str());
    exit(-1);
  }

  predictor_ = std::move(result.value());
  auto input_names = predictor_->GetInputNames();
  for (const auto &name : input_names) {
    auto handle = predictor_->GetInputHandle(name);
    input_handles_.emplace_back(std::move(handle));
  }
  auto output_names = predictor_->GetOutputNames();
  for (const auto &name : output_names) {
    auto handle = predictor_->GetOutputHandle(name);
    output_handles_.emplace_back(std::move(handle));
  }
}

absl::StatusOr<std::shared_ptr<paddle_infer::Predictor>> PaddleInfer::Create() {
  auto model_paths = Utility::GetModelPaths(model_dir_, model_file_prefix_);
  if (!model_paths.ok()) {
    return model_paths.status();
  }
  if (model_paths->find("paddle") == model_paths->end()) {
    return absl::NotFoundError("No valid PaddlePaddle model found");
  }

  auto result_check = CheckRunMode();
  if (!result_check.ok()) {
    return result_check;
  }

  auto model_files = model_paths.value()["paddle"];
  std::string model_file = model_files.first;
  std::string params_file = model_files.second;

  if (option_.DeviceType() == "cpu" && option_.DeviceId() >= 0) {
    auto result_set = option_.SetDeviceId(0);
    if (!result_set.ok()) {
      return result_set;
    }
    INFO("`device_id` has been set to nullptr");
  }

  if (option_.DeviceType() == "gpu" && option_.DeviceId() < 0) {
    auto result_device_id = option_.SetDeviceId(0);
    if (!result_device_id.ok()) {
      return result_device_id;
    }
    INFO("`device_id` has been set to 0");
  }

  paddle_infer::Config config;
  config.SetModel(model_file, params_file);

  if (option_.DeviceType() == "gpu") {
    std::unordered_set<std::string> mixed_op_set = {"feed", "fetch"};
    config.Exp_DisableMixedPrecisionOps(mixed_op_set);

    paddle_infer::PrecisionType precision =
        paddle_infer::PrecisionType::kFloat32;
    if (option_.RunMode() == "paddle_fp16") {
      precision = paddle_infer::PrecisionType::kHalf;
    }

    config.DisableMKLDNN();
    config.EnableUseGpu(100, option_.DeviceId(), precision);
    config.EnableNewIR(option_.EnableNewIR());
    if (option_.EnableNewIR() && option_.EnableCinn()) {
      config.EnableCINN();
    }
    config.EnableNewExecutor();
    config.SetOptimizationLevel(3);
  } else if (option_.DeviceType() == "cpu") {
    config.DisableGpu();
    if (option_.RunMode().find("mkldnn") != std::string::npos) {
      config.EnableMKLDNN();
      if (option_.RunMode().find("bf16") != std::string::npos) {
        config.EnableMkldnnBfloat16();
      }
      config.SetMkldnnCacheCapacity(option_.MkldnnCacheCapacity());
    } else {
      config.DisableMKLDNN();
    }
    config.SetCpuMathLibraryNumThreads(option_.CpuThreads());
    config.EnableNewIR(option_.EnableNewIR());
    config.EnableNewExecutor();
    config.SetOptimizationLevel(3);
  } else {
    return absl::InvalidArgumentError("Not supported device type: " +
                                      option_.DeviceType());
  }

  config.EnableMemoryOptim();
  for (const auto &del_p : option_.DeletePass()) {
    config.DeletePass(del_p);
  }
  config.DisableGlogInfo();

  auto predictor_shared = paddle_infer::CreatePredictor(config);

  return predictor_shared;
};

absl::StatusOr<std::vector<cv::Mat>>
PaddleInfer::Apply(const std::vector<cv::Mat> &x) {
  for (size_t i = 0; i < x.size(); ++i) {
    auto &input_handle = input_handles_[i];
    std::vector<int> input_shape(x[0].dims);
    for (int i = 0; i < x[0].dims; i++) {
      input_shape[i] = x[0].size[i];
    }
    input_handle->Reshape(input_shape);
    input_handle->CopyFromCpu<float>((float *)x[i].data);
  }
  try {
    predictor_->Run();
  } catch (const std::exception &e) {
    INFOE("static Infer fail: %s", e.what());
    exit(-1);
  }

  std::vector<std::vector<float>> outputs;
  std::vector<int> output_shape = {};
  for (auto &output_handle : output_handles_) {
    output_shape = output_handle->shape();
    size_t numel = 1;
    for (auto dim : output_shape)
      numel *= dim;
    std::vector<float> out_data(numel);
    output_handle->CopyToCpu(out_data.data());
    outputs.push_back(std::move(out_data));
  }
  auto size_v = outputs[0].size();
  cv::Mat pred(output_shape.size(), output_shape.data(), CV_32F);
  memcpy(pred.ptr<float>(), outputs[0].data(),
         outputs[0].size() * sizeof(float));
  std::vector<cv::Mat> pred_outputs = {pred};
  return pred_outputs;
};

absl::Status PaddleInfer::CheckRunMode() {
  if (option_.RunMode().rfind("mkldnn", 0) == 0 &&
      Mkldnn::MKLDNN_BLOCKLIST.count(model_name_) > 0 &&
      option_.DeviceType() == "cpu") {
    INFOW("The model %s is not supported to run in MKLDNN mode! Using `paddle` "
          "instead!",
          model_name_.c_str());

    auto result = option_.SetRunMode("paddle");
    if (!result.ok()) {
      return result;
    }
  }
  if (model_name_ == "LaTeX_OCR_rec" && option_.DeviceType() == "cpu") {
    if (Utility::IsMkldnnAvailable() && option_.RunMode() != "mkldnn") {
      INFOE("Now, the `LaTeX_OCR_rec` model only support `mkldnn` mode when "
            "running on Intel CPU devices. So using `mkldnn` instead.");
      exit(-1);
      auto result = option_.SetRunMode("mkldnn");
      if (!result.ok()) {
        return result;
      }
    }
  }

  return absl::OkStatus();
};
