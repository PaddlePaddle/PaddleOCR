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

#include "text_recognition.h"

#include "src/utils/args.h"
#include "src/utils/yaml_config.h"

#define COPY_PARAMS(field) to.field = from.field;

TextRecognition::TextRecognition(const TextRecognitionParams &params)
    : params_(params) {
  OverrideConfig();
  auto status = CheckParams();
  if (!status.ok()) {
    INFOE("Init TextRecognition fail : %s", status.ToString().c_str());
    exit(-1);
  }
  CreateModel();
};
std::vector<std::unique_ptr<BaseCVResult>>
TextRecognition::Predict(const std::vector<std::string> &input) {
  return model_infer_->Predict(input);
}
void TextRecognition::CreateModel() {
  model_infer_ = std::unique_ptr<BasePredictor>(
      new TextRecPredictor(ToTextRecognitionModelParams(params_)));
}
void TextRecognition::OverrideConfig() {
  if (!FLAGS_text_recognition_model_name.empty()) {
    params_.model_name = FLAGS_text_recognition_model_name;
  }
  if (!FLAGS_text_recognition_model_dir.empty()) {
    params_.model_dir = FLAGS_text_recognition_model_dir;
  }
  if (!FLAGS_text_recognition_batch_size.empty()) {
    params_.batch_size = std::stoi(FLAGS_text_recognition_batch_size);
  }
  if (!FLAGS_text_rec_input_shape.empty()) {
    params_.input_shape =
        YamlConfig::SmartParseVector(FLAGS_text_rec_input_shape).vec_int;
  }
  if (!FLAGS_vis_font_dir.empty()) {
    params_.vis_font_dir = FLAGS_vis_font_dir;
  }
  if (!FLAGS_device.empty()) {
    params_.device = FLAGS_device;
  }
  if (!FLAGS_precision.empty()) {
    params_.precision = FLAGS_precision;
  }
  if (!FLAGS_enable_mkldnn.empty()) {
    params_.enable_mkldnn = Utility::StringToBool(FLAGS_enable_mkldnn);
  }
  if (!FLAGS_mkldnn_cache_capacity.empty()) {
    params_.mkldnn_cache_capacity = std::stoi(FLAGS_mkldnn_cache_capacity);
  }
  if (!FLAGS_cpu_threads.empty()) {
    params_.cpu_threads = std::stoi(FLAGS_cpu_threads);
  }
}

absl::Status TextRecognition::CheckParams() {
  if (!params_.model_dir.has_value()) {
    return absl::NotFoundError("Require text recognition model_dir.");
  }
  return absl::OkStatus();
}

TextRecPredictorParams TextRecognition::ToTextRecognitionModelParams(
    const TextRecognitionParams &from) {
  TextRecPredictorParams to;
  COPY_PARAMS(model_name)
  COPY_PARAMS(model_dir)
  COPY_PARAMS(batch_size)
  COPY_PARAMS(input_shape)
  COPY_PARAMS(vis_font_dir)
  COPY_PARAMS(device)
  COPY_PARAMS(enable_mkldnn)
  COPY_PARAMS(mkldnn_cache_capacity)
  COPY_PARAMS(precision)
  COPY_PARAMS(cpu_threads)
  return to;
}
