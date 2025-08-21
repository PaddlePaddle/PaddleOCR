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

#include "text_detection.h"

#include "src/utils/args.h"
#include "src/utils/yaml_config.h"

#define COPY_PARAMS(field) to.field = from.field;

TextDetection::TextDetection(const TextDetectionParams &params)
    : params_(params) {
  OverrideConfig();
  auto status = CheckParams();
  if (!status.ok()) {
    INFOE("Init TextDetection fail : %s", status.ToString().c_str());
    exit(-1);
  }
  CreateModel();
};
std::vector<std::unique_ptr<BaseCVResult>>
TextDetection::Predict(const std::vector<std::string> &input) {
  return model_infer_->Predict(input);
}
void TextDetection::CreateModel() {
  model_infer_ = std::unique_ptr<BasePredictor>(
      new TextDetPredictor(ToTextDetectionModelParams(params_)));
}
void TextDetection::OverrideConfig() {
  if (!FLAGS_text_detection_model_name.empty()) {
    params_.model_name = FLAGS_text_detection_model_name;
  }
  if (!FLAGS_text_detection_model_dir.empty()) {
    params_.model_dir = FLAGS_text_detection_model_dir;
  }
  if (!FLAGS_text_det_limit_side_len.empty()) {
    params_.limit_side_len = std::stoi(FLAGS_text_det_limit_side_len);
  }
  if (!FLAGS_text_det_limit_type.empty()) {
    params_.limit_type = FLAGS_text_det_limit_type;
  }
  if (!FLAGS_text_det_thresh.empty()) {
    params_.thresh = std::stof(FLAGS_text_det_thresh);
  }
  if (!FLAGS_text_det_box_thresh.empty()) {
    params_.box_thresh = std::stof(FLAGS_text_det_box_thresh);
  }
  if (!FLAGS_text_det_unclip_ratio.empty()) {
    params_.unclip_ratio = std::stof(FLAGS_text_det_unclip_ratio);
  }
  if (!FLAGS_text_det_input_shape.empty()) {
    params_.input_shape =
        YamlConfig::SmartParseVector(FLAGS_text_det_input_shape).vec_int;
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

absl::Status TextDetection::CheckParams() {
  if (!params_.model_dir.has_value()) {
    return absl::NotFoundError("Require text detection model dir.");
  }
  return absl::OkStatus();
}

TextDetPredictorParams
TextDetection::ToTextDetectionModelParams(const TextDetectionParams &from) {
  TextDetPredictorParams to;
  COPY_PARAMS(model_name)
  COPY_PARAMS(model_dir)
  COPY_PARAMS(limit_side_len)
  COPY_PARAMS(limit_type)
  COPY_PARAMS(thresh)
  COPY_PARAMS(box_thresh)
  COPY_PARAMS(unclip_ratio)
  COPY_PARAMS(input_shape)
  COPY_PARAMS(device)
  COPY_PARAMS(enable_mkldnn)
  COPY_PARAMS(mkldnn_cache_capacity)
  COPY_PARAMS(precision)
  COPY_PARAMS(cpu_threads)
  return to;
}
