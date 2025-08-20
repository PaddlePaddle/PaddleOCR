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

#include "ocr.h"

#include "src/utils/args.h"
#include "src/utils/yaml_config.h"

#define COPY_PARAMS(field) to.field = from.field;

PaddleOCR::PaddleOCR(const PaddleOCRParams &params) : params_(params) {
  OverrideConfig();
  auto status = CheckParams();
  if (!status.ok()) {
    INFOE("Init paddleOCR fail : %s", status.ToString().c_str());
    exit(-1);
  }
  CreatePipeline();
};
std::vector<std::unique_ptr<BaseCVResult>>
PaddleOCR::Predict(const std::vector<std::string> &input) {
  return pipeline_infer_->Predict(input);
}
void PaddleOCR::CreatePipeline() {
  pipeline_infer_ = std::unique_ptr<BasePipeline>(
      new OCRPipeline(ToOCRPipelineParams(params_)));
}
void PaddleOCR::OverrideConfig() {
  if (!FLAGS_doc_orientation_classify_model_name.empty()) {
    params_.doc_orientation_classify_model_name =
        FLAGS_doc_orientation_classify_model_name;
  }
  if (!FLAGS_doc_orientation_classify_model_dir.empty()) {
    params_.doc_orientation_classify_model_dir =
        FLAGS_doc_orientation_classify_model_dir;
  }
  if (!FLAGS_doc_unwarping_model_name.empty()) {
    params_.doc_unwarping_model_name = FLAGS_doc_unwarping_model_name;
  }
  if (!FLAGS_doc_unwarping_model_dir.empty()) {
    params_.doc_unwarping_model_dir = FLAGS_doc_unwarping_model_dir;
  }
  if (!FLAGS_text_detection_model_name.empty()) {
    params_.text_detection_model_name = FLAGS_text_detection_model_name;
  }
  if (!FLAGS_text_detection_model_dir.empty()) {
    params_.text_detection_model_dir = FLAGS_text_detection_model_dir;
  }
  if (!FLAGS_textline_orientation_model_name.empty()) {
    params_.textline_orientation_model_name =
        FLAGS_textline_orientation_model_name;
  }
  if (!FLAGS_textline_orientation_model_dir.empty()) {
    params_.textline_orientation_model_dir =
        FLAGS_textline_orientation_model_dir;
  }
  if (!FLAGS_textline_orientation_batch_size.empty()) {
    params_.textline_orientation_batch_size =
        std::stoi(FLAGS_textline_orientation_batch_size);
  }
  if (!FLAGS_text_recognition_model_name.empty()) {
    params_.text_recognition_model_name = FLAGS_text_recognition_model_name;
  }
  if (!FLAGS_text_recognition_model_dir.empty()) {
    params_.text_recognition_model_dir = FLAGS_text_recognition_model_dir;
  }
  if (!FLAGS_text_recognition_batch_size.empty()) {
    params_.text_recognition_batch_size =
        std::stoi(FLAGS_text_recognition_batch_size);
  }
  if (!FLAGS_use_doc_orientation_classify.empty()) {
    params_.use_doc_orientation_classify =
        Utility::StringToBool(FLAGS_use_doc_orientation_classify);
  }
  if (!FLAGS_use_doc_unwarping.empty()) {
    params_.use_doc_unwarping = Utility::StringToBool(FLAGS_use_doc_unwarping);
  }
  if (!FLAGS_use_textline_orientation.empty()) {
    params_.use_textline_orientation =
        Utility::StringToBool(FLAGS_use_textline_orientation);
  }
  if (!FLAGS_text_det_limit_side_len.empty()) {
    params_.text_det_limit_side_len = std::stoi(FLAGS_text_det_limit_side_len);
  }
  if (!FLAGS_text_det_limit_type.empty()) {
    params_.text_det_limit_type = FLAGS_text_det_limit_type;
  }
  if (!FLAGS_text_det_thresh.empty()) {
    params_.text_det_thresh = std::stof(FLAGS_text_det_thresh);
  }
  if (!FLAGS_text_det_box_thresh.empty()) {
    params_.text_det_box_thresh = std::stof(FLAGS_text_det_box_thresh);
  }
  if (!FLAGS_text_det_unclip_ratio.empty()) {
    params_.text_det_unclip_ratio = std::stof(FLAGS_text_det_unclip_ratio);
  }
  if (!FLAGS_text_det_input_shape.empty()) {
    params_.text_det_input_shape =
        YamlConfig::SmartParseVector(FLAGS_text_det_input_shape).vec_int;
  }
  if (!FLAGS_text_rec_score_thresh.empty()) {
    params_.text_rec_score_thresh = std::stof(FLAGS_text_rec_score_thresh);
  }
  if (!FLAGS_text_rec_input_shape.empty()) {
    params_.text_rec_input_shape =
        YamlConfig::SmartParseVector(FLAGS_text_rec_input_shape).vec_int;
  }
  if (!FLAGS_lang.empty()) {
    params_.lang = FLAGS_lang;
  }
  if (!FLAGS_ocr_version.empty()) {
    params_.ocr_version = FLAGS_ocr_version;
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
  if (!FLAGS_thread_num.empty()) {
    params_.thread_num = std::stoi(FLAGS_thread_num);
  }
  if (!FLAGS_paddlex_config.empty()) {
    params_.paddlex_config = FLAGS_paddlex_config;
  }
}

absl::Status PaddleOCR::CheckParams() {
  if (!params_.doc_orientation_classify_model_dir.has_value() &&
      !(params_.use_doc_orientation_classify.has_value() &&
        !params_.use_doc_orientation_classify.value())) {
    return absl::NotFoundError("Require doc orientation classify model dir.");
  }
  if (!params_.doc_unwarping_model_dir.has_value() &&
      !(params_.use_doc_unwarping.has_value() &&
        !params_.use_doc_unwarping.value())) {
    return absl::NotFoundError("Require doc unwarping model dir.");
  }
  if (!params_.textline_orientation_model_dir.has_value() &&
      !(params_.use_textline_orientation.has_value() &&
        !params_.use_textline_orientation.value())) {
    return absl::NotFoundError("Require textline orientation model_dir.");
  }
  if (!params_.text_detection_model_dir.has_value()) {
    return absl::NotFoundError("Require text detection model dir.");
  }
  if (!params_.text_recognition_model_dir.has_value()) {
    return absl::NotFoundError("Require text recognition model_dir.");
  }
  return absl::OkStatus();
}

OCRPipelineParams PaddleOCR::ToOCRPipelineParams(const PaddleOCRParams &from) {
  OCRPipelineParams to;
  COPY_PARAMS(doc_orientation_classify_model_name)
  COPY_PARAMS(doc_orientation_classify_model_dir)
  COPY_PARAMS(doc_unwarping_model_name)
  COPY_PARAMS(doc_unwarping_model_dir)
  COPY_PARAMS(text_detection_model_name)
  COPY_PARAMS(text_detection_model_dir)
  COPY_PARAMS(textline_orientation_model_name)
  COPY_PARAMS(textline_orientation_model_dir)
  COPY_PARAMS(textline_orientation_batch_size)
  COPY_PARAMS(text_recognition_model_name)
  COPY_PARAMS(text_recognition_model_dir)
  COPY_PARAMS(text_recognition_batch_size)
  COPY_PARAMS(use_doc_orientation_classify)
  COPY_PARAMS(use_doc_unwarping)
  COPY_PARAMS(use_textline_orientation)
  COPY_PARAMS(text_det_limit_side_len)
  COPY_PARAMS(text_det_limit_type)
  COPY_PARAMS(text_det_thresh)
  COPY_PARAMS(text_det_box_thresh)
  COPY_PARAMS(text_det_unclip_ratio)
  COPY_PARAMS(text_det_input_shape)
  COPY_PARAMS(text_rec_score_thresh)
  COPY_PARAMS(text_rec_input_shape)
  COPY_PARAMS(lang)
  COPY_PARAMS(ocr_version)
  COPY_PARAMS(vis_font_dir)
  COPY_PARAMS(device)
  COPY_PARAMS(enable_mkldnn)
  COPY_PARAMS(mkldnn_cache_capacity)
  COPY_PARAMS(precision)
  COPY_PARAMS(cpu_threads)
  COPY_PARAMS(thread_num)
  COPY_PARAMS(paddlex_config)
  return to;
}
