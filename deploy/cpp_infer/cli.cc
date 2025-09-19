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

#include "src/api/models/doc_img_orientation_classification.h"
#include "src/api/models/text_detection.h"
#include "src/api/models/text_image_unwarping.h"
#include "src/api/models/text_recognition.h"
#include "src/api/models/textline_orientation_classification.h"
#include "src/api/pipelines/doc_preprocessor.h"
#include "src/api/pipelines/ocr.h"
#include "src/utils/args.h"
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

static const std::unordered_set<std::string> SUPPORT_MODE_PIPELINE = {
    "ocr",
    "doc_preprocessor",
};

static const std::unordered_set<std::string> SUPPORT_MODE_MODEL = {
    "text_image_unwarping", "doc_img_orientation_classification",
    "textline_orientation_classification", "text_detection",
    "text_recognition"};

void PrintErrorInfo(const std::string &msg, const std::string &main_mode = "") {
  auto join_modes =
      [](const std::unordered_set<std::string> &modes) -> std::string {
    std::string result;
    for (const auto &mode : modes) {
      result += mode + ", ";
    }
    if (!result.empty()) {
      result.pop_back();
      result.pop_back();
    }
    return result;
  };

  std::string pipeline_modes = join_modes(SUPPORT_MODE_PIPELINE);
  std::string model_modes = join_modes(SUPPORT_MODE_MODEL);

  INFOE("%s%s", msg.c_str(),
        main_mode.empty() ? "" : (": \"" + main_mode + "\"").c_str());
  INFO("==========================================");
  INFO("Supported pipeline : [%s]", pipeline_modes.c_str());
  INFO("Supported model    : [%s]", model_modes.c_str());
  INFO("==========================================");
}

std::tuple<PaddleOCRParams, DocPreprocessorParams,
           DocImgOrientationClassificationParams, TextImageUnwarpingParams,
           TextDetectionParams, TextLineOrientationClassificationParams,
           TextRecognitionParams>
GetPipelineMoudleParams() {
  PaddleOCRParams ocr_params;
  DocPreprocessorParams doc_pre_params;
  DocImgOrientationClassificationParams doc_orient_params;
  TextImageUnwarpingParams unwarp_params;
  TextDetectionParams det_params;
  TextLineOrientationClassificationParams teline_orient_params;
  TextRecognitionParams rec_params;
  if (!FLAGS_doc_orientation_classify_model_name.empty()) {
    ocr_params.doc_orientation_classify_model_name =
        FLAGS_doc_orientation_classify_model_name;
    doc_pre_params.doc_orientation_classify_model_name =
        FLAGS_doc_orientation_classify_model_name;
    doc_orient_params.model_name = FLAGS_doc_orientation_classify_model_name;
  }
  if (!FLAGS_doc_orientation_classify_model_dir.empty()) {
    ocr_params.doc_orientation_classify_model_dir =
        FLAGS_doc_orientation_classify_model_dir;
    doc_pre_params.doc_orientation_classify_model_dir =
        FLAGS_doc_orientation_classify_model_dir;
    doc_orient_params.model_dir = FLAGS_doc_orientation_classify_model_dir;
  }
  if (!FLAGS_doc_unwarping_model_name.empty()) {
    ocr_params.doc_unwarping_model_name = FLAGS_doc_unwarping_model_name;
    doc_pre_params.doc_unwarping_model_name = FLAGS_doc_unwarping_model_name;
    unwarp_params.model_name = FLAGS_doc_unwarping_model_name;
  }
  if (!FLAGS_doc_unwarping_model_dir.empty()) {
    ocr_params.doc_unwarping_model_dir = FLAGS_doc_unwarping_model_dir;
    doc_pre_params.doc_unwarping_model_dir = FLAGS_doc_unwarping_model_dir;
    unwarp_params.model_dir = FLAGS_doc_unwarping_model_dir;
  }
  if (!FLAGS_text_detection_model_name.empty()) {
    ocr_params.text_detection_model_name = FLAGS_text_detection_model_name;
    det_params.model_name = FLAGS_text_detection_model_name;
  }
  if (!FLAGS_text_detection_model_dir.empty()) {
    ocr_params.text_detection_model_dir = FLAGS_text_detection_model_dir;
    det_params.model_dir = FLAGS_text_detection_model_dir;
  }
  if (!FLAGS_textline_orientation_model_name.empty()) {
    ocr_params.textline_orientation_model_name =
        FLAGS_textline_orientation_model_name;
    teline_orient_params.model_name = FLAGS_textline_orientation_model_name;
  }
  if (!FLAGS_textline_orientation_model_dir.empty()) {
    ocr_params.textline_orientation_model_dir =
        FLAGS_textline_orientation_model_dir;
    teline_orient_params.model_dir = FLAGS_textline_orientation_model_dir;
  }
  if (!FLAGS_textline_orientation_batch_size.empty()) {
    ocr_params.textline_orientation_batch_size =
        std::stoi(FLAGS_textline_orientation_batch_size);
  }
  if (!FLAGS_text_recognition_model_name.empty()) {
    ocr_params.text_recognition_model_name = FLAGS_text_recognition_model_name;
    rec_params.model_name = FLAGS_text_recognition_model_name;
  }
  if (!FLAGS_text_recognition_model_dir.empty()) {
    ocr_params.text_recognition_model_dir = FLAGS_text_recognition_model_dir;
    rec_params.model_dir = FLAGS_text_recognition_model_dir;
  }
  if (!FLAGS_text_recognition_batch_size.empty()) {
    ocr_params.text_recognition_batch_size =
        std::stoi(FLAGS_text_recognition_batch_size);
    rec_params.batch_size = std::stoi(FLAGS_text_recognition_batch_size);
    rec_params.input_shape =
        YamlConfig::SmartParseVector(FLAGS_text_rec_input_shape).vec_int;
  }
  if (!FLAGS_use_doc_orientation_classify.empty()) {
    ocr_params.use_doc_orientation_classify =
        Utility::StringToBool(FLAGS_use_doc_orientation_classify);
    doc_pre_params.use_doc_orientation_classify =
        Utility::StringToBool(FLAGS_use_doc_orientation_classify);
  }
  if (!FLAGS_use_doc_unwarping.empty()) {
    ocr_params.use_doc_unwarping =
        Utility::StringToBool(FLAGS_use_doc_unwarping);
    doc_pre_params.use_doc_unwarping =
        Utility::StringToBool(FLAGS_use_doc_unwarping);
  }
  if (!FLAGS_use_textline_orientation.empty()) {
    ocr_params.use_textline_orientation =
        Utility::StringToBool(FLAGS_use_textline_orientation);
  }
  if (!FLAGS_text_det_limit_side_len.empty()) {
    ocr_params.text_det_limit_side_len =
        std::stoi(FLAGS_text_det_limit_side_len);
  }
  if (!FLAGS_text_det_limit_type.empty()) {
    ocr_params.text_det_limit_type = FLAGS_text_det_limit_type;
    det_params.limit_type = FLAGS_text_det_limit_type;
  }
  if (!FLAGS_text_det_thresh.empty()) {
    ocr_params.text_det_thresh = std::stof(FLAGS_text_det_thresh);
    det_params.thresh = std::stof(FLAGS_text_det_thresh);
  }
  if (!FLAGS_text_det_box_thresh.empty()) {
    ocr_params.text_det_box_thresh = std::stof(FLAGS_text_det_box_thresh);
    det_params.box_thresh = std::stof(FLAGS_text_det_box_thresh);
  }
  if (!FLAGS_text_det_unclip_ratio.empty()) {
    ocr_params.text_det_unclip_ratio = std::stof(FLAGS_text_det_unclip_ratio);
    det_params.unclip_ratio = std::stof(FLAGS_text_det_unclip_ratio);
  }
  if (!FLAGS_text_det_input_shape.empty()) {
    ocr_params.text_det_input_shape =
        YamlConfig::SmartParseVector(FLAGS_text_det_input_shape).vec_int;
    det_params.input_shape =
        YamlConfig::SmartParseVector(FLAGS_text_det_input_shape).vec_int;
  }
  if (!FLAGS_text_rec_score_thresh.empty()) {
    ocr_params.text_rec_score_thresh = std::stof(FLAGS_text_rec_score_thresh);
  }
  if (!FLAGS_text_rec_input_shape.empty()) {
    ocr_params.text_rec_input_shape =
        YamlConfig::SmartParseVector(FLAGS_text_rec_input_shape).vec_int;
  }
  if (!FLAGS_lang.empty()) {
    ocr_params.lang = FLAGS_lang;
  }
  if (!FLAGS_ocr_version.empty()) {
    ocr_params.ocr_version = FLAGS_ocr_version;
  }
  if (!FLAGS_vis_font_dir.empty()) {
    ocr_params.vis_font_dir = FLAGS_vis_font_dir;
    rec_params.vis_font_dir = FLAGS_vis_font_dir;
  }
  if (!FLAGS_device.empty()) {
    ocr_params.device = FLAGS_device;
    doc_pre_params.device = FLAGS_device;
    doc_orient_params.device = FLAGS_device;
    unwarp_params.device = FLAGS_device;
    teline_orient_params.device = FLAGS_device;
    det_params.device = FLAGS_device;
    rec_params.device = FLAGS_device;
  }
  if (!FLAGS_precision.empty()) {
    ocr_params.precision = FLAGS_precision;
    doc_pre_params.precision = FLAGS_precision;
    doc_orient_params.precision = FLAGS_precision;
    unwarp_params.precision = FLAGS_precision;
    teline_orient_params.precision = FLAGS_precision;
    det_params.precision = FLAGS_precision;
    rec_params.precision = FLAGS_precision;
  }
  if (!FLAGS_enable_mkldnn.empty()) {
    ocr_params.enable_mkldnn = Utility::StringToBool(FLAGS_enable_mkldnn);
    doc_pre_params.enable_mkldnn = Utility::StringToBool(FLAGS_enable_mkldnn);
    doc_orient_params.enable_mkldnn =
        Utility::StringToBool(FLAGS_enable_mkldnn);
    unwarp_params.enable_mkldnn = Utility::StringToBool(FLAGS_enable_mkldnn);
    teline_orient_params.enable_mkldnn =
        Utility::StringToBool(FLAGS_enable_mkldnn);
    det_params.enable_mkldnn = Utility::StringToBool(FLAGS_enable_mkldnn);
    rec_params.enable_mkldnn = Utility::StringToBool(FLAGS_enable_mkldnn);
  }
  if (!FLAGS_mkldnn_cache_capacity.empty()) {
    ocr_params.mkldnn_cache_capacity = std::stoi(FLAGS_mkldnn_cache_capacity);
    doc_pre_params.mkldnn_cache_capacity =
        std::stoi(FLAGS_mkldnn_cache_capacity);
    doc_orient_params.mkldnn_cache_capacity =
        std::stoi(FLAGS_mkldnn_cache_capacity);
    unwarp_params.mkldnn_cache_capacity =
        std::stoi(FLAGS_mkldnn_cache_capacity);
    teline_orient_params.mkldnn_cache_capacity =
        std::stoi(FLAGS_mkldnn_cache_capacity);
    det_params.mkldnn_cache_capacity = std::stoi(FLAGS_mkldnn_cache_capacity);
    rec_params.mkldnn_cache_capacity = std::stoi(FLAGS_mkldnn_cache_capacity);
  }
  if (!FLAGS_cpu_threads.empty()) {
    ocr_params.cpu_threads = std::stoi(FLAGS_cpu_threads);
    doc_pre_params.cpu_threads = std::stoi(FLAGS_cpu_threads);
    doc_orient_params.cpu_threads = std::stoi(FLAGS_cpu_threads);
    unwarp_params.cpu_threads = std::stoi(FLAGS_cpu_threads);
    teline_orient_params.cpu_threads = std::stoi(FLAGS_cpu_threads);
    det_params.cpu_threads = std::stoi(FLAGS_cpu_threads);
    rec_params.cpu_threads = std::stoi(FLAGS_cpu_threads);
  }
  if (!FLAGS_thread_num.empty()) {
    ocr_params.thread_num = std::stoi(FLAGS_thread_num);
    doc_pre_params.thread_num = std::stoi(FLAGS_thread_num);
  }
  if (!FLAGS_paddlex_config.empty()) {
    ocr_params.paddlex_config = FLAGS_paddlex_config;
    doc_pre_params.paddlex_config = FLAGS_paddlex_config;
  }
  return std::make_tuple(ocr_params, doc_pre_params, doc_orient_params,
                         unwarp_params, det_params, teline_orient_params,
                         rec_params);
}

int main(int argc, char *argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_input.empty()) {
    INFOE("Require input, such as ./build/ppocr <pipeline_or_module> --input "
          "your_image_path [--param1] [--param2] [...]");
    exit(-1);
  }
  std::string main_mode = "";
  if (argc > 1) {
    main_mode = argv[1];
    if (SUPPORT_MODE_PIPELINE.count(main_mode) == 0 &&
        SUPPORT_MODE_MODEL.count(main_mode) == 0) {
      PrintErrorInfo("ERROR: Unsupported pipeline or module", main_mode);
      exit(-1);
    }
  } else {
    PrintErrorInfo(
        "Must provide pipeline or module name, such as ./build/ppocr "
        "<pipeline_or_module> [--param1] [--param2] [...]");
    exit(-1);
  }
  auto params = GetPipelineMoudleParams();
  using PredFunc = std::function<std::vector<std::unique_ptr<BaseCVResult>>(
      const std::string &)>;
  std::unordered_map<std::string, PredFunc> pred_map = {
      {"ocr",
       [&params](const std::string &input) {
         return PaddleOCR(std::get<0>(params)).Predict(input);
       }},
      {"doc_preprocessor",
       [&params](const std::string &input) {
         return DocPreprocessor(std::get<1>(params)).Predict(input);
       }},
      {"doc_img_orientation_classification",
       [&params](const std::string &input) {
         return DocImgOrientationClassification(std::get<2>(params))
             .Predict(input);
       }},
      {"text_image_unwarping",
       [&params](const std::string &input) {
         return TextImageUnwarping(std::get<3>(params)).Predict(input);
       }},
      {"text_detection",
       [&params](const std::string &input) {
         return TextDetection(std::get<4>(params)).Predict(input);
       }},
      {"textline_orientation_classification",
       [&params](const std::string &input) {
         return TextLineOrientationClassification(std::get<5>(params))
             .Predict(input);
       }},
      {"text_recognition",
       [&params](const std::string &input) {
         return TextRecognition(std::get<6>(params)).Predict(input);
       }},

  };
  auto it = pred_map.find(main_mode);
  auto outputs = it->second(FLAGS_input);
  for (auto &output : outputs) {
    output->Print();
    output->SaveToImg(FLAGS_save_path);
    output->SaveToJson(FLAGS_save_path);
  }
  return 0;
}
