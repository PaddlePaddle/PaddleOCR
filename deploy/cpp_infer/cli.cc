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

#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "src/api/models/doc_img_orientation_classification.h"
#include "src/api/models/text_detection.h"
#include "src/api/models/text_image_unwarping.h"
#include "src/api/models/text_recognition.h"
#include "src/api/models/textline_orientation_classification.h"
#include "src/api/pipelines/doc_preprocessor.h"
#include "src/api/pipelines/ocr.h"
#include "src/utils/args.h"

static const std::unordered_set<std::string> SUPPORT_MODE_PIPELINE = {
    "paddleocr",
    "doc_preprocessor",
};

static const std::unordered_set<std::string> SUPPORT_MODE_MODEL = {
    "text_image_unwarping", "doc_img_orientation_classification",
    "textline_orientation_classification", "text_detection",
    "text_recognition"};

int main(int argc, char *argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_input.empty()) {
    INFOE("Require input.");
    exit(-1);
  }
  std::string main_mode = "";
  if (argc > 1) {
    main_mode = argv[1];
    if (SUPPORT_MODE_PIPELINE.count(main_mode) == 0 &&
        SUPPORT_MODE_MODEL.count(main_mode) == 0) {
      INFOE("Unsupport mode: %s", main_mode.c_str());
      exit(-1);
    }
  } else {
    INFOE("Too few params.");
    exit(-1);
  }

  using PredFunc = std::function<std::vector<std::unique_ptr<BaseCVResult>>(
      const std::string &)>;
  std::unordered_map<std::string, PredFunc> pred_map = {
      {"paddleocr",
       [](const std::string &input) { return PaddleOCR().Predict(input); }},
      {"doc_preprocessor",
       [](const std::string &input) {
         return DocPreprocessor().Predict(input);
       }},
      {"text_image_unwarping",
       [](const std::string &input) {
         return TextImageUnwarping().Predict(input);
       }},
      {"doc_img_orientation_classification",
       [](const std::string &input) {
         return DocImgOrientationClassification().Predict(input);
       }},
      {"textline_orientation_classification",
       [](const std::string &input) {
         return TextLineOrientationClassification().Predict(input);
       }},
      {"text_detection",
       [](const std::string &input) { return TextDetection().Predict(input); }},
      {"text_recognition",
       [](const std::string &input) {
         return TextRecognition().Predict(input);
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
