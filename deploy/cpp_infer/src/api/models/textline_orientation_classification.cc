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

#include "textline_orientation_classification.h"

#include "src/utils/args.h"
#include "src/utils/yaml_config.h"

#define COPY_PARAMS(field) to.field = from.field;

TextLineOrientationClassification::TextLineOrientationClassification(
    const TextLineOrientationClassificationParams &params)
    : params_(params) {
  auto status = CheckParams();
  if (!status.ok()) {
    INFOE("Init TextLineOrientationClassification fail : %s",
          status.ToString().c_str());
    exit(-1);
  }
  CreateModel();
};
std::vector<std::unique_ptr<BaseCVResult>>
TextLineOrientationClassification::Predict(
    const std::vector<std::string> &input) {
  return model_infer_->Predict(input);
}
void TextLineOrientationClassification::CreateModel() {
  model_infer_ = std::unique_ptr<BasePredictor>(new ClasPredictor(
      ToTextLineOrientationClassificationModelParams(params_)));
}

absl::Status TextLineOrientationClassification::CheckParams() {
  if (!params_.model_dir.has_value()) {
    return absl::NotFoundError(
        "Require textLine orientation classification model dir.");
  }
  return absl::OkStatus();
}

ClasPredictorParams TextLineOrientationClassification::
    ToTextLineOrientationClassificationModelParams(
        const TextLineOrientationClassificationParams &from) {
  ClasPredictorParams to;
  COPY_PARAMS(model_name)
  COPY_PARAMS(model_dir)
  COPY_PARAMS(device)
  COPY_PARAMS(enable_mkldnn)
  COPY_PARAMS(mkldnn_cache_capacity)
  COPY_PARAMS(precision)
  COPY_PARAMS(cpu_threads)
  return to;
}
