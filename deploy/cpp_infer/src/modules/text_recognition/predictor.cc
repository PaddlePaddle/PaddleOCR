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

#include "predictor.h"

#include <algorithm>

#include "result.h"
#include "src/common/image_batch_sampler.h"

TextRecPredictor::TextRecPredictor(const TextRecPredictorParams &params)
    : BasePredictor(params.model_dir, params.model_name, params.device,
                    params.precision, params.enable_mkldnn,
                    params.mkldnn_cache_capacity, params.cpu_threads,
                    params.batch_size, "image"),
      params_(params) {
  auto status = CheckRecModelParams();
  auto status_build = Build();
  if (!status_build.ok()) {
    INFOE("Build fail: %s", status_build.ToString().c_str());
    exit(-1);
  }
};

absl::Status TextRecPredictor::Build() {
  const auto &pre_params = config_.PreProcessOpInfo();
  Register<ReadImage>("Read", "BGR"); //******
  Register<OCRReisizeNormImg>("ReisizeNorm", params_.input_shape);
  Register<ToBatchUniform>("ToBatch");
  infer_ptr_ = CreateStaticInfer();
  const auto &post_params = config_.PostProcessOpInfo();
  post_op_["CTCLabelDecode"] = std::unique_ptr<CTCLabelDecode>(
      new CTCLabelDecode(YamlConfig::SmartParseVector(
                             post_params.at("PostProcess.character_dict"))
                             .vec_string));
  return absl::OkStatus();
};

std::vector<std::unique_ptr<BaseCVResult>>
TextRecPredictor::Process(std::vector<cv::Mat> &batch_data) {
  std::vector<cv::Mat> origin_image = {};
  origin_image.reserve(batch_data.size());
  for (const auto &mat : batch_data) {
    origin_image.push_back(mat.clone());
  }
  auto batch_read = pre_op_.at("Read")->Apply(batch_data);
  if (!batch_read.ok()) {
    INFOE(batch_read.status().ToString().c_str());
    exit(-1);
  }

  auto batch_resize_norm = pre_op_.at("ReisizeNorm")->Apply(batch_read.value());
  if (!batch_resize_norm.ok()) {
    INFOE(batch_resize_norm.status().ToString().c_str());
    exit(-1);
  }

  auto batch_tobatch = pre_op_.at("ToBatch")->Apply(batch_resize_norm.value());
  if (!batch_tobatch.ok()) {
    INFOE(batch_tobatch.status().ToString().c_str());
    exit(-1);
  }
  auto batch_infer = infer_ptr_->Apply(batch_tobatch.value());
  if (!batch_infer.ok()) {
    INFOE(batch_infer.status().ToString().c_str());
    exit(-1);
  }

  auto ctc_result =
      post_op_.at("CTCLabelDecode")->Apply(batch_infer.value()[0]);

  if (!ctc_result.ok()) {
    INFOE(ctc_result.status().ToString().c_str());
    exit(-1);
  }

  std::vector<std::unique_ptr<BaseCVResult>> base_cv_result_ptr_vec = {};
  for (int i = 0; i < ctc_result.value().size(); i++, input_index_++) {
    TextRecPredictorResult predictor_result;
    if (!input_path_.empty()) {
      if (input_index_ == input_path_.size())
        input_index_ = 0;
      predictor_result.input_path = input_path_[input_index_];
    }
    predictor_result.input_image = origin_image[i];
    predictor_result.rec_text = ctc_result.value()[i].first;
    predictor_result.rec_score = ctc_result.value()[i].second;
    predictor_result.vis_font = params_.vis_font_dir.value_or("");
    predictor_result_vec_.push_back(predictor_result);
    base_cv_result_ptr_vec.push_back(
        std::unique_ptr<BaseCVResult>(new TextRecResult(predictor_result)));
  }
  return base_cv_result_ptr_vec;
}

absl::Status TextRecPredictor::CheckRecModelParams() {
  auto result_models_check = Utility::GetOcrModelInfo(
      params_.lang.value_or(""), params_.ocr_version.value_or(""));
  if (!result_models_check.ok()) {
    return absl::InvalidArgumentError("lang and ocr_version is invalid : " +
                                      result_models_check.status().ToString());
  }
  auto result_model_name = ModelName();
  if (!result_model_name.ok()) {
    return absl::InternalError("Get model name fail : " +
                               result_model_name.status().ToString());
  }
  size_t pos_model_name = result_model_name.value().find('_');
  size_t pos_model_check = std::get<1>(result_models_check.value()).find('_');
  std::string prefix_model_name =
      result_model_name.value().substr(0, pos_model_name);
  std::string prefix_model_check =
      std::get<1>(result_models_check.value()).substr(0, pos_model_check);
  auto result =
      Utility::GetOcrModelInfo(params_.lang.value_or(""), prefix_model_name);
  if (!result.ok()) {
    return absl::InternalError("Model and lang do not match : " +
                               result.status().ToString());
  }
  if (params_.ocr_version.has_value()) {
    if (prefix_model_name != params_.ocr_version.value()) {
      INFOW("Rec model ocr_version and ocr_verision params do not match");
    }
  }

#ifdef USE_FREETYPE
  if (!params_.vis_font_dir.has_value()) {
    return absl::InvalidArgumentError(
        "Visualization font path is empty, please provide " +
        std::get<2>(result_models_check.value()) + " path.");
  } else {
    size_t pos = params_.vis_font_dir.value().find_last_of("/\\");
    std::string filename = params_.vis_font_dir.value().substr(pos + 1);
    if (filename != std::get<2>(result_models_check.value())) {
      return absl::NotFoundError("Expected visualization font is " +
                                 std::get<2>(result_models_check.value()) +
                                 ", but get is " + filename);
    }
  }
#endif
  return absl::OkStatus();
}
