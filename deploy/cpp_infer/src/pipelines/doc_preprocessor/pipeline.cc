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

#include "pipeline.h"

#include "result.h"
#include "src/modules/image_classification/predictor.h"
#include "src/modules/image_unwarping/predictor.h"

_DocPreprocessorPipeline::_DocPreprocessorPipeline(
    const DocPreprocessorPipelineParams &params)
    : BasePipeline(), params_(params) {
  if (params.paddlex_config.has_value()) {
    if (params.paddlex_config.value().IsStr()) {
      config_ = YamlConfig(params.paddlex_config.value().GetStr());
    } else {
      config_ = YamlConfig(params.paddlex_config.value().GetMap());
    }
  } else {
    auto config_path = Utility::GetDefaultConfig("doc_preprocessor");
    if (!config_path.ok()) {
      INFOE("Could not find doc_preprocessors pipeline config file : %s",
            config_path.status().ToString().c_str());
      exit(-1);
    }
    config_ = YamlConfig(config_path.value());
  }
  OverrideConfig();
  auto result_doc = config_.GetBool("use_doc_orientation_classify", true);
  if (!result_doc.ok()) {
    INFOE("use_doc_orientation_classify set fail : %s",
          result_doc.status().ToString().c_str());
    exit(-1);
  }
  use_doc_orientation_classify_ = result_doc.value();
  auto result_batch = config_.GetInt("batch_size", 1);
  if (!result_batch.ok()) {
    INFOE("batch_size get fail: %s", result_batch.status().ToString().c_str());
    exit(-1);
  }

  if (use_doc_orientation_classify_) {
    ClasPredictorParams doc_ori_classify_params;
    auto result_model_dir =
        config_.GetString("DocOrientationClassify.model_dir");
    if (!result_model_dir.ok()) {
      INFOE("Could not find DocOrientationClassify model dir : %s",
            result_model_dir.status().ToString().c_str());
      exit(-1);
    }
    auto result_model_name =
        config_.GetString("DocOrientationClassify.model_name");
    if (!result_model_name.ok()) {
      INFOE("Could not find DocOrientationClassify model name : %s",
            result_model_name.status().ToString().c_str());
      exit(-1);
    }
    doc_ori_classify_params.model_dir = result_model_dir.value();
    doc_ori_classify_params.model_name = result_model_name.value();
    doc_ori_classify_params.device = params_.device;
    doc_ori_classify_params.precision = params_.precision;
    doc_ori_classify_params.enable_mkldnn = params_.enable_mkldnn;
    doc_ori_classify_params.mkldnn_cache_capacity =
        params_.mkldnn_cache_capacity;
    doc_ori_classify_params.cpu_threads = params_.cpu_threads;
    doc_ori_classify_params.batch_size = result_batch.value();

    doc_ori_classify_model_ =
        CreateModule<ClasPredictor>(doc_ori_classify_params);
  }

  auto result_unwarping = config_.GetBool("use_doc_unwarping", true);
  if (!result_unwarping.ok()) {
    INFOE("use_doc_unwarping get fail:%s",
          result_unwarping.status().ToString().c_str());
    exit(-1);
  }
  use_doc_unwarping_ = result_unwarping.value();

  if (use_doc_unwarping_) {
    WarpPredictorParams doc_unwarping_params;
    auto result_model_dir = config_.GetString("DocUnwarping.model_dir");
    if (!result_model_dir.ok()) {
      INFOE("Could not find DocUnwarping model dir : %s",
            result_model_dir.status().ToString().c_str());
      exit(-1);
    }
    auto result_model_name = config_.GetString("DocUnwarping.model_name");
    if (!result_model_name.ok()) {
      INFOE("Could not find DocUnwarping model name : %s",
            result_model_name.status().ToString().c_str());
      exit(-1);
    }
    doc_unwarping_params.model_dir = result_model_dir.value();
    doc_unwarping_params.model_name = result_model_name.value();
    doc_unwarping_params.device = params_.device;
    doc_unwarping_params.precision = params_.precision;
    doc_unwarping_params.enable_mkldnn = params_.enable_mkldnn;
    doc_unwarping_params.mkldnn_cache_capacity = params_.mkldnn_cache_capacity;
    doc_unwarping_params.cpu_threads = params_.cpu_threads;
    doc_unwarping_params.batch_size = result_batch.value();

    doc_unwarping_model_ = CreateModule<WarpPredictor>(doc_unwarping_params);
  }

  batch_sampler_ptr_ = std::unique_ptr<BaseBatchSampler>(
      new ImageBatchSampler(result_batch.value()));
};

std::vector<std::unique_ptr<BaseCVResult>>
_DocPreprocessorPipeline::Predict(const std::vector<std::string> &input) {
  auto model_setting = GetModelSettings();
  auto status = CheckModelSettingsVaild(model_setting);
  if (!status.ok()) {
    INFOE("the input params for model settings are invalid!: %s",
          status.ToString().c_str());
    exit(-1);
  }
  auto batches = batch_sampler_ptr_->Apply(input);
  if (!batches.ok()) {
    INFOE("pipeline get sample fail : %s", batches.status().ToString().c_str());
    exit(-1);
  }
  auto input_path = batch_sampler_ptr_->InputPath();
  int index = 0;
  std::vector<cv::Mat> origin_image = {};

  std::vector<std::unique_ptr<BaseCVResult>> base_cv_result_ptr_vec = {};
  std::vector<DocPreprocessorPipelineResult> pipeline_result_vec = {};
  pipeline_result_vec_.clear();
  for (auto &batch_data : batches.value()) {
    origin_image.reserve(batch_data.size());
    for (const auto &mat : batch_data) {
      origin_image.push_back(mat.clone());
    }
    std::vector<int> angles = {};
    std::vector<cv::Mat> rotate_images = {};
    if (model_setting["use_doc_orientation_classify"]) {
      doc_ori_classify_model_->Predict(batch_data);
      ClasPredictor *derived =
          static_cast<ClasPredictor *>(doc_ori_classify_model_.get());
      std::vector<ClasPredictorResult> preds = derived->PredictorResult();
      for (auto &pred : preds) {
        auto result_angle = Utility::StringToInt(pred.label_names[0]);
        if (!result_angle.ok()) {
          INFOE("angle is invalid : %s",
                result_angle.status().ToString().c_str());
          exit(-1);
        }
        angles.push_back(result_angle.value());
        auto result_rotate = ComponentsProcessor::RotateImage(
            pred.input_image, result_angle.value());
        if (!result_rotate.ok()) {
          INFOE("RotateImage fail : %s",
                result_rotate.status().ToString().c_str());
          exit(-1);
        }
        rotate_images.push_back(result_rotate.value());
      }
    } else {
      angles = std::vector<int>(batch_data.size(), -1);
      rotate_images = batch_data;
    }
    std::vector<cv::Mat> output_imgs = {};
    if (model_setting["use_doc_unwarping"]) {
      doc_unwarping_model_->Predict(rotate_images);
      WarpPredictor *derived =
          static_cast<WarpPredictor *>(doc_unwarping_model_.get());
      std::vector<WarpPredictorResult> preds = derived->PredictorResult();
      for (auto &pred : preds) {
        output_imgs.push_back(pred.doctr_img); //***"RGB" "BGR"
      }
    } else {
      output_imgs = rotate_images;
    }

    pipeline_result_vec.clear();
    for (int i = 0; i < output_imgs.size(); i++, index++) {
      DocPreprocessorPipelineResult pipeline_result;
      pipeline_result.input_path = input_path[index];
      pipeline_result.input_image = origin_image[i];
      pipeline_result.model_settings = model_setting;
      pipeline_result.angle = angles[i];
      pipeline_result.rotate_image = rotate_images[i];
      pipeline_result.output_image = output_imgs[i];
      pipeline_result_vec.push_back(pipeline_result);
    }
    origin_image.clear();
    pipeline_result_vec_.insert(pipeline_result_vec_.end(),
                                pipeline_result_vec.begin(),
                                pipeline_result_vec.end());
    for (auto &pipeline_result : pipeline_result_vec) {
      std::unique_ptr<BaseCVResult> base_cv_result_ptr =
          std::unique_ptr<BaseCVResult>(
              new DocPreprocessorResult(pipeline_result));
      base_cv_result_ptr_vec.emplace_back(std::move(base_cv_result_ptr));
    }
  }
  return base_cv_result_ptr_vec;
};

std::unordered_map<std::string, bool>
_DocPreprocessorPipeline::GetModelSettings(
    absl::optional<bool> use_doc_orientation_classify,
    absl::optional<bool> use_doc_unwarping) const {
  if (!use_doc_orientation_classify.has_value()) {
    use_doc_orientation_classify = use_doc_orientation_classify_;
  }
  if (!use_doc_unwarping.has_value()) {
    use_doc_unwarping = use_doc_unwarping_;
  }
  std::unordered_map<std::string, bool> model_settings = {};
  model_settings["use_doc_orientation_classify"] =
      use_doc_orientation_classify.value();
  model_settings["use_doc_unwarping"] = use_doc_unwarping.value();
  return model_settings;
};

absl::Status _DocPreprocessorPipeline::CheckModelSettingsVaild(
    std::unordered_map<std::string, bool> model_settings) const {
  if (model_settings["use_doc_orientation_classify"] &&
      !use_doc_orientation_classify_) {
    return absl::InvalidArgumentError(
        "Set use_doc_orientation_classify, but the model for doc orientation "
        "classify is not initialized.");
  }

  if (model_settings["use_doc_unwarping"] && !use_doc_unwarping_) {
    return absl::InvalidArgumentError(
        "Set use_doc_unwarping, but the model for doc unwarping is not "
        "initialized.");
  }
  return absl::OkStatus();
}

std::vector<std::unique_ptr<BaseCVResult>>
DocPreprocessorPipeline::Predict(const std::vector<std::string> &input) {
  if (thread_num_ == 1) {
    return infer_->Predict(input);
  }
  batch_sampler_ptr_ =
      std::unique_ptr<BaseBatchSampler>(new ImageBatchSampler(1));
  auto nomeaning = batch_sampler_ptr_->Apply(input);
  int input_num = nomeaning.value().size();
  if (thread_num_ > input_num) {
    INFOW("thread num exceed input num, will set %d", input_num);
    thread_num_ = input_num;
  }
  int infer_batch_num = input_num / thread_num_;
  auto status = batch_sampler_ptr_->SetBatchSize(infer_batch_num);
  if (!status.ok()) {
    INFOE("Set batch size fail : %s", status.ToString().c_str());
    exit(-1);
  }
  auto infer_batch_data =
      batch_sampler_ptr_->SampleFromVectorToStringVector(input);
  if (!infer_batch_data.ok()) {
    INFOE("Get infer batch data fail : %s",
          infer_batch_data.status().ToString().c_str());
    exit(-1);
  }
  std::vector<std::unique_ptr<BaseCVResult>> results = {};
  results.reserve(input_num);
  for (auto &infer_data : infer_batch_data.value()) {
    auto status =
        AutoParallelSimpleInferencePipeline::PredictThread(infer_data);
    if (!status.ok()) {
      INFOE("Infer fail : %s", status.ToString().c_str());
      exit(-1);
    }
  }
  for (int i = 0; i < infer_batch_data.value().size(); i++) {
    auto infer_data_result = GetResult();
    if (!infer_data_result.ok()) {
      INFOE("Get infer result fail : %s",
            infer_batch_data.status().ToString().c_str());
      exit(-1);
    }
    results.insert(results.end(),
                   std::make_move_iterator(infer_data_result.value().begin()),
                   std::make_move_iterator(infer_data_result.value().end()));
  }
  return results;
}

void _DocPreprocessorPipeline::OverrideConfig() {
  auto &data = config_.Data();
  if (params_.doc_orientation_classify_model_name.has_value()) {
    auto it = config_.FindKey("DocOrientationClassify.model_name");
    if (!it.ok()) {
      data["DocPreprocessor.SubModules.DocOrientationClassify."
           "model_name"] = params_.doc_orientation_classify_model_name.value();
    } else {
      auto key = it.value().first;
      data.erase(data.find(key));
      data[key] = params_.doc_orientation_classify_model_name.value();
    }
  }
  if (params_.doc_orientation_classify_model_dir.has_value()) {
    auto it = config_.FindKey("DocOrientationClassify.model_dir");
    if (!it.ok()) {
      data["DocPreprocessor.SubModules.DocOrientationClassify."
           "model_dir"] = params_.doc_orientation_classify_model_dir.value();
    } else {
      auto key = it.value().first;
      data.erase(data.find(key));
      data[key] = params_.doc_orientation_classify_model_dir.value();
    }
  }
  if (params_.doc_unwarping_model_name.has_value()) {
    auto it = config_.FindKey("DocUnwarping.model_name");
    if (!it.ok()) {
      data["DocPreprocessor.SubModules.DocUnwarping.model_name"] =
          params_.doc_unwarping_model_name.value();
    } else {
      auto key = it.value().first;
      data.erase(data.find(key));
      data[key] = params_.doc_unwarping_model_name.value();
    }
  }
  if (params_.doc_unwarping_model_dir.has_value()) {
    auto it = config_.FindKey("DocUnwarping.model_dir");
    if (!it.ok()) {
      data["DocPreprocessor.SubModules.DocUnwarping.model_dir"] =
          params_.doc_unwarping_model_dir.value();
    } else {
      auto key = it.value().first;
      data.erase(data.find(key));
      data[key] = params_.doc_unwarping_model_dir.value();
    }
  }

  if (params_.use_doc_orientation_classify.has_value()) {
    auto it = config_.FindKey("DocPreprocessor.use_doc_orientation_classify");
    if (!it.ok()) {
      data["DocPreprocessor.use_doc_orientation_classify"] =
          params_.use_doc_orientation_classify.value() ? "true" : "false";
    } else {
      auto key = it.value().first;
      data.erase(data.find(key));
      data[key] =
          params_.use_doc_orientation_classify.value() ? "true" : "false";
    }
  }
  if (params_.use_doc_unwarping.has_value()) {
    auto it = config_.FindKey("DocPreprocessor.use_doc_unwarping");
    if (!it.ok()) {
      data["DocPreprocessor.use_doc_unwarping"] =
          params_.use_doc_unwarping.value() ? "true" : "false";
    } else {
      auto key = it.value().first;
      data.erase(data.find(key));
      data[key] = params_.use_doc_unwarping.value() ? "true" : "false";
    }
  }
}
