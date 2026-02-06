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
#include "src/utils/args.h"
_OCRPipeline::_OCRPipeline(const OCRPipelineParams &params)
    : BasePipeline(), params_(params) {
  if (params.paddlex_config.has_value()) {
    if (params.paddlex_config.value().IsStr()) {
      config_ = YamlConfig(params.paddlex_config.value().GetStr());
    } else {
      config_ = YamlConfig(params.paddlex_config.value().GetMap());
    }
  } else {
    auto config_path = Utility::GetDefaultConfig("OCR");
    if (!config_path.ok()) {
      INFOE("Could not find OCR pipeline config file: %s",
            config_path.status().ToString().c_str());
      exit(-1);
    }
    config_ = YamlConfig(config_path.value());
  }
  OverrideConfig();
  auto result_use_doc_orientation_classify =
      config_.GetBool("use_doc_orientation_classify", true);
  if (!result_use_doc_orientation_classify.ok()) {
    INFOE("use_doc_orientation_classify config error : %s",
          result_use_doc_orientation_classify.status().ToString().c_str());
    exit(-1);
  }
  auto result_use_use_doc_unwarping =
      config_.GetBool("use_doc_unwarping", true);
  if (!result_use_use_doc_unwarping.ok()) {
    INFOE("use_doc_unwarping config error : %s",
          result_use_use_doc_unwarping.status().ToString().c_str());
    exit(-1);
  }
  if (result_use_doc_orientation_classify.value() ||
      result_use_use_doc_unwarping.value()) {
    use_doc_preprocessor_ = true;
  } else {
    use_doc_preprocessor_ = false;
  }
  if (use_doc_preprocessor_) {
    auto result_doc_preprocessor_config = config_.GetSubModule("SubPipelines");
    if (!result_doc_preprocessor_config.ok()) {
      INFOE("Get doc preprocessors subpipelines config fail : ",
            result_doc_preprocessor_config.status().ToString().c_str());
      exit(-1);
    }
    DocPreprocessorPipelineParams params;
    params.device = params_.device;
    params.precision = params_.precision;
    params.enable_mkldnn = params_.enable_mkldnn;
    params.mkldnn_cache_capacity = params_.mkldnn_cache_capacity;
    params.cpu_threads = params_.cpu_threads;
    params.paddlex_config = result_doc_preprocessor_config.value();
    doc_preprocessors_pipeline_ =
        CreatePipeline<_DocPreprocessorPipeline>(params);

    use_doc_orientation_classify_ =
        config_.GetBool("DocPreprocessor.use_doc_orientation_classify", true)
            .value();
    use_doc_unwarping_ =
        config_.GetBool("DocPreprocessor.use_doc_unwarping", true).value();
  }
  auto result_use_textline_orientation =
      config_.GetBool("use_textline_orientation", true);
  if (!result_use_textline_orientation.ok()) {
    INFOE("use_textline_orientation config error : %s",
          result_use_textline_orientation.status().ToString().c_str());
    exit(-1);
  }
  use_textline_orientation_ = result_use_textline_orientation.value();
  if (use_textline_orientation_) {
    ClasPredictorParams params;
    params.device = params_.device;
    params.precision = params_.precision;
    params.enable_mkldnn = params_.enable_mkldnn;
    params.mkldnn_cache_capacity = params_.mkldnn_cache_capacity;
    params.cpu_threads = params_.cpu_threads;
    auto result_batch_size =
        config_.GetInt("TextLineOrientation.batch_size", 1);
    if (!result_batch_size.ok()) {
      INFOE("Get TextLineOrientation batch size fail: %s",
            result_batch_size.status().ToString().c_str());
      exit(-1);
    }
    params.batch_size = result_batch_size.value();

    auto result_model_name =
        config_.GetString("TextLineOrientation.model_name");
    if (!result_model_name.ok()) {
      INFOE("Could not find TextLineOrientation model name : %s",
            result_model_name.status().ToString().c_str());
      exit(-1);
    }
    params.model_name = result_model_name.value();
    auto result_model_dir = config_.GetString("TextLineOrientation.model_dir");
    if (!result_model_dir.ok()) {
      INFOE("Could not find TextLineOrientation model dir : %s",
            result_model_dir.status().ToString().c_str());
      exit(-1);
    }
    params.model_dir = result_model_dir.value();
    textline_orientation_model_ = CreateModule<ClasPredictor>(params);
  }
  auto text_type = config_.GetString("text_type");
  if (!text_type.ok()) {
    INFOE("Get text type fail : %s", text_type.status().ToString().c_str());
    exit(-1);
  }
  text_type_ = text_type.value();
  TextDetPredictorParams params_det;
  auto result_text_det_model_name =
      config_.GetString("TextDetection.model_name");
  if (!result_text_det_model_name.ok()) {
    INFOE("Could not find TextDetection model name : %s",
          result_text_det_model_name.status().ToString().c_str());
    exit(-1);
  }
  params_det.model_name = result_text_det_model_name.value();
  auto result_text_det_model_dir = config_.GetString("TextDetection.model_dir");
  if (!result_text_det_model_dir.ok()) {
    INFOE("Could not find TextDetection model dir : %s",
          result_text_det_model_dir.status().ToString().c_str());
    exit(-1);
  }
  params_det.model_dir = result_text_det_model_dir.value();
  auto result_det_input_shape = config_.GetString("TextDetection.input_shape");
  if (!result_det_input_shape.value().empty()) {
    params_det.input_shape =
        config_.SmartParseVector(result_det_input_shape.value()).vec_int;
  }
  params_det.device = params_.device;
  params_det.precision = params_.precision;
  params_det.enable_mkldnn = params_.enable_mkldnn;
  params_det.mkldnn_cache_capacity = params_.mkldnn_cache_capacity;
  params_det.cpu_threads = params_.cpu_threads;
  params_det.batch_size = config_.GetInt("TextDetection.batch_size", 1).value();
  if (text_type_ == "general") {
    params_det.limit_side_len =
        config_.GetInt("TextDetection.limit_side_len", 960).value();
    params_det.limit_type =
        config_.GetString("TextDetection.limit_type", "max").value();
    params_det.max_side_limit =
        config_.GetInt("TextDetection.max_side_limit", 4000).value();
    params_det.thresh = config_.GetFloat("TextDetection.thresh", 0.3).value();
    params_det.box_thresh =
        config_.GetFloat("TextDetection.box_thresh", 0.6).value();
    params_det.unclip_ratio =
        config_.GetFloat("TextDetection.unclip_ratio", 2.0).value();
    sort_boxes_ = ComponentsProcessor::SortQuadBoxes;
    crop_by_polys_ = std::unique_ptr<CropByPolys>(new CropByPolys("quad"));
  } else if (text_type_ == "seal") {
    params_det.limit_side_len =
        config_.GetInt("TextDetection.limit_side_len", 736).value();
    params_det.limit_type =
        config_.GetString("TextDetection.limit_type", "min").value();
    params_det.max_side_limit =
        config_.GetInt("TextDetection.max_side_limit", 4000).value();
    params_det.thresh = config_.GetFloat("TextDetection.thresh", 0.2).value();
    params_det.box_thresh =
        config_.GetFloat("TextDetection.box_thresh", 0.6).value();
    params_det.unclip_ratio =
        config_.GetFloat("TextDetection.unclip_ratio", 0.5).value();
    sort_boxes_ = ComponentsProcessor::SortPolyBoxes;
    crop_by_polys_ = std::unique_ptr<CropByPolys>(new CropByPolys("poly"));
  } else {
    INFOE("Unsupported text type We %s", text_type.value().c_str());
    exit(-1);
  }
  text_det_model_ = CreateModule<TextDetPredictor>(params_det);

  text_det_params_.text_det_limit_side_len = params_det.limit_side_len.value();
  text_det_params_.text_det_limit_type = params_det.limit_type.value();
  text_det_params_.text_det_max_side_limit = params_det.max_side_limit.value();
  text_det_params_.text_det_thresh = params_det.thresh.value();
  text_det_params_.text_det_box_thresh = params_det.box_thresh.value();
  text_det_params_.text_det_unclip_ratio = params_det.unclip_ratio.value();

  TextRecPredictorParams params_rec;
  auto result_text_rec_model_name =
      config_.GetString("TextRecognition.model_name");
  if (!result_text_rec_model_name.ok()) {
    INFOE("Could not find TextRecognition model name : %s",
          result_text_rec_model_name.status().ToString().c_str());
    exit(-1);
  }
  params_rec.model_name = result_text_rec_model_name.value();
  auto result_text_rec_model_dir =
      config_.GetString("TextRecognition.model_dir");
  if (!result_text_rec_model_dir.ok()) {
    INFOE("Could not find TextRecognition model dir : %s",
          result_text_rec_model_dir.status().ToString().c_str());
    exit(-1);
  }
  auto result_rec_input_shape =
      config_.GetString("TextRecognition.input_shape");
  if (!result_rec_input_shape.value().empty()) {
    params_rec.input_shape =
        config_.SmartParseVector(result_rec_input_shape.value()).vec_int;
  }
  params_rec.model_dir = result_text_rec_model_dir.value();
  params_rec.lang = params_.lang;
  params_rec.ocr_version = params_.ocr_version;
  params_rec.vis_font_dir = params_.vis_font_dir;
  params_rec.device = params_.device;
  params_rec.precision = params_.precision;
  params_rec.enable_mkldnn = params_.enable_mkldnn;
  params_rec.mkldnn_cache_capacity = params_.mkldnn_cache_capacity;
  params_rec.cpu_threads = params_.cpu_threads;
  params_rec.batch_size =
      config_.GetInt("TextRecognition.batch_size", 1).value();

  text_rec_model_ = CreateModule<TextRecPredictor>(params_rec);
  text_rec_score_thresh_ =
      config_.GetFloat("TextRecognition.score_thresh", 0.0).value();

  batch_sampler_ptr_ = std::unique_ptr<BaseBatchSampler>(
      new ImageBatchSampler(1)); //** pipeline batch_size
};

absl::StatusOr<std::vector<cv::Mat>>
_OCRPipeline::RotateImage(const std::vector<cv::Mat> &image_array_list,
                          const std::vector<int> &rotate_angle_list) {
  if (image_array_list.size() != rotate_angle_list.size()) {
    return absl::InvalidArgumentError(
        "Length of image_array_list (" +
        std::to_string(image_array_list.size()) +
        ") must match length of rotate_angle_list (" +
        std::to_string(rotate_angle_list.size()) + ")");
  }
  std::vector<cv::Mat> rotated_images;
  rotated_images.reserve(image_array_list.size());
  for (std::size_t i = 0; i < image_array_list.size(); ++i) {
    int angle_indicator = rotate_angle_list[i];
    if (angle_indicator != 0 && angle_indicator != 1) {
      return absl::InvalidArgumentError(
          "rotate_angle must be 0 or 1, now it's: " +
          std::to_string(angle_indicator));
    }
    int rotate_angle = angle_indicator * 180;
    auto result_rotated_image =
        ComponentsProcessor::RotateImage(image_array_list[i], rotate_angle);
    if (!result_rotated_image.ok()) {
      return result_rotated_image.status();
    }
    cv::Mat rotated_image = result_rotated_image.value();
    rotated_images.push_back(rotated_image);
  }
  return rotated_images;
}

std::unordered_map<std::string, bool> _OCRPipeline::GetModelSettings() const {
  std::unordered_map<std::string, bool> model_settings = {};
  model_settings["use_doc_preprocessor"] = use_doc_preprocessor_;
  model_settings["use_textline_orientation"] = use_textline_orientation_;
  return model_settings;
}

std::vector<std::unique_ptr<BaseCVResult>>
_OCRPipeline::Predict(const std::vector<std::string> &input) {
  auto model_settings = GetModelSettings();
  auto batches = batch_sampler_ptr_->Apply(input);
  auto batches_string =
      batch_sampler_ptr_->SampleFromVectorToStringVector(input);
  if (!batches.ok()) {
    INFOE("pipeline get sample fail : %s", batches.status().ToString().c_str());
    exit(-1);
  }
  if (!batches_string.ok()) {
    INFOE("pipeline get sample fail : %s",
          batches_string.status().ToString().c_str());
    exit(-1);
  }
  auto input_path = batch_sampler_ptr_->InputPath();
  int index = 0;
  std::vector<cv::Mat> origin_image = {};
  std::vector<std::unique_ptr<BaseCVResult>> base_results = {};
  pipeline_result_vec_.clear();
  for (int i = 0; i < batches.value().size(); i++) {
    origin_image.reserve(batches.value()[i].size());
    for (const auto &mat : batches.value()[i]) {
      origin_image.push_back(mat.clone());
    }
    std::vector<DocPreprocessorPipelineResult>
        doc_preprocessors_pipeline_results = {};
    if (use_doc_preprocessor_) {
      doc_preprocessors_pipeline_->Predict(batches_string.value()[i]);
      doc_preprocessors_pipeline_results =
          static_cast<_DocPreprocessorPipeline *>(
              doc_preprocessors_pipeline_.get())
              ->PipelineResult();
    } else {
      DocPreprocessorPipelineResult result;
      for (auto &image : batches.value()[i]) {
        result.output_image = image.clone();
        doc_preprocessors_pipeline_results.push_back(result);
      }
    }
    std::vector<cv::Mat> doc_preprocessor_pipeline_images = {};
    std::vector<cv::Mat> doc_preprocessor_pipeline_images_copy = {};
    for (auto &item : doc_preprocessors_pipeline_results) {
      doc_preprocessor_pipeline_images.push_back(item.output_image);
      doc_preprocessor_pipeline_images_copy.push_back(
          item.output_image.clone());
    }
    text_det_model_->Predict(doc_preprocessor_pipeline_images_copy);
    std::vector<TextDetPredictorResult> det_results =
        static_cast<TextDetPredictor *>(text_det_model_.get())
            ->PredictorResult();
    std::vector<std::vector<std::vector<cv::Point2f>>> dt_polys_list = {};
    for (auto &item : det_results) {
      if (!item.dt_polys.empty()) {
        auto sort_item = sort_boxes_(item.dt_polys);
        dt_polys_list.push_back(sort_item);
      } else {
        dt_polys_list.push_back(std::vector<std::vector<cv::Point2f>>{});
      }
    }

    std::vector<int> indices = {};
    for (int j = 0; j < doc_preprocessor_pipeline_images.size(); j++) {
      if (!dt_polys_list.empty() && !dt_polys_list[j].empty()) {
        indices.push_back(j);
      }
    }
    std::vector<OCRPipelineResult> results(
        doc_preprocessor_pipeline_images.size());
    for (int k = 0; k < results.size(); k++, index++) {
      results[k].input_path = input_path[index];
      results[k].doc_preprocessor_res = doc_preprocessors_pipeline_results[k];
      results[k].dt_polys = dt_polys_list[k];
      results[k].model_settings = model_settings;
      results[k].text_det_params = text_det_params_;
      results[k].text_type = text_type_;
      results[k].text_rec_score_thresh = text_rec_score_thresh_;
    }
    if (!indices.empty()) {
      std::vector<cv::Mat> all_subs_of_imgs = {};
      std::vector<cv::Mat> all_subs_of_imgs_copy = {};
      std::vector<int> chunk_indices(1, 0);
      for (auto &idx : indices) {
        auto result_all_subs_of_img = (*crop_by_polys_)(
            doc_preprocessor_pipeline_images[idx], dt_polys_list[idx]);
        if (!result_all_subs_of_img.ok()) {
          INFOE("Split image fail : ",
                result_all_subs_of_img.status().ToString().c_str());
          exit(-1);
        }
        all_subs_of_imgs.insert(all_subs_of_imgs.end(),
                                result_all_subs_of_img.value().begin(),
                                result_all_subs_of_img.value().end());
        chunk_indices.emplace_back(chunk_indices.back() +
                                   result_all_subs_of_img.value().size());
      }
      for (auto &item : all_subs_of_imgs) {
        all_subs_of_imgs_copy.push_back(item.clone());
      }
      std::vector<int> angles = {};
      if (model_settings["use_textline_orientation"]) {
        textline_orientation_model_->Predict(all_subs_of_imgs_copy);
        auto textline_orientation_model_results =
            static_cast<ClasPredictor *>(textline_orientation_model_.get())
                ->PredictorResult();
        textline_orientation_model_results[0].input_image;
        for (auto &result_angle : textline_orientation_model_results) {
          angles.push_back(result_angle.class_ids[0]);
        }
        auto result_all_subs_of_imgs = RotateImage(all_subs_of_imgs, angles);
        if (!result_all_subs_of_imgs.ok()) {
          INFOE("Rotate images fail : %s",
                result_all_subs_of_imgs.status().ToString().c_str());
          exit(-1);
        }
        all_subs_of_imgs = result_all_subs_of_imgs.value();
      } else {
        angles = std::vector<int>(all_subs_of_imgs.size(), -1);
      }
      for (int l = 0; l < indices.size(); l++) {
        for (int m = chunk_indices[l]; m < chunk_indices[l + 1]; m++) {
          results[indices[l]].textline_orientation_angles.push_back(angles[m]);
        }
      }
      for (int l = 0; l < indices.size(); l++) {
        std::vector<cv::Mat> all_subs_of_img = {};
        for (int m = chunk_indices[l]; m < chunk_indices[l + 1]; m++) {
          all_subs_of_img.push_back(all_subs_of_imgs[m]);
        }
        std::vector<std::pair<std::pair<int, float>, TextRecPredictorResult>>
            sub_img_info_list = {};

        for (int m = 0; m < all_subs_of_img.size(); m++) {
          int sub_img_id = m;
          float sub_img_ratio = (float)all_subs_of_img[m].size[1] /
                                (float)all_subs_of_img[m].size[0];
          TextRecPredictorResult result;
          sub_img_info_list.push_back({{sub_img_id, sub_img_ratio}, result});
        }
        std::vector<std::pair<int, float>> sorted_subs_info = {};
        for (auto &item : sub_img_info_list) {
          sorted_subs_info.push_back(item.first);
        }
        std::sort(
            sorted_subs_info.begin(), sorted_subs_info.end(),
            [](const std::pair<int, float> &a, const std::pair<int, float> &b) {
              return a.second < b.second;
            });
        std::vector<cv::Mat> sorted_subs_of_img = {};
        for (auto &item : sorted_subs_info) {
          sorted_subs_of_img.push_back(all_subs_of_img[item.first]);
        }
        text_rec_model_->Predict(sorted_subs_of_img);
        auto text_rec_model_results =
            static_cast<TextRecPredictor *>(text_rec_model_.get())
                ->PredictorResult();
        for (int m = 0; m < text_rec_model_results.size(); m++) {
          int sub_img_id = sorted_subs_info[m].first;
          sub_img_info_list[sub_img_id].second = text_rec_model_results[m];
        }
        for (int sno = 0; sno < sub_img_info_list.size(); sno++) {
          auto rec_res = sub_img_info_list[sno].second;
          if (rec_res.rec_score >= text_rec_score_thresh_) {
            results[l].rec_texts.push_back(rec_res.rec_text);
            results[l].rec_scores.push_back(rec_res.rec_score);
            results[l].rec_polys.push_back(dt_polys_list[l][sno]);
            results[l].vis_fonts = rec_res.vis_font;
          }
        }
      }
    }
    for (auto &res : results) {
      if (text_type_ == "general") {
        res.rec_boxes =
            ComponentsProcessor::ConvertPointsToBoxes(res.rec_polys);
      }
      pipeline_result_vec_.push_back(res);
      base_results.push_back(std::unique_ptr<BaseCVResult>(new OCRResult(res)));
    }
  }
  return base_results;
}

std::vector<std::unique_ptr<BaseCVResult>>
OCRPipeline::Predict(const std::vector<std::string> &input) {
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

void _OCRPipeline::OverrideConfig() {
  auto &data = config_.Data();
  if (params_.doc_orientation_classify_model_name.has_value()) {
    auto it = config_.FindKey("DocOrientationClassify.model_name");
    if (!it.ok()) {
      data["SubPipelines.DocPreprocessor.SubModules.DocOrientationClassify."
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
      data["SubPipelines.DocPreprocessor.SubModules.DocOrientationClassify."
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
      data["SubPipelines.DocPreprocessor.SubModules.DocUnwarping.model_name"] =
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
      data["SubPipelines.DocPreprocessor.SubModules.DocUnwarping.model_dir"] =
          params_.doc_unwarping_model_dir.value();
    } else {
      auto key = it.value().first;
      data.erase(data.find(key));
      data[key] = params_.doc_unwarping_model_dir.value();
    }
  }
  if (params_.text_detection_model_name.has_value()) {
    auto it = config_.FindKey("TextDetection.model_name");
    if (!it.ok()) {
      data["SubModules.TextDetection.model_name"] =
          params_.text_detection_model_name.value();
    } else {
      auto key = it.value().first;
      data.erase(data.find(key));
      data[key] = params_.text_detection_model_name.value();
    }
  }
  if (params_.text_detection_model_dir.has_value()) {
    auto it = config_.FindKey("TextDetection.model_dir");
    if (!it.ok()) {
      data["SubModules.TextDetection.model_dir"] =
          params_.text_detection_model_dir.value();
    } else {
      auto key = it.value().first;
      data.erase(data.find(key));
      data[key] = params_.text_detection_model_dir.value();
    }
  }
  if (params_.textline_orientation_model_name.has_value()) {
    auto it = config_.FindKey("TextLineOrientation.model_name");
    if (!it.ok()) {
      data["SubModules.TextLineOrientation.model_name"] =
          params_.textline_orientation_model_name.value();
    } else {
      auto key = it.value().first;
      data.erase(data.find(key));
      data[key] = params_.textline_orientation_model_name.value();
    }
  }
  if (params_.textline_orientation_model_dir.has_value()) {
    auto it = config_.FindKey("TextLineOrientation.model_dir");
    if (!it.ok()) {
      data["SubModules.TextLineOrientation.model_dir"] =
          params_.textline_orientation_model_dir.value();
    } else {
      auto key = it.value().first;
      data.erase(data.find(key));
      data[key] = params_.textline_orientation_model_dir.value();
    }
  }
  if (params_.textline_orientation_batch_size.has_value()) {
    auto it = config_.FindKey("TextLineOrientation.batch_size");
    if (!it.ok()) {
      data["SubModules.TextLineOrientation.batch_size"] =
          std::to_string(params_.textline_orientation_batch_size.value());
    } else {
      auto key = it.value().first;
      data.erase(data.find(key));
      data[key] =
          std::to_string(params_.textline_orientation_batch_size.value());
    }
  }

  if (params_.text_recognition_model_name.has_value()) {
    auto it = config_.FindKey("TextRecognition.model_name");
    if (!it.ok()) {
      data["SubModules.TextRecognition.model_name"] =
          params_.text_recognition_model_name.value();
    } else {
      auto key = it.value().first;
      data.erase(data.find(key));
      data[key] = params_.text_recognition_model_name.value();
    }
  }
  if (params_.text_recognition_model_dir.has_value()) {
    auto it = config_.FindKey("TextRecognition.model_dir");
    if (!it.ok()) {
      data["SubModules.TextRecognition.model_dir"] =
          params_.text_recognition_model_dir.value();
    } else {
      auto key = it.value().first;
      data.erase(data.find(key));
      data[key] = params_.text_recognition_model_dir.value();
    }
  }
  if (params_.text_recognition_batch_size.has_value()) {
    auto it = config_.FindKey("TextRecognition.batch_size");
    if (!it.ok()) {
      data["SubModules.TextRecognition.batch_size"] =
          std::to_string(params_.text_recognition_batch_size.value());
    } else {
      auto key = it.value().first;
      data.erase(data.find(key));
      data[key] = std::to_string(params_.text_recognition_batch_size.value());
    }
  }

  if (params_.use_doc_orientation_classify.has_value()) {
    auto it = config_.FindKey("DocPreprocessor.use_doc_orientation_classify");
    if (!it.ok()) {
      data["SubPipelines.DocPreprocessor.use_doc_orientation_classify"] =
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
      data["SubPipelines.DocPreprocessor.use_doc_unwarping"] =
          params_.use_doc_unwarping.value() ? "true" : "false";
    } else {
      auto key = it.value().first;
      data.erase(data.find(key));
      data[key] = params_.use_doc_unwarping.value() ? "true" : "false";
    }
  }
  if (params_.use_textline_orientation.has_value()) {
    auto it = config_.FindKey("use_textline_orientation");
    if (!it.ok()) {
      data["use_textline_orientation"] =
          params_.use_textline_orientation.value() ? "true" : "false";
    } else {
      auto key = it.value().first;
      data.erase(data.find(key));
      data[key] = params_.use_textline_orientation.value() ? "true" : "false";
    }
  }
  if (params_.text_det_limit_side_len.has_value()) {
    auto it = config_.FindKey("TextDetection.limit_side_len");
    if (!it.ok()) {
      data["SubModules.TextDetection.limit_side_len"] =
          std::to_string(params_.text_det_limit_side_len.value());
    } else {
      auto key = it.value().first;
      data.erase(data.find(key));
      data[key] = std::to_string(params_.text_det_limit_side_len.value());
    }
  }
  if (params_.text_det_limit_type.has_value()) {
    auto it = config_.FindKey("TextDetection.limit_type");
    if (!it.ok()) {
      data["SubModules.TextDetection.limit_type"] =
          params_.text_det_limit_type.value();
    } else {
      auto key = it.value().first;
      data.erase(data.find(key));
      data[key] = params_.text_det_limit_type.value();
    }
  }
  if (params_.text_det_thresh.has_value()) {
    auto it = config_.FindKey("TextDetection.thresh");
    if (!it.ok()) {
      data["SubModules.TextDetection.thresh"] =
          std::to_string(params_.text_det_thresh.value());
    } else {
      auto key = it.value().first;
      data.erase(data.find(key));
      data[key] = std::to_string(params_.text_det_thresh.value());
    }
  }
  if (params_.text_det_box_thresh.has_value()) {
    auto it = config_.FindKey("TextDetection.box_thresh");
    if (!it.ok()) {
      data["SubModules.TextDetection.box_thresh"] =
          std::to_string(params_.text_det_box_thresh.value());
    } else {
      auto key = it.value().first;
      data.erase(data.find(key));
      data[key] = std::to_string(params_.text_det_box_thresh.value());
    }
  }
  if (params_.text_det_unclip_ratio.has_value()) {
    auto it = config_.FindKey("TextDetection.unclip_ratio");
    if (!it.ok()) {
      data["SubModules.TextDetection.unclip_ratio"] =
          std::to_string(params_.text_det_unclip_ratio.value());
    } else {
      auto key = it.value().first;
      data.erase(data.find(key));
      data[key] = std::to_string(params_.text_det_unclip_ratio.value());
    }
  }
  if (params_.text_det_input_shape.has_value()) {
    auto it = config_.FindKey("TextDetection.input_shape");
    if (!it.ok()) {
      data["SubModules.TextDetection.input_shape"] =
          Utility::VecToString(params_.text_det_input_shape.value());
    } else {
      auto key = it.value().first;
      data.erase(data.find(key));
      data[key] = Utility::VecToString(params_.text_det_input_shape.value());
    }
  }
  if (params_.text_rec_score_thresh.has_value()) {
    auto it = config_.FindKey("TextRecognition.score_thresh");
    if (!it.ok()) {
      data["SubModules.TextRecognition.score_thresh"] =
          std::to_string(params_.text_rec_score_thresh.value());
    } else {
      auto key = it.value().first;
      data.erase(data.find(key));
      data[key] = std::to_string(params_.text_rec_score_thresh.value());
    }
  }
  if (params_.text_rec_input_shape.has_value()) {
    auto it = config_.FindKey("TextRecognition.input_shape");
    if (!it.ok()) {
      data["SubModules.TextRecognition.input_shape"] =
          Utility::VecToString(params_.text_rec_input_shape.value());
    } else {
      auto key = it.value().first;
      data.erase(data.find(key));
      data[key] = Utility::VecToString(params_.text_rec_input_shape.value());
    }
  }
}
