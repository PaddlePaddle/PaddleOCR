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

#pragma once

#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/optional.h"
#include "src/base/base_pipeline.h"
#include "src/common/image_batch_sampler.h"
#include "src/common/processors.h"
#include "src/modules/image_classification/predictor.h"
#include "src/modules/text_detection/predictor.h"
#include "src/modules/text_recognition/predictor.h"
#include "src/pipelines/doc_preprocessor/pipeline.h"
#include "src/utils/ilogger.h"
#include "src/utils/utility.h"

struct TextDetParams {
  int text_det_limit_side_len = -1;
  std::string text_det_limit_type = "";
  int text_det_max_side_limit = -1;
  float text_det_thresh = -1;
  float text_det_box_thresh = -1;
  float text_det_unclip_ratio = -1;
};

struct OCRPipelineResult {
  std::string input_path = "";
  DocPreprocessorPipelineResult doc_preprocessor_res;
  std::vector<std::vector<cv::Point2f>> dt_polys = {};
  std::unordered_map<std::string, bool> model_settings = {};
  TextDetParams text_det_params;
  std::string text_type = "";
  float text_rec_score_thresh = 0.0;
  std::vector<std::string> rec_texts = {};
  std::vector<float> rec_scores = {};
  std::vector<int> textline_orientation_angles = {};
  std::vector<std::vector<cv::Point2f>> rec_polys = {};
  std::vector<std::array<float, 4>> rec_boxes = {};
  std::string vis_fonts = "";
};

struct OCRPipelineParams {
  absl::optional<std::string> doc_orientation_classify_model_name =
      absl::nullopt;
  absl::optional<std::string> doc_orientation_classify_model_dir =
      absl::nullopt;
  absl::optional<std::string> doc_unwarping_model_name = absl::nullopt;
  absl::optional<std::string> doc_unwarping_model_dir = absl::nullopt;
  absl::optional<std::string> text_detection_model_name = absl::nullopt;
  absl::optional<std::string> text_detection_model_dir = absl::nullopt;
  absl::optional<std::string> textline_orientation_model_name = absl::nullopt;
  absl::optional<std::string> textline_orientation_model_dir = absl::nullopt;
  absl::optional<int> textline_orientation_batch_size = absl::nullopt;
  absl::optional<std::string> text_recognition_model_name = absl::nullopt;
  absl::optional<std::string> text_recognition_model_dir = absl::nullopt;
  absl::optional<int> text_recognition_batch_size = absl::nullopt;
  absl::optional<bool> use_doc_orientation_classify = absl::nullopt;
  absl::optional<bool> use_doc_unwarping = absl::nullopt;
  absl::optional<bool> use_textline_orientation = absl::nullopt;
  absl::optional<int> text_det_limit_side_len = absl::nullopt;
  absl::optional<std::string> text_det_limit_type = absl::nullopt;
  absl::optional<float> text_det_thresh = absl::nullopt;
  absl::optional<float> text_det_box_thresh = absl::nullopt;
  absl::optional<float> text_det_unclip_ratio = absl::nullopt;
  absl::optional<std::vector<int>> text_det_input_shape = absl::nullopt;
  absl::optional<float> text_rec_score_thresh = absl::nullopt;
  absl::optional<std::vector<int>> text_rec_input_shape = absl::nullopt;
  absl::optional<std::string> lang = absl::nullopt;
  absl::optional<std::string> ocr_version = absl::nullopt;
  absl::optional<std::string> vis_font_dir = absl::nullopt;
  absl::optional<std::string> device = absl::nullopt;
  bool enable_mkldnn = true;
  int mkldnn_cache_capacity = 10;
  std::string precision = "fp32";
  int cpu_threads = 8;
  int thread_num = 1;
  absl::optional<Utility::PaddleXConfigVariant> paddlex_config = absl::nullopt;
};

class _OCRPipeline : public BasePipeline {
public:
  explicit _OCRPipeline(const OCRPipelineParams &params);
  virtual ~_OCRPipeline() = default;
  _OCRPipeline() = delete;

  std::vector<std::unique_ptr<BaseCVResult>>
  Predict(const std::vector<std::string> &input) override;

  std::vector<OCRPipelineResult> PipelineResult() const {
    return pipeline_result_vec_;
  };

  static absl::StatusOr<std::vector<cv::Mat>>
  RotateImage(const std::vector<cv::Mat> &image_array_list,
              const std::vector<int> &rotate_angle_list);

  std::unordered_map<std::string, bool> GetModelSettings() const;
  TextDetParams GetTextDetParams() const { return text_det_params_; };

  void OverrideConfig();

private:
  OCRPipelineParams params_;
  YamlConfig config_;
  std::unique_ptr<BaseBatchSampler> batch_sampler_ptr_;
  std::vector<OCRPipelineResult> pipeline_result_vec_;
  bool use_doc_preprocessor_ = false;
  bool use_doc_orientation_classify_ = false;
  bool use_doc_unwarping_ = false;
  std::unique_ptr<BasePipeline> doc_preprocessors_pipeline_;
  bool use_textline_orientation_ = false;
  std::unique_ptr<BasePredictor> textline_orientation_model_;
  std::unique_ptr<BasePredictor> text_det_model_;
  std::unique_ptr<BasePredictor> text_rec_model_;
  std::unique_ptr<CropByPolys> crop_by_polys_;
  std::function<std::vector<std::vector<cv::Point2f>>(
      const std::vector<std::vector<cv::Point2f>> &)>
      sort_boxes_;
  float text_rec_score_thresh_ = 0.0;
  std::string text_type_;
  TextDetParams text_det_params_;
};

class OCRPipeline
    : public AutoParallelSimpleInferencePipeline<
          _OCRPipeline, OCRPipelineParams, std::vector<std::string>,
          std::vector<std::unique_ptr<BaseCVResult>>> {
public:
  OCRPipeline(const OCRPipelineParams &params)
      : AutoParallelSimpleInferencePipeline(params),
        thread_num_(params.thread_num) {
    if (thread_num_ == 1) {
      infer_ = std::unique_ptr<BasePipeline>(new _OCRPipeline(params));
    }
  };

  std::vector<std::unique_ptr<BaseCVResult>>
  Predict(const std::vector<std::string> &input) override;

private:
  int thread_num_;
  std::unique_ptr<BasePipeline> infer_;
  std::unique_ptr<BaseBatchSampler> batch_sampler_ptr_;
};
