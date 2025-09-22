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

#include "result.h"
#include "src/common/image_batch_sampler.h"

WarpPredictor::WarpPredictor(const WarpPredictorParams &params)
    : BasePredictor(params.model_dir, params.model_name, params.device,
                    params.precision, params.enable_mkldnn,
                    params.mkldnn_cache_capacity, params.cpu_threads,
                    params.batch_size, "image"),
      params_(params) {
  auto status = Build();
  if (!status.ok()) {
    INFOE("Build fail: %s", status.ToString().c_str());
    exit(-1);
  }
};

absl::Status WarpPredictor::Build() {
  const auto &pre_params = config_.PreProcessOpInfo();
  Register<ReadImage>("Read", "BGR");
  Register<Normalize>("Normalize", 1.0 / 255.0, 0.0, 1.0);
  Register<ToCHWImage>("ToCHW");
  Register<ToBatch>("ToBatch");

  infer_ptr_ = CreateStaticInfer();
  const auto &post_params = config_.PostProcessOpInfo();
  post_op_["DocTr"] = std::unique_ptr<DocTrPostProcess>(new DocTrPostProcess());
  return absl::OkStatus();
};

std::vector<std::unique_ptr<BaseCVResult>>
WarpPredictor::Process(std::vector<cv::Mat> &batch_data) {
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

  auto batch_normalize = pre_op_.at("Normalize")->Apply(batch_read.value());
  if (!batch_normalize.ok()) {
    INFOE(batch_normalize.status().ToString().c_str());
    exit(-1);
  }
  auto batch_tochw = pre_op_.at("ToCHW")->Apply(batch_normalize.value());
  if (!batch_tochw.ok()) {
    INFOE(batch_tochw.status().ToString().c_str());
    exit(-1);
  }
  auto batch_tobatch = pre_op_.at("ToBatch")->Apply(batch_tochw.value());
  if (!batch_tobatch.ok()) {
    INFOE(batch_tobatch.status().ToString().c_str());
    exit(-1);
  }
  auto batch_infer = infer_ptr_->Apply(batch_tobatch.value());
  if (!batch_infer.ok()) {
    INFOE(batch_infer.status().ToString().c_str());
    exit(-1);
  }
  auto warp_result = post_op_.at("DocTr")->Apply(batch_infer.value()[0]);

  if (!warp_result.ok()) {
    INFOE(warp_result.status().ToString().c_str());
    exit(-1);
  }
  std::vector<std::unique_ptr<BaseCVResult>> base_cv_result_ptr_vec = {};
  for (int i = 0; i < warp_result.value().size(); i++, input_index_++) {
    WarpPredictorResult predictor_result;
    if (!input_path_.empty()) {
      if (input_index_ == input_path_.size())
        input_index_ = 0;
      predictor_result.input_path = input_path_[input_index_];
    }
    predictor_result.input_image = origin_image[i];
    predictor_result.doctr_img = warp_result.value()[i];
    predictor_result_vec_.push_back(predictor_result);
    base_cv_result_ptr_vec.push_back(
        std::unique_ptr<BaseCVResult>(new DocTrResult(predictor_result)));
  }
  return base_cv_result_ptr_vec;
}
