// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <include/ocr_cls.h>

namespace PaddleOCR {

cv::Mat Classifier::Run(cv::Mat &img) {
  cv::Mat src_img;
  img.copyTo(src_img);
  cv::Mat resize_img;

  std::vector<int> cls_image_shape = {3, 48, 192};
  int index = 0;
  float wh_ratio = float(img.cols) / float(img.rows);

  this->resize_op_.Run(img, resize_img, cls_image_shape);

  this->normalize_op_.Run(&resize_img, this->mean_, this->scale_,
                          this->is_scale_);

  std::vector<float> input(1 * 3 * resize_img.rows * resize_img.cols, 0.0f);

  this->permute_op_.Run(&resize_img, input.data());

  // Inference.
  if (this->use_zero_copy_run_) {
    auto input_names = this->predictor_->GetInputNames();
    auto input_t = this->predictor_->GetInputTensor(input_names[0]);
    input_t->Reshape({1, 3, resize_img.rows, resize_img.cols});
    input_t->copy_from_cpu(input.data());
    this->predictor_->ZeroCopyRun();
  } else {
    paddle::PaddleTensor input_t;
    input_t.shape = {1, 3, resize_img.rows, resize_img.cols};
    input_t.data =
        paddle::PaddleBuf(input.data(), input.size() * sizeof(float));
    input_t.dtype = PaddleDType::FLOAT32;
    std::vector<paddle::PaddleTensor> outputs;
    this->predictor_->Run({input_t}, &outputs, 1);
  }

  std::vector<float> softmax_out;
  std::vector<int64_t> label_out;
  auto output_names = this->predictor_->GetOutputNames();
  auto softmax_out_t = this->predictor_->GetOutputTensor(output_names[0]);
  auto softmax_shape_out = softmax_out_t->shape();

  int softmax_out_num =
      std::accumulate(softmax_shape_out.begin(), softmax_shape_out.end(), 1,
                      std::multiplies<int>());

  softmax_out.resize(softmax_out_num);

  softmax_out_t->copy_to_cpu(softmax_out.data());

  float score = 0;
  int label = 0;
  for (int i = 0; i < softmax_out_num; i++) {
    if (softmax_out[i] > score) {
      score = softmax_out[i];
      label = i;
    }
  }
  if (label % 2 == 1 && score > this->cls_thresh) {
    cv::rotate(src_img, src_img, 1);
  }
  return src_img;
}

void Classifier::LoadModel(const std::string &model_dir) {
  AnalysisConfig config;
  config.SetModel(model_dir + "/inference.pdmodel",
                  model_dir + "/inference.pdiparams");

  if (this->use_gpu_) {
    config.EnableUseGpu(this->gpu_mem_, this->gpu_id_);
  } else {
    config.DisableGpu();
    if (this->use_mkldnn_) {
      config.EnableMKLDNN();
    }
    config.SetCpuMathLibraryNumThreads(this->cpu_math_library_num_threads_);
  }

  // false for zero copy tensor
  config.SwitchUseFeedFetchOps(!this->use_zero_copy_run_);
  // true for multiple input
  config.SwitchSpecifyInputNames(true);

  config.SwitchIrOptim(true);

  config.EnableMemoryOptim();
  config.DisableGlogInfo();

  this->predictor_ = CreatePaddlePredictor(config);
}
} // namespace PaddleOCR
