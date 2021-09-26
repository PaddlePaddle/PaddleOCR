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

#include <include/ocr_rec.h>

namespace PaddleOCR {

void CRNNRecognizer::Run(cv::Mat &img, std::vector<double> *times) {
  cv::Mat srcimg;
  img.copyTo(srcimg);
  cv::Mat resize_img;

  float wh_ratio = float(srcimg.cols) / float(srcimg.rows);
  auto preprocess_start = std::chrono::steady_clock::now();
  this->resize_op_.Run(srcimg, resize_img, wh_ratio, this->use_tensorrt_);

  this->normalize_op_.Run(&resize_img, this->mean_, this->scale_,
                          this->is_scale_);

  std::vector<float> input(1 * 3 * resize_img.rows * resize_img.cols, 0.0f);

  this->permute_op_.Run(&resize_img, input.data());
  auto preprocess_end = std::chrono::steady_clock::now();

  // Inference.
  auto input_names = this->predictor_->GetInputNames();
  auto input_t = this->predictor_->GetInputHandle(input_names[0]);
  input_t->Reshape({1, 3, resize_img.rows, resize_img.cols});
  auto inference_start = std::chrono::steady_clock::now();
  input_t->CopyFromCpu(input.data());
  this->predictor_->Run();

  std::vector<float> predict_batch;
  auto output_names = this->predictor_->GetOutputNames();
  auto output_t = this->predictor_->GetOutputHandle(output_names[0]);
  auto predict_shape = output_t->shape();

  int out_num = std::accumulate(predict_shape.begin(), predict_shape.end(), 1,
                                std::multiplies<int>());
  predict_batch.resize(out_num);

  output_t->CopyToCpu(predict_batch.data());
  auto inference_end = std::chrono::steady_clock::now();

  // ctc decode
  auto postprocess_start = std::chrono::steady_clock::now();
  std::vector<std::string> str_res;
  int argmax_idx;
  int last_index = 0;
  float score = 0.f;
  int count = 0;
  float max_value = 0.0f;

  for (int n = 0; n < predict_shape[1]; n++) {
    argmax_idx =
        int(Utility::argmax(&predict_batch[n * predict_shape[2]],
                            &predict_batch[(n + 1) * predict_shape[2]]));
    max_value =
        float(*std::max_element(&predict_batch[n * predict_shape[2]],
                                &predict_batch[(n + 1) * predict_shape[2]]));

    if (argmax_idx > 0 && (!(n > 0 && argmax_idx == last_index))) {
      score += max_value;
      count += 1;
      str_res.push_back(label_list_[argmax_idx]);
    }
    last_index = argmax_idx;
  }
  auto postprocess_end = std::chrono::steady_clock::now();
  score /= count;
  for (int i = 0; i < str_res.size(); i++) {
    std::cout << str_res[i];
  }
  std::cout << "\tscore: " << score << std::endl;

  std::chrono::duration<float> preprocess_diff = preprocess_end - preprocess_start;
  times->push_back(double(preprocess_diff.count() * 1000));
  std::chrono::duration<float> inference_diff = inference_end - inference_start;
  times->push_back(double(inference_diff.count() * 1000));
  std::chrono::duration<float> postprocess_diff = postprocess_end - postprocess_start;
  times->push_back(double(postprocess_diff.count() * 1000));
}

void CRNNRecognizer::LoadModel(const std::string &model_dir) {
  //   AnalysisConfig config;
  paddle_infer::Config config;
  config.SetModel(model_dir + "/inference.pdmodel",
                  model_dir + "/inference.pdiparams");

  if (this->use_gpu_) {
    config.EnableUseGpu(this->gpu_mem_, this->gpu_id_);
    if (this->use_tensorrt_) {
      auto precision = paddle_infer::Config::Precision::kFloat32;
      if (this->precision_ == "fp16") {
        precision = paddle_infer::Config::Precision::kHalf;
      }
     if (this->precision_ == "int8") {
        precision = paddle_infer::Config::Precision::kInt8;
      } 
      config.EnableTensorRtEngine(
          1 << 20, 10, 3,
          precision,
          false, false);

      std::map<std::string, std::vector<int>> min_input_shape = {
          {"x", {1, 3, 32, 10}},
          {"lstm_0.tmp_0", {10, 1, 96}}};
      std::map<std::string, std::vector<int>> max_input_shape = {
          {"x", {1, 3, 32, 2000}},
          {"lstm_0.tmp_0", {1000, 1, 96}}};
      std::map<std::string, std::vector<int>> opt_input_shape = {
          {"x", {1, 3, 32, 320}},
          {"lstm_0.tmp_0", {25, 1, 96}}};

      config.SetTRTDynamicShapeInfo(min_input_shape, max_input_shape,
                                    opt_input_shape);
    }
  } else {
    config.DisableGpu();
    if (this->use_mkldnn_) {
      config.EnableMKLDNN();
      // cache 10 different shapes for mkldnn to avoid memory leak
      config.SetMkldnnCacheCapacity(10);
    }
    config.SetCpuMathLibraryNumThreads(this->cpu_math_library_num_threads_);
  }

  config.SwitchUseFeedFetchOps(false);
  // true for multiple input
  config.SwitchSpecifyInputNames(true);

  config.SwitchIrOptim(true);

  config.EnableMemoryOptim();
//   config.DisableGlogInfo();

  this->predictor_ = CreatePredictor(config);
}

} // namespace PaddleOCR
