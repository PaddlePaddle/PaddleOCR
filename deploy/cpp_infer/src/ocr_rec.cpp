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

void CRNNRecognizer::Run(std::vector<cv::Mat> img_list,
                         std::vector<std::string> &rec_texts,
                         std::vector<float> &rec_text_scores,
                         std::vector<double> &times) {
  std::chrono::duration<float> preprocess_diff =
      std::chrono::steady_clock::now() - std::chrono::steady_clock::now();
  std::chrono::duration<float> inference_diff =
      std::chrono::steady_clock::now() - std::chrono::steady_clock::now();
  std::chrono::duration<float> postprocess_diff =
      std::chrono::steady_clock::now() - std::chrono::steady_clock::now();

  int img_num = img_list.size();
  std::vector<float> width_list;
  for (int i = 0; i < img_num; i++) {
    width_list.push_back(float(img_list[i].cols) / img_list[i].rows);
  }
  std::vector<int> indices = Utility::argsort(width_list);

  for (int beg_img_no = 0; beg_img_no < img_num;
       beg_img_no += this->rec_batch_num_) {
    auto preprocess_start = std::chrono::steady_clock::now();
    int end_img_no = min(img_num, beg_img_no + this->rec_batch_num_);
    int batch_num = end_img_no - beg_img_no;
    int imgH = this->rec_image_shape_[1];
    int imgW = this->rec_image_shape_[2];
    float max_wh_ratio = imgW * 1.0 / imgH;
    for (int ino = beg_img_no; ino < end_img_no; ino++) {
      int h = img_list[indices[ino]].rows;
      int w = img_list[indices[ino]].cols;
      float wh_ratio = w * 1.0 / h;
      max_wh_ratio = max(max_wh_ratio, wh_ratio);
    }

    int batch_width = imgW;
    std::vector<cv::Mat> norm_img_batch;
    for (int ino = beg_img_no; ino < end_img_no; ino++) {
      cv::Mat srcimg;
      img_list[indices[ino]].copyTo(srcimg);
      cv::Mat resize_img;
      this->resize_op_.Run(srcimg, resize_img, max_wh_ratio,
                           this->use_tensorrt_, this->rec_image_shape_);
      this->normalize_op_.Run(&resize_img, this->mean_, this->scale_,
                              this->is_scale_);
      norm_img_batch.push_back(resize_img);
      batch_width = max(resize_img.cols, batch_width);
    }

    std::vector<float> input(batch_num * 3 * imgH * batch_width, 0.0f);
    this->permute_op_.Run(norm_img_batch, input.data());
    auto preprocess_end = std::chrono::steady_clock::now();
    preprocess_diff += preprocess_end - preprocess_start;
    // Inference.
    auto input_names = this->predictor_->GetInputNames();
    auto input_t = this->predictor_->GetInputHandle(input_names[0]);
    input_t->Reshape({batch_num, 3, imgH, batch_width});
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
    inference_diff += inference_end - inference_start;
    // ctc decode
    auto postprocess_start = std::chrono::steady_clock::now();
    for (int m = 0; m < predict_shape[0]; m++) {
      std::string str_res;
      int argmax_idx;
      int last_index = 0;
      float score = 0.f;
      int count = 0;
      float max_value = 0.0f;

      for (int n = 0; n < predict_shape[1]; n++) {
        argmax_idx = int(Utility::argmax(
            &predict_batch[(m * predict_shape[1] + n) * predict_shape[2]],
            &predict_batch[(m * predict_shape[1] + n + 1) * predict_shape[2]]));
        max_value = float(*std::max_element(
            &predict_batch[(m * predict_shape[1] + n) * predict_shape[2]],
            &predict_batch[(m * predict_shape[1] + n + 1) * predict_shape[2]]));

        if (argmax_idx > 0 && (!(n > 0 && argmax_idx == last_index))) {
          score += max_value;
          count += 1;
          str_res += label_list_[argmax_idx];
        }
        last_index = argmax_idx;
      }
      score /= count;
      if (isnan(score)) {
        continue;
      }
      rec_texts[indices[beg_img_no + m]] = str_res;
      rec_text_scores[indices[beg_img_no + m]] = score;
    }
    auto postprocess_end = std::chrono::steady_clock::now();
    postprocess_diff += postprocess_end - postprocess_start;
  }
  times.push_back(double(preprocess_diff.count() * 1000));
  times.push_back(double(inference_diff.count() * 1000));
  times.push_back(double(postprocess_diff.count() * 1000));
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
      config.EnableTensorRtEngine(1 << 20, 10, 3, precision, false, false);
      int imgH = this->rec_image_shape_[1];
      int imgW = this->rec_image_shape_[2];
      std::map<std::string, std::vector<int>> min_input_shape = {
          {"x", {1, 3, imgH, 10}}, {"lstm_0.tmp_0", {10, 1, 96}}};
      std::map<std::string, std::vector<int>> max_input_shape = {
          {"x", {1, 3, imgH, 2000}}, {"lstm_0.tmp_0", {1000, 1, 96}}};
      std::map<std::string, std::vector<int>> opt_input_shape = {
          {"x", {1, 3, imgH, imgW}}, {"lstm_0.tmp_0", {25, 1, 96}}};

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

  // get pass_builder object
  auto pass_builder = config.pass_builder();
  // delete "matmul_transpose_reshape_fuse_pass"
  pass_builder->DeletePass("matmul_transpose_reshape_fuse_pass");
  config.SwitchUseFeedFetchOps(false);
  // true for multiple input
  config.SwitchSpecifyInputNames(true);

  config.SwitchIrOptim(true);

  config.EnableMemoryOptim();
  //   config.DisableGlogInfo();

  this->predictor_ = CreatePredictor(config);
}

} // namespace PaddleOCR
