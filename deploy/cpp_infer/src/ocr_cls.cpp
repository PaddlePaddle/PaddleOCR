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

void Classifier::Run(std::vector<cv::Mat> img_list,
                     std::vector<int> &cls_labels,
                     std::vector<float> &cls_scores,
                     std::vector<double> &times) {
  std::chrono::duration<float> preprocess_diff =
      std::chrono::steady_clock::now() - std::chrono::steady_clock::now();
  std::chrono::duration<float> inference_diff =
      std::chrono::steady_clock::now() - std::chrono::steady_clock::now();
  std::chrono::duration<float> postprocess_diff =
      std::chrono::steady_clock::now() - std::chrono::steady_clock::now();

  int img_num = img_list.size();
  std::vector<int> cls_image_shape = {3, 48, 192};
  for (int beg_img_no = 0; beg_img_no < img_num;
       beg_img_no += this->cls_batch_num_) {
    auto preprocess_start = std::chrono::steady_clock::now();
    int end_img_no = min(img_num, beg_img_no + this->cls_batch_num_);
    int batch_num = end_img_no - beg_img_no;
    // preprocess
    std::vector<cv::Mat> norm_img_batch;
    for (int ino = beg_img_no; ino < end_img_no; ino++) {
      cv::Mat srcimg;
      img_list[ino].copyTo(srcimg);
      cv::Mat resize_img;
      this->resize_op_.Run(srcimg, resize_img, this->use_tensorrt_,
                           cls_image_shape);

      this->normalize_op_.Run(&resize_img, this->mean_, this->scale_,
                              this->is_scale_);
      norm_img_batch.push_back(resize_img);
    }
    std::vector<float> input(batch_num * cls_image_shape[0] *
                                 cls_image_shape[1] * cls_image_shape[2],
                             0.0f);
    this->permute_op_.Run(norm_img_batch, input.data());
    auto preprocess_end = std::chrono::steady_clock::now();
    preprocess_diff += preprocess_end - preprocess_start;

    // inference.
    auto input_names = this->predictor_->GetInputNames();
    auto input_t = this->predictor_->GetInputHandle(input_names[0]);
    input_t->Reshape({batch_num, cls_image_shape[0], cls_image_shape[1],
                      cls_image_shape[2]});
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

    // postprocess
    auto postprocess_start = std::chrono::steady_clock::now();
    for (int batch_idx = 0; batch_idx < predict_shape[0]; batch_idx++) {
      int label = int(
          Utility::argmax(&predict_batch[batch_idx * predict_shape[1]],
                          &predict_batch[(batch_idx + 1) * predict_shape[1]]));
      float score = float(*std::max_element(
          &predict_batch[batch_idx * predict_shape[1]],
          &predict_batch[(batch_idx + 1) * predict_shape[1]]));
      cls_labels[beg_img_no + batch_idx] = label;
      cls_scores[beg_img_no + batch_idx] = score;
    }
    auto postprocess_end = std::chrono::steady_clock::now();
    postprocess_diff += postprocess_end - postprocess_start;
  }
  times.push_back(double(preprocess_diff.count() * 1000));
  times.push_back(double(inference_diff.count() * 1000));
  times.push_back(double(postprocess_diff.count() * 1000));
}

void Classifier::LoadModel(const std::string &model_dir) {
  AnalysisConfig config;
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
      if (!Utility::PathExists("./trt_cls_shape.txt")){
        config.CollectShapeRangeInfo("./trt_cls_shape.txt");
      } else { 
        config.EnableTunedTensorRtDynamicShape("./trt_cls_shape.txt", true);
      }
    }
  } else {
    config.DisableGpu();
    if (this->use_mkldnn_) {
      config.EnableMKLDNN();
    }
    config.SetCpuMathLibraryNumThreads(this->cpu_math_library_num_threads_);
  }

  // false for zero copy tensor
  config.SwitchUseFeedFetchOps(false);
  // true for multiple input
  config.SwitchSpecifyInputNames(true);

  config.SwitchIrOptim(true);

  config.EnableMemoryOptim();
  config.DisableGlogInfo();

  this->predictor_ = CreatePredictor(config);
}
} // namespace PaddleOCR
