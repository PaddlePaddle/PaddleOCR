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

#include <include/structure_layout.h>

namespace PaddleOCR {

void StructureLayoutRecognizer::Run(cv::Mat img,
                                    std::vector<StructurePredictResult> &result,
                                    std::vector<double> &times) {
  std::chrono::duration<float> preprocess_diff =
      std::chrono::steady_clock::now() - std::chrono::steady_clock::now();
  std::chrono::duration<float> inference_diff =
      std::chrono::steady_clock::now() - std::chrono::steady_clock::now();
  std::chrono::duration<float> postprocess_diff =
      std::chrono::steady_clock::now() - std::chrono::steady_clock::now();

  // preprocess
  auto preprocess_start = std::chrono::steady_clock::now();

  cv::Mat srcimg;
  img.copyTo(srcimg);
  cv::Mat resize_img;
  this->resize_op_.Run(srcimg, resize_img, 800, 608);
  this->normalize_op_.Run(&resize_img, this->mean_, this->scale_,
                          this->is_scale_);

  std::vector<float> input(1 * 3 * resize_img.rows * resize_img.cols, 0.0f);
  this->permute_op_.Run(&resize_img, input.data());
  auto preprocess_end = std::chrono::steady_clock::now();
  preprocess_diff += preprocess_end - preprocess_start;

  // inference.
  auto input_names = this->predictor_->GetInputNames();
  auto input_t = this->predictor_->GetInputHandle(input_names[0]);
  input_t->Reshape({1, 3, resize_img.rows, resize_img.cols});
  auto inference_start = std::chrono::steady_clock::now();
  input_t->CopyFromCpu(input.data());

  this->predictor_->Run();

  // Get output tensor
  std::vector<std::vector<float>> out_tensor_list;
  std::vector<std::vector<int>> output_shape_list;
  auto output_names = this->predictor_->GetOutputNames();
  for (int j = 0; j < output_names.size(); j++) {
    auto output_tensor = this->predictor_->GetOutputHandle(output_names[j]);
    std::vector<int> output_shape = output_tensor->shape();
    int out_num = std::accumulate(output_shape.begin(), output_shape.end(), 1,
                                  std::multiplies<int>());
    output_shape_list.push_back(output_shape);

    std::vector<float> out_data;
    out_data.resize(out_num);
    output_tensor->CopyToCpu(out_data.data());
    out_tensor_list.push_back(out_data);
  }
  auto inference_end = std::chrono::steady_clock::now();
  inference_diff += inference_end - inference_start;

  // postprocess
  auto postprocess_start = std::chrono::steady_clock::now();

  std::vector<int> bbox_num;
  int reg_max = 0;
  for (int i = 0; i < out_tensor_list.size(); i++) {
    if (i == this->post_processor_.fpn_stride_.size()) {
      reg_max = output_shape_list[i][2] / 4;
      break;
    }
  }
  std::vector<int> ori_shape = {srcimg.rows, srcimg.cols};
  std::vector<int> resize_shape = {resize_img.rows, resize_img.cols};
  this->post_processor_.Run(result, out_tensor_list, ori_shape, resize_shape,
                            reg_max);
  bbox_num.push_back(result.size());

  auto postprocess_end = std::chrono::steady_clock::now();
  postprocess_diff += postprocess_end - postprocess_start;
  times.push_back(double(preprocess_diff.count() * 1000));
  times.push_back(double(inference_diff.count() * 1000));
  times.push_back(double(postprocess_diff.count() * 1000));
}

void StructureLayoutRecognizer::LoadModel(const std::string &model_dir) {
  paddle_infer::Config config;
  if (Utility::PathExists(model_dir + "/inference.pdmodel") &&
      Utility::PathExists(model_dir + "/inference.pdiparams")) {
    config.SetModel(model_dir + "/inference.pdmodel",
                    model_dir + "/inference.pdiparams");
  } else if (Utility::PathExists(model_dir + "/model.pdmodel") &&
             Utility::PathExists(model_dir + "/model.pdiparams")) {
    config.SetModel(model_dir + "/model.pdmodel",
                    model_dir + "/model.pdiparams");
  } else {
    std::cerr << "[ERROR] not find model.pdiparams or inference.pdiparams in "
              << model_dir << std::endl;
    exit(1);
  }

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
      if (!Utility::PathExists("./trt_layout_shape.txt")) {
        config.CollectShapeRangeInfo("./trt_layout_shape.txt");
      } else {
        config.EnableTunedTensorRtDynamicShape("./trt_layout_shape.txt", true);
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

  this->predictor_ = paddle_infer::CreatePredictor(config);
}
} // namespace PaddleOCR
