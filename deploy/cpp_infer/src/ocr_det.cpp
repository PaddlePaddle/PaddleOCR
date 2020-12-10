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

#include <include/ocr_det.h>

namespace PaddleOCR {

void DBDetector::LoadModel(const std::string &model_dir) {
  AnalysisConfig config;
  config.SetModel(model_dir + ".pdmodel", model_dir + ".pdiparams");

  if (this->use_gpu_) {
    config.EnableUseGpu(this->gpu_mem_, this->gpu_id_);
  } else {
    config.DisableGpu();
    if (this->use_mkldnn_) {
      config.EnableMKLDNN();
      // cache 10 different shapes for mkldnn to avoid memory leak
      config.SetMkldnnCacheCapacity(10);
    }
    config.SetCpuMathLibraryNumThreads(this->cpu_math_library_num_threads_);
  }

  // false for zero copy tensor
  // true for commom tensor
  config.SwitchUseFeedFetchOps(!this->use_zero_copy_run_);
  // true for multiple input
  config.SwitchSpecifyInputNames(true);

  config.SwitchIrOptim(true);

  config.EnableMemoryOptim();
  config.DisableGlogInfo();

  this->predictor_ = CreatePaddlePredictor(config);
}

void DBDetector::Run(cv::Mat &img,
                     std::vector<std::vector<std::vector<int>>> &boxes) {
  float ratio_h{};
  float ratio_w{};

  cv::Mat srcimg;
  cv::Mat resize_img;
  img.copyTo(srcimg);
  this->resize_op_.Run(img, resize_img, this->max_side_len_, ratio_h, ratio_w);

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

  std::vector<float> out_data;
  auto output_names = this->predictor_->GetOutputNames();
  auto output_t = this->predictor_->GetOutputTensor(output_names[0]);
  std::vector<int> output_shape = output_t->shape();
  int out_num = std::accumulate(output_shape.begin(), output_shape.end(), 1,
                                std::multiplies<int>());

  out_data.resize(out_num);
  output_t->copy_to_cpu(out_data.data());

  int n2 = output_shape[2];
  int n3 = output_shape[3];
  int n = n2 * n3;

  std::vector<float> pred(n, 0.0);
  std::vector<unsigned char> cbuf(n, ' ');

  for (int i = 0; i < n; i++) {
    pred[i] = float(out_data[i]);
    cbuf[i] = (unsigned char)((out_data[i]) * 255);
  }

  cv::Mat cbuf_map(n2, n3, CV_8UC1, (unsigned char *)cbuf.data());
  cv::Mat pred_map(n2, n3, CV_32F, (float *)pred.data());

  const double threshold = this->det_db_thresh_ * 255;
  const double maxvalue = 255;
  cv::Mat bit_map;
  cv::threshold(cbuf_map, bit_map, threshold, maxvalue, cv::THRESH_BINARY);
  cv::Mat dilation_map;
  cv::Mat dila_ele = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
  cv::dilate(bit_map, dilation_map, dila_ele);
  boxes = post_processor_.BoxesFromBitmap(pred_map, dilation_map,
                                          this->det_db_box_thresh_,
                                          this->det_db_unclip_ratio_);

  boxes = post_processor_.FilterTagDetRes(boxes, ratio_h, ratio_w, srcimg);

  //// visualization
  if (this->visualize_) {
    Utility::VisualizeBboxes(srcimg, boxes);
  }
}

} // namespace PaddleOCR
