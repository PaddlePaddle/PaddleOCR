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
          {"x", {1, 3, 50, 50}},
          {"conv2d_92.tmp_0", {1, 96, 20, 20}},
          {"conv2d_91.tmp_0", {1, 96, 10, 10}},
          {"nearest_interp_v2_1.tmp_0", {1, 96, 10, 10}},
          {"nearest_interp_v2_2.tmp_0", {1, 96, 20, 20}},
          {"nearest_interp_v2_3.tmp_0", {1, 24, 20, 20}},
          {"nearest_interp_v2_4.tmp_0", {1, 24, 20, 20}},
          {"nearest_interp_v2_5.tmp_0", {1, 24, 20, 20}},
          {"elementwise_add_7", {1, 56, 2, 2}},
          {"nearest_interp_v2_0.tmp_0", {1, 96, 2, 2}}};
      std::map<std::string, std::vector<int>> max_input_shape = {
          {"x", {1, 3, this->max_side_len_, this->max_side_len_}},
          {"conv2d_92.tmp_0", {1, 96, 400, 400}},
          {"conv2d_91.tmp_0", {1, 96, 200, 200}},
          {"nearest_interp_v2_1.tmp_0", {1, 96, 200, 200}},
          {"nearest_interp_v2_2.tmp_0", {1, 96, 400, 400}},
          {"nearest_interp_v2_3.tmp_0", {1, 24, 400, 400}},
          {"nearest_interp_v2_4.tmp_0", {1, 24, 400, 400}},
          {"nearest_interp_v2_5.tmp_0", {1, 24, 400, 400}},
          {"elementwise_add_7", {1, 56, 400, 400}},
          {"nearest_interp_v2_0.tmp_0", {1, 96, 400, 400}}};
      std::map<std::string, std::vector<int>> opt_input_shape = {
          {"x", {1, 3, 640, 640}},
          {"conv2d_92.tmp_0", {1, 96, 160, 160}},
          {"conv2d_91.tmp_0", {1, 96, 80, 80}},
          {"nearest_interp_v2_1.tmp_0", {1, 96, 80, 80}},
          {"nearest_interp_v2_2.tmp_0", {1, 96, 160, 160}},
          {"nearest_interp_v2_3.tmp_0", {1, 24, 160, 160}},
          {"nearest_interp_v2_4.tmp_0", {1, 24, 160, 160}},
          {"nearest_interp_v2_5.tmp_0", {1, 24, 160, 160}},
          {"elementwise_add_7", {1, 56, 40, 40}},
          {"nearest_interp_v2_0.tmp_0", {1, 96, 40, 40}}};

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
  // use zero_copy_run as default
  config.SwitchUseFeedFetchOps(false);
  // true for multiple input
  config.SwitchSpecifyInputNames(true);

  config.SwitchIrOptim(true);

  config.EnableMemoryOptim();
  // config.DisableGlogInfo();

  this->predictor_ = CreatePredictor(config);
}

void DBDetector::Run(cv::Mat &img,
                     std::vector<std::vector<std::vector<int>>> &boxes,
                     std::vector<double> *times) {
  float ratio_h{};
  float ratio_w{};

  cv::Mat srcimg;
  cv::Mat resize_img;
  img.copyTo(srcimg);
  
  auto preprocess_start = std::chrono::steady_clock::now();
  this->resize_op_.Run(img, resize_img, this->max_side_len_, ratio_h, ratio_w,
                       this->use_tensorrt_);

  this->normalize_op_.Run(&resize_img, this->mean_, this->scale_,
                          this->is_scale_);

  std::vector<float> input(1 * 3 * resize_img.rows * resize_img.cols, 0.0f);
  this->permute_op_.Run(&resize_img, input.data());
  auto preprocess_end = std::chrono::steady_clock::now();
    
  // Inference.
  auto inference_start = std::chrono::steady_clock::now();
  auto input_names = this->predictor_->GetInputNames();
  auto input_t = this->predictor_->GetInputHandle(input_names[0]);
  input_t->Reshape({1, 3, resize_img.rows, resize_img.cols});
  input_t->CopyFromCpu(input.data());
  
  this->predictor_->Run();
    
  std::vector<float> out_data;
  auto output_names = this->predictor_->GetOutputNames();
  auto output_t = this->predictor_->GetOutputHandle(output_names[0]);
  std::vector<int> output_shape = output_t->shape();
  int out_num = std::accumulate(output_shape.begin(), output_shape.end(), 1,
                                std::multiplies<int>());

  out_data.resize(out_num);
  output_t->CopyToCpu(out_data.data());
  auto inference_end = std::chrono::steady_clock::now();
  
  auto postprocess_start = std::chrono::steady_clock::now();
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
  boxes = post_processor_.BoxesFromBitmap(
      pred_map, dilation_map, this->det_db_box_thresh_,
      this->det_db_unclip_ratio_, this->use_polygon_score_);

  boxes = post_processor_.FilterTagDetRes(boxes, ratio_h, ratio_w, srcimg);
  std::cout << "Detected boxes num: " << boxes.size() << endl;
  auto postprocess_end = std::chrono::steady_clock::now();

  std::chrono::duration<float> preprocess_diff = preprocess_end - preprocess_start;
  times->push_back(double(preprocess_diff.count() * 1000));
  std::chrono::duration<float> inference_diff = inference_end - inference_start;
  times->push_back(double(inference_diff.count() * 1000));
  std::chrono::duration<float> postprocess_diff = postprocess_end - postprocess_start;
  times->push_back(double(postprocess_diff.count() * 1000));
    
  //// visualization
  if (this->visualize_) {
    Utility::VisualizeBboxes(srcimg, boxes);
  }
}

} // namespace PaddleOCR
