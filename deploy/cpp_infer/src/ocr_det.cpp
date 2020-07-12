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

#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "paddle_api.h"
#include "paddle_inference_api.h"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <vector>

#include <cstring>
#include <fstream>
#include <numeric>

#include <include/ocr_det.h>

namespace PaddleOCR {

void DBDetector::LoadModel(const std::string &model_dir, bool use_gpu,
                           const int gpu_id, const int min_subgraph_size,
                           const int batch_size) {
  AnalysisConfig config;
  config.SetModel(model_dir + "/model", model_dir + "/params");

  // for cpu
  config.DisableGpu();
  config.EnableMKLDNN(); // 开启MKLDNN加速
  config.SetCpuMathLibraryNumThreads(10);

  // 使用ZeroCopyTensor，此处必须设置为false
  config.SwitchUseFeedFetchOps(false);
  // 若输入为多个，此处必须设置为true
  config.SwitchSpecifyInputNames(true);
  // config.SwitchIrDebug(true); //
  // 可视化调试选项，若开启，则会在每个图优化过程后生成dot文件
  // config.SwitchIrOptim(false);// 默认为true。如果设置为false，关闭所有优化
  config.EnableMemoryOptim(); // 开启内存/显存复用

  this->predictor_ = CreatePaddlePredictor(config);
  //   predictor_ = std::move(CreatePaddlePredictor(config)); // PaddleDetection
  //   usage
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

  float *input = new float[1 * 3 * resize_img.rows * resize_img.cols];
  this->permute_op_.Run(&resize_img, input);

  auto input_names = this->predictor_->GetInputNames();
  auto input_t = this->predictor_->GetInputTensor(input_names[0]);
  input_t->Reshape({1, 3, resize_img.rows, resize_img.cols});
  input_t->copy_from_cpu(input);

  this->predictor_->ZeroCopyRun();

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

  float *pred = new float[n];
  unsigned char *cbuf = new unsigned char[n];

  for (int i = 0; i < n; i++) {
    pred[i] = float(out_data[i]);
    cbuf[i] = (unsigned char)((out_data[i]) * 255);
  }

  cv::Mat cbuf_map(n2, n3, CV_8UC1, (unsigned char *)cbuf);
  cv::Mat pred_map(n2, n3, CV_32F, (float *)pred);

  const double threshold = 0.3 * 255;
  const double maxvalue = 255;
  cv::Mat bit_map;
  cv::threshold(cbuf_map, bit_map, threshold, maxvalue, cv::THRESH_BINARY);

  boxes = post_processor_.boxes_from_bitmap(pred_map, bit_map);

  boxes = post_processor_.filter_tag_det_res(boxes, ratio_h, ratio_w, srcimg);

  //// visualization
  cv::Point rook_points[boxes.size()][4];
  for (int n = 0; n < boxes.size(); n++) {
    for (int m = 0; m < boxes[0].size(); m++) {
      rook_points[n][m] = cv::Point(int(boxes[n][m][0]), int(boxes[n][m][1]));
    }
  }

  cv::Mat img_vis;
  srcimg.copyTo(img_vis);
  for (int n = 0; n < boxes.size(); n++) {
    const cv::Point *ppt[1] = {rook_points[n]};
    int npt[] = {4};
    cv::polylines(img_vis, ppt, npt, 1, 1, CV_RGB(0, 255, 0), 2, 8, 0);
  }

  imwrite("./det_res.png", img_vis);

  std::cout << "The detection visualized image saved in ./det_res.png"
            << std::endl;

  delete[] input;
  delete[] pred;
  delete[] cbuf;
}

} // namespace PaddleOCR