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

#pragma once

#include <include/ocr_cls.h>
#include <include/ocr_det.h>
#include <include/ocr_rec.h>

namespace PaddleOCR {

class PPOCR {
public:
  explicit PPOCR();
  ~PPOCR();

  std::vector<std::vector<OCRPredictResult>> ocr(std::vector<cv::Mat> img_list,
                                                 bool det = true,
                                                 bool rec = true,
                                                 bool cls = true);
  std::vector<OCRPredictResult> ocr(cv::Mat img, bool det = true,
                                    bool rec = true, bool cls = true);

  void reset_timer();
  void benchmark_log(int img_num);

protected:
  std::vector<double> time_info_det = {0, 0, 0};
  std::vector<double> time_info_rec = {0, 0, 0};
  std::vector<double> time_info_cls = {0, 0, 0};

  void det(cv::Mat img, std::vector<OCRPredictResult> &ocr_results);
  void rec(std::vector<cv::Mat> img_list,
           std::vector<OCRPredictResult> &ocr_results);
  void cls(std::vector<cv::Mat> img_list,
           std::vector<OCRPredictResult> &ocr_results);

private:
  DBDetector *detector_ = nullptr;
  Classifier *classifier_ = nullptr;
  CRNNRecognizer *recognizer_ = nullptr;
};

} // namespace PaddleOCR
