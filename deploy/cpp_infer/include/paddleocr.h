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

#include <include/ocr_cls.h>
#include <include/ocr_det.h>
#include <include/ocr_rec.h>
#include <include/preprocess_op.h>
#include <include/utility.h>

using namespace paddle_infer;

namespace PaddleOCR {

class PaddleOCR {
public:
  explicit PaddleOCR();
  ~PaddleOCR();
  std::vector<std::vector<OCRPredictResult>>
  ocr(std::vector<cv::String> cv_all_img_names, bool det = true,
      bool rec = true, bool cls = true);

private:
  DBDetector *detector_ = nullptr;
  Classifier *classifier_ = nullptr;
  CRNNRecognizer *recognizer_ = nullptr;

  void det(cv::Mat img, std::vector<OCRPredictResult> &ocr_results,
           std::vector<double> &times);
  void rec(std::vector<cv::Mat> img_list,
           std::vector<OCRPredictResult> &ocr_results,
           std::vector<double> &times);
  void cls(std::vector<cv::Mat> img_list,
           std::vector<OCRPredictResult> &ocr_results,
           std::vector<double> &times);
  void log(std::vector<double> &det_times, std::vector<double> &rec_times,
           std::vector<double> &cls_times, int img_num);
};

} // namespace PaddleOCR
