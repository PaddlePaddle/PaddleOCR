// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

#include "predictor.h"
#include "src/base/base_cv_result.h"

class TopkResult : public BaseCVResult {
public:
  TopkResult(ClasPredictorResult predictor_result)
      : BaseCVResult(), predictor_result_(predictor_result){};
  // std::unordered_map<std::string, cv::Mat> ToImg() const override;
  void SaveToImg(const std::string &save_path) override;
  void Print() const override;
  void SaveToJson(const std::string &save_path) const override;
  static int getAdaptiveFontScale(const std::string &text, int imgWidth,
                                  int maxWidth, int minFont, int maxFont,
                                  int thickness, int &outBaseline,
                                  int &outFontFace);

private:
  ClasPredictorResult predictor_result_;
};
