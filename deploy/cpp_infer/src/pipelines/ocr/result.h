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

#ifdef USE_FREETYPE
#include <opencv2/freetype.hpp>
#endif
#include "pipeline.h"
#include "src/base/base_cv_result.h"

class OCRResult : public BaseCVResult {
public:
  OCRResult(OCRPipelineResult pipeline_result_)
      : BaseCVResult(), pipeline_result_(pipeline_result_){};

  void SaveToImg(const std::string &save_path) override;
  void Print() const override;
  void SaveToJson(const std::string &save_path) const override;

#ifdef USE_FREETYPE
  static cv::Mat DrawBoxTextFine(const cv::Size &img_ize,
                                 const std::vector<cv::Point2f> &box,
                                 const std::string &txt,
                                 const std::string &vis_font);

  static void DrawVerticalText(cv::Ptr<cv::freetype::FreeType2> &ft2,
                               cv::Mat &img, const std::string &text, int x,
                               int y, int font_height, cv::Scalar color,
                               float line_spacing = 2);
  static int CreateFont(cv::Ptr<cv::freetype::FreeType2> &ft2,
                        const std::string &text, int region_height,
                        int region_width);

  static int CreateFontVertical(cv::Ptr<cv::freetype::FreeType2> &ft2,
                                const std::string &text, int region_height,
                                int region_width, float scale = 1.2f);
  static cv::Size getActualCharSize(cv::Ptr<cv::freetype::FreeType2> &ft2,
                                    const std::string &utf8_char,
                                    int font_height);
#endif
  static std::vector<cv::Point>
  GetMinareaRect(const std::vector<cv::Point> &points);

private:
  OCRPipelineResult pipeline_result_;
};
