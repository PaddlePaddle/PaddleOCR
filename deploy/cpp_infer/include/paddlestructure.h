// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include <include/paddleocr.h>
#include <include/structure_layout.h>
#include <include/structure_table.h>

namespace PaddleOCR {

class PaddleStructure : public PPOCR {
public:
  explicit PaddleStructure();
  ~PaddleStructure() = default;

  std::vector<StructurePredictResult> structure(cv::Mat img,
                                                bool layout = false,
                                                bool table = true,
                                                bool ocr = false);

  void reset_timer();
  void benchmark_log(int img_num);

private:
  std::vector<double> time_info_table = {0, 0, 0};
  std::vector<double> time_info_layout = {0, 0, 0};

  std::unique_ptr<StructureTableRecognizer> table_model_;
  std::unique_ptr<StructureLayoutRecognizer> layout_model_;

  void layout(cv::Mat img,
              std::vector<StructurePredictResult> &structure_result);

  void table(cv::Mat img, StructurePredictResult &structure_result);

  std::string rebuild_table(std::vector<std::string> rec_html_tags,
                            std::vector<std::vector<int>> rec_boxes,
                            std::vector<OCRPredictResult> &ocr_result);

  float dis(std::vector<int> &box1, std::vector<int> &box2);

  static bool comparison_dis(const std::vector<float> &dis1,
                             const std::vector<float> &dis2) {
    if (dis1[1] < dis2[1]) {
      return true;
    } else if (dis1[1] == dis2[1]) {
      return dis1[0] < dis2[0];
    } else {
      return false;
    }
  }
};

} // namespace PaddleOCR
