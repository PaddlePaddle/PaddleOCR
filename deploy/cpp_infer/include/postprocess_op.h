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

#include "include/clipper.h"
#include "include/utility.h"

namespace PaddleOCR {

class DBPostProcessor {
public:
  void GetContourArea(const std::vector<std::vector<float>> &box,
                      float unclip_ratio, float &distance);

  cv::RotatedRect UnClip(std::vector<std::vector<float>> box,
                         const float &unclip_ratio);

  float **Mat2Vec(cv::Mat mat);

  std::vector<std::vector<int>>
  OrderPointsClockwise(std::vector<std::vector<int>> pts);

  std::vector<std::vector<float>> GetMiniBoxes(cv::RotatedRect box,
                                               float &ssid);

  float BoxScoreFast(std::vector<std::vector<float>> box_array, cv::Mat pred);
  float PolygonScoreAcc(std::vector<cv::Point> contour, cv::Mat pred);

  std::vector<std::vector<std::vector<int>>>
  BoxesFromBitmap(const cv::Mat pred, const cv::Mat bitmap,
                  const float &box_thresh, const float &det_db_unclip_ratio,
                  const std::string &det_db_score_mode);

  std::vector<std::vector<std::vector<int>>>
  FilterTagDetRes(std::vector<std::vector<std::vector<int>>> boxes,
                  float ratio_h, float ratio_w, cv::Mat srcimg);

private:
  static bool XsortInt(std::vector<int> a, std::vector<int> b);

  static bool XsortFp32(std::vector<float> a, std::vector<float> b);

  std::vector<std::vector<float>> Mat2Vector(cv::Mat mat);

  inline int _max(int a, int b) { return a >= b ? a : b; }

  inline int _min(int a, int b) { return a >= b ? b : a; }

  template <class T> inline T clamp(T x, T min, T max) {
    if (x > max)
      return max;
    if (x < min)
      return min;
    return x;
  }

  inline float clampf(float x, float min, float max) {
    if (x > max)
      return max;
    if (x < min)
      return min;
    return x;
  }
};

class TablePostProcessor {
public:
  void init(std::string label_path, bool merge_no_span_structure = true);
  void Run(std::vector<float> &loc_preds, std::vector<float> &structure_probs,
           std::vector<float> &rec_scores, std::vector<int> &loc_preds_shape,
           std::vector<int> &structure_probs_shape,
           std::vector<std::vector<std::string>> &rec_html_tag_batch,
           std::vector<std::vector<std::vector<int>>> &rec_boxes_batch,
           std::vector<int> &width_list, std::vector<int> &height_list);

private:
  std::vector<std::string> label_list_;
  std::string end = "eos";
  std::string beg = "sos";
};

class PicodetPostProcessor {
public:
  void init(std::string label_path, const double score_threshold = 0.4,
            const double nms_threshold = 0.5,
            const std::vector<int> &fpn_stride = {8, 16, 32, 64});
  void Run(std::vector<StructurePredictResult> &results,
           std::vector<std::vector<float>> outs, std::vector<int> ori_shape,
           std::vector<int> resize_shape, int eg_max);
  std::vector<int> fpn_stride_ = {8, 16, 32, 64};

private:
  StructurePredictResult disPred2Bbox(std::vector<float> bbox_pred, int label,
                                      float score, int x, int y, int stride,
                                      std::vector<int> im_shape, int reg_max);
  void nms(std::vector<StructurePredictResult> &input_boxes,
           float nms_threshold);

  std::vector<std::string> label_list_;
  double score_threshold_ = 0.4;
  double nms_threshold_ = 0.5;
  int num_class_ = 5;
};

} // namespace PaddleOCR
