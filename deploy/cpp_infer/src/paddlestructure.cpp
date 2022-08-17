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

#include <include/args.h>
#include <include/paddlestructure.h>

#include "auto_log/autolog.h"
#include <numeric>
#include <sys/stat.h>

namespace PaddleOCR {

PaddleStructure::PaddleStructure() {
  if (FLAGS_table) {
    this->recognizer_ = new StructureTableRecognizer(
        FLAGS_table_model_dir, FLAGS_use_gpu, FLAGS_gpu_id, FLAGS_gpu_mem,
        FLAGS_cpu_threads, FLAGS_enable_mkldnn, FLAGS_table_char_dict_path,
        FLAGS_use_tensorrt, FLAGS_precision, FLAGS_table_batch_num,
        FLAGS_table_max_len);
  }
};

std::vector<std::vector<StructurePredictResult>>
PaddleStructure::structure(std::vector<cv::String> cv_all_img_names,
                           bool layout, bool table) {
  std::vector<double> time_info_det = {0, 0, 0};
  std::vector<double> time_info_rec = {0, 0, 0};
  std::vector<double> time_info_cls = {0, 0, 0};
  std::vector<double> time_info_table = {0, 0, 0};

  std::vector<std::vector<StructurePredictResult>> structure_results;

  if (!Utility::PathExists(FLAGS_output) && FLAGS_det) {
    mkdir(FLAGS_output.c_str(), 0777);
  }
  for (int i = 0; i < cv_all_img_names.size(); ++i) {
    std::vector<StructurePredictResult> structure_result;
    cv::Mat srcimg = cv::imread(cv_all_img_names[i], cv::IMREAD_COLOR);
    if (!srcimg.data) {
      std::cerr << "[ERROR] image read failed! image path: "
                << cv_all_img_names[i] << endl;
      exit(1);
    }
    if (layout) {
    } else {
      StructurePredictResult res;
      res.type = "table";
      res.box = std::vector<int>(4, 0);
      res.box[2] = srcimg.cols;
      res.box[3] = srcimg.rows;
      structure_result.push_back(res);
    }
    cv::Mat roi_img;
    for (int i = 0; i < structure_result.size(); i++) {
      // crop image
      roi_img = Utility::crop_image(srcimg, structure_result[i].box);
      if (structure_result[i].type == "table") {
        this->table(roi_img, structure_result[i], time_info_table,
                    time_info_det, time_info_rec, time_info_cls);
      }
    }
    structure_results.push_back(structure_result);
  }
  return structure_results;
};

void PaddleStructure::table(cv::Mat img,
                            StructurePredictResult &structure_result,
                            std::vector<double> &time_info_table,
                            std::vector<double> &time_info_det,
                            std::vector<double> &time_info_rec,
                            std::vector<double> &time_info_cls) {
  // predict structure
  std::vector<std::vector<std::string>> structure_html_tags;
  std::vector<float> structure_scores(1, 0);
  std::vector<std::vector<std::vector<std::vector<int>>>> structure_boxes;
  std::vector<double> structure_imes;
  std::vector<cv::Mat> img_list;
  img_list.push_back(img);
  this->recognizer_->Run(img_list, structure_html_tags, structure_scores,
                         structure_boxes, structure_imes);
  time_info_table[0] += structure_imes[0];
  time_info_table[1] += structure_imes[1];
  time_info_table[2] += structure_imes[2];

  std::vector<OCRPredictResult> ocr_result;
  std::string html;
  int expand_pixel = 3;

  for (int i = 0; i < img_list.size(); i++) {
    // det
    this->det(img_list[i], ocr_result, time_info_det);
    // crop image
    std::vector<cv::Mat> rec_img_list;
    for (int j = 0; j < ocr_result.size(); j++) {
      int x_collect[4] = {ocr_result[j].box[0][0], ocr_result[j].box[1][0],
                          ocr_result[j].box[2][0], ocr_result[j].box[3][0]};
      int y_collect[4] = {ocr_result[j].box[0][1], ocr_result[j].box[1][1],
                          ocr_result[j].box[2][1], ocr_result[j].box[3][1]};
      int left = int(*std::min_element(x_collect, x_collect + 4));
      int right = int(*std::max_element(x_collect, x_collect + 4));
      int top = int(*std::min_element(y_collect, y_collect + 4));
      int bottom = int(*std::max_element(y_collect, y_collect + 4));
      std::vector<int> box{max(0, left - expand_pixel),
                           max(0, top - expand_pixel),
                           min(img_list[i].cols, right + expand_pixel),
                           min(img_list[i].rows, bottom + expand_pixel)};
      cv::Mat crop_img = Utility::crop_image(img_list[i], box);
      rec_img_list.push_back(crop_img);
    }
    // rec
    this->rec(rec_img_list, ocr_result, time_info_rec);
    // rebuild table
    html = this->rebuild_table(structure_html_tags[i], structure_boxes[i],
                               ocr_result);
    structure_result.html = html;
    structure_result.html_score = structure_scores[i];
  }
};

std::string PaddleStructure::rebuild_table(
    std::vector<std::string> structure_html_tags,
    std::vector<std::vector<std::vector<int>>> structure_boxes,
    std::vector<OCRPredictResult> &ocr_result) {
  // match text in same cell
  std::vector<std::vector<string>> matched(structure_boxes.size(),
                                           std::vector<std::string>());

  for (int i = 0; i < ocr_result.size(); i++) {
    std::vector<std::vector<float>> dis_list(structure_boxes.size(),
                                             std::vector<float>(3, 100000.0));
    for (int j = 0; j < structure_boxes.size(); j++) {
      int x_collect[4] = {ocr_result[i].box[0][0], ocr_result[i].box[1][0],
                          ocr_result[i].box[2][0], ocr_result[i].box[3][0]};
      int y_collect[4] = {ocr_result[i].box[0][1], ocr_result[i].box[1][1],
                          ocr_result[i].box[2][1], ocr_result[i].box[3][1]};
      int left = int(*std::min_element(x_collect, x_collect + 4));
      int right = int(*std::max_element(x_collect, x_collect + 4));
      int top = int(*std::min_element(y_collect, y_collect + 4));
      int bottom = int(*std::max_element(y_collect, y_collect + 4));
      std::vector<std::vector<int>> box(2, std::vector<int>(2, 0));
      box[0][0] = left - 1;
      box[0][1] = top - 1;
      box[1][0] = right + 1;
      box[1][1] = bottom + 1;

      dis_list[j][0] = this->dis(box, structure_boxes[j]);
      dis_list[j][1] = 1 - this->iou(box, structure_boxes[j]);
      dis_list[j][2] = j;
    }
    // find min dis idx
    std::sort(dis_list.begin(), dis_list.end(),
              PaddleStructure::comparison_dis);
    matched[dis_list[0][2]].push_back(ocr_result[i].text);
  }
  // get pred html
  std::string html_str = "";
  int td_tag_idx = 0;
  for (int i = 0; i < structure_html_tags.size(); i++) {
    if (structure_html_tags[i].find("</td>") != std::string::npos) {
      if (structure_html_tags[i].find("<td></td>") != std::string::npos) {
        html_str += "<td>";
      }
      if (matched[td_tag_idx].size() > 0) {
        bool b_with = false;
        if (matched[td_tag_idx][0].find("<b>") != std::string::npos &&
            matched[td_tag_idx].size() > 1) {
          b_with = true;
          html_str += "<b>";
        }
        for (int j = 0; j < matched[td_tag_idx].size(); j++) {
          std::string content = matched[td_tag_idx][j];
          if (matched[td_tag_idx].size() > 1) {
            // remove blank, <b> and </b>
            if (content.length() > 0 && content.at(0) == ' ') {
              content = content.substr(0);
            }
            if (content.length() > 2 && content.substr(0, 3) == "<b>") {
              content = content.substr(3);
            }
            if (content.length() > 4 &&
                content.substr(content.length() - 4) == "</b>") {
              content = content.substr(0, content.length() - 4);
            }
            if (content.empty()) {
              continue;
            }
            // add blank
            if (j != matched[td_tag_idx].size() - 1 &&
                content.at(content.length() - 1) != ' ') {
              content += ' ';
            }
          }
          html_str += content;
        }
        if (b_with) {
          html_str += "</b>";
        }
      }
      if (structure_html_tags[i].find("<td></td>") != std::string::npos) {
        html_str += "</td>";
      } else {
        html_str += structure_html_tags[i];
      }
      td_tag_idx += 1;
    } else {
      html_str += structure_html_tags[i];
    }
  }
  return html_str;
}

float PaddleStructure::iou(std::vector<std::vector<int>> &box1,
                           std::vector<std::vector<int>> &box2) {
  int area1 = max(0, box1[1][0] - box1[0][0]) * max(0, box1[1][1] - box1[0][1]);
  int area2 = max(0, box2[1][0] - box2[0][0]) * max(0, box2[1][1] - box2[0][1]);

  // computing the sum_area
  int sum_area = area1 + area2;

  // find the each point of intersect rectangle
  int x1 = max(box1[0][0], box2[0][0]);
  int y1 = max(box1[0][1], box2[0][1]);
  int x2 = min(box1[1][0], box2[1][0]);
  int y2 = min(box1[1][1], box2[1][1]);

  // judge if there is an intersect
  if (y1 >= y2 || x1 >= x2) {
    return 0.0;
  } else {
    int intersect = (x2 - x1) * (y2 - y1);
    return intersect / (sum_area - intersect + 0.00000001);
  }
}

float PaddleStructure::dis(std::vector<std::vector<int>> &box1,
                           std::vector<std::vector<int>> &box2) {
  int x1_1 = box1[0][0];
  int y1_1 = box1[0][1];
  int x2_1 = box1[1][0];
  int y2_1 = box1[1][1];

  int x1_2 = box2[0][0];
  int y1_2 = box2[0][1];
  int x2_2 = box2[1][0];
  int y2_2 = box2[1][1];

  float dis =
      abs(x1_2 - x1_1) + abs(y1_2 - y1_1) + abs(x2_2 - x2_1) + abs(y2_2 - y2_1);
  float dis_2 = abs(x1_2 - x1_1) + abs(y1_2 - y1_1);
  float dis_3 = abs(x2_2 - x2_1) + abs(y2_2 - y2_1);
  return dis + min(dis_2, dis_3);
}

PaddleStructure::~PaddleStructure() {
  if (this->recognizer_ != nullptr) {
    delete this->recognizer_;
  }
};

} // namespace PaddleOCR