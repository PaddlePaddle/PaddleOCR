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

#include "result.h"

#include <algorithm>
#include <fstream>
#include <string>

#include "src/utils/utility.h"
#include "third_party/nlohmann/json.hpp"

using json = nlohmann::json;
void DocPreprocessorResult::SaveToImg(const std::string &save_path) {
  cv::Mat input_img = pipeline_result_.input_image.clone();
  cv::Mat rot_img = pipeline_result_.rotate_image.clone();
  cv::Mat output_img = pipeline_result_.output_image.clone();
  bool use_doc_orientation_classify =
      pipeline_result_.model_settings.at("use_doc_orientation_classify");
  bool use_doc_unwarping =
      pipeline_result_.model_settings.at("use_doc_unwarping");
  int angle = pipeline_result_.angle;
  int h1 = input_img.size[0], w1 = input_img.size[1];
  int h2 = rot_img.size[0], w2 = rot_img.size[1];
  int h3 = output_img.size[0], w3 = output_img.size[1];
  int h = std::max(h1, std::max(h2, h3));
  int total_w = w1 + w2 + w3;
  int final_h = h + 25;

  cv::Mat img_show(final_h, total_w, CV_8UC3, cv::Scalar(255, 255, 255));

  input_img.copyTo(img_show(cv::Rect(0, 0, w1, h1)));
  rot_img.copyTo(img_show(cv::Rect(w1, 0, w2, h2)));
  output_img.copyTo(img_show(cv::Rect(w1 + w2, 0, w3, h3)));
  pipeline_result_.image_all = img_show.clone();
  std::vector<std::string> txt_list = {
      "Original Image",
      "Rotated Image (" +
          std::string(use_doc_orientation_classify ? "True" : "False") + ", " +
          std::to_string(angle) + ")",
      "Unwarping Image (" + std::string(use_doc_unwarping ? "True" : "False") +
          ")"};
  std::vector<int> region_w_list = {w1, w2, w3};
  std::vector<int> beg_w_list = {0, w1, w1 + w2};
  for (int tno = 0; tno < 3; ++tno) {
    DrawText(img_show, txt_list[tno], beg_w_list[tno], h, region_w_list[tno]);
  }
  auto full_path = Utility::SmartCreateDirectoryForImage(
      save_path, pipeline_result_.input_path);
  if (!full_path.ok()) {
    INFOE(full_path.status().ToString().c_str());
    exit(-1);
  }
  bool success = cv::imwrite(full_path.value(), img_show);
  if (!success) {
    INFOE("Error: Failed to write the image : %s", full_path.value().c_str());
    exit(-1);
  }
}

void DocPreprocessorResult::Print() const {
  std::cout << "{\n  \"res\": {" << std::endl;

  std::cout << "    \"input_path\": {" << pipeline_result_.input_path << "},"
            << std::endl;
  std::cout << "    \"model_settings\": {"
            << "use_doc_orientation_classify: " +
                   std::string(pipeline_result_.model_settings.at(
                                   "use_doc_orientation_classify")
                                   ? "True"
                                   : "False")
            << ", use_doc_unwarping: " +
                   std::string(
                       pipeline_result_.model_settings.at("use_doc_unwarping")
                           ? "True"
                           : "False")
            << "}," << std::endl;
  std::cout << "    \"angle\": {" << pipeline_result_.angle << "},"
            << std::endl;
  std::cout << "}" << std::endl;
}

void DocPreprocessorResult::SaveToJson(const std::string &save_path) const {
  nlohmann::ordered_json j;

  j["input_path"] = pipeline_result_.input_path;
  j["page_index"] = nullptr; //********
  j["model_settings"] = pipeline_result_.model_settings;
  j["angle"] = pipeline_result_.angle;

  auto full_path = Utility::SmartCreateDirectoryForJson(
      save_path, pipeline_result_.input_path);
  if (!full_path.ok()) {
    INFOE(full_path.status().ToString().c_str());
    exit(-1);
  }
  std::ofstream file(full_path.value());
  if (file.is_open()) {
    file << j.dump(4);
    file.close();
  } else {
    INFOE("Could not open file for writing : %s", save_path.c_str());
    exit(-1);
  }
}

void DocPreprocessorResult::DrawText(cv::Mat &img, const std::string &text,
                                     int x, int y, int width) {
  int fontFace = cv::FONT_HERSHEY_SIMPLEX;
  double fontScale = 0.7;
  int thickness = 2;
  int baseline = 0;
  cv::Size textSize =
      cv::getTextSize(text, fontFace, fontScale, thickness, &baseline);

  putText(img, text, cv::Point(x + 10, y + textSize.height + 2), fontFace,
          fontScale, cv::Scalar(0, 0, 0), thickness, cv::LINE_AA);
}
