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

#include <fstream>
#include <string>

#include "src/utils/utility.h"
#include "third_party/nlohmann/json.hpp"

using json = nlohmann::json;
void TopkResult::SaveToImg(const std::string &save_path) {
  cv::Mat img = predictor_result_.input_image.clone();

  std::ostringstream oss;
  oss << predictor_result_.label_names[0] << " " << std::fixed
      << std::setprecision(2) << predictor_result_.scores[0];
  std::string label_str = oss.str();

  int imgWidth = img.cols;
  int minFont = std::max(12, imgWidth * 2 / 100);
  int maxFont = std::max(18, imgWidth * 5 / 100);
  int baseline = 0;
  int fontFace = 0;
  double fontScale = getAdaptiveFontScale(
      label_str, imgWidth, imgWidth, minFont, maxFont, 2, baseline, fontFace);

  cv::Size textSize =
      cv::getTextSize(label_str, fontFace, fontScale, 2, &baseline);

  int rect_left = 3, rect_top = 3;
  int rect_right = rect_left + textSize.width + 6;
  int rect_bottom = rect_top + textSize.height + 6;
  cv::Scalar bgColor(0, 0, 255);
  cv::rectangle(img, cv::Point(rect_left, rect_top),
                cv::Point(rect_right, rect_bottom), bgColor, cv::FILLED);

  int text_x = rect_left + 3;
  int text_y = rect_top + textSize.height + 2;
  cv::Scalar fontColor(255, 255, 255);
  cv::putText(img, label_str, cv::Point(text_x, text_y), fontFace, fontScale,
              fontColor, 2, cv::LINE_AA);
  absl::StatusOr<std::string> full_path;
  if (predictor_result_.input_path.empty()) {
    auto now = std::chrono::system_clock::now();
    auto now_time = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << "output_" << std::put_time(std::localtime(&now_time), "%Y%m%d_%H%M%S")
       << ".jpg";
    std::string timestamp_filename = ss.str();
    INFOW("Input path is empty, will use %s instead!",
          timestamp_filename.c_str());
    predictor_result_.input_path = timestamp_filename;
    full_path =
        Utility::SmartCreateDirectoryForImage(save_path, timestamp_filename);
  } else {
    full_path = Utility::SmartCreateDirectoryForImage(
        save_path, predictor_result_.input_path);
  }
  if (!full_path.ok()) {
    INFOE(full_path.status().ToString().c_str());
    exit(-1);
  }
  bool success = cv::imwrite(full_path.value(), img);
  if (!success) {
    INFOE("Failed to write the image %s ", full_path.value().c_str());
    exit(-1);
  }
}

void TopkResult::Print() const {
  std::cout << "{\n  \"res\": {" << std::endl;

  std::cout << "    \"input_path\": {" << predictor_result_.input_path << "},"
            << std::endl;
  std::cout << "    \"class_ids\": {" << predictor_result_.class_ids[0] << "},"
            << std::endl;
  std::cout << "    \"scores\": {" << predictor_result_.scores[0] << "},"
            << std::endl;
  std::cout << "    \"label_names\": {" << predictor_result_.label_names[0]
            << "}," << std::endl;
  std::cout << "}" << std::endl;
}

void TopkResult::SaveToJson(const std::string &save_path) const {
  nlohmann::ordered_json j;

  j["input_path"] = predictor_result_.input_path;
  j["page_index"] = nullptr; //********

  json class_ids = json::array();
  for (const auto &item : predictor_result_.class_ids) {
    class_ids.push_back(item);
  }
  json scores = json::array();
  for (const auto &item : predictor_result_.scores) {
    scores.push_back(item);
  }
  json label_names = json::array();
  for (const auto &item : predictor_result_.label_names) {
    label_names.push_back(item);
  }

  j["class_ids"] = class_ids;
  j["scores"] = scores;
  j["label_names"] = label_names;

  auto full_path = Utility::SmartCreateDirectoryForJson(
      save_path, predictor_result_.input_path);
  if (!full_path.ok()) {
    INFOE(full_path.status().ToString().c_str());
    exit(-1);
  }
  std::ofstream file(full_path.value());
  if (file.is_open()) {
    file << j.dump(4);
    file.close();
  } else {
    INFOE("Could not open file for writing: %s", save_path.c_str());
    exit(-1);
  }
}

int TopkResult::getAdaptiveFontScale(const std::string &text, int imgWidth,
                                     int maxWidth, int minFont, int maxFont,
                                     int thickness, int &outBaseline,
                                     int &outFontFace) {
  int fontFace = cv::FONT_HERSHEY_SIMPLEX;
  double fontScale = 1.0;
  int baseline = 0;
  int bestFontSize = minFont;

  for (int fontSize = maxFont; fontSize >= minFont; --fontSize) {
    fontScale = fontSize / 20.0; // 20为基准比例，可调
    int base;
    cv::Size textSize =
        cv::getTextSize(text, fontFace, fontScale, thickness, &base);
    if (textSize.width <= maxWidth) {
      bestFontSize = fontSize;
      outBaseline = base;
      outFontFace = fontFace;
      return fontScale;
    }
  }
  outBaseline = 0;
  outFontFace = fontFace;
  return minFont / 20.0;
}
