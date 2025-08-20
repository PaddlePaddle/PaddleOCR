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

#ifdef USE_FREETYPE
#include <opencv2/freetype.hpp>
#endif
#include <string>

#include "src/utils/utility.h"
#include "third_party/nlohmann/json.hpp"

using json = nlohmann::json;

#ifdef USE_FREETYPE
void TextRecResult::SaveToImg(const std::string &save_path) {
  int image_width = predictor_result_.input_image.size[1];
  int image_height = predictor_result_.input_image.size[0];
  std::string text = predictor_result_.rec_text + "(" +
                     std::to_string(predictor_result_.rec_score) + ")";
  int font = AdjustFontSize(image_width, text);
  cv::Ptr<cv::freetype::FreeType2> ft2 = cv::freetype::createFreeType2();
  ft2->loadFontData(predictor_result_.vis_font, 0);
  int baseline = 0;
  cv::Size text_size = ft2->getTextSize(text, font, -1, &baseline);
  int row_height = text_size.height;
  int new_image_height = image_height + static_cast<int>(row_height * 1.2);
  cv::Mat new_image(new_image_height, image_width, CV_8UC3,
                    cv::Scalar(255, 255, 255));
  predictor_result_.input_image.copyTo(
      new_image(cv::Rect(0, 0, image_width, image_height)));
  cv::Point org(0, image_height + row_height);
  ft2->putText(new_image, text, org, font, cv::Scalar(0, 0, 0), -1, cv::LINE_AA,
               true);

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
  bool success = cv::imwrite(full_path.value(), new_image);
  if (!success) {
    INFOE("Error: Failed to write the image :%s ", full_path.value().c_str());
    exit(-1);
  }
}
int TextRecResult::AdjustFontSize(int image_width,
                                  const std::string &text) const {
  cv::Ptr<cv::freetype::FreeType2> ft2 = cv::freetype::createFreeType2();
  int font_size = static_cast<int>(image_width * 0.06);

  ft2->loadFontData(predictor_result_.vis_font, 0);

  cv::Size text_size;
  int baseline = 0;

  do {
    text_size = ft2->getTextSize(text, font_size, -1, &baseline);
    if (text_size.width <= image_width)
      break;
    font_size--;
  } while (font_size > 0);

  return font_size;
}
#else
void TextRecResult::SaveToImg(const std::string &save_path) {
  INFOW(
      "OpenCV was not compiled with the freetype module (opencv_freetype), rec "
      "image will be not saved.");
}
#endif
void TextRecResult::Print() const {
  std::cout << "{\n  \"res\": {" << std::endl;

  std::cout << "    \"input_path\": {" << predictor_result_.input_path << " },"
            << std::endl;
  std::cout << "    \"rec_text\": {" << predictor_result_.rec_text << " }"
            << std::endl;
  std::cout << "    \"rec_score\": {" << predictor_result_.rec_score << " }"
            << std::endl;
  std::cout << "}" << std::endl;
}

void TextRecResult::SaveToJson(const std::string &save_path) const {
  nlohmann::ordered_json j;

  j["input_path"] = predictor_result_.input_path;
  j["page_index"] = nlohmann::json::value_t::null; //********

  j["rec_text"] = predictor_result_.rec_text;
  j["rec_score"] = predictor_result_.rec_score;

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
