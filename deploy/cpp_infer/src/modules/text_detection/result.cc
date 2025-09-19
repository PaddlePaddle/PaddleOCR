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
void TextDetResult::SaveToImg(const std::string &save_path) {
  cv::Mat img = predictor_result_.input_image.clone();

  const auto &dt_polys = predictor_result_.dt_polys;
  for (const auto &poly : dt_polys) {
    std::vector<cv::Point> pts;
    for (const auto &pt : poly) {
      pts.emplace_back(cv::Point(cvRound(pt.x), cvRound(pt.y)));
    }

    const cv::Point *pts_ptr = pts.data();
    int npts = pts.size();
    cv::polylines(img, &pts_ptr, &npts, 1, true, cv::Scalar(0, 0, 255), 2);
  }

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
    INFOE("Failed to write the image : %s", full_path.value().c_str());
    exit(-1);
  }
}

void TextDetResult::Print() const {
  std::cout << "{\n  \"res\": {" << std::endl;

  std::cout << "    \"input_path\": {" << predictor_result_.input_path
            << "    }," << std::endl;

  std::cout << "    \"dt_polys\": [" << std::endl;
  for (const auto &polygon : predictor_result_.dt_polys) {
    std::cout << "        [";
    for (size_t i = 0; i < polygon.size(); ++i) {
      std::cout << "[" << static_cast<int>(polygon[i].x) << ", "
                << static_cast<int>(polygon[i].y) << "]";
      if (i < polygon.size() - 1)
        std::cout << ", ";
    }
    std::cout << "]," << std::endl;
  }

  std::cout << "      ]}," << std::endl;

  std::cout << "    \"dt_scores\": [" << std::endl;
  for (auto it = predictor_result_.dt_scores.begin();
       it != predictor_result_.dt_scores.end(); ++it) {
    std::cout << *it;
    if (it < predictor_result_.dt_scores.end() - 1)
      std::cout << ", ";
  }
  std::cout << "]}" << std::endl;

  std::cout << "    ]" << std::endl;

  std::cout << "  }\n}" << std::endl;
}

void TextDetResult::SaveToJson(const std::string &save_path) const {
  nlohmann::ordered_json j;

  j["input_path"] = predictor_result_.input_path;

  j["page_index"] = nullptr; //********
  json polys_json = json::array();
  for (const auto &polygon : predictor_result_.dt_polys) {
    json poly_json = json::array();
    for (const auto &point : polygon) {
      poly_json.push_back(
          {static_cast<int>(point.x), static_cast<int>(point.y)});
    }
    polys_json.push_back(poly_json);
  }
  j["dt_polys"] = polys_json;
  j["dt_score"] = predictor_result_.dt_scores;

  absl::StatusOr<std::string> full_path;

  full_path = Utility::SmartCreateDirectoryForJson(
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
