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
#include <codecvt>
#include <fstream>
#include <locale>
#include <random>
#include <string>

#include "src/utils/utility.h"
#include "third_party/nlohmann/json.hpp"

using json = nlohmann::json;

void OCRResult::SaveToImg(const std::string &save_path) {
  cv::Mat image = pipeline_result_.doc_preprocessor_res.output_image;
  auto texts = pipeline_result_.rec_texts;
  std::vector<std::vector<cv::Point>> boxes;
  std::vector<std::vector<cv::Point2f>> boxes_float =
      pipeline_result_.rec_polys;
  for (const auto &floatPolygon : pipeline_result_.rec_polys) {
    std::vector<cv::Point> intPolygon;
    for (const auto &point : floatPolygon) {
      intPolygon.push_back(cv::Point(cvRound(point.x), cvRound(point.y)));
    }
    boxes.push_back(intPolygon);
  }

  if (image.empty()) {
    INFOE("Input image is empty.");
    exit(-1);
  }

  int h = image.rows;
  int w = image.cols;

  cv::Mat img_left = image.clone();

  cv::Mat img_right(h, w, CV_8UC3, cv::Scalar(255, 255, 255));

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, 255);

  for (size_t i = 0; i < boxes.size(); ++i) {
    auto &box = boxes[i];
    const auto &box_float = boxes_float[i];
    const auto &text = texts[i];

    cv::Scalar color(dis(gen), dis(gen), dis(gen));

    if (box.size() > 4) {
      const std::vector<std::vector<cv::Point>> polygons{box};
      cv::fillPoly(img_left, polygons, color);
      cv::polylines(img_left, polygons, true, color, 8);
      box = GetMinareaRect(box);
      std::vector<int> ys;
      for (const auto &pt : box)
        ys.push_back(pt.y);
      int min_y = *std::min_element(ys.begin(), ys.end());
      int max_y = *std::max_element(ys.begin(), ys.end());
      int height = static_cast<int>(0.5 * (max_y - min_y));
      double mean_y = std::accumulate(ys.begin(), ys.end(), 0.0) / ys.size();
      if (box.size() >= 4) {
        box[0].y = static_cast<int>(mean_y);
        box[1].y = static_cast<int>(mean_y);
        box[2].y = static_cast<int>(mean_y + std::min(20, height));
        box[3].y = static_cast<int>(mean_y + std::min(20, height));
      }
    } else {
      cv::fillPoly(img_left, std::vector<std::vector<cv::Point>>{box}, color);
    }
#ifdef USE_FREETYPE
    cv::Mat img_right_text = DrawBoxTextFine(cv::Size(w, h), box_float, text,
                                             pipeline_result_.vis_fonts);
    cv::polylines(img_right_text, box, true, color, 1);
    cv::bitwise_and(img_right, img_right_text, img_right);
#endif
  }
  cv::Mat blended;
  cv::addWeighted(image, 0.5, img_left, 0.5, 0, blended);
#ifdef USE_FREETYPE
  cv::Mat ocr_res_image(h, w * 2, CV_8UC3, cv::Scalar(255, 255, 255));
  blended.copyTo(ocr_res_image(cv::Rect(0, 0, w, h)));
  img_right.copyTo(ocr_res_image(cv::Rect(w, 0, w, h)));
#else
  cv::Mat ocr_res_image = blended;
#endif

  auto model_settings = pipeline_result_.model_settings;
  std::unordered_map<std::string, cv::Mat> res_img_dict;
  res_img_dict["ocr_res_img"] = ocr_res_image;

  auto ocr_path = Utility::SmartCreateDirectoryForImage(
      save_path, pipeline_result_.input_path, "_ocr_res_img");
  if (!ocr_path.ok()) {
    INFOE(ocr_path.status().ToString().c_str());
    exit(-1);
  }
  auto doc_pre_path = Utility::SmartCreateDirectoryForImage(
      save_path, pipeline_result_.input_path, "_doc_preprocessor_res");
  if (!doc_pre_path.ok()) {
    INFOE(doc_pre_path.status().ToString().c_str());
    exit(-1);
  }
  cv::imwrite(ocr_path.value(), ocr_res_image);
  if (model_settings["use_doc_preprocessor"]) {
    int h1 = pipeline_result_.doc_preprocessor_res.input_image.size[0];
    int w1 = pipeline_result_.doc_preprocessor_res.input_image.size[1];
    int h2 = pipeline_result_.doc_preprocessor_res.rotate_image.size[0];
    int w2 = pipeline_result_.doc_preprocessor_res.rotate_image.size[1];
    int h3 = pipeline_result_.doc_preprocessor_res.output_image.size[0];
    int w3 = pipeline_result_.doc_preprocessor_res.output_image.size[1];
    int h_all = std::max(h1, std::max(h2, h3));
    int total_w = w1 + w2 + w3;

    cv::Mat doc_pre_res_image(h_all, total_w, CV_8UC3,
                              cv::Scalar(255, 255, 255));

    pipeline_result_.doc_preprocessor_res.input_image.copyTo(
        doc_pre_res_image(cv::Rect(0, 0, w1, h1)));
    pipeline_result_.doc_preprocessor_res.rotate_image.copyTo(
        doc_pre_res_image(cv::Rect(w1, 0, w2, h2)));
    pipeline_result_.doc_preprocessor_res.output_image.copyTo(
        doc_pre_res_image(cv::Rect(w1 + w2, 0, w3, h3)));
    cv::imwrite(doc_pre_path.value(), doc_pre_res_image);
    res_img_dict["doc_preprocessor_res"] = doc_pre_res_image;
  }
}

#ifdef USE_FREETYPE
cv::Mat OCRResult::DrawBoxTextFine(const cv::Size &img_size,
                                   const std::vector<cv::Point2f> &box,
                                   const std::string &txt,
                                   const std::string &vis_font) {
  int box_height = cv::norm(box[0] - box[3]);
  int box_width = cv::norm(box[0] - box[1]);
  auto ft2 = cv::freetype::createFreeType2();
  ft2->loadFontData(vis_font, 0);

  bool vertical_mode = box_height > 2 * box_width && box_height > 30;
  int n = std::max(int(txt.size()), 1);

  int font_height = 10;
  if (!txt.empty()) {
    if (vertical_mode) {
      font_height = CreateFontVertical(ft2, txt, box_height, box_width);
    } else {
      font_height = CreateFont(ft2, txt, box_height, box_width);
    }
  }
  cv::Mat img_text(box_height, box_width, CV_8UC3, cv::Scalar(255, 255, 255));
  int x = 0, y = 0;

  if (!txt.empty()) {
    if (vertical_mode) {
      DrawVerticalText(ft2, img_text, txt, x, y, font_height,
                       cv::Scalar(0, 0, 0));
    } else {
      int baseline = 0;
      cv::Size textsize = ft2->getTextSize(txt, font_height, -1, &baseline);
      x = (box_width - textsize.width) / 2;
      y = (box_height + textsize.height) / 2 - baseline;
      ft2->putText(img_text, txt, cv::Point(x, y), font_height,
                   cv::Scalar(0, 0, 0), -1, cv::LINE_AA, true);
    }
  }
  std::vector<cv::Point2f> src_pts = {{0, 0},
                                      {float(box_width), 0},
                                      {float(box_width), float(box_height)},
                                      {0, float(box_height)}};
  cv::Mat M = cv::getPerspectiveTransform(src_pts, box);

  cv::Mat dst(img_size, CV_8UC3, cv::Scalar(255, 255, 255));
  cv::warpPerspective(img_text, dst, M, img_size, cv::INTER_NEAREST,
                      cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));
  return dst;
}
cv::Size OCRResult::getActualCharSize(cv::Ptr<cv::freetype::FreeType2> &ft2,
                                      const std::string &utf8_char,
                                      int font_height) {
  cv::Mat temp = cv::Mat::zeros(300, 300, CV_8UC1);

  cv::Point pos(100, 150);

  ft2->putText(temp, utf8_char, pos, font_height, cv::Scalar(255), -1,
               cv::LINE_AA, false);

  std::vector<cv::Point> nonZeroPoints;
  cv::findNonZero(temp, nonZeroPoints);

  if (nonZeroPoints.empty()) {
    return cv::Size(0, 0);
  }

  cv::Rect boundingRect = cv::boundingRect(nonZeroPoints);
  return cv::Size(boundingRect.width, boundingRect.height);
}
void OCRResult::DrawVerticalText(cv::Ptr<cv::freetype::FreeType2> &ft2,
                                 cv::Mat &img, const std::string &text, int x,
                                 int y, int font_height, cv::Scalar color,
                                 float line_spacing) {
  std::wstring wtext =
      std::wstring_convert<std::codecvt_utf8<wchar_t>>().from_bytes(text);
  for (size_t i = 0; i < wtext.size(); ++i) {
    std::wstring single_char(1, wtext[i]);
    std::string utf8_char =
        std::wstring_convert<std::codecvt_utf8<wchar_t>>().to_bytes(
            single_char);
    ft2->putText(img, utf8_char, cv::Point(x, y), font_height, color, -1,
                 cv::LINE_AA, true);
    int baseline = 0;
    cv::Size size = ft2->getTextSize(utf8_char, font_height, -1, &baseline);
    size.height += baseline;
    y += size.height * 1.1 + line_spacing;
  }
}
int OCRResult::CreateFont(cv::Ptr<cv::freetype::FreeType2> &ft2,
                          const std::string &text, int region_height,
                          int region_width) {
  int font_height = std::max(int(region_height * 0.8), 10);
  int baseline = 0;
  cv::Size text_size = ft2->getTextSize(text, font_height, -1, &baseline);
  if (text_size.width > region_width) {
    font_height =
        static_cast<int>(font_height * region_width / text_size.width);
    text_size = ft2->getTextSize(text, font_height, -1, &baseline);
  }
  return font_height;
}
int OCRResult::CreateFontVertical(cv::Ptr<cv::freetype::FreeType2> &ft2,
                                  const std::string &text, int region_height,
                                  int region_width, float scale) {
  std::wstring_convert<std::codecvt_utf8<wchar_t>> conv;
  std::wstring wtext = conv.from_bytes(text);
  int n = static_cast<int>(wtext.length());
  int baseFontSize = static_cast<int>(region_height / n * 0.8 * scale);
  baseFontSize = std::max(baseFontSize, 10);

  int maxCharWidth = 0;
  for (size_t i = 0; i < wtext.length(); ++i) {
    std::wstring singleChar(1, wtext[i]);
    std::string utf8Char =
        std::wstring_convert<std::codecvt_utf8<wchar_t>>().to_bytes(singleChar);
    cv::Size textSize = getActualCharSize(ft2, utf8Char, baseFontSize);
    maxCharWidth = std::max(maxCharWidth, textSize.width);
  }

  int finalFontSize = baseFontSize;
  if (maxCharWidth > region_width) {
    finalFontSize =
        static_cast<int>(baseFontSize * region_width / maxCharWidth);
    finalFontSize = std::max(finalFontSize, 10);
  }

  return finalFontSize;
}
#endif

std::vector<cv::Point>
OCRResult::GetMinareaRect(const std::vector<cv::Point> &points) {
  cv::RotatedRect bounding_box = cv::minAreaRect(points);

  cv::Point2f boxPts[4];
  bounding_box.points(boxPts);
  std::vector<cv::Point2f> ptsVec(boxPts, boxPts + 4);

  std::sort(
      ptsVec.begin(), ptsVec.end(),
      [](const cv::Point2f &a, const cv::Point2f &b) { return a.x < b.x; });
  int index_a, index_b, index_c, index_d;
  if (ptsVec[1].y > ptsVec[0].y) {
    index_a = 0;
    index_d = 1;
  } else {
    index_a = 1;
    index_d = 0;
  }
  if (ptsVec[3].y > ptsVec[2].y) {
    index_b = 2;
    index_c = 3;
  } else {
    index_b = 3;
    index_c = 2;
  }

  std::vector<cv::Point> box = {ptsVec[index_a], ptsVec[index_b],
                                ptsVec[index_c], ptsVec[index_d]};

  for (auto &pt : box) {
    pt.x = static_cast<int>(std::round(pt.x));
    pt.y = static_cast<int>(std::round(pt.y));
  }

  return box;
}

void OCRResult::SaveToJson(const std::string &save_path) const {
  nlohmann::ordered_json j;
  j["input_path"] = pipeline_result_.input_path;

  j["page_index"] = nullptr;

  j["model_settings"] = pipeline_result_.model_settings;

  auto it = pipeline_result_.model_settings.find("use_doc_preprocessor");
  if (it != pipeline_result_.model_settings.end() && it->second) {
    nlohmann::ordered_json j_doc_pre;
    j_doc_pre["model_settings"] =
        pipeline_result_.doc_preprocessor_res.model_settings;
    j_doc_pre["angle"] = pipeline_result_.doc_preprocessor_res.angle;
    j["doc_preprocessor_res"] = j_doc_pre;
  }
  json polys_json = json::array();
  for (const auto &polygon : pipeline_result_.dt_polys) {
    json poly_json = json::array();
    for (const auto &point : polygon) {
      poly_json.push_back(
          {static_cast<int>(point.x), static_cast<int>(point.y)});
    }
    polys_json.push_back(poly_json);
  }
  j["dt_polys"] = polys_json;
  nlohmann::ordered_json j_text_det_params;
  j_text_det_params["limit_side_len"] =
      pipeline_result_.text_det_params.text_det_limit_side_len;
  j_text_det_params["limit_type"] =
      pipeline_result_.text_det_params.text_det_limit_type;
  j_text_det_params["thresh"] =
      pipeline_result_.text_det_params.text_det_thresh;
  j_text_det_params["max_side_limit"] =
      pipeline_result_.text_det_params.text_det_max_side_limit;
  j_text_det_params["box_thresh"] =
      pipeline_result_.text_det_params.text_det_box_thresh;
  j_text_det_params["unclip_ratio"] =
      pipeline_result_.text_det_params.text_det_unclip_ratio;
  j["text_det_params"] = j_text_det_params;
  j["text_type"] = pipeline_result_.text_type;

  if (!pipeline_result_.textline_orientation_angles.empty()) {
    j["textline_orientation_angles"] =
        pipeline_result_.textline_orientation_angles;
  }
  j["text_rec_score_thresh"] = pipeline_result_.text_rec_score_thresh;
  j["rec_texts"] = pipeline_result_.rec_texts;
  j["rec_scores"] = pipeline_result_.rec_scores;
  json rec_polys_json = json::array();
  for (const auto &polygon : pipeline_result_.rec_polys) {
    json poly_json = json::array();
    for (const auto &point : polygon) {
      poly_json.push_back(
          {static_cast<int>(point.x), static_cast<int>(point.y)});
    }
    rec_polys_json.push_back(poly_json);
  }
  j["rec_polys"] = rec_polys_json;

  std::vector<std::array<int, 4>> int_vec;
  int_vec.reserve(pipeline_result_.rec_boxes.size());

  std::transform(pipeline_result_.rec_boxes.begin(),
                 pipeline_result_.rec_boxes.end(), std::back_inserter(int_vec),
                 [](const std::array<float, 4> &arr) {
                   std::array<int, 4> res;
                   for (size_t i = 0; i < 4; ++i) {
                     res[i] = static_cast<int>(arr[i]);
                   }
                   return res;
                 });
  j["rec_boxes"] = int_vec;

  absl::StatusOr<std::string> full_path;
  if (pipeline_result_.input_path.empty()) {
    INFOW("Input path is empty, will use output_res.json instead!");
    full_path = Utility::SmartCreateDirectoryForJson(save_path, "output");
  } else {
    full_path = Utility::SmartCreateDirectoryForJson(
        save_path, pipeline_result_.input_path);
  }
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

void PrintDocPreprocessorPipelineResult(
    const DocPreprocessorPipelineResult &doc) {
  std::cout << "{\n";
  std::cout << "    \"model_settings\": {";
  bool first = true;
  for (const auto &kv : doc.model_settings) {
    if (!first)
      std::cout << ", ";
    std::cout << "\"" << kv.first << "\": " << (kv.second ? "true" : "false");
    first = false;
  }
  std::cout << "},\n";
  std::cout << "    \"angle\": " << doc.angle << "\n";
  std::cout << "  }";
}

void PrintPolys(const std::vector<std::vector<cv::Point2f>> &polys) {
  std::cout << "[";
  for (size_t i = 0; i < polys.size(); ++i) {
    if (i != 0)
      std::cout << ",\n ";
    std::cout << "[";
    for (size_t j = 0; j < polys[i].size(); ++j) {
      if (j != 0)
        std::cout << ", ";
      std::cout << "[" << polys[i][j].x << ", " << polys[i][j].y << "]";
    }
    std::cout << "]";
  }
  std::cout << "]";
}

void PrintModelSettings(const std::unordered_map<std::string, bool> &ms) {
  std::cout << "{";
  bool first = true;
  for (const auto &kv : ms) {
    if (!first)
      std::cout << ", ";
    std::cout << "\"" << kv.first << "\": " << (kv.second ? "true" : "false");
    first = false;
  }
  std::cout << "}";
}

void PrintArray(const std::vector<float> &arr) {
  std::cout << "[";
  for (size_t i = 0; i < arr.size(); ++i) {
    if (i != 0)
      std::cout << ", ";
    std::cout << arr[i];
  }
  std::cout << "]";
}

void PrintStringArray(const std::vector<std::string> &arr) {
  std::cout << "[";
  for (size_t i = 0; i < arr.size(); ++i) {
    if (i != 0)
      std::cout << ", ";
    std::cout << "\"" << arr[i] << "\"";
  }
  std::cout << "]";
}

void PrintIntArray(const std::vector<int> &arr) {
  std::cout << "[";
  for (size_t i = 0; i < arr.size(); ++i) {
    if (i != 0)
      std::cout << ", ";
    std::cout << arr[i];
  }
  std::cout << "]";
}

void PrintRecBoxes(const std::vector<std::array<float, 4>> &arr) {
  std::cout << "[";
  for (size_t i = 0; i < arr.size(); ++i) {
    if (i != 0)
      std::cout << ", ";
    std::cout << "[" << arr[i][0] << ", " << arr[i][1] << ", " << arr[i][2]
              << ", " << arr[i][3] << "]";
  }
  std::cout << "],";
}

void PrintTextDetParams(const TextDetParams &p) {
  std::cout << "{";
  std::cout << "\"limit_side_len\": " << p.text_det_limit_side_len << ", ";
  std::cout << "\"limit_type\": \"" << p.text_det_limit_type << "\", ";
  std::cout << "\"thresh\": " << p.text_det_thresh << ", ";
  std::cout << "\"max_side_limit\": " << p.text_det_max_side_limit << ", ";
  std::cout << "\"box_thresh\": " << p.text_det_box_thresh << ", ";
  std::cout << "\"unclip_ratio\": " << p.text_det_unclip_ratio;
  std::cout << "}";
}

void OCRResult::Print() const {
  std::cout << "{\n";
  std::cout << "  \"input_path\": \"" << pipeline_result_.input_path << "\",\n";
  if (pipeline_result_.model_settings.at("use_doc_preprocessor")) {
    std::cout << "  \"doc_preprocessor_res\": ";
    PrintDocPreprocessorPipelineResult(pipeline_result_.doc_preprocessor_res);
    std::cout << ",\n";
  }
  std::cout << "  \"dt_polys\": ";
  PrintPolys(pipeline_result_.dt_polys);
  std::cout << ",\n";
  std::cout << "  \"model_settings\": ";
  PrintModelSettings(pipeline_result_.model_settings);
  std::cout << ",\n";
  std::cout << "  \"text_det_params\": ";
  PrintTextDetParams(pipeline_result_.text_det_params);
  std::cout << ",\n";
  std::cout << "  \"text_type\": \"" << pipeline_result_.text_type << "\",\n";
  std::cout << "  \"text_rec_score_thresh\": "
            << pipeline_result_.text_rec_score_thresh << ",\n";
  std::cout << "  \"rec_texts\": ";
  PrintStringArray(pipeline_result_.rec_texts);
  std::cout << ",\n";
  std::cout << "  \"rec_scores\": ";
  PrintArray(pipeline_result_.rec_scores);
  std::cout << ",\n";
  std::cout << "  \"textline_orientation_angles\": ";
  PrintIntArray(pipeline_result_.textline_orientation_angles);
  std::cout << ",\n";
  std::cout << "  \"rec_polys\": ";
  PrintPolys(pipeline_result_.rec_polys);
  std::cout << ",\n";
  std::cout << "  \"rec_boxes\": ";
  PrintRecBoxes(pipeline_result_.rec_boxes);
  std::cout << "\n}" << std::endl;
}
