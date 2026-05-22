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

#include "processors.h"

#include <codecvt>
#include <locale>
#include <numeric>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include "src/utils/utility.h"

absl::StatusOr<std::vector<cv::Mat>>
OCRResizeNormImg::Apply(std::vector<cv::Mat> &input, const void *param) const {
  std::vector<cv::Mat> output = {};
  output.reserve(input.size());
  if (input_shape_.empty()) {
    for (auto &image : input) {
      auto result = Resize(image);
      if (!result.ok()) {
        return result.status();
      }
      output.push_back(result.value());
    }
  } else {
    for (auto &image : input) {
      auto result = StaticResize(image);
      if (!result.ok()) {
        return result.status();
      }
      output.push_back(result.value());
    }
  }
  return output;
}

absl::StatusOr<cv::Mat> OCRResizeNormImg::Resize(cv::Mat &image) const {
  float rec_wh_ratio = (float)rec_image_shape_[2] / (float)rec_image_shape_[1];
  float image_wh_ratio = (float)image.size[1] / (float)image.size[0];
  float max_wh_ratio = std::max(rec_wh_ratio, image_wh_ratio);
  auto image_result = ResizeNormImg(image, max_wh_ratio);
  if (!image_result.ok()) {
    return image_result.status();
  }
  return image_result.value();
}

absl::StatusOr<cv::Mat> OCRResizeNormImg::StaticResize(cv::Mat &image) const {
  cv::Mat resize_image;
  int img_c = input_shape_[0];
  int img_h = input_shape_[1];
  int img_w = input_shape_[2];
  cv::resize(image, resize_image, cv::Size(img_w, img_h));
  resize_image.convertTo(resize_image, CV_32F);
  std::vector<cv::Mat> mat_split(resize_image.channels());
  cv::split(resize_image, mat_split);
  for (auto &item : mat_split) {
    item /= 255;
    item -= 0.5;
    item /= 0.5;
    item = item.reshape(1, 1);
  }
  cv::Mat resize_image_process;
  cv::hconcat(mat_split, resize_image_process);
  std::vector<int> resize_shape = {img_c, img_h, img_w};
  resize_image_process = resize_image_process.reshape(1, resize_shape);
  return resize_image_process;
}

absl::StatusOr<cv::Mat>
OCRResizeNormImg::ResizeNormImg(cv::Mat &image, float max_wh_ratio) const {
  assert(rec_image_shape_[0] == image.channels());
  int rec_c = rec_image_shape_[0];
  int rec_h = rec_image_shape_[1];
  int rec_w = rec_image_shape_[2];

  rec_w = rec_h * max_wh_ratio;
  cv::Mat resize_image;
  int resize_w = 0;
  if (rec_w > MAX_IMG_W) {
    rec_w = MAX_IMG_W;
    resize_w = MAX_IMG_W;
    cv::resize(image, resize_image, cv::Size(resize_w, rec_h));
  } else {
    float wh_ratio = (float)image.size[1] / (float)image.size[0];
    if (std::ceil(rec_h * wh_ratio) > rec_w) {
      resize_w = rec_w;
    } else {
      resize_w = std::ceil(rec_h * wh_ratio);
    }
    cv::resize(image, resize_image, cv::Size(resize_w, rec_h));
  }
  resize_image.convertTo(resize_image, CV_32F);
  std::vector<cv::Mat> mat_split(resize_image.channels());
  cv::split(resize_image, mat_split);
  for (auto &item : mat_split) {
    item /= 255;
    item -= 0.5;
    item /= 0.5;
    item = item.reshape(1, 1);
  }
  cv::Mat resize_image_process;
  cv::hconcat(mat_split, resize_image_process);
  std::vector<int> resize_shape = {rec_c, rec_h, resize_w};
  resize_image_process = resize_image_process.reshape(1, resize_shape);
  std::vector<int> image_shape = {rec_c, rec_h, rec_w};
  cv::Mat padding_im =
      cv::Mat::zeros(image_shape.size(), &image_shape[0], CV_32F);
  for (int c = 0; c < rec_c; ++c) {
    for (int row = 0; row < rec_h; ++row) {
      float *dst = padding_im.ptr<float>(c) + row * rec_w;
      float *src = resize_image_process.ptr<float>(c) + row * resize_w;
      std::copy(src, src + resize_w, dst);
    }
  }
  return padding_im;
}

CTCLabelDecode::CTCLabelDecode(const std::vector<std::string> &character_list,
                               bool use_space_char)
    : character_list_(character_list), use_space_char_(use_space_char) {
  if (character_list_.empty()) {
    const std::string normal = "0123456789abcdefghijklmnopqrstuvwxyz";
    for (const auto &item : normal) {
      character_list_.emplace_back(std::string(1, item));
    }
  }
  if (use_space_char) {
    character_list_.emplace_back(std::string(" "));
  }
  AddSpecialChar();
  dict_.reserve(character_list_.size());
  for (int i = 0; i < character_list_.size(); i++) {
    dict_[i] = character_list_[i];
  }
}

absl::StatusOr<std::vector<CTCLabelDecodeResult>>
CTCLabelDecode::Apply(const cv::Mat &preds, const bool return_word_box,
                      std::vector<float> wh_ratio_list,
                      float max_wh_ratio) const {
  auto preds_batch = Utility::SplitBatch(preds);
  std::vector<CTCLabelDecodeResult> ctc_result = {};
  ctc_result.reserve(preds_batch.value().size());
  if (!preds_batch.ok()) {
    return preds_batch.status();
  }
  for (const auto &pred : preds_batch.value()) {
    auto result = Process(pred, return_word_box);
    if (!result.ok()) {
      return result.status();
    }
    ctc_result.push_back(result.value());
  }
  if (return_word_box) {
    for (int i = 0; i < ctc_result.size(); i++) {
      float wh_ratio = wh_ratio_list[i];
      ctc_result[i].sentence_len =
          ctc_result[i].sentence_len * (wh_ratio / max_wh_ratio);
    }
  }
  return ctc_result;
}

absl::StatusOr<CTCLabelDecodeResult>
CTCLabelDecode::Process(const cv::Mat &pred_data, bool return_word_box) const {
  std::vector<int> shape_squeeze = {};
  for (int i = 1; i < pred_data.dims; i++) {
    shape_squeeze.push_back(pred_data.size[i]);
  }
  cv::Mat pred_data_process;
  pred_data_process = pred_data.reshape(1, shape_squeeze);

  int seq_len = pred_data_process.size[0];
  int num_classes = pred_data_process.size[1];
  std::list<int> text_index;
  std::list<float> text_prob;
  for (int t = 0; t < seq_len; ++t) {
    const float *row_ptr = pred_data_process.ptr<float>(t);
    float max_val = row_ptr[0];
    int max_idx = 0;
    for (int c = 1; c < num_classes; ++c) {
      if (row_ptr[c] > max_val) {
        max_val = row_ptr[c];
        max_idx = c;
      }
    }
    text_index.push_back(max_idx);
    text_prob.push_back(max_val);
  }
  auto decode_result = Decode(text_index, text_prob, true, return_word_box);
  if (!decode_result.ok()) {
    return decode_result.status();
  }
  return decode_result.value();
}

absl::StatusOr<CTCLabelDecodeResult>
CTCLabelDecode::Decode(std::list<int> &text_index, std::list<float> &text_prob,
                       bool is_remove_duplicate, bool return_word_box) const {
  std::vector<bool> selection(text_index.size(), true);
  if (is_remove_duplicate && text_index.size() > 1) {
    auto prev = text_index.begin();
    auto curr = std::next(prev);
    size_t idx = 1;
    for (; curr != text_index.end(); ++curr, ++prev, ++idx) {
      if (*curr == *prev)
        selection[idx] = false;
    }
  }
  for (const auto &ignore_item : IGNORE_TOKEN) {
    size_t idx = 0;
    for (auto item_list = text_index.begin(); item_list != text_index.end();
         ++item_list, idx++) {
      if (*item_list == ignore_item) {
        selection[idx] = false;
      }
    }
  }
  auto sel_it = selection.begin();
  for (auto it = text_index.begin(); it != text_index.end();) {
    if (!(*sel_it)) {
      it = text_index.erase(it);
    } else {
      ++it;
    }
    ++sel_it;
  }
  auto sel_it_prob = selection.begin();
  for (auto it = text_prob.begin(); it != text_prob.end();) {
    if (!(*sel_it_prob)) {
      it = text_prob.erase(it);
    } else {
      ++it;
    }
    ++sel_it_prob;
  }
  std::vector<std::string> char_list = {};
  for (auto list_index = text_index.begin(); list_index != text_index.end();
       ++list_index) {
    if ((*list_index) < character_list_.size()) {
      char_list.push_back(character_list_[*list_index]);
    } else {
      char_list.push_back(" ");
    }
  }
  std::list<float> conf_list = {};
  if (!text_prob.empty()) {
    conf_list = text_prob;
  } else {
    conf_list = std::list<float>(selection.size(), 0);
  }
  std::string text;
  for (const auto &item_char : char_list) {
    text += item_char;
  }
  float sum = std::accumulate(conf_list.begin(), conf_list.end(), 0.0f);
  float mean = sum / conf_list.size();
  CTCLabelDecodeResult result;
  if (return_word_box) {
    auto word_info_tuple = GetWordInfo(text, selection);
    result.sentence_len = selection.size();
    result.word_list = std::get<0>(word_info_tuple);
    result.word_col_list = std::get<1>(word_info_tuple);
    result.state_list = std::get<2>(word_info_tuple);
  }
  result.sentence = std::pair<std::string, float>(text, mean);

  return result;
}

void CTCLabelDecode::AddSpecialChar() {
  character_list_.insert(character_list_.begin(), "blank");
}

std::tuple<std::vector<std::wstring>, std::vector<std::vector<int>>,
           std::vector<std::string>>
CTCLabelDecode::GetWordInfo(const std::string &text_origin,
                            const std::vector<bool> &selection) const {

  std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
  std::wstring text = converter.from_bytes(text_origin);

  std::string state = "";
  std::wstring word_content = L"";
  std::vector<int> word_col_content;

  std::vector<std::wstring> word_list = {};
  std::vector<std::vector<int>> word_col_list = {};
  std::vector<std::string> state_list = {};

  std::vector<int> valid_col;
  for (int i = 0; i < selection.size(); ++i) {
    if (selection[i]) {
      valid_col.push_back(i);
    }
  }

  std::wregex en_num_pattern(L"[a-zA-Z0-9]");
  std::wregex num_pattern(L"[0-9]");

  for (int c_i = 0; c_i < text.length(); ++c_i) {
    wchar_t ch = text[c_i];
    std::string c_state;

    if (ch >= L'\u4e00' && ch <= L'\u9fff') {
      c_state = "cn";
    } else if (std::regex_search(std::wstring(1, ch), en_num_pattern)) { // [5]
      c_state = "en&num";
    } else {
      c_state = "symbol";
    }

    if (ch == L'.' && state == "en&num" && c_i + 1 < text.length()) {
      if (std::regex_search(std::wstring(1, text[c_i + 1]), num_pattern)) {
        c_state = "en&num";
      }
    }

    if (ch == L'-' && state == "en&num") {
      c_state = "en&num";
    }

    if (state.empty()) {
      state = c_state;
    }

    if (state != c_state && !word_content.empty()) {
      if (!word_content.empty()) {
        word_list.push_back(word_content);
        word_col_list.push_back(word_col_content);
        state_list.push_back(state);
        word_content.clear();
        word_col_content.clear();
      }
      state = c_state;
    }

    word_content += ch;
    if (c_i < valid_col.size()) {
      word_col_content.push_back(valid_col[c_i]);
    }
  }

  if (!word_content.empty()) {
    word_list.push_back(word_content);
    word_col_list.push_back(word_col_content);
    state_list.push_back(state);
  }
  auto result = std::make_tuple(word_list, word_col_list, state_list);
  return result;
}
