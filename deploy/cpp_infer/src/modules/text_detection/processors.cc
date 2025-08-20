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

#include <stdexcept>

#include "src/utils/utility.h"

DetResizeForTest::DetResizeForTest(const DetResizeForTestParam &params) {
  if (params.input_shape.has_value()) {
    input_shape_ = params.input_shape.value();
    resize_type_ = 3;
  } else if (params.image_shape.has_value()) {
    image_shape_ = params.image_shape.value();
    resize_type_ = 1;
    if (params.keep_ratio.has_value()) {
      keep_ratio_ = params.keep_ratio.value();
    }
  } else if (params.limit_side_len.has_value()) {
    limit_side_len_ = params.limit_side_len.value();
    limit_type_ = params.limit_type.value_or("min");
  } else if (params.resize_long.has_value()) {
    resize_type_ = 2;
    resize_long_ = params.resize_long.value_or(960);
  } else {
    limit_side_len_ = 736;
    limit_type_ = "min";
  }
  if (params.max_side_limit.has_value()) {
    max_side_limit_ = params.max_side_limit.value();
  }
}

absl::StatusOr<std::vector<cv::Mat>>
DetResizeForTest::Apply(std::vector<cv::Mat> &input,
                        const void *param_ptr) const {
  if (input.empty()) {
    return absl::InvalidArgumentError("Input image vector is empty.");
  }
  std::vector<cv::Mat> results;
  if (param_ptr != nullptr) {
    const DetResizeForTestParam *param =
        static_cast<const DetResizeForTestParam *>(param_ptr);
    for (const auto &img : input) {
      auto res = Resize(
          img,
          param->limit_side_len.has_value() ? param->limit_side_len.value()
                                            : limit_side_len_,
          param->limit_type.has_value() ? param->limit_type.value()
                                        : limit_type_,
          param->max_side_limit.has_value() ? param->max_side_limit.value()
                                            : max_side_limit_);
      if (!res.ok())
        return res.status();
      results.push_back(res.value());
    }
  } else {
    for (const auto &img : input) {
      auto res = Resize(img, limit_side_len_, limit_type_, max_side_limit_);
      if (!res.ok())
        return res.status();
      results.push_back(res.value());
    }
  }
  return results;
}

absl::StatusOr<cv::Mat> DetResizeForTest::Resize(const cv::Mat &img,
                                                 int limit_side_len,
                                                 const std::string &limit_type,
                                                 int max_side_limit) const {
  int src_h = img.rows;
  int src_w = img.cols;
  if (src_h + src_w < 64) {
    cv::Mat padded = ImagePadding(img);
    src_h = padded.rows;
    src_w = padded.cols;
    return Resize(padded, limit_side_len, limit_type, max_side_limit);
  }

  switch (resize_type_) {
  case 0:
    return ResizeImageType0(img, limit_side_len, limit_type, max_side_limit);
  case 1:
    return ResizeImageType1(img);
  case 2:
    return ResizeImageType2(img);
  case 3:
    return ResizeImageType3(img);
  default:
    return absl::InvalidArgumentError("Unknown resize_type: " +
                                      std::to_string(resize_type_));
  }
}

cv::Mat DetResizeForTest::ImagePadding(const cv::Mat &img, int value) const {
  int h = img.rows, w = img.cols, c = img.channels();
  int pad_h = std::max(32, h);
  int pad_w = std::max(32, w);
  cv::Mat im_pad = cv::Mat::zeros(pad_h, pad_w, img.type());
  im_pad.setTo(cv::Scalar::all(value));
  img.copyTo(im_pad(cv::Rect(0, 0, w, h)));
  return im_pad;
}

absl::StatusOr<cv::Mat>
DetResizeForTest::ResizeImageType0(const cv::Mat &img, int limit_side_len,
                                   const std::string &limit_type,
                                   int max_side_limit) const {
  int h = img.rows, w = img.cols;
  float ratio = 1.f;
  if (limit_type == "max") {
    if (std::max(h, w) > limit_side_len)
      ratio = float(limit_side_len) / std::max(h, w);
  } else if (limit_type == "min") {
    if (std::min(h, w) < limit_side_len)
      ratio = float(limit_side_len) / std::min(h, w);
  } else if (limit_type == "resize_long") {
    ratio = float(limit_side_len) / std::max(h, w);
  } else {
    return absl::InvalidArgumentError("Not supported limit_type: " +
                                      limit_type);
  }
  int resize_h = int(h * ratio);
  int resize_w = int(w * ratio);

  if (std::max(resize_h, resize_w) > max_side_limit) {
    ratio = float(max_side_limit) / std::max(resize_h, resize_w);
    resize_h = int(resize_h * ratio);
    resize_w = int(resize_w * ratio);
  }
  resize_h = std::max(int(std::round(resize_h / 32.0) * 32), 32);
  resize_w = std::max(int(std::round(resize_w / 32.0) * 32), 32);

  if (resize_h == h && resize_w == w)
    return img;
  if (resize_h <= 0 || resize_w <= 0)
    return absl::InvalidArgumentError("resize_w/h <= 0");
  cv::Mat resized;
  cv::resize(img, resized, cv::Size(resize_w, resize_h));
  return resized;
}

absl::StatusOr<cv::Mat>
DetResizeForTest::ResizeImageType1(const cv::Mat &img) const {
  int resize_h = image_shape_[0];
  int resize_w = image_shape_[1];
  int ori_h = img.rows, ori_w = img.cols;
  if (keep_ratio_) {
    resize_w = int(ori_w * (float(resize_h) / ori_h));
    int N = int(std::ceil(resize_w / 32.0));
    resize_w = N * 32;
  }
  if (resize_h == ori_h && resize_w == ori_w)
    return img;
  cv::Mat resized;
  cv::resize(img, resized, cv::Size(resize_w, resize_h));
  return resized;
}

absl::StatusOr<cv::Mat>
DetResizeForTest::ResizeImageType2(const cv::Mat &img) const {
  int h = img.rows, w = img.cols;
  int resize_h = h, resize_w = w;
  float ratio;
  if (resize_h > resize_w)
    ratio = float(resize_long_) / resize_h;
  else
    ratio = float(resize_long_) / resize_w;

  resize_h = int(resize_h * ratio);
  resize_w = int(resize_w * ratio);

  int max_stride = 128;
  resize_h = ((resize_h + max_stride - 1) / max_stride) * max_stride;
  resize_w = ((resize_w + max_stride - 1) / max_stride) * max_stride;

  if (resize_h == h && resize_w == w)
    return img;
  cv::Mat resized;
  cv::resize(img, resized, cv::Size(resize_w, resize_h));
  return resized;
}

absl::StatusOr<cv::Mat>
DetResizeForTest::ResizeImageType3(const cv::Mat &img) const {
  if (input_shape_.size() != INPUTSHAPE)
    return absl::InvalidArgumentError("input_shape not set for type " +
                                      std::to_string(INPUTSHAPE));
  int resize_h = input_shape_[1];
  int resize_w = input_shape_[2];
  int ori_h = img.rows, ori_w = img.cols;
  if (resize_h == ori_h && resize_w == ori_w)
    return img;
  cv::Mat resized;
  cv::resize(img, resized, cv::Size(resize_w, resize_h));
  return resized;
}

DBPostProcess::DBPostProcess(const DBPostProcessParams &params)
    : thresh_(params.thresh.value_or(0.3)),
      box_thresh_(params.box_thresh.value_or(0.7)),
      unclip_ratio_(params.unclip_ratio.value_or(2.0)),
      max_candidates_(params.max_candidates), min_size_(3),
      use_dilation_(params.use_dilation), score_mode_(params.score_mode),
      box_type_(params.box_type) {
  assert(score_mode_ == "slow" || score_mode_ == "fast");
  assert(box_type_ == "quad" || box_type_ == "poly");
}

absl::StatusOr<
    std::pair<std::vector<std::vector<cv::Point2f>>, std::vector<float>>>
DBPostProcess::operator()(const cv::Mat &preds,
                          const std::vector<int> &img_shapes,
                          absl::optional<float> thresh,
                          absl::optional<float> box_thresh,
                          absl::optional<float> unclip_ratio) {
  std::vector<std::vector<cv::Point2f>> all_boxes;
  std::vector<float> all_scores;
  auto preds_batch = Utility::SplitBatch(preds);
  if (!preds_batch.ok()) {
    return preds_batch.status();
  }
  for (const auto &preds_data : *preds_batch) {
    auto result = Process(preds_data, img_shapes, thresh.value_or(thresh_),
                          box_thresh.value_or(box_thresh_),
                          unclip_ratio.value_or(unclip_ratio_));

    if (!result.ok()) {
      return result.status();
    }

    auto boxes_result = *result;
    auto boxes = boxes_result.first;
    auto scores = boxes_result.second;
    all_boxes.insert(all_boxes.end(), boxes.begin(), boxes.end());
    all_scores.insert(all_scores.end(), scores.begin(), scores.end());
  }

  return std::make_pair(all_boxes, all_scores);
}

absl::StatusOr<std::vector<
    std::pair<std::vector<std::vector<cv::Point2f>>, std::vector<float>>>>
DBPostProcess::Apply(const cv::Mat &preds, const std::vector<int> &img_shapes,
                     absl::optional<float> thresh,
                     absl::optional<float> box_thresh,
                     absl::optional<float> unclip_ratio) {
  std::vector<
      std::pair<std::vector<std::vector<cv::Point2f>>, std::vector<float>>>
      db_result = {};

  auto preds_batch = Utility::SplitBatch(preds);

  if (!preds_batch.ok()) {
    return preds_batch.status();
  }
  for (const auto &pred : preds_batch.value()) {
    auto result = Process(pred, img_shapes, thresh.value_or(thresh_),
                          box_thresh.value_or(box_thresh_),
                          unclip_ratio.value_or(unclip_ratio_));

    if (!result.ok()) {
      return result.status();
    }
    db_result.push_back(result.value());
  }

  return db_result;
}

absl::StatusOr<
    std::pair<std::vector<std::vector<cv::Point2f>>, std::vector<float>>>
DBPostProcess::Process(const cv::Mat &pred, const std::vector<int> &img_shape,
                       float thresh, float box_thresh, float unclip_ratio) {
  cv::Mat pred_single = pred.clone();
  std::vector<int> shape_pred = {pred_single.size[pred_single.dims - 2],
                                 pred_single.size[pred_single.dims - 1]};
  pred_single = pred_single.reshape(1, shape_pred);
  cv::Mat segmentation = pred_single > thresh;
  cv::Mat mask;
  if (use_dilation_) {
    cv::Mat kernel = (cv::Mat_<uchar>(2, 2) << 1, 1, 1, 1); //暂时未测试
    cv::dilate(segmentation, mask, kernel);
  } else {
    mask = segmentation;
  }

  int src_h = img_shape[0];
  int src_w = img_shape[1];

  if (box_type_ == "poly") {
    return PolygonsFromBitmap(pred_single, mask, src_w, src_h, box_thresh,
                              unclip_ratio);
  } else if (box_type_ == "quad") {
    return BoxesFromBitmap(pred_single, mask, src_w, src_h, box_thresh,
                           unclip_ratio);
  }

  return absl::InvalidArgumentError(
      "box_type can only be one of ['quad', 'poly']");
}

absl::StatusOr<
    std::pair<std::vector<std::vector<cv::Point2f>>, std::vector<float>>>
DBPostProcess::PolygonsFromBitmap(const cv::Mat &pred, const cv::Mat &bitmap,
                                  int dest_width, int dest_height,
                                  float box_thresh, float unclip_ratio) {
  std::vector<std::vector<cv::Point2f>> boxes;
  std::vector<float> scores;

  float width_scale = static_cast<float>(dest_width) / bitmap.cols;
  float height_scale = static_cast<float>(dest_height) / bitmap.rows;

  cv::Mat bitmap_uint8;
  bitmap.convertTo(bitmap_uint8, CV_8UC1, 255.0);

  std::vector<std::vector<cv::Point2f>> contours;
  cv::findContours(bitmap_uint8, contours, cv::RETR_LIST,
                   cv::CHAIN_APPROX_SIMPLE);

  int num_contours =
      std::min(static_cast<int>(contours.size()), max_candidates_);

  for (int i = 0; i < num_contours; ++i) {
    const auto &contour = contours[i];

    std::vector<cv::Point2f> approx;
    double epsilon = 0.002 * cv::arcLength(contour, true);
    cv::approxPolyDP(contour, approx, epsilon, true);

    if (approx.size() < 4) {
      continue;
    }

    float score = BoxScoreFast(pred, approx);
    if (box_thresh > score) {
      continue;
    }

    std::vector<cv::Point2f> box;
    if (approx.size() > 2) {
      auto unclip_result = Unclip(approx, unclip_ratio);
      if (!unclip_result.ok()) {
        continue;
      }
      box = *unclip_result;
      if (box.size() > 1) {
        continue;
      }
    } else {
      continue;
    }

    if (!box.empty()) {
      auto min_box_result = GetMiniBoxes(box);
      auto min_box = min_box_result.first;
      auto sside = min_box_result.second;
      if (sside < min_size_ + 2) {
        continue;
      }

      for (auto &point : box) {
        point.x = std::max(
            0, std::min(static_cast<int>(std::round(point.x * width_scale)),
                        dest_width - 1));
        point.y = std::max(
            0, std::min(static_cast<int>(std::round(point.y * height_scale)),
                        dest_height - 1));
      }

      boxes.push_back(box);
      scores.push_back(score);
    }
  }

  return std::make_pair(boxes, scores);
}

absl::StatusOr<
    std::pair<std::vector<std::vector<cv::Point2f>>, std::vector<float>>>
DBPostProcess::BoxesFromBitmap(const cv::Mat &pred, const cv::Mat &bitmap,
                               int dest_width, int dest_height,
                               float box_thresh, float unclip_ratio) {
  std::vector<std::vector<cv::Point2f>> boxes;
  std::vector<float> scores;

  float width_scale = static_cast<float>(dest_width) / bitmap.cols;
  float height_scale = static_cast<float>(dest_height) / bitmap.rows;

  cv::Mat bitmap_uint8;
  bitmap.convertTo(bitmap_uint8, CV_8UC1, 255.0);

  std::vector<std::vector<cv::Point>> contours_;
  cv::findContours(bitmap_uint8, contours_, cv::RETR_LIST,
                   cv::CHAIN_APPROX_SIMPLE);
  std::vector<std::vector<cv::Point2f>> contours;
  for (const auto &contour : contours_) { // 这里可以优化
    std::vector<cv::Point2f> float_contour;
    for (const auto &point : contour) {
      float_contour.push_back(cv::Point2f(point.x, point.y));
    }
    contours.push_back(float_contour);
  }
  int num_contours =
      std::min(static_cast<int>(contours.size()), max_candidates_);

  for (int i = 0; i < num_contours; ++i) {
    const auto &contour = contours[i];

    auto contour_result = GetMiniBoxes(contour);
    auto points = contour_result.first;
    auto sside = contour_result.second;
    if (sside < min_size_) {
      continue;
    }

    float score = 0;
    if (score_mode_ == "fast") {
      score = BoxScoreFast(pred, points);
    } else {
      score = BoxScoreSlow(pred, contour);
    }

    if (box_thresh > score) {
      continue;
    }

    auto unclip_result = Unclip(points, unclip_ratio);
    if (!unclip_result.ok()) {
      continue;
    }

    auto box = *unclip_result;
    auto min_box_result = GetMiniBoxes(box);
    auto min_box = min_box_result.first;
    auto new_sside = min_box_result.second;
    if (new_sside < min_size_ + 2) {
      continue;
    }

    for (auto &point : min_box) {
      point.x = std::max(
          0, std::min(static_cast<int>(std::round(point.x * width_scale)),
                      dest_width - 1));
      point.y = std::max(
          0, std::min(static_cast<int>(std::round(point.y * height_scale)),
                      dest_height - 1));
    }

    boxes.push_back(min_box);
    scores.push_back(score);
  }

  return std::make_pair(boxes, scores);
}

absl::StatusOr<std::vector<cv::Point2f>>
DBPostProcess::Unclip(const std::vector<cv::Point2f> &box, float unclip_ratio) {
  float area = cv::contourArea(box);
  float length = cv::arcLength(box, true);
  float distance = area * unclip_ratio / length;

  ClipperLib::Path path;
  for (const auto &point : box) {
    path << ClipperLib::IntPoint(point.x, point.y);
  }

  ClipperLib::ClipperOffset co;
  co.AddPath(path, ClipperLib::jtRound, ClipperLib::etClosedPolygon);

  ClipperLib::Paths solution;
  co.Execute(solution, distance);

  if (solution.empty()) {
    return absl::InternalError("Failed to unclip polygon");
  }

  std::vector<cv::Point2f> result;
  for (const auto &p : solution[0]) {
    result.emplace_back(p.X, p.Y);
  }

  return result;
}

std::pair<std::vector<cv::Point2f>, float>
DBPostProcess::GetMiniBoxes(const std::vector<cv::Point2f> &contour) {
  cv::RotatedRect box = cv::minAreaRect(contour);

  std::vector<cv::Point2f> points(4);
  box.points(points.data());

  std::sort(
      points.begin(), points.end(),
      [](const cv::Point2f &a, const cv::Point2f &b) { return a.x < b.x; });

  int index_1 = 0, index_2 = 1, index_3 = 2, index_4 = 3;
  if (points[1].y > points[0].y) {
    index_1 = 0;
    index_4 = 1;
  } else {
    index_1 = 1;
    index_4 = 0;
  }

  if (points[3].y > points[2].y) {
    index_2 = 2;
    index_3 = 3;
  } else {
    index_2 = 3;
    index_3 = 2;
  }

  std::vector<cv::Point2f> box_points = {points[index_1], points[index_2],
                                         points[index_3], points[index_4]};

  float sside = std::min(box.size.width, box.size.height);
  return std::make_pair(box_points, sside);
}

float DBPostProcess::BoxScoreFast(const cv::Mat &bitmap,
                                  const std::vector<cv::Point2f> &contour) {
  int h = bitmap.size[bitmap.dims - 2]; // must be CHW
  int w = bitmap.size[bitmap.dims - 1];

  std::vector<cv::Point2f> contour_copy = contour;

  int xmin = std::max(
      0, static_cast<int>(std::floor(
             std::min_element(contour_copy.begin(), contour_copy.end(),
                              [](const cv::Point2f &a, const cv::Point2f &b) {
                                return a.x < b.x;
                              })
                 ->x)));
  int xmax = std::max(
      0, static_cast<int>(std::ceil(
             std::max_element(contour_copy.begin(), contour_copy.end(),
                              [](const cv::Point2f &a, const cv::Point2f &b) {
                                return a.x < b.x;
                              })
                 ->x)));
  int ymin = std::max(
      0, static_cast<int>(std::floor(
             std::min_element(contour_copy.begin(), contour_copy.end(),
                              [](const cv::Point2f &a, const cv::Point2f &b) {
                                return a.y < b.y;
                              })
                 ->y)));
  int ymax = std::max(
      0, static_cast<int>(std::ceil(
             std::max_element(contour_copy.begin(), contour_copy.end(),
                              [](const cv::Point2f &a, const cv::Point2f &b) {
                                return a.y < b.y;
                              })
                 ->y)));

  xmin = std::min(xmin, w - 1);
  xmax = std::min(xmax, w - 1);
  ymin = std::min(ymin, h - 1);
  ymax = std::min(ymax, h - 1);

  cv::Mat mask = cv::Mat::zeros(ymax - ymin + 1, xmax - xmin + 1, CV_8UC1);

  std::vector<cv::Point> contour_copy_int;
  for (auto &point : contour_copy) {
    point.x -= xmin;
    point.y -= ymin;
    contour_copy_int.push_back(
        cv::Point(static_cast<int>(point.x), static_cast<int>(point.y)));
  }

  std::vector<std::vector<cv::Point>> contours = {contour_copy_int};
  cv::fillPoly(mask, contours, cv::Scalar(1));

  cv::Mat roi = bitmap(cv::Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1));
  cv::Scalar mean_val = cv::mean(roi, mask);

  return mean_val[0];
}

float DBPostProcess::BoxScoreSlow(const cv::Mat &bitmap,
                                  const std::vector<cv::Point2f> &contour) {
  int h = bitmap.size[bitmap.dims - 2]; // must be CHW
  int w = bitmap.size[bitmap.dims - 1];

  std::vector<cv::Point2f> contour_copy = contour;

  int xmin = std::max(
      0, static_cast<int>(std::floor(
             std::min_element(contour_copy.begin(), contour_copy.end(),
                              [](const cv::Point2f &a, const cv::Point2f &b) {
                                return a.x < b.x;
                              })
                 ->x)));
  int xmax = std::max(
      0, static_cast<int>(std::ceil(
             std::max_element(contour_copy.begin(), contour_copy.end(),
                              [](const cv::Point2f &a, const cv::Point2f &b) {
                                return a.x < b.x;
                              })
                 ->x)));
  int ymin = std::max(
      0, static_cast<int>(std::floor(
             std::min_element(contour_copy.begin(), contour_copy.end(),
                              [](const cv::Point2f &a, const cv::Point2f &b) {
                                return a.y < b.y;
                              })
                 ->y)));
  int ymax = std::max(
      0, static_cast<int>(std::ceil(
             std::max_element(contour_copy.begin(), contour_copy.end(),
                              [](const cv::Point2f &a, const cv::Point2f &b) {
                                return a.y < b.y;
                              })
                 ->y)));

  xmin = std::min(xmin, w - 1);
  xmax = std::min(xmax, w - 1);
  ymin = std::min(ymin, h - 1);
  ymax = std::min(ymax, h - 1);

  cv::Mat mask = cv::Mat::zeros(ymax - ymin + 1, xmax - xmin + 1, CV_8UC1);

  std::vector<cv::Point> contour_copy_int;
  for (auto &point : contour_copy) {
    point.x -= xmin;
    point.y -= ymin;
    contour_copy_int.push_back(
        cv::Point(static_cast<int>(point.x), static_cast<int>(point.y)));
  }

  std::vector<std::vector<cv::Point>> contours = {contour_copy_int};
  cv::fillPoly(mask, contours, 1);

  cv::Scalar mean = cv::mean(
      bitmap(cv::Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1)), mask);
  return static_cast<float>(mean[0]);
}
