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

#include <include/clipper.h>
#include <include/postprocess_op.h>

namespace PaddleOCR {

void DBPostProcessor::GetContourArea(const std::vector<std::vector<float>> &box,
                                     float unclip_ratio,
                                     float &distance) noexcept {
  int pts_num = 4;
  float area = 0.0f;
  float dist = 0.0f;
  for (int i = 0; i < pts_num; ++i) {
    area += box[i][0] * box[(i + 1) % pts_num][1] -
            box[i][1] * box[(i + 1) % pts_num][0];
    dist += sqrtf((box[i][0] - box[(i + 1) % pts_num][0]) *
                      (box[i][0] - box[(i + 1) % pts_num][0]) +
                  (box[i][1] - box[(i + 1) % pts_num][1]) *
                      (box[i][1] - box[(i + 1) % pts_num][1]));
  }
  area = fabs(float(area / 2.0));

  distance = area * unclip_ratio / dist;
}

cv::RotatedRect
DBPostProcessor::UnClip(const std::vector<std::vector<float>> &box,
                        const float &unclip_ratio) noexcept {
  float distance = 1.0;

  GetContourArea(box, unclip_ratio, distance);

  ClipperLib::ClipperOffset offset;
  ClipperLib::Path p;
  p.emplace_back(int(box[0][0]), int(box[0][1]));
  p.emplace_back(int(box[1][0]), int(box[1][1]));
  p.emplace_back(int(box[2][0]), int(box[2][1]));
  p.emplace_back(int(box[3][0]), int(box[3][1]));
  offset.AddPath(p, ClipperLib::jtRound, ClipperLib::etClosedPolygon);

  ClipperLib::Paths soln;
  if (!offset.Execute(soln, distance))
    return cv::RotatedRect();

  std::vector<cv::Point2f> points;

  for (size_t j = 0; j < soln.size(); ++j) {
    for (size_t i = 0; i < soln[soln.size() - 1].size(); ++i) {
      points.emplace_back(soln[j][i].X, soln[j][i].Y);
    }
  }
  cv::RotatedRect res;
  if (points.size() <= 0) {
    res = cv::RotatedRect(cv::Point2f(0, 0), cv::Size2f(1, 1), 0);
  } else {
    res = cv::minAreaRect(points);
  }
  return res;
}

float **DBPostProcessor::Mat2Vec(const cv::Mat &mat) noexcept {
  auto **array = new float *[mat.rows];
  for (int i = 0; i < mat.rows; ++i)
    array[i] = new float[mat.cols];
  for (int i = 0; i < mat.rows; ++i) {
    for (int j = 0; j < mat.cols; ++j) {
      array[i][j] = mat.at<float>(i, j);
    }
  }

  return array;
}

std::vector<std::vector<int>> DBPostProcessor::OrderPointsClockwise(
    const std::vector<std::vector<int>> &pts) noexcept {
  std::vector<std::vector<int>> box = pts;
  std::sort(box.begin(), box.end(), XsortInt);

  std::vector<std::vector<int>> leftmost = {box[0], box[1]};
  std::vector<std::vector<int>> rightmost = {box[2], box[3]};

  if (leftmost[0][1] > leftmost[1][1])
    std::swap(leftmost[0], leftmost[1]);

  if (rightmost[0][1] > rightmost[1][1])
    std::swap(rightmost[0], rightmost[1]);

  std::vector<std::vector<int>> rect = {leftmost[0], rightmost[0], rightmost[1],
                                        leftmost[1]};
  return rect;
}

std::vector<std::vector<float>>
DBPostProcessor::Mat2Vector(const cv::Mat &mat) noexcept {
  std::vector<std::vector<float>> img_vec;

  for (int i = 0; i < mat.rows; ++i) {
    std::vector<float> tmp;
    for (int j = 0; j < mat.cols; ++j) {
      tmp.emplace_back(mat.at<float>(i, j));
    }
    img_vec.emplace_back(std::move(tmp));
  }
  return img_vec;
}

bool DBPostProcessor::XsortFp32(const std::vector<float> &a,
                                const std::vector<float> &b) noexcept {
  if (a[0] != b[0])
    return a[0] < b[0];
  return false;
}

bool DBPostProcessor::XsortInt(const std::vector<int> &a,
                               const std::vector<int> &b) noexcept {
  if (a[0] != b[0])
    return a[0] < b[0];
  return false;
}

std::vector<std::vector<float>>
DBPostProcessor::GetMiniBoxes(const cv::RotatedRect &box,
                              float &ssid) noexcept {
  ssid = std::max(box.size.width, box.size.height);

  cv::Mat points;
  cv::boxPoints(box, points);

  auto array = Mat2Vector(points);
  std::sort(array.begin(), array.end(), XsortFp32);

  std::vector<float> idx1 = array[0], idx2 = array[1], idx3 = array[2],
                     idx4 = array[3];
  if (array[3][1] <= array[2][1]) {
    idx2 = array[3];
    idx3 = array[2];
  } else {
    idx2 = array[2];
    idx3 = array[3];
  }
  if (array[1][1] <= array[0][1]) {
    idx1 = array[1];
    idx4 = array[0];
  } else {
    idx1 = array[0];
    idx4 = array[1];
  }

  array[0] = idx1;
  array[1] = idx2;
  array[2] = idx3;
  array[3] = idx4;

  return array;
}

float DBPostProcessor::PolygonScoreAcc(const std::vector<cv::Point> &contour,
                                       const cv::Mat &pred) noexcept {
  int width = pred.cols;
  int height = pred.rows;
  std::vector<float> box_x;
  std::vector<float> box_y;
  for (size_t i = 0; i < contour.size(); ++i) {
    box_x.emplace_back(contour[i].x);
    box_y.emplace_back(contour[i].y);
  }

  int xmin =
      clamp(int(std::floor(*(std::min_element(box_x.begin(), box_x.end())))), 0,
            width - 1);
  int xmax =
      clamp(int(std::ceil(*(std::max_element(box_x.begin(), box_x.end())))), 0,
            width - 1);
  int ymin =
      clamp(int(std::floor(*(std::min_element(box_y.begin(), box_y.end())))), 0,
            height - 1);
  int ymax =
      clamp(int(std::ceil(*(std::max_element(box_y.begin(), box_y.end())))), 0,
            height - 1);

  cv::Mat mask;
  mask = cv::Mat::zeros(ymax - ymin + 1, xmax - xmin + 1, CV_8UC1);

  cv::Point *rook_point = new cv::Point[contour.size()];

  for (size_t i = 0; i < contour.size(); ++i) {
    rook_point[i] = cv::Point(int(box_x[i]) - xmin, int(box_y[i]) - ymin);
  }
  const cv::Point *ppt[1] = {rook_point};
  int npt[] = {int(contour.size())};

  cv::fillPoly(mask, ppt, npt, 1, cv::Scalar(1));

  cv::Mat croppedImg;
  pred(cv::Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1))
      .copyTo(croppedImg);
  float score = cv::mean(croppedImg, mask)[0];

  delete[] rook_point;
  return score;
}

float DBPostProcessor::BoxScoreFast(
    const std::vector<std::vector<float>> &box_array,
    const cv::Mat &pred) noexcept {
  const auto &array = box_array;
  int width = pred.cols;
  int height = pred.rows;

  float box_x[4] = {array[0][0], array[1][0], array[2][0], array[3][0]};
  float box_y[4] = {array[0][1], array[1][1], array[2][1], array[3][1]};

  int xmin = clamp(int(std::floor(*(std::min_element(box_x, box_x + 4)))), 0,
                   width - 1);
  int xmax = clamp(int(std::ceil(*(std::max_element(box_x, box_x + 4)))), 0,
                   width - 1);
  int ymin = clamp(int(std::floor(*(std::min_element(box_y, box_y + 4)))), 0,
                   height - 1);
  int ymax = clamp(int(std::ceil(*(std::max_element(box_y, box_y + 4)))), 0,
                   height - 1);

  cv::Mat mask;
  mask = cv::Mat::zeros(ymax - ymin + 1, xmax - xmin + 1, CV_8UC1);

  cv::Point root_point[4];
  root_point[0] = cv::Point(int(array[0][0]) - xmin, int(array[0][1]) - ymin);
  root_point[1] = cv::Point(int(array[1][0]) - xmin, int(array[1][1]) - ymin);
  root_point[2] = cv::Point(int(array[2][0]) - xmin, int(array[2][1]) - ymin);
  root_point[3] = cv::Point(int(array[3][0]) - xmin, int(array[3][1]) - ymin);
  const cv::Point *ppt[1] = {root_point};
  int npt[] = {4};
  cv::fillPoly(mask, ppt, npt, 1, cv::Scalar(1));

  cv::Mat croppedImg;
  pred(cv::Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1))
      .copyTo(croppedImg);

  auto score = cv::mean(croppedImg, mask)[0];
  return score;
}

std::vector<std::vector<std::vector<int>>> DBPostProcessor::BoxesFromBitmap(
    const cv::Mat &pred, const cv::Mat &bitmap, const float &box_thresh,
    const float &det_db_unclip_ratio,
    const std::string &det_db_score_mode) noexcept {
  const int min_size = 3;
  const int max_candidates = 1000;

  int width = bitmap.cols;
  int height = bitmap.rows;

  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Vec4i> hierarchy;

  cv::findContours(bitmap, contours, hierarchy, cv::RETR_LIST,
                   cv::CHAIN_APPROX_SIMPLE);

  int num_contours =
      contours.size() >= max_candidates ? max_candidates : contours.size();

  std::vector<std::vector<std::vector<int>>> boxes;

  for (int _i = 0; _i < num_contours; ++_i) {
    if (contours[_i].size() <= 2) {
      continue;
    }
    float ssid;
    cv::RotatedRect box = cv::minAreaRect(contours[_i]);
    auto array = GetMiniBoxes(box, ssid);

    auto box_for_unclip = array;
    // end get_mini_box

    if (ssid < min_size) {
      continue;
    }

    float score;
    if (det_db_score_mode == "slow")
      /* compute using polygon*/
      score = PolygonScoreAcc(contours[_i], pred);
    else
      score = BoxScoreFast(array, pred);

    if (score < box_thresh)
      continue;

    // start for unclip
    cv::RotatedRect points = UnClip(box_for_unclip, det_db_unclip_ratio);
    if (points.size.height < 1.001 && points.size.width < 1.001) {
      continue;
    }
    // end for unclip

    cv::RotatedRect clipbox = points;
    auto cliparray = GetMiniBoxes(clipbox, ssid);

    if (ssid < min_size + 2)
      continue;

    int dest_width = pred.cols;
    int dest_height = pred.rows;
    std::vector<std::vector<int>> intcliparray;

    for (int num_pt = 0; num_pt < 4; ++num_pt) {
      std::vector<int> a{int(clampf(roundf(cliparray[num_pt][0] / float(width) *
                                           float(dest_width)),
                                    0, float(dest_width))),
                         int(clampf(roundf(cliparray[num_pt][1] /
                                           float(height) * float(dest_height)),
                                    0, float(dest_height)))};
      intcliparray.emplace_back(std::move(a));
    }
    boxes.emplace_back(std::move(intcliparray));

  } // end for
  return boxes;
}

void DBPostProcessor::FilterTagDetRes(
    std::vector<std::vector<std::vector<int>>> &boxes, float ratio_h,
    float ratio_w, const cv::Mat &srcimg) noexcept {
  int oriimg_h = srcimg.rows;
  int oriimg_w = srcimg.cols;

  std::vector<std::vector<std::vector<int>>> root_points;
  for (size_t n = 0; n < boxes.size(); ++n) {
    boxes[n] = OrderPointsClockwise(boxes[n]);
    for (size_t m = 0; m < boxes[0].size(); ++m) {
      boxes[n][m][0] /= ratio_w;
      boxes[n][m][1] /= ratio_h;

      boxes[n][m][0] = int(_min(_max(boxes[n][m][0], 0), oriimg_w - 1));
      boxes[n][m][1] = int(_min(_max(boxes[n][m][1], 0), oriimg_h - 1));
    }
  }

  for (size_t n = 0; n < boxes.size(); ++n) {
    int rect_width, rect_height;
    rect_width = int(sqrt(pow(boxes[n][0][0] - boxes[n][1][0], 2) +
                          pow(boxes[n][0][1] - boxes[n][1][1], 2)));
    rect_height = int(sqrt(pow(boxes[n][0][0] - boxes[n][3][0], 2) +
                           pow(boxes[n][0][1] - boxes[n][3][1], 2)));
    if (rect_width <= 4 || rect_height <= 4)
      continue;
    root_points.emplace_back(boxes[n]);
  }
  boxes = std::move(root_points);
}

void TablePostProcessor::init(const std::string &label_path,
                              bool merge_no_span_structure) noexcept {
  this->label_list_ = Utility::ReadDict(label_path);
  if (merge_no_span_structure) {
    this->label_list_.emplace_back("<td></td>");
    std::vector<std::string>::iterator it;
    for (it = this->label_list_.begin(); it != this->label_list_.end();) {
      if (*it == "<td>") {
        it = this->label_list_.erase(it);
      } else {
        ++it;
      }
    }
  }
  // add_special_char
  this->label_list_.emplace(this->label_list_.begin(), this->beg);
  this->label_list_.emplace_back(this->end);
}

void TablePostProcessor::Run(
    const std::vector<float> &loc_preds,
    const std::vector<float> &structure_probs, std::vector<float> &rec_scores,
    const std::vector<int> &loc_preds_shape,
    const std::vector<int> &structure_probs_shape,
    std::vector<std::vector<std::string>> &rec_html_tag_batch,
    std::vector<std::vector<std::vector<int>>> &rec_boxes_batch,
    const std::vector<int> &width_list,
    const std::vector<int> &height_list) noexcept {
  for (int batch_idx = 0; batch_idx < structure_probs_shape[0]; ++batch_idx) {
    // image tags and boxes
    std::vector<std::string> rec_html_tags;
    std::vector<std::vector<int>> rec_boxes;

    float score = 0.f;
    int count = 0;
    float char_score = 0.f;
    int char_idx = 0;

    // step
    for (int step_idx = 0; step_idx < structure_probs_shape[1]; ++step_idx) {
      std::string html_tag;
      std::vector<int> rec_box;
      // html tag
      int step_start_idx = (batch_idx * structure_probs_shape[1] + step_idx) *
                           structure_probs_shape[2];
      char_idx = int(Utility::argmax(
          &structure_probs[step_start_idx],
          &structure_probs[step_start_idx + structure_probs_shape[2]]));
      char_score = float(*std::max_element(
          &structure_probs[step_start_idx],
          &structure_probs[step_start_idx + structure_probs_shape[2]]));
      html_tag = this->label_list_[char_idx];

      if (step_idx > 0 && html_tag == this->end) {
        break;
      }
      if (html_tag == this->beg) {
        continue;
      }
      count += 1;
      score += char_score;
      rec_html_tags.emplace_back(html_tag);

      // box
      if (html_tag == "<td>" || html_tag == "<td" || html_tag == "<td></td>") {
        for (int point_idx = 0; point_idx < loc_preds_shape[2]; ++point_idx) {
          step_start_idx = (batch_idx * structure_probs_shape[1] + step_idx) *
                               loc_preds_shape[2] +
                           point_idx;
          float point = loc_preds[step_start_idx];
          if (point_idx % 2 == 0) {
            point = int(point * width_list[batch_idx]);
          } else {
            point = int(point * height_list[batch_idx]);
          }
          rec_box.emplace_back(point);
        }
        rec_boxes.emplace_back(std::move(rec_box));
      }
    }
    score /= count;
    if (std::isnan(score) || rec_boxes.size() == 0) {
      score = -1;
    }
    rec_scores.emplace_back(score);
    rec_boxes_batch.emplace_back(std::move(rec_boxes));
    rec_html_tag_batch.emplace_back(std::move(rec_html_tags));
  }
}

void PicodetPostProcessor::init(const std::string &label_path,
                                const double score_threshold,
                                const double nms_threshold,
                                const std::vector<int> &fpn_stride) noexcept {
  this->label_list_ = Utility::ReadDict(label_path);
  this->score_threshold_ = score_threshold;
  this->nms_threshold_ = nms_threshold;
  this->num_class_ = label_list_.size();
  this->fpn_stride_ = fpn_stride;
}

void PicodetPostProcessor::Run(std::vector<StructurePredictResult> &results,
                               const std::vector<std::vector<float>> &outs,
                               const std::vector<int> &ori_shape,
                               const std::vector<int> &resize_shape,
                               int reg_max) noexcept {
  int in_h = resize_shape[0];
  int in_w = resize_shape[1];
  float scale_factor_h = resize_shape[0] / float(ori_shape[0]);
  float scale_factor_w = resize_shape[1] / float(ori_shape[1]);

  std::vector<std::vector<StructurePredictResult>> bbox_results;
  bbox_results.resize(this->num_class_);
  for (size_t i = 0; i < this->fpn_stride_.size(); ++i) {
    const int feature_h = std::ceil((float)in_h / this->fpn_stride_[i]);
    const int feature_w = std::ceil((float)in_w / this->fpn_stride_[i]);
    const size_t hxw = feature_h * feature_w;
    for (size_t idx = 0; idx < hxw; ++idx) {
      // score and label
      float score = 0;
      int cur_label = 0;
      for (size_t label = 0; label < this->label_list_.size(); ++label) {
        float osc = outs[i][idx * this->label_list_.size() + label];
        if (osc > score) {
          score = osc;
          cur_label = label;
        }
      }
      // bbox
      if (score > this->score_threshold_) {
        int row = idx / feature_w;
        int col = idx % feature_w;
        std::vector<float>::const_iterator itemp =
            outs[i + this->fpn_stride_.size()].begin() + idx * 4 * reg_max;
        std::vector<float> bbox_pred(itemp, itemp + 4 * reg_max);
        bbox_results[cur_label].emplace_back(std::move(
            this->disPred2Bbox(bbox_pred, cur_label, score, col, row,
                               this->fpn_stride_[i], resize_shape, reg_max)));
      }
    }
  }
#if 0
  for (size_t i = 0; i < bbox_results.size(); ++i) {
    bool flag = bbox_results[i].size() <= 0;
  }
#endif
  for (size_t i = 0; i < bbox_results.size(); ++i) {
    // bool flag = bbox_results[i].size() <= 0;
    if (bbox_results[i].size() <= 0) {
      continue;
    }
    this->nms(bbox_results[i], this->nms_threshold_);
    for (auto &box : bbox_results[i]) {
      box.box[0] = box.box[0] / scale_factor_w;
      box.box[2] = box.box[2] / scale_factor_w;
      box.box[1] = box.box[1] / scale_factor_h;
      box.box[3] = box.box[3] / scale_factor_h;
      results.emplace_back(std::move(box));
    }
  }
}

StructurePredictResult PicodetPostProcessor::disPred2Bbox(
    const std::vector<float> &bbox_pred, int label, float score, int x, int y,
    int stride, const std::vector<int> &im_shape, int reg_max) noexcept {
  float ct_x = (x + 0.5) * stride;
  float ct_y = (y + 0.5) * stride;
  std::vector<float> dis_pred;
  dis_pred.resize(4);
  for (int i = 0; i < 4; ++i) {
    float dis = 0;
    std::vector<float>::const_iterator itemp = bbox_pred.begin() + i * reg_max;
    std::vector<float> bbox_pred_i(itemp, itemp + reg_max);
    std::vector<float> dis_after_sm(
        std::move(Utility::activation_function_softmax(bbox_pred_i)));
    for (int j = 0; j < reg_max; ++j) {
      dis += j * dis_after_sm[j];
    }
    dis *= stride;
    dis_pred[i] = dis;
  }

  float xmin = (std::max)(ct_x - dis_pred[0], .0f);
  float ymin = (std::max)(ct_y - dis_pred[1], .0f);
  float xmax = (std::min)(ct_x + dis_pred[2], (float)im_shape[1]);
  float ymax = (std::min)(ct_y + dis_pred[3], (float)im_shape[0]);

  StructurePredictResult result_item;
  result_item.box = {xmin, ymin, xmax, ymax};
  result_item.type = this->label_list_[label];
  result_item.confidence = score;

  return result_item;
}

void PicodetPostProcessor::nms(std::vector<StructurePredictResult> &input_boxes,
                               float nms_threshold) noexcept {
  std::sort(input_boxes.begin(), input_boxes.end(),
            [](StructurePredictResult a, StructurePredictResult b) noexcept {
              return a.confidence > b.confidence;
            });
  std::vector<int> picked(input_boxes.size(), 1);

  for (size_t i = 0; i < input_boxes.size(); ++i) {
    if (picked[i] == 0) {
      continue;
    }
    for (size_t j = i + 1; j < input_boxes.size(); ++j) {
      if (picked[j] == 0) {
        continue;
      }
      float iou = Utility::iou(input_boxes[i].box, input_boxes[j].box);
      if (iou > nms_threshold) {
        picked[j] = 0;
      }
    }
  }
  std::vector<StructurePredictResult> input_boxes_nms;
  for (size_t i = 0; i < input_boxes.size(); ++i) {
    if (picked[i] == 1) {
      input_boxes_nms.emplace_back(input_boxes[i]);
    }
  }
  input_boxes = std::move(input_boxes_nms);
}

} // namespace PaddleOCR
