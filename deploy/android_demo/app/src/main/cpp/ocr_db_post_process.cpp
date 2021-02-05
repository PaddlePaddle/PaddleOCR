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

#include "ocr_clipper.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <math.h>
#include <vector>

static void getcontourarea(float **box, float unclip_ratio, float &distance) {
  int pts_num = 4;
  float area = 0.0f;
  float dist = 0.0f;
  for (int i = 0; i < pts_num; i++) {
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

static cv::RotatedRect unclip(float **box) {
  float unclip_ratio = 2.0;
  float distance = 1.0;

  getcontourarea(box, unclip_ratio, distance);

  ClipperLib::ClipperOffset offset;
  ClipperLib::Path p;
  p << ClipperLib::IntPoint(int(box[0][0]), int(box[0][1]))
    << ClipperLib::IntPoint(int(box[1][0]), int(box[1][1]))
    << ClipperLib::IntPoint(int(box[2][0]), int(box[2][1]))
    << ClipperLib::IntPoint(int(box[3][0]), int(box[3][1]));
  offset.AddPath(p, ClipperLib::jtRound, ClipperLib::etClosedPolygon);

  ClipperLib::Paths soln;
  offset.Execute(soln, distance);
  std::vector<cv::Point2f> points;

  for (int j = 0; j < soln.size(); j++) {
    for (int i = 0; i < soln[soln.size() - 1].size(); i++) {
      points.emplace_back(soln[j][i].X, soln[j][i].Y);
    }
  }
  cv::RotatedRect res = cv::minAreaRect(points);

  return res;
}

static float **Mat2Vec(cv::Mat mat) {
  auto **array = new float *[mat.rows];
  for (int i = 0; i < mat.rows; ++i) {
    array[i] = new float[mat.cols];
  }
  for (int i = 0; i < mat.rows; ++i) {
    for (int j = 0; j < mat.cols; ++j) {
      array[i][j] = mat.at<float>(i, j);
    }
  }

  return array;
}

static void quickSort(float **s, int l, int r) {
  if (l < r) {
    int i = l, j = r;
    float x = s[l][0];
    float *xp = s[l];
    while (i < j) {
      while (i < j && s[j][0] >= x) {
        j--;
      }
      if (i < j) {
        std::swap(s[i++], s[j]);
      }
      while (i < j && s[i][0] < x) {
        i++;
      }
      if (i < j) {
        std::swap(s[j--], s[i]);
      }
    }
    s[i] = xp;
    quickSort(s, l, i - 1);
    quickSort(s, i + 1, r);
  }
}

static void quickSort_vector(std::vector<std::vector<int>> &box, int l, int r,
                             int axis) {
  if (l < r) {
    int i = l, j = r;
    int x = box[l][axis];
    std::vector<int> xp(box[l]);
    while (i < j) {
      while (i < j && box[j][axis] >= x) {
        j--;
      }
      if (i < j) {
        std::swap(box[i++], box[j]);
      }
      while (i < j && box[i][axis] < x) {
        i++;
      }
      if (i < j) {
        std::swap(box[j--], box[i]);
      }
    }
    box[i] = xp;
    quickSort_vector(box, l, i - 1, axis);
    quickSort_vector(box, i + 1, r, axis);
  }
}

static std::vector<std::vector<int>>
order_points_clockwise(std::vector<std::vector<int>> pts) {
  std::vector<std::vector<int>> box = pts;
  quickSort_vector(box, 0, int(box.size() - 1), 0);
  std::vector<std::vector<int>> leftmost = {box[0], box[1]};
  std::vector<std::vector<int>> rightmost = {box[2], box[3]};

  if (leftmost[0][1] > leftmost[1][1]) {
    std::swap(leftmost[0], leftmost[1]);
  }

  if (rightmost[0][1] > rightmost[1][1]) {
    std::swap(rightmost[0], rightmost[1]);
  }

  std::vector<std::vector<int>> rect = {leftmost[0], rightmost[0], rightmost[1],
                                        leftmost[1]};
  return rect;
}

static float **get_mini_boxes(cv::RotatedRect box, float &ssid) {
  ssid = box.size.width >= box.size.height ? box.size.height : box.size.width;

  cv::Mat points;
  cv::boxPoints(box, points);
  // sorted box points
  auto array = Mat2Vec(points);
  quickSort(array, 0, 3);

  float *idx1 = array[0], *idx2 = array[1], *idx3 = array[2], *idx4 = array[3];
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

template <class T> T clamp(T x, T min, T max) {
  if (x > max) {
    return max;
  }
  if (x < min) {
    return min;
  }
  return x;
}

static float clampf(float x, float min, float max) {
  if (x > max)
    return max;
  if (x < min)
    return min;
  return x;
}

float box_score_fast(float **box_array, cv::Mat pred) {
  auto array = box_array;
  int width = pred.cols;
  int height = pred.rows;

  float box_x[4] = {array[0][0], array[1][0], array[2][0], array[3][0]};
  float box_y[4] = {array[0][1], array[1][1], array[2][1], array[3][1]};

  int xmin = clamp(int(std::floorf(*(std::min_element(box_x, box_x + 4)))), 0,
                   width - 1);
  int xmax = clamp(int(std::ceilf(*(std::max_element(box_x, box_x + 4)))), 0,
                   width - 1);
  int ymin = clamp(int(std::floorf(*(std::min_element(box_y, box_y + 4)))), 0,
                   height - 1);
  int ymax = clamp(int(std::ceilf(*(std::max_element(box_y, box_y + 4)))), 0,
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

std::vector<std::vector<std::vector<int>>>
boxes_from_bitmap(const cv::Mat &pred, const cv::Mat &bitmap) {
  const int min_size = 3;
  const int max_candidates = 1000;
  const float box_thresh = 0.5;

  int width = bitmap.cols;
  int height = bitmap.rows;

  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Vec4i> hierarchy;

  cv::findContours(bitmap, contours, hierarchy, cv::RETR_LIST,
                   cv::CHAIN_APPROX_SIMPLE);

  int num_contours =
      contours.size() >= max_candidates ? max_candidates : contours.size();

  std::vector<std::vector<std::vector<int>>> boxes;

  for (int _i = 0; _i < num_contours; _i++) {
    float ssid;
    cv::RotatedRect box = cv::minAreaRect(contours[_i]);
    auto array = get_mini_boxes(box, ssid);

    auto box_for_unclip = array;
    // end get_mini_box

    if (ssid < min_size) {
      continue;
    }

    float score;
    score = box_score_fast(array, pred);
    // end box_score_fast
    if (score < box_thresh) {
      continue;
    }

    // start for unclip
    cv::RotatedRect points = unclip(box_for_unclip);
    // end for unclip

    cv::RotatedRect clipbox = points;
    auto cliparray = get_mini_boxes(clipbox, ssid);

    if (ssid < min_size + 2)
      continue;

    int dest_width = pred.cols;
    int dest_height = pred.rows;
    std::vector<std::vector<int>> intcliparray;

    for (int num_pt = 0; num_pt < 4; num_pt++) {
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

int _max(int a, int b) { return a >= b ? a : b; }

int _min(int a, int b) { return a >= b ? b : a; }

std::vector<std::vector<std::vector<int>>>
filter_tag_det_res(const std::vector<std::vector<std::vector<int>>> &o_boxes,
                   float ratio_h, float ratio_w, const cv::Mat &srcimg) {
  int oriimg_h = srcimg.rows;
  int oriimg_w = srcimg.cols;
  std::vector<std::vector<std::vector<int>>> boxes{o_boxes};
  std::vector<std::vector<std::vector<int>>> root_points;
  for (int n = 0; n < boxes.size(); n++) {
    boxes[n] = order_points_clockwise(boxes[n]);
    for (int m = 0; m < boxes[0].size(); m++) {
      boxes[n][m][0] /= ratio_w;
      boxes[n][m][1] /= ratio_h;

      boxes[n][m][0] = int(_min(_max(boxes[n][m][0], 0), oriimg_w - 1));
      boxes[n][m][1] = int(_min(_max(boxes[n][m][1], 0), oriimg_h - 1));
    }
  }

  for (int n = 0; n < boxes.size(); n++) {
    int rect_width, rect_height;
    rect_width = int(sqrt(pow(boxes[n][0][0] - boxes[n][1][0], 2) +
                          pow(boxes[n][0][1] - boxes[n][1][1], 2)));
    rect_height = int(sqrt(pow(boxes[n][0][0] - boxes[n][3][0], 2) +
                           pow(boxes[n][0][1] - boxes[n][3][1], 2)));
    if (rect_width <= 10 || rect_height <= 10)
      continue;
    root_points.push_back(boxes[n]);
  }
  return root_points;
}