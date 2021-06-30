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

#include "db_post_process.h" // NOLINT
#include <algorithm>
#include <utility>

void GetContourArea(std::vector<std::vector<float>> box, float unclip_ratio,
                    float &distance) {
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

cv::RotatedRect Unclip(std::vector<std::vector<float>> box,
                       float unclip_ratio) {
  float distance = 1.0;

  GetContourArea(box, unclip_ratio, distance);

  ClipperLib::ClipperOffset offset;
  ClipperLib::Path p;
  p << ClipperLib::IntPoint(static_cast<int>(box[0][0]),
                            static_cast<int>(box[0][1]))
    << ClipperLib::IntPoint(static_cast<int>(box[1][0]),
                            static_cast<int>(box[1][1]))
    << ClipperLib::IntPoint(static_cast<int>(box[2][0]),
                            static_cast<int>(box[2][1]))
    << ClipperLib::IntPoint(static_cast<int>(box[3][0]),
                            static_cast<int>(box[3][1]));
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

std::vector<std::vector<float>> Mat2Vector(cv::Mat mat) {
  std::vector<std::vector<float>> img_vec;
  std::vector<float> tmp;

  for (int i = 0; i < mat.rows; ++i) {
    tmp.clear();
    for (int j = 0; j < mat.cols; ++j) {
      tmp.push_back(mat.at<float>(i, j));
    }
    img_vec.push_back(tmp);
  }
  return img_vec;
}

bool XsortFp32(std::vector<float> a, std::vector<float> b) {
  if (a[0] != b[0])
    return a[0] < b[0];
  return false;
}

bool XsortInt(std::vector<int> a, std::vector<int> b) {
  if (a[0] != b[0])
    return a[0] < b[0];
  return false;
}

std::vector<std::vector<int>>
OrderPointsClockwise(std::vector<std::vector<int>> pts) {
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

std::vector<std::vector<float>> GetMiniBoxes(cv::RotatedRect box, float &ssid) {
  ssid = std::min(box.size.width, box.size.height);

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

float BoxScoreFast(std::vector<std::vector<float>> box_array, cv::Mat pred) {
  auto array = box_array;
  int width = pred.cols;
  int height = pred.rows;

  float box_x[4] = {array[0][0], array[1][0], array[2][0], array[3][0]};
  float box_y[4] = {array[0][1], array[1][1], array[2][1], array[3][1]};

  int xmin = clamp(
      static_cast<int>(std::floorf(*(std::min_element(box_x, box_x + 4)))), 0,
      width - 1);
  int xmax =
      clamp(static_cast<int>(std::ceilf(*(std::max_element(box_x, box_x + 4)))),
            0, width - 1);
  int ymin = clamp(
      static_cast<int>(std::floorf(*(std::min_element(box_y, box_y + 4)))), 0,
      height - 1);
  int ymax =
      clamp(static_cast<int>(std::ceilf(*(std::max_element(box_y, box_y + 4)))),
            0, height - 1);

  cv::Mat mask;
  mask = cv::Mat::zeros(ymax - ymin + 1, xmax - xmin + 1, CV_8UC1);

  cv::Point root_point[4];
  root_point[0] = cv::Point(static_cast<int>(array[0][0]) - xmin,
                            static_cast<int>(array[0][1]) - ymin);
  root_point[1] = cv::Point(static_cast<int>(array[1][0]) - xmin,
                            static_cast<int>(array[1][1]) - ymin);
  root_point[2] = cv::Point(static_cast<int>(array[2][0]) - xmin,
                            static_cast<int>(array[2][1]) - ymin);
  root_point[3] = cv::Point(static_cast<int>(array[3][0]) - xmin,
                            static_cast<int>(array[3][1]) - ymin);
  const cv::Point *ppt[1] = {root_point};
  int npt[] = {4};
  cv::fillPoly(mask, ppt, npt, 1, cv::Scalar(1));

  cv::Mat croppedImg;
  pred(cv::Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1))
      .copyTo(croppedImg);

  auto score = cv::mean(croppedImg, mask)[0];
  return score;
}

float PolygonScoreAcc(std::vector<cv::Point> contour, cv::Mat pred) {
  int width = pred.cols;
  int height = pred.rows;
  std::vector<float> box_x;
  std::vector<float> box_y;
  for (int i = 0; i < contour.size(); ++i) {
    box_x.push_back(contour[i].x);
    box_y.push_back(contour[i].y);
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

  for (int i = 0; i < contour.size(); ++i) {
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

std::vector<std::vector<std::vector<int>>>
BoxesFromBitmap(const cv::Mat pred, const cv::Mat bitmap,
                std::map<std::string, double> Config) {
  const int min_size = 3;
  const int max_candidates = 1000;
  const float box_thresh = static_cast<float>(Config["det_db_box_thresh"]);
  const float unclip_ratio = static_cast<float>(Config["det_db_unclip_ratio"]);
  const int det_use_polygon_score = int(Config["det_use_polygon_score"]);

  int width = bitmap.cols;
  int height = bitmap.rows;

  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Vec4i> hierarchy;

  cv::findContours(bitmap, contours, hierarchy, cv::RETR_LIST,
                   cv::CHAIN_APPROX_SIMPLE);

  int num_contours =
      contours.size() >= max_candidates ? max_candidates : contours.size();

  std::vector<std::vector<std::vector<int>>> boxes;

  for (int i = 0; i < num_contours; i++) {
    float ssid;
    if (contours[i].size() <= 2)
      continue;

    cv::RotatedRect box = cv::minAreaRect(contours[i]);
    auto array = GetMiniBoxes(box, ssid);

    auto box_for_unclip = array;
    // end get_mini_box

    if (ssid < min_size) {
      continue;
    }

    float score;
    if (det_use_polygon_score) {
      score = PolygonScoreAcc(contours[i], pred);
    } else {
      score = BoxScoreFast(array, pred);
    }
    // end box_score_fast
    if (score < box_thresh)
      continue;

    // start for unclip
    cv::RotatedRect points = Unclip(box_for_unclip, unclip_ratio);
    if (points.size.height < 1.001 && points.size.width < 1.001)
      continue;
    // end for unclip

    cv::RotatedRect clipbox = points;
    auto cliparray = GetMiniBoxes(clipbox, ssid);

    if (ssid < min_size + 2)
      continue;

    int dest_width = pred.cols;
    int dest_height = pred.rows;
    std::vector<std::vector<int>> intcliparray;

    for (int num_pt = 0; num_pt < 4; num_pt++) {
      std::vector<int> a{
          static_cast<int>(clamp(
              roundf(cliparray[num_pt][0] / float(width) * float(dest_width)),
              float(0), float(dest_width))),
          static_cast<int>(clamp(
              roundf(cliparray[num_pt][1] / float(height) * float(dest_height)),
              float(0), float(dest_height)))};
      intcliparray.push_back(a);
    }
    boxes.push_back(intcliparray);

  } // end for
  return boxes;
}

std::vector<std::vector<std::vector<int>>>
FilterTagDetRes(std::vector<std::vector<std::vector<int>>> boxes, float ratio_h,
                float ratio_w, cv::Mat srcimg) {
  int oriimg_h = srcimg.rows;
  int oriimg_w = srcimg.cols;

  std::vector<std::vector<std::vector<int>>> root_points;
  for (int n = 0; n < static_cast<int>(boxes.size()); n++) {
    boxes[n] = OrderPointsClockwise(boxes[n]);
    for (int m = 0; m < static_cast<int>(boxes[0].size()); m++) {
      boxes[n][m][0] /= ratio_w;
      boxes[n][m][1] /= ratio_h;

      boxes[n][m][0] =
          static_cast<int>(std::min(std::max(boxes[n][m][0], 0), oriimg_w - 1));
      boxes[n][m][1] =
          static_cast<int>(std::min(std::max(boxes[n][m][1], 0), oriimg_h - 1));
    }
  }

  for (int n = 0; n < boxes.size(); n++) {
    int rect_width, rect_height;
    rect_width =
        static_cast<int>(sqrt(pow(boxes[n][0][0] - boxes[n][1][0], 2) +
                              pow(boxes[n][0][1] - boxes[n][1][1], 2)));
    rect_height =
        static_cast<int>(sqrt(pow(boxes[n][0][0] - boxes[n][3][0], 2) +
                              pow(boxes[n][0][1] - boxes[n][3][1], 2)));
    if (rect_width <= 4 || rect_height <= 4)
      continue;
    root_points.push_back(boxes[n]);
  }
  return root_points;
}
