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

#include <dirent.h>
#include <include/utility.h>
#include <opencv2/imgcodecs.hpp>

#include <fstream>
#include <iostream>
#include <vector>

#ifdef _MSC_VER
#include <direct.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace PaddleOCR {

std::vector<std::string> Utility::ReadDict(const std::string &path) noexcept {
  std::vector<std::string> m_vec;
  std::ifstream in(path);
  if (in) {
    for (;;) {
      std::string line;
      if (!getline(in, line))
        break;
      m_vec.emplace_back(std::move(line));
    }
  } else {
    std::cout << "no such label file: " << path << ", exit the program..."
              << std::endl;
    exit(1);
  }
  return m_vec;
}

void Utility::VisualizeBboxes(const cv::Mat &srcimg,
                              const std::vector<OCRPredictResult> &ocr_result,
                              const std::string &save_path) noexcept {
  cv::Mat img_vis;
  srcimg.copyTo(img_vis);
  for (size_t n = 0; n < ocr_result.size(); ++n) {
    cv::Point rook_points[4];
    for (size_t m = 0; m < ocr_result[n].box.size(); ++m) {
      rook_points[m] =
          cv::Point(int(ocr_result[n].box[m][0]), int(ocr_result[n].box[m][1]));
    }

    const cv::Point *ppt[1] = {rook_points};
    int npt[] = {4};
    cv::polylines(img_vis, ppt, npt, 1, 1, CV_RGB(0, 255, 0), 2, 8, 0);
  }

  cv::imwrite(save_path, img_vis);
  std::cout << "The detection visualized image saved in " + save_path
            << std::endl;
}

void Utility::VisualizeBboxes(const cv::Mat &srcimg,
                              const StructurePredictResult &structure_result,
                              const std::string &save_path) noexcept {
  cv::Mat img_vis;
  srcimg.copyTo(img_vis);
  img_vis = crop_image(img_vis, structure_result.box);
  for (size_t n = 0; n < structure_result.cell_box.size(); ++n) {
    if (structure_result.cell_box[n].size() == 8) {
      cv::Point rook_points[4];
      for (size_t m = 0; m < structure_result.cell_box[n].size(); m += 2) {
        rook_points[m / 2] =
            cv::Point(int(structure_result.cell_box[n][m]),
                      int(structure_result.cell_box[n][m + 1]));
      }
      const cv::Point *ppt[1] = {rook_points};
      int npt[] = {4};
      cv::polylines(img_vis, ppt, npt, 1, 1, CV_RGB(0, 255, 0), 2, 8, 0);
    } else if (structure_result.cell_box[n].size() == 4) {
      cv::Point rook_points[2] = {
          cv::Point(int(structure_result.cell_box[n][0]),
                    int(structure_result.cell_box[n][1])),
          cv::Point(int(structure_result.cell_box[n][2]),
                    int(structure_result.cell_box[n][3]))};
      cv::rectangle(img_vis, rook_points[0], rook_points[1], CV_RGB(0, 255, 0),
                    2, 8, 0);
    }
  }

  cv::imwrite(save_path, img_vis);
  std::cout << "The table visualized image saved in " + save_path << std::endl;
}

// list all files under a directory
void Utility::GetAllFiles(const char *dir_name,
                          std::vector<std::string> &all_inputs) noexcept {
  if (NULL == dir_name) {
    std::cout << " dir_name is null ! " << std::endl;
    return;
  }
  struct stat s;
  stat(dir_name, &s);
  if (!S_ISDIR(s.st_mode)) {
    std::cout << "dir_name is not a valid directory !" << std::endl;
    all_inputs.emplace_back(dir_name);
    return;
  } else {
    struct dirent *filename; // return value for readdir()
    DIR *dir;                // return value for opendir()
    dir = opendir(dir_name);
    if (NULL == dir) {
      std::cout << "Can not open dir " << dir_name << std::endl;
      return;
    }
    std::cout << "Successfully opened the dir !" << std::endl;
    while ((filename = readdir(dir)) != NULL) {
      if (strcmp(filename->d_name, ".") == 0 ||
          strcmp(filename->d_name, "..") == 0)
        continue;
      // img_dir + std::string("/") + all_inputs[0];
      all_inputs.emplace_back(dir_name + std::string("/") +
                              std::string(filename->d_name));
    }
  }
}

cv::Mat
Utility::GetRotateCropImage(const cv::Mat &srcimage,
                            const std::vector<std::vector<int>> &box) noexcept {
  cv::Mat image;
  srcimage.copyTo(image);
  std::vector<std::vector<int>> points = box;

  int x_collect[4] = {box[0][0], box[1][0], box[2][0], box[3][0]};
  int y_collect[4] = {box[0][1], box[1][1], box[2][1], box[3][1]};
  int left = int(*std::min_element(x_collect, x_collect + 4));
  int right = int(*std::max_element(x_collect, x_collect + 4));
  int top = int(*std::min_element(y_collect, y_collect + 4));
  int bottom = int(*std::max_element(y_collect, y_collect + 4));

  cv::Mat img_crop;
  image(cv::Rect(left, top, right - left, bottom - top)).copyTo(img_crop);

  for (size_t i = 0; i < points.size(); ++i) {
    points[i][0] -= left;
    points[i][1] -= top;
  }

  int img_crop_width = int(sqrt(pow(points[0][0] - points[1][0], 2) +
                                pow(points[0][1] - points[1][1], 2)));
  int img_crop_height = int(sqrt(pow(points[0][0] - points[3][0], 2) +
                                 pow(points[0][1] - points[3][1], 2)));

  const cv::Point2f pts_std[4] = {
      {0., 0.},
      {(float)img_crop_width, 0.},
      {(float)img_crop_width, (float)img_crop_height},
      {0.f, (float)img_crop_height}};

  const cv::Point2f pointsf[4] = {{(float)points[0][0], (float)points[0][1]},
                                  {(float)points[1][0], (float)points[1][1]},
                                  {(float)points[2][0], (float)points[2][1]},
                                  {(float)points[3][0], (float)points[3][1]}};

  cv::Mat M = cv::getPerspectiveTransform(pointsf, pts_std);

  cv::Mat dst_img;
  cv::warpPerspective(img_crop, dst_img, M,
                      cv::Size(img_crop_width, img_crop_height),
                      cv::BORDER_REPLICATE);

  if (float(dst_img.rows) >= float(dst_img.cols) * 1.5) {
    cv::Mat srcCopy(dst_img.rows, dst_img.cols, dst_img.depth());
    cv::transpose(dst_img, srcCopy);
    cv::flip(srcCopy, srcCopy, 0);
    return srcCopy;
  } else {
    return dst_img;
  }
}

std::vector<size_t> Utility::argsort(const std::vector<float> &array) noexcept {
  std::vector<size_t> array_index(array.size(), 0);
  for (size_t i = 0; i < array.size(); ++i)
    array_index[i] = i;

  std::sort(array_index.begin(), array_index.end(),
            [&array](size_t pos1, size_t pos2) noexcept {
              return (array[pos1] < array[pos2]);
            });

  return array_index;
}

std::string Utility::basename(const std::string &filename) noexcept {
  if (filename.empty()) {
    return "";
  }

  auto len = filename.length();
  auto index = filename.find_last_of("/\\");

  if (index == std::string::npos) {
    return filename;
  }

  if (index + 1 >= len) {

    --len;
    index = filename.substr(0, len).find_last_of("/\\");

    if (len == 0) {
      return filename;
    }

    if (index == 0) {
      return filename.substr(1, len - 1);
    }

    if (index == std::string::npos) {
      return filename.substr(0, len);
    }

    return filename.substr(index + 1, len - index - 1);
  }

  return filename.substr(index + 1, len - index);
}

bool Utility::PathExists(const char *path) noexcept {
#ifdef _WIN32
  struct _stat buffer;
  return (_stat(path, &buffer) == 0);
#else
  struct stat buffer;
  return (stat(path, &buffer) == 0);
#endif // !_WIN32
}

void Utility::CreateDir(const char *path) noexcept {
#ifdef _MSC_VER
  _mkdir(path);
#elif defined __MINGW32__
  mkdir(path);
#else
  mkdir(path, 0777);
#endif // !_WIN32
}

void Utility::print_result(
    const std::vector<OCRPredictResult> &ocr_result) noexcept {
  for (size_t i = 0; i < ocr_result.size(); ++i) {
    std::cout << i << "\t";
    // det
    const std::vector<std::vector<int>> &boxes = ocr_result[i].box;
    if (boxes.size() > 0) {
      std::cout << "det boxes: [";
      for (size_t n = 0; n < boxes.size(); ++n) {
        std::cout << '[' << boxes[n][0] << ',' << boxes[n][1] << "]";
        if (n != boxes.size() - 1) {
          std::cout << ',';
        }
      }
      std::cout << "] ";
    }
    // rec
    if (ocr_result[i].score != -1.0) {
      std::cout << "rec text: " << ocr_result[i].text
                << " rec score: " << ocr_result[i].score << " ";
    }

    // cls
    if (ocr_result[i].cls_label != -1) {
      std::cout << "cls label: " << ocr_result[i].cls_label
                << " cls score: " << ocr_result[i].cls_score;
    }
    std::cout << std::endl;
  }
}

cv::Mat Utility::crop_image(const cv::Mat &img,
                            const std::vector<int> &box) noexcept {
  cv::Mat crop_im = cv::Mat::zeros(box[3] - box[1], box[2] - box[0], 16);
  int crop_x1 = std::max(0, box[0]);
  int crop_y1 = std::max(0, box[1]);
  int crop_x2 = std::min(img.cols - 1, box[2] - 1);
  int crop_y2 = std::min(img.rows - 1, box[3] - 1);

  cv::Mat crop_im_window =
      crop_im(cv::Range(crop_y1 - box[1], crop_y2 + 1 - box[1]),
              cv::Range(crop_x1 - box[0], crop_x2 + 1 - box[0]));
  cv::Mat roi_img =
      img(cv::Range(crop_y1, crop_y2 + 1), cv::Range(crop_x1, crop_x2 + 1));
  crop_im_window += roi_img;
  return crop_im;
}

cv::Mat Utility::crop_image(const cv::Mat &img,
                            const std::vector<float> &box) noexcept {
  std::vector<int> box_int = {(int)box[0], (int)box[1], (int)box[2],
                              (int)box[3]};
  return crop_image(img, box_int);
}

void Utility::sort_boxes(std::vector<OCRPredictResult> &ocr_result) noexcept {
  std::sort(ocr_result.begin(), ocr_result.end(), Utility::comparison_box);
  if (ocr_result.size() > 1) {
    for (size_t i = 0; i < ocr_result.size() - 1; ++i) {
      for (size_t j = i; j != size_t(-1); --j) {
        if (abs(ocr_result[j + 1].box[0][1] - ocr_result[j].box[0][1]) < 10 &&
            (ocr_result[j + 1].box[0][0] < ocr_result[j].box[0][0])) {
          std::swap(ocr_result[i], ocr_result[i + 1]);
        }
      }
    }
  }
}

std::vector<int>
Utility::xyxyxyxy2xyxy(const std::vector<std::vector<int>> &box) noexcept {
  int x_collect[4] = {box[0][0], box[1][0], box[2][0], box[3][0]};
  int y_collect[4] = {box[0][1], box[1][1], box[2][1], box[3][1]};
  int left = int(*std::min_element(x_collect, x_collect + 4));
  int right = int(*std::max_element(x_collect, x_collect + 4));
  int top = int(*std::min_element(y_collect, y_collect + 4));
  int bottom = int(*std::max_element(y_collect, y_collect + 4));
  return {left, top, right, bottom};
}

std::vector<int> Utility::xyxyxyxy2xyxy(const std::vector<int> &box) noexcept {
  int x_collect[4] = {box[0], box[2], box[4], box[6]};
  int y_collect[4] = {box[1], box[3], box[5], box[7]};
  int left = int(*std::min_element(x_collect, x_collect + 4));
  int right = int(*std::max_element(x_collect, x_collect + 4));
  int top = int(*std::min_element(y_collect, y_collect + 4));
  int bottom = int(*std::max_element(y_collect, y_collect + 4));
  return {left, top, right, bottom};
}

float Utility::fast_exp(float x) noexcept {
  union {
    uint32_t i;
    float f;
  } v{};
  v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
  return v.f;
}

std::vector<float>
Utility::activation_function_softmax(const std::vector<float> &src) noexcept {
  size_t length = src.size();
  std::vector<float> dst;
  dst.resize(length);
  const float alpha = float(*std::max_element(&src[0], &src[length]));
  float denominator{0};

  for (size_t i = 0; i < length; ++i) {
    dst[i] = fast_exp(src[i] - alpha);
    denominator += dst[i];
  }

  for (size_t i = 0; i < length; ++i) {
    dst[i] /= denominator;
  }
  return dst;
}

float Utility::iou(const std::vector<int> &box1,
                   const std::vector<int> &box2) noexcept {
  int area1 = std::max(0, box1[2] - box1[0]) * std::max(0, box1[3] - box1[1]);
  int area2 = std::max(0, box2[2] - box2[0]) * std::max(0, box2[3] - box2[1]);

  // computing the sum_area
  int sum_area = area1 + area2;

  // find the each point of intersect rectangle
  int x1 = std::max(box1[0], box2[0]);
  int y1 = std::max(box1[1], box2[1]);
  int x2 = std::min(box1[2], box2[2]);
  int y2 = std::min(box1[3], box2[3]);

  // judge if there is an intersect
  if (y1 >= y2 || x1 >= x2) {
    return 0.0;
  } else {
    int intersect = (x2 - x1) * (y2 - y1);
    return intersect / (sum_area - intersect + 0.00000001);
  }
}

float Utility::iou(const std::vector<float> &box1,
                   const std::vector<float> &box2) noexcept {
  float area1 = std::max((float)0.0, box1[2] - box1[0]) *
                std::max((float)0.0, box1[3] - box1[1]);
  float area2 = std::max((float)0.0, box2[2] - box2[0]) *
                std::max((float)0.0, box2[3] - box2[1]);

  // computing the sum_area
  float sum_area = area1 + area2;

  // find the each point of intersect rectangle
  float x1 = std::max(box1[0], box2[0]);
  float y1 = std::max(box1[1], box2[1]);
  float x2 = std::min(box1[2], box2[2]);
  float y2 = std::min(box1[3], box2[3]);

  // judge if there is an intersect
  if (y1 >= y2 || x1 >= x2) {
    return 0.0;
  } else {
    float intersect = (x2 - x1) * (y2 - y1);
    return intersect / (sum_area - intersect + 0.00000001);
  }
}

} // namespace PaddleOCR
