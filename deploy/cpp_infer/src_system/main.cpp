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

#include "glog/logging.h"
#include "omp.h"
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <vector>

#include <cstring>
#include <fstream>
#include <numeric>

#include <glog/logging.h>
// #include <include/config.h>
#include <include/ocr_det.h>
#include <include/ocr_rec.h>
// #include <include/utility.h>
#include <sys/stat.h>

#include <gflags/gflags.h>

DEFINE_bool(use_gpu, false, "Infering with GPU or CPU.");
DEFINE_int32(gpu_id, 0, "Device id of GPU to execute.");
DEFINE_int32(gpu_mem, 4000, "GPU id when infering with GPU.");
DEFINE_int32(cpu_math_library_num_threads, 10, "Num of threads with CPU.");
DEFINE_bool(use_mkldnn, false, "Whether use mkldnn with CPU.");

DEFINE_string(image_dir, "", "Dir of input image.");
DEFINE_string(det_model_dir, "", "Path of det inference model.");
DEFINE_int32(max_side_len, 960, "max_side_len of input image.");
DEFINE_double(det_db_thresh, 0.3, "Threshold of det_db_thresh.");
DEFINE_double(det_db_box_thresh, 0.5, "Threshold of det_db_box_thresh.");
DEFINE_double(det_db_unclip_ratio, 1.6, "Threshold of det_db_unclip_ratio.");
DEFINE_bool(use_polygon_score, false, "Whether use polygon score.");
DEFINE_bool(visualize, true, "Whether show the detection results.");

DEFINE_bool(use_angle_cls, false, "Whether use use_angle_cls.");
DEFINE_string(cls_model_dir, "", "Path of cls inference model.");
DEFINE_double(cls_thresh, 0.9, "Threshold of cls_thresh.");

DEFINE_string(rec_model_dir, "", "Path of rec inference model.");
DEFINE_string(char_list_file, "../../ppocr/utils/ppocr_keys_v1.txt", "Path of dictionary.");

DEFINE_bool(use_tensorrt, false, "Whether use tensorrt.");
DEFINE_bool(use_fp16, false, "Whether use fp16 when use tensorrt.");

using namespace std;
using namespace cv;
using namespace PaddleOCR;


static bool PathExists(const std::string& path){
#ifdef _WIN32
  struct _stat buffer;
  return (_stat(path.c_str(), &buffer) == 0);
#else
  struct stat buffer;
  return (stat(path.c_str(), &buffer) == 0);
#endif  // !_WIN32
}


cv::Mat GetRotateCropImage(const cv::Mat &srcimage,
                            std::vector<std::vector<int>> box) {
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

  for (int i = 0; i < points.size(); i++) {
    points[i][0] -= left;
    points[i][1] -= top;
  }

  int img_crop_width = int(sqrt(pow(points[0][0] - points[1][0], 2) +
                                pow(points[0][1] - points[1][1], 2)));
  int img_crop_height = int(sqrt(pow(points[0][0] - points[3][0], 2) +
                                 pow(points[0][1] - points[3][1], 2)));

  cv::Point2f pts_std[4];
  pts_std[0] = cv::Point2f(0., 0.);
  pts_std[1] = cv::Point2f(img_crop_width, 0.);
  pts_std[2] = cv::Point2f(img_crop_width, img_crop_height);
  pts_std[3] = cv::Point2f(0.f, img_crop_height);

  cv::Point2f pointsf[4];
  pointsf[0] = cv::Point2f(points[0][0], points[0][1]);
  pointsf[1] = cv::Point2f(points[1][0], points[1][1]);
  pointsf[2] = cv::Point2f(points[2][0], points[2][1]);
  pointsf[3] = cv::Point2f(points[3][0], points[3][1]);

  cv::Mat M = cv::getPerspectiveTransform(pointsf, pts_std);

  cv::Mat dst_img;
  cv::warpPerspective(img_crop, dst_img, M,
                      cv::Size(img_crop_width, img_crop_height),
                      cv::BORDER_REPLICATE);

  if (float(dst_img.rows) >= float(dst_img.cols) * 1.5) {
    cv::Mat srcCopy = cv::Mat(dst_img.rows, dst_img.cols, dst_img.depth());
    cv::transpose(dst_img, srcCopy);
    cv::flip(srcCopy, srcCopy, 0);
    return srcCopy;
  } else {
    return dst_img;
  }
}


int main(int argc, char **argv) {
  // Parsing command-line
  google::ParseCommandLineFlags(&argc, &argv, true);
  if ((FLAGS_det_model_dir.empty() || FLAGS_rec_model_dir.empty() || FLAGS_image_dir.empty()) ||
     (FLAGS_use_angle_cls && FLAGS_cls_model_dir.empty())) {
    std::cout << "Usage[default]: ./ocr_system --det_model_dir=/PATH/TO/DET_INFERENCE_MODEL/ "
                << "--rec_model_dir=/PATH/TO/REC_INFERENCE_MODEL/ "
                << "--image_dir=/PATH/TO/INPUT/IMAGE/" << std::endl;
    std::cout << "Usage[use angle cls]: ./ocr_system --det_model_dir=/PATH/TO/DET_INFERENCE_MODEL/ "
                << "--use_angle_cls=true "
                << "--cls_model_dir=/PATH/TO/CLS_INFERENCE_MODEL/ "
                << "--rec_model_dir=/PATH/TO/REC_INFERENCE_MODEL/ "
                << "--image_dir=/PATH/TO/INPUT/IMAGE/" << std::endl;
    return -1;
  }

  if (!PathExists(FLAGS_image_dir)) {
      std::cerr << "[ERROR] image path not exist! image_dir: " << FLAGS_image_dir << endl;
      exit(1);      
  }
  std::vector<cv::String> cv_all_img_names;
  cv::glob(FLAGS_image_dir, cv_all_img_names);
  std::cout << "total images num: " << cv_all_img_names.size() << endl;

  DBDetector det(FLAGS_det_model_dir, FLAGS_use_gpu, FLAGS_gpu_id,
                 FLAGS_gpu_mem, FLAGS_cpu_math_library_num_threads, 
                 FLAGS_use_mkldnn, FLAGS_max_side_len, FLAGS_det_db_thresh,
                 FLAGS_det_db_box_thresh, FLAGS_det_db_unclip_ratio,
                 FLAGS_use_polygon_score, FLAGS_visualize,
                 FLAGS_use_tensorrt, FLAGS_use_fp16);

  Classifier *cls = nullptr;
  if (FLAGS_use_angle_cls) {
    cls = new Classifier(FLAGS_cls_model_dir, FLAGS_use_gpu, FLAGS_gpu_id,
                         FLAGS_gpu_mem, FLAGS_cpu_math_library_num_threads,
                         FLAGS_use_mkldnn, FLAGS_cls_thresh,
                         FLAGS_use_tensorrt, FLAGS_use_fp16);
  }

  CRNNRecognizer rec(FLAGS_rec_model_dir, FLAGS_use_gpu, FLAGS_gpu_id,
                     FLAGS_gpu_mem, FLAGS_cpu_math_library_num_threads,
                     FLAGS_use_mkldnn, FLAGS_char_list_file,
                     FLAGS_use_tensorrt, FLAGS_use_fp16);

  auto start = std::chrono::system_clock::now();

  for (int i = 0; i < cv_all_img_names.size(); ++i) {
    LOG(INFO) << "The predict img: " << cv_all_img_names[i];

    cv::Mat srcimg = cv::imread(FLAGS_image_dir, cv::IMREAD_COLOR);
    if (!srcimg.data) {
      std::cerr << "[ERROR] image read failed! image path: " << cv_all_img_names[i] << endl;
      exit(1);
    }
    std::vector<std::vector<std::vector<int>>> boxes;

    det.Run(srcimg, boxes);
  
    cv::Mat crop_img;
    for (int j = 0; j < boxes.size(); j++) {
      crop_img = GetRotateCropImage(srcimg, boxes[j]);

      if (cls != nullptr) {
        crop_img = cls->Run(crop_img);
      }
      rec.Run(crop_img);
    }
      
    auto end = std::chrono::system_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Cost  "
              << double(duration.count()) *
                     std::chrono::microseconds::period::num /
                     std::chrono::microseconds::period::den
              << "s" << std::endl;
  }

  return 0;
}
