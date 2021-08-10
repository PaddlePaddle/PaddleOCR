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
#include <include/ocr_det.h>
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


int main(int argc, char **argv) {
  // Parsing command-line
  google::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_det_model_dir.empty() || FLAGS_image_dir.empty()) {
    std::cout << "Usage: ./ocr_det --det_model_dir=/PATH/TO/INFERENCE_MODEL/ "
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

  auto start = std::chrono::system_clock::now();

  for (int i = 0; i < cv_all_img_names.size(); ++i) {
    LOG(INFO) << "The predict img: " << cv_all_img_names[i];

    cv::Mat srcimg = cv::imread(cv_all_img_names[i], cv::IMREAD_COLOR);
    if (!srcimg.data) {
      std::cerr << "[ERROR] image read failed! image path: " << cv_all_img_names[i] << endl;
      exit(1);
    }
    std::vector<std::vector<std::vector<int>>> boxes;

    det.Run(srcimg, boxes);

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
