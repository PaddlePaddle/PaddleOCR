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

#include <include/config.h>
#include <include/ocr_det.h>
#include <include/ocr_rec.h>

using namespace std;
using namespace cv;
using namespace PaddleOCR;

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "[ERROR] usage: " << argv[0]
              << " configure_filepath image_path\n";
    exit(1);
  }

  Config config(argv[1]);

  config.PrintConfigInfo();

  std::string img_path(argv[2]);

  cv::Mat srcimg = cv::imread(img_path, cv::IMREAD_COLOR);

  DBDetector det(
      config.det_model_dir, config.use_gpu, config.gpu_id, config.gpu_mem,
      config.cpu_math_library_num_threads, config.use_mkldnn,
      config.use_zero_copy_run, config.max_side_len, config.det_db_thresh,
      config.det_db_box_thresh, config.det_db_unclip_ratio, config.visualize);

  Classifier *cls = nullptr;
  if (config.use_angle_cls == true) {
    cls = new Classifier(config.cls_model_dir, config.use_gpu, config.gpu_id,
                         config.gpu_mem, config.cpu_math_library_num_threads,
                         config.use_mkldnn, config.use_zero_copy_run,
                         config.cls_thresh);
  }

  CRNNRecognizer rec(config.rec_model_dir, config.use_gpu, config.gpu_id,
                     config.gpu_mem, config.cpu_math_library_num_threads,
                     config.use_mkldnn, config.use_zero_copy_run,
                     config.char_list_file);

#ifdef USE_MKL
#pragma omp parallel
  for (auto i = 0; i < 10; i++) {
    LOG_IF(WARNING,
           config.cpu_math_library_num_threads != omp_get_num_threads())
        << "WARNING! MKL is running on " << omp_get_num_threads()
        << " threads while cpu_math_library_num_threads is set to "
        << config.cpu_math_library_num_threads
        << ". Possible reason could be 1. You have set omp_set_num_threads() "
           "somewhere; 2. MKL is not linked properly";
  }
#endif

  auto start = std::chrono::system_clock::now();
  std::vector<std::vector<std::vector<int>>> boxes;
  det.Run(srcimg, boxes);

  rec.Run(boxes, srcimg, cls);
  auto end = std::chrono::system_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cout << "花费了"
            << double(duration.count()) *
                   std::chrono::microseconds::period::num /
                   std::chrono::microseconds::period::den
            << "秒" << std::endl;

  return 0;
}
