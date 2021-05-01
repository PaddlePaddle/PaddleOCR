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
#include <include/config.h>
#include <include/ocr_det.h>
#include <include/ocr_rec.h>
#include <include/utility.h>
#include <numeric>
#include <sys/stat.h>

#include <glog/logging.h>

using namespace std;
using namespace cv;
using namespace PaddleOCR;

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "[ERROR] usage: " << argv[0]
              << " configure_filepath image_path\n";
    exit(1);
  }
  if (argc > 3) {
    std::cout << "[-H] uasge: " << argv[0]
              << " configure_filepath image_path use_gpu gpu_id  "
                 "cpu_math_library_num_threads  use_mkldnn  use_tensorrt  "
                 "use_fp16"
              << std::endl;
  }

  OCRConfig config(argv[1], argc, argv);

  config.PrintConfigInfo();

  std::string img_path(argv[2]);
  std::vector<std::string> all_img_names;

  Utility::GetAllFiles((char *)img_path.c_str(), all_img_names);

  DBDetector det(config.det_model_dir, config.use_gpu, config.gpu_id,
                 config.gpu_mem, config.cpu_math_library_num_threads,
                 config.use_mkldnn, config.max_side_len, config.det_db_thresh,
                 config.det_db_box_thresh, config.det_db_unclip_ratio,
                 config.visualize, config.use_tensorrt, config.use_fp16);

  Classifier *cls = nullptr;
  if (config.use_angle_cls == true) {
    cls = new Classifier(config.cls_model_dir, config.use_gpu, config.gpu_id,
                         config.gpu_mem, config.cpu_math_library_num_threads,
                         config.use_mkldnn, config.cls_thresh,
                         config.use_tensorrt, config.use_fp16);
  }

  CRNNRecognizer rec(config.rec_model_dir, config.use_gpu, config.gpu_id,
                     config.gpu_mem, config.cpu_math_library_num_threads,
                     config.use_mkldnn, config.char_list_file,
                     config.use_tensorrt, config.use_fp16);

  std::vector<double> det_t = {0, 0, 0};
  std::vector<double> rec_t = {0, 0, 0};
  int rec_img_num = 0;
  Timer timer;
  std::vector<Timer> warmpupTimer = {timer, timer, timer};
  // warmup 20 times
  for (int warmup = 0; warmup <= 20; warmup++) {
    cv::Mat srcimg = cv::imread(all_img_names[0], cv::IMREAD_COLOR);

    std::vector<std::vector<std::vector<int>>> boxes;
    det.Run(srcimg, boxes, warmpupTimer);
    rec.Run(boxes, srcimg, cls, warmpupTimer);
  }

  // start to run
  std::vector<Timer> detTimer = {timer, timer, timer};
  std::vector<Timer> recTimer = {timer, timer, timer};
  for (auto img_dir : all_img_names) {
    LOG(INFO) << "The predict img: " << img_dir;
    cv::Mat srcimg = cv::imread(img_dir, cv::IMREAD_COLOR);

    std::vector<std::vector<std::vector<int>>> boxes;
    det.Run(srcimg, boxes, detTimer);

    rec.Run(boxes, srcimg, cls, recTimer);

    rec_img_num += boxes.size();
  }

  LOG(INFO) << "----------------------- Cong info -----------------------";
  LOG(INFO) << "runtime_device: " << (config.use_gpu ? "gpu" : "cpu");
  LOG(INFO) << "ir_optim: "
            << "True";
  LOG(INFO) << "enable_memory_optim: "
            << "True";
  LOG(INFO) << "enable_tensorrt: " << config.use_tensorrt;
  LOG(INFO) << "precision: " << (config.use_fp16 ? "fp16" : "fp32");
  LOG(INFO) << "enable_mkldnn: " << config.use_mkldnn;
  LOG(INFO) << "cpu_math_library_num_threads: "
            << config.cpu_math_library_num_threads;
  LOG(INFO) << "----------------------- Data info -----------------------";
  LOG(INFO) << "batch_size: " << 1;
  LOG(INFO) << "input_shape: "
            << "dynamic shape";
  LOG(INFO) << "\n";
  LOG(INFO) << "----------------------- Det Model info -----------------------";
  LOG(INFO) << "[Det] model_name: " << config.det_model_dir;

  LOG(INFO) << "----------------------- Det Perf info -----------------------"
            << std::endl;
  LOG(INFO) << "[Det] total number of predicted data: " << all_img_names.size()
            << " and total time spent(s): "
            << (detTimer[0].report() + detTimer[1].report() +
                detTimer[2].report()) /
                   1000
            << std::endl;
  LOG(INFO) << "[Det] preprocess_time(ms): "
            << detTimer[0].report() / all_img_names.size()
            << ", inference_time(ms): "
            << detTimer[1].report() / all_img_names.size()
            << ", postprocess_time(ms): "
            << detTimer[2].report() / all_img_names.size() << std::endl;
  LOG(INFO) << "\n" << std::endl;
  LOG(INFO) << "----------------------- Rec Model info -----------------------"
            << std::endl;
  LOG(INFO) << "[Rec] model_name: " << config.rec_model_dir;
  LOG(INFO) << "----------------------- Rec Perf info -----------------------"
            << std::endl;
  LOG(INFO) << "[Rec] total number of predicted data: " << rec_img_num
            << " and total time spent(s): "
            << (recTimer[0].report() + recTimer[1].report() +
                recTimer[2].report()) /
                   1000
            << std::endl;
  LOG(INFO) << "[Rec] preprocess_time(ms): "
            << recTimer[0].report() / rec_img_num
            << ", inference_time(ms): " << recTimer[1].report() / rec_img_num
            << ", postprocess_time(ms): " << recTimer[2].report() / rec_img_num
            << std::endl;

  return 0;
}
