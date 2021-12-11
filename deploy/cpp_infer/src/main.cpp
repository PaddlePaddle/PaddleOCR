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

#ifndef OCR_EXPORTS

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
#include <include/ocr_cls.h>
#include <include/ocr_rec.h>
#include <include/utility.h>
#include <sys/stat.h>

#include "auto_log/autolog.h"
#include "include/ocr_defines.h"

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


int main_det(std::vector<cv::String> cv_all_img_names) {
    std::vector<double> time_info = {0, 0, 0};
    DBDetector det(FLAGS_det_model_dir, FLAGS_use_gpu, FLAGS_gpu_id,
                   FLAGS_gpu_mem, FLAGS_cpu_threads, 
                   FLAGS_enable_mkldnn, FLAGS_max_side_len, FLAGS_det_db_thresh,
                   FLAGS_det_db_box_thresh, FLAGS_det_db_unclip_ratio,
                   FLAGS_use_polygon_score, FLAGS_visualize,
                   FLAGS_use_tensorrt, FLAGS_precision);
    
    for (int i = 0; i < cv_all_img_names.size(); ++i) {
      LOG(INFO) << "The predict img: " << cv_all_img_names[i];

      cv::Mat srcimg = cv::imread(cv_all_img_names[i], cv::IMREAD_COLOR);
      if (!srcimg.data) {
        std::cerr << "[ERROR] image read failed! image path: " << cv_all_img_names[i] << endl;
        exit(1);
      }
      std::vector<std::vector<std::vector<int>>> boxes;
      std::vector<double> det_times;

      det.Run(srcimg, boxes, &det_times);
  
      time_info[0] += det_times[0];
      time_info[1] += det_times[1];
      time_info[2] += det_times[2];
    }
    
    if (FLAGS_benchmark) {
        AutoLogger autolog("ocr_det", 
                           FLAGS_use_gpu,
                           FLAGS_use_tensorrt,
                           FLAGS_enable_mkldnn,
                           FLAGS_cpu_threads,
                           1, 
                           "dynamic", 
                           FLAGS_precision, 
                           time_info, 
                           cv_all_img_names.size());
        autolog.report();
    }
    return 0;
}


int main_rec(std::vector<cv::String> cv_all_img_names) {
    std::vector<double> time_info = {0, 0, 0};
    CRNNRecognizer rec(FLAGS_rec_model_dir, FLAGS_use_gpu, FLAGS_gpu_id,
                       FLAGS_gpu_mem, FLAGS_cpu_threads,
                       FLAGS_enable_mkldnn, FLAGS_char_list_file,
                       FLAGS_use_tensorrt, FLAGS_precision);

    for (int i = 0; i < cv_all_img_names.size(); ++i) {
      LOG(INFO) << "The predict img: " << cv_all_img_names[i];

      cv::Mat srcimg = cv::imread(cv_all_img_names[i], cv::IMREAD_COLOR);
      if (!srcimg.data) {
        std::cerr << "[ERROR] image read failed! image path: " << cv_all_img_names[i] << endl;
        exit(1);
      }
      std::vector<string> strs_res;
      std::vector<float> scores;
      std::vector<double> rec_times;

      rec.Run(srcimg,strs_res, scores, &rec_times);
        
      time_info[0] += rec_times[0];
      time_info[1] += rec_times[1];
      time_info[2] += rec_times[2];
    }
    
    return 0;
}


int main_system(std::vector<cv::String> cv_all_img_names) {
    DBDetector det(FLAGS_det_model_dir, FLAGS_use_gpu, FLAGS_gpu_id,
                   FLAGS_gpu_mem, FLAGS_cpu_threads, 
                   FLAGS_enable_mkldnn, FLAGS_max_side_len, FLAGS_det_db_thresh,
                   FLAGS_det_db_box_thresh, FLAGS_det_db_unclip_ratio,
                   FLAGS_use_polygon_score, FLAGS_visualize,
                   FLAGS_use_tensorrt, FLAGS_precision);

    Classifier *cls = nullptr;
    if (FLAGS_use_angle_cls) {
      cls = new Classifier(FLAGS_cls_model_dir, FLAGS_use_gpu, FLAGS_gpu_id,
                           FLAGS_gpu_mem, FLAGS_cpu_threads,
                           FLAGS_enable_mkldnn, FLAGS_cls_thresh,
                           FLAGS_use_tensorrt, FLAGS_precision);
    }

    CRNNRecognizer rec(FLAGS_rec_model_dir, FLAGS_use_gpu, FLAGS_gpu_id,
                       FLAGS_gpu_mem, FLAGS_cpu_threads,
                       FLAGS_enable_mkldnn, FLAGS_char_list_file,
                       FLAGS_use_tensorrt, FLAGS_precision);

    auto start = std::chrono::system_clock::now();

    for (int i = 0; i < cv_all_img_names.size(); ++i) {
      LOG(INFO) << "The predict img: " << cv_all_img_names[i];

      cv::Mat srcimg = cv::imread(cv_all_img_names[i], cv::IMREAD_COLOR);
      if (!srcimg.data) {
        std::cerr << "[ERROR] image read failed! image path: " << cv_all_img_names[i] << endl;
        exit(1);
      }
      std::vector<std::vector<std::vector<int>>> boxes;
      std::vector<double> det_times;
      std::vector<double> rec_times;
        
      det.Run(srcimg, boxes, &det_times);
    
      cv::Mat crop_img;
      for (int j = 0; j < boxes.size(); j++) {
        crop_img = Utility::GetRotateCropImage(srcimg, boxes[j]);

        if (cls != nullptr) {
          crop_img = cls->Run(crop_img);
        }
        std::vector<string> strs_res;
        std::vector<float> scores;
        rec.Run(crop_img, strs_res, scores, &rec_times);
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


void check_params(char* mode) {
    if (strcmp(mode, "det")==0) {
        if (FLAGS_det_model_dir.empty() || FLAGS_image_dir.empty()) {
            std::cout << "Usage[det]: ./ppocr --det_model_dir=/PATH/TO/DET_INFERENCE_MODEL/ "
                      << "--image_dir=/PATH/TO/INPUT/IMAGE/" << std::endl;      
            exit(1);      
        }
    }
    if (strcmp(mode, "rec")==0) {
        if (FLAGS_rec_model_dir.empty() || FLAGS_image_dir.empty()) {
            std::cout << "Usage[rec]: ./ppocr --rec_model_dir=/PATH/TO/REC_INFERENCE_MODEL/ "
                      << "--image_dir=/PATH/TO/INPUT/IMAGE/" << std::endl;      
            exit(1);
        }
    }
    if (strcmp(mode, "system")==0) {
        if ((FLAGS_det_model_dir.empty() || FLAGS_rec_model_dir.empty() || FLAGS_image_dir.empty()) ||
           (FLAGS_use_angle_cls && FLAGS_cls_model_dir.empty())) {
            std::cout << "Usage[system without angle cls]: ./ppocr --det_model_dir=/PATH/TO/DET_INFERENCE_MODEL/ "
                        << "--rec_model_dir=/PATH/TO/REC_INFERENCE_MODEL/ "
                        << "--image_dir=/PATH/TO/INPUT/IMAGE/" << std::endl;
            std::cout << "Usage[system with angle cls]: ./ppocr --det_model_dir=/PATH/TO/DET_INFERENCE_MODEL/ "
                        << "--use_angle_cls=true "
                        << "--cls_model_dir=/PATH/TO/CLS_INFERENCE_MODEL/ "
                        << "--rec_model_dir=/PATH/TO/REC_INFERENCE_MODEL/ "
                        << "--image_dir=/PATH/TO/INPUT/IMAGE/" << std::endl;
            exit(1);      
        }
    }
    if (FLAGS_precision != "fp32" && FLAGS_precision != "fp16" && FLAGS_precision != "int8") {
        cout << "precison should be 'fp32'(default), 'fp16' or 'int8'. " << endl;
        exit(1);
    }
}


int main(int argc, char **argv) {
    if (argc<=1 || (strcmp(argv[1], "det")!=0 && strcmp(argv[1], "rec")!=0 && strcmp(argv[1], "system")!=0)) {
        std::cout << "Please choose one mode of [det, rec, system] !" << std::endl;
        return -1;
    }
    std::cout << "mode: " << argv[1] << endl;

    // Parsing command-line
    google::ParseCommandLineFlags(&argc, &argv, true);
    check_params(argv[1]);
        
    if (!PathExists(FLAGS_image_dir)) {
        std::cerr << "[ERROR] image path not exist! image_dir: " << FLAGS_image_dir << endl;
        exit(1);      
    }
    
    std::vector<cv::String> cv_all_img_names;
    cv::glob(FLAGS_image_dir, cv_all_img_names);
    std::cout << "total images num: " << cv_all_img_names.size() << endl;
    
    if (strcmp(argv[1], "det")==0) {
        return main_det(cv_all_img_names);
    }
    if (strcmp(argv[1], "rec")==0) {
        return main_rec(cv_all_img_names);
    }    
    if (strcmp(argv[1], "system")==0) {
        return main_system(cv_all_img_names);
    } 

}

#endif // ! OCR_EXPORTS
