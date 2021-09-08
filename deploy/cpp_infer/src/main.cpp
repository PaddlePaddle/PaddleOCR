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
#include <include/ocr_cls.h>
#include <include/ocr_rec.h>
#include <include/utility.h>
#include <sys/stat.h>

#include <gflags/gflags.h>
#include "auto_log/autolog.h"

DEFINE_bool(use_gpu, false, "Infering with GPU or CPU.");
DEFINE_int32(gpu_id, 0, "Device id of GPU to execute.");
DEFINE_int32(gpu_mem, 4000, "GPU id when infering with GPU.");
DEFINE_int32(cpu_threads, 10, "Num of threads with CPU.");
DEFINE_bool(enable_mkldnn, false, "Whether use mkldnn with CPU.");
DEFINE_bool(use_tensorrt, false, "Whether use tensorrt.");
DEFINE_string(precision, "fp32", "Precision be one of fp32/fp16/int8");
DEFINE_bool(benchmark, true, "Whether use benchmark.");
DEFINE_string(save_log_path, "./log_output/", "Save benchmark log path.");
// detection related
DEFINE_string(image_dir, "", "Dir of input image.");
DEFINE_string(det_model_dir, "", "Path of det inference model.");
DEFINE_int32(max_side_len, 960, "max_side_len of input image.");
DEFINE_double(det_db_thresh, 0.3, "Threshold of det_db_thresh.");
DEFINE_double(det_db_box_thresh, 0.5, "Threshold of det_db_box_thresh.");
DEFINE_double(det_db_unclip_ratio, 1.6, "Threshold of det_db_unclip_ratio.");
DEFINE_bool(use_polygon_score, false, "Whether use polygon score.");
DEFINE_bool(visualize, true, "Whether show the detection results.");
// classification related
DEFINE_bool(use_angle_cls, false, "Whether use use_angle_cls.");
DEFINE_string(cls_model_dir, "", "Path of cls inference model.");
DEFINE_double(cls_thresh, 0.9, "Threshold of cls_thresh.");
// recognition related
DEFINE_string(rec_model_dir, "", "Path of rec inference model.");
DEFINE_int32(rec_batch_num, 1, "rec_batch_num.");
DEFINE_string(char_list_file, "../../ppocr/utils/ppocr_keys_v1.txt", "Path of dictionary.");
// DEFINE_string(char_list_file, "./ppocr/utils/ppocr_keys_v1.txt", "Path of dictionary.");


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

      std::vector<double> rec_times;
      rec.Run(srcimg, &rec_times);
        
      time_info[0] += rec_times[0];
      time_info[1] += rec_times[1];
      time_info[2] += rec_times[2];
    }
        
    if (FLAGS_benchmark) {
        AutoLogger autolog("ocr_rec", 
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


int main_system(std::vector<cv::String> cv_all_img_names) {
    std::vector<double> time_info_det = {0, 0, 0};
    std::vector<double> time_info_rec = {0, 0, 0};

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
      time_info_det[0] += det_times[0];
      time_info_det[1] += det_times[1];
      time_info_det[2] += det_times[2];
        
      cv::Mat crop_img;
      for (int j = 0; j < boxes.size(); j++) {
        crop_img = Utility::GetRotateCropImage(srcimg, boxes[j]);

        if (cls != nullptr) {
          crop_img = cls->Run(crop_img);
        }
        rec.Run(crop_img, &rec_times);
        time_info_rec[0] += rec_times[0];
        time_info_rec[1] += rec_times[1];
        time_info_rec[2] += rec_times[2];
      }
    }
    if (FLAGS_benchmark) {
        AutoLogger autolog_det("ocr_det", 
                            FLAGS_use_gpu,
                            FLAGS_use_tensorrt,
                            FLAGS_enable_mkldnn,
                            FLAGS_cpu_threads,
                            1, 
                            "dynamic", 
                            FLAGS_precision, 
                            time_info_det, 
                            cv_all_img_names.size());
        AutoLogger autolog_rec("ocr_rec", 
                            FLAGS_use_gpu,
                            FLAGS_use_tensorrt,
                            FLAGS_enable_mkldnn,
                            FLAGS_cpu_threads,
                            1, 
                            "dynamic", 
                            FLAGS_precision, 
                            time_info_rec, 
                            cv_all_img_names.size());
        autolog_det.report();
        std::cout << endl;
        autolog_rec.report();
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
