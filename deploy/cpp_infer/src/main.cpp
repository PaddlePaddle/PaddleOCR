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
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <vector>

#include <include/args.h>
#include <include/paddleocr.h>
#include <include/paddlestructure.h>

using namespace PaddleOCR;

void check_params() {
  if (FLAGS_det) {
    if (FLAGS_det_model_dir.empty() || FLAGS_image_dir.empty()) {
      std::cout << "Usage[det]: ./ppocr "
                   "--det_model_dir=/PATH/TO/DET_INFERENCE_MODEL/ "
                << "--image_dir=/PATH/TO/INPUT/IMAGE/" << std::endl;
      exit(1);
    }
  }
  if (FLAGS_rec) {
    std::cout
        << "In PP-OCRv3, rec_image_shape parameter defaults to '3, 48, 320',"
           "if you are using recognition model with PP-OCRv2 or an older "
           "version, "
           "please set --rec_image_shape='3,32,320"
        << std::endl;
    if (FLAGS_rec_model_dir.empty() || FLAGS_image_dir.empty()) {
      std::cout << "Usage[rec]: ./ppocr "
                   "--rec_model_dir=/PATH/TO/REC_INFERENCE_MODEL/ "
                << "--image_dir=/PATH/TO/INPUT/IMAGE/" << std::endl;
      exit(1);
    }
  }
  if (FLAGS_cls && FLAGS_use_angle_cls) {
    if (FLAGS_cls_model_dir.empty() || FLAGS_image_dir.empty()) {
      std::cout << "Usage[cls]: ./ppocr "
                << "--cls_model_dir=/PATH/TO/REC_INFERENCE_MODEL/ "
                << "--image_dir=/PATH/TO/INPUT/IMAGE/" << std::endl;
      exit(1);
    }
  }
  if (FLAGS_table) {
    if (FLAGS_table_model_dir.empty() || FLAGS_det_model_dir.empty() ||
        FLAGS_rec_model_dir.empty() || FLAGS_image_dir.empty()) {
      std::cout << "Usage[table]: ./ppocr "
                << "--det_model_dir=/PATH/TO/DET_INFERENCE_MODEL/ "
                << "--rec_model_dir=/PATH/TO/REC_INFERENCE_MODEL/ "
                << "--table_model_dir=/PATH/TO/TABLE_INFERENCE_MODEL/ "
                << "--image_dir=/PATH/TO/INPUT/IMAGE/" << std::endl;
      exit(1);
    }
  }
  if (FLAGS_precision != "fp32" && FLAGS_precision != "fp16" &&
      FLAGS_precision != "int8") {
    cout << "precison should be 'fp32'(default), 'fp16' or 'int8'. " << endl;
    exit(1);
  }
}

void ocr(std::vector<cv::String> &cv_all_img_names) {
  PPOCR ocr = PPOCR();

  std::vector<std::vector<OCRPredictResult>> ocr_results =
      ocr.ocr(cv_all_img_names, FLAGS_det, FLAGS_rec, FLAGS_cls);

  for (int i = 0; i < cv_all_img_names.size(); ++i) {
    if (FLAGS_benchmark) {
      cout << cv_all_img_names[i] << '\t';
      if (FLAGS_rec && FLAGS_det) {
        Utility::print_result(ocr_results[i]);
      } else if (FLAGS_det) {
        for (int n = 0; n < ocr_results[i].size(); n++) {
          for (int m = 0; m < ocr_results[i][n].box.size(); m++) {
            cout << ocr_results[i][n].box[m][0] << ' '
                 << ocr_results[i][n].box[m][1] << ' ';
          }
        }
        cout << endl;
      } else {
        Utility::print_result(ocr_results[i]);
      }
    } else {
      cout << cv_all_img_names[i] << "\n";
      Utility::print_result(ocr_results[i]);
      if (FLAGS_visualize && FLAGS_det) {
        cv::Mat srcimg = cv::imread(cv_all_img_names[i], cv::IMREAD_COLOR);
        if (!srcimg.data) {
          std::cerr << "[ERROR] image read failed! image path: "
                    << cv_all_img_names[i] << endl;
          exit(1);
        }
        std::string file_name = Utility::basename(cv_all_img_names[i]);

        Utility::VisualizeBboxes(srcimg, ocr_results[i],
                                 FLAGS_output + "/" + file_name);
      }
      cout << "***************************" << endl;
    }
  }
}

void structure(std::vector<cv::String> &cv_all_img_names) {
  PaddleOCR::PaddleStructure engine = PaddleOCR::PaddleStructure();
  std::vector<std::vector<StructurePredictResult>> structure_results =
      engine.structure(cv_all_img_names, false, FLAGS_table);
  for (int i = 0; i < cv_all_img_names.size(); i++) {
    cout << "predict img: " << cv_all_img_names[i] << endl;
    cv::Mat srcimg = cv::imread(cv_all_img_names[i], cv::IMREAD_COLOR);
    for (int j = 0; j < structure_results[i].size(); j++) {
      std::cout << j << "\ttype: " << structure_results[i][j].type
                << ", region: [";
      std::cout << structure_results[i][j].box[0] << ","
                << structure_results[i][j].box[1] << ","
                << structure_results[i][j].box[2] << ","
                << structure_results[i][j].box[3] << "], res: ";
      if (structure_results[i][j].type == "table") {
        std::cout << structure_results[i][j].html << std::endl;
        std::string file_name = Utility::basename(cv_all_img_names[i]);

        Utility::VisualizeBboxes(srcimg, structure_results[i][j],
                                 FLAGS_output + "/" + std::to_string(j) + "_" +
                                     file_name);
      } else {
        Utility::print_result(structure_results[i][j].text_res);
      }
    }
  }
}

int main(int argc, char **argv) {
  // Parsing command-line
  google::ParseCommandLineFlags(&argc, &argv, true);
  check_params();

  if (!Utility::PathExists(FLAGS_image_dir)) {
    std::cerr << "[ERROR] image path not exist! image_dir: " << FLAGS_image_dir
              << endl;
    exit(1);
  }

  std::vector<cv::String> cv_all_img_names;
  cv::glob(FLAGS_image_dir, cv_all_img_names);
  std::cout << "total images num: " << cv_all_img_names.size() << endl;

  if (FLAGS_type == "ocr") {
    ocr(cv_all_img_names);
  } else if (FLAGS_type == "structure") {
    structure(cv_all_img_names);
  } else {
    std::cout << "only value in ['ocr','structure'] is supported" << endl;
  }
}
