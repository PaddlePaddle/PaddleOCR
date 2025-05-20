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

#pragma once

#include <fstream>
#include <include/preprocess_op.h>
#include <include/utility.h>
#include <iostream>
#include <memory>
#include <yaml-cpp/yaml.h>

namespace paddle_infer {
class Predictor;
}

namespace PaddleOCR {

class CRNNRecognizer {
public:
  explicit CRNNRecognizer(const std::string &model_dir, const bool &use_gpu,
                          const int &gpu_id, const int &gpu_mem,
                          const int &cpu_math_library_num_threads,
                          const bool &use_mkldnn, const std::string &label_path,
                          const bool &use_tensorrt,
                          const std::string &precision,
                          const int &rec_batch_num, const int &rec_img_h,
                          const int &rec_img_w) noexcept {
    this->use_gpu_ = use_gpu;
    this->gpu_id_ = gpu_id;
    this->gpu_mem_ = gpu_mem;
    this->cpu_math_library_num_threads_ = cpu_math_library_num_threads;
    this->use_mkldnn_ = use_mkldnn;
    this->use_tensorrt_ = use_tensorrt;
    this->precision_ = precision;
    this->rec_batch_num_ = rec_batch_num;
    this->rec_img_h_ = rec_img_h;
    this->rec_img_w_ = rec_img_w;
    std::vector<int> rec_image_shape = {3, rec_img_h, rec_img_w};
    this->rec_image_shape_ = rec_image_shape;

    std::string new_label_path = label_path;
    std::string yaml_file_path = model_dir + "/inference.yml";
    std::ifstream yaml_file(yaml_file_path);
    if (yaml_file.is_open()) {
      std::string model_name;
      std::vector<std::string> rec_char_list;
      try {
        YAML::Node config = YAML::LoadFile(yaml_file_path);
        if (config["Global"] && config["Global"]["model_name"]) {
          model_name = config["Global"]["model_name"].as<std::string>();
        }
        if (!model_name.empty()) {
          std::cerr << "Error: " << model_name << " is currently not supported."
                    << std::endl;
          std::exit(EXIT_FAILURE);
        }
        if (config["PostProcess"] && config["PostProcess"]["character_dict"]) {
          rec_char_list = config["PostProcess"]["character_dict"]
                              .as<std::vector<std::string>>();
        }
      } catch (const YAML::Exception &e) {
        std::cerr << "Failed to load YAML file: " << e.what() << std::endl;
      }
      if (label_path == "../../ppocr/utils/ppocr_keys_v1.txt" &&
          !rec_char_list.empty()) {
        std::string new_rec_char_dict_path = model_dir + "/ppocr_keys.txt";
        std::ofstream new_file(new_rec_char_dict_path);
        if (new_file.is_open()) {
          for (const auto &character : rec_char_list) {
            new_file << character << '\n';
          }
          new_label_path = new_rec_char_dict_path;
        }
      }
    }

    this->label_list_ = Utility::ReadDict(new_label_path);
    this->label_list_.emplace(this->label_list_.begin(),
                              "#"); // blank char for ctc
    this->label_list_.emplace_back(" ");

    LoadModel(model_dir);
  }

  // Load Paddle inference model
  void LoadModel(const std::string &model_dir) noexcept;

  void Run(const std::vector<cv::Mat> &img_list,
           std::vector<std::string> &rec_texts,
           std::vector<float> &rec_text_scores,
           std::vector<double> &times) noexcept;

private:
  std::shared_ptr<paddle_infer::Predictor> predictor_;

  bool use_gpu_ = false;
  int gpu_id_ = 0;
  int gpu_mem_ = 4000;
  int cpu_math_library_num_threads_ = 4;
  bool use_mkldnn_ = false;

  std::vector<std::string> label_list_;

  std::vector<float> mean_ = {0.5f, 0.5f, 0.5f};
  std::vector<float> scale_ = {1 / 0.5f, 1 / 0.5f, 1 / 0.5f};
  bool is_scale_ = true;
  bool use_tensorrt_ = false;
  std::string precision_ = "fp32";
  int rec_batch_num_ = 6;
  int rec_img_h_ = 32;
  int rec_img_w_ = 320;
  std::vector<int> rec_image_shape_ = {3, rec_img_h_, rec_img_w_};
  // pre-process
  CrnnResizeImg resize_op_;
  Normalize normalize_op_;
  PermuteBatch permute_op_;

}; // class CrnnRecognizer

} // namespace PaddleOCR
