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

#include <iomanip>
#include <iostream>
#include <map>
#include <ostream>
#include <string>
#include <vector>

#include "include/utility.h"

namespace PaddleOCR {

class Config {
public:
  explicit Config(const std::string &config_file) {
    config_map_ = LoadConfig(config_file);

    this->use_gpu = bool(stoi(config_map_["use_gpu"]));

    this->gpu_id = stoi(config_map_["gpu_id"]);

    this->gpu_mem = stoi(config_map_["gpu_mem"]);

    this->cpu_math_library_num_threads =
        stoi(config_map_["cpu_math_library_num_threads"]);

    this->use_mkldnn = bool(stoi(config_map_["use_mkldnn"]));

    this->max_side_len = stoi(config_map_["max_side_len"]);

    this->det_db_thresh = stod(config_map_["det_db_thresh"]);

    this->det_db_box_thresh = stod(config_map_["det_db_box_thresh"]);

    this->det_db_box_thresh = stod(config_map_["det_db_box_thresh"]);

    this->det_model_dir.assign(config_map_["det_model_dir"]);

    this->rec_model_dir.assign(config_map_["rec_model_dir"]);

    this->char_list_file.assign(config_map_["char_list_file"]);

    this->visualize = bool(stoi(config_map_["visualize"]));
  }

  bool use_gpu = false;

  int gpu_id = 0;

  int gpu_mem = 4000;

  int cpu_math_library_num_threads = 1;

  bool use_mkldnn = false;

  int max_side_len = 960;

  double det_db_thresh = 0.3;

  double det_db_box_thresh = 0.5;

  double det_db_unclip_ratio = 2.0;

  std::string det_model_dir;

  std::string rec_model_dir;

  std::string char_list_file;

  bool visualize = true;

  void PrintConfigInfo();

private:
  // Load configuration
  std::map<std::string, std::string> LoadConfig(const std::string &config_file);

  std::vector<std::string> split(const std::string &str,
                                 const std::string &delim);

  std::map<std::string, std::string> config_map_;
};

} // namespace PaddleOCR
