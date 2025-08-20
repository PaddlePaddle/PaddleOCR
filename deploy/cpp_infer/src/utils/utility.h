// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <algorithm>
#include <fstream>
#include <map>
#include <opencv2/opencv.hpp>
#include <string>
#include <unordered_set>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#ifdef _WIN32
#include <direct.h>
#include <io.h>
#define mkdir _mkdir
static const char PATH_SEPARATOR = '\\';
#else
#include <sys/stat.h>
#include <sys/types.h>
static const char PATH_SEPARATOR = '/';
#endif

#include <errno.h>

class Utility {
public:
  struct PaddleXConfigVariant {
    enum class Type { NONE, STR, MAP };
    Type type;
    std::string str_val;
    std::unordered_map<std::string, std::string> map_val;
    PaddleXConfigVariant() : type(Type::NONE) {}
    PaddleXConfigVariant(const std::string &val)
        : type(Type::STR), str_val(val) {}
    PaddleXConfigVariant(const char *val)
        : type(Type::STR), str_val(val ? val : "") {}
    PaddleXConfigVariant(
        const std::unordered_map<std::string, std::string> &val)
        : type(Type::MAP), map_val(val) {}
    bool IsStr() const { return type == Type::STR; }
    bool IsMap() const { return type == Type::MAP; }
    const std::string &GetStr() const {
      assert(IsStr());
      return str_val;
    }
    const std::unordered_map<std::string, std::string> &GetMap() const {
      assert(IsMap());
      return map_val;
    }
  };
  static constexpr const char *MODEL_FILE_PREFIX = "inference";
  static const std::set<std::string> kImgSuffixes;

  static absl::StatusOr<
      std::map<std::string, std::pair<std::string, std::string>>>
  GetModelPaths(const std::string &model_dir,
                const std::string &model_file_prefix = MODEL_FILE_PREFIX);

  static absl::StatusOr<std::string>
  FindModelPath(const std::string &model_dir, const std::string &model_name);
  static absl::StatusOr<std::string>
  GetConfigPaths(const std::string &model_dir,
                 const std::string &model_file_prefix = MODEL_FILE_PREFIX);

  static absl::StatusOr<std::string>
  GetDefaultConfig(std::string pipeline_name);

  static absl::Status FileExists(const std::string &path);

  // TODO windows
  static bool IsMkldnnAvailable();

  static void PrintShape(const cv::Mat &img);

  static absl::Status MyCreateDirectory(const std::string &path);
  static absl::Status MyCreatePath(const std::string &path);
  static absl::Status MyCreateFile(const std::string &filepath);

  static absl::StatusOr<std::vector<cv::Mat>> SplitBatch(const cv::Mat &batch);

  static absl::StatusOr<cv::Mat> MyLoadImage(const std::string &file_path);
  static bool IsDirectory(const std::string &path);
  static std::string GetFileExtension(const std::string &file_path);
  static void GetFilesRecursive(const std::string &dir_path,
                                std::vector<std::string> &file_list);
  static std::string ToLower(const std::string &str);
  static bool IsImageFile(const std::string &file_path);
  static int MakeDir(const std::string &path);
  static absl::Status CreateDirectoryRecursive(const std::string &path);
  static absl::Status CreateDirectoryForFile(const std::string &filePath);
  static absl::StatusOr<std::string>
  SmartCreateDirectoryForImage(std::string save_path,
                               const std::string &input_path,
                               const std::string &suffix = "_res");
  static absl::StatusOr<std::string>
  SmartCreateDirectoryForJson(const std::string &save_path,
                              const std::string &input_path,
                              const std::string &suffix = "_res");

  static absl::StatusOr<int> StringToInt(std::string s);
  static bool StringToBool(const std::string &str);
  static std::string VecToString(const std::vector<int> &input);

  static absl::StatusOr<std::tuple<std::string, std::string, std::string>>
  GetOcrModelInfo(std::string lang, std::string ppocr_version);
};
