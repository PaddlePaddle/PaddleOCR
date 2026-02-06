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

#include "yaml_config.h"

#include <iostream>
#include <sstream>

#include "absl/strings/str_cat.h"
#include "src/utils/ilogger.h"

YamlConfig::YamlConfig(const std::string &model_dir) {
  auto status_get = GetConfigYamlPaths(model_dir);
  if (!status_get.ok()) {
    INFOE("Could find files with the .yaml or .yml in %s %s", model_dir.c_str(),
          status_get.ToString().c_str());
    exit(-1);
  }
  auto status = LoadYamlFile();
  if (!status.ok()) {
    INFOE("Failed to load config: ", status.ToString().c_str());
    exit(-1);
  }
  Init();
}

absl::Status YamlConfig::GetConfigYamlPaths(const std::string &model_dir) {
  if (Utility::GetFileExtension(model_dir) == "yaml" ||
      Utility::GetFileExtension(model_dir) == "yml") {
    config_yaml_path_ = model_dir;
    return absl::OkStatus();
  }
  std::string config_path_yml =
      model_dir + "/" + Utility::MODEL_FILE_PREFIX + ".yml";
  std::string config_path_yaml =
      model_dir + "/" + Utility::MODEL_FILE_PREFIX + ".yaml";
  if (Utility::FileExists(config_path_yml).ok()) {
    config_yaml_path_ = config_path_yml;
    return absl::OkStatus();
  } else if (Utility::FileExists(config_path_yaml).ok()) {
    config_yaml_path_ = config_path_yaml;
    return absl::OkStatus();
  } else {
    return absl::NotFoundError("file is not exist!");
  }
};

absl::Status YamlConfig::LoadYamlFile() {
  try {
    YAML::Node config = YAML::LoadFile(config_yaml_path_);
    ParseNode(config);
    return absl::OkStatus();
  } catch (const YAML::BadFile &e) {
    return absl::NotFoundError(
        absl::StrCat("Failed to open YAML file: ", config_yaml_path_));
  } catch (const YAML::ParserException &e) {
    return absl::InvalidArgumentError(
        absl::StrCat("Failed to parse YAML file: ", e.what()));
  } catch (const YAML::Exception &e) {
    return absl::InternalError(absl::StrCat("YAML error: ", e.what()));
  } catch (const std::exception &e) {
    return absl::InternalError(absl::StrCat("Unexpected error: ", e.what()));
  }
}

void YamlConfig::Init() {
  for (const auto &info : data_) {
    if (info.first.find("DecodeImage.channel_first") != std::string::npos) {
      pre_process_op_info_["DecodeImage.channel_first"] = info.second;
    } else if (info.first.find("DecodeImage.img_mode") != std::string::npos) {
      pre_process_op_info_["DecodeImage.img_mode"] = info.second;
    } else if (info.first.find("DetLabelEncode") != std::string::npos) {
      pre_process_op_info_["DetLabelEncode"] = info.second;
    } else if (info.first.find("DetResizeForTest.resize_long") !=
               std::string::npos) {
      pre_process_op_info_["DetResizeForTest.resize_long"] = info.second;
    } else if (info.first.find("NormalizeImage.mean") != std::string::npos) {
      size_t pos = info.first.find("NormalizeImage.mean");
      size_t after = pos + std::string("NormalizeImage.mean").size();
      if (info.first[after] != '[') {
        pre_process_op_info_["NormalizeImage.mean"] = info.second;
      }
    } else if (info.first.find("NormalizeImage.order") != std::string::npos) {
      pre_process_op_info_["NormalizeImage.order"] = info.second;
    } else if (info.first.find("NormalizeImage.scale") != std::string::npos) {
      pre_process_op_info_["NormalizeImage.scale"] = info.second;
    } else if (info.first.find("NormalizeImage.std") != std::string::npos) {
      size_t pos = info.first.find("NormalizeImage.std");
      size_t after = pos + std::string("NormalizeImage.std").size();
      if (info.first[after] != '[') {
        pre_process_op_info_["NormalizeImage.std"] = info.second;
      }
    } else if (info.first.find("ResizeImage.size") != std::string::npos) {
      size_t pos = info.first.find("ResizeImage.size");
      size_t after = pos + std::string("ResizeImage.size").size();
      if (info.first[after] != '[') {
        pre_process_op_info_["ResizeImage.size"] = info.second;
      }
    } else if (info.first.find("ResizeImage.resize_short") !=
               std::string::npos) {
      pre_process_op_info_["ResizeImage.resize_short"] = info.second;
    } else if (info.first.find("CropImage.size") != std::string::npos) {
      pre_process_op_info_["CropImage.size"] = info.second;
    } else if (info.first.find("ToCHWImage") != std::string::npos) {
      pre_process_op_info_["ToCHWImage"] = info.second;
    } else if (info.first.find("KeepKeys.keep_keys") != std::string::npos) {
      pre_process_op_info_["KeepKeys.keep_keys"] = info.second;
    } else if (info.first.find("PostProcess.name") != std::string::npos) {
      post_process_op_info_["PostProcess.name"] = info.second;
    } else if (info.first.find("PostProcess.thresh") != std::string::npos) {
      post_process_op_info_["PostProcess.thresh"] = info.second;
    } else if (info.first.find("PostProcess.box_thresh") != std::string::npos) {
      post_process_op_info_["PostProcess.box_thresh"] = info.second;
    } else if (info.first.find("PostProcess.max_candidates") !=
               std::string::npos) {
      post_process_op_info_["PostProcess.max_candidates"] = info.second;
    } else if (info.first.find("PostProcess.unclip_ratio") !=
               std::string::npos) {
      post_process_op_info_["PostProcess.unclip_ratio"] = info.second;
    } else if (info.first.find("PostProcess.Topk.topk") != std::string::npos) {
      post_process_op_info_["PostProcess.Topk.topk"] = info.second;
    } else if (info.first.find("PostProcess.Topk.label_list") !=
               std::string::npos) {
      size_t pos = info.first.find("PostProcess.Topk.label_list");
      size_t after = pos + std::string("PostProcess.Topk.label_list").size();
      if (info.first[after] != '[') {
        post_process_op_info_["PostProcess.Topk.label_list"] = info.second;
      }
    } else if (info.first.find("PostProcess.character_dict") !=
               std::string::npos) {
      size_t pos = info.first.find("PostProcess.character_dict");
      size_t after = pos + std::string("PostProcess.character_dict").size();
      if (info.first[after] != '[') {
        post_process_op_info_["PostProcess.character_dict"] = info.second;
      }
    }
  }
}

void YamlConfig::ParseNode(const YAML::Node &node, const std::string &prefix) {
  if (node.IsMap()) {
    for (auto it = node.begin(); it != node.end(); ++it) {
      std::string key = prefix.empty()
                            ? it->first.as<std::string>()
                            : prefix + "." + it->first.as<std::string>();
      ParseNode(it->second, key);
    }
  } else if (node.IsSequence()) {
    std::stringstream ss;
    ss << "[";
    for (size_t i = 0; i < node.size(); ++i) {
      std::string index_key = prefix + "[" + std::to_string(i) + "]";
      if (node[i].IsScalar()) {
        data_[index_key] = node[i].as<std::string>();
        if (i > 0)
          ss << ", ";
        ss << node[i].as<std::string>();
      } else {
        ParseNode(node[i], index_key);
      }
    }
    ss << "]";
    data_[prefix] = ss.str();
  } else if (node.IsScalar()) {
    data_[prefix] = node.as<std::string>();
  } else if (node.IsNull()) {
    data_[prefix] = "null";
  }
}

absl::StatusOr<std::string>
YamlConfig::GetString(const std::string &key,
                      const std::string &default_value) const {
  for (const auto &info : data_) {
    if (info.first.find(key) != std::string::npos) {
      return info.second;
    }
  }
  return default_value;
}

absl::StatusOr<int> YamlConfig::GetInt(const std::string &key,
                                       int default_value) const {
  for (const auto &info : data_) {
    if (info.first.find(key) != std::string::npos) {
      for (int i = 0; i < info.second.size(); i++) {
        if (!std::isdigit(static_cast<uchar>(info.second[i]))) {
          return absl::InvalidArgumentError("the " + key + " is not int type");
        }
      }
      return std::stoi(info.second);
    }
  }
  return default_value;
}

absl::StatusOr<float> YamlConfig::GetFloat(const std::string &key,
                                           float default_value) const {
  for (const auto &info : data_) {
    if (info.first.find(key) != std::string::npos) {
      return std::stof(info.second);
    }
  }
  INFOW("Key not found %s,will use default value %f.", key.c_str(),
        default_value);
  return default_value;
}

absl::StatusOr<double> YamlConfig::GetDouble(const std::string &key) const {
  auto it = data_.find(key);
  if (it == data_.end()) {
    return absl::NotFoundError(absl::StrCat("Key not found: ", key));
  }
  try {
    return std::stod(it->second);
  } catch (const std::invalid_argument &) {
    return absl::InvalidArgumentError(
        absl::StrCat("Invalid double value for key '", key, "': ", it->second));
  } catch (const std::out_of_range &) {
    return absl::OutOfRangeError(absl::StrCat(
        "Double value out of range for key '", key, "': ", it->second));
  }
}

absl::StatusOr<bool> YamlConfig::GetBool(const std::string &key,
                                         bool default_value) const {
  for (const auto &info : data_) {
    if (info.first.find(key) != std::string::npos) {
      if (Utility::ToLower(info.second) == "true") {
        return true;
      } else if (Utility::ToLower(info.second) == "false") {
        return false;
      } else {
        return absl::InvalidArgumentError("the " + key + " is not bool type");
      }
    }
  }
  return default_value;
}
absl::StatusOr<std::unordered_map<std::string, std::string>>
YamlConfig::GetSubModule(const std::string &key) const {
  std::unordered_map<std::string, std::string> submodule_result = {};
  for (const auto &info : data_) {
    if (info.first.find(key) != std::string::npos) {
      submodule_result[info.first] = info.second;
    }
  }
  if (submodule_result.empty()) {
    return absl::NotFoundError("the " + key + " is not exits!");
  }
  return submodule_result;
}
absl::Status YamlConfig::HasKey(const std::string &key) const {
  if (data_.find(key) != data_.end()) {
    return absl::OkStatus();
  }
  return absl::NotFoundError(absl::StrCat("Key not found: ", key));
}

absl::Status YamlConfig::PrintAll() const {
  for (const auto &it : data_) {
    std::cout << it.first << ": " << it.second << std::endl;
  }
  return absl::OkStatus();
}

absl::Status YamlConfig::PrintWithPrefix(const std::string &prefix) const {
  for (const auto &it : data_) {
    if (it.first.find(prefix) == 0) {
      std::cout << it.first << ": " << it.second << std::endl;
    }
  }
  return absl::OkStatus();
}

absl::Status YamlConfig::FindPreProcessOp(const std::string &prefix) const {
  std::unordered_map<std::string, std::string> pre_process_op_info{};
  for (const auto &it : data_) {
    if (it.first.find(prefix) == 0) {
      std::cout << it.first << ": " << it.second << std::endl;
    }
  }
  return absl::OkStatus();
}

VectorVariant YamlConfig::SmartParseVector(const std::string &input) {
  auto trimBracketAndSpace = [](const std::string &str) -> std::string {
    std::string s = str;
    s.erase(std::remove(s.begin(), s.end(), ' '), s.end());
    if (!s.empty() && s.front() == '[')
      s.erase(0, 1);
    if (!s.empty() && s.back() == ']')
      s.pop_back();
    return s;
  };
  auto splitComma = [](const std::string &s) -> std::vector<std::string> {
    std::vector<std::string> res;
    std::string cur;
    bool inQuotes = false;
    for (size_t i = 0; i < s.size(); ++i) {
      char ch = s[i];
      if (ch == '"' && s[i + 1] != ',') {
        inQuotes = !inQuotes;
      }
      if (ch == ',' && s[i - 1] == ',' && s[i + 1] == ',') {
        cur += ch;
        continue;
      }
      if (ch == ',' && !inQuotes) {
        res.push_back(cur);
        cur.clear();
      } else {
        cur += ch;
      }
    }
    if (!cur.empty())
      res.push_back(cur);
    return res;
  };
  auto isInt = [](const std::string &s) -> bool {
    if (s.empty())
      return false;
    size_t i = 0;
    if (s[0] == '-' || s[0] == '+')
      i = 1;
    if (i == s.size())
      return false;
    for (; i < s.size(); ++i) {
      if (!isdigit(s[i]))
        return false;
    }
    return true;
  };
  auto isFloat = [](const std::string &s) -> bool {
    std::istringstream iss(s);
    float f;
    char c;
    return (iss >> f) && !(iss >> c);
  };
  VectorVariant result;
  result.type = VECTOR_UNKNOWN;

  std::string s = trimBracketAndSpace(input);
  std::vector<std::string> items = splitComma(s);

  bool allString = true, allInt = true, allFloat = true;
  for (size_t i = 0; i < items.size(); ++i) {
    std::string tmp = items[i];
    if (tmp.size() >= 2 && tmp.front() == '"' && tmp.back() == '"') {
      continue;
    }
    allString = false;
    if (!isInt(tmp))
      allInt = false;
    if (!isFloat(tmp))
      allFloat = false;
  }

  if (allString) {
    result.type = VECTOR_STRING;
    for (size_t i = 0; i < items.size(); ++i) {
      std::string tmp = items[i];
      result.vec_string.push_back(tmp.substr(1, tmp.size() - 2));
    }
  } else if (allInt) {
    result.type = VECTOR_INT; // int maybe is string
    for (size_t i = 0; i < items.size(); ++i) {
      result.vec_int.push_back(std::stoi(items[i]));
    }
    for (size_t i = 0; i < items.size(); ++i) {
      result.vec_string.push_back(items[i]);
    }
  } else if (allFloat) {
    result.type = VECTOR_FLOAT;
    for (size_t i = 0; i < items.size(); ++i) {
      result.vec_float.push_back(std::stof(items[i]));
    }
  } else {
    result.type = VECTOR_STRING;
    for (size_t i = 0; i < items.size(); ++i) {
      result.vec_string.push_back(items[i]);
    }
  }
  return result;
}

absl::StatusOr<std::pair<std::string, std::string>>
YamlConfig::FindKey(const std::string &key) {
  for (const auto &info : data_) {
    if (info.first.find(key) != std::string::npos) {
      return info;
    }
  }
  return absl::NotFoundError("Could find key " + key);
}
