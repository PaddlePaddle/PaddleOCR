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

#include <include/config.h>

namespace PaddleOCR {

std::vector<std::string> OCRConfig::split(const std::string &str,
                                          const std::string &delim) {
  std::vector<std::string> res;
  if ("" == str)
    return res;
  char *strs = new char[str.length() + 1];
  std::strcpy(strs, str.c_str());

  char *d = new char[delim.length() + 1];
  std::strcpy(d, delim.c_str());

  char *p = std::strtok(strs, d);
  while (p) {
    std::string s = p;
    res.push_back(s);
    p = std::strtok(NULL, d);
  }

  return res;
}

std::map<std::string, std::string>
OCRConfig::LoadConfig(const std::string &config_path) {
  auto config = Utility::ReadDict(config_path);

  std::map<std::string, std::string> dict;
  for (int i = 0; i < config.size(); i++) {
    // pass for empty line or comment
    if (config[i].size() <= 1 || config[i][0] == '#') {
      continue;
    }
    std::vector<std::string> res = split(config[i], " ");
    dict[res[0]] = res[1];
  }
  return dict;
}

void OCRConfig::PrintConfigInfo() {
  std::cout << "=======Paddle OCR inference config======" << std::endl;
  for (auto iter = config_map_.begin(); iter != config_map_.end(); iter++) {
    std::cout << iter->first << " : " << iter->second << std::endl;
  }
  std::cout << "=======End of Paddle OCR inference config======" << std::endl;
}

} // namespace PaddleOCR