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

#include <dirent.h>
#include <include/utility.h>
#include <iostream>
#include <ostream>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>

namespace PaddleOCR {

std::vector<std::string> Utility::ReadDict(const std::string &path) {
  std::ifstream in(path);
  std::string line;
  std::vector<std::string> m_vec;
  if (in) {
    while (getline(in, line)) {
      m_vec.push_back(line);
    }
  } else {
    std::cout << "no such label file: " << path << ", exit the program..."
              << std::endl;
    exit(1);
  }
  return m_vec;
}

void Utility::VisualizeBboxes(
    const cv::Mat &srcimg,
    const std::vector<std::vector<std::vector<int>>> &boxes) {
  cv::Mat img_vis;
  srcimg.copyTo(img_vis);
  for (int n = 0; n < boxes.size(); n++) {
    cv::Point rook_points[4];
    for (int m = 0; m < boxes[n].size(); m++) {
      rook_points[m] = cv::Point(int(boxes[n][m][0]), int(boxes[n][m][1]));
    }

    const cv::Point *ppt[1] = {rook_points};
    int npt[] = {4};
    cv::polylines(img_vis, ppt, npt, 1, 1, CV_RGB(0, 255, 0), 2, 8, 0);
  }

  cv::imwrite("./ocr_vis.png", img_vis);
  std::cout << "The detection visualized image saved in ./ocr_vis.png"
            << std::endl;
}

// list all files under a directory
void Utility::GetAllFiles(const char *dir_name,
                          std::vector<std::string> &all_inputs) {
  if (NULL == dir_name) {
    std::cout << " dir_name is null ! " << std::endl;
    return;
  }
  struct stat s;
  lstat(dir_name, &s);
  if (!S_ISDIR(s.st_mode)) {
    std::cout << "dir_name is not a valid directory !" << std::endl;
    all_inputs.push_back(dir_name);
    return;
  } else {
    struct dirent *filename; // return value for readdir()
    DIR *dir;                // return value for opendir()
    dir = opendir(dir_name);
    if (NULL == dir) {
      std::cout << "Can not open dir " << dir_name << std::endl;
      return;
    }
    std::cout << "Successfully opened the dir !" << std::endl;
    while ((filename = readdir(dir)) != NULL) {
      if (strcmp(filename->d_name, ".") == 0 ||
          strcmp(filename->d_name, "..") == 0)
        continue;
      // img_dir + std::string("/") + all_inputs[0];
      all_inputs.push_back(dir_name + std::string("/") +
                           std::string(filename->d_name));
    }
  }
}

} // namespace PaddleOCR