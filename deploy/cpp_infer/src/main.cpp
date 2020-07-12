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
#include <chrono>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <vector>

#include <cstring>
#include <fstream>
#include <numeric>

#include <include/ocr_det.h>
#include <include/ocr_rec.h>

using namespace std;
using namespace cv;
using namespace PaddleOCR;

int main(int argc, char **argv) {
  if (argc < 4) {
    std::cerr << "[ERROR] usage: " << argv[0]
              << " det_model_file rec_model_file image_path\n";
    exit(1);
  }
  std::string det_model_file = argv[1];
  std::string rec_model_file = argv[2];
  std::string img_path = argv[3];

  auto start = std::chrono::system_clock::now();

  cv::Mat srcimg = cv::imread(img_path, cv::IMREAD_COLOR);

  DBDetector det(det_model_file);
  CRNNRecognizer rec(rec_model_file);

  std::vector<std::vector<std::vector<int>>> boxes;
  det.Run(srcimg, boxes);

  rec.Run(boxes, srcimg);

  auto end = std::chrono::system_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cout << "花费了"
            << double(duration.count()) *
                   std::chrono::microseconds::period::num /
                   std::chrono::microseconds::period::den
            << "秒" << std::endl;

  return 0;
}
