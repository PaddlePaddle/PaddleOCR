// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "fastdeploy/vision.h"
#ifdef WIN32
const char sep = '\\';
#else
const char sep = '/';
#endif

void InitAndInfer(const std::string &rec_model_dir,
                  const std::string &rec_label_file,
                  const std::string &image_file,
                  const fastdeploy::RuntimeOption &option) {
  auto rec_model_file = rec_model_dir + sep + "inference.pdmodel";
  auto rec_params_file = rec_model_dir + sep + "inference.pdiparams";
  auto rec_option = option;

  auto rec_model = fastdeploy::vision::ocr::Recognizer(
      rec_model_file, rec_params_file, rec_label_file, rec_option);

  assert(rec_model.Initialized());

  auto im = cv::imread(image_file);
  auto im_bak = im.clone();

  fastdeploy::vision::OCRResult result;

  if (!rec_model.Predict(im, &result)) {
    std::cerr << "Failed to predict." << std::endl;
    return;
  }

  // User can infer a batch of images by following code.
  // if (!rec_model.BatchPredict({im}, &result)) {
  //   std::cerr << "Failed to predict." << std::endl;
  //   return;
  // }

  std::cout << result.Str() << std::endl;
}

int main(int argc, char *argv[]) {
  if (argc < 5) {
    std::cout << "Usage: infer_demo"
                 "path/to/rec_model path/to/rec_label_file path/to/image "
                 "run_option, "
                 "e.g ./infer_demo "
                 "./ch_PP-OCRv3_rec_infer "
                 "./ppocr_keys_v1.txt ./12.jpg 0"
              << std::endl;
    std::cout << "The data type of run_option is int, 0: run with cpu; 1: run "
                 "with gpu;"
              << std::endl;
    return -1;
  }

  fastdeploy::RuntimeOption option;
  int flag = std::atoi(argv[4]);

  if (flag == 0) {
    option.UseCpu();
  } else if (flag == 1) {
    option.UseGpu();
  }

  std::string rec_model_dir = argv[1];
  std::string rec_label_file = argv[2];
  std::string test_image = argv[3];
  InitAndInfer(rec_model_dir, rec_label_file, test_image, option);
  return 0;
}
