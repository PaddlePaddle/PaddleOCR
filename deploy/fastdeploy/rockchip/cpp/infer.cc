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

void InitAndInfer(const std::string &det_model_file,
                  const std::string &cls_model_file,
                  const std::string &rec_model_file,
                  const std::string &rec_label_file,
                  const std::string &image_file,
                  const fastdeploy::RuntimeOption &option,
                  const fastdeploy::ModelFormat &format) {
  auto det_params_file = "";
  auto cls_params_file = "";
  auto rec_params_file = "";

  auto det_option = option;
  auto cls_option = option;
  auto rec_option = option;

  if (format == fastdeploy::ONNX) {
    std::cout << "ONNX Model" << std::endl;
  }

  auto det_model = fastdeploy::vision::ocr::DBDetector(
      det_model_file, det_params_file, det_option, format);
  auto cls_model = fastdeploy::vision::ocr::Classifier(
      cls_model_file, cls_params_file, cls_option, format);
  auto rec_model = fastdeploy::vision::ocr::Recognizer(
      rec_model_file, rec_params_file, rec_label_file, rec_option, format);

  if (format == fastdeploy::RKNN) {
    cls_model.GetPreprocessor().DisableNormalize();
    cls_model.GetPreprocessor().DisablePermute();

    det_model.GetPreprocessor().DisableNormalize();
    det_model.GetPreprocessor().DisablePermute();

    rec_model.GetPreprocessor().DisableNormalize();
    rec_model.GetPreprocessor().DisablePermute();
  }
  det_model.GetPreprocessor().SetStaticShapeInfer(true);
  rec_model.GetPreprocessor().SetStaticShapeInfer(true);

  assert(det_model.Initialized());
  assert(cls_model.Initialized());
  assert(rec_model.Initialized());

  // The classification model is optional, so the PP-OCR can also be connected
  // in series as follows auto ppocr_v3 =
  // fastdeploy::pipeline::PPOCRv3(&det_model, &rec_model);
  auto ppocr_v3 =
      fastdeploy::pipeline::PPOCRv3(&det_model, &cls_model, &rec_model);

  // When users enable static shape infer for rec model, the batch size of cls
  // and rec model must to be set to 1.
  ppocr_v3.SetClsBatchSize(1);
  ppocr_v3.SetRecBatchSize(1);

  if (!ppocr_v3.Initialized()) {
    std::cerr << "Failed to initialize PP-OCR." << std::endl;
    return;
  }

  auto im = cv::imread(image_file);

  fastdeploy::vision::OCRResult result;
  if (!ppocr_v3.Predict(im, &result)) {
    std::cerr << "Failed to predict." << std::endl;
    return;
  }

  std::cout << result.Str() << std::endl;

  auto vis_im = fastdeploy::vision::VisOcr(im, result);
  cv::imwrite("vis_result.jpg", vis_im);
  std::cout << "Visualized result saved in ./vis_result.jpg" << std::endl;
}

int main(int argc, char *argv[]) {
  if (argc < 7) {
    std::cout << "Usage: infer_demo path/to/det_model path/to/cls_model "
                 "path/to/rec_model path/to/rec_label_file path/to/image "
                 "run_option, "
                 "e.g ./infer_demo ./ch_PP-OCRv3_det_infer "
                 "./ch_ppocr_mobile_v2.0_cls_infer ./ch_PP-OCRv3_rec_infer "
                 "./ppocr_keys_v1.txt ./12.jpg 0"
              << std::endl;
    std::cout << "The data type of run_option is int, 0: run with cpu; 1: run "
                 "with ascend."
              << std::endl;
    return -1;
  }

  fastdeploy::RuntimeOption option;
  fastdeploy::ModelFormat format;
  int flag = std::atoi(argv[6]);

  if (flag == 0) {
    option.UseCpu();
    format = fastdeploy::ONNX;
  } else if (flag == 1) {
    option.UseRKNPU2();
    format = fastdeploy::RKNN;
  }

  std::string det_model_dir = argv[1];
  std::string cls_model_dir = argv[2];
  std::string rec_model_dir = argv[3];
  std::string rec_label_file = argv[4];
  std::string test_image = argv[5];
  InitAndInfer(det_model_dir, cls_model_dir, rec_model_dir, rec_label_file,
               test_image, option, format);
  return 0;
}
