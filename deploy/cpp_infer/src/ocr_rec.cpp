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

#include <include/ocr_rec.h>

namespace PaddleOCR {

void CRNNRecognizer::Run(std::vector<std::vector<std::vector<int>>> boxes,
                         cv::Mat &img, Classifier *cls) {
  cv::Mat srcimg;
  img.copyTo(srcimg);
  cv::Mat crop_img;
  cv::Mat resize_img;

  std::cout << "The predicted text is :" << std::endl;
  int index = 0;
  for (int i = 0; i < boxes.size(); i++) {
    crop_img = GetRotateCropImage(srcimg, boxes[i]);

    if (cls != nullptr) {
      crop_img = cls->Run(crop_img);
    }

    float wh_ratio = float(crop_img.cols) / float(crop_img.rows);

    this->resize_op_.Run(crop_img, resize_img, wh_ratio, this->use_tensorrt_);

    this->normalize_op_.Run(&resize_img, this->mean_, this->scale_,
                            this->is_scale_);

    std::vector<float> input(1 * 3 * resize_img.rows * resize_img.cols, 0.0f);

    this->permute_op_.Run(&resize_img, input.data());

    // Inference.
    auto input_names = this->predictor_->GetInputNames();
    auto input_t = this->predictor_->GetInputHandle(input_names[0]);
    input_t->Reshape({1, 3, resize_img.rows, resize_img.cols});
    input_t->CopyFromCpu(input.data());
    this->predictor_->Run();

    std::vector<float> predict_batch;
    auto output_names = this->predictor_->GetOutputNames();
    auto output_t = this->predictor_->GetOutputHandle(output_names[0]);
    auto predict_shape = output_t->shape();

    int out_num = std::accumulate(predict_shape.begin(), predict_shape.end(), 1,
                                  std::multiplies<int>());
    predict_batch.resize(out_num);

    output_t->CopyToCpu(predict_batch.data());

    // ctc decode
    std::vector<std::string> str_res;
    int argmax_idx;
    int last_index = 0;
    float score = 0.f;
    int count = 0;
    float max_value = 0.0f;

    for (int n = 0; n < predict_shape[1]; n++) {
      argmax_idx =
          int(Utility::argmax(&predict_batch[n * predict_shape[2]],
                              &predict_batch[(n + 1) * predict_shape[2]]));
      max_value =
          float(*std::max_element(&predict_batch[n * predict_shape[2]],
                                  &predict_batch[(n + 1) * predict_shape[2]]));

      if (argmax_idx > 0 && (!(n > 0 && argmax_idx == last_index))) {
        score += max_value;
        count += 1;
        str_res.push_back(label_list_[argmax_idx]);
      }
      last_index = argmax_idx;
    }
    score /= count;
    for (int i = 0; i < str_res.size(); i++) {
      std::cout << str_res[i];
    }
    std::cout << "\tscore: " << score << std::endl;
  }
}

void CRNNRecognizer::LoadModel(const std::string &model_dir) {
  //   AnalysisConfig config;
  paddle_infer::Config config;
  config.SetModel(model_dir + "/inference.pdmodel",
                  model_dir + "/inference.pdiparams");

  if (this->use_gpu_) {
    config.EnableUseGpu(this->gpu_mem_, this->gpu_id_);
    if (this->use_tensorrt_) {
      config.EnableTensorRtEngine(
          1 << 20, 10, 3,
          this->use_fp16_ ? paddle_infer::Config::Precision::kHalf
                          : paddle_infer::Config::Precision::kFloat32,
          false, false);
      std::map<std::string, std::vector<int>> min_input_shape = {
          {"x", {1, 3, 32, 10}}};
      std::map<std::string, std::vector<int>> max_input_shape = {
          {"x", {1, 3, 32, 2000}}};
      std::map<std::string, std::vector<int>> opt_input_shape = {
          {"x", {1, 3, 32, 320}}};

      config.SetTRTDynamicShapeInfo(min_input_shape, max_input_shape,
                                    opt_input_shape);
    }
  } else {
    config.DisableGpu();
    if (this->use_mkldnn_) {
      config.EnableMKLDNN();
      // cache 10 different shapes for mkldnn to avoid memory leak
      config.SetMkldnnCacheCapacity(10);
    }
    config.SetCpuMathLibraryNumThreads(this->cpu_math_library_num_threads_);
  }

  config.SwitchUseFeedFetchOps(false);
  // true for multiple input
  config.SwitchSpecifyInputNames(true);

  config.SwitchIrOptim(true);

  config.EnableMemoryOptim();
  config.DisableGlogInfo();

  this->predictor_ = CreatePredictor(config);
}

cv::Mat CRNNRecognizer::GetRotateCropImage(const cv::Mat &srcimage,
                                           std::vector<std::vector<int>> box) {
  cv::Mat image;
  srcimage.copyTo(image);
  std::vector<std::vector<int>> points = box;

  int x_collect[4] = {box[0][0], box[1][0], box[2][0], box[3][0]};
  int y_collect[4] = {box[0][1], box[1][1], box[2][1], box[3][1]};
  int left = int(*std::min_element(x_collect, x_collect + 4));
  int right = int(*std::max_element(x_collect, x_collect + 4));
  int top = int(*std::min_element(y_collect, y_collect + 4));
  int bottom = int(*std::max_element(y_collect, y_collect + 4));

  cv::Mat img_crop;
  image(cv::Rect(left, top, right - left, bottom - top)).copyTo(img_crop);

  for (int i = 0; i < points.size(); i++) {
    points[i][0] -= left;
    points[i][1] -= top;
  }

  int img_crop_width = int(sqrt(pow(points[0][0] - points[1][0], 2) +
                                pow(points[0][1] - points[1][1], 2)));
  int img_crop_height = int(sqrt(pow(points[0][0] - points[3][0], 2) +
                                 pow(points[0][1] - points[3][1], 2)));

  cv::Point2f pts_std[4];
  pts_std[0] = cv::Point2f(0., 0.);
  pts_std[1] = cv::Point2f(img_crop_width, 0.);
  pts_std[2] = cv::Point2f(img_crop_width, img_crop_height);
  pts_std[3] = cv::Point2f(0.f, img_crop_height);

  cv::Point2f pointsf[4];
  pointsf[0] = cv::Point2f(points[0][0], points[0][1]);
  pointsf[1] = cv::Point2f(points[1][0], points[1][1]);
  pointsf[2] = cv::Point2f(points[2][0], points[2][1]);
  pointsf[3] = cv::Point2f(points[3][0], points[3][1]);

  cv::Mat M = cv::getPerspectiveTransform(pointsf, pts_std);

  cv::Mat dst_img;
  cv::warpPerspective(img_crop, dst_img, M,
                      cv::Size(img_crop_width, img_crop_height),
                      cv::BORDER_REPLICATE);

  if (float(dst_img.rows) >= float(dst_img.cols) * 1.5) {
    cv::Mat srcCopy = cv::Mat(dst_img.rows, dst_img.cols, dst_img.depth());
    cv::transpose(dst_img, srcCopy);
    cv::flip(srcCopy, srcCopy, 0);
    return srcCopy;
  } else {
    return dst_img;
  }
}

} // namespace PaddleOCR
