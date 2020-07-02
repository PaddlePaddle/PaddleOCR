// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include <iostream>
#include <vector>
#include <chrono>
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "paddle_api.h"  // NOLINT

#include "utils/db_post_process.cpp"
#include "utils/crnn_process.cpp"
#include <cstring>
#include <fstream>

using namespace paddle::lite_api;  // NOLINT

struct Object {
  cv::Rect rec;
  int class_id;
  float prob;
};

int64_t ShapeProduction(const shape_t& shape) {
  int64_t res = 1;
  for (auto i : shape) res *= i;
  return res;
}

// fill tensor with mean and scale and trans layout: nhwc -> nchw, neon speed up
void neon_mean_scale(const float* din,
                     float* dout,
                     int size,
                     const std::vector<float> mean,
                     const std::vector<float> scale) {
  if (mean.size() != 3 || scale.size() != 3) {
    std::cerr << "[ERROR] mean or scale size must equal to 3\n";
    exit(1);
  }
  float32x4_t vmean0 = vdupq_n_f32(mean[0]);
  float32x4_t vmean1 = vdupq_n_f32(mean[1]);
  float32x4_t vmean2 = vdupq_n_f32(mean[2]);
  float32x4_t vscale0 = vdupq_n_f32(scale[0]);
  float32x4_t vscale1 = vdupq_n_f32(scale[1]);
  float32x4_t vscale2 = vdupq_n_f32(scale[2]);

  float* dout_c0 = dout;
  float* dout_c1 = dout + size;
  float* dout_c2 = dout + size * 2;

  int i = 0;
  for (; i < size - 3; i += 4) {
    float32x4x3_t vin3 = vld3q_f32(din);
    float32x4_t vsub0 = vsubq_f32(vin3.val[0], vmean0);
    float32x4_t vsub1 = vsubq_f32(vin3.val[1], vmean1);
    float32x4_t vsub2 = vsubq_f32(vin3.val[2], vmean2);
    float32x4_t vs0 = vmulq_f32(vsub0, vscale0);
    float32x4_t vs1 = vmulq_f32(vsub1, vscale1);
    float32x4_t vs2 = vmulq_f32(vsub2, vscale2);
    vst1q_f32(dout_c0, vs0);
    vst1q_f32(dout_c1, vs1);
    vst1q_f32(dout_c2, vs2);

    din += 12;
    dout_c0 += 4;
    dout_c1 += 4;
    dout_c2 += 4;
  }
  for (; i < size; i++) {
    *(dout_c0++) = (*(din++) - mean[0]) * scale[0];
    *(dout_c1++) = (*(din++) - mean[1]) * scale[1];
    *(dout_c2++) = (*(din++) - mean[2]) * scale[2];
  }
}

// resize image to a size multiple of 32 which is required by the network
cv::Mat resize_img_type0(const cv::Mat img, int max_size_len, float *ratio_h, float *ratio_w){
  int w = img.cols;
  int h = img.rows;

  float ratio = 1.f;
  int max_wh = w >=h ? w : h;
  if (max_wh > max_size_len){
    if (h > w){
      ratio = float(max_size_len) / float(h);
    } else {
      ratio = float(max_size_len) / float(w);
    }
  }

  int resize_h = int(float(h) * ratio);
  int resize_w = int(float(w) * ratio);
  if (resize_h % 32 == 0)
    resize_h = resize_h;
  else if (resize_h / 32 < 1)
    resize_h = 32;
  else
    resize_h = (resize_h / 32 - 1) * 32;

  if (resize_w % 32 == 0)
    resize_w = resize_w;
  else if (resize_w /32 < 1)
    resize_w = 32;
  else
    resize_w = (resize_w/32 - 1)*32;

  cv::Mat resize_img;
  cv::resize(img, resize_img, cv::Size(resize_w, resize_h));

  *ratio_h = float(resize_h) / float(h);
  *ratio_w = float(resize_w) / float(w);
  return resize_img;
}

using namespace std;

void RunRecModel(std::vector<std::vector<std::vector<int>>> boxes, cv::Mat img, std::string rec_model_file){

  MobileConfig config;
  config.set_model_from_file(rec_model_file);

  std::shared_ptr<PaddlePredictor> predictor_crnn =
        CreatePaddlePredictor<MobileConfig>(config);

  std::vector<float> mean = {0.5f, 0.5f, 0.5f};
  std::vector<float> scale = {1 / 0.5f, 1 / 0.5f, 1 / 0.5f};

  cv::Mat srcimg;
  img.copyTo(srcimg);
  cv::Mat crop_img;
  cv::Mat resize_img;

  std::string dict_path = "./ppocr_keys_v1.txt";
  auto charactor_dict = read_dict(dict_path);

  std::cout << "The predicted text is :" << std::endl;
  int index = 0;
  for (int i=boxes.size()-1; i >= 0; i--) {
    crop_img = get_rotate_crop_image(srcimg, boxes[i]);

    float wh_ratio = float(crop_img.cols) / float(crop_img.rows);

    resize_img = crnn_resize_img(crop_img, wh_ratio);
    resize_img.convertTo(resize_img, CV_32FC3, 1 / 255.f);

    const float *dimg = reinterpret_cast<const float *>(resize_img.data);

    std::unique_ptr <Tensor> input_tensor0(std::move(predictor_crnn->GetInput(0)));
    input_tensor0->Resize({1, 3, resize_img.rows, resize_img.cols});
    auto *data0 = input_tensor0->mutable_data<float>();

    neon_mean_scale(dimg, data0, resize_img.rows * resize_img.cols, mean, scale);

    //// Run CRNN predictor
    predictor_crnn->Run();

    // Get output and run postprocess
    std::unique_ptr<const Tensor> output_tensor0(
            std::move(predictor_crnn->GetOutput(0)));
    auto *rec_idx = output_tensor0->data<int>();

    auto rec_idx_lod = output_tensor0->lod();
    auto shape_out = output_tensor0->shape();

    std::vector<int> pred_idx;
    for (int n = int(rec_idx_lod[0][0]); n < int(rec_idx_lod[0][1] * 2); n += 2) {
      pred_idx.push_back(int(rec_idx[n]));
    }

    if (pred_idx.size() < 1e-3)
      continue;
    std::cout << std::endl;

    index += 1;
    std::cout << index << "\t";
    for (int n = 0; n < pred_idx.size(); n++) {
      std::cout << charactor_dict[pred_idx[n]];
    }

    ////get score
    std::unique_ptr<const Tensor> output_tensor1(std::move(predictor_crnn->GetOutput(1)));
    auto *predict_batch = output_tensor1->data<float>();
    auto predict_shape = output_tensor1->shape();

    auto predict_lod = output_tensor1->lod();

    int argmax_idx;
    int blank = predict_shape[1];
    float score = 0.f;
    int count = 0;
    float max_value = 0.0f;

    for (int n = predict_lod[0][0]; n < predict_lod[0][1] - 1; n++) {
      argmax_idx = int(argmax(&predict_batch[n * predict_shape[1]], &predict_batch[(n + 1) * predict_shape[1]]));
      max_value = float(
              *std::max_element(&predict_batch[n * predict_shape[1]], &predict_batch[(n + 1) * predict_shape[1]]));

      if (blank - 1 - argmax_idx > 1e-5) {
        score += max_value;
        count += 1;
      }

    }
    score /= count;
    std::cout << "\tscore: " << score << std::endl;
  }
}

std::vector<std::vector<std::vector<int>>> RunDetModel(std::string model_file, cv::Mat img) {
  // Set MobileConfig
  MobileConfig config;
  config.set_model_from_file(model_file);

  std::shared_ptr<PaddlePredictor> predictor =
      CreatePaddlePredictor<MobileConfig>(config);

  // Read img
  int max_side_len = 960;
  float ratio_h{};
  float ratio_w{};

  cv::Mat srcimg;
  img.copyTo(srcimg);

  img = resize_img_type0(img, max_side_len, &ratio_h, &ratio_w);
  cv::Mat img_fp;
  img.convertTo(img_fp, CV_32FC3, 1.0 / 255.f);

  // Prepare input data from image
  std::unique_ptr<Tensor> input_tensor0(std::move(predictor->GetInput(0)));
  input_tensor0->Resize({1, 3, img_fp.rows, img_fp.cols});
  auto* data0 = input_tensor0->mutable_data<float>();

  std::vector<float> mean = {0.485f, 0.456f, 0.406f};
  std::vector<float> scale = {1/0.229f, 1/0.224f, 1/0.225f};
  const float* dimg = reinterpret_cast<const float*>(img_fp.data);
  neon_mean_scale(dimg, data0, img_fp.rows * img_fp.cols, mean, scale);

  // Run predictor
  predictor->Run();

  // Get output and post process
  std::unique_ptr<const Tensor> output_tensor(std::move(predictor->GetOutput(0)));
  auto* outptr = output_tensor->data<float>();
  auto shape_out = output_tensor->shape();

  int64_t out_numl = 1;
  double sum = 0;
  for (auto i : shape_out) {
    out_numl *= i;
  }

  // Save output
  float pred[shape_out[2]][shape_out[3]];
  unsigned char cbuf[shape_out[2]][shape_out[3]];

  for (int i=0; i< int(shape_out[2]*shape_out[3]); i++){
    pred[int(i/int(shape_out[3]))][int(i%shape_out[3])] = float(outptr[i]);
    cbuf[int(i/int(shape_out[3]))][int(i%shape_out[3])] = (unsigned char) ((outptr[i])*255);
  }

  cv::Mat cbuf_map(shape_out[2], shape_out[3], CV_8UC1, (unsigned char*)cbuf);
  cv::Mat pred_map(shape_out[2], shape_out[3], CV_32F, (float *)pred);

  const double threshold = 0.3*255;
  const double maxvalue = 255;
  cv::Mat bit_map;
  cv::threshold(cbuf_map, bit_map, threshold, maxvalue, cv::THRESH_BINARY);

  auto boxes = boxes_from_bitmap(pred_map, bit_map);

  std::vector<std::vector<std::vector<int>>> filter_boxes = filter_tag_det_res(boxes, ratio_h, ratio_w, srcimg);

  //// visualization
  cv::Point rook_points[filter_boxes.size()][4];
  for (int n=0; n<filter_boxes.size(); n++){
    for (int m=0; m< filter_boxes[0].size(); m++){
      rook_points[n][m] = cv::Point(int(filter_boxes[n][m][0]), int(filter_boxes[n][m][1]));
    }
  }

  cv::Mat img_vis;
  srcimg.copyTo(img_vis);
  for (int n=0; n<boxes.size(); n++){
    const cv::Point* ppt[1] = { rook_points[n] };
    int npt[] = { 4 };
    cv::polylines(img_vis, ppt, npt,1,1,CV_RGB(0,255,0),2,8,0);
  }

  cv::imwrite("./imgs_vis/vis.jpg", img_vis);
  std::cout << "The detection visualized image saved in ./imgs_vis/" <<std::endl;

  return filter_boxes;
}


int main(int argc, char** argv) {
  if (argc < 4) {
    std::cerr << "[ERROR] usage: " << argv[0] << " det_model_file rec_model_file image_path\n";
    exit(1);
  }
  std::string det_model_file = argv[1];
  std::string rec_model_file = argv[2];
  std::string img_path = argv[3];

  auto start = std::chrono::system_clock::now();

  cv::Mat srcimg = cv::imread(img_path, cv::IMREAD_COLOR);
  auto boxes = RunDetModel(det_model_file, srcimg);

  RunRecModel(boxes, srcimg, rec_model_file);

  auto end   = std::chrono::system_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cout <<  "花费了"
            << double(duration.count()) * std::chrono::microseconds::period::num /std::chrono::microseconds::period::den
            << "秒" << std::endl;

  return 0;
}

