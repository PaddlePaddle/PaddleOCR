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

#include "core/general-server/op/general_detection_op.h"
#include "core/predictor/framework/infer.h"
#include "core/predictor/framework/memory.h"
#include "core/predictor/framework/resource.h"
#include "core/util/include/timer.h"
#include <algorithm>
#include <iostream>
#include <memory>
#include <sstream>

/*
#include "opencv2/imgcodecs/legacy/constants_c.h"
#include "opencv2/imgproc/types_c.h"
*/

namespace baidu {
namespace paddle_serving {
namespace serving {

using baidu::paddle_serving::Timer;
using baidu::paddle_serving::predictor::InferManager;
using baidu::paddle_serving::predictor::MempoolWrapper;
using baidu::paddle_serving::predictor::PaddleGeneralModelConfig;
using baidu::paddle_serving::predictor::general_model::Request;
using baidu::paddle_serving::predictor::general_model::Response;
using baidu::paddle_serving::predictor::general_model::Tensor;

int GeneralDetectionOp::inference() {
  VLOG(2) << "Going to run inference";
  const std::vector<std::string> pre_node_names = pre_names();
  if (pre_node_names.size() != 1) {
    LOG(ERROR) << "This op(" << op_name()
               << ") can only have one predecessor op, but received "
               << pre_node_names.size();
    return -1;
  }
  const std::string pre_name = pre_node_names[0];

  const GeneralBlob *input_blob = get_depend_argument<GeneralBlob>(pre_name);
  if (!input_blob) {
    LOG(ERROR) << "input_blob is nullptr,error";
    return -1;
  }
  uint64_t log_id = input_blob->GetLogId();
  VLOG(2) << "(logid=" << log_id << ") Get precedent op name: " << pre_name;

  GeneralBlob *output_blob = mutable_data<GeneralBlob>();
  if (!output_blob) {
    LOG(ERROR) << "output_blob is nullptr,error";
    return -1;
  }
  output_blob->SetLogId(log_id);

  if (!input_blob) {
    LOG(ERROR) << "(logid=" << log_id
               << ") Failed mutable depended argument, op:" << pre_name;
    return -1;
  }

  const TensorVector *in = &input_blob->tensor_vector;
  TensorVector *out = &output_blob->tensor_vector;

  int batch_size = input_blob->_batch_size;
  VLOG(2) << "(logid=" << log_id << ") input batch size: " << batch_size;

  output_blob->_batch_size = batch_size;

  std::vector<int> input_shape;
  int in_num = 0;
  void *databuf_data = NULL;
  char *databuf_char = NULL;
  size_t databuf_size = 0;
  // now only support single string
  char *total_input_ptr = static_cast<char *>(in->at(0).data.data());
  std::string base64str = total_input_ptr;

  float ratio_h{};
  float ratio_w{};

  cv::Mat img = Base2Mat(base64str);
  cv::Mat srcimg;
  cv::Mat resize_img;

  cv::Mat resize_img_rec;
  cv::Mat crop_img;
  img.copyTo(srcimg);

  this->resize_op_.Run(img, resize_img, this->max_side_len_, ratio_h, ratio_w,
                       this->use_tensorrt_);

  this->normalize_op_.Run(&resize_img, this->mean_det, this->scale_det,
                          this->is_scale_);

  std::vector<float> input(1 * 3 * resize_img.rows * resize_img.cols, 0.0f);
  this->permute_op_.Run(&resize_img, input.data());

  TensorVector *real_in = new TensorVector();
  if (!real_in) {
    LOG(ERROR) << "real_in is nullptr,error";
    return -1;
  }

  for (int i = 0; i < in->size(); ++i) {
    input_shape = {1, 3, resize_img.rows, resize_img.cols};
    in_num = std::accumulate(input_shape.begin(), input_shape.end(), 1,
                             std::multiplies<int>());
    databuf_size = in_num * sizeof(float);
    databuf_data = MempoolWrapper::instance().malloc(databuf_size);
    if (!databuf_data) {
      LOG(ERROR) << "Malloc failed, size: " << databuf_size;
      return -1;
    }
    memcpy(databuf_data, input.data(), databuf_size);
    databuf_char = reinterpret_cast<char *>(databuf_data);
    paddle::PaddleBuf paddleBuf(databuf_char, databuf_size);
    paddle::PaddleTensor tensor_in;
    tensor_in.name = in->at(i).name;
    tensor_in.dtype = paddle::PaddleDType::FLOAT32;
    tensor_in.shape = {1, 3, resize_img.rows, resize_img.cols};
    tensor_in.lod = in->at(i).lod;
    tensor_in.data = paddleBuf;
    real_in->push_back(tensor_in);
  }

  Timer timeline;
  int64_t start = timeline.TimeStampUS();
  timeline.Start();

  if (InferManager::instance().infer(engine_name().c_str(), real_in, out,
                                     batch_size)) {
    LOG(ERROR) << "(logid=" << log_id
               << ") Failed do infer in fluid model: " << engine_name().c_str();
    return -1;
  }
  delete real_in;

  std::vector<int> output_shape;
  int out_num = 0;
  void *databuf_data_out = NULL;
  char *databuf_char_out = NULL;
  size_t databuf_size_out = 0;
  // this is special add for PaddleOCR postprecess
  int infer_outnum = out->size();
  for (int k = 0; k < infer_outnum; ++k) {
    int n2 = out->at(k).shape[2];
    int n3 = out->at(k).shape[3];
    int n = n2 * n3;

    float *out_data = static_cast<float *>(out->at(k).data.data());
    std::vector<float> pred(n, 0.0);
    std::vector<unsigned char> cbuf(n, ' ');

    for (int i = 0; i < n; i++) {
      pred[i] = float(out_data[i]);
      cbuf[i] = (unsigned char)((out_data[i]) * 255);
    }

    cv::Mat cbuf_map(n2, n3, CV_8UC1, (unsigned char *)cbuf.data());
    cv::Mat pred_map(n2, n3, CV_32F, (float *)pred.data());

    const double threshold = this->det_db_thresh_ * 255;
    const double maxvalue = 255;
    cv::Mat bit_map;
    cv::threshold(cbuf_map, bit_map, threshold, maxvalue, cv::THRESH_BINARY);
    cv::Mat dilation_map;
    cv::Mat dila_ele =
        cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
    cv::dilate(bit_map, dilation_map, dila_ele);
    boxes = post_processor_.BoxesFromBitmap(pred_map, dilation_map,
                                            this->det_db_box_thresh_,
                                            this->det_db_unclip_ratio_);

    boxes = post_processor_.FilterTagDetRes(boxes, ratio_h, ratio_w, srcimg);

    float max_wh_ratio = 0.0f;
    std::vector<cv::Mat> crop_imgs;
    std::vector<cv::Mat> resize_imgs;
    int max_resize_w = 0;
    int max_resize_h = 0;
    int box_num = boxes.size();
    std::vector<std::vector<float>> output_rec;
    for (int i = 0; i < box_num; ++i) {
      cv::Mat line_img = GetRotateCropImage(img, boxes[i]);
      float wh_ratio = float(line_img.cols) / float(line_img.rows);
      max_wh_ratio = max_wh_ratio > wh_ratio ? max_wh_ratio : wh_ratio;
      crop_imgs.push_back(line_img);
    }

    for (int i = 0; i < box_num; ++i) {
      cv::Mat resize_img;
      crop_img = crop_imgs[i];
      this->resize_op_rec.Run(crop_img, resize_img, max_wh_ratio,
                              this->use_tensorrt_);

      this->normalize_op_.Run(&resize_img, this->mean_rec, this->scale_rec,
                              this->is_scale_);

      max_resize_w = std::max(max_resize_w, resize_img.cols);
      max_resize_h = std::max(max_resize_h, resize_img.rows);
      resize_imgs.push_back(resize_img);
    }
    int buf_size = 3 * max_resize_h * max_resize_w;
    output_rec = std::vector<std::vector<float>>(
        box_num, std::vector<float>(buf_size, 0.0f));
    for (int i = 0; i < box_num; ++i) {
      resize_img_rec = resize_imgs[i];

      this->permute_op_.Run(&resize_img_rec, output_rec[i].data());
    }

    // Inference.
    output_shape = {box_num, 3, max_resize_h, max_resize_w};
    out_num = std::accumulate(output_shape.begin(), output_shape.end(), 1,
                              std::multiplies<int>());
    databuf_size_out = out_num * sizeof(float);
    databuf_data_out = MempoolWrapper::instance().malloc(databuf_size_out);
    if (!databuf_data_out) {
      LOG(ERROR) << "Malloc failed, size: " << databuf_size_out;
      return -1;
    }
    int offset = buf_size * sizeof(float);
    for (int i = 0; i < box_num; ++i) {
      memcpy(databuf_data_out + i * offset, output_rec[i].data(), offset);
    }
    databuf_char_out = reinterpret_cast<char *>(databuf_data_out);
    paddle::PaddleBuf paddleBuf(databuf_char_out, databuf_size_out);
    paddle::PaddleTensor tensor_out;
    tensor_out.name = "x";
    tensor_out.dtype = paddle::PaddleDType::FLOAT32;
    tensor_out.shape = output_shape;
    tensor_out.data = paddleBuf;
    out->push_back(tensor_out);
  }
  out->erase(out->begin(), out->begin() + infer_outnum);

  int64_t end = timeline.TimeStampUS();
  CopyBlobInfo(input_blob, output_blob);
  AddBlobInfo(output_blob, start);
  AddBlobInfo(output_blob, end);
  return 0;
}

cv::Mat GeneralDetectionOp::Base2Mat(std::string &base64_data) {
  cv::Mat img;
  std::string s_mat;
  s_mat = base64Decode(base64_data.data(), base64_data.size());
  std::vector<char> base64_img(s_mat.begin(), s_mat.end());
  img = cv::imdecode(base64_img, cv::IMREAD_COLOR); // CV_LOAD_IMAGE_COLOR
  return img;
}

std::string GeneralDetectionOp::base64Decode(const char *Data, int DataByte) {
  const char DecodeTable[] = {
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
      0,  0,  0,  0,  0,  0,  0,  0,  0,
      62, // '+'
      0,  0,  0,
      63,                                     // '/'
      52, 53, 54, 55, 56, 57, 58, 59, 60, 61, // '0'-'9'
      0,  0,  0,  0,  0,  0,  0,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,
      10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, // 'A'-'Z'
      0,  0,  0,  0,  0,  0,  26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
      37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, // 'a'-'z'
  };

  std::string strDecode;
  int nValue;
  int i = 0;
  while (i < DataByte) {
    if (*Data != '\r' && *Data != '\n') {
      nValue = DecodeTable[*Data++] << 18;
      nValue += DecodeTable[*Data++] << 12;
      strDecode += (nValue & 0x00FF0000) >> 16;
      if (*Data != '=') {
        nValue += DecodeTable[*Data++] << 6;
        strDecode += (nValue & 0x0000FF00) >> 8;
        if (*Data != '=') {
          nValue += DecodeTable[*Data++];
          strDecode += nValue & 0x000000FF;
        }
      }
      i += 4;
    } else // 回车换行,跳过
    {
      Data++;
      i++;
    }
  }
  return strDecode;
}

cv::Mat
GeneralDetectionOp::GetRotateCropImage(const cv::Mat &srcimage,
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

DEFINE_OP(GeneralDetectionOp);

} // namespace serving
} // namespace paddle_serving
} // namespace baidu
