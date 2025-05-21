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

#include <chrono>
#include "paddle_api.h" // NOLINT
#include "paddle_place.h"

#include "cls_process.h"
#include "crnn_process.h"
#include "db_post_process.h"
#include "AutoLog/auto_log/lite_autolog.h"

using namespace paddle::lite_api; // NOLINT
using namespace std;

// fill tensor with mean and scale and trans layout: nhwc -> nchw, neon speed up
void NeonMeanScale(const float *din, float *dout, int size,
                   const std::vector<float> mean,
                   const std::vector<float> scale) {
  if (mean.size() != 3 || scale.size() != 3) {
    std::cerr << "[ERROR] mean or scale size must equal to 3" << std::endl;
    exit(1);
  }
  float32x4_t vmean0 = vdupq_n_f32(mean[0]);
  float32x4_t vmean1 = vdupq_n_f32(mean[1]);
  float32x4_t vmean2 = vdupq_n_f32(mean[2]);
  float32x4_t vscale0 = vdupq_n_f32(scale[0]);
  float32x4_t vscale1 = vdupq_n_f32(scale[1]);
  float32x4_t vscale2 = vdupq_n_f32(scale[2]);

  float *dout_c0 = dout;
  float *dout_c1 = dout + size;
  float *dout_c2 = dout + size * 2;

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
cv::Mat DetResizeImg(const cv::Mat img, int max_size_len,
                     std::vector<float> &ratio_hw) {
  int w = img.cols;
  int h = img.rows;

  float ratio = 1.f;
  int max_wh = w >= h ? w : h;
  if (max_wh > max_size_len) {
    if (h > w) {
      ratio = static_cast<float>(max_size_len) / static_cast<float>(h);
    } else {
      ratio = static_cast<float>(max_size_len) / static_cast<float>(w);
    }
  }

  int resize_h = static_cast<int>(float(h) * ratio);
  int resize_w = static_cast<int>(float(w) * ratio);
  if (resize_h % 32 == 0)
    resize_h = resize_h;
  else if (resize_h / 32 < 1 + 1e-5)
    resize_h = 32;
  else
    resize_h = (resize_h / 32 - 1) * 32;

  if (resize_w % 32 == 0)
    resize_w = resize_w;
  else if (resize_w / 32 < 1 + 1e-5)
    resize_w = 32;
  else
    resize_w = (resize_w / 32 - 1) * 32;

  cv::Mat resize_img;
  cv::resize(img, resize_img, cv::Size(resize_w, resize_h));

  ratio_hw.push_back(static_cast<float>(resize_h) / static_cast<float>(h));
  ratio_hw.push_back(static_cast<float>(resize_w) / static_cast<float>(w));
  return resize_img;
}

cv::Mat RunClsModel(cv::Mat img, std::shared_ptr<PaddlePredictor> predictor_cls,
                    const float thresh = 0.9) {
  std::vector<float> mean = {0.5f, 0.5f, 0.5f};
  std::vector<float> scale = {1 / 0.5f, 1 / 0.5f, 1 / 0.5f};

  cv::Mat srcimg;
  img.copyTo(srcimg);
  cv::Mat crop_img;
  img.copyTo(crop_img);
  cv::Mat resize_img;

  int index = 0;
  float wh_ratio =
      static_cast<float>(crop_img.cols) / static_cast<float>(crop_img.rows);

  resize_img = ClsResizeImg(crop_img);
  resize_img.convertTo(resize_img, CV_32FC3, 1 / 255.f);

  const float *dimg = reinterpret_cast<const float *>(resize_img.data);

  std::unique_ptr<Tensor> input_tensor0(std::move(predictor_cls->GetInput(0)));
  input_tensor0->Resize({1, 3, resize_img.rows, resize_img.cols});
  auto *data0 = input_tensor0->mutable_data<float>();

  NeonMeanScale(dimg, data0, resize_img.rows * resize_img.cols, mean, scale);
  // Run CLS predictor
  predictor_cls->Run();

  // Get output and run postprocess
  std::unique_ptr<const Tensor> softmax_out(
      std::move(predictor_cls->GetOutput(0)));
  auto *softmax_scores = softmax_out->mutable_data<float>();
  auto softmax_out_shape = softmax_out->shape();
  float score = 0;
  int label = 0;
  for (int i = 0; i < softmax_out_shape[1]; i++) {
    if (softmax_scores[i] > score) {
      score = softmax_scores[i];
      label = i;
    }
  }
  if (label % 2 == 1 && score > thresh) {
    cv::rotate(srcimg, srcimg, 1);
  }
  return srcimg;
}

void RunRecModel(std::vector<std::vector<std::vector<int>>> boxes, cv::Mat img,
                 std::shared_ptr<PaddlePredictor> predictor_crnn,
                 std::vector<std::string> &rec_text,
                 std::vector<float> &rec_text_score,
                 std::vector<std::string> charactor_dict,
                 std::shared_ptr<PaddlePredictor> predictor_cls,
                 int use_direction_classify,
                 std::vector<double> *times,
                 int rec_image_height) {
  std::vector<float> mean = {0.5f, 0.5f, 0.5f};
  std::vector<float> scale = {1 / 0.5f, 1 / 0.5f, 1 / 0.5f};

  cv::Mat srcimg;
  img.copyTo(srcimg);
  cv::Mat crop_img;
  cv::Mat resize_img;

  int index = 0;

  std::vector<double> time_info = {0, 0, 0};
  for (int i = boxes.size() - 1; i >= 0; i--) {
    auto preprocess_start = std::chrono::steady_clock::now();
    crop_img = GetRotateCropImage(srcimg, boxes[i]);
    if (use_direction_classify >= 1) {
      crop_img = RunClsModel(crop_img, predictor_cls);
    }
    float wh_ratio =
        static_cast<float>(crop_img.cols) / static_cast<float>(crop_img.rows);

    resize_img = CrnnResizeImg(crop_img, wh_ratio, rec_image_height);
    resize_img.convertTo(resize_img, CV_32FC3, 1 / 255.f);

    const float *dimg = reinterpret_cast<const float *>(resize_img.data);

    std::unique_ptr<Tensor> input_tensor0(
        std::move(predictor_crnn->GetInput(0)));
    input_tensor0->Resize({1, 3, resize_img.rows, resize_img.cols});
    auto *data0 = input_tensor0->mutable_data<float>();

    NeonMeanScale(dimg, data0, resize_img.rows * resize_img.cols, mean, scale);
    auto preprocess_end = std::chrono::steady_clock::now();
    //// Run CRNN predictor
    auto inference_start = std::chrono::steady_clock::now();
    predictor_crnn->Run();

    // Get output and run postprocess
    std::unique_ptr<const Tensor> output_tensor0(
        std::move(predictor_crnn->GetOutput(0)));
    auto *predict_batch = output_tensor0->data<float>();
    auto predict_shape = output_tensor0->shape();
    auto inference_end = std::chrono::steady_clock::now();

    // ctc decode
    auto postprocess_start = std::chrono::steady_clock::now();
    std::string str_res;
    int argmax_idx;
    int last_index = 0;
    float score = 0.f;
    int count = 0;
    float max_value = 0.0f;

    for (int n = 0; n < predict_shape[1]; n++) {
      argmax_idx = int(Argmax(&predict_batch[n * predict_shape[2]],
                              &predict_batch[(n + 1) * predict_shape[2]]));
      max_value =
          float(*std::max_element(&predict_batch[n * predict_shape[2]],
                                  &predict_batch[(n + 1) * predict_shape[2]]));
      if (argmax_idx > 0 && (!(n > 0 && argmax_idx == last_index))) {
        score += max_value;
        count += 1;
        str_res += charactor_dict[argmax_idx];
      }
      last_index = argmax_idx;
    }
    score /= count;
    rec_text.push_back(str_res);
    rec_text_score.push_back(score);
    auto postprocess_end = std::chrono::steady_clock::now();

    std::chrono::duration<float> preprocess_diff = preprocess_end - preprocess_start;
    time_info[0] += double(preprocess_diff.count() * 1000);
    std::chrono::duration<float> inference_diff = inference_end - inference_start;
    time_info[1] += double(inference_diff.count() * 1000);
    std::chrono::duration<float> postprocess_diff = postprocess_end - postprocess_start;
    time_info[2] += double(postprocess_diff.count() * 1000);

  }

times->push_back(time_info[0]);
times->push_back(time_info[1]);
times->push_back(time_info[2]);
}

std::vector<std::vector<std::vector<int>>>
RunDetModel(std::shared_ptr<PaddlePredictor> predictor, cv::Mat img,
            std::map<std::string, double> Config, std::vector<double> *times) {
  // Read img
  int max_side_len = int(Config["max_side_len"]);
  int det_db_use_dilate = int(Config["det_db_use_dilate"]);

  cv::Mat srcimg;
  img.copyTo(srcimg);
  
  auto preprocess_start = std::chrono::steady_clock::now();
  std::vector<float> ratio_hw;
  img = DetResizeImg(img, max_side_len, ratio_hw);
  cv::Mat img_fp;
  img.convertTo(img_fp, CV_32FC3, 1.0 / 255.f);

  // Prepare input data from image
  std::unique_ptr<Tensor> input_tensor0(std::move(predictor->GetInput(0)));
  input_tensor0->Resize({1, 3, img_fp.rows, img_fp.cols});
  auto *data0 = input_tensor0->mutable_data<float>();

  std::vector<float> mean = {0.485f, 0.456f, 0.406f};
  std::vector<float> scale = {1 / 0.229f, 1 / 0.224f, 1 / 0.225f};
  const float *dimg = reinterpret_cast<const float *>(img_fp.data);
  NeonMeanScale(dimg, data0, img_fp.rows * img_fp.cols, mean, scale);
  auto preprocess_end = std::chrono::steady_clock::now();

  // Run predictor
  auto inference_start = std::chrono::steady_clock::now();
  predictor->Run();

  // Get output and post process
  std::unique_ptr<const Tensor> output_tensor(
      std::move(predictor->GetOutput(0)));
  auto *outptr = output_tensor->data<float>();
  auto shape_out = output_tensor->shape();
  auto inference_end = std::chrono::steady_clock::now();

  // Save output
  auto postprocess_start = std::chrono::steady_clock::now();
  float pred[shape_out[2] * shape_out[3]];
  unsigned char cbuf[shape_out[2] * shape_out[3]];

  for (int i = 0; i < int(shape_out[2] * shape_out[3]); i++) {
    pred[i] = static_cast<float>(outptr[i]);
    cbuf[i] = static_cast<unsigned char>((outptr[i]) * 255);
  }

  cv::Mat cbuf_map(shape_out[2], shape_out[3], CV_8UC1,
                   reinterpret_cast<unsigned char *>(cbuf));
  cv::Mat pred_map(shape_out[2], shape_out[3], CV_32F,
                   reinterpret_cast<float *>(pred));

  const double threshold = double(Config["det_db_thresh"]) * 255;
  const double max_value = 255;
  cv::Mat bit_map;
  cv::threshold(cbuf_map, bit_map, threshold, max_value, cv::THRESH_BINARY);
  if (det_db_use_dilate == 1) {
    cv::Mat dilation_map;
    cv::Mat dila_ele =
        cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
    cv::dilate(bit_map, dilation_map, dila_ele);
    bit_map = dilation_map;
  }
  auto boxes = BoxesFromBitmap(pred_map, bit_map, Config);

  std::vector<std::vector<std::vector<int>>> filter_boxes =
      FilterTagDetRes(boxes, ratio_hw[0], ratio_hw[1], srcimg);
  auto postprocess_end = std::chrono::steady_clock::now();

  std::chrono::duration<float> preprocess_diff = preprocess_end - preprocess_start;
  times->push_back(double(preprocess_diff.count() * 1000));
  std::chrono::duration<float> inference_diff = inference_end - inference_start;
  times->push_back(double(inference_diff.count() * 1000));
  std::chrono::duration<float> postprocess_diff = postprocess_end - postprocess_start;
  times->push_back(double(postprocess_diff.count() * 1000));

  return filter_boxes;
}

std::shared_ptr<PaddlePredictor> loadModel(std::string model_file, int num_threads) {
  MobileConfig config;
  config.set_model_from_file(model_file);

  config.set_threads(num_threads);
  std::shared_ptr<PaddlePredictor> predictor =
      CreatePaddlePredictor<MobileConfig>(config);
  return predictor;
}

cv::Mat Visualization(cv::Mat srcimg,
                      std::vector<std::vector<std::vector<int>>> boxes) {
  cv::Point rook_points[boxes.size()][4];
  for (int n = 0; n < boxes.size(); n++) {
    for (int m = 0; m < boxes[0].size(); m++) {
      rook_points[n][m] = cv::Point(static_cast<int>(boxes[n][m][0]),
                                    static_cast<int>(boxes[n][m][1]));
    }
  }
  cv::Mat img_vis;
  srcimg.copyTo(img_vis);
  for (int n = 0; n < boxes.size(); n++) {
    const cv::Point *ppt[1] = {rook_points[n]};
    int npt[] = {4};
    cv::polylines(img_vis, ppt, npt, 1, 1, CV_RGB(0, 255, 0), 2, 8, 0);
  }

  cv::imwrite("./vis.jpg", img_vis);
  std::cout << "The detection visualized image saved in ./vis.jpg" << std::endl;
  return img_vis;
}

std::vector<std::string> split(const std::string &str,
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
    string s = p;
    res.push_back(s);
    p = std::strtok(NULL, d);
  }

  return res;
}

std::map<std::string, double> LoadConfigTxt(std::string config_path) {
  auto config = ReadDict(config_path);

  std::map<std::string, double> dict;
  for (int i = 0; i < config.size(); i++) {
    std::vector<std::string> res = split(config[i], " ");
    dict[res[0]] = stod(res[1]);
  }
  return dict;
}

void check_params(int argc, char **argv) {
  if (argc<=1 || (strcmp(argv[1], "det")!=0 && strcmp(argv[1], "rec")!=0 && strcmp(argv[1], "system")!=0)) {
    std::cerr << "Please choose one mode of [det, rec, system] !" << std::endl;
    exit(1);
  }
  if (strcmp(argv[1], "det") == 0) {
      if (argc < 9){
        std::cerr << "[ERROR] usage:" << argv[0]
                  << " det det_model runtime_device num_threads batchsize img_dir det_config lite_benchmark_value" << std::endl;
        exit(1);
      }
  }

  if (strcmp(argv[1], "rec") == 0) {
      if (argc < 9){
        std::cerr << "[ERROR] usage:" << argv[0]
                  << " rec rec_model runtime_device num_threads batchsize img_dir key_txt lite_benchmark_value" << std::endl;
        exit(1);
      }
  }

  if (strcmp(argv[1], "system") == 0) {
      if (argc < 12){
        std::cerr << "[ERROR] usage:" << argv[0]
                  << " system det_model rec_model clas_model runtime_device num_threads batchsize img_dir det_config key_txt lite_benchmark_value" << std::endl;
        exit(1);
      }
  }
}

void system(char **argv){
  std::string det_model_file = argv[2];
  std::string rec_model_file = argv[3];
  std::string cls_model_file = argv[4];
  std::string runtime_device = argv[5];
  std::string precision = argv[6];
  std::string num_threads = argv[7];
  std::string batchsize = argv[8];
  std::string img_dir = argv[9];
  std::string det_config_path = argv[10];
  std::string dict_path = argv[11];

  if (strcmp(argv[6], "FP32") != 0 && strcmp(argv[6], "INT8") != 0) {
      std::cerr << "Only support FP32 or INT8." << std::endl;
      exit(1);
  }

  std::vector<cv::String> cv_all_img_names;
  cv::glob(img_dir, cv_all_img_names);

  //// load config from txt file
  auto Config = LoadConfigTxt(det_config_path);
  int use_direction_classify = int(Config["use_direction_classify"]);
  int rec_image_height = int(Config["rec_image_height"]);
  auto charactor_dict = ReadDict(dict_path);
  charactor_dict.insert(charactor_dict.begin(), "#"); // blank char for ctc
  charactor_dict.push_back(" ");

  auto det_predictor = loadModel(det_model_file, std::stoi(num_threads));
  auto rec_predictor = loadModel(rec_model_file, std::stoi(num_threads));
  auto cls_predictor = loadModel(cls_model_file, std::stoi(num_threads));

  std::vector<double> det_time_info = {0, 0, 0};
  std::vector<double> rec_time_info = {0, 0, 0};

  for (int i = 0; i < cv_all_img_names.size(); ++i) {
    std::cout << "The predict img: " << cv_all_img_names[i] << std::endl;
    cv::Mat srcimg = cv::imread(cv_all_img_names[i], cv::IMREAD_COLOR);

    if (!srcimg.data) {
      std::cerr << "[ERROR] image read failed! image path: " << cv_all_img_names[i] << std::endl;
      exit(1);
    }

    std::vector<double> det_times;
    auto boxes = RunDetModel(det_predictor, srcimg, Config, &det_times);
  
    std::vector<std::string> rec_text;
    std::vector<float> rec_text_score;
  
    std::vector<double> rec_times;
    RunRecModel(boxes, srcimg, rec_predictor, rec_text, rec_text_score,
                charactor_dict, cls_predictor, use_direction_classify, &rec_times, rec_image_height);
  
    //// visualization
    auto img_vis = Visualization(srcimg, boxes);
  
    //// print recognized text
    for (int i = 0; i < rec_text.size(); i++) {
      std::cout << i << "\t" << rec_text[i] << "\t" << rec_text_score[i]
                <<  std::endl;

    }

    det_time_info[0] += det_times[0];
    det_time_info[1] += det_times[1];
    det_time_info[2] += det_times[2];
    rec_time_info[0] += rec_times[0];
    rec_time_info[1] += rec_times[1];
    rec_time_info[2] += rec_times[2];
  }
  if (strcmp(argv[12], "True") == 0) {
    AutoLogger autolog_det(det_model_file, 
                       runtime_device,
                       std::stoi(num_threads),
                       std::stoi(batchsize), 
                       "dynamic", 
                       precision, 
                       det_time_info, 
                       cv_all_img_names.size());
    AutoLogger autolog_rec(rec_model_file, 
                       runtime_device,
                       std::stoi(num_threads),
                       std::stoi(batchsize), 
                       "dynamic", 
                       precision, 
                       rec_time_info, 
                       cv_all_img_names.size());

    autolog_det.report();
    std::cout << std::endl;
    autolog_rec.report();
  }
}

void det(int argc, char **argv) {
  std::string det_model_file = argv[2];
  std::string runtime_device = argv[3];
  std::string precision = argv[4];
  std::string num_threads = argv[5];
  std::string batchsize = argv[6];
  std::string img_dir = argv[7];
  std::string det_config_path = argv[8];

  if (strcmp(argv[4], "FP32") != 0 && strcmp(argv[4], "INT8") != 0) {
      std::cerr << "Only support FP32 or INT8." << std::endl;
      exit(1);
  }

  std::vector<cv::String> cv_all_img_names;
  cv::glob(img_dir, cv_all_img_names);

  //// load config from txt file
  auto Config = LoadConfigTxt(det_config_path);

  auto det_predictor = loadModel(det_model_file, std::stoi(num_threads));

  std::vector<double> time_info = {0, 0, 0};
  for (int i = 0; i < cv_all_img_names.size(); ++i) {
    std::cout << "The predict img: " << cv_all_img_names[i] << std::endl;
    cv::Mat srcimg = cv::imread(cv_all_img_names[i], cv::IMREAD_COLOR);

    if (!srcimg.data) {
      std::cerr << "[ERROR] image read failed! image path: " << cv_all_img_names[i] << std::endl;
      exit(1);
    }

    std::vector<double> times;
    auto boxes = RunDetModel(det_predictor, srcimg, Config, &times);

    //// visualization
    auto img_vis = Visualization(srcimg, boxes);
    std::cout << boxes.size() << " bboxes have detected:" << std::endl;

    for (int i=0; i<boxes.size(); i++){
      std::cout << "The " << i << " box:" << std::endl;
      for (int j=0; j<4; j++){
        for (int k=0; k<2; k++){
          std::cout << boxes[i][j][k] << "\t";
        }
      }
      std::cout << std::endl;
    }
    time_info[0] += times[0];
    time_info[1] += times[1];
    time_info[2] += times[2];
  }

  if (strcmp(argv[9], "True") == 0) {
    AutoLogger autolog(det_model_file, 
                       runtime_device,
                       std::stoi(num_threads),
                       std::stoi(batchsize), 
                       "dynamic", 
                       precision, 
                       time_info, 
                       cv_all_img_names.size());
    autolog.report();
  }
}

void rec(int argc, char **argv) {
  std::string rec_model_file = argv[2];
  std::string runtime_device = argv[3];
  std::string precision = argv[4];
  std::string num_threads = argv[5];
  std::string batchsize = argv[6];
  std::string img_dir = argv[7];
  std::string dict_path = argv[8];
  std::string config_path = argv[9];

  if (strcmp(argv[4], "FP32") != 0 && strcmp(argv[4], "INT8") != 0) {
      std::cerr << "Only support FP32 or INT8." << std::endl;
      exit(1);
  }

  auto Config = LoadConfigTxt(config_path);
  int rec_image_height = int(Config["rec_image_height"]);

  std::vector<cv::String> cv_all_img_names;
  cv::glob(img_dir, cv_all_img_names);

  auto charactor_dict = ReadDict(dict_path);
  charactor_dict.insert(charactor_dict.begin(), "#"); // blank char for ctc
  charactor_dict.push_back(" ");

  auto rec_predictor = loadModel(rec_model_file, std::stoi(num_threads));

  std::shared_ptr<PaddlePredictor> cls_predictor;

  std::vector<double> time_info = {0, 0, 0};
  for (int i = 0; i < cv_all_img_names.size(); ++i) {
    std::cout << "The predict img: " << cv_all_img_names[i] << std::endl;
    cv::Mat srcimg = cv::imread(cv_all_img_names[i], cv::IMREAD_COLOR);

    if (!srcimg.data) {
      std::cerr << "[ERROR] image read failed! image path: " << cv_all_img_names[i] << std::endl;
      exit(1);
    }

    int width = srcimg.cols;
    int height = srcimg.rows;
    std::vector<int> upper_left = {0, 0};
    std::vector<int> upper_right = {width, 0};
    std::vector<int> lower_right = {width, height};
    std::vector<int> lower_left  = {0, height};
    std::vector<std::vector<int>> box = {upper_left, upper_right, lower_right, lower_left};
    std::vector<std::vector<std::vector<int>>> boxes = {box};

    std::vector<std::string> rec_text;
    std::vector<float> rec_text_score;
    std::vector<double> times;
    RunRecModel(boxes, srcimg, rec_predictor, rec_text, rec_text_score,
                charactor_dict, cls_predictor, 0, &times, rec_image_height);
  
    //// print recognized text
    for (int i = 0; i < rec_text.size(); i++) {
      std::cout << i << "\t" << rec_text[i] << "\t" << rec_text_score[i]
                << std::endl;
    }
    time_info[0] += times[0];
    time_info[1] += times[1];
    time_info[2] += times[2];
  }
  // TODO: support autolog
  if (strcmp(argv[9], "True") == 0) {
    AutoLogger autolog(rec_model_file, 
                       runtime_device,
                       std::stoi(num_threads),
                       std::stoi(batchsize), 
                       "dynamic", 
                       precision, 
                       time_info, 
                       cv_all_img_names.size());
    autolog.report();
  }
}

int main(int argc, char **argv) {
  check_params(argc, argv);
  std::cout << "mode: " << argv[1] << endl;

  if (strcmp(argv[1], "system") == 0) {
    system(argv);
  }

  if (strcmp(argv[1], "det") == 0) {
    det(argc, argv);
  }

  if (strcmp(argv[1], "rec") == 0) {
    rec(argc, argv);
  }

  return 0;
}
