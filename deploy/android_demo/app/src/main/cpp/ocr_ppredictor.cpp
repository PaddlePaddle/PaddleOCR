//
// Created by fujiayi on 2020/7/1.
//

#include "ocr_ppredictor.h"
#include "common.h"
#include "ocr_cls_process.h"
#include "ocr_crnn_process.h"
#include "ocr_db_post_process.h"
#include "preprocess.h"

namespace ppredictor {

OCR_PPredictor::OCR_PPredictor(const OCR_Config &config) : _config(config) {}

int OCR_PPredictor::init(const std::string &det_model_content,
                         const std::string &rec_model_content,
                         const std::string &cls_model_content) {
  _det_predictor = std::unique_ptr<PPredictor>(new PPredictor{
      _config.use_opencl, _config.thread_num, NET_OCR, _config.mode});
  _det_predictor->init_nb(det_model_content);

  _rec_predictor = std::unique_ptr<PPredictor>(new PPredictor{
      _config.use_opencl, _config.thread_num, NET_OCR_INTERNAL, _config.mode});
  _rec_predictor->init_nb(rec_model_content);

  _cls_predictor = std::unique_ptr<PPredictor>(new PPredictor{
      _config.use_opencl, _config.thread_num, NET_OCR_INTERNAL, _config.mode});
  _cls_predictor->init_nb(cls_model_content);
  return RETURN_OK;
}

int OCR_PPredictor::init_from_file(const std::string &det_model_path,
                                   const std::string &rec_model_path,
                                   const std::string &cls_model_path) {
  _det_predictor = std::unique_ptr<PPredictor>(new PPredictor{
      _config.use_opencl, _config.thread_num, NET_OCR, _config.mode});
  _det_predictor->init_from_file(det_model_path);

  _rec_predictor = std::unique_ptr<PPredictor>(new PPredictor{
      _config.use_opencl, _config.thread_num, NET_OCR_INTERNAL, _config.mode});
  _rec_predictor->init_from_file(rec_model_path);

  _cls_predictor = std::unique_ptr<PPredictor>(new PPredictor{
      _config.use_opencl, _config.thread_num, NET_OCR_INTERNAL, _config.mode});
  _cls_predictor->init_from_file(cls_model_path);
  return RETURN_OK;
}
/**
 * for debug use, show result of First Step
 * @param filter_boxes
 * @param boxes
 * @param srcimg
 */
static void
visual_img(const std::vector<std::vector<std::vector<int>>> &filter_boxes,
           const std::vector<std::vector<std::vector<int>>> &boxes,
           const cv::Mat &srcimg) {
  // visualization
  cv::Point rook_points[filter_boxes.size()][4];
  for (int n = 0; n < filter_boxes.size(); n++) {
    for (int m = 0; m < filter_boxes[0].size(); m++) {
      rook_points[n][m] =
          cv::Point(int(filter_boxes[n][m][0]), int(filter_boxes[n][m][1]));
    }
  }

  cv::Mat img_vis;
  srcimg.copyTo(img_vis);
  for (int n = 0; n < boxes.size(); n++) {
    const cv::Point *ppt[1] = {rook_points[n]};
    int npt[] = {4};
    cv::polylines(img_vis, ppt, npt, 1, 1, CV_RGB(0, 255, 0), 2, 8, 0);
  }
  // 调试用，自行替换需要修改的路径
  cv::imwrite("/sdcard/1/vis.png", img_vis);
}

std::vector<OCRPredictResult>
OCR_PPredictor::infer_ocr(cv::Mat &origin, int max_size_len, int run_det,
                          int run_cls, int run_rec) {
  LOGI("ocr cpp start *****************");
  LOGI("ocr cpp det: %d, cls: %d, rec: %d", run_det, run_cls, run_rec);
  std::vector<OCRPredictResult> ocr_results;
  if (run_det) {
    infer_det(origin, max_size_len, ocr_results);
  }
  if (run_rec) {
    if (ocr_results.size() == 0) {
      OCRPredictResult res;
      ocr_results.emplace_back(std::move(res));
    }
    for (int i = 0; i < ocr_results.size(); i++) {
      infer_rec(origin, run_cls, ocr_results[i]);
    }
  } else if (run_cls) {
    ClsPredictResult cls_res = infer_cls(origin);
    OCRPredictResult res;
    res.cls_score = cls_res.cls_score;
    res.cls_label = cls_res.cls_label;
    ocr_results.push_back(res);
  }

  LOGI("ocr cpp end *****************");
  return ocr_results;
}

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

void OCR_PPredictor::infer_det(cv::Mat &origin, int max_size_len,
                               std::vector<OCRPredictResult> &ocr_results) {
  std::vector<float> mean = {0.485f, 0.456f, 0.406f};
  std::vector<float> scale = {1 / 0.229f, 1 / 0.224f, 1 / 0.225f};

  PredictorInput input = _det_predictor->get_first_input();

  std::vector<float> ratio_hw;
  cv::Mat input_image = DetResizeImg(origin, max_size_len, ratio_hw);
  input_image.convertTo(input_image, CV_32FC3, 1 / 255.0f);
  const float *dimg = reinterpret_cast<const float *>(input_image.data);
  int input_size = input_image.rows * input_image.cols;

  input.set_dims({1, 3, input_image.rows, input_image.cols});

  neon_mean_scale(dimg, input.get_mutable_float_data(), input_size, mean,
                  scale);
  LOGI("ocr cpp det shape %d,%d", input_image.rows, input_image.cols);
  std::vector<PredictorOutput> results = _det_predictor->infer();
  PredictorOutput &res = results.at(0);
  std::vector<std::vector<std::vector<int>>> filtered_box =
      calc_filtered_boxes(res.get_float_data(), res.get_size(),
                          input_image.rows, input_image.cols, origin);
  LOGI("ocr cpp det Filter_box size %ld", filtered_box.size());

  for (int i = 0; i < filtered_box.size(); i++) {
    LOGI("ocr cpp box  %d,%d,%d,%d,%d,%d,%d,%d", filtered_box[i][0][0],
         filtered_box[i][0][1], filtered_box[i][1][0], filtered_box[i][1][1],
         filtered_box[i][2][0], filtered_box[i][2][1], filtered_box[i][3][0],
         filtered_box[i][3][1]);
    OCRPredictResult res;
    res.points = filtered_box[i];
    ocr_results.push_back(res);
  }
}

void OCR_PPredictor::infer_rec(const cv::Mat &origin_img, int run_cls,
                               OCRPredictResult &ocr_result) {
  std::vector<float> mean = {0.5f, 0.5f, 0.5f};
  std::vector<float> scale = {1 / 0.5f, 1 / 0.5f, 1 / 0.5f};
  std::vector<int64_t> dims = {1, 3, 0, 0};

  PredictorInput input = _rec_predictor->get_first_input();

  const std::vector<std::vector<int>> &box = ocr_result.points;
  cv::Mat crop_img;
  if (box.size() > 0) {
    crop_img = get_rotate_crop_image(origin_img, box);
  } else {
    crop_img = origin_img;
  }

  if (run_cls) {
    ClsPredictResult cls_res = infer_cls(crop_img);
    crop_img = cls_res.img;
    ocr_result.cls_score = cls_res.cls_score;
    ocr_result.cls_label = cls_res.cls_label;
  }

  float wh_ratio = float(crop_img.cols) / float(crop_img.rows);
  cv::Mat input_image = crnn_resize_img(crop_img, wh_ratio);
  input_image.convertTo(input_image, CV_32FC3, 1 / 255.0f);
  const float *dimg = reinterpret_cast<const float *>(input_image.data);
  int input_size = input_image.rows * input_image.cols;

  dims[2] = input_image.rows;
  dims[3] = input_image.cols;
  input.set_dims(dims);

  neon_mean_scale(dimg, input.get_mutable_float_data(), input_size, mean,
                  scale);

  std::vector<PredictorOutput> results = _rec_predictor->infer();
  const float *predict_batch = results.at(0).get_float_data();
  const std::vector<int64_t> predict_shape = results.at(0).get_shape();

  // ctc decode
  int argmax_idx;
  int last_index = 0;
  float score = 0.f;
  int count = 0;
  float max_value = 0.0f;

  for (int n = 0; n < predict_shape[1]; n++) {
    argmax_idx = int(argmax(&predict_batch[n * predict_shape[2]],
                            &predict_batch[(n + 1) * predict_shape[2]]));
    max_value =
        float(*std::max_element(&predict_batch[n * predict_shape[2]],
                                &predict_batch[(n + 1) * predict_shape[2]]));
    if (argmax_idx > 0 && (!(n > 0 && argmax_idx == last_index))) {
      score += max_value;
      count += 1;
      ocr_result.word_index.push_back(argmax_idx);
    }
    last_index = argmax_idx;
  }
  score /= count;
  ocr_result.score = score;
  LOGI("ocr cpp rec word size %ld", count);
}

ClsPredictResult OCR_PPredictor::infer_cls(const cv::Mat &img, float thresh) {
  std::vector<float> mean = {0.5f, 0.5f, 0.5f};
  std::vector<float> scale = {1 / 0.5f, 1 / 0.5f, 1 / 0.5f};
  std::vector<int64_t> dims = {1, 3, 0, 0};

  PredictorInput input = _cls_predictor->get_first_input();

  cv::Mat input_image = cls_resize_img(img);
  input_image.convertTo(input_image, CV_32FC3, 1 / 255.0f);
  const float *dimg = reinterpret_cast<const float *>(input_image.data);
  int input_size = input_image.rows * input_image.cols;

  dims[2] = input_image.rows;
  dims[3] = input_image.cols;
  input.set_dims(dims);

  neon_mean_scale(dimg, input.get_mutable_float_data(), input_size, mean,
                  scale);

  std::vector<PredictorOutput> results = _cls_predictor->infer();

  const float *scores = results.at(0).get_float_data();
  float score = 0;
  int label = 0;
  for (int64_t i = 0; i < results.at(0).get_size(); i++) {
    LOGI("ocr cpp cls output scores [%f]", scores[i]);
    if (scores[i] > score) {
      score = scores[i];
      label = i;
    }
  }
  cv::Mat srcimg;
  img.copyTo(srcimg);
  if (label % 2 == 1 && score > thresh) {
    cv::rotate(srcimg, srcimg, 1);
  }
  ClsPredictResult res;
  res.cls_label = label;
  res.cls_score = score;
  res.img = srcimg;
  LOGI("ocr cpp cls word cls %ld, %f", label, score);
  return res;
}

std::vector<std::vector<std::vector<int>>>
OCR_PPredictor::calc_filtered_boxes(const float *pred, int pred_size,
                                    int output_height, int output_width,
                                    const cv::Mat &origin) {
  const double threshold = 0.3;
  const double maxvalue = 1;

  cv::Mat pred_map = cv::Mat::zeros(output_height, output_width, CV_32F);
  memcpy(pred_map.data, pred, pred_size * sizeof(float));
  cv::Mat cbuf_map;
  pred_map.convertTo(cbuf_map, CV_8UC1);

  cv::Mat bit_map;
  cv::threshold(cbuf_map, bit_map, threshold, maxvalue, cv::THRESH_BINARY);

  std::vector<std::vector<std::vector<int>>> boxes =
      boxes_from_bitmap(pred_map, bit_map);
  float ratio_h = output_height * 1.0f / origin.rows;
  float ratio_w = output_width * 1.0f / origin.cols;
  std::vector<std::vector<std::vector<int>>> filter_boxes =
      filter_tag_det_res(boxes, ratio_h, ratio_w, origin);
  return filter_boxes;
}

std::vector<int>
OCR_PPredictor::postprocess_rec_word_index(const PredictorOutput &res) {
  const int *rec_idx = res.get_int_data();
  const std::vector<std::vector<uint64_t>> rec_idx_lod = res.get_lod();

  std::vector<int> pred_idx;
  for (int n = int(rec_idx_lod[0][0]); n < int(rec_idx_lod[0][1] * 2); n += 2) {
    pred_idx.emplace_back(rec_idx[n]);
  }
  return pred_idx;
}

float OCR_PPredictor::postprocess_rec_score(const PredictorOutput &res) {
  const float *predict_batch = res.get_float_data();
  const std::vector<int64_t> predict_shape = res.get_shape();
  const std::vector<std::vector<uint64_t>> predict_lod = res.get_lod();
  int blank = predict_shape[1];
  float score = 0.f;
  int count = 0;
  for (int n = predict_lod[0][0]; n < predict_lod[0][1] - 1; n++) {
    int argmax_idx = argmax(predict_batch + n * predict_shape[1],
                            predict_batch + (n + 1) * predict_shape[1]);
    float max_value = predict_batch[n * predict_shape[1] + argmax_idx];
    if (blank - 1 - argmax_idx > 1e-5) {
      score += max_value;
      count += 1;
    }
  }
  if (count == 0) {
    LOGE("calc score count 0");
  } else {
    score /= count;
  }
  LOGI("calc score: %f", score);
  return score;
}

NET_TYPE OCR_PPredictor::get_net_flag() const { return NET_OCR; }
} // namespace ppredictor
