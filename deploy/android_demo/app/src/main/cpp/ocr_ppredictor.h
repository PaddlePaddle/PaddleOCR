//
// Created by fujiayi on 2020/7/1.
//

#pragma once

#include "ppredictor.h"
#include <opencv2/opencv.hpp>
#include <paddle_api.h>
#include <string>

namespace ppredictor {

/**
 * Config
 */
struct OCR_Config {
  int use_opencl = 0;
  int thread_num = 4; // Thread num
  paddle::lite_api::PowerMode mode =
      paddle::lite_api::LITE_POWER_HIGH; // PaddleLite Mode
};

/**
 * Polygons Result
 */
struct OCRPredictResult {
  std::vector<int> word_index;
  std::vector<std::vector<int>> points;
  float score;
  float cls_score;
  int cls_label = -1;
};

struct ClsPredictResult {
  float cls_score;
  int cls_label = -1;
  cv::Mat img;
};
/**
 * OCR there are 2 models
 * 1. First model（det），select polygons to show where are the texts
 * 2. crop from the origin images, use these polygons to infer
 */
class OCR_PPredictor : public PPredictor_Interface {
public:
  OCR_PPredictor(const OCR_Config &config);

  virtual ~OCR_PPredictor() {}

  /**
   * 初始化二个模型的Predictor
   * @param det_model_content
   * @param rec_model_content
   * @return
   */
  int init(const std::string &det_model_content,
           const std::string &rec_model_content,
           const std::string &cls_model_content);
  int init_from_file(const std::string &det_model_path,
                     const std::string &rec_model_path,
                     const std::string &cls_model_path);
  /**
   * Return OCR result
   * @param dims
   * @param input_data
   * @param input_len
   * @param net_flag
   * @param origin
   * @return
   */
  virtual std::vector<OCRPredictResult> infer_ocr(cv::Mat &origin,
                                                  int max_size_len, int run_det,
                                                  int run_cls, int run_rec);

  virtual NET_TYPE get_net_flag() const;

private:
  /**
   * calculate polygons from the result image of first model
   * @param pred
   * @param output_height
   * @param output_width
   * @param origin
   * @return
   */
  std::vector<std::vector<std::vector<int>>>
  calc_filtered_boxes(const float *pred, int pred_size, int output_height,
                      int output_width, const cv::Mat &origin);

  void infer_det(cv::Mat &origin, int max_side_len,
                 std::vector<OCRPredictResult> &ocr_results);
  /**
   * infer for rec model
   *
   * @param boxes
   * @param origin
   * @return
   */
  void infer_rec(const cv::Mat &origin, int run_cls,
                 OCRPredictResult &ocr_result);

  /**
   * infer for cls model
   *
   * @param boxes
   * @param origin
   * @return
   */
  ClsPredictResult infer_cls(const cv::Mat &origin, float thresh = 0.9);

  /**
   * Postprocess or second model to extract text
   * @param res
   * @return
   */
  std::vector<int> postprocess_rec_word_index(const PredictorOutput &res);

  /**
   * calculate confidence of second model text result
   * @param res
   * @return
   */
  float postprocess_rec_score(const PredictorOutput &res);

  std::unique_ptr<PPredictor> _det_predictor;
  std::unique_ptr<PPredictor> _rec_predictor;
  std::unique_ptr<PPredictor> _cls_predictor;
  OCR_Config _config;
};
} // namespace ppredictor
