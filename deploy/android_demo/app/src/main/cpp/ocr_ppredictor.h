//
// Created by fujiayi on 2020/7/1.
//

#pragma once

#include <string>
#include <opencv2/opencv.hpp>
#include <paddle_api.h>
#include "ppredictor.h"

namespace ppredictor {

/**
 * 配置
 */
struct OCR_Config {
    int thread_num = 4; // 线程数
    paddle::lite_api::PowerMode mode = paddle::lite_api::LITE_POWER_HIGH; // PaddleLite Mode
};

/**
 * 一个四边形内图片的推理结果,
 */
struct OCRPredictResult {
    std::vector<int> word_index; //
    std::vector<std::vector<int>> points;
    float score;
};

/**
 * OCR 一共有2个模型进行推理，
 * 1. 使用第一个模型（det），框选出多个四边形
 * 2. 从原图从抠出这些多边形，使用第二个模型（rec），获取文本
 */
class OCR_PPredictor : public PPredictor_Interface {
public:
    OCR_PPredictor(const OCR_Config &config);

    virtual ~OCR_PPredictor() {

    }

    /**
     * 初始化二个模型的Predictor
     * @param det_model_content
     * @param rec_model_content
     * @return
     */
    int init(const std::string &det_model_content, const std::string &rec_model_content);
    int init_from_file(const std::string &det_model_path, const std::string &rec_model_path);
    /**
     * 返回OCR结果
     * @param dims
     * @param input_data
     * @param input_len
     * @param net_flag
     * @param origin
     * @return
     */
    virtual std::vector<OCRPredictResult>
    infer_ocr(const std::vector<int64_t> &dims, const float *input_data, int input_len,
              int net_flag, cv::Mat &origin);


    virtual NET_TYPE get_net_flag() const;


private:

    /**
     * 从第一个模型的结果中计算有文字的四边形
     * @param pred
     * @param output_height
     * @param output_width
     * @param origin
     * @return
     */
    std::vector<std::vector<std::vector<int>>>
    calc_filtered_boxes(const float *pred, int pred_size, int output_height, int output_width,
                        const cv::Mat &origin);

    /**
     * 第二个模型的推理
     *
     * @param boxes
     * @param origin
     * @return
     */
    std::vector<OCRPredictResult>
    infer_rec(const std::vector<std::vector<std::vector<int>>> &boxes, const cv::Mat &origin);

    /**
     * 第二个模型提取文字的后处理
     * @param res
     * @return
     */
    std::vector<int> postprocess_rec_word_index(const PredictorOutput &res);

    /**
     * 计算第二个模型的文字的置信度
     * @param res
     * @return
     */
    float postprocess_rec_score(const PredictorOutput &res);

    std::unique_ptr<PPredictor> _det_predictor;
    std::unique_ptr<PPredictor> _rec_predictor;
    OCR_Config _config;

};
}
