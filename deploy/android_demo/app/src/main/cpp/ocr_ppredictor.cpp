//
// Created by fujiayi on 2020/7/1.
//

#include "ocr_ppredictor.h"
#include "preprocess.h"
#include "common.h"
#include "ocr_db_post_process.h"
#include "ocr_crnn_process.h"

namespace ppredictor {

OCR_PPredictor::OCR_PPredictor(const OCR_Config &config) : _config(config) {

}

int
OCR_PPredictor::init(const std::string &det_model_content, const std::string &rec_model_content) {
    _det_predictor = std::unique_ptr<PPredictor>(
        new PPredictor{_config.thread_num, NET_OCR, _config.mode});
    _det_predictor->init_nb(det_model_content);

    _rec_predictor = std::unique_ptr<PPredictor>(
        new PPredictor{_config.thread_num, NET_OCR_INTERNAL, _config.mode});
    _rec_predictor->init_nb(rec_model_content);
    return RETURN_OK;
}

int OCR_PPredictor::init_from_file(const std::string &det_model_path, const std::string &rec_model_path){
    _det_predictor = std::unique_ptr<PPredictor>(
        new PPredictor{_config.thread_num, NET_OCR, _config.mode});
    _det_predictor->init_from_file(det_model_path);

    _rec_predictor = std::unique_ptr<PPredictor>(
        new PPredictor{_config.thread_num, NET_OCR_INTERNAL, _config.mode});
    _rec_predictor->init_from_file(rec_model_path);
    return RETURN_OK;
}
/**
 * for debug use, show result of First Step
 * @param filter_boxes
 * @param boxes
 * @param srcimg
 */
static void visual_img(const std::vector<std::vector<std::vector<int>>> &filter_boxes,
                       const std::vector<std::vector<std::vector<int>>> &boxes,
                       const cv::Mat &srcimg) {
    // visualization
    cv::Point rook_points[filter_boxes.size()][4];
    for (int n = 0; n < filter_boxes.size(); n++) {
        for (int m = 0; m < filter_boxes[0].size(); m++) {
            rook_points[n][m] = cv::Point(int(filter_boxes[n][m][0]), int(filter_boxes[n][m][1]));
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
OCR_PPredictor::infer_ocr(const std::vector<int64_t> &dims, const float *input_data, int input_len,
                          int net_flag, cv::Mat &origin) {

    PredictorInput input = _det_predictor->get_first_input();
    input.set_dims(dims);
    input.set_data(input_data, input_len);
    std::vector<PredictorOutput> results = _det_predictor->infer();
    PredictorOutput &res = results.at(0);
    std::vector<std::vector<std::vector<int>>> filtered_box
        = calc_filtered_boxes(res.get_float_data(), res.get_size(), (int) dims[2], (int) dims[3],
                              origin);
    LOGI("Filter_box size %ld", filtered_box.size());
    return infer_rec(filtered_box, origin);
}

std::vector<OCRPredictResult>
OCR_PPredictor::infer_rec(const std::vector<std::vector<std::vector<int>>> &boxes,
                          const cv::Mat &origin_img) {
    std::vector<float> mean = {0.5f, 0.5f, 0.5f};
    std::vector<float> scale = {1 / 0.5f, 1 / 0.5f, 1 / 0.5f};
    std::vector<int64_t> dims = {1, 3, 0, 0};
    std::vector<OCRPredictResult> ocr_results;

    PredictorInput input = _rec_predictor->get_first_input();
    for (auto bp = boxes.crbegin(); bp != boxes.crend(); ++bp) {
        const std::vector<std::vector<int>> &box = *bp;
        cv::Mat crop_img = get_rotate_crop_image(origin_img, box);
        float wh_ratio = float(crop_img.cols) / float(crop_img.rows);
        cv::Mat input_image = crnn_resize_img(crop_img, wh_ratio);
        input_image.convertTo(input_image, CV_32FC3, 1 / 255.0f);
        const float *dimg = reinterpret_cast<const float *>(input_image.data);
        int input_size = input_image.rows * input_image.cols;

        dims[2] = input_image.rows;
        dims[3] = input_image.cols;
        input.set_dims(dims);

        neon_mean_scale(dimg, input.get_mutable_float_data(), input_size, mean, scale);

        std::vector<PredictorOutput> results = _rec_predictor->infer();

        OCRPredictResult res;
        res.word_index = postprocess_rec_word_index(results.at(0));
        if (res.word_index.empty()) {
            continue;
        }
        res.score = postprocess_rec_score(results.at(1));
        res.points = box;
        ocr_results.emplace_back(std::move(res));
    }
    LOGI("ocr_results finished %lu", ocr_results.size());
    return ocr_results;
}

std::vector<std::vector<std::vector<int>>>
OCR_PPredictor::calc_filtered_boxes(const float *pred, int pred_size, int output_height,
                                    int output_width, const cv::Mat &origin) {
    const double threshold = 0.3;
    const double maxvalue = 1;

    cv::Mat pred_map = cv::Mat::zeros(output_height, output_width, CV_32F);
    memcpy(pred_map.data, pred, pred_size * sizeof(float));
    cv::Mat cbuf_map;
    pred_map.convertTo(cbuf_map, CV_8UC1);

    cv::Mat bit_map;
    cv::threshold(cbuf_map, bit_map, threshold, maxvalue, cv::THRESH_BINARY);

    std::vector<std::vector<std::vector<int>>> boxes = boxes_from_bitmap(pred_map, bit_map);
    float ratio_h = output_height * 1.0f / origin.rows;
    float ratio_w = output_width * 1.0f / origin.cols;
    std::vector<std::vector<std::vector<int>>> filter_boxes = filter_tag_det_res(boxes, ratio_h,
                                                                                 ratio_w, origin);
    return filter_boxes;
}

std::vector<int> OCR_PPredictor::postprocess_rec_word_index(const PredictorOutput &res) {
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


NET_TYPE OCR_PPredictor::get_net_flag() const {
    return NET_OCR;
}
}