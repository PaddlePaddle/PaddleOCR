#ifdef OCR_EXPORTS

#include "include/paddle_ocr.h"

#include <memory>
#include <vector>

#include "include/ocr_cls.h"
#include "include/ocr_defines.h"
#include "include/ocr_det.h"
#include "include/ocr_rec.h"

struct paddle_ocr_t {
    std::unique_ptr<PaddleOCR::DBDetector> det_ptr = nullptr;
    std::unique_ptr<PaddleOCR::CRNNRecognizer> rec_ptr = nullptr;
    std::unique_ptr<PaddleOCR::Classifier> cls_ptr = nullptr;
};

cv::Mat decode(const uint8_t* buffer, size_t size)
{
    std::vector<uint8_t> buf(buffer, buffer + size);
    return cv::imdecode(buf, 1);
}

void stringvec_to_charpp(const std::vector<std::string>& stringvec, char** out_strs, size_t* out_size)
{
    for (size_t i = 0; i != stringvec.size(); ++i) {
        strcpy(*(out_strs + i), stringvec[i].data());
    }
    *out_size = stringvec.size();
}

template <typename T>
void vector2pointer(const std::vector<T>& vec, T* out, size_t* out_size)
{
    if (out) {
        memcpy(out, vec.data(), vec.size() * sizeof(T));
        if (out_size) {
            *out_size = vec.size();
        }
    }
    else if (out_size) {
        *out_size = 0;
    }
}

paddle_ocr_t* PaddleOcrCreate(const char* det_model_dir, const char* rec_model_dir,
    const char* char_list_file, const char* cls_model_dir)
{
    paddle_ocr_t* ocr_ptr = new paddle_ocr_t();

    if (det_model_dir) {
        ocr_ptr->det_ptr = std::make_unique<PaddleOCR::DBDetector>(
            det_model_dir,
            FLAGS_use_gpu, FLAGS_gpu_id,
            FLAGS_gpu_mem, FLAGS_cpu_threads,
            FLAGS_enable_mkldnn, FLAGS_max_side_len, FLAGS_det_db_thresh,
            FLAGS_det_db_box_thresh, FLAGS_det_db_unclip_ratio,
            FLAGS_use_polygon_score, FLAGS_visualize,
            FLAGS_use_tensorrt, FLAGS_precision);
    }
    if (rec_model_dir && char_list_file) {
        ocr_ptr->rec_ptr = std::make_unique<PaddleOCR::CRNNRecognizer>(
            rec_model_dir,
            FLAGS_use_gpu, FLAGS_gpu_id,
            FLAGS_gpu_mem, FLAGS_cpu_threads,
            FLAGS_enable_mkldnn, char_list_file,
            FLAGS_use_tensorrt, FLAGS_precision);
    }
    if (cls_model_dir) {
        ocr_ptr->cls_ptr = std::make_unique<PaddleOCR::Classifier>(
            cls_model_dir,
            FLAGS_use_gpu, FLAGS_gpu_id,
            FLAGS_gpu_mem, FLAGS_cpu_threads,
            FLAGS_enable_mkldnn, FLAGS_cls_thresh,
            FLAGS_use_tensorrt, FLAGS_precision);
    }

    return ocr_ptr;
}

void PaddleOcrDestroy(paddle_ocr_t* ocr_ptr)
{
    if (ocr_ptr == nullptr) {
        return;
    }
    delete ocr_ptr;
    ocr_ptr = nullptr;
}

OCR_ERROR _PaddleOcrDet(
    paddle_ocr_t* ocr_ptr, cv::Mat& srcimg,
    int* out_boxes, size_t* out_boxes_size,
    double* out_times, size_t* out_times_size)
{
    std::vector<std::vector<std::vector<int>>> boxes;
    std::vector<double> det_times;

    ocr_ptr->det_ptr->Run(srcimg, boxes, &det_times);

    *out_boxes_size = boxes.size();
    // each box has 8 values
    for (size_t i = 0; i != boxes.size(); ++i) {
        for (size_t j = 0; j != 4; ++j) {
            *(out_boxes + i * 8 + j * 2) = boxes[i][j][0];
            *(out_boxes + i * 8 + j * 2 + 1) = boxes[i][j][1];
        }
    }

    vector2pointer(det_times, out_times, out_times_size);

    return OCR_SUCCESS;
}

OCR_ERROR PaddleOcrDet(
    paddle_ocr_t* ocr_ptr, const uint8_t* encode_buf, size_t encode_buf_size,
    int* out_boxes, size_t* out_boxes_size,
    double* out_times, size_t* out_times_size)
{
    if (ocr_ptr == nullptr
        || encode_buf == nullptr
        || out_boxes == nullptr
        || out_boxes_size == nullptr) {
        return OCR_FAILURE;
    }

    cv::Mat srcimg = decode(encode_buf, encode_buf_size);
    if (srcimg.empty()) {
        return OCR_FAILURE;
    }

    return _PaddleOcrDet(ocr_ptr, srcimg,
        out_boxes, out_boxes_size,
        out_times, out_times_size);
}

OCR_ERROR PaddleOcrDetWithData(
    paddle_ocr_t* ocr_ptr, int rows, int cols, int type, void* data,
    int* out_boxes, size_t* out_boxes_size,
    double* out_times, size_t* out_times_size)
{
    if (ocr_ptr == nullptr
        || data == nullptr
        || out_boxes == nullptr
        || out_boxes_size == nullptr) {
        return OCR_FAILURE;
    }

    cv::Mat srcimg(rows, cols, type, data);
    if (srcimg.empty()) {
        return OCR_FAILURE;
    }

    return _PaddleOcrDet(ocr_ptr, srcimg,
        out_boxes, out_boxes_size,
        out_times, out_times_size);
}

OCR_ERROR _PaddleOcrRec(
    paddle_ocr_t* ocr_ptr, cv::Mat& srcimg,
    char** out_strs, float* out_scores, size_t* out_size,
    double* out_times, size_t* out_times_size)
{
    std::vector<string> strs_res;
    std::vector<float> scores;
    std::vector<double> rec_times;

    ocr_ptr->rec_ptr->Run(srcimg, strs_res, scores, &rec_times);

    stringvec_to_charpp(strs_res, out_strs, out_size);
    memcpy(out_scores, scores.data(), scores.size() * sizeof(float));
    vector2pointer(rec_times, out_times, out_times_size);

    return OCR_SUCCESS;
}

OCR_ERROR PaddleOcrRec(
    paddle_ocr_t* ocr_ptr, const uint8_t* encode_buf, size_t encode_buf_size,
    char** out_strs, float* out_scores, size_t* out_size,
    double* out_times, size_t* out_times_size)
{
    if (ocr_ptr == nullptr
        || encode_buf == nullptr
        || out_strs == nullptr
        || out_scores == nullptr
        || out_size == nullptr) {
        return OCR_FAILURE;
    }

    cv::Mat srcimg = decode(encode_buf, encode_buf_size);
    if (srcimg.empty()) {
        return OCR_FAILURE;
    }

    return _PaddleOcrRec(ocr_ptr, srcimg,
        out_strs, out_scores, out_size,
        out_times, out_times_size);
}

OCR_ERROR PaddleOcrRecWithData(
    paddle_ocr_t* ocr_ptr, int rows, int cols, int type, void* data,
    char** out_strs, float* out_scores, size_t* out_size,
    double* out_times, size_t* out_times_size)
{
    if (ocr_ptr == nullptr
        || data == nullptr
        || out_strs == nullptr
        || out_scores == nullptr
        || out_size == nullptr) {
        return OCR_FAILURE;
    }

    cv::Mat srcimg(rows, cols, type, data);
    if (srcimg.empty()) {
        return OCR_FAILURE;
    }

    return _PaddleOcrRec(ocr_ptr, srcimg,
        out_strs, out_scores, out_size,
        out_times, out_times_size);
}

OCR_ERROR _PaddleOcrSystem(
    paddle_ocr_t* ocr_ptr, cv::Mat& srcimg,
    bool with_cls,
    int* out_boxes, char** out_strs, float* out_scores, size_t* out_size,
    double* out_times, size_t* out_times_size)
{
    std::vector<std::vector<std::vector<int>>> boxes;
    std::vector<double> all_times;

    ocr_ptr->det_ptr->Run(srcimg, boxes, &all_times);

    *out_size = boxes.size();
    // each box has 8 values
    for (size_t i = 0; i != boxes.size(); ++i) {
        for (size_t j = 0; j != 4; ++j) {
            *(out_boxes + i * 8 + j * 2) = boxes[i][j][0];
            *(out_boxes + i * 8 + j * 2 + 1) = boxes[i][j][1];
        }
    }

    std::vector<string> strs_res;
    std::vector<float> scores;

    for (const auto& box : boxes) {
        cv::Mat crop = PaddleOCR::Utility::GetRotateCropImage(srcimg, box);
        if (ocr_ptr->cls_ptr && with_cls) {
            ocr_ptr->cls_ptr->Run(crop);
        }
        ocr_ptr->rec_ptr->Run(crop, strs_res, scores, &all_times);
    }

    stringvec_to_charpp(strs_res, out_strs, out_size);
    memcpy(out_scores, scores.data(), scores.size() * sizeof(float));

    vector2pointer(all_times, out_times, out_times_size);

    return OCR_SUCCESS;
}

OCR_ERROR PaddleOcrSystem(
    paddle_ocr_t* ocr_ptr, const uint8_t* encode_buf, size_t encode_buf_size,
    bool with_cls,
    int* out_boxes, char** out_strs, float* out_scores, size_t* out_size,
    double* out_times, size_t* out_times_size)
{
    if (ocr_ptr == nullptr
        || encode_buf == nullptr
        || out_boxes == nullptr
        || out_strs == nullptr
        || out_scores == nullptr
        || out_size == nullptr) {
        return OCR_FAILURE;
    }

    cv::Mat srcimg = decode(encode_buf, encode_buf_size);
    if (srcimg.empty()) {
        return OCR_FAILURE;
    }

    return _PaddleOcrSystem(ocr_ptr, srcimg, with_cls,
        out_boxes, out_strs, out_scores, out_size,
        out_times, out_times_size);
}

OCR_ERROR OCRAPI PaddleOcrSystemWithData(
    paddle_ocr_t* ocr_ptr, int rows, int cols, int type, void* data,
    bool with_cls,
    int* out_boxes, char** out_strs, float* out_scores, size_t* out_size,
    double* out_times, size_t* out_times_size)
{
    if (ocr_ptr == nullptr
        || data == nullptr
        || out_boxes == nullptr
        || out_strs == nullptr
        || out_scores == nullptr
        || out_size == nullptr) {
        return OCR_FAILURE;
    }

    cv::Mat srcimg(rows, cols, type, data);
    if (srcimg.empty()) {
        return OCR_FAILURE;
    }

    return _PaddleOcrSystem(ocr_ptr, srcimg, with_cls,
        out_boxes, out_strs, out_scores, out_size,
        out_times, out_times_size);
}

#endif // OCR_EXPORTS