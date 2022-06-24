//
// Created by fujiayi on 2020/7/5.
//

#include "native.h"
#include "ocr_ppredictor.h"
#include <algorithm>
#include <paddle_api.h>
#include <string>

static paddle::lite_api::PowerMode str_to_cpu_mode(const std::string &cpu_mode);

extern "C" JNIEXPORT jlong JNICALL
Java_com_baidu_paddle_lite_demo_ocr_OCRPredictorNative_init(
    JNIEnv *env, jobject thiz, jstring j_det_model_path,
    jstring j_rec_model_path, jstring j_cls_model_path, jint j_use_opencl, jint j_thread_num,
    jstring j_cpu_mode) {
  std::string det_model_path = jstring_to_cpp_string(env, j_det_model_path);
  std::string rec_model_path = jstring_to_cpp_string(env, j_rec_model_path);
  std::string cls_model_path = jstring_to_cpp_string(env, j_cls_model_path);
  int thread_num = j_thread_num;
  std::string cpu_mode = jstring_to_cpp_string(env, j_cpu_mode);
  ppredictor::OCR_Config conf;
  conf.use_opencl = j_use_opencl;
  conf.thread_num = thread_num;
  conf.mode = str_to_cpu_mode(cpu_mode);
  ppredictor::OCR_PPredictor *orc_predictor =
      new ppredictor::OCR_PPredictor{conf};
  orc_predictor->init_from_file(det_model_path, rec_model_path, cls_model_path);
  return reinterpret_cast<jlong>(orc_predictor);
}

/**
 * "LITE_POWER_HIGH" convert to paddle::lite_api::LITE_POWER_HIGH
 * @param cpu_mode
 * @return
 */
static paddle::lite_api::PowerMode
str_to_cpu_mode(const std::string &cpu_mode) {
  static std::map<std::string, paddle::lite_api::PowerMode> cpu_mode_map{
      {"LITE_POWER_HIGH", paddle::lite_api::LITE_POWER_HIGH},
      {"LITE_POWER_LOW", paddle::lite_api::LITE_POWER_HIGH},
      {"LITE_POWER_FULL", paddle::lite_api::LITE_POWER_FULL},
      {"LITE_POWER_NO_BIND", paddle::lite_api::LITE_POWER_NO_BIND},
      {"LITE_POWER_RAND_HIGH", paddle::lite_api::LITE_POWER_RAND_HIGH},
      {"LITE_POWER_RAND_LOW", paddle::lite_api::LITE_POWER_RAND_LOW}};
  std::string upper_key;
  std::transform(cpu_mode.cbegin(), cpu_mode.cend(), upper_key.begin(),
                 ::toupper);
  auto index = cpu_mode_map.find(upper_key.c_str());
  if (index == cpu_mode_map.end()) {
    LOGE("cpu_mode not found %s", upper_key.c_str());
    return paddle::lite_api::LITE_POWER_HIGH;
  } else {
    return index->second;
  }
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_baidu_paddle_lite_demo_ocr_OCRPredictorNative_forward(
    JNIEnv *env, jobject thiz, jlong java_pointer, jobject original_image,jint j_max_size_len, jint j_run_det, jint j_run_cls, jint j_run_rec) {
  LOGI("begin to run native forward");
  if (java_pointer == 0) {
    LOGE("JAVA pointer is NULL");
    return cpp_array_to_jfloatarray(env, nullptr, 0);
  }

  cv::Mat origin = bitmap_to_cv_mat(env, original_image);
  if (origin.size == 0) {
    LOGE("origin bitmap cannot convert to CV Mat");
    return cpp_array_to_jfloatarray(env, nullptr, 0);
  }

  int max_size_len = j_max_size_len;
  int run_det = j_run_det;
  int run_cls = j_run_cls;
  int run_rec = j_run_rec;

  ppredictor::OCR_PPredictor *ppredictor =
      (ppredictor::OCR_PPredictor *)java_pointer;
  std::vector<int64_t> dims_arr;
  std::vector<ppredictor::OCRPredictResult> results =
      ppredictor->infer_ocr(origin, max_size_len, run_det, run_cls, run_rec);
  LOGI("infer_ocr finished with boxes %ld", results.size());

  // 这里将std::vector<ppredictor::OCRPredictResult> 序列化成
  // float数组，传输到java层再反序列化
  std::vector<float> float_arr;
  for (const ppredictor::OCRPredictResult &r : results) {
    float_arr.push_back(r.points.size());
    float_arr.push_back(r.word_index.size());
    float_arr.push_back(r.score);
    // add det point
    for (const std::vector<int> &point : r.points) {
      float_arr.push_back(point.at(0));
      float_arr.push_back(point.at(1));
    }
    // add rec word idx
    for (int index : r.word_index) {
      float_arr.push_back(index);
    }
    // add cls result
    float_arr.push_back(r.cls_label);
    float_arr.push_back(r.cls_score);
  }
  return cpp_array_to_jfloatarray(env, float_arr.data(), float_arr.size());
}

extern "C" JNIEXPORT void JNICALL
Java_com_baidu_paddle_lite_demo_ocr_OCRPredictorNative_release(
    JNIEnv *env, jobject thiz, jlong java_pointer) {
  if (java_pointer == 0) {
    LOGE("JAVA pointer is NULL");
    return;
  }
  ppredictor::OCR_PPredictor *ppredictor =
      (ppredictor::OCR_PPredictor *)java_pointer;
  delete ppredictor;
}
