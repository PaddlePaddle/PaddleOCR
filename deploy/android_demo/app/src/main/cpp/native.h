//
// Created by fujiayi on 2020/7/5.
//

#pragma once

#include "common.h"
#include <android/bitmap.h>
#include <jni.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

inline std::string jstring_to_cpp_string(JNIEnv *env, jstring jstr) {
  // In java, a unicode char will be encoded using 2 bytes (utf16).
  // so jstring will contain characters utf16. std::string in c++ is
  // essentially a string of bytes, not characters, so if we want to
  // pass jstring from JNI to c++, we have convert utf16 to bytes.
  if (!jstr) {
    return "";
  }
  const jclass stringClass = env->GetObjectClass(jstr);
  const jmethodID getBytes =
      env->GetMethodID(stringClass, "getBytes", "(Ljava/lang/String;)[B");
  const jbyteArray stringJbytes = (jbyteArray)env->CallObjectMethod(
      jstr, getBytes, env->NewStringUTF("UTF-8"));

  size_t length = (size_t)env->GetArrayLength(stringJbytes);
  jbyte *pBytes = env->GetByteArrayElements(stringJbytes, NULL);

  std::string ret = std::string(reinterpret_cast<char *>(pBytes), length);
  env->ReleaseByteArrayElements(stringJbytes, pBytes, JNI_ABORT);

  env->DeleteLocalRef(stringJbytes);
  env->DeleteLocalRef(stringClass);
  return ret;
}

inline jstring cpp_string_to_jstring(JNIEnv *env, std::string str) {
  auto *data = str.c_str();
  jclass strClass = env->FindClass("java/lang/String");
  jmethodID strClassInitMethodID =
      env->GetMethodID(strClass, "<init>", "([BLjava/lang/String;)V");

  jbyteArray bytes = env->NewByteArray(strlen(data));
  env->SetByteArrayRegion(bytes, 0, strlen(data),
                          reinterpret_cast<const jbyte *>(data));

  jstring encoding = env->NewStringUTF("UTF-8");
  jstring res = (jstring)(env->NewObject(strClass, strClassInitMethodID, bytes,
                                         encoding));

  env->DeleteLocalRef(strClass);
  env->DeleteLocalRef(encoding);
  env->DeleteLocalRef(bytes);

  return res;
}

inline jfloatArray cpp_array_to_jfloatarray(JNIEnv *env, const float *buf,
                                            int64_t len) {
  if (len == 0) {
    return env->NewFloatArray(0);
  }
  jfloatArray result = env->NewFloatArray(len);
  env->SetFloatArrayRegion(result, 0, len, buf);
  return result;
}

inline jintArray cpp_array_to_jintarray(JNIEnv *env, const int *buf,
                                        int64_t len) {
  jintArray result = env->NewIntArray(len);
  env->SetIntArrayRegion(result, 0, len, buf);
  return result;
}

inline jbyteArray cpp_array_to_jbytearray(JNIEnv *env, const int8_t *buf,
                                          int64_t len) {
  jbyteArray result = env->NewByteArray(len);
  env->SetByteArrayRegion(result, 0, len, buf);
  return result;
}

inline jlongArray int64_vector_to_jlongarray(JNIEnv *env,
                                             const std::vector<int64_t> &vec) {
  jlongArray result = env->NewLongArray(vec.size());
  jlong *buf = new jlong[vec.size()];
  for (size_t i = 0; i < vec.size(); ++i) {
    buf[i] = (jlong)vec[i];
  }
  env->SetLongArrayRegion(result, 0, vec.size(), buf);
  delete[] buf;
  return result;
}

inline std::vector<int64_t> jlongarray_to_int64_vector(JNIEnv *env,
                                                       jlongArray data) {
  int data_size = env->GetArrayLength(data);
  jlong *data_ptr = env->GetLongArrayElements(data, nullptr);
  std::vector<int64_t> data_vec(data_ptr, data_ptr + data_size);
  env->ReleaseLongArrayElements(data, data_ptr, 0);
  return data_vec;
}

inline std::vector<float> jfloatarray_to_float_vector(JNIEnv *env,
                                                      jfloatArray data) {
  int data_size = env->GetArrayLength(data);
  jfloat *data_ptr = env->GetFloatArrayElements(data, nullptr);
  std::vector<float> data_vec(data_ptr, data_ptr + data_size);
  env->ReleaseFloatArrayElements(data, data_ptr, 0);
  return data_vec;
}

inline cv::Mat bitmap_to_cv_mat(JNIEnv *env, jobject bitmap) {
  AndroidBitmapInfo info;
  int result = AndroidBitmap_getInfo(env, bitmap, &info);
  if (result != ANDROID_BITMAP_RESULT_SUCCESS) {
    LOGE("AndroidBitmap_getInfo failed, result: %d", result);
    return cv::Mat{};
  }
  if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
    LOGE("Bitmap format is not RGBA_8888 !");
    return cv::Mat{};
  }
  unsigned char *srcData = NULL;
  AndroidBitmap_lockPixels(env, bitmap, (void **)&srcData);
  cv::Mat mat = cv::Mat::zeros(info.height, info.width, CV_8UC4);
  memcpy(mat.data, srcData, info.height * info.width * 4);
  AndroidBitmap_unlockPixels(env, bitmap);
  cv::cvtColor(mat, mat, cv::COLOR_RGBA2BGR);
  /**
  if (!cv::imwrite("/sdcard/1/copy.jpg", mat)){
      LOGE("Write image failed " );
  }
   */
  return mat;
}
