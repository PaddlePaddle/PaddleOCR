//
// Created by fu on 4/25/18.
//

#pragma once
#import <numeric>
#import <vector>

#ifdef __ANDROID__

#include <android/log.h>

#define LOG_TAG "OCR_NDK"

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#else
#include <stdio.h>
#define LOGI(format, ...)                                                      \
  fprintf(stdout, "[" LOG_TAG "]" format "\n", ##__VA_ARGS__)
#define LOGW(format, ...)                                                      \
  fprintf(stdout, "[" LOG_TAG "]" format "\n", ##__VA_ARGS__)
#define LOGE(format, ...)                                                      \
  fprintf(stderr, "[" LOG_TAG "]Error: " format "\n", ##__VA_ARGS__)
#endif

enum RETURN_CODE { RETURN_OK = 0 };

enum NET_TYPE { NET_OCR = 900100, NET_OCR_INTERNAL = 991008 };

template <typename T> inline T product(const std::vector<T> &vec) {
  if (vec.empty()) {
    return 0;
  }
  return std::accumulate(vec.begin(), vec.end(), 1, std::multiplies<T>());
}
