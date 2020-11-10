//
// Created by fujiayi on 2020/7/3.
//
#pragma once

#include <vector>
#include <opencv2/opencv.hpp>
#include "common.h"

extern const std::vector<int> REC_IMAGE_SHAPE;

cv::Mat get_rotate_crop_image(const cv::Mat &srcimage, const std::vector<std::vector<int>> &box);

cv::Mat crnn_resize_img(const cv::Mat &img, float wh_ratio);

template<class ForwardIterator>
inline size_t argmax(ForwardIterator first, ForwardIterator last) {
    return std::distance(first, std::max_element(first, last));
}