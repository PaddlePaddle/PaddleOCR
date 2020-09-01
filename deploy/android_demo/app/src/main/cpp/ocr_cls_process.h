//
// Created by fujiayi on 2020/7/3.
//
#pragma once

#include "common.h"
#include <opencv2/opencv.hpp>
#include <vector>

extern const std::vector<int> CLS_IMAGE_SHAPE;

cv::Mat cls_resize_img(const cv::Mat &img);