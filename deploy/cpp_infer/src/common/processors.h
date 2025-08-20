// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "polyclipping/clipper.hpp"
#include "src/utils/func_register.h"

class Resize : public BaseProcessor {
public:
  Resize(const std::vector<int> &target_size, bool keep_ratio = false,
         int size_divisor = 0, const std::string &interp = "LINEAR");
  absl::Status CheckImageSize() const;
  std::pair<std::vector<int>, double>
  RescaleSize(const std::vector<int> &img_size) const;
  absl::StatusOr<std::vector<cv::Mat>>
  Apply(std::vector<cv::Mat> &input,
        const void *param = nullptr) const override;
  absl::StatusOr<cv::Mat> ResizeOne(const cv::Mat &img) const;
  static absl::StatusOr<int> GetInterp(const std::string &interp);

private:
  std::vector<int> target_size_;
  bool keep_ratio_;
  int size_divisor_;
  int interp_;
};

class ResizeByShort : public BaseProcessor {
public:
  ResizeByShort(int target_short_edge, int size_divisor = 0,
                const std::string &interp = "LINEAR");
  absl::StatusOr<std::vector<cv::Mat>>
  Apply(std::vector<cv::Mat> &input,
        const void *param = nullptr) const override;
  absl::StatusOr<cv::Mat> ResizeOne(const cv::Mat &img) const;

private:
  int target_short_edge_;
  int size_divisor_;
  int interp_;
};

class ReadImage : public BaseProcessor {
public:
  enum class Format { BGR, RGB, GRAY };

  ReadImage(const std::string &format = "RGB");

  ReadImage(const ReadImage &) = delete;
  ReadImage &operator=(const ReadImage &) = delete;

  absl::StatusOr<std::vector<cv::Mat>>
  Apply(std::vector<cv::Mat> &input,
        const void *param_ptr = nullptr) const override;

private:
  static absl::StatusOr<Format> StringToFormat(const std::string &format);
  Format format_;
};

class Normalize : public BaseProcessor {
public:
  Normalize(float scale = 1.0 / 255.0,
            const std::vector<float> &mean = {0.5, 0.5, 0.5},
            const std::vector<float> &std = {0.5, 0.5, 0.5});
  Normalize(float scale = 1.0 / 255.0, const float &mean = 0.5,
            const float &std = 0.5);
  absl::StatusOr<cv::Mat> NormalizeOne(const cv::Mat &input) const;
  absl::StatusOr<std::vector<cv::Mat>>
  Apply(std::vector<cv::Mat> &input,
        const void *param = nullptr) const override;
  static constexpr int CHANNEL = 3;

private:
  std::vector<float> alpha_;
  std::vector<float> beta_;
};

class NormalizeImage : public BaseProcessor {
public:
  NormalizeImage(float scale = 1.0 / 255.0,
                 const std::vector<float> &mean = {0.485, 0.456, 0.406},
                 const std::vector<float> &std = {0.229, 0.224, 0.225});

  absl::StatusOr<std::vector<cv::Mat>>
  Apply(std::vector<cv::Mat> &input,
        const void *param = nullptr) const override;

private:
  std::vector<float> alpha_;
  std::vector<float> beta_;

  absl::StatusOr<cv::Mat> Normalize(const cv::Mat &img) const;
  NormalizeImage(const NormalizeImage &) = delete;
  NormalizeImage &operator=(const NormalizeImage &) = delete;
  static constexpr int CHANNEL = 3;
};

class ToCHWImage : public BaseProcessor {
public:
  absl::StatusOr<std::vector<cv::Mat>>
  operator()(const std::vector<cv::Mat> &imgs_batch);
  absl::StatusOr<std::vector<cv::Mat>>
  Apply(std::vector<cv::Mat> &input,
        const void *param = nullptr) const override;
};

class ToBatch : public BaseProcessor {
public:
  absl::StatusOr<std::vector<cv::Mat>>
  operator()(const std::vector<cv::Mat> &imgs) const;
  absl::StatusOr<std::vector<cv::Mat>>
  Apply(std::vector<cv::Mat> &input,
        const void *param = nullptr) const override;
};

class ComponentsProcessor {
public:
  static absl::StatusOr<cv::Mat> RotateImage(const cv::Mat &image, int angle);
  static std::vector<std::vector<cv::Point2f>>
  SortQuadBoxes(const std::vector<std::vector<cv::Point2f>> &dt_polys);
  static std::vector<std::vector<cv::Point2f>>
  SortPolyBoxes(const std::vector<std::vector<cv::Point2f>> &dt_polys);
  static std::vector<std::array<float, 4>>
  ConvertPointsToBoxes(const std::vector<std::vector<cv::Point2f>> &dt_polys);
};

class CropByPolys {
public:
  enum class DetBoxType { kQuad, kPoly };

  CropByPolys(const std::string &box_type = "quad");

  absl::StatusOr<std::vector<cv::Mat>>
  operator()(const cv::Mat &img,
             const std::vector<std::vector<cv::Point2f>> &dt_polys);

  absl::StatusOr<cv::Mat>
  GetMinAreaRectCrop(const cv::Mat &img,
                     const std::vector<cv::Point2f> &points) const;

  absl::StatusOr<cv::Mat>
  GetPolyRectCrop(const cv::Mat &img,
                  const std::vector<cv::Point2f> &poly) const;

  absl::StatusOr<cv::Mat>
  GetRotateCropImage(const cv::Mat &img,
                     const std::vector<cv::Point2f> &box) const;

  std::vector<cv::Point2f>
  GetMinAreaRectPoints(const std::vector<cv::Point2f> &poly) const;

  static double IoU(const std::vector<cv::Point2f> &poly1,
                    const std::vector<cv::Point2f> &poly2);

  static ClipperLib::Path
  CvPolyToClipperPath(const std::vector<cv::Point2f> &poly);

  static const double SCALE;

private:
  DetBoxType box_type_;
};
