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

#include "processors.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <unordered_map>

#include "src/utils/ilogger.h"
#include "src/utils/utility.h"

absl::StatusOr<int> Resize::GetInterp(const std::string &interp) {
  static const std::unordered_map<std::string, int> interp_map = {
      {"NEAREST", cv::INTER_NEAREST},
      {"LINEAR", cv::INTER_LINEAR},
      {"BICUBIC", cv::INTER_CUBIC},
      {"AREA", cv::INTER_AREA},
      {"LANCZOS4", cv::INTER_LANCZOS4}};
  auto it = interp_map.find(interp);
  if (it == interp_map.end())
    return -1;
  return it->second;
}

std::pair<std::vector<int>, double>
Resize::RescaleSize(const std::vector<int> &img_size) const {
  int img_w = img_size[0], img_h = img_size[1];
  int target_w = target_size_[0], target_h = target_size_[1];
  double scale = std::min(static_cast<double>(std::max(target_w, target_h)) /
                              std::max(img_w, img_h),
                          static_cast<double>(std::min(target_w, target_h)) /
                              std::min(img_w, img_h));
  std::vector<int> rescaled_size = {
      static_cast<int>(std::round(img_w * scale)),
      static_cast<int>(std::round(img_h * scale))};
  return std::make_pair(rescaled_size, scale);
}

absl::Status Resize::CheckImageSize() const {
  if (target_size_.size() != 2) {
    return absl::InvalidArgumentError("Size must be a vector of two elements.");
  }
  if (target_size_[0] <= 0 || target_size_[1] <= 0) {
    return absl::InvalidArgumentError("Width and height must be positive.");
  }
  return absl::OkStatus();
}

Resize::Resize(const std::vector<int> &target_size, bool keep_ratio,
               int size_divisor, const std::string &interp)
    : keep_ratio_(keep_ratio), size_divisor_(size_divisor) {
  if (target_size.size() == 1) {
    target_size_ = {target_size[0], target_size[0]};
  } else {
    target_size_ = target_size;
  }
  absl::Status status = CheckImageSize();
  if (!status.ok()) {
    INFOE("image check fail : %s", status.ToString().c_str());
    exit(-1);
  }
  std::string interp_upper = interp;
  std::transform(interp_upper.begin(), interp_upper.end(), interp_upper.begin(),
                 ::toupper);

  auto interp_value = GetInterp(interp_upper);
  if (!interp_value.ok()) {
    INFOE("Unknown type: %s", interp_value.status().ToString().c_str());
    exit(-1);
  }
  interp_ = interp_value.value();
}

absl::StatusOr<std::vector<cv::Mat>> Resize::Apply(std::vector<cv::Mat> &input,
                                                   const void *param) const {
  std::vector<cv::Mat> out_imgs;
  for (const auto &img : input) {
    auto out = ResizeOne(img);
    if (!out.ok())
      return out.status();
    out_imgs.push_back(std::move(out.value()));
  }
  return out_imgs;
}

absl::StatusOr<cv::Mat> Resize::ResizeOne(const cv::Mat &img) const {
  if (img.empty()) {
    return absl::InvalidArgumentError("Input image is empty.");
  }

  std::vector<int> cur_target = target_size_;
  auto size_test = img.size();
  cv::Size orig_size = img.size();
  int orig_w = orig_size.width, orig_h = orig_size.height;

  if (keep_ratio_) {
    std::vector<int> wh = {orig_w, orig_h};
    auto rescale = RescaleSize(wh);
    cur_target = rescale.first;
  }

  if (size_divisor_ > 0) {
    for (auto &x : cur_target) {
      x = static_cast<int>(std::ceil(static_cast<double>(x) / size_divisor_)) *
          size_divisor_;
    }
  }

  cv::Mat out;
  cv::resize(img, out, cv::Size(cur_target[0], cur_target[1]), 0, 0, interp_);
  return out;
}

ResizeByShort::ResizeByShort(int target_short_edge, int size_divisor,
                             const std::string &interp)
    : target_short_edge_(target_short_edge), size_divisor_(size_divisor) {
  std::string interp_upper = interp;
  std::transform(interp_upper.begin(), interp_upper.end(), interp_upper.begin(),
                 ::toupper);

  auto interp_value = Resize::GetInterp(interp_upper);
  if (!interp_value.ok()) {
    INFOE("Unknown type: %s", interp_value.status().ToString().c_str());
    exit(-1);
  }
  interp_ = interp_value.value();
}
absl::StatusOr<std::vector<cv::Mat>>
ResizeByShort::Apply(std::vector<cv::Mat> &input, const void *param) const {
  std::vector<cv::Mat> out_imgs;
  for (auto &image : input) {
    auto out = ResizeOne(image);
    if (!out.ok())
      return out.status();
    out_imgs.push_back(std::move(out.value()));
  }
  return out_imgs;
}

absl::StatusOr<cv::Mat> ResizeByShort::ResizeOne(const cv::Mat &img) const {
  if (img.empty()) {
    return absl::InvalidArgumentError("Input image is empty.");
  }
  int h = img.size[0];
  int w = img.size[1];
  int short_edge = std::min(h, w);
  float scale = static_cast<double>(target_short_edge_) / short_edge;
  int h_resize = static_cast<int>(std::round(h * scale));
  int w_resize = static_cast<int>(std::round(w * scale));

  if (size_divisor_ > 0) {
    h_resize = static_cast<int>(std::ceil(h_resize / (float)size_divisor_)) *
               size_divisor_;
    w_resize = static_cast<int>(std::ceil(w_resize / (float)size_divisor_)) *
               size_divisor_;
  }

  cv::Mat dst;
  cv::resize(img, dst, cv::Size(w_resize, h_resize), 0, 0, interp_);
  return dst;
}

ReadImage::ReadImage(const std::string &format) {
  auto fmt = StringToFormat(format);
  if (!fmt.ok()) {
    INFOE(fmt.status().ToString().c_str());
    exit(-1);
  }
  format_ = *fmt;
}

absl::StatusOr<std::vector<cv::Mat>>
ReadImage::Apply(std::vector<cv::Mat> &input, const void *param_ptr) const {
  if (input.empty()) {
    return absl::InvalidArgumentError("Input image vector is empty.");
  }
  std::vector<cv::Mat> output;
  output.reserve(input.size());

  for (size_t i = 0; i < input.size(); ++i) {
    const cv::Mat &img = input[i];
    if (img.empty()) {
      return absl::InvalidArgumentError("Image at index " + std::to_string(i) +
                                        " is empty.");
    }

    cv::Mat converted;
    switch (format_) {
    case Format::BGR:
      if (img.channels() == 3) {
        converted = img.clone();
      } else if (img.channels() == 1) {
        cv::cvtColor(img, converted, cv::COLOR_GRAY2BGR);
      } else {
        return absl::InvalidArgumentError("Image at index " +
                                          std::to_string(i) +
                                          " channel not supported for BGR.");
      }
      break;
    case Format::RGB:
      if (img.channels() == 3) {
        cv::cvtColor(img, converted, cv::COLOR_BGR2RGB);
      } else if (img.channels() == 1) {
        cv::cvtColor(img, converted, cv::COLOR_GRAY2RGB);
      } else {
        return absl::InvalidArgumentError("Image at index " +
                                          std::to_string(i) +
                                          " channel not supported for RGB.");
      }
      break;
    case Format::GRAY:
      if (img.channels() == 3) {
        cv::cvtColor(img, converted, cv::COLOR_BGR2GRAY);
      } else if (img.channels() == 1) {
        converted = img.clone();
      } else {
        return absl::InvalidArgumentError("Image at index " +
                                          std::to_string(i) +
                                          " channel not supported for GRAY.");
      }
      break;
    default:
      return absl::InvalidArgumentError("Unknown format.");
    }
    output.push_back(std::move(converted));
  }
  return output;
}

absl::StatusOr<ReadImage::Format>
ReadImage::StringToFormat(const std::string &format) {
  if (format == "BGR")
    return Format::BGR;
  if (format == "RGB")
    return Format::RGB;
  if (format == "GRAY")
    return Format::GRAY;
  return absl::InvalidArgumentError("Unsupported format: " + format);
}

absl::StatusOr<std::vector<cv::Mat>>
ToCHWImage::operator()(const std::vector<cv::Mat> &imgs_batch) {
  std::vector<std::vector<cv::Mat>> chw_imgs_batch;

  std::vector<cv::Mat> chw_imgs;
  for (const auto &img : imgs_batch) {
    if (img.empty()) {
      return absl::InvalidArgumentError("Input image is empty!");
    }
    if (img.channels() != 3) {
      return absl::InvalidArgumentError(
          "Input image must have 3 channels (HWC format)!");
    }

    cv::Mat chw_img(3, img.rows * img.cols, CV_32F);
    float *ptr = chw_img.ptr<float>();

    for (int h = 0; h < img.rows; ++h) {
      for (int w = 0; w < img.cols; ++w) {
        const cv::Vec3b &pixel = img.at<cv::Vec3b>(h, w);
        ptr[0 * img.total() + h * img.cols + w] = pixel[0];
        ptr[1 * img.total() + h * img.cols + w] = pixel[1];
        ptr[2 * img.total() + h * img.cols + w] = pixel[2];
      }
    }

    chw_imgs.push_back(chw_img);
  }

  return chw_imgs;
};

Normalize::Normalize(float scale, const std::vector<float> &mean,
                     const std::vector<float> &std)
    : alpha_(CHANNEL), beta_(CHANNEL) {
  assert(mean.size() == CHANNEL && std.size() == CHANNEL);
  for (size_t i = 0; i < CHANNEL; ++i) {
    alpha_[i] = scale / std.at(i);
    beta_[i] = -mean.at(i) / std.at(i);
  }
}
Normalize::Normalize(float scale, const float &mean, const float &std)
    : alpha_(CHANNEL), beta_(CHANNEL) {
  for (size_t i = 0; i < CHANNEL; ++i) {
    alpha_[i] = scale / std;
    beta_[i] = -mean / std;
  }
}

absl::StatusOr<cv::Mat> Normalize::NormalizeOne(const cv::Mat &image) const {
  if (image.empty()) {
    return absl::InvalidArgumentError("Input image is empty.");
  }
  if (image.channels() != CHANNEL) {
    return absl::InvalidArgumentError("Input image must have 3 dims");
  }
  if (image.depth() != CV_8U && image.depth() != CV_32F) {
    return absl::InvalidArgumentError("Input image must be CV_8U or CV_32F.");
  }
  cv::Mat input;
  if (image.depth() == CV_8U) {
    image.convertTo(input, CV_32F);
  } else {
    input = image.clone(); // note origin type is CV_8U
  }
  if (input.channels() == CHANNEL) {
    cv::Mat processed = input;
    std::vector<cv::Mat> channels(input.channels());
    cv::split(processed, channels);

    for (int c = 0; c < input.channels(); ++c) {
      channels[c] = channels[c] * alpha_[c] + beta_[c];
    }
    cv::merge(channels, processed);
    return processed;
  } else { // dims >= 3
    assert(input.isContinuous());
    int total = 1;
    for (int i = 0; i < input.dims - 1; i++) {
      total *= input.size[i];
    }
    float *data = input.ptr<float>();
    for (int i = 0; i < total; i++) {
      float *group = data + i * CHANNEL;
      for (int j = 0; j < CHANNEL; j++) {
        group[j] = group[j] * alpha_[j] + beta_[j];
      }
    }
    return input;
  }
}
absl::StatusOr<std::vector<cv::Mat>>
Normalize::Apply(std::vector<cv::Mat> &input, const void *param) const {
  std::vector<cv::Mat> results_norm;
  results_norm.reserve(input.size());
  for (const auto &img : input) {
    auto norm_single = NormalizeOne(img);
    if (!norm_single.ok()) {
      return norm_single.status();
    }
    results_norm.emplace_back(norm_single.value());
  }
  return results_norm;
}

NormalizeImage::NormalizeImage(float scale, const std::vector<float> &mean,
                               const std::vector<float> &std)
    : alpha_(CHANNEL), beta_(CHANNEL) {
  assert(mean.size() == CHANNEL && std.size() == CHANNEL);
  for (size_t i = 0; i < CHANNEL; ++i) {
    alpha_[i] = scale / std.at(i);
    beta_[i] = -mean.at(i) / std.at(i);
  }
}

absl::StatusOr<cv::Mat> NormalizeImage::Normalize(const cv::Mat &img) const {
  if (img.empty()) {
    return absl::InvalidArgumentError("Input image is empty.");
  }
  if (img.channels() != CHANNEL) {
    return absl::InvalidArgumentError("Input image must have 3 channels.");
  }
  if (img.depth() != CV_8U && img.depth() != CV_32F) {
    return absl::InvalidArgumentError("Input image must be CV_8U or CV_32F.");
  }

  cv::Mat input;
  if (img.depth() == CV_8U) {
    img.convertTo(input, CV_32F);
  } else {
    input = img.clone();
  }

  cv::Mat processed = input;

  std::vector<cv::Mat> channels(CHANNEL);

  cv::split(processed, channels);

  for (int c = 0; c < CHANNEL; ++c) {
    channels[c] = channels[c] * alpha_[c] + beta_[c];
  }

  cv::merge(channels, processed);
  return processed;
}

absl::StatusOr<std::vector<cv::Mat>>
NormalizeImage::Apply(std::vector<cv::Mat> &imgs, const void *param) const {
  std::vector<cv::Mat> results;
  results.reserve(imgs.size());
  for (const auto &img : imgs) {
    auto normed = this->Normalize(img);
    if (!normed.ok()) {
      return normed.status();
    }
    results.push_back(std::move(normed).value());
  }
  return results;
}

// absl::StatusOr<std::vector<cv::Mat>> ToCHWImage::Apply(
//   std::vector<cv::Mat>& input, const void* param) const {
//   std::vector<cv::Mat> chw_imgs;
//   for (const auto& img : input) {
//     if (img.empty()) {
//       return absl::InvalidArgumentError("Input image is empty!");
//     }
//     if (img.channels() != 3) {
//       return absl::InvalidArgumentError(
//           "Input image must have 3 channels (HWC format)!");
//     }

//     std::vector<int> shape_chw = {img.channels(), img.rows, img.cols};  //
//     Define sizes for CHW cv::Mat chw_img(shape_chw.size(), shape_chw.data(),
//     CV_32F); float* ptr = chw_img.ptr<float>(); for (int h = 0; h < img.rows;
//     ++h) {
//       for (int w = 0; w < img.cols; ++w) {
//         const cv::Vec3f& pixel = img.at<cv::Vec3f>(h, w);
//         ptr[0 * img.total() + h * img.cols + w] = pixel[0];
//         ptr[1 * img.total() + h * img.cols + w] = pixel[1];
//         ptr[2 * img.total() + h * img.cols + w] = pixel[2];
//       }
//     }

//     chw_imgs.push_back(chw_img);
//   }

//   return chw_imgs;
// }

absl::StatusOr<std::vector<cv::Mat>>
ToCHWImage::Apply(std::vector<cv::Mat> &input, const void *param) const {
  std::vector<cv::Mat> chw_imgs;
  for (const auto &img : input) {
    if (img.empty()) {
      return absl::InvalidArgumentError("Input image is empty!");
    }
    if (img.channels() != 3) {
      return absl::InvalidArgumentError(
          "Input image must have 3 channels (HWC format)!");
    }

    std::vector<cv::Mat> vec_split = {};
    cv::split(img, vec_split);
    cv::Mat chw_img;
    for (auto &split : vec_split)
      split = split.reshape(1, 1);
    cv::hconcat(vec_split, chw_img);
    std::vector<int> shape = {img.channels(), img.size[0], img.size[1]};
    chw_img = chw_img.reshape(1, shape);
    chw_imgs.push_back(chw_img);
  }

  return chw_imgs;
}

absl::StatusOr<std::vector<cv::Mat>>
ToBatch::operator()(const std::vector<cv::Mat> &imgs) const {
  if (imgs.empty()) {
    return absl::InvalidArgumentError("Input image vector is empty.");
  }
  const int batch = imgs.size();
  const int rows = imgs[0].rows;
  const int cols = imgs[0].cols;
  const int channels = imgs[0].channels();

  for (size_t i = 0; i < imgs.size(); ++i) {
    if (imgs[i].rows != rows || imgs[i].cols != cols ||
        imgs[i].channels() != channels) {
      return absl::InvalidArgumentError(
          "All images must have the same size and number of channels.");
    }
  }

  std::vector<int> sizes = {batch, rows, cols, channels};
  cv::Mat out(4, sizes.data(), CV_32F);

  for (int b = 0; b < batch; ++b) {
    cv::Mat img_float;
    if (imgs[b].depth() != CV_32F) {
      imgs[b].convertTo(img_float, CV_32F);
    } else {
      img_float = imgs[b];
    }

    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < cols; ++c) {
        if (channels == 1) {
          float v = img_float.at<float>(r, c);
          int idx[4] = {b, r, c, 0};
          out.at<float>(idx) = v;
        } else if (channels == 3) {
          cv::Vec3f v = img_float.at<cv::Vec3f>(r, c);
          for (int ch = 0; ch < 3; ++ch) {
            int idx[4] = {b, r, c, ch};
            out.at<float>(idx) = v[ch];
          }
        } else {
          const float *pix = img_float.ptr<float>(r, c);
          for (int ch = 0; ch < channels; ++ch) {
            int idx[4] = {b, r, c, ch};
            out.at<float>(idx) = pix[ch];
          }
        }
      }
    }
  }
  std::vector<cv::Mat> result{out};
  return result;
}

absl::StatusOr<std::vector<cv::Mat>> ToBatch::Apply(std::vector<cv::Mat> &input,
                                                    const void *param) const {
  if (input.empty()) {
    return absl::InvalidArgumentError("Input image vector is empty.");
  }

  std::vector<int> batch_shape = {(int)input.size()};
  for (const auto &image : input) {
    if (image.dims != input[0].dims) {
      return absl::InvalidArgumentError("All images must have the same dims.");
    } else {
      for (int i = 0; i < input[0].dims; i++) {
        if (image.size[i] != input[0].size[i]) {
          return absl::InvalidArgumentError(
              "All images must have the same size and number of channels.");
        }
        if (&image == &(*std::begin(input)))
          batch_shape.emplace_back(input[0].size[i]);
      }
    }
  }
  cv::Mat batch_out;
  for (auto &image : input)
    image = image.reshape(1, 1);
  cv::vconcat(input, batch_out);
  batch_out = batch_out.reshape(1, batch_shape);
  std::vector<cv::Mat> out = {batch_out};
  return out;
}

absl::StatusOr<cv::Mat> ComponentsProcessor::RotateImage(const cv::Mat &image,
                                                         int angle) {
  if (image.empty() || image.channels() != 3) {
    return absl::InvalidArgumentError("image is invalid");
  }
  if (angle < 0 || angle >= 360) {
    return absl::InvalidArgumentError("`angle` should be in range [0, 360)");
  }
  if (std::abs(angle) < 1e-7) {
    return image.clone();
  }

  int h = image.rows;
  int w = image.cols;
  cv::Point2f center(w / 2.0f, h / 2.0f);
  double scale = 1.0;
  cv::Mat rot_mat = cv::getRotationMatrix2D(center, angle, scale);

  double abs_cos = std::abs(rot_mat.at<double>(0, 0));
  double abs_sin = std::abs(rot_mat.at<double>(0, 1));
  int new_w = int(h * abs_sin + w * abs_cos);
  int new_h = int(h * abs_cos + w * abs_sin);

  rot_mat.at<double>(0, 2) += (new_w - w) / 2.0;
  rot_mat.at<double>(1, 2) += (new_h - h) / 2.0;

  cv::Mat rotated;
  cv::warpAffine(image, rotated, rot_mat, cv::Size(new_w, new_h),
                 cv::INTER_CUBIC);

  return rotated;
}

std::vector<std::vector<cv::Point2f>> ComponentsProcessor::SortQuadBoxes(
    const std::vector<std::vector<cv::Point2f>> &dt_polys) {
  std::vector<std::vector<cv::Point2f>> dt_boxes = dt_polys;

  std::sort(
      dt_boxes.begin(), dt_boxes.end(),
      [](const std::vector<cv::Point2f> &a, const std::vector<cv::Point2f> &b) {
        return (a[0].y < b[0].y) || (a[0].y == b[0].y && a[0].x < b[0].x);
      });

  for (size_t i = 0; i < dt_boxes.size() - 1; ++i) {
    for (size_t j = i + 1; j > 0; --j) {
      if (std::abs(dt_boxes[j][0].y - dt_boxes[j - 1][0].y) < 10 &&
          dt_boxes[j][0].x < dt_boxes[j - 1][0].x) {
        std::swap(dt_boxes[j], dt_boxes[j - 1]);
      } else {
        break;
      }
    }
  }
  return dt_boxes;
}

std::vector<std::vector<cv::Point2f>> ComponentsProcessor::SortPolyBoxes(
    const std::vector<std::vector<cv::Point2f>> &dt_polys) {
  size_t num_boxes = dt_polys.size();
  if (num_boxes == 0)
    return dt_polys;
  std::vector<int> y_min_list(num_boxes);
  for (size_t i = 0; i < num_boxes; ++i) {
    int y_min = dt_polys[i][0].y;
    for (size_t j = 1; j < dt_polys[i].size(); ++j) {
      if (dt_polys[i][j].y < y_min) {
        y_min = dt_polys[i][j].y;
      }
    }
    y_min_list[i] = y_min;
  }
  std::vector<size_t> rank(num_boxes);
  std::iota(rank.begin(), rank.end(), 0);
  std::sort(rank.begin(), rank.end(),
            [&](size_t a, size_t b) { return y_min_list[a] < y_min_list[b]; });
  std::vector<std::vector<cv::Point2f>> dt_polys_rank(num_boxes);
  for (size_t i = 0; i < num_boxes; ++i) {
    dt_polys_rank[i] = dt_polys[rank[i]];
  }
  return dt_polys_rank;
}

std::vector<std::array<float, 4>> ComponentsProcessor::ConvertPointsToBoxes(
    const std::vector<std::vector<cv::Point2f>> &dt_polys) {
  std::vector<std::array<float, 4>> dt_boxes;
  for (const auto &poly : dt_polys) {
    if (poly.empty()) {
      continue;
    }
    float left = std::numeric_limits<float>::max();
    float right = std::numeric_limits<float>::lowest();
    float top = std::numeric_limits<float>::max();
    float bottom = std::numeric_limits<float>::lowest();

    for (const auto &pt : poly) {
      if (pt.x < left)
        left = pt.x;
      if (pt.x > right)
        right = pt.x;
      if (pt.y < top)
        top = pt.y;
      if (pt.y > bottom)
        bottom = pt.y;
    }
    dt_boxes.push_back({left, top, right, bottom});
  }
  return dt_boxes;
}

CropByPolys::CropByPolys(const std::string &box_type) {
  assert(box_type == "quad" || box_type == "poly");
  if (box_type == "quad") {
    box_type_ = DetBoxType::kQuad;
  } else {
    box_type_ = DetBoxType::kPoly;
  }
}

absl::StatusOr<std::vector<cv::Mat>>
CropByPolys::operator()(const cv::Mat &img,
                        const std::vector<std::vector<cv::Point2f>> &dt_polys) {
  if (img.empty())
    return absl::InvalidArgumentError("Input image is empty.");
  std::vector<cv::Mat> output_list;
  try {
    if (box_type_ == DetBoxType::kQuad) {
      for (const auto &poly : dt_polys) {
        auto out = GetMinAreaRectCrop(img, poly);
        if (!out.ok())
          return out.status();
        output_list.push_back(*out);
      }
    } else if (box_type_ == DetBoxType::kPoly) {
      for (const auto &poly : dt_polys) {
        auto out = GetPolyRectCrop(img, poly);
        if (!out.ok())
          return out.status();
        output_list.push_back(*out);
      }
    } else {
      return absl::UnimplementedError("Unknown box type.");
    }
  } catch (const std::exception &e) {
    return absl::InternalError(std::string("Exception: ") + e.what());
  }
  return output_list;
}

absl::StatusOr<cv::Mat>
CropByPolys::GetMinAreaRectCrop(const cv::Mat &img,
                                const std::vector<cv::Point2f> &points) const {
  if (points.size() < 4)
    return absl::InvalidArgumentError("Less than 4 points for min area rect.");
  std::vector<cv::Point2f> box = GetMinAreaRectPoints(points);
  return GetRotateCropImage(img, box);
}

absl::StatusOr<cv::Mat>
CropByPolys::GetRotateCropImage(const cv::Mat &img,
                                const std::vector<cv::Point2f> &box) const {
  if (box.size() != 4)
    return absl::InvalidArgumentError("Box must have 4 points.");
  float widthTop = cv::norm(box[0] - box[1]);
  float widthBottom = cv::norm(box[2] - box[3]);
  float maxWidth = std::max(widthTop, widthBottom);

  float heightLeft = cv::norm(box[0] - box[3]);
  float heightRight = cv::norm(box[1] - box[2]);
  float maxHeight = std::max(heightLeft, heightRight);

  std::vector<cv::Point2f> dst = {
      cv::Point2f(0, 0), cv::Point2f(maxWidth - 1, 0),
      cv::Point2f(maxWidth - 1, maxHeight - 1), cv::Point2f(0, maxHeight - 1)};
  cv::Mat M = cv::getPerspectiveTransform(box, dst);
  cv::Mat out;
  cv::warpPerspective(img, out, M, cv::Size((int)maxWidth, (int)maxHeight),
                      cv::INTER_CUBIC, cv::BORDER_REPLICATE);
  if (out.rows != 0 && 1.0 * out.rows / out.cols >= 1.5)
    cv::rotate(out, out, cv::ROTATE_90_COUNTERCLOCKWISE);
  return out;
}

std::vector<cv::Point2f>
CropByPolys::GetMinAreaRectPoints(const std::vector<cv::Point2f> &poly) const {
  auto pts = poly;
  if (pts.size() < 4)
    return {};
  cv::RotatedRect minRect = cv::minAreaRect(pts);
  std::vector<cv::Point2f> box(4);
  minRect.points(box.data());
  std::sort(box.begin(), box.end(),
            [](const cv::Point2f &a, const cv::Point2f &b) {
              return a.x < b.x || (a.x == b.x && a.y < b.y);
            });
  size_t index_a = 0, index_d = 1;
  if (box[1].y > box[0].y) {
    index_a = 0;
    index_d = 1;
  } else {
    index_a = 1;
    index_d = 0;
  }
  size_t index_b = 2, index_c = 3;
  if (box[3].y > box[2].y) {
    index_b = 2;
    index_c = 3;
  } else {
    index_b = 3;
    index_c = 2;
  }
  return {box[index_a], box[index_b], box[index_c], box[index_d]};
}

absl::StatusOr<cv::Mat>
CropByPolys::GetPolyRectCrop(const cv::Mat &img,
                             const std::vector<cv::Point2f> &poly) const {
  if (poly.size() < 4)
    return absl::InvalidArgumentError(
        "Less than 4 points for GetPolyRectCrop.");
  // 对Poly和最小外接矩形做IoU判断
  std::vector<cv::Point2f> minrect = GetMinAreaRectPoints(poly);
  if (minrect.size() != 4)
    return absl::InternalError("Failed to get minarea rect.");
  double iou = IoU(poly, minrect);
  // 若IoU>0.7则返回直接crop，否则可做更复杂处理，如透视矫正，可进一步实现自定义变形矫正
  auto crop_result = GetRotateCropImage(img, minrect);
  if (!crop_result.ok())
    return crop_result.status();
  // 测试下如果IoU很高就用直接的最小外接矩形crop，否则复杂矫正（本实现只用直接crop）
  // 若需更强几何修复，可集成TPS、ThinPlateSpline或AutoRectifier
  return *crop_result;
}

const double CropByPolys::SCALE = 10000.0;

ClipperLib::Path
CropByPolys::CvPolyToClipperPath(const std::vector<cv::Point2f> &poly) {
  ClipperLib::Path path;
  for (const auto &pt : poly)
    path.emplace_back(static_cast<ClipperLib::cInt>(std::round(pt.x * SCALE)),
                      static_cast<ClipperLib::cInt>(std::round(pt.y * SCALE)));
  return path;
}

double CropByPolys::IoU(const std::vector<cv::Point2f> &poly1,
                        const std::vector<cv::Point2f> &poly2) {
  auto path1 = CvPolyToClipperPath(poly1);
  auto path2 = CvPolyToClipperPath(poly2);
  ClipperLib::Paths inter_solution, union_solution;
  ClipperLib::Clipper c_inter, c_union;
  c_inter.AddPath(path1, ClipperLib::ptSubject, true);
  c_inter.AddPath(path2, ClipperLib::ptClip, true);
  c_inter.Execute(ClipperLib::ctIntersection, inter_solution,
                  ClipperLib::pftNonZero, ClipperLib::pftNonZero);
  double area_inter = 0.0;
  for (const auto &p : inter_solution)
    area_inter += std::fabs(ClipperLib::Area(p));
  c_union.AddPath(path1, ClipperLib::ptSubject, true);
  c_union.AddPath(path2, ClipperLib::ptClip, true);
  c_union.Execute(ClipperLib::ctUnion, union_solution, ClipperLib::pftNonZero,
                  ClipperLib::pftNonZero);
  double area_union = 0.0;
  for (const auto &p : union_solution)
    area_union += std::fabs(ClipperLib::Area(p));
  area_inter /= (SCALE * SCALE);
  area_union /= (SCALE * SCALE);
  if (area_union < 1e-8)
    return 0.0;
  return area_inter / area_union;
}
