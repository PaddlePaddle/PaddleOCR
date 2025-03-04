
// This code is refer from:
// https://github.com/open-mmlab/mmcv/blob/master/mmcv/ops/csrc/pytorch/cpu/roi_align_rotated.cpp

#include <cassert>
#include <cmath>
#include <vector>

#include "paddle/extension.h"

#define PADDLE_WITH_CUDA
#define CHECK_INPUT_SAME(x1, x2)                                               \
  PD_CHECK(x1.place() == x2.place(), "input must be same place.")
#define CHECK_INPUT_CPU(x) PD_CHECK(x.is_cpu(), #x " must be a CPU Tensor.")

template <typename T> struct PreCalc {
  int pos1;
  int pos2;
  int pos3;
  int pos4;
  T w1;
  T w2;
  T w3;
  T w4;
};

template <typename T>
void pre_calc_for_bilinear_interpolate(
    const int height, const int width, const int pooled_height,
    const int pooled_width, const int iy_upper, const int ix_upper,
    T roi_start_h, T roi_start_w, T bin_size_h, T bin_size_w,
    int roi_bin_grid_h, int roi_bin_grid_w, T roi_center_h, T roi_center_w,
    T cos_theta, T sin_theta, std::vector<PreCalc<T>> &pre_calc) {
  int pre_calc_index = 0;
  for (int ph = 0; ph < pooled_height; ph++) {
    for (int pw = 0; pw < pooled_width; pw++) {
      for (int iy = 0; iy < iy_upper; iy++) {
        const T yy = roi_start_h + ph * bin_size_h +
                     static_cast<T>(iy + .5f) * bin_size_h /
                         static_cast<T>(roi_bin_grid_h); // e.g., 0.5, 1.5
        for (int ix = 0; ix < ix_upper; ix++) {
          const T xx = roi_start_w + pw * bin_size_w +
                       static_cast<T>(ix + .5f) * bin_size_w /
                           static_cast<T>(roi_bin_grid_w);

          // Rotate by theta around the center and translate
          // In image space, (y, x) is the order for Right Handed System,
          // and this is essentially multiplying the point by a rotation matrix
          // to rotate it counterclockwise through angle theta.
          T y = yy * cos_theta - xx * sin_theta + roi_center_h;
          T x = yy * sin_theta + xx * cos_theta + roi_center_w;
          // deal with: inverse elements are out of feature map boundary
          if (y < -1.0 || y > height || x < -1.0 || x > width) {
            // empty
            PreCalc<T> pc;
            pc.pos1 = 0;
            pc.pos2 = 0;
            pc.pos3 = 0;
            pc.pos4 = 0;
            pc.w1 = 0;
            pc.w2 = 0;
            pc.w3 = 0;
            pc.w4 = 0;
            pre_calc[pre_calc_index] = pc;
            pre_calc_index += 1;
            continue;
          }

          if (y < 0) {
            y = 0;
          }
          if (x < 0) {
            x = 0;
          }

          int y_low = (int)y;
          int x_low = (int)x;
          int y_high;
          int x_high;

          if (y_low >= height - 1) {
            y_high = y_low = height - 1;
            y = (T)y_low;
          } else {
            y_high = y_low + 1;
          }

          if (x_low >= width - 1) {
            x_high = x_low = width - 1;
            x = (T)x_low;
          } else {
            x_high = x_low + 1;
          }

          T ly = y - y_low;
          T lx = x - x_low;
          T hy = 1. - ly, hx = 1. - lx;
          T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

          // save weights and indices
          PreCalc<T> pc;
          pc.pos1 = y_low * width + x_low;
          pc.pos2 = y_low * width + x_high;
          pc.pos3 = y_high * width + x_low;
          pc.pos4 = y_high * width + x_high;
          pc.w1 = w1;
          pc.w2 = w2;
          pc.w3 = w3;
          pc.w4 = w4;
          pre_calc[pre_calc_index] = pc;

          pre_calc_index += 1;
        }
      }
    }
  }
}

template <typename T>
void roi_align_rotated_cpu_forward(const int nthreads, const T *input,
                                   const T &spatial_scale, const bool aligned,
                                   const bool clockwise, const int channels,
                                   const int height, const int width,
                                   const int pooled_height,
                                   const int pooled_width,
                                   const int sampling_ratio, const T *rois,
                                   T *output) {
  int n_rois = nthreads / channels / pooled_width / pooled_height;
  // (n, c, ph, pw) is an element in the pooled output
  // can be parallelized using omp
  // #pragma omp parallel for num_threads(32)
  for (int n = 0; n < n_rois; n++) {
    int index_n = n * channels * pooled_width * pooled_height;

    const T *current_roi = rois + n * 6;
    int roi_batch_ind = current_roi[0];

    // Do not use rounding; this implementation detail is critical
    T offset = aligned ? (T)0.5 : (T)0.0;
    T roi_center_w = current_roi[1] * spatial_scale - offset;
    T roi_center_h = current_roi[2] * spatial_scale - offset;
    T roi_width = current_roi[3] * spatial_scale;
    T roi_height = current_roi[4] * spatial_scale;
    T theta = current_roi[5];
    if (clockwise) {
      theta = -theta; // If clockwise, the angle needs to be reversed.
    }
    T cos_theta = cos(theta);
    T sin_theta = sin(theta);

    if (aligned) {
      assert(roi_width >= 0 && roi_height >= 0);
    } else { // for backward-compatibility only
      roi_width = std::max(roi_width, (T)1.);
      roi_height = std::max(roi_height, (T)1.);
    }

    T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
    T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h = (sampling_ratio > 0)
                             ? sampling_ratio
                             : ceilf(roi_height / pooled_height); // e.g., = 2
    int roi_bin_grid_w =
        (sampling_ratio > 0) ? sampling_ratio : ceilf(roi_width / pooled_width);

    // We do average (integral) pooling inside a bin
    const T count = std::max(roi_bin_grid_h * roi_bin_grid_w, 1); // e.g. = 4

    // we want to precalculate indices and weights shared by all channels,
    // this is the key point of optimization
    std::vector<PreCalc<T>> pre_calc(roi_bin_grid_h * roi_bin_grid_w *
                                     pooled_width * pooled_height);

    // roi_start_h and roi_start_w are computed wrt the center of RoI (x, y).
    // Appropriate translation needs to be applied after.
    T roi_start_h = -roi_height / 2.0;
    T roi_start_w = -roi_width / 2.0;

    pre_calc_for_bilinear_interpolate(
        height, width, pooled_height, pooled_width, roi_bin_grid_h,
        roi_bin_grid_w, roi_start_h, roi_start_w, bin_size_h, bin_size_w,
        roi_bin_grid_h, roi_bin_grid_w, roi_center_h, roi_center_w, cos_theta,
        sin_theta, pre_calc);

    for (int c = 0; c < channels; c++) {
      int index_n_c = index_n + c * pooled_width * pooled_height;
      const T *offset_input =
          input + (roi_batch_ind * channels + c) * height * width;
      int pre_calc_index = 0;

      for (int ph = 0; ph < pooled_height; ph++) {
        for (int pw = 0; pw < pooled_width; pw++) {
          int index = index_n_c + ph * pooled_width + pw;

          T output_val = 0.;
          for (int iy = 0; iy < roi_bin_grid_h; iy++) {
            for (int ix = 0; ix < roi_bin_grid_w; ix++) {
              PreCalc<T> pc = pre_calc[pre_calc_index];
              output_val += pc.w1 * offset_input[pc.pos1] +
                            pc.w2 * offset_input[pc.pos2] +
                            pc.w3 * offset_input[pc.pos3] +
                            pc.w4 * offset_input[pc.pos4];

              pre_calc_index += 1;
            }
          }
          output_val /= count;

          output[index] = output_val;
        } // for pw
      }   // for ph
    }     // for c
  }       // for n
}

template <typename T>
void bilinear_interpolate_gradient(const int height, const int width, T y, T x,
                                   T &w1, T &w2, T &w3, T &w4, int &x_low,
                                   int &x_high, int &y_low, int &y_high) {
  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    // empty
    w1 = w2 = w3 = w4 = 0.;
    x_low = x_high = y_low = y_high = -1;
    return;
  }

  if (y < 0) {
    y = 0;
  }

  if (x < 0) {
    x = 0;
  }

  y_low = (int)y;
  x_low = (int)x;

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (T)y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (T)x_low;
  } else {
    x_high = x_low + 1;
  }

  T ly = y - y_low;
  T lx = x - x_low;
  T hy = 1. - ly, hx = 1. - lx;

  // reference in forward
  // T v1 = input[y_low * width + x_low];
  // T v2 = input[y_low * width + x_high];
  // T v3 = input[y_high * width + x_low];
  // T v4 = input[y_high * width + x_high];
  // T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

  w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  return;
}

template <class T> inline void add(T *address, const T &val) {
  *address += val;
}

template <typename T>
void roi_align_rotated_cpu_backward(
    const int nthreads,
    // may not be contiguous. should index using n_stride, etc
    const T *grad_output, const T &spatial_scale, const bool aligned,
    const bool clockwise, const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width, const int sampling_ratio,
    T *grad_input, const T *rois, const int n_stride, const int c_stride,
    const int h_stride, const int w_stride) {
  for (int index = 0; index < nthreads; index++) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    const T *current_roi = rois + n * 6;
    int roi_batch_ind = current_roi[0];

    // Do not use rounding; this implementation detail is critical
    T offset = aligned ? (T)0.5 : (T)0.0;
    T roi_center_w = current_roi[1] * spatial_scale - offset;
    T roi_center_h = current_roi[2] * spatial_scale - offset;
    T roi_width = current_roi[3] * spatial_scale;
    T roi_height = current_roi[4] * spatial_scale;
    T theta = current_roi[5];
    if (clockwise) {
      theta = -theta; // If clockwise, the angle needs to be reversed.
    }
    T cos_theta = cos(theta);
    T sin_theta = sin(theta);

    if (aligned) {
      assert(roi_width >= 0 && roi_height >= 0);
    } else { // for backward-compatibility only
      roi_width = std::max(roi_width, (T)1.);
      roi_height = std::max(roi_height, (T)1.);
    }

    T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
    T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

    T *offset_grad_input =
        grad_input + ((roi_batch_ind * channels + c) * height * width);

    int output_offset = n * n_stride + c * c_stride;
    const T *offset_grad_output = grad_output + output_offset;
    const T grad_output_this_bin =
        offset_grad_output[ph * h_stride + pw * w_stride];

    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h = (sampling_ratio > 0)
                             ? sampling_ratio
                             : ceilf(roi_height / pooled_height); // e.g., = 2
    int roi_bin_grid_w =
        (sampling_ratio > 0) ? sampling_ratio : ceilf(roi_width / pooled_width);

    // roi_start_h and roi_start_w are computed wrt the center of RoI (x, y).
    // Appropriate translation needs to be applied after.
    T roi_start_h = -roi_height / 2.0;
    T roi_start_w = -roi_width / 2.0;

    // We do average (integral) pooling inside a bin
    const T count = roi_bin_grid_h * roi_bin_grid_w; // e.g. = 4

    for (int iy = 0; iy < roi_bin_grid_h; iy++) {
      const T yy = roi_start_h + ph * bin_size_h +
                   static_cast<T>(iy + .5f) * bin_size_h /
                       static_cast<T>(roi_bin_grid_h); // e.g., 0.5, 1.5
      for (int ix = 0; ix < roi_bin_grid_w; ix++) {
        const T xx = roi_start_w + pw * bin_size_w +
                     static_cast<T>(ix + .5f) * bin_size_w /
                         static_cast<T>(roi_bin_grid_w);

        // Rotate by theta around the center and translate
        T y = yy * cos_theta - xx * sin_theta + roi_center_h;
        T x = yy * sin_theta + xx * cos_theta + roi_center_w;

        T w1, w2, w3, w4;
        int x_low, x_high, y_low, y_high;

        bilinear_interpolate_gradient(height, width, y, x, w1, w2, w3, w4,
                                      x_low, x_high, y_low, y_high);

        T g1 = grad_output_this_bin * w1 / count;
        T g2 = grad_output_this_bin * w2 / count;
        T g3 = grad_output_this_bin * w3 / count;
        T g4 = grad_output_this_bin * w4 / count;

        if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0) {
          // atomic add is not needed for now since it is single threaded
          add(offset_grad_input + y_low * width + x_low, static_cast<T>(g1));
          add(offset_grad_input + y_low * width + x_high, static_cast<T>(g2));
          add(offset_grad_input + y_high * width + x_low, static_cast<T>(g3));
          add(offset_grad_input + y_high * width + x_high, static_cast<T>(g4));
        } // if
      }   // ix
    }     // iy
  }       // for
} // ROIAlignRotatedBackward

std::vector<paddle::Tensor>
RoIAlignRotatedCPUForward(const paddle::Tensor &input,
                          const paddle::Tensor &rois, int aligned_height,
                          int aligned_width, float spatial_scale,
                          int sampling_ratio, bool aligned, bool clockwise) {
  CHECK_INPUT_CPU(input);
  CHECK_INPUT_CPU(rois);

  auto num_rois = rois.shape()[0];

  auto channels = input.shape()[1];
  auto height = input.shape()[2];
  auto width = input.shape()[3];

  auto output =
      paddle::empty({num_rois, channels, aligned_height, aligned_width},
                    input.type(), paddle::CPUPlace());
  auto output_size = output.numel();

  PD_DISPATCH_FLOATING_TYPES(
      input.type(), "roi_align_rotated_cpu_forward", ([&] {
        roi_align_rotated_cpu_forward<data_t>(
            output_size, input.data<data_t>(),
            static_cast<data_t>(spatial_scale), aligned, clockwise, channels,
            height, width, aligned_height, aligned_width, sampling_ratio,
            rois.data<data_t>(), output.data<data_t>());
      }));

  return {output};
}

std::vector<paddle::Tensor> RoIAlignRotatedCPUBackward(
    const paddle::Tensor &input, const paddle::Tensor &rois,
    const paddle::Tensor &grad_output, int aligned_height, int aligned_width,
    float spatial_scale, int sampling_ratio, bool aligned, bool clockwise) {

  auto batch_size = input.shape()[0];
  auto channels = input.shape()[1];
  auto height = input.shape()[2];
  auto width = input.shape()[3];

  auto grad_input = paddle::full({batch_size, channels, height, width}, 0.0,
                                 input.type(), paddle::CPUPlace());

  // get stride values to ensure indexing into gradients is correct.
  int n_stride = grad_output.shape()[0];
  int c_stride = grad_output.shape()[1];
  int h_stride = grad_output.shape()[2];
  int w_stride = grad_output.shape()[3];

  PD_DISPATCH_FLOATING_TYPES(
      grad_output.type(), "roi_align_rotated_cpu_backward", [&] {
        roi_align_rotated_cpu_backward<data_t>(
            grad_output.numel(), grad_output.data<data_t>(),
            static_cast<data_t>(spatial_scale), aligned, clockwise, channels,
            height, width, aligned_height, aligned_width, sampling_ratio,
            grad_input.data<data_t>(), rois.data<data_t>(), n_stride, c_stride,
            h_stride, w_stride);
      });
  return {grad_input};
}

#ifdef PADDLE_WITH_CUDA
std::vector<paddle::Tensor>
RoIAlignRotatedCUDAForward(const paddle::Tensor &input,
                           const paddle::Tensor &rois, int aligned_height,
                           int aligned_width, float spatial_scale,
                           int sampling_ratio, bool aligned, bool clockwise);
#endif

#ifdef PADDLE_WITH_CUDA
std::vector<paddle::Tensor> RoIAlignRotatedCUDABackward(
    const paddle::Tensor &input, const paddle::Tensor &rois,
    const paddle::Tensor &grad_output, int aligned_height, int aligned_width,
    float spatial_scale, int sampling_ratio, bool aligned, bool clockwise);
#endif

std::vector<paddle::Tensor>
RoIAlignRotatedForward(const paddle::Tensor &input, const paddle::Tensor &rois,
                       int aligned_height, int aligned_width,
                       float spatial_scale, int sampling_ratio, bool aligned,
                       bool clockwise) {
  CHECK_INPUT_SAME(input, rois);
  if (input.is_cpu()) {
    return RoIAlignRotatedCPUForward(input, rois, aligned_height, aligned_width,
                                     spatial_scale, sampling_ratio, aligned,
                                     clockwise);
#ifdef PADDLE_WITH_CUDA
  } else if (input.is_gpu()) {
    return RoIAlignRotatedCUDAForward(input, rois, aligned_height,
                                      aligned_width, spatial_scale,
                                      sampling_ratio, aligned, clockwise);
#endif
  } else {
    PD_THROW("Unsupported device type for forward function of roi align "
             "rotated operator.");
  }
}

std::vector<paddle::Tensor>
RoIAlignRotatedBackward(const paddle::Tensor &input, const paddle::Tensor &rois,
                        const paddle::Tensor &grad_output, int aligned_height,
                        int aligned_width, float spatial_scale,
                        int sampling_ratio, bool aligned, bool clockwise) {
  CHECK_INPUT_SAME(input, rois);
  if (input.is_cpu()) {
    return RoIAlignRotatedCPUBackward(input, rois, grad_output, aligned_height,
                                      aligned_width, spatial_scale,
                                      sampling_ratio, aligned, clockwise);
#ifdef PADDLE_WITH_CUDA
  } else if (input.is_gpu()) {
    return RoIAlignRotatedCUDABackward(input, rois, grad_output, aligned_height,
                                       aligned_width, spatial_scale,
                                       sampling_ratio, aligned, clockwise);
#endif
  } else {
    PD_THROW("Unsupported device type for forward function of roi align "
             "rotated operator.");
  }
}

std::vector<std::vector<int64_t>> InferShape(std::vector<int64_t> input_shape,
                                             std::vector<int64_t> rois_shape) {
  return {{rois_shape[0], input_shape[1], input_shape[2], input_shape[3]}};
}

std::vector<std::vector<int64_t>>
InferBackShape(std::vector<int64_t> input_shape,
               std::vector<int64_t> rois_shape) {
  return {input_shape};
}

std::vector<paddle::DataType> InferDtype(paddle::DataType input_dtype,
                                         paddle::DataType rois_dtype) {
  return {input_dtype};
}

PD_BUILD_OP(roi_align_rotated)
    .Inputs({"Input", "Rois"})
    .Outputs({"Output"})
    .Attrs({"aligned_height: int", "aligned_width: int", "spatial_scale: float",
            "sampling_ratio: int", "aligned: bool", "clockwise: bool"})
    .SetKernelFn(PD_KERNEL(RoIAlignRotatedForward))
    .SetInferShapeFn(PD_INFER_SHAPE(InferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(InferDtype));

PD_BUILD_GRAD_OP(roi_align_rotated)
    .Inputs({"Input", "Rois", paddle::Grad("Output")})
    .Attrs({"aligned_height: int", "aligned_width: int", "spatial_scale: float",
            "sampling_ratio: int", "aligned: bool", "clockwise: bool"})
    .Outputs({paddle::Grad("Input")})
    .SetKernelFn(PD_KERNEL(RoIAlignRotatedBackward))
    .SetInferShapeFn(PD_INFER_SHAPE(InferBackShape));
