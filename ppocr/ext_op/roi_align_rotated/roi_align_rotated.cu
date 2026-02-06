
// This code is refer from:
// https://github.com/open-mmlab/mmcv/blob/master/mmcv/ops/csrc/common/cuda/roi_align_rotated_cuda_kernel.cuh

#include <cassert>
#include <cmath>
#include <vector>

#include "paddle/extension.h"
#include <cuda.h>

#define CUDA_1D_KERNEL_LOOP(i, n)                                              \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n);                 \
       i += blockDim.x * gridDim.x)

#define THREADS_PER_BLOCK 512

inline int GET_BLOCKS(const int N) {
  int optimal_block_num = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  int max_block_num = 4096;
  return min(optimal_block_num, max_block_num);
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600

static __inline__ __device__ double atomicAdd(double *address, double val) {
  unsigned long long int *address_as_ull = (unsigned long long int *)address;
  unsigned long long int old = *address_as_ull, assumed;
  if (val == 0.0)
    return __longlong_as_double(old);
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}

#endif

template <typename T>
__device__ T bilinear_interpolate(const T *input, const int height,
                                  const int width, T y, T x,
                                  const int index /* index for debug only*/) {
  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width)
    return 0;

  if (y <= 0)
    y = 0;
  if (x <= 0)
    x = 0;

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
  // do bilinear interpolation
  T v1 = input[y_low * width + x_low];
  T v2 = input[y_low * width + x_high];
  T v3 = input[y_high * width + x_low];
  T v4 = input[y_high * width + x_high];
  T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

  return val;
}

template <typename T>
__device__ void
bilinear_interpolate_gradient(const int height, const int width, T y, T x,
                              T &w1, T &w2, T &w3, T &w4, int &x_low,
                              int &x_high, int &y_low, int &y_high,
                              const int index /* index for debug only*/) {
  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    // empty
    w1 = w2 = w3 = w4 = 0.;
    x_low = x_high = y_low = y_high = -1;
    return;
  }

  if (y <= 0)
    y = 0;
  if (x <= 0)
    x = 0;

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

/*** Forward ***/
template <typename scalar_t>
__global__ void roi_align_rotated_cuda_forward_kernel(
    const int nthreads, const scalar_t *bottom_data,
    const scalar_t *bottom_rois, const scalar_t spatial_scale,
    const int sample_num, const bool aligned, const bool clockwise,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width, scalar_t *top_data) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    const scalar_t *offset_bottom_rois = bottom_rois + n * 6;
    int roi_batch_ind = offset_bottom_rois[0];

    // Do not using rounding; this implementation detail is critical
    scalar_t offset = aligned ? (scalar_t)0.5 : (scalar_t)0.0;
    scalar_t roi_center_w = offset_bottom_rois[1] * spatial_scale - offset;
    scalar_t roi_center_h = offset_bottom_rois[2] * spatial_scale - offset;
    scalar_t roi_width = offset_bottom_rois[3] * spatial_scale;
    scalar_t roi_height = offset_bottom_rois[4] * spatial_scale;
    // scalar_t theta = offset_bottom_rois[5] * M_PI / 180.0;
    scalar_t theta = offset_bottom_rois[5];
    if (clockwise) {
      theta = -theta; // If clockwise, the angle needs to be reversed.
    }
    if (!aligned) { // for backward-compatibility only
      // Force malformed ROIs to be 1x1
      roi_width = max(roi_width, (scalar_t)1.);
      roi_height = max(roi_height, (scalar_t)1.);
    }
    scalar_t bin_size_h = static_cast<scalar_t>(roi_height) /
                          static_cast<scalar_t>(pooled_height);
    scalar_t bin_size_w =
        static_cast<scalar_t>(roi_width) / static_cast<scalar_t>(pooled_width);

    const scalar_t *offset_bottom_data =
        bottom_data + (roi_batch_ind * channels + c) * height * width;

    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h = (sample_num > 0)
                             ? sample_num
                             : ceilf(roi_height / pooled_height); // e.g., = 2
    int roi_bin_grid_w =
        (sample_num > 0) ? sample_num : ceilf(roi_width / pooled_width);

    // roi_start_h and roi_start_w are computed wrt the center of RoI (x, y).
    // Appropriate translation needs to be applied after.
    scalar_t roi_start_h = -roi_height / 2.0;
    scalar_t roi_start_w = -roi_width / 2.0;
    scalar_t cosscalar_theta = cos(theta);
    scalar_t sinscalar_theta = sin(theta);

    // We do average (integral) pooling inside a bin
    const scalar_t count = max(roi_bin_grid_h * roi_bin_grid_w, 1); // e.g. = 4

    scalar_t output_val = 0.;
    for (int iy = 0; iy < roi_bin_grid_h; iy++) { // e.g., iy = 0, 1
      const scalar_t yy =
          roi_start_h + ph * bin_size_h +
          static_cast<scalar_t>(iy + .5f) * bin_size_h /
              static_cast<scalar_t>(roi_bin_grid_h); // e.g., 0.5, 1.5
      for (int ix = 0; ix < roi_bin_grid_w; ix++) {
        const scalar_t xx = roi_start_w + pw * bin_size_w +
                            static_cast<scalar_t>(ix + .5f) * bin_size_w /
                                static_cast<scalar_t>(roi_bin_grid_w);

        // Rotate by theta (counterclockwise) around the center and translate
        scalar_t y = yy * cosscalar_theta - xx * sinscalar_theta + roi_center_h;
        scalar_t x = yy * sinscalar_theta + xx * cosscalar_theta + roi_center_w;

        scalar_t val = bilinear_interpolate<scalar_t>(
            offset_bottom_data, height, width, y, x, index);
        output_val += val;
      }
    }
    output_val /= count;

    top_data[index] = output_val;
  }
}

/*** Backward ***/
template <typename scalar_t>
__global__ void roi_align_rotated_backward_cuda_kernel(
    const int nthreads, const scalar_t *top_diff, const scalar_t *bottom_rois,
    const scalar_t spatial_scale, const int sample_num, const bool aligned,
    const bool clockwise, const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width, scalar_t *bottom_diff) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    const scalar_t *offset_bottom_rois = bottom_rois + n * 6;
    int roi_batch_ind = offset_bottom_rois[0];

    // Do not round
    scalar_t offset = aligned ? (scalar_t)0.5 : (scalar_t)0.0;
    scalar_t roi_center_w = offset_bottom_rois[1] * spatial_scale - offset;
    scalar_t roi_center_h = offset_bottom_rois[2] * spatial_scale - offset;
    scalar_t roi_width = offset_bottom_rois[3] * spatial_scale;
    scalar_t roi_height = offset_bottom_rois[4] * spatial_scale;
    // scalar_t theta = offset_bottom_rois[5] * M_PI / 180.0;
    scalar_t theta = offset_bottom_rois[5];
    if (clockwise) {
      theta = -theta; // If clockwise, the angle needs to be reversed.
    }
    if (!aligned) { // for backward-compatibility only
      // Force malformed ROIs to be 1x1
      roi_width = max(roi_width, (scalar_t)1.);
      roi_height = max(roi_height, (scalar_t)1.);
    }
    scalar_t bin_size_h = static_cast<scalar_t>(roi_height) /
                          static_cast<scalar_t>(pooled_height);
    scalar_t bin_size_w =
        static_cast<scalar_t>(roi_width) / static_cast<scalar_t>(pooled_width);

    scalar_t *offset_bottom_diff =
        bottom_diff + (roi_batch_ind * channels + c) * height * width;

    int top_offset = (n * channels + c) * pooled_height * pooled_width;
    const scalar_t *offset_top_diff = top_diff + top_offset;
    const scalar_t top_diff_this_bin = offset_top_diff[ph * pooled_width + pw];

    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h = (sample_num > 0)
                             ? sample_num
                             : ceilf(roi_height / pooled_height); // e.g., = 2
    int roi_bin_grid_w =
        (sample_num > 0) ? sample_num : ceilf(roi_width / pooled_width);

    // roi_start_h and roi_start_w are computed wrt the center of RoI (x, y).
    // Appropriate translation needs to be applied after.
    scalar_t roi_start_h = -roi_height / 2.0;
    scalar_t roi_start_w = -roi_width / 2.0;
    scalar_t cosTheta = cos(theta);
    scalar_t sinTheta = sin(theta);

    // We do average (integral) pooling inside a bin
    const scalar_t count = roi_bin_grid_h * roi_bin_grid_w; // e.g. = 4

    for (int iy = 0; iy < roi_bin_grid_h; iy++) { // e.g., iy = 0, 1
      const scalar_t yy =
          roi_start_h + ph * bin_size_h +
          static_cast<scalar_t>(iy + .5f) * bin_size_h /
              static_cast<scalar_t>(roi_bin_grid_h); // e.g., 0.5, 1.5
      for (int ix = 0; ix < roi_bin_grid_w; ix++) {
        const scalar_t xx = roi_start_w + pw * bin_size_w +
                            static_cast<scalar_t>(ix + .5f) * bin_size_w /
                                static_cast<scalar_t>(roi_bin_grid_w);

        // Rotate by theta around the center and translate
        scalar_t y = yy * cosTheta - xx * sinTheta + roi_center_h;
        scalar_t x = yy * sinTheta + xx * cosTheta + roi_center_w;

        scalar_t w1, w2, w3, w4;
        int x_low, x_high, y_low, y_high;

        bilinear_interpolate_gradient<scalar_t>(height, width, y, x, w1, w2, w3,
                                                w4, x_low, x_high, y_low,
                                                y_high, index);

        scalar_t g1 = top_diff_this_bin * w1 / count;
        scalar_t g2 = top_diff_this_bin * w2 / count;
        scalar_t g3 = top_diff_this_bin * w3 / count;
        scalar_t g4 = top_diff_this_bin * w4 / count;

        if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0) {
          atomicAdd(offset_bottom_diff + y_low * width + x_low, g1);
          atomicAdd(offset_bottom_diff + y_low * width + x_high, g2);
          atomicAdd(offset_bottom_diff + y_high * width + x_low, g3);
          atomicAdd(offset_bottom_diff + y_high * width + x_high, g4);
        } // if
      }   // ix
    }     // iy
  }       // CUDA_1D_KERNEL_LOOP
} // RoIAlignBackward

std::vector<paddle::Tensor>
RoIAlignRotatedCUDAForward(const paddle::Tensor &input,
                           const paddle::Tensor &rois, int aligned_height,
                           int aligned_width, float spatial_scale,
                           int sampling_ratio, bool aligned, bool clockwise) {

  auto num_rois = rois.shape()[0];

  auto channels = input.shape()[1];
  auto height = input.shape()[2];
  auto width = input.shape()[3];

  auto output =
      paddle::empty({num_rois, channels, aligned_height, aligned_width},
                    input.type(), paddle::GPUPlace());
  auto output_size = output.numel();

  PD_DISPATCH_FLOATING_TYPES(
      input.type(), "roi_align_rotated_cuda_forward_kernel", ([&] {
        roi_align_rotated_cuda_forward_kernel<data_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK>>>(
                output_size, input.data<data_t>(), rois.data<data_t>(),
                static_cast<data_t>(spatial_scale), sampling_ratio, aligned,
                clockwise, channels, height, width, aligned_height,
                aligned_width, output.data<data_t>());
      }));

  return {output};
}

std::vector<paddle::Tensor> RoIAlignRotatedCUDABackward(
    const paddle::Tensor &input, const paddle::Tensor &rois,
    const paddle::Tensor &grad_output, int aligned_height, int aligned_width,
    float spatial_scale, int sampling_ratio, bool aligned, bool clockwise) {

  auto num_rois = rois.shape()[0];

  auto batch_size = input.shape()[0];
  auto channels = input.shape()[1];
  auto height = input.shape()[2];
  auto width = input.shape()[3];

  auto grad_input = paddle::full({batch_size, channels, height, width}, 0.0,
                                 input.type(), paddle::GPUPlace());

  const int output_size = num_rois * aligned_height * aligned_width * channels;

  PD_DISPATCH_FLOATING_TYPES(
      grad_output.type(), "roi_align_rotated_backward_cuda_kernel", ([&] {
        roi_align_rotated_backward_cuda_kernel<data_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK>>>(
                output_size, grad_output.data<data_t>(), rois.data<data_t>(),
                spatial_scale, sampling_ratio, aligned, clockwise, channels,
                height, width, aligned_height, aligned_width,
                grad_input.data<data_t>());
      }));
  return {grad_input};
}
