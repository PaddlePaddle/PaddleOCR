#include "preprocess.h"
#include <android/bitmap.h>

cv::Mat bitmap_to_cv_mat(JNIEnv *env, jobject bitmap) {
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

cv::Mat resize_img(const cv::Mat &img, int height, int width) {
  if (img.rows == height && img.cols == width) {
    return img;
  }
  cv::Mat new_img;
  cv::resize(img, new_img, cv::Size(height, width));
  return new_img;
}

// fill tensor with mean and scale and trans layout: nhwc -> nchw, neon speed up
void neon_mean_scale(const float *din, float *dout, int size,
                     const std::vector<float> &mean,
                     const std::vector<float> &scale) {
  if (mean.size() != 3 || scale.size() != 3) {
    LOGE("[ERROR] mean or scale size must equal to 3");
    return;
  }

  float32x4_t vmean0 = vdupq_n_f32(mean[0]);
  float32x4_t vmean1 = vdupq_n_f32(mean[1]);
  float32x4_t vmean2 = vdupq_n_f32(mean[2]);
  float32x4_t vscale0 = vdupq_n_f32(scale[0]);
  float32x4_t vscale1 = vdupq_n_f32(scale[1]);
  float32x4_t vscale2 = vdupq_n_f32(scale[2]);

  float *dout_c0 = dout;
  float *dout_c1 = dout + size;
  float *dout_c2 = dout + size * 2;

  int i = 0;
  for (; i < size - 3; i += 4) {
    float32x4x3_t vin3 = vld3q_f32(din);
    float32x4_t vsub0 = vsubq_f32(vin3.val[0], vmean0);
    float32x4_t vsub1 = vsubq_f32(vin3.val[1], vmean1);
    float32x4_t vsub2 = vsubq_f32(vin3.val[2], vmean2);
    float32x4_t vs0 = vmulq_f32(vsub0, vscale0);
    float32x4_t vs1 = vmulq_f32(vsub1, vscale1);
    float32x4_t vs2 = vmulq_f32(vsub2, vscale2);
    vst1q_f32(dout_c0, vs0);
    vst1q_f32(dout_c1, vs1);
    vst1q_f32(dout_c2, vs2);

    din += 12;
    dout_c0 += 4;
    dout_c1 += 4;
    dout_c2 += 4;
  }
  for (; i < size; i++) {
    *(dout_c0++) = (*(din++) - mean[0]) * scale[0];
    *(dout_c1++) = (*(din++) - mean[1]) * scale[1];
    *(dout_c2++) = (*(din++) - mean[2]) * scale[2];
  }
}