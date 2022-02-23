/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <stdint.h>

#include <cmath>
#include <iostream>
#include <limits>

#ifdef PADDLE_WITH_CUDA
#include <cuda.h>
#endif  // PADDLE_WITH_CUDA

#ifdef PADDLE_WITH_HIP
#include <hip/hip_runtime.h>
#endif

#if defined(__CUDACC__) && CUDA_VERSION >= 7050
#define PADDLE_CUDA_FP16
#include <cuda_fp16.h>
#endif

#ifdef __HIPCC__
#define PADDLE_CUDA_FP16
#include <hip/hip_fp16.h>
#endif

#if !defined(_WIN32)
#define PADDLE_ALIGN(x) __attribute__((aligned(x)))
#else
#define PADDLE_ALIGN(x) __declspec(align(x))
#endif

#define CUDA_ARCH_FP16_SUPPORTED(CUDA_ARCH) (CUDA_ARCH >= 600)

#if (defined(__CUDACC__) || defined(__HIPCC__))
#define HOSTDEVICE __host__ __device__
#define DEVICE __device__
#define HOST __host__
#else
#define HOSTDEVICE
#define DEVICE
#define HOST
#endif

namespace paddle {
namespace platform {

// Use PADDLE_ALIGNED(2) to ensure that each float16 will be allocated
// and aligned at least on a 2-byte boundary, which leads to efficient
// memory access of float16 struct and also makes float16 compatible
// with CUDA half, ARM float16_t data types.
struct PADDLE_ALIGN(2) float16 {
 public:
  uint16_t x;

  // The following defaulted special class member functions
  // are added to make float16 pass the std::is_trivial test
  float16() = default;
  float16(const float16& o) = default;
  float16& operator=(const float16& o) = default;
  float16(float16&& o) = default;
  float16& operator=(float16&& o) = default;
  ~float16() = default;

// Constructors
#ifdef PADDLE_CUDA_FP16
  HOSTDEVICE inline explicit float16(const half& h) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#if defined(PADDLE_WITH_HIP) || CUDA_VERSION >= 9000
    x = reinterpret_cast<__half_raw*>(const_cast<half*>(&h))->x;
#else
    x = h.x;
#endif  // CUDA_VERSION >= 9000
#endif
  }
#endif  // PADDLE_CUDA_FP16

#ifdef PADDLE_WITH_NATIVE_FP16
  // __fp16 is a native half precision data type for arm cpu,
  // float16_t is an alias for __fp16
  HOSTDEVICE inline explicit float16(const float16_t& h) {
    x = *reinterpret_cast<const uint16_t*>(&h);
  }
#endif

  HOSTDEVICE inline explicit float16(float val) {
#if defined(PADDLE_CUDA_FP16) && \
    (defined(__HIPCC__) || (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300))
    half tmp = __float2half(val);
    x = *reinterpret_cast<uint16_t*>(&tmp);

#elif defined(PADDLE_WITH_NATIVE_FP16)
    float32x4_t tmp = vld1q_dup_f32(&val);
    float16_t res = vget_lane_f16(vcvt_f16_f32(tmp), 0);
    x = *reinterpret_cast<uint16_t*>(&res);

#elif defined(__F16C__)
    x = _cvtss_sh(val, 0);

#else
    // Conversion routine adapted from
    // http://stackoverflow.com/questions/1659440/32-bit-to-16-bit-floating-point-conversion
    Bits v, s;
    v.f = val;
    uint32_t sign = v.si & sigN;
    v.si ^= sign;
    sign >>= shiftSign;  // logical shift
    s.si = mulN;
    s.si = s.f * v.f;  // correct subnormals
    v.si ^= (s.si ^ v.si) & -(minN > v.si);
    v.si ^= (infN ^ v.si) & -((infN > v.si) & (v.si > maxN));
    v.si ^= (nanN ^ v.si) & -((nanN > v.si) & (v.si > infN));
    v.ui >>= shift;  // logical shift
    v.si ^= ((v.si - maxD) ^ v.si) & -(v.si > maxC);
    v.si ^= ((v.si - minD) ^ v.si) & -(v.si > subC);
    x = v.ui | sign;

#endif
  }

  HOSTDEVICE inline explicit float16(bool b) : x(b ? 0x3c00 : 0) {}

  template <class T>
  HOSTDEVICE inline explicit float16(const T& val)
      : x(float16(static_cast<float>(val)).x) {}

// Assignment operators
#ifdef PADDLE_CUDA_FP16
  HOSTDEVICE inline float16& operator=(const half& rhs) {
#if defined(PADDLE_WITH_HIP) || CUDA_VERSION >= 9000
    x = reinterpret_cast<__half_raw*>(const_cast<half*>(&rhs))->x;
#else
    x = rhs.x;
#endif
    return *this;
  }
#endif

#ifdef PADDLE_WITH_NATIVE_FP16
  HOSTDEVICE inline float16& operator=(const float16_t& rhs) {
    x = *reinterpret_cast<const uint16_t*>(&rhs);
    return *this;
  }
#endif

  HOSTDEVICE inline float16& operator=(bool b) {
    x = b ? 0x3c00 : 0;
    return *this;
  }

  HOSTDEVICE inline float16& operator=(int8_t val) {
    x = float16(val).x;
    return *this;
  }

  HOSTDEVICE inline float16& operator=(uint8_t val) {
    x = float16(val).x;
    return *this;
  }

  HOSTDEVICE inline float16& operator=(int16_t val) {
    x = float16(val).x;
    return *this;
  }

  HOSTDEVICE inline float16& operator=(uint16_t val) {
    x = float16(val).x;
    return *this;
  }

  HOSTDEVICE inline float16& operator=(int32_t val) {
    x = float16(val).x;
    return *this;
  }

  HOSTDEVICE inline float16& operator=(uint32_t val) {
    x = float16(val).x;
    return *this;
  }

  HOSTDEVICE inline float16& operator=(int64_t val) {
    x = float16(val).x;
    return *this;
  }

  HOSTDEVICE inline float16& operator=(uint64_t val) {
    x = float16(val).x;
    return *this;
  }

  HOSTDEVICE inline float16& operator=(float val) {
    x = float16(val).x;
    return *this;
  }

  HOSTDEVICE inline float16& operator=(double val) {
    x = float16(val).x;
    return *this;
  }

// Conversion opertors
#ifdef PADDLE_CUDA_FP16
  HOSTDEVICE inline explicit operator half() const {
#if defined(PADDLE_WITH_HIP) || CUDA_VERSION >= 9000
    __half_raw h;
    h.x = x;
    return half(h);
#else
    half h;
    h.x = x;
    return h;
#endif  // CUDA_VERSION >= 9000
  }
#endif  // PADDLE_CUDA_FP16

#ifdef PADDLE_WITH_NATIVE_FP16
  HOSTDEVICE inline explicit operator float16_t() const {
    return *reinterpret_cast<const float16_t*>(this);
  }
#endif

  HOSTDEVICE inline explicit operator float() const {
#if defined(PADDLE_CUDA_FP16) && \
    (defined(__HIPCC__) || (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300))
    half tmp = *reinterpret_cast<const half*>(this);
    return __half2float(tmp);

#elif defined(PADDLE_WITH_NATIVE_FP16)
    float16x4_t res = vld1_dup_f16(reinterpret_cast<const float16_t*>(this));
    return vgetq_lane_f32(vcvt_f32_f16(res), 0);

#elif defined(__F16C__)
    return _cvtsh_ss(this->x);

#else
    // Conversion routine adapted from
    // http://stackoverflow.com/questions/1659440/32-bit-to-16-bit-floating-point-conversion
    Bits v;
    v.ui = this->x;
    int32_t sign = v.si & sigC;
    v.si ^= sign;
    sign <<= shiftSign;
    v.si ^= ((v.si + minD) ^ v.si) & -(v.si > subC);
    v.si ^= ((v.si + maxD) ^ v.si) & -(v.si > maxC);
    Bits s;
    s.si = mulC;
    s.f *= v.si;
    int32_t mask = -(norC > v.si);
    v.si <<= shift;
    v.si ^= (s.si ^ v.si) & mask;
    v.si |= sign;
    return v.f;

#endif
  }

  HOSTDEVICE inline explicit operator bool() const { return (x & 0x7fff) != 0; }

  HOSTDEVICE inline explicit operator int8_t() const {
    return static_cast<int8_t>(static_cast<float>(*this));
  }

  HOSTDEVICE inline explicit operator uint8_t() const {
    return static_cast<uint8_t>(static_cast<float>(*this));
  }

  HOSTDEVICE inline explicit operator int16_t() const {
    return static_cast<int16_t>(static_cast<float>(*this));
  }

  HOSTDEVICE inline explicit operator uint16_t() const {
    return static_cast<uint16_t>(static_cast<float>(*this));
  }

  HOSTDEVICE inline explicit operator int32_t() const {
    return static_cast<int32_t>(static_cast<float>(*this));
  }

  HOSTDEVICE inline explicit operator uint32_t() const {
    return static_cast<uint32_t>(static_cast<float>(*this));
  }

  HOSTDEVICE inline explicit operator int64_t() const {
    return static_cast<int64_t>(static_cast<float>(*this));
  }

  HOSTDEVICE inline explicit operator uint64_t() const {
    return static_cast<uint64_t>(static_cast<float>(*this));
  }

  HOSTDEVICE inline explicit operator double() const {
    return static_cast<double>(static_cast<float>(*this));
  }

 private:
  union Bits {
    float f;
    int32_t si;
    uint32_t ui;
  };

  static const int shift = 13;
  static const int shiftSign = 16;

  static const int32_t infN = 0x7F800000;
  static const int32_t maxN = 0x477FE000;  // max flt16 as flt32
  static const int32_t minN = 0x38800000;  // min flt16 normal as flt32
  static const int32_t sigN = 0x80000000;  // sign bit

  static constexpr int32_t infC = infN >> shift;
  static constexpr int32_t nanN = (infC + 1)
                                  << shift;  // minimum flt16 nan as float32
  static constexpr int32_t maxC = maxN >> shift;
  static constexpr int32_t minC = minN >> shift;
  static constexpr int32_t sigC = sigN >> shiftSign;

  static const int32_t mulN = 0x52000000;  // (1 << 23) / minN
  static const int32_t mulC = 0x33800000;  // minN / (1 << (23 - shift))
  static const int32_t subC = 0x003FF;     // max flt32 subnormal downshifted
  static const int32_t norC = 0x00400;     // min flt32 normal downshifted

  static constexpr int32_t maxD = infC - maxC - 1;
  static constexpr int32_t minD = minC - subC - 1;
};

// Arithmetic operators on GPU
// CUDA 9.0 provides built-in arithmetic operators for half while
// CUDA 7.5 and 8.0 do not. The arithmetic operators defined here are
// for users to write similar CUDA code in CUDA 7.5 and 8.0 as in
// CUDA 9.0 regarding the half data type.
// ROCM has built-in arithmetic operators as not defined
// __HIP_NO_HALF_OPERATORS__
#if defined(PADDLE_CUDA_FP16) && !defined(__HIPCC__) && CUDA_VERSION < 9000
DEVICE inline half operator+(const half& a, const half& b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
  return __hadd(a, b);
#else
  float res = static_cast<float>(float16(a)) + static_cast<float>(float16(b));
  return half(float16(res));
#endif
}

DEVICE inline half operator-(const half& a, const half& b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
  return __hsub(a, b);
#else
  float res = static_cast<float>(float16(a)) - static_cast<float>(float16(b));
  return half(float16(res));
#endif
}

DEVICE inline half operator*(const half& a, const half& b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
  return __hmul(a, b);
#else
  float res = static_cast<float>(float16(a)) * static_cast<float>(float16(b));
  return half(float16(res));
#endif
}

DEVICE inline half operator/(const half& a, const half& b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
  float num = __half2float(a);
  float denom = __half2float(b);
  return __float2half(num / denom);
#else
  float res = static_cast<float>(float16(a)) / static_cast<float>(float16(b));
  return half(float16(res));
#endif
}

DEVICE inline half operator-(const half& a) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
  return __hneg(a);
#else
  float res = -static_cast<float>(float16(a));
  return half(float16(res));
#endif
}

#ifndef PADDLE_WITH_HIP  // not defined __HIP_NO_HALF_OPERATORS__
DEVICE inline half& operator+=(half& a, const half& b) {  // NOLINT
  a = a + b;
  return a;
}

DEVICE inline half& operator-=(half& a, const half& b) {  // NOLINT
  a = a - b;
  return a;
}

DEVICE inline half& operator*=(half& a, const half& b) {  // NOLINT
  a = a * b;
  return a;
}

DEVICE inline half& operator/=(half& a, const half& b) {  // NOLINT
  a = a / b;
  return a;
}
#endif

DEVICE inline bool operator==(const half& a, const half& b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
  return __heq(a, b);
#else
  return static_cast<float>(float16(a)) == static_cast<float>(float16(b));
#endif
}

DEVICE inline bool operator!=(const half& a, const half& b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
  return __hne(a, b);
#else
  return static_cast<float>(float16(a)) != static_cast<float>(float16(b));
#endif
}

DEVICE inline bool operator<(const half& a, const half& b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
  return __hlt(a, b);
#else
  return static_cast<float>(float16(a)) < static_cast<float>(float16(b));
#endif
}

DEVICE inline bool operator<=(const half& a, const half& b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
  return __hle(a, b);
#else
  return static_cast<float>(float16(a)) <= static_cast<float>(float16(b));
#endif
}

DEVICE inline bool operator>(const half& a, const half& b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
  return __hgt(a, b);
#else
  return static_cast<float>(float16(a)) > static_cast<float>(float16(b));
#endif
}

DEVICE inline bool operator>=(const half& a, const half& b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
  return __hge(a, b);
#else
  return static_cast<float>(float16(a)) >= static_cast<float>(float16(b));
#endif
}

#endif  // PADDLE_CUDA_FP16

// Arithmetic operators for float16 on GPU
#if defined(PADDLE_CUDA_FP16)
// HIPCC has compile error if call __device__ function __hadd, __hsub, etc.
// in __host__ __device__ function
#if defined(__HIPCC__)
DEVICE inline float16 operator+(const float16& a, const float16& b) {
  return float16(__hadd(half(a), half(b)));
}
HOST inline float16 operator+(const float16& a, const float16& b) {
  return float16(static_cast<float>(a) + static_cast<float>(b));
}
#else
HOSTDEVICE inline float16 operator+(const float16& a, const float16& b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
  return float16(__hadd(half(a), half(b)));
#else
  return float16(static_cast<float>(a) + static_cast<float>(b));
#endif
}
#endif

#if defined(__HIPCC__)
DEVICE inline float16 operator-(const float16& a, const float16& b) {
  return float16(__hsub(half(a), half(b)));
}
HOST inline float16 operator-(const float16& a, const float16& b) {
  return float16(static_cast<float>(a) - static_cast<float>(b));
}
#else
HOSTDEVICE inline float16 operator-(const float16& a, const float16& b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
  return float16(__hsub(half(a), half(b)));
#else
  return float16(static_cast<float>(a) - static_cast<float>(b));
#endif
}
#endif

#if defined(__HIPCC__)
DEVICE inline float16 operator*(const float16& a, const float16& b) {
  return float16(__hmul(half(a), half(b)));
}
HOST inline float16 operator*(const float16& a, const float16& b) {
  return float16(static_cast<float>(a) * static_cast<float>(b));
}
#else
HOSTDEVICE inline float16 operator*(const float16& a, const float16& b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
  return float16(__hmul(half(a), half(b)));
#else
  return float16(static_cast<float>(a) * static_cast<float>(b));
#endif
}
#endif

#if defined(__HIPCC__)
DEVICE inline float16 operator/(const float16& a, const float16& b) {
  return float16(__hdiv(half(a), half(b)));
}
HOST inline float16 operator/(const float16& a, const float16& b) {
  return float16(static_cast<float>(a) / static_cast<float>(b));
}
#else
HOSTDEVICE inline float16 operator/(const float16& a, const float16& b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
  // TODO(kexinzhao): check which cuda version starts to support __hdiv
  float num = __half2float(half(a));
  float denom = __half2float(half(b));
  return float16(num / denom);
#else
  return float16(static_cast<float>(a) / static_cast<float>(b));
#endif
}
#endif

#if defined(__HIPCC__)
DEVICE inline float16 operator-(const float16& a) {
  return float16(__hneg(half(a)));
}
HOST inline float16 operator-(const float16& a) {
  float16 res;
  res.x = a.x ^ 0x8000;
  return res;
}
#else
HOSTDEVICE inline float16 operator-(const float16& a) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
  return float16(__hneg(half(a)));
#else
  float16 res;
  res.x = a.x ^ 0x8000;
  return res;
#endif
}
#endif

HOSTDEVICE inline float16& operator+=(float16& a, const float16& b) {  // NOLINT
  a = a + b;
  return a;
}

HOSTDEVICE inline float16& operator-=(float16& a, const float16& b) {  // NOLINT
  a = a - b;
  return a;
}

HOSTDEVICE inline float16& operator*=(float16& a, const float16& b) {  // NOLINT
  a = a * b;
  return a;
}

HOSTDEVICE inline float16& operator/=(float16& a, const float16& b) {  // NOLINT
  a = a / b;
  return a;
}

// HIPCC has compile error if call __device__ function __heq, __hne, etc.
// in __host__ __device__ function
#if defined(__HIPCC__)
DEVICE inline bool operator==(const float16& a, const float16& b) {
  return __heq(half(a), half(b));
}
HOST inline bool operator==(const float16& a, const float16& b) {
  return static_cast<float>(a) == static_cast<float>(b);
}
#else  // __HIPCC__
HOSTDEVICE inline bool operator==(const float16& a, const float16& b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
  return __heq(half(a), half(b));
#else
  return static_cast<float>(a) == static_cast<float>(b);
#endif
}
#endif  // __HIPCC__

#if defined(__HIPCC__)
DEVICE inline bool operator!=(const float16& a, const float16& b) {
  return __hne(half(a), half(b));
}
HOST inline bool operator!=(const float16& a, const float16& b) {
  return static_cast<float>(a) != static_cast<float>(b);
}
#else  // __HIPCC__
HOSTDEVICE inline bool operator!=(const float16& a, const float16& b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
  return __hne(half(a), half(b));
#else
  return static_cast<float>(a) != static_cast<float>(b);
#endif
}
#endif  // __HIPCC__

#if defined(__HIPCC__)
DEVICE inline bool operator<(const float16& a, const float16& b) {
  return __hlt(half(a), half(b));
}
HOST inline bool operator<(const float16& a, const float16& b) {
  return static_cast<float>(a) < static_cast<float>(b);
}
#else  // __HIPCC__
HOSTDEVICE inline bool operator<(const float16& a, const float16& b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
  return __hlt(half(a), half(b));
#else
  return static_cast<float>(a) < static_cast<float>(b);
#endif
}
#endif  // __HIPCC__

#if defined(__HIPCC__)
DEVICE inline bool operator<=(const float16& a, const float16& b) {
  return __hle(half(a), half(b));
}
HOST inline bool operator<=(const float16& a, const float16& b) {
  return static_cast<float>(a) <= static_cast<float>(b);
}
#else  // __HIPCC__
HOSTDEVICE inline bool operator<=(const float16& a, const float16& b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
  return __hle(half(a), half(b));
#else
  return static_cast<float>(a) <= static_cast<float>(b);
#endif
}
#endif  // __HIPCC__

#if defined(__HIPCC__)
DEVICE inline bool operator>(const float16& a, const float16& b) {
  return __hgt(half(a), half(b));
}
HOST inline bool operator>(const float16& a, const float16& b) {
  return static_cast<float>(a) > static_cast<float>(b);
}
#else  // __HIPCC__
HOSTDEVICE inline bool operator>(const float16& a, const float16& b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
  return __hgt(half(a), half(b));
#else
  return static_cast<float>(a) > static_cast<float>(b);
#endif
}
#endif  // __HIPCC__

#if defined(__HIPCC__)
DEVICE inline bool operator>=(const float16& a, const float16& b) {
  return __hge(half(a), half(b));
}
HOST inline bool operator>=(const float16& a, const float16& b) {
  return static_cast<float>(a) >= static_cast<float>(b);
}
#else  // __HIPCC__
HOSTDEVICE inline bool operator>=(const float16& a, const float16& b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
  return __hge(half(a), half(b));
#else
  return static_cast<float>(a) >= static_cast<float>(b);
#endif
}
#endif  // __HIPCC__

// Arithmetic operators for float16 on ARMv8.2-A CPU
#elif defined(PADDLE_WITH_NATIVE_FP16)
inline float16 operator+(const float16& a, const float16& b) {
  float16 res;
  asm volatile(
      "ld1 {v0.h}[0], [%[a_ptr]]\n"
      "ld1 {v1.h}[0], [%[b_ptr]]\n"
      "fadd h0, h0, h1\n"
      "st1 {v0.h}[0], [%[res_ptr]]\n"
      :  // outputs
      :  // inputs
      [a_ptr] "r"(&(a.x)), [b_ptr] "r"(&(b.x)),
      [res_ptr] "r"(&(res.x))
      :  // clobbers
      "memory", "v0", "v1");
  return res;
}

inline float16 operator-(const float16& a, const float16& b) {
  float16 res;
  asm volatile(
      "ld1 {v0.h}[0], [%[a_ptr]]\n"
      "ld1 {v1.h}[0], [%[b_ptr]]\n"
      "fsub h0, h0, h1\n"
      "st1 {v0.h}[0], [%[res_ptr]]\n"
      :  // outputs
      :  // inputs
      [a_ptr] "r"(&(a.x)), [b_ptr] "r"(&(b.x)),
      [res_ptr] "r"(&(res.x))
      :  // clobbers
      "memory", "v0", "v1");
  return res;
}

inline float16 operator*(const float16& a, const float16& b) {
  float16 res;
  asm volatile(
      "ld1 {v0.h}[0], [%[a_ptr]]\n"
      "ld1 {v1.h}[0], [%[b_ptr]]\n"
      "fmul h0, h0, h1\n"
      "st1 {v0.h}[0], [%[res_ptr]]\n"
      :  // outputs
      :  // inputs
      [a_ptr] "r"(&(a.x)), [b_ptr] "r"(&(b.x)),
      [res_ptr] "r"(&(res.x))
      :  // clobbers
      "memory", "v0", "v1");
  return res;
}

inline float16 operator/(const float16& a, const float16& b) {
  float16 res;
  asm volatile(
      "ld1 {v0.h}[0], [%[a_ptr]]\n"
      "ld1 {v1.h}[0], [%[b_ptr]]\n"
      "fdiv h0, h0, h1\n"
      "st1 {v0.h}[0], [%[res_ptr]]\n"
      :  // outputs
      :  // inputs
      [a_ptr] "r"(&(a.x)), [b_ptr] "r"(&(b.x)),
      [res_ptr] "r"(&(res.x))
      :  // clobbers
      "memory", "v0", "v1");
  return res;
}

inline float16 operator-(const float16& a) {
  float16 res;
  asm volatile(
      "ld1 {v0.h}[0], [%[a_ptr]]\n"
      "fneg h0, h0\n"
      "st1 {v0.h}[0], [%[res_ptr]]\n"
      :  // outputs
      :  // inputs
      [a_ptr] "r"(&(a.x)),
      [res_ptr] "r"(&(res.x))
      :  // clobbers
      "memory", "v0");
  return res;
}

inline float16& operator+=(float16& a, const float16& b) {  // NOLINT
  a = a + b;
  return a;
}

inline float16& operator-=(float16& a, const float16& b) {  // NOLINT
  a = a - b;
  return a;
}

inline float16& operator*=(float16& a, const float16& b) {  // NOLINT
  a = a * b;
  return a;
}

inline float16& operator/=(float16& a, const float16& b) {  // NOLINT
  a = a / b;
  return a;
}

inline bool operator==(const float16& a, const float16& b) {
  uint16_t res;
  asm volatile(
      "ld1 {v0.h}[0], [%[a_ptr]]\n"
      "ld1 {v1.h}[0], [%[b_ptr]]\n"
      "fcmeq h0, h0, h1\n"
      "st1 {v0.h}[0], [%[res_ptr]]\n"
      :  // outputs
      :  // inputs
      [a_ptr] "r"(&(a.x)), [b_ptr] "r"(&(b.x)),
      [res_ptr] "r"(&res)
      :  // clobbers
      "memory", "v0", "v1");
  return (res & 0xffff) != 0;
}

inline bool operator!=(const float16& a, const float16& b) { return !(a == b); }

inline bool operator<(const float16& a, const float16& b) {
  uint16_t res;
  asm volatile(
      "ld1 {v1.h}[0], [%[a_ptr]]\n"
      "ld1 {v0.h}[0], [%[b_ptr]]\n"
      "fcmgt h0, h0, h1\n"
      "st1 {v0.h}[0], [%[res_ptr]]\n"
      :  // outputs
      :  // inputs
      [a_ptr] "r"(&(a.x)), [b_ptr] "r"(&(b.x)),
      [res_ptr] "r"(&res)
      :  // clobbers
      "memory", "v0", "v1");
  return (res & 0xffff) != 0;
}

inline bool operator<=(const float16& a, const float16& b) {
  uint16_t res;
  asm volatile(
      "ld1 {v1.h}[0], [%[a_ptr]]\n"
      "ld1 {v0.h}[0], [%[b_ptr]]\n"
      "fcmge h0, h0, h1\n"
      "st1 {v0.h}[0], [%[res_ptr]]\n"
      :  // outputs
      :  // inputs
      [a_ptr] "r"(&(a.x)), [b_ptr] "r"(&(b.x)),
      [res_ptr] "r"(&res)
      :  // clobbers
      "memory", "v0", "v1");
  return (res & 0xffff) != 0;
}

inline bool operator>(const float16& a, const float16& b) {
  uint16_t res;
  asm volatile(
      "ld1 {v0.h}[0], [%[a_ptr]]\n"
      "ld1 {v1.h}[0], [%[b_ptr]]\n"
      "fcmgt h0, h0, h1\n"
      "st1 {v0.h}[0], [%[res_ptr]]\n"
      :  // outputs
      :  // inputs
      [a_ptr] "r"(&(a.x)), [b_ptr] "r"(&(b.x)),
      [res_ptr] "r"(&res)
      :  // clobbers
      "memory", "v0", "v1");
  return (res & 0xffff) != 0;
}

inline bool operator>=(const float16& a, const float16& b) {
  uint16_t res;
  asm volatile(
      "ld1 {v0.h}[0], [%[a_ptr]]\n"
      "ld1 {v1.h}[0], [%[b_ptr]]\n"
      "fcmge h0, h0, h1\n"
      "st1 {v0.h}[0], [%[res_ptr]]\n"
      :  // outputs
      :  // inputs
      [a_ptr] "r"(&(a.x)), [b_ptr] "r"(&(b.x)),
      [res_ptr] "r"(&res)
      :  // clobbers
      "memory", "v0", "v1");
  return (res & 0xffff) != 0;
}

// Arithmetic operators for float16, software emulated on other CPU
#else
inline float16 operator+(const float16& a, const float16& b) {
  return float16(static_cast<float>(a) + static_cast<float>(b));
}

inline float16 operator-(const float16& a, const float16& b) {
  return float16(static_cast<float>(a) - static_cast<float>(b));
}

inline float16 operator*(const float16& a, const float16& b) {
  return float16(static_cast<float>(a) * static_cast<float>(b));
}

inline float16 operator/(const float16& a, const float16& b) {
  return float16(static_cast<float>(a) / static_cast<float>(b));
}

inline float16 operator-(const float16& a) {
  float16 res;
  res.x = a.x ^ 0x8000;
  return res;
}

inline float16& operator+=(float16& a, const float16& b) {  // NOLINT
  a = float16(static_cast<float>(a) + static_cast<float>(b));
  return a;
}

inline float16& operator-=(float16& a, const float16& b) {  // NOLINT
  a = float16(static_cast<float>(a) - static_cast<float>(b));
  return a;
}

inline float16& operator*=(float16& a, const float16& b) {  // NOLINT
  a = float16(static_cast<float>(a) * static_cast<float>(b));
  return a;
}

inline float16& operator/=(float16& a, const float16& b) {  // NOLINT
  a = float16(static_cast<float>(a) / static_cast<float>(b));
  return a;
}

inline bool operator==(const float16& a, const float16& b) {
  return static_cast<float>(a) == static_cast<float>(b);
}

inline bool operator!=(const float16& a, const float16& b) {
  return static_cast<float>(a) != static_cast<float>(b);
}

inline bool operator<(const float16& a, const float16& b) {
  return static_cast<float>(a) < static_cast<float>(b);
}

inline bool operator<=(const float16& a, const float16& b) {
  return static_cast<float>(a) <= static_cast<float>(b);
}

inline bool operator>(const float16& a, const float16& b) {
  return static_cast<float>(a) > static_cast<float>(b);
}

inline bool operator>=(const float16& a, const float16& b) {
  return static_cast<float>(a) >= static_cast<float>(b);
}
#endif

HOSTDEVICE inline float16 raw_uint16_to_float16(uint16_t a) {
  float16 res;
  res.x = a;
  return res;
}

// HIPCC has compile error if call __device__ function __hisnan in __host__
// __device__ function
#if defined(PADDLE_CUDA_FP16) && defined(__HIPCC__)
DEVICE inline bool(isnan)(const float16& a) { return __hisnan(half(a)); }
HOST inline bool(isnan)(const float16& a) { return (a.x & 0x7fff) > 0x7c00; }
#else
HOSTDEVICE inline bool(isnan)(const float16& a) {
#if defined(PADDLE_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
  return __hisnan(half(a));
#else
  return (a.x & 0x7fff) > 0x7c00;
#endif
}
#endif

HOSTDEVICE inline bool(isinf)(const float16& a) {
  return (a.x & 0x7fff) == 0x7c00;
}

HOSTDEVICE inline bool(isfinite)(const float16& a) {
  return !((isnan)(a)) && !((isinf)(a));
}

HOSTDEVICE inline float16(abs)(const float16& a) {
#if defined(PADDLE_CUDA_FP16) && \
    (defined(__HIPCC__) || (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530))
  return float16(::fabs(static_cast<float>(a)));
#else
  return float16(std::abs(static_cast<float>(a)));
#endif
}

inline std::ostream& operator<<(std::ostream& os, const float16& a) {
  os << static_cast<float>(a);
  return os;
}

}  // namespace platform
}  // namespace paddle

namespace std {

// Override the std::is_pod::value for float16
// The reason is that different compilers implemented std::is_pod based on
// different C++ standards. float16 class is a plain old data in C++11 given
// that it is both trivial and standard_layout.
// However, std::is_pod in nvcc 8.0 host c++ compiler follows C++0x and is
// more restricted in that you cannot provide any customized
// constructor in float16. Hence, we override is_pod here following C++11
// so that .cu files can be successfully compiled by nvcc.
template <>
struct is_pod<paddle::platform::float16> {
  static const bool value =
      is_trivial<paddle::platform::float16>::value &&
      is_standard_layout<paddle::platform::float16>::value;
};

template <>
struct is_floating_point<paddle::platform::float16>
    : std::integral_constant<
          bool, std::is_same<paddle::platform::float16,
                             typename std::remove_cv<
                                 paddle::platform::float16>::type>::value> {};
template <>
struct is_signed<paddle::platform::float16> {
  static const bool value = true;
};

template <>
struct is_unsigned<paddle::platform::float16> {
  static const bool value = false;
};

inline bool isnan(const paddle::platform::float16& a) {
  return paddle::platform::isnan(a);
}

inline bool isinf(const paddle::platform::float16& a) {
  return paddle::platform::isinf(a);
}

template <>
struct numeric_limits<paddle::platform::float16> {
  static const bool is_specialized = true;
  static const bool is_signed = true;
  static const bool is_integer = false;
  static const bool is_exact = false;
  static const bool has_infinity = true;
  static const bool has_quiet_NaN = true;
  static const bool has_signaling_NaN = true;
  static const float_denorm_style has_denorm = denorm_present;
  static const bool has_denorm_loss = false;
  static const std::float_round_style round_style = std::round_to_nearest;
  static const bool is_iec559 = false;
  static const bool is_bounded = false;
  static const bool is_modulo = false;
  static const int digits = 11;
  static const int digits10 = 3;
  static const int max_digits10 = 5;
  static const int radix = 2;
  static const int min_exponent = -13;
  static const int min_exponent10 = -4;
  static const int max_exponent = 16;
  static const int max_exponent10 = 4;
  static const bool traps = true;
  static const bool tinyness_before = false;

  HOSTDEVICE static paddle::platform::float16(min)() {
    return paddle::platform::raw_uint16_to_float16(0x400);
  }
  HOSTDEVICE static paddle::platform::float16 lowest() {
    return paddle::platform::raw_uint16_to_float16(0xfbff);
  }
  HOSTDEVICE static paddle::platform::float16(max)() {
    return paddle::platform::raw_uint16_to_float16(0x7bff);
  }
  HOSTDEVICE static paddle::platform::float16 epsilon() {
    return paddle::platform::raw_uint16_to_float16(0x0800);
  }
  HOSTDEVICE static paddle::platform::float16 round_error() {
    return paddle::platform::float16(0.5);
  }
  HOSTDEVICE static paddle::platform::float16 infinity() {
    return paddle::platform::raw_uint16_to_float16(0x7c00);
  }
  HOSTDEVICE static paddle::platform::float16 quiet_NaN() {
    return paddle::platform::raw_uint16_to_float16(0x7e00);
  }
  HOSTDEVICE static paddle::platform::float16 signaling_NaN() {
    return paddle::platform::raw_uint16_to_float16(0x7e00);
  }
  HOSTDEVICE static paddle::platform::float16 denorm_min() {
    return paddle::platform::raw_uint16_to_float16(0x1);
  }
};

HOSTDEVICE inline paddle::platform::float16 abs(
    const paddle::platform::float16& a) {
  return paddle::platform::abs(a);
}

}  // namespace std
