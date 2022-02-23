/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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

#include <iostream>
#include <sstream>
#include <string>

namespace paddle {

//////////////// Exception handling and Error Message  /////////////////
#if !defined(_WIN32)
#define PD_UNLIKELY(expr) (__builtin_expect(static_cast<bool>(expr), 0))
#define PD_LIKELY(expr) (__builtin_expect(static_cast<bool>(expr), 1))
#else
#define PD_UNLIKELY(expr) (expr)
#define PD_LIKELY(expr) (expr)
#endif

struct PD_Exception : public std::exception {
 public:
  template <typename... Args>
  explicit PD_Exception(const std::string& msg, const char* file, int line,
                        const char* default_msg) {
    std::ostringstream sout;
    if (msg.empty()) {
      sout << default_msg << "\n  [" << file << ":" << line << "]";
    } else {
      sout << msg << "\n  [" << file << ":" << line << "]";
    }
    err_msg_ = sout.str();
  }

  const char* what() const noexcept override { return err_msg_.c_str(); }

 private:
  std::string err_msg_;
};

class ErrorMessage {
 public:
  template <typename... Args>
  explicit ErrorMessage(const Args&... args) {
    build_string(args...);
  }

  void build_string() { oss << ""; }

  template <typename T>
  void build_string(const T& t) {
    oss << t;
  }

  template <typename T, typename... Args>
  void build_string(const T& t, const Args&... args) {
    build_string(t);
    build_string(args...);
  }

  std::string to_string() { return oss.str(); }

 private:
  std::ostringstream oss;
};

#if defined _WIN32
#define HANDLE_THE_ERROR try {
#define END_HANDLE_THE_ERROR            \
  }                                     \
  catch (const std::exception& e) {     \
    std::cerr << e.what() << std::endl; \
    throw e;                            \
  }
#else
#define HANDLE_THE_ERROR
#define END_HANDLE_THE_ERROR
#endif

#define PD_CHECK(COND, ...)                                               \
  do {                                                                    \
    if (PD_UNLIKELY(!(COND))) {                                           \
      auto __message__ = ::paddle::ErrorMessage(__VA_ARGS__).to_string(); \
      throw ::paddle::PD_Exception(__message__, __FILE__, __LINE__,       \
                                   "Expected " #COND                      \
                                   ", but it's not satisfied.");          \
    }                                                                     \
  } while (0)

#define PD_THROW(...)                                                   \
  do {                                                                  \
    auto __message__ = ::paddle::ErrorMessage(__VA_ARGS__).to_string(); \
    throw ::paddle::PD_Exception(__message__, __FILE__, __LINE__,       \
                                 "An error occurred.");                 \
  } while (0)

}  // namespace paddle
