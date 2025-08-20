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

// Based on https://github.com/shouxieai/tensorRT_Pro

// Copyright (c) 2022 TensorRTPro

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software && associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, &&/||  sell
// copies of the Software, && to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice && this permission notice shall be included in
// all copies ||  substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <time.h>

#include <string>
#include <tuple>
#include <vector>

#if defined(_WIN32)
#define U_OS_WINDOWS
#else
#define U_OS_LINUX
#endif

namespace iLogger {

using namespace std;

enum class LogLevel : int {
  Debug = 5,
  Verbose = 4,
  Info = 3,
  Warning = 2,
  Error = 1,
  Fatal = 0
};

#define INFOD(...)                                                             \
  iLogger::__log_func(__FILE__, __LINE__, iLogger::LogLevel::Debug, __VA_ARGS__)
#define INFOV(...)                                                             \
  iLogger::__log_func(__FILE__, __LINE__, iLogger::LogLevel::Verbose,          \
                      __VA_ARGS__)
#define INFO(...)                                                              \
  iLogger::__log_func(__FILE__, __LINE__, iLogger::LogLevel::Info, __VA_ARGS__)
#define INFOW(...)                                                             \
  iLogger::__log_func(__FILE__, __LINE__, iLogger::LogLevel::Warning,          \
                      __VA_ARGS__)
#define INFOE(...)                                                             \
  iLogger::__log_func(__FILE__, __LINE__, iLogger::LogLevel::Error, __VA_ARGS__)
#define INFOF(...)                                                             \
  iLogger::__log_func(__FILE__, __LINE__, iLogger::LogLevel::Fatal, __VA_ARGS__)

string date_now();
string time_now();
string gmtime_now();
string gmtime(time_t t);
time_t gmtime2ctime(const string &gmt);
void sleep(int ms);

bool isfile(const string &file);
bool mkdir(const string &path);
bool mkdirs(const string &path);
bool delete_file(const string &path);
bool rmtree(const string &directory, bool ignore_fail = false);
bool exists(const string &path);
string format(const char *fmt, ...);
FILE *fopen_mkdirs(const string &path, const string &mode);
string file_name(const string &path, bool include_suffix = true);
string directory(const string &path);
long long timestamp_now();
double timestamp_now_float();
time_t last_modify(const string &file);
vector<uint8_t> load_file(const string &file);
string load_text_file(const string &file);
size_t file_size(const string &file);

bool begin_with(const string &str, const string &with);
bool end_with(const string &str, const string &with);
vector<string> split_string(const string &str, const std::string &spstr);
string replace_string(const string &str, const string &token,
                      const string &value, int nreplace = -1,
                      int *out_num_replace = nullptr);

// h[0-1], s[0-1], v[0-1]
// return, 0-255, 0-255, 0-255
tuple<uint8_t, uint8_t, uint8_t> hsv2rgb(float h, float s, float v);
tuple<uint8_t, uint8_t, uint8_t> random_color(int id);

//   abcdefg.pnga          *.png      > false
//   abcdefg.png           *.png      > true
//   abcdefg.png          a?cdefg.png > true
bool pattern_match(const char *str, const char *matcher,
                   bool igrnoe_case = true);
vector<string> find_files(const string &directory, const string &filter = "*",
                          bool findDirectory = false,
                          bool includeSubDirectory = false);

string align_blank(const string &input, int align_size, char blank = ' ');
bool save_file(const string &file, const vector<uint8_t> &data,
               bool mk_dirs = true);
bool save_file(const string &file, const string &data, bool mk_dirs = true);
bool save_file(const string &file, const void *data, size_t length,
               bool mk_dirs = true);

// 捕获：SIGINT(2)、SIGQUIT(3)
int while_loop();

// 关于logger的api
const char *level_string(LogLevel level);
void set_logger_save_directory(const string &loggerDirectory);

void set_log_level(LogLevel level);
LogLevel get_log_level();
void __log_func(const char *file, int line, LogLevel level, const char *fmt,
                ...);
void destroy_logger();

string base64_decode(const string &base64);
string base64_encode(const void *data, size_t size);

inline int upbound(int n, int align = 32) {
  return (n + align - 1) / align * align;
}
string join_dims(const vector<int64_t> &dims);
}; // namespace iLogger
