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

#include "ilogger.h"

#include <math.h>
#include <signal.h>
#include <stdarg.h>
#include <string.h>
#include <time.h>

#include <algorithm>
#include <atomic>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <sstream>
#include <stack>
#include <string>
#include <thread>
#include <vector>

#if defined(U_OS_WINDOWS)
#define HAS_UUID
#include <Shlwapi.h>
#include <Windows.h>
#include <wingdi.h>
#pragma comment(lib, "shlwapi.lib")
#pragma comment(lib, "ole32.lib")
#pragma comment(lib, "gdi32.lib")
#undef min
#undef max
#endif

#if defined(U_OS_LINUX)
//# include <sys/io.h>
#include <dirent.h>
#include <stdarg.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#define strtok_s strtok_r
#endif

#if defined(U_OS_LINUX)
#define __GetTimeBlock                                                         \
  time_t timep;                                                                \
  time(&timep);                                                                \
  tm &t = *(tm *)localtime(&timep);
#endif

#if defined(U_OS_WINDOWS)
#define __GetTimeBlock                                                         \
  tm t;                                                                        \
  _getsystime(&t);
#endif

namespace iLogger {

using namespace std;

const char *level_string(LogLevel level) {
  switch (level) {
  case LogLevel::Debug:
    return "debug";
  case LogLevel::Verbose:
    return "verbo";
  case LogLevel::Info:
    return "info";
  case LogLevel::Warning:
    return "warn";
  case LogLevel::Error:
    return "error";
  case LogLevel::Fatal:
    return "fatal";
  default:
    return "unknown";
  }
}

std::tuple<uint8_t, uint8_t, uint8_t> hsv2bgr(float h, float s, float v) {
  const int h_i = static_cast<int>(h * 6);
  const float f = h * 6 - h_i;
  const float p = v * (1 - s);
  const float q = v * (1 - f * s);
  const float t = v * (1 - (1 - f) * s);
  float r, g, b;
  switch (h_i) {
  case 0:
    r = v;
    g = t;
    b = p;
    break;
  case 1:
    r = q;
    g = v;
    b = p;
    break;
  case 2:
    r = p;
    g = v;
    b = t;
    break;
  case 3:
    r = p;
    g = q;
    b = v;
    break;
  case 4:
    r = t;
    g = p;
    b = v;
    break;
  case 5:
    r = v;
    g = p;
    b = q;
    break;
  default:
    r = 1;
    g = 1;
    b = 1;
    break;
  }
  return make_tuple(static_cast<uint8_t>(b * 255),
                    static_cast<uint8_t>(g * 255),
                    static_cast<uint8_t>(r * 255));
}

std::tuple<uint8_t, uint8_t, uint8_t> random_color(int id) {
  float h_plane = ((((unsigned int)id << 2) ^ 0x937151) % 100) / 100.0f;
  ;
  float s_plane = ((((unsigned int)id << 3) ^ 0x315793) % 100) / 100.0f;
  return hsv2bgr(h_plane, s_plane, 1);
}

string date_now() {
  char time_string[20];
  __GetTimeBlock;

  sprintf(time_string, "%04d-%02d-%02d", t.tm_year + 1900, t.tm_mon + 1,
          t.tm_mday);
  return time_string;
}

string time_now() {
  char time_string[20];
  __GetTimeBlock;

  sprintf(time_string, "%04d-%02d-%02d %02d:%02d:%02d", t.tm_year + 1900,
          t.tm_mon + 1, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec);
  return time_string;
}

size_t file_size(const string &file) {
#if defined(U_OS_LINUX)
  struct stat st;
  stat(file.c_str(), &st);
  return st.st_size;
#elif defined(U_OS_WINDOWS)
  WIN32_FIND_DATAA find_data;
  HANDLE hFind = FindFirstFileA(file.c_str(), &find_data);
  if (hFind == INVALID_HANDLE_VALUE)
    return 0;

  FindClose(hFind);
  return (uint64_t)find_data.nFileSizeLow |
         ((uint64_t)find_data.nFileSizeHigh << 32);
#endif
}

time_t last_modify(const string &file) {
#if defined(U_OS_LINUX)
  struct stat st;
  stat(file.c_str(), &st);
  return st.st_mtim.tv_sec;
#elif defined(U_OS_WINDOWS)
  INFOW("LastModify has not support on windows os");
  return 0;
#endif
}

void sleep(int ms) { this_thread::sleep_for(std::chrono::milliseconds(ms)); }

int get_month_by_name(char *month) {
  if (strcmp(month, "Jan") == 0)
    return 0;
  if (strcmp(month, "Feb") == 0)
    return 1;
  if (strcmp(month, "Mar") == 0)
    return 2;
  if (strcmp(month, "Apr") == 0)
    return 3;
  if (strcmp(month, "May") == 0)
    return 4;
  if (strcmp(month, "Jun") == 0)
    return 5;
  if (strcmp(month, "Jul") == 0)
    return 6;
  if (strcmp(month, "Aug") == 0)
    return 7;
  if (strcmp(month, "Sep") == 0)
    return 8;
  if (strcmp(month, "Oct") == 0)
    return 9;
  if (strcmp(month, "Nov") == 0)
    return 10;
  if (strcmp(month, "Dec") == 0)
    return 11;
  return -1;
}

int get_week_day_by_name(char *wday) {
  if (strcmp(wday, "Sun") == 0)
    return 0;
  if (strcmp(wday, "Mon") == 0)
    return 1;
  if (strcmp(wday, "Tue") == 0)
    return 2;
  if (strcmp(wday, "Wed") == 0)
    return 3;
  if (strcmp(wday, "Thu") == 0)
    return 4;
  if (strcmp(wday, "Fri") == 0)
    return 5;
  if (strcmp(wday, "Sat") == 0)
    return 6;
  return -1;
}

time_t gmtime2ctime(const string &gmt) {
  char week[4] = {0};
  char month[4] = {0};
  tm date;
  sscanf(gmt.c_str(), "%3s, %2d %3s %4d %2d:%2d:%2d GMT", week, &date.tm_mday,
         month, &date.tm_year, &date.tm_hour, &date.tm_min, &date.tm_sec);
  date.tm_mon = get_month_by_name(month);
  date.tm_wday = get_week_day_by_name(week);
  date.tm_year = date.tm_year - 1900;
  return mktime(&date);
}

string gmtime(time_t t) {
  t += 28800;
  tm *gmt = ::gmtime(&t);

  // http://en.cppreference.com/w/c/chrono/strftime
  // e.g.: Sat, 22 Aug 2015 11:48:50 GMT
  //       5+   3+4+   5+   9+       3   = 29
  const char *fmt = "%a, %d %b %Y %H:%M:%S GMT";
  char tstr[30];
  strftime(tstr, sizeof(tstr), fmt, gmt);
  return tstr;
}

string gmtime_now() { return gmtime(time(nullptr)); }

bool mkdir(const string &path) {
#ifdef U_OS_WINDOWS
  return CreateDirectoryA(path.c_str(), nullptr);
#else
  return ::mkdir(path.c_str(), 0755) == 0;
#endif
}

bool mkdirs(const string &path) {
  if (path.empty())
    return false;
  if (exists(path))
    return true;

  string _path = path;
  char *dir_ptr = (char *)_path.c_str();
  char *iter_ptr = dir_ptr;

  bool keep_going = *iter_ptr != 0;
  while (keep_going) {
    if (*iter_ptr == 0)
      keep_going = false;

#ifdef U_OS_WINDOWS
    if (*iter_ptr == '/' || *iter_ptr == '\\' || *iter_ptr == 0) {
#else
    if ((*iter_ptr == '/' && iter_ptr != dir_ptr) || *iter_ptr == 0) {
#endif
      char old = *iter_ptr;
      *iter_ptr = 0;
      if (!exists(dir_ptr)) {
        if (!mkdir(dir_ptr)) {
          if (!exists(dir_ptr)) {
            INFOE("mkdirs %s return false.", dir_ptr);
            return false;
          }
        }
      }
      *iter_ptr = old;
    }
    iter_ptr++;
  }
  return true;
}

bool isfile(const string &file) {
#if defined(U_OS_LINUX)
  struct stat st;
  stat(file.c_str(), &st);
  return S_ISREG(st.st_mode);
#elif defined(U_OS_WINDOWS)
  INFOW("is_file has not support on windows os");
  return 0;
#endif
}

FILE *fopen_mkdirs(const string &path, const string &mode) {
  FILE *f = fopen(path.c_str(), mode.c_str());
  if (f)
    return f;

  int p = path.rfind('/');

#if defined(U_OS_WINDOWS)
  int e = path.rfind('\\');
  p = std::max(p, e);
#endif
  if (p == -1)
    return nullptr;

  string directory = path.substr(0, p);
  if (!mkdirs(directory))
    return nullptr;

  return fopen(path.c_str(), mode.c_str());
}

bool exists(const string &path) {
#ifdef U_OS_WINDOWS
  return ::PathFileExistsA(path.c_str());
#elif defined(U_OS_LINUX)
  return access(path.c_str(), R_OK) == 0;
#endif
}

string format(const char *fmt, ...) {
  va_list vl;
  va_start(vl, fmt);
  char buffer[2048];
  vsnprintf(buffer, sizeof(buffer), fmt, vl);
  return buffer;
}

string file_name(const string &path, bool include_suffix) {
  if (path.empty())
    return "";

  int p = path.rfind('/');

#ifdef U_OS_WINDOWS
  int e = path.rfind('\\');
  p = std::max(p, e);
#endif
  p += 1;

  // include suffix
  if (include_suffix)
    return path.substr(p);

  int u = path.rfind('.');
  if (u == -1)
    return path.substr(p);

  if (u <= p)
    u = path.size();
  return path.substr(p, u - p);
}

string directory(const string &path) {
  if (path.empty())
    return ".";

  int p = path.rfind('/');

#ifdef U_OS_WINDOWS
  int e = path.rfind('\\');
  p = std::max(p, e);
#endif
  if (p == -1)
    return ".";

  return path.substr(0, p + 1);
}

bool begin_with(const string &str, const string &with) {
  if (str.length() < with.length())
    return false;
  return strncmp(str.c_str(), with.c_str(), with.length()) == 0;
}

bool end_with(const string &str, const string &with) {
  if (str.length() < with.length())
    return false;

  return strncmp(str.c_str() + str.length() - with.length(), with.c_str(),
                 with.length()) == 0;
}

long long timestamp_now() {
#ifdef _WIN32
  FILETIME ft;
  GetSystemTimeAsFileTime(&ft);
  ULARGE_INTEGER li;
  li.LowPart = ft.dwLowDateTime;
  li.HighPart = ft.dwHighDateTime;
  return (li.QuadPart - 116444736000000000LL) / 10000LL;
#else
  return chrono::duration_cast<chrono::milliseconds>(
             chrono::system_clock::now().time_since_epoch())
      .count();
#endif
}

double timestamp_now_float() {
#ifdef _WIN32
  FILETIME ft;
  GetSystemTimeAsFileTime(&ft);
  ULARGE_INTEGER li;
  li.LowPart = ft.dwLowDateTime;
  li.HighPart = ft.dwHighDateTime;
  return ((li.QuadPart - 116444736000000000LL) / 10000.0);
#else
  return chrono::duration_cast<chrono::microseconds>(
             chrono::system_clock::now().time_since_epoch())
             .count() /
         1000.0;
#endif
}

static struct Logger {
  mutex logger_lock_;
  string logger_directory;
  LogLevel logger_level{LogLevel::Info};
  vector<string> cache_, local_;
  shared_ptr<thread> flush_thread_;
  atomic<bool> keep_run_{false};
  shared_ptr<FILE> handler;
  bool logger_shutdown{false};

  void write(const string &line) {
    lock_guard<mutex> l(logger_lock_);
    if (logger_shutdown)
      return;

    if (!keep_run_) {
      if (flush_thread_)
        return;

      cache_.reserve(1000);
      keep_run_ = true;
      flush_thread_.reset(new thread(std::bind(&Logger::flush_job, this)));
    }
    cache_.emplace_back(line);
  }

  void flush() {
    if (cache_.empty())
      return;

    {
      std::lock_guard<mutex> l(logger_lock_);
      std::swap(local_, cache_);
    }

    if (!local_.empty() && !logger_directory.empty()) {
      auto now = date_now();
      auto file = format("%s%s.txt", logger_directory.c_str(), now.c_str());
      if (!exists(file)) {
        handler.reset(fopen_mkdirs(file, "wb"), fclose);
      } else if (!handler) {
        handler.reset(fopen_mkdirs(file, "a+"), fclose);
      }

      if (handler) {
        for (auto &line : local_)
          fprintf(handler.get(), "%s\n", line.c_str());
        fflush(handler.get());
        handler.reset();
      }
    }
    local_.clear();
  }

  void flush_job() {
    auto tick_begin = timestamp_now();
    std::vector<string> local;
    while (keep_run_) {
      if (timestamp_now() - tick_begin < 1000) {
        this_thread::sleep_for(std::chrono::milliseconds(100));
        continue;
      }

      tick_begin = timestamp_now();
      flush();
    }
    flush();
  }

  void set_save_directory(const string &loggerDirectory) {
    logger_directory = loggerDirectory;

    if (logger_directory.empty())
      logger_directory = ".";

#if defined(U_OS_LINUX)
    if (logger_directory.back() != '/') {
      logger_directory.push_back('/');
    }
#endif

#if defined(U_OS_WINDOWS)
    if (logger_directory.back() != '/' && logger_directory.back() != '\\') {
      logger_directory.push_back('/');
    }
#endif
  }

  void set_logger_level(LogLevel level) { logger_level = level; }

  void close() {
    {
      lock_guard<mutex> l(logger_lock_);
      if (logger_shutdown)
        return;

      logger_shutdown = true;
    };

    if (!keep_run_)
      return;
    keep_run_ = false;
    flush_thread_->join();
    flush_thread_.reset();
    handler.reset();
  }

  virtual ~Logger() { close(); }
} __g_logger;

void destroy_logger() { __g_logger.close(); }

static void remove_color_text(char *buffer) {
  //"\033[31m%s\033[0m"
  char *p = buffer;
  while (*p) {
    if (*p == 0x1B) {
      char np = *(p + 1);
      if (np == '[') {
        // has token
        char *t = p + 2;
        while (*t) {
          if (*t == 'm') {
            t = t + 1;
            char *k = p;
            while (*t) {
              *k++ = *t++;
            }
            *k = 0;
            break;
          }
          t++;
        }
      }
    }
    p++;
  }
}

void set_logger_save_directory(const string &loggerDirectory) {
  __g_logger.set_save_directory(loggerDirectory);
}

void set_log_level(LogLevel level) { __g_logger.set_logger_level(level); }

LogLevel get_log_level() { return __g_logger.logger_level; }

void __log_func(const char *file, int line, LogLevel level, const char *fmt,
                ...) {
  if (level > __g_logger.logger_level)
    return;

  string now = time_now();
  va_list vl;
  va_start(vl, fmt);

  char buffer[2048];
  string filename = file_name(file, true);
  int n = snprintf(buffer, sizeof(buffer), "[%s]", now.c_str());

#if defined(U_OS_WINDOWS)
  if (level == LogLevel::Fatal || level == LogLevel::Error) {
    n += snprintf(buffer + n, sizeof(buffer) - n, "[%s]", level_string(level));
  } else if (level == LogLevel::Warning) {
    n += snprintf(buffer + n, sizeof(buffer) - n, "[%s]", level_string(level));
  } else {
    n += snprintf(buffer + n, sizeof(buffer) - n, "[%s]", level_string(level));
  }
#elif defined(U_OS_LINUX)
  if (level == LogLevel::Fatal || level == LogLevel::Error) {
    n += snprintf(buffer + n, sizeof(buffer) - n, "[\033[31m%s\033[0m]",
                  level_string(level));
  } else if (level == LogLevel::Warning) {
    n += snprintf(buffer + n, sizeof(buffer) - n, "[\033[33m%s\033[0m]",
                  level_string(level));
  } else if (level == LogLevel::Info) {
    n += snprintf(buffer + n, sizeof(buffer) - n, "[\033[35m%s\033[0m]",
                  level_string(level));
  } else if (level == LogLevel::Verbose) {
    n += snprintf(buffer + n, sizeof(buffer) - n, "[\033[34m%s\033[0m]",
                  level_string(level));
  } else {
    n += snprintf(buffer + n, sizeof(buffer) - n, "[%s]", level_string(level));
  }
#endif

  n += snprintf(buffer + n, sizeof(buffer) - n, "[%s:%d]:", filename.c_str(),
                line);
  vsnprintf(buffer + n, sizeof(buffer) - n, fmt, vl);

  if (level == LogLevel::Fatal || level == LogLevel::Error) {
    fprintf(stderr, "%s\n", buffer);
  } else if (level == LogLevel::Warning) {
    fprintf(stdout, "%s\n", buffer);
  } else {
    fprintf(stdout, "%s\n", buffer);
  }

  if (!__g_logger.logger_directory.empty()) {
#ifdef U_OS_LINUX
    // remove save color txt
    remove_color_text(buffer);
#endif
    __g_logger.write(buffer);
    if (level == LogLevel::Fatal) {
      __g_logger.flush();
    }
  }

  if (level == LogLevel::Fatal) {
    fflush(stdout);
    abort();
  }
}

string load_text_file(const string &file) {
  ifstream in(file, ios::in | ios::binary);
  if (!in.is_open())
    return {};

  in.seekg(0, ios::end);
  size_t length = in.tellg();

  string data;
  if (length > 0) {
    in.seekg(0, ios::beg);
    data.resize(length);

    in.read((char *)&data[0], length);
  }
  in.close();
  return data;
}

std::vector<uint8_t> load_file(const string &file) {
  ifstream in(file, ios::in | ios::binary);
  if (!in.is_open())
    return {};

  in.seekg(0, ios::end);
  size_t length = in.tellg();

  std::vector<uint8_t> data;
  if (length > 0) {
    in.seekg(0, ios::beg);
    data.resize(length);

    in.read((char *)&data[0], length);
  }
  in.close();
  return data;
}

bool alphabet_equal(char a, char b, bool ignore_case) {
  if (ignore_case) {
    a = a > 'a' && a < 'z' ? a - 'a' + 'A' : a;
    b = b > 'a' && b < 'z' ? b - 'a' + 'A' : b;
  }
  return a == b;
}

static bool pattern_match_body(const char *str, const char *matcher,
                               bool igrnoe_case) {
  //   abcdefg.pnga          *.png      > false
  //   abcdefg.png           *.png      > true
  //   abcdefg.png          a?cdefg.png > true

  if (!matcher || !*matcher || !str || !*str)
    return false;

  const char *ptr_matcher = matcher;
  while (*str) {
    if (*ptr_matcher == '?') {
      ptr_matcher++;
    } else if (*ptr_matcher == '*') {
      if (*(ptr_matcher + 1)) {
        if (pattern_match_body(str, ptr_matcher + 1, igrnoe_case))
          return true;
      } else {
        return true;
      }
    } else if (!alphabet_equal(*ptr_matcher, *str, igrnoe_case)) {
      return false;
    } else {
      if (*ptr_matcher)
        ptr_matcher++;
      else
        return false;
    }
    str++;
  }

  while (*ptr_matcher) {
    if (*ptr_matcher != '*')
      return false;
    ptr_matcher++;
  }
  return true;
}

bool pattern_match(const char *str, const char *matcher, bool igrnoe_case) {
  //   abcdefg.pnga          *.png      > false
  //   abcdefg.png           *.png      > true
  //   abcdefg.png          a?cdefg.png > true

  if (!matcher || !*matcher || !str || !*str)
    return false;

  char filter[500];
  strcpy(filter, matcher);

  vector<const char *> arr;
  char *ptr_str = filter;
  char *ptr_prev_str = ptr_str;
  while (*ptr_str) {
    if (*ptr_str == ';') {
      *ptr_str = 0;
      arr.push_back(ptr_prev_str);
      ptr_prev_str = ptr_str + 1;
    }
    ptr_str++;
  }

  if (*ptr_prev_str)
    arr.push_back(ptr_prev_str);

  for (int i = 0; i < arr.size(); ++i) {
    if (pattern_match_body(str, arr[i], igrnoe_case))
      return true;
  }
  return false;
}

#ifdef U_OS_WINDOWS
vector<string> find_files(const string &directory, const string &filter,
                          bool findDirectory, bool includeSubDirectory) {
  string realpath = directory;
  if (realpath.empty())
    realpath = "./";

  char backchar = realpath.back();
  if (backchar != '\\' && backchar != '/')
    realpath += "/";

  vector<string> out;
  _WIN32_FIND_DATAA find_data;
  stack<string> ps;
  ps.push(realpath);

  while (!ps.empty()) {
    string search_path = ps.top();
    ps.pop();

    HANDLE hFind = FindFirstFileA((search_path + "*").c_str(), &find_data);
    if (hFind != INVALID_HANDLE_VALUE) {
      do {
        if (strcmp(find_data.cFileName, ".") == 0 ||
            strcmp(find_data.cFileName, "..") == 0)
          continue;

        if (!findDirectory &&
                (find_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) !=
                    FILE_ATTRIBUTE_DIRECTORY ||
            findDirectory &&
                (find_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) ==
                    FILE_ATTRIBUTE_DIRECTORY) {
          if (PathMatchSpecA(find_data.cFileName, filter.c_str()))
            out.push_back(search_path + find_data.cFileName);
        }

        if (includeSubDirectory &&
            (find_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) ==
                FILE_ATTRIBUTE_DIRECTORY)
          ps.push(search_path + find_data.cFileName + "/");

      } while (FindNextFileA(hFind, &find_data));
      FindClose(hFind);
    }
  }
  return out;
}
#endif

#ifdef U_OS_LINUX
vector<string> find_files(const string &directory, const string &filter,
                          bool findDirectory, bool includeSubDirectory) {
  string realpath = directory;
  if (realpath.empty())
    realpath = "./";

  char backchar = realpath.back();
  if (backchar != '\\' && backchar != '/')
    realpath += "/";

  struct dirent *fileinfo;
  DIR *handle;
  stack<string> ps;
  vector<string> out;
  ps.push(realpath);

  while (!ps.empty()) {
    string search_path = ps.top();
    ps.pop();

    handle = opendir(search_path.c_str());
    if (handle != 0) {
      while (fileinfo = readdir(handle)) {
        struct stat file_stat;
        if (strcmp(fileinfo->d_name, ".") == 0 ||
            strcmp(fileinfo->d_name, "..") == 0)
          continue;

        if (lstat((search_path + fileinfo->d_name).c_str(), &file_stat) < 0)
          continue;

        if (!findDirectory && !S_ISDIR(file_stat.st_mode) ||
            findDirectory && S_ISDIR(file_stat.st_mode)) {
          if (pattern_match(fileinfo->d_name, filter.c_str()))
            out.push_back(search_path + fileinfo->d_name);
        }

        if (includeSubDirectory && S_ISDIR(file_stat.st_mode))
          ps.push(search_path + fileinfo->d_name + "/");
      }
      closedir(handle);
    }
  }
  return out;
}
#endif

string align_blank(const string &input, int align_size, char blank) {
  if (input.size() >= align_size)
    return input;
  string output = input;
  for (int i = 0; i < align_size - input.size(); ++i)
    output.push_back(blank);
  return output;
}

vector<string> split_string(const string &str, const std::string &spstr) {
  vector<string> res;
  if (str.empty())
    return res;
  if (spstr.empty())
    return {str};

  auto p = str.find(spstr);
  if (p == string::npos)
    return {str};

  res.reserve(5);
  string::size_type prev = 0;
  int lent = spstr.length();
  const char *ptr = str.c_str();

  while (p != string::npos) {
    int len = p - prev;
    if (len > 0) {
      res.emplace_back(str.substr(prev, len));
    }
    prev = p + lent;
    p = str.find(spstr, prev);
  }

  int len = str.length() - prev;
  if (len > 0) {
    res.emplace_back(str.substr(prev, len));
  }
  return res;
}

string replace_string(const string &str, const string &token,
                      const string &value, int nreplace, int *out_num_replace) {
  if (nreplace == -1) {
    nreplace = str.size();
  }

  if (nreplace == 0) {
    return str;
  }

  string result;
  result.resize(str.size());

  char *dest = &result[0];
  const char *src = str.c_str();
  const char *value_ptr = value.c_str();
  int old_nreplace = nreplace;
  bool keep = true;
  string::size_type pos = 0;
  string::size_type prev = 0;
  size_t token_length = token.length();
  size_t value_length = value.length();

  do {
    pos = str.find(token, pos);
    if (pos == string::npos) {
      keep = false;
      pos = str.length();
    } else {
      if (nreplace == 0) {
        pos = str.length();
        keep = false;
      } else {
        nreplace--;
      }
    }

    size_t copy_length = pos - prev;
    if (copy_length > 0) {
      size_t dest_length = dest - &result[0];

      // Extended memory space if needed.
      if (dest_length + copy_length > result.size()) {
        result.resize((result.size() + copy_length) * 1.2);
        dest = &result[dest_length];
      }

      memcpy(dest, src + prev, copy_length);
      dest += copy_length;
    }

    if (keep) {
      pos += token_length;
      prev = pos;

      size_t dest_length = dest - &result[0];

      // Extended memory space if needed.
      if (dest_length + value_length > result.size()) {
        result.resize((result.size() + value_length) * 1.2);
        dest = &result[dest_length];
      }
      memcpy(dest, value_ptr, value_length);
      dest += value_length;
    }
  } while (keep);

  if (out_num_replace) {
    *out_num_replace = old_nreplace - nreplace;
  }

  // Crop extra space
  size_t valid_size = dest - &result[0];
  result.resize(valid_size);
  return result;
}

bool save_file(const string &file, const void *data, size_t length,
               bool mk_dirs) {
  if (mk_dirs) {
    int p = (int)file.rfind('/');

#ifdef U_OS_WINDOWS
    int e = (int)file.rfind('\\');
    p = std::max(p, e);
#endif
    if (p != -1) {
      if (!mkdirs(file.substr(0, p)))
        return false;
    }
  }

  FILE *f = fopen(file.c_str(), "wb");
  if (!f)
    return false;

  if (data && length > 0) {
    if (fwrite(data, 1, length, f) != length) {
      fclose(f);
      return false;
    }
  }
  fclose(f);
  return true;
}

bool save_file(const string &file, const string &data, bool mk_dirs) {
  return save_file(file, data.data(), data.size(), mk_dirs);
}

bool save_file(const string &file, const vector<uint8_t> &data, bool mk_dirs) {
  return save_file(file, data.data(), data.size(), mk_dirs);
}

static volatile bool g_has_exit_signal = false;
static int g_signum = 0;
static void signal_callback_handler(int signum) {
  INFO("Capture interrupt signal.");
  g_has_exit_signal = true;
  g_signum = signum;
}

int while_loop() {
  signal(SIGINT, signal_callback_handler);
  // signal(SIGQUIT, signal_callback_handler);
  while (!g_has_exit_signal) {
    this_thread::yield();
  }
  INFO("Loop over.");
  return g_signum;
}

static unsigned char from_b64(unsigned char ch) {
  /* Inverse lookup map */
  static const unsigned char tab[128] = {
      255, 255, 255, 255,
      255, 255, 255, 255, /*  0 */
      255, 255, 255, 255,
      255, 255, 255, 255, /*  8 */
      255, 255, 255, 255,
      255, 255, 255, 255, /*  16 */
      255, 255, 255, 255,
      255, 255, 255, 255, /*  24 */
      255, 255, 255, 255,
      255, 255, 255, 255, /*  32 */
      255, 255, 255, 62,
      255, 255, 255, 63, /*  40 */
      52,  53,  54,  55,
      56,  57,  58,  59, /*  48 */
      60,  61,  255, 255,
      255, 200, 255, 255, /*  56   '=' is 200, on index 61 */
      255, 0,   1,   2,
      3,   4,   5,   6, /*  64 */
      7,   8,   9,   10,
      11,  12,  13,  14, /*  72 */
      15,  16,  17,  18,
      19,  20,  21,  22, /*  80 */
      23,  24,  25,  255,
      255, 255, 255, 255, /*  88 */
      255, 26,  27,  28,
      29,  30,  31,  32, /*  96 */
      33,  34,  35,  36,
      37,  38,  39,  40, /*  104 */
      41,  42,  43,  44,
      45,  46,  47,  48, /*  112 */
      49,  50,  51,  255,
      255, 255, 255, 255, /*  120 */
  };
  return tab[ch & 127];
}

string base64_decode(const string &base64) {
  if (base64.empty())
    return "";

  int len = base64.size();
  auto s = (const unsigned char *)base64.data();
  unsigned char a, b, c, d;
  int orig_len = len;
  int dec_len = 0;
  string out_data;

  auto end_s = s + base64.size();
  int count_eq = 0;
  while (*--end_s == '=') {
    count_eq++;
  }
  out_data.resize(len / 4 * 3 - count_eq);

  char *dst = const_cast<char *>(out_data.data());
  char *orig_dst = dst;
  while (len >= 4 && (a = from_b64(s[0])) != 255 &&
         (b = from_b64(s[1])) != 255 && (c = from_b64(s[2])) != 255 &&
         (d = from_b64(s[3])) != 255) {
    s += 4;
    len -= 4;
    if (a == 200 || b == 200)
      break; /* '=' can't be there */
    *dst++ = a << 2 | b >> 4;
    if (c == 200)
      break;
    *dst++ = b << 4 | c >> 2;
    if (d == 200)
      break;
    *dst++ = c << 6 | d;
  }
  dec_len = (dst - orig_dst);
  return out_data;
}

string base64_encode(const void *data, size_t size) {
  string encode_result;
  encode_result.reserve(size / 3 * 4 + (size % 3 != 0 ? 4 : 0));

  const unsigned char *current = static_cast<const unsigned char *>(data);
  static const char *base64_table =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  while (size > 2) {
    encode_result += base64_table[current[0] >> 2];
    encode_result +=
        base64_table[((current[0] & 0x03) << 4) + (current[1] >> 4)];
    encode_result +=
        base64_table[((current[1] & 0x0f) << 2) + (current[2] >> 6)];
    encode_result += base64_table[current[2] & 0x3f];

    current += 3;
    size -= 3;
  }

  if (size > 0) {
    encode_result += base64_table[current[0] >> 2];
    if (size % 3 == 1) {
      encode_result += base64_table[(current[0] & 0x03) << 4];
      encode_result += "==";
    } else if (size % 3 == 2) {
      encode_result +=
          base64_table[((current[0] & 0x03) << 4) + (current[1] >> 4)];
      encode_result += base64_table[(current[1] & 0x0f) << 2];
      encode_result += "=";
    }
  }
  return encode_result;
}

bool delete_file(const string &path) {
#ifdef U_OS_WINDOWS
  return DeleteFileA(path.c_str());
#else
  return ::remove(path.c_str()) == 0;
#endif
}

bool rmtree(const string &directory, bool ignore_fail) {
  if (directory.empty())
    return false;
  auto files = find_files(directory, "*", false);
  auto dirs = find_files(directory, "*", true);

  bool success = true;
  for (int i = 0; i < files.size(); ++i) {
    if (::remove(files[i].c_str()) != 0) {
      success = false;

      if (!ignore_fail) {
        return false;
      }
    }
  }

  dirs.insert(dirs.begin(), directory);
  for (int i = (int)dirs.size() - 1; i >= 0; --i) {
#ifdef U_OS_WINDOWS
    if (!::RemoveDirectoryA(dirs[i].c_str())) {
#else
    if (::rmdir(dirs[i].c_str()) != 0) {
#endif
      success = false;
      if (!ignore_fail)
        return false;
    }
  }
  return success;
}

string join_dims(const vector<int64_t> &dims) {
  stringstream output;
  char buf[64];
  const char *fmts[] = {"%d", " x %d"};
  for (int i = 0; i < dims.size(); ++i) {
    snprintf(buf, sizeof(buf), fmts[i != 0], dims[i]);
    output << buf;
  }
  return output.str();
}

}; // namespace iLogger
