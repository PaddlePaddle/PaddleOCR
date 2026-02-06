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

#include <set>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#ifdef WITH_GPU
static constexpr const char *DEVICE = "gpu:0";
#else
static constexpr const char *DEVICE = "cpu";
#endif
class PaddlePredictorOption {
public:
  const std::vector<std::string> SUPPORT_RUN_MODE = {"paddle", "paddle_fp16",
                                                     "mkldnn", "mkldnn_bf16"};

  const std::vector<std::string> SUPPORT_DEVICE = {"gpu", "cpu"};

  const std::string &RunMode() const;
  const std::string &DeviceType() const;
  int DeviceId() const;
  int CpuThreads() const;
  const std::vector<std::string> &DeletePass() const;
  bool EnableNewIR() const;
  bool EnableCinn() const;
  int MkldnnCacheCapacity() const;
  const std::vector<std::string> &GetSupportRunMode() const;
  const std::vector<std::string> &GetSupportDevice() const;
  std::string DebugString() const;

  absl::Status SetRunMode(const std::string &run_mode);
  absl::Status SetDeviceType(const std::string &device_type);
  absl::Status SetDeviceId(int device_id);
  absl::Status SetCpuThreads(int cpu_threads);
  absl::Status SetMkldnnCacheCapacity(int mkldnn_cache_capacity);
  void SetDeletePass(const std::vector<std::string> &delete_pass);
  void SetEnableNewIR(bool enable_new_ir);
  void SetEnableCinn(bool enable_cinn);

private:
  std::string run_mode_ = "paddle";
  std::string device_type_ = DEVICE;
  int device_id_ = 0;
  int cpu_threads_ = 10;
  std::vector<std::string> delete_pass_ = {};
  bool enable_new_ir_ = true;
  bool enable_cinn_ = false;
  int mkldnn_cache_capacity_ = 10;
};
