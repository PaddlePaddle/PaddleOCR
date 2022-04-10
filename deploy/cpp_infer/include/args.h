// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <gflags/gflags.h>

// common args
DECLARE_bool(use_gpu);
DECLARE_bool(use_tensorrt);
DECLARE_int32(gpu_id);
DECLARE_int32(gpu_mem);
DECLARE_int32(cpu_threads);
DECLARE_bool(enable_mkldnn);
DECLARE_string(precision);
DECLARE_bool(benchmark);
DECLARE_string(output);
DECLARE_string(image_dir);
DECLARE_string(type);
// detection related
DECLARE_string(det_model_dir);
DECLARE_int32(max_side_len);
DECLARE_double(det_db_thresh);
DECLARE_double(det_db_box_thresh);
DECLARE_double(det_db_unclip_ratio);
DECLARE_bool(use_dilation);
DECLARE_string(det_db_score_mode);
DECLARE_bool(visualize);
// classification related
DECLARE_bool(use_angle_cls);
DECLARE_string(cls_model_dir);
DECLARE_double(cls_thresh);
DECLARE_int32(cls_batch_num);
// recognition related
DECLARE_string(rec_model_dir);
DECLARE_int32(rec_batch_num);
DECLARE_string(rec_char_dict_path);
// forward related
DECLARE_bool(det);
DECLARE_bool(rec);
DECLARE_bool(cls);
