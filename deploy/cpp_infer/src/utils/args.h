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

#include <gflags/gflags.h>

DECLARE_string(input);
DECLARE_string(save_path);
DECLARE_string(doc_orientation_classify_model_name);
DECLARE_string(doc_orientation_classify_model_dir);
DECLARE_string(doc_unwarping_model_name);
DECLARE_string(doc_unwarping_model_dir);
DECLARE_string(text_detection_model_name);
DECLARE_string(text_detection_model_dir);
DECLARE_string(textline_orientation_model_name);
DECLARE_string(textline_orientation_model_dir);
DECLARE_string(textline_orientation_batch_size);
DECLARE_string(text_recognition_model_name);
DECLARE_string(text_recognition_model_dir);
DECLARE_string(text_recognition_batch_size);
DECLARE_string(use_doc_orientation_classify);
DECLARE_string(use_doc_unwarping);
DECLARE_string(use_textline_orientation);
DECLARE_string(text_det_limit_side_len);
DECLARE_string(text_det_limit_type);
DECLARE_string(text_det_thresh);
DECLARE_string(text_det_box_thresh);
DECLARE_string(text_det_unclip_ratio);
DECLARE_string(text_det_input_shape);
DECLARE_string(text_rec_score_thresh);
DECLARE_string(text_rec_input_shape);
DECLARE_string(lang);
DECLARE_string(ocr_version);
DECLARE_string(device);
DECLARE_string(vis_font_dir);
DECLARE_string(precision);
DECLARE_string(enable_mkldnn);
DECLARE_string(mkldnn_cache_capacity);
DECLARE_string(cpu_threads);
DECLARE_string(thread_num);
DECLARE_string(paddlex_config);
