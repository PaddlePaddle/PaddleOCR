#pragma once

#include <gflags/gflags.h>

DEFINE_bool(use_gpu, false, "Infering with GPU or CPU.");
DEFINE_int32(gpu_id, 0, "Device id of GPU to execute.");
DEFINE_int32(gpu_mem, 4000, "GPU id when infering with GPU.");
DEFINE_int32(cpu_threads, 10, "Num of threads with CPU.");
DEFINE_bool(enable_mkldnn, false, "Whether use mkldnn with CPU.");
DEFINE_bool(use_tensorrt, false, "Whether use tensorrt.");
DEFINE_string(precision, "fp32", "Precision be one of fp32/fp16/int8");
#ifndef OCR_EXPORTS
DEFINE_bool(benchmark, true, "Whether use benchmark.");
DEFINE_string(save_log_path, "./log_output/", "Save benchmark log path.");
#endif
// detection related
#ifndef OCR_EXPORTS
DEFINE_string(image_dir, "", "Dir of input image.");
DEFINE_string(det_model_dir, "", "Path of det inference model.");
#endif
DEFINE_int32(max_side_len, 960, "max_side_len of input image.");
DEFINE_double(det_db_thresh, 0.3, "Threshold of det_db_thresh.");
DEFINE_double(det_db_box_thresh, 0.5, "Threshold of det_db_box_thresh.");
DEFINE_double(det_db_unclip_ratio, 1.6, "Threshold of det_db_unclip_ratio.");
DEFINE_bool(use_polygon_score, false, "Whether use polygon score.");
DEFINE_bool(visualize, true, "Whether show the detection results.");
// classification related
#ifndef OCR_EXPORTS
DEFINE_bool(use_angle_cls, false, "Whether use use_angle_cls.");
DEFINE_string(cls_model_dir, "", "Path of cls inference model.");
#endif
DEFINE_double(cls_thresh, 0.9, "Threshold of cls_thresh.");
// recognition related
#ifndef OCR_EXPORTS
DEFINE_string(rec_model_dir, "", "Path of rec inference model.");
#endif
DEFINE_int32(rec_batch_num, 1, "rec_batch_num.");
#ifndef OCR_EXPORTS
DEFINE_string(char_list_file, "./ppocr_keys_v1.txt", "Path of dictionary.");
#endif
