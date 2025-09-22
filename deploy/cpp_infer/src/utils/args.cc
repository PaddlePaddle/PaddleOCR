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

#include "args.h"

DEFINE_string(input, "",
              "Data to be predicted, required. Local path of an image file.");
DEFINE_string(save_path, "./output/", "Path to save inference result files.");
DEFINE_string(doc_orientation_classify_model_name, "PP-LCNet_x1_0_doc_ori",
              "Name of the document image orientation classification model.");
DEFINE_string(
    doc_orientation_classify_model_dir, "",
    "Path to the document image orientation classification model directory.");
DEFINE_string(doc_unwarping_model_name, "UVDoc",
              "Name of the text image unwarping model.");
DEFINE_string(doc_unwarping_model_dir, "",
              "Path to the image unwarping model directory.");
DEFINE_string(text_detection_model_name, "PP-OCRv5_server_det",
              "Name of the text detection model.");
DEFINE_string(text_detection_model_dir, "",
              "Path to the text detection model directory.");
DEFINE_string(textline_orientation_model_name, "PP-LCNet_x1_0_textline_ori",
              "Name of the text line orientation classification model.");
DEFINE_string(
    textline_orientation_model_dir, "",
    "Path to the text line orientation classification model directory.");
DEFINE_string(textline_orientation_batch_size, "6",
              "Batch size for the text line orientation classification model.");
DEFINE_string(text_recognition_model_name, "PP-OCRv5_server_rec",
              "Name of the text recognition model.");
DEFINE_string(text_recognition_model_dir, "",
              "Path to the text recognition model directory.");
DEFINE_string(text_recognition_batch_size, "6",
              "Batch size for the text recognition model.");
DEFINE_string(use_doc_orientation_classify, "true",
              "Whether to use document image orientation classification.");
DEFINE_string(use_doc_unwarping, "true",
              "Whether to use text image unwarping.");
DEFINE_string(use_textline_orientation, "true",
              "Whether to use text line orientation classification.");
DEFINE_string(text_det_limit_side_len, "64",
              "This sets a limit on the side length of the input image for the "
              "text detection model.");
DEFINE_string(text_det_limit_type, "min",
              "This determines how the side length limit is applied to the "
              "input image before feeding it into the text deteciton model.");
DEFINE_string(text_det_thresh, "0.3",
              "Detection pixel threshold for the text detection model. Pixels "
              "with scores greater than this threshold in the output "
              "probability map are considered text pixels.");
DEFINE_string(
    text_det_box_thresh, "0.6",
    "Detection box threshold for the text detection model. A detection result "
    "is considered a text region if the average score of all pixels within the "
    "border of the result is greater than this threshold.");
DEFINE_string(
    text_det_unclip_ratio, "1.5",
    "Text detection expansion coefficient, which expands the text region using "
    "this method. The larger the value, the larger the expansion area.");
DEFINE_string(text_det_input_shape, "",
              "Input shape of the text detection model.eg C,H,W");
DEFINE_string(text_rec_score_thresh, "0",
              "Text recognition threshold. Text results with scores greater "
              "than this threshold are retained.");
DEFINE_string(text_rec_input_shape, "",
              "Input shape of the text recognition model.eg C,H,W");
DEFINE_string(lang, "", "Language in the input image for OCR processing.");
DEFINE_string(ocr_version, "", "PP-OCR version to use.");
#ifdef WITH_GPU
DEFINE_string(device, "gpu:0",
              "Device for inference. Supports specifying a specific card "
              "number: gpu:0.");
#else
DEFINE_string(device, "cpu",
              "Device for inference. Supports specifying a specific card "
              "number: gpu:0.");
#endif
DEFINE_string(vis_font_dir, "",
              "When enable USE_FREETYPE, required. Path to the visualization "
              "font, render the detected texts on images");
DEFINE_string(precision, "fp32",
              "Computational precision, such as fp32, fp16.");
DEFINE_string(enable_mkldnn, "true", "enable_mkldnn");
DEFINE_string(mkldnn_cache_capacity, "10", "MKL-DNN cache capacity.");
DEFINE_string(cpu_threads, "8",
              "Number of threads used for paddlepaddle inference on CPU.");
DEFINE_string(thread_num, "1",
              "Number of threads used for pipeline instance inference on CPU.");
DEFINE_string(paddlex_config, "",
              "Path to the PaddleX pipeline configuration file.");
