# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import fastdeploy as fd
import cv2
import os


def parse_arguments():
    import argparse
    import ast

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--det_model", required=True, help="Path of Detection model of PPOCR."
    )
    parser.add_argument(
        "--cls_model", required=True, help="Path of Classification model of PPOCR."
    )
    parser.add_argument(
        "--rec_model", required=True, help="Path of Recognization model of PPOCR."
    )
    parser.add_argument(
        "--rec_label_file", required=True, help="Path of Recognization model of PPOCR."
    )
    parser.add_argument(
        "--image", type=str, required=True, help="Path of test image file."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Type of inference device, support 'cpu' or 'gpu'.",
    )
    parser.add_argument(
        "--device_id",
        type=int,
        default=0,
        help="Define which GPU card used to run model.",
    )
    parser.add_argument(
        "--cls_bs",
        type=int,
        default=1,
        help="Classification model inference batch size.",
    )
    parser.add_argument(
        "--rec_bs", type=int, default=6, help="Recognition model inference batch size"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="default",
        help="Type of inference backend, support ort/trt/paddle/openvino, default 'openvino' for cpu, 'tensorrt' for gpu",
    )

    return parser.parse_args()


def build_option(args):
    det_option = fd.RuntimeOption()
    cls_option = fd.RuntimeOption()
    rec_option = fd.RuntimeOption()

    if args.device.lower() == "gpu":
        det_option.use_gpu(args.device_id)
        cls_option.use_gpu(args.device_id)
        rec_option.use_gpu(args.device_id)

    if args.backend.lower() == "trt":
        assert (
            args.device.lower() == "gpu"
        ), "TensorRT backend require inference on device GPU."
        det_option.use_trt_backend()
        cls_option.use_trt_backend()
        rec_option.use_trt_backend()

        # If use TRT backend, the dynamic shape will be set as follow.
        # We recommend that users set the length and height of the detection model to a multiple of 32.
        # We also recommend that users set the Trt input shape as follow.
        det_option.set_trt_input_shape(
            "x", [1, 3, 64, 64], [1, 3, 640, 640], [1, 3, 960, 960]
        )
        cls_option.set_trt_input_shape(
            "x", [1, 3, 48, 10], [args.cls_bs, 3, 48, 320], [args.cls_bs, 3, 48, 1024]
        )
        rec_option.set_trt_input_shape(
            "x", [1, 3, 48, 10], [args.rec_bs, 3, 48, 320], [args.rec_bs, 3, 48, 2304]
        )

        # Users could save TRT cache file to disk as follow.
        det_option.set_trt_cache_file(args.det_model + "/det_trt_cache.trt")
        cls_option.set_trt_cache_file(args.cls_model + "/cls_trt_cache.trt")
        rec_option.set_trt_cache_file(args.rec_model + "/rec_trt_cache.trt")

    elif args.backend.lower() == "pptrt":
        assert (
            args.device.lower() == "gpu"
        ), "Paddle-TensorRT backend require inference on device GPU."
        det_option.use_paddle_infer_backend()
        det_option.paddle_infer_option.collect_trt_shape = True
        det_option.paddle_infer_option.enable_trt = True

        cls_option.use_paddle_infer_backend()
        cls_option.paddle_infer_option.collect_trt_shape = True
        cls_option.paddle_infer_option.enable_trt = True

        rec_option.use_paddle_infer_backend()
        rec_option.paddle_infer_option.collect_trt_shape = True
        rec_option.paddle_infer_option.enable_trt = True

        # If use TRT backend, the dynamic shape will be set as follow.
        # We recommend that users set the length and height of the detection model to a multiple of 32.
        # We also recommend that users set the Trt input shape as follow.
        det_option.set_trt_input_shape(
            "x", [1, 3, 64, 64], [1, 3, 640, 640], [1, 3, 960, 960]
        )
        cls_option.set_trt_input_shape(
            "x", [1, 3, 48, 10], [args.cls_bs, 3, 48, 320], [args.cls_bs, 3, 48, 1024]
        )
        rec_option.set_trt_input_shape(
            "x", [1, 3, 48, 10], [args.rec_bs, 3, 48, 320], [args.rec_bs, 3, 48, 2304]
        )

        # Users could save TRT cache file to disk as follow.
        det_option.set_trt_cache_file(args.det_model)
        cls_option.set_trt_cache_file(args.cls_model)
        rec_option.set_trt_cache_file(args.rec_model)

    elif args.backend.lower() == "ort":
        det_option.use_ort_backend()
        cls_option.use_ort_backend()
        rec_option.use_ort_backend()

    elif args.backend.lower() == "paddle":
        det_option.use_paddle_infer_backend()
        cls_option.use_paddle_infer_backend()
        rec_option.use_paddle_infer_backend()

    elif args.backend.lower() == "openvino":
        assert (
            args.device.lower() == "cpu"
        ), "OpenVINO backend require inference on device CPU."
        det_option.use_openvino_backend()
        cls_option.use_openvino_backend()
        rec_option.use_openvino_backend()

    elif args.backend.lower() == "pplite":
        assert (
            args.device.lower() == "cpu"
        ), "Paddle Lite backend require inference on device CPU."
        det_option.use_lite_backend()
        cls_option.use_lite_backend()
        rec_option.use_lite_backend()

    return det_option, cls_option, rec_option


args = parse_arguments()

det_model_file = os.path.join(args.det_model, "inference.pdmodel")
det_params_file = os.path.join(args.det_model, "inference.pdiparams")

cls_model_file = os.path.join(args.cls_model, "inference.pdmodel")
cls_params_file = os.path.join(args.cls_model, "inference.pdiparams")

rec_model_file = os.path.join(args.rec_model, "inference.pdmodel")
rec_params_file = os.path.join(args.rec_model, "inference.pdiparams")
rec_label_file = args.rec_label_file

det_option, cls_option, rec_option = build_option(args)

det_model = fd.vision.ocr.DBDetector(
    det_model_file, det_params_file, runtime_option=det_option
)

cls_model = fd.vision.ocr.Classifier(
    cls_model_file, cls_params_file, runtime_option=cls_option
)

rec_model = fd.vision.ocr.Recognizer(
    rec_model_file, rec_params_file, rec_label_file, runtime_option=rec_option
)

# Parameters settings for pre and post processing of Det/Cls/Rec Models.
# All parameters are set to default values.
det_model.preprocessor.max_side_len = 960
det_model.postprocessor.det_db_thresh = 0.3
det_model.postprocessor.det_db_box_thresh = 0.6
det_model.postprocessor.det_db_unclip_ratio = 1.5
det_model.postprocessor.det_db_score_mode = "slow"
det_model.postprocessor.use_dilation = False
cls_model.postprocessor.cls_thresh = 0.9

# Create PP-OCRv3, if cls_model is not needed, just set cls_model=None .
ppocr_v3 = fd.vision.ocr.PPOCRv3(
    det_model=det_model, cls_model=cls_model, rec_model=rec_model
)

# Set inference batch size for cls model and rec model, the value could be -1 and 1 to positive infinity.
# When inference batch size is set to -1, it means that the inference batch size
# of the cls and rec models will be the same as the number of boxes detected by the det model.
ppocr_v3.cls_batch_size = args.cls_bs
ppocr_v3.rec_batch_size = args.rec_bs

# Read the input image
im = cv2.imread(args.image)

# Predict and return the results
result = ppocr_v3.predict(im)

print(result)

# Visuliaze the results.
vis_im = fd.vision.vis_ppocr(im, result)
cv2.imwrite("visualized_result.jpg", vis_im)
print("Visualized result save in ./visualized_result.jpg")
