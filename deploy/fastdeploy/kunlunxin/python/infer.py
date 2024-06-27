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
        "--cls_bs",
        type=int,
        default=1,
        help="Classification model inference batch size.",
    )
    parser.add_argument(
        "--rec_bs", type=int, default=6, help="Recognition model inference batch size"
    )
    return parser.parse_args()


def build_option(args):
    det_option = fd.RuntimeOption()
    cls_option = fd.RuntimeOption()
    rec_option = fd.RuntimeOption()

    det_option.use_kunlunxin()
    cls_option.use_kunlunxin()
    rec_option.use_kunlunxin()

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

# Create PP-OCRv3, if cls_model is not needed,
# just set cls_model=None .
ppocr_v3 = fd.vision.ocr.PPOCRv3(
    det_model=det_model, cls_model=cls_model, rec_model=rec_model
)

# Set inference batch size for cls model and rec model, the value could be -1 and 1 to positive infinity.
# When inference batch size is set to -1, it means that the inference batch size
# of the cls and rec models will be the same as the number of boxes detected by the det model.
ppocr_v3.cls_batch_size = args.cls_bs
ppocr_v3.rec_batch_size = args.rec_bs

# Prepare image.
im = cv2.imread(args.image)

# Print the results.
result = ppocr_v3.predict(im)

print(result)

# Visuliaze the output.
vis_im = fd.vision.vis_ppocr(im, result)
cv2.imwrite("visualized_result.jpg", vis_im)
print("Visualized result save in ./visualized_result.jpg")
