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
        help="Type of inference device, support 'cpu', 'kunlunxin' or 'gpu'.",
    )
    parser.add_argument(
        "--device_id",
        type=int,
        default=0,
        help="Define which GPU card used to run model.",
    )
    return parser.parse_args()


def build_option(args):
    rec_option = fd.RuntimeOption()

    if args.device.lower() == "gpu":
        rec_option.use_gpu(args.device_id)

    return rec_option


args = parse_arguments()

rec_model_file = os.path.join(args.rec_model, "inference.pdmodel")
rec_params_file = os.path.join(args.rec_model, "inference.pdiparams")
rec_label_file = args.rec_label_file

# Set the runtime option
rec_option = build_option(args)

# Create the rec_model
rec_model = fd.vision.ocr.Recognizer(
    rec_model_file, rec_params_file, rec_label_file, runtime_option=rec_option
)

# Read the image
im = cv2.imread(args.image)

# Predict and return the result
result = rec_model.predict(im)

# User can infer a batch of images by following code.
# result = rec_model.batch_predict([im])

print(result)
