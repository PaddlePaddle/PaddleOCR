# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import argparse
import os
import cv2
import paddle
from paddle_msssim import ms_ssim, ssim
from GeoTr_PP import GeoTr
from utils import to_image, to_tensor, unwarp
import paddle.nn as nn


def run(args):
    image_path = args.image
    model_path = args.model
    # 如果使用预训练模型需要开启下行代码，因为只提供了权重
    state_dict=paddle.load(model_path)
    # 如果使用自己训练的模型需要开启下行代码
    # checkpoint = paddle.load(model_path)
    # state_dict = checkpoint["model"]
    model = GeoTr()
    model.set_state_dict(state_dict)
    model.eval()

    with paddle.no_grad():
        for i in range(1, 131):
            img_org = cv2.imread(image_path)
            y = to_tensor(img_org)
            img = cv2.resize(img_org, (288, 288))
            x = to_tensor(img)
            bm = model(x)
            # 如果使用预训练模型需要开启下行代码，如果使用自己训练的模型则关闭
            bm = bm / 288.
            bm = (bm - 0.5) * 2

            out = unwarp(y, bm)
            out = to_image(out.cpu())
            cv2.imwrite('output.png', out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="predict")
    parser.add_argument(
        "--image",
        "-i",
        nargs="?",
        type=str,
        default="",
        help="The path of image", )

    parser.add_argument(
        "--model",
        "-m",
        nargs="?",
        type=str,
        default="",
        help="The path of model", )

    args = parser.parse_args()

    print(args)

    run(args)
