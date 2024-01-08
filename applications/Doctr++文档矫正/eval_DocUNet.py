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
    output_path = args.output
    gt_path = args.gt_path
    # 如果使用预训练模型需要开启下行代码，因为只提供了权重
    #state_dict=paddle.load(model_path)
    checkpoint = paddle.load(model_path)
    state_dict = checkpoint["model"]
    model = GeoTr()
    model.set_state_dict(state_dict)
    model.eval()

    avg_ms_ssim = 0
    avg_ssim = 0
    l1_loss_fn = nn.L1Loss()
    with paddle.no_grad():
        for i in range(1, 131):
            print("EVAL number {0}".format(i))
            if i % 2 == 0:
                id_ = 2
            else:
                id_ = 1
            image_path_i = os.path.join(
                image_path,
                str(int((i + 1) / 2)) + '_' + str(id_) + " copy.png")
            gt_path_i = os.path.join(gt_path, str(int((i + 1) / 2)) + ".png")
            output_path_i = os.path.join(
                output_path, str(int((i + 1) / 2)) + '_' + str(id_) + ".png")
            img_org = cv2.imread(image_path_i)
            gt = cv2.imread(gt_path_i)
            y = to_tensor(img_org)
            img = cv2.resize(img_org, (288, 288))
            gt = cv2.resize(gt, (img_org.shape[1], img_org.shape[0]))
            gt_ = to_tensor(gt)
            x = to_tensor(img)
            bm = model(x)
            # 如果使用预训练模型需要开启下行代码
            # bm = bm / 288.
            # bm = (bm - 0.5) * 2
            out = unwarp(y, bm)
            ssim_val = ssim(out, gt_, data_range=1.0)
            ms_ssim_val = ms_ssim(out, gt_, data_range=1.0)
            avg_ssim += ssim_val
            avg_ms_ssim += ms_ssim_val
            out = to_image(out.cpu())
            cv2.imwrite(output_path_i, out)
        print("avg_ssim:", avg_ssim / 130)
        print("avg_ms_ssim", avg_ms_ssim / 130)


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

    parser.add_argument(
        "--output",
        "-o",
        nargs="?",
        type=str,
        default="",
        help="The path of output", )

    parser.add_argument(
        "--gt_path",
        "-g",
        nargs="?",
        type=str,
        default="",
        help="The path of ground truth", )

    args = parser.parse_args()

    print(args)

    run(args)
