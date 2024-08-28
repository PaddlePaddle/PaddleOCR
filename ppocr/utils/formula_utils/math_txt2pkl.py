# copyright (c) 2024 PaddlePaddle Authors. All Rights Reserve.
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

import pickle
from tqdm import tqdm
import os
import math
from paddle.utils import try_import
from collections import defaultdict
import glob
from os.path import join
import argparse


def txt2pickle(images, equations, save_dir):
    imagesize = try_import("imagesize")
    save_p = os.path.join(save_dir, "latexocr_{}.pkl".format(images.split("/")[-1]))
    min_dimensions = (32, 32)
    max_dimensions = (672, 192)
    max_length = 512
    data = defaultdict(lambda: [])
    if images is not None and equations is not None:
        images_list = [
            path.replace("\\", "/") for path in glob.glob(join(images, "*.png"))
        ]
        indices = [int(os.path.basename(img).split(".")[0]) for img in images_list]
        eqs = open(equations, "r").read().split("\n")
        for i, im in tqdm(enumerate(images_list), total=len(images_list)):
            width, height = imagesize.get(im)
            if (
                min_dimensions[0] <= width <= max_dimensions[0]
                and min_dimensions[1] <= height <= max_dimensions[1]
            ):
                divide_h = math.ceil(height / 16) * 16
                divide_w = math.ceil(width / 16) * 16
                im = os.path.basename(im)
                data[(divide_w, divide_h)].append((eqs[indices[i]], im))
        data = dict(data)
        with open(save_p, "wb") as file:
            pickle.dump(data, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image_dir",
        type=str,
        default=".",
        help="Input_label or input path to be converted",
    )
    parser.add_argument(
        "--mathtxt_path",
        type=str,
        default=".",
        help="Input_label or input path to be converted",
    )
    parser.add_argument(
        "--output_dir", type=str, default="out_label.txt", help="Output file name"
    )

    args = parser.parse_args()
    txt2pickle(args.image_dir, args.mathtxt_path, args.output_dir)
