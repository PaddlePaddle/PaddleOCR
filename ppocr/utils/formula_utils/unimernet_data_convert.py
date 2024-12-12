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

import os
import cv2
import glob
import argparse
from os.path import join
from tqdm import tqdm


def latexocr2paddleocr_train(image_path, math_unimernet_file, math_hwe_file, save_path):
    convert_f = open(save_path, "w")
    sub_dir = "UniMER-1M/images"
    img_sub_dir = os.path.join(image_path, sub_dir)
    with open(math_unimernet_file, "r") as f:
        lines = f.readlines()
        formula_num = len(lines)
        for i, line in tqdm(enumerate(lines), total=formula_num):
            image_name = "{0:07d}.png".format(i)
            math_gt = line.strip()
            image_p = os.path.join(img_sub_dir, image_name)
            img_name_subdir = os.path.join(sub_dir, image_name)
            if os.path.exists(image_p):
                convert_f.writelines("{}\t{}\n".format(img_name_subdir, math_gt))

    sub_dir = "HME100K/train_images"
    img_sub_dir = os.path.join(image_path, sub_dir)
    with open(math_hwe_file, "r") as f:
        lines = f.readlines()
        formula_num = len(lines)
        for i, line in tqdm(enumerate(lines), total=formula_num):
            img_name, math_gt = line.strip().split("\t")
            image_path = os.path.join(img_sub_dir, img_name)
            img_name_subdir = os.path.join(sub_dir, img_name)
            convert_f.writelines("{}\t{}\n".format(img_name_subdir, math_gt))

    convert_f.close()


def unimernet2paddleocr_test(image_path, math_file, save_path):
    convert_f = open(save_path, "w")
    with open(math_file, "r") as f:
        # load maths which
        lines = f.readlines()
        formula_num = len(lines)
        for i, line in tqdm(enumerate(lines), total=formula_num):
            image_name = "{0:07d}.png".format(i)
            math_gt = line.strip()
            image_p = os.path.join(image_path, image_name)
            if os.path.exists(image_p):
                convert_f.writelines("{}\t{}\n".format(image_name, math_gt))
    convert_f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image_dir",
        type=str,
        default=".",
        help="Input_label or input path to be converted",
    )
    parser.add_argument(
        "--unimernet_txt_path",
        type=str,
        default="",
        help="Input_label or input path to be converted",
    )
    parser.add_argument(
        "--hme100k_txt_path",
        type=str,
        default="",
        help="Input_label or input path to be converted",
    )
    parser.add_argument(
        "--output_path", type=str, default="out_label.txt", help="Output file name"
    )
    parser.add_argument(
        "--datatype", type=str, default="out_label.txt", help="datatype"
    )
    args = parser.parse_args()
    if args.datatype == "unimernet_train":
        latexocr2paddleocr_train(
            args.image_dir,
            args.unimernet_txt_path,
            args.hme100k_txt_path,
            args.output_path,
        )
    elif args.datatype == "unimernet_test":
        unimernet2paddleocr_test(
            args.image_dir, args.unimernet_txt_path, args.output_path
        )
    else:
        raise NotImplementedError("the datatype is not supported")
