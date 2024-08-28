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

"""
This code is refer from:
https://github.com/lukas-blecher/LaTeX-OCR/blob/main/pix2tex/dataset/dataset.py
"""

import numpy as np
import cv2
import math
import os
import json
import pickle
import random
import traceback
import paddle
from paddle.io import Dataset
from .imaug.label_ops import LatexOCRLabelEncode
from .imaug import transform, create_operators


class LaTeXOCRDataSet(Dataset):
    def __init__(self, config, mode, logger, seed=None):
        super(LaTeXOCRDataSet, self).__init__()
        self.logger = logger
        self.mode = mode.lower()

        global_config = config["Global"]
        dataset_config = config[mode]["dataset"]
        loader_config = config[mode]["loader"]

        pkl_path = dataset_config.pop("data")
        self.data_dir = dataset_config["data_dir"]
        self.min_dimensions = dataset_config.pop("min_dimensions")
        self.max_dimensions = dataset_config.pop("max_dimensions")
        self.batchsize = dataset_config.pop("batch_size_per_pair")
        self.keep_smaller_batches = dataset_config.pop("keep_smaller_batches")
        self.max_seq_len = global_config.pop("max_seq_len")
        self.rec_char_dict_path = global_config.pop("rec_char_dict_path")
        self.tokenizer = LatexOCRLabelEncode(self.rec_char_dict_path)

        file = open(pkl_path, "rb")
        data = pickle.load(file)
        temp = {}
        for k in data:
            if (
                self.min_dimensions[0] <= k[0] <= self.max_dimensions[0]
                and self.min_dimensions[1] <= k[1] <= self.max_dimensions[1]
            ):
                temp[k] = data[k]
        self.data = temp
        self.do_shuffle = loader_config["shuffle"]
        self.seed = seed

        if self.mode == "train" and self.do_shuffle:
            random.seed(self.seed)
        self.pairs = []
        for k in self.data:
            info = np.array(self.data[k], dtype=object)
            p = (
                paddle.randperm(len(info))
                if self.mode == "train" and self.do_shuffle
                else paddle.arange(len(info))
            )
            for i in range(0, len(info), self.batchsize):
                batch = info[p[i : i + self.batchsize]]
                if len(batch.shape) == 1:
                    batch = batch[None, :]
                if len(batch) < self.batchsize and not self.keep_smaller_batches:
                    continue
                self.pairs.append(batch)
        if self.do_shuffle:
            self.pairs = np.random.permutation(np.array(self.pairs, dtype=object))
        else:
            self.pairs = np.array(self.pairs, dtype=object)

        self.size = len(self.pairs)
        self.set_epoch_as_seed(self.seed, dataset_config)

        self.ops = create_operators(dataset_config["transforms"], global_config)
        self.ext_op_transform_idx = dataset_config.get("ext_op_transform_idx", 2)
        self.need_reset = True

    def set_epoch_as_seed(self, seed, dataset_config):
        if self.mode == "train":
            try:
                border_map_id = [
                    index
                    for index, dictionary in enumerate(dataset_config["transforms"])
                    if "MakeBorderMap" in dictionary
                ][0]
                shrink_map_id = [
                    index
                    for index, dictionary in enumerate(dataset_config["transforms"])
                    if "MakeShrinkMap" in dictionary
                ][0]
                dataset_config["transforms"][border_map_id]["MakeBorderMap"][
                    "epoch"
                ] = (seed if seed is not None else 0)
                dataset_config["transforms"][shrink_map_id]["MakeShrinkMap"][
                    "epoch"
                ] = (seed if seed is not None else 0)
            except Exception as E:
                print(E)
                return

    def shuffle_data_random(self):
        random.seed(self.seed)
        random.shuffle(self.data_lines)
        return

    def __getitem__(self, idx):
        batch = self.pairs[idx]
        eqs, ims = batch.T
        try:
            max_width, max_height, max_length = 0, 0, 0

            images_transform = []

            for file_name in ims:
                img_path = os.path.join(self.data_dir, file_name)
                data = {
                    "img_path": img_path,
                }
                with open(data["img_path"], "rb") as f:
                    img = f.read()
                    data["image"] = img
                    item = transform(data, self.ops)
                    images_transform.append(np.array(item[0]))
            image_concat = np.concatenate(images_transform, axis=0)[:, np.newaxis, :, :]
            images_transform = image_concat.astype(np.float32)
            labels, attention_mask, max_length = self.tokenizer(list(eqs))
            if self.max_seq_len < max_length:
                rnd_idx = (
                    np.random.randint(self.__len__())
                    if self.mode == "train"
                    else (idx + 1) % self.__len__()
                )
                return self.__getitem__(rnd_idx)
            return (images_transform, labels, attention_mask)

        except:

            self.logger.error(
                "When parsing line {}, error happened with msg: {}".format(
                    data["img_path"], traceback.format_exc()
                )
            )
            outs = None

        if outs is None:
            # during evaluation, we should fix the idx to get same results for many times of evaluation.
            rnd_idx = (
                np.random.randint(self.__len__())
                if self.mode == "train"
                else (idx + 1) % self.__len__()
            )
            return self.__getitem__(rnd_idx)
        return outs

    def __len__(self):
        return self.size
