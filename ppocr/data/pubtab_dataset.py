# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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
import numpy as np
import os
import random
from paddle.io import Dataset
import json
from copy import deepcopy

from .imaug import transform, create_operators


class PubTabDataSet(Dataset):
    def __init__(self, config, mode, logger, seed=None):
        super(PubTabDataSet, self).__init__()
        self.logger = logger

        global_config = config["Global"]
        dataset_config = config[mode]["dataset"]
        loader_config = config[mode]["loader"]

        label_file_list = dataset_config.pop("label_file_list")
        data_source_num = len(label_file_list)
        ratio_list = dataset_config.get("ratio_list", [1.0])
        if isinstance(ratio_list, (float, int)):
            ratio_list = [float(ratio_list)] * int(data_source_num)

        assert (
            len(ratio_list) == data_source_num
        ), "The length of ratio_list should be the same as the file_list."

        self.data_dir = dataset_config["data_dir"]
        self.do_shuffle = loader_config["shuffle"]

        self.seed = seed
        self.mode = mode.lower()
        logger.info("Initialize indexs of datasets:%s" % label_file_list)
        self.data_lines = self.get_image_info_list(label_file_list, ratio_list)
        # self.check(config['Global']['max_text_length'])

        if mode.lower() == "train" and self.do_shuffle:
            self.shuffle_data_random()
        self.ops = create_operators(dataset_config["transforms"], global_config)
        self.need_reset = True in [x < 1 for x in ratio_list]

    def get_image_info_list(self, file_list, ratio_list):
        if isinstance(file_list, str):
            file_list = [file_list]
        data_lines = []
        for idx, file in enumerate(file_list):
            with open(file, "rb") as f:
                lines = f.readlines()
                if self.mode == "train" or ratio_list[idx] < 1.0:
                    random.seed(self.seed)
                    lines = random.sample(lines, round(len(lines) * ratio_list[idx]))
                data_lines.extend(lines)
        return data_lines

    def check(self, max_text_length):
        data_lines = []
        for line in self.data_lines:
            data_line = line.decode("utf-8").strip("\n")
            info = json.loads(data_line)
            file_name = info["filename"]
            cells = info["html"]["cells"].copy()
            structure = info["html"]["structure"]["tokens"].copy()

            img_path = os.path.join(self.data_dir, file_name)
            if not os.path.exists(img_path):
                self.logger.warning("{} does not exist!".format(img_path))
                continue
            if len(structure) == 0 or len(structure) > max_text_length:
                continue
            # data = {'img_path': img_path, 'cells': cells, 'structure':structure,'file_name':file_name}
            data_lines.append(line)
        self.data_lines = data_lines

    def shuffle_data_random(self):
        if self.do_shuffle:
            random.seed(self.seed)
            random.shuffle(self.data_lines)
        return

    def __getitem__(self, idx):
        try:
            data_line = self.data_lines[idx]
            data_line = data_line.decode("utf-8").strip("\n")
            info = json.loads(data_line)
            file_name = info["filename"]
            cells = info["html"]["cells"].copy()
            structure = info["html"]["structure"]["tokens"].copy()

            img_path = os.path.join(self.data_dir, file_name)
            if not os.path.exists(img_path):
                raise Exception("{} does not exist!".format(img_path))
            data = {
                "img_path": img_path,
                "cells": cells,
                "structure": structure,
                "file_name": file_name,
            }

            with open(data["img_path"], "rb") as f:
                img = f.read()
                data["image"] = img
            outs = transform(data, self.ops)
        except:
            import traceback

            err = traceback.format_exc()
            self.logger.error(
                "When parsing line {}, error happened with msg: {}".format(
                    data_line, err
                )
            )
            outs = None
        if outs is None:
            rnd_idx = (
                np.random.randint(self.__len__())
                if self.mode == "train"
                else (idx + 1) % self.__len__()
            )
            return self.__getitem__(rnd_idx)
        return outs

    def __len__(self):
        return len(self.data_lines)
