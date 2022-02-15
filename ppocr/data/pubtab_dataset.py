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

from .imaug import transform, create_operators


class PubTabDataSet(Dataset):
    def __init__(self, config, mode, logger, seed=None):
        super(PubTabDataSet, self).__init__()
        self.logger = logger

        global_config = config['Global']
        dataset_config = config[mode]['dataset']
        loader_config = config[mode]['loader']

        label_file_path = dataset_config.pop('label_file_path')

        self.data_dir = dataset_config['data_dir']
        self.do_shuffle = loader_config['shuffle']
        self.do_hard_select = False
        if 'hard_select' in loader_config:
            self.do_hard_select = loader_config['hard_select']
            self.hard_prob = loader_config['hard_prob']
        if self.do_hard_select:
            self.img_select_prob = self.load_hard_select_prob()
        self.table_select_type = None
        if 'table_select_type' in loader_config:
            self.table_select_type = loader_config['table_select_type']
            self.table_select_prob = loader_config['table_select_prob']

        self.seed = seed
        logger.info("Initialize indexs of datasets:%s" % label_file_path)
        with open(label_file_path, "rb") as f:
            self.data_lines = f.readlines()
        self.data_idx_order_list = list(range(len(self.data_lines)))
        if mode.lower() == "train":
            self.shuffle_data_random()
        self.ops = create_operators(dataset_config['transforms'], global_config)

        ratio_list = dataset_config.get("ratio_list", [1.0])
        self.need_reset = True in [x < 1 for x in ratio_list]

    def shuffle_data_random(self):
        if self.do_shuffle:
            random.seed(self.seed)
            random.shuffle(self.data_lines)
        return

    def __getitem__(self, idx):
        try:
            data_line = self.data_lines[idx]
            data_line = data_line.decode('utf-8').strip("\n")
            info = json.loads(data_line)
            file_name = info['filename']
            select_flag = True
            if self.do_hard_select:
                prob = self.img_select_prob[file_name]
                if prob < random.uniform(0, 1):
                    select_flag = False

            if self.table_select_type:
                structure = info['html']['structure']['tokens'].copy()
                structure_str = ''.join(structure)
                table_type = "simple"
                if 'colspan' in structure_str or 'rowspan' in structure_str:
                    table_type = "complex"
                if table_type == "complex":
                    if self.table_select_prob < random.uniform(0, 1):
                        select_flag = False

            if select_flag:
                cells = info['html']['cells'].copy()
                structure = info['html']['structure'].copy()
                img_path = os.path.join(self.data_dir, file_name)
                data = {
                    'img_path': img_path,
                    'cells': cells,
                    'structure': structure
                }
                if not os.path.exists(img_path):
                    raise Exception("{} does not exist!".format(img_path))
                with open(data['img_path'], 'rb') as f:
                    img = f.read()
                    data['image'] = img
                outs = transform(data, self.ops)
            else:
                outs = None
        except Exception as e:
            self.logger.error(
                "When parsing line {}, error happened with msg: {}".format(
                    data_line, e))
            outs = None
        if outs is None:
            return self.__getitem__(np.random.randint(self.__len__()))
        return outs

    def __len__(self):
        return len(self.data_idx_order_list)
