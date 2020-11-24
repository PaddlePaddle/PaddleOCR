# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

from .imaug import transform, create_operators


class SimpleDataSet(Dataset):
    def __init__(self, config, mode, logger):
        super(SimpleDataSet, self).__init__()
        self.logger = logger

        global_config = config['Global']
        dataset_config = config[mode]['dataset']
        loader_config = config[mode]['loader']
        batch_size = loader_config['batch_size_per_card']

        self.delimiter = dataset_config.get('delimiter', '\t')
        label_file_list = dataset_config.pop('label_file_list')
        data_source_num = len(label_file_list)
        if data_source_num == 1:
            ratio_list = [1.0]
        else:
            ratio_list = dataset_config.pop('ratio_list')

        assert sum(ratio_list) == 1, "The sum of the ratio_list should be 1."
        assert len(
            ratio_list
        ) == data_source_num, "The length of ratio_list should be the same as the file_list."
        self.data_dir = dataset_config['data_dir']
        self.do_shuffle = loader_config['shuffle']

        logger.info("Initialize indexs of datasets:%s" % label_file_list)
        self.data_lines_list, data_num_list = self.get_image_info_list(
            label_file_list)
        self.data_idx_order_list = self.dataset_traversal(
            data_num_list, ratio_list, batch_size)
        self.shuffle_data_random()

        self.ops = create_operators(dataset_config['transforms'], global_config)

    def get_image_info_list(self, file_list):
        if isinstance(file_list, str):
            file_list = [file_list]
        data_lines_list = []
        data_num_list = []
        for file in file_list:
            with open(file, "rb") as f:
                lines = f.readlines()
                data_lines_list.append(lines)
                data_num_list.append(len(lines))
        return data_lines_list, data_num_list

    def dataset_traversal(self, data_num_list, ratio_list, batch_size):
        select_num_list = []
        dataset_num = len(data_num_list)
        for dno in range(dataset_num):
            select_num = round(batch_size * ratio_list[dno])
            select_num = max(select_num, 1)
            select_num_list.append(select_num)
        data_idx_order_list = []
        cur_index_sets = [0] * dataset_num
        while True:
            finish_read_num = 0
            for dataset_idx in range(dataset_num):
                cur_index = cur_index_sets[dataset_idx]
                if cur_index >= data_num_list[dataset_idx]:
                    finish_read_num += 1
                else:
                    select_num = select_num_list[dataset_idx]
                    for sno in range(select_num):
                        cur_index = cur_index_sets[dataset_idx]
                        if cur_index >= data_num_list[dataset_idx]:
                            break
                        data_idx_order_list.append((dataset_idx, cur_index))
                        cur_index_sets[dataset_idx] += 1
            if finish_read_num == dataset_num:
                break
        return data_idx_order_list

    def shuffle_data_random(self):
        if self.do_shuffle:
            for dno in range(len(self.data_lines_list)):
                random.shuffle(self.data_lines_list[dno])
        return

    def __getitem__(self, idx):
        dataset_idx, file_idx = self.data_idx_order_list[idx]
        data_line = self.data_lines_list[dataset_idx][file_idx]
        try:
            data_line = data_line.decode('utf-8')
            substr = data_line.strip("\n").split(self.delimiter)
            file_name = substr[0]
            label = substr[1]
            img_path = os.path.join(self.data_dir, file_name)
            data = {'img_path': img_path, 'label': label}
            with open(data['img_path'], 'rb') as f:
                img = f.read()
                data['image'] = img
            outs = transform(data, self.ops)
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
