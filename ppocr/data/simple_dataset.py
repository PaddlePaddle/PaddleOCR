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
import json
import math
import os
import pickle
import random
import traceback
from multiprocessing import Pool, cpu_count

import numpy as np
from paddle.io import Dataset

from .imaug import create_operators, transform


class SimpleDataSet(Dataset):
    def __init__(self, config, mode, logger, seed=None):
        super(SimpleDataSet, self).__init__()
        self.logger = logger
        self.mode = mode.lower()

        global_config = config['Global']
        dataset_config = config[mode]['dataset']
        loader_config = config[mode]['loader']

        self.delimiter = dataset_config.get('delimiter', '\t')
        label_file_list = dataset_config.pop('label_file_list')
        data_source_num = len(label_file_list)
        ratio_list = dataset_config.get("ratio_list", 1.0)
        if isinstance(ratio_list, (float, int)):
            ratio_list = [float(ratio_list)] * int(data_source_num)

        assert len(
            ratio_list
        ) == data_source_num, "The length of ratio_list should be the same as the file_list."
        self.data_dir = dataset_config['data_dir']
        self.do_shuffle = loader_config['shuffle']
        self.seed = seed
        logger.info("Initialize indexs of datasets:%s" % label_file_list)
        self.data_lines = self.get_image_info_list(label_file_list, ratio_list)
        self.data_idx_order_list = list(range(len(self.data_lines)))
        if self.mode == "train" and self.do_shuffle:
            self.shuffle_data_random()
        self.ops = create_operators(dataset_config['transforms'], global_config)
        self.ext_op_transform_idx = dataset_config.get("ext_op_transform_idx",
                                                       2)
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
                    lines = random.sample(lines,
                                          round(len(lines) * ratio_list[idx]))
                data_lines.extend(lines)
        return data_lines

    def shuffle_data_random(self):
        random.seed(self.seed)
        random.shuffle(self.data_lines)
        return

    def _try_parse_filename_list(self, file_name):
        # multiple images -> one gt label
        if len(file_name) > 0 and file_name[0] == "[":
            try:
                info = json.loads(file_name)
                file_name = random.choice(info)
            except:
                pass
        return file_name

    def get_ext_data(self):
        ext_data_num = 0
        for op in self.ops:
            if hasattr(op, 'ext_data_num'):
                ext_data_num = getattr(op, 'ext_data_num')
                break
        load_data_ops = self.ops[:self.ext_op_transform_idx]
        ext_data = []

        while len(ext_data) < ext_data_num:
            file_idx = self.data_idx_order_list[np.random.randint(self.__len__(
            ))]
            data_line = self.data_lines[file_idx]
            data_line = data_line.decode('utf-8')
            substr = data_line.strip("\n").split(self.delimiter)
            file_name = substr[0]
            file_name = self._try_parse_filename_list(file_name)
            label = substr[1]
            img_path = os.path.join(self.data_dir, file_name)
            data = {'img_path': img_path, 'label': label}
            if not os.path.exists(img_path):
                continue
            with open(data['img_path'], 'rb') as f:
                img = f.read()
                data['image'] = img
            data = transform(data, load_data_ops)

            if data is None:
                continue
            if 'polys' in data.keys():
                if data['polys'].shape[1] != 4:
                    continue
            ext_data.append(data)
        return ext_data

    def __getitem__(self, idx):
        file_idx = self.data_idx_order_list[idx]
        data_line = self.data_lines[file_idx]
        try:
            data_line = data_line.decode('utf-8')
            substr = data_line.strip("\n").split(self.delimiter)
            file_name = substr[0]
            file_name = self._try_parse_filename_list(file_name)
            label = substr[1]
            img_path = os.path.join(self.data_dir, file_name)
            data = {'img_path': img_path, 'label': label}
            if not os.path.exists(img_path):
                raise Exception("{} does not exist!".format(img_path))
            with open(data['img_path'], 'rb') as f:
                img = f.read()
                data['image'] = img
            data['ext_data'] = self.get_ext_data()
            outs = transform(data, self.ops)
        except:
            self.logger.error(
                "When parsing line {}, error happened with msg: {}".format(
                    data_line, traceback.format_exc()))
            outs = None
        if outs is None:
            # during evaluation, we should fix the idx to get same results for many times of evaluation.
            rnd_idx = np.random.randint(self.__len__(
            )) if self.mode == "train" else (idx + 1) % self.__len__()
            return self.__getitem__(rnd_idx)
        return outs

    def __len__(self):
        return len(self.data_idx_order_list)


class SpeedupDataset(SimpleDataSet):
    def __init__(self, config, mode, logger, seed=None):
        super().__init__(config, mode, logger, seed)
        dataset_config = config[mode]['dataset']
        self.dataset_mode = mode
        self.save_parts_dir = dataset_config.get('save_parts_dir')
        self.save_gear = dataset_config.get('save_gear')
        self.save_nums = self.save_gear*config[mode]['loader']['batch_size_per_card']
        self.save_reset = dataset_config.get('save_reset', False)
        self.save_files = [f"{self.save_parts_dir}/{i}.pkl" for i in range(math.ceil(self.__len__()/self.save_nums))]

        # save_data
        self.prepare_data()

        # initialize_cursor
        self.cursor_file = None
        self.cursor = []

    def hash_idx_2_pkl(self, idx):
        ifi = int(idx/self.save_nums)
        idd = idx % self.save_nums
        return self.save_files[ifi], idd

    def hash_filei_2_idxs(self, i): return list(range(i*self.save_nums, min((i+1)*self.save_nums, self.__len__())))
    def shuffle_data_random(self): ...

    def prepare_data(self):
        print(f"Start prepare dataset `{self.dataset_mode}`")
        os.makedirs(self.save_parts_dir, exist_ok=True)
        eis = []
        for ei in enumerate(self.save_files):
            _, save_file = ei
            if self.save_reset or (not os.path.exists(save_file)):
                eis.append(ei)
        # for ei in enumerate(self.save_files):
        #     self.save_process(ei)
        with Pool(cpu_count()//4) as p:
            p.map(self.save_process, enumerate(self.save_files))
        print(f"Prepare dataset `{self.dataset_mode}` finish")

    def save_process(self, args):
        ifi, save_file = args
        idxs = self.hash_filei_2_idxs(ifi)
        datas = self.load_idxs(idxs)
        with open(save_file, "wb") as f:
            pickle.dump(datas, f)

    def load_idxs(self, idxs):
        datas = []
        for idx in idxs:
            datas.append(self.get_old(idx))
        return datas

    def load_cursor(self, save_file):
        with open(save_file, "rb") as f:
            self.cursor = pickle.load(f)
            self.cursor_file = save_file

    def __getitem__(self, idx):
        cursor_file, idd = self.hash_idx_2_pkl(idx)
        if cursor_file != self.cursor_file:
            self.load_cursor(cursor_file)
            return self.__getitem__(idx)
        return self.cursor[idd]

    def get_old(self, idx):
        file_idx = self.data_idx_order_list[idx]
        data_line = self.data_lines[file_idx]
        try:
            data_line = data_line.decode('utf-8')
            substr = data_line.strip("\n").split(self.delimiter)
            file_name = substr[0]
            file_name = self._try_parse_filename_list(file_name)
            label = substr[1]
            img_path = os.path.join(self.data_dir, file_name)
            data = {'img_path': img_path, 'label': label}
            if not os.path.exists(img_path):
                raise Exception("{} does not exist!".format(img_path))
            with open(data['img_path'], 'rb') as f:
                img = f.read()
                data['image'] = img
            data['ext_data'] = self.get_ext_data()
            outs = transform(data, self.ops)
        except:
            self.logger.info("When parsing line {}, error happened with msg: {}".format(data_line, traceback.format_exc()))
            outs = None
        if outs is None:
            # during evaluation, we should fix the idx to get same results for many times of evaluation.
            rnd_idx = (idx + 1) % self.__len__()
            return self.get_old(rnd_idx)
        return outs
