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
import os, json, random, traceback
import numpy as np

from PIL import Image
from paddle.io import Dataset

from .imaug import transform, create_operators


class HMERDataSet(Dataset):
    def __init__(self, config, mode, logger, seed=None):
        super(HMERDataSet, self).__init__()

        self.logger = logger
        self.seed = seed
        self.mode = mode

        global_config = config['Global']
        dataset_config = config[mode]['dataset']
        self.data_dir = config[mode]['dataset']['data_dir']

        label_file_list = dataset_config['label_file_list']
        data_source_num = len(label_file_list)
        ratio_list = dataset_config.get("ratio_list", [1.0])

        self.data_lines, self.labels = self.get_image_info_list(label_file_list,
                                                                ratio_list)
        self.data_idx_order_list = list(range(len(self.data_lines)))
        if self.mode == "train" and self.do_shuffle:
            self.shuffle_data_random()

        if isinstance(ratio_list, (float, int)):
            ratio_list = [float(ratio_list)] * int(data_source_num)

        assert len(
            ratio_list
        ) == data_source_num, "The length of ratio_list should be the same as the file_list."

        self.ops = create_operators(dataset_config['transforms'], global_config)
        self.need_reset = True in [x < 1 for x in ratio_list]

    def get_image_info_list(self, file_list, ratio_list):
        if isinstance(file_list, str):
            file_list = [file_list]
        labels = {}
        for idx, file in enumerate(file_list):
            with open(file, "r") as f:
                lines = json.load(f)
                labels.update(lines)
        data_lines = [name for name in labels.keys()]
        return data_lines, labels

    def shuffle_data_random(self):
        random.seed(self.seed)
        random.shuffle(self.data_lines)
        return

    def __len__(self):
        return len(self.data_idx_order_list)

    def __getitem__(self, idx):
        file_idx = self.data_idx_order_list[idx]
        data_name = self.data_lines[file_idx]
        try:
            file_name = data_name + '.jpg'
            img_path = os.path.join(self.data_dir, file_name)
            if not os.path.exists(img_path):
                raise Exception("{} does not exist!".format(img_path))
            with open(img_path, 'rb') as f:
                img = f.read()

            label = self.labels.get(data_name).split()
            label = np.array([int(item) for item in label])

            data = {'image': img, 'label': label}
            outs = transform(data, self.ops)
        except:
            self.logger.error(
                "When parsing line {}, error happened with msg: {}".format(
                    file_name, traceback.format_exc()))
            outs = None
        if outs is None:
            # during evaluation, we should fix the idx to get same results for many times of evaluation.
            rnd_idx = np.random.randint(self.__len__())
            return self.__getitem__(rnd_idx)
        return outs
