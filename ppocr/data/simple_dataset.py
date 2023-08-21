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
import cv2
import math
import os
import json
import random
import traceback
from paddle.io import Dataset
from .imaug import transform, create_operators


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

        self.set_epoch_as_seed(self.seed, dataset_config)

        self.ops = create_operators(dataset_config['transforms'], global_config)
        self.ext_op_transform_idx = dataset_config.get("ext_op_transform_idx",
                                                       2)
        self.need_reset = True in [x < 1 for x in ratio_list]

    def set_epoch_as_seed(self, seed, dataset_config):
        if self.mode == 'train':
            try:
                border_map_id = [index
                    for index, dictionary in enumerate(dataset_config['transforms'])
                    if 'MakeBorderMap' in dictionary][0]
                shrink_map_id = [index
                    for index, dictionary in enumerate(dataset_config['transforms'])
                    if 'MakeShrinkMap' in dictionary][0]
                dataset_config['transforms'][border_map_id]['MakeBorderMap'][
                    'epoch'] = seed if seed is not None else 0
                dataset_config['transforms'][shrink_map_id]['MakeShrinkMap'][
                    'epoch'] = seed if seed is not None else 0
            except Exception as E:
                print(E)
                return

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


class MultiScaleDataSet(SimpleDataSet):
    def __init__(self, config, mode, logger, seed=None):
        super(MultiScaleDataSet, self).__init__(config, mode, logger, seed)
        self.ds_width = config[mode]['dataset'].get('ds_width', False)
        if self.ds_width:
            self.wh_aware()

    def wh_aware(self):
        data_line_new = []
        wh_ratio = []
        for lins in self.data_lines:
            data_line_new.append(lins)
            lins = lins.decode('utf-8')
            name, label, w, h = lins.strip("\n").split(self.delimiter)
            wh_ratio.append(float(w) / float(h))

        self.data_lines = data_line_new
        self.wh_ratio = np.array(wh_ratio)
        self.wh_ratio_sort = np.argsort(self.wh_ratio)
        self.data_idx_order_list = list(range(len(self.data_lines)))

    def resize_norm_img(self, data, imgW, imgH, padding=True):
        img = data['image']
        h = img.shape[0]
        w = img.shape[1]
        if not padding:
            resized_image = cv2.resize(
                img, (imgW, imgH), interpolation=cv2.INTER_LINEAR)
            resized_w = imgW
        else:
            ratio = w / float(h)
            if math.ceil(imgH * ratio) > imgW:
                resized_w = imgW
            else:
                resized_w = int(math.ceil(imgH * ratio))
            resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype('float32')

        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((3, imgH, imgW), dtype=np.float32)
        padding_im[:, :, :resized_w] = resized_image
        valid_ratio = min(1.0, float(resized_w / imgW))
        data['image'] = padding_im
        data['valid_ratio'] = valid_ratio
        return data

    def __getitem__(self, properties):
        # properites is a tuple, contains (width, height, index)
        img_height = properties[1]
        idx = properties[2]
        if self.ds_width and properties[3] is not None:
            wh_ratio = properties[3]
            img_width = img_height * (1 if int(round(wh_ratio)) == 0 else
                                      int(round(wh_ratio)))
            file_idx = self.wh_ratio_sort[idx]
        else:
            file_idx = self.data_idx_order_list[idx]
            img_width = properties[0]
            wh_ratio = None

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
            outs = transform(data, self.ops[:-1])
            if outs is not None:
                outs = self.resize_norm_img(outs, img_width, img_height)
                outs = transform(outs, self.ops[-1:])
        except:
            self.logger.error(
                "When parsing line {}, error happened with msg: {}".format(
                    data_line, traceback.format_exc()))
            outs = None
        if outs is None:
            # during evaluation, we should fix the idx to get same results for many times of evaluation.
            rnd_idx = (idx + 1) % self.__len__()
            return self.__getitem__([img_width, img_height, rnd_idx, wh_ratio])
        return outs
