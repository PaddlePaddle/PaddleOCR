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
from paddle.io import Dataset
from .imaug import transform, create_operators
import random


class PGDataSet(Dataset):
    def __init__(self, config, mode, logger, seed=None):
        super(PGDataSet, self).__init__()

        self.logger = logger
        self.seed = seed
        global_config = config['Global']
        dataset_config = config[mode]['dataset']
        loader_config = config[mode]['loader']

        label_file_list = dataset_config.pop('label_file_list')
        data_source_num = len(label_file_list)
        ratio_list = dataset_config.get("ratio_list", [1.0])
        if isinstance(ratio_list, (float, int)):
            ratio_list = [float(ratio_list)] * int(data_source_num)
        self.data_format = dataset_config.get('data_format', 'icdar')
        assert len(
            ratio_list
        ) == data_source_num, "The length of ratio_list should be the same as the file_list."
        self.do_shuffle = loader_config['shuffle']

        logger.info("Initialize indexs of datasets:%s" % label_file_list)
        self.data_lines = self.get_image_info_list(label_file_list, ratio_list,
                                                   self.data_format)
        self.data_idx_order_list = list(range(len(self.data_lines)))
        if mode.lower() == "train":
            self.shuffle_data_random()

        self.ops = create_operators(dataset_config['transforms'], global_config)

    def shuffle_data_random(self):
        if self.do_shuffle:
            random.seed(self.seed)
            random.shuffle(self.data_lines)
        return

    def extract_polys(self, poly_txt_path):
        """
        Read text_polys, txt_tags, txts from give txt file.
        """
        text_polys, txt_tags, txts = [], [], []
        with open(poly_txt_path) as f:
            for line in f.readlines():
                poly_str, txt = line.strip().split('\t')
                poly = map(float, poly_str.split(','))
                text_polys.append(
                    np.array(
                        list(poly), dtype=np.float32).reshape(-1, 2))
                txts.append(txt)
                if txt == '###':
                    txt_tags.append(True)
                else:
                    txt_tags.append(False)

        return np.array(list(map(np.array, text_polys))), \
               np.array(txt_tags, dtype=np.bool), txts

    def extract_info_textnet(self, im_fn, img_dir=''):
        """
        Extract information from line in textnet format.
        """
        info_list = im_fn.split('\t')
        img_path = ''
        for ext in [
                'jpg', 'bmp', 'png', 'jpeg', 'rgb', 'tif', 'tiff', 'gif', 'JPG'
        ]:
            if os.path.exists(os.path.join(img_dir, info_list[0] + ext)):
                img_path = os.path.join(img_dir, info_list[0] + ext)
                break

        if img_path == '':
            print('Image {0} NOT found in {1}, and it will be ignored.'.format(
                info_list[0], img_dir))

        nBox = (len(info_list) - 1) // 9
        wordBBs, txts, txt_tags = [], [], []
        for n in range(0, nBox):
            wordBB = list(map(float, info_list[n * 9 + 1:(n + 1) * 9]))
            txt = info_list[(n + 1) * 9]
            wordBBs.append([[wordBB[0], wordBB[1]], [wordBB[2], wordBB[3]],
                            [wordBB[4], wordBB[5]], [wordBB[6], wordBB[7]]])
            txts.append(txt)
            if txt == '###':
                txt_tags.append(True)
            else:
                txt_tags.append(False)
        return img_path, np.array(wordBBs, dtype=np.float32), txt_tags, txts

    def get_image_info_list(self, file_list, ratio_list, data_format='textnet'):
        if isinstance(file_list, str):
            file_list = [file_list]
        data_lines = []
        for idx, data_source in enumerate(file_list):
            image_files = []
            if data_format == 'icdar':
                image_files = [(data_source, x) for x in
                               os.listdir(os.path.join(data_source, 'rgb'))
                               if x.split('.')[-1] in [
                                   'jpg', 'bmp', 'png', 'jpeg', 'rgb', 'tif',
                                   'tiff', 'gif', 'JPG'
                               ]]
            elif data_format == 'textnet':
                with open(data_source) as f:
                    image_files = [(data_source, x.strip())
                                   for x in f.readlines()]
            else:
                print("Unrecognized data format...")
                exit(-1)
            random.seed(self.seed)
            image_files = random.sample(
                image_files, round(len(image_files) * ratio_list[idx]))
            data_lines.extend(image_files)
        return data_lines

    def __getitem__(self, idx):
        file_idx = self.data_idx_order_list[idx]
        data_path, data_line = self.data_lines[file_idx]
        try:
            if self.data_format == 'icdar':
                im_path = os.path.join(data_path, 'rgb', data_line)
                poly_path = os.path.join(data_path, 'poly',
                                         data_line.split('.')[0] + '.txt')
                text_polys, text_tags, text_strs = self.extract_polys(poly_path)
            else:
                image_dir = os.path.join(os.path.dirname(data_path), 'image')
                im_path, text_polys, text_tags, text_strs = self.extract_info_textnet(
                    data_line, image_dir)

            data = {
                'img_path': im_path,
                'polys': text_polys,
                'tags': text_tags,
                'strs': text_strs
            }
            with open(data['img_path'], 'rb') as f:
                img = f.read()
                data['image'] = img
            outs = transform(data, self.ops)

        except Exception as e:
            self.logger.error(
                "When parsing line {}, error happened with msg: {}".format(
                    self.data_idx_order_list[idx], e))
            outs = None
        if outs is None:
            return self.__getitem__(np.random.randint(self.__len__()))
        return outs

    def __len__(self):
        return len(self.data_idx_order_list)
