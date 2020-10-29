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

import os
import sys
import math
import random
import numpy as np
import cv2

from ppocr.utils.utility import initial_logger
from ppocr.utils.utility import get_image_file_list

logger = initial_logger()

from ppocr.data.rec.img_tools import resize_norm_img, warp
from ppocr.data.cls.randaugment import RandAugment


def random_crop(img):
    img_h, img_w = img.shape[:2]
    if img_w > img_h * 4:
        w = random.randint(img_h * 2, img_w)
        i = random.randint(0, img_w - w)

        img = img[:, i:i + w, :]
    return img


class SimpleReader(object):
    def __init__(self, params):
        if params['mode'] != 'train':
            self.num_workers = 1
        else:
            self.num_workers = params['num_workers']
        if params['mode'] != 'test':
            self.img_set_dir = params['img_set_dir']
            self.label_file_path = params['label_file_path']
        self.use_gpu = params['use_gpu']
        self.image_shape = params['image_shape']
        self.mode = params['mode']
        self.infer_img = params['infer_img']
        self.use_distort = params['mode'] == 'train' and params['distort']
        self.randaug = RandAugment()
        self.label_list = params['label_list']
        if "distort" in params:
            self.use_distort = params['distort'] and params['use_gpu']
            if not params['use_gpu']:
                logger.info(
                    "Distort operation can only support in GPU.Distort will be set to False."
                )
        if params['mode'] == 'train':
            self.batch_size = params['train_batch_size_per_card']
            self.drop_last = True
        else:
            self.batch_size = params['test_batch_size_per_card']
            self.drop_last = False
            self.use_distort = False

    def __call__(self, process_id):
        if self.mode != 'train':
            process_id = 0

        def get_device_num():
            if self.use_gpu:
                gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "1")
                gpu_num = len(gpus.split(','))
                return gpu_num
            else:
                cpu_num = os.environ.get("CPU_NUM", 1)
                return int(cpu_num)

        def sample_iter_reader():
            if self.mode != 'train' and self.infer_img is not None:
                image_file_list = get_image_file_list(self.infer_img)
                for single_img in image_file_list:
                    img = cv2.imread(single_img)
                    if img.shape[-1] == 1 or len(list(img.shape)) == 2:
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    norm_img = resize_norm_img(img, self.image_shape)

                    norm_img = norm_img[np.newaxis, :]
                    yield norm_img
            else:
                with open(self.label_file_path, "rb") as fin:
                    label_infor_list = fin.readlines()
                img_num = len(label_infor_list)
                img_id_list = list(range(img_num))
                random.shuffle(img_id_list)
                if sys.platform == "win32" and self.num_workers != 1:
                    print("multiprocess is not fully compatible with Windows."
                          "num_workers will be 1.")
                    self.num_workers = 1
                if self.batch_size * get_device_num(
                ) * self.num_workers > img_num:
                    raise Exception(
                        "The number of the whole data ({}) is smaller than the batch_size * devices_num * num_workers ({})".
                        format(img_num, self.batch_size * get_device_num() *
                               self.num_workers))
                for img_id in range(process_id, img_num, self.num_workers):
                    label_infor = label_infor_list[img_id_list[img_id]]
                    substr = label_infor.decode('utf-8').strip("\n").split("\t")
                    label = self.label_list.index(substr[1])

                    img_path = self.img_set_dir + "/" + substr[0]
                    img = cv2.imread(img_path)
                    if img is None:
                        logger.info("{} does not exist!".format(img_path))
                        continue
                    if img.shape[-1] == 1 or len(list(img.shape)) == 2:
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

                    if self.use_distort:
                        img = warp(img, 10)
                        img = self.randaug(img)
                    norm_img = resize_norm_img(img, self.image_shape)
                    norm_img = norm_img[np.newaxis, :]
                    yield (norm_img, label)

        def batch_iter_reader():
            batch_outs = []
            for outs in sample_iter_reader():
                batch_outs.append(outs)
                if len(batch_outs) == self.batch_size:
                    yield batch_outs
                    batch_outs = []
            if not self.drop_last:
                if len(batch_outs) != 0:
                    yield batch_outs

        if self.infer_img is None:
            return batch_iter_reader
        return sample_iter_reader
