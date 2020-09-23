#copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import os
import sys
import math
import random
import functools
import numpy as np
import cv2
import string
from ppocr.utils.utility import initial_logger
logger = initial_logger()
from ppocr.utils.utility import create_module
from ppocr.utils.utility import get_image_file_list
import time


class TrainReader(object):
    def __init__(self, params):
        self.num_workers = params['num_workers']
        self.label_file_path = params['label_file_path']
        print(self.label_file_path)
        self.use_mul_data = False
        if isinstance(self.label_file_path, list):
            self.use_mul_data = True
            self.data_ratio_list = params['data_ratio_list']
        self.batch_size = params['train_batch_size_per_card']
        assert 'process_function' in params,\
            "absence process_function in Reader"
        self.process = create_module(params['process_function'])(params)

    def __call__(self, process_id):     
        def sample_iter_reader():
            with open(self.label_file_path, "rb") as fin:
                label_infor_list = fin.readlines()
            img_num = len(label_infor_list)
            img_id_list = list(range(img_num))
            random.shuffle(img_id_list)
            if sys.platform == "win32" and self.num_workers != 1:
                print("multiprocess is not fully compatible with Windows."
                      "num_workers will be 1.")
                self.num_workers = 1
            for img_id in range(process_id, img_num, self.num_workers):
                label_infor = label_infor_list[img_id_list[img_id]]
                outs = self.process(label_infor)
                if outs is None:
                    continue
                yield outs

        def sample_iter_reader_mul():
            batch_size = 1000
            data_source_list = self.label_file_path
            batch_size_list = list(map(int, [max(1.0, batch_size * x) for x in self.data_ratio_list]))
            print(self.data_ratio_list, batch_size_list)

            data_filename_list, data_size_list, fetch_record_list = [], [], []
            for data_source in data_source_list:
                image_files = open(data_source, "rb").readlines()
                random.shuffle(image_files)
                data_filename_list.append(image_files)
                data_size_list.append(len(image_files))
                fetch_record_list.append(0)

            image_batch = []
            # get a batch of img_fns and poly_fns
            for i in range(0, len(batch_size_list)):
                bs = batch_size_list[i]
                ds = data_size_list[i]
                image_names = data_filename_list[i]
                fetch_record = fetch_record_list[i]
                data_path = data_source_list[i]
                for j in range(fetch_record, fetch_record + bs):
                    index = j % ds
                    image_batch.append(image_names[index])

                if (fetch_record + bs) > ds:
                    fetch_record_list[i] = 0
                    random.shuffle(data_filename_list[i])
                else:
                    fetch_record_list[i] = fetch_record + bs

            if sys.platform == "win32":
                print("multiprocess is not fully compatible with Windows."
                      "num_workers will be 1.")
                self.num_workers = 1

            for label_infor in image_batch:
                outs = self.process(label_infor)
                if outs is None:
                    continue
                yield outs

        def batch_iter_reader():
            batch_outs = []
            if self.use_mul_data:
                print("Sample date from multiple datasets!")
                for outs in sample_iter_reader_mul():
                    batch_outs.append(outs)
                    if len(batch_outs) == self.batch_size:
                        yield batch_outs
                        batch_outs = []                
            else:
                for outs in sample_iter_reader():
                    batch_outs.append(outs)
                    if len(batch_outs) == self.batch_size:
                        yield batch_outs
                        batch_outs = []

        return batch_iter_reader


class EvalTestReader(object):
    def __init__(self, params):
        self.params = params
        assert 'process_function' in params,\
            "absence process_function in EvalTestReader"

    def __call__(self, mode):
        process_function = create_module(self.params['process_function'])(
            self.params)
        batch_size = self.params['test_batch_size_per_card']

        img_list = []
        if mode != "test":
            img_set_dir = self.params['img_set_dir']
            img_name_list_path = self.params['label_file_path']
            with open(img_name_list_path, "rb") as fin:
                lines = fin.readlines()
                for line in lines:
                    img_name = line.decode().strip("\n").split("\t")[0]
                    img_path = os.path.join(img_set_dir, img_name)
                    img_list.append(img_path)
        else:
            img_path = self.params['infer_img']
            img_list = get_image_file_list(img_path)

        def batch_iter_reader():
            batch_outs = []
            for img_path in img_list:
                img = cv2.imread(img_path)
                if img is None:
                    logger.info("{} does not exist!".format(img_path))
                    continue
                elif len(list(img.shape)) == 2 or img.shape[2] == 1:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                outs = process_function(img)
                outs.append(img_path)
                batch_outs.append(outs)
                if len(batch_outs) == batch_size:
                    yield batch_outs
                    batch_outs = []
            if len(batch_outs) != 0:
                yield batch_outs

        return batch_iter_reader
