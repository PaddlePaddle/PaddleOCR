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
import numpy as np
import cv2

import string
import lmdb

from ppocr.utils.utility import initial_logger
from ppocr.utils.utility import get_image_file_list
logger = initial_logger()

from .img_tools import process_image, get_img_data


class LMDBReader(object):
    def __init__(self, params):
        if params['mode'] != 'train':
            self.num_workers = 1
        else:
            self.num_workers = params['num_workers']
        self.lmdb_sets_dir = params['lmdb_sets_dir']
        self.char_ops = params['char_ops']
        self.image_shape = params['image_shape']
        self.loss_type = params['loss_type']
        self.max_text_length = params['max_text_length']
        self.mode = params['mode']
        if params['mode'] == 'train':
            self.batch_size = params['train_batch_size_per_card']
        else:
            self.batch_size = params['test_batch_size_per_card']
        self.infer_img = params['infer_img']

    def load_hierarchical_lmdb_dataset(self):
        lmdb_sets = {}
        dataset_idx = 0
        for dirpath, dirnames, filenames in os.walk(self.lmdb_sets_dir + '/'):
            if not dirnames:
                env = lmdb.open(
                    dirpath,
                    max_readers=32,
                    readonly=True,
                    lock=False,
                    readahead=False,
                    meminit=False)
                txn = env.begin(write=False)
                num_samples = int(txn.get('num-samples'.encode()))
                lmdb_sets[dataset_idx] = {"dirpath":dirpath, "env":env, \
                    "txn":txn, "num_samples":num_samples}
                dataset_idx += 1
        return lmdb_sets

    def print_lmdb_sets_info(self, lmdb_sets):
        lmdb_info_strs = []
        for dataset_idx in range(len(lmdb_sets)):
            tmp_str = " %s:%d," % (lmdb_sets[dataset_idx]['dirpath'],
                                   lmdb_sets[dataset_idx]['num_samples'])
            lmdb_info_strs.append(tmp_str)
        lmdb_info_strs = ''.join(lmdb_info_strs)
        logger.info("DataSummary:" + lmdb_info_strs)
        return

    def close_lmdb_dataset(self, lmdb_sets):
        for dataset_idx in lmdb_sets:
            lmdb_sets[dataset_idx]['env'].close()
        return

    def get_lmdb_sample_info(self, txn, index):
        label_key = 'label-%09d'.encode() % index
        label = txn.get(label_key)
        if label is None:
            return None
        label = label.decode('utf-8')
        img_key = 'image-%09d'.encode() % index
        imgbuf = txn.get(img_key)
        img = get_img_data(imgbuf)
        if img is None:
            return None
        return img, label

    def __call__(self, process_id):
        if self.mode != 'train':
            process_id = 0

        def sample_iter_reader():
            if self.infer_img is not None:
                image_file_list = get_image_file_list(self.infer_img)
                for single_img in image_file_list:
                    img = cv2.imread(single_img)
                    if img.shape[-1] == 1 or len(list(img.shape)) == 2:
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    norm_img = process_image(img, self.image_shape)
                    yield norm_img
            else:
                lmdb_sets = self.load_hierarchical_lmdb_dataset()
                if process_id == 0:
                    self.print_lmdb_sets_info(lmdb_sets)
                cur_index_sets = [1 + process_id] * len(lmdb_sets)
                while True:
                    finish_read_num = 0
                    for dataset_idx in range(len(lmdb_sets)):
                        cur_index = cur_index_sets[dataset_idx]
                        if cur_index > lmdb_sets[dataset_idx]['num_samples']:
                            finish_read_num += 1
                        else:
                            sample_info = self.get_lmdb_sample_info(
                                lmdb_sets[dataset_idx]['txn'], cur_index)
                            cur_index_sets[dataset_idx] += self.num_workers
                            if sample_info is None:
                                continue
                            img, label = sample_info
                            outs = process_image(img, self.image_shape, label,
                                                 self.char_ops, self.loss_type,
                                                 self.max_text_length)
                            if outs is None:
                                continue
                            yield outs

                    if finish_read_num == len(lmdb_sets):
                        break
                self.close_lmdb_dataset(lmdb_sets)

        def batch_iter_reader():
            batch_outs = []
            for outs in sample_iter_reader():
                batch_outs.append(outs)
                if len(batch_outs) == self.batch_size:
                    yield batch_outs
                    batch_outs = []
            if len(batch_outs) != 0:
                yield batch_outs

        if self.infer_img is None:
            return batch_iter_reader
        return sample_iter_reader


class SimpleReader(object):
    def __init__(self, params):
        if params['mode'] != 'train':
            self.num_workers = 1
        else:
            self.num_workers = params['num_workers']
        if params['mode'] != 'test':
            self.img_set_dir = params['img_set_dir']
            self.label_file_path = params['label_file_path']
        self.char_ops = params['char_ops']
        self.image_shape = params['image_shape']
        self.loss_type = params['loss_type']
        self.max_text_length = params['max_text_length']
        self.mode = params['mode']
        self.infer_img = params['infer_img']
        if params['mode'] == 'train':
            self.batch_size = params['train_batch_size_per_card']
        else:
            self.batch_size = params['test_batch_size_per_card']

    def __call__(self, process_id):
        if self.mode != 'train':
            process_id = 0

        def sample_iter_reader():
            if self.infer_img is not None:
                image_file_list = get_image_file_list(self.infer_img)
                for single_img in image_file_list:
                    img = cv2.imread(single_img)
                    if img.shape[-1] == 1 or len(list(img.shape)) == 2:
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    norm_img = process_image(img, self.image_shape)
                    yield norm_img
            else:
                with open(self.label_file_path, "rb") as fin:
                    label_infor_list = fin.readlines()
                img_num = len(label_infor_list)
                img_id_list = list(range(img_num))
                random.shuffle(img_id_list)
                if sys.platform == "win32":
                    print("multiprocess is not fully compatible with Windows."
                          "num_workers will be 1.")
                    self.num_workers = 1
                for img_id in range(process_id, img_num, self.num_workers):
                    label_infor = label_infor_list[img_id_list[img_id]]
                    substr = label_infor.decode('utf-8').strip("\n").split("\t")
                    img_path = self.img_set_dir + "/" + substr[0]
                    img = cv2.imread(img_path)
                    if img is None:
                        logger.info("{} does not exist!".format(img_path))
                        continue
                    if img.shape[-1] == 1 or len(list(img.shape)) == 2:
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

                    label = substr[1]
                    outs = process_image(img, self.image_shape, label,
                                         self.char_ops, self.loss_type,
                                         self.max_text_length)
                    if outs is None:
                        continue
                    yield outs

        def batch_iter_reader():
            batch_outs = []
            for outs in sample_iter_reader():
                batch_outs.append(outs)
                if len(batch_outs) == self.batch_size:
                    yield batch_outs
                    batch_outs = []
            if len(batch_outs) != 0:
                yield batch_outs

        if self.infer_img is None:
            return batch_iter_reader
        return sample_iter_reader
