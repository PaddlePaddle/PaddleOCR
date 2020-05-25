# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import numpy as np

import paddle.fluid as fluid

__all__ = ['eval_rec_run', 'test_rec_benchmark']

import logging

FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)

from ppocr.utils.character import cal_predicts_accuracy
from ppocr.utils.character import convert_rec_label_to_lod
from ppocr.utils.character import convert_rec_attention_infer_res
from ppocr.utils.utility import create_module
import json
from copy import deepcopy
import cv2
from ppocr.data.reader_main import reader_main


def eval_rec_run(exe, config, eval_info_dict, mode):
    """
    Run evaluation program, return program outputs.
    """
    char_ops = config['Global']['char_ops']
    total_loss = 0
    total_sample_num = 0
    total_acc_num = 0
    total_batch_num = 0
    if mode == "test":
        is_remove_duplicate = False
    else:
        is_remove_duplicate = True

    for data in eval_info_dict['reader']():
        img_num = len(data)
        img_list = []
        label_list = []
        for ino in range(img_num):
            img_list.append(data[ino][0])
            label_list.append(data[ino][1])
        img_list = np.concatenate(img_list, axis=0)
        outs = exe.run(eval_info_dict['program'], \
                       feed={'image': img_list}, \
                       fetch_list=eval_info_dict['fetch_varname_list'], \
                       return_numpy=False)
        preds = np.array(outs[0])
        if preds.shape[1] != 1:
            preds, preds_lod = convert_rec_attention_infer_res(preds)
        else:
            preds_lod = outs[0].lod()[0]
        labels, labels_lod = convert_rec_label_to_lod(label_list)
        acc, acc_num, sample_num = cal_predicts_accuracy(
            char_ops, preds, preds_lod, labels, labels_lod, is_remove_duplicate)
        total_acc_num += acc_num
        total_sample_num += sample_num
        total_batch_num += 1
    avg_acc = total_acc_num * 1.0 / total_sample_num
    metrics = {'avg_acc': avg_acc, "total_acc_num": total_acc_num, \
               "total_sample_num": total_sample_num}
    return metrics


def test_rec_benchmark(exe, config, eval_info_dict):
    " 评估lmdb 数据"
    eval_data_list = ['IIIT5k_3000', 'SVT', 'IC03_860', 'IC03_867', \
                      'IC13_857', 'IC13_1015', 'IC15_1811', 'IC15_2077', 'SVTP', 'CUTE80']
    eval_data_dir = config['TestReader']['lmdb_sets_dir']
    total_evaluation_data_number = 0
    total_correct_number = 0
    eval_data_acc_info = {}
    for eval_data in eval_data_list:
        config['EvalReader']['lmdb_sets_dir'] = \
            eval_data_dir + "/" + eval_data
        eval_reader = reader_main(config=config, mode="eval")
        eval_info_dict['reader'] = eval_reader
        metrics = eval_rec_run(exe, config, eval_info_dict, "eval")
        total_evaluation_data_number += metrics['total_sample_num']
        total_correct_number += metrics['total_acc_num']
        eval_data_acc_info[eval_data] = metrics

    avg_acc = total_correct_number * 1.0 / total_evaluation_data_number
    logger.info('-' * 50)
    strs = ""
    for eval_data in eval_data_list:
        eval_acc = eval_data_acc_info[eval_data]['avg_acc']
        strs += "\n {}, accuracy:{:.6f}".format(eval_data, eval_acc)
    strs += "\n average, accuracy:{:.6f}".format(avg_acc)
    logger.info(strs)
    logger.info('-' * 50)
