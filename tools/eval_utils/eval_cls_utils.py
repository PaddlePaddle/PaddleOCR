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

import numpy as np

__all__ = ['eval_cls_run']

import logging

FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


def eval_cls_run(exe, eval_info_dict):
    """
    Run evaluation program, return program outputs.
    """
    total_sample_num = 0
    total_acc_num = 0
    total_batch_num = 0

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
        softmax_outs = np.array(outs[1])
        if len(softmax_outs.shape) != 1:
            softmax_outs = np.array(outs[0])
        acc, acc_num = cal_cls_acc(softmax_outs, label_list)
        total_acc_num += acc_num
        total_sample_num += len(label_list)
        # logger.info("eval batch id: {}, acc: {}".format(total_batch_num, acc))
        total_batch_num += 1
    avg_acc = total_acc_num * 1.0 / total_sample_num
    metrics = {'avg_acc': avg_acc, "total_acc_num": total_acc_num, \
               "total_sample_num": total_sample_num}
    return metrics


def cal_cls_acc(preds, labels):
    acc_num = 0
    for pred, label in zip(preds, labels):
        if pred == label:
            acc_num += 1
    return acc_num / len(preds), acc_num
