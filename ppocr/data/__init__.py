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
from __future__ import unicode_literals

import os
import sys
import numpy as np
import paddle

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))

import copy
from paddle.io import DataLoader, DistributedBatchSampler, BatchSampler
import paddle.distributed as dist

from ppocr.data.imaug import transform, create_operators

__all__ = ['build_dataloader', 'transform', 'create_operators']


def build_dataset(config, global_config):
    from ppocr.data.dataset import SimpleDataSet, LMDBDateSet
    support_dict = ['SimpleDataSet', 'LMDBDateSet']

    module_name = config.pop('name')
    assert module_name in support_dict, Exception(
        'DataSet only support {}'.format(support_dict))

    dataset = eval(module_name)(config, global_config)
    return dataset


def build_dataloader(config, device, distributed=False, global_config=None):
    from ppocr.data.dataset import BatchBalancedDataLoader

    config = copy.deepcopy(config)
    dataset_config = config['dataset']

    _dataset_list = []
    file_list = dataset_config.pop('file_list')
    if len(file_list) == 1:
        ratio_list = [1.0]
    else:
        ratio_list = dataset_config.pop('ratio_list')
    for file in file_list:
        dataset_config['file_list'] = file
        _dataset = build_dataset(dataset_config, global_config)
        _dataset_list.append(_dataset)
    data_loader = BatchBalancedDataLoader(_dataset_list, ratio_list,
                                          distributed, device, config['loader'])
    return data_loader, _dataset.info_dict


def test_loader():
    import time
    from tools.program import load_config, ArgsParser

    FLAGS = ArgsParser().parse_args()
    config = load_config(FLAGS.config)

    place = paddle.CPUPlace()
    paddle.disable_static(place)
    import time

    data_loader, _ = build_dataloader(
        config['TRAIN'], place, global_config=config['Global'])
    start = time.time()
    print(len(data_loader))
    for epoch in range(1):
        print('epoch {} ****************'.format(epoch))
        for i, batch in enumerate(data_loader):
            if i > len(data_loader):
                break
            t = time.time() - start
            start = time.time()
            print('{}, batch : {} ,time {}'.format(i, len(batch[0]), t))

            continue
            import matplotlib.pyplot as plt

            from matplotlib import pyplot as plt
            import cv2
            fig = plt.figure()
            # # cv2.imwrite('img.jpg',batch[0].numpy()[0].transpose((1,2,0)))
            # # cv2.imwrite('bmap.jpg',batch[1].numpy()[0])
            # # cv2.imwrite('bmask.jpg',batch[2].numpy()[0])
            # # cv2.imwrite('smap.jpg',batch[3].numpy()[0])
            # # cv2.imwrite('smask.jpg',batch[4].numpy()[0])
            plt.title('img')
            plt.imshow(batch[0].numpy()[0].transpose((1, 2, 0)))
            # plt.figure()
            # plt.title('bmap')
            # plt.imshow(batch[1].numpy()[0],cmap='Greys')
            # plt.figure()
            # plt.title('bmask')
            # plt.imshow(batch[2].numpy()[0],cmap='Greys')
            # plt.figure()
            # plt.title('smap')
            # plt.imshow(batch[3].numpy()[0],cmap='Greys')
            # plt.figure()
            # plt.title('smask')
            # plt.imshow(batch[4].numpy()[0],cmap='Greys')
            # plt.show()
            # break


if __name__ == '__main__':
    test_loader()
