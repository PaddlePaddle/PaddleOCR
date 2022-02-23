# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import collections
import json
import os

from paddle.dataset.common import md5file
from paddle.utils.download import get_path_from_url
from paddlenlp.utils.env import DATA_HOME
from . import DatasetBuilder

__all__ = ['DuConv']


class DuConv(DatasetBuilder):
    """
    Duconv is an dialogue dataset based on knowledge map released by Baidu. 
    Duconv contains two test sets, test_1 and test_2. And the test_1 contains 
    the response of the conversation but test_2 not. More information please 
    refer to `https://arxiv.org/abs/1503.02364`.
    """
    URL = 'https://bj.bcebos.com/paddlenlp/datasets/DuConv.tar.gz'
    MD5 = 'ef496871787f66718e567d62bd8f3546'
    META_INFO = collections.namedtuple('META_INFO', ('file', 'md5'))
    SPLITS = {
        'train': META_INFO(
            os.path.join('DuConv', 'train.txt'),
            '26192809b8740f620b95c9e18c65edf4'),
        'dev': META_INFO(
            os.path.join('DuConv', 'dev.txt'),
            '2e5ee6396b0467309cad75d37d6460b1'),
        'test_1': META_INFO(
            os.path.join('DuConv', 'test_1.txt'),
            '8ec83a72318d004691962647905cc345'),
        'test_2': META_INFO(
            os.path.join('DuConv', 'test_2.txt'),
            'e8d5f04a5d0a03ab110b1605d0a632ad')
    }

    def _get_data(self, mode, **kwargs):
        default_root = os.path.join(DATA_HOME, self.__class__.__name__)
        filename, data_hash = self.SPLITS[mode]
        fullname = os.path.join(default_root, filename)
        if not os.path.exists(fullname) or (data_hash and
                                            not md5file(fullname) == data_hash):
            get_path_from_url(self.URL, default_root, self.MD5)
        return fullname

    def _read(self, filename, *args):
        with open(filename, 'r', encoding='utf-8') as fin:
            for line in fin:
                example = json.loads(line.strip())
                yield example
