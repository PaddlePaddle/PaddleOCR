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

__all__ = ['SIGHAN_CN']


class SIGHAN_CN(DatasetBuilder):
    URL = "https://bj.bcebos.com/paddlenlp/datasets/sighan-cn.zip"
    MD5 = "cd67b9b36a5908f848cbf04b5d83c005"
    META_INFO = collections.namedtuple('META_INFO', ('file', 'md5'))
    SPLITS = {
        'train': META_INFO(
            os.path.join('sighan-cn', 'train.txt'),
            '5eb7b7847722f3bf69bf978d1a5f99cc'),
        'dev': META_INFO(
            os.path.join('sighan-cn', 'dev.txt'),
            'bc34d119aeb7ca022aa66e2f448ded95'),
    }

    def _get_data(self, mode, **kwargs):
        """Downloads dataset."""
        default_root = os.path.join(DATA_HOME, self.__class__.__name__)
        filename, data_hash = self.SPLITS[mode]
        fullname = os.path.join(default_root, filename)
        if not os.path.exists(fullname) or (data_hash and
                                            not md5file(fullname) == data_hash):
            get_path_from_url(self.URL, default_root, self.MD5)

        return fullname

    def _read(self, filename, *args):
        """Reads data."""
        with open(filename, "r", encoding="utf8") as fr:
            for line in fr:
                source, target = line.strip('\n').split('\t')[0:2]
                yield {'source': source, 'target': target}
