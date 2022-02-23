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

__all__ = ['LCQMC']


class LCQMC(DatasetBuilder):
    """
    LCQMC:A Large-scale Chinese Question Matching Corpus
    More information please refer to `https://www.aclweb.org/anthology/C18-1166/`

    """

    URL = "https://bj.bcebos.com/paddlenlp/datasets/lcqmc.zip"
    MD5 = "7069fa0cffbd2110845869c61f83814a"
    META_INFO = collections.namedtuple('META_INFO', ('file', 'md5'))
    SPLITS = {
        'train': META_INFO(
            os.path.join('lcqmc', 'lcqmc', 'train.tsv'),
            '479d94fe575981f236319f2a5b8b3c03'),
        'dev': META_INFO(
            os.path.join('lcqmc', 'lcqmc', 'dev.tsv'),
            '089329fb44ef26155baef9c9c8c823ba'),
        'test': META_INFO(
            os.path.join('lcqmc', 'lcqmc', 'test.tsv'),
            'a4a483f2f871d57e0f3894fca0d0f8f0'),
    }

    def _get_data(self, mode, **kwargs):
        default_root = os.path.join(DATA_HOME, self.__class__.__name__)
        filename, data_hash = self.SPLITS[mode]
        fullname = os.path.join(default_root, filename)
        if not os.path.exists(fullname) or (data_hash and
                                            not md5file(fullname) == data_hash):
            get_path_from_url(self.URL, default_root, self.MD5)

        return fullname

    def _read(self, filename):
        """Reads data."""
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                data = line.strip().split("\t")
                if len(data) == 3:
                    query, title, label = data
                    yield {"query": query, "title": title, "label": label}
                elif len(data) == 2:
                    query, title = data
                    yield {"query": query, "title": title, "label": ''}
                else:
                    continue

    def get_labels(self):
        """
        Return labels of the LCQMC object.
        """
        return ["0", "1"]
