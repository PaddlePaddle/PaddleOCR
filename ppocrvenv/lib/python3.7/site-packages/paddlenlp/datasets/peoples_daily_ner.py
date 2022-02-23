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

import collections
import os
import warnings

from paddle.io import Dataset
from paddle.dataset.common import md5file
from paddle.utils.download import get_path_from_url
from paddlenlp.utils.env import DATA_HOME
from . import DatasetBuilder

__all__ = ['PeoplesDailyNER']


class PeoplesDailyNER(DatasetBuilder):
    """
    Chinese Named Entity Recognition dataset published by People's Daily.
    The dataset is in the BIO scheme with tags: LOC, ORG and PER.
    """
    URL = "https://bj.bcebos.com/paddlenlp/datasets/peoples_daily_ner.tar.gz"
    MD5 = 'a44ff9c4b37b48add9ddc17994d5620c'
    META_INFO = collections.namedtuple('META_INFO', ('file', 'md5'))
    SPLITS = {
        'train': META_INFO(
            os.path.join('peoples_daily_ner', 'train.tsv'),
            '67d3c93a37daba60ef43c03271f119d7'),
        'dev': META_INFO(
            os.path.join('peoples_daily_ner', 'dev.tsv'),
            'ec772f3ba914bca5269f6e785bb3375d'),
        'test': META_INFO(
            os.path.join('peoples_daily_ner', 'test.tsv'),
            '2f27ae68b5f61d6553ffa28bb577c8a7')
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
        with open(filename, 'r', encoding='utf-8') as f:
            next(f)
            for line in f:
                line_stripped = line.strip().split('\t')
                if not line_stripped:
                    break
                if len(line_stripped) == 2:
                    tokens = line_stripped[0].split("\002")
                    tags = line_stripped[1].split("\002")
                else:
                    tokens = line_stripped.split("\002")
                    tags = []
                yield {"tokens": tokens, "labels": tags}

    def get_labels(self):

        return ["B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "O"]
