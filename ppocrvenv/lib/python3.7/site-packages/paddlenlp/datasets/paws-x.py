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

__all__ = ['PAWSX']


class PAWSX(DatasetBuilder):
    """
    PAWS-X: A Cross-lingual Adversarial Dataset for Paraphrase Identification
    More information please refer to `https://arxiv.org/abs/1908.11828`
    Here we only store simplified Chinese(zh) version.
    """
    URL = "https://bj.bcebos.com/paddlenlp/datasets/paws-x-zh.zip"
    MD5 = "f1c6f2ab8afb1f29fe04a0c929e3ab1c"
    META_INFO = collections.namedtuple('META_INFO', ('file', 'md5'))
    SPLITS = {
        'train': META_INFO(
            os.path.join('paws-x-zh', 'paws-x-zh', 'train.tsv'),
            '3422ba98e5151c91bbb0a785c4873a4c'),
        'dev': META_INFO(
            os.path.join('paws-x-zh', 'paws-x-zh', 'dev.tsv'),
            'dc163453e728cf118e17b4065d6602c8'),
        'test': META_INFO(
            os.path.join('paws-x-zh', 'paws-x-zh', 'test.tsv'),
            '5b7320760e70559591092cb01b6f5955'),
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
                    sentence1, sentence2, label = data
                    yield {
                        "sentence1": sentence1,
                        "sentence2": sentence2,
                        "label": label
                    }
                elif len(data) == 2:
                    sentence1, sentence2 = data
                    yield {
                        "sentence1": sentence1,
                        "sentence2": sentence2,
                        "label": ''
                    }
                else:
                    continue

    def get_labels(self):
        """
        Return labels of the PAWS-X object.
        """
        return ["0", "1"]
