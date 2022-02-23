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

__all__ = ['ChnSentiCorp']


class ChnSentiCorp(DatasetBuilder):
    """
    ChnSentiCorp (by Tan Songbo at ICT of Chinese Academy of Sciences, and for
    opinion mining)

    """

    URL = "https://bj.bcebos.com/paddlenlp/datasets/ChnSentiCorp.zip"
    MD5 = "7ef61b08ad10fbddf2ba97613f071561"
    META_INFO = collections.namedtuple('META_INFO', ('file', 'md5'))
    SPLITS = {
        'train': META_INFO(
            os.path.join('ChnSentiCorp', 'ChnSentiCorp', 'train.tsv'),
            '689360c4a4a9ce8d8719ed500ae80907'),
        'dev': META_INFO(
            os.path.join('ChnSentiCorp', 'ChnSentiCorp', 'dev.tsv'),
            '20c77cc2371634731a367996b097ec0a'),
        'test': META_INFO(
            os.path.join('ChnSentiCorp', 'ChnSentiCorp', 'test.tsv'),
            '9b4dc7d1e4ada48c645b7e938592f49c'),
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

    def _read(self, filename, split):
        """Reads data."""
        with open(filename, 'r', encoding='utf-8') as f:
            head = None
            for line in f:
                data = line.strip().split("\t")
                if not head:
                    head = data
                else:
                    if split == 'train':
                        label, text = data
                        yield {"text": text, "label": label, "qid": ''}
                    elif split == 'dev':
                        qid, label, text = data
                        yield {"text": text, "label": label, "qid": qid}
                    elif split == 'test':
                        qid, text = data
                        yield {"text": text, "label": '', "qid": qid}

    def get_labels(self):
        """
        Return labels of the ChnSentiCorp object.
        """
        return ["0", "1"]
