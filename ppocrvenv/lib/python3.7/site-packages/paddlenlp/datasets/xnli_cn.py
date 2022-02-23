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
import os

from paddle.dataset.common import md5file
from paddle.utils.download import get_path_from_url
from paddlenlp.utils.env import DATA_HOME
from . import DatasetBuilder

__all__ = ['XNLI_CN']


class XNLI_CN(DatasetBuilder):
    """
    XNLI dataset for chinese.

    XNLI is an evaluation corpus for language transfer and cross-lingual
    sentence classification in 15 languages. Here, XNLI only contrains
    chinese corpus.

    For more information, please visit https://github.com/facebookresearch/XNLI
    """

    URL = "https://bj.bcebos.com/paddlenlp/datasets/xnli_cn.tar.gz"
    MD5 = "aaf6de381a2553d61d8e6fad4ba96499"
    META_INFO = collections.namedtuple('META_INFO', ('file', 'md5'))
    SPLITS = {
        'train': META_INFO(
            os.path.join('xnli_cn.tar', 'xnli_cn', 'train', 'part-0'),
            'b0e4df29af8413eb935a2204de8958b7'),
        'dev': META_INFO(
            os.path.join('xnli_cn.tar', 'xnli_cn', 'dev', 'part-0'),
            '401a2178e15f4b0c35812ab4a322bd94'),
        'test': META_INFO(
            os.path.join('xnli_cn.tar', 'xnli_cn', 'test', 'part-0'),
            '71b043be8207e54185e761fca00ba3d7'),
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
                        text_a, text_b, label = data
                        yield {
                            "text_a": text_a,
                            "text_b": text_b,
                            "label": label
                        }
                    elif split == 'dev':
                        text_a, text_b, label = data
                        yield {
                            "text_a": text_a,
                            "text_b": text_b,
                            "label": label
                        }
                    elif split == 'test':
                        text_a, text_b, label = data
                        yield {
                            "text_a": text_a,
                            "text_b": text_b,
                            "label": label
                        }

    def get_labels(self):
        """
        Return labels of XNLI dataset.

        Note:
            Contradictory and contradiction are the same label
        """
        return ["contradictory", "entailment", "neutral"]
