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

import os
import collections

from paddle.io import Dataset

from paddle.utils.download import get_path_from_url
from paddle.dataset.common import md5file
from paddlenlp.utils.env import DATA_HOME
from . import DatasetBuilder

__all__ = ['YahooAnswer100K']


class YahooAnswer100K(DatasetBuilder):
    """
    The data is from https://arxiv.org/pdf/1702.08139.pdf, which samples 100k
    documents from original Yahoo Answer data, and vocabulary size is 200k.
    """
    URL = 'https://bj.bcebos.com/paddlenlp/datasets/yahoo-answer-100k.tar.gz'
    MD5 = "68b88fd3f2cc9918a78047d99bcc6532"
    META_INFO = collections.namedtuple('META_INFO', ('file', 'md5'))
    SPLITS = {
        'train': META_INFO(
            os.path.join('yahoo-answer-100k', 'yahoo.train.txt'),
            "3fb31bad56bae7c65fa084f702398c3b"),
        'valid': META_INFO(
            os.path.join('yahoo-answer-100k', 'yahoo.valid.txt'),
            "2680dd89b4fe882359846b5accfb7647"),
        'test': META_INFO(
            os.path.join('yahoo-answer-100k', 'yahoo.test.txt'),
            "3e6dcb643282e3543303980f1e21bb9d")
    }
    VOCAB_INFO = (os.path.join("yahoo-answer-100k", "vocab.txt"),
                  "2c17c7120e6240d34d19490404b5133d")
    UNK_TOKEN = '_UNK'

    def _get_data(self, mode, **kwargs):
        default_root = os.path.join(DATA_HOME, self.__class__.__name__)
        filename, data_hash = self.SPLITS[mode]
        fullname = os.path.join(default_root, filename)
        vocab_filename, vocab_hash = self.VOCAB_INFO
        vocab_fullname = os.path.join(default_root, vocab_filename)

        if (not os.path.exists(fullname)) or (
                data_hash and not md5file(fullname) == data_hash) or (
                    not os.path.exists(vocab_fullname) or
                    (vocab_hash and not md5file(vocab_fullname) == vocab_hash)):

            get_path_from_url(self.URL, default_root, self.MD5)

        return fullname

    def _read(self, filename, *args):
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                line_stripped = line.strip()
                yield {"sentence": line_stripped}

    def get_vocab(self):
        vocab_fullname = os.path.join(DATA_HOME, self.__class__.__name__,
                                      self.VOCAB_INFO[0])

        # Construct vocab_info to match the form of the input of `Vocab.load_vocabulary()` function
        vocab_info = {'filepath': vocab_fullname, 'unk_token': self.UNK_TOKEN}
        return vocab_info
