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

__all__ = ['Couplet']


class Couplet(DatasetBuilder):
    """
    Couplet dataset. The couplet data is from this github repository:
    https://github.com/v-zich/couplet-clean-dataset, which filters dirty data
    from the original repository https://github.com/wb14123/couplet-dataset.
    """
    URL = "https://bj.bcebos.com/paddlenlp/datasets/couplet.tar.gz"
    META_INFO = collections.namedtuple('META_INFO', ('src_file', 'tgt_file',
                                                     'src_md5', 'tgt_md5'))
    MD5 = '5c0dcde8eec6a517492227041c2e2d54'
    SPLITS = {
        'train': META_INFO(
            os.path.join("couplet", "train_src.tsv"),
            os.path.join("couplet", "train_tgt.tsv"),
            "ad137385ad5e264ac4a54fe8c95d1583",
            "daf4dd79dbf26040696eee0d645ef5ad"),
        'dev': META_INFO(
            os.path.join("couplet", "dev_src.tsv"),
            os.path.join("couplet", "dev_tgt.tsv"),
            "65bf9e72fa8fdf0482751c1fd6b6833c",
            "3bc3b300b19d170923edfa8491352951"),
        'test': META_INFO(
            os.path.join("couplet", "test_src.tsv"),
            os.path.join("couplet", "test_tgt.tsv"),
            "f0a7366dfa0acac884b9f4901aac2cc1",
            "56664bff3f2edfd7a751a55a689f90c2")
    }
    VOCAB_INFO = (os.path.join("couplet", "vocab.txt"),
                  "0bea1445c7c7fb659b856bb07e54a604")
    UNK_TOKEN = '<unk>'
    BOS_TOKEN = '<s>'
    EOS_TOKEN = '</s>'

    def _get_data(self, mode, **kwargs):
        default_root = os.path.join(DATA_HOME, self.__class__.__name__)
        src_filename, tgt_filename, src_data_hash, tgt_data_hash = self.SPLITS[
            mode]
        src_fullname = os.path.join(default_root, src_filename)
        tgt_fullname = os.path.join(default_root, tgt_filename)

        vocab_filename, vocab_hash = self.VOCAB_INFO
        vocab_fullname = os.path.join(default_root, vocab_filename)

        if (not os.path.exists(src_fullname) or
            (src_data_hash and not md5file(src_fullname) == src_data_hash)) or (
                not os.path.exists(tgt_fullname) or
                (tgt_data_hash and
                 not md5file(tgt_fullname) == tgt_data_hash)) or (
                     not os.path.exists(vocab_fullname) or
                     (vocab_hash and
                      not md5file(vocab_fullname) == vocab_hash)):
            get_path_from_url(self.URL, default_root, self.MD5)

        return src_fullname, tgt_fullname

    def _read(self, filename, *args):
        src_filename, tgt_filename = filename
        with open(src_filename, 'r', encoding='utf-8') as src_f:
            with open(tgt_filename, 'r', encoding='utf-8') as tgt_f:
                for src_line, tgt_line in zip(src_f, tgt_f):
                    src_line = src_line.strip()
                    tgt_line = tgt_line.strip()
                    if not src_line and not tgt_line:
                        continue
                    yield {"first": src_line, "second": tgt_line}

    def get_vocab(self):
        vocab_fullname = os.path.join(DATA_HOME, self.__class__.__name__,
                                      self.VOCAB_INFO[0])

        # Construct vocab_info to match the form of the input of `Vocab.load_vocabulary()` function
        vocab_info = {
            'filepath': vocab_fullname,
            'unk_token': self.UNK_TOKEN,
            'bos_token': self.BOS_TOKEN,
            'eos_token': self.EOS_TOKEN
        }
        return vocab_info
