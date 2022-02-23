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

__all__ = ['IWSLT15']


class IWSLT15(DatasetBuilder):
    '''
    Created by Stanford at 2015, the IWSLT 15 English-Vietnamese Sentence 
    pairs for translation., in Multi-Lingual language. Containing 133 in Text 
    file format.
    '''

    URL = "https://bj.bcebos.com/paddlenlp/datasets/iwslt15.en-vi.tar.gz"
    META_INFO = collections.namedtuple('META_INFO', ('src_file', 'tgt_file',
                                                     'src_md5', 'tgt_md5'))
    MD5 = 'aca22dc3f90962e42916dbb36d8f3e8e'
    SPLITS = {
        'train': META_INFO(
            os.path.join("iwslt15.en-vi", "train.en"),
            os.path.join("iwslt15.en-vi", "train.vi"),
            "5b6300f46160ab5a7a995546d2eeb9e6",
            "858e884484885af5775068140ae85dab"),
        'dev': META_INFO(
            os.path.join("iwslt15.en-vi", "tst2012.en"),
            os.path.join("iwslt15.en-vi", "tst2012.vi"),
            "c14a0955ed8b8d6929fdabf4606e3875",
            "dddf990faa149e980b11a36fca4a8898"),
        'test': META_INFO(
            os.path.join("iwslt15.en-vi", "tst2013.en"),
            os.path.join("iwslt15.en-vi", "tst2013.vi"),
            "c41c43cb6d3b122c093ee89608ba62bd",
            "a3185b00264620297901b647a4cacf38")
    }
    VOCAB_INFO = (os.path.join("iwslt15.en-vi", "vocab.en"), os.path.join(
        "iwslt15.en-vi", "vocab.vi"), "98b5011e1f579936277a273fd7f4e9b4",
                  "e8b05f8c26008a798073c619236712b4")
    UNK_TOKEN = '<unk>'
    BOS_TOKEN = '<s>'
    EOS_TOKEN = '</s>'

    def _get_data(self, mode, **kwargs):
        default_root = os.path.join(DATA_HOME, self.__class__.__name__)
        src_filename, tgt_filename, src_data_hash, tgt_data_hash = self.SPLITS[
            mode]
        src_fullname = os.path.join(default_root, src_filename)
        tgt_fullname = os.path.join(default_root, tgt_filename)

        src_vocab_filename, src_vocab_hash, tgt_vocab_filename, tgt_vocab_hash = self.VOCAB_INFO
        src_vocab_fullname = os.path.join(default_root, src_vocab_filename)
        tgt_vocab_fullname = os.path.join(default_root, tgt_vocab_filename)

        if (not os.path.exists(src_fullname) or
            (src_data_hash and not md5file(src_fullname) == src_data_hash)) or (
                not os.path.exists(tgt_fullname) or
                (tgt_data_hash and
                 not md5file(tgt_fullname) == tgt_data_hash)) or (
                     not os.path.exists(src_vocab_fullname) or
                     (src_vocab_hash and
                      not md5file(src_vocab_fullname) == src_vocab_hash)) or (
                          not os.path.exists(tgt_vocab_fullname) or
                          (tgt_vocab_hash and
                           not md5file(tgt_vocab_fullname) == tgt_vocab_hash)):
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
                    yield {"en": src_line, "vi": tgt_line}

    def get_vocab(self):
        en_vocab_fullname = os.path.join(DATA_HOME, self.__class__.__name__,
                                         self.VOCAB_INFO[0])
        vi_vocab_fullname = os.path.join(DATA_HOME, self.__class__.__name__,
                                         self.VOCAB_INFO[1])

        # Construct vocab_info to match the form of the input of `Vocab.load_vocabulary()` function
        vocab_info = {
            'en': {
                'filepath': en_vocab_fullname,
                'unk_token': self.UNK_TOKEN,
                'bos_token': self.BOS_TOKEN,
                'eos_token': self.EOS_TOKEN
            },
            'vi': {
                'filepath': vi_vocab_fullname,
                'unk_token': self.UNK_TOKEN,
                'bos_token': self.BOS_TOKEN,
                'eos_token': self.EOS_TOKEN
            }
        }
        return vocab_info
