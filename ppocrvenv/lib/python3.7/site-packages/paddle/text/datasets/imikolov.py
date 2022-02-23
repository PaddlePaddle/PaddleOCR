#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import six
import tarfile
import numpy as np
import collections

from paddle.io import Dataset
from paddle.dataset.common import _check_exists_and_download

__all__ = []

URL = 'https://dataset.bj.bcebos.com/imikolov%2Fsimple-examples.tgz'
MD5 = '30177ea32e27c525793142b6bf2c8e2d'


class Imikolov(Dataset):
    """
    Implementation of imikolov dataset.

    Args:
        data_file(str): path to data tar file, can be set None if
            :attr:`download` is True. Default None
        data_type(str): 'NGRAM' or 'SEQ'. Default 'NGRAM'.
        window_size(int): sliding window size for 'NGRAM' data. Default -1.
        mode(str): 'train' 'test' mode. Default 'train'.
        min_word_freq(int): minimal word frequence for building word dictionary. Default 50.
        download(bool): whether to download dataset automatically if
            :attr:`data_file` is not set. Default True

    Returns:
        Dataset: instance of imikolov dataset

    Examples:

        .. code-block:: python

            import paddle
            from paddle.text.datasets import Imikolov

            class SimpleNet(paddle.nn.Layer):
                def __init__(self):
                    super(SimpleNet, self).__init__()

                def forward(self, src, trg):
                    return paddle.sum(src), paddle.sum(trg)


            imikolov = Imikolov(mode='train', data_type='SEQ', window_size=2)

            for i in range(10):
                src, trg = imikolov[i]
                src = paddle.to_tensor(src)
                trg = paddle.to_tensor(trg)

                model = SimpleNet()
                src, trg = model(src, trg)
                print(src.numpy().shape, trg.numpy().shape)

    """

    def __init__(self,
                 data_file=None,
                 data_type='NGRAM',
                 window_size=-1,
                 mode='train',
                 min_word_freq=50,
                 download=True):
        assert data_type.upper() in ['NGRAM', 'SEQ'], \
            "data type should be 'NGRAM', 'SEQ', but got {}".format(data_type)
        self.data_type = data_type.upper()

        assert mode.lower() in ['train', 'test'], \
            "mode should be 'train', 'test', but got {}".format(mode)
        self.mode = mode.lower()

        self.window_size = window_size
        self.min_word_freq = min_word_freq

        self.data_file = data_file
        if self.data_file is None:
            assert download, "data_file is not set and downloading automatically disabled"
            self.data_file = _check_exists_and_download(data_file, URL, MD5,
                                                        'imikolov', download)

        # Build a word dictionary from the corpus
        self.word_idx = self._build_work_dict(min_word_freq)

        # read dataset into memory
        self._load_anno()

    def word_count(self, f, word_freq=None):
        if word_freq is None:
            word_freq = collections.defaultdict(int)

        for l in f:
            for w in l.strip().split():
                word_freq[w] += 1
            word_freq['<s>'] += 1
            word_freq['<e>'] += 1

        return word_freq

    def _build_work_dict(self, cutoff):
        train_filename = './simple-examples/data/ptb.train.txt'
        test_filename = './simple-examples/data/ptb.valid.txt'
        with tarfile.open(self.data_file) as tf:
            trainf = tf.extractfile(train_filename)
            testf = tf.extractfile(test_filename)
            word_freq = self.word_count(testf, self.word_count(trainf))
            if '<unk>' in word_freq:
                # remove <unk> for now, since we will set it as last index
                del word_freq['<unk>']

            word_freq = [
                x for x in six.iteritems(word_freq) if x[1] > self.min_word_freq
            ]

            word_freq_sorted = sorted(word_freq, key=lambda x: (-x[1], x[0]))
            words, _ = list(zip(*word_freq_sorted))
            word_idx = dict(list(zip(words, six.moves.range(len(words)))))
            word_idx['<unk>'] = len(words)

        return word_idx

    def _load_anno(self):
        self.data = []
        with tarfile.open(self.data_file) as tf:
            filename = './simple-examples/data/ptb.{}.txt'.format(self.mode)
            f = tf.extractfile(filename)

            UNK = self.word_idx['<unk>']
            for l in f:
                if self.data_type == 'NGRAM':
                    assert self.window_size > -1, 'Invalid gram length'
                    l = ['<s>'] + l.strip().split() + ['<e>']
                    if len(l) >= self.window_size:
                        l = [self.word_idx.get(w, UNK) for w in l]
                        for i in six.moves.range(self.window_size, len(l) + 1):
                            self.data.append(tuple(l[i - self.window_size:i]))
                elif self.data_type == 'SEQ':
                    l = l.strip().split()
                    l = [self.word_idx.get(w, UNK) for w in l]
                    src_seq = [self.word_idx['<s>']] + l
                    trg_seq = l + [self.word_idx['<e>']]
                    if self.window_size > 0 and len(src_seq) > self.window_size:
                        continue
                    self.data.append((src_seq, trg_seq))
                else:
                    assert False, 'Unknow data type'

    def __getitem__(self, idx):
        return tuple([np.array(d) for d in self.data[idx]])

    def __len__(self):
        return len(self.data)
