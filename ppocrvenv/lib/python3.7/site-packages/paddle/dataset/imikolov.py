# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
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
"""
imikolov's simple dataset.

This module will download dataset from
http://www.fit.vutbr.cz/~imikolov/rnnlm/ and parse training set and test set
into paddle reader creators.
"""

from __future__ import print_function

import paddle.dataset.common
import paddle.utils.deprecated as deprecated
import collections
import tarfile
import six

__all__ = []

#URL = 'http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz'
URL = 'https://dataset.bj.bcebos.com/imikolov%2Fsimple-examples.tgz'
MD5 = '30177ea32e27c525793142b6bf2c8e2d'


class DataType(object):
    NGRAM = 1
    SEQ = 2


def word_count(f, word_freq=None):
    if word_freq is None:
        word_freq = collections.defaultdict(int)

    for l in f:
        for w in l.strip().split():
            word_freq[w] += 1
        word_freq['<s>'] += 1
        word_freq['<e>'] += 1

    return word_freq


def build_dict(min_word_freq=50):
    """
    Build a word dictionary from the corpus,  Keys of the dictionary are words,
    and values are zero-based IDs of these words.
    """
    train_filename = './simple-examples/data/ptb.train.txt'
    test_filename = './simple-examples/data/ptb.valid.txt'
    with tarfile.open(
            paddle.dataset.common.download(paddle.dataset.imikolov.URL,
                                           'imikolov',
                                           paddle.dataset.imikolov.MD5)) as tf:
        trainf = tf.extractfile(train_filename)
        testf = tf.extractfile(test_filename)
        word_freq = word_count(testf, word_count(trainf))
        if '<unk>' in word_freq:
            # remove <unk> for now, since we will set it as last index
            del word_freq['<unk>']

        word_freq = [
            x for x in six.iteritems(word_freq) if x[1] > min_word_freq
        ]

        word_freq_sorted = sorted(word_freq, key=lambda x: (-x[1], x[0]))
        words, _ = list(zip(*word_freq_sorted))
        word_idx = dict(list(zip(words, six.moves.range(len(words)))))
        word_idx['<unk>'] = len(words)

    return word_idx


def reader_creator(filename, word_idx, n, data_type):
    def reader():
        with tarfile.open(
                paddle.dataset.common.download(
                    paddle.dataset.imikolov.URL, 'imikolov',
                    paddle.dataset.imikolov.MD5)) as tf:
            f = tf.extractfile(filename)

            UNK = word_idx['<unk>']
            for l in f:
                if DataType.NGRAM == data_type:
                    assert n > -1, 'Invalid gram length'
                    l = ['<s>'] + l.strip().split() + ['<e>']
                    if len(l) >= n:
                        l = [word_idx.get(w, UNK) for w in l]
                        for i in six.moves.range(n, len(l) + 1):
                            yield tuple(l[i - n:i])
                elif DataType.SEQ == data_type:
                    l = l.strip().split()
                    l = [word_idx.get(w, UNK) for w in l]
                    src_seq = [word_idx['<s>']] + l
                    trg_seq = l + [word_idx['<e>']]
                    if n > 0 and len(src_seq) > n: continue
                    yield src_seq, trg_seq
                else:
                    assert False, 'Unknow data type'

    return reader


@deprecated(
    since="2.0.0",
    update_to="paddle.text.datasets.Imikolov",
    level=1,
    reason="Please use new dataset API which supports paddle.io.DataLoader")
def train(word_idx, n, data_type=DataType.NGRAM):
    """
    imikolov training set creator.

    It returns a reader creator, each sample in the reader is a word ID
    tuple.

    :param word_idx: word dictionary
    :type word_idx: dict
    :param n: sliding window size if type is ngram, otherwise max length of sequence
    :type n: int
    :param data_type: data type (ngram or sequence)
    :type data_type: member variable of DataType (NGRAM or SEQ)
    :return: Training reader creator
    :rtype: callable
    """
    return reader_creator('./simple-examples/data/ptb.train.txt', word_idx, n,
                          data_type)


@deprecated(
    since="2.0.0",
    update_to="paddle.text.datasets.Imikolov",
    level=1,
    reason="Please use new dataset API which supports paddle.io.DataLoader")
def test(word_idx, n, data_type=DataType.NGRAM):
    """
    imikolov test set creator.

    It returns a reader creator, each sample in the reader is a word ID
    tuple.

    :param word_idx: word dictionary
    :type word_idx: dict
    :param n: sliding window size if type is ngram, otherwise max length of sequence
    :type n: int
    :param data_type: data type (ngram or sequence)
    :type data_type: member variable of DataType (NGRAM or SEQ)
    :return: Test reader creator
    :rtype: callable
    """
    return reader_creator('./simple-examples/data/ptb.valid.txt', word_idx, n,
                          data_type)


@deprecated(
    since="2.0.0",
    update_to="paddle.text.datasets.Imikolov",
    level=1,
    reason="Please use new dataset API which supports paddle.io.DataLoader")
def fetch():
    paddle.dataset.common.download(URL, "imikolov", MD5)
