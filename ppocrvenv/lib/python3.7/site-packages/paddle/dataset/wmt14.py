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
WMT14 dataset.
The original WMT14 dataset is too large and a small set of data for set is
provided. This module will download dataset from
http://paddlepaddle.bj.bcebos.com/demo/wmt_shrinked_data/wmt14.tgz and
parse training set and test set into paddle reader creators.

"""

from __future__ import print_function

import six
import tarfile
import gzip

import paddle.dataset.common
import paddle.compat as cpt
import paddle.utils.deprecated as deprecated

__all__ = []

URL_DEV_TEST = ('http://www-lium.univ-lemans.fr/~schwenk/'
                'cslm_joint_paper/data/dev+test.tgz')
MD5_DEV_TEST = '7d7897317ddd8ba0ae5c5fa7248d3ff5'
# this is a small set of data for test. The original data is too large and
# will be add later.
URL_TRAIN = ('http://paddlemodels.bj.bcebos.com/wmt/wmt14.tgz')
MD5_TRAIN = '0791583d57d5beb693b9414c5b36798c'
# BLEU of this trained model is 26.92
URL_MODEL = 'http://paddlemodels.bj.bcebos.com/wmt%2Fwmt14.tgz'
MD5_MODEL = '0cb4a5366189b6acba876491c8724fa3'

START = "<s>"
END = "<e>"
UNK = "<unk>"
UNK_IDX = 2


def __read_to_dict(tar_file, dict_size):
    def __to_dict(fd, size):
        out_dict = dict()
        for line_count, line in enumerate(fd):
            if line_count < size:
                out_dict[cpt.to_text(line.strip())] = line_count
            else:
                break
        return out_dict

    with tarfile.open(tar_file, mode='r') as f:
        names = [
            each_item.name for each_item in f
            if each_item.name.endswith("src.dict")
        ]
        assert len(names) == 1
        src_dict = __to_dict(f.extractfile(names[0]), dict_size)
        names = [
            each_item.name for each_item in f
            if each_item.name.endswith("trg.dict")
        ]
        assert len(names) == 1
        trg_dict = __to_dict(f.extractfile(names[0]), dict_size)
        return src_dict, trg_dict


def reader_creator(tar_file, file_name, dict_size):
    def reader():
        src_dict, trg_dict = __read_to_dict(tar_file, dict_size)
        with tarfile.open(tar_file, mode='r') as f:
            names = [
                each_item.name for each_item in f
                if each_item.name.endswith(file_name)
            ]
            for name in names:
                for line in f.extractfile(name):
                    line = cpt.to_text(line)
                    line_split = line.strip().split('\t')
                    if len(line_split) != 2:
                        continue
                    src_seq = line_split[0]  # one source sequence
                    src_words = src_seq.split()
                    src_ids = [
                        src_dict.get(w, UNK_IDX)
                        for w in [START] + src_words + [END]
                    ]

                    trg_seq = line_split[1]  # one target sequence
                    trg_words = trg_seq.split()
                    trg_ids = [trg_dict.get(w, UNK_IDX) for w in trg_words]

                    # remove sequence whose length > 80 in training mode
                    if len(src_ids) > 80 or len(trg_ids) > 80:
                        continue
                    trg_ids_next = trg_ids + [trg_dict[END]]
                    trg_ids = [trg_dict[START]] + trg_ids

                    yield src_ids, trg_ids, trg_ids_next

    return reader


@deprecated(
    since="2.0.0",
    update_to="paddle.text.datasets.WMT14",
    level=1,
    reason="Please use new dataset API which supports paddle.io.DataLoader")
def train(dict_size):
    """
    WMT14 training set creator.

    It returns a reader creator, each sample in the reader is source language
    word ID sequence, target language word ID sequence and next word ID
    sequence.

    :return: Training reader creator
    :rtype: callable
    """
    return reader_creator(
        paddle.dataset.common.download(URL_TRAIN, 'wmt14', MD5_TRAIN),
        'train/train', dict_size)


@deprecated(
    since="2.0.0",
    update_to="paddle.text.datasets.WMT14",
    level=1,
    reason="Please use new dataset API which supports paddle.io.DataLoader")
def test(dict_size):
    """
    WMT14 test set creator.

    It returns a reader creator, each sample in the reader is source language
    word ID sequence, target language word ID sequence and next word ID
    sequence.

    :return: Test reader creator
    :rtype: callable
    """
    return reader_creator(
        paddle.dataset.common.download(URL_TRAIN, 'wmt14', MD5_TRAIN),
        'test/test', dict_size)


@deprecated(
    since="2.0.0",
    update_to="paddle.text.datasets.WMT14",
    level=1,
    reason="Please use new dataset API which supports paddle.io.DataLoader")
def gen(dict_size):
    return reader_creator(
        paddle.dataset.common.download(URL_TRAIN, 'wmt14', MD5_TRAIN),
        'gen/gen', dict_size)


@deprecated(
    since="2.0.0",
    update_to="paddle.text.datasets.WMT14",
    level=1,
    reason="Please use new dataset API which supports paddle.io.DataLoader")
def get_dict(dict_size, reverse=True):
    # if reverse = False, return dict = {'a':'001', 'b':'002', ...}
    # else reverse = true, return dict = {'001':'a', '002':'b', ...}
    tar_file = paddle.dataset.common.download(URL_TRAIN, 'wmt14', MD5_TRAIN)
    src_dict, trg_dict = __read_to_dict(tar_file, dict_size)
    if reverse:
        src_dict = {v: k for k, v in six.iteritems(src_dict)}
        trg_dict = {v: k for k, v in six.iteritems(trg_dict)}
    return src_dict, trg_dict


@deprecated(
    since="2.0.0",
    update_to="paddle.text.datasets.WMT14",
    level=1,
    reason="Please use new dataset API which supports paddle.io.DataLoader")
def fetch():
    paddle.dataset.common.download(URL_TRAIN, 'wmt14', MD5_TRAIN)
    paddle.dataset.common.download(URL_MODEL, 'wmt14', MD5_MODEL)
