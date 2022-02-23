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
UCI Housing dataset.

This module will download dataset from
https://archive.ics.uci.edu/ml/machine-learning-databases/housing/ and
parse training set and test set into paddle reader creators.
"""

from __future__ import print_function

import numpy as np
import six
import tempfile
import tarfile
import os
import paddle.dataset.common
import paddle.utils.deprecated as deprecated

__all__ = []

URL = 'http://paddlemodels.bj.bcebos.com/uci_housing/housing.data'
MD5 = 'd4accdce7a25600298819f8e28e8d593'
feature_names = [
    'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
    'PTRATIO', 'B', 'LSTAT'
]

UCI_TRAIN_DATA = None
UCI_TEST_DATA = None

FLUID_URL_MODEL = 'https://github.com/PaddlePaddle/book/raw/develop/01.fit_a_line/fluid/fit_a_line.fluid.tar'
FLUID_MD5_MODEL = '6e6dd637ccd5993961f68bfbde46090b'


def feature_range(maximums, minimums):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    feature_num = len(maximums)
    ax.bar(list(range(feature_num)),
           maximums - minimums,
           color='r',
           align='center')
    ax.set_title('feature scale')
    plt.xticks(list(range(feature_num)), feature_names)
    plt.xlim([-1, feature_num])
    fig.set_figheight(6)
    fig.set_figwidth(10)
    if not os.path.exists('./image'):
        os.makedirs('./image')
    fig.savefig('image/ranges.png', dpi=48)
    plt.close(fig)


def load_data(filename, feature_num=14, ratio=0.8):
    global UCI_TRAIN_DATA, UCI_TEST_DATA
    if UCI_TRAIN_DATA is not None and UCI_TEST_DATA is not None:
        return

    data = np.fromfile(filename, sep=' ')
    data = data.reshape(data.shape[0] // feature_num, feature_num)
    maximums, minimums, avgs = data.max(axis=0), data.min(axis=0), data.sum(
        axis=0) / data.shape[0]
    # if you want to print the distribution of input data, you could use function of feature_range
    #feature_range(maximums[:-1], minimums[:-1])
    for i in six.moves.range(feature_num - 1):
        data[:, i] = (data[:, i] - avgs[i]) / (maximums[i] - minimums[i])
    offset = int(data.shape[0] * ratio)
    UCI_TRAIN_DATA = data[:offset]
    UCI_TEST_DATA = data[offset:]


@deprecated(
    since="2.0.0",
    update_to="paddle.text.datasets.UCIHousing",
    level=1,
    reason="Please use new dataset API which supports paddle.io.DataLoader")
def train():
    """
    UCI_HOUSING training set creator.

    It returns a reader creator, each sample in the reader is features after
    normalization and price number.

    :return: Training reader creator
    :rtype: callable
    """
    global UCI_TRAIN_DATA
    load_data(paddle.dataset.common.download(URL, 'uci_housing', MD5))

    def reader():
        for d in UCI_TRAIN_DATA:
            yield d[:-1], d[-1:]

    return reader


@deprecated(
    since="2.0.0",
    update_to="paddle.text.datasets.UCIHousing",
    level=1,
    reason="Please use new dataset API which supports paddle.io.DataLoader")
def test():
    """
    UCI_HOUSING test set creator.

    It returns a reader creator, each sample in the reader is features after
    normalization and price number.

    :return: Test reader creator
    :rtype: callable
    """
    global UCI_TEST_DATA
    load_data(paddle.dataset.common.download(URL, 'uci_housing', MD5))

    def reader():
        for d in UCI_TEST_DATA:
            yield d[:-1], d[-1:]

    return reader


def fluid_model():
    parameter_tar = paddle.dataset.common.download(
        FLUID_URL_MODEL, 'uci_housing', FLUID_MD5_MODEL, 'fit_a_line.fluid.tar')

    tar = tarfile.TarFile(parameter_tar, mode='r')
    dirpath = tempfile.mkdtemp()
    tar.extractall(path=dirpath)

    return dirpath


@deprecated(
    since="2.0.0",
    update_to="paddle.text.datasets.UCIHousing",
    level=1,
    reason="Please use new dataset API which supports paddle.io.DataLoader")
def predict_reader():
    """
    It returns just one tuple data to do inference.

    :return: one tuple data
    :rtype: tuple
    """
    global UCI_TEST_DATA
    load_data(paddle.dataset.common.download(URL, 'uci_housing', MD5))
    return (UCI_TEST_DATA[0][:-1], )


@deprecated(
    since="2.0.0",
    update_to="paddle.text.datasets.UCIHousing",
    level=1,
    reason="Please use new dataset API which supports paddle.io.DataLoader")
def fetch():
    paddle.dataset.common.download(URL, 'uci_housing', MD5)
