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
MNIST dataset.

This module will download dataset from http://yann.lecun.com/exdb/mnist/ and
parse training set and test set into paddle reader creators.
"""

from __future__ import print_function

import paddle.dataset.common
import paddle.utils.deprecated as deprecated
import gzip
import numpy
import struct
from six.moves import range

__all__ = []

URL_PREFIX = 'https://dataset.bj.bcebos.com/mnist/'
TEST_IMAGE_URL = URL_PREFIX + 't10k-images-idx3-ubyte.gz'
TEST_IMAGE_MD5 = '9fb629c4189551a2d022fa330f9573f3'
TEST_LABEL_URL = URL_PREFIX + 't10k-labels-idx1-ubyte.gz'
TEST_LABEL_MD5 = 'ec29112dd5afa0611ce80d1b7f02629c'
TRAIN_IMAGE_URL = URL_PREFIX + 'train-images-idx3-ubyte.gz'
TRAIN_IMAGE_MD5 = 'f68b3c2dcbeaaa9fbdd348bbdeb94873'
TRAIN_LABEL_URL = URL_PREFIX + 'train-labels-idx1-ubyte.gz'
TRAIN_LABEL_MD5 = 'd53e105ee54ea40749a09fcbcd1e9432'


def reader_creator(image_filename, label_filename, buffer_size):
    def reader():
        with gzip.GzipFile(image_filename, 'rb') as image_file:
            img_buf = image_file.read()
            with gzip.GzipFile(label_filename, 'rb') as label_file:
                lab_buf = label_file.read()

                step_label = 0

                offset_img = 0
                # read from Big-endian
                # get file info from magic byte
                # image file : 16B
                magic_byte_img = '>IIII'
                magic_img, image_num, rows, cols = struct.unpack_from(
                    magic_byte_img, img_buf, offset_img)
                offset_img += struct.calcsize(magic_byte_img)

                offset_lab = 0
                # label file : 8B
                magic_byte_lab = '>II'
                magic_lab, label_num = struct.unpack_from(magic_byte_lab,
                                                          lab_buf, offset_lab)
                offset_lab += struct.calcsize(magic_byte_lab)

                while True:
                    if step_label >= label_num:
                        break
                    fmt_label = '>' + str(buffer_size) + 'B'
                    labels = struct.unpack_from(fmt_label, lab_buf, offset_lab)
                    offset_lab += struct.calcsize(fmt_label)
                    step_label += buffer_size

                    fmt_images = '>' + str(buffer_size * rows * cols) + 'B'
                    images_temp = struct.unpack_from(fmt_images, img_buf,
                                                     offset_img)
                    images = numpy.reshape(images_temp, (
                        buffer_size, rows * cols)).astype('float32')
                    offset_img += struct.calcsize(fmt_images)

                    images = images / 255.0
                    images = images * 2.0
                    images = images - 1.0

                    for i in range(buffer_size):
                        yield images[i, :], int(labels[i])

    return reader


@deprecated(
    since="2.0.0",
    update_to="paddle.vision.datasets.MNIST",
    level=1,
    reason="Please use new dataset API which supports paddle.io.DataLoader")
def train():
    """
    MNIST training set creator.

    It returns a reader creator, each sample in the reader is image pixels in
    [-1, 1] and label in [0, 9].

    :return: Training reader creator
    :rtype: callable
    """
    return reader_creator(
        paddle.dataset.common.download(TRAIN_IMAGE_URL, 'mnist',
                                       TRAIN_IMAGE_MD5),
        paddle.dataset.common.download(TRAIN_LABEL_URL, 'mnist',
                                       TRAIN_LABEL_MD5), 100)


@deprecated(
    since="2.0.0",
    update_to="paddle.vision.datasets.MNIST",
    level=1,
    reason="Please use new dataset API which supports paddle.io.DataLoader")
def test():
    """
    MNIST test set creator.

    It returns a reader creator, each sample in the reader is image pixels in
    [-1, 1] and label in [0, 9].

    :return: Test reader creator.
    :rtype: callable
    """
    return reader_creator(
        paddle.dataset.common.download(TEST_IMAGE_URL, 'mnist', TEST_IMAGE_MD5),
        paddle.dataset.common.download(TEST_LABEL_URL, 'mnist', TEST_LABEL_MD5),
        100)


@deprecated(
    since="2.0.0",
    update_to="paddle.vision.datasets.MNIST",
    level=1,
    reason="Please use new dataset API which supports paddle.io.DataLoader")
def fetch():
    paddle.dataset.common.download(TRAIN_IMAGE_URL, 'mnist', TRAIN_IMAGE_MD5)
    paddle.dataset.common.download(TRAIN_LABEL_URL, 'mnist', TRAIN_LABEL_MD5)
    paddle.dataset.common.download(TEST_IMAGE_URL, 'mnist', TEST_IMAGE_MD5)
    paddle.dataset.common.download(TEST_LABEL_URL, 'mnist', TEST_LABEL_MD5)
