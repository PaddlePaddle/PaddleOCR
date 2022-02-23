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
This module will download dataset from
http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html
and parse train/test set intopaddle reader creators.

This set contains images of flowers belonging to 102 different categories.
The images were acquired by searching the web and taking pictures. There are a
minimum of 40 images for each category.

The database was used in:

Nilsback, M-E. and Zisserman, A. Automated flower classification over a large
 number of classes.Proceedings of the Indian Conference on Computer Vision,
Graphics and Image Processing (2008)
http://www.robots.ox.ac.uk/~vgg/publications/papers/nilsback08.{pdf,ps.gz}.

"""

from __future__ import print_function

import itertools
import functools
from .common import download
import tarfile

from paddle.dataset.image import load_image_bytes
from paddle.dataset.image import load_image
from paddle.dataset.image import simple_transform
from paddle.dataset.image import batch_images_from_tar

from paddle.reader import map_readers, xmap_readers
from paddle import compat as cpt
import paddle.utils.deprecated as deprecated
import os
import numpy as np
from multiprocessing import cpu_count
import six
from six.moves import cPickle as pickle
from paddle.utils import try_import

__all__ = []

DATA_URL = 'http://paddlemodels.bj.bcebos.com/flowers/102flowers.tgz'
LABEL_URL = 'http://paddlemodels.bj.bcebos.com/flowers/imagelabels.mat'
SETID_URL = 'http://paddlemodels.bj.bcebos.com/flowers/setid.mat'
DATA_MD5 = '52808999861908f626f3c1f4e79d11fa'
LABEL_MD5 = 'e0620be6f572b9609742df49c70aed4d'
SETID_MD5 = 'a5357ecc9cb78c4bef273ce3793fc85c'
# In official 'readme', tstid is the flag of test data
# and trnid is the flag of train data. But test data is more than train data.
# So we exchange the train data and test data.
TRAIN_FLAG = 'tstid'
TEST_FLAG = 'trnid'
VALID_FLAG = 'valid'


def default_mapper(is_train, sample):
    '''
    map image bytes data to type needed by model input layer
    '''
    img, label = sample
    img = load_image_bytes(img)
    img = simple_transform(
        img, 256, 224, is_train, mean=[103.94, 116.78, 123.68])
    return img.flatten().astype('float32'), label


train_mapper = functools.partial(default_mapper, True)
test_mapper = functools.partial(default_mapper, False)


def reader_creator(data_file,
                   label_file,
                   setid_file,
                   dataset_name,
                   mapper,
                   buffered_size=1024,
                   use_xmap=True,
                   cycle=False):
    '''
    1. read images from tar file and
        merge images into batch files in 102flowers.tgz_batch/
    2. get a reader to read sample from batch file

    :param data_file: downloaded data file
    :type data_file: string
    :param label_file: downloaded label file
    :type label_file: string
    :param setid_file: downloaded setid file containing information
                        about how to split dataset
    :type setid_file: string
    :param dataset_name: data set name (tstid|trnid|valid)
    :type dataset_name: string
    :param mapper: a function to map image bytes data to type
                    needed by model input layer
    :type mapper: callable
    :param buffered_size: the size of buffer used to process images
    :type buffered_size: int
    :param cycle: whether to cycle through the dataset
    :type cycle: bool
    :return: data reader
    :rtype: callable
    '''

    def reader():
        scio = try_import('scipy.io')

        labels = scio.loadmat(label_file)['labels'][0]
        indexes = scio.loadmat(setid_file)[dataset_name][0]

        img2label = {}
        for i in indexes:
            img = "jpg/image_%05d.jpg" % i
            img2label[img] = labels[i - 1]

        tf = tarfile.open(data_file)
        mems = tf.getmembers()
        file_id = 0
        for mem in mems:
            if mem.name in img2label:
                image = tf.extractfile(mem).read()
                label = img2label[mem.name]
                yield image, int(label) - 1

    if use_xmap:
        return xmap_readers(mapper, reader, min(4, cpu_count()), buffered_size)
    else:
        return map_readers(mapper, reader)


@deprecated(
    since="2.0.0",
    update_to="paddle.vision.datasets.Flowers",
    level=1,
    reason="Please use new dataset API which supports paddle.io.DataLoader")
def train(mapper=train_mapper, buffered_size=1024, use_xmap=True, cycle=False):
    '''
    Create flowers training set reader.
    It returns a reader, each sample in the reader is
    image pixels in [0, 1] and label in [1, 102]
    translated from original color image by steps:
    1. resize to 256*256
    2. random crop to 224*224
    3. flatten
    :param mapper:  a function to map sample.
    :type mapper: callable
    :param buffered_size: the size of buffer used to process images
    :type buffered_size: int
    :param cycle: whether to cycle through the dataset
    :type cycle: bool
    :return: train data reader
    :rtype: callable
    '''
    return reader_creator(
        download(DATA_URL, 'flowers', DATA_MD5),
        download(LABEL_URL, 'flowers', LABEL_MD5),
        download(SETID_URL, 'flowers', SETID_MD5),
        TRAIN_FLAG,
        mapper,
        buffered_size,
        use_xmap,
        cycle=cycle)


@deprecated(
    since="2.0.0",
    update_to="paddle.vision.datasets.Flowers",
    level=1,
    reason="Please use new dataset API which supports paddle.io.DataLoader")
def test(mapper=test_mapper, buffered_size=1024, use_xmap=True, cycle=False):
    '''
    Create flowers test set reader.
    It returns a reader, each sample in the reader is
    image pixels in [0, 1] and label in [1, 102]
    translated from original color image by steps:
    1. resize to 256*256
    2. random crop to 224*224
    3. flatten
    :param mapper:  a function to map sample.
    :type mapper: callable
    :param buffered_size: the size of buffer used to process images
    :type buffered_size: int
    :param cycle: whether to cycle through the dataset
    :type cycle: bool
    :return: test data reader
    :rtype: callable
    '''
    return reader_creator(
        download(DATA_URL, 'flowers', DATA_MD5),
        download(LABEL_URL, 'flowers', LABEL_MD5),
        download(SETID_URL, 'flowers', SETID_MD5),
        TEST_FLAG,
        mapper,
        buffered_size,
        use_xmap,
        cycle=cycle)


@deprecated(
    since="2.0.0",
    update_to="paddle.vision.datasets.Flowers",
    level=1,
    reason="Please use new dataset API which supports paddle.io.DataLoader")
def valid(mapper=test_mapper, buffered_size=1024, use_xmap=True):
    '''
    Create flowers validation set reader.
    It returns a reader, each sample in the reader is
    image pixels in [0, 1] and label in [1, 102]
    translated from original color image by steps:
    1. resize to 256*256
    2. random crop to 224*224
    3. flatten
    :param mapper:  a function to map sample.
    :type mapper: callable
    :param buffered_size: the size of buffer used to process images
    :type buffered_size: int
    :return: test data reader
    :rtype: callable
    '''
    return reader_creator(
        download(DATA_URL, 'flowers', DATA_MD5),
        download(LABEL_URL, 'flowers', LABEL_MD5),
        download(SETID_URL, 'flowers', SETID_MD5), VALID_FLAG, mapper,
        buffered_size, use_xmap)


def fetch():
    download(DATA_URL, 'flowers', DATA_MD5)
    download(LABEL_URL, 'flowers', LABEL_MD5)
    download(SETID_URL, 'flowers', SETID_MD5)
