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

import os
import gzip
import struct
import numpy as np
from PIL import Image

import paddle
from paddle.io import Dataset
from paddle.dataset.common import _check_exists_and_download

__all__ = []


class MNIST(Dataset):
    """
    Implementation of `MNIST <http://yann.lecun.com/exdb/mnist/>`_ dataset

    Args:
        image_path(str): path to image file, can be set None if
            :attr:`download` is True. Default None, default data path: ~/.cache/paddle/dataset/mnist
        label_path(str): path to label file, can be set None if
            :attr:`download` is True. Default None, default data path: ~/.cache/paddle/dataset/mnist
        mode(str): 'train' or 'test' mode. Default 'train'.
        download(bool): download dataset automatically if
            :attr:`image_path` :attr:`label_path` is not set. Default True
        backend(str, optional): Specifies which type of image to be returned: 
            PIL.Image or numpy.ndarray. Should be one of {'pil', 'cv2'}. 
            If this option is not set, will get backend from ``paddle.vsion.get_image_backend`` ,
            default backend is 'pil'. Default: None.
            
    Returns:
        Dataset: MNIST Dataset.

    Examples:
        
        .. code-block:: python

            from paddle.vision.datasets import MNIST

            mnist = MNIST(mode='test')

            for i in range(len(mnist)):
                sample = mnist[i]
                print(sample[0].size, sample[1])

    """
    NAME = 'mnist'
    URL_PREFIX = 'https://dataset.bj.bcebos.com/mnist/'
    TEST_IMAGE_URL = URL_PREFIX + 't10k-images-idx3-ubyte.gz'
    TEST_IMAGE_MD5 = '9fb629c4189551a2d022fa330f9573f3'
    TEST_LABEL_URL = URL_PREFIX + 't10k-labels-idx1-ubyte.gz'
    TEST_LABEL_MD5 = 'ec29112dd5afa0611ce80d1b7f02629c'
    TRAIN_IMAGE_URL = URL_PREFIX + 'train-images-idx3-ubyte.gz'
    TRAIN_IMAGE_MD5 = 'f68b3c2dcbeaaa9fbdd348bbdeb94873'
    TRAIN_LABEL_URL = URL_PREFIX + 'train-labels-idx1-ubyte.gz'
    TRAIN_LABEL_MD5 = 'd53e105ee54ea40749a09fcbcd1e9432'

    def __init__(self,
                 image_path=None,
                 label_path=None,
                 mode='train',
                 transform=None,
                 download=True,
                 backend=None):
        assert mode.lower() in ['train', 'test'], \
                "mode should be 'train' or 'test', but got {}".format(mode)

        if backend is None:
            backend = paddle.vision.get_image_backend()
        if backend not in ['pil', 'cv2']:
            raise ValueError(
                "Expected backend are one of ['pil', 'cv2'], but got {}"
                .format(backend))
        self.backend = backend

        self.mode = mode.lower()
        self.image_path = image_path
        if self.image_path is None:
            assert download, "image_path is not set and downloading automatically is disabled"
            image_url = self.TRAIN_IMAGE_URL if mode == 'train' else self.TEST_IMAGE_URL
            image_md5 = self.TRAIN_IMAGE_MD5 if mode == 'train' else self.TEST_IMAGE_MD5
            self.image_path = _check_exists_and_download(
                image_path, image_url, image_md5, self.NAME, download)

        self.label_path = label_path
        if self.label_path is None:
            assert download, "label_path is not set and downloading automatically is disabled"
            label_url = self.TRAIN_LABEL_URL if self.mode == 'train' else self.TEST_LABEL_URL
            label_md5 = self.TRAIN_LABEL_MD5 if self.mode == 'train' else self.TEST_LABEL_MD5
            self.label_path = _check_exists_and_download(
                label_path, label_url, label_md5, self.NAME, download)

        self.transform = transform

        # read dataset into memory
        self._parse_dataset()

        self.dtype = paddle.get_default_dtype()

    def _parse_dataset(self, buffer_size=100):
        self.images = []
        self.labels = []
        with gzip.GzipFile(self.image_path, 'rb') as image_file:
            img_buf = image_file.read()
            with gzip.GzipFile(self.label_path, 'rb') as label_file:
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
                    images = np.reshape(images_temp, (buffer_size, rows *
                                                      cols)).astype('float32')
                    offset_img += struct.calcsize(fmt_images)

                    for i in range(buffer_size):
                        self.images.append(images[i, :])
                        self.labels.append(
                            np.array([labels[i]]).astype('int64'))

    def __getitem__(self, idx):
        image, label = self.images[idx], self.labels[idx]
        image = np.reshape(image, [28, 28])

        if self.backend == 'pil':
            image = Image.fromarray(image.astype('uint8'), mode='L')

        if self.transform is not None:
            image = self.transform(image)

        if self.backend == 'pil':
            return image, label.astype('int64')

        return image.astype(self.dtype), label.astype('int64')

    def __len__(self):
        return len(self.labels)


class FashionMNIST(MNIST):
    """
    Implementation `Fashion-MNIST <https://github.com/zalandoresearch/fashion-mnist>`_ dataset.

    Args:
        image_path(str): path to image file, can be set None if
            :attr:`download` is True. Default None
        label_path(str): path to label file, can be set None if
            :attr:`download` is True. Default None
        mode(str): 'train' or 'test' mode. Default 'train'.
        download(bool): whether to download dataset automatically if
            :attr:`image_path` :attr:`label_path` is not set. Default True
        backend(str, optional): Specifies which type of image to be returned: 
            PIL.Image or numpy.ndarray. Should be one of {'pil', 'cv2'}. 
            If this option is not set, will get backend from ``paddle.vsion.get_image_backend`` ,
            default backend is 'pil'. Default: None.
            
    Returns:
        Dataset: Fashion-MNIST Dataset.

    Examples:
        
        .. code-block:: python

            from paddle.vision.datasets import FashionMNIST

            mnist = FashionMNIST(mode='test')

            for i in range(len(mnist)):
                sample = mnist[i]
                print(sample[0].size, sample[1])
    """

    NAME = 'fashion-mnist'
    URL_PREFIX = 'https://dataset.bj.bcebos.com/fashion_mnist/'
    TEST_IMAGE_URL = URL_PREFIX + 't10k-images-idx3-ubyte.gz'
    TEST_IMAGE_MD5 = 'bef4ecab320f06d8554ea6380940ec79'
    TEST_LABEL_URL = URL_PREFIX + 't10k-labels-idx1-ubyte.gz'
    TEST_LABEL_MD5 = 'bb300cfdad3c16e7a12a480ee83cd310'
    TRAIN_IMAGE_URL = URL_PREFIX + 'train-images-idx3-ubyte.gz'
    TRAIN_IMAGE_MD5 = '8d4fb7e6c68d591d4c3dfef9ec88bf0d'
    TRAIN_LABEL_URL = URL_PREFIX + 'train-labels-idx1-ubyte.gz'
    TRAIN_LABEL_MD5 = '25c81989df183df01b3e8a0aad5dffbe'
