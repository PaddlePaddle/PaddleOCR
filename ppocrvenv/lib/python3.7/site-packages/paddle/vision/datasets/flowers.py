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
import io
import tarfile
import numpy as np
from PIL import Image

import paddle
from paddle.io import Dataset
from paddle.utils import try_import
from paddle.dataset.common import _check_exists_and_download

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
MODE_FLAG_MAP = {'train': 'tstid', 'test': 'trnid', 'valid': 'valid'}


class Flowers(Dataset):
    """
    Implementation of `Flowers <https://www.robots.ox.ac.uk/~vgg/data/flowers/>`_
    dataset

    Args:
        data_file(str): path to data file, can be set None if
            :attr:`download` is True. Default None, default data path: ~/.cache/paddle/dataset/flowers/
        label_file(str): path to label file, can be set None if
            :attr:`download` is True. Default None, default data path: ~/.cache/paddle/dataset/flowers/
        setid_file(str): path to subset index file, can be set
            None if :attr:`download` is True. Default None
        mode(str): 'train', 'valid' or 'test' mode. Default 'train'.
        transform(callable): transform to perform on image, None for no transform.
        download(bool): download dataset automatically if :attr:`data_file` is None. Default True
        backend(str, optional): Specifies which type of image to be returned: 
            PIL.Image or numpy.ndarray. Should be one of {'pil', 'cv2'}. 
            If this option is not set, will get backend from ``paddle.vsion.get_image_backend`` ,
            default backend is 'pil'. Default: None.

    Examples:
        
        .. code-block:: python

            from paddle.vision.datasets import Flowers

            flowers = Flowers(mode='test')

            for i in range(len(flowers)):
                sample = flowers[i]
                print(sample[0].size, sample[1])

    """

    def __init__(self,
                 data_file=None,
                 label_file=None,
                 setid_file=None,
                 mode='train',
                 transform=None,
                 download=True,
                 backend=None):
        assert mode.lower() in ['train', 'valid', 'test'], \
                "mode should be 'train', 'valid' or 'test', but got {}".format(mode)

        if backend is None:
            backend = paddle.vision.get_image_backend()
        if backend not in ['pil', 'cv2']:
            raise ValueError(
                "Expected backend are one of ['pil', 'cv2'], but got {}"
                .format(backend))
        self.backend = backend

        flag = MODE_FLAG_MAP[mode.lower()]

        if not data_file:
            assert download, "data_file is not set and downloading automatically is disabled"
            data_file = _check_exists_and_download(
                data_file, DATA_URL, DATA_MD5, 'flowers', download)

        if not label_file:
            assert download, "label_file is not set and downloading automatically is disabled"
            label_file = _check_exists_and_download(
                label_file, LABEL_URL, LABEL_MD5, 'flowers', download)

        if not setid_file:
            assert download, "setid_file is not set and downloading automatically is disabled"
            setid_file = _check_exists_and_download(
                setid_file, SETID_URL, SETID_MD5, 'flowers', download)

        self.transform = transform

        data_tar = tarfile.open(data_file)
        self.data_path = data_file.replace(".tgz", "/")
        if not os.path.exists(self.data_path):
            os.mkdir(self.data_path)
        data_tar.extractall(self.data_path)

        scio = try_import('scipy.io')
        self.labels = scio.loadmat(label_file)['labels'][0]
        self.indexes = scio.loadmat(setid_file)[flag][0]

    def __getitem__(self, idx):
        index = self.indexes[idx]
        label = np.array([self.labels[index - 1]])
        img_name = "jpg/image_%05d.jpg" % index
        image = os.path.join(self.data_path, img_name)
        if self.backend == 'pil':
            image = Image.open(image)
        elif self.backend == 'cv2':
            image = np.array(Image.open(image))

        if self.transform is not None:
            image = self.transform(image)

        if self.backend == 'pil':
            return image, label.astype('int64')

        return image.astype(paddle.get_default_dtype()), label.astype('int64')

    def __len__(self):
        return len(self.indexes)
