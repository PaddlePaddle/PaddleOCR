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

import io
import tarfile
import numpy as np
from PIL import Image

import paddle
from paddle.io import Dataset
from paddle.dataset.common import _check_exists_and_download

__all__ = []

VOC_URL = 'https://dataset.bj.bcebos.com/voc/VOCtrainval_11-May-2012.tar'

VOC_MD5 = '6cd6e144f989b92b3379bac3b3de84fd'
SET_FILE = 'VOCdevkit/VOC2012/ImageSets/Segmentation/{}.txt'
DATA_FILE = 'VOCdevkit/VOC2012/JPEGImages/{}.jpg'
LABEL_FILE = 'VOCdevkit/VOC2012/SegmentationClass/{}.png'

CACHE_DIR = 'voc2012'

MODE_FLAG_MAP = {'train': 'trainval', 'test': 'train', 'valid': "val"}


class VOC2012(Dataset):
    """
    Implementation of `VOC2012 <http://host.robots.ox.ac.uk/pascal/VOC/voc2012/>`_ dataset

    To speed up the download, we put the data on https://dataset.bj.bcebos.com/voc/VOCtrainval_11-May-2012.tar. 
    Original data can get from http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar.

    Args:
        data_file(str): path to data file, can be set None if
            :attr:`download` is True. Default None,  default data path: ~/.cache/paddle/dataset/voc2012
        mode(str): 'train', 'valid' or 'test' mode. Default 'train'.
        download(bool): download dataset automatically if :attr:`data_file` is None. Default True
        backend(str, optional): Specifies which type of image to be returned: 
            PIL.Image or numpy.ndarray. Should be one of {'pil', 'cv2'}. 
            If this option is not set, will get backend from ``paddle.vsion.get_image_backend`` ,
            default backend is 'pil'. Default: None.

    Examples:

        .. code-block:: python

            import paddle
            from paddle.vision.datasets import VOC2012
            from paddle.vision.transforms import Normalize

            class SimpleNet(paddle.nn.Layer):
                def __init__(self):
                    super(SimpleNet, self).__init__()

                def forward(self, image, label):
                    return paddle.sum(image), label


            normalize = Normalize(mean=[0.5, 0.5, 0.5],
                                  std=[0.5, 0.5, 0.5],
                                  data_format='HWC')
            voc2012 = VOC2012(mode='train', transform=normalize, backend='cv2')

            for i in range(10):
                image, label= voc2012[i]
                image = paddle.cast(paddle.to_tensor(image), 'float32')
                label = paddle.to_tensor(label)

                model = SimpleNet()
                image, label= model(image, label)
                print(image.numpy().shape, label.numpy().shape)

    """

    def __init__(self,
                 data_file=None,
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

        self.flag = MODE_FLAG_MAP[mode.lower()]

        self.data_file = data_file
        if self.data_file is None:
            assert download, "data_file is not set and downloading automatically is disabled"
            self.data_file = _check_exists_and_download(
                data_file, VOC_URL, VOC_MD5, CACHE_DIR, download)
        self.transform = transform

        # read dataset into memory
        self._load_anno()

        self.dtype = paddle.get_default_dtype()

    def _load_anno(self):
        self.name2mem = {}
        self.data_tar = tarfile.open(self.data_file)
        for ele in self.data_tar.getmembers():
            self.name2mem[ele.name] = ele

        set_file = SET_FILE.format(self.flag)
        sets = self.data_tar.extractfile(self.name2mem[set_file])

        self.data = []
        self.labels = []

        for line in sets:
            line = line.strip()
            data = DATA_FILE.format(line.decode('utf-8'))
            label = LABEL_FILE.format(line.decode('utf-8'))
            self.data.append(data)
            self.labels.append(label)

    def __getitem__(self, idx):
        data_file = self.data[idx]
        label_file = self.labels[idx]

        data = self.data_tar.extractfile(self.name2mem[data_file]).read()
        label = self.data_tar.extractfile(self.name2mem[label_file]).read()
        data = Image.open(io.BytesIO(data))
        label = Image.open(io.BytesIO(label))

        if self.backend == 'cv2':
            data = np.array(data)
            label = np.array(label)

        if self.transform is not None:
            data = self.transform(data)

        if self.backend == 'cv2':
            return data.astype(self.dtype), label.astype(self.dtype)

        return data, label

    def __len__(self):
        return len(self.data)

    def __del__(self):
        if self.data_tar:
            self.data_tar.close()
