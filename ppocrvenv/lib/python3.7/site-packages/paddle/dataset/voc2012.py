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
Image dataset for segmentation.
The 2012 dataset contains images from 2008-2011 for which additional
segmentations have been prepared. As in previous years the assignment
to training/test sets has been maintained. The total number of images
with segmentation has been increased from 7,062 to 9,993.
"""

from __future__ import print_function

import tarfile
import io
import numpy as np
from paddle.dataset.common import download
import paddle.utils.deprecated as deprecated
from PIL import Image

__all__ = []

VOC_URL = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/\
VOCtrainval_11-May-2012.tar'

VOC_MD5 = '6cd6e144f989b92b3379bac3b3de84fd'
SET_FILE = 'VOCdevkit/VOC2012/ImageSets/Segmentation/{}.txt'
DATA_FILE = 'VOCdevkit/VOC2012/JPEGImages/{}.jpg'
LABEL_FILE = 'VOCdevkit/VOC2012/SegmentationClass/{}.png'

CACHE_DIR = 'voc2012'


def reader_creator(filename, sub_name):

    tarobject = tarfile.open(filename)
    name2mem = {}
    for ele in tarobject.getmembers():
        name2mem[ele.name] = ele

    def reader():
        set_file = SET_FILE.format(sub_name)
        sets = tarobject.extractfile(name2mem[set_file])
        for line in sets:
            line = line.strip()
            data_file = DATA_FILE.format(line)
            label_file = LABEL_FILE.format(line)
            data = tarobject.extractfile(name2mem[data_file]).read()
            label = tarobject.extractfile(name2mem[label_file]).read()
            data = Image.open(io.BytesIO(data))
            label = Image.open(io.BytesIO(label))
            data = np.array(data)
            label = np.array(label)
            yield data, label

    return reader


@deprecated(
    since="2.0.0",
    update_to="paddle.vision.datasets.VOC2012",
    level=1,
    reason="Please use new dataset API which supports paddle.io.DataLoader")
def train():
    """
    Create a train dataset reader containing 2913 images in HWC order.
    """
    return reader_creator(download(VOC_URL, CACHE_DIR, VOC_MD5), 'trainval')


@deprecated(
    since="2.0.0",
    update_to="paddle.vision.datasets.VOC2012",
    level=1,
    reason="Please use new dataset API which supports paddle.io.DataLoader")
def test():
    """
    Create a test dataset reader containing 1464 images in HWC order.
    """
    return reader_creator(download(VOC_URL, CACHE_DIR, VOC_MD5), 'train')


@deprecated(
    since="2.0.0",
    update_to="paddle.vision.datasets.VOC2012",
    level=1,
    reason="Please use new dataset API which supports paddle.io.DataLoader")
def val():
    """
    Create a val dataset reader containing 1449 images in HWC order.
    """
    return reader_creator(download(VOC_URL, CACHE_DIR, VOC_MD5), 'val')
