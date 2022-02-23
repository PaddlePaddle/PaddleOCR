# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import collections
import json
import io
import os

import numpy as np

from paddle.dataset.common import md5file
from paddlenlp.utils.downloader import get_path_from_url
from paddlenlp.utils.env import DATA_HOME
from . import DatasetBuilder

__all__ = ['Imdb']


class Imdb(DatasetBuilder):
    """
    Subsets of IMDb data are available for access to customers for personal and non-commercial use.
    Each dataset is contained in a gzipped, tab-separated-values (TSV) formatted file in the UTF-8 character set.
    The first line in each file contains headers that describe what is in each column.
    Implementation of `IMDB <https://www.imdb.com/interfaces/>`_ dataset.

    """
    URL = 'https://bj.bcebos.com/dataset/imdb%2FaclImdb_v1.tar.gz'
    MD5 = '7c2ac02c03563afcf9b574c7e56c153a'
    META_INFO = collections.namedtuple('META_INFO', ('data_dir', 'md5'))
    SPLITS = {
        'train': META_INFO(os.path.join('aclImdb', 'train'), None),
        'test': META_INFO(os.path.join('aclImdb', 'test'), None),
    }

    def _get_data(self, mode, **kwargs):
        """Downloads dataset."""
        default_root = os.path.join(DATA_HOME, self.__class__.__name__)
        filename, _ = self.SPLITS[mode]
        data_dir = os.path.join(default_root, filename)
        if not os.path.exists(data_dir):
            path = get_path_from_url(self.URL, default_root, self.MD5)
        return data_dir

    def _read(self, data_dir, *args):
        for label in ["pos", "neg"]:
            root = os.path.join(data_dir, label)
            data_files = os.listdir(root)
            data_files.sort()

            if label == "pos":
                label_id = "1"
            elif label == "neg":
                label_id = "0"
            for f in data_files:
                f = os.path.join(root, f)
                with io.open(f, 'r', encoding='utf8') as fr:
                    data = fr.readlines()
                    data = data[0]
                    yield {"text": data, "label": label_id}

    def get_labels(self):
        """
        Return labels of the Imdb object.
        """
        return ["0", "1"]
