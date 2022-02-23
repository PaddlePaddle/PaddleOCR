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

import tarfile
import numpy as np
import gzip
import six

from paddle.io import Dataset
import paddle.compat as cpt
from paddle.dataset.common import _check_exists_and_download

__all__ = []

URL_DEV_TEST = ('http://www-lium.univ-lemans.fr/~schwenk/'
                'cslm_joint_paper/data/dev+test.tgz')
MD5_DEV_TEST = '7d7897317ddd8ba0ae5c5fa7248d3ff5'
# this is a small set of data for test. The original data is too large and
# will be add later.
URL_TRAIN = ('http://paddlemodels.bj.bcebos.com/wmt/wmt14.tgz')
MD5_TRAIN = '0791583d57d5beb693b9414c5b36798c'

START = "<s>"
END = "<e>"
UNK = "<unk>"
UNK_IDX = 2


class WMT14(Dataset):
    """
    Implementation of `WMT14 <http://www.statmt.org/wmt14/>`_ test dataset.
    The original WMT14 dataset is too large and a small set of data for set is
    provided. This module will download dataset from
    http://paddlemodels.bj.bcebos.com/wmt/wmt14.tgz .

    Args:
        data_file(str): path to data tar file, can be set None if
            :attr:`download` is True. Default None
        mode(str): 'train', 'test' or 'gen'. Default 'train'
        dict_size(int): word dictionary size. Default -1.
        download(bool): whether to download dataset automatically if
            :attr:`data_file` is not set. Default True

    Returns:
        Dataset: instance of WMT14 dataset

    Examples:

        .. code-block:: python

            import paddle
            from paddle.text.datasets import WMT14

            class SimpleNet(paddle.nn.Layer):
                def __init__(self):
                    super(SimpleNet, self).__init__()

                def forward(self, src_ids, trg_ids, trg_ids_next):
                    return paddle.sum(src_ids), paddle.sum(trg_ids), paddle.sum(trg_ids_next)

            wmt14 = WMT14(mode='train', dict_size=50)

            for i in range(10):
                src_ids, trg_ids, trg_ids_next = wmt14[i]
                src_ids = paddle.to_tensor(src_ids)
                trg_ids = paddle.to_tensor(trg_ids)
                trg_ids_next = paddle.to_tensor(trg_ids_next)

                model = SimpleNet()
                src_ids, trg_ids, trg_ids_next = model(src_ids, trg_ids, trg_ids_next)
                print(src_ids.numpy(), trg_ids.numpy(), trg_ids_next.numpy())

    """

    def __init__(self,
                 data_file=None,
                 mode='train',
                 dict_size=-1,
                 download=True):
        assert mode.lower() in ['train', 'test', 'gen'], \
            "mode should be 'train', 'test' or 'gen', but got {}".format(mode)
        self.mode = mode.lower()

        self.data_file = data_file
        if self.data_file is None:
            assert download, "data_file is not set and downloading automatically is disabled"
            self.data_file = _check_exists_and_download(
                data_file, URL_TRAIN, MD5_TRAIN, 'wmt14', download)

        # read dataset into memory
        assert dict_size > 0, "dict_size should be set as positive number"
        self.dict_size = dict_size
        self._load_data()

    def _load_data(self):
        def __to_dict(fd, size):
            out_dict = dict()
            for line_count, line in enumerate(fd):
                if line_count < size:
                    out_dict[cpt.to_text(line.strip())] = line_count
                else:
                    break
            return out_dict

        self.src_ids = []
        self.trg_ids = []
        self.trg_ids_next = []
        with tarfile.open(self.data_file, mode='r') as f:
            names = [
                each_item.name for each_item in f
                if each_item.name.endswith("src.dict")
            ]
            assert len(names) == 1
            self.src_dict = __to_dict(f.extractfile(names[0]), self.dict_size)
            names = [
                each_item.name for each_item in f
                if each_item.name.endswith("trg.dict")
            ]
            assert len(names) == 1
            self.trg_dict = __to_dict(f.extractfile(names[0]), self.dict_size)

            file_name = "{}/{}".format(self.mode, self.mode)
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
                        self.src_dict.get(w, UNK_IDX)
                        for w in [START] + src_words + [END]
                    ]

                    trg_seq = line_split[1]  # one target sequence
                    trg_words = trg_seq.split()
                    trg_ids = [self.trg_dict.get(w, UNK_IDX) for w in trg_words]

                    # remove sequence whose length > 80 in training mode
                    if len(src_ids) > 80 or len(trg_ids) > 80:
                        continue
                    trg_ids_next = trg_ids + [self.trg_dict[END]]
                    trg_ids = [self.trg_dict[START]] + trg_ids

                    self.src_ids.append(src_ids)
                    self.trg_ids.append(trg_ids)
                    self.trg_ids_next.append(trg_ids_next)

    def __getitem__(self, idx):
        return (np.array(self.src_ids[idx]), np.array(self.trg_ids[idx]),
                np.array(self.trg_ids_next[idx]))

    def __len__(self):
        return len(self.src_ids)

    def get_dict(self, reverse=False):
        """
        Get the source and target dictionary.

        Args:
            reverse (bool): wether to reverse key and value in dictionary,
                i.e. key: value to value: key.
    
        Returns:
            Two dictionaries, the source and target dictionary.
    
        Examples:
    
            .. code-block:: python
    
                from paddle.text.datasets import WMT14
                wmt14 = WMT14(mode='train', dict_size=50)
                src_dict, trg_dict = wmt14.get_dict()
        """
        src_dict, trg_dict = self.src_dict, self.trg_dict
        if reverse:
            src_dict = {v: k for k, v in six.iteritems(src_dict)}
            trg_dict = {v: k for k, v in six.iteritems(trg_dict)}
        return src_dict, trg_dict
