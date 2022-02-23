# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import os

from paddle.dataset.common import md5file
from paddle.utils.download import get_path_from_url
from paddlenlp.utils.env import DATA_HOME
from . import DatasetBuilder

__all__ = ['NLPCC14SC']


class NLPCC14SC(DatasetBuilder):
    """
    NLPCC14-SC is the dataset for sentiment classification. There are 2 classes
    in the datasets: Negative (0) and Positive (1). The following is a part of
    the train data:
      '''
      label	                  text_a
      1	                      超级值得看的一个电影
      0	                      我感觉卓越的东西现在好垃圾，还贵，关键贵。
      '''
    Please note that the test data contains no corresponding labels. 

    NLPCC14-SC datasets only contain train and test data, so we remove the dev
    data in META_INFO. By Fiyen at Beijing Jiaotong University.
    """

    URL = "https://bj.bcebos.com/paddlenlp/datasets/NLPCC14-SC.zip"
    MD5 = "4792a0982bc64b83d9a76dcce8bc00ad"
    META_INFO = collections.namedtuple('META_INFO', ('file', 'md5'))
    SPLITS = {
        'train': META_INFO(
            os.path.join('NLPCC14-SC', 'NLPCC14-SC', 'train.tsv'),
            'b0c6f74bb8d41020067c8f103c6e08c0'),
        'test': META_INFO(
            os.path.join('NLPCC14-SC', 'NLPCC14-SC', 'test.tsv'),
            '57526ba07510fdc901777e7602a26774'),
    }

    def _get_data(self, mode, **kwargs):
        """Downloads dataset."""
        default_root = os.path.join(DATA_HOME, self.__class__.__name__)
        filename, data_hash = self.SPLITS[mode]
        fullname = os.path.join(default_root, filename)
        if not os.path.exists(fullname) or (data_hash and
                                            not md5file(fullname) == data_hash):
            get_path_from_url(self.URL, default_root, self.MD5)

        return fullname

    def _read(self, filename, split):
        """Reads data."""
        with open(filename, 'r', encoding='utf-8') as f:
            head = None
            for line in f:
                data = line.strip().split("\t")
                if not head:
                    head = data
                else:
                    if split == 'train':
                        label, text = data
                        yield {"text": text, "label": label, "qid": ''}
                    elif split == 'test':
                        qid, text = data
                        yield {"text": text, "label": '', "qid": qid}

    def get_labels(self):
        """
        Return labels of the NLPCC14-SC object.
        """
        return ["0", "1"]
