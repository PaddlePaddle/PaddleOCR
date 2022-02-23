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
import os

from paddle.dataset.common import md5file
from paddle.utils.download import get_path_from_url
from paddlenlp.utils.env import DATA_HOME
from . import DatasetBuilder


class THUCNews(DatasetBuilder):
    """
    A subset of THUCNews dataset. THUCNews is a text classification dataset.
    See descrition about this subset version at https://github.com/gaussic/text-classification-cnn-rnn#%E6%95%B0%E6%8D%AE%E9%9B%86
    The whole dataset can be downloaded at https://thunlp.oss-cn-qingdao.aliyuncs.com/THUCNews.zip
    """
    URL = "https://bj.bcebos.com/paddlenlp/datasets/thucnews.zip"
    MD5 = "97626b2268f902662a29aadf222f22cc"
    META_INFO = collections.namedtuple('META_INFO', ('file', 'md5'))
    LABEL_PATH = os.path.join('thucnews', 'label.txt')
    SPLITS = {
        'train': META_INFO(
            os.path.join('thucnews', 'train.txt'),
            "beda43dfb4f7bd9bd3d465edb35fbb7f"),
        'dev': META_INFO(
            os.path.join('thucnews', 'val.txt'),
            "1abe8fe2c75dde701407a9161dcd223a"),
        'test': META_INFO(
            os.path.join('thucnews', 'test.txt'),
            "201f558b7d0b3419ddebcd695f3070f0")
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
        with open(filename, "r", encoding='utf8') as f:
            examples = f.readlines()
            for example in examples:
                split_idx = example.find('\t')
                label = example[:split_idx]
                text = example[split_idx + 1:].strip()
                yield {'text': text, 'label': label}

    def get_labels(self):
        labels = []
        filename = os.path.join(DATA_HOME, self.__class__.__name__,
                                self.LABEL_PATH)
        with open(filename, "r", encoding='utf8') as f:
            while True:
                label = f.readline().strip()
                if label == '':
                    break
                labels.append(label)
        return labels