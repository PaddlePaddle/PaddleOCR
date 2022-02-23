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
import os
import warnings
import json

from paddle.io import Dataset
from paddle.dataset.common import md5file
from paddle.utils.download import get_path_from_url
from paddlenlp.utils.env import DATA_HOME
from . import DatasetBuilder

__all__ = ['DuReaderYesNo']


class DuReaderYesNo(DatasetBuilder):
    '''
    DuReaderYesNo is a dataset with the judgment of opinion polarity as the 
    target task. Polarity of opinion is divided into three categories 
    {Yes, No, Depends}.
    '''

    URL = "https://bj.bcebos.com/paddlenlp/datasets/dureader_yesno-data.tar.gz"
    MD5 = '30c744d65e87fdce00cdc707fd008138'
    META_INFO = collections.namedtuple('META_INFO', ('file', 'md5'))
    SPLITS = {
        'train': META_INFO(
            os.path.join('dureader_yesno-data', 'train.json'),
            'c469a0ef3f975cfd705e3553ddb27cc1'),
        'dev': META_INFO(
            os.path.join('dureader_yesno-data', 'dev.json'),
            'c38544f8b5a7b567492314e3232057b5'),
        'test': META_INFO(
            os.path.join('dureader_yesno-data', 'test.json'),
            '1c7a1a3ea5b8992eeaeea017fdc2d55f')
    }

    def _get_data(self, mode, **kwargs):
        default_root = os.path.join(DATA_HOME, self.__class__.__name__)
        filename, data_hash = self.SPLITS[mode]
        fullname = os.path.join(default_root, filename)
        if not os.path.exists(fullname) or (data_hash and
                                            not md5file(fullname) == data_hash):

            get_path_from_url(self.URL, default_root, self.MD5)

        return fullname

    def _read(self, filename, *args):
        with open(filename, "r", encoding="utf8") as f:
            for entry in f:
                source = json.loads(entry.strip())
                yield {
                    'id': source['id'],
                    'question': source['question'],
                    'answer': source['answer'],
                    'labels': source['yesno_answer']
                }

    def get_labels(self):

        return ["Yes", "No", "Depends"]
