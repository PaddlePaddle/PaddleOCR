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
import os

from paddle.dataset.common import md5file
from paddle.utils.download import get_path_from_url
from paddlenlp.utils.env import DATA_HOME
from . import DatasetBuilder

__all__ = ['CAIL2019_SCM']


class CAIL2019_SCM(DatasetBuilder):
    '''
    CAIL2019-SCM contains 8,964 triplets of cases published by the Supreme People's 
    Court of China. The input of CAIL2019-SCM is a triplet (A, B, C), where A, B, C 
    are fact descriptions of three cases. The task of CAIL2019-SCM is to predict 
    whether sim(A, B) > sim(A, C) or sim(A, C) > sim(A, B).

    See more details on https://arxiv.org/abs/1911.08962.
    '''
    META_INFO = collections.namedtuple('META_INFO', ('file', 'md5', 'URL'))
    SPLITS = {
        'train': META_INFO(
            os.path.join('cail2019_scm_train.json'),
            'd50a105f9689e72be7d79adbba0ae224',
            'https://bj.bcebos.com/paddlenlp/datasets/cail2019/scm/cail2019_scm_train.json'
        ),
        'dev': META_INFO(
            os.path.join('cail2019_scm_dev.json'),
            'e36a295c1cb8c6b9fb28015907a42d9e',
            'https://bj.bcebos.com/paddlenlp/datasets/cail2019/scm/cail2019_scm_dev.json'
        ),
        'test': META_INFO(
            os.path.join('cail2019_scm_test.json'),
            '91a6cf060e1283f05fcc6a2027238379',
            'https://bj.bcebos.com/paddlenlp/datasets/cail2019/scm/cail2019_scm_test.json'
        )
    }

    def _get_data(self, mode, **kwargs):
        default_root = os.path.join(DATA_HOME, self.__class__.__name__)
        filename, data_hash, URL = self.SPLITS[mode]
        fullname = os.path.join(default_root, filename)
        if not os.path.exists(fullname) or (data_hash and
                                            not md5file(fullname) == data_hash):
            get_path_from_url(URL, default_root)

        return fullname

    def _read(self, filename, *args):
        with open(filename, "r", encoding="utf8") as f:
            for line in f.readlines():
                dic = json.loads(line)
                yield {
                    "text_a": dic["A"],
                    "text_b": dic["B"],
                    "text_c": dic["C"],
                    "label": dic["label"]
                }

    def get_labels(self):
        """
        Return labels of the CAIL2019_SCM object.
        """
        return ["B", "C"]
