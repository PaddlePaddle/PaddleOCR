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
import xml.dom.minidom

from paddle.dataset.common import md5file
from paddle.utils.download import get_path_from_url
from paddlenlp.utils.env import DATA_HOME
from . import DatasetBuilder


class HYP(DatasetBuilder):
    """
    Hyperpartisan News Detection
    Task: Given a news article text, decide whether it follows a hyperpartisan 
    argumentation, i.e., whether it exhibits blind, prejudiced, or unreasoning 
    allegiance to one party, faction, cause, or person.
    
    More detail at https://pan.webis.de/semeval19/semeval19-web/
    """
    URL = "https://bj.bcebos.com/paddlenlp/datasets/hyp.zip"
    MD5 = "125c504b4da6882c2d163ae9962b6220"
    META_INFO = collections.namedtuple('META_INFO', ('file', 'md5'))
    SPLITS = {
        'train': META_INFO(
            os.path.join('hyp', 'train.xml'),
            "f9dc8cb583db4c061a5abfb556d8c164"),
        'dev': META_INFO(
            os.path.join('hyp', 'eval.xml'),
            "20a7a7e82ae695a7fac4b8c48d0e4932"),
        'test': META_INFO(
            os.path.join('hyp', 'test.xml'), "5b1a166e7966fa744b402b033b9ed3ae")
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
        dom = xml.dom.minidom.parse(filename)
        example_nodes = dom.documentElement.getElementsByTagName('article')
        for example in example_nodes:
            text = ''.join([
                nodes.toprettyxml(
                    indent='', newl='') for nodes in example.childNodes
            ])
            label = example.getAttribute('hyperpartisan')
            yield {'text': text, 'label': label}

    def get_labels(self):
        """
        Return labels of the HYP object.
        """
        return ["false", "true"]
