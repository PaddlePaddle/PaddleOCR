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

__all__ = ['CMRC2018']


class CMRC2018(DatasetBuilder):
    '''
    This dataset is a Span-Extraction dataset for Chinese machine reading 
    comprehension. The dataset is composed by near 20,000 real questions 
    annotated on Wikipedia paragraphs by human experts.
    '''

    META_INFO = collections.namedtuple('META_INFO', ('file', 'md5', 'URL'))
    SPLITS = {
        'train': META_INFO(
            os.path.join('cmrc2018_train.json'),
            '7fb714b479c7f40fbb16acabd7af0ede',
            'https://bj.bcebos.com/paddlenlp/datasets/cmrc/cmrc2018_train.json'),
        'dev': META_INFO(
            os.path.join('cmrc2018_dev.json'),
            '853b80709ff2d071f9fce196521b843c',
            'https://bj.bcebos.com/paddlenlp/datasets/cmrc/cmrc2018_dev.json'),
        'trial': META_INFO(
            os.path.join('cmrc2018_trial.json'),
            '070f8ade5b15cfdb095c1fcef9cf43c1',
            'https://bj.bcebos.com/paddlenlp/datasets/cmrc/cmrc2018_trial.json')
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
            input_data = json.load(f)["data"]
        for entry in input_data:
            title = entry.get("title", "").strip()
            for paragraph in entry["paragraphs"]:
                context = paragraph["context"].strip()
                for qa in paragraph["qas"]:
                    qas_id = qa["id"]
                    question = qa["question"].strip()
                    answer_starts = [
                        answer["answer_start"]
                        for answer in qa.get("answers", [])
                    ]
                    answers = [
                        answer["text"].strip()
                        for answer in qa.get("answers", [])
                    ]

                    yield {
                        'id': qas_id,
                        'title': title,
                        'context': context,
                        'question': question,
                        'answers': answers,
                        'answer_starts': answer_starts
                    }
