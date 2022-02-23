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

__all__ = ['DuReaderChecklist']


class DuReaderChecklist(DatasetBuilder):
    '''
    A high-quality Chinese machine reading comprehension dataset for real 
    application scenarios. It that focus on challenging the MRC models 
    from multiple aspects, including understanding of vocabulary, phrase, 
    semantic role, reasoning and so on. 
    '''

    META_INFO = collections.namedtuple('META_INFO', ('file', 'md5', 'URL'))
    SPLITS = {
        'train': META_INFO(
            os.path.join('dataset', 'train.json'),
            '28881033c067c690826a841d2d72a18a',
            'https://bj.bcebos.com/paddlenlp/datasets/lic2021/dureader_checklist.dataset.tar.gz'
        ),
        'dev': META_INFO(
            os.path.join('dataset', 'dev.json'),
            '28881033c067c690826a841d2d72a18a',
            'https://bj.bcebos.com/paddlenlp/datasets/lic2021/dureader_checklist.dataset.tar.gz'
        ),
        'test1': META_INFO(
            os.path.join('test1', 'test1.json'),
            'd7047ada5fb6734b4e58bfa198d47f6e',
            'https://bj.bcebos.com/paddlenlp/datasets/lic2021/dureader_checklist.test1.tar.gz'
        )
    }

    def _get_data(self, mode, **kwargs):
        default_root = os.path.join(DATA_HOME, self.__class__.__name__)
        filename, data_hash, URL = self.SPLITS[mode]
        fullname = os.path.join(default_root, filename)
        if not os.path.exists(fullname) or (data_hash and
                                            not md5file(fullname) == data_hash):
            get_path_from_url(URL, default_root, data_hash)

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
                    answer_starts = []
                    answers = []
                    is_impossible = False
                    qa_type = qa.get('type', '')

                    if "is_impossible" in qa.keys():
                        is_impossible = qa["is_impossible"]

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
                        'type': qa_type,
                        'context': context,
                        'question': question,
                        'answers': answers,
                        'answer_starts': answer_starts,
                        'is_impossible': is_impossible
                    }
