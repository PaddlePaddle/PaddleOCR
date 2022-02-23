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

__all__ = ['TriviaQA']


class TriviaQA(DatasetBuilder):
    '''
    TriviaQA is a reading comprehension dataset containing over 650K question-answer-evidence 
    triples. TriviaQA includes 95K question-answer pairs authored by trivia enthusiasts and 
    independently gathered evidence documents, six per question on average, that provide high 
    quality distant supervision for answering the questions. The details can be found ACL 
    17 paper: https://arxiv.org/abs/1705.03551.
    '''
    META_INFO = collections.namedtuple('META_INFO', ('file', 'md5', 'URL'))
    SPLITS = {
        'train': META_INFO(
            os.path.join('wikipedia-train.json'),
            'e4b3c74e781472d92e68da9c4b7418fe',
            'https://bj.bcebos.com/paddlenlp/datasets/triviaqa/wikipedia-train.zip'
        ),
        'dev': META_INFO(
            os.path.join('wikipedia-dev.json'),
            '20d23a2f668a46fe5c590d126f4d2b95',
            'https://bj.bcebos.com/paddlenlp/datasets/triviaqa/wikipedia-dev.zip'
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
            input_data = json.load(f)["data"]
        for entry in input_data:
            title = entry.get("title", "").strip()
            for paragraph in entry["paragraphs"]:
                context = paragraph["context"]
                for qa in paragraph["qas"]:
                    qas_id = qa["qid"]
                    question = qa["question"]
                    answer_starts = [
                        answer["answer_start"]
                        for answer in qa.get("answers", [])
                    ]
                    answers = [
                        answer["text"] for answer in qa.get("answers", [])
                    ]
                    if len(answers) == 1:
                        yield {
                            'id': qas_id,
                            'title': title,
                            'context': context,
                            'question': question,
                            'answers': answers,
                            'answer_starts': answer_starts
                        }
