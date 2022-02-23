# coding:utf-8
# Copyright (c) 2021  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

import glob
import json
import math
import os
import copy
import csv
import itertools

from .utils import download_file
from .utils import TermTree
from .knowledge_mining import WordTagTask
from .utils import Customization

usage = r"""
          from paddlenlp import Taskflow 

          ner = Taskflow("ner")
          ner("《孤女》是2010年九州出版社出版的小说，作者是余兼羽")
          '''
          [('《', 'w'), ('孤女', '作品类_实体'), ('》', 'w'), ('是', '肯定词'), ('2010年', '时间类'), ('九州出版社', '组织机构类'), ('出版', '场景事件'), ('的', '助词'), ('小说', '作品类_概念'), ('，', 'w'), ('作者', '人物类_概念'), ('是', '肯定词'), ('余兼羽', '人物类_实体')]
          '''

          ner = Taskflow("ner")
          ner(["热梅茶是一道以梅子为主要原料制作的茶饮", "《孤女》是2010年九州出版社出版的小说，作者是余兼羽"])
          '''
          [[('热梅茶', '饮食类_饮品'), ('是', '肯定词'), ('一道', '数量词'), ('以', '介词'), ('梅子', '饮食类'), ('为', '肯定词'), ('主要原料', '物体类'), ('制作', '场景事件'), ('的', '助词'), ('茶饮', '饮食类_饮品')], [('《', 'w'), ('孤女', '作品类_实体'), ('》', 'w'), ('是', '肯定词'), ('2010年', '时间类'), ('九州出版社', '组织机构类'), ('出版', '场景事件'), ('的', '助词'), ('小说', '作品类_概念'), ('，', 'w'), ('作者', '人物类_概念'), ('是', '肯定词'), ('余兼羽', '人物类_实体')]]
          '''
          """


class NERTask(WordTagTask):
    """
    This the NER(Named Entity Recognition) task that convert the raw text to entities. And the task with the `wordtag` 
    model will link the more meesage with the entity.
    Args:
        task(string): The name of task.
        model(string): The model name in the task.
        kwargs (dict, optional): Additional keyword arguments passed along to the specific task. 

    """

    resource_files_names = {
        "model_state": "model_state.pdparams",
        "model_config": "model_config.json",
        "tags": "tags.txt",
    }
    resource_files_urls = {
        "wordtag": {
            "model_state": [
                "https://bj.bcebos.com/paddlenlp/taskflow/knowledge_mining/wordtag/model_state.pdparams",
                "12685d1d84c09fb851b6c1541af1146e"
            ],
            "model_config": [
                "https://bj.bcebos.com/paddlenlp/taskflow/knowledge_mining/wordtag/model_config.json",
                "aa47cdf7c270943a24495bd5ff59dc00"
            ],
            "tags": [
                "https://bj.bcebos.com/paddlenlp/taskflow/knowledge_mining/wordtag/tags.txt",
                "87db06ae6ca42565157045ab3e9a996f"
            ],
        }
    }

    def __init__(self, model, task, **kwargs):
        super().__init__(model=model, task=task, **kwargs)
        if self._user_dict:
            self._custom = Customization()
            self._custom.load_customization(self._user_dict)
        else:
            self._custom = None

    def _decode(self, batch_texts, batch_pred_tags):
        batch_results = []
        for sent_index in range(len(batch_texts)):
            sent = batch_texts[sent_index]
            tags = [
                self._index_to_tags[index]
                for index in batch_pred_tags[sent_index][self.summary_num:len(
                    sent) + self.summary_num]
            ]
            if self._custom:
                self._custom.parse_customization(sent, tags, prefix=True)
            sent_out = []
            tags_out = []
            partial_word = ""
            for ind, tag in enumerate(tags):
                if partial_word == "":
                    partial_word = sent[ind]
                    tags_out.append(tag.split('-')[-1])
                    continue
                if tag.startswith("B") or tag.startswith("S") or tag.startswith(
                        "O"):
                    sent_out.append(partial_word)
                    tags_out.append(tag.split('-')[-1])
                    partial_word = sent[ind]
                    continue
                partial_word += sent[ind]

            if len(sent_out) < len(tags_out):
                sent_out.append(partial_word)

            pred_words = []
            for s, t in zip(sent_out, tags_out):
                pred_words.append({"item": s, "wordtag_label": t})

            result = {"text": sent, "items": pred_words}
            batch_results.append(result)
        return batch_results

    def _concat_short_text_reuslts(self, input_texts, results):
        """
        Concat the model output of short texts to the total result of long text.
        """
        long_text_lens = [len(text) for text in input_texts]
        concat_results = []
        single_results = {}
        count = 0
        for text in input_texts:
            text_len = len(text)
            while True:
                if len(single_results) == 0 or len(single_results[
                        "text"]) < text_len:
                    if len(single_results) == 0:
                        single_results = copy.deepcopy(results[count])
                    else:
                        single_results["text"] += results[count]["text"]
                        single_results["items"].extend(results[count]["items"])
                    count += 1
                elif len(single_results["text"]) == text_len:
                    concat_results.append(single_results)
                    single_results = {}
                    break
                else:
                    raise Exception(
                        "The length of input text and raw text is not equal.")
        simple_results = []
        for result in concat_results:
            simple_result = []
            if 'items' in result:
                for item in result['items']:
                    simple_result.append((item['item'], item['wordtag_label']))
            simple_results.append(simple_result)
        simple_results = simple_results[0] if len(
            simple_results) == 1 else simple_results
        return simple_results

    def _postprocess(self, inputs):
        """
        The model output is the tag ids, this function will convert the model output to raw text.
        """
        results = self._decode(inputs['short_input_texts'],
                               inputs['all_pred_tags'])
        results = self._concat_short_text_reuslts(inputs['inputs'], results)
        return results
