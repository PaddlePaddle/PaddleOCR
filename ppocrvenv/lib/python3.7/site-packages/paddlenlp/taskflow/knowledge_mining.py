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
from collections import OrderedDict

import numpy as np
import paddle
import paddle.nn as nn
from paddlenlp.layers.crf import LinearChainCrf
from paddlenlp.utils.tools import compare_version
if compare_version(paddle.version.full_version, "2.2.0") >= 0:
    # paddle.text.ViterbiDecoder is supported by paddle after version 2.2.0
    from paddle.text import ViterbiDecoder
else:
    from paddlenlp.layers.crf import ViterbiDecoder

from ..datasets import MapDataset, load_dataset
from ..data import Stack, Pad, Tuple
from ..transformers import ErnieCtmWordtagModel, ErnieCtmNptagModel, ErnieCtmTokenizer
from .utils import download_file, add_docstrings, static_mode_guard, dygraph_mode_guard
from .utils import TermTree, BurkhardKellerTree
from .utils import Customization
from .task import Task

LABEL_TO_SCHEMA = {
    "人物类_实体": ["人物|E", "虚拟角色|E", "演艺团体|E"],
    "人物类_概念": ["人物|C", "虚拟角色|C"],
    "作品类_实体": ["作品与出版物|E"],
    "作品类_概念": ["作品与出版物|C", "文化类"],
    "组织机构类": ["组织机构"],
    "组织机构类_企事业单位": ["企事业单位", "品牌", "组织机构"],
    "组织机构类_医疗卫生机构": ["医疗卫生机构", "组织机构"],
    "组织机构类_国家机关": ["国家机关", "组织机构"],
    "组织机构类_体育组织机构": ["体育组织机构", "组织机构"],
    "组织机构类_教育组织机构": ["教育组织机构", "组织机构"],
    "组织机构类_军事组织机构": ["军事组织机构", "组织机构"],
    "物体类": ["物体与物品", "品牌", "虚拟物品", "虚拟物品"],
    "物体类_兵器": ["兵器"],
    "物体类_化学物质": ["物体与物品", "化学术语"],
    "其他角色类": ["角色"],
    "文化类": ["文化", "作品与出版物|C", "体育运动项目", "语言文字"],
    "文化类_语言文字": ["语言学术语"],
    "文化类_奖项赛事活动": ["奖项赛事活动", "特殊日", "事件"],
    "文化类_制度政策协议": ["制度政策协议", "法律法规"],
    "文化类_姓氏与人名": ["姓氏与人名"],
    "生物类": ["生物"],
    "生物类_植物": ["植物", "生物"],
    "生物类_动物": ["动物", "生物"],
    "品牌名": ["品牌", "企事业单位"],
    "场所类": ["区域场所", "居民服务机构", "医疗卫生机构"],
    "场所类_交通场所": ["交通场所", "设施"],
    "位置方位": ["位置方位"],
    "世界地区类": ["世界地区", "区域场所", "政权朝代"],
    "饮食类": ["饮食", "生物类", "药物"],
    "饮食类_菜品": ["饮食"],
    "饮食类_饮品": ["饮食"],
    "药物类": ["药物", "生物类"],
    "药物类_中药": ["药物", "生物类"],
    "医学术语类": ["医药学术语"],
    "术语类_生物体": ["生物学术语"],
    "疾病损伤类": ["疾病损伤", "动物疾病", "医药学术语"],
    "疾病损伤类_植物病虫害": ["植物病虫害", "医药学术语"],
    "宇宙类": ["天文学术语"],
    "事件类": ["事件", "奖项赛事活动"],
    "时间类": ["时间阶段", "政权朝代"],
    "术语类": ["术语"],
    "术语类_符号指标类": ["编码符号指标", "术语"],
    "信息资料": ["生活用语"],
    "链接地址": ["生活用语"],
    "个性特征": ["个性特点", "生活用语"],
    "感官特征": ["生活用语"],
    "场景事件": ["场景事件", "情绪", "态度", "个性特点"],
    "介词": ["介词"],
    "介词_方位介词": ["介词"],
    "助词": ["助词"],
    "代词": ["代词"],
    "连词": ["连词"],
    "副词": ["副词"],
    "疑问词": ["疑问词"],
    "肯定词": ["肯定否定词"],
    "否定词": ["肯定否定词"],
    "数量词": ["数量词", "量词"],
    "叹词": ["叹词"],
    "拟声词": ["拟声词"],
    "修饰词": ["修饰词", "生活用语"],
    "外语单词": ["日文假名", "词汇用语"],
    "汉语拼音": ["汉语拼音"],
}

usage = r"""
          from paddlenlp import Taskflow 

          # 默认使用WordTag词类知识标注工具
          wordtag = Taskflow("knowledge_mining", model="wordtag")
          wordtag("《孤女》是2010年九州出版社出版的小说，作者是余兼羽")
          '''
          [{'text': '《孤女》是2010年九州出版社出版的小说，作者是余兼羽', 'items': [{'item': '《', 'offset': 0, 'wordtag_label': 'w', 'length': 1}, {'item': '孤女', 'offset': 1, 'wordtag_label': '作品类_实体', 'length': 2}, {'item': '》', 'offset': 3, 'wordtag_label': 'w', 'length': 1}, {'item': '是', 'offset': 4, 'wordtag_label': '肯定词', 'length': 1, 'termid': '肯定否定词_cb_是'}, {'item': '2010年', 'offset': 5, 'wordtag_label': '时间类', 'length': 5, 'termid': '时间阶段_cb_2010年'}, {'item': '九州出版社', 'offset': 10, 'wordtag_label': '组织机构类', 'length': 5, 'termid': '组织机构_eb_九州出版社'}, {'item': '出版', 'offset': 15, 'wordtag_label': '场景事件', 'length': 2, 'termid': '场景事件_cb_出版'}, {'item': '的', 'offset': 17, 'wordtag_label': '助词', 'length': 1, 'termid': '助词_cb_的'}, {'item': '小说', 'offset': 18, 'wordtag_label': '作品类_概念', 'length': 2, 'termid': '小说_cb_小说'}, {'item': '，', 'offset': 20, 'wordtag_label': 'w', 'length': 1}, {'item': '作者', 'offset': 21, 'wordtag_label': '人物类_概念', 'length': 2, 'termid': '人物_cb_作者'}, {'item': '是', 'offset': 23, 'wordtag_label': '肯定词', 'length': 1, 'termid': '肯定否定词_cb_是'}, {'item': '余兼羽', 'offset': 24, 'wordtag_label': '人物类_实体', 'length': 3}]}]
          '''

          wordtag= Taskflow("knowledge_mining", batch_size=2)
          wordtag(["热梅茶是一道以梅子为主要原料制作的茶饮",
                   "《孤女》是2010年九州出版社出版的小说，作者是余兼羽"])
          '''
          [{'text': '热梅茶是一道以梅子为主要原料制作的茶饮', 'items': [{'item': '热梅茶', 'offset': 0, 'wordtag_label': '饮食类_饮品', 'length': 3}, {'item': '是', 'offset': 3, 'wordtag_label': '肯定词', 'length': 1, 'termid': '肯定否定词_cb_是'}, {'item': '一道', 'offset': 4, 'wordtag_label': '数量词', 'length': 2}, {'item': '以', 'offset': 6, 'wordtag_label': '介词', 'length': 1, 'termid': '介词_cb_以'}, {'item': '梅子', 'offset': 7, 'wordtag_label': '饮食类', 'length': 2, 'termid': '饮食_cb_梅'}, {'item': '为', 'offset': 9, 'wordtag_label': '肯定词', 'length': 1, 'termid': '肯定否定词_cb_为'}, {'item': '主要原料', 'offset': 10, 'wordtag_label': '物体类', 'length': 4, 'termid': '物品_cb_主要原料'}, {'item': '制作', 'offset': 14, 'wordtag_label': '场景事件', 'length': 2, 'termid': '场景事件_cb_制作'}, {'item': '的', 'offset': 16, 'wordtag_label': '助词', 'length': 1, 'termid': '助词_cb_的'}, {'item': '茶饮', 'offset': 17, 'wordtag_label': '饮食类_饮品', 'length': 2, 'termid': '饮品_cb_茶饮'}]}, {'text': '《孤女》是2010年九州出版社出版的小说，作者是余兼羽', 'items': [{'item': '《', 'offset': 0, 'wordtag_label': 'w', 'length': 1}, {'item': '孤女', 'offset': 1, 'wordtag_label': '作品类_实体', 'length': 2}, {'item': '》', 'offset': 3, 'wordtag_label': 'w', 'length': 1}, {'item': '是', 'offset': 4, 'wordtag_label': '肯定词', 'length': 1, 'termid': '肯定否定词_cb_是'}, {'item': '2010年', 'offset': 5, 'wordtag_label': '时间类', 'length': 5, 'termid': '时间阶段_cb_2010年'}, {'item': '九州出版社', 'offset': 10, 'wordtag_label': '组织机构类', 'length': 5, 'termid': '组织机构_eb_九州出版社'}, {'item': '出版', 'offset': 15, 'wordtag_label': '场景事件', 'length': 2, 'termid': '场景事件_cb_出版'}, {'item': '的', 'offset': 17, 'wordtag_label': '助词', 'length': 1, 'termid': '助词_cb_的'}, {'item': '小说', 'offset': 18, 'wordtag_label': '作品类_概念', 'length': 2, 'termid': '小说_cb_小说'}, {'item': '，', 'offset': 20, 'wordtag_label': 'w', 'length': 1}, {'item': '作者', 'offset': 21, 'wordtag_label': '人物类_概念', 'length': 2, 'termid': '人物_cb_作者'}, {'item': '是', 'offset': 23, 'wordtag_label': '肯定词', 'length': 1, 'termid': '肯定否定词_cb_是'}, {'item': '余兼羽', 'offset': 24, 'wordtag_label': '人物类_实体', 'length': 3}]}]
          '''

          # 切换为NPTag名词短语标注工具
          nptag = Taskflow("knowledge_mining", model="nptag")
          nptag("糖醋排骨")
          '''
          [{'text': '糖醋排骨', 'label': '菜品'}]
          '''

          nptag(["糖醋排骨", "红曲霉菌"])
          '''
          [{'text': '糖醋排骨', 'label': '菜品'}, {'text': '红曲霉菌', 'label': '微生物'}]
          '''

          # 输出粗粒度类别标签`category`，即WordTag的词汇标签。
          nptag = Taskflow("knowledge_mining", model="nptag", linking=True)
          nptag(["糖醋排骨", "红曲霉菌"])
          '''
          [{'text': '糖醋排骨', 'label': '菜品', 'category': '饮食类_菜品'}, {'text': '红曲霉菌', 'label': '微生物', 'category': '生物类_微生物'}]
          '''
         """


@add_docstrings(usage)
class WordTagTask(Task):
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
        "termtree_schema": "termtree_type.csv",
        "termtree_data": "termtree_data",
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
            "termtree_schema": [
                "https://bj.bcebos.com/paddlenlp/taskflow/knowledge_mining/wordtag/termtree_type.csv",
                "062cb9ac24f4135bf836e2a2fc5a1209"
            ],
            "termtree_data": [
                "https://bj.bcebos.com/paddlenlp/taskflow/knowledge_mining/wordtag/termtree_data",
                "a0efe723f84cf90540ac727be5b62e59"
            ],
            "tags": [
                "https://bj.bcebos.com/paddlenlp/taskflow/knowledge_mining/wordtag/tags.txt",
                "87db06ae6ca42565157045ab3e9a996f"
            ],
        }
    }

    def __init__(self,
                 model,
                 task,
                 params_path=None,
                 tag_path=None,
                 term_schema_path=None,
                 term_data_path=None,
                 user_dict=None,
                 linking=True,
                 **kwargs):
        super().__init__(model=model, task=task, **kwargs)
        self._tag_path = tag_path
        self._params_path = params_path
        self._term_schema_path = term_schema_path
        self._term_data_path = term_data_path
        self._user_dict = user_dict
        self._linking = linking
        self._check_task_files()
        self._load_task_resources()
        self._construct_tokenizer(model)
        self._usage = usage
        self._summary_num = 2
        self._get_inference_model()

        if self._user_dict:
            self._custom = Customization()
            self._custom.load_customization(self._user_dict)
        else:
            self._custom = None

    @property
    def summary_num(self):
        """
        Number of model summary token
        """
        return self._summary_num

    @property
    def linking(self):
        """
        Whether to do term linking.
        """
        return self._linking

    @staticmethod
    def _load_labels(tag_path):
        tags_to_idx = {}
        all_tags = []
        i = 0
        with open(tag_path, encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                tag = line.split("-")[-1]
                if tag not in all_tags:
                    all_tags.append(tag)
                tags_to_idx[line] = i
                i += 1
        idx_to_tags = dict(zip(*(tags_to_idx.values(), tags_to_idx.keys())))
        return tags_to_idx, idx_to_tags, all_tags

    def _load_task_resources(self):
        if self._tag_path is None:
            self._tag_path = os.path.join(self._task_path, "tags.txt")
        self._tags_to_index, self._index_to_tags, self._all_tags = self._load_labels(
            self._tag_path)
        if self._term_schema_path is None:
            self._term_schema_path = os.path.join(self._task_path,
                                                  "termtree_type.csv")
        if self._term_data_path is None:
            self._term_data_path = os.path.join(self._task_path,
                                                "termtree_data")
        if self._linking is True:
            self._termtree = TermTree.from_dir(
                self._term_schema_path, self._term_data_path, self._linking)

    def _split_long_text_input(self, input_texts, max_text_len):
        """
        Split the long text to list of short text, the max_seq_len of input text is 512,
        if the text length greater than 512, will this function that spliting the long text.
        """
        short_input_texts = []
        for text in input_texts:
            if len(text) <= max_text_len:
                short_input_texts.append(text)
            else:
                lens = len(text)
                temp_text_list = text.split("？。！")
                temp_text_list = [
                    temp_text for temp_text in temp_text_list
                    if len(temp_text) > 0
                ]
                if len(temp_text_list) <= 1:
                    temp_text_list = [
                        text[i:i + max_text_len]
                        for i in range(0, len(text), max_text_len)
                    ]
                    short_input_texts.extend(temp_text_list)
                else:
                    list_len = len(temp_text_list)
                    start = 0
                    end = 0
                    for i in range(0, list_len):
                        if len(temp_text_list[i]) + 1 >= max_text_len:
                            if start != end:
                                short_input_texts.extend(
                                    self._split_long_text_input(
                                        [text[start:end]], max_text_len))
                            short_input_texts.extend(
                                self._split_long_text_input([
                                    text[end:end + len(temp_text_list[i]) + 1]
                                ], max_text_len))
                            start = end + len(temp_text_list[i]) + 1
                            end = start
                        else:
                            if start + len(temp_text_list[
                                    i]) + 1 > max_text_len:
                                short_input_texts.extend(
                                    self._split_long_text_input(
                                        [text[start:end]], max_text_len))
                                start = end
                                end = end + len(temp_text_list[i]) + 1
                            else:
                                end = len(temp_text_list[i]) + 1
                    if start != end:
                        short_input_texts.extend(
                            self._split_long_text_input([text[start:end]],
                                                        max_text_len))
        return short_input_texts

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
        for result in concat_results:
            pred_words = result['items']
            pred_words = self._reset_offset(pred_words)
            result['items'] = pred_words
        return concat_results

    def _preprocess_text(self, input_texts):
        """
        Create the dataset and dataloader for the predict.
        """
        batch_size = self.kwargs[
            'batch_size'] if 'batch_size' in self.kwargs else 1
        num_workers = self.kwargs[
            'num_workers'] if 'num_workers' in self.kwargs else 0

        max_seq_length = 512
        if 'max_seq_length' in self.kwargs:
            max_seq_length = self.kwargs['max_seq_length']
        infer_data = []
        max_predict_len = max_seq_length - self.summary_num - 1
        filter_input_texts = []
        for input_text in input_texts:
            if not (isinstance(input_text, str) and len(input_text) > 0):
                continue
            filter_input_texts.append(input_text)
        input_texts = filter_input_texts

        short_input_texts = self._split_long_text_input(input_texts,
                                                        max_predict_len)

        def read(inputs):
            for text in inputs:
                tokenized_output = self._tokenizer(
                    list(text),
                    return_length=True,
                    is_split_into_words=True,
                    max_seq_len=max_seq_length)
                yield tokenized_output['input_ids'], tokenized_output[
                    'token_type_ids'], tokenized_output['seq_len']

        infer_ds = load_dataset(read, inputs=short_input_texts, lazy=False)
        batchify_fn = lambda samples, fn=Tuple(
            Pad(axis=0, pad_val=self._tokenizer.pad_token_id, dtype='int64'
                ),  # input_ids
            Pad(axis=0,
                pad_val=self._tokenizer.pad_token_type_id,
                dtype='int64'),  # token_type_ids
            Stack(dtype='int64'), # seq_len
        ): fn(samples)

        infer_data_loader = paddle.io.DataLoader(
            infer_ds,
            collate_fn=batchify_fn,
            num_workers=num_workers,
            batch_size=batch_size,
            shuffle=False,
            return_list=True)

        outputs = {}
        outputs['data_loader'] = infer_data_loader
        outputs['short_input_texts'] = short_input_texts
        outputs['inputs'] = input_texts
        return outputs

    def _reset_offset(self, pred_words):
        for i in range(0, len(pred_words)):
            if i > 0:
                pred_words[i]["offset"] = pred_words[i - 1]["offset"] + len(
                    pred_words[i - 1]["item"])
            pred_words[i]["length"] = len(pred_words[i]["item"])
        return pred_words

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
                pred_words.append({"item": s, "offset": 0, "wordtag_label": t})

            pred_words = self._reset_offset(pred_words)
            result = {"text": sent, "items": pred_words}
            batch_results.append(result)
        return batch_results

    def _term_linking(self, wordtag_res):
        for item in wordtag_res["items"]:
            flag, _ = self._termtree.find_term(item["item"])
            if flag is False:
                continue
            if item["wordtag_label"] not in LABEL_TO_SCHEMA:
                # Custom label defined by user
                if item["wordtag_label"] not in self._all_tags:
                    target_type_can = [item["wordtag_label"]]
                else:
                    continue
            else:
                target_type_can = LABEL_TO_SCHEMA[item["wordtag_label"]]
            for target_type_raw in target_type_can:
                target_type_ = target_type_raw.split("|")
                target_src = None
                if len(target_type_) == 2:
                    target_src = target_type_[1]
                target_type = target_type_[0]
                flag, term_id = self._termtree.find_term(item["item"],
                                                         target_type)
                if flag is False:
                    continue
                term_id = list(
                    filter(lambda d: self._termtree[d].node_type == "term",
                           term_id))
                if len(term_id) == 0:
                    continue
                if target_src is not None:
                    term_id = list(
                        filter(
                            lambda d: self._termtree[d].base.startswith(target_src.lower()),
                            term_id))
                    if len(term_id) == 0:
                        continue
                term_id.sort(
                    key=lambda d: (self._termtree[d].termtype == target_type or target_type in self._termtree[d].subtype, self._termtree[d].term == item["item"]),
                    reverse=True)
                item["termid"] = term_id[0]

    def _construct_input_spec(self):
        """
        Construct the input spec for the convert dygraph model to static model.
        """
        self._input_spec = [
            paddle.static.InputSpec(
                shape=[None, None], dtype="int64",
                name="input_ids"),  # input_ids
            paddle.static.InputSpec(
                shape=[None, None], dtype="int64",
                name="token_type_ids"),  # token_type_ids
            paddle.static.InputSpec(
                shape=[None], dtype="int64", name="seq_len"),  # seq_len
        ]

    def _construct_model(self, model):
        """
        Construct the inference model for the predictor.
        """
        model_instance = ErnieCtmWordtagModel.from_pretrained(
            model, num_tag=len(self._tags_to_index))
        if self._params_path is not None:
            state_dict = paddle.load(self._params_path)
            model_instance.set_dict(state_dict)
        self._model = model_instance
        self._model.eval()

    def _construct_tokenizer(self, model):
        """
        Construct the tokenizer for the predictor.
        """
        tokenizer_instance = ErnieCtmTokenizer.from_pretrained(model)
        self._tokenizer = tokenizer_instance

    def _preprocess(self, inputs, padding=True, add_special_tokens=True):
        """
        Transform the raw text to the model inputs, two steps involved:
           1) Transform the raw text to token ids.
           2) Generate the other model inputs from the raw text and token ids.
        """
        inputs = self._check_input_text(inputs)
        outputs = self._preprocess_text(inputs)
        return outputs

    def _run_model(self, inputs):
        """
        Run the task model from the outputs of the `_tokenize` function. 
        """
        all_pred_tags = []

        for batch in inputs['data_loader']:
            input_ids, token_type_ids, seq_len = batch
            self.input_handles[0].copy_from_cpu(input_ids.numpy())
            self.input_handles[1].copy_from_cpu(token_type_ids.numpy())
            self.input_handles[2].copy_from_cpu(seq_len.numpy())
            self.predictor.run()
            pred_tags = self.output_handle[0].copy_to_cpu()
            all_pred_tags.extend(pred_tags.tolist())
        inputs['all_pred_tags'] = all_pred_tags
        return inputs

    def _postprocess(self, inputs):
        """
        The model output is the tag ids, this function will convert the model output to raw text.
        """
        results = self._decode(inputs['short_input_texts'],
                               inputs['all_pred_tags'])
        results = self._concat_short_text_reuslts(inputs['inputs'], results)
        if self.linking is True:
            for res in results:
                self._term_linking(res)
        return results


@add_docstrings(usage)
class NPTagTask(Task):
    """
    Noun phrase tagging task that convert the noun phrase to POS tag.
    Args:
        task(string): The name of task.
        model(string): The model name in the task.
        batch_size(int): Numbers of examples a batch.
        linking(bool): Returns the categories. If `linking` is True, the fine-grained label (label) will link with the coarse-grained label (category).
    """

    resource_files_names = {
        "model_state": "model_state.pdparms",
        "model_config": "model_config.json",
        "name_category_map": "name_category_map.json",
    }
    resource_files_urls = {
        "nptag": {
            "model_state": [
                "https://bj.bcebos.com/paddlenlp/taskflow/knowledge_mining/nptag/model_state.pdparams",
                "05ed1906b42126d3e04b4ac5c210b010"
            ],
            "model_config": [
                "https://bj.bcebos.com/paddlenlp/taskflow/knowledge_mining/nptag/model_config.json",
                "17c9e5216abfc9bd94c7586574a3cbc4"
            ],
            "name_category_map": [
                "https://bj.bcebos.com/paddlenlp/taskflow/knowledge_mining/nptag/name_category_map.json",
                "c60810205993d307d919a26a3b96786f"
            ],
        }
    }

    def __init__(self,
                 task,
                 model,
                 batch_size=1,
                 max_seq_len=64,
                 linking=False,
                 **kwargs):
        super().__init__(task=task, model=model, **kwargs)
        self._usage = usage
        self._batch_size = batch_size
        self._max_seq_len = max_seq_len
        self._linking = linking
        self._check_task_files()
        self._construct_tokenizer(model)
        self._name_dict = None
        self._summary_num = 2
        self._max_cls_len = 5
        self._construct_dict_map()

        self._get_inference_model()
        if paddle.get_device().startswith("gpu"):
            inference_model_path = os.path.join(self._task_path, "static",
                                                "inference")
            model_file = inference_model_path + ".pdmodel"
            params_file = inference_model_path + ".pdiparams"
            self._config = paddle.inference.Config(model_file, params_file)
            self._config.enable_use_gpu(100, 0)
            self._config.switch_use_feed_fetch_ops(False)
            self._config.disable_glog_info()
            # TODO(linjieccc): enable embedding_eltwise_layernorm_fuse_pass after fixed
            self._config.delete_pass("embedding_eltwise_layernorm_fuse_pass")
            self.predictor = paddle.inference.create_predictor(self._config)
            self.input_handles = [
                self.predictor.get_input_handle(name)
                for name in self.predictor.get_input_names()
            ]
            self.output_handle = [
                self.predictor.get_output_handle(name)
                for name in self.predictor.get_output_names()
            ]

    @property
    def summary_num(self):
        """
        Number of model summary token
        """
        return self._summary_num

    def _construct_dict_map(self):
        """
        Construct dict map for the predictor.
        """
        name_dict_path = os.path.join(self._task_path, "name_category_map.json")
        with open(name_dict_path, encoding="utf-8") as fp:
            self._name_dict = json.load(fp)
        self._tree = BurkhardKellerTree()
        self._cls_vocabs = OrderedDict()
        for k in self._name_dict:
            self._tree.add(k)
            for c in k:
                if c not in self._cls_vocabs:
                    self._cls_vocabs[c] = len(self._cls_vocabs)
        self._cls_vocabs["[PAD]"] = len(self._cls_vocabs)
        self._id_vocabs = dict(
            zip(self._cls_vocabs.values(), self._cls_vocabs.keys()))
        self._vocab_ids = self._tokenizer.vocab.to_indices(
            list(self._cls_vocabs.keys()))

    def _decode(self, pred_ids):
        tokens = [self._id_vocabs[i] for i in pred_ids]
        valid_token = []
        for token in tokens:
            if token == "[PAD]":
                break
            valid_token.append(token)
        return "".join(valid_token)

    def _search(self, scores_can, pred_ids_can, depth, path, score):
        if depth >= 5:
            return [(path, score)]
        res = []
        for i in range(len(pred_ids_can[0])):
            tmp_res = self._search(scores_can, pred_ids_can, depth + 1,
                                   path + [pred_ids_can[depth][i]],
                                   score + scores_can[depth][i])
            res.extend(tmp_res)
        return res

    def _find_topk(self, a, k, axis=-1, largest=True, sorted=True):
        if axis is None:
            axis_size = a.size
        else:
            axis_size = a.shape[axis]
        assert 1 <= k <= axis_size

        a = np.asanyarray(a)
        if largest:
            index_array = np.argpartition(a, axis_size - k, axis=axis)
            topk_indices = np.take(index_array, -np.arange(k) - 1, axis=axis)
        else:
            index_array = np.argpartition(a, k - 1, axis=axis)
            topk_indices = np.take(index_array, np.arange(k), axis=axis)
        topk_values = np.take_along_axis(a, topk_indices, axis=axis)
        if sorted:
            sorted_indices_in_topk = np.argsort(topk_values, axis=axis)
            if largest:
                sorted_indices_in_topk = np.flip(
                    sorted_indices_in_topk, axis=axis)
            sorted_topk_values = np.take_along_axis(
                topk_values, sorted_indices_in_topk, axis=axis)
            sorted_topk_indices = np.take_along_axis(
                topk_indices, sorted_indices_in_topk, axis=axis)
            return sorted_topk_values, sorted_topk_indices
        return topk_values, topk_indices

    def _construct_input_spec(self):
        """
        Construct the input spec for the convert dygraph model to static model.
        """
        self._input_spec = [
            paddle.static.InputSpec(
                shape=[None, None], dtype="int64",
                name="input_ids"),  # input_ids
            paddle.static.InputSpec(
                shape=[None, None], dtype="int64",
                name="token_type_ids"),  # token_type_ids
        ]

    def _construct_model(self, model):
        """
        Construct the inference model for the predictor.
        """
        model_instance = ErnieCtmNptagModel.from_pretrained(self._task_path)
        self._model = model_instance
        self._model.eval()

    def _construct_tokenizer(self, model):
        """
        Construct the tokenizer for the predictor.
        """
        tokenizer_instance = ErnieCtmTokenizer.from_pretrained(model)
        self._tokenizer = tokenizer_instance

    def _preprocess(self, inputs):
        """
        Create the dataset and dataloader for the predict.
        """
        inputs = self._check_input_text(inputs)
        self._max_cls_len = 5
        num_workers = self.kwargs[
            'num_workers'] if 'num_workers' in self.kwargs else 0
        lazy_load = self.kwargs[
            'lazy_load'] if 'lazy_load' in self.kwargs else False

        # Prompt template: input_text + "是" + "[MASK]" * cls_seq_length
        prompt_template = ["是"] + ["[MASK]"] * self._max_cls_len

        def read(inputs):
            for text in inputs:
                if len(
                        text
                ) + self._max_cls_len + 1 + self._summary_num + 1 > self._max_seq_len:
                    text = text[:(self._max_seq_len - (self._max_cls_len + 1 +
                                                       self._summary_num + 1))]

                tokens = list(text) + prompt_template
                tokenized_output = self._tokenizer(
                    tokens,
                    return_length=True,
                    is_split_into_words=True,
                    pad_to_max_seq_len=True,
                    max_seq_len=self._max_seq_len)
                label_indices = list(
                    range(tokenized_output["seq_len"] - 1 - self._max_cls_len,
                          tokenized_output["seq_len"] - 1))

                yield tokenized_output["input_ids"], tokenized_output[
                    "token_type_ids"], label_indices

        infer_ds = load_dataset(read, inputs=inputs, lazy=lazy_load)
        batchify_fn = lambda samples, fn=Tuple(
            Stack(dtype='int64'),  # input_ids
            Stack(dtype='int64'),  # token_type_ids
            Stack(dtype='int64'),  # label_indices
        ): fn(samples)

        infer_data_loader = paddle.io.DataLoader(
            infer_ds,
            collate_fn=batchify_fn,
            num_workers=num_workers,
            batch_size=self._batch_size,
            shuffle=False,
            return_list=True)

        outputs = {}
        outputs['data_loader'] = infer_data_loader
        outputs['texts'] = inputs
        return outputs

    def _run_model(self, inputs):
        all_scores_can = []
        all_preds_can = []
        pred_ids = []

        for batch in inputs['data_loader']:
            input_ids, token_type_ids, label_indices = batch
            self.input_handles[0].copy_from_cpu(input_ids.numpy())
            self.input_handles[1].copy_from_cpu(token_type_ids.numpy())
            self.predictor.run()
            logits = self.output_handle[0].copy_to_cpu()
            for i, l in zip(label_indices, logits):
                score = l[i[0]:i[-1] + 1, self._vocab_ids]
                # Find topk candidates of scores and predicted indices.
                score_can, pred_id_can = self._find_topk(score, k=4, axis=-1)

                all_scores_can.extend([score_can.tolist()])
                all_preds_can.extend([pred_id_can.tolist()])
                pred_ids.extend([pred_id_can[:, 0].tolist()])

        inputs['all_scores_can'] = all_scores_can
        inputs['all_preds_can'] = all_preds_can
        inputs['pred_ids'] = pred_ids
        return inputs

    def _postprocess(self, inputs):
        results = []

        for i in range(len(inputs['texts'])):
            cls_label = self._decode(inputs['pred_ids'][i])

            result = {
                'text': inputs['texts'][i],
                'label': cls_label,
            }

            if cls_label not in self._name_dict:
                scores_can = inputs['all_scores_can'][i]
                pred_ids_can = inputs['all_preds_can'][i]
                labels_can = self._search(scores_can, pred_ids_can, 0, [], 0)
                labels_can.sort(key=lambda d: -d[1])
                for labels in labels_can:
                    cls_label_can = self._decode(labels[0])
                    if cls_label_can in self._name_dict:
                        result['label'] = cls_label_can
                        break
                    else:
                        labels_can = self._tree.search_similar_word(cls_label)
                        result['label'] = labels_can[0][0]

            if self._linking:
                result['category'] = self._name_dict[result['label']]
            results.append(result)
        return results
