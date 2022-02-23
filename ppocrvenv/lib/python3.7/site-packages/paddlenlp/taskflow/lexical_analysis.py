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
import itertools

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ..datasets import load_dataset, MapDataset
from ..data import Stack, Pad, Tuple, Vocab, JiebaTokenizer
from .utils import download_file, add_docstrings, dygraph_mode_guard
from .utils import Customization
from .task import Task
from .models import BiGruCrf

usage = r"""
           from paddlenlp import Taskflow 

           lac = Taskflow("lexical_analysis")
           lac("LAC是个优秀的分词工具")
           '''
           [{'text': 'LAC是个优秀的分词工具', 'segs': ['LAC', '是', '个', '优秀', '的', '分词', '工具'], 'tags': ['nz', 'v', 'q', 'a', 'u', 'n', 'n']}]
           '''

           lac(["LAC是个优秀的分词工具", "三亚是一个美丽的城市"])
           '''
           [{'text': 'LAC是个优秀的分词工具', 'segs': ['LAC', '是', '个', '优秀', '的', '分词', '工具'], 'tags': ['nz', 'v', 'q', 'a', 'u', 'n', 'n']}, 
            {'text': '三亚是一个美丽的城市', 'segs': ['三亚', '是', '一个', '美丽', '的', '城市'], 'tags': ['LOC', 'v', 'm', 'a', 'u', 'n']}
           ]
           '''

         """


def load_vocab(dict_path):
    """
    Load vocab from file
    """
    vocab = {}
    reverse = None
    with open(dict_path, "r", encoding='utf8') as fin:
        for i, line in enumerate(fin):
            terms = line.strip("\n").split("\t")
            if len(terms) == 2:
                if reverse == None:
                    reverse = True if terms[0].isdigit() else False
                if reverse:
                    value, key = terms
                else:
                    key, value = terms
            elif len(terms) == 1:
                key, value = terms[0], str(i)
            else:
                raise ValueError("Error line: %s in file: %s" %
                                 (line, dict_path))
            vocab[key] = value
    return vocab


class LacTask(Task):
    """
    Lexical analysis of Chinese task to segement the chinese sentence. 
    Args:
        task(string): The name of task.
        model(string): The model name in the task.
        kwargs (dict, optional): Additional keyword arguments passed along to the specific task. 
    """

    resource_files_names = {
        "model_state": "model_state.pdparams",
        "tags": "tag.dic",
        "q2b": "q2b.dic",
        "word": "word.dic",
    }
    resource_files_urls = {
        "lac": {
            "model_state": [
                "https://bj.bcebos.com/paddlenlp/taskflow/lexical_analysis/lac/model_state.pdparams", 
                "3d4008c6c9d29424465829c9acf909bd"
            ],
            "tags": [
                "https://bj.bcebos.com/paddlenlp/taskflow/lexical_analysis/lac/tag.dic",
                "b11b616926b9f7f0a40a8087f84a8a99"
            ],
            "q2b": [
                "https://bj.bcebos.com/paddlenlp/taskflow/lexical_analysis/lac/q2b.dic",
                "4ef2cd16f8002fe7cd7dd31cdff47e0d"
            ],
            "word": [
                "https://bj.bcebos.com/paddlenlp/taskflow/lexical_analysis/lac/word.dic",
                "f1dfc68139bb6dd58c9c4313c341e436"
            ],
        }
    }

    def __init__(self, 
                 task, 
                 model,
                 user_dict=None,
                 **kwargs):
        super().__init__(task=task, model=model, **kwargs)
        self._usage = usage
        self._user_dict = user_dict
        self._check_task_files()
        self._construct_vocabs()
        self._get_inference_model()
        if self._user_dict:
            self._custom = Customization()
            self._custom.load_customization(self._user_dict)
        else:
            self._custom = None

    def _construct_input_spec(self):
        """
       Construct the input spec for the convert dygraph model to static model.
       """
        self._input_spec = [
            paddle.static.InputSpec(
                shape=[None, None], dtype="int64", name='token_ids'),
            paddle.static.InputSpec(
                shape=[None], dtype="int64", name='length')
        ]

    def _construct_vocabs(self):
        word_dict_path = os.path.join(self._task_path, "word.dic")
        tag_dict_path = os.path.join(self._task_path, "tag.dic")
        q2b_dict_path = os.path.join(self._task_path, "q2b.dic")
        self._word_vocab = load_vocab(word_dict_path)
        self._tag_vocab = load_vocab(tag_dict_path)
        self._q2b_vocab = load_vocab(q2b_dict_path)
        self._id2word_dict = dict(
            zip(self._word_vocab.values(), self._word_vocab.keys()))
        self._id2tag_dict = dict(
            zip(self._tag_vocab.values(), self._tag_vocab.keys()))

    def _construct_model(self, model):
        """
        Construct the inference model for the predictor.
        """
        model_instance = BiGruCrf(self.kwargs['emb_dim'],
                                  self.kwargs['hidden_size'],
                                  len(self._word_vocab), 
                                  len(self._tag_vocab))
        # Load the model parameter for the predict
        state_dict = paddle.load(
            os.path.join(self._task_path, "model_state.pdparams"))
        model_instance.set_dict(state_dict)
        self._model = model_instance
        self._model.eval()

    def _construct_tokenizer(self, model):
        """
        Construct the tokenizer for the predictor.
        """
        return None

    def _preprocess(self, inputs, padding=True, add_special_tokens=True):
        """
        Transform the raw text to the model inputs, two steps involved:
           1) Transform the raw text to token ids.
           2) Generate the other model inputs from the raw text and token ids.
        """
        inputs = self._check_input_text(inputs)
        # Get the config from the kwargs
        batch_size = self.kwargs[
            'batch_size'] if 'batch_size' in self.kwargs else 1
        num_workers = self.kwargs[
            'num_workers'] if 'num_workers' in self.kwargs else 0
        infer_data = []
        oov_token_id = self._word_vocab.get("OOV")

        filter_inputs = []

        def read(inputs):
            for input_tokens in inputs:
                if not (isinstance(input_tokens, str) and
                        len(input_tokens.strip()) > 0):
                    continue
                filter_inputs.append(input_tokens)
                ids = []
                for token in input_tokens:
                    token = self._q2b_vocab.get(token, token)
                    token_id = self._word_vocab.get(token, oov_token_id)
                    ids.append(token_id)
                lens = len(ids)
                yield ids, lens

        infer_ds = load_dataset(read, inputs=inputs, lazy=False)
        batchify_fn = lambda samples, fn=Tuple(
            Pad(axis=0, pad_val=0, dtype="int64"),  # input_ids
            Stack(dtype='int64'),  # seq_len
        ): fn(samples)
        infer_data_loader = paddle.io.DataLoader(
            infer_ds,
            collate_fn=batchify_fn,
            num_workers=num_workers,
            batch_size=batch_size,
            shuffle=False,
            return_list=True)
        outputs = {}
        outputs['text'] = filter_inputs
        outputs['data_loader'] = infer_data_loader
        return outputs

    def _run_model(self, inputs):
        """
        Run the task model from the outputs of the `_tokenize` function. 
        """
        results = []
        lens = []
        for batch in inputs['data_loader']:
            input_ids, seq_len = batch
            self.input_handles[0].copy_from_cpu(input_ids.numpy())
            self.input_handles[1].copy_from_cpu(seq_len.numpy())
            self.predictor.run()
            tags_ids = self.output_handle[0].copy_to_cpu()
            results.extend(tags_ids.tolist())
            lens.extend(seq_len.tolist())
        inputs['result'] = results
        inputs['lens'] = lens
        return inputs

    def _postprocess(self, inputs):
        """
        The model output is the tag ids, this function will convert the model output to raw text.
        """
        batch_out = []
        lengths = inputs['lens']
        preds = inputs['result']
        sents = inputs['text']
        final_results = []
        for sent_index in range(len(lengths)):
            single_result = {}
            tags = [
                self._id2tag_dict[str(index)]
                for index in preds[sent_index][:lengths[sent_index]]
            ]
            sent = sents[sent_index]
            if self._custom:
                self._custom.parse_customization(sent, tags)
            sent_out = []
            tags_out = []
            parital_word = ""
            for ind, tag in enumerate(tags):
                if parital_word == "":
                    parital_word = sent[ind]
                    tags_out.append(tag.split('-')[0])
                    continue
                if tag.endswith("-B") or (tag == "O" and tags[ind - 1] != "O"):
                    sent_out.append(parital_word)
                    tags_out.append(tag.split('-')[0])
                    parital_word = sent[ind]
                    continue
                parital_word += sent[ind]

            if len(sent_out) < len(tags_out):
                sent_out.append(parital_word)

            single_result['text'] = sent
            single_result['segs'] = sent_out
            single_result['tags'] = tags_out
            final_results.append(single_result)
        return final_results
