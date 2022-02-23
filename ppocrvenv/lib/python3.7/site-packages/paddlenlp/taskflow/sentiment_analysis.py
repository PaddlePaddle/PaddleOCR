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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ..datasets import load_dataset, MapDataset
from ..data import Stack, Pad, Tuple, Vocab, JiebaTokenizer
from ..transformers import SkepTokenizer
from .utils import download_file, add_docstrings, static_mode_guard, dygraph_mode_guard
from .models import BoWModel, LSTMModel, SkepSequenceModel
from .task import Task

usage = r"""
           from paddlenlp import Taskflow 

           senta = Taskflow("sentiment_analysis")
           senta("怀着十分激动的心情放映，可是看着看着发现，在放映完毕后，出现一集米老鼠的动画片")
           '''
           [{'text': '怀着十分激动的心情放映，可是看着看着发现，在放映完毕后，出现一集米老鼠的动画片', 'label': 'negative', 'score': 0.6691398620605469}]
           '''

           senta(["怀着十分激动的心情放映，可是看着看着发现，在放映完毕后，出现一集米老鼠的动画片", 
                  "作为老的四星酒店，房间依然很整洁，相当不错。机场接机服务很好，可以在车上办理入住手续，节省时间"])
           '''
           [{'text': '怀着十分激动的心情放映，可是看着看着发现，在放映完毕后，出现一集米老鼠的动画片', 'label': 'negative', 'score': 0.6691398620605469}, 
            {'text': '作为老的四星酒店，房间依然很整洁，相当不错。机场接机服务很好，可以在车上办理入住手续，节省时间', 'label': 'positive', 'score': 0.9857505559921265}
           ]
           '''

           senta = Taskflow("sentiment_analysis", model="skep_ernie_1.0_large_ch")
           senta("作为老的四星酒店，房间依然很整洁，相当不错。机场接机服务很好，可以在车上办理入住手续，节省时间。")
           '''
           [{'text': '作为老的四星酒店，房间依然很整洁，相当不错。机场接机服务很好，可以在车上办理入住手续，节省时间。', 'label': 'positive', 'score': 0.984320878982544}]
           '''
         """


class SentaTask(Task):
    """
    Sentiment analysis task using RNN or BOW model to predict sentiment opinion on Chinese text. 
    Args:
        task(string): The name of task.
        model(string): The model name in the task.
        kwargs (dict, optional): Additional keyword arguments passed along to the specific task. 
    """

    resource_files_names = {
        "model_state": "model_state.pdparams",
        "vocab": "vocab.txt"
    }
    resource_files_urls = {
        "bilstm": {
            "vocab": [
                "https://bj.bcebos.com/paddlenlp/taskflow/sentiment_analysis/bilstm/vocab.txt",
                "df714f0bfd6d749f88064679b4c97fd5"
            ],
            "model_state": [
                "https://bj.bcebos.com/paddlenlp/taskflow/sentiment_analysis/bilstm/model_state.pdparams",
                "609fc068aa35339e20f8310b5c20887c"
            ],
        }
    }

    def __init__(self, task, model, **kwargs):
        super().__init__(task=task, model=model, **kwargs)
        self._static_mode = True
        self._label_map = {0: 'negative', 1: 'positive'}
        self._check_task_files()
        self._construct_tokenizer(model)
        if self._static_mode:
            self._get_inference_model()
        else:
            self._construct_model(model)
        self._usage = usage

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

    def _construct_model(self, model):
        """
        Construct the inference model for the predictor.
        """
        vocab_size = self.kwargs['vocab_size']
        pad_token_id = self.kwargs['pad_token_id']
        num_classes = 2

        # Select the senta network for the inference
        model_instance = LSTMModel(
            vocab_size,
            num_classes,
            direction='bidirect',
            padding_idx=pad_token_id,
            pooling_type='max')
        model_path = os.path.join(self._task_path, "model_state.pdparams")

        # Load the model parameter for the predict
        state_dict = paddle.load(model_path)
        model_instance.set_dict(state_dict)
        self._model = model_instance
        self._model.eval()

    def _construct_tokenizer(self, model):
        """
        Construct the tokenizer for the predictor.
        """
        vocab_path = os.path.join(self._task_path, "vocab.txt")
        vocab = Vocab.load_vocabulary(
            vocab_path, unk_token='[UNK]', pad_token='[PAD]')

        vocab_size = len(vocab)
        pad_token_id = vocab.to_indices('[PAD]')
        # Construct the tokenizer form the JiebaToeknizer
        self.kwargs['pad_token_id'] = pad_token_id
        self.kwargs['vocab_size'] = vocab_size
        tokenizer = JiebaTokenizer(vocab)
        self._tokenizer = tokenizer

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
        examples = []
        filter_inputs = []
        for input_data in inputs:
            if not (isinstance(input_data, str) and len(input_data) > 0):
                continue
            filter_inputs.append(input_data)
            ids = self._tokenizer.encode(input_data)
            lens = len(ids)
            examples.append((ids, lens))

        batchify_fn = lambda samples, fn=Tuple(
            Pad(axis=0, pad_val=self._tokenizer.vocab.token_to_idx.get('[PAD]', 0)),  # input_ids
            Stack(dtype='int64'),  # seq_len
        ): fn(samples)
        batches = [
            examples[idx:idx + batch_size]
            for idx in range(0, len(examples), batch_size)
        ]
        outputs = {}
        outputs['data_loader'] = batches
        outputs['text'] = filter_inputs
        self.batchify_fn = batchify_fn
        return outputs

    def _run_model(self, inputs):
        """
        Run the task model from the outputs of the `_tokenize` function. 
        """
        results = []
        scores = []
        with static_mode_guard():
            for batch in inputs['data_loader']:
                ids, lens = self.batchify_fn(batch)
                self.input_handles[0].copy_from_cpu(ids)
                self.input_handles[1].copy_from_cpu(lens)
                self.predictor.run()
                idx = self.output_handle[0].copy_to_cpu().tolist()
                probs = self.output_handle[1].copy_to_cpu().tolist()
                labels = [self._label_map[i] for i in idx]
                score = [max(prob) for prob in probs]
                results.extend(labels)
                scores.extend(score)

        inputs['result'] = results
        inputs['score'] = scores
        return inputs

    def _postprocess(self, inputs):
        """
        This function will convert the model output to raw text.
        """
        final_results = []
        for text, label, score in zip(inputs['text'], inputs['result'],
                                      inputs['score']):
            result = {}
            result['text'] = text
            result['label'] = label
            result['score'] = score
            final_results.append(result)
        return final_results


class SkepTask(Task):
    """
    Sentiment analysis task using ERNIE-Gram model to predict sentiment opinion on Chinese text. 
    Args:
        task(string): The name of task.
        model(string): The model name in the task.
        kwargs (dict, optional): Additional keyword arguments passed along to the specific task. 
    """

    resource_files_names = {
        "model_state": "model_state.pdparams",
        "model_config": "model_config.json",
    }
    resource_files_urls = {
        "skep_ernie_1.0_large_ch": {
            "model_state": [
                "https://bj.bcebos.com/paddlenlp/taskflow/sentiment_analysis/skep_ernie_1.0_large_ch/model_state.pdparams",
                "cf7aa5f5ffa834b329bbcb1dca54e9fc"
            ],
            "model_config": [
                "https://bj.bcebos.com/paddlenlp/taskflow/sentiment_analysis/skep_ernie_1.0_large_ch/model_config.json",
                "847b84ab08611a2f5a01a22c18b0be23",
            ],
        }
    }

    def __init__(self, task, model, **kwargs):
        super().__init__(task=task, model=model, **kwargs)
        self._static_mode = True
        self._label_map = {0: 'negative', 1: 'positive'}
        self._check_task_files()
        self._construct_tokenizer(model)
        if self._static_mode:
            self._get_inference_model()
        else:
            self._construct_model(model)
        self._usage = usage

    def _construct_model(self, model):
        """
        Construct the inference model for the predictor.
        """
        model_instance = SkepSequenceModel.from_pretrained(
            self._task_path, num_classes=len(self._label_map))
        self._model = model_instance
        self._model.eval()

    def _construct_input_spec(self):
        """
       Construct the input spec for the convert dygraph model to static model.
       """
        self._input_spec = [
            paddle.static.InputSpec(
                shape=[None, None], dtype="int64"),  # input_ids
            paddle.static.InputSpec(
                shape=[None, None], dtype="int64")  # segment_ids
        ]

    def _construct_tokenizer(self, model):
        """
        Construct the tokenizer for the predictor.
        """
        tokenizer = SkepTokenizer.from_pretrained(model)
        self._tokenizer = tokenizer

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

        examples = []
        filter_inputs = []
        for input_data in inputs:
            if not (isinstance(input_data, str) and
                    len(input_data.strip()) > 0):
                continue
            filter_inputs.append(input_data)
            encoded_inputs = self._tokenizer(text=input_data, max_seq_len=128)
            ids = encoded_inputs["input_ids"]
            segment_ids = encoded_inputs["token_type_ids"]
            examples.append((ids, segment_ids))

        batchify_fn = lambda samples, fn=Tuple(
            Pad(axis=0, pad_val=self._tokenizer.pad_token_id),  # input ids
            Pad(axis=0, pad_val=self._tokenizer.pad_token_type_id),  # token type ids
        ): [data for data in fn(samples)]
        batches = [
            examples[idx:idx + batch_size]
            for idx in range(0, len(examples), batch_size)
        ]
        outputs = {}
        outputs['text'] = filter_inputs
        outputs['data_loader'] = batches
        self._batchify_fn = batchify_fn
        return outputs

    def _run_model(self, inputs):
        """
        Run the task model from the outputs of the `_tokenize` function. 
        """
        results = []
        scores = []
        with static_mode_guard():
            for batch in inputs['data_loader']:
                ids, segment_ids = self._batchify_fn(batch)
                self.input_handles[0].copy_from_cpu(ids)
                self.input_handles[1].copy_from_cpu(segment_ids)
                self.predictor.run()
                idx = self.output_handle[0].copy_to_cpu().tolist()
                probs = self.output_handle[1].copy_to_cpu().tolist()
                labels = [self._label_map[i] for i in idx]
                score = [max(prob) for prob in probs]
                results.extend(labels)
                scores.extend(score)

        inputs['result'] = results
        inputs['score'] = scores
        return inputs

    def _postprocess(self, inputs):
        """
        The model output is tag ids, this function will convert the model output to raw text.
        """
        final_results = []
        for text, label, score in zip(inputs['text'], inputs['result'],
                                      inputs['score']):
            result = {}
            result['text'] = text
            result['label'] = label
            result['score'] = score
            final_results.append(result)
        return final_results
