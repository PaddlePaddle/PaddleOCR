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
from ..transformers import GPTForGreedyGeneration
from ..transformers import GPTChineseTokenizer, GPTTokenizer
from ..datasets import load_dataset
from ..data import Stack, Pad, Tuple
from .utils import download_file, add_docstrings, static_mode_guard, dygraph_mode_guard
from .task import Task

usage = r"""
         """

URLS = {
    "gpt-cpm-large-cn": [
        "https://bj.bcebos.com/paddlenlp/taskflow/text_generation/gpt-cpm/gpt-cpm-large-cn_params.tar",
        "5aad6f81053cfdbba4797f044fcf66d1"
    ],
}


class TextGenerationTask(Task):
    """
    The text generation model to predict the question or chinese  poetry. 
    Args:
        task(string): The name of task.
        model(string): The model name in the task.
        kwargs (dict, optional): Additional keyword arguments passed along to the specific task. 
    """

    def __init__(self, task, model, **kwargs):
        super().__init__(task=task, model=model, **kwargs)
        self._static_mode = True
        self._usage = usage
        if self._static_mode:
            download_file(self._task_path,
                          "static" + os.path.sep + "inference.pdiparams",
                          URLS[self.model][0], URLS[self.model][1])
            self._get_inference_model()
        else:
            self._construct_model(model)
        self._construct_tokenizer(model)
        self.kwargs['generation_task'] = task

    def _construct_input_spec(self):
        """
       Construct the input spec for the convert dygraph model to static model.
       """
        self._input_spec = [
            paddle.static.InputSpec(
                shape=[None, None], dtype="int64", name='token_ids')
        ]

    def _construct_model(self, model):
        """
        Construct the inference model for the predictor.
        """
        model_instance = GPTForGreedyGeneration.from_pretrained(
            self.model, max_predict_len=32)
        # Load the model parameter for the predict
        model_instance.eval()
        self._model = model_instance

    def _construct_tokenizer(self, model):
        """
        Construct the tokenizer for the predictor.
        """
        if self.model == "gpt-cpm-large-cn":
            tokenizer_instance = GPTChineseTokenizer.from_pretrained(model)
        else:
            tokenizer_instance = GPTTokenizer.from_pretrained(model)

        self._tokenizer = tokenizer_instance

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
        generation_task = self.kwargs[
            'generation_task'] if 'generation_task' in self.kwargs else 'question_answering'
        max_seq_len = 32

        def select_few_shot_input(model_name, generation_task):
            pre_input = ""
            if generation_task not in [
                    'question_answering', 'poetry_generation'
            ]:
                raise ValueError(
                    "The generation task must be question or poetry")
            if model_name == "gpt-cpm-large-cn":
                if generation_task == "question_answering":
                    pre_input = '问题：中国的首都是哪里？答案：北京。\n问题：{} 答案：'
                else:
                    pre_input = '默写古诗: 大漠孤烟直，长河落日圆。\n{}'
            return pre_input

        pre_input = select_few_shot_input(self.model, generation_task)

        infer_data = []

        examples = []
        filter_inputs = []
        for input_text in inputs:
            if not (isinstance(input_text, str) and len(input_text) > 0):
                continue
            filter_inputs.append(input_text)
            few_shot_input = pre_input.format(input_text)
            ids = self._tokenizer(few_shot_input)["input_ids"]
            examples.append((ids, len(ids)))

        batchify_fn = lambda samples, fn=Tuple(
            Pad(axis=0, pad_val=0, dtype="int64"),
            Stack(dtype='int64'),  # seq_len
        ): fn(samples)

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
        lens = []
        with static_mode_guard():
            for batch in inputs['data_loader']:
                ids, seq_len = self._batchify_fn(batch)
                self.input_handles[0].copy_from_cpu(ids)
                self.predictor.run()
                result = self.output_handle[0].copy_to_cpu().tolist()
                results.extend(result)
                lens.extend(seq_len.tolist())
        inputs['results'] = results
        inputs['lens'] = lens
        return inputs

    def _postprocess(self, inputs):
        """
        The model output is tag ids, this function will convert the model output to raw text.
        """
        batch_out = []
        preds = inputs['results']
        for index in range(0, len(preds)):
            seq_len = inputs['lens'][index]
            single_result = {}
            single_result['text'] = inputs['text'][index]
            single_result['answer'] = self._tokenizer.convert_ids_to_string(
                preds[index][seq_len:-1])
            batch_out.append(single_result)
        return batch_out
