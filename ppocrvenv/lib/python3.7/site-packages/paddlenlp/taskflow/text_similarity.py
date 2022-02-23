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

import paddle
from paddlenlp.transformers import BertModel, BertTokenizer

from ..data import Pad, Tuple
from .utils import static_mode_guard
from .task import Task

usage = r"""
         from paddlenlp import Taskflow

         similarity = Taskflow("text_similarity")
         similarity([["世界上什么东西最小", "世界上什么东西最小？"]])
         '''
         [{'text1': '世界上什么东西最小', 'text2': '世界上什么东西最小？', 'similarity': 0.992725}]
         '''

         similarity = Taskflow("text_similarity", batch_size=2)
         similarity([["光眼睛大就好看吗", "眼睛好看吗？"], ["小蝌蚪找妈妈怎么样", "小蝌蚪找妈妈是谁画的"]])
         '''
         [{'text1': '光眼睛大就好看吗', 'text2': '眼睛好看吗？', 'similarity': 0.74502707}, {'text1': '小蝌蚪找妈妈怎么样', 'text2': '小蝌蚪找妈妈是谁画的', 'similarity': 0.8192149}]
         '''
         """


class TextSimilarityTask(Task):
    """
    Text similarity task using SimBERT to predict the similarity of sentence pair.
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
        "simbert-base-chinese": {
            "model_state": [
                "https://bj.bcebos.com/paddlenlp/taskflow/text_similarity/simbert-base-chinese/model_state.pdparams",
                "27d9ef240c2e8e736bdfefea52af2542"
            ],
            "model_config": [
                "https://bj.bcebos.com/paddlenlp/taskflow/text_similarity/simbert-base-chinese/model_config.json",
                "1254bbd7598457a9dad0afcb2e24b70c"
            ],
        }
    }

    def __init__(self, task, model, batch_size=1, max_seq_len=128, **kwargs):
        super().__init__(task=task, model=model, **kwargs)
        self._check_task_files()
        self._construct_tokenizer(model)
        self._get_inference_model()
        self._batch_size = batch_size
        self._max_seq_len = max_seq_len
        self._usage = usage

    def _construct_input_spec(self):
        """
        Construct the input spec for the convert dygraph model to static model.
        """
        self._input_spec = [
            paddle.static.InputSpec(
                shape=[None, None], dtype="int64", name='input_ids'),
            paddle.static.InputSpec(
                shape=[None], dtype="int64", name='token_type_ids'),
        ]

    def _construct_model(self, model):
        """
        Construct the inference model for the predictor.
        """
        self._model = BertModel.from_pretrained(
            self._task_path, pool_act='linear')
        self._model.eval()

    def _construct_tokenizer(self, model):
        """
        Construct the tokenizer for the predictor.
        """
        self._tokenizer = BertTokenizer.from_pretrained(model)

    def _check_input_text(self, inputs):
        inputs = inputs[0]
        if not all([isinstance(i, list) and i \
            and all(i) and len(i) == 2 for i in inputs]):
            raise TypeError("Invalid input format.")
        return inputs

    def _preprocess(self, inputs):
        """
        Transform the raw text to the model inputs, two steps involved:
           1) Transform the raw text to token ids.
           2) Generate the other model inputs from the raw text and token ids.
        """
        inputs = self._check_input_text(inputs)
        num_workers = self.kwargs[
            'num_workers'] if 'num_workers' in self.kwargs else 0
        lazy_load = self.kwargs[
            'lazy_load'] if 'lazy_load' in self.kwargs else False

        examples = []

        for data in inputs:
            text1, text2 = data[0], data[1]

            text1_encoded_inputs = self._tokenizer(
                text=text1, max_seq_len=self._max_seq_len)
            text1_input_ids = text1_encoded_inputs["input_ids"]
            text1_token_type_ids = text1_encoded_inputs["token_type_ids"]

            text2_encoded_inputs = self._tokenizer(
                text=text2, max_seq_len=self._max_seq_len)
            text2_input_ids = text2_encoded_inputs["input_ids"]
            text2_token_type_ids = text2_encoded_inputs["token_type_ids"]

            examples.append((text1_input_ids, text1_token_type_ids,
                             text2_input_ids, text2_token_type_ids))

        batches = [
            examples[idx:idx + self._batch_size]
            for idx in range(0, len(examples), self._batch_size)
        ]

        batchify_fn = lambda samples, fn=Tuple(
            Pad(axis=0, pad_val=self._tokenizer.pad_token_id, dtype='int64'),  # text1_input_ids
            Pad(axis=0, pad_val=self._tokenizer.pad_token_type_id, dtype='int64'),  # text1_token_type_ids
            Pad(axis=0, pad_val=self._tokenizer.pad_token_id, dtype='int64'),  # text2_input_ids
            Pad(axis=0, pad_val=self._tokenizer.pad_token_type_id, dtype='int64'),  # text2_token_type_ids
        ): [data for data in fn(samples)]

        outputs = {}
        outputs['data_loader'] = batches
        outputs['text'] = inputs
        self._batchify_fn = batchify_fn
        return outputs

    def _run_model(self, inputs):
        """
        Run the task model from the outputs of the `_tokenize` function.
        """
        results = []
        with static_mode_guard():
            for batch in inputs['data_loader']:
                text1_ids, text1_segment_ids, text2_ids, text2_segment_ids = self._batchify_fn(
                    batch)
                self.input_handles[0].copy_from_cpu(text1_ids)
                self.input_handles[1].copy_from_cpu(text1_segment_ids)
                self.predictor.run()
                vecs_text1 = self.output_handle[1].copy_to_cpu()

                self.input_handles[0].copy_from_cpu(text2_ids)
                self.input_handles[1].copy_from_cpu(text2_segment_ids)
                self.predictor.run()
                vecs_text2 = self.output_handle[1].copy_to_cpu()

                vecs_text1 = vecs_text1 / (vecs_text1**2).sum(
                    axis=1, keepdims=True)**0.5
                vecs_text2 = vecs_text2 / (vecs_text2**2).sum(
                    axis=1, keepdims=True)**0.5
                similarity = (vecs_text1 * vecs_text2).sum(axis=1)
                results.extend(similarity)
        inputs['result'] = results
        return inputs

    def _postprocess(self, inputs):
        """
        The model output is tag ids, this function will convert the model output to raw text.
        """
        final_results = []
        for text, similarity in zip(inputs['text'], inputs['result']):
            result = {}
            result['text1'] = text[0]
            result['text2'] = text[1]
            result['similarity'] = similarity
            final_results.append(result)
        return final_results
