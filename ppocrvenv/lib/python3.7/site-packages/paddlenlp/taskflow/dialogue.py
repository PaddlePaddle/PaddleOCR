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

import os
import contextlib
from collections import deque

import numpy as np
import paddle

from ..transformers import UnifiedTransformerLMHeadModel, UnifiedTransformerTokenizer
from ..datasets import load_dataset
from ..data import Pad
from .utils import dygraph_mode_guard
from .task import Task

usage = r"""
           from paddlenlp import Taskflow 

           # 非交互模式
           dialogue = Taskflow("dialogue")
           dialogue(["吃饭了吗"])
           '''
           ['刚吃完饭,你在干什么呢?']
           '''
           dialogue(["你好", "吃饭了吗"], ["你是谁？"])
           '''
           ['吃过了,你呢', '我是李明啊']
           '''

           dialogue = Taskflow("dialogue")
           # 进入交互模式 (输入exit退出)
           dialogue.interactive_mode(max_turn=3)

           '''
           [Human]:你好
           [Bot]:你好,很高兴认识你,我想问你一下,你喜欢运动吗?
           [Human]:喜欢
           [Bot]:那你喜欢什么运动啊?
           [Human]:篮球,你喜欢篮球吗
           [Bot]:当然了,我很喜欢打篮球的。
           '''           
         """


class DialogueTask(Task):
    """
    Task of Chinese open domain dialogue.
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
        "plato-mini": {
            "model_state": [
                "https://bj.bcebos.com/paddlenlp/taskflow/dialogue/plato-mini/model_state.pdparams",
                "450be85b9b7f0bc03b12252a75af04f3"
            ],
            "model_config": [
                "https://bj.bcebos.com/paddlenlp/taskflow/dialogue/plato-mini/model_config.json",
                "5e853fda9a9b573815ad112e494a65af"
            ],
        }
    }

    def __init__(self, task, model, batch_size=1, max_seq_len=512, **kwargs):
        super().__init__(task=task, model=model, **kwargs)
        self._static_mode = False
        self._usage = usage
        self._check_task_files()
        self._construct_tokenizer(model)
        self._batch_size = batch_size
        self._max_seq_len = max_seq_len
        self._interactive_mode = False
        if self._static_mode:
            self._get_inference_model()
        else:
            self._construct_model(model)

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
        model_instance = UnifiedTransformerLMHeadModel.from_pretrained(
            self._task_path)
        model_path = os.path.join(self._task_path, "model_state.pdparams")
        state_dict = paddle.load(model_path)
        model_instance.set_state_dict(state_dict)
        self._model = model_instance
        self._model.eval()

    def _construct_tokenizer(self, model):
        """
        Construct the tokenizer for the predictor.
        """
        self._tokenizer = UnifiedTransformerTokenizer.from_pretrained(model)

    def _batchify_fn(self, batch_examples):
        #padding = False if self._batch_size == 1 else True
        pad_func = Pad(pad_val=self._tokenizer.pad_token_id,
                       pad_right=False,
                       dtype='int64')

        def pad_mask(batch_attention_mask):
            batch_size = len(batch_attention_mask)
            max_len = max(map(len, batch_attention_mask))
            attention_mask = np.ones(
                (batch_size, max_len, max_len), dtype='float32') * -1e4
            for i, mask_data in enumerate(attention_mask):
                seq_len = len(batch_attention_mask[i])
                mask_data[-seq_len:, -seq_len:] = np.array(
                    batch_attention_mask[i], dtype='float32')
            # In order to ensure the correct broadcasting mechanism, expand one
            # dimension to the second dimension (n_head of Transformer).
            attention_mask = np.expand_dims(attention_mask, axis=1)
            return attention_mask

        input_ids = pad_func(
            [example['input_ids'] for example in batch_examples])
        token_type_ids = pad_func(
            [example['token_type_ids'] for example in batch_examples])
        position_ids = pad_func(
            [example['position_ids'] for example in batch_examples])
        attention_mask = pad_mask(
            [example['attention_mask'] for example in batch_examples])

        return input_ids, token_type_ids, position_ids, attention_mask

    def _check_input_text(self, inputs):
        if self._interactive_mode:
            if isinstance(inputs, str):
                self.context.append(inputs.strip())
                inputs = [list(self.context)]
                return inputs
            else:
                raise ValueError(
                    "In the interactive mode, the input data shold be a string")
        elif not isinstance(inputs[0], list):
            raise ValueError(
                "If not in the interactive mode, the input data should be a list."
            )
        return inputs

    def _batchify(self, data, max_seq_len, batch_size):
        """
        Generate input batches.
        """
        padding = False if batch_size == 1 else True
        pad_func = Pad(pad_val=self._tokenizer.pad_token_id,
                       pad_right=False,
                       dtype=np.int64)

        def pad_mask(batch_attention_mask):
            batch_size = len(batch_attention_mask)
            max_len = max(map(len, batch_attention_mask))
            attention_mask = np.ones(
                (batch_size, max_len, max_len), dtype='float32') * -1e4
            for i, mask_data in enumerate(attention_mask):
                seq_len = len(batch_attention_mask[i])
                mask_data[-seq_len:, -seq_len:] = np.array(
                    batch_attention_mask[i], dtype='float32')
            # In order to ensure the correct broadcasting mechanism, expand one
            # dimension to the second dimension (n_head of Transformer).
            attention_mask = np.expand_dims(attention_mask, axis=1)
            return attention_mask

        def _parse_batch(batch_examples):
            if padding:
                input_ids = pad_func(
                    [example['input_ids'] for example in batch_examples])
                token_type_ids = pad_func(
                    [example['token_type_ids'] for example in batch_examples])
                position_ids = pad_func(
                    [example['position_ids'] for example in batch_examples])
                attention_mask = pad_mask(
                    [example['attention_mask'] for example in batch_examples])
            else:
                input_ids = np.asarray(
                    [example['input_ids'] for example in batch_examples],
                    dtype=np.int64)
                token_type_ids = np.asarray(
                    [example['token_type_ids'] for example in batch_examples],
                    dtype=np.int64)
                position_ids = np.asarray(
                    [example['position_ids'] for example in batch_examples],
                    dtype=np.int64)
                attention_mask = np.asarray(
                    [example['attention_mask'] for example in batch_examples])
                attention_mask = np.expand_dims(attention_mask, 0)

            return input_ids, token_type_ids, position_ids, attention_mask

        examples = []
        for texts in data:
            examples.append(self._convert_text_to_input(texts, max_seq_len))

        # Seperates data into some batches.
        one_batch = []
        for example in examples:
            one_batch.append(example)
            if len(one_batch) == batch_size:
                yield _parse_batch(one_batch)
                one_batch = []
        if one_batch:
            yield _parse_batch(one_batch)

    def _convert_text_to_input(self, texts, max_seq_len):
        """
        Convert input strings to tokens.
        """
        return self._tokenizer.dialogue_encode(
            texts,
            max_seq_len=max_seq_len,
            add_start_token_as_response=True,
            is_split_into_words=False)

    def _preprocess(self, inputs):
        """
        Transform the raw text to the model inputs, two steps involved:
           1) Transform the raw text to token ids.
           2) Generate the other model inputs from the raw text and token ids.
        """
        inputs = self._check_input_text(inputs)
        # Get the config from the kwargs
        num_workers = self.kwargs[
            'num_workers'] if 'num_workers' in self.kwargs else 0
        lazy_load = self.kwargs[
            'lazy_load'] if 'lazy_load' in self.kwargs else False

        batches = self._batchify(inputs, self._max_seq_len, self._batch_size)

        outputs = {}
        outputs['batches'] = batches
        outputs['text'] = inputs
        return outputs

    def _run_model(self, inputs):
        """
        Run the task model from the outputs of the `_tokenize` function.
        """
        all_ids = []
        all_scores = []

        for batch in inputs["batches"]:
            input_ids, token_type_ids, position_ids, attention_mask = map(
                paddle.to_tensor, batch)
            ids, scores = self._model.generate(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                max_length=64,
                min_length=1,
                decode_strategy='sampling',
                temperature=1.0,
                top_k=5,
                top_p=1.0,
                num_beams=0,
                length_penalty=1.0,
                early_stopping=False,
                use_faster=False,
                num_return_sequences=1)
            all_ids.extend([ids])
            all_scores.extend([scores])
        inputs['ids'] = all_ids
        inputs['scores'] = all_scores
        return inputs

    def _post_process_response(self, token_ids, tokenizer):
        '''
        Post-process the decoded sequence. Truncate from the first <eos>.
        '''
        eos_pos = len(token_ids)
        for i, tok_id in enumerate(token_ids):
            if tok_id == tokenizer.sep_token_id:
                eos_pos = i
                break
        token_ids = token_ids[:eos_pos]
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        tokens = tokenizer.merge_subword(tokens)
        return token_ids, tokens

    @contextlib.contextmanager
    def interactive_mode(self, max_turn=3):
        """
        Enter the interactive mode.
        """
        self._interactive_mode = True
        self.max_turn = max_turn
        self.context = deque(maxlen=self.max_turn)
        yield
        self.context.clear()
        self._interactive_mode = False

    def _get_in_turn_repetition(self, pred, is_cn=False):
        '''
        Get in-turn repetition.
        '''
        if len(pred) == 0:
            return 1.0
        if isinstance(pred[0], str):
            pred = [tok.lower() for tok in pred]
            if is_cn:
                pred = "".join(pred)
        tri_grams = set()
        for i in range(len(pred) - 2):
            tri_gram = tuple(pred[i:i + 3])
            if tri_gram in tri_grams:
                return True
            tri_grams.add(tri_gram)
        return False

    def _select_response(self,
                         ids,
                         scores,
                         tokenizer,
                         max_dec_len=None,
                         num_return_sequences=1,
                         keep_space=True):
        '''
        Select response with the highest score.
        '''
        ids = ids.numpy().tolist()
        scores = scores.numpy()

        if len(ids) != len(scores) or (len(ids) % num_return_sequences) != 0:
            raise ValueError(
                "the length of `ids` is {}, but the `num_return_sequences` is {}".
                format(len(ids), num_return_sequences))

        group = []
        tmp = []
        for pred, score in zip(ids, scores):
            pred_token_ids, pred_tokens = self._post_process_response(pred,
                                                                      tokenizer)
            num_token = len(pred_token_ids)
            if keep_space:
                response = " ".join(pred_tokens)
            else:
                response = "".join(pred_tokens)

            in_turn_repetition = self._get_in_turn_repetition(pred_tokens, True) \
                or self._get_in_turn_repetition(pred_token_ids)
            # not ending
            if max_dec_len is not None and num_token >= max_dec_len:
                score -= 1e3
            elif in_turn_repetition:
                score -= 1e3

            tmp.append([response, score])
            if len(tmp) == num_return_sequences:
                group.append(tmp)
                tmp = []

        results = []
        for preds in group:
            preds = sorted(preds, key=lambda x: -x[1])
            results.append(preds[0][0])
        return results

    def _postprocess(self, inputs):
        all_ids = inputs['ids']
        all_scores = inputs['scores']
        texts = inputs['text']

        results = []
        for ids, scores, text in zip(all_ids, all_scores, texts):
            results.extend(
                self._select_response(
                    ids,
                    scores,
                    self._tokenizer,
                    num_return_sequences=1,
                    keep_space=False))

        if self._interactive_mode:
            self.context.append(results[0].strip())
        return results
