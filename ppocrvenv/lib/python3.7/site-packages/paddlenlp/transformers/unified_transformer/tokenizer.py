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
"""Tokenization class for UnifiedTransformer model."""

import copy
import io
import json
import os
import six
import re
import unicodedata
from shutil import copyfile

import numpy as np
import jieba
import paddle
from paddle.utils import try_import

from .. import PretrainedTokenizer
from ..tokenizer_utils import convert_to_unicode, whitespace_tokenize, _is_whitespace, _is_control
from ...data.vocab import Vocab

__all__ = ['UnifiedTransformerTokenizer']


class UnifiedTransformerTokenizer(PretrainedTokenizer):
    """
    Constructs an UnifiedTransformer tokenizer based on `SentencePiece <https://github.com/google/sentencepiece>`__.

    This tokenizer inherits from :class:`~paddlenlp.transformers.tokenizer_utils.PretrainedTokenizer`
    which contains most of the main methods. For more information regarding those methods,
    please refer to this superclass.

    Args:
        vocab_file (str):
            The path of file to construct vocabulary.
        sentencepiece_model_file (str):
            The sentencepiece model file (ends with '.spm') required to instantiate a
            `SentencePiece <https://github.com/google/sentencepiece>`__.
        do_lower_case (bool, optional):
            Whether or not to lowercase the input when tokenizing. Defaults to 
            False and **does not** lowercase the input.
        unk_token (str, optional):
            A special token representing the *unknown (out-of-vocabulary)* token.
            An unknown token is set to be `unk_token` inorder to be converted 
            to an ID. Defaults to "[UNK]".
        pad_token (str, optional):
            A special token used to make arrays of tokens the same size for 
            batching purposes. Defaults to "[PAD]".
        cls_token (str, optional):
            A special token representing the beginning of a sequence. Defaults 
            to "[CLS]".
        sep_token (str, optional):
            A special token representing the end of a sequence or separating 
            two different sentences in the same input. Defaults to "[SEP]".
        mask_token (str, optional):
            A special token representing a masked token. Defaults to "[MASK]".
        special_tokens_file (str, optional):
            The path of file that contains additional special tokens to be used 
            by the tokenizer. Defaults to "".
    """

    resource_files_names = {
        "vocab_file": "vocab.txt",
        "sentencepiece_model_file": "spm.model",
    }  # for save_pretrained
    pretrained_resource_files_map = {
        "vocab_file": {
            "unified_transformer-12L-cn":
            "https://bj.bcebos.com/paddlenlp/models/transformers/unified_transformer/unified_transformer-12L-cn-vocab.txt",
            "unified_transformer-12L-cn-luge":
            "https://bj.bcebos.com/paddlenlp/models/transformers/unified_transformer/unified_transformer-12L-cn-vocab.txt",
            "plato-mini":
            "https://bj.bcebos.com/paddlenlp/models/transformers/unified_transformer/plato-mini-vocab.txt",
        },
        "sentencepiece_model_file": {
            "unified_transformer-12L-cn":
            "https://bj.bcebos.com/paddlenlp/models/transformers/unified_transformer/unified_transformer-12L-cn-spm.model",
            "unified_transformer-12L-cn-luge":
            "https://bj.bcebos.com/paddlenlp/models/transformers/unified_transformer/unified_transformer-12L-cn-spm.model",
            "plato-mini":
            "https://bj.bcebos.com/paddlenlp/models/transformers/unified_transformer/plato-mini-spm.model",
        },
    }
    pretrained_init_configuration = {
        "unified_transformer-12L-cn": {
            "do_lower_case": False
        },
        "unified_transformer-12L-cn-luge": {
            "do_lower_case": False
        },
        "plato-mini": {
            "do_lower_case": False
        },
    }

    TASK_TO_SPECIAL_TOKEN = {
        'chitchat': "[CHAT]",
        'knowledge': "[KNOW]",
        'recommend': "[RECO]",
    }

    def __init__(self,
                 vocab_file,
                 sentencepiece_model_file,
                 do_lower_case=False,
                 unk_token="[UNK]",
                 pad_token="[PAD]",
                 cls_token="[CLS]",
                 sep_token="[SEP]",
                 mask_token="[MASK]",
                 special_tokens_file="",
                 **kwargs):
        mod = try_import('sentencepiece')
        self.spm_model = mod.SentencePieceProcessor()

        self.do_lower_case = do_lower_case
        if not os.path.isfile(vocab_file):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'. To load the "
                "vocabulary from a pretrained model please use "
                "`tokenizer = ErnieTinyTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
                .format(vocab_file))
        self.vocab = self.load_vocabulary(
            vocab_file,
            unk_token,
            pad_token,
            cls_token,
            sep_token,
            mask_token=mask_token)

        # if the sentencepiece_model_file is not exists, just the default sentence-piece model 
        if os.path.isfile(sentencepiece_model_file):
            self.spm_model.Load(sentencepiece_model_file)

        pat_str = ""
        if os.path.isfile(special_tokens_file):
            self.specials = self.read_file(special_tokens_file)
            for special in self.specials:
                pat_str += "(" + re.escape(special) + ")|"
        else:
            self.specials = {}

        pat_str += r"([a-zA-Z0-9\S]+)"
        self.pat = re.compile(pat_str)

        self.vocab_file = vocab_file
        self.sentencepiece_model_file = sentencepiece_model_file

    @property
    def vocab_size(self):
        """
        Returns the size of vocabulary.

        Example:
            .. code-block::

                from paddlenlp.transformers import UnifiedTransformerTokenizer

                tokenizer = UnifiedTransformerTokenizer.from_pretrained('plato-mini')
                print(tokenizer.vocab_size)
                # 30001
        """
        return len(self.vocab)

    def preprocess_text(self,
                        inputs,
                        remove_space=True,
                        lower=False,
                        is_split_into_words=True):
        # preprocess data by removing extra space and normalize data.
        if not is_split_into_words:
            inputs = " ".join(jieba.lcut(inputs))
        outputs = inputs
        if remove_space:
            outputs = " ".join(inputs.strip().split())
        outputs = unicodedata.normalize("NFKD", outputs)
        outputs = "".join([c for c in outputs if not unicodedata.combining(c)])
        if lower:
            outputs = outputs.lower()
        return outputs

    def clean_text(self, text):
        # Performs invalid character removal and whitespace cleanup on text.
        text = text.replace(u"“", u'"')\
            .replace(u'”', u'"')\
            .replace(u'‘', "'")\
            .replace(u'’', u"'")\
            .replace(u'—', u'-')
        output = []
        for char in text:
            if _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def encode_pieces(self, spm_model, text, return_unicode=True, sample=False):
        # turn sentences into word pieces.
        # liujiaxiang: add for ernie-albert, mainly consider for “/”/‘/’/— causing too many unk
        text = self.clean_text(text)
        if not sample:
            pieces = spm_model.EncodeAsPieces(text)
        else:
            pieces = spm_model.SampleEncodeAsPieces(text, 64, 0.1)
        return pieces

    def _tokenize(self, text, is_split_into_words=True):
        """
        End-to-end tokenization for UnifiedTransformer models.

        Args:
            text (str): 
                The text to be tokenized.
        
        Returns:
            list: A list of string representing converted tokens.
        """
        text = self.preprocess_text(
            text,
            lower=self.do_lower_case,
            is_split_into_words=is_split_into_words)
        tokens = []
        for match in self.pat.finditer(text):
            part_text = match.group(0)
            if part_text in self.specials:
                tokens.append(part_text)
                continue
            part_tokens = self.encode_pieces(self.spm_model, part_text)
            tokens.extend(part_tokens)
        return tokens

    def merge_subword(self, tokens):
        # Merge subword.
        ret = []
        for token in tokens:
            if token.startswith(u"▁"):
                ret.append(token[1:])
            else:
                if len(ret):
                    ret[-1] += token
                else:
                    ret.append(token)

        ret = [token for token in ret if token]
        return ret

    def convert_tokens_to_string(self, tokens, keep_space=True):
        """
        Converts a sequence of tokens (list of string) in a single string. Since
        the usage of WordPiece introducing `__` to concat subwords, also remove
        `__` when converting.

        Args:
            tokens (list[str]): 
                A list of string representing tokens to be converted.
            keep_space (bool, optinal): 
                Whether or not to keep the segmentation with space. Defaults to 
                True.

        Returns:
            str: Converted string from tokens.

        Example:
            .. code-block::

                from paddlenlp.transformers import UnifiedTransformerTokenizer

                tokenizer = UnifiedTransformerTokenizer.from_pretrained('plato-mini')
                print(tokenizer.convert_tokens_to_string(['▁欢迎', '▁使用', '▁百度', '▁飞', '桨', '▁!']))
                # 欢迎 使用 百度 飞桨 !
                print(tokenizer.convert_tokens_to_string(['▁欢迎', '▁使用', '▁百度', '▁飞', '桨', '▁!'], keep_space=False))
                # 欢迎使用百度飞桨!
        """
        tokens = self.merge_subword(tokens)
        if keep_space:
            out_string = " ".join(tokens).replace("<s>", "")
        else:
            out_string = "".join(tokens).replace("<s>", "")
        out_string = out_string.replace("</s>", "\n").replace("\n ",
                                                              "\n").strip()
        return out_string

    def convert_ids_to_string(self, ids, keep_space=True):
        """
        Converts a single index or a sequence of indices to a token or a 
        sequence of tokens.

        Args:
            ids (int|list[int]):
                The token id (or token ids) to be converted to token(s).
            keep_space (bool, optional):
                Whether or not to keep the segmentation with space. Defaults to 
                True.

        Returns:
            str|list[str]: The decoded token(s).

        Example:
            .. code-block::

                from paddlenlp.transformers import UnifiedTransformerTokenizer

                tokenizer = UnifiedTransformerTokenizer.from_pretrained('plato-mini')
                tokens = tokenizer.tokenize('欢迎使用百度飞桨！', is_split_into_words=False)
                ids = tokenizer.convert_tokens_to_ids(tokens)
                print(ids)
                # [6, 121, 26907, 25475]

                print(tokenizer.convert_ids_to_string(ids))
                # 我 爱祖国
                print(tokenizer.convert_ids_to_string(ids, keep_space=False))
                # 我爱祖国
        """
        tokens = self.convert_ids_to_tokens(ids)
        out_string = self.convert_tokens_to_string(tokens, keep_space)
        return out_string

    def num_special_tokens_to_add(self, pair=False):
        token_ids_0 = []
        token_ids_1 = []
        return len(
            self.build_inputs_with_special_tokens(token_ids_0, token_ids_1
                                                  if pair else None))

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        _cls = [self.cls_token_id]
        _sep = [self.sep_token_id]
        if token_ids_1 is None:
            return _cls + token_ids_0 + _sep
        return _cls + token_ids_0 + _sep + token_ids_1 + _sep

    def build_offset_mapping_with_special_tokens(self,
                                                 offset_mapping_0,
                                                 offset_mapping_1=None):
        if offset_mapping_1 is None:
            return [(0, 0)] + offset_mapping_0 + [(0, 0)]

        return [(0, 0)] + offset_mapping_0 + [(0, 0)
                                              ] + offset_mapping_1 + [(0, 0)]

    def create_token_type_ids_from_sequences(self,
                                             token_ids_0,
                                             token_ids_1=None):
        _cls = [self.cls_token_id]
        _sep = [self.sep_token_id]
        if token_ids_1 is None:
            return [0] * len(_cls + token_ids_0 + _sep)
        return [0] * len(_cls + token_ids_0 + _sep) + [1] * len(token_ids_1 +
                                                                _sep)

    def get_special_tokens_mask(self,
                                token_ids_0,
                                token_ids_1=None,
                                already_has_special_tokens=False):
        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formatted with special tokens for the model."
                )
            return list(
                map(lambda x: 1 if x in [self.sep_token_id, self.cls_token_id] else 0,
                    token_ids_0))

        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + (
                [0] * len(token_ids_1)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1]

    def save_resources(self, save_directory):
        # Rewrite the :meth:`save_resources` method of superclass to save 
        # related resources under `save_directory`.
        for name, file_name in self.resource_files_names.items():
            src_path = getattr(self, name)
            save_path = os.path.join(save_directory, file_name)
            if os.path.abspath(src_path) != os.path.abspath(save_path):
                copyfile(src_path, save_path)

    @staticmethod
    def read_file(filepath):
        token_to_idx = {}
        with open(filepath, 'r', encoding='utf-8') as f:
            for num, line in enumerate(f):
                items = convert_to_unicode(line.rstrip()).split("\t")
                if len(items) > 2:
                    break
                token = items[0]
                index = int(items[1]) if len(items) == 2 else num
                token = token.strip()
                token_to_idx[token] = index
        return token_to_idx

    @staticmethod
    def load_vocabulary(filepath,
                        unk_token=None,
                        pad_token=None,
                        bos_token=None,
                        eos_token=None,
                        **kwargs):
        # Rewrite the :meth:`load_vocabulary` method of superclass to deal with 
        # the inconsistency of the vocabulary format.
        token_to_idx = UnifiedTransformerTokenizer.read_file(filepath)
        vocab = Vocab.from_dict(
            token_to_idx,
            unk_token=unk_token,
            pad_token=pad_token,
            bos_token=bos_token,
            eos_token=eos_token,
            **kwargs)
        # Filtered the tokens that are mapped to the same id
        idx_to_token = {v: k for k, v in vocab._token_to_idx.items()}
        vocab._idx_to_token = [
            idx_to_token[idx] for idx in sorted(idx_to_token.keys())
        ]
        return vocab

    def dialogue_encode(self,
                        history,
                        response=None,
                        knowledge=None,
                        task_type=None,
                        max_seq_len=512,
                        max_response_len=128,
                        max_knowledge_len=128,
                        return_position_ids=True,
                        return_token_type_ids=True,
                        return_attention_mask=True,
                        return_length=False,
                        add_start_token_as_response=False,
                        pad_to_max_seq_len=False,
                        return_tensors=False,
                        is_split_into_words=True):
        """
        Main method to encode the single-turn or multi-turn dialogue conversation. 
        It will return a dictionary containing the encoded sequence and other 
        relative informations which meets the input format requirements of the 
        UnifiedTransformer model. 
        See detail at 
        https://github.com/PaddlePaddle/Knover/tree/luge-dialogue/luge-dialogue

        Args:
            history (str|list|tuple): The history of dialogue conversation. It 
                is an utterance or list of utterances to be encoded. Each 
                utterance is a string. 
            response (str, optional): The response of dialogue conversation. 
                It should be set when training the model. It should not be set 
                when running inference. Defaults to None.
            knowledge (str, optional): The knowledge information of dialogue 
                conversation. It should be set if the `task_type` is "knowledge" 
                or "recommend". Defaults to None.
            task_type (str, optional): The type of dialogue conversation. It is 
                one of "chitchat", "knowledge" and "recommend". They represent 
                the chitchat dialogue, knowledge grounded dialogue and 
                conversational recommendation respectively. Defaults to None, 
                which means there is no `special_token` added in output sequence 
                for identifying different conversation types.
            max_seq_len (int, optional): The maximum encoded sequence length.
                Defaults to 512.
            max_response_len (int, optional): The maximum encoded sequence 
                length of the input `response`. Defaults to 128.
            max_knowledge_len (int, optional): The maximum encoded sequence 
                length of the input `knowledge`. Defaults to 128.
            return_position_ids (bool, optional): Whether to return the 
                position_ids. Defaults to True.
            return_token_type_ids (bool, optional): Whether to return the 
                token_type_ids. Defaults to True.
            return_attention_mask (bool, optional): Whether to return the 
                attention_mask. Defaults to True.
            return_length (bool, optional): Whether to return the length of the
                encoded sequence. Defaults to False.
            add_start_token_as_response (bool, optional): Whether to add the 
                special token "[CLS]" at the end of sequence as the begining of 
                the response when running inference to force the model to start 
                generating response sequence. Defaults to False.
            pad_to_max_seq_len (bool, optional): Whether to pad the returned 
                sequences to the `max_seq_len`. Note that, in this method, 
                returned sequences will be padded on the left. Defaults to False.
            return_tensors (bool, optional): Whether to convert the returned 
                sequences to Tensor. Defaults to False.
            is_split_into_words(bool, optinal): Whether or not the input text 
                (`history`, `response` and `knowledge`) has been pretokenized. 
                Defaults to True.

        Returns: 
            dict: A dictionary containing the encoded sequence and other 
            relative informations.

            With the corresponding fields:

            - input_ids (list[int]|Tensor):
                A list of indices of input tokens to be feed to UnifiedTransformer 
                model. If `return_tensors` is True, it is a Tensor with shape 
                [1, sequence_length] and data type 'int64'.
            - token_type_ids (list[int]|Tensor, optional):
                A list of segment token indices to indicate whether the token 
                belongs to the dialogue response. If `return_tensors` is True, 
                it is a Tensor with shape [1, sequence_length] and data type 
                'int64'. 
                Being returned when `return_token_type_ids` is set to True.
            - position_ids (list[int]|Tensor, optional):
                A list of The position indices. If `return_tensors` is True, 
                it is a Tensor with shape [1, sequence_length] and data type 
                'int64'.
                Being returned when `return_position_ids` is set to True.
            - attention_mask (numpy.ndarray|Tensor, optional):
                A numpy.ndarray to prevents attention to some unwanted positions, 
                with shape [sequence_length, sequence_length] and data type 
                'float32'. If `return_tensors` is True, it is a Tensor with shape 
                [1, 1, sequence_length, sequence_length] and data type 'float32'.
                Being returned when `return_attention_mask` is set to True.
            - seq_len (int, optional):
                The actual length of the `input_ids`, excluding the pad token. 
                Being returned when `return_length` is set to True.

        Example:
            .. code-block::

                from paddlenlp.transformers import UnifiedTransformerTokenizer

                tokenizer = UnifiedTransformerTokenizer.from_pretrained('plato-mini')

                inputs = tokenizer.dialogue_encode('我爱祖国')
                for key in inputs:
                    print(key + ':')
                    print(inputs[key])
                # input_ids: [1, 6, 25445, 26907, 25475, 2]
                # token_type_ids: [0, 0, 0, 0, 0, 0]
                # position_ids: [0, 1, 2, 3, 4, 5]
                # attention_mask: [[0. 0. 0. 0. 0. 0.]
                # [0. 0. 0. 0. 0. 0.]
                # [0. 0. 0. 0. 0. 0.]
                # [0. 0. 0. 0. 0. 0.]
                # [0. 0. 0. 0. 0. 0.]
                # [0. 0. 0. 0. 0. 0.]]
        """

        # Input type checking for clearer error
        assert isinstance(history, str) or (
            isinstance(history, (list, tuple)) and
            (len(history) == 0 or len(history) != 0 and
             isinstance(history[0], str))), (
                 "The input `history` must be with type `str` (single context) "
                 "or `List[str]` (multi-turn context). But received: {}".format(
                     history))
        assert response is None or isinstance(response, str), (
            "The input `response` must of be with type `str`. But received: {}".
            format(response))
        assert knowledge is None or isinstance(knowledge, str), (
            "The input `knowledge` must of be with type `str`. But received: {}".
            format(knowledge))
        assert task_type is None or task_type in self.TASK_TO_SPECIAL_TOKEN, (
            "The input `task_type` must be None or one of {}.".format(", ".join(
                self.TASK_TO_SPECIAL_TOKEN.keys())))
        assert max_seq_len > max_response_len + max_knowledge_len, (
            "`max_seq_len` must be greater than the sum of `max_response_len` "
            "and `max_knowledge_len`. But received `max_seq_len` is {}, "
            "`max_response_len` is {}, `max_knowledge_len` is {}.".format(
                max_seq_len, max_response_len, max_knowledge_len))
        assert response is None or not add_start_token_as_response, (
            "`add_start_token_as_response` only works when `response` is "
            "`None`. But received `add_start_token_as_response`: `{}`, "
            "`response`: {}.".format(add_start_token_as_response, response))

        knowledge_ids = []
        if knowledge is not None:
            tokens = self._tokenize(knowledge, is_split_into_words)
            knowledge_ids = self.convert_tokens_to_ids(tokens)
            if len(knowledge_ids) > max_knowledge_len - 1:
                knowledge_ids = knowledge_ids[:max_knowledge_len - 1]
            knowledge_ids += [self.sep_token_id]

        response_ids = []
        if response is not None:
            tokens = self._tokenize(response, is_split_into_words)
            response_ids = [self.cls_token_id] + self.convert_tokens_to_ids(
                tokens)
            if len(response_ids) > max_response_len - 1:
                response_ids = response_ids[:max_response_len - 1]
            response_ids += [self.sep_token_id]
        elif add_start_token_as_response:
            response_ids = [self.cls_token_id]

        if task_type is not None:
            special_token = self.TASK_TO_SPECIAL_TOKEN[task_type]
            assert special_token in self.vocab._token_to_idx, (
                "The vocab file should contain the special token corresponding "
                "to the task: {}.".format(task_type))
            special_token_id = self.vocab._token_to_idx[special_token]
            knowledge_ids = [self.cls_token_id, special_token_id
                             ] + knowledge_ids
        else:
            knowledge_ids = [self.cls_token_id] + knowledge_ids

        max_history_len = max_seq_len - len(knowledge_ids) - len(response_ids)
        if isinstance(history, str):
            history = [history]
        history_ids = []
        for i in range(len(history) - 1, -1, -1):
            tokens = self._tokenize(history[i], is_split_into_words)
            if len(history_ids) + len(tokens) + 1 > max_history_len:
                if i == len(history) - 1:
                    tokens = tokens[1 - max_history_len:]
                    history_ids = (self.convert_tokens_to_ids(tokens) +
                                   [self.sep_token_id])
                break
            history_ids = (self.convert_tokens_to_ids(tokens) +
                           [self.sep_token_id]) + history_ids

        history_ids = knowledge_ids + history_ids
        # Build output dictionnary
        encoded_inputs = {}
        encoded_inputs["input_ids"] = history_ids + response_ids
        # Check lengths
        sequence_length = len(encoded_inputs["input_ids"])
        assert sequence_length <= max_seq_len

        # Considering that the logits at the last time step in the API of 
        # generative task are taken to generate the next token. In order to 
        # avoid the last time step being a pad, so take padding on the left.
        pad_length = max_seq_len - sequence_length if pad_to_max_seq_len else 0
        if pad_length > 0:
            encoded_inputs["input_ids"] = [
                self.pad_token_id
            ] * pad_length + encoded_inputs["input_ids"]
        if return_tensors:
            # Add dimention for batch_size
            encoded_inputs["input_ids"] = paddle.to_tensor(encoded_inputs[
                "input_ids"]).unsqueeze(0)

        if return_token_type_ids:
            encoded_inputs["token_type_ids"] = [0] * len(
                history_ids) + [1] * len(response_ids)
            if pad_length > 0:
                encoded_inputs["token_type_ids"] = [
                    self.pad_token_id
                ] * pad_length + encoded_inputs["token_type_ids"]
            if return_tensors:
                # Add dimention for batch_size
                encoded_inputs["token_type_ids"] = paddle.to_tensor(
                    encoded_inputs["token_type_ids"]).unsqueeze(0)

        if return_length:
            encoded_inputs["seq_len"] = sequence_length

        if return_position_ids:
            encoded_inputs["position_ids"] = list(range(sequence_length))
            if pad_length > 0:
                encoded_inputs["position_ids"] = [
                    self.pad_token_id
                ] * pad_length + encoded_inputs["position_ids"]
            if return_tensors:
                # Add dimention for batch_size
                encoded_inputs["position_ids"] = paddle.to_tensor(
                    encoded_inputs["position_ids"]).unsqueeze(0)

        if return_attention_mask:
            attention_mask = np.ones(
                (sequence_length, sequence_length), dtype='float32') * -1e4
            start = len(history_ids)
            end = sequence_length
            attention_mask[:end, :start] = 0.0
            # Generate the lower triangular matrix using the slice of matrix
            tmp = np.triu(
                np.ones(
                    [end - start, end - start], dtype='float32') * -1e4, 1)
            attention_mask[start:end, start:end] = tmp
            encoded_inputs["attention_mask"] = attention_mask
            if pad_length > 0:
                new_mask = np.ones(
                    (max_seq_len, max_seq_len), dtype='float32') * -1e4
                new_mask[-sequence_length:, -sequence_length:] = attention_mask
                encoded_inputs["attention_mask"] = new_mask
            if return_tensors:
                # Add dimentions for batch_size and num_heads
                encoded_inputs["attention_mask"] = paddle.to_tensor(
                    encoded_inputs["attention_mask"]).unsqueeze((0, 1))

        return encoded_inputs
