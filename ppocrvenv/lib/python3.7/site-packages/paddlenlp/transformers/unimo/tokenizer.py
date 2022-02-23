# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.

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
import pickle
import six
import shutil
import paddle
from paddle.utils import try_import
from paddlenlp.utils.env import MODEL_HOME
import numpy as np
from ...data.vocab import Vocab

from .. import BasicTokenizer, PretrainedTokenizer, WordpieceTokenizer

__all__ = ['UNIMOTokenizer']


class UNIMOTokenizer(PretrainedTokenizer):
    r"""
    Constructs an UNIMO tokenizer. It uses a basic tokenizer to do punctuation
    splitting, lower casing and so on, and follows a WordPiece tokenizer to
    tokenize as subwords.

    This tokenizer inherits from :class:`~paddlenlp.transformers.tokenizer_utils.PretrainedTokenizer`
    which contains most of the main methods. For more information regarding those methods,
    please refer to this superclass.

    Args:
        vocab_file (str): 
            The vocabulary file path (ends with '.txt') required to instantiate
            a `WordpieceTokenizer`.
        do_lower_case (str, optional):
            Whether or not to lowercase the input when tokenizing.
            Defaults to`True`.
        unk_token (str):
            A special token representing the *unknown (out-of-vocabulary)* token.
            An unknown token is set to be `unk_token` inorder to be converted to an ID.
            Defaults to "[UNK]".
        sep_token (str):
            A special token separating two different sentences in the same input.
            Defaults to "[SEP]".
        pad_token (str):
            A special token used to make arrays of tokens the same size for batching purposes.
            Defaults to "[PAD]".
        cls_token (str):
            A special token used for sequence classification. It is the last token
            of the sequence when built with special tokens. Defaults to "[CLS]".
        mask_token (str):
            A special token representing a masked token. This is the token used
            in the masked language modeling task which the model tries to predict the original unmasked ones.
            Defaults to "[MASK]".

    Examples:
        .. code-block::

            from paddlenlp.transformers import UNIMOTokenizer
            tokenizer = UNIMOTokenizer.from_pretrained('unimo-text-1.0')
            encoded_inputs = tokenizer('He was a puppeteer')
            # encoded_inputs
            #{
            #   'input_ids': [1, 4444, 4385, 1545, 6712, 10062, 9568, 9756, 9500, 2],
            #   'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            #}

    """
    resource_files_names = {"vocab_file": "vocab.txt"}  # for save_pretrained
    pretrained_resource_files_map = {
        "vocab_file": {
            "unimo-text-1.0":
            "https://bj.bcebos.com/paddlenlp/models/transformers/unimo/unimo-text-1.0-vocab.txt",
            "unimo-text-1.0-lcsts-new":
            "https://bj.bcebos.com/paddlenlp/models/transformers/unimo/unimo-text-1.0-vocab.txt",
            "unimo-text-1.0-large":
            "https://bj.bcebos.com/paddlenlp/models/transformers/unimo/unimo-text-1.0-large-vocab.txt",
        }
    }
    pretrained_init_configuration = {
        "unimo-text-1.0": {
            "do_lower_case": True
        },
        "unimo-text-1.0-lcsts-new": {
            "do_lower_case": True
        },
        "unimo-text-1.0-large": {
            "do_lower_case": True
        },
    }

    def __init__(self,
                 vocab_file,
                 do_lower_case=True,
                 unk_token="[UNK]",
                 sep_token="[SEP]",
                 pad_token="[PAD]",
                 cls_token="[CLS]",
                 mask_token="[MASK]",
                 **kwargs):

        if not os.path.isfile(vocab_file):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'. To load the "
                "vocabulary from a pretrained model please use "
                "`tokenizer = UNIMOTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
                .format(vocab_file))
        self.vocab = self.load_vocabulary(vocab_file, unk_token=unk_token)
        self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
        self.wordpiece_tokenizer = WordpieceTokenizer(
            vocab=self.vocab, unk_token=unk_token)

    @property
    def vocab_size(self):
        """
        Return the size of vocabulary.

        Returns:
            int: The size of vocabulary.
        """
        return len(self.vocab)

    @staticmethod
    def load_vocabulary(filepath,
                        unk_token=None,
                        pad_token=None,
                        bos_token=None,
                        eos_token=None,
                        **kwargs):
        token_to_idx = {}
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                token, index = line.rstrip('\n').split('\t')
                token_to_idx[token] = int(index)
        vocab = Vocab.from_dict(
            token_to_idx,
            unk_token=unk_token,
            pad_token=pad_token,
            bos_token=bos_token,
            eos_token=eos_token,
            **kwargs)
        return vocab

    def _tokenize(self, text):
        r"""
        End-to-end tokenization for UNIMO models.

        Args:
            text (str): The text to be tokenized.

        Returns:
            List[str]: A list of string representing converted tokens.
        """
        split_tokens = []
        for token in self.basic_tokenizer.tokenize(text):
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)
        return split_tokens

    def convert_tokens_to_string(self, tokens):
        r"""
        Converts a sequence of tokens (list of string) in a single string. Since
        the usage of WordPiece introducing `##` to concat subwords, also remove
        `##` when converting.

        Args:
            tokens (list): A list of string representing tokens to be converted.

        Returns:
            str: Converted string from tokens.

        Examples:
            .. code-block::

                from paddlenlp.transformers import UNIMOTokenizer

                tokenizer = UNIMOTokenizer.from_pretrained('unimo-text-1.0')
                tokens = tokenizer.tokenize('He was a puppeteer')

                strings = tokenizer.convert_tokens_to_string(tokens)
                '''
                he was a puppeteer
                '''

        """
        out_string = " ".join(tokens).replace(" ##", "").strip()
        return out_string

    def num_special_tokens_to_add(self, pair=False):
        r"""
        Returns the number of added tokens when encoding a sequence with special tokens.

        Args:
            pair(bool):
                Whether the input is a sequence pair or a single sequence.
                Defaults to `False` and the input is a single sequence.

        Returns:
            int: Number of tokens added to sequences.
        """
        token_ids_0 = []
        token_ids_1 = []
        return len(
            self.build_inputs_with_special_tokens(token_ids_0, token_ids_1
                                                  if pair else None))

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        r"""
        Build model inputs from a sequence or a pair of sequence for sequence
        classification tasks by concatenating and adding special tokens.

        A UNIMO sequence has the following format:

        - single sequence:      ``[CLS] X [SEP]``
        - pair of sequences:        ``[CLS] A [SEP] B [SEP]``

        Args:
            token_ids_0 (List[int]):
                List of IDs to which the special tokens will be added.
            token_ids_1 (List[int], optional):
                Optional second list of IDs for sequence pairs.
                Defaults to `None`.

        Returns:
            List[int]: List of input_id with the appropriate special tokens.
        """
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        _cls = [self.cls_token_id]
        _sep = [self.sep_token_id]
        return _cls + token_ids_0 + _sep + token_ids_1 + _sep

    def merge_subword(self, tokens):
        r"""
        Converts the subwords in a sequence of tokens (list of string) to whole
        words, also remove `##` when converting.

        Args:
            tokens (List[str]): A list of string representing tokens to be converted.

        Returns:
            List[str]: Converted sequence of whole words.
        """
        ret = []
        for token in tokens:
            if token.startswith("##"):
                real_token = token[2:]
                if len(ret):
                    ret[-1] += real_token
                else:
                    ret.append(real_token)
            else:
                ret.append(token)

        return ret

    def build_offset_mapping_with_special_tokens(self,
                                                 offset_mapping_0,
                                                 offset_mapping_1=None):
        r"""
        Build offset map from a pair of offset map by concatenating and adding
        offsets of special tokens.

        A UNIMO offset_mapping has the following format:
        ::
            - single sequence: ``(0,0) X (0,0)``
            - pair of sequences: `(0,0) A (0,0) B (0,0)``

        Args:
            offset_mapping_ids_0 (List[tuple]):
                List of char offsets to which the special tokens will be added.
            offset_mapping_ids_1 (List[tuple], optional):
                Optional second list of char offsets for offset mapping pairs.
                Defaults to `None`.

        Returns:
            List[tuple]: List of char offsets with the appropriate offsets
                of special tokens.
        """
        if offset_mapping_1 is None:
            return [(0, 0)] + offset_mapping_0 + [(0, 0)]

        return [(0, 0)] + offset_mapping_0 + [(0, 0)
                                              ] + offset_mapping_1 + [(0, 0)]

    def create_token_type_ids_from_sequences(self,
                                             token_ids_0,
                                             token_ids_1=None):
        r"""
        Create a mask from the two sequences passed to be used in a sequence-pair
        classification task.

        A UNIMO sequence pair mask has the following format:
        ::

            0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
            | first sequence    | second sequence |

        If `token_ids_1` is `None`, this method only returns the first portion
        of the mask (0s).

        Args:
            token_ids_0 (List[int]):
                List of IDs.
            token_ids_1 (List[int], optional):
                Optional second list of IDs for sequence pairs.
                Defaults to `None`.

        Returns:
            List[int]: List of token_type_id according to the given sequence(s).
        """
        _sep = [self.sep_token_id]
        _cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(_cls + token_ids_0 + _sep) * [0]
        return len(_cls + token_ids_0 + _sep) * [0] + len(token_ids_1 +
                                                          _sep) * [1]

    def gen_encode(self,
                   source,
                   title=None,
                   target=None,
                   max_seq_len=512,
                   max_title_len=128,
                   max_target_len=128,
                   return_position_ids=True,
                   return_token_type_ids=True,
                   return_attention_mask=True,
                   return_length=False,
                   add_start_token_for_decoding=False,
                   pad_to_max_seq_len=False,
                   return_tensors=False,
                   is_split_into_words=False,
                   continuous_position=False):
        """
        Main method for encoding the source for generation. It will return a
        dictionary containing the encoded sequence and other relative informations
        which meets the input format requirements of the UNIMO-text model.

        Args:
            source (str): The source text of generation. It should be a string.
            target (str, optional): The target text of generation. It should be
                set when training the model and should be None when running
                inference. Defaults to None.
            title (str, optional): The additional information of some of the
                generation tasks such as summary. Defaults to None.
            max_seq_len (int, optional): The maximum encoded sequence length.
                Defaults to 512.
            max_target_len (int, optional): The maximum encoded sequence
                length of the input `target`. Defaults to 128.
            max_title_len (int, optional): The maximum encoded sequence
                length of the input `title`. Defaults to 128.
            return_position_ids (bool, optional): Whether to return the
                position_ids. Defaults to True.
            return_token_type_ids (bool, optional): Whether to return the
                token_type_ids. Defaults to True.
            return_attention_mask (bool, optional): Whether to return the
                attention_mask. Defaults to True.
            return_length (bool, optional): Whether to return the length of the
                encoded sequence. Defaults to False.
            add_start_token_for_decoding (bool, optional): Whether to add the
                special token "[CLS]" at the end of sequence as the begining of
                the target when running inference to force the model to start
                generating target sequence. Defaults to False.
            pad_to_max_seq_len (bool, optional): Whether to pad the returned
                sequences to the `max_seq_len`. Note that, in this method,
                returned sequences will be padded on the left. Defaults to False.
            return_tensors (bool, optional): Whether to convert the returned
                sequences to Tensor. Defaults to False.
            is_split_into_words(bool, optinal): Whether or not the input text
                (`source`, `target` and `title`) has been pretokenized.
                Defaults to False.
            continuous_position(bool, optinal): Whether the position ids is
                continuous between source ids and target ids. Defaults to False.

        Returns:
            dict: A dictionary containing the encoded sequence and other
            relative informations.

            With the corresponding fields:

            - input_ids (list[int]|Tensor):
                A list of indices of input tokens to be feed to UNIMO-text
                model. If `return_tensors` is True, it is a Tensor with shape
                [1, sequence_length] and data type 'int64'.
            - token_type_ids (list[int]|Tensor, optional):
                A list of segment token indices to indicate whether the token
                belongs to the dialogue target. If `return_tensors` is True,
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

                from paddlenlp.transformers import UNIMOTokenizer
                tokenizer = UNIMOTokenizer.from_pretrained('unimo-text-1.0')
                inputs = tokenizer.gen_encode('He was a puppeteer')
                #{'input_ids': [1, 4444, 4385, 1545, 6712, 10062, 9568, 9756, 9500, 2],
                #'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                #'position_ids': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                #'attention_mask': array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                #[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                #[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                #[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                #[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                #[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                #[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                #[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                #[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                #[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]], dtype=float32)}
        """

        # Input type checking for clearer error
        assert isinstance(source, str), (
            "The input `source` must be with type `str` (single context). "
            " But received: {}".format(source))
        assert target is None or isinstance(target, str), (
            "The input `target` must of be with type `str`. But received: {}".
            format(target))
        assert title is None or isinstance(title, str), (
            "The input `title` must of be with type `str`. But received: {}".
            format(title))
        assert max_seq_len > max_title_len + max_target_len, (
            "`max_seq_len` must be greater than the sum of `max_target_len` "
            "and `max_title_len`. But received `max_seq_len` is {}, "
            "`max_target_len` is {}, `max_title_len` is {}.".format(
                max_seq_len, max_title_len, max_target_len))
        assert target is None or not add_start_token_for_decoding, (
            "`add_start_token_for_decoding` only works when `target` is "
            "`None`. But received `add_start_token_for_decoding`: `{}`, "
            "`target`: {}.".format(add_start_token_for_decoding, target))

        title_ids = []
        if title is not None:
            tokens = self._tokenize(title)
            title_ids = self.convert_tokens_to_ids(tokens)
            if len(title_ids) > max_title_len - 1:
                title_ids = title_ids[:max_title_len - 1]
            title_ids += [self.sep_token_id]

        target_ids = []
        if target is not None:
            tokens = self._tokenize(target)
            target_ids = [self.cls_token_id] + self.convert_tokens_to_ids(
                tokens)
            if len(target_ids) > max_target_len - 1:
                target_ids = target_ids[:max_target_len - 1]
            target_ids += [self.mask_token_id]
        elif add_start_token_for_decoding:
            target_ids = [self.cls_token_id]

        title_ids = [self.cls_token_id] + title_ids

        max_source_len = max_seq_len - len(title_ids) - len(target_ids)
        source_ids = []
        tokens = self._tokenize(source)
        source_ids = self.convert_tokens_to_ids(tokens)

        if len(source_ids) > max_source_len - 1:
            source_ids = source_ids[:max_source_len - 1]

        source_ids += [self.sep_token_id]
        source_ids = title_ids + source_ids
        # Build output dictionnary

        encoded_inputs = {}
        encoded_inputs["input_ids"] = source_ids + target_ids
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
                source_ids) + [1] * len(target_ids)
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
            if continuous_position:
                encoded_inputs["position_ids"] = list(range(sequence_length))
            else:
                encoded_inputs["position_ids"] = list(range(len(
                    source_ids))) + list(range(len(target_ids)))
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
            start = len(source_ids)
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
