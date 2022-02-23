#encoding=utf8
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# MIT License

# Copyright (c) 2021 ShannonAI

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from functools import lru_cache

from paddle.utils import try_import

from paddlenlp.transformers import BertTokenizer


class ChineseBertTokenizer(BertTokenizer):
    """
    Construct a ChineseBert tokenizer. `ChineseBertTokenizer` is similar to `BertTokenizerr`.
    The difference between them is that ChineseBert has the extra process about pinyin id.
    For more information regarding those methods, please refer to this superclass.

    Args:
        vocab_file (str):
            The vocabulary file path (ends with '.txt') required to instantiate
            a `WordpieceTokenizer`.
        do_lower_case (bool):
            Whether or not to lowercase the input when tokenizing.
            Defaults to `True`.
        pinyin_map (dict):
            A dict of pinyin map, the map between pinyin char and id. pinyin char is 26 Romanian characters and 0-5 numbers.
            Defaults to None.
        id2pinyin (dict):
            A dict of char id map tensor.
            Defaults to None.
        pinyin2tensor (dict):
            A dict of pinyin map tensor.
            Defaults to None.
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

            from paddlenlp.transformers import ChineseBertTokenizer
            tokenizer = ChineseBertTokenizer.from_pretrained('ChineseBERT-base')

            inputs = tokenizer('欢迎使用飞桨！')
            print(inputs)

            '''
            {'input_ids': [101, 3614, 6816, 886, 4500, 7607, 3444, 8013, 102], 
            'pinyin_ids': [0, 0, 0, 0, 0, 0, 0, 0, 13, 26, 6, 19, 1, 0, 0, 0, 30, 14, 19, 12, 2, 0, 0, 0, 24, 13, 14, 3, 0, 0, 0, 0, 30, 20, 19, 12, 4, 0, 0, 0, 11, 10, 14, 1, 0, 0, 0, 0, 15, 14, 6, 19, 12, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
            'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0]}
            '''

    """

    pretrained_resource_files_map = {
        "vocab_file": {
            "ChineseBERT-base":
            "https://bj.bcebos.com/paddlenlp/models/transformers/chinese_bert/chinesebert-base/vocab.txt",
            "ChineseBERT-large":
            "https://bj.bcebos.com/paddlenlp/models/transformers/chinese_bert/chinesebert-base/vocab.txt",
        },
        "tokenizer_config_file": {
            "ChineseBERT-base":
            "https://bj.bcebos.com/paddlenlp/models/transformers/chinese_bert/chinesebert-large/tokenizer_config.json",
            "ChineseBERT-large":
            "https://bj.bcebos.com/paddlenlp/models/transformers/chinese_bert/chinesebert-large/tokenizer_config.json",
        },
    }
    pretrained_init_configuration = {
        "ChineseBERT-base": {
            "do_lower_case": True
        },
        "ChineseBERT-large": {
            "do_lower_case": True
        },
    }
    padding_side = "right"

    def __init__(
            self,
            vocab_file,
            do_lower_case=True,
            pinyin_map=None,
            id2pinyin=None,
            pinyin2tensor=None,
            unk_token="[UNK]",
            sep_token="[SEP]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            mask_token="[MASK]", ):
        super().__init__(
            vocab_file,
            do_lower_case,
            unk_token,
            sep_token,
            pad_token,
            cls_token,
            mask_token, )
        self.pinyin_dict = pinyin_map
        self.id2pinyin = id2pinyin
        self.pinyin2tensor = pinyin2tensor
        self.special_tokens_pinyin_ids = [0] * 8

    def encode(self,
               text,
               text_pair=None,
               max_seq_len=512,
               pad_to_max_seq_len=False,
               truncation_strategy="longest_first",
               return_position_ids=False,
               return_token_type_ids=True,
               return_attention_mask=False,
               return_length=False,
               return_overflowing_tokens=False,
               return_special_tokens_mask=False):
        """
        Performs tokenization and uses the tokenized tokens to prepare model
        inputs. It supports sequence or sequence pair as input, and batch input
        is not allowed.

        Args:
            text (str, List[str] or List[int]):
                The sequence to be processed. One sequence is a string, a list
                of strings, or a list of integers depending on whether it has
                been pretokenized and converted to ids. 
            text_pair (str, List[str] or List[List[str]]):
                Same as `text` argument, while it represents for the latter
                sequence of the sequence pair.
            max_seq_len (int, optional):
                If set to a number, will limit the total sequence returned so
                that it has a maximum length. If there are overflowing tokens,
                those overflowing tokens will be added to the returned dictionary
                when `return_overflowing_tokens` is `True`. Defaults to `None`.
            stride (int, optional):
                Only available for batch input of sequence pair and mainly for
                question answering usage. When for QA, `text` represents questions
                and `text_pair` represents contexts. If `stride` is set to a
                positive number, the context will be split into multiple spans
                where `stride` defines the number of (tokenized) tokens to skip
                from the start of one span to get the next span, thus will produce
                a bigger batch than inputs to include all spans. Moreover, 'overflow_to_sample'
                and 'offset_mapping' preserving the original example and position
                information will be added to the returned dictionary. Defaults to 0.
            pad_to_max_seq_len (bool, optional):
                If set to `True`, the returned sequences would be padded up to
                `max_seq_len` specified length according to padding side
                (`self.padding_side`) and padding token id. Defaults to `False`.
            truncation_strategy (str, optional):
                String selected in the following options:

                - 'longest_first' (default) Iteratively reduce the inputs sequence
                until the input is under `max_seq_len` starting from the longest
                one at each token (when there is a pair of input sequences).
                - 'only_first': Only truncate the first sequence.
                - 'only_second': Only truncate the second sequence.
                - 'do_not_truncate': Do not truncate (raise an error if the input
                sequence is longer than `max_seq_len`).

                Defaults to 'longest_first'.
            return_position_ids (bool, optional):
                Whether to include tokens position ids in the returned dictionary.
                Defaults to `False`.
            return_token_type_ids (bool, optional):
                Whether to include token type ids in the returned dictionary.
                Defaults to `True`.
            return_attention_mask (bool, optional):
                Whether to include the attention mask in the returned dictionary.
                Defaults to `False`.
            return_length (bool, optional):
                Whether to include the length of each encoded inputs in the
                returned dictionary. Defaults to `False`.
            return_overflowing_tokens (bool, optional):
                Whether to include overflowing token information in the returned
                dictionary. Defaults to `False`.
            return_special_tokens_mask (bool, optional):
                Whether to include special tokens mask information in the returned
                dictionary. Defaults to `False`.

        Returns:
            dict:
                The dict has the following optional items:

                - **input_ids** (list[int]): List of token ids to be fed to a model.
                - **pinyin_ids** (list[int]): List of pinyin ids to be fed to a model.
                - **position_ids** (list[int], optional): List of token position ids to be
                  fed to a model. Included when `return_position_ids` is `True`
                - **token_type_ids** (list[int], optional): List of token type ids to be
                  fed to a model. Included when `return_token_type_ids` is `True`.
                - **attention_mask** (list[int], optional): List of integers valued 0 or 1,
                  where 0 specifies paddings and should not be attended to by the
                  model. Included when `return_attention_mask` is `True`.
                - **seq_len** (int, optional): The input_ids length. Included when `return_length`
                  is `True`.
                - **overflowing_tokens** (list[int], optional): List of overflowing tokens.
                  Included when if `max_seq_len` is specified and `return_overflowing_tokens`
                  is True.
                - **num_truncated_tokens** (int, optional): The number of overflowing tokens.
                  Included when if `max_seq_len` is specified and `return_overflowing_tokens`
                  is True.
                - **special_tokens_mask** (list[int], optional): List of integers valued 0 or 1,
                  with 0 specifying special added tokens and 1 specifying sequence tokens.
                  Included when `return_special_tokens_mask` is `True`.
        """

        def get_input_ids(text):
            if isinstance(text, str):
                tokens = self.tokenize(text)
                return self.convert_tokens_to_ids(tokens)
            elif isinstance(text,
                            (list, tuple)) and len(text) > 0 and isinstance(
                                text[0], str):
                return self.convert_tokens_to_ids(text)
            elif isinstance(text,
                            (list, tuple)) and len(text) > 0 and isinstance(
                                text[0], int):
                return text
            else:
                raise ValueError(
                    "Input is not valid. Should be a string, a list/tuple of strings or a list/tuple of integers."
                )

        ids = get_input_ids(text)
        pair_ids = get_input_ids(text_pair) if text_pair is not None else None

        pair = bool(pair_ids is not None)
        len_ids = len(ids)
        len_pair_ids = len(pair_ids) if pair else 0

        encoded_inputs = {}

        # Truncation: Handle max sequence length
        total_len = len_ids + len_pair_ids + (self.num_special_tokens_to_add(
            pair=pair))

        token_offset_mapping = self.get_offset_mapping(text)

        if pair:
            token_pair_offset_mapping = self.get_offset_mapping(text_pair)
        else:
            token_pair_offset_mapping = None

        if max_seq_len and total_len > max_seq_len:
            ids, pair_ids, token_offset_mapping, token_pair_offset_mapping, overflowing_tokens = self.truncate_sequences(
                ids,
                pair_ids=pair_ids,
                token_offset_mapping=token_offset_mapping,
                token_pair_offset_mapping=token_pair_offset_mapping,
                num_tokens_to_remove=total_len - max_seq_len,
                truncation_strategy=truncation_strategy, )

            if return_overflowing_tokens:
                encoded_inputs["overflowing_tokens"] = overflowing_tokens
                encoded_inputs["num_truncated_tokens"] = total_len - max_seq_len

        # Add special tokens

        sequence = self.build_inputs_with_special_tokens(ids, pair_ids)
        token_type_ids = self.create_token_type_ids_from_sequences(ids,
                                                                   pair_ids)

        offset_mapping = self.build_offset_mapping_with_special_tokens(
            token_offset_mapping, token_pair_offset_mapping)

        # Build output dictionnary
        encoded_inputs["input_ids"] = sequence
        encoded_inputs["pinyin_ids"] = self.get_pinyin_ids(text, text_pair,
                                                           offset_mapping)

        if return_token_type_ids:
            encoded_inputs["token_type_ids"] = token_type_ids
        if return_special_tokens_mask:
            encoded_inputs[
                "special_tokens_mask"] = self.get_special_tokens_mask(ids,
                                                                      pair_ids)
        if return_length:
            encoded_inputs["seq_len"] = len(encoded_inputs["input_ids"])

        # Check lengths
        assert max_seq_len is None or len(encoded_inputs[
            "input_ids"]) <= max_seq_len

        # Padding
        needs_to_be_padded = pad_to_max_seq_len and \
                             max_seq_len and len(encoded_inputs["input_ids"]) < max_seq_len

        if needs_to_be_padded:
            difference = max_seq_len - len(encoded_inputs["input_ids"])
            encoded_inputs["pinyin_ids"] = encoded_inputs[
                "pinyin_ids"] + self.special_tokens_pinyin_ids * difference
            if self.padding_side == 'right':
                if return_attention_mask:
                    encoded_inputs["attention_mask"] = [1] * len(encoded_inputs[
                        "input_ids"]) + [0] * difference
                if return_token_type_ids:
                    encoded_inputs["token_type_ids"] = (
                        encoded_inputs["token_type_ids"] +
                        [self.pad_token_type_id] * difference)
                if return_special_tokens_mask:
                    encoded_inputs["special_tokens_mask"] = encoded_inputs[
                        "special_tokens_mask"] + [1] * difference
                encoded_inputs["input_ids"] = encoded_inputs[
                    "input_ids"] + [self.pad_token_id] * difference
            elif self.padding_side == 'left':
                if return_attention_mask:
                    encoded_inputs["attention_mask"] = [0] * difference + [
                        1
                    ] * len(encoded_inputs["input_ids"])
                if return_token_type_ids:
                    encoded_inputs["token_type_ids"] = (
                        [self.pad_token_type_id] * difference +
                        encoded_inputs["token_type_ids"])
                if return_special_tokens_mask:
                    encoded_inputs["special_tokens_mask"] = [
                        1
                    ] * difference + encoded_inputs["special_tokens_mask"]
                encoded_inputs["input_ids"] = [
                    self.pad_token_id
                ] * difference + encoded_inputs["input_ids"]
        else:
            if return_attention_mask:
                encoded_inputs["attention_mask"] = [1] * len(encoded_inputs[
                    "input_ids"])

        if return_position_ids:
            encoded_inputs["position_ids"] = list(
                range(len(encoded_inputs["input_ids"])))

        return encoded_inputs

    def batch_encode(self,
                     batch_text_or_text_pairs,
                     max_seq_len=512,
                     pad_to_max_seq_len=False,
                     stride=0,
                     is_split_into_words=False,
                     truncation_strategy="longest_first",
                     return_position_ids=False,
                     return_token_type_ids=True,
                     return_attention_mask=False,
                     return_length=False,
                     return_overflowing_tokens=False,
                     return_special_tokens_mask=False):
        """
        Performs tokenization and uses the tokenized tokens to prepare model
        inputs. It supports batch inputs of sequence or sequence pair.

        Args:
            batch_text_or_text_pairs (list):
                The element of list can be sequence or sequence pair, and the
                sequence is a string or a list of strings depending on whether
                it has been pretokenized. If each sequence is provided as a list
                of strings (pretokenized), you must set `is_split_into_words` as
                `True` to disambiguate with a sequence pair.
            max_seq_len (int, optional):
                If set to a number, will limit the total sequence returned so
                that it has a maximum length. If there are overflowing tokens,
                those overflowing tokens will be added to the returned dictionary
                when `return_overflowing_tokens` is `True`. Defaults to `None`.
            stride (int, optional):
                Only available for batch input of sequence pair and mainly for
                question answering usage. When for QA, `text` represents questions
                and `text_pair` represents contexts. If `stride` is set to a
                positive number, the context will be split into multiple spans
                where `stride` defines the number of (tokenized) tokens to skip
                from the start of one span to get the next span, thus will produce
                a bigger batch than inputs to include all spans. Moreover, 'overflow_to_sample'
                and 'offset_mapping' preserving the original example and position
                information will be added to the returned dictionary. Defaults to 0.
            pad_to_max_seq_len (bool, optional):
                If set to `True`, the returned sequences would be padded up to
                `max_seq_len` specified length according to padding side
                (`self.padding_side`) and padding token id. Defaults to `False`.
            truncation_strategy (str, optional):
                String selected in the following options:

                - 'longest_first' (default) Iteratively reduce the inputs sequence
                until the input is under `max_seq_len` starting from the longest
                one at each token (when there is a pair of input sequences).
                - 'only_first': Only truncate the first sequence.
                - 'only_second': Only truncate the second sequence.
                - 'do_not_truncate': Do not truncate (raise an error if the input
                sequence is longer than `max_seq_len`).

                Defaults to 'longest_first'.
            return_position_ids (bool, optional):
                Whether to include tokens position ids in the returned dictionary.
                Defaults to `False`.
            return_token_type_ids (bool, optional):
                Whether to include token type ids in the returned dictionary.
                Defaults to `True`.
            return_attention_mask (bool, optional):
                Whether to include the attention mask in the returned dictionary.
                Defaults to `False`.
            return_length (bool, optional):
                Whether to include the length of each encoded inputs in the
                returned dictionary. Defaults to `False`.
            return_overflowing_tokens (bool, optional):
                Whether to include overflowing token information in the returned
                dictionary. Defaults to `False`.
            return_special_tokens_mask (bool, optional):
                Whether to include special tokens mask information in the returned
                dictionary. Defaults to `False`.

        Returns:
            list[dict]:
                The dict has the following optional items:

                - **input_ids** (list[int]): List of token ids to be fed to a model.
                - **pinyin_ids** (list[int]): List of pinyin ids to be fed to a model.
                - **position_ids** (list[int], optional): List of token position ids to be
                  fed to a model. Included when `return_position_ids` is `True`
                - **token_type_ids** (list[int], optional): List of token type ids to be
                  fed to a model. Included when `return_token_type_ids` is `True`.
                - **attention_mask** (list[int], optional): List of integers valued 0 or 1,
                  where 0 specifies paddings and should not be attended to by the
                  model. Included when `return_attention_mask` is `True`.
                - **seq_len** (int, optional): The input_ids length. Included when `return_length`
                  is `True`.
                - **overflowing_tokens** (list[int], optional): List of overflowing tokens.
                  Included when if `max_seq_len` is specified and `return_overflowing_tokens`
                  is True.
                - **num_truncated_tokens** (int, optional): The number of overflowing tokens.
                  Included when if `max_seq_len` is specified and `return_overflowing_tokens`
                  is True.
                - **special_tokens_mask** (list[int], optional): List of integers valued 0 or 1,
                  with 0 specifying special added tokens and 1 specifying sequence tokens.
                  Included when `return_special_tokens_mask` is `True`.
                - **offset_mapping** (list[int], optional): list of pair preserving the
                  index of start and end char in original input for each token.
                  For a sqecial token, the index pair is `(0, 0)`. Included when
                  `stride` works.
                - **overflow_to_sample** (int, optional): Index of example from which this
                  feature is generated. Included when `stride` works.
        """

        def get_input_ids(text):
            if isinstance(text, str):
                tokens = self.tokenize(text)
                return self.convert_tokens_to_ids(tokens)
            elif isinstance(text,
                            (list, tuple)) and len(text) > 0 and isinstance(
                                text[0], str):
                return self.convert_tokens_to_ids(text)
            elif isinstance(text,
                            (list, tuple)) and len(text) > 0 and isinstance(
                                text[0], int):
                return text
            else:
                raise ValueError(
                    "Input is not valid. Should be a string, a list/tuple of strings or a list/tuple of integers."
                )

        batch_encode_inputs = []
        for example_id, tokens_or_pair_tokens in enumerate(
                batch_text_or_text_pairs):
            if not isinstance(tokens_or_pair_tokens, (list, tuple)):
                text, text_pair = tokens_or_pair_tokens, None
            elif is_split_into_words and not isinstance(
                    tokens_or_pair_tokens[0], (list, tuple)):
                text, text_pair = tokens_or_pair_tokens, None
            else:
                text, text_pair = tokens_or_pair_tokens

            if stride > 0 and text_pair is not None:
                first_ids = get_input_ids(text)
                second_ids = get_input_ids(text_pair)

                max_len_for_pair = max_seq_len - len(
                    first_ids) - self.num_special_tokens_to_add(pair=True)
                token_offset_mapping = self.get_offset_mapping(text)
                token_pair_offset_mapping = self.get_offset_mapping(text_pair)

                while True:
                    encoded_inputs = {}

                    ids = first_ids
                    mapping = token_offset_mapping
                    if len(second_ids) <= max_len_for_pair:
                        pair_ids = second_ids
                        pair_mapping = token_pair_offset_mapping
                    else:
                        pair_ids = second_ids[:max_len_for_pair]
                        pair_mapping = token_pair_offset_mapping[:
                                                                 max_len_for_pair]

                    offset_mapping = self.build_offset_mapping_with_special_tokens(
                        mapping, pair_mapping)

                    sequence = self.build_inputs_with_special_tokens(ids,
                                                                     pair_ids)
                    token_type_ids = self.create_token_type_ids_from_sequences(
                        ids, pair_ids)

                    # Build output dictionnary
                    encoded_inputs["input_ids"] = sequence
                    # add_pinyin_ids
                    encoded_inputs["pinyin_ids"] = self.get_pinyin_ids(
                        text, text_pair, offset_mapping)
                    if return_token_type_ids:
                        encoded_inputs["token_type_ids"] = token_type_ids
                    if return_special_tokens_mask:
                        encoded_inputs[
                            "special_tokens_mask"] = self.get_special_tokens_mask(
                                ids, pair_ids)
                    if return_length:
                        encoded_inputs["seq_len"] = len(encoded_inputs[
                            "input_ids"])

                    # Check lengths
                    assert max_seq_len is None or len(encoded_inputs[
                        "input_ids"]) <= max_seq_len

                    # Padding
                    needs_to_be_padded = pad_to_max_seq_len and \
                                        max_seq_len and len(encoded_inputs["input_ids"]) < max_seq_len

                    encoded_inputs['offset_mapping'] = offset_mapping

                    if needs_to_be_padded:
                        difference = max_seq_len - len(encoded_inputs[
                            "input_ids"])
                        # padding pinyin_ids
                        encoded_inputs["pinyin_ids"] = encoded_inputs[
                            "pinyin_ids"] + self.special_tokens_pinyin_ids * difference
                        if self.padding_side == 'right':
                            if return_attention_mask:
                                encoded_inputs["attention_mask"] = [1] * len(
                                    encoded_inputs[
                                        "input_ids"]) + [0] * difference
                            if return_token_type_ids:
                                # 0 for padding token mask
                                encoded_inputs["token_type_ids"] = (
                                    encoded_inputs["token_type_ids"] +
                                    [self.pad_token_type_id] * difference)
                            if return_special_tokens_mask:
                                encoded_inputs[
                                    "special_tokens_mask"] = encoded_inputs[
                                        "special_tokens_mask"] + [1
                                                                  ] * difference
                            encoded_inputs["input_ids"] = encoded_inputs[
                                "input_ids"] + [self.pad_token_id] * difference
                            encoded_inputs['offset_mapping'] = encoded_inputs[
                                'offset_mapping'] + [(0, 0)] * difference
                        elif self.padding_side == 'left':
                            if return_attention_mask:
                                encoded_inputs["attention_mask"] = [
                                    0
                                ] * difference + [1] * len(encoded_inputs[
                                    "input_ids"])
                            if return_token_type_ids:
                                # 0 for padding token mask
                                encoded_inputs["token_type_ids"] = (
                                    [self.pad_token_type_id] * difference +
                                    encoded_inputs["token_type_ids"])
                            if return_special_tokens_mask:
                                encoded_inputs["special_tokens_mask"] = [
                                    1
                                ] * difference + encoded_inputs[
                                    "special_tokens_mask"]
                            encoded_inputs["input_ids"] = [
                                self.pad_token_id
                            ] * difference + encoded_inputs["input_ids"]
                            encoded_inputs['offset_mapping'] = [
                                (0, 0)
                            ] * difference + encoded_inputs['offset_mapping']
                    else:
                        if return_attention_mask:
                            encoded_inputs["attention_mask"] = [1] * len(
                                encoded_inputs["input_ids"])

                    if return_position_ids:
                        encoded_inputs["position_ids"] = list(
                            range(len(encoded_inputs["input_ids"])))

                    encoded_inputs['overflow_to_sample'] = example_id
                    batch_encode_inputs.append(encoded_inputs)

                    if len(second_ids) <= max_len_for_pair:
                        break
                    else:
                        second_ids = second_ids[max_len_for_pair - stride:]
                        token_pair_offset_mapping = token_pair_offset_mapping[
                            max_len_for_pair - stride:]

            else:
                batch_encode_inputs.append(
                    self.encode(
                        text,
                        text_pair,
                        max_seq_len=max_seq_len,
                        pad_to_max_seq_len=pad_to_max_seq_len,
                        truncation_strategy=truncation_strategy,
                        return_position_ids=return_position_ids,
                        return_token_type_ids=return_token_type_ids,
                        return_attention_mask=return_attention_mask,
                        return_length=return_length,
                        return_overflowing_tokens=return_overflowing_tokens,
                        return_special_tokens_mask=return_special_tokens_mask))

        return batch_encode_inputs

    def truncate_sequences(self,
                           ids,
                           pair_ids=None,
                           token_offset_mapping=None,
                           token_pair_offset_mapping=None,
                           num_tokens_to_remove=0,
                           truncation_strategy='longest_first',
                           stride=0):
        """
        Truncates a sequence pair in place to the maximum length.

        Args:
            ids: list of tokenized input ids. Can be obtained from a string by chaining the
                `tokenize` and `convert_tokens_to_ids` methods.
            pair_ids: Optional second list of input ids. Can be obtained from a string by chaining the
                `tokenize` and `convert_tokens_to_ids` methods.
            token_offset_mapping (list): The map of tokens and the start and end index of their start and end character
            token_pair_offset_mapping(list): The map of token pairs and the start and end index of their start and end character
            num_tokens_to_remove (:obj:`int`, `optional`, defaults to ``0``):
                number of tokens to remove using the truncation strategy
            truncation_strategy: string selected in the following options:
                - 'longest_first' (default) Iteratively reduce the inputs sequence until the input is under max_seq_len
                    starting from the longest one at each token (when there is a pair of input sequences).
                    Overflowing tokens only contains overflow from the first sequence.
                - 'only_first': Only truncate the first sequence. raise an error if the first sequence is shorter or equal to than num_tokens_to_remove.
                - 'only_second': Only truncate the second sequence
                - 'do_not_truncate': Does not truncate (raise an error if the input sequence is longer than max_seq_len)
            stride (:obj:`int`, `optional`, defaults to ``0``):
                If set to a number along with max_seq_len, the overflowing tokens returned will contain some tokens
                from the main sequence returned. The value of this argument defines the number of additional tokens.
        """

        if num_tokens_to_remove <= 0:
            return ids, pair_ids, []

        if truncation_strategy == 'longest_first':
            overflowing_tokens = []
            for _ in range(num_tokens_to_remove):
                if pair_ids is None or len(ids) > len(pair_ids):
                    overflowing_tokens = [ids[-1]] + overflowing_tokens
                    ids = ids[:-1]
                    token_offset_mapping = token_offset_mapping[:-1]
                else:
                    pair_ids = pair_ids[:-1]
                    token_pair_offset_mapping = token_pair_offset_mapping[:-1]
            window_len = min(len(ids), stride)
            if window_len > 0:
                overflowing_tokens = ids[-window_len:] + overflowing_tokens
        elif truncation_strategy == 'only_first':
            assert len(ids) > num_tokens_to_remove
            window_len = min(len(ids), stride + num_tokens_to_remove)
            overflowing_tokens = ids[-window_len:]
            ids = ids[:-num_tokens_to_remove]
            token_offset_mapping = token_offset_mapping[:-num_tokens_to_remove]
        elif truncation_strategy == 'only_second':
            assert pair_ids is not None and len(pair_ids) > num_tokens_to_remove
            window_len = min(len(pair_ids), stride + num_tokens_to_remove)
            overflowing_tokens = pair_ids[-window_len:]
            pair_ids = pair_ids[:-num_tokens_to_remove]
            token_pair_offset_mapping = token_pair_offset_mapping[:
                                                                  -num_tokens_to_remove]
        elif truncation_strategy == 'do_not_truncate':
            raise ValueError(
                "Input sequence are too long for max_length. Please select a truncation strategy."
            )
        else:
            raise ValueError(
                "Truncation_strategy should be selected in ['longest_first', 'only_first', 'only_second', 'do_not_truncate']"
            )
        return (ids, pair_ids, token_offset_mapping, token_pair_offset_mapping,
                overflowing_tokens)

    @lru_cache(9999)
    def pinyin_locs_map(self, text):
        """
        Get the map of pinyin locations and pinyin tensor.

        Args:
            text (str):
                The sequence to be processed.
 
        Returns:
            dict: the map of pinyin locations and pinyin tensor.
        """
        pinyin = try_import("pypinyin.pinyin")
        Style = try_import("pypinyin.Style")
        pinyin_list = pinyin(
            text,
            style=Style.TONE3,
            heteronym=True,
            errors=lambda x: [["not chinese"] for _ in x], )
        pinyin_locs = {}
        # get pinyin of each location
        for index, item in enumerate(pinyin_list):
            pinyin_string = item[0]
            # not a Chinese character, pass
            if pinyin_string == "not chinese":
                continue
            if pinyin_string in self.pinyin2tensor:
                pinyin_locs[index] = self.pinyin2tensor[pinyin_string]
            else:
                ids = [0] * 8
                for i, p in enumerate(pinyin_string):
                    if p not in self.pinyin_dict["char2idx"]:
                        ids = [0] * 8
                        break
                    ids[i] = self.pinyin_dict["char2idx"][p]
                pinyin_locs[index] = ids
        return pinyin_locs

    def get_pinyin_ids(self, text, text_pair=None, offset_mapping=None):
        """
        Find chinese character location, and generate pinyin ids.

        Args:
            text (str):
                The sequence to be processed.
            text_pair (str, optional): 
                Same as `text` argument, while it represents for the latter sequence of the sequence pair.
                Defaults to `None`.
            offset_mapping (list, optional):
                A list of wordpiece offsets with the appropriate offsets of special tokens.
                Defaults to `None`.
                
        Returns:
            list: The list of pinyin id tensor.
        """

        text_pinyin_locs = self.pinyin_locs_map(text)
        if text_pair:
            text_pair_pinyin_locs = self.pinyin_locs_map(text_pair)
        else:
            text_pair_pinyin_locs = None

        pinyin_ids = []
        special_token_count = 0

        for offset in offset_mapping:
            if offset == (0, 0):
                special_token_count += 1

            if special_token_count <= 1:
                pinyin_locs_maps = text_pinyin_locs
            else:
                pinyin_locs_maps = text_pair_pinyin_locs

            if offset[1] - offset[0] != 1:
                pinyin_ids.extend([0] * 8)
                continue
            if offset[0] in pinyin_locs_maps:
                pinyin_ids.extend(pinyin_locs_maps[offset[0]])
            else:
                pinyin_ids.extend([0] * 8)

        return pinyin_ids
