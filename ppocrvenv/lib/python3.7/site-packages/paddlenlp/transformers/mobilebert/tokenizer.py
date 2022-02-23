# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2021 The HuggingFace Inc. team.
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

from .. import BertTokenizer

__all__ = ['MobileBertTokenizer']


class MobileBertTokenizer(BertTokenizer):
    r"""
    Construct a MobileBERT tokenizer.
    :class:`~paddlenlp.transformers.MobileBertTokenizer is identical to :class:`~paddlenlp.transformers.BertTokenizer` and runs end-to-end
    tokenization: punctuation splitting and wordpiece.
    Refer to superclass :class:`~~paddlenlp.transformers.BertTokenizer` for usage examples and documentation concerning
    parameters.
    """
    resource_files_names = {"vocab_file": "vocab.txt"}
    pretrained_resource_files_map = {
        "vocab_file": {
            "mobilebert-uncased":
            "https://bj.bcebos.com/paddlenlp/models/transformers/mobilebert/mobilebert-uncased/vocab.txt"
        }
    }
    pretrained_init_configuration = {
        "mobilebert-uncased": {
            "do_lower_case": True
        }
    }

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
            dict:
                The dict has the following optional items:
                - **input_ids** (list[int]): List of token ids to be fed to a model.
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
                tokens = self._tokenize(text)
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

            first_ids = get_input_ids(text)
            second_ids = get_input_ids(
                text_pair) if text_pair is not None else None

            if stride > 0 and second_ids is not None:

                max_len_for_pair = max_seq_len - len(
                    first_ids) - self.num_special_tokens_to_add(
                        pair=True)  # need -4  <sep> A </sep> </sep> B <sep>

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
                        first_ids,
                        second_ids,
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
