# encoding=utf-8
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2021 The Facebook, Inc. and The HuggingFace Inc. team.
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

from ..gpt.tokenizer import GPTTokenizer
import re

__all__ = ['BlenderbotSmallTokenizer']


# Copy from paddlenlp.transformers.gpt.tokenizer.get_pairs
def get_pairs(word):
    """
    Args:
        word (tuple): tuple of symbols (symbols being variable-length strings).

    Returns:
        set: symbol pairs in a word.
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


class BlenderbotSmallTokenizer(GPTTokenizer):
    r"""
    Constructs a BlenderbotSmall tokenizer based on Byte-Pair-Encoding.

    This tokenizer inherits from :class:`~paddlenlp.transformers.GPTTokenizer`,
    which contains most of the main methods.
    Please should refer to the superclass for more information regarding methods.
    Args:
        vocab_file (str): file path of the vocabulary
        merges_file (str): file path of the merges file.
        errors (str): The method to handle errors in decoding
        max_len (int): The specified maximum sequence length. Default: "None".
        special_tokens (dict): The additional special tokens. Default: "None".
        bos_token (str): The special token for beginning of sequence token. Default: "__start__".
        eos_token (str): The special token for end of sequence token. Default: "__end__".
        unk_token (str): The special token for unknown tokens. Default: "__unk__"
        pad_token (str): The special token for padding. Default: "__null__".
        eol_token (str): The special token for newline. Default: "__newln__".
    Examples:
        .. code-block:: python
            from paddlenlp.transformers import BlenderbotSmallTokenizer
            tokenizer = BlenderbotSmallTokenizer.from_pretrained("blenderbot_small-90M")
            text = "My friends are cool but they eat too many carbs."
            inputs = tokenizer(text)
            # above line outputs:
            #   {'input_ids': [42, 643, 46, 1430, 45, 52, 1176, 146, 177, 753, 2430, 5],
            #   'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
    """
    resource_files_names = {
        "vocab_file": "vocab.json",
        "merges_file": "merges.txt"
    }
    pretrained_resource_files_map = {
        "vocab_file": {
            "blenderbot_small-90M":
            "https://bj.bcebos.com/paddlenlp/models/transformers/blenderbot_small/blenderbot_small-90M-vocab.json",
        },
        "merges_file": {
            "blenderbot_small-90M":
            "https://bj.bcebos.com/paddlenlp/models/transformers/blenderbot_small/blenderbot_small-90M-merges.txt",
        }
    }
    pretrained_init_configuration = {"blenderbot_small-90M": {}}

    def __init__(self,
                 vocab_file,
                 merges_file,
                 errors='replace',
                 max_len=None,
                 special_tokens=None,
                 bos_token="__start__",
                 eos_token="__end__",
                 unk_token="__unk__",
                 pad_token="__null__",
                 eol_token="__newln__",
                 **kwargs):
        super(BlenderbotSmallTokenizer, self).__init__(
            vocab_file=vocab_file,
            merges_file=merges_file,
            errors=errors,
            max_len=max_len,
            special_tokens=special_tokens,
            pad_token=pad_token,
            eos_token=eos_token,
            eol_token=eol_token)
        self.pat = r"\S+\n?"  # String matching pattern of BlenderbotSmall is different from Blenderbot
        self.unk_id = self.encoder[unk_token]
        self.eol_token = eol_token

    def bpe(self, token):
        """
        Apply Byte-Pair-Encoding on token.
        The process of bpe in BlenderbotSmall is different from Blenderbot.
        Args:
            token (str): The token to be converted.

        Returns:
            str: Converted token.
        """
        if token in self.cache:
            return self.cache[token]
        token = re.sub("([.,!?()])", r" \1", token)
        token = re.sub("(')", r" \1 ", token)
        token = re.sub(r"\s{2,}", " ", token)
        if "\n" in token:
            token = token.replace("\n", self.eol_token)

        tokens = token.split(" ")
        words = []
        for token in tokens:
            if not len(token):
                continue

            token = token.lower()
            word = tuple(token)
            word = tuple(list(word[:-1]) + [word[-1] + "</w>"])
            pairs = get_pairs(word)

            if not pairs:
                words.append(token)
                continue

            while True:
                bigram = min(
                    pairs,
                    key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
                if bigram not in self.bpe_ranks:
                    break
                first, second = bigram
                new_word = []
                i = 0

                while i < len(word):
                    try:
                        j = word.index(first, i)
                        new_word.extend(word[i:j])
                        i = j
                    except ValueError:
                        new_word.extend(word[i:])
                        break

                    if word[i] == first and i < len(word) - 1 and word[
                            i + 1] == second:
                        new_word.append(first + second)
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                new_word = tuple(new_word)
                word = new_word
                if len(word) == 1:
                    break
                else:
                    pairs = get_pairs(word)
            word = "@@ ".join(word)
            word = word[:-4]

            self.cache[token] = word
            words.append(word)
        return " ".join(words)

    def convert_tokens_to_string(self, tokens):
        """
        Converts a sequence of tokens (list of string) to a single string.
        Args:
            tokens (list[str]): A sequence of tokens.

        Returns:
            str: Converted string.
        """
        return " ".join(tokens).replace("@@ ", "").strip()

    def convert_ids_to_string(self,
                              ids,
                              skip_special_tokens=True,
                              clean_up_tokenization_spaces=True):
        """
        Converts a sequence of ids (list of integers) to a single string.
        Args:
            ids (list[int]):
                A sequence of ids corresponding to tokens.
            skip_special_tokens (bool, optional):
                Whether to skip and not decode special tokens when converting. Defaults to `False`.
            clean_up_tokenization_spaces (bool, optional):
                Whether to Clean up a list of simple English tokenization artifacts
                like spaces before punctuations and abbreviated forms.
        Returns:
            str: Converted string.
        """
        tokens = self.convert_ids_to_tokens(
            ids, skip_special_tokens=skip_special_tokens)
        output_string = self.convert_tokens_to_string(tokens)
        if clean_up_tokenization_spaces:
            output_string = (output_string.replace(" .", ".").replace(" ?", "?")
                             .replace(" !", "!").replace(" ,", ",")
                             .replace(" ' ", "'").replace(" n't", "n't")
                             .replace(" 'm", "'m").replace(" 's", "'s")
                             .replace(" 've", "'ve").replace(" 're", "'re"))
        return output_string
