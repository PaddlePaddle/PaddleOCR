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

from .. import GPTTokenizer, AddedToken
from paddle.utils import try_import

__all__ = ['BlenderbotTokenizer']


class BlenderbotTokenizer(GPTTokenizer):
    r"""
    Construct a Blenderbot tokenizer, derived from the GPT tokenizer, using
    byte-level Byte-Pair-Encoding.

    This tokenizer inherits from :class:`~paddlenlp.transformers.GPTTokenizer`,
    which contains most of the main methods.
    Please should refer to the superclass for more information regarding methods.
    Args:
        vocab_file (str): file path of the vocabulary
        merges_file (str): file path of the merges file.
        errors (str): The method to handle errors in decoding
        max_len (int): The specified maximum sequence length. Default: "None".
        special_tokens (dict): The additional special tokens. Default: "None".
        bos_token (str): The special token for beginning of sequence token. Default: "<s>".
        eos_token (str): The special token for end of sequence token. Default: "</s>".
        cls_token (str): The special token for cls. Default: "<s>".
        sep_token (str): The special token for separator token . Default: "</s>".
        pad_token (str): The special token for padding. Default: "<pad>".
        eol_token (str): The special token for newline. Default: "\u010a".
        add_prefix (bool): Whether or not to add an initial space to the input.
            This allows to treat the leading word just as any other word.
            (Blenderbot adds an initial space when tokenizes input text, which
             is differnt from BlenderbotSmall)
    Examples:
        .. code-block:: python
            from paddlenlp.transformers import BlenderbotTokenizer
            tokenizer = BlenderbotTokenizer.from_pretrained("blenderbot-400M-distill")
            text = "My friends are cool but they eat too many carbs."
            inputs = tokenizer(text)
            # above line outputs:
            # {'input_ids': [863, 1329, 366, 1449, 373, 382, 1861, 618, 847, 911, 1372, 21, 2],
            # 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
    """
    resource_files_names = {
        "vocab_file": "vocab.json",
        "merges_file": "merges.txt"
    }
    pretrained_resource_files_map = {
        "vocab_file": {
            "blenderbot-400M-distill":
            "https://bj.bcebos.com/paddlenlp/models/transformers/blenderbot/blenderbot-400M-distill-vocab.json",
            "blenderbot-3B":
            "https://bj.bcebos.com/paddlenlp/models/transformers/blenderbot/blenderbot-3B-vocab.json",
            "blenderbot-1B-distill":
            "https://bj.bcebos.com/paddlenlp/models/transformers/blenderbot/blenderbot-1B-distill-vocab.json"
        },
        "merges_file": {
            "blenderbot-400M-distill":
            "https://bj.bcebos.com/paddlenlp/models/transformers/blenderbot/blenderbot-400M-distill-merges.txt",
            "blenderbot-3B":
            "https://bj.bcebos.com/paddlenlp/models/transformers/blenderbot/blenderbot-3B-merges.txt",
            "blenderbot-1B-distill":
            "https://bj.bcebos.com/paddlenlp/models/transformers/blenderbot/blenderbot-1B-distill-merges.txt"
        }
    }
    pretrained_init_configuration = {
        "blenderbot-3B": {
            "add_prefix": True
        },
        "blenderbot-400M-distill": {
            "add_prefix": True
        },
        "blenderbot-1B-distill": {
            "add_prefix": True
        }
    }

    def __init__(self,
                 vocab_file,
                 merges_file,
                 errors='replace',
                 max_len=None,
                 special_tokens=None,
                 bos_token="<s>",
                 eos_token="</s>",
                 cls_token="<s>",
                 sep_token="</s>",
                 pad_token="<pad>",
                 unk_token="<unk>",
                 mask_token="<mask>",
                 eol_token='\u010a',
                 add_prefix=True,
                 **kwargs):

        sep_token = AddedToken(
            sep_token,
            lstrip=False,
            rstrip=False,
            single_word=False,
            normalized=True) if isinstance(sep_token, str) else sep_token

        self._build_special_tokens_map_extended(sep_token=sep_token)

        super(BlenderbotTokenizer, self).__init__(
            vocab_file=vocab_file,
            merges_file=merges_file,
            errors=errors,
            max_len=max_len,
            special_tokens=special_tokens,
            pad_token=pad_token,
            eos_token=eos_token,
            eol_token=eol_token)
        self.add_prefix = add_prefix

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        A Blenderbot sequence has the following format:
        ::
            - single sequence: ``X </s>``

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (:obj:`List[int]`, `optional`):
                token_ids_1 Will be ignored

        Returns:
            :obj:`List[int]`: List of input_id with the appropriate special tokens.
        """
        return token_ids_0 + [self.eos_token_id]

    def _tokenize(self, text):
        """
        End-to-end tokenization for Blenderbot models.
        Args:
            text (str): The text to be tokenized.

        Returns:
            list: A list of string representing converted tokens.
        """
        bpe_tokens = []
        re = try_import("regex")
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(
                bpe_token for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def prepare_for_tokenization(self,
                                 text,
                                 is_split_into_words=False,
                                 **kwargs):
        add_prefix = kwargs.pop("add_prefix", self.add_prefix)
        if is_split_into_words or add_prefix:
            text = " " + text
        return text
