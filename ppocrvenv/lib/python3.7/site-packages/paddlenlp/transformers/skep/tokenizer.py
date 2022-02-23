# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
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

import json
import os
import shutil

from paddle.utils import try_import
from paddlenlp.transformers import BasicTokenizer, PretrainedTokenizer, WordpieceTokenizer
from paddlenlp.utils.log import logger
from paddlenlp.utils.env import MODEL_HOME

__all__ = ['SkepTokenizer', ]


def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(33, 126 + 1)) + list(range(161, 172 + 1)) + list(
        range(174, 255 + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


class BpeEncoder(object):
    """BpeEncoder"""

    def __init__(self, encoder_json_file, vocab_bpe_file, errors='replace'):
        """
        Constructs a BpeEncoder.

        Args:
            encoder_json_file (`str`): The path to bpe encode json file.
            vocab_bpe_file (`str`): The path to bpe vocab file.
        """
        self.encoder = self.__get_encoder(encoder_json_file)
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.errors = errors  # how to handle errors in decoding
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.bpe_ranks = self.__get_bpe_ranks(vocab_bpe_file)
        self.cache = {}
        re = try_import("regex")
        self.pat = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )

    def __get_encoder(self, encoder_json_file):
        with open(encoder_json_file, 'r') as f:
            encoder = json.load(f)
        return encoder

    def __get_bpe_ranks(self, vocab_bpe_file):
        with open(vocab_bpe_file, 'r', encoding="utf-8") as f:
            bpe_data = f.read()
        bpe_merges = [
            tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]
        ]
        bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        return bpe_ranks

    def bpe(self, token):
        """
        bpe
        """
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(
                pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
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
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i +
                                                                   1] == second:
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
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        """
        encode
        """
        bpe_tokens = []
        re = try_import("regex")
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_token]
                              for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def decode(self, tokens):
        """
        decode
        """
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode(
            'utf-8', errors=self.errors)
        return text


class SkepTokenizer(PretrainedTokenizer):
    r"""
    Constructs a Skep tokenizer. It uses a basic tokenizer to do punctuation
    splitting, lower casing and so on, and follows a WordPiece tokenizer to
    tokenize as subwords.

    This tokenizer inherits from :class:`~paddlenlp.transformers.tokenizer_utils.PretrainedTokenizer`
    which contains most of the main methods. For more information regarding those methods,
    please refer to this superclass.

    Args:
        vocab_file (str):
            The vocabulary file path (ends with '.txt') required to instantiate
            a `WordpieceTokenizer`.
        bpe_vocab_file (str, optional):
            The vocabulary file path of a `BpeTokenizer`. Defaults to `None`.
        bpe_json_file (str, optional):
            The json file path of a `BpeTokenizer`. Defaults to `None`.
        use_bpe_encoder (bool, optional):
            Whether or not to use BPE Encoder. Defaults to `False`.
        need_token_type_id (bool, optional):
            Whether or not to use token type id. Defaults to `True`.
        add_two_sep_token_inter (bool, optional):
            Whether or not to add two different `sep_token`. Defaults to `False`.
        unk_token (str, optional):
            The special token for unknown words.
            Defaults to "[UNK]".
        sep_token (str, optional):
            The special token for separator token.
            Defaults to "[SEP]".
        pad_token (str, optional):
            The special token for padding.
            Defaults to "[PAD]".
        cls_token (str, optional):
            The special token for cls.
            Defaults to "[CLS]".
        mask_token (str, optional):
            The special token for mask.
            Defaults to "[MASK]".

    Examples:
        .. code-block::

            from paddlenlp.transformers import SkepTokenizer
            tokenizer = SkepTokenizer.from_pretrained('skep_ernie_2.0_large_en')
            encoded_inputs = tokenizer('He was a puppeteer')
            # encoded_inputs:
            # {
            #    'input_ids': [101, 2002, 2001, 1037, 13997, 11510, 102],
            #    'token_type_ids': [0, 0, 0, 0, 0, 0, 0]
            # }
    """
    resource_files_names = {
        "vocab_file": "vocab.txt",
        "bpe_vocab_file": "vocab.bpe",
        "bpe_json_file": "encoder.json"
    }  # for save_pretrained
    pretrained_resource_files_map = {
        "vocab_file": {
            "skep_ernie_1.0_large_ch":
            "https://bj.bcebos.com/paddlenlp/models/transformers/skep/skep_ernie_1.0_large_ch.vocab.txt",
            "skep_ernie_2.0_large_en":
            "https://bj.bcebos.com/paddlenlp/models/transformers/skep/skep_ernie_2.0_large_en.vocab.txt",
            "skep_roberta_large_en":
            "https://bj.bcebos.com/paddlenlp/models/transformers/skep/skep_roberta_large_en.vocab.txt",
        },
        "bpe_vocab_file": {
            "skep_ernie_1.0_large_ch": None,
            "skep_ernie_2.0_large_en": None,
            "skep_roberta_large_en":
            "https://bj.bcebos.com/paddlenlp/models/transformers/skep/skep_roberta_large_en.vocab.bpe",
        },
        "bpe_json_file": {
            "skep_ernie_1.0_large_ch": None,
            "skep_ernie_2.0_large_en": None,
            "skep_roberta_large_en":
            "https://bj.bcebos.com/paddlenlp/models/transformers/skep/skep_roberta_large_en.encoder.json",
        }
    }

    pretrained_init_configuration = {
        "skep_ernie_1.0_large_ch": {
            "do_lower_case": True,
            "use_bpe_encoder": False,
            "need_token_type_id": True,
            "add_two_sep_token_inter": False,
        },
        "skep_ernie_2.0_large_en": {
            "do_lower_case": True,
            "use_bpe_encoder": False,
            "need_token_type_id": True,
            "add_two_sep_token_inter": False,
        },
        "skep_roberta_large_en": {
            "do_lower_case": True,
            "use_bpe_encoder": True,
            "need_token_type_id": False,
            "add_two_sep_token_inter": True,
        },
    }

    def __init__(self,
                 vocab_file,
                 bpe_vocab_file=None,
                 bpe_json_file=None,
                 do_lower_case=True,
                 use_bpe_encoder=False,
                 need_token_type_id=True,
                 add_two_sep_token_inter=False,
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
                "`tokenizer = SkepTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
                .format(vocab_file))
        self.vocab_file = vocab_file
        self.bpe_vocab_file = bpe_vocab_file
        self.bpe_json_file = bpe_json_file
        self.vocab = self.load_vocabulary(
            vocab_file, unk_token=unk_token, pad_token=pad_token)

        self.use_bpe_encoder = use_bpe_encoder
        self.need_token_type_id = need_token_type_id
        self.add_two_sep_token_inter = add_two_sep_token_inter

        if not self.use_bpe_encoder:
            self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
            self.wordpiece_tokenizer = WordpieceTokenizer(
                vocab=self.vocab, unk_token=unk_token)
        else:
            assert (bpe_vocab_file and bpe_json_file) is not None, (
                f"bpe_vocab_file and bpe_json_file must be not None.")
            if os.path.isfile(bpe_vocab_file) and os.path.isfile(bpe_json_file):
                self.bpe_tokenizer = BpeEncoder(bpe_json_file, bpe_vocab_file)

    @property
    def vocab_size(self):
        r"""
        Return the size of vocabulary.

        Returns:
            int: the size of vocabulary.
        """
        return len(self.vocab)

    def _tokenize(self, text):
        r"""
        End-to-end tokenization for Skep models.

        Args:
            text (str): The text to be tokenized.

        Returns:
            list: A list of string representing converted tokens.
        """
        split_tokens = []
        if not self.use_bpe_encoder:
            for token in self.basic_tokenizer.tokenize(text):
                for sub_token in self.wordpiece_tokenizer.tokenize(token):
                    split_tokens.append(sub_token)
        else:
            for token in self.bpe_tokenizer.encode(text):
                split_tokens.append(str(token))

        return split_tokens

    def num_special_tokens_to_add(self, pair=False):
        r"""
        Returns the number of added tokens when encoding a sequence with special tokens.

        Args:
            pair (bool, optional):
                Returns the number of added tokens in the case of a sequence
                pair if set to True, returns the number of added tokens in the case of a single sequence if set to False.
                Defaults to False.

        Returns:
            int: Number of tokens added to sequences
        """
        token_ids_0 = []
        token_ids_1 = []
        return len(
            self.build_inputs_with_special_tokens(token_ids_0, token_ids_1
                                                  if pair else None))

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        r"""
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens.

        A skep_ernie_1.0_large_ch/skep_ernie_2.0_large_en sequence has the following format:

        - single sequence:      ``[CLS] X [SEP]``
        - pair of sequences:        ``[CLS] A [SEP] B [SEP]``

        A skep_roberta_large_en sequence has the following format:

        - single sequence:      ``[CLS] X [SEP]``
        - pair of sequences:        ``[CLS] A [SEP] [SEP] B [SEP]``

        Args:
            token_ids_0 (List[int]):
                List of IDs to which the special tokens will be added.
            token_ids_1 (List[int], optional):
                Optional second list of IDs for sequence pairs.
                Defaults to `None`.

        Returns:
            list[int]: List of input_id with the appropriate special tokens.
        """
        if not self.add_two_sep_token_inter:
            if token_ids_1 is None:
                return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
            _cls = [self.cls_token_id]
            _sep = [self.sep_token_id]
            return _cls + token_ids_0 + _sep + token_ids_1 + _sep
        else:
            if token_ids_1 is None:
                return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
            _cls = [self.cls_token_id]
            _sep = [self.sep_token_id]
            return _cls + token_ids_0 + _sep + _sep + token_ids_1 + _sep

    def create_token_type_ids_from_sequences(self,
                                             token_ids_0,
                                             token_ids_1=None):
        r"""
        Create a mask from the two sequences passed to be used in a sequence-pair classification task.

        A skep_ernie_1.0_large_ch/skep_ernie_2.0_large_en sequence pair mask has the following format:
        ::

            0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
            | first sequence    | second sequence |

        If `token_ids_1` is `None`, this method only returns the first portion of the mask (0s).

        note: There is no need token type ids for skep_roberta_large_ch model.

        Args:
            token_ids_0 (List[int]):
                List of IDs.
            token_ids_1 (List[int], optional):
                Optional second list of IDs for sequence pairs. 
                Defaults to `None`.

        Returns:
            List[int]: List of token_type_id according to the given sequence(s).
        """
        if self.need_token_type_id:
            _sep = [self.sep_token_id]
            _cls = [self.cls_token_id]
            if token_ids_1 is None:
                return len(_cls + token_ids_0 + _sep) * [0]
            return len(_cls + token_ids_0 + _sep) * [0] + len(token_ids_1 +
                                                              _sep) * [1]
        else:
            # For model skep-roberta-large-en, token type ids is no need.
            return None

    def save_resources(self, save_directory):
        """
        Save tokenizer related resources to files under `save_directory`.

        Args:
            save_directory (str): Directory to save files into.
        """
        for name, file_name in self.resource_files_names.items():
            save_path = os.path.join(save_directory, file_name)
            source_file = getattr(self, name)
            if source_file is not None:
                shutil.copyfile(source_file, save_path)
