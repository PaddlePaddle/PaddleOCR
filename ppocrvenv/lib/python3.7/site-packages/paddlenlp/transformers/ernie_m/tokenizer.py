# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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

from paddle.utils import try_import

from .. import PretrainedTokenizer
from ..tokenizer_utils import _is_control, _is_whitespace

__all__ = ['ErnieMTokenizer']

SPIECE_UNDERLINE = "▁"


class ErnieMTokenizer(PretrainedTokenizer):
    r"""
    Constructs a ErnieM tokenizer. It uses the `sentencepiece` tools to cut the words to sub-words.
    
    Args:
        vocab_file (str): 
            The file path of the vocabulary.
        sentencepiece_model_file (str):
            The file path of sentencepiece model.
        do_lower_case (str, optional): 
            Whether or not to lowercase the input when tokenizing.
            Defaults to`True`.
        unk_token (str, optional): 
            A special token representing the *unknown (out-of-vocabulary)* token.
            An unknown token is set to be `unk_token` inorder to be converted to an ID.
            Defaults to "[UNK]".
        sep_token (str, optional): 
            A special token separating two different sentences in the same input.
            Defaults to "[SEP]".
        pad_token (str, optional): 
            A special token used to make arrays of tokens the same size for batching purposes.
            Defaults to "[PAD]".
        cls_token (str, optional): 
            A special token used for sequence classification. It is the last token
            of the sequence when built with special tokens. Defaults to "[CLS]".
        mask_token (str, optional): 
            A special token representing a masked token. This is the token used
            in the masked language modeling task which the model tries to predict the original unmasked ones.
            Defaults to "[MASK]".
    """
    resource_files_names = {
        "sentencepiece_model_file": "sentencepiece.bpe.model",
        "vocab_file": "vocab.txt",
    }  # for save_pretrained
    pretrained_resource_files_map = {
        "vocab_file": {
            "ernie-m-base":
            "https://bj.bcebos.com/paddlenlp/models/transformers/ernie_m/ernie_m.vocab.txt",
            "ernie-m-large":
            "https://bj.bcebos.com/paddlenlp/models/transformers/ernie_m/ernie_m.vocab.txt"
        },
        "sentencepiece_model_file": {
            "ernie-m-base":
            "https://bj.bcebos.com/paddlenlp/models/transformers/ernie_m/ernie_m.sentencepiece.bpe.model",
            "ernie-m-large":
            "https://bj.bcebos.com/paddlenlp/models/transformers/ernie_m/ernie_m.sentencepiece.bpe.model"
        }
    }
    pretrained_init_configuration = {
        "ernie-m-base": {
            "do_lower_case": True
        },
        "ernie-m-large": {
            "do_lower_case": True
        }
    }

    def __init__(self,
                 vocab_file,
                 sentencepiece_model_file,
                 do_lower_case=True,
                 encoding="utf8",
                 unk_token="[UNK]",
                 sep_token="[SEP]",
                 pad_token="[PAD]",
                 cls_token="[CLS]",
                 mask_token="[MASK]"):
        mod = try_import('sentencepiece')
        self.sp_model = mod.SentencePieceProcessor()

        self.do_lower_case = do_lower_case
        self.encoding = encoding
        if not os.path.isfile(vocab_file):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'.".format(vocab_file))
        self.vocab = self.load_vocabulary(vocab_file, unk_token=unk_token)

        if os.path.isfile(sentencepiece_model_file):
            self.sp_model.Load(sentencepiece_model_file)

    def __call__(self,
                 text,
                 text_pair=None,
                 max_seq_len=None,
                 stride=0,
                 is_split_into_words=False,
                 pad_to_max_seq_len=False,
                 truncation_strategy="longest_first",
                 return_position_ids=True,
                 return_token_type_ids=False,
                 return_attention_mask=True,
                 return_length=False,
                 return_overflowing_tokens=False,
                 return_special_tokens_mask=False):
        return super(ErnieMTokenizer, self).__call__(
            text, text_pair, max_seq_len, stride, is_split_into_words,
            pad_to_max_seq_len, truncation_strategy, return_position_ids,
            return_token_type_ids, return_attention_mask, return_length,
            return_overflowing_tokens, return_special_tokens_mask)

    @property
    def vocab_size(self):
        r"""
        Return the size of vocabulary.

        Returns:
            int: The size of vocabulary.
        """
        return len(self.vocab)

    def clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
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

    def _tokenize(self, text, sample=False):
        """Tokenize a string."""
        text = self.clean_text(text)

        if not sample:
            pieces = self.sp_model.EncodeAsPieces(text)
        else:
            pieces = self.sp_model.SampleEncodeAsPieces(text, 64, 0.1)
        new_pieces = []
        for piece in pieces:
            if len(piece) > 1 and piece[-1] == str(",") and piece[-2].isdigit():
                cur_pieces = self.sp_model.EncodeAsPieces(piece[:-1].replace(
                    SPIECE_UNDERLINE, ""))
                if piece[0] != SPIECE_UNDERLINE and cur_pieces[0][
                        0] == SPIECE_UNDERLINE:
                    if len(cur_pieces[0]) == 1:
                        cur_pieces = cur_pieces[1:]
                    else:
                        cur_pieces[0] = cur_pieces[0][1:]
                cur_pieces.append(piece[-1])
                new_pieces.extend(cur_pieces)
            else:
                new_pieces.append(piece)

        return new_pieces

    def tokenize(self, text):
        r"""
        Converts a string to a list of tokens.
        
        Args:
            text (str): The text to be tokenized.
        Returns:
            List(str): A list of string representing converted tokens.
        """
        return self._tokenize(text)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (strings for sub-words) in a single string."""
        out_string = "".join(tokens).replace(SPIECE_UNDERLINE, " ").strip()
        return out_string

    def convert_ids_to_string(self, ids):
        """
        Converts a sequence of tokens (strings for sub-words) in a single string.
        """
        tokens = self.convert_ids_to_tokens(ids)
        out_string = "".join(tokens).replace(SPIECE_UNDERLINE, " ").strip()
        return out_string

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        r"""
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. 
        
        An ERNIE-M sequence has the following format:
        - single sequence:       ``[CLS] X [SEP]``
        - pair of sequences:        ``[CLS] A [SEP] [SEP] B [SEP]``
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
        return _cls + token_ids_0 + _sep + _sep + token_ids_1 + _sep

    def build_offset_mapping_with_special_tokens(self,
                                                 offset_mapping_0,
                                                 offset_mapping_1=None):
        r"""
        Build offset map from a pair of offset map by concatenating and adding offsets of special tokens. 
        
        An ERNIE-M offset_mapping has the following format:
        - single sequence:      ``(0,0) X (0,0)``
        - pair of sequences:        ``(0,0) A (0,0) (0,0) B (0,0)``
        
        Args:
            offset_mapping_ids_0 (List[tuple]):
                List of char offsets to which the special tokens will be added.
            offset_mapping_ids_1 (List[tuple], optional):
                Optional second list of wordpiece offsets for offset mapping pairs.
                Defaults to `None`.
        Returns:
            List[tuple]: List of wordpiece offsets with the appropriate offsets of special tokens.
        """
        if offset_mapping_1 is None:
            return [(0, 0)] + offset_mapping_0 + [(0, 0)]

        return [(0, 0)] + offset_mapping_0 + [(0, 0), (0, 0)
                                              ] + offset_mapping_1 + [(0, 0)]

    def get_special_tokens_mask(self,
                                token_ids_0,
                                token_ids_1=None,
                                already_has_special_tokens=False):
        r"""
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer ``encode`` methods.
        Args:
            token_ids_0 (List[int]): 
                List of ids of the first sequence.
            token_ids_1 (List[int], optinal): 
                Optional second list of IDs for sequence pairs.
                Defaults to `None`.
            already_has_special_tokens (str, optional): 
                Whether or not the token list is already formatted with special tokens for the model. 
                Defaults to `False`.
        Returns:
            List[int]: 
                The list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """

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
            return [1] + ([0] * len(token_ids_0)) + [1, 1] + (
                [0] * len(token_ids_1)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1]
