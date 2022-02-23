# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from enum import Enum
import os
import os.path as osp
import numpy as np
import logging

import paddle
import paddle.nn as nn
from paddle.utils.download import get_path_from_url
from paddlenlp.utils.env import _get_sub_home, MODEL_HOME
from paddlenlp.utils.log import logger
from paddlenlp.data import Vocab, get_idx_from_word
from .constant import EMBEDDING_URL_ROOT, PAD_TOKEN, UNK_TOKEN,\
                      EMBEDDING_NAME_LIST

EMBEDDING_HOME = _get_sub_home('embeddings', parent_home=MODEL_HOME)

__all__ = ['list_embedding_name', 'TokenEmbedding']


def list_embedding_name():
    """
    Lists all names of pretrained embedding models paddlenlp provides.
    """
    return list(EMBEDDING_NAME_LIST)


class TokenEmbedding(nn.Embedding):
    """
    A `TokenEmbedding` can load pre-trained embedding model which paddlenlp provides by
    specifying embedding name. Furthermore, a `TokenEmbedding` can load extended vocabulary
    by specifying extended_vocab_path.

    Args:
        embedding_name (`str`, optional):
            The pre-trained embedding model name. Use `paddlenlp.embeddings.list_embedding_name()` to
            list the names of all embedding models that we provide. 
            Defaults to `w2v.baidu_encyclopedia.target.word-word.dim300`.
        unknown_token (`str`, optional):
            Specifies unknown token.
            Defaults to `[UNK]`.
        unknown_token_vector (`list`, optional):
            To initialize the vector of unknown token. If it's none, use normal distribution to
            initialize the vector of unknown token.
            Defaults to `None`.
        extended_vocab_path (`str`, optional):
            The file path of extended vocabulary.
            Defaults to `None`.
        trainable (`bool`, optional):
            Whether the weight of embedding can be trained.
            Defaults to True.
        keep_extended_vocab_only (`bool`, optional):
            Whether to keep the extended vocabulary only, will be effective only if provides extended_vocab_path.
            Defaults to False.
    """

    def __init__(self,
                 embedding_name=EMBEDDING_NAME_LIST[0],
                 unknown_token=UNK_TOKEN,
                 unknown_token_vector=None,
                 extended_vocab_path=None,
                 trainable=True,
                 keep_extended_vocab_only=False):
        vector_path = osp.join(EMBEDDING_HOME, embedding_name + ".npz")
        if not osp.exists(vector_path):
            # download
            url = EMBEDDING_URL_ROOT + "/" + embedding_name + ".tar.gz"
            get_path_from_url(url, EMBEDDING_HOME)

        logger.info("Loading token embedding...")
        vector_np = np.load(vector_path)
        self.embedding_dim = vector_np['embedding'].shape[1]
        self.unknown_token = unknown_token
        if unknown_token_vector is not None:
            unk_vector = np.array(unknown_token_vector).astype(
                paddle.get_default_dtype())
        else:
            unk_vector = np.random.normal(
                scale=0.02,
                size=self.embedding_dim).astype(paddle.get_default_dtype())
        pad_vector = np.array(
            [0] * self.embedding_dim).astype(paddle.get_default_dtype())
        if extended_vocab_path is not None:
            embedding_table = self._extend_vocab(extended_vocab_path, vector_np,
                                                 pad_vector, unk_vector,
                                                 keep_extended_vocab_only)
            trainable = True
        else:
            embedding_table = self._init_without_extend_vocab(
                vector_np, pad_vector, unk_vector)

        self.vocab = Vocab.from_dict(
            self._word_to_idx, unk_token=unknown_token, pad_token=PAD_TOKEN)
        self.num_embeddings = embedding_table.shape[0]
        # import embedding
        super(TokenEmbedding, self).__init__(
            self.num_embeddings,
            self.embedding_dim,
            padding_idx=self._word_to_idx[PAD_TOKEN])
        self.weight.set_value(embedding_table)
        self.set_trainable(trainable)
        logger.info("Finish loading embedding vector.")
        s = "Token Embedding info:\
             \nUnknown index: {}\
             \nUnknown token: {}\
             \nPadding index: {}\
             \nPadding token: {}\
             \nShape :{}".format(
            self._word_to_idx[self.unknown_token], self.unknown_token,
            self._word_to_idx[PAD_TOKEN], PAD_TOKEN, self.weight.shape)
        logger.info(s)

    def _init_without_extend_vocab(self, vector_np, pad_vector, unk_vector):
        """
        Constructs index to word list, word to index dict and embedding weight.
        """
        self._idx_to_word = list(vector_np['vocab'])
        self._idx_to_word.append(self.unknown_token)
        self._idx_to_word.append(PAD_TOKEN)
        self._word_to_idx = self._construct_word_to_idx(self._idx_to_word)
        # insert unk, pad embedding
        embedding_table = np.append(
            vector_np['embedding'], [unk_vector, pad_vector], axis=0)

        return embedding_table

    def _read_vocab_list_from_file(self, extended_vocab_path):
        # load new vocab table from file
        vocab_list = []
        with open(extended_vocab_path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                vocab = line.rstrip("\n").split("\t")[0]
                vocab_list.append(vocab)
        return vocab_list

    def _extend_vocab(self, extended_vocab_path, vector_np, pad_vector,
                      unk_vector, keep_extended_vocab_only):
        """
        Constructs index to word list, word to index dict and embedding weight using
        extended vocab.
        """
        logger.info("Start extending vocab.")
        extend_vocab_list = self._read_vocab_list_from_file(extended_vocab_path)
        extend_vocab_set = set(extend_vocab_list)
        # update idx_to_word
        self._idx_to_word = extend_vocab_list
        self._word_to_idx = self._construct_word_to_idx(self._idx_to_word)

        # use the Xavier init the embedding
        xavier_scale = np.sqrt(
            6.0 / float(len(self._idx_to_word) + self.embedding_dim))
        embedding_table = np.random.uniform(
            low=-1.0 * xavier_scale,
            high=xavier_scale,
            size=(len(self._idx_to_word),
                  self.embedding_dim)).astype(paddle.get_default_dtype())

        pretrained_idx_to_word = list(vector_np['vocab'])
        pretrained_word_to_idx = self._construct_word_to_idx(
            pretrained_idx_to_word)
        pretrained_embedding_table = np.array(vector_np['embedding'])

        pretrained_vocab_set = set(pretrained_idx_to_word)
        extend_vocab_set = set(self._idx_to_word)
        vocab_intersection = pretrained_vocab_set & extend_vocab_set
        vocab_subtraction = pretrained_vocab_set - extend_vocab_set

        # assignment from pretrained_vocab_embedding to extend_vocab_embedding
        pretrained_vocab_intersect_index = [
            pretrained_word_to_idx[word] for word in vocab_intersection
        ]
        pretrained_vocab_subtract_index = [
            pretrained_word_to_idx[word] for word in vocab_subtraction
        ]
        extend_vocab_intersect_index = [
            self._word_to_idx[word] for word in vocab_intersection
        ]
        embedding_table[
            extend_vocab_intersect_index] = pretrained_embedding_table[
                pretrained_vocab_intersect_index]
        if not keep_extended_vocab_only:
            for idx in pretrained_vocab_subtract_index:
                word = pretrained_idx_to_word[idx]
                self._idx_to_word.append(word)
                self._word_to_idx[word] = len(self._idx_to_word) - 1

            embedding_table = np.append(
                embedding_table,
                pretrained_embedding_table[pretrained_vocab_subtract_index],
                axis=0)

        if self.unknown_token not in extend_vocab_set:
            self._idx_to_word.append(self.unknown_token)
            self._word_to_idx[self.unknown_token] = len(self._idx_to_word) - 1
            embedding_table = np.append(embedding_table, [unk_vector], axis=0)
        else:
            unk_idx = self._word_to_idx[self.unknown_token]
            embedding_table[unk_idx] = unk_vector

        if PAD_TOKEN not in extend_vocab_set:
            self._idx_to_word.append(PAD_TOKEN)
            self._word_to_idx[PAD_TOKEN] = len(self._idx_to_word) - 1
            embedding_table = np.append(embedding_table, [pad_vector], axis=0)
        else:
            embedding_table[self._word_to_idx[PAD_TOKEN]] = pad_vector

        logger.info("Finish extending vocab.")
        return embedding_table

    def set_trainable(self, trainable):
        """
        Whether or not to set the weights of token embedding to be trainable.

        Args:
            trainable (`bool`):
                The weights can be trained if trainable is set to True, or the weights are fixed if trainable is False.

        """
        self.weight.stop_gradient = not trainable

    def search(self, words):
        """
        Gets the vectors of specifying words.

        Args:
            words (`list` or `str` or `int`): The words which need to be searched.

        Returns:
            `numpy.array`: The vectors of specifying words.

        Examples:
            .. code-block::

                from paddlenlp.embeddings import TokenEmbedding

                embed = TokenEmbedding()
                vector =  embed.search('Welcome to use PaddlePaddle and PaddleNLP!')

        """
        idx_list = self.get_idx_list_from_words(words)
        idx_tensor = paddle.to_tensor(idx_list)
        return self(idx_tensor).numpy()

    def get_idx_from_word(self, word):
        """
        Gets the index of specifying word by searching word_to_idx dict. 

        Args:
            word (`list` or `str` or `int`): The input token word which we want to get the token index converted from.

        Returns:
            `int`: The index of specifying word.

        """
        return get_idx_from_word(word, self.vocab.token_to_idx,
                                 self.unknown_token)

    def get_idx_list_from_words(self, words):
        """
        Gets the index list of specifying words by searching word_to_idx dict.

        Args:
            words (`list` or `str` or `int`): The input token words which we want to get the token indices converted from.

        Returns:
            `list`: The indexes list of specifying words.

        Examples:
            .. code-block::

                from paddlenlp.embeddings import TokenEmbedding

                embed = TokenEmbedding()
                index =  embed.get_idx_from_word('Welcome to use PaddlePaddle and PaddleNLP!')
                #635963

        """
        if isinstance(words, str):
            idx_list = [self.get_idx_from_word(words)]
        elif isinstance(words, int):
            idx_list = [words]
        elif isinstance(words, list) or isinstance(words, tuple):
            idx_list = [
                self.get_idx_from_word(word) if isinstance(word, str) else word
                for word in words
            ]
        else:
            raise TypeError
        return idx_list

    def _dot_np(self, array_a, array_b):
        return np.sum(array_a * array_b)

    def _calc_word(self, word_a, word_b, calc_kernel):
        embeddings = self.search([word_a, word_b])
        embedding_a = embeddings[0]
        embedding_b = embeddings[1]
        return calc_kernel(embedding_a, embedding_b)

    def dot(self, word_a, word_b):
        """
        Calculates the dot product of 2 words. Dot product or scalar product is an
        algebraic operation that takes two equal-length sequences of numbers (usually
        coordinate vectors), and returns a single number.

        Args:
            word_a (`str`): The first word string.
            word_b (`str`): The second word string.

        Returns:
            float: The dot product of 2 words.

        Examples:
            .. code-block::

                from paddlenlp.embeddings import TokenEmbedding

                embed = TokenEmbedding()
                dot_product =  embed.dot('PaddlePaddle', 'PaddleNLP!')
                #0.11827179

        """
        dot = self._dot_np
        return self._calc_word(word_a, word_b, lambda x, y: dot(x, y))

    def cosine_sim(self, word_a, word_b):
        """
        Calculates the cosine similarity of 2 word vectors. Cosine similarity is the
        cosine of the angle between two n-dimensional vectors in an n-dimensional space.

        Args:
            word_a (`str`): The first word string.
            word_b (`str`): The second word string.

        Returns:
            float: The cosine similarity of 2 words.

        Examples:
            .. code-block::

                from paddlenlp.embeddings import TokenEmbedding

                embed = TokenEmbedding()
                cosine_simi =  embed.cosine_sim('PaddlePaddle', 'PaddleNLP!')
                #0.99999994

        """
        dot = self._dot_np
        return self._calc_word(
            word_a, word_b,
            lambda x, y: dot(x, y) / (np.sqrt(dot(x, x)) * np.sqrt(dot(y, y))))

    def _construct_word_to_idx(self, idx_to_word):
        """
        Constructs word to index dict.

        Args:
            idx_to_word ('list'):

        Returns:
            `Dict`: The word to index dict constructed by idx_to_word.

        """
        word_to_idx = {}
        for i, word in enumerate(idx_to_word):
            word_to_idx[word] = i
        return word_to_idx

    def __repr__(self):
        """
        Returns:
            `Str`: The token embedding infomation.

        """
        info = "Object   type: {}\
             \nUnknown index: {}\
             \nUnknown token: {}\
             \nPadding index: {}\
             \nPadding token: {}\
             \n{}".format(
            super(TokenEmbedding, self).__repr__(),
            self._word_to_idx[self.unknown_token], self.unknown_token,
            self._word_to_idx[PAD_TOKEN], PAD_TOKEN, self.weight)
        return info
