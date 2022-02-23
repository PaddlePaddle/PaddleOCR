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

import math
import sys
from collections import defaultdict

import paddle

from .utils import default_trans_func

__all__ = ["BLEU", "BLEUForDuReader"]


def get_match_size(cand_ngram, refs_ngram):
    ref_set = defaultdict(int)
    for ref_ngram in refs_ngram:
        tmp_ref_set = defaultdict(int)
        for ngram in ref_ngram:
            tmp_ref_set[tuple(ngram)] += 1
        for ngram, count in tmp_ref_set.items():
            ref_set[tuple(ngram)] = max(ref_set[tuple(ngram)], count)
    cand_set = defaultdict(int)
    for ngram in cand_ngram:
        cand_set[tuple(ngram)] += 1
    match_size = 0
    for ngram, count in cand_set.items():
        match_size += min(count, ref_set.get(tuple(ngram), 0))
    cand_size = len(cand_ngram)
    return match_size, cand_size


def get_ngram(sent, n_size, label=None):
    def _ngram(sent, n_size):
        ngram_list = []
        for left in range(len(sent) - n_size):
            ngram_list.append(sent[left:left + n_size + 1])
        return ngram_list

    ngram_list = _ngram(sent, n_size)
    if label is not None:
        ngram_list = [ngram + '_' + label for ngram in ngram_list]
    return ngram_list


class BLEU(paddle.metric.Metric):
    r'''
    BLEU (bilingual evaluation understudy) is an algorithm for evaluating the
    quality of text which has been machine-translated from one natural language
    to another. This metric uses a modified form of precision to compare a
    candidate translation against multiple reference translations.

    BLEU could be used as `paddle.metric.Metric` class, or an ordinary
    class. When BLEU is used as `paddle.metric.Metric` class. A function is
    needed that transforms the network output to reference string list, and
    transforms the label to candidate string. By default, a default function
    `default_trans_func` is provided, which gets target sequence id by
    calculating the maximum probability of each step. In this case, user must
    provide `vocab`. It should be noted that the BLEU here is different from
    the BLEU calculated in prediction, and it is only for observation during
    training and evaluation.
    
    .. math::

        BP & =
        \begin{cases} 
        1,  & \text{if }c>r \\
        e_{1-r/c}, & \text{if }c\leq r
        \end{cases}

        BLEU & = BP\exp(\sum_{n=1}^N w_{n} \log{p_{n}})

    where `c` is the length of candidate sentence, and `r` is the length of reference sentence.

    Args:
        trans_func (callable, optional): `trans_func` transforms the network
            output to string to calculate.
        vocab (dict|paddlenlp.data.vocab, optional): Vocab for target language.
            If `trans_func` is None and BLEU is used as `paddle.metric.Metric`
            instance, `default_trans_func` will be performed and `vocab` must
            be provided.
        n_size (int, optional): Number of gram for BLEU metric. Defaults to 4.
        weights (list, optional): The weights of precision of each gram.
            Defaults to None.
        name (str, optional): Name of `paddle.metric.Metric` instance.
            Defaults to "bleu".

    Examples:
        1. Using as a general evaluation object.

        .. code-block:: python

            from paddlenlp.metrics import BLEU
            bleu = BLEU()
            cand = ["The","cat","The","cat","on","the","mat"]
            ref_list = [["The","cat","is","on","the","mat"], ["There","is","a","cat","on","the","mat"]]
            bleu.add_inst(cand, ref_list)
            print(bleu.score()) # 0.4671379777282001

        2. Using as an instance of `paddle.metric.Metric`.
                
        .. code-block:: python

            # You could add the code below to Seq2Seq example in this repo to
            # use BLEU as `paddlenlp.metric.Metric' class. If you run the
            # following code alone, you may get an error.
            # log example:
            # Epoch 1/12
            # step 100/507 - loss: 308.7948 - Perplexity: 541.5600 - bleu: 2.2089e-79 - 923ms/step
            # step 200/507 - loss: 264.2914 - Perplexity: 334.5099 - bleu: 0.0093 - 865ms/step
            # step 300/507 - loss: 236.3913 - Perplexity: 213.2553 - bleu: 0.0244 - 849ms/step

            from paddlenlp.data import Vocab
            from paddlenlp.metrics import BLEU

            bleu_metric = BLEU(vocab=src_vocab.idx_to_token)
            model.prepare(optimizer, CrossEntropyCriterion(), [ppl_metric, bleu_metric])

    '''

    def __init__(self,
                 trans_func=None,
                 vocab=None,
                 n_size=4,
                 weights=None,
                 name="bleu"):
        super(BLEU, self).__init__()
        if not weights:
            weights = [1 / n_size for _ in range(n_size)]
        assert len(weights) == n_size, (
            "Number of weights and n-gram should be the same, got Number of weights: '%d' and n-gram: '%d'"
            % (len(weights), n_size))
        self._name = name
        self.match_ngram = {}
        self.candi_ngram = {}
        self.weights = weights
        self.bp_r = 0
        self.bp_c = 0
        self.n_size = n_size
        self.vocab = vocab
        self.trans_func = trans_func

    def update(self, output, label, seq_mask=None):
        if self.trans_func is None:
            if self.vocab is None:
                raise AttributeError(
                    "The `update` method requires users to provide `trans_func` or `vocab` when initializing BLEU."
                )
            cand_list, ref_list = default_trans_func(
                output, label, seq_mask=seq_mask, vocab=self.vocab)
        else:
            cand_list, ref_list = self.trans_func(output, label, seq_mask)
        if len(cand_list) != len(ref_list):
            raise ValueError(
                "Length error! Please check the output of network.")
        for i in range(len(cand_list)):
            self.add_inst(cand_list[i], ref_list[i])

    def add_inst(self, cand, ref_list):
        '''
        Update the states based on a pair of candidate and references.

        Args:
            cand (list): Tokenized candidate sentence.
            ref_list (list of list): List of tokenized ground truth sentences.
        '''
        for n_size in range(self.n_size):
            self.count_ngram(cand, ref_list, n_size)
        self.count_bp(cand, ref_list)

    def count_ngram(self, cand, ref_list, n_size):
        cand_ngram = get_ngram(cand, n_size)
        refs_ngram = []
        for ref in ref_list:
            refs_ngram.append(get_ngram(ref, n_size))
        if n_size not in self.match_ngram:
            self.match_ngram[n_size] = 0
            self.candi_ngram[n_size] = 0
        match_size, cand_size = get_match_size(cand_ngram, refs_ngram)

        self.match_ngram[n_size] += match_size
        self.candi_ngram[n_size] += cand_size

    def count_bp(self, cand, ref_list):
        self.bp_c += len(cand)
        self.bp_r += min([(abs(len(cand) - len(ref)), len(ref))
                          for ref in ref_list])[1]

    def reset(self):
        self.match_ngram = {}
        self.candi_ngram = {}
        self.bp_r = 0
        self.bp_c = 0

    def accumulate(self):
        '''
        Calculates and returns the final bleu metric.

        Returns:
            Tensor: Returns the accumulated metric `bleu` and its data type is float64.
        '''
        prob_list = []
        for n_size in range(self.n_size):
            try:
                if self.candi_ngram[n_size] == 0:
                    _score = 0.0
                else:
                    _score = self.match_ngram[n_size] / float(self.candi_ngram[
                        n_size])
            except:
                _score = 0
            if _score == 0:
                _score = sys.float_info.min
            prob_list.append(_score)

        logs = math.fsum(w_i * math.log(p_i)
                         for w_i, p_i in zip(self.weights, prob_list))
        bp = math.exp(min(1 - self.bp_r / float(self.bp_c), 0))
        bleu = bp * math.exp(logs)
        return bleu

    def score(self):
        return self.accumulate()

    def name(self):
        return self._name


class BLEUForDuReader(BLEU):
    '''
    BLEU metric with bonus for DuReader contest.

    Please refer to `DuReader Homepage<https://ai.baidu.com//broad/subordinate?dataset=dureader>`_ for more details.

    Args:
        n_size (int, optional): Number of gram for BLEU metric. Defaults to 4.
        alpha (float, optional): Weight of YesNo dataset when adding bonus for DuReader contest. Defaults to 1.0.
        beta (float, optional): Weight of Entity dataset when adding bonus for DuReader contest. Defaults to 1.0.

    '''

    def __init__(self, n_size=4, alpha=1.0, beta=1.0):
        super(BLEUForDuReader, self).__init__(n_size)
        self.alpha = alpha
        self.beta = beta

    def add_inst(self,
                 cand,
                 ref_list,
                 yn_label=None,
                 yn_ref=None,
                 entity_ref=None):
        BLEU.add_inst(self, cand, ref_list)
        if yn_label is not None and yn_ref is not None:
            self.add_yn_bonus(cand, ref_list, yn_label, yn_ref)
        elif entity_ref is not None:
            self.add_entity_bonus(cand, entity_ref)

    def add_yn_bonus(self, cand, ref_list, yn_label, yn_ref):
        for n_size in range(self.n_size):
            cand_ngram = get_ngram(cand, n_size, label=yn_label)
            ref_ngram = []
            for ref_id, r in enumerate(yn_ref):
                ref_ngram.append(get_ngram(ref_list[ref_id], n_size, label=r))
            match_size, cand_size = get_match_size(cand_ngram, ref_ngram)
            self.match_ngram[n_size] += self.alpha * match_size
            self.candi_ngram[n_size] += self.alpha * match_size

    def add_entity_bonus(self, cand, entity_ref):
        for n_size in range(self.n_size):
            cand_ngram = get_ngram(cand, n_size, label='ENTITY')
            ref_ngram = []
            for reff_id, r in enumerate(entity_ref):
                ref_ngram.append(get_ngram(r, n_size, label='ENTITY'))
            match_size, cand_size = get_match_size(cand_ngram, ref_ngram)
            self.match_ngram[n_size] += self.beta * match_size
            self.candi_ngram[n_size] += self.beta * match_size
