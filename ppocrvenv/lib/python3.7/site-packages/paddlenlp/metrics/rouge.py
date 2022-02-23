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

import numpy as np

import paddle
from .utils import default_trans_func

__all__ = ['RougeL', 'RougeLForDuReader']


class RougeN():
    def __init__(self, n):
        self.n = n

    def _get_ngrams(self, words):
        """Calculates word n-grams for multiple sentences.
        """
        ngram_set = set()
        max_index_ngram_start = len(words) - self.n
        for i in range(max_index_ngram_start + 1):
            ngram_set.add(tuple(words[i:i + self.n]))
        return ngram_set

    def score(self, evaluated_sentences_ids, reference_sentences_ids):
        overlapping_count, reference_count = self.compute(
            evaluated_sentences_ids, reference_sentences_ids)
        return overlapping_count / reference_count

    def compute(self, evaluated_sentences_ids, reference_sentences_ids):
        """
        Args:
            evaluated_sentences (list): the sentences ids predicted by the model.
            reference_sentences (list): the referenced sentences ids. Its size should be same as evaluated_sentences.

        Returns:
            overlapping_count (int): the overlapping n-gram count.
            reference_count (int): the reference sentences n-gram count. 
        """
        if len(evaluated_sentences_ids) <= 0 or len(
                reference_sentences_ids) <= 0:
            raise ValueError("Collections must contain at least 1 sentence.")

        reference_count = 0
        overlapping_count = 0

        for evaluated_sentence_ids, reference_sentence_ids in zip(
                evaluated_sentences_ids, reference_sentences_ids):
            evaluated_ngrams = self._get_ngrams(evaluated_sentence_ids)
            reference_ngrams = self._get_ngrams(reference_sentence_ids)
            reference_count += len(reference_ngrams)

            # Gets the overlapping ngrams between evaluated and reference
            overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
            overlapping_count += len(overlapping_ngrams)

        return overlapping_count, reference_count

    def accumulate(self):
        """
        This function returns the mean precision, recall and f1 score for all accumulated minibatches.

        Returns:
            float: mean precision, recall and f1 score.
        """
        rouge_score = self.overlapping_count / self.reference_count
        return rouge_score

    def reset(self):
        """
        Reset function empties the evaluation memory for previous mini-batches.
        """
        self.overlapping_count = 0
        self.reference_count = 0

    def name(self):
        """
        Return name of metric instance.
        """
        return "Rouge-%s" % self.n

    def update(self, overlapping_count, reference_count):
        """
        Args:
        """
        self.overlapping_count += overlapping_count
        self.reference_count += reference_count


class Rouge1(RougeN):
    def __init__(self):
        super(Rouge1, self).__init__(n=1)


class Rouge2(RougeN):
    def __init__(self):
        super(Rouge2, self).__init__(n=2)


class RougeL(paddle.metric.Metric):
    r'''
    Rouge-L is Recall-Oriented Understudy for Gisting Evaluation based on Longest Common Subsequence (LCS).
    Longest common subsequence problem takes into account sentence level structure
    similarity naturally and identifies longest co-occurring
    in sequence n-grams automatically.

    .. math::

        R_{LCS} & = \frac{LCS(C,S)}{len(S)}

        P_{LCS} & = \frac{LCS(C,S)}{len(C)}

        F_{LCS} & = \frac{(1 + \gamma^2)R_{LCS}P_{LCS}}{R_{LCS}} + \gamma^2{R_{LCS}}

    where `C` is the candidate sentence, and `S` is the reference sentence.

    Args:
        trans_func (callable, optional): `trans_func` transforms the network
            output to string to calculate.
        vocab (dict|paddlenlp.data.vocab, optional): Vocab for target language.
            If `trans_func` is None and RougeL is used as `paddle.metric.Metric`
            instance, `default_trans_func` will be performed and `vocab` must
            be provided.
        gamma (float): A hyperparameter to decide the weight of recall. Defaults to 1.2.
        name (str, optional): Name of `paddle.metric.Metric` instance. Defaults to "rouge-l".

    Examples:
        .. code-block:: python

            from paddlenlp.metrics import RougeL
            rougel = RougeL()
            cand = ["The","cat","The","cat","on","the","mat"]
            ref_list = [["The","cat","is","on","the","mat"], ["There","is","a","cat","on","the","mat"]]
            rougel.add_inst(cand, ref_list)
            print(rougel.score()) # 0.7800511508951408

    '''

    def __init__(self,
                 trans_func=None,
                 vocab=None,
                 gamma=1.2,
                 name="rouge-l",
                 *args,
                 **kwargs):
        super(RougeL, self).__init__(*args, **kwargs)
        self.gamma = gamma
        self.inst_scores = []
        self._name = name
        self.vocab = vocab
        self.trans_func = trans_func

    def lcs(self, string, sub):
        """
        Calculate the length of longest common subsequence of string and sub.

        Args:
            string (str):
                The string to be calculated, usually longer the sub string.
            sub (str):
                The sub string to be calculated.

        Returns:
            float: Returns the length of the longest common subsequence of string and sub.
        """
        if len(string) < len(sub):
            sub, string = string, sub
        lengths = np.zeros((len(string) + 1, len(sub) + 1))
        for j in range(1, len(sub) + 1):
            for i in range(1, len(string) + 1):
                if string[i - 1] == sub[j - 1]:
                    lengths[i][j] = lengths[i - 1][j - 1] + 1
                else:
                    lengths[i][j] = max(lengths[i - 1][j], lengths[i][j - 1])
        return lengths[len(string)][len(sub)]

    def add_inst(self, cand, ref_list):
        '''
        Update the states based on the a pair of candidate and references.

        Args:
            cand (str): The candidate sentence generated by model.
            ref_list (list): List of ground truth sentences.
        '''
        precs, recalls = [], []
        for ref in ref_list:
            basic_lcs = self.lcs(cand, ref)
            prec = basic_lcs / len(cand) if len(cand) > 0. else 0.
            rec = basic_lcs / len(ref) if len(ref) > 0. else 0.
            precs.append(prec)
            recalls.append(rec)

        prec_max = max(precs)
        rec_max = max(recalls)

        if prec_max != 0 and rec_max != 0:
            score = ((1 + self.gamma**2) * prec_max * rec_max) / \
                    float(rec_max + self.gamma**2 * prec_max)
        else:
            score = 0.0
        self.inst_scores.append(score)

    def update(self, output, label, seq_mask=None):
        if self.trans_func is None:
            if self.vocab is None:
                raise AttributeError(
                    "The `update` method requires users to provide `trans_func` or `vocab` when initializing RougeL."
                )
            cand_list, ref_list = default_trans_func(output, label, seq_mask,
                                                     self.vocab)
        else:
            cand_list, ref_list = self.trans_func(output, label, seq_mask)
        if len(cand_list) != len(ref_list):
            raise ValueError(
                "Length error! Please check the output of network.")
        for i in range(len(cand_list)):
            self.add_inst(cand_list[i], ref_list[i])

    def accumulate(self):
        '''
        Calculate the final rouge-l metric.
        '''
        return 1. * sum(self.inst_scores) / len(self.inst_scores)

    def score(self):
        return self.accumulate()

    def reset(self):
        self.inst_scores = []

    def name(self):
        return self._name


class RougeLForDuReader(RougeL):
    '''
    Rouge-L metric with bonus for DuReader contest.

    Please refer to `DuReader Homepage<https://ai.baidu.com//broad/subordinate?dataset=dureader>`_ for more details.

    Args:
        alpha (float, optional): Weight of YesNo dataset when adding bonus for DuReader contest. Defaults to 1.0.
        beta (float, optional): Weight of Entity dataset when adding bonus for DuReader contest. Defaults to 1.0.
    '''

    def __init__(self, alpha=1.0, beta=1.0, gamma=1.2):
        super(RougeLForDuReader, self).__init__(gamma)
        self.alpha = alpha
        self.beta = beta

    def add_inst(self,
                 cand,
                 ref_list,
                 yn_label=None,
                 yn_ref=None,
                 entity_ref=None):
        precs, recalls = [], []
        for i, ref in enumerate(ref_list):
            basic_lcs = self.lcs(cand, ref)
            yn_bonus, entity_bonus = 0.0, 0.0
            if yn_ref is not None and yn_label is not None:
                yn_bonus = self.add_yn_bonus(cand, ref, yn_label, yn_ref[i])
            elif entity_ref is not None:
                entity_bonus = self.add_entity_bonus(cand, entity_ref)
            p_denom = len(
                cand) + self.alpha * yn_bonus + self.beta * entity_bonus
            r_denom = len(
                ref) + self.alpha * yn_bonus + self.beta * entity_bonus
            prec = (basic_lcs + self.alpha * yn_bonus + self.beta * entity_bonus) \
                    / p_denom if p_denom > 0. else 0.
            rec = (basic_lcs + self.alpha * yn_bonus + self.beta * entity_bonus) \
                    / r_denom if r_denom > 0. else 0.
            precs.append(prec)
            recalls.append(rec)

        prec_max = max(precs)
        rec_max = max(recalls)
        if prec_max != 0 and rec_max != 0:
            score = ((1 + self.gamma**2) * prec_max * rec_max) / \
                    float(rec_max + self.gamma**2 * prec_max)
        else:
            score = 0.0
        self.inst_scores.append(score)

    def add_yn_bonus(self, cand, ref, yn_label, yn_ref):
        if yn_label != yn_ref:
            return 0.0
        lcs_ = self.lcs(cand, ref)
        return lcs_

    def add_entity_bonus(self, cand, entity_ref):
        lcs_ = 0.0
        for ent in entity_ref:
            if ent in cand:
                lcs_ += len(ent)
        return lcs_
