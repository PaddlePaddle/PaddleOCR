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

import paddle
import numpy as np

import paddle.nn.functional as F


class Perplexity(paddle.metric.Metric):
    """
    Perplexity is a metric used to judge how good a language model is.
    We can define perplexity as the inverse probability of the test set,
    normalised by the number of the words in the test set.
    Perplexity is calculated using cross entropy. It supports both padding data
    and no padding data.

    If data is not padded, users should provide `seq_len` for `Metric`
    initialization. If data is padded, your label should contain `seq_mask`,
    which indicates the actual length of samples.

    This Perplexity requires that the output of your network is prediction,
    label and sequence length (optional). If the Perplexity here doesn't meet
    your needs, you could override the `compute` or `update` method for
    calculating Perplexity.

    Args:
        seq_len(int): Sequence length of each sample, it must be provided while
            data is not padded. Defaults to 20.
        name(str): Name of `Metric` instance. Defaults to 'Perplexity'.

    Example:
        .. code-block::

            import paddle
            from paddlenlp.transformers import BertTokenizer
            from paddlenlp.metrics import Perplexity

            paddle.seed(2021)
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            batch_size, seq_len, vocab_size = 1, 4, tokenizer.vocab_size
            logits = paddle.rand([batch_size, seq_len, vocab_size])
            labels= paddle.to_tensor([[1,0,1,1]])

            perplexity = Perplexity()
            correct = perplexity.compute(logits,labels)
            perplexity.update(correct.numpy())
            res = perplexity.accumulate()
            print(res)
            # 48263.528820122105
    """

    def __init__(self, name='Perplexity', *args, **kwargs):
        super(Perplexity, self).__init__(*args, **kwargs)
        self._name = name
        self.total_ce = 0
        self.total_word_num = 0

    def compute(self, pred, label, seq_mask=None):
        """
        Computes cross entropy loss.

        Args:
            pred (Tensor):
                Predictor tensor, and its dtype is float32 or float64, and has
                a shape of [batch_size, sequence_length, vocab_size].
            label(Tensor):
                Label tensor, and its dtype is int64, and has a shape of
                [batch_size, sequence_length, 1] or [batch_size, sequence_length].
            seq_mask(Tensor, optional):
                Sequence mask tensor, and its type could be float32, float64,
                int32 or int64, and has a shape of [batch_size, sequence_length].
                It's used to calculate loss. Defaults to None.

        Returns:
            tuple or Tensor: Returns tuple (`ce, word_num`) if `seq_mask` is not None. Otherwise, returns tensor `ce`.
            `ce` it the cross entropy loss, its shape is [batch_size, sequence_length] and its data type should be float32.

        """
        if label.dim() == 2:
            label = paddle.unsqueeze(label, axis=2)
        ce = F.cross_entropy(
            input=pred, label=label, reduction='none', soft_label=False)
        ce = paddle.squeeze(ce, axis=[2])
        if seq_mask is not None:
            ce = ce * seq_mask
            word_num = paddle.sum(seq_mask)
            return ce, word_num
        return ce

    def update(self, ce, word_num=None):
        """
        Updates metric states.

        Args:
            ce (numpy.ndarray):
                Cross entropy loss, it's calculated by `compute` and converted
                to `numpy.ndarray`.
            word_num (numpy.ndarray):
               The number of words of sequence, it's calculated by `compute`
               and converted to `numpy.ndarray`. Defaults to None.

        """
        batch_ce = np.sum(ce)
        if word_num is None:
            word_num = ce.shape[0] * ce.shape[1]
        else:
            word_num = word_num[0]
        self.total_ce += batch_ce
        self.total_word_num += word_num

    def reset(self):
        """
        Resets all metric states.
        """
        self.total_ce = 0
        self.total_word_num = 0

    def accumulate(self):
        """
        Calculates and returns the value of perplexity.

        Returns:
            float: Returns `perplexity`, the calculation results.
        """
        return np.exp(self.total_ce / self.total_word_num)

    def name(self):
        """
        Returns name of the metric instance.

        Returns:
           str: The name of the metric instance.

        """
        return self._name
