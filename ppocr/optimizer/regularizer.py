# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import paddle


class L1Decay(object):
    """
    L1 Weight Decay Regularization, which encourages the weights to be sparse.
    Args:
        factor(float): regularization coeff. Default:0.0.
    """

    def __init__(self, factor=0.0):
        super(L1Decay, self).__init__()
        self.coeff = factor

    def __call__(self):
        reg = paddle.regularizer.L1Decay(self.coeff)
        return reg


class L2Decay(object):
    """
    L2 Weight Decay Regularization, which helps to prevent the model over-fitting.
    Args:
        factor(float): regularization coeff. Default:0.0.
    """

    def __init__(self, factor=0.0):
        super(L2Decay, self).__init__()
        self.coeff = float(factor)

    def __call__(self):
        return self.coeff


class CosineL2Decay(object):
    """
    L2 Weight Decay with cosine annealing schedule.

    Anneals the weight decay coefficient from `factor` to `end_factor`
    following a cosine curve over total training steps, with optional
    linear warmup. Avoids over-regularizing small-capacity models.

    Reference: EfficientNetV2 (Tan & Le, 2021) - "annealing the loss
    incurred by weight decay regularization over the course of training".

    Args:
        factor(float): initial weight decay coefficient.
        end_factor(float): final weight decay coefficient. Default: 0.0.
        warmup_epoch(int|float): warmup epochs (same as lr warmup). Default: 0.
    """

    def __init__(self, factor=5e-5, end_factor=0.0, warmup_epoch=0):
        super(CosineL2Decay, self).__init__()
        self.start_factor = float(factor)
        self.end_factor = float(end_factor)
        self.warmup_epoch = warmup_epoch

    def __call__(self):
        return self.start_factor
