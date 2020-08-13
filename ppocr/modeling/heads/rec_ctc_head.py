#copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import paddle
import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from .rec_seq_encoder import SequenceEncoder
from ..common_functions import get_para_bias_attr
import numpy as np


class CTCPredict(object):
    def __init__(self, params):
        super(CTCPredict, self).__init__()
        self.char_num = params['char_num']
        self.encoder = SequenceEncoder(params)
        self.encoder_type = params['encoder_type']
        self.fc_decay = params.get("fc_decay", 0.0004)

    def __call__(self, inputs, labels=None, mode=None):
        encoder_features = self.encoder(inputs)
        if self.encoder_type != "reshape":
            encoder_features = fluid.layers.concat(encoder_features, axis=1)
        name = "ctc_fc"
        para_attr, bias_attr = get_para_bias_attr(
            l2_decay=self.fc_decay, k=encoder_features.shape[1], name=name)
        predict = fluid.layers.fc(input=encoder_features,
                                  size=self.char_num + 1,
                                  param_attr=para_attr,
                                  bias_attr=bias_attr,
                                  name=name)
        decoded_out = fluid.layers.ctc_greedy_decoder(
            input=predict, blank=self.char_num)
        predicts = {'predict': predict, 'decoded_out': decoded_out}
        return predicts
