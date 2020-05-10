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
import paddle.fluid as fluid
import paddle.fluid.layers as layers


class EncoderWithReshape(object):
    def __init__(self, params):
        super(EncoderWithReshape, self).__init__()

    def __call__(self, inputs):
        sliced_feature = layers.im2sequence(
            input=inputs,
            stride=[1, 1],
            filter_size=[inputs.shape[2], 1],
            name="sliced_feature")
        return sliced_feature


class EncoderWithRNN(object):
    def __init__(self, params):
        super(EncoderWithRNN, self).__init__()
        self.rnn_hidden_size = params['SeqRNN']['hidden_size']

    def __call__(self, inputs):
        lstm_list = []
        name_prefix = "lstm"
        rnn_hidden_size = self.rnn_hidden_size
        for no in range(1, 3):
            if no == 1:
                is_reverse = False
            else:
                is_reverse = True
            name = "%s_st1_fc%d" % (name_prefix, no)
            fc = layers.fc(input=inputs,
                           size=rnn_hidden_size * 4,
                           param_attr=fluid.ParamAttr(name=name + "_w"),
                           bias_attr=fluid.ParamAttr(name=name + "_b"),
                           name=name)
            name = "%s_st1_out%d" % (name_prefix, no)
            lstm, _ = layers.dynamic_lstm(
                input=fc,
                size=rnn_hidden_size * 4,
                is_reverse=is_reverse,
                param_attr=fluid.ParamAttr(name=name + "_w"),
                bias_attr=fluid.ParamAttr(name=name + "_b"),
                use_peepholes=False)
            name = "%s_st2_fc%d" % (name_prefix, no)
            fc = layers.fc(input=lstm,
                           size=rnn_hidden_size * 4,
                           param_attr=fluid.ParamAttr(name=name + "_w"),
                           bias_attr=fluid.ParamAttr(name=name + "_b"),
                           name=name)
            name = "%s_st2_out%d" % (name_prefix, no)
            lstm, _ = layers.dynamic_lstm(
                input=fc,
                size=rnn_hidden_size * 4,
                is_reverse=is_reverse,
                param_attr=fluid.ParamAttr(name=name + "_w"),
                bias_attr=fluid.ParamAttr(name=name + "_b"),
                use_peepholes=False)
            lstm_list.append(lstm)
        return lstm_list


class SequenceEncoder(object):
    def __init__(self, params):
        super(SequenceEncoder, self).__init__()
        self.encoder_type = params['encoder_type']
        self.encoder_reshape = EncoderWithReshape(params)
        if self.encoder_type == "rnn":
            self.encoder_rnn = EncoderWithRNN(params)

    def __call__(self, inputs):
        if self.encoder_type == "reshape":
            encoder_features = self.encoder_reshape(inputs)
        elif self.encoder_type == "rnn":
            inputs = self.encoder_reshape(inputs)
            encoder_features = self.encoder_rnn(inputs)
        else:
            assert False, "Unsupport encoder_type:%s"\
                % self.encoder_type
        return encoder_features
