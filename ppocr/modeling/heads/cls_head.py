#copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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


class ClsHead(object):
    """
    Class orientation

    Args:

        params(dict): super parameters for build Class network
    """

    def __init__(self, params):
        super(ClsHead, self).__init__()
        self.class_dim = params['class_dim']

    def __call__(self, inputs, labels=None, mode=None):
        pool = fluid.layers.pool2d(
            input=inputs, pool_type='avg', global_pooling=True)
        stdv = 1.0 / math.sqrt(pool.shape[1] * 1.0)

        out = fluid.layers.fc(
            input=pool,
            size=self.class_dim,
            param_attr=fluid.param_attr.ParamAttr(
                name="fc_0.w_0",
                initializer=fluid.initializer.Uniform(-stdv, stdv)),
            bias_attr=fluid.param_attr.ParamAttr(name="fc_0.b_0"))

        softmax_out = fluid.layers.softmax(out, use_cudnn=False)
        out_label = fluid.layers.argmax(out, axis=1)
        predicts = {'predict': softmax_out, 'decoded_out': out_label}
        return predicts
