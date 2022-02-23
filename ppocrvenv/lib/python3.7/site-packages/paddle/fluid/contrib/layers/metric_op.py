# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
"""
Contrib layers just related to metric.
"""

from __future__ import print_function

import warnings
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.initializer import Normal, Constant
from paddle.fluid.framework import Variable
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.layers import nn

__all__ = ['ctr_metric_bundle']


def ctr_metric_bundle(input, label):
    """
    ctr related metric layer

    This function help compute the ctr related metrics: RMSE, MAE, predicted_ctr, q_value.
    To compute the final values of these metrics, we should do following computations using
    total instance number:
    MAE = local_abserr / instance number
    RMSE = sqrt(local_sqrerr / instance number)
    predicted_ctr = local_prob / instance number
    q = local_q / instance number
    Note that if you are doing distribute job, you should all reduce these metrics and instance
    number first

    Args:
        input(Variable): A floating-point 2D Variable, values are in the range
                         [0, 1]. Each row is sorted in descending order. This
                         input should be the output of topk. Typically, this
                         Variable indicates the probability of each label.
        label(Variable): A 2D int Variable indicating the label of the training
                         data. The height is batch size and width is always 1.

    Returns:
        local_sqrerr(Variable): Local sum of squared error
        local_abserr(Variable): Local sum of abs error
        local_prob(Variable): Local sum of predicted ctr
        local_q(Variable): Local sum of q value

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            data = fluid.layers.data(name="data", shape=[32, 32], dtype="float32")
            label = fluid.layers.data(name="label", shape=[1], dtype="int32")
            predict = fluid.layers.sigmoid(fluid.layers.fc(input=data, size=1))
            auc_out = fluid.contrib.layers.ctr_metric_bundle(input=predict, label=label)
    """
    assert input.shape == label.shape
    helper = LayerHelper("ctr_metric_bundle", **locals())

    local_abserr = helper.create_global_variable(
        persistable=True, dtype='float32', shape=[1])
    local_sqrerr = helper.create_global_variable(
        persistable=True, dtype='float32', shape=[1])
    local_prob = helper.create_global_variable(
        persistable=True, dtype='float32', shape=[1])
    local_q = helper.create_global_variable(
        persistable=True, dtype='float32', shape=[1])
    local_pos_num = helper.create_global_variable(
        persistable=True, dtype='float32', shape=[1])
    local_ins_num = helper.create_global_variable(
        persistable=True, dtype='float32', shape=[1])

    tmp_res_elesub = helper.create_global_variable(
        persistable=False, dtype='float32', shape=[-1])
    tmp_res_sigmoid = helper.create_global_variable(
        persistable=False, dtype='float32', shape=[-1])
    tmp_ones = helper.create_global_variable(
        persistable=False, dtype='float32', shape=[-1])

    batch_prob = helper.create_global_variable(
        persistable=False, dtype='float32', shape=[1])
    batch_abserr = helper.create_global_variable(
        persistable=False, dtype='float32', shape=[1])
    batch_sqrerr = helper.create_global_variable(
        persistable=False, dtype='float32', shape=[1])
    batch_q = helper.create_global_variable(
        persistable=False, dtype='float32', shape=[1])
    batch_pos_num = helper.create_global_variable(
        persistable=False, dtype='float32', shape=[1])
    batch_ins_num = helper.create_global_variable(
        persistable=False, dtype='float32', shape=[1])
    for var in [
            local_abserr, batch_abserr, local_sqrerr, batch_sqrerr, local_prob,
            batch_prob, local_q, batch_q, batch_pos_num, batch_ins_num,
            local_pos_num, local_ins_num
    ]:
        helper.set_variable_initializer(
            var, Constant(
                value=0.0, force_cpu=True))

    helper.append_op(
        type="elementwise_sub",
        inputs={"X": [input],
                "Y": [label]},
        outputs={"Out": [tmp_res_elesub]})

    helper.append_op(
        type="squared_l2_norm",
        inputs={"X": [tmp_res_elesub]},
        outputs={"Out": [batch_sqrerr]})
    helper.append_op(
        type="elementwise_add",
        inputs={"X": [batch_sqrerr],
                "Y": [local_sqrerr]},
        outputs={"Out": [local_sqrerr]})

    helper.append_op(
        type="l1_norm",
        inputs={"X": [tmp_res_elesub]},
        outputs={"Out": [batch_abserr]})
    helper.append_op(
        type="elementwise_add",
        inputs={"X": [batch_abserr],
                "Y": [local_abserr]},
        outputs={"Out": [local_abserr]})

    helper.append_op(
        type="reduce_sum", inputs={"X": [input]},
        outputs={"Out": [batch_prob]})
    helper.append_op(
        type="elementwise_add",
        inputs={"X": [batch_prob],
                "Y": [local_prob]},
        outputs={"Out": [local_prob]})
    helper.append_op(
        type="sigmoid",
        inputs={"X": [input]},
        outputs={"Out": [tmp_res_sigmoid]})
    helper.append_op(
        type="reduce_sum",
        inputs={"X": [tmp_res_sigmoid]},
        outputs={"Out": [batch_q]})
    helper.append_op(
        type="elementwise_add",
        inputs={"X": [batch_q],
                "Y": [local_q]},
        outputs={"Out": [local_q]})

    helper.append_op(
        type="reduce_sum",
        inputs={"X": [label]},
        outputs={"Out": [batch_pos_num]})
    helper.append_op(
        type="elementwise_add",
        inputs={"X": [batch_pos_num],
                "Y": [local_pos_num]},
        outputs={"Out": [local_pos_num]})

    helper.append_op(
        type='fill_constant_batch_size_like',
        inputs={"Input": label},
        outputs={'Out': [tmp_ones]},
        attrs={
            'shape': [-1, 1],
            'dtype': tmp_ones.dtype,
            'value': float(1.0),
        })
    helper.append_op(
        type="reduce_sum",
        inputs={"X": [tmp_ones]},
        outputs={"Out": [batch_ins_num]})
    helper.append_op(
        type="elementwise_add",
        inputs={"X": [batch_ins_num],
                "Y": [local_ins_num]},
        outputs={"Out": [local_ins_num]})

    return local_sqrerr, local_abserr, local_prob, local_q, local_pos_num, local_ins_num
