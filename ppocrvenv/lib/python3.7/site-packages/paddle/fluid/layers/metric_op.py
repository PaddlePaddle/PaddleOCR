#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
All layers just related to metric.
"""

from __future__ import print_function

import warnings
from ..layer_helper import LayerHelper
from ..initializer import Normal, Constant
from ..framework import Variable, in_dygraph_mode, _varbase_creator
from .. import core
from ..param_attr import ParamAttr
from . import nn
from ..data_feeder import check_variable_and_dtype
from paddle import _C_ops

__all__ = ['accuracy', 'auc']


def accuracy(input, label, k=1, correct=None, total=None):
    """
    accuracy layer.
    Refer to the https://en.wikipedia.org/wiki/Precision_and_recall

    This function computes the accuracy using the input and label.
    If the correct label occurs in top k predictions, then correct will increment by one.
    Note: the dtype of accuracy is determined by input. the input and label dtype can be different.

    Args:
        input(Variable): The input of accuracy layer, which is the predictions of network. A LoDTensor or Tensor with type float32,float64.
            The shape is ``[sample_number, class_dim]`` .
        label(Variable): The label of dataset.  LoDTensor or Tensor with type int32,int64. The shape is ``[sample_number, 1]`` .
        k(int): The top k predictions for each class will be checked. Data type is int64 or int32.
        correct(Variable): The correct predictions count. A Tensor with type int64 or int32.
        total(Variable): The total entries count. A tensor with type int64 or int32.

    Returns:
        Variable: The correct rate. A Tensor with type float32.

    Examples:
        .. code-block:: python

            import numpy as np

            import paddle
            import paddle.static as static
            import paddle.nn.functional as F

            paddle.enable_static()
            data = static.data(name="input", shape=[-1, 32, 32], dtype="float32")
            label = static.data(name="label", shape=[-1,1], dtype="int")
            fc_out = static.nn.fc(x=data, size=10)
            predict = F.softmax(x=fc_out)
            result = static.accuracy(input=predict, label=label, k=5)

            place = paddle.CPUPlace()
            exe = static.Executor(place)

            exe.run(static.default_startup_program())
            x = np.random.rand(3, 32, 32).astype("float32")
            y = np.array([[1],[0],[1]])
            output= exe.run(feed={"input": x,"label": y},
                        fetch_list=[result[0]])
            print(output)

            #[array([0.], dtype=float32)]
    """
    if in_dygraph_mode():
        if correct is None:
            correct = _varbase_creator(dtype="int32")
        if total is None:
            total = _varbase_creator(dtype="int32")

        _k = k.numpy().item(0) if isinstance(k, Variable) else k
        topk_out, topk_indices = _C_ops.top_k_v2(input, 'k', _k, 'sorted',
                                                 False)
        _acc, _, _ = _C_ops.accuracy(topk_out, topk_indices, label, correct,
                                     total)
        return _acc

    helper = LayerHelper("accuracy", **locals())
    check_variable_and_dtype(input, 'input', ['float16', 'float32', 'float64'],
                             'accuracy')
    topk_out = helper.create_variable_for_type_inference(dtype=input.dtype)
    topk_indices = helper.create_variable_for_type_inference(dtype="int64")
    inputs = {"X": [input]}
    if isinstance(k, Variable):
        inputs['K'] = [k]
    else:
        attrs = {'k': k}
    attrs['sorted'] = False
    helper.append_op(
        type="top_k_v2",
        inputs=inputs,
        attrs=attrs,
        outputs={"Out": [topk_out],
                 "Indices": [topk_indices]})
    acc_out = helper.create_variable_for_type_inference(dtype="float32")
    if correct is None:
        correct = helper.create_variable_for_type_inference(dtype="int32")
    if total is None:
        total = helper.create_variable_for_type_inference(dtype="int32")
    helper.append_op(
        type="accuracy",
        inputs={
            "Out": [topk_out],
            "Indices": [topk_indices],
            "Label": [label]
        },
        outputs={
            "Accuracy": [acc_out],
            "Correct": [correct],
            "Total": [total],
        })
    return acc_out


def auc(input,
        label,
        curve='ROC',
        num_thresholds=2**12 - 1,
        topk=1,
        slide_steps=1):
    r"""
    **Area Under the Curve (AUC) Layer**

    This implementation computes the AUC according to forward output and label.
    It is used very widely in binary classification evaluation.

    Note: If input label contains values other than 0 and 1, it will be cast
    to `bool`. Find the relevant definitions `here <https://en.wikipedia.org\
    /wiki/Receiver_operating_characteristic#Area_under_the_curve>`_.

    There are two types of possible curves:

        1. ROC: Receiver operating characteristic;
        2. PR: Precision Recall

    Args:
        input(Variable): A floating-point 2D Variable, values are in the range
                         [0, 1]. Each row is sorted in descending order. This
                         input should be the output of topk. Typically, this
                         Variable indicates the probability of each label.
                         A LoDTensor or Tensor with type float32,float64.
        label(Variable): A 2D int Variable indicating the label of the training
                         data. The height is batch size and width is always 1.
                         A LoDTensor or Tensor with type int32,int64.
        curve(str): Curve type, can be 'ROC' or 'PR'. Default 'ROC'.
        num_thresholds(int): The number of thresholds to use when discretizing
                             the roc curve. Default 200.
        topk(int): only topk number of prediction output will be used for auc.
        slide_steps: when calc batch auc, we can not only use step currently but the previous steps can be used. slide_steps=1 means use the current step, slide_steps=3 means use current step and the previous second steps, slide_steps=0 use all of the steps.


    Returns:
        Variable: A tuple representing the current AUC.
        The return tuple is auc_out, batch_auc_out, [
        batch_stat_pos, batch_stat_neg, stat_pos, stat_neg ]
        Data type is Tensor, supporting float32, float64.

    Examples:
        .. code-block:: python

            import numpy as np

            import paddle
            import paddle.static as static
            import paddle.nn.functional as F

            paddle.enable_static()
            data = static.data(name="input", shape=[-1, 32,32], dtype="float32")
            label = static.data(name="label", shape=[-1], dtype="int")
            fc_out = static.nn.fc(x=data, size=2)
            predict = F.softmax(x=fc_out)
            result = static.auc(input=predict, label=label)

            place = paddle.CPUPlace()
            exe = static.Executor(place)

            exe.run(static.default_startup_program())
            x = np.random.rand(3,32,32).astype("float32")
            y = np.array([1,0,1])
            output= exe.run(feed={"input": x,"label": y},
                        fetch_list=[result[0]])
            print(output)
            #[array([0.])]
    """
    helper = LayerHelper("auc", **locals())
    check_variable_and_dtype(input, 'input', ['float32', 'float64'], 'auc')
    check_variable_and_dtype(label, 'label', ['int32', 'int64'], 'auc')
    auc_out = helper.create_variable_for_type_inference(dtype="float64")
    batch_auc_out = helper.create_variable_for_type_inference(dtype="float64")
    # make tp, tn, fp, fn persistable, so that can accumulate all batches.

    # for batch auc
    # we create slide_step+1 buckets, the first slide_steps buckets store 
    # historical batch-level values, and the last bucket stores the sum values of 
    # previous slide_step buckets.
    # The index of bucket that the newest batch will use is determined by batch_id mod slide_steps,
    # and batch_id is store in the last posision of following variable
    batch_stat_pos = helper.create_global_variable(
        persistable=True,
        dtype='int64',
        shape=[(1 + slide_steps) * (num_thresholds + 1) + 1])
    batch_stat_neg = helper.create_global_variable(
        persistable=True,
        dtype='int64',
        shape=[(1 + slide_steps) * (num_thresholds + 1) + 1])

    # for global auc
    # Needn't maintain the batch id
    stat_pos = helper.create_global_variable(
        persistable=True, dtype='int64', shape=[1, num_thresholds + 1])
    stat_neg = helper.create_global_variable(
        persistable=True, dtype='int64', shape=[1, num_thresholds + 1])

    for var in [batch_stat_pos, batch_stat_neg, stat_pos, stat_neg]:
        helper.set_variable_initializer(
            var, Constant(
                value=0.0, force_cpu=False))

    # Batch AUC
    helper.append_op(
        type="auc",
        inputs={
            "Predict": [input],
            "Label": [label],
            "StatPos": [batch_stat_pos],
            "StatNeg": [batch_stat_neg]
        },
        attrs={
            "curve": curve,
            "num_thresholds": num_thresholds,
            "slide_steps": slide_steps
        },
        outputs={
            "AUC": [batch_auc_out],
            "StatPosOut": [batch_stat_pos],
            "StatNegOut": [batch_stat_neg]
        })
    # Global AUC
    helper.append_op(
        type="auc",
        inputs={
            "Predict": [input],
            "Label": [label],
            "StatPos": [stat_pos],
            "StatNeg": [stat_neg]
        },
        attrs={
            "curve": curve,
            "num_thresholds": num_thresholds,
            "slide_steps": 0
        },
        outputs={
            "AUC": [auc_out],
            "StatPosOut": [stat_pos],
            "StatNegOut": [stat_neg]
        })
    return auc_out, batch_auc_out, [
        batch_stat_pos, batch_stat_neg, stat_pos, stat_neg
    ]
