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

from __future__ import print_function

import numpy as np
from functools import partial, reduce
import paddle
from paddle.utils import deprecated
from . import nn
from .layer_function_generator import templatedoc
from ..layer_helper import LayerHelper
from ..framework import Variable, in_dygraph_mode, static_only
from .. import core
from ..data_feeder import check_variable_and_dtype, check_type
from ..param_attr import ParamAttr
from ..initializer import NumpyArrayInitializer, Constant
from .. import core
import warnings
from paddle import _C_ops

__all__ = [
    'center_loss',
    'bpr_loss',
    'cross_entropy',
    'square_error_cost',
    'edit_distance',
    'warpctc',
    'nce',
    'hsigmoid',
    'sampled_softmax_with_cross_entropy',
    'softmax_with_cross_entropy',
    'rank_loss',
    'margin_rank_loss',
    'sigmoid_cross_entropy_with_logits',
    'teacher_student_sigmoid_loss',
    'huber_loss',
    'kldiv_loss',
    'npair_loss',
    'mse_loss',
]

kIgnoreIndex = -100


def center_loss(input,
                label,
                num_classes,
                alpha,
                param_attr,
                update_center=True):
    r"""
    :api_attr: Static Graph

    **Center loss Cost layer**
    
    This OP accepts input (deep features,the output of the last hidden layer)
    and target label and return the center loss cost. The average of the 
    distances of each sample in the mini-batch from the center of the 
    corresponding category is calculated as the center loss.
    
    For deep features, :math:`X`, and target labels, :math:`Y`, the equation is:
    
    .. math::

        Out = \\frac{1}{2}(X - Y)^2

    Args:
        input (Variable): a 2-D tensor with shape[N x M]. Its dtype should be float32 or float64.
        label (Variable): the groud truth which is a 2-D tensor
                         with shape[N x 1],where N is the batch size. Its dtype should be int32.
        num_classes (int): the number of classification categories.
        alpha (float|Variable): learning rate of centers.
        param_attr (ParamAttr): Attribute initializer of centers. 
        update_center (bool): whether to update value of center.

    Returns:
        Variable: 2-D tensor with shape [N * 1] 

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid 
          import paddle
          paddle.enable_static()

          input = fluid.data(name='x',shape=[20,30],dtype='float32')
          label = fluid.data(name='y',shape=[20,1],dtype='int64')
          num_classes = 1000
          alpha = 0.01
          param_attr = fluid.initializer.Xavier(uniform=False)
          center_loss=fluid.layers.center_loss(input=input,
                 label=label,
                 num_classes=1000,
                 alpha=alpha,
                 param_attr=fluid.initializer.Xavier(uniform=False),
                 update_center=True)
    """
    helper = LayerHelper('center_loss', **locals())
    dtype = helper.input_dtype()
    check_variable_and_dtype(input, 'input', ['float32', 'float64'],
                             'center_loss')
    check_variable_and_dtype(label, 'label', ['int32', 'int64'], 'center_loss')

    centers_shape = [num_classes, input.shape[1]]
    centers_param = helper.create_parameter(
        attr=param_attr, shape=centers_shape, dtype=dtype)
    centers_param.stop_gradient = True

    if isinstance(alpha, Variable):
        alpha_param = alpha
        check_variable_and_dtype(alpha, 'alpha', ['float32', 'float64'],
                                 'center_loss')
    else:
        assert isinstance(alpha, float)
        alpha_param = helper.create_variable(
            name="centerloss_alpha",
            shape=[1],
            dtype="float32",
            type=core.VarDesc.VarType.LOD_TENSOR,
            persistable=True,
            stop_gradient=True,
            initializer=Constant(alpha))

    centersdiff = helper.create_variable_for_type_inference(dtype=input.dtype)
    loss = helper.create_variable_for_type_inference(dtype=input.dtype)
    helper.append_op(
        type='center_loss',
        inputs={
            'X': [input],
            'Label': [label],
            'Centers': [centers_param],
            'CenterUpdateRate': [alpha_param]
        },
        outputs={
            'SampleCenterDiff': [centersdiff],
            'Loss': [loss],
            'CentersOut': [centers_param]
        },
        attrs={'cluster_num': num_classes,
               'need_update': update_center})
    return loss


def bpr_loss(input, label, name=None):
    r"""

    **Bayesian Personalized Ranking Loss Operator**

    This operator belongs to pairwise ranking loss. Label is the desired item.
    The loss at a given point in one session is defined as:

    .. math::
        Y[i] = 1/(N[i] - 1) * \sum_j{\log(\sigma(X[i, Label[i]]-X[i, j]))}

    Learn more details by reading paper <session-based recommendations with recurrent
    neural networks>.

    Args:
        input (Variable|list):  a 2-D tensor with shape [N x D], where N is the
                                batch size and D is the number of positive classes and negative classes
                                This input is not probability but logits.
        label (Variable|list):  the ground truth which is a 2-D tensor.  `label`
                                is a tensor<int64> with shape [N x 1].
        name (str|None):        A name for this layer(optional). If set None, the
                                layer will be named automatically. Default: None.
    Returns:
        A 2-D tensor with shape [N x 1], the bpr loss.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          import paddle

          paddle.enable_static()

          neg_size = 10
          label = fluid.data(
                    name="label", shape=[3, 1], dtype="int64")
          predict = fluid.data(
                    name="predict", shape=[3, neg_size + 1], dtype="float32")
          cost = fluid.layers.bpr_loss(input=predict, label=label)
    """
    helper = LayerHelper('bpr_loss', **locals())
    out = helper.create_variable_for_type_inference(dtype=input.dtype)
    check_variable_and_dtype(input, 'input', ['float16', 'float32', 'float64'],
                             'bpr_loss')
    helper.append_op(
        type='bpr_loss',
        inputs={'X': [input],
                'Label': [label]},
        outputs={'Y': [out]})
    return out


def cross_entropy(input, label, soft_label=False, ignore_index=kIgnoreIndex):
    r"""
    :alias_main: paddle.nn.functional.cross_entropy
	:alias: paddle.nn.functional.cross_entropy,paddle.nn.functional.loss.cross_entropy
	:old_api: paddle.fluid.layers.cross_entropy

    This operator computes the cross entropy between input and label. It
    supports both hard-label and and soft-label cross entropy computation.

    1. Hard-label cross entropy: if soft_label=False, :math:`label[i_1, i_2, ..., i_k]`
       is the hard label of each sample.

        .. math::

           output[i_1, i_2, ..., i_k]=-log(input[i_1, i_2, ..., i_k, j]), label[i_1, i_2, ..., i_k] = j, j != ignore\_index

    2. Soft-label cross entropy: if soft_label=True,  :math:`label[i_1, i_2, ..., i_k, j]`
       is the soft label of each sample corresponding to the j-th class.

        .. math::

           output[i_1, i_2, ..., i_k]= -\sum_{j}label[i_1,i_2,...,i_k,j]*log(input[i_1, i_2, ..., i_k,j])

    Args:
        input (Variable): a multidimensional Tensor with shape
                :math:`[N_1, N_2, ..., N_k, D]`, where the last dimension D is
                the class number. The data type should be float32 or float64.
        label (Variable): label value corresponding to input. If
                soft_label=False, the dimension of label should be :math:`[N_1, N_2, ..., N_k]`
                or :math:`[N_1, N_2, ..., N_k, 1]` , and its data type should be int64,
                and the value must be inside [0, D). If soft_label=True, the shape,
                data type of label should be the same with input, and the sum of
                soft label value of each sample should be 1.
        soft_label (bool): indicate whether label is soft. Default False, meaning that
                the label is hard. If soft_label=True, the label is soft.
        ignore_index (int): specify an ignorable label value. The ignored label would be
                omitted when computing. If it is a negative integer, no label would
                be ignored. Only valid when soft_label=False. Default -100.

    Returns:
         A Variable holding Tensor representing the cross entropy, whose data type is the same with input.
         If soft_label=False, the shape of output is the same with label.
         If soft_label=True, the shape of output is :math:`[N_1, N_2, ..., N_k, 1]` .

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            class_num = 7
            x = fluid.data(name='x', shape=[None, 3, 10], dtype='float32')
            label = fluid.data(name='label', shape=[None, 1], dtype='int64')
            predict = fluid.layers.fc(input=x, size=class_num, act='softmax')
            cost = fluid.layers.cross_entropy(input=predict, label=label)
    """
    if not soft_label:
        return cross_entropy2(input, label, ignore_index)

    if in_dygraph_mode():
        return _C_ops.cross_entropy(input, label, "soft_label", soft_label,
                                    "ignore_index", ignore_index)

    inputs = {'X': [input], 'Label': [label]}
    attrs = {"soft_label": soft_label, "ignore_index": ignore_index}

    check_variable_and_dtype(input, 'input', ['float16', 'float32', 'float64'],
                             'cross_entropy')
    helper = LayerHelper('cross_entropy', **locals())
    out = helper.create_variable_for_type_inference(dtype=input.dtype)
    helper.append_op(
        type='cross_entropy', inputs=inputs, outputs={'Y': [out]}, attrs=attrs)
    return out


def cross_entropy2(input, label, ignore_index=kIgnoreIndex):
    if in_dygraph_mode():
        loss, _, _ = _C_ops.cross_entropy2(input, label, 'ignore_index',
                                           ignore_index)
        return loss

    inputs = {'X': [input], 'Label': [label]}
    attrs = {'ignore_index': ignore_index}
    check_variable_and_dtype(input, 'input', ['float16', 'float32', 'float64'],
                             'cross_entropy2')
    helper = LayerHelper('cross_entropy2', **locals())
    out = helper.create_variable_for_type_inference(dtype=input.dtype)
    xshape = helper.create_variable_for_type_inference(dtype=input.dtype)
    match_x = helper.create_variable_for_type_inference(dtype=input.dtype)
    helper.append_op(
        type='cross_entropy2',
        inputs=inputs,
        outputs={'Y': [out],
                 'MatchX': [match_x],
                 'XShape': [xshape]},
        attrs=attrs)
    return out


def square_error_cost(input, label):
    r"""

    This op accepts input predictions and target label and returns the
    squared error cost.

    For predictions label, and target label, the equation is:

    .. math::

        Out = (input - label)^2

    Parameters:
        input (Tensor): Input tensor, the data type should be float32.
        label (Tensor): Label tensor, the data type should be float32.

    Returns:
        The tensor storing the element-wise squared error \
                  difference between input and label.

    Return type: Tensor.

    Examples:

        .. code-block:: python

            import paddle
            input = paddle.to_tensor([1.1, 1.9])
            label = paddle.to_tensor([1.0, 2.0])
            output = paddle.nn.functional.square_error_cost(input, label)
            print(output)
            # [0.01, 0.01]

    """
    if in_dygraph_mode():
        minus_out = _C_ops.elementwise_sub(input, label)
        square_out = _C_ops.square(minus_out)
        return square_out

    check_variable_and_dtype(input, "input", ['float32', 'float64'],
                             'square_error_cost')
    check_variable_and_dtype(label, "label", ['float32', 'float64'],
                             'square_error_cost')
    helper = LayerHelper('square_error_cost', **locals())
    minus_out = helper.create_variable_for_type_inference(dtype=input.dtype)
    helper.append_op(
        type='elementwise_sub',
        inputs={'X': [input],
                'Y': [label]},
        outputs={'Out': [minus_out]})

    square_out = helper.create_variable_for_type_inference(dtype=input.dtype)
    helper.append_op(
        type='square', inputs={'X': [minus_out]},
        outputs={'Out': [square_out]})
    return square_out


def edit_distance(input,
                  label,
                  normalized=True,
                  ignored_tokens=None,
                  input_length=None,
                  label_length=None):
    """
    This op computes the edit distances, also called Levenshtein distance, between a batch of
    hypothesis strings and their references. It measures how dissimilar two strings are by counting
    the minimum number of operations to transform one string into another.
    The operations include insertion, deletion, and substitution.

    For example, given hypothesis string A = "kitten" and reference
    B = "sitting", A will be transformed into B
    at least after two substitutions and one insertion:

    "kitten" -> "sitten" -> "sittin" -> "sitting"

    So the edit distance between A and B is 3.

    The input is a Tensor, the input_length and label_length should be supported.

    The `batch_size` of labels should be same as `input`.

    The output include the edit distance value between every pair of input and related label, and the number of sequence.
    If Attr(normalized) is true,
    the edit distance value will be divided by the length of label.

    Parameters:
        input(Tensor): The input tensor, its rank should be equal to 2 and its data type should be int64.
        label(Tensor): The label tensor, its rank should be equal to 2 and its data type should be int64.
        normalized(bool, default True): Indicated whether to normalize the edit distance.
        ignored_tokens(list<int>, default None): Tokens that will be removed before
                                     calculating edit distance.
        input_length(Tensor): The length for each sequence in `input` if it's of Tensor type, it should have shape `(batch_size, )` and its data type should be int64.
        label_length(Tensor): The length for each sequence in `label` if it's of Tensor type, it should have shape `(batch_size, )` and its data type should be int64.
        NOTE: To be avoid unexpected result, the value of every elements in input_length and label_length should be equal to the value of the second dimension of input and label. For example, The input: [[1,2,3,4],[5,6,7,8],[9,10,11,12]], the shape of input is [3,4] and the input_length should be [4,4,4]
        NOTE: This Api is different from fluid.metrics.EditDistance

    Returns:
	Tuple:

        distance(Tensor): edit distance result, its data type is float32, and its shape is (batch_size, 1).
        sequence_num(Tensor): sequence number, its data type is float32, and its shape is (1,).

    Examples:
        .. code-block:: python

            import paddle
            import paddle.nn.functional as F

            input = paddle.to_tensor([[1,2,3],[4,5,6],[4,4,4],[1,1,1]], dtype='int64')
            label = paddle.to_tensor([[1,3,4,1],[4,5,8,1],[7,7,7,1],[1,1,1,1]], dtype='int64')
            input_len = paddle.to_tensor([3,3,3,3], dtype='int64')
            label_len = paddle.to_tensor([4,4,4,4], dtype='int64')

            distance, sequence_num = F.loss.edit_distance(input=input, label=label, input_length=input_len, label_length=label_len, normalized=False)

            # print(distance)
            # [[3.]
            #  [2.]
            #  [4.]
            #  [1.]]
            # if set normalized to True
            # [[0.75]
            #  [0.5 ]
            #  [1.  ]
            #  [0.25]
            #
            # print(sequence_num)
            # [4]

    """
    check_variable_and_dtype(input, 'input', ['int64'], 'edit_distance')
    check_variable_and_dtype(label, 'label', ['int64'], 'edit_distance')
    helper = LayerHelper("edit_distance", **locals())

    # remove some tokens from input and labels
    if ignored_tokens is not None and len(ignored_tokens) > 0:
        erased_input = helper.create_variable_for_type_inference(dtype="int64")
        erased_label = helper.create_variable_for_type_inference(dtype="int64")

        helper.append_op(
            type="sequence_erase",
            inputs={"X": [input]},
            outputs={"Out": [erased_input]},
            attrs={"tokens": ignored_tokens})
        input = erased_input

        helper.append_op(
            type="sequence_erase",
            inputs={"X": [label]},
            outputs={"Out": [erased_label]},
            attrs={"tokens": ignored_tokens})
        label = erased_label

    this_inputs = {"Hyps": [input], "Refs": [label]}
    if input_length is not None and label_length is not None:
        this_inputs['HypsLength'] = [input_length]
        this_inputs['RefsLength'] = [label_length]

    # edit distance op
    edit_distance_out = helper.create_variable_for_type_inference(dtype="int64")
    sequence_num = helper.create_variable_for_type_inference(dtype="int64")
    helper.append_op(
        type="edit_distance",
        inputs=this_inputs,
        outputs={"Out": [edit_distance_out],
                 "SequenceNum": [sequence_num]},
        attrs={"normalized": normalized})

    return edit_distance_out, sequence_num


def warpctc(input,
            label,
            blank=0,
            norm_by_times=False,
            input_length=None,
            label_length=None):
    """
    An operator integrating the open source Warp-CTC library
    (https://github.com/baidu-research/warp-ctc)
    to compute Connectionist Temporal Classification (CTC) loss.
    It can be aliased as softmax with CTC, since a native softmax activation is
    interated to the Warp-CTC library to normalize values for each row of the
    input tensor.

    Args:
       input (Variable): The unscaled probabilities of variable-length sequences,
         which is a 2-D Tensor with LoD information, or a 3-D Tensor without Lod
         information. When it is a 2-D LodTensor, its shape is 
         `[Lp, num_classes + 1]`, where `Lp` is the sum of all input
         sequences' length and `num_classes` is the true number of classes.
         (not including the blank label). When it is a 3-D Tensor, its shape 
         is `[max_logit_length, batch_size, num_classes + 1]`,
         where `max_logit_length` is the longest length of
         input logit sequence. The data type should be float32 or float64.
       label (Variable): The ground truth of variable-length sequence,
         which must be a 2-D Tensor with LoD information or a 3-D Tensor without
         LoD information, needs to be consistent with the coressponding input. 
         When it is a 2-D LoDTensor, its shape is `[Lg, 1]`, where `Lg` is the sum 
         of all labels' length. When it is a 3-D Tensor, its shape is 
         `[batch_size, max_label_length]`, where `max_label_length` is the longest
         length of label sequence. Data type must be int32.
       blank (int, default 0): The blank label index of Connectionist
         Temporal Classification (CTC) loss, which is in the
         half-opened interval `[0, num_classes + 1)`. The data type must be int32. 
       norm_by_times(bool, default false): Whether to normalize the gradients
         by the number of time-step, which is also the sequence's length.
         There is no need to normalize the gradients if warpctc layer was
         followed by a mean_op.
       input_length(Variable): The length for each input sequence if it is 
         of Tensor type, it should have shape `[batch_size]` and dtype int64.
       label_length(Variable): The length for each label sequence if it is
         of Tensor type, it should have shape `[batch_size]` and dtype int64.

    Returns:
        Variable: The Connectionist Temporal Classification (CTC) loss,
        which is a 2-D Tensor with the shape `[batch_size, 1]`.
        The date type is the same as input.

    Examples:

        .. code-block:: python

            # using LoDTensor
            import paddle
            import paddle.fluid as fluid
            import numpy as np

            # lengths of logit sequences
            seq_lens = [2,6]
            # lengths of label sequences
            label_lens = [2,3]
            # class num
            class_num = 5

            paddle.enable_static()
            logits = fluid.data(name='logits',shape=[None, class_num+1],
                                 dtype='float32',lod_level=1)
            label = fluid.data(name='label', shape=[None, 1],
                               dtype='int32', lod_level=1)
            cost = fluid.layers.warpctc(input=logits, label=label)
            place = fluid.CPUPlace()
            x = fluid.create_lod_tensor(
                     np.random.rand(np.sum(seq_lens), class_num+1).astype("float32"), 
                     [seq_lens], place)
            y = fluid.create_lod_tensor(
                     np.random.randint(0, class_num, [np.sum(label_lens), 1]).astype("int32"), 
                     [label_lens], place)
            exe = fluid.Executor(place)
            output= exe.run(fluid.default_main_program(),
                            feed={"logits": x,"label": y},
                            fetch_list=[cost.name])
            print(output)

        .. code-block:: python

            # using Tensor
            import paddle
            import paddle.fluid as fluid
            import numpy as np

            # length of the longest logit sequence
            max_seq_length = 5
            #length of the longest label sequence
            max_label_length = 3
            # number of logit sequences
            batch_size = 16
            # class num
            class_num = 5
            paddle.enable_static()
            logits = fluid.data(name='logits',
                           shape=[max_seq_length, batch_size, class_num+1],
                           dtype='float32')
            logits_length = fluid.data(name='logits_length', shape=[None],
                             dtype='int64')
            label = fluid.data(name='label', shape=[batch_size, max_label_length],
                           dtype='int32')
            label_length = fluid.data(name='labels_length', shape=[None],
                             dtype='int64')
            cost = fluid.layers.warpctc(input=logits, label=label,
                            input_length=logits_length,
                            label_length=label_length)
            place = fluid.CPUPlace()
            x = np.random.rand(max_seq_length, batch_size, class_num+1).astype("float32")
            y = np.random.randint(0, class_num, [batch_size, max_label_length]).astype("int32")
            exe = fluid.Executor(place)
            output= exe.run(fluid.default_main_program(),
                            feed={"logits": x,
                                  "label": y,
                                  "logits_length": np.array([max_seq_length]*batch_size).astype("int64"),
                                  "labels_length": np.array([max_label_length]*batch_size).astype("int64")},
                                  fetch_list=[cost.name])
            print(output)
    """
    if in_dygraph_mode():
        if input_length is None or label_length is None:
            raise ValueError(
                "input_length and label_length must not be None in dygraph mode!"
            )
        grad, loss_out = _C_ops.warpctc(
            input,
            label,
            input_length,
            label_length,
            'blank',
            blank,
            'norm_by_times',
            norm_by_times, )
        return loss_out
    helper = LayerHelper('warpctc', **locals())
    check_variable_and_dtype(input, 'input', ['float32', 'float64'], "warpctc")
    check_variable_and_dtype(label, 'label', ['int32'], "warpctc")
    this_inputs = {'Logits': [input], 'Label': [label]}
    if input_length is not None and label_length is not None:
        check_variable_and_dtype(input_length, 'LogitsLength', ['int64'],
                                 "warpctc")
        check_variable_and_dtype(label_length, 'LabelLength', ['int64'],
                                 "warpctc")
        this_inputs['LogitsLength'] = [input_length]
        this_inputs['LabelLength'] = [label_length]

    loss_out = helper.create_variable_for_type_inference(dtype=input.dtype)
    grad_out = helper.create_variable_for_type_inference(dtype=input.dtype)

    helper.append_op(
        type='warpctc',
        inputs=this_inputs,
        outputs={'WarpCTCGrad': [grad_out],
                 'Loss': [loss_out]},
        attrs={
            'blank': blank,
            'norm_by_times': norm_by_times,
        })
    return loss_out


# FIXME(wuyi): let docstring_checker.py understand @autodoc.
# For now, the comments in c++ use types like Tensor, but in python side
# the type is often "Variable", and arguments may vary.
@static_only
@templatedoc(op_type="nce")
def nce(input,
        label,
        num_total_classes,
        sample_weight=None,
        param_attr=None,
        bias_attr=None,
        num_neg_samples=None,
        name=None,
        sampler="uniform",
        custom_dist=None,
        seed=0,
        is_sparse=False):
    """
    :api_attr: Static Graph

    ${comment}

    Args:
        input (Tensor): Input tensor, 2-D tensor with shape [batch_size, dim], 
            and data type is float32 or float64.
        label (Tensor): Input label, 2-D tensor with shape [batch_size, num_true_class],
            and data type is int64.
        num_total_classes (int):${num_total_classes_comment}.
        sample_weight (Tensor|None): A Tensor of shape [batch_size, 1]
            storing a weight for each sample. The default weight for each
            sample is 1.0.
        param_attr (ParamAttr|None): To specify the weight parameter attribute. 
            Default: None, which means the default weight parameter property is 
            used. See usage for details in :ref:`api_fluid_ParamAttr` .
        bias_attr (ParamAttr|None): To specify the bias parameter attribute. 
            Default: None, which means the default bias parameter property is 
            used. See usage for details in :ref:`api_fluid_ParamAttr` .
        num_neg_samples (int): ${num_neg_samples_comment}.
        name(str|None): For detailed information, please refer to 
            :ref:`api_guide_Name` . Usually name is no need to set and None by default.
        sampler (str, optional): The sampler used to sample class from negative classes.
                       It can be 'uniform', 'log_uniform' or 'custom_dist'.
                       default: 'uniform'.
        custom_dist (nd.array|None): A numpy ndarray with size=num_total_classes.
                       It is used when sampler is set to 'custom_dist'.
                       custom_dist[i] is the probability of i-th class to be sampled.
                       default: None.
        seed (int, optional): The seed used in sampler. Default 0, means no random seed.
        is_sparse(bool, optional): The flag indicating whether to use sparse update, 
            the weight@GRAD and bias@GRAD will be changed to SelectedRows. Default False.

    Returns:
        Tensor: The output nce loss.

    Examples:
        .. code-block:: python


            import paddle
            import numpy as np

            paddle.enable_static()

            window_size = 5
            words = []
            for i in range(window_size):
                words.append(paddle.static.data(
                    name='word_{0}'.format(i), shape=[-1, 1], dtype='int64'))

            dict_size = 10000
            label_word = int(window_size / 2) + 1

            embs = []
            for i in range(window_size):
                if i == label_word:
                    continue

                emb = paddle.static.nn.embedding(input=words[i], size=[dict_size, 32],
                                    param_attr='embed', is_sparse=True)
                embs.append(emb)

            embs = paddle.concat(x=embs, axis=1)
            loss = paddle.static.nn.nce(input=embs, label=words[label_word],
                        num_total_classes=dict_size, param_attr='nce.w_0',
                        bias_attr='nce.b_0')

            #or use custom distribution
            dist = np.array([0.05,0.5,0.1,0.3,0.05])
            loss = paddle.static.nn.nce(input=embs, label=words[label_word],
                    num_total_classes=5, param_attr='nce.w_1',
                    bias_attr='nce.b_1',
                    num_neg_samples=3,
                    sampler="custom_dist",
                    custom_dist=dist)
    """
    helper = LayerHelper('nce', **locals())
    check_variable_and_dtype(input, 'input', ['float32', 'float64'], 'nce')
    check_variable_and_dtype(label, 'label', ['int64'], 'nce')

    dim = input.shape[1]
    num_true_class = label.shape[1]
    w = helper.create_parameter(
        attr=helper.param_attr,
        shape=[num_total_classes, dim],
        is_bias=False,
        dtype=input.dtype)
    inputs = {}
    if helper.bias_attr:
        b = helper.create_parameter(
            attr=helper.bias_attr,
            shape=[num_total_classes, 1],
            is_bias=True,
            dtype=input.dtype)
        inputs['Bias'] = b
    cost = helper.create_variable_for_type_inference(dtype=input.dtype)
    sample_logits = helper.create_variable_for_type_inference(dtype=input.dtype)
    sample_labels = helper.create_variable_for_type_inference(dtype=label.dtype)

    inputs['Input'] = input
    inputs['Label'] = label
    inputs['Weight'] = w
    inputs['SampleWeight'] = sample_weight if sample_weight is not None else []

    if sampler == "uniform":
        sampler = 0
    elif sampler == "log_uniform":
        sampler = 1
    elif sampler == "custom_dist":
        assert custom_dist is not None

        custom_dist_len = num_total_classes
        alias_probs_ = [0] * custom_dist_len
        alias_ = [0] * custom_dist_len
        bigs = []
        littles = []
        for i in range(custom_dist_len):
            normal_prob = custom_dist[i] * custom_dist_len
            if normal_prob - 1.0 > 0:
                bigs.append((i, normal_prob))
            elif 1.0 - normal_prob > 0:
                littles.append((i, normal_prob))
            else:
                alias_probs_[i] = normal_prob
                alias_[i] = -1

        while len(bigs) and len(littles):
            big = bigs.pop(0)
            little = littles.pop(0)

            big_idx = big[0]
            big_prob = big[1]

            alias_probs_[little[0]] = little[1]
            alias_[little[0]] = big_idx
            big_left = big[1] + little[1] - 1
            if big_left - 1.0 > 0:
                bigs.append((big_idx, big_left))
            elif 1.0 - big_left > 0:
                littles.append((big_idx, big_left))
            else:
                alias_probs_[big_idx] = big_left
                alias_[big_idx] = -1

        if len(bigs):
            big = bigs.pop(0)
            alias_probs_[big[0]] = 1.0
            alias_[big[0]] = -1
        if len(littles):
            little = littles.pop(0)
            alias_probs_[little[0]] = 1.0
            alias_[little[0]] = -1

        def _init_by_numpy_array(numpy_array):
            ret = helper.create_parameter(
                attr=ParamAttr(),
                shape=numpy_array.shape,
                dtype=numpy_array.dtype,
                default_initializer=NumpyArrayInitializer(numpy_array))
            ret.stop_gradient = True
            return ret

        inputs['CustomDistProbs'] = _init_by_numpy_array(
            np.array(custom_dist).astype('float32'))
        inputs['CustomDistAlias'] = _init_by_numpy_array(
            np.array(alias_).astype('int32'))
        inputs['CustomDistAliasProbs'] = _init_by_numpy_array(
            np.array(alias_probs_).astype('float32'))
        sampler = 2
    else:
        raise Exception("Unsupported sampler type.")

    if num_neg_samples is None:
        num_neg_samples = 10
    else:
        num_neg_samples = int(num_neg_samples)

    remote_prefetch = is_sparse
    print(
        "With sparse mode, if your models has only small parameter prefetch may cause speed down"
    )

    attrs = {
        'num_total_classes': int(num_total_classes),
        'num_neg_samples': num_neg_samples,
        'seed': seed,
        'sampler': sampler,
        'is_sparse': is_sparse,
        'remote_prefetch': remote_prefetch
    }

    helper.append_op(
        type='nce',
        inputs=inputs,
        outputs={
            'Cost': cost,
            'SampleLogits': sample_logits,
            'SampleLabels': sample_labels
        },
        attrs=attrs)
    return cost / (num_neg_samples + 1)


def hsigmoid(input,
             label,
             num_classes,
             param_attr=None,
             bias_attr=None,
             name=None,
             path_table=None,
             path_code=None,
             is_custom=False,
             is_sparse=False):
    """
    :api_attr: Static Graph
    
    The hierarchical sigmoid organizes the classes into a complete binary tree to reduce the computational complexity
    and speed up the model training, especially the training of language model.
    Each leaf node of the complete binary tree represents a class(word) and each non-leaf node acts as a binary classifier.
    For each class(word), there's a unique path from root to itself, hsigmoid calculate the cost for each non-leaf node on
    the path, and sum them to get a total cost.
    Comparing to softmax, the OP can reduce the computational complexity from :math:`O(N)` to :math:`O(logN)`, where :math:`N`
    represents the number of classes or the size of word dict.

    The OP supports default tree and custom tree. For the default tree, you can refer to `Hierarchical Probabilistic Neural
    Network Language Model <http://www.iro.umontreal.ca/~lisa/pointeurs/hierarchical-nnlm-aistats05.pdf>`. For the custom
    tree, you need to set :attr:`is_custom` to True, and do the following steps (take the language model as an example):

    1. Using a custom word dict to build a binary tree, each leaf node should be an word in the word dict.
    2. Creating a dict map word_id -> path that from the word to the root node, we call it path_table.
    3. Creating a dict map word_id -> code of path that from the word to the root node, we call it path_code.
       Code means the label of each binary classifier, 1 indicate true, 0 indicate false.
    4. Now, each word should has its path and code along the path, you can pass a batch of path and code related
       to the same batch of inputs.

    Parameters:
        input (Variable): A tensor with the shape [N, D], where N is the size of mini-batch,
            and D is the feature size. Its data type supports float32 and float64.
        label (Variable): A tensor contains the labels of training data. Its shape is [N, 1]
            and data type is int64.
        num_classes (int): The number of classes or the size of word dict, must be greater than 2.
            If the default tree is used (:attr:`is_custom` is set to False), :attr:`num_classes`
            should not be None. If the custom tree is used (:attr:`is_custom` is set to True),
            :attr:`num_classes` should be the number of non-leaf nodes, which indicates the num of
            classes using by the binary classifier.
        param_attr (ParamAttr, optional): The parameter attribute for the learnable parameters/weights
            of hsigmoid. If it is set to None or one attribute of ParamAttr, hsigmoid will create a
            ParamAttr as param_attr. If the Initializer of the param_attr is not set, the parameter is
            initialized with Xavier. Default: None.
        bias_attr (ParamAttr|bool, optional): The parameter attribute for the bias of hsigmoid. If it
            is set to False, no bias will be added. If it is set to None or one attribute of ParamAttr,
            hsigmoid will create a ParamAttr as bias_attr. If the Initializer of the bias_attr is not
            set, the bias is initialized zero. Default: None.
        name (str, optional): Normally there is no need for user to set this property. For more information,
            please refer to :ref:`api_guide_Name`. Default: None.
        path_table (Variable, optional): A tensor that stores each batch of samples' path from leaf to root
            node, its shape is [N, L] and data type is int64, where L is the length of path. For each sample i,
            path_table[i] is a np.array like structure and each element in this array is the indexes in parent
            nodes' weight matrix. Default: None.
        path_code (Variable, optional): A tensor that stores each batch of samples' code of path from leaf
            to root node, its shape is [N, L] and data type is int64, which is the same as :attr:`path_table`.
            Each code of path is consisted with the code of nodes from leaf to root node. Default: None.
        is_custom (bool, optional): Whether use custom binary tree. If it's True, :attr:`path_table`,
            :attr:`path_code` and :attr:`num_classes` should be set, otherwise :attr:`num_classes` should
            be set. Default: False.
        is_sparse (bool, optional): Whether use sparse updating instead of dense updating, if it's True, the
            gradient of W and input will be sparse. Default: False.

    Returns:
        Variable: A tensor with the cost of hierarchical sigmoid, its shape is [N, 1] and data type is the same as :attr:`input`.

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid
            x = fluid.layers.fill_constant(shape=[4, 3], value=0.9, dtype='float32')
            # x = [[0.9, 0.9, 0.9], [0.9, 0.9, 0.9], [0.9, 0.9, 0.9], [0.9, 0.9, 0.9]]
            y = fluid.layers.fill_constant(
                shape=[4, 1], value=1, dtype='int64')
            # y = [[1], [1], [1], [1]]
            out = fluid.layers.hsigmoid(input=x, label=y, num_classes=2, param_attr=fluid.initializer.Constant(
                value=0.05), bias_attr=fluid.initializer.Constant(value=.0))
            # out = [[0.62792355], [0.62792355], [0.62792355], [0.62792355]]
    """
    check_variable_and_dtype(input, 'input', ['float32', 'float64'], 'hsigmoid')
    check_variable_and_dtype(label, 'label', ['int64'], 'hsigmoid')

    helper = LayerHelper('hierarchical_sigmoid', **locals())
    dtype = helper.input_dtype()
    out = helper.create_variable_for_type_inference(dtype)
    pre_out = helper.create_variable_for_type_inference(dtype)
    dim = input.shape[1]
    if ((num_classes is None) or (num_classes < 2)) and (not is_custom):
        raise ValueError(
            "num_classes must not be less than 2 with default tree")

    if (not is_custom) and (is_sparse):
        print("Sparse mode should not be used without custom tree")
        is_sparse = False

    if (not is_custom) and ((path_table is not None) or
                            (path_code is not None)):
        raise ValueError(
            "only num_classes should be passed without custom tree")

    if (is_custom) and (path_code is None):
        raise ValueError("path_code should not be None with custom tree")
    elif (is_custom) and (path_table is None):
        raise ValueError("path_table should not be None with custom tree")
    elif (is_custom) and (num_classes is None):
        raise ValueError("num_classes should not be None with custom tree")
    else:
        pass

    weights = None
    remote_prefetch = is_sparse
    print(
        "With sparse mode, if your models has only small parameter prefetch may cause speed down"
    )
    if not is_custom:
        weights = helper.create_parameter(
            attr=helper.param_attr,
            shape=[num_classes - 1, dim],
            is_bias=False,
            dtype=input.dtype)
    else:
        weights = helper.create_parameter(
            attr=helper.param_attr,
            shape=[num_classes, dim],
            is_bias=False,
            dtype=input.dtype)
    inputs = {
        "X": input,
        "W": weights,
        "PathTable": path_table,
        "PathCode": path_code,
        "Label": label
    }
    if helper.bias_attr:
        if not is_custom:
            bias = helper.create_parameter(
                attr=helper.bias_attr,
                shape=[num_classes - 1, 1],
                is_bias=True,
                dtype=input.dtype)
            inputs['Bias'] = bias
        else:
            bias = helper.create_parameter(
                attr=helper.bias_attr,
                shape=[num_classes, 1],
                is_bias=True,
                dtype=input.dtype)
            inputs['Bias'] = bias
    helper.append_op(
        type="hierarchical_sigmoid",
        inputs=inputs,
        outputs={"Out": out,
                 "PreOut": pre_out,
                 "W_Out": weights},
        attrs={
            "num_classes": num_classes,
            "is_sparse": is_sparse,
            "remote_prefetch": remote_prefetch
        })
    return out


def sampled_softmax_with_cross_entropy(logits,
                                       label,
                                       num_samples,
                                       num_true=1,
                                       remove_accidental_hits=True,
                                       use_customized_samples=False,
                                       customized_samples=None,
                                       customized_probabilities=None,
                                       seed=0):
    """
    **Sampled Softmax With Cross Entropy Operator.**

    Cross entropy loss with sampled softmax is used as the output layer for 
    larger output classes extensively. This operator samples a number of samples
    for all examples, and computes the softmax normalized values for each 
    row of the sampled tensor, after which cross-entropy loss is computed. 

    Because this operator performs a softmax on logits internally, it expects
    unscaled logits. This operator should not be used with the output of
    softmax operator since that would produce incorrect results.
    
    For examples with T true labels (T >= 1), we assume that each true label has
    a probability of 1/T. For each sample, S samples are generated using a
    log uniform distribution. True labels are concatenated with these samples to
    form T + S samples for each example. So, assume the shape of logits is
    [N x K], the shape for samples is [N x (T+S)]. For each sampled label, a 
    probability is calculated, which corresponds to the Q(y|x) in 
    [Jean et al., 2014](http://arxiv.org/abs/1412.2007).
    
    Logits are sampled according to the sampled labels. Then if 
    remove_accidental_hits is True, if a sample[i, j] accidentally hits true 
    labels, then the corresponding sampled_logits[i, j] is minus by 1e20 to 
    make its softmax result close to zero. Then sampled logits are subtracted by
    logQ(y|x), these sampled logits and re-indexed labels are used to compute 
    a softmax with cross entropy.

    Args:
        logits (Variable): The unscaled log probabilities, which is a 2-D tensor
            with shape [N x K]. N is the batch_size, and K is the class number.
        label (Variable): The ground truth which is a 2-D tensor. Label is a 
            Tensor<int64> with shape [N x T], where T is the number of true 
            labels per example. 
        num_samples (int): The number for each example, num_samples should be 
            less than the number of class.
        num_true(int): The number of target classes per training example.
        remove_accidental_hits (bool): A flag indicating whether to remove 
            accidental hits when sampling. If True and if a sample[i, j] 
            accidentally hits true labels, then the corresponding 
            sampled_logits[i, j] is minus by 1e20 to make its softmax result 
            close to zero. Default is True.
        use_customized_samples (bool): Whether to use custom samples and probabities to sample
            logits.
        customized_samples (Variable): User defined samples, which is a 2-D tensor
            with shape [N, T + S]. S is the num_samples, and T is the number of true 
            labels per example. 
        customized_probabilities (Variable): User defined probabilities of samples, 
            a 2-D tensor which has the same shape with customized_samples.
        seed (int): The random seed for generating random number, which is used
            in the process of sampling. Default is 0.

    Returns:
        Variable: Return the cross entropy loss which is a 2-D tensor with shape
                  [N x 1].

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid

            input = fluid.layers.data(name='data', shape=[256], dtype='float32')
            label = fluid.layers.data(name='label', shape=[1], dtype='int64')
            fc = fluid.layers.fc(input=input, size=100)
            out = fluid.layers.sampled_softmax_with_cross_entropy(
                      logits=fc, label=label, num_samples=25)
    """
    helper = LayerHelper('sample_logits', **locals())
    samples = customized_samples if use_customized_samples else helper.create_variable_for_type_inference(
        dtype='int64')
    probabilities = customized_probabilities if use_customized_samples else helper.create_variable_for_type_inference(
        dtype=logits.dtype)
    sampled_logits \
        = helper.create_variable_for_type_inference(dtype=logits.dtype)
    sampled_label = helper.create_variable_for_type_inference(dtype='int64')
    sampled_softlabel = helper.create_variable_for_type_inference(
        dtype=logits.dtype)
    logits_dim = helper.create_variable_for_type_inference(dtype=logits.dtype)
    labels_dim = helper.create_variable_for_type_inference(dtype=label.type)

    helper.append_op(
        type='sample_logits',
        inputs={
            'Logits': logits,
            'Labels': label,
            'CustomizedSamples': customized_samples,
            'CustomizedProbabilities': customized_probabilities
        },
        outputs={
            'Samples': samples,
            'Probabilities': probabilities,
            'SampledLabels': sampled_label,
            'SampledLogits': sampled_logits,
            'LogitsDim': logits_dim,
            'LabelsDim': labels_dim
        },
        attrs={
            'use_customized_samples': use_customized_samples,
            'uniq': True,
            'remove_accidental_hits': remove_accidental_hits,
            'num_samples': num_samples,
            'seed': seed
        })
    loss = helper.create_variable_for_type_inference(dtype=logits.dtype)
    softmax = helper.create_variable_for_type_inference(dtype=logits.dtype)
    helper.append_op(
        type='one_hot',
        inputs={'X': sampled_label},
        attrs={'depth': num_samples + 1},
        outputs={'Out': sampled_softlabel})

    helper.append_op(
        type='softmax_with_cross_entropy',
        inputs={'Logits': sampled_logits,
                'Label': sampled_softlabel},
        outputs={'Softmax': softmax,
                 'Loss': loss},
        attrs={
            'soft_label': True,
            'ignore_index': False,
            'numeric_stable_mode': False
        })
    return loss / num_true


def softmax_with_cross_entropy(logits,
                               label,
                               soft_label=False,
                               ignore_index=kIgnoreIndex,
                               numeric_stable_mode=True,
                               return_softmax=False,
                               axis=-1):
    r"""

    This operator implements the cross entropy loss function with softmax. This function 
    combines the calculation of the softmax operation and the cross entropy loss function 
    to provide a more numerically stable gradient.

    Because this operator performs a softmax on logits internally, it expects
    unscaled logits. This operator should not be used with the output of
    softmax operator since that would produce incorrect results.

    When the attribute :attr:`soft_label` is set :attr:`False`, this operators 
    expects mutually exclusive hard labels, each sample in a batch is in exactly 
    one class with a probability of 1.0. Each sample in the batch will have a 
    single label.

    The equation is as follows:

    1) Hard label (one-hot label, so every sample has exactly one class)

    .. math::

        loss_j =  -\\text{logits}_{label_j} +
        \\log\\left(\\sum_{i=0}^{K}\\exp(\\text{logits}_i)\\right), j = 1,..., K

    2) Soft label (each sample can have a distribution over all classes)

    .. math::

        loss_j =  -\\sum_{i=0}^{K}\\text{label}_i
        \\left(\\text{logits}_i - \\log\\left(\\sum_{i=0}^{K}
        \\exp(\\text{logits}_i)\\right)\\right), j = 1,...,K

    3) If :attr:`numeric_stable_mode` is :attr:`True`, softmax is calculated first by:

    .. math::

        max_j &= \\max_{i=0}^{K}{\\text{logits}_i}

        log\\_max\\_sum_j &= \\log\\sum_{i=0}^{K}\\exp(logits_i - max_j)

        softmax_j &= \\exp(logits_j - max_j - {log\\_max\\_sum}_j)

    and then cross entropy loss is calculated by softmax and label.

    Args:
        logits (Tensor): A multi-dimension ``Tensor`` , and the data type is float32 or float64. The input tensor of unscaled log probabilities.
        label (Tensor): The ground truth  ``Tensor`` , data type is the same
            as the ``logits`` . If :attr:`soft_label` is set to :attr:`True`, 
            Label is a ``Tensor``  in the same shape with :attr:`logits`. 
            If :attr:`soft_label` is set to :attr:`True`, Label is a ``Tensor`` 
            in the same shape with :attr:`logits` expect shape in dimension :attr:`axis` as 1.
        soft_label (bool, optional): A flag to indicate whether to interpretant the given
            labels as soft labels. Default False.
        ignore_index (int, optional): Specifies a target value that is ignored and does
                                      not contribute to the input gradient. Only valid
                                      if :attr:`soft_label` is set to :attr:`False`. 
                                      Default: kIgnoreIndex(-100).
        numeric_stable_mode (bool, optional): A flag to indicate whether to use a more
                                              numerically stable algorithm. Only valid
                                              when :attr:`soft_label` is :attr:`False` 
                                              and GPU is used. When :attr:`soft_label` 
                                              is :attr:`True` or CPU is used, the 
                                              algorithm is always numerically stable.
                                              Note that the speed may be slower when use
                                              stable algorithm. Default: True.
        return_softmax (bool, optional): A flag indicating whether to return the softmax
                                         along with the cross entropy loss. Default: False.
        axis (int, optional): The index of dimension to perform softmax calculations. It 
                              should be in range :math:`[-1, rank - 1]`, while :math:`rank`
                              is the rank of input :attr:`logits`. Default: -1.

    Returns:
        ``Tensor`` or Tuple of two ``Tensor`` : Return the cross entropy loss if \
                                                    `return_softmax` is False, otherwise the tuple \
                                                    (loss, softmax), softmax is in the same shape \
                                                    with input logits and cross entropy loss is in \
                                                    the same shape with input logits except shape \
                                                    in dimension :attr:`axis` as 1.

    Examples:
        .. code-block:: python

            import paddle
            import numpy as np

            data = np.random.rand(128).astype("float32")
            label = np.random.rand(1).astype("int64")
            data = paddle.to_tensor(data)
            label = paddle.to_tensor(label)
            linear = paddle.nn.Linear(128, 100)
            x = linear(data)
            out = paddle.nn.functional.softmax_with_cross_entropy(logits=x, label=label)
            print(out)
    """
    if in_dygraph_mode():
        if core.is_compiled_with_npu():
            softmax, backprop, loss = _C_ops.softmax_with_cross_entropy(
                logits, label, 'soft_label', soft_label, 'ignore_index',
                ignore_index, 'numeric_stable_mode', numeric_stable_mode,
                'axis', axis)
        else:
            softmax, loss = _C_ops.softmax_with_cross_entropy(
                logits, label, 'soft_label', soft_label, 'ignore_index',
                ignore_index, 'numeric_stable_mode', numeric_stable_mode,
                'axis', axis)
        if not return_softmax:
            return loss
        else:
            return loss, softmax

    attrs = {
        'soft_label': soft_label,
        'ignore_index': ignore_index,
        'numeric_stable_mode': numeric_stable_mode,
        'axis': axis
    }
    helper = LayerHelper('softmax_with_cross_entropy', **locals())
    softmax = helper.create_variable_for_type_inference(dtype=logits.dtype)
    loss = helper.create_variable_for_type_inference(dtype=logits.dtype)

    outputs = {'Softmax': softmax, 'Loss': loss}
    if core.is_compiled_with_npu():
        backprop = helper.create_variable_for_type_inference(dtype=logits.dtype)
        outputs['Backprop'] = backprop
    helper.append_op(
        type='softmax_with_cross_entropy',
        inputs={'Logits': logits,
                'Label': label},
        outputs=outputs,
        attrs=attrs)

    if return_softmax:
        return loss, softmax

    return loss


def rank_loss(label, left, right, name=None):
    r"""

    This operator implements the sort loss layer in the RankNet model. RankNet is a pairwise ranking model 
    with a training sample consisting of a pair of documents (A and B), The label (P) 
    indicates whether A is ranked higher than B or not. Please refer to more details: 
    `RankNet <http://icml.cc/2015/wp-content/uploads/2015/06/icml_ranking.pdf>`_

    Rank loss layer takes three inputs: left ( :math:`o_i` ), right ( :math:`o_j` ) and
    label ( :math:`P_{i,j}` ). The inputs respectively represent RankNet's output scores
    for documents A and B and the value of label P. Rank loss layer takes batch inputs 
    with size batch_size (batch_size >= 1), P = {0, 1} or {0, 0.5, 1}, 
    where 0.5 means that there is no information about the rank of the input pair.
    The following equation computes rank loss C_{i,j} from the inputs:

    .. math::
      C_{i,j} &= -\\tilde{P_{ij}} * o_{i,j} + \log(1 + e^{o_{i,j}}) \\\\
    .. math::
      o_{i,j} &=  o_i - o_j  \\\\
    .. math::
      \\tilde{P_{i,j}} &= \\left \{0, 0.5, 1 \\right \} \ or \ \\left \{0, 1 \\right \}

    Parameters:
        label (Variable): 2-D ``Tensor`` with the shape of :math:`[batch,1]`, the data type is float32, batch indicates the size of the data. Indicats whether A ranked higher than B or not.
        left (Variable): 2-D ``Tensor`` with the shape of :math:`[batch,1]`, the data type is float32. RankNet's output score for doc A.
        right (Variable): 2-D ``Tensor`` with the shape of :math:`[batch,1]`, the data type is float32. RankNet's output score for doc B.
        name(str|None): The default value is None. Normally there is no need for user to set this property. For more information, please refer to :ref:`api_guide_Name` .

    Returns:
        Variable: ``Tensor`` indicating the output value of the sort loss layer, the data type is float32, and the return value's shape is :math:`[batch,1]` .

    Raises:
        ValueError: Any of label, left, and right is not a ``Variable`` .

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid
            import paddle
            paddle.enable_static()
            label = fluid.data(name="label", shape=[-1, 1], dtype="float32")
            left = fluid.data(name="left", shape=[-1, 1], dtype="float32")
            right = fluid.data(name="right", shape=[-1, 1], dtype="float32")
            out = fluid.layers.rank_loss(label, left, right)

    """
    helper = LayerHelper('rank_loss', **locals())
    check_variable_and_dtype(label, 'label', ['float32'], "rank_loss")
    check_variable_and_dtype(left, 'left', ['float32'], "rank_loss")
    check_variable_and_dtype(right, 'right', ['float32'], "rank_loss")

    out = helper.create_variable_for_type_inference("float32")

    helper.append_op(
        type='rank_loss',
        inputs={"Label": label,
                "Left": left,
                "Right": right},
        outputs={'Out': out})
    return out


def margin_rank_loss(label, left, right, margin=0.1, name=None):
    r"""
    Margin Ranking Loss Layer for ranking problem,
    which compares left score and right score passed in.
    The ranking loss can be defined as following equation:

    .. math::

        rank\_loss = max(0, -label * (left - right) + margin)

    Args:
       label (Variable): Indicates whether the left is ranked higher than the right or not.
           Data type is float32.
       left (Variable): Ranking score for left. Data type float32.
       right (Variable): Ranking score for right. Data type float32.
       margin (float): Indicates the given margin.
       name(str|None): For detailed information, please refer to 
           :ref:`api_guide_Name` . Usually name is no need to set and None by default.

    Returns:
       Variable: The ranking loss.

    Raises:
       ValueError: Any of label, left, and right is not a Variable.

    Examples:

        .. code-block:: python

           import paddle.fluid as fluid
           label = fluid.data(name="label", shape=[-1, 1], dtype="float32")
           left = fluid.data(name="left", shape=[-1, 1], dtype="float32")
           right = fluid.data(name="right", shape=[-1, 1], dtype="float32")
           out = fluid.layers.margin_rank_loss(label, left, right)
    """
    helper = LayerHelper('margin_rank_loss', **locals())
    check_variable_and_dtype(label, 'label', ['float32'], 'margin_rank_loss')
    check_variable_and_dtype(label, 'left', ['float32'], 'margin_rank_loss')
    check_variable_and_dtype(label, 'right', ['float32'], 'margin_rank_loss')
    out = helper.create_variable_for_type_inference(left.dtype)
    act = helper.create_variable_for_type_inference(left.dtype)
    helper.append_op(
        type='margin_rank_loss',
        inputs={"Label": label,
                "X1": left,
                "X2": right},
        outputs={'Out': out,
                 'Activated': act},
        attrs={'margin': margin})
    return out


@templatedoc()
def sigmoid_cross_entropy_with_logits(x,
                                      label,
                                      ignore_index=kIgnoreIndex,
                                      name=None,
                                      normalize=False):
    """

    ${comment}

    Args:
        x(Tensor): a 2-D tensor with shape N x D, where N is the batch size and
                D is the number of classes. This input is a tensor of logits computed
                by the previous operator. Logits are unscaled log probabilities given
                as log(p/(1-p)) The data type should be float32 or float64.
        label (Tensor): a 2-D tensor of the same type and shape as X.
                This input is a tensor of probabalistic labels for each logit.
        ignore_index(int): Specifies a target value that is ignored and 
                does not contribute to the input gradient.
        name(str|None): The default value is None.  Normally there is
            no need for user to set this property.  For more information,
            please refer to :ref:`api_guide_Name`
        normalize(bool): If true, divide the output by the number of
            targets != ignore_index.

    Returns:
        out(Tensor): ${out_comment}

    Examples:
        .. code-block:: python


            import paddle

            input = paddle.rand(shape=[10], dtype='float32')
            label = paddle.rand(shape=[10], dtype='float32')
            loss = paddle.fluid.layers.sigmoid_cross_entropy_with_logits(input, label, 
                                                            ignore_index=-1, normalize=True)
            print(loss)
    """
    check_variable_and_dtype(x, 'input', ['float16', 'float32', 'float64'],
                             'sigmoid_cross_entropy_with_logits')

    helper = LayerHelper("sigmoid_cross_entropy_with_logits", **locals())

    out = helper.create_variable_for_type_inference(dtype=x.dtype)

    helper.append_op(
        type="sigmoid_cross_entropy_with_logits",
        inputs={"X": x,
                "Label": label},
        attrs={"ignore_index": ignore_index,
               'normalize': normalize},
        outputs={"Out": out})
    return out


def teacher_student_sigmoid_loss(input,
                                 label,
                                 soft_max_up_bound=15.0,
                                 soft_max_lower_bound=-15.0):
    """

    **Teacher Student Log Loss Layer**

    This layer accepts input predictions and target label and returns the
    teacher_student loss. Z is click or not, z' is value of teacher loss, label = {-2, -1, [0, 2]}
    when z' is not exist, clk = 0 : label = -2; when z' is not exist, clk = 1 : label = -1;
    when z' is exist    , clk = 0 : label = 0 + z'; when z' is exist    , clk = 1 : label = 1 + z'

    .. math::
        loss = max(x, 0) - x * z + log(1 + exp(-abs(x))) + max(x, 0) - x * z' + log(1 + exp(-abs(x)))

    Args:
        input (Variable|list):  a 2-D tensor with shape [N x 1], where N is the
                                batch size. This input is a probability computed
                                by the previous operator.
        label (Variable|list):  the ground truth which is a 2-D tensor with
                                shape [N x 1], where N is the batch size.
        soft_max_up_bound  (float):  if input > soft_max_up_bound, will be bound
        soft_max_lower_bound (float): if input < soft_max_lower_bound, will be bound

    Returns:
        Variable: A 2-D tensor with shape [N x 1], the teacher_student_sigmoid_loss.

    Examples:
        .. code-block:: python
          
          import paddle.fluid as fluid
          import paddle
          paddle.enable_static()
          batch_size = 64
          label = fluid.data(
                    name="label", shape=[batch_size, 1], dtype="int64")
          similarity = fluid.data(
                    name="similarity", shape=[batch_size, 1], dtype="float32")
          cost = fluid.layers.teacher_student_sigmoid_loss(input=similarity, label=label)

    """
    check_variable_and_dtype(input, "input",
                             ['float32', 'float64', 'int32', 'int64'],
                             'teacher_student_sigmoid_loss')
    check_variable_and_dtype(label, "label",
                             ['float32', 'float64', 'int32', 'int64'],
                             'teacher_student_sigmoid_loss')

    helper = LayerHelper('teacher_student_sigmoid_loss', **locals())
    out = helper.create_variable(dtype=input.dtype)
    helper.append_op(
        type='teacher_student_sigmoid_loss',
        inputs={'X': [input],
                'Label': [label]},
        outputs={'Y': [out]},
        attrs={"soft_max_lower_bound": float(soft_max_lower_bound), \
                "soft_max_up_bound": float(soft_max_up_bound)})
    return out


def huber_loss(input, label, delta):
    r"""
    This operator computes the Huber loss between input and label.
    Huber loss is commonly used in regression tasks. Compared to square_error_cost, Huber loss is more robust and less sensitivity to outliers.

    When the absolute difference between input and label is greater than delta, the linear error is calculated:

    .. math::
            huber\_loss = delta * (label - input) - 0.5 * delta * delta

    When the absolute difference between input and label is greater than delta, the square error is calculated:

    .. math::
            huber\_loss = 0.5 * (label - input) * (label - input)


    Args:
        input (Variable): Predicted data, 2D-Tensor with the shape of [batch_size, 1]. The data type should be float32.
        label (Variable): Ground truth label, 2D-Tensor with the shape of [batch_size, 1]. The data type should be float32.
        delta (float): The threshold for Huber loss, which is used to control the balance between the linear error and square error. The data type should be float32.

    Returns:
        Variable: The huber loss, a tensor with the same shape and data type as input.


    Examples:

    ..  code-block:: python

        import paddle.fluid as fluid
        import numpy as np

        DATATYPE='float32'
        input_data = np.array([[1.],[2.],[3.],[4.]]).astype(DATATYPE)
        label_data = np.array([[3.],[3.],[4.],[4.]]).astype(DATATYPE)

        x = fluid.data(name='input', shape=[None, 1], dtype=DATATYPE)
        y = fluid.data(name='label', shape=[None, 1], dtype=DATATYPE)
        loss = fluid.layers.huber_loss(input=x, label=y, delta=1.0)

        place = fluid.CPUPlace()
        #place = fluid.CUDAPlace(0)
        exe = fluid.Executor(place)
        HuberLoss, = exe.run(feed={'input':input_data ,'label':label_data}, fetch_list=[loss.name])
        print(HuberLoss)  #[[1.5], [0.5], [0.5], [0. ]], dtype=float32
    """
    helper = LayerHelper('huber_loss', **locals())
    check_variable_and_dtype(input, 'input', ['float32', 'float64'],
                             'huber_loss')
    check_variable_and_dtype(label, 'label', ['float32', 'float64'],
                             'huber_loss')
    residual = helper.create_variable_for_type_inference(
        dtype=helper.input_dtype())
    out = helper.create_variable_for_type_inference(dtype=helper.input_dtype())
    helper.append_op(
        type='huber_loss',
        inputs={'X': input,
                'Y': label},
        outputs={'Out': out,
                 'Residual': residual},
        attrs={'delta': delta})
    return out


@deprecated(since="2.0.0", update_to="paddle.nn.functional.kl_div")
@templatedoc()
def kldiv_loss(x, target, reduction='mean', name=None):
    """

    ${comment}

    Args:
        x (Tensor): ${x_comment}
        target (Tensor): ${target_comment}
        reduction (Tensor): ${reduction_comment}
        name(str, optional): For detailed information, please refer
                             to :ref:`api_guide_Name`. Usually name is no need to set and
                             None by default.

    Returns:
        Tensor: The KL divergence loss. The data type is same as input tensor

    Examples:
        .. code-block:: python

            import paddle
            import paddle.fluid as fluid
            
            x = paddle.rand(shape=[3,4,2,2], dtype='float32')
            target = paddle.rand(shape=[3,4,2,2], dtype='float32')

            # 'batchmean' reduction, loss shape will be [1]
            loss = fluid.layers.kldiv_loss(x=x, target=target, reduction='batchmean')
            print(loss.shape) # shape=[1]
            
            # 'mean' reduction, loss shape will be [1]
            loss = fluid.layers.kldiv_loss(x=x, target=target, reduction='mean')
            print(loss.shape) # shape=[1]
            
            # 'sum' reduction, loss shape will be [1]
            loss = fluid.layers.kldiv_loss(x=x, target=target, reduction='sum')
            print(loss.shape) # shape=[1]
            
            # 'none' reduction, loss shape is same with X shape
            loss = fluid.layers.kldiv_loss(x=x, target=target, reduction='none')
            print(loss.shape) # shape=[3, 4, 2, 2]

    """
    helper = LayerHelper('kldiv_loss', **locals())
    check_variable_and_dtype(x, 'x', ['float32', 'float64'], 'kldiv_loss')
    check_variable_and_dtype(target, 'target', ['float32', 'float64'],
                             'kldiv_loss')
    check_type(reduction, 'reduction', str, 'kldiv_loss')
    loss = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type='kldiv_loss',
        inputs={'X': x,
                'Target': target},
        outputs={'Loss': loss},
        attrs={'reduction': reduction})
    return loss


from .ops import square
from .control_flow import equal


def npair_loss(anchor, positive, labels, l2_reg=0.002):
    """ 
  
    Npair loss requires paired data. Npair loss has two parts: the first part is L2
    regularizer on the embedding vector; the second part is cross entropy loss which
    takes the similarity matrix of anchor and positive as logits.
  
    For more information, please refer to:
    `Improved Deep Metric Learning with Multi class N pair Loss Objective <http://www.nec-labs.com/uploads/images/Department-Images/MediaAnalytics/papers/nips16_npairmetriclearning.pdf>`_
  
    Args:
      anchor(Tensor): embedding vector for the anchor image. shape=[batch_size, embedding_dims], 
                        the data type is float32 or float64.
      positive(Tensor): embedding vector for the positive image. shape=[batch_size, embedding_dims], 
                        the data type is float32 or float64.
      labels(Tensor): 1-D tensor. shape=[batch_size], the data type is float32 or float64 or int64.
      l2_reg(float32): L2 regularization term on embedding vector, default: 0.002.

  
    Returns:
      A Tensor representing the npair loss, the data type is the same as anchor, the shape is [1].
  
    Examples:

      .. code-block:: python
  
          import paddle
          
          DATATYPE = "float32"
  
          anchor = paddle.rand(shape=(18, 6), dtype=DATATYPE)
          positive = paddle.rand(shape=(18, 6), dtype=DATATYPE)
          labels = paddle.rand(shape=(18,), dtype=DATATYPE)
          
          npair_loss = paddle.nn.functional.npair_loss(anchor, positive, labels, l2_reg = 0.002)
          print(npair_loss)
  
    """
    check_variable_and_dtype(anchor, 'anchor', ['float32', 'float64'],
                             'npair_loss')
    check_variable_and_dtype(positive, 'positive', ['float32', 'float64'],
                             'positive')
    check_variable_and_dtype(labels, 'labels', ['float32', 'float64', 'int64'],
                             'labels')
    Beta = 0.25
    batch_size = labels.shape[0]

    labels = nn.reshape(labels, shape=[batch_size, 1])
    labels = paddle.tile(labels, repeat_times=[1, batch_size])

    labels = equal(labels, nn.transpose(labels, perm=[1, 0])).astype('float32')
    labels = labels / nn.reduce_sum(labels, dim=1, keep_dim=True)

    l2loss = nn.reduce_mean(nn.reduce_sum(square(anchor), 1)) \
             + nn.reduce_mean(nn.reduce_sum(square(positive), 1))
    l2loss = l2loss * Beta * l2_reg

    similarity_matrix = paddle.matmul(
        anchor, positive, transpose_x=False, transpose_y=True)
    softmax_ce = softmax_with_cross_entropy(
        logits=similarity_matrix, label=labels, soft_label=True)
    cross_entropy = nn.reduce_sum(labels * softmax_ce, 0)
    celoss = nn.reduce_mean(cross_entropy)

    return l2loss + celoss


def mse_loss(input, label):
    """

    This op accepts input predications and target label and returns the mean square error.

    The loss can be described as:

    .. math::
        
        Out = MEAN((input - label)^2)

    Parameters: 
        input (Tensor): Input tensor, the data type should be float32.
        label (Tensor): Label tensor, the data type should be float32.

    Returns:
        Tensor: The tensor storing the mean square error difference of input and label.

    Return type: Tensor.
    
    Examples:
        .. code-block:: python

            import paddle
            input = paddle.to_tensor([1.1, 1.9])
            label = paddle.to_tensor([1.0, 2.0])
            output = paddle.fluid.layers.mse_loss(input, label)
            print(output.numpy())
            # [0.01]
    """
    check_variable_and_dtype(input, "input", ['float32', 'float64'], 'mse_loss')
    check_variable_and_dtype(label, "label", ['float32', 'float64'], 'mse_loss')
    return nn.reduce_mean(square_error_cost(input, label))
