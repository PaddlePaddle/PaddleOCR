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

from .layer_function_generator import templatedoc
from ..framework import Variable, in_dygraph_mode
from ..layer_helper import LayerHelper
from ..data_feeder import check_variable_and_dtype, check_type, check_dtype
from ..core import VarDesc

__all__ = [
    'sequence_conv',
    'sequence_softmax',
    'sequence_pool',
    'sequence_concat',
    'sequence_first_step',
    'sequence_last_step',
    'sequence_slice',
    'sequence_expand',
    'sequence_expand_as',
    'sequence_pad',
    'sequence_unpad',
    'sequence_reshape',
    'sequence_scatter',
    'sequence_enumerate',
    'sequence_mask',
    'sequence_reverse',
]


@templatedoc()
def sequence_conv(input,
                  num_filters,
                  filter_size=3,
                  filter_stride=1,
                  padding=True,
                  padding_start=None,
                  bias_attr=None,
                  param_attr=None,
                  act=None,
                  name=None):
    r"""
	:api_attr: Static Graph

    **Notes: The Op only receives LoDTensor as input. If your input is Tensor, please use conv2d Op.(fluid.layers.** :ref:`api_fluid_layers_conv2d` ).

    This operator receives input sequences with variable length and other convolutional
    configuration parameters(num_filters, filter_size) to apply the convolution operation.
    It fills all-zero padding data on both sides of the sequence by default to ensure that
    the output is the same length as the input. You can customize the padding behavior by
    configuring the parameter :attr:`padding\_start` .
    
    **Warning:** the parameter :attr:`padding` take no effect and will be deprecated in the future.

    .. code-block:: text

            Here we will illustrate the details of the padding operation:
            For a mini-batch of 2 variable lengths sentences, containing 3, and 1 time-steps:
            Assumed input (X) is a [4, N] float LoDTensor, and for the sake of simplicity, we assume N=2.
            input.data = [[1, 1],
                          [2, 2],
                          [3, 3],
                          [4, 4]]

            This is to say that input (X) has 4 words and the dimension of each word
            representation is 2.

            * Case1:

                If padding_start is -1 and filter_size is 3.
                The length of padding data is calculated as follows:
                up_pad_len = max(0, -padding_start) = 1
                down_pad_len = max(0, filter_size + padding_start - 1) = 1

                The output of the input sequence after padding is:
                data_aftet_padding = [[0, 0, 1, 1, 2, 2],
                                      [1, 1, 2, 2, 3, 3],
                                      [2, 2, 3, 3, 0, 0],
                                      [0, 0, 4, 4, 0, 0]]

                It will be multiplied by the filter weight to get the final output.
                Assume num_filters = 3
                output.data = [[ 0.3234, -0.2334,  0.7433],
                               [ 0.5646,  0.9464, -0.1223],
                               [-0.1343,  0.5653,  0.4555],
                               [ 0.9954, -0.1234, -0.1234]]
                output.shape = [4, 3]     # 3 = num_filters
                output.lod = [[0, 3, 4]]  # Remain the same


    Args:
        input (Variable): LoDTensor with shape :math:`(M, K)`, where M is the total time-step of mini-batch
            and K is hidden_size of input. Only lod_level of 1 is supported. The data type should be float32 or
            float64.
        num_filters (int): the number of filters.
        filter_size (int): the height of filter. Specified filter width is not supported, the width is
            hidden_size by default. Default: 3.
        filter_stride (int): stride of the filter. Currently only supports :attr:`stride` = 1.
        padding (bool): the parameter :attr:`padding` take no effect and will be discarded in the
            future. Currently, it will always pad input to make sure the length of the output is
            the same as input whether :attr:`padding` is set true or false. Because the length of
            input sequence may be shorter than :attr:`filter\_size`, which will cause the convolution
            result to not be computed correctly. These padding data will not be trainable or updated
            while training. Default: True.
        padding_start (int): It is used to indicate the start index for padding the input
            sequence, which can be negative. The negative number means to pad
            :attr:`|padding_start|` time-steps of all-zero data at the beginning of each instance.
            The positive number means to skip :attr:`padding_start` time-steps of each instance,
            and it will pad :math:`filter\_size + padding\_start - 1` time-steps of all-zero data
            at the end of the sequence to ensure that the output is the same length as the input.
            If set None, the same length :math:`\\frac{filter\_size}{2}` of data will be filled
            on both sides of the sequence. If set 0, the length of :math:`filter\_size - 1` data
            is padded at the end of each input sequence. Default: None.
        bias_attr (ParamAttr): To specify the bias parameter property. Default: None, which means the
            default bias parameter property is used. See usage for details in :ref:`api_fluid_ParamAttr` .
        param_attr (ParamAttr): To specify the weight parameter property. Default: None, which means the
            default weight parameter property is used. See usage for details in :ref:`api_fluid_ParamAttr` .
        act (str): Activation to be applied to the output of this layer, such as tanh, softmax,
            sigmoid, relu. For more information, please refer to :ref:`api_guide_activations_en` . Default: None.
        name (str, optional): The default value is None.  Normally there is no need for user to set this property.
            For more information, please refer to :ref:`api_guide_Name` .

    Returns:
        Variable: LoDTensor with the same length as input. The data type is float32 or float64, which is same as input.

    Examples:

        .. code-block:: python

             import paddle
             paddle.enable_static()

             x = paddle.static.data(name='x', shape=[-1, 10], dtype='float32', lod_level=1)
             x_conved = paddle.static.nn.sequence_conv(input=x, num_filters=2, filter_size=3, padding_start=-1)
    """

    assert not in_dygraph_mode(), (
        "sequence layer is not supported in dygraph mode yet.")
    check_variable_and_dtype(input, 'input', ['float32', 'float64'],
                             'sequence_conv')
    helper = LayerHelper('sequence_conv', **locals())
    dtype = helper.input_dtype()
    filter_shape = [filter_size * input.shape[1], num_filters]
    filter_param = helper.create_parameter(
        attr=helper.param_attr, shape=filter_shape, dtype=dtype)
    pre_bias = helper.create_variable_for_type_inference(dtype)
    if padding_start is None:
        padding_start = -int(filter_size // 2)

    helper.append_op(
        type='sequence_conv',
        inputs={
            'X': [input],
            'Filter': [filter_param],
        },
        outputs={"Out": pre_bias},
        attrs={
            'contextStride': filter_stride,
            'contextStart': padding_start,
            'contextLength': filter_size,
        })
    pre_act = helper.append_bias_op(pre_bias)
    return helper.append_activation(pre_act)


def sequence_softmax(input, use_cudnn=False, name=None):
    r"""
	:api_attr: Static Graph

    **Note**:
    
    **The input type of the OP must be LoDTensor. For Tensor, use:** :ref:`api_fluid_layers_softmax` 

    A LoD-tensor can be regarded as several sequences, and this op apply softmax algo on each sequence.
    The shape of input Tensor can be :math:`[N, 1]` or :math:`[N]`, where :math:`N`
    is the sum of the length of all sequences. Recommended usage: :math:`[N]`.

    For i-th sequence in a mini-batch:

    .. math::

        Out(X[lod[i]:lod[i+1]], :) = \\frac{\exp(X[lod[i]:lod[i+1], :])}{\sum(\exp(X[lod[i]:lod[i+1], :]))}

    For example, for a LoD-Tensor with 6 sequences ([3, 2, 4, 1, 2, 3] - sequence length list in order), 
    the lod in the runtime is [[0, 3, 5, 9, 10, 12, 15]],
    then softmax will be computed among :math:`X[0:3,:],X[3:5,:],X[5:9,:],X[9:10,:],X[10:12,:],X[12:15,:]`,
    and :math:`N` turns out to be 15.

    .. code-block:: text

        *Case 1:

            Given:
                input.data = [0.7, 1, 0.6,
                              1.5, 1.1,
                              1.2, 0.2, 0.6, 1.9,
                              3.1,
                              2.5, 0.8,
                              0.1, 2.4, 1.3]
                input.lod = [[0, 3, 5, 9, 10, 12, 15]]
            then:
                 output.data = [0.30724832, 0.41474187, 0.2780098,
                                0.59868765, 0.40131235,
                                0.2544242, 0.09359743, 0.13963096, 0.5123474, 
                                1.,
                                0.84553474, 0.15446526,
                                0.06995796, 0.69777346, 0.23226859]
                 output.lod = [[0, 3, 5, 9, 10, 12, 15]]    
    

    Args:
        input (Variable):A LoDTensor with shape of  :math:`[N, 1]` or  :math:`[N]`, Recommended usage: :math:`[N]`. 
                         Supported data types: float32, float64. 
        use_cudnn (bool, optional): Use cudnn kernel or not. Effective only when the cudnn version of the paddle 
                                    library is installed and GPU is used for training or reasoning. Default: False.
        name (str, optional): The default value is None. Normally there is no need for user to set this property. 
                              For more information, please refer to :ref:`api_guide_Name`

    Returns:
        Variable: A LoD-Tensor which has the same shape and data type with input.

    Examples:

        .. code-block:: python
             
             import paddle
             paddle.enable_static()
             
             x = paddle.static.data(name='x', shape=[7, 1],
                              dtype='float32', lod_level=1)
             x_sequence_softmax_1 = paddle.static.nn.sequence_softmax(input=x)  

             y = paddle.static.data(name='y', shape=[7],
                 dtype='float32', lod_level=1)
             x_sequence_softmax_2 = paddle.static.nn.sequence_softmax(input=y)  
    """
    assert not in_dygraph_mode(), (
        "sequence layer is not supported in dygraph mode yet.")
    helper = LayerHelper('sequence_softmax', **locals())
    check_variable_and_dtype(input, 'input', ['float32', 'float64'],
                             'sequence_softmax')
    dtype = helper.input_dtype()
    softmax_out = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type="sequence_softmax",
        inputs={"X": input},
        outputs={"Out": softmax_out},
        attrs={"use_cudnn": use_cudnn})
    return softmax_out


def sequence_pool(input, pool_type, is_test=False, pad_value=0.0):
    r"""
	:api_attr: Static Graph

    **Notes: The Op only receives LoDTensor as input. If your input is Tensor, please use pool2d Op.(fluid.layers.** :ref:`api_fluid_layers_pool2d` ).

    This operator only supports LoDTensor as input. It will apply specified pooling
    operation on the input LoDTensor. It pools features of all time-steps of each
    sequence at the last lod_level using :attr:`pool_type` mentioned in the parameters,
    such as sum, average, sqrt, etc.

    It supports six pool_type:

    - average: :math:`Out[i] = \\frac{\sum_i X_i}{N}`
    - sum:     :math:`Out[i] = \sum_jX_{ij}`
    - sqrt:    :math:`Out[i] = \\frac{\sum_jX_{ij}}{\sqrt{len(X_i)}}`
    - max:     :math:`Out[i] = max(X_i)`
    - last:    :math:`Out[i] = X_{N_i}`
    - first:   :math:`Out[i]` = X_0

    where :math:`N_i` is the length of i-th input sequence.

    .. code-block:: text

        Case 1:
        input is a 1-level LoDTensor and pad_value = 0.0:
            input.lod = [[0, 2, 5, 7, 7]]
            input.data = [[1.], [3.], [2.], [4.], [6.], [5.], [1.]]
            input.shape = [7, 1]

        output is LoDTensor:
            out.shape = [4, 1]
            with condition out.shape[0] == len(x.lod[-1]) == 4

        for different pool_type:
            average: out.data = [[2.], [4.], [3.], [0.0]], where 2.=(1. + 3.)/2, 4.=(2. + 4. + 6.)/3, 3.=(5. + 1.)/2
            sum    : out.data = [[4.], [12.], [6.], [0.0]], where 4.=1. + 3., 12.=2. + 4. + 6., 6.=5. + 1.
            sqrt   : out.data = [[2.82], [6.93], [4.24], [0.0]], where 2.82=(1. + 3.)/sqrt(2), 6.93=(2. + 4. + 6.)/sqrt(3), 4.24=(5. + 1.)/sqrt(2)
            max    : out.data = [[3.], [6.], [5.], [0.0]], where 3.=max(1., 3.), 6.=max(2., 4., 6.), 5.=max(5., 1.)
            last   : out.data = [[3.], [6.], [1.], [0.0]], where 3.=last(1., 3.), 6.=last(2., 4., 6.), 1.=last(5., 1.)
            first  : out.data = [[1.], [2.], [5.], [0.0]], where 1.=first(1., 3.), 2.=first(2., 4., 6.), 5.=first(5., 1.)

            and all above [0.0] at last of out.data is padding data.

        Case 2:
        input is a 2-level LoDTensor containing 3 sequences with length info [2, 0, 3],
        where 0 means empty sequence.
        The first sequence contains 2 subsequence with length info [1, 2];
        The last sequence contains 3 subsequence with length info [1, 0, 3].
            input.lod = [[0, 2, 2, 5], [0, 1, 3, 4, 4, 7]]
            input.data = [[1.], [3.], [2.], [4.], [6.], [5.], [1.]]
            input.shape = [7, 1]

        If pool_typ = sum, it will apply pooling on last lod_level [0, 1, 3, 4, 4, 7]. pad_value = 0.0
        output is LoDTensor:
            out.shape= [5, 1]
            out.lod = [[0, 2, 2, 5]]
            where out.shape[0] == len(x.lod[-1]) == 5
            sum: out.data = [[1.], [5.], [4.], [0.0], [12.]]
            where 1.=1., 5.=3. + 2., 4.=4., 0.0=pad_value, 12.=6. + 5. + 1.

    Args:
        input (variable): LoDTensor with lod_level no more than 2. The data type should be float32 or float64.
        pool_type (str): The pooling type that supports average, sum, sqrt, max, last or first.
        is_test (bool): Only works when :attr:`pool_type` is max. If set False, a temporary Tenosr maxIndex is
            created to record the index information corresponding to the maximum value, which is used for backward
            gradient calculation in the training phase. Default: False.
        pad_value (float): Used to pad the pooling result for empty input sequence. Default: 0.0

    Returns:
        Variable: LoDTensor after pooling with data type float32 or float64.

    Examples:

        .. code-block:: python

            import paddle
            paddle.enable_static()

            x = paddle.static.data(name='x', shape=[None, 10], dtype='float32', lod_level=1)
            avg_x = paddle.static.nn.sequence_pool(input=x, pool_type='average')
            sum_x = paddle.static.nn.sequence_pool(input=x, pool_type='sum')
            sqrt_x = paddle.static.nn.sequence_pool(input=x, pool_type='sqrt')
            max_x = paddle.static.nn.sequence_pool(input=x, pool_type='max')
            last_x = paddle.static.nn.sequence_pool(input=x, pool_type='last')
            first_x = paddle.static.nn.sequence_pool(input=x, pool_type='first')
    """
    assert not in_dygraph_mode(), (
        "sequence layer is not supported in dygraph mode yet.")
    check_variable_and_dtype(input, 'input', ['float32', 'float64'],
                             'sequence_pool')
    helper = LayerHelper('sequence_pool', **locals())
    dtype = helper.input_dtype()
    pool_out = helper.create_variable_for_type_inference(dtype)
    max_index = helper.create_variable_for_type_inference(dtype)

    helper.append_op(
        type="sequence_pool",
        inputs={"X": input},
        outputs={"Out": pool_out,
                 "MaxIndex": max_index},
        attrs={
            "pooltype": pool_type.upper(),
            "is_test": is_test,
            "pad_value": pad_value
        })

    # when pool_type is max, variable max_index is initialized,
    # so we stop the gradient explicitly here
    if pool_type == 'max':
        max_index.stop_gradient = True

    return pool_out


@templatedoc()
def sequence_concat(input, name=None):
    """
	:api_attr: Static Graph

    **Notes: The Op only receives LoDTensor as input. If your input is Tensor, please use concat Op.(fluid.layers.** :ref:`api_fluid_layers_concat` ).

    This operator only supports LoDTensor as input. It concatenates the multiple LoDTensor from input by the LoD information,
    and outputs the concatenated LoDTensor.

    .. code-block:: text

        input is a list of LoDTensor:
            input = [x1, x2]
        where:
            x1.lod = [[0, 3, 5]]
            x1.data = [[1], [2], [3], [4], [5]]
            x1.shape = [5, 1]

            x2.lod = [[0, 2, 4]]
            x2.data = [[6], [7], [8], [9]]
            x2.shape = [4, 1]
        and should satisfy: len(x1.lod[0]) == len(x2.lod[0])

        output is LoDTensor:
            out.lod = [[0, 3+2, 5+4]]
            out.data = [[1], [2], [3], [6], [7], [4], [5], [8], [9]]
            out.shape = [9, 1]

    Args:
        input(list of Variable): List of LoDTensor to be concatenated. The length of each LoDTensor should be same.
            The data type can be float32, float64 or int64.
        name(str, optional): The default value is None.  Normally there is no need for user to set this property.
            For more information, please refer to :ref:`api_guide_Name` .

    Returns:
        Variable: Output the concatenated LoDTensor. The data type is same as input.

    Examples:
        .. code-block:: python

            import paddle
            paddle.enable_static()

            x = paddle.static.data(name='x', shape=[-1, 10], dtype='float32', lod_level=1)
            y = paddle.static.data(name='y', shape=[-1, 10], dtype='float32', lod_level=1)
            out = paddle.static.nn.sequence_concat(input=[x, y])
    """
    assert not in_dygraph_mode(), (
        "sequence layer is not supported in dygraph mode yet.")
    helper = LayerHelper('sequence_concat', **locals())

    check_type(input, 'input', list, 'fluid.layers.sequence_concat')
    for i, input_x in enumerate(input):
        check_variable_and_dtype(input_x, 'input[' + str(i) + ']',
                                 ['int64', 'float32', 'float64'],
                                 'fluid.layers.sequence_concat')

    out = helper.create_variable_for_type_inference(dtype=helper.input_dtype())
    helper.append_op(
        type='sequence_concat', inputs={'X': input}, outputs={'Out': [out]})
    return out


def sequence_first_step(input):
    """
	:api_attr: Static Graph

    This operator only supports LoDTensor as input. Given the input LoDTensor, it will
    select first time-step feature of each sequence as output.

    .. code-block:: text

       Case 1:
        input is 1-level LoDTensor:
            input.lod = [[0, 2, 5, 7]]
            input.data = [[1.], [3.], [2.], [4.], [6.], [5.], [1.]]
            input.shape = [7, 1]

        output is a LoDTensor:
            out.shape = [3, 1]
            out.shape[0] == len(x.lod[-1]) == 3
            out.data = [[1.], [2.], [5.]], where 1.=first(1., 3.), 2.=first(2., 4., 6.), 5.=first(5., 1.)

        Case 2:
        input is a 2-level LoDTensor containing 3 sequences with length info [2, 0, 3],
        where 0 means empty sequence.
        The first sequence contains 2 subsequence with length info [1, 2];
        The last sequence contains 3 subsequence with length info [1, 0, 3].
            input.lod = [[0, 2, 2, 5], [0, 1, 3, 4, 4, 7]]
            input.data = [[1.], [3.], [2.], [4.], [6.], [5.], [1.]]
            input.shape = [7, 1]

        It will apply pooling on last lod_level [0, 1, 3, 4, 4, 7]. pad_value = 0.0
        output is a LoDTensor:
            out.shape= [5, 1]
            out.lod = [[0, 2, 2, 5]]
            out.shape[0] == len(x.lod[-1]) == 5
            out.data = [[1.], [3.], [4.], [0.0], [6.]]
            where 1.=first(1.), 3.=first(3., 2.), 4.=first(4.), 0.0 = pad_value, 6.=first(6., 5., 1.)

    Args:
        input(Variable): LoDTensor with lod_level no more than 2. The data type should be float32 or float64.

    Returns:
        Variable: LoDTensor consist of the sequence's first step vector. The data type is float32 or float64.

    Examples:

        .. code-block:: python

             import paddle
             paddle.enable_static()

             x = paddle.static.data(name='x', shape=[None, 10], dtype='float32', lod_level=1)
             x_first_step = paddle.static.nn.sequence_first_step(input=x)
    """
    check_variable_and_dtype(input, 'input', ['float32', 'float64'],
                             'sequence_first_step')
    return sequence_pool(input=input, pool_type="first")


def sequence_last_step(input):
    """
	:api_attr: Static Graph

    This operator only supports LoDTensor as input. Given the input LoDTensor, it will
    select last time-step feature of each sequence as output.

    .. code-block:: text

        Case 1:
        input is 1-level LoDTensor:
            input.lod = [[0, 2, 5, 7]]
            input.data = [[1.], [3.], [2.], [4.], [6.], [5.], [1.]]
            input.shape = [7, 1]

        output is a LoDTensor:
            out.shape = [3, 1]
            out.shape[0] == len(x.lod[-1]) == 3
            out.data = [[3.], [6.], [1.]], where 3.=last(1., 3.), 6.=last(2., 4., 6.), 1.=last(5., 1.)

        Case 2:
        input is a 2-level LoDTensor containing 3 sequences with length info [2, 0, 3],
        where 0 means empty sequence.
        The first sequence contains 2 subsequence with length info [1, 2];
        The last sequence contains 3 subsequence with length info [1, 0, 3].
            input.lod = [[0, 2, 2, 5], [0, 1, 3, 4, 4, 7]]
            input.data = [[1.], [3.], [2.], [4.], [6.], [5.], [1.]]
            input.shape = [7, 1]

        It will apply pooling on last lod_level [0, 1, 3, 4, 4, 7]. pad_value = 0.0
        output is a LoDTensor:
            out.shape= [5, 1]
            out.lod = [[0, 2, 2, 5]]
            out.shape[0] == len(x.lod[-1]) == 5
            out.data = [[1.], [2.], [4.], [0.0], [1.]]
            where 1.=last(1.), 2.=last(3., 2.), 4.=last(4.), 0.0 = pad_value, 1=last(6., 5., 1.)


    Args:
        input(Variable): LoDTensor with lod_level no more than 2. The data type should be float32.

    Returns:
        Variable: LoDTensor consist of the sequence's last step vector. The data type is float32.

    Examples:

        .. code-block:: python

             import paddle
             paddle.enable_static()
             
             x = paddle.static.data(name='x', shape=[None, 10], dtype='float32', lod_level=1)
             x_last_step = paddle.static.nn.sequence_last_step(input=x)
    """
    check_variable_and_dtype(input, 'input', ['float32', 'float64'],
                             'sequence_last_step')
    return sequence_pool(input=input, pool_type="last")


def sequence_slice(input, offset, length, name=None):
    """
	:api_attr: Static Graph

    **Sequence Slice Layer**

    The layer crops a subsequence from given sequence with given start
    offset and subsequence length.

    It only supports sequence data (LoDTensor with lod_level equal to 1).

    .. code-block:: text

              - Case:

            Given the input Variable **input**:

                input.data = [[a1, a2], [b1, b2], [c1, c2], [d1, d2], [e1, e2]],
                input.lod = [[3, 2]],
                input.dims = (5, 2),

            with offset.data = [[0], [1]] and length.data = [[2], [1]],

            the output Variable will be

                out.data = [[a1, a2], [b1, b2], [e1, e2]],
                out.lod = [[2, 1]],
                out.dims = (3, 2).

    Note:
          The first dimension size of **input**, **offset** and **length**
          should be equal. The **offset** should start from 0.

    Args:
        input(Variable): LoDTensor, The input Variable which consists of the complete
                         sequences.The data type can be float32, float64, int32 or int64
        offset(Variable): LoDTensor, The offset to slice each sequence. The data
                         type is int32 or int64.
        length(Variable): LoDTensor, The length of each subsequence. The data
                         type is int32 or int64.
        name(str|None): The default value is None.  Normally there is no need
                        for user to set this property.  For more information,
                        please refer to :ref:`api_guide_Name`

    Returns:
        Variable: The output subsequences.

    Examples:

        .. code-block:: python

             import paddle
             paddle.enable_static()
             
             import numpy as np
             seqs = paddle.static.data(name='x', shape=[10, 5],
                              dtype='float32', lod_level=1)
             offset = paddle.assign(np.array([[0, 1]]).astype("int32"))
             length = paddle.assign(np.array([[2, 1]]).astype("int32"))
             subseqs = paddle.static.nn.sequence_slice(input=seqs, offset=offset,
                                                   length=length)
    """
    assert not in_dygraph_mode(), (
        "sequence layer is not supported in dygraph mode yet.")
    helper = LayerHelper("sequence_slice", **locals())

    check_variable_and_dtype(input, 'input',
                             ['float32', 'float64', 'int32', 'int64'],
                             'sequence_slice')
    check_variable_and_dtype(offset, 'offset', ['int32', 'int64'],
                             'sequence_slice')
    check_variable_and_dtype(length, 'length', ['int32', 'int64'],
                             'sequence_slice')

    dtype = helper.input_dtype()
    out = helper.create_variable_for_type_inference(dtype)

    offset.stop_gradient = True
    length.stop_gradient = True

    helper.append_op(
        type="sequence_slice",
        inputs={"X": input,
                "Offset": offset,
                "Length": length},
        outputs={"Out": out})

    return out


def sequence_expand(x, y, ref_level=-1, name=None):
    r"""
	:api_attr: Static Graph

        Sequence Expand Layer. This layer will expand the input variable ``x`` \
        according to specified level ``ref_level`` lod of ``y``. Please note that \
        the lod level of ``x`` is at most 1. If the lod level of ``x`` is 1, than \
        the size of lod of ``x`` must be equal to the length of ``ref_level`` lod \
        of ``y``. If the lod level of ``x`` is 0, then the first dim of ``x`` should \
        be equal to the size of ``ref_level`` of ``y``. The rank of **x** is at least 2. \
        When rank of ``x`` is greater than 2, then it would be viewed as a 2-D tensor.

    Please note that the input ``x`` should be LodTensor or Tensor, \
        and input ``y`` must be LodTensor.

    Following examples will explain how sequence_expand works:

    .. code-block:: text

        Case 1

        Consider 2 sequences [a][b] and [c][d], now we want to expand them to [a][b], [a][b], [c][d] and [c][d].
        Sequence [a][b] expand twice and [c][d] expands twice, so the lod which according to is [2, 2].

        Input x is a 1-level LoDTensor:
            x.lod  = [[2,        2]]    #lod based on length may be easier to understand
            x.data = [[a], [b], [c], [d]]
            x.dims = [4, 1]

        input y is a LoDTensor:
            y.lod = [[2,    2],    #the 0th level lod, according to this level
                     [3, 3, 1, 1]] #the 1st level lod, it has nothing to do with this level

        ref_level: 0

        then output is a 1-level LoDTensor out:
            out.lod =  [[2,        2,        2,        2]]    #lod based on offset
            out.data = [[a], [b], [a], [b], [c], [d], [c], [d]]
            out.dims = [8, 1]


        Case 2

        Consider 3 sequences [a], [b], [c], now we want to expand them to [a][a], [c][c][c].
        It's obvious that the lod info of expanded sequences is [2, 0, 3].

        x is a Tensor:
            x.data = [[a], [b], [c]]
            x.dims = [3, 1]

        y is a LoDTensor:
            y.lod = [[2, 0, 3]]

        ref_level: -1

        then output is a 1-level LodTensor:
            out.data = [[a], [a], [c], [c], [c]]
            out.dims = [5, 1]

    Args:
        x (Variable): The input variable which is a Tensor or LoDTensor, with the \
            dims ``[M, K]``. The lod level is at most 1. The data type should be \
            float32, float64, int32 or int64.
        y (Variable): The input variable which is a LoDTensor, the lod level is \
            at least 1.
        ref_level (int): Lod level of ``y`` to be referred by ``x``. If set to -1, \
                         refer the last level of lod.
        name(str, optional): For detailed information, please refer \
            to :ref:`api_guide_Name`. Usually name is no need to set and \
            None by default. 

    Returns: The expanded variable which is a LoDTensor, with dims ``[N, K]``. \
            ``N`` depends on the lod info of ``x`` and ``y``. \
            The data type is same as input.

    Return Type: Variable

    Examples:
        .. code-block:: python
	
            import paddle
            from paddle import fluid
            paddle.enable_static()
            import numpy as np

            x = paddle.static.data(name='x', shape=[4, 1], dtype='float32')
            y = paddle.static.data(name='y', shape=[8, 1],
                        dtype='float32', lod_level=1)
            out = paddle.static.nn.sequence_expand(x=x, y=y, ref_level=0)

            exe = paddle.static.Executor(fluid.CPUPlace())
            place = paddle.CPUPlace()

            np_data = np.array([[1], [2], [3], [4]]).astype('float32')
            x_lod_tensor = fluid.create_lod_tensor(np_data, [[2, 2]], place)
            print(x_lod_tensor)
            #lod: [[0, 2, 4]]
            #    dim: 4, 1
            #    layout: NCHW
            #    dtype: float
            #    data: [1 2 3 4]

            np_data = np.array([[1], [2], [3], [4], [5], [6], [7], [8]]).astype('float32')
	    y_lod_tensor = fluid.create_lod_tensor(np_data, [[2, 2], [3,3,1,1]], place)
            print(y_lod_tensor)
            #lod: [[0, 2, 4][0, 3, 6, 7, 8]]
            #    dim: 8, 1
            #    layout: NCHW
            #    dtype: int64_t
            #    data: [0 0 1 1 1 1 1 0]

            out_main = exe.run(fluid.default_main_program(),
                            feed={'x': x_lod_tensor, 'y': y_lod_tensor},
                            fetch_list=[out], return_numpy=False)
            print(out_main[0])
            #lod: [[0, 2, 4, 6, 8]]
            #    dim: 8, 1
            #    layout: NCHW
            #    dtype: float
            #    data: [1 2 1 2 3 4 3 4]
    """
    assert not in_dygraph_mode(), (
        "sequence layer is not supported in dygraph mode yet.")
    check_variable_and_dtype(x, 'x', ['float32', 'float64', 'int32', 'int64'],
                             'sequence_expand')
    helper = LayerHelper('sequence_expand', **locals())
    dtype = helper.input_dtype(input_param_name='x')
    tmp = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type='sequence_expand',
        inputs={'X': x,
                'Y': y},
        outputs={'Out': tmp},
        attrs={'ref_level': ref_level})
    return tmp


def sequence_expand_as(x, y, name=None):
    r"""
	:api_attr: Static Graph

        Sequence Expand As Layer. This OP will expand the input variable ``x`` \
        according to the zeroth level lod of ``y``. Current implementation requires \
        the level number of ``y``'s lod must be 1, and the first dimension of \
        ``x`` should be equal to the size of ``y``'s zeroth level lod, thus \
        the expanded LodTensor has the same lod info as ``y``. The expanded result \
        has nothing to do with ``x``'s lod, so the lod of Input(X) is not considered.

    Please note that the input ``x`` should be LodTensor or Tensor, \
        and input ``y`` must be LodTensor.

    Following examples will explain how sequence_expand_as works:

    .. code-block:: text

        Case 1:

        Consider 4 sequences [a], [b], [c], [d], now we want to expand them to [a][a][a], [b][b][b], [c] and [d].
        It's obvious that the lod info of expanded sequences is [0, 3, 6, 7, 8].
        Given a 1-level LodTensor ``x``: 
            x.data = [[a], [b], [c], [d]]
            x.dims = [4, 1]
        and input ``y``
            y.lod = [[3, 3, 1, 1]] #lod based on length may be easier to understand

        then we get 1-level LoDTensor out:
            Out.lod =  [[0,            3,              6,  7,  8]] #based on offset
            Out.data = [[a], [a], [a], [b], [b], [b], [c], [d]]
            Out.dims = [8, 1]


        Case 2:

        Given a common Tensor ``x``:
            x.data = [[a, b], [c, d], [e, f]]
            x.dims = [3, 2]
        and input ``y``:
            y.lod = [[0, 2, 3, 6]]

        then we get a 1-level LoDTensor:
            out.lod =  [[0,             2,     3,                    6]]
            out.data = [[a, b], [a, b] [c, d], [e, f], [e, f], [e, f]]
            out.dims = [6, 2]

    Args:
        x (Variable): The input variable which is a Tensor or LoDTensor, with the \
            dims ``[M, K]``. The data type should be float32, float64, int32 \
            or int64.
        y (Variable): The input variable which is a LoDTensor with 1-level lod.
        name (str, optional): For detailed information, please refer \
            to :ref:`api_guide_Name`. Usually name is no need to set and \
            None by default.

    Returns: The expanded variable which is a LoDTensor with the dims ``[N, K]``. \
            ``N`` depends on the lod of ``y``, and the lod level must be 1. \
            The data type is same as input.

    Return Type: Variable

    Examples:
        .. code-block:: python

            import paddle
            import paddle.fluid as fluid
            paddle.enable_static()
            import numpy as np

            x = paddle.static.data(name='x', shape=[4, 1], dtype='float32')
            y = paddle.static.data(name='y', shape=[8, 1], dtype='float32', lod_level=1)
            out = paddle.static.nn.sequence_expand_as(x=x, y=y)

            exe = fluid.Executor(fluid.CPUPlace())
            place = fluid.CPUPlace()

            np_data = np.array([[1], [2], [3], [4]]).astype('float32')
            x_lod_tensor = fluid.create_lod_tensor(np_data, [[2, 2]], place)
            print(x_lod_tensor)
            #lod: [[0, 2, 4]]
            #    dim: 4, 1
            #    layout: NCHW
            #    dtype: float
            #    data: [1 2 3 4]

            np_data = np.array([[1], [2], [3], [4], [5], [6], [7], [8]]).astype('float32')
	    y_lod_tensor = fluid.create_lod_tensor(np_data, [[3,3,1,1]], place)
            print(y_lod_tensor)
            #lod: [[0, 3, 6, 7, 8]]
            #    dim: 8, 1
            #    layout: NCHW
            #    dtype: int64_t
            #    data: [0 0 1 0 1 1 1 0]

            out_main = exe.run(fluid.default_main_program(),
                            feed={'x': x_lod_tensor, 'y': y_lod_tensor},
                            fetch_list=[out], return_numpy=False)
            print(out_main[0])
            #lod: [[0, 3, 6, 7, 8]]
            #    dim: 8, 1
            #    layout: NCHW
            #    dtype: float
            #    data: [1 1 1 2 2 2 3 4]
    """
    assert not in_dygraph_mode(), (
        "sequence layer is not supported in dygraph mode yet.")
    check_variable_and_dtype(x, 'x', ['float32', 'float64', 'int32', 'int64'],
                             'sequence_expand_as')
    check_type(y, 'y', Variable, 'sequence_expand_as')
    helper = LayerHelper('sequence_expand_as', **locals())
    dtype = helper.input_dtype(input_param_name='x')
    tmp = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type='sequence_expand_as',
        inputs={'X': x,
                'Y': y},
        outputs={'Out': tmp})
    return tmp


def sequence_pad(x, pad_value, maxlen=None, name=None):
    r"""
	:api_attr: Static Graph

        This layer padding the sequences in a same batch to a common length (according 
        to ``maxlen``). The padding value is defined by ``pad_value``, and will be 
        appended to the tail of sequences. The result is a Python tuple ``(Out, Length)``: 
        the LodTensor ``Out`` is the padded sequences, and LodTensor ``Length`` is 
        the length information of input sequences. For removing padding data (unpadding operation), See :ref:`api_fluid_layers_sequence_unpad`.

        Please note that the input ``x`` should be LodTensor.

    .. code-block:: text

        Case 1:
        Given input 1-level LoDTensor x:
            x.lod = [[0,  2,   5]]
            x.data = [[a],[b],[c],[d],[e]]
        pad_value:
            pad_value.data = [0]
        maxlen = 4

        the output tuple (Out, Length):
            Out.data = [[[a],[b],[0],[0]],[[c],[d],[e],[0]]]
            Length.data = [2, 3]      #Original sequences length

        Case 2:
        Given input 1-level LoDTensor x:
            x.lod =  [[0,             2,                     5]]
            x.data = [[a1,a2],[b1,b2],[c1,c2],[d1,d2],[e1,e2]]
        pad_value:
            pad_value.data = [0]
        default maxlen = None, (the virtual value is 3, according to the shape of x)

        the output tuple (Out, Length):
            Out.data = [[[a1,a2],[b1,b2],[0,0]],[[c1,c2],[d1,d2],[e1,e2]]]
            Length.data = [2, 3]

        Case 3:
        Given input 1-level LoDTensor x:
            x.lod =  [[0,             2,                     5]]
            x.data = [[a1,a2],[b1,b2],[c1,c2],[d1,d2],[e1,e2]]
        pad_value:
            pad_value.data = [p1,p2]
        default maxlen = None, (the virtual value is 3)

        get tuple (Out, Length):
            Out.data = [[[a1,a2],[b1,b2],[p1,p2]],[[c1,c2],[d1,d2],[e1,e2]]]
            Length.data = [2, 3]



    Args:
        x (Variable): Input 1-level LodTensor with dims ``[M, K]``. The batch \
            size is described by lod infor (the number of sequences ). \
            The data type should be float32, float64, int8, int32 or int64.
        pad_value (Variable): Padding value. It can be a scalar or a 1D tensor \
            with length ``K``. If it's a scalar, it will be automatically broadcasted \
            to a Tensor. The data type should be as same as ``x``.
        maxlen (int, optional): The length of padded sequences, None by default. \
            When it is None, all sequences will be padded up to the length of the \
            longest one among them; when it a certain positive value, it must be \
            greater than the length of the longest original sequence.
        name (str, optional): For detailed information, please refer \
            to :ref:`api_guide_Name`. Usually name is no need to set and \
            None by default.

    Returns: A Python tuple (Out, Length): the 1st is a 0 level LodTensor \
            ``Out``, with the shape ``[batch_size, maxlen, K]``; the second is the original \
            sequences length infor ``Length``, which should be a 0-level 1D LodTensor. \
            The size of ``Length`` is equal to batch size, and the data type is int64.

    Return Type: tuple

    Examples:
        .. code-block:: python

            import paddle
            paddle.enable_static()
            import paddle.fluid as fluid
            import numpy

            x = paddle.static.data(name='x', shape=[10, 5], dtype='float32', lod_level=1)
            pad_value = paddle.assign(
                numpy.array([0.0], dtype=numpy.float32))
            out = paddle.static.nn.sequence_pad(x=x, pad_value=pad_value)
    """

    assert not in_dygraph_mode(), (
        "sequence layer is not supported in dygraph mode yet.")
    helper = LayerHelper('sequence_pad', **locals())
    check_variable_and_dtype(x, 'x', ['float32', 'float64', 'int32', 'int64'],
                             'fluid.layers.sequence_pad')
    check_variable_and_dtype(pad_value, 'pad_value',
                             ['float32', 'float64', 'int32', 'int64'],
                             'fluid.layers.sequence_pad')
    dtype = helper.input_dtype(input_param_name='x')
    out = helper.create_variable_for_type_inference(dtype)
    length = helper.create_variable_for_type_inference(VarDesc.VarType.INT64)

    pad_value.stop_gradient = True
    length.stop_gradient = True

    if maxlen is None:
        maxlen = -1
    helper.append_op(
        type='sequence_pad',
        inputs={'X': x,
                'PadValue': pad_value},
        outputs={'Out': out,
                 'Length': length},
        attrs={'padded_length': maxlen})
    return out, length


def sequence_unpad(x, length, name=None):
    """
	:api_attr: Static Graph

    **Note**:
    
    **The input of the OP is Tensor and the output is LoDTensor.  For padding operation, See:**  :ref:`api_fluid_layers_sequence_pad`  
     
    The OP removes the padding data from the input based on the length information and returns a LoDTensor.

    .. code-block:: text

	Case 1:

	Given input Variable **x**:
	    x.data = [[ 1.0,  2.0,  3.0,  4.0,  5.0],
		      [ 6.0,  7.0,  8.0,  9.0, 10.0],
		      [11.0, 12.0, 13.0, 14.0, 15.0]],

	in which there are 3 sequences padded to length 5, and the actual length
	specified by input Variable **length**:

	    length.data = [2, 3, 4],

	after unpadding, the output Variable will be:

	    out.data = [[1.0, 2.0, 6.0, 7.0, 8.0, 11.0, 12.0, 13.0, 14.0]]
	    out.lod = [[0, 2, 5, 9]]

    Args:
        x(Variable): A Tensor which contains padding data, and its shape size can not be less than 2.
                     Supported data types: float32, float64, int32, int64.
        length(Variable): A 1D Tensor that stores the actual length of each sample, and the Tensor 
                          has the same shape with the 0th dimension of the X . Supported data types: int64.
        name(str|None):  The default value is None.  Normally there is no need for user to set this property.  
                         For more information, please refer to :ref:`api_guide_Name`

    Returns:
        Variable: A LoDTensor whose recursive sequence length is consistent with the information of the length parameter and it has the same data type with input.

    Examples:
        .. code-block:: python

            import paddle
            paddle.enable_static()
            import paddle.fluid as fluid
            import numpy

            # pad data
            x = paddle.static.data(name='x', shape=[10, 5], dtype='float32', lod_level=1)
            pad_value = paddle.assign(numpy.array([0.0], dtype=numpy.float32))
            pad_data, len = paddle.static.nn.sequence_pad(x=x, pad_value=pad_value)
            
            # unpad data
            unpad_data = paddle.static.nn.sequence_unpad(x=pad_data, length=len)
    """

    assert not in_dygraph_mode(), (
        "sequence layer is not supported in dygraph mode yet.")
    helper = LayerHelper('sequence_unpad', **locals())
    check_variable_and_dtype(x, 'x', ['float32', 'float64', 'int32', 'int64'],
                             'fluid.layers.sequence_unpad')
    check_variable_and_dtype(length, 'length', ['int64'],
                             'fluid.layers.sequence_unpad')
    dtype = helper.input_dtype(input_param_name='x')
    out = helper.create_variable_for_type_inference(dtype)

    length.stop_gradient = True

    helper.append_op(
        type='sequence_unpad',
        inputs={'X': x,
                'Length': length},
        outputs={'Out': out})
    return out


def sequence_reshape(input, new_dim):
    """
	:api_attr: Static Graph

    **Notes: The Op only receives LoDTensor as input. If your input is Tensor, please use reshape Op.(fluid.layers.** :ref:`api_fluid_layers_reshape` ).

    This operator only supports LoDTensor as input. Given :attr:`new_dim` ,
    it will compute new shape according to original length of each sequence,
    original dimensions and :attr:`new_dim` . Then it will output a new LoDTensor
    containing :attr:`new_dim` . Currently it only supports 1-level LoDTensor.
    Please make sure that (original length * original dimensions) can be divided
    by the :attr:`new_dim` with no remainder for each sequence.

    .. code-block:: text

        input is a LoDTensor:
            input.lod  = [[0, 2, 6]]
            input.data = [[1,  2], [3,  4],
                          [5,  6], [7,  8],
                          [9, 10], [11, 12]]
            input.shape = [6, 2]

        set new_dim = 4
        out is a LoDTensor:
            out.lod  = [[0, 1, 3]]
            out.data = [[1,  2,  3,  4],
                        [5,  6,  7,  8],
                        [9, 10, 11, 12]]
            out.shape = [3, 4]


    Args:

       input (Variable): 1-level LoDTensor with shape :math:`[M, K]` . The data type should
            be int32, int64, float32 or float64.
       new_dim (int): New dimension that the input LoDTensor is reshaped to.

    Returns:
        Variable: Reshaped LoDTensor according to new dimension. The data type is same as input.

    Examples:
        .. code-block:: python

            import paddle
            paddle.enable_static()

            x = paddle.static.data(name='x', shape=[None, 16], dtype='float32', lod_level=1)
            x_reshaped = paddle.static.nn.sequence_reshape(input=x, new_dim=4)
    """
    assert not in_dygraph_mode(), (
        "sequence layer is not supported in dygraph mode yet.")
    helper = LayerHelper('sequence_reshape', **locals())
    check_variable_and_dtype(input, 'input',
                             ['float32', 'float64', 'int32', 'int64'],
                             'fluid.layers.sequence_reshape')
    out = helper.create_variable_for_type_inference(helper.input_dtype())
    helper.append_op(
        type='sequence_reshape',
        inputs={'X': [input]},
        outputs={'Out': [out]},
        attrs={'new_dim': new_dim})
    return out


def sequence_scatter(input, index, updates, name=None):
    """
	:api_attr: Static Graph

    **Note**:
    
    **The index and updates parameters of the OP must be LoDTensor.**
     
    Plus the updates data to the corresponding input according to the index.
 
    The updated algorithm is as follows: output[instance_index][index [pos]] = input[instance_index][index [pos]] +  updates[pos], 
    where instance_idx is the K sample corresponding to pos in batch.

    The value of output[i][j] depends on whether j can be found in the i+1th interval of the index. If found, 
    out[i][j] = input[i][j] + update[m] [n], otherwise, out[i][j] = input[i][j].

    For example, in the following example, the lod information for index is divided into three sequences. Among 
    them, because the element 0 can be found in the first interval of the index, it is updated with the value of 
    the corresponding position of the updates, out[0][0] = input[0][0]+updates[0][0] . Because element 1 cannot 
    be found in the third interval of index, out[2][1] = input[2][1].

    .. code-block:: text
        
        *Case 1:

            Given:
                input.data = [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                              [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                              [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]
                              input.dims = [3, 6]

                index.data = [[0], [1], [2], [5], [4], [3], [2], [1], [3], [2], [5], [4]]
                index.lod =  [[0,        3,                       8,                 12]]

                updates.data = [[0.3], [0.3], [0.4], [0.1], [0.2], [0.3], [0.4], [0.0], [0.2], [0.3], [0.1], [0.4]]
                updates.lod =  [[  0,            3,                                 8,                         12]]

            Then:
                out.data = [[1.3, 1.3, 1.4, 1.0, 1.0, 1.0],
                            [1.0, 1.0, 1.4, 1.3, 1.2, 1.1],
                            [1.0, 1.0, 1.3, 1.2, 1.4, 1.1]]
                out.dims = X.dims = [3, 6]

    Args:
        input (Variable): A Tensor with shape of  :math:`[N, k_1... k_n]`. Supported data types: float32, float64, int32, int64.
        index (Variable):  A LoDTensor contains index information. Its LoD level must be 1 and its data type can be int32 or int64.
        updates (Variable): A LodTensor contains updates information. It has the same  LoD level with the index and has the 
                            same data type  with the input. Supported data types: float32, float64, int32, int64.
        name (str, optional): The default value is None.  Normally there is no need for user to set this property.  For more information, 
                              please refer to :ref:`api_guide_Name`

    Returns:
        Variable: A Tensor which has been updated. It has the same shape and data type with input.

    Examples:

        .. code-block:: python
	
            import paddle
            paddle.enable_static()

            input = paddle.static.data(name="x", shape=[None, 3, 6], dtype='float32' )
            index = paddle.static.data(name='index', shape=[12, 1],  dtype='int64', lod_level=1)
            updates = paddle.static.data(name='updates', shape=[12, 1], dtype='float32', lod_level=1)
            output = paddle.static.nn.sequence_scatter(input, index, updates)

    """
    assert not in_dygraph_mode(), (
        "sequence layer is not supported in dygraph mode yet.")
    helper = LayerHelper('sequence_scatter', **locals())

    check_variable_and_dtype(input, 'input',
                             ['float32', 'float64', 'int32', 'int64'],
                             'sequence_scatter')
    check_variable_and_dtype(index, 'index', ['int32', 'int64'],
                             'sequence_scatter')
    check_variable_and_dtype(updates, 'updates',
                             ['float32', 'float64', 'int32', 'int64'],
                             'sequence_scatter')

    dtype = helper.input_dtype()
    out = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type="sequence_scatter",
        inputs={"X": input,
                "Ids": index,
                "Updates": updates},
        outputs={"Out": out})
    return out


def sequence_enumerate(input, win_size, pad_value=0, name=None):
    r"""
	:api_attr: Static Graph

    Generate a new sequence for the input index sequence with \
        shape ``[d_1, win_size]``, which enumerates all the \
        sub-sequences with length ``win_size`` of the input with \
        shape ``[d_1, 1]``, and padded by ``pad_value`` if necessary in generation.

    Please note that the `input` must be LodTensor.

    .. code-block:: text

        Input x:
            x.lod = [[0, 3, 5]]
            x.data = [[1], [2], [3], [4], [5]]
            x.dims = [5, 1]

        Attrs:
            win_size = 2
            pad_value = 0

        Output:
            out.lod = [[0, 3, 5]]
            out.data = [[1, 2], [2, 3], [3, 0], [4, 5], [5, 0]]
            out.dims = [5, 2]


    Args:
        input (Variable): The input variable which is a index sequence, \
            which should be a LodTensor with shape ``[d_1, 1]`` and 1-level lod info. \
            The data type should be int32 or int64.
        win_size (int): The window size for enumerating all sub-sequences.
        pad_value (int, optional): The padding value, default 0.
        name(str, optional): For detailed information, please refer \
            to :ref:`api_guide_Name`. Usually name is no need to set and \
            None by default.

    Returns: The enumerate sequence variable which is a LoDTensor with \
            shape ``[d_1, win_size]`` and 1-level lod info. \
            The data type is same as ``input``.

    Return Type: Variable

    Examples:
        .. code-block:: python

            import paddle
            paddle.enable_static()
            
            x = paddle.static.data(name='x', shape=[-1, 1], dtype='int32', lod_level=1)
            out = paddle.static.nn.sequence_enumerate(input=x, win_size=3, pad_value=0)
    """
    assert not in_dygraph_mode(), (
        "sequence layer is not supported in dygraph mode yet.")
    check_variable_and_dtype(input, 'input', ['int32', 'int64'],
                             'sequence_enumerate')
    helper = LayerHelper('sequence_enumerate', **locals())
    out = helper.create_variable_for_type_inference(
        helper.input_dtype(), stop_gradient=True)
    helper.append_op(
        type='sequence_enumerate',
        inputs={'X': input},
        outputs={'Out': out},
        attrs={'win_size': win_size,
               'pad_value': pad_value})
    return out


def sequence_mask(x, maxlen=None, dtype='int64', name=None):
    r"""
    **SequenceMask Layer**

    This layer outputs a mask according to the input :code:`x` and
    :code:`maxlen` with data type of :code:`dtype`.

    Supposing :code:`x` is a Tensor with shape [d_1, d_2, ..., d_n], the
    :code:`y` is a mask with shape [d_1, d_2, ..., d_n, maxlen], where:

    .. math::

        y(i_1, i_2,..., i_n, j) = (j < x(i_1, i_2,..., i_n))

    .. code-block:: text

        Case:

        Consider input:
            x = [3, 1, 1, 0]    max_len = 4

        then we get out:
            mask = [[1, 1, 1, 0],
                    [1, 0, 0, 0],
                    [1, 0, 0, 0],
                    [0, 0, 0, 0]]

    Args:
        x (Variable): Input tensor of sequence_mask layer, \
            whose elements are integers less than :code:`maxlen`. \
            Tensor or LodTensor with shape [d_1, d_2, ..., d_n].
        maxlen (int, optional): Maximum length of the sequence. If :code:`maxlen` \
                           is None, it would be replace with :math:`max(x)`.
        dtype (np.dtype|paddle.dtype|str, optional): Data type of the output, \
             ``int64`` by default.
        name(str, optional): For detailed information, please refer \
            to :ref:`api_guide_Name`. Usually name is no need to set and \
            None by default.

    Returns: The output sequence mask. Tensor with shape [d_1, d_2, ..., d_n, maxlen] \
            and data type of :code:`dtype`. The data type should be bool, float32, float64, int8, \
            int32 or int64.

    Return Type: Tensor

    Examples:
        .. code-block:: python

            import paddle

            lengths = paddle.to_tensor([10, 9, 8])
            mask = paddle.nn.functional.sequence_mask(lengths)

            print(mask.numpy())
            # [[1 1 1 1 1 1 1 1 1 1]
            #  [1 1 1 1 1 1 1 1 1 0]
            #  [1 1 1 1 1 1 1 1 0 0]]

    """
    helper = LayerHelper('sequence_mask', **locals())
    out = helper.create_variable_for_type_inference(dtype=dtype)

    inputs = {'X': [x]}
    attrs = {'out_dtype': out.dtype}
    if maxlen is not None:
        if isinstance(maxlen, Variable):
            inputs['MaxLenTensor'] = maxlen
        else:
            attrs['maxlen'] = maxlen

    helper.append_op(
        type='sequence_mask', inputs=inputs, outputs={'Y': out}, attrs=attrs)

    out.stop_gradient = True
    return out


@templatedoc()
def sequence_reverse(x, name=None):
    """
    **Notes: The Op only receives LoDTensor as input. If your input is Tensor, please use reverse Op.(fluid.layers.** :ref:`api_fluid_layers_reverse` ).

    This operator only supports LoDTensor as input. It will reverse each sequence for input LoDTensor.
    Currently it only supports 1-level LoDTensor. This operator is very useful when building a
    reverse :ref:`api_fluid_layers_DynamicRNN` network.

    .. code-block:: text

        input(x) is a LoDTensor:
            x.lod  = [[0, 2, 5]]
            x.data = [[1,  2,  3,  4],
                      [5,  6,  7,  8],
                      [9, 10, 11, 12],
                      [13,14, 15, 16],
                      [17,18, 19, 20]]
            x.shape = [5, 4]

        output LoDTensor with same shape and LoD info:
            out.lod  = [[0, 2, 5]]
            out.data = [[5,  6,  7,  8],
                        [1,  2,  3,  4],
                        [17,18, 19, 20],
                        [13,14, 15, 16],
                        [9, 10, 11, 12]]
            out.shape = [5, 4]

    Args:
        x(Variable): LoDTensor with 1-level LoD info. Currently it only supports 1-level LoDTensor.
            The data type should be float32, float64, int8, int32 or int64.
        name(str, optional): The default value is None.  Normally there is no need for user to set this property.
            For more information, please refer to :ref:`api_guide_Name` .

    Returns:
        Variable: LoDTensor reversed from input. The data type is same with input.

    Examples:
        .. code-block:: python

            import paddle
            paddle.enable_static()

            x = paddle.static.data(name='x', shape=[None, 10], dtype='float32', lod_level=1)
            x_reversed = paddle.static.nn.sequence_reverse(x)
    """
    assert not in_dygraph_mode(), (
        "sequence layer is not supported in dygraph mode yet.")
    helper = LayerHelper("sequence_reverse", **locals())
    check_variable_and_dtype(x, 'x',
                             ['float32', 'float64', 'int8', 'int32', 'int64'],
                             'fluid.layers.sequence_reverse')
    out = helper.create_variable_for_type_inference(dtype=x.dtype)

    helper.append_op(
        type="sequence_reverse",
        inputs={"X": x},
        outputs={"Y": out},
        attrs=dict())
    return out
