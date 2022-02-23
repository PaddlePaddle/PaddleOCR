#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.fluid.framework import _global_flags

import numpy as np
from ...device import get_cudnn_version
from ...fluid.framework import in_dygraph_mode
from ...static import Variable
from ...fluid import core, dygraph_utils, get_flags
from ...fluid.layers.utils import convert_to_list, _is_symmetric_padding
from ...fluid.data_feeder import check_variable_and_dtype
from ...framework import ParamAttr
from ...fluid.layer_helper import LayerHelper
from paddle import _C_ops
from ...tensor.manipulation import unsqueeze, squeeze
from ...tensor.math import add
from ...fluid.layers import nn

__all__ = []


def _is_list_or_tuple(input):
    return isinstance(input, (list, tuple))


def _zero_padding_in_batch_and_channel(padding, channel_last):
    if channel_last:
        return list(padding[0]) == [0, 0] and list(padding[-1]) == [0, 0]
    else:
        return list(padding[0]) == [0, 0] and list(padding[1]) == [0, 0]


def _exclude_padding_in_batch_and_channel(padding, channel_last):
    padding_ = padding[1:-1] if channel_last else padding[2:]
    padding_ = [elem for pad_a_dim in padding_ for elem in pad_a_dim]
    return padding_


def _update_padding_nd(padding, channel_last, num_dims):
    if isinstance(padding, str):
        padding = padding.upper()
        if padding not in ["SAME", "VALID"]:
            raise ValueError(
                "Unknown padding: '{}'. It can only be 'SAME' or 'VALID'.".
                format(padding))
        if padding == "VALID":
            padding_algorithm = "VALID"
            padding = [0] * num_dims
        else:
            padding_algorithm = "SAME"
            padding = [0] * num_dims
    elif _is_list_or_tuple(padding):
        # for padding like
        # [(pad_before, pad_after), (pad_before, pad_after), ...]
        # padding for batch_dim and channel_dim included
        if len(padding) == 2 + num_dims and _is_list_or_tuple(padding[0]):
            if not _zero_padding_in_batch_and_channel(padding, channel_last):
                raise ValueError(
                    "Non-zero padding({}) in the batch or channel dimensions "
                    "is not supported.".format(padding))
            padding_algorithm = "EXPLICIT"
            padding = _exclude_padding_in_batch_and_channel(padding,
                                                            channel_last)
            if _is_symmetric_padding(padding, num_dims):
                padding = padding[0::2]
        # for padding like [pad_before, pad_after, pad_before, pad_after, ...]
        elif len(padding) == 2 * num_dims and isinstance(padding[0], int):
            padding_algorithm = "EXPLICIT"
            padding = convert_to_list(padding, 2 * num_dims, 'padding')
            if _is_symmetric_padding(padding, num_dims):
                padding = padding[0::2]
        # for padding like [pad_d1, pad_d2, ...]
        elif len(padding) == num_dims and isinstance(padding[0], int):
            padding_algorithm = "EXPLICIT"
            padding = convert_to_list(padding, num_dims, 'padding')
        else:
            raise ValueError("In valid padding: {}".format(padding))
    # for integer padding
    else:
        padding_algorithm = "EXPLICIT"
        padding = convert_to_list(padding, num_dims, 'padding')
    if not all([p >= 0 for p in padding]):
        raise ValueError(
            "Invalid padding, all value should be larger than or equal to 0, but received: {}".
            format(padding))
    return padding, padding_algorithm


def _conv_nd(x,
             weight,
             bias=None,
             stride=1,
             padding=0,
             padding_algorithm=None,
             dilation=1,
             groups=1,
             data_format="NCHW",
             channel_dim=1,
             op_type="conv2d",
             use_cudnn=True,
             use_mkldnn=False,
             name=None):

    # Due to the poor performance of NHWC, we transpose the input to NCHW.
    if in_dygraph_mode():
        attrs = ('strides', stride, 'paddings', padding, 'dilations', dilation,
                 'groups', groups, 'use_cudnn', use_cudnn, 'use_mkldnn',
                 use_mkldnn, 'fuse_relu_before_depthwise_conv', False,
                 "padding_algorithm", padding_algorithm, "data_format",
                 data_format)
        pre_bias = getattr(_C_ops, op_type)(x, weight, *attrs)
        if bias is not None:
            out = nn.elementwise_add(pre_bias, bias, axis=channel_dim)
        else:
            out = pre_bias
    else:
        inputs = {'Input': [x], 'Filter': [weight]}
        attrs = {
            'strides': stride,
            'paddings': padding,
            'dilations': dilation,
            'groups': groups,
            'use_cudnn': use_cudnn,
            'use_mkldnn': use_mkldnn,
            'fuse_relu_before_depthwise_conv': False,
            "padding_algorithm": padding_algorithm,
            "data_format": data_format
        }
        check_variable_and_dtype(x, 'x', ['float16', 'float32', 'float64'],
                                 op_type)
        helper = LayerHelper(op_type, **locals())
        dtype = helper.input_dtype(input_param_name='x')
        pre_bias = helper.create_variable_for_type_inference(dtype)
        outputs = {"Output": [pre_bias]}
        helper.append_op(
            type=op_type, inputs=inputs, outputs=outputs, attrs=attrs)
        if bias is not None:
            out = helper.create_variable_for_type_inference(dtype)
            helper.append_op(
                type='elementwise_add',
                inputs={'X': [pre_bias],
                        'Y': [bias]},
                outputs={'Out': [out]},
                attrs={'axis': channel_dim,
                       'use_mkldnn': use_mkldnn})
        else:
            out = pre_bias
    return out


def conv1d(x,
           weight,
           bias=None,
           stride=1,
           padding=0,
           dilation=1,
           groups=1,
           data_format='NCL',
           name=None):
    r"""
    The convolution1D layer calculates the output based on the input, filter
    and strides, paddings, dilations, groups parameters. Input and
    Output are in NCL format, where N is batch size, C is the number of
    channels, L is the length of the feature.
    Filter is in MCK format, where M is the number of output image channels,
    C is the number of input image channels, K is the size of the kernel.
    If the groups is greater than 1, C will equal the number of input image
    channels divided by the groups. If bias attribution and activation type
    are provided, bias is added to the output of the convolution, and the
    corresponding activation function is applied to the final result.

    For each input :math:`X`, the equation is:

    .. math::

        Out = \sigma (W \ast X + b)

    Where:

    * :math:`X`: Input value, a tensor with NCL format.
    * :math:`W`: Kernel value, a tensor with MCK format.
    * :math:`\\ast`: Convolution operation.
    * :math:`b`: Bias value, a 2-D tensor with shape [M, 1].
    * :math:`\\sigma`: Activation function.
    * :math:`Out`: Output value, the shape of :math:`Out` and :math:`X` may be different.

    Example:

        - Input:

          Input shape: :math:`(N, C_{in}, L_{in})`

          Filter shape: :math:`(C_{out}, C_{in}, L_f)`

        - Output:

          Output shape: :math:`(N, C_{out}, L_{out})`

        Where

        .. math::

            L_{out} = \frac{(L_{in} + 2 * padding - (dilation * (L_f - 1) + 1))}{stride} + 1

    Args:
        x (Tensor): The input is 3-D Tensor with shape [N, C, L], the data type 
            of input is float16 or float32 or float64.
        weight (Tensor): The convolution kernel with shape [M, C/g, K], where M is
            the number of output channels, g is the number of groups, K is the kernel's size. 
        bias (Tensor, optional): The bias with shape [M,]. Default: None.
        stride (int|list|tuple, optional): The stride size. If stride is a list/tuple, it must
            contain one integers, (stride_size). Default: 1.
        padding(int|str|tuple|list, optional): The padding size. Padding could be in one of the following forms.
            1. a string in ['valid', 'same'].
            2. an int, which means the feature map is zero paded by size of `padding` on both sides.
            3. a list[int] or tuple[int] whose length is 1, which means the feature map is zero paded by size of `padding[0]` on both sides.
            4. a list[int] or tuple[int] whose length is 2. It has the form  [pad_before, pad_after].
            5. a list or tuple of pairs of ints. It has the form [[pad_before, pad_after], [pad_before, pad_after], ...]. Note that, the batch dimension and channel dimension are also included. Each pair of integers correspond to the amount of padding for a dimension of the input. Padding in batch dimension and channel dimension should be [0, 0] or (0, 0).
            The default value is 0.
        dilation (int|list|tuple, optional): The dilation size. If dilation is a list/tuple, it must
            contain one integer, (dilation_size). Default: 1.
        groups (int, optional): The groups number of the conv1d function. According to grouped
            convolution in Alex Krizhevsky's Deep CNN paper: when group=2,
            the first half of the filters is only connected to the first half
            of the input channels, while the second half of the filters is only
            connected to the second half of the input channels. Default: 1.
        data_format (str, optional): Specify the data format of the input, and the data format of the output 
            will be consistent with that of the input. An optional string from: `"NCL"`, `"NLC"`.
            The default is `"NCL"`. When it is `"NCL"`, the data is stored in the order of:
            `[batch_size, input_channels, feature_length]`.
        name(str, optional): For detailed information, please refer 
           to :ref:`api_guide_Name`. Usually name is no need to set and 
           None by default.

    Returns:
        A tensor representing the conv1d, whose data type is the 
        same with input.

    Raises:
        ValueError: If the channel dimension of the input is less than or equal to zero.
        ValueError: If `data_format` is not "NCL" or "NLC".
        ValueError: If `padding` is a string, but not "SAME" or "VALID".
        ValueError: If `padding` is a list/tuple, but the element corresponding to the input's batch size is not 0 
            or the element corresponding to the input's channel is not 0.
        ShapeError: If the input is not 3-D Tensor.
        ShapeError: If the input's dimension size and filter's dimension size not equal.
        ShapeError: If the dimension size of input minus the size of `stride` is not 1.
        ShapeError: If the number of input channels is not equal to filter's channels * groups.
        ShapeError: If the number of output channels is not be divided by groups.

    Examples:
        .. code-block:: python

          import paddle
          import paddle.nn.functional as F
          import numpy as np
          x = np.array([[[4, 8, 1, 9],
            [7, 2, 0, 9],
            [6, 9, 2, 6]]]).astype(np.float32)
          w=np.array(
          [[[9, 3, 4],
            [0, 0, 7],
            [2, 5, 6]],
           [[0, 3, 4],
            [2, 9, 7],
            [5, 6, 8]]]).astype(np.float32)
          
          x_var = paddle.to_tensor(x)
          w_var = paddle.to_tensor(w)
          y_var = F.conv1d(x_var, w_var)
          y_np = y_var.numpy()
          print(y_np)
          
          # [[[133. 238.]
          #   [160. 211.]]]
    """
    cudnn_version = get_cudnn_version()
    if cudnn_version is not None:
        use_cudnn = True
    else:
        use_cudnn = False

    if data_format not in ["NCL", "NLC"]:
        raise ValueError("Attr(data_format) should be 'NCL' or 'NLC'. "
                         "Received Attr(data_format): {}.".format(data_format))

    channel_last = (data_format == "NLC")
    channel_dim = -1 if channel_last else 1
    conv2d_data_format = "NHWC" if channel_last else "NCHW"
    if len(x.shape) != 3:
        raise ValueError(
            "Input x should be 3D tensor, but received x with the shape of {}".
            format(x.shape))
    num_channels = x.shape[channel_dim]
    num_filters = weight.shape[0]
    if num_channels < 0:
        raise ValueError("The channel dimension of the input({}) "
                         "should be defined. Received: {}.".format(
                             x.shape, num_channels))
    if groups <= 0:
        raise ValueError(
            "The groups of conv1d should be greater than 0. Received groups: {}".
            format(groups))
    if num_channels % groups != 0:
        raise ValueError(
            "the channel of input must be divisible by groups,"
            "received: the channel of input is {}, the shape of input is {}"
            ", the groups is {}".format(num_channels, x.shape, groups))
    if num_filters % groups != 0:
        raise ValueError(
            "the number of filters must be divisible by groups,"
            "received: the number of filters is {}, the shape of weight is {}"
            ", the groups is {}".format(num_filters, weight.shape, groups))

    # update attrs
    padding, padding_algorithm = _update_padding_nd(padding, channel_last, 1)
    if len(padding) == 2:
        padding = padding + [0] * 2
    elif len(padding) == 1:
        padding = padding + [0]
    else:
        raise ValueError(
            "The size of padding's dimension should be 1 or 2. But got padding={}".
            format(padding))

    stride = convert_to_list(stride, 1, 'stride') + [1]
    dilation = convert_to_list(dilation, 1, 'dilation') + [1]

    l_type = "conv2d"
    if (num_channels == groups and num_channels != 1 and
            num_filters % num_channels == 0 and not use_cudnn):
        l_type = 'depthwise_conv2d'
        use_cudnn = False

    # NPU only supports depthwise_conv2d when  "input_channel = output_channel = groups"
    if core.is_compiled_with_npu():
        if (num_channels == groups and num_channels == num_filters):
            l_type = 'depthwise_conv2d'
        else:
            l_type = 'conv2d'

    squeeze_aixs = -2 if channel_last else -1
    x = unsqueeze(x, axis=[squeeze_aixs])
    weight = unsqueeze(weight, axis=[-1])
    if in_dygraph_mode():
        attrs = ('strides', stride, 'paddings', padding, 'dilations', dilation,
                 'groups', groups, 'use_cudnn', use_cudnn, 'use_mkldnn', False,
                 'fuse_relu_before_depthwise_conv', False, "padding_algorithm",
                 padding_algorithm, "data_format", conv2d_data_format)
        out = getattr(_C_ops, l_type)(x, weight, *attrs)
        if bias is not None:
            out = nn.elementwise_add(out, bias, axis=channel_dim)
    else:
        inputs = {'Input': [x], 'Filter': [weight]}
        attrs = {
            'strides': stride,
            'paddings': padding,
            'dilations': dilation,
            'groups': groups,
            'use_cudnn': use_cudnn,
            'use_mkldnn': False,
            'fuse_relu_before_depthwise_conv': False,
            "padding_algorithm": padding_algorithm,
            "data_format": conv2d_data_format
        }
        check_variable_and_dtype(x, 'input', ['float16', 'float32', 'float64'],
                                 'conv2d')
        helper = LayerHelper(l_type, **locals())
        dtype = helper.input_dtype(input_param_name='x')
        out = helper.create_variable_for_type_inference(dtype)
        outputs = {"Output": [out]}
        helper.append_op(
            type=l_type, inputs=inputs, outputs=outputs, attrs=attrs)
        if bias is not None:
            out = nn.elementwise_add(out, bias, axis=channel_dim)
    out = squeeze(out, axis=[squeeze_aixs])
    return out


def conv2d(x,
           weight,
           bias=None,
           stride=1,
           padding=0,
           dilation=1,
           groups=1,
           data_format="NCHW",
           name=None):
    r"""

    The convolution2D layer calculates the output based on the input, filter
    and strides, paddings, dilations, groups parameters. Input and
    Output are in NCHW or NHWC format, where N is batch size, C is the number of
    channels, H is the height of the feature, and W is the width of the feature.
    Filter is in MCHW format, where M is the number of output image channels,
    C is the number of input image channels, H is the height of the filter,
    and W is the width of the filter. If the groups is greater than 1,
    C will equal the number of input image channels divided by the groups.
    Please refer to UFLDL's `convolution
    <http://ufldl.stanford.edu/tutorial/supervised/FeatureExtractionUsingConvolution/>`_
    for more details.
    If bias attribution and activation type are provided, bias is added to the
    output of the convolution, and the corresponding activation function is
    applied to the final result.

    For each input :math:`X`, the equation is:

    ..  math::

        Out = \sigma (W \ast X + b)

    Where:

    * :math:`X`: Input value, a tensor with NCHW or NHWC format.
    * :math:`W`: Filter value, a tensor with MCHW format.
    * :math:`\\ast`: Convolution operation.
    * :math:`b`: Bias value, a 2-D tensor with shape [M, 1].
    * :math:`\\sigma`: Activation function.
    * :math:`Out`: Output value, the shape of :math:`Out` and :math:`X` may be different.

    Example:

        - Input:

          Input shape: :math:`(N, C_{in}, H_{in}, W_{in})`

          Filter shape: :math:`(C_{out}, C_{in}, H_f, W_f)`

        - Output:

          Output shape: :math:`(N, C_{out}, H_{out}, W_{out})`

        Where

        ..  math::

            H_{out}&= \frac{(H_{in} + 2 * paddings[0] - (dilations[0] * (H_f - 1) + 1))}{strides[0]} + 1 \\\\
            W_{out}&= \frac{(W_{in} + 2 * paddings[1] - (dilations[1] * (W_f - 1) + 1))}{strides[1]} + 1

    Args:
        x (Tensor): The input is 4-D Tensor with shape [N, C, H, W], the data type 
            of input is float16 or float32 or float64.
        weight (Tensor): The convolution kernel with shape [M, C/g, kH, kW], where M is
            the number of output channels, g is the number of groups, kH is the filter's
            height, kW is the filter's width. 
        bias (Tensor, optional): The bias with shape [M,].
        stride (int|list|tuple): The stride size. It means the stride in convolution. 
            If stride is a list/tuple, it must contain two integers, (stride_height, stride_width). 
            Otherwise, stride_height = stride_width = stride. Default: stride = 1.
        padding (string|int|list|tuple): The padding size. It means the number of zero-paddings
            on both sides for each dimension.If `padding` is a string, either 'VALID' or
            'SAME' which is the padding algorithm. If padding size is a tuple or list,
            it could be in three forms: `[pad_height, pad_width]` or
            `[pad_height_top, pad_height_bottom, pad_width_left, pad_width_right]`, and when 
            `data_format` is `"NCHW"`, `padding` can be in the form `[[0,0], [0,0], 
            [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right]]`.
            when `data_format` is `"NHWC"`, `padding` can be in the form
            `[[0,0], [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right], [0,0]]`.
            Default: padding = 0.
        dilation (int|list|tuple): The dilation size. It means the spacing between the kernel
            points. If dilation is a list/tuple, it must contain two integers, (dilation_height, 
            dilation_width). Otherwise, dilation_height = dilation_width = dilation. 
            Default: dilation = 1.
        groups (int): The groups number of the Conv2D Layer. According to grouped
            convolution in Alex Krizhevsky's Deep CNN paper: when group=2,
            the first half of the filters is only connected to the first half
            of the input channels, while the second half of the filters is only
            connected to the second half of the input channels. Default: groups=1.
        data_format (str, optional): Specify the data format of the input, and the data format of the output 
            will be consistent with that of the input. An optional string from: `"NCHW"`, `"NHWC"`.
            The default is `"NCHW"`. When it is `"NCHW"`, the data is stored in the order of:
            `[batch_size, input_channels, input_height, input_width]`.
        name(str, optional): For detailed information, please refer 
           to :ref:`api_guide_Name`. Usually name is no need to set and 
           None by default.

    Returns:
        A Tensor representing the conv2d result, whose data type is the same with input. 

    Raises:
        ValueError: If `data_format` is not "NCHW" or "NHWC".
        ValueError: If the channel dimension of the input is less than or equal to zero.
        ValueError: If `padding` is a string, but not "SAME" or "VALID".
        ValueError: If `padding` is a list/tuple, but the element corresponding to the input's batch size is not 0 
            or the element corresponding to the input's channel is not 0.
        ShapeError: If the input is not 4-D Tensor.
        ShapeError: If the input's dimension size and filter's dimension size not equal.
        ShapeError: If the dimension size of input minus the size of `stride` is not 2.
        ShapeError: If the number of input channels is not equal to filter's channels * groups.
        ShapeError: If the number of output channels is not be divided by groups.

    Examples:
        .. code-block:: python

          import paddle
          import paddle.nn.functional as F

          x_var = paddle.randn((2, 3, 8, 8), dtype='float32')
          w_var = paddle.randn((6, 3, 3, 3), dtype='float32')

          y_var = F.conv2d(x_var, w_var)
          y_np = y_var.numpy()

          print(y_np.shape)
          # (2, 6, 6, 6)
    """
    # entry checks
    if data_format not in ["NCHW", "NHWC"]:
        raise ValueError("Attr(data_format) should be 'NCHW' or 'NHWC'. "
                         "Received Attr(data_format): {}.".format(data_format))

    channel_last = (data_format == "NHWC")
    channel_dim = -1 if channel_last else 1
    if len(x.shape) != 4:
        raise ValueError(
            "Input x should be 4D tensor, but received x with the shape of {}".
            format(x.shape))
    num_channels = x.shape[channel_dim]
    num_filters = weight.shape[0]
    if num_channels < 0:
        raise ValueError("The channel dimension of the input({}) "
                         "should be defined. Received: {}.".format(
                             x.shape, num_channels))
    if groups <= 0:
        raise ValueError(
            "The groups of conv2d should be greater than 0. Received groups: {}".
            format(groups))
    if num_channels % groups != 0:
        raise ValueError(
            "the channel of input must be divisible by groups,"
            "received: the channel of input is {}, the shape of input is {}"
            ", the groups is {}".format(num_channels, x.shape, groups))
    if num_filters % groups != 0:
        raise ValueError(
            "the number of filters must be divisible by groups,"
            "received: the number of filters is {}, the shape of weight is {}"
            ", the groups is {}".format(num_filters, weight.shape, groups))

    cudnn_version = get_cudnn_version()

    use_cudnn = True if (core.is_compiled_with_cuda() and
                         cudnn_version is not None) else False

    use_mkldnn = _global_flags()["FLAGS_use_mkldnn"]

    # update attrs
    padding, padding_algorithm = _update_padding_nd(padding, channel_last, 2)
    stride = convert_to_list(stride, 2, 'stride')
    dilation = convert_to_list(dilation, 2, 'dilation')

    l_type = "conv2d"
    if (num_channels == groups and num_channels != 1 and
            num_filters % num_channels == 0):
        l_type = 'depthwise_conv2d'
        if core.is_compiled_with_rocm():
            use_cudnn = True
        else:
            use_cudnn = False

    # NPU only supports depthwise_conv2d when  "input_channel = output_channel = groups"
    if core.is_compiled_with_npu():
        if (num_channels == groups and num_channels == num_filters):
            l_type = 'depthwise_conv2d'
        else:
            l_type = 'conv2d'

    if (core.is_compiled_with_cuda() and get_flags("FLAGS_conv2d_disable_cudnn")
        ["FLAGS_conv2d_disable_cudnn"]):
        use_cudnn = False

    return _conv_nd(x, weight, bias, stride, padding, padding_algorithm,
                    dilation, groups, data_format, channel_dim, l_type,
                    use_cudnn, use_mkldnn, name)


def conv1d_transpose(x,
                     weight,
                     bias=None,
                     stride=1,
                     padding=0,
                     output_padding=0,
                     groups=1,
                     dilation=1,
                     output_size=None,
                     data_format="NCL",
                     name=None):
    r"""
    The 1-D convolution transpose layer calculates the output based on the input,
    filter, and dilation, stride, padding. Input(Input) and output(Output)
    are in 'NCL' format or 'NLC' where N is batch size, C is the number of channels,
    L is the length of the feature. The details of convolution transpose
    layer, please refer to the following explanation and references
    `therein <https://arxiv.org/pdf/1603.07285.pdf>`_.
    If bias attribution and activation type are provided, bias is added to
    the output of the convolution, and the corresponding activation function
    is applied to the final result.

    For each input :math:`X`, the equation is:

    .. math::

        Out = \sigma (W \ast X + b)

    Where:

    * :math:`X`: Input value, a 3-D Tensor with 'NCL' format or 'NLC' format.
    * :math:`W`: Filter value, a 3-D Tensor with 'MCK' format.
    * :math:`\\ast`: Convolution operation.
    * :math:`b`: Bias value, a 2-D Tensor with shape [M, 1].
    * :math:`\\sigma`: Activation function.
    * :math:`Out`: Output value, a 3-D Tensor with data format 'NCL' or 'NLC', the shape of :math:`Out` and :math:`X` may be different.

    Example:

        - Input:

          Input shape: :math:`(N, C_{in}, L_{in})`

          Filter shape: :math:`(C_{in}, C_{out}, L_f)`

        - Output:

          Output shape: :math:`(N, C_{out}, L_{out})`

        Where

        .. math::

           L^\prime_{out} &= (L_{in} - 1) * stride - pad_top - pad_bottom + dilation * (L_f - 1) + 1 + output_padding \\\\
           L_{out} &\in [ L^\prime_{out}, L^\prime_{out} + stride ]

    Note:
          The conv1d_transpose can be seen as the backward of the conv1d. For conv1d,
          when stride > 1, conv1d maps multiple input shape to the same output shape,
          so for conv1d_transpose, when stride > 1, input shape maps multiple output shape.
          If output_size is None, :math:`L_{out} = L^\prime_{out}`;
          else, the :math:`L_{out}` of the output size must between :math:`L^\prime_{out}`
          and :math:`L^\prime_{out} + stride`.

    Args:
        x(Tensor): 3-D tensor with [N, C, L] or [N, L, C] format,
                         its data type is float32 or float64.
        weight(Tensor): The convolution kernel, a Tensor with shape [C, M/g, K],
            where M is the number of output channels(filters), g is the number of groups,
            K is the size of the kernel.
        bias(Tensor, optional): The bias, a Tensor with shape [M, ].
        stride(int|tuple|list, optional): The stride size. It means the stride in transposed convolution.
            If stride is a list/tuple, it must contain one integer, `(stride_size)`.
            Default: stride = 1.
        padding(int|list|str|tuple, optional): The padding size. The padding argument effectively adds
             `dilation * (kernel - 1)` amount of zero-padding on both sides of input. If `padding` is a
             string, either 'VALID' or 'SAME' supported, which is the padding algorithm.
             If `padding` is a tuple or list, it could be in two forms:
             `[pad]` or `[pad_left, pad_right]`. Default: padding = 0.
        output_padding(int|list|tuple, optional): The count of zeros to be added to tail of each dimension.
             If it is a list/tuple, it must contain one integer. Default: 0.
        groups(int, optional): The groups number of the conv1d transpose function. Inspired by
            grouped convolution in Alex Krizhevsky's Deep CNN paper, in which
            when group=2, the first half of the filters is only connected to the
            first half of the input channels, while the second half of the
            filters is only connected to the second half of the input channels.
            Default: groups = 1.
        dilation(int|tuple|list, optional): The dilation size. It means the spacing between the kernel points.
            If dilation is a list/tuple, it must contain one integer, `(dilation_size)`.
            Default: dilation = 1.
        output_size(int|tuple|list, optional): The output image size. If output size is a
            tuple/list, it must contain one integer, `(feature_length)`. None if use
            filter_size(shape of weight), padding, and stride to calculate output_size.
        data_format (str, optional): Specify the data format of the input, and the data format of the output 
            will be consistent with that of the input. An optional string from: `"NCL"`, `"NLC"`.
            The default is `"NCL"`. When it is `"NCL"`, the data is stored in the order of:
            `[batch_size, input_channels, input_length]`.
        name(str, optional): For detailed information, please refer 
           to :ref:`api_guide_Name`. Usually name is no need to set and 
           None by default.

    Returns:
        A  tensor representing the result of 1-D transpose convolution, whose
        data type is the same with input. And its shape is (num_batches, channels, length)
        when data_format is `"NCL"` and (num_batches, length, channels) when data_format is
        `"NLC"`.

    Raises:
        ValueError: If `data_format` is a string, but not "NCL" or "NLC".
        ValueError: If `padding` is a string, but not "SAME" or "VALID".
        ValueError: If `padding` is a list/tuple, but the element corresponding to the input's batch size is not 0 
            or the element corresponding to the input's channel is not 0.
        ValueError: If `output_size` and filter_size are None at the same time.
        ValueError: If `output_padding` is greater than `stride`.
        ShapeError: If the input is not 3-D Tensor.
        ShapeError: If the input's dimension size and filter's dimension size not equal.
        ShapeError: If the dimension size of input minus the size of `stride` is not 1.
        ShapeError: If the number of input channels is not equal to filter's channels.
        ShapeError: If the size of `output_size` is not equal to that of `stride`.

    Examples:
        .. code-block:: python



          import paddle
          import paddle.nn.functional as F
          import numpy as np
          
          # shape: (1, 2, 4)
          x=np.array([[[4, 0, 9, 7],
                       [8, 0, 9, 2,]]]).astype(np.float32)
          # shape: (2, 1, 2)
          w=np.array([[[7, 0]],
                      [[4, 2]]]).astype(np.float32)
          x_var = paddle.to_tensor(x)
          w_var = paddle.to_tensor(w)
          y_var = F.conv1d_transpose(x_var, w_var)
          print(y_var)
          
          # [[[60. 16. 99. 75.  4.]]]
    """
    cudnn_version = get_cudnn_version()
    if cudnn_version is not None:
        use_cudnn = True
    else:
        use_cudnn = False

    if data_format not in ['NCL', 'NLC']:
        raise ValueError(
            "Attr(data_format) of conv2d_transpose got wrong value: "
            "received {}, but only 'NCL' or 'NLC' are supported.".format(
                data_format))
    channel_last = (data_format == "NLC")
    channel_dim = -1 if channel_last else 1
    if len(x.shape) != 3:
        raise ValueError(
            "Input x should be 3D tensor, but received x with the shape of {}".
            format(x.shape))

    num_channels = x.shape[channel_dim]
    if num_channels < 0:
        raise ValueError("The channel dimension of the input({}) "
                         "should be defined. Received: {}.".format(
                             x.shape, num_channels))
    if groups <= 0:
        raise ValueError(
            "The groups of conv1d_transpose should be greater than 0. Received groups: {}".
            format(groups))
    if num_channels % groups != 0:
        raise ValueError(
            "the channel of input must be divisible by groups,"
            "received: the channel of input is {}, the shape of input is {}"
            ", the groups is {}".format(num_channels, x.shape, groups))

    # update attrs
    padding, padding_algorithm = _update_padding_nd(padding, channel_last, 1)

    if len(padding) == 2:
        padding = padding + [0] * 2
    elif len(padding) == 1:
        padding = padding + [0]
    else:
        raise ValueError(
            "The size of padding's dimension should 1 or 2. But got padding={}".
            format(padding))

    stride = convert_to_list(stride, 1, 'stride') + [1]
    dilation = convert_to_list(dilation, 1, 'dilation') + [1]

    if output_size is None:
        output_size = []
    else:
        if output_padding != 0:
            raise ValueError('output_padding option is mutually exclusive with '
                             'output_size')
        if isinstance(output_size, (list, tuple, int)):
            output_size = convert_to_list(output_size, 1, 'output_size') + [1]
        else:
            raise ValueError(
                "output_size should be int, or list, tuple of ints")

    if output_padding == 0:
        output_padding = []
    else:
        output_padding = convert_to_list(output_padding, 1,
                                         'output_padding') + [0]

    if len(output_padding) > 0 and output_padding[0] > stride[0]:
        raise ValueError(
            "The size of output_padding should not be greater than stride."
            "But got output_padding={} and stride={}".format(output_padding[0],
                                                             stride[0]))

    op_type = 'conv2d_transpose'
    num_filters = weight.shape[1]
    if (num_channels == groups and num_channels != 1 and num_filters == 1 and
            not use_cudnn):
        op_type = 'depthwise_conv2d_transpose'
        use_cudnn = False

    squeeze_axis = -2 if channel_last else -1
    conv2d_data_format = "NHWC" if channel_last else "NCHW"

    x = unsqueeze(x, axis=[squeeze_axis])
    weight = unsqueeze(weight, axis=[-1])

    if in_dygraph_mode():
        attrs = ('output_padding', output_padding, 'output_size', output_size,
                 'strides', stride, 'paddings', padding, 'padding_algorithm',
                 padding_algorithm, 'dilations', dilation, 'groups', groups,
                 'use_cudnn', use_cudnn, 'data_format', conv2d_data_format)
        out = getattr(_C_ops, op_type)(x, weight, *attrs)
        if bias is not None:
            out = nn.elementwise_add(out, bias, axis=channel_dim)
    else:
        inputs = {'Input': [x], 'Filter': [weight]}
        attrs = {
            'output_padding': output_padding,
            'output_size': output_size,
            'strides': stride,
            'paddings': padding,
            'padding_algorithm': padding_algorithm,
            'dilations': dilation,
            'groups': groups,
            'use_cudnn': use_cudnn,
            'data_format': conv2d_data_format
        }
        check_variable_and_dtype(x, 'input', ['float16', 'float32', 'float64'],
                                 'conv2d_transpose')
        helper = LayerHelper(op_type, **locals())
        dtype = helper.input_dtype(input_param_name='x')
        out = helper.create_variable_for_type_inference(dtype)
        outputs = {"Output": [out]}
        helper.append_op(
            type=op_type, inputs=inputs, outputs=outputs, attrs=attrs)
        if bias is not None:
            out = nn.elementwise_add(out, bias, axis=channel_dim)

    out = squeeze(out, axis=[squeeze_axis])
    return out


def conv2d_transpose(x,
                     weight,
                     bias=None,
                     stride=1,
                     padding=0,
                     output_padding=0,
                     dilation=1,
                     groups=1,
                     output_size=None,
                     data_format='NCHW',
                     name=None):
    r"""

    The convolution2D transpose layer calculates the output based on the input,
    filter, and dilations, strides, paddings. Input(Input) and output(Output)
    are in NCHW or NHWC format. Where N is batch size, C is the number of channels,
    H is the height of the feature, and W is the width of the feature.
    Parameters(dilations, strides, paddings) are two elements. These two elements
    represent height and width, respectively. The details of convolution transpose
    layer, please refer to the following explanation and references
    `therein <https://arxiv.org/pdf/1603.07285.pdf>`_.
    If bias attribution and activation type are provided, bias is added to
    the output of the convolution, and the corresponding activation function
    is applied to the final result.
    See more detail in :ref:`api_nn_conv_ConvTranspose2d` .

    For each input :math:`X`, the equation is:

    ..  math::

        Out = \sigma (W \ast X + b)

    Where:

    * :math:`X`: Input value, a 4-D Tensor with NCHW or NHWC format.
    * :math:`W`: Filter value, a 4-D Tensor with MCHW format.
    * :math:`\\ast`: Convolution operation.
    * :math:`b`: Bias value, a 2-D Tensor with shape [M, 1].
    * :math:`\\sigma`: Activation function.
    * :math:`Out`: Output value, a 4-D Tensor with data format 'NCHW' or 'NHWC', the shape of :math:`Out` and :math:`X` may be different.

    Example:

        - Input:

          Input shape: :math:`(N, C_{in}, H_{in}, W_{in})`

          Filter shape: :math:`(C_{in}, C_{out}, H_f, W_f)`

        - Output:

          Output shape: :math:`(N, C_{out}, H_{out}, W_{out})`

        Where

        ..  math::

           H^\prime_{out} &= (H_{in} - 1) * strides[0] - pad_height_top - pad_height_bottom + dilations[0] * (H_f - 1) + 1 \\\\
           W^\prime_{out} &= (W_{in} - 1) * strides[1] - pad_width_left - pad_width_right + dilations[1] * (W_f - 1) + 1 \\\\
           H_{out} &\in [ H^\prime_{out}, H^\prime_{out} + strides[0] ] \\\\
           W_{out} &\in [ W^\prime_{out}, W^\prime_{out} + strides[1] ]

    Note:
          The conv2d_transpose can be seen as the backward of the conv2d. For conv2d, 
          when stride > 1, conv2d maps multiple input shape to the same output shape, 
          so for conv2d_transpose, when stride > 1, input shape maps multiple output shape.
          If output_size is None, :math:`H_{out} = H^\prime_{out}, W_{out} = W^\prime_{out}`; 
          else, the :math:`H_{out}` of the output size must between :math:`H^\prime_{out}` 
          and :math:`H^\prime_{out} + strides[0]`, and the :math:`W_{out}` of the output size must 
          between :math:`W^\prime_{out}` and :math:`W^\prime_{out} + strides[1]`.

    Args:
        x(Tensor): 4-D Tensor with [N, C, H, W] or [N, H, W, C] format,
            whose data type is float32 or float64.
        weight(Tensor): The convolution kernel, a Tensor with shape [C, M/g, kH, kW],
            where M is the number of output channels(filters), g is the number of groups,
            kH is the height of the kernel, and kW is the width of the kernel.
        bias(Tensor, optional): The bias, a Tensor with shape [M, ].
        stride(int|list|tuple, optional): The stride size. It means the stride in transposed convolution. 
            If stride is a list/tuple, it must contain two integers, (stride_height, stride_width). 
            Otherwise, stride_height = stride_width = stride. Default: stride = 1.
        padding(str|int|list|tuple, optional): The padding size. It means the number of zero-paddings 
            on both sides for each dimension. If `padding` is a string, either 'VALID' or 
            'SAME' which is the padding algorithm. If padding size is a tuple or list,
            it could be in three forms: `[pad_height, pad_width]` or 
            `[pad_height_top, pad_height_bottom, pad_width_left, pad_width_right]`,
            and when `data_format` is `"NCHW"`, `padding` can be in the form 
            `[[0,0], [0,0], [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right]]`.
            when `data_format` is `"NHWC"`, `padding` can be in the form 
            `[[0,0], [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right], [0,0]]`.
            Default: padding = 0.
        output_padding(int|list|tuple, optional): Additional size added to one side
            of each dimension in the output shape. Default: 0.
        groups(int, optional): The groups number of the Conv2D transpose layer. Inspired by
            grouped convolution in Alex Krizhevsky's Deep CNN paper, in which
            when group=2, the first half of the filters is only connected to the
            first half of the input channels, while the second half of the
            filters is only connected to the second half of the input channels.
            Default: groups = 1.
        dilation(int|list|tuple, optional): The dilation size. It means the spacing between the kernel points. 
            If dilation is a list/tuple, it must contain two integers, (dilation_height, dilation_width). 
            Otherwise, dilation_height = dilation_width = dilation. Default: dilation = 1.
        output_size(int|tuple|list, optional): The output image size. If output size is a
            tuple/list, it must contain two integers, (image_height, image_width). None if use
            filter_size(shape of weight), padding, and stride to calculate output_size.
        data_format (str, optional): Specify the data format of the input, and the data format of the output 
            will be consistent with that of the input. An optional string from: `"NCHW"`, `"NHWC"`.
            The default is `"NCHW"`. When it is `"NCHW"`, the data is stored in the order of:
            `[batch_size, input_channels, input_height, input_width]`.
        name(str, optional): For detailed information, please refer 
           to :ref:`api_guide_Name`. Usually name is no need to set and 
           None by default.

    Returns:
        A Tensor representing the conv2d_transpose, whose
        data type is the same with input and shape is (num_batches, channels, out_h, 
        out_w) or (num_batches, out_h, out_w, channels). The tensor variable storing 
        transposed convolution result.

    Raises:
        ValueError: If `data_format` is not "NCHW" or "NHWC".
        ValueError: If `padding` is a string, but not "SAME" or "VALID".
        ValueError: If `padding` is a list/tuple, but the element corresponding to the input's batch size is not 0 
            or the element corresponding to the input's channel is not 0.
        ValueError: If `output_size` and kernel_size are None at the same time.
        ShapeError: If the input is not 4-D Tensor.
        ShapeError: If the input's dimension size and filter's dimension size not equal.
        ShapeError: If the dimension size of input minus the size of `stride` is not 2.
        ShapeError: If the number of input channels is not equal to filter's channels.
        ShapeError: If the size of `output_size` is not equal to that of `stride`.

    Examples:
        .. code-block:: python

          import paddle
          import paddle.nn.functional as F

          x_var = paddle.randn((2, 3, 8, 8), dtype='float32')
          w_var = paddle.randn((3, 6, 3, 3), dtype='float32')

          y_var = F.conv2d_transpose(x_var, w_var)
          y_np = y_var.numpy()

          print(y_np.shape)
          # (2, 6, 10, 10)
    """

    if data_format not in ['NCHW', 'NHWC']:
        raise ValueError(
            "Attr(data_format) of conv2d_transpose got wrong value: "
            "received {}, but only 'NCHW' or 'NHWC' are supported.".format(
                data_format))
    channel_last = (data_format == "NHWC")
    channel_dim = -1 if channel_last else 1
    if len(x.shape) != 4:
        raise ValueError(
            "Input x should be 4D tensor, but received x with the shape of {}".
            format(x.shape))
    num_channels = x.shape[channel_dim]
    if num_channels < 0:
        raise ValueError("The channel dimension of the input({}) "
                         "should be defined. Received: {}.".format(
                             x.shape, num_channels))
    if groups <= 0:
        raise ValueError(
            "The groups of conv2d_transpose should be greater than 0. Received groups: {}".
            format(groups))
    if num_channels % groups != 0:
        raise ValueError(
            "the channel of input must be divisible by groups,"
            "received: the channel of input is {}, the shape of input is {}"
            ", the groups is {}".format(num_channels, x.shape, groups))

    cudnn_version = get_cudnn_version()

    use_cudnn = True if (core.is_compiled_with_cuda() and
                         cudnn_version is not None) else False

    # update attrs
    padding, padding_algorithm = _update_padding_nd(padding, channel_last, 2)
    stride = convert_to_list(stride, 2, 'stride')
    dilation = convert_to_list(dilation, 2, 'dilation')

    if output_size is None:
        output_size = []
    else:
        if output_padding != 0:
            raise ValueError('output_padding option is mutually exclusive with '
                             'output_size')
        if isinstance(output_size, (list, tuple, int)):
            output_size = convert_to_list(output_size, 2, 'output_size')
        else:
            raise ValueError(
                "output_size should be int, or list, tuple of ints")

    if output_padding == 0:
        output_padding = []
    else:
        output_padding = convert_to_list(output_padding, 2, 'output_padding')

    op_type = 'conv2d_transpose'
    num_filters = weight.shape[1]
    if (num_channels == groups and num_channels != 1 and num_filters == 1):
        op_type = 'depthwise_conv2d_transpose'
        use_cudnn = False

    if in_dygraph_mode():
        attrs = ('output_padding', output_padding, 'output_size', output_size,
                 'strides', stride, 'paddings', padding, 'padding_algorithm',
                 padding_algorithm, 'dilations', dilation, 'groups', groups,
                 'use_cudnn', use_cudnn, 'data_format', data_format)
        pre_bias = getattr(_C_ops, op_type)(x, weight, *attrs)
        if bias is not None:
            out = nn.elementwise_add(pre_bias, bias, axis=channel_dim)
        else:
            out = pre_bias
    else:
        inputs = {'Input': [x], 'Filter': [weight]}
        attrs = {
            'output_padding': output_padding,
            'output_size': output_size,
            'strides': stride,
            'paddings': padding,
            'padding_algorithm': padding_algorithm,
            'dilations': dilation,
            'groups': groups,
            'use_cudnn': use_cudnn,
            'data_format': data_format
        }
        check_variable_and_dtype(x, 'x', ['float16', 'float32', 'float64'],
                                 'conv2d_transpose')
        helper = LayerHelper(op_type, **locals())
        pre_bias = helper.create_variable_for_type_inference(x.dtype)
        outputs = {"Output": [pre_bias]}
        helper.append_op(
            type=op_type, inputs=inputs, outputs=outputs, attrs=attrs)

        if bias is not None:
            out = nn.elementwise_add(pre_bias, bias, axis=channel_dim)
        else:
            out = pre_bias

    return out


def conv3d(x,
           weight,
           bias=None,
           stride=1,
           padding=0,
           dilation=1,
           groups=1,
           data_format="NCDHW",
           name=None):
    r"""

    The convolution3D layer calculates the output based on the input, filter
    and strides, paddings, dilations, groups parameters. Input(Input) and
    Output(Output) are in NCDHW or NDHWC format. Where N is batch size C is the number of
    channels, D is the depth of the feature, H is the height of the feature,
    and W is the width of the feature. Convlution3D is similar with Convlution2D
    but adds one dimension(depth). If bias attribution and activation type are
    provided, bias is added to the output of the convolution, and the
    corresponding activation function is applied to the final result.

    For each input :math:`X`, the equation is:

    ..  math::

        Out = \sigma (W \ast X + b)

    In the above equation:

    * :math:`X`: Input value, a tensor with NCDHW or NDHWC format.
    * :math:`W`: Filter value, a tensor with MCDHW format.
    * :math:`\\ast`: Convolution operation.
    * :math:`b`: Bias value, a 2-D tensor with shape [M, 1].
    * :math:`\\sigma`: Activation function.
    * :math:`Out`: Output value, the shape of :math:`Out` and :math:`X` may be different.

    Example:

        - Input:

          Input shape: :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})`

          Filter shape: :math:`(C_{out}, C_{in}, D_f, H_f, W_f)`

        - Output:
          Output shape: :math:`(N, C_{out}, D_{out}, H_{out}, W_{out})`

        Where

        ..  math::

            D_{out}&= \\frac{(D_{in} + 2 * paddings[0] - (dilations[0] * (D_f - 1) + 1))}{strides[0]} + 1 \\\\
            H_{out}&= \\frac{(H_{in} + 2 * paddings[1] - (dilations[1] * (H_f - 1) + 1))}{strides[1]} + 1 \\\\
            W_{out}&= \\frac{(W_{in} + 2 * paddings[2] - (dilations[2] * (W_f - 1) + 1))}{strides[2]} + 1

    Args:
        x (Tensor): The input is 5-D Tensor with shape [N, C, D, H, W], the data 
            type of input is float16 or float32 or float64.
        weight (Tensor): The convolution kernel, a Tensor with shape [M, C/g, kD, kH, kW],
            where M is the number of filters(output channels), g is the number of groups,
            kD, kH, kW are the filter's depth, height and width respectively.
        bias (Tensor, optional): The bias, a Tensor of shape [M, ].
        stride (int|list|tuple): The stride size. It means the stride in convolution. If stride is a 
            list/tuple, it must contain three integers, (stride_depth, stride_height, stride_width). 
            Otherwise, stride_depth = stride_height = stride_width = stride. Default: stride = 1.
        padding (string|int|list|tuple): The padding size. It means the number of zero-paddings 
            on both sides for each dimension. If `padding` is a string, either 'VALID' or
            'SAME' which is the padding algorithm. If padding size is a tuple or list,
            it could be in three forms: `[pad_depth, pad_height, pad_width]` or
            `[pad_depth_front, pad_depth_back, pad_height_top, pad_height_bottom, pad_width_left, pad_width_right]`,
            and when `data_format` is `"NCDHW"`, `padding` can be in the form
            `[[0,0], [0,0], [pad_depth_front, pad_depth_back], [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right]]`.
            when `data_format` is `"NDHWC"`, `padding` can be in the form
            `[[0,0], [pad_depth_front, pad_depth_back], [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right], [0,0]]`.
            Default: padding = 0.
        dilation (int|list|tuple): The dilation size. It means the spacing between the kernel points. 
            If dilation is a list/tuple, it must contain three integers, (dilation_depth, dilation_height,
            dilation_width). Otherwise, dilation_depth = dilation_height = dilation_width = dilation. 
            Default: dilation = 1.
        groups (int): The groups number of the Conv3D Layer. According to grouped
            convolution in Alex Krizhevsky's Deep CNN paper: when group=2,
            the first half of the filters is only connected to the first half
            of the input channels, while the second half of the filters is only
            connected to the second half of the input channels. Default: groups=1
        data_format (str, optional): Specify the data format of the input, and the data format of the output 
            will be consistent with that of the input. An optional string from: `"NCHW"`, `"NHWC"`.
            The default is `"NCHW"`. When it is `"NCHW"`, the data is stored in the order of:
            `[batch_size, input_channels, input_height, input_width]`.
        name(str|None): For detailed information, please refer 
           to :ref:`api_guide_Name`. Usually name is no need to set and 
           None by default.

    Returns:
        A Tensor representing the conv3d, whose data type is 
        the same with input. If act is None, the tensor storing the 
        convolution result, and if act is not None, the tensor storing 
        convolution and non-linearity activation result.

    Examples:
        .. code-block:: python

            import paddle
            import paddle.nn.functional as F

            x_var = paddle.randn((2, 3, 8, 8, 8), dtype='float32')
            w_var = paddle.randn((6, 3, 3, 3, 3), dtype='float32')

            y_var = F.conv3d(x_var, w_var)
            y_np = y_var.numpy()

            print(y_np.shape)
            # (2, 6, 6, 6, 6)
    """
    # entry check
    if data_format not in ["NCDHW", "NDHWC"]:
        raise ValueError(
            "Attr(data_format) should be 'NCDHW' or 'NDHWC'. Received "
            "Attr(data_format): {}.".format(data_format))

    channel_last = (data_format == "NDHWC")
    channel_dim = -1 if channel_last else 1
    if len(x.shape) != 5:
        raise ValueError(
            "Input x should be 5D tensor, but received x with the shape of {}".
            format(x.shape))
    num_channels = x.shape[channel_dim]
    num_filters = weight.shape[0]
    if num_channels < 0:
        raise ValueError(
            "The channel dimension of the input({}) should be defined. "
            "Received: {}.".format(x.shape, num_channels))
    if groups <= 0:
        raise ValueError(
            "The groups of conv3d should be greater than 0. Received groups: {}".
            format(groups))
    if num_channels % groups != 0:
        raise ValueError(
            "The number of input channels must be divisible by Attr(groups). "
            "Received: number of channels({}), groups({}).".format(num_channels,
                                                                   groups))
    if num_filters % groups != 0:
        raise ValueError(
            "The number of filters must be divisible by Attr(groups). "
            "Received: number of filters({}), groups({}).".format(num_filters,
                                                                  groups))

    cudnn_version = get_cudnn_version()
    use_cudnn = True if (core.is_compiled_with_cuda() and
                         cudnn_version is not None) else False

    padding, padding_algorithm = _update_padding_nd(padding, channel_last, 3)
    stride = convert_to_list(stride, 3, 'stride')
    dilation = convert_to_list(dilation, 3, 'dilation')
    op_type = "conv3d"

    return _conv_nd(x, weight, bias, stride, padding, padding_algorithm,
                    dilation, groups, data_format, channel_dim, op_type,
                    use_cudnn, False, name)


def conv3d_transpose(x,
                     weight,
                     bias=None,
                     stride=1,
                     padding=0,
                     output_padding=0,
                     groups=1,
                     dilation=1,
                     output_size=None,
                     data_format='NCDHW',
                     name=None):
    r"""
    The convolution3d transpose layer calculates the output based on the input,
    filter, and dilations, strides, paddings. Input(Input) and output(Output)
    are in NCDHW or NDHWC format. Where N is batch size, C is the number of channels,
    D is the depth of the feature, H is the height of the feature, and W
    is the width of the feature. Parameters(dilations, strides, paddings) are
    two elements. These two elements represent height and width, respectively.
    The details of convolution transpose layer, please refer to the following
    explanation and references `therein <https://arxiv.org/pdf/1603.07285.pdf>`_.
    If bias attribution and activation type are provided, bias is added to
    the output of the convolution, and the corresponding activation function
    is applied to the final result.
    See more detail in :ref:`api_nn_conv_ConvTranspose3d` .

    For each input :math:`X`, the equation is:

    ..  math::

        Out = \sigma (W \ast X + b)

    In the above equation:

    * :math:`X`: Input value, a Tensor with NCDHW or NDHWC format.
    * :math:`W`: Filter value, a Tensor with MCDHW format.
    * :math:`\\ast`: Convolution operation.
    * :math:`b`: Bias value, a 2-D Tensor with shape [M, 1].
    * :math:`\\sigma`: Activation function.
    * :math:`Out`: Output value, the shape of :math:`Out` and :math:`X` may be different.

    Example:

        - Input:

          Input shape: :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})`

          Filter shape: :math:`(C_{in}, C_{out}, D_f, H_f, W_f)`

        - Output:

          Output shape: :math:`(N, C_{out}, D_{out}, H_{out}, W_{out})`

        Where

        ..  math::

           D^\prime_{out} &= (D_{in} - 1) * strides[0] - 2 * paddings[0] + dilations[0] * (D_f - 1) + 1 \\\\
           H^\prime_{out} &= (H_{in} - 1) * strides[1] - 2 * paddings[1] + dilations[1] * (H_f - 1) + 1 \\\\
           W^\prime_{out} &= (W_{in} - 1) * strides[2] - 2 * paddings[2] + dilations[2] * (W_f - 1) + 1 \\\\
           D_{out} &\in [ D^\prime_{out}, D^\prime_{out} + strides[0] ] \\\\
           H_{out} &\in [ H^\prime_{out}, H^\prime_{out} + strides[1] ] \\\\
           W_{out} &\in [ W^\prime_{out}, W^\prime_{out} + strides[2] ]

    Note:
          The conv3d_transpose can be seen as the backward of the conv3d. For conv3d, 
          when stride > 1, conv3d maps multiple input shape to the same output shape, 
          so for conv3d_transpose, when stride > 1, input shape maps multiple output shape.
          If output_size is None, :math:`H_{out} = H^\prime_{out}, :math:`H_{out} = \
          H^\prime_{out}, W_{out} = W^\prime_{out}`; else, the :math:`D_{out}` of the output 
          size must between :math:`D^\prime_{out}` and :math:`D^\prime_{out} + strides[0]`, 
          the :math:`H_{out}` of the output size must between :math:`H^\prime_{out}` 
          and :math:`H^\prime_{out} + strides[1]`, and the :math:`W_{out}` of the output size must 
          between :math:`W^\prime_{out}` and :math:`W^\prime_{out} + strides[2]`.

    Args:
        x(Tensor): The input is 5-D Tensor with shape [N, C, D, H, W] or [N, D, H, W, C], the data type 
            of input is float32 or float64.
        weight (Tensor): The convolution kernel, a Tensor with shape [C, M/g, kD, kH, kW],
            where M is the number of filters(output channels), g is the number of groups,
            kD, kH, kW are the filter's depth, height and width respectively.
        bias (Tensor, optional): The bias, a Tensor of shape [M, ].
        stride(int|list|tuple, optional): The stride size. It means the stride in transposed convolution. 
            If stride is a list/tuple, it must contain three integers, (stride_depth, stride_height, 
            stride_width). Otherwise, stride_depth = stride_height = stride_width = stride. 
            Default: stride = 1.
        padding (string|int|list|tuple, optional): The padding size. It means the number of zero-paddings 
            on both sides for each dimension. If `padding` is a string, either 'VALID' or
            'SAME' which is the padding algorithm. If padding size is a tuple or list,
            it could be in three forms: `[pad_depth, pad_height, pad_width]` or
            `[pad_depth_front, pad_depth_back, pad_height_top, pad_height_bottom, pad_width_left, pad_width_right]`,
            and when `data_format` is `"NCDHW"`, `padding` can be in the form
            `[[0,0], [0,0], [pad_depth_front, pad_depth_back], [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right]]`.
            when `data_format` is `"NDHWC"`, `padding` can be in the form
            `[[0,0], [pad_depth_front, pad_depth_back], [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right], [0,0]]`.
            Default: padding = 0.
        output_padding(int|list|tuple, optional): Additional size added to one side
            of each dimension in the output shape. Default: 0.
        groups(int, optional): The groups number of the Conv3D transpose layer. Inspired by
            grouped convolution in Alex Krizhevsky's Deep CNN paper, in which
            when group=2, the first half of the filters is only connected to the
            first half of the input channels, while the second half of the
            filters is only connected to the second half of the input channels.
            Default: groups=1
        dilation(int|list|tuple, optional): The dilation size. It means the spacing between the kernel points. 
            If dilation is a list/tuple, it must contain three integers, (dilation_depth, dilation_height, 
            dilation_width). Otherwise, dilation_depth = dilation_height = dilation_width = dilation. 
            Default: dilation = 1.
        output_size(int|list|tuple, optional): The output image size. If output size is a
            list/tuple, it must contain three integers, (image_depth, image_height, image_width).
            None if use filter_size(shape of weight), padding, and stride to calculate output_size.
        data_format (str, optional): Specify the data format of the input, and the data format of the output 
            will be consistent with that of the input. An optional string from: `"NCHW"`, `"NHWC"`.
            The default is `"NCHW"`. When it is `"NCHW"`, the data is stored in the order of:
            `[batch_size, input_channels, input_height, input_width]`.
        name(str, optional): For detailed information, please refer 
           to :ref:`api_guide_Name`. Usually name is no need to set and 
           None by default.

    Returns:
        A Tensor representing the conv3d_transpose, whose data
        type is the same with input and shape is (num_batches, channels, out_d, out_h, 
        out_w) or (num_batches, out_d, out_h, out_w, channels). If act is None, the tensor 
        variable storing the transposed convolution result, and if act is not None, the tensor 
        variable storing transposed convolution and non-linearity activation result.

    Raises:
        ValueError: If `data_format` is not "NCDHW" or "NDHWC".
        ValueError: If `padding` is a string, but not "SAME" or "VALID".
        ValueError: If `padding` is a list/tuple, but the element corresponding to the input's batch size is not 0 
            or the element corresponding to the input's channel is not 0.
        ValueError: If `output_size` and kernel_size are None at the same time.
        ShapeError: If the input is not 5-D Tensor.
        ShapeError: If the input's dimension size and filter's dimension size not equal.
        ShapeError: If the dimension size of input minus the size of `stride` is not 2.
        ShapeError: If the number of input channels is not equal to filter's channels.
        ShapeError: If the size of `output_size` is not equal to that of `stride`.

    Examples:
       .. code-block:: python
          
          import paddle
          import paddle.nn.functional as F

          x_var = paddle.randn((2, 3, 8, 8, 8), dtype='float32')
          w_var = paddle.randn((3, 6, 3, 3, 3), dtype='float32')

          y_var = F.conv3d_transpose(x_var, w_var)
          y_np = y_var.numpy()

          print(y_np.shape)
          # (2, 6, 10, 10, 10)
    """
    # entry checks
    if data_format not in ["NCDHW", "NDHWC"]:
        raise ValueError(
            "Attr(data_format) should be 'NCDHW' or 'NDHWC'. Received "
            "Attr(data_format): {}.".format(data_format))

    channel_last = (data_format == "NDHWC")
    channel_dim = -1 if channel_last else 1
    if len(x.shape) != 5:
        raise ValueError(
            "Input x should be 5D tensor, but received x with the shape of {}".
            format(x.shape))
    num_channels = x.shape[channel_dim]
    num_filters = weight.shape[1]
    if num_channels < 0:
        raise ValueError(
            "The channel dimension of the input({}) should be defined. "
            "Received: {}.".format(x.shape, num_channels))
    if groups <= 0:
        raise ValueError(
            "The groups of conv3d_transpose should be greater than 0. Received groups: {}".
            format(groups))
    if num_channels % groups != 0:
        raise ValueError(
            "The number of input channels must be divisible by Attr(groups). "
            "Received: number of channels({}), groups({}).".format(num_channels,
                                                                   groups))

    padding, padding_algorithm = _update_padding_nd(padding, channel_last, 3)
    stride = convert_to_list(stride, 3, 'stride')
    dilation = convert_to_list(dilation, 3, 'dilation')
    if output_size is None:
        output_size = []
    else:
        if output_padding != 0:
            raise ValueError('output_padding option is mutually exclusive with '
                             'output_size')
        if isinstance(output_size, (list, tuple, int)):
            output_size = convert_to_list(output_size, 3, 'output_size')
        else:
            raise ValueError(
                "output_size should be int, or list, tuple of ints")

    if output_padding == 0:
        output_padding = []
    else:
        output_padding = convert_to_list(output_padding, 3, 'output_padding')

    cudnn_version = get_cudnn_version()

    #TODO(LielinJiang): whether to use cudnn according to the version of cudnn
    use_cudnn = True if (core.is_compiled_with_cuda() and
                         cudnn_version is not None) else False

    op_type = 'conv3d_transpose'
    data_format_ = "NHWC" if channel_last else "NCHW"

    if in_dygraph_mode():
        attrs = ('output_padding', output_padding, 'output_size', output_size,
                 'paddings', padding, "padding_algorithm", padding_algorithm,
                 'strides', stride, 'dilations', dilation, 'groups', groups,
                 'use_cudnn', use_cudnn, "data_format", data_format_)
        pre_bias = getattr(_C_ops, op_type)(x, weight, *attrs)
        if bias is not None:
            out = nn.elementwise_add(pre_bias, bias, axis=channel_dim)
        else:
            out = pre_bias
    else:
        inputs = {'Input': [x], 'Filter': [weight]}
        attrs = {
            'output_padding': output_padding,
            'output_size': output_size,
            'paddings': padding,
            "padding_algorithm": padding_algorithm,
            'strides': stride,
            'dilations': dilation,
            'groups': groups,
            'use_cudnn': use_cudnn,
            "data_format": data_format_
        }
        helper = LayerHelper(op_type, **locals())
        check_variable_and_dtype(x, 'x', ['float16', 'float32', 'float64'],
                                 'conv3d')

        pre_bias = helper.create_variable_for_type_inference(x.dtype)
        outputs = {"Output": [pre_bias]}

        helper.append_op(
            type=op_type, inputs=inputs, outputs=outputs, attrs=attrs)
        if bias is not None:
            out = nn.elementwise_add(pre_bias, bias, axis=channel_dim)
        else:
            out = pre_bias

    return out
