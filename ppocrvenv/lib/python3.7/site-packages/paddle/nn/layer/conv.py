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

# TODO: define classes of convolutional neural network

import numpy as np

from ...fluid import get_flags
from ...fluid import core
from ...device import get_cudnn_version
from .. import Layer
from ..initializer import Normal
from .. import functional as F
from ...fluid.layers import utils
from ..functional.conv import _update_padding_nd

__all__ = []


def _get_default_param_initializer(num_channels, filter_size):
    filter_elem_num = num_channels * np.prod(filter_size)
    std = (2.0 / filter_elem_num)**0.5
    return Normal(0.0, std)


def _reverse_repeat_list(t, n):
    """Reverse the order of `t` and repeat each element for `n` times.
    This can be used to translate padding arg used by Conv and Pooling modules
    to the ones used by `F.pad`.
    """
    return list(x for x in reversed(t) for _ in range(n))


class _ConvNd(Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 transposed,
                 dims,
                 stride=1,
                 padding=0,
                 padding_mode='zeros',
                 output_padding=0,
                 dilation=1,
                 groups=1,
                 weight_attr=None,
                 bias_attr=None,
                 data_format="NCHW"):
        super(_ConvNd, self).__init__()
        assert weight_attr is not False, "weight_attr should not be False in Conv."
        self._param_attr = weight_attr
        self._bias_attr = bias_attr
        self._groups = groups
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._data_format = data_format

        valid_padding_modes = {'zeros', 'reflect', 'replicate', 'circular'}
        if padding_mode not in valid_padding_modes:
            raise ValueError(
                "padding_mode must be one of {}, but got padding_mode='{}'".
                format(valid_padding_modes, padding_mode))

        if padding_mode in {'reflect', 'replicate', 'circular'
                            } and not isinstance(padding, np.int):
            raise TypeError(
                "when padding_mode in ['reflect', 'replicate', 'circular'], type of padding must be int"
            )

        valid_format = {'NHWC', 'NCHW', 'NDHWC', 'NCDHW', 'NLC', 'NCL'}
        if data_format not in valid_format:
            raise ValueError(
                "data_format must be one of {}, but got data_format='{}'".
                format(valid_format, data_format))

        channel_last = (data_format == "NHWC") or (data_format == "NDHWC") or (
            data_format == "NLC")
        if channel_last:
            self._channel_dim = len(data_format) - 1
        else:
            self._channel_dim = 1

        self._stride = utils.convert_to_list(stride, dims, 'stride')
        self._dilation = utils.convert_to_list(dilation, dims, 'dilation')
        self._kernel_size = utils.convert_to_list(kernel_size, dims,
                                                  'kernel_size')
        self._padding = padding
        self._padding_mode = padding_mode
        self.output_padding = output_padding
        if dims != 1:
            self._updated_padding, self._padding_algorithm = _update_padding_nd(
                padding, channel_last, dims)

        if transposed:
            filter_shape = [self._in_channels, out_channels // groups
                            ] + self._kernel_size
        else:
            if in_channels % groups != 0:
                raise ValueError("in_channels must be divisible by groups.")

            if padding_mode in {'reflect', 'replicate', 'circular'}:
                _paired_padding = utils.convert_to_list(padding, dims,
                                                        'padding')
                self._reversed_padding_repeated_twice = _reverse_repeat_list(
                    _paired_padding, 2)

                self._updated_padding, self._padding_algorithm = _update_padding_nd(
                    0, channel_last, dims)

            filter_shape = [out_channels, in_channels // groups
                            ] + self._kernel_size

        def _get_default_param_initializer():
            if transposed:
                return None
            filter_elem_num = np.prod(self._kernel_size) * self._in_channels
            std = (2.0 / filter_elem_num)**0.5
            return Normal(0.0, std)

        self.weight = self.create_parameter(
            shape=filter_shape,
            attr=self._param_attr,
            default_initializer=_get_default_param_initializer())
        self.bias = self.create_parameter(
            attr=self._bias_attr, shape=[self._out_channels], is_bias=True)

        cudnn_version = get_cudnn_version()

        self._use_cudnn = True if (core.is_compiled_with_cuda() and
                                   cudnn_version is not None) else False

        self._op_type = "conv" + str(dims) + 'd'
        if self._op_type == 'conv2d' and (in_channels == groups and
                                          in_channels != 1 and
                                          out_channels % in_channels == 0):
            self._op_type = 'depthwise_conv2d'
            if core.is_compiled_with_rocm():
                self._use_cudnn = True
            else:
                self._use_cudnn = False

        if (core.is_compiled_with_cuda() and get_flags(
                "FLAGS_conv2d_disable_cudnn")["FLAGS_conv2d_disable_cudnn"]):
            self._use_cudnn = False

    def extra_repr(self):
        main_str = '{_in_channels}, {_out_channels}, kernel_size={_kernel_size}'
        if self._stride != [1] * len(self._stride):
            main_str += ', stride={_stride}'
        if self._padding != 0:
            main_str += ', padding={_padding}'
        if self._padding_mode is not 'zeros':
            main_str += ', padding_mode={_padding_mode}'
        if self.output_padding != 0:
            main_str += ', output_padding={output_padding}'
        if self._dilation != [1] * len(self._dilation):
            main_str += ', dilation={_dilation}'
        if self._groups != 1:
            main_str += ', groups={_groups}'
        main_str += ', data_format={_data_format}'
        return main_str.format(**self.__dict__)


class Conv1D(_ConvNd):
    r"""
    This interface is used to construct a callable object of the ``Conv1D`` class.
    For more details, refer to code examples.
    The convolution1D layer calculates the output based on the input, filter
    and stride, padding, dilation, groups parameters. Input and
    Output are in NCL format or NLC format, where N is batch size, C is the number of
    the feature map, L is the length of the feature map.
    Filter's shape is [MCK] , where M is the number of output feature map,
    C is the number of input feature map, K is the size of the kernel. 
    If the groups is greater than 1, C will equal the number of input feature map divided by the groups.
    If bias attribution and activation type are provided, bias is added to the
    output of the convolution, and the corresponding activation function is
    applied to the final result.

    For each input :math:`X` , the equation is:

    .. math::

        Out = \sigma (W \ast X + b)

    Where:

    * :math:`X`: Input value, a ``Tensor`` with 'NCL' format or 'NLC' format.
    * :math:`W`: Filter value, a ``Tensor`` with shape [MCK] .
    * :math:`\\ast`: Convolution operation.
    * :math:`b`: Bias value, a 1-D ``Tensor`` with shape [M].
    * :math:`\\sigma`: Activation function.
    * :math:`Out`: Output value, the shape of :math:`Out` and :math:`X` may be different.

    Example:

        - Input:

          Input shape: :math:`(N, C_{in}, L_{in})`

          Kernel shape: :math:`(C_{out}, C_{in}, K)`

        - Output:

          Output shape: :math:`(N, C_{out}, L_{out})`

        Where

        .. math::

            L_{out}&= \frac{(L_{in} + 2 * padding - (dilation * (L_f - 1) + 1))}{stride} + 1 \\

    Parameters:
        in_channels(int): The number of channels in the input image.
        out_channels(int): The number of filter. It is as same as the output
            feature map.
        kernel_size (int|tuple|list): The filter size. If kernel_size is a tuple/list,
            it must contain one integer, (kernel_size).
        stride (int|tuple|list, optional): The stride size. If stride is a tuple/list, it must
            contain one integer, (stride_size). Default: 1.
        padding(int|str|tuple|list, optional): The size of zeros to be padded. It must be in one of the following forms.
            1. a string in ['valid', 'same'].
            2. an int, which means the feature map is zero paded by size of `padding` on both sides.
            3. a list[int] or tuple[int] whose length is 1, which means the feature map is zero paded by size of `padding[0]` on both sides.
            The default value is 0.
        dilation (int|tuple|list, optional): The dilation size. If dilation is a tuple/list, it must
            contain one integer, (dilation_size). Default: 1.
        groups (int, optional): The groups number of the conv2d Layer. According to grouped
            convolution in Alex Krizhevsky's Deep CNN paper: when group=2,
            the first half of the filters is only connected to the first half
            of the input channels, while the second half of the filters is only
            connected to the second half of the input channels. Default: 1.
        padding_mode(str, optional): Four modes: 'zeros', 'reflect', 'replicate', 'circular'.
            When in 'zeros' mode, this op uses zeros to pad the input tensor.
            When in 'reflect' mode, uses reflection of the input boundaries to pad the input tensor.
            When in 'replicate' mode, uses input boundaries to pad the input tensor.
            When in 'circular' mode, uses circular input to pad the input tensor.
            Default is 'zeros'.
        weight_attr (ParamAttr, optional): The parameter attribute for learnable weights(Parameter)
            of conv1d. If it is set to None or one attribute of ParamAttr, conv1d
            will create ParamAttr as param_attr. If the Initializer of the param_attr
            is not set, the parameter is initialized with :math:`Normal(0.0, std)`,
            and the :math:`std` is :math:`(\frac{2.0 }{filter\_elem\_num})^{0.5}`. Default: None.
        bias_attr (ParamAttr or bool, optional): The attribute for the bias of conv1d.
            If it is set to False, no bias will be added to the output units.
            If it is set to None or one attribute of ParamAttr, conv1d
            will create ParamAttr as bias_attr. If the Initializer of the bias_attr
            is not set, the bias is initialized zero. Default: None.

    Attribute:

        **weight** (Parameter): the learnable weights of filter of this layer.

        **bias** (Parameter or None): the learnable bias of this layer.

    Shape:
        - x: 3-D tensor with shape: (batch, in_channels, length) or (batch, length, in_channels).
        - weight: 3-D tensor with shape: (out_channels, in_channels, kernel_size)
        - bias: 1-D tensor with shape: (out_channels)
        - output: 3-D tensor with same shape as input x.
    
    Raises:
        None

    Examples:
        .. code-block:: python

          import paddle
          from paddle.nn import Conv1D
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
          x_t = paddle.to_tensor(x)
          conv = Conv1D(3, 2, 3)
          conv.weight.set_value(w)
          y_t = conv(x_t)
          print(y_t)
          # [[[133. 238.]
          #   [160. 211.]]]
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 padding_mode='zeros',
                 weight_attr=None,
                 bias_attr=None,
                 data_format="NCL"):
        super(Conv1D, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            False,
            1,
            stride=stride,
            padding=padding,
            padding_mode=padding_mode,
            dilation=dilation,
            groups=groups,
            weight_attr=weight_attr,
            bias_attr=bias_attr,
            data_format=data_format)

    def forward(self, x):
        padding = 0
        if self._padding_mode != "zeros":
            x = F.pad(x,
                      self._reversed_padding_repeated_twice,
                      mode=self._padding_mode,
                      data_format=self._data_format)
        else:
            padding = self._padding

        out = F.conv1d(
            x,
            self.weight,
            bias=self.bias,
            padding=padding,
            stride=self._stride,
            dilation=self._dilation,
            groups=self._groups,
            data_format=self._data_format)
        return out


class Conv1DTranspose(_ConvNd):
    r"""
    This interface is used to construct a callable object of the ``Conv1DTranspose`` class.
    For more details, refer to code examples.
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
    * :math:`W`: Kernel value, a 3-D Tensor with 'MCK' format.
    * :math:`\\ast`: Convolution operation.
    * :math:`b`: Bias value, a 2-D Tensor with shape [M, 1].
    * :math:`\\sigma`: Activation function.
    * :math:`Out`: Output value, a 3-D Tensor with data format 'NCL' of 'NLC', the shape of :math:`Out` and :math:`X` may be different.

    Example:

        - Input:

          Input shape: :math:`(N, C_{in}, L_{in})`

          Filter shape: :math:`(C_{in}, C_{out}, L_f)`

        - Output:

          Output shape: :math:`(N, C_{out}, L_{out})`

        Where

        .. math::

           L^\prime_{out} &= (L_{in} - 1) * stride - pad_top - pad_bottom + dilation * (L_f - 1) + 1 \\\\
           L_{out} &\in [ L^\prime_{out}, L^\prime_{out} + stride ]

    Note:
          The conv1d_transpose can be seen as the backward of the conv1d. For conv1d,
          when stride > 1, conv1d maps multiple input shape to the same output shape,
          so for conv1d_transpose, when stride > 1, input shape maps multiple output shape.
          If output_size is None, :math:`L_{out} = L^\prime_{out}`;
          else, the :math:`L_{out}` of the output size must between :math:`L^\prime_{out}`
          and :math:`L^\prime_{out} + stride`.

    Args:
        in_channels(int): The number of channels in the input image.
        out_channels(int): The number of the filter. It is as same as the output
            feature map.
        kernel_size(int|tuple|list, optional): The filter size. If kernel_size is a tuple/list,
            it must contain one integers, (kernel_size). None if
            use output size to calculate kernel_size. Default: None. kernel_size and
            output_size should not be None at the same time.
        stride(int|tuple|list, optional): The stride size. It means the stride in transposed convolution.
            If stride is a tuple/list, it must contain one integer, (stride_size).
            Default: stride = 1.
        padding(int|list|str|tuple, optional): The padding size. The padding argument effectively adds
             `dilation * (kernel - 1)` amount of zero-padding on both sides of input. If `padding` is a
             string, either 'VALID' or 'SAME' supported, which is the padding algorithm.
             If `padding` is a tuple or list, it could be in two forms:
             `[pad]` or `[pad_left, pad_right]`. Default: padding = 0.
        output_padding(int|list|tuple, optional): The count of zeros to be added to tail of each dimension.
             If it is a tuple/list, it must contain one integer. Default: 0.
        groups(int, optional): The groups number of the Conv2D transpose layer. Inspired by
            grouped convolution in Alex Krizhevsky's Deep CNN paper, in which
            when group=2, the first half of the filters is only connected to the
            first half of the input channels, while the second half of the
            filters is only connected to the second half of the input channels.
            Default: groups = 1.
        bias(bool, optional): Whether to use bias. Default: True.
        dilation(int|tuple|list, optional): The dilation size. It means the spacing between the kernel points.
            If dilation is a tuple/list, it must contain one integer, (dilation_size).
            Default: dilation = 1.
        weight_attr (ParamAttr, optional): The parameter attribute for learnable parameters/weights
            of conv1d_transpose. If it is set to None or one attribute of ParamAttr, conv1d_transpose
            will create ParamAttr as param_attr. If the Initializer of the param_attr
            is not set, the parameter is initialized with Xavier. Default: None.
        bias_attr (ParamAttr|bool, optional): The parameter attribute for the bias of conv1d_transpose.
            If it is set to False, no bias will be added to the output units.
            If it is set to None or one attribute of ParamAttr, conv1d_transpose
            will create ParamAttr as bias_attr. If the Initializer of the bias_attr
            is not set, the bias is initialized zero. Default: None.

    Attribute:
        **weight** (Parameter): the learnable weights of filters of this layer.
        **bias** (Parameter or None): the learnable bias of this layer.

    Shape:

        - x(Tensor): 3-D tensor with shape (batch, in_channels, length) when data_format is "NCL" or shape (batch, length, in_channels) when data_format is "NLC".
        - weight(Tensor): 3-D tensor with shape (in_channels, out_channels, kernel_length).
        - bias(Tensor): 1-D tensor with shape (out_channels).
        - output_size(int|tuple|list, optional): The output image size. If output size is a tuple/list, it must contain one integer, (feature_length). None if use kernel_size, padding, output_padding and stride to calculate output_size. If output_size and kernel_size are specified at the same time, They should follow the formula above. Default: None. output_size and kernel_size should not be None at the same time.
        - output(Tensor): 3-D tensor with same shape as input x.

    Examples:
       .. code-block:: python

          import paddle
          from paddle.nn import Conv1DTranspose
          import numpy as np
          
          # shape: (1, 2, 4)
          x=np.array([[[4, 0, 9, 7],
                       [8, 0, 9, 2]]]).astype(np.float32)
          # shape: (2, 1, 2)
          y=np.array([[[7, 0]],
                      [[4, 2]]]).astype(np.float32)
          x_t = paddle.to_tensor(x)
          conv = Conv1DTranspose(2, 1, 2)
          conv.weight.set_value(y)
          y_t = conv(x_t)
          print(y_t)
          
          # [[[60. 16. 99. 75.  4.]]]
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 output_padding=0,
                 groups=1,
                 dilation=1,
                 weight_attr=None,
                 bias_attr=None,
                 data_format="NCL"):
        super(Conv1DTranspose, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            True,
            1,
            stride=stride,
            padding=padding,
            dilation=dilation,
            output_padding=output_padding,
            groups=groups,
            weight_attr=weight_attr,
            bias_attr=bias_attr,
            data_format=data_format)

    def forward(self, x, output_size=None):
        out = F.conv1d_transpose(
            x,
            self.weight,
            bias=self.bias,
            output_size=output_size,
            output_padding=self.output_padding,
            padding=self._padding,
            stride=self._stride,
            dilation=self._dilation,
            groups=self._groups,
            data_format=self._data_format)
        return out


class Conv2D(_ConvNd):
    r"""
    This interface is used to construct a callable object of the ``Conv2D`` class.
    For more details, refer to code examples.
    The convolution2D layer calculates the output based on the input, filter
    and strides, paddings, dilations, groups parameters. Input and
    Output are in NCHW format, where N is batch size, C is the number of
    the feature map, H is the height of the feature map, and W is the width of the feature map.
    Filter's shape is [MCHW] , where M is the number of output feature map,
    C is the number of input feature map, H is the height of the filter,
    and W is the width of the filter. If the groups is greater than 1,
    C will equal the number of input feature map divided by the groups.
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

    * :math:`X`: Input value, a ``Tensor`` with NCHW format.
    * :math:`W`: Filter value, a ``Tensor`` with shape [MCHW] .
    * :math:`\\ast`: Convolution operation.
    * :math:`b`: Bias value, a 1-D ``Tensor`` with shape [M].
    * :math:`\\sigma`: Activation function.
    * :math:`Out`: Output value, the shape of :math:`Out` and :math:`X` may be different.
    
    Parameters:
        in_channels(int): The number of input channels in the input image.
        out_channels(int): The number of output channels produced by the convolution.
        kernel_size(int|list|tuple, optional): The size of the convolving kernel.
        stride(int|list|tuple, optional): The stride size. If stride is a list/tuple, it must
            contain three integers, (stride_H, stride_W). Otherwise, the
            stride_H = stride_W = stride. The default value is 1.
        padding(int|str|tuple|list, optional): The padding size. Padding coule be in one of the following forms.
            1. a string in ['valid', 'same'].
            2. an int, which means each spartial dimension(depth, height, width) is zero paded by size of `padding` 
            3. a list[int] or tuple[int] whose length is the number of spartial dimensions, which contains the amount of padding on each side for each spartial dimension. It has the form [pad_d1, pad_d2, ...].
            4. a list[int] or tuple[int] whose length is 2 * number of spartial dimensions. It has the form  [pad_before, pad_after, pad_before, pad_after, ...] for all spartial dimensions.
            5. a list or tuple of pairs of ints. It has the form [[pad_before, pad_after], [pad_before, pad_after], ...]. Note that, the batch dimension and channel dimension are also included. Each pair of integers correspond to the amount of padding for a dimension of the input. Padding in batch dimension and channel dimension should be [0, 0] or (0, 0).
            The default value is 0.
        dilation(int|list|tuple, optional): The dilation size. If dilation is a list/tuple, it must
            contain three integers, (dilation_D, dilation_H, dilation_W). Otherwise, the
            dilation_D = dilation_H = dilation_W = dilation. The default value is 1.
        groups(int, optional): The groups number of the Conv3D Layer. According to grouped
            convolution in Alex Krizhevsky's Deep CNN paper: when group=2,
            the first half of the filters is only connected to the first half
            of the input channels, while the second half of the filters is only
            connected to the second half of the input channels. The default value is 1.
        padding_mode(str, optional): ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``. Default: ``'zeros'``.
        weight_attr(ParamAttr, optional): The parameter attribute for learnable parameters/weights
            of conv2d. If it is set to None or one attribute of ParamAttr, conv2d
            will create ParamAttr as param_attr. If it is set to None, the parameter
            is initialized with :math:`Normal(0.0, std)`, and the :math:`std` is
            :math:`(\frac{2.0 }{filter\_elem\_num})^{0.5}`. The default value is None.
        bias_attr(ParamAttr|bool, optional): The parameter attribute for the bias of conv2d.
            If it is set to False, no bias will be added to the output units.
            If it is set to None or one attribute of ParamAttr, conv2d
            will create ParamAttr as bias_attr. If the Initializer of the bias_attr
            is not set, the bias is initialized zero. The default value is None.
        data_format(str, optional): Data format that specifies the layout of input.
            It can be "NCHW" or "NHWC". Default: "NCHW".

    Attribute:

        **weight** (Parameter): the learnable weights of filter of this layer.

        **bias** (Parameter or None): the learnable bias of this layer.

    Shape:

        - x: :math:`(N, C_{in}, H_{in}, W_{in})`

        - weight: :math:`(C_{out}, C_{in}, K_{h}, K_{w})`

        - bias: :math:`(C_{out})`

        - output: :math:`(N, C_{out}, H_{out}, W_{out})`

        Where

        ..  math::

           H_{out}&= \frac{(H_{in} + 2 * paddings[0] - (dilations[0] * (kernel\_size[0] - 1) + 1))}{strides[0]} + 1

           W_{out}&= \frac{(W_{in} + 2 * paddings[1] - (dilations[1] * (kernel\_size[1] - 1) + 1))}{strides[1]} + 1

    Examples:

        .. code-block:: python

          import paddle
          import paddle.nn as nn
          
          paddle.disable_static()
          
          x_var = paddle.uniform((2, 4, 8, 8), dtype='float32', min=-1., max=1.)
          
          conv = nn.Conv2D(4, 6, (3, 3))
          y_var = conv(x_var)
          y_np = y_var.numpy()
          print(y_np.shape)
          # (2, 6, 6, 6)
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 padding_mode='zeros',
                 weight_attr=None,
                 bias_attr=None,
                 data_format="NCHW"):
        super(Conv2D, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            False,
            2,
            stride=stride,
            padding=padding,
            padding_mode=padding_mode,
            dilation=dilation,
            groups=groups,
            weight_attr=weight_attr,
            bias_attr=bias_attr,
            data_format=data_format)

    def forward(self, x):
        if self._padding_mode != 'zeros':
            x = F.pad(x,
                      self._reversed_padding_repeated_twice,
                      mode=self._padding_mode,
                      data_format=self._data_format)

        out = F.conv._conv_nd(
            x,
            self.weight,
            bias=self.bias,
            stride=self._stride,
            padding=self._updated_padding,
            padding_algorithm=self._padding_algorithm,
            dilation=self._dilation,
            groups=self._groups,
            data_format=self._data_format,
            channel_dim=self._channel_dim,
            op_type=self._op_type,
            use_cudnn=self._use_cudnn)
        return out


class Conv2DTranspose(_ConvNd):
    r"""
    This interface is used to construct a callable object of the ``Conv2DTranspose`` class.
    For more details, refer to code examples.
    The convolution2D transpose layer calculates the output based on the input,
    filter, and dilations, strides, paddings. Input and output
    are in NCHW format. Where N is batch size, C is the number of feature map,
    H is the height of the feature map, and W is the width of the feature map.
    Filter's shape is [CMHW] , where C is the number of input feature map,
    M is the number of output feature map, H is the height of the filter,
    and W is the width of the filter. If the groups is greater than 1,
    C will equal the number of input feature map divided by the groups.
    If bias attribution and activation type are provided, bias is added to
    the output of the convolution, and the corresponding activation function
    is applied to the final result.
    The details of convolution transpose layer, please refer to the following explanation and references
    `conv2dtranspose <https://arxiv.org/pdf/1603.07285.pdf>`_ .
    For each input :math:`X`, the equation is:

    ..  math::

        Out = \sigma (W \ast X + b)

    Where:

    * :math:`X`: Input value, a ``Tensor`` with NCHW format.
    * :math:`W`: Filter value, a ``Tensor`` with shape [CMHW] .
    * :math:`\\ast`: Convolution operation.
    * :math:`b`: Bias value, a 1-D ``Tensor`` with shape [M].
    * :math:`\\sigma`: Activation function.
    * :math:`Out`: Output value, the shape of :math:`Out` and :math:`X` may be different.
    
    Parameters:
        in_channels(int): The number of channels in the input image.
        out_channels(int): The number of channels produced by the convolution.
        kernel_size(int|list|tuple): The kernel size. If kernel_size is a list/tuple,
            it must contain two integers, (kernel_size_H, kernel_size_W).
            Otherwise, the kernel will be a square.
        stride(int|list|tuple, optional): The stride size. If stride is a list/tuple, it must
            contain two integers, (stride_H, stride_W). Otherwise, the
            stride_H = stride_W = stride. Default: 1.
        padding(int|str|tuple|list, optional): The padding size. Padding coule be in one of the following forms.
            1. a string in ['valid', 'same'].
            2. an int, which means each spartial dimension(depth, height, width) is zero paded by size of `padding` on both sides 
            3. a list[int] or tuple[int] whose length is the number of spartial dimensions, which contains the amount of padding on each side for each spartial dimension. It has the form [pad_d1, pad_d2, ...].
            4. a list[int] or tuple[int] whose length is 2 * number of spartial dimensions. It has the form  [pad_before, pad_after, pad_before, pad_after, ...] for all spartial dimensions.
            5. a list or tuple of pairs of ints. It has the form [[pad_before, pad_after], [pad_before, pad_after], ...]. Note that, the batch dimension and channel dimension are also included. Each pair of integers correspond to the amount of padding for a dimension of the input. Padding in batch dimension and channel dimension should be [0, 0] or (0, 0).
            The default value is 0.
        output_padding(int|list|tuple, optional): Additional size added to one side
            of each dimension in the output shape. Default: 0.
        dilation(int|list|tuple, optional): The dilation size. If dilation is a list/tuple, it must
            contain two integers, (dilation_H, dilation_W). Otherwise, the
            dilation_H = dilation_W = dilation. Default: 1.
        groups(int, optional): The groups number of the Conv2D transpose layer. Inspired by
            grouped convolution in Alex Krizhevsky's Deep CNN paper, in which
            when group=2, the first half of the filters is only connected to the
            first half of the input channels, while the second half of the
            filters is only connected to the second half of the input channels.
            Default: 1.
        weight_attr(ParamAttr, optional): The parameter attribute for learnable weights(Parameter)
            of conv2d_transpose. If it is set to None or one attribute of ParamAttr, conv2d_transpose
            will create ParamAttr as param_attr. If the Initializer of the param_attr
            is not set, the parameter is initialized with Xavier. Default: None.
        bias_attr(ParamAttr|bool, optional): The attribute for the bias of conv2d_transpose.
            If it is set to False, no bias will be added to the output units.
            If it is set to None or one attribute of ParamAttr, conv2d_transpose
            will create ParamAttr as bias_attr. If the Initializer of the bias_attr
            is not set, the bias is initialized zero. Default: None.
        data_format(str, optional): Data format that specifies the layout of input.
            It can be "NCHW" or "NHWC". Default: "NCHW".

    Attribute:

        **weight** (Parameter): the learnable weights of filters of this layer.

        **bias** (Parameter or None): the learnable bias of this layer.

    Shape:

        - x: :math:`(N, C_{in}, H_{in}, W_{in})`

        - weight: :math:`(C_{in}, C_{out}, K_{h}, K_{w})`

        - bias: :math:`(C_{out})`

        - output: :math:`(N, C_{out}, H_{out}, W_{out})`

        Where

        ..  math::

           H^\prime_{out} &= (H_{in} - 1) * strides[0] - 2 * paddings[0] + dilations[0] * (kernel\_size[0] - 1) + 1

           W^\prime_{out} &= (W_{in} - 1) * strides[1] - 2 * paddings[1] + dilations[1] * (kernel\_size[1] - 1) + 1

           H_{out} &\in [ H^\prime_{out}, H^\prime_{out} + strides[0] )

           W_{out} &\in [ W^\prime_{out}, W^\prime_{out} + strides[1] )

    Examples:

       .. code-block:: python

          import paddle
          import paddle.nn as nn
          
          paddle.disable_static()

          x_var = paddle.uniform((2, 4, 8, 8), dtype='float32', min=-1., max=1.)

          conv = nn.Conv2DTranspose(4, 6, (3, 3))
          y_var = conv(x_var)
          y_np = y_var.numpy()
          print(y_np.shape)
          # (2, 6, 10, 10)
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 output_padding=0,
                 dilation=1,
                 groups=1,
                 weight_attr=None,
                 bias_attr=None,
                 data_format="NCHW"):
        super(Conv2DTranspose, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            True,
            2,
            stride=stride,
            padding=padding,
            dilation=dilation,
            output_padding=output_padding,
            groups=groups,
            weight_attr=weight_attr,
            bias_attr=bias_attr,
            data_format=data_format)

    def forward(self, x, output_size=None):
        if output_size is None:
            output_padding = self.output_padding
        else:
            output_padding = 0

        out = F.conv2d_transpose(
            x,
            self.weight,
            bias=self.bias,
            padding=self._padding,
            output_padding=output_padding,
            stride=self._stride,
            dilation=self._dilation,
            groups=self._groups,
            output_size=output_size,
            data_format=self._data_format)
        return out


class Conv3D(_ConvNd):
    r"""
    **Convlution3d Layer**
    The convolution3d layer calculates the output based on the input, filter
    and strides, paddings, dilations, groups parameters. Input(Input) and
    Output(Output) are multidimensional tensors with a shape of 
    :math:`[N, C, D, H, W]` . Where N is batch size, C is the number of
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
    * :math:`b`: Bias value, a 1-D tensor with shape [M].
    * :math:`\\sigma`: Activation function.
    * :math:`Out`: Output value, the shape of :math:`Out` and :math:`X` may be different.

    Parameters:
        in_channels(int): The number of input channels in the input image.
        out_channels(int): The number of output channels produced by the convolution.
        kernel_size(int|list|tuple, optional): The size of the convolving kernel.
        stride(int|list|tuple, optional): The stride size. If stride is a list/tuple, it must
            contain three integers, (stride_D, stride_H, stride_W). Otherwise, the
            stride_D = stride_H = stride_W = stride. The default value is 1.
        padding(int|str|tuple|list, optional): The padding size. Padding coule be in one of the following forms.
            1. a string in ['valid', 'same'].
            2. an int, which means each spartial dimension(depth, height, width) is zero paded by size of `padding` 
            3. a list[int] or tuple[int] whose length is the number of spartial dimensions, which contains the amount of padding on each side for each spartial dimension. It has the form [pad_d1, pad_d2, ...].
            4. a list[int] or tuple[int] whose length is 2 * number of spartial dimensions. It has the form  [pad_before, pad_after, pad_before, pad_after, ...] for all spartial dimensions.
            5. a list or tuple of pairs of ints. It has the form [[pad_before, pad_after], [pad_before, pad_after], ...]. Note that, the batch dimension and channel dimension are also included. Each pair of integers correspond to the amount of padding for a dimension of the input. Padding in batch dimension and channel dimension should be [0, 0] or (0, 0).
            The default value is 0.
        dilation(int|list|tuple, optional): The dilation size. If dilation is a list/tuple, it must
            contain three integers, (dilation_D, dilation_H, dilation_W). Otherwise, the
            dilation_D = dilation_H = dilation_W = dilation. The default value is 1.
        groups(int, optional): The groups number of the Conv3D Layer. According to grouped
            convolution in Alex Krizhevsky's Deep CNN paper: when group=2,
            the first half of the filters is only connected to the first half
            of the input channels, while the second half of the filters is only
            connected to the second half of the input channels. The default value is 1.
        padding_mode(str, optional): ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``. Default: ``'zeros'``.
        weight_attr(ParamAttr, optional): The parameter attribute for learnable parameters/weights
            of conv3d. If it is set to None or one attribute of ParamAttr, conv3d
            will create ParamAttr as param_attr. If it is set to None, the parameter
            is initialized with :math:`Normal(0.0, std)`, and the :math:`std` is
            :math:`(\frac{2.0 }{filter\_elem\_num})^{0.5}`. The default value is None.
        bias_attr(ParamAttr|bool, optional): The parameter attribute for the bias of conv3d.
            If it is set to False, no bias will be added to the output units.
            If it is set to None or one attribute of ParamAttr, conv3d
            will create ParamAttr as bias_attr. If the Initializer of the bias_attr
            is not set, the bias is initialized zero. The default value is None.
        data_format(str, optional): Data format that specifies the layout of input.
            It can be "NCDHW" or "NDHWC". Default: "NCDHW".

    Attribute:

        **weight** (Parameter): the learnable weights of filters of this layer.

        **bias** (Parameter): the learnable bias of this layer.

    Shape:

        - x: :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})`

        - weight: :math:`(C_{out}, C_{in}, K_{d}, K_{h}, K_{w})`

        - bias: :math:`(C_{out})`

        - output: :math:`(N, C_{out}, D_{out}, H_{out}, W_{out})`

        Where

        ..  math::

           D_{out}&= \frac{(D_{in} + 2 * paddings[0] - (dilations[0] * (kernel\_size[0] - 1) + 1))}{strides[0]} + 1

           H_{out}&= \frac{(H_{in} + 2 * paddings[1] - (dilations[1] * (kernel\_size[1] - 1) + 1))}{strides[1]} + 1

           W_{out}&= \frac{(W_{in} + 2 * paddings[2] - (dilations[2] * (kernel\_size[2] - 1) + 1))}{strides[2]} + 1

    Raises:
        ValueError: If the shapes of input, filter_size, stride, padding and
                    groups mismatch.

    Examples:

        .. code-block:: python

          import paddle
          import paddle.nn as nn
          
          paddle.disable_static()

          x_var = paddle.uniform((2, 4, 8, 8, 8), dtype='float32', min=-1., max=1.)
          
          conv = nn.Conv3D(4, 6, (3, 3, 3))
          y_var = conv(x_var)
          y_np = y_var.numpy()
          print(y_np.shape)
          # (2, 6, 6, 6, 6)
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 padding_mode='zeros',
                 weight_attr=None,
                 bias_attr=None,
                 data_format="NCDHW"):
        super(Conv3D, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            False,
            3,
            stride=stride,
            padding=padding,
            padding_mode=padding_mode,
            dilation=dilation,
            groups=groups,
            weight_attr=weight_attr,
            bias_attr=bias_attr,
            data_format=data_format)

    def forward(self, x):
        if self._padding_mode != 'zeros':
            x = F.pad(x,
                      self._reversed_padding_repeated_twice,
                      mode=self._padding_mode,
                      data_format=self._data_format)

        out = F.conv._conv_nd(
            x,
            self.weight,
            bias=self.bias,
            stride=self._stride,
            padding=self._updated_padding,
            padding_algorithm=self._padding_algorithm,
            dilation=self._dilation,
            groups=self._groups,
            data_format=self._data_format,
            channel_dim=self._channel_dim,
            op_type=self._op_type,
            use_cudnn=self._use_cudnn)
        return out


class Conv3DTranspose(_ConvNd):
    r"""
    **Convlution3D transpose layer**
    The convolution3D transpose layer calculates the output based on the input,
    filter, and dilations, strides, paddings. Input(Input) and output(Output)
    are in NCDHW format. Where N is batch size, C is the number of channels,
    D is the depth of the feature, H is the height of the feature, and W
    is the width of the feature. Parameters(dilations, strides, paddings) are
    two elements. These two elements represent height and width, respectively.
    The details of convolution transpose layer, please refer to the following
    explanation and references `therein <https://arxiv.org/pdf/1603.07285.pdf>`_.
    If bias attribution and activation type are provided, bias is added to
    the output of the convolution, and the corresponding activation function
    is applied to the final result.
    For each input :math:`X`, the equation is:
    
    ..  math::

        Out = \sigma (W \ast X + b)

    In the above equation:

    * :math:`X`: Input value, a tensor with NCDHW format.
    * :math:`W`: Filter value, a tensor with CMDHW format.
    * :math:`\\ast`: Convolution operation.
    * :math:`b`: Bias value, a 1-D tensor with shape [M].
    * :math:`\\sigma`: Activation function.
    * :math:`Out`: Output value, the shape of :math:`Out` and :math:`X` may be different.

    **Note**:

          The conv3d_transpose can be seen as the backward of the conv3d. For conv3d,
          when stride > 1, conv3d maps multiple input shape to the same output shape, 
          so for conv3d_transpose, when stride > 1, input shape maps multiple output shape.
          If output_size is None, :math:`H_{out} = H^\prime_{out}, :math:`H_{out} = \
          H^\prime_{out}, W_{out} = W^\prime_{out}`; else, the :math:`D_{out}` of the output 
          size must between :math:`D^\prime_{out}` and :math:`D^\prime_{out} + strides[0]`, 
          the :math:`H_{out}` of the output size must between :math:`H^\prime_{out}` 
          and :math:`H^\prime_{out} + strides[1]`, and the :math:`W_{out}` of the output size must 
          between :math:`W^\prime_{out}` and :math:`W^\prime_{out} + strides[2]`, 
          conv3d_transpose can compute the kernel size automatically.

    Parameters:
        in_channels(int): The number of channels in the input image.
        out_channels(int): The number of channels produced by the convolution.
        kernel_size(int|list|tuple): The kernel size. If kernel_size is a list/tuple,
            it must contain three integers, (kernel_size_D, kernel_size_H, kernel_size_W).
            Otherwise, the kernel will be a square.
        stride(int|list|tuple, optional): The stride size. It means the stride in transposed convolution. 
            If stride is a list/tuple, it must contain three integers, (stride_depth, stride_height, 
            stride_width). Otherwise, stride_depth = stride_height = stride_width = stride. 
            The default value is 1.
        padding(int|str|tuple|list, optional): The padding size. Padding coule be in one of the following forms.
            1. a string in ['valid', 'same'].
            2. an int, which means each spartial dimension(depth, height, width) is zero paded by size of `padding` 
            3. a list[int] or tuple[int] whose length is the number of spartial dimensions, which contains the amount of padding on each side for each spartial dimension. It has the form [pad_d1, pad_d2, ...].
            4. a list[int] or tuple[int] whose length is 2 * number of spartial dimensions. It has the form  [pad_before, pad_after, pad_before, pad_after, ...] for all spartial dimensions.
            5. a list or tuple of pairs of ints. It has the form [[pad_before, pad_after], [pad_before, pad_after], ...]. Note that, the batch dimension and channel dimension are also included. Each pair of integers correspond to the amount of padding for a dimension of the input. Padding in batch dimension and channel dimension should be [0, 0] or (0, 0).
            The default value is 0.
        output_padding(int|list|tuple, optional): Additional size added to one side
            of each dimension in the output shape. Default: 0.
        dilation(int|list|tuple, optional): The dilation size. If dilation is a list/tuple, it must
            contain three integers, (dilation_D, dilation_H, dilation_W). Otherwise, the
            dilation_D = dilation_H = dilation_W = dilation. The default value is 1.
        groups(int, optional): The groups number of the Conv3D transpose layer. Inspired by
            grouped convolution in Alex Krizhevsky's Deep CNN paper, in which
            when group=2, the first half of the filters is only connected to the
            first half of the input channels, while the second half of the
            filters is only connected to the second half of the input channels.
            The default value is 1.
        weight_attr(ParamAttr, optional): The parameter attribute for learnable parameters/weights
            of conv3d_transpose. If it is set to None or one attribute of ParamAttr, conv3d_transpose
            will create ParamAttr as param_attr. If the Initializer of the param_attr
            is not set, the parameter is initialized with Xavier. The default value is None.
        bias_attr(ParamAttr|bool, optional): The parameter attribute for the bias of conv3d_transpose.
            If it is set to False, no bias will be added to the output units.
            If it is set to None or one attribute of ParamAttr, conv3d_transpose
            will create ParamAttr as bias_attr. If the Initializer of the bias_attr
            is not set, the bias is initialized zero. The default value is None.
        data_format(str, optional): Data format that specifies the layout of input.
            It can be "NCDHW" or "NDHWC". Default: "NCDHW".

    Attribute:

        **weight** (Parameter): the learnable weights of filters of this layer.

        **bias** (Parameter): the learnable bias of this layer.

    Shape:

        - x: :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})`

        - weight: :math:`(C_{in}, C_{out}, K_{d}, K_{h}, K_{w})`

        - bias: :math:`(C_{out})`

        - output: :math:`(N, C_{out}, D_{out}, H_{out}, W_{out})`

        Where

        ..  math::

           D^\prime_{out} &= (D_{in} - 1) * strides[0] - 2 * paddings[0] + dilations[0] * (kernel\_size[0] - 1) + 1
           
           H^\prime_{out} &= (H_{in} - 1) * strides[1] - 2 * paddings[1] + dilations[1] * (kernel\_size[1] - 1) + 1
           
           W^\prime_{out} &= (W_{in} - 1) * strides[2] - 2 * paddings[2] + dilations[2] * (kernel\_size[2] - 1) + 1
           
    Raises:
        ValueError: If the shapes of input, filter_size, stride, padding and
                    groups mismatch.
    Examples:

       .. code-block:: python

          import paddle
          import paddle.nn as nn
          
          paddle.disable_static()

          x_var = paddle.uniform((2, 4, 8, 8, 8), dtype='float32', min=-1., max=1.)
          
          conv = nn.Conv3DTranspose(4, 6, (3, 3, 3))
          y_var = conv(x_var)
          y_np = y_var.numpy()
          print(y_np.shape)
          # (2, 6, 10, 10, 10)
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 output_padding=0,
                 dilation=1,
                 groups=1,
                 weight_attr=None,
                 bias_attr=None,
                 data_format="NCDHW"):
        super(Conv3DTranspose, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            True,
            3,
            stride=stride,
            padding=padding,
            dilation=dilation,
            output_padding=output_padding,
            groups=groups,
            weight_attr=weight_attr,
            bias_attr=bias_attr,
            data_format=data_format)

    def forward(self, x, output_size=None):
        if output_size is None:
            output_padding = self.output_padding
        else:
            output_padding = 0

        out = F.conv3d_transpose(
            x,
            self.weight,
            bias=self.bias,
            padding=self._padding,
            output_padding=output_padding,
            stride=self._stride,
            dilation=self._dilation,
            groups=self._groups,
            output_size=output_size,
            data_format=self._data_format)
        return out
