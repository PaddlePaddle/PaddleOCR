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

# TODO: define pooling functions
from ...fluid import core
from ...fluid.framework import in_dygraph_mode
from ...fluid.layers import utils, LayerHelper
from ...tensor.manipulation import unsqueeze, squeeze
from ...fluid.data_feeder import check_type, check_variable_and_dtype
from paddle import _C_ops
from paddle import _C_ops

__all__ = []


def _is_list_or_tuple(input):
    return isinstance(input, (list, tuple))


def _check_input(x, dimension):
    if len(x.shape) != dimension:
        raise ValueError(
            "Excepted Input X is {}-D tensor, but received {}-D {}".format(
                dimension, len(x.shape), type(x)))


def _check_instance(x, x_name, types=(int, float)):

    if not isinstance(x, types):
        raise ValueError("Excepted {} type for {} but received type: {}. ".
                         format(types, x_name, type(x)))


def _check_value_limitation(x, x_name, min_limit=1e-3):
    def _check_value(x, x_name, min_limit=1e-3):
        if isinstance(x, int) and min_limit is not None and x < min_limit:
            raise ValueError(
                "Excepted the input {} to be greater than {} but received x: {}. ".
                format(x_name, min_limit, x))

    for ele in x:
        _check_value(ele, x_name)


def _zero_padding_in_batch_and_channel(padding, channel_last):
    if channel_last:
        return list(padding[0]) == [0, 0] and list(padding[-1]) == [0, 0]
    else:
        return list(padding[0]) == [0, 0] and list(padding[1]) == [0, 0]


def _exclude_padding_in_batch_and_channel(padding, channel_last):
    padding_ = padding[1:-1] if channel_last else padding[2:]
    padding_ = [elem for pad_a_dim in padding_ for elem in pad_a_dim]
    return padding_


def _channel_last(data_format, num_dims):
    if num_dims == 1:
        if data_format not in ['NCL', 'NLC']:
            raise ValueError(
                "Attr(data_format) should be 'NCL' or 'NLC'. Received "
                "Attr(data_format): %s" % str(data_format))
        else:
            return True if data_format == "NLC" else False
    if num_dims == 2:
        if data_format not in ['NCHW', 'NHWC']:
            raise ValueError(
                "Attr(data_format) should be 'NCHW' or 'NHWC'. Received "
                "Attr(data_format): %s" % str(data_format))
        else:
            return True if data_format == "NHWC" else False
    if num_dims == 3:
        if data_format not in ['NCDHW', 'NDHWC']:
            raise ValueError(
                "Attr(data_format) should be 'NCDHW' or 'NDHWC'. Received "
                "Attr(data_format): %s" % str(data_format))
        else:
            return True if data_format == "NDHWC" else False


def _update_padding_nd(padding, num_dims, channel_last=False, ceil_mode=False):
    if isinstance(padding, str):
        padding = padding.upper()
        if padding not in ["SAME", "VALID"]:
            raise ValueError(
                "Unknown padding: '{}'. It can only be 'SAME' or 'VALID'.".
                format(padding))
        if padding == "VALID":
            if ceil_mode != False:
                raise ValueError(
                    "When Attr(padding) is \"VALID\", Attr(ceil_mode) must be False. "
                    "Received ceil_mode: True.")

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
            if utils._is_symmetric_padding(padding, num_dims):
                padding = padding[0::2]
        # for padding like [pad_before, pad_after, pad_before, pad_after, ...]
        elif len(padding) == 2 * num_dims and isinstance(padding[0], int):
            padding_algorithm = "EXPLICIT"
            padding = utils.convert_to_list(padding, 2 * num_dims, 'padding')
            if utils._is_symmetric_padding(padding, num_dims):
                padding = padding[0::2]
        # for padding like [pad_d1, pad_d2, ...]
        elif len(padding) == num_dims and isinstance(padding[0], int):
            padding_algorithm = "EXPLICIT"
            padding = utils.convert_to_list(padding, num_dims, 'padding')
        else:
            raise ValueError("Invalid padding: {}".format(padding))
    # for integer padding
    else:
        padding_algorithm = "EXPLICIT"
        padding = utils.convert_to_list(padding, num_dims, 'padding')
    return padding, padding_algorithm


def _expand_low_nd_padding(padding):
    #1d to 2d fake input
    if len(padding) == 2:
        padding = [0] * 2 + padding
    elif len(padding) == 1:
        padding = [0] + padding
    else:
        raise ValueError(
            "The size of padding's dimmention should be 1 or 2. But got padding={}".
            format(padding))
    return padding


def avg_pool1d(x,
               kernel_size,
               stride=None,
               padding=0,
               exclusive=True,
               ceil_mode=False,
               name=None):
    """
    This API implements average pooling 1d operation,
    See more details in :ref:`api_nn_pooling_AvgPool1d` .

    Args:
        x (Tensor): The input tensor of pooling operator which is a 3-D tensor with
                          shape [N, C, L]. where `N` is batch size, `C` is the number of channels,
                          `L` is the length of the feature. The data type is float32 or float64.
        kernel_size (int|list|tuple): The pool kernel size. If pool kernel size is a tuple or list,
            it must contain an integer.
        stride (int|list|tuple): The pool stride size. If pool stride size is a tuple or list,
            it must contain an integer.
        padding (string|int|list|tuple): The padding size. Padding could be in one of the following forms.
            1. A string in ['valid', 'same'].
            2. An int, which means the feature map is zero padded by size of `padding` on every sides.
            3. A list[int] or tuple(int) whose length is 1, which means the feature map is zero padded by the size of `padding[0]` on every sides.
            4. A list[int] or tuple(int) whose length is 2. It has the form [pad_before, pad_after].
            5. A list or tuple of pairs of integers. It has the form [[pad_before, pad_after], [pad_before, pad_after], ...]. Note that, the batch dimension and channel dimension should be [0,0] or (0,0).
            The default value is 0.
        exclusive (bool): Whether to exclude padding points in average pooling
                          mode, default is `True`.
        ceil_mode (bool): ${ceil_mode_comment}Whether to use the ceil function to calculate output height and width.
            If it is set to False, the floor function will be used. The default value is False.
        name(str, optional): For detailed information, please refer
                             to :ref:`api_guide_Name`. Usually name is no need to set and
                             None by default.
    Returns:
        Tensor: The output tensor of pooling result. The data type is same as input tensor.

    Raises:
        ValueError: If `padding` is a string, but not "SAME" or "VALID".
        ValueError: If `padding` is "VALID", but `ceil_mode` is True.
        ValueError: If `padding` is a list or tuple but its length is greater than 1.
        ShapeError: If the input is not a 3-D tensor.
        ShapeError: If the output's shape calculated is not greater than 0.

    Examples:
        .. code-block:: python
          
            import paddle
            import paddle.nn.functional as F
            import numpy as np

            data = paddle.to_tensor(np.random.uniform(-1, 1, [1, 3, 32]).astype(np.float32))
            out = F.avg_pool1d(data, kernel_size=2, stride=2, padding=0)
            # out shape: [1, 3, 16]
    """
    """NCL to NCHW"""
    data_format = "NCHW"
    if not in_dygraph_mode():
        check_variable_and_dtype(x, 'x', ['float32', 'float64'], 'avg_pool1d')
    _check_input(x, 3)
    x = unsqueeze(x, [2])
    kernel_size = utils.convert_to_list(kernel_size, 1, 'kernel_size')
    kernel_size = [1] + kernel_size
    if stride is None:
        stride = kernel_size
    else:
        stride = utils.convert_to_list(stride, 1, 'pool_stride')
        stride = [1] + stride

    _check_value_limitation(kernel_size, "kernel_size", min_limit=1e-3)
    _check_value_limitation(stride, "stride", min_limit=1e-3)

    channel_last = _channel_last("NCL", 1)
    padding, padding_algorithm = _update_padding_nd(
        padding, 1, channel_last=channel_last, ceil_mode=ceil_mode)

    # use 2d to implenment 1d should expand padding in advance.
    padding = _expand_low_nd_padding(padding)

    if in_dygraph_mode():
        output = _C_ops.pool2d(
            x, 'pooling_type', 'avg', 'ksize', kernel_size, 'global_pooling',
            False, 'strides', stride, 'paddings', padding, 'padding_algorithm',
            padding_algorithm, 'use_cudnn', True, 'ceil_mode', ceil_mode,
            'use_mkldnn', False, 'exclusive', exclusive, 'data_format',
            data_format)
        return squeeze(output, [2])

    op_type = 'pool2d'
    helper = LayerHelper(op_type, **locals())
    dtype = helper.input_dtype(input_param_name='x')
    pool_out = helper.create_variable_for_type_inference(dtype)

    helper.append_op(
        type=op_type,
        inputs={"X": x},
        outputs={"Out": pool_out},
        attrs={
            "pooling_type": 'avg',
            "ksize": kernel_size,
            "global_pooling": False,
            "strides": stride,
            "paddings": padding,
            "padding_algorithm": padding_algorithm,
            "use_cudnn": True,
            "ceil_mode": ceil_mode,
            "use_mkldnn": False,
            "exclusive": exclusive,
            "data_format": data_format,
        })

    return squeeze(pool_out, [2])


def avg_pool2d(x,
               kernel_size,
               stride=None,
               padding=0,
               ceil_mode=False,
               exclusive=True,
               divisor_override=None,
               data_format="NCHW",
               name=None):
    """
    This API implements average pooling 2d operation.
    See more details in :ref:`api_nn_pooling_AvgPool2d` .

    Args:
        x (Tensor): The input tensor of pooling operator which is a 4-D tensor with
                          shape [N, C, H, W]. The format of input tensor is `"NCHW"` or
                          `"NHWC"`, where `N` is batch size, `C` is the number of channels,
                          `H` is the height of the feature, and `W` is the width of the
                          feature. The data type if float32 or float64.
        kernel_size (int|list|tuple): The pool kernel size. If it is a tuple or list,
            it must contain two integers, (kernel_size_Height, kernel_size_Width).
            Otherwise, the pool kernel size will be a square of an int.
        stride (int|list|tuple): The stride size. If it is a tuple or list,
            it must contain two integers, (stride_Height, stride_Width).
            Otherwise, the stride size will be a square of an int.

        padding (string|int|list|tuple): The padding size. Padding could be in one of the following forms.
            1. A string in ['valid', 'same'].
            2. An int, which means the feature map is zero padded by size of `padding` on every sides.
            3. A list[int] or tuple(int) whose length is 2, [pad_height, pad_weight] whose value means the padding size of each dimension.
            4. A list[int] or tuple(int) whose length is 4. [pad_height_top, pad_height_bottom, pad_width_left, pad_width_right] whose value means the padding size of each side.
            5. A list or tuple of pairs of integers. It has the form [[pad_before, pad_after], [pad_before, pad_after], ...]. Note that, the batch dimension and channel dimension should be [0,0] or (0,0).
            The default value is 0.
        ceil_mode (bool): when True, will use `ceil` instead of `floor` to compute the output shape
        exclusive (bool): Whether to exclude padding points in average pooling
                          mode, default is `true`.
        divisor_override (float): if specified, it will be used as divisor, otherwise kernel_size will be used. Default None.
        data_format (string): The data format of the input and output data. An optional string from: `"NCHW"`, `"NHWC"`.
                        The default is `"NCHW"`. When it is `"NCHW"`, the data is stored in the order of:
                        `[batch_size, input_channels, input_height, input_width]`.
        name(str, optional): For detailed information, please refer
                             to :ref:`api_guide_Name`. Usually name is no need to set and
                             None by default.
    
    Returns:
        Tensor: The output tensor of pooling result. The data type is same as input tensor.
    
    Raises:
        ValueError: If `padding` is a string, but not "SAME" or "VALID".
        ValueError: If `padding` is "VALID", but `ceil_mode` is True.
        ShapeError: If the output's shape calculated is not greater than 0.
    
    Examples:
        .. code-block:: python
          
            import paddle
            import paddle.nn.functional as F
            import numpy as np
            
            # avg pool2d
            x = paddle.to_tensor(np.random.uniform(-1, 1, [1, 3, 32, 32]).astype(np.float32))
            out = F.avg_pool2d(x,
                            kernel_size=2,
                            stride=2, padding=0)
            # out.shape [1, 3, 16, 16]
    """
    kernel_size = utils.convert_to_list(kernel_size, 2, 'pool_size')
    if stride is None:
        stride = kernel_size
    else:
        stride = utils.convert_to_list(stride, 2, 'pool_stride')

    _check_value_limitation(kernel_size, "kernel_size", min_limit=1e-3)
    _check_value_limitation(stride, "stride", min_limit=1e-3)

    channel_last = _channel_last(data_format, 2)
    padding, padding_algorithm = _update_padding_nd(
        padding, 2, channel_last, ceil_mode=ceil_mode)

    if in_dygraph_mode():
        output = _C_ops.pool2d(x, 'pooling_type', 'avg', 'ksize', kernel_size,
                               'global_pooling', False, 'padding_algorithm',
                               padding_algorithm, 'strides', stride, 'paddings',
                               padding, 'use_cudnn', True, 'ceil_mode',
                               ceil_mode, 'use_mkldnn', False, 'exclusive',
                               exclusive, 'data_format', data_format)
        if divisor_override is None:
            return output
        else:
            _check_instance(divisor_override, "divisor_override")
            return output * (kernel_size[0] * kernel_size[1]) / divisor_override

    op_type = 'pool2d'
    helper = LayerHelper(op_type, **locals())
    check_variable_and_dtype(x, 'x', ['float32', 'float64'], 'avg_pool2d')
    dtype = helper.input_dtype(input_param_name='x')
    pool_out = helper.create_variable_for_type_inference(dtype)

    helper.append_op(
        type=op_type,
        inputs={"X": x},
        outputs={"Out": pool_out},
        attrs={
            "pooling_type": "avg",
            "ksize": kernel_size,
            "global_pooling": False,
            "strides": stride,
            "paddings": padding,
            "padding_algorithm": padding_algorithm,
            "use_cudnn": True,
            "ceil_mode": ceil_mode,
            "use_mkldnn": False,
            "exclusive": exclusive,
            "data_format": data_format,
        })

    if divisor_override is None:
        return pool_out
    else:
        _check_instance(divisor_override, "divisor_override")
        return pool_out * (kernel_size[0] * kernel_size[1]) / divisor_override


def avg_pool3d(x,
               kernel_size,
               stride=None,
               padding=0,
               ceil_mode=False,
               exclusive=True,
               divisor_override=None,
               data_format="NCDHW",
               name=None):
    """
    This API implements average pooling 3d operation.
    See more details in :ref:`api_nn_pooling_AvgPool3d` .

    Args:
        x (Tensor): The input tensor of pooling operator, which is a 5-D tensor with
                          shape [N, C, D, H, W], where `N` represents the batch size, `C` represents
                          the number of channels, `D`, `H` and `W` represent the depth, height and width of the feature respectively.
        kernel_size (int|list|tuple): The pool kernel size. If pool kernel size
            is a tuple or list, it must contain three integers,
            (kernel_size_Depth, kernel_size_Height, kernel_size_Width).
            Otherwise, the pool kernel size will be the cube of an int.
        stride (int|list|tuple): The pool stride size. If pool stride size is a tuple or list,
            it must contain three integers, [stride_Depth, stride_Height, stride_Width).
            Otherwise, the pool stride size will be a cube of an int.
        padding (string|int|list|tuple): The padding size. Padding could be in one of the following forms.
            1. A string in ['valid', 'same'].
            2. An int, which means the feature map is zero padded by size of `padding` on every sides.
            3. A list[int] or tuple(int) whose length is 3, [pad_depth, pad_height, pad_weight] whose value means the padding size of each dimension.
            4. A list[int] or tuple(int) whose length is 6. [pad_depth_front, pad_depth_back, pad_height_top, pad_height_bottom, pad_width_left, pad_width_right] whose value means the padding size of each side.
            5. A list or tuple of pairs of integers. It has the form [[pad_before, pad_after], [pad_before, pad_after], ...]. Note that, the batch dimension and channel dimension should be [0,0] or (0,0).
            The default value is 0.
        ceil_mode (bool): ${ceil_mode_comment}
        exclusive (bool): Whether to exclude padding points in average pooling
                          mode, default is True.
        divisor_override (int|float) if specified, it will be used as divisor, otherwise kernel_size will be used. Default None.
        data_format (string): The data format of the input and output data. An optional string from: `"NCDHW"`, `"NDHWC"`.
                        The default is `"NCDHW"`. When it is `"NCDHW"`, the data is stored in the order of:
                        `[batch_size, input_channels, input_depth, input_height, input_width]`.
        name(str, optional): For detailed information, please refer
                             to :ref:`api_guide_Name`. Usually name is no need to set and
                             None by default.
    
    Returns:
        Tensor: The output tensor of pooling result. The data type is same as input tensor.
    
    Raises:
        ValueError: If `padding` is a string, but not "SAME" or "VALID".
        ValueError: If `padding` is "VALID", but `ceil_mode` is True.
        ShapeError: If the output's shape calculated is not greater than 0.
    
    Examples:
        .. code-block:: python
          
          import paddle
          import numpy as np

          x = paddle.to_tensor(np.random.uniform(-1, 1, [1, 3, 32, 32, 32]).astype(np.float32))
          # avg pool3d
          out = paddle.nn.functional.avg_pool3d(
                                            x,
                                            kernel_size = 2,
                                            stride = 2,
                                            padding=0)
          # out.shape: [1, 3, 16, 16, 16]
    """
    kernel_size = utils.convert_to_list(kernel_size, 3, 'pool_size')
    if stride is None:
        stride = kernel_size
    else:
        stride = utils.convert_to_list(stride, 3, 'pool_stride')

    channel_last = _channel_last(data_format, 3)
    padding, padding_algorithm = _update_padding_nd(
        padding, 3, channel_last=channel_last, ceil_mode=ceil_mode)

    _check_value_limitation(kernel_size, "kernel_size", min_limit=1e-3)
    _check_value_limitation(stride, "stride", min_limit=1e-3)

    if in_dygraph_mode():
        output = _C_ops.pool3d(
            x, 'pooling_type', 'avg', 'ksize', kernel_size, 'strides', stride,
            'paddings', padding, 'global_pooling', False, 'padding_algorithm',
            padding_algorithm, 'use_cudnn', True, 'ceil_mode', ceil_mode,
            'use_mkldnn', False, 'exclusive', exclusive, 'data_format',
            data_format)
        if divisor_override is None:
            return output
        else:
            _check_instance(divisor_override, "divisor_override")
            return output * (kernel_size[0] * kernel_size[1] *
                             kernel_size[2]) / divisor_override

    op_type = "pool3d"
    helper = LayerHelper(op_type, **locals())
    check_variable_and_dtype(x, 'x', ['float32', 'float64'], 'max_pool3d')
    dtype = helper.input_dtype(input_param_name='x')
    pool_out = helper.create_variable_for_type_inference(dtype)
    outputs = {"Out": pool_out}

    helper.append_op(
        type=op_type,
        inputs={"X": x},
        outputs=outputs,
        attrs={
            "pooling_type": 'avg',
            "ksize": kernel_size,
            "global_pooling": False,
            "strides": stride,
            "paddings": padding,
            "padding_algorithm": padding_algorithm,
            "use_cudnn": True,
            "ceil_mode": ceil_mode,
            "use_mkldnn": False,
            "exclusive": exclusive,
            "data_format": data_format,
        })

    if divisor_override is None:
        return pool_out
    else:
        _check_instance(divisor_override, "divisor_override")
        return pool_out * (kernel_size[0] * kernel_size[1] *
                           kernel_size[2]) / divisor_override


def max_pool1d(x,
               kernel_size,
               stride=None,
               padding=0,
               return_mask=False,
               ceil_mode=False,
               name=None):
    """
    This API implements max pooling 1d opereation.
    See more details in :ref:`api_nn_pooling_MaxPool1d` .

    Args:
        x (Tensor): The input tensor of pooling operator which is a 3-D tensor with
                          shape [N, C, L], where `N` is batch size, `C` is the number of channels,
                          `L` is the length of the feature. The data type if float32 or float64.
        kernel_size (int|list|tuple): The pool kernel size. If pool kernel size is a tuple or list,
            it must contain an integer.
        stride (int|list|tuple): The pool stride size. If pool stride size is a tuple or list,
            it must contain an integer.
        padding (string|int|list|tuple): The padding size. Padding could be in one of the following forms.
            1. A string in ['valid', 'same'].
            2. An integer, which means the feature map is zero padded by size of `padding` on every sides.
            3. A list[int] or tuple(int) whose length is 1, which means the feature map is zero padded by the size of `padding[0]` on every sides.
            4. A list[int] or tuple(int) whose length is 2. It has the form [pad_before, pad_after].
            5. A list or tuple of pairs of integers. It has the form [[pad_before, pad_after], [pad_before, pad_after], ...]. Note that, the batch dimension and channel dimension should be [0,0] or (0,0).
            The default value is 0.
        return_mask (bool): Whether return the max indices along with the outputs. default is `False`.
        ceil_mode (bool): Whether to use the ceil function to calculate output height and width. False is the default.
            If it is set to False, the floor function will be used. Default False.
        name(str, optional): For detailed information, please refer
                             to :ref:`api_guide_Name`. Usually name is no need to set and
                             None by default.
    Returns:
        Tensor: The output tensor of pooling result. The data type is same as input tensor.

    Raises:
        ValueError: If `padding` is a string, but not "SAME" or "VALID".
        ValueError: If `padding` is "VALID", but `ceil_mode` is True.
        ShapeError: If the input is not a 3-D tensor.
        ShapeError: If the output's shape calculated is not greater than 0.

    Examples:
        .. code-block:: python

          import paddle
          import paddle.nn.functional as F
          import numpy as np

          data = paddle.to_tensor(np.random.uniform(-1, 1, [1, 3, 32]).astype(np.float32))
          pool_out = F.max_pool1d(data, kernel_size=2, stride=2, padding=0)
          # pool_out shape: [1, 3, 16]
          pool_out, indices = F.max_pool1d(data, kernel_size=2, stride=2, padding=0, return_mask=True)
          # pool_out shape: [1, 3, 16],  indices shape: [1, 3, 16]
    """
    """NCL to NCHW"""
    data_format = "NCHW"
    if not in_dygraph_mode():
        check_variable_and_dtype(x, 'x', ['float32', 'float64'], 'max_pool1d')
    _check_input(x, 3)
    x = unsqueeze(x, [2])
    kernel_size = [1] + utils.convert_to_list(kernel_size, 1, 'pool_size')
    if stride is None:
        stride = kernel_size
    else:
        stride = [1] + utils.convert_to_list(stride, 1, 'pool_stride')

    padding, padding_algorithm = _update_padding_nd(
        padding, 1, ceil_mode=ceil_mode)

    # use 2d to implenment 1d should expand padding in advance.
    padding = _expand_low_nd_padding(padding)

    if in_dygraph_mode():
        if return_mask:
            pool_out = _C_ops.max_pool2d_with_index(
                x, 'ksize', kernel_size, 'global_pooling', False, 'strides',
                stride, 'paddings', padding, 'padding_algorithm',
                padding_algorithm, 'use_cudnn', True, 'ceil_mode', ceil_mode,
                'use_mkldnn', False, 'exclusive', True, 'data_format',
                data_format)
            return (squeeze(pool_out[0], [2]),
                    squeeze(pool_out[1],
                            [2])) if return_mask else squeeze(pool_out[0], [2])
        else:
            pool_out = _C_ops.pool2d(
                x, 'pooling_type', 'max', 'ksize', kernel_size,
                'global_pooling', False, 'padding_algorithm', padding_algorithm,
                'strides', stride, 'paddings', padding, 'use_cudnn', True,
                'ceil_mode', ceil_mode, 'use_mkldnn', False, 'exclusive', True,
                'data_format', data_format)
            return squeeze(pool_out, [2])

    op_type = 'max_pool2d_with_index' if return_mask else "pool2d"
    helper = LayerHelper(op_type, **locals())
    dtype = helper.input_dtype(input_param_name='x')
    pool_out = helper.create_variable_for_type_inference(dtype)
    mask = helper.create_variable_for_type_inference(dtype)
    outputs = {"Out": pool_out, "Mask": mask}

    helper.append_op(
        type=op_type,
        inputs={"X": x},
        outputs=outputs,
        attrs={
            "pooling_type": 'max',
            "ksize": kernel_size,
            "global_pooling": False,
            "strides": stride,
            "paddings": padding,
            "padding_algorithm": padding_algorithm,
            "use_cudnn": True,
            "ceil_mode": ceil_mode,
            "use_mkldnn": False,
            "exclusive": True,
            "data_format": data_format,
        })

    return (squeeze(pool_out, [2]),
            squeeze(mask, [2])) if return_mask else squeeze(pool_out, [2])


def _unpool_output_size(x, kernel_size, stride, padding, output_size):
    input_size = x.shape
    default_size = []
    for d in range(len(kernel_size)):
        default_size.append((input_size[-len(kernel_size) + d] - 1) * stride[d]
                            + kernel_size[d] - 2 * padding[d])
    if output_size is None:
        ret = default_size
    else:
        if len(output_size) == len(kernel_size) + 2:
            output_size = output_size[2:]
        if len(output_size) != len(kernel_size):
            raise ValueError(
                "output_size should be a sequence containing "
                "{} or {} elements, but it has a length of '{}'".format(
                    len(kernel_size), len(kernel_size) + 2, len(output_size)))
        for d in range(len(kernel_size)):
            min_size = default_size[d] - stride[d]
            max_size = default_size[d] + stride[d]
            if not (min_size < output_size[d] < max_size):
                raise ValueError(
                    'invalid output_size "{}" (dim {} must be between {} and {})'.
                    format(output_size, d, min_size, max_size))

        ret = output_size
    return ret


def max_unpool2d(x,
                 indices,
                 kernel_size,
                 stride=None,
                 padding=0,
                 data_format="NCHW",
                 output_size=None,
                 name=None):
    """
    This API implements max unpooling 2d opereation.
    See more details in :ref:`api_nn_pooling_MaxUnPool2D` .


    Args:
        x (Tensor): The input tensor of unpooling operator which is a 4-D tensor with
                          shape [N, C, H, W]. The format of input tensor is `"NCHW"`, 
                          where `N` is batch size, `C` is the number of channels,
                          `H` is the height of the feature, and `W` is the width of the
                          feature. The data type if float32 or float64.
        indices (Tensor): The indices given out by maxpooling2d which is a 4-D tensor with
                          shape [N, C, H, W]. The format of input tensor is `"NCHW"` , 
                          where `N` is batch size, `C` is the number of channels,
                          `H` is the height of the feature, and `W` is the width of the
                          feature. The data type if float32 or float64.
        kernel_size (int|list|tuple): The unpool kernel size. If unpool kernel size is a tuple or list,
            it must contain an integer.
        stride (int|list|tuple): The unpool stride size. If unpool stride size is a tuple or list,
            it must contain an integer.
        kernel_size (int|tuple): Size of the max unpooling window.
        padding (int | tuple): Padding that was added to the input.
        output_size(list|tuple, optional): The target output size. If output_size is not specified, 
                           the actual output shape will be automatically calculated by (input_shape,
                           kernel_size, padding).
        name(str, optional): For detailed information, please refer
                             to :ref:`api_guide_Name`. Usually name is no need to set and
                             None by default.


        - Input: :math:`(N, C, H_{in}, W_{in})`
        - Output: :math:`(N, C, H_{out}, W_{out})`, where

          .. math::
            H_{out} = (H_{in} - 1) \times \text{stride[0]} - 2 \times \text{padding[0]} + \text{kernel\_size[0]}

          .. math::
            W_{out} = (W_{in} - 1) \times \text{stride[1]} - 2 \times \text{padding[1]} + \text{kernel\_size[1]}

          or as given by :attr:`output_size` in the call operator

        Returns:
            Tensor: The output tensor of unpooling result. 

        Raises:
            ValueError: If the input is not a 4-D tensor.
            ValueError: If indeces shape is not equal input shape.
            

        Examples:
            .. code-block:: python
          
            import paddle
            import paddle.nn.functional as F

            data = paddle.rand(shape=[1,1,6,6])
            pool_out, indices = F.max_pool2d(data, kernel_size=2, stride=2, padding=0, return_mask=True)
            # pool_out shape: [1, 1, 3, 3],  indices shape: [1, 1, 3, 3]
            unpool_out = F.max_unpool2d(pool_out, indices, kernel_size=2, padding=0)
            # unpool_out shape: [1, 1, 6, 6]

            # specify a different output size than input size 
            unpool_out = F.max_unpool2d(pool_out, indices, kernel_size=2, padding=0, output_size=[7,7])
            # unpool_out shape: [1, 1, 7, 7] 

    """
    kernel_size = utils.convert_to_list(kernel_size, 2, 'pool_size')
    if stride is None:
        stride = kernel_size
    else:
        stride = utils.convert_to_list(stride, 2, 'pool_stride')
    padding = utils.convert_to_list(padding, 2, 'padding')

    if data_format not in ["NCHW"]:
        raise ValueError("Attr(data_format) should be 'NCHW'. Received "
                         "Attr(data_format): %s." % str(data_format))

    output_size = _unpool_output_size(x, kernel_size, stride, padding,
                                      output_size)

    if in_dygraph_mode():
        output = _C_ops.unpool(x, indices, 'unpooling_type', 'max', 'ksize',
                               kernel_size, 'strides', stride, 'paddings',
                               padding, "output_size", output_size,
                               "data_format", data_format)
        return output

    op_type = "unpool"
    helper = LayerHelper(op_type, **locals())
    dtype = helper.input_dtype(input_param_name="x")
    unpool_out = helper.create_variable_for_type_inference(dtype)

    helper.append_op(
        type=op_type,
        inputs={"X": x,
                "Indices": indices},
        outputs={"Out": unpool_out},
        attrs={
            "unpooling_type": "max",
            "ksize": kernel_size,
            "strides": stride,
            "paddings": padding,
            "output_size": output_size
        })
    return unpool_out


def max_pool2d(x,
               kernel_size,
               stride=None,
               padding=0,
               return_mask=False,
               ceil_mode=False,
               data_format="NCHW",
               name=None):
    kernel_size = utils.convert_to_list(kernel_size, 2, 'pool_size')
    if stride is None:
        stride = kernel_size
    else:
        stride = utils.convert_to_list(stride, 2, 'pool_stride')

    if data_format not in ["NCHW", "NHWC"]:
        raise ValueError(
            "Attr(data_format) should be 'NCHW' or 'NHWC'. Received "
            "Attr(data_format): %s." % str(data_format))

    channel_last = True if data_format == "NHWC" else False

    padding, padding_algorithm = _update_padding_nd(
        padding, num_dims=2, channel_last=channel_last, ceil_mode=ceil_mode)

    if data_format == "NHWC" and return_mask:
        raise ValueError(
            "When setting return_mask to true, data_format must be set to NCHW in API:max_pool2d"
        )

    if in_dygraph_mode():
        if return_mask:
            output = _C_ops.max_pool2d_with_index(
                x, 'ksize', kernel_size, 'global_pooling', False, 'strides',
                stride, 'paddings', padding, 'padding_algorithm',
                padding_algorithm, 'use_cudnn', True, 'ceil_mode', ceil_mode,
                'use_mkldnn', False, 'exclusive', True, 'data_format',
                data_format)
            return output if return_mask else output[0]
        else:
            output = _C_ops.pool2d(
                x, 'pooling_type', 'max', 'ksize', kernel_size,
                'global_pooling', False, 'padding_algorithm', padding_algorithm,
                'strides', stride, 'paddings', padding, 'use_cudnn', True,
                'ceil_mode', ceil_mode, 'use_mkldnn', False, 'exclusive', True,
                'data_format', data_format)
            return output

    op_type = 'max_pool2d_with_index' if return_mask else "pool2d"
    helper = LayerHelper(op_type, **locals())
    check_variable_and_dtype(x, 'x', ['float16', 'float32', 'float64'],
                             'max_pool2d')
    dtype = helper.input_dtype(input_param_name='x')
    pool_out = helper.create_variable_for_type_inference(dtype)
    mask = helper.create_variable_for_type_inference(dtype)
    outputs = {"Out": pool_out, "Mask": mask}

    helper.append_op(
        type=op_type,
        inputs={"X": x},
        outputs=outputs,
        attrs={
            "pooling_type": 'max',
            "ksize": kernel_size,
            "global_pooling": False,
            "strides": stride,
            "paddings": padding,
            "padding_algorithm": padding_algorithm,
            "use_cudnn": True,
            "ceil_mode": ceil_mode,
            "use_mkldnn": False,
            "exclusive": True,
            "data_format": data_format,
        })

    return (pool_out, mask) if return_mask else pool_out


def max_pool3d(x,
               kernel_size,
               stride=None,
               padding=0,
               return_mask=False,
               ceil_mode=False,
               data_format="NCDHW",
               name=None):
    """
    This API implements max pooling 2d operation.
    See more details in :ref:`api_nn_pooling_MaxPool3d` .
    Args:
        x (Tensor): The input tensor of pooling operator, which is a 5-D tensor with
                          shape [N, C, D, H, W]. The format of input tensor is `"NCDHW"` or `"NDHWC"`, where N represents batch size, C represents the number of channels, D, H and W represent the depth, height and width of the feature respectively.
        kernel_size (int|list|tuple): The pool kernel size. If the kernel size
            is a tuple or list, it must contain three integers,
            (kernel_size_Depth, kernel_size_Height, kernel_size_Width).
            Otherwise, the pool kernel size will be the cube of an int.
        stride (int|list|tuple): The pool stride size. If pool stride size is a tuple or list,
            it must contain three integers, [stride_Depth, stride_Height, stride_Width).
            Otherwise, the pool stride size will be a cube of an int.
        padding (string|int|list|tuple): The padding size. Padding could be in one of the following forms.
            1. A string in ['valid', 'same'].
            2. An int, which means the feature map is zero padded by size of `padding` on every sides.
            3. A list[int] or tuple(int) whose length is 3, [pad_depth, pad_height, pad_weight] whose value means the padding size of each dimension.
            4. A list[int] or tuple(int) whose length is 6. [pad_depth_front, pad_depth_back, pad_height_top, pad_height_bottom, pad_width_left, pad_width_right] whose value means the padding size of each side.
            5. A list or tuple of pairs of integers. It has the form [[pad_before, pad_after], [pad_before, pad_after], ...]. Note that, the batch dimension and channel dimension should be [0,0] or (0,0).
            The default value is 0.
        ceil_mode (bool): ${ceil_mode_comment}
        return_mask (bool): Whether to return the max indices along with the outputs. Default False. Only support "NDCHW" data_format.
        data_format (string): The data format of the input and output data. An optional string from: `"NCDHW"`, `"NDHWC"`.
                        The default is `"NCDHW"`. When it is `"NCDHW"`, the data is stored in the order of:
                        `[batch_size, input_channels, input_depth, input_height, input_width]`.
        name(str, optional): For detailed information, please refer
                             to :ref:`api_guide_Name`. Usually name is no need to set and
                             None by default.
    
    Returns:
        Tensor: The output tensor of pooling result. The data type is same as input tensor.
    
    Raises:
        ValueError: If `padding` is a string, but not "SAME" or "VALID".
        ValueError: If `padding` is "VALID", but `ceil_mode` is True.
        ShapeError: If the output's shape calculated is not greater than 0.
    
    Examples:
        .. code-block:: python

            import paddle
            import paddle.nn.functional as F
            import numpy as np

            # max pool3d
            x = paddle.to_tensor(np.random.uniform(-1, 1, [1, 3, 32, 32, 32]).astype(np.float32))
            output = F.max_pool2d(x,
                                  kernel_size=2,
                                  stride=2, padding=0)
            output.shape [1, 3, 16, 16, 16]
            # for return_mask=True
            x = paddle.to_tensor(np.random.uniform(-1, 1, [1, 3, 32, 32, 32]).astype(np.float32))
            output, max_indices = paddle.nn.functional.max_pool3d(x,
                                          kernel_size = 2,
                                          stride = 2,
                                          padding=0,
                                          return_mask=True)
            # output.shape [None, 3, 16, 16, 16], max_indices.shape [None, 3, 16, 16, 16],
    """
    kernel_size = utils.convert_to_list(kernel_size, 3, 'pool_size')
    if stride is None:
        stride = kernel_size
    else:
        stride = utils.convert_to_list(stride, 3, 'pool_stride')

    channel_last = _channel_last(data_format, 3)

    padding, padding_algorithm = _update_padding_nd(
        padding, 3, channel_last=channel_last, ceil_mode=ceil_mode)

    if data_format == "NDHWC" and return_mask:
        raise ValueError(
            "When setting return_mask to true, data_format must be set to NCDHW in API:max_pool3d"
        )

    if in_dygraph_mode():
        if return_mask:
            output = _C_ops.max_pool3d_with_index(
                x, 'pooling_type', 'max', 'ksize', kernel_size, 'strides',
                stride, 'paddings', padding, 'global_pooling', False,
                'padding_algorithm', padding_algorithm, 'use_cudnn', True,
                'ceil_mode', ceil_mode, 'use_mkldnn', False, 'exclusive', True,
                'data_format', data_format)
            return output if return_mask else output[0]
        else:
            output = _C_ops.pool3d(
                x, 'pooling_type', 'max', 'ksize', kernel_size,
                'global_pooling', False, 'padding_algorithm', padding_algorithm,
                'strides', stride, 'paddings', padding, 'use_cudnn', True,
                'ceil_mode', ceil_mode, 'use_mkldnn', False, 'exclusive', True,
                'data_format', data_format)
            return output

    op_type = "max_pool3d_with_index" if return_mask else "pool3d"
    helper = LayerHelper(op_type, **locals())
    check_variable_and_dtype(x, 'x', ['float32', 'float64'], 'max_pool3d')
    dtype = helper.input_dtype(input_param_name='x')
    pool_out = helper.create_variable_for_type_inference(dtype)
    mask = helper.create_variable_for_type_inference(dtype)
    outputs = {"Out": pool_out, "Mask": mask}

    helper.append_op(
        type=op_type,
        inputs={"X": x},
        outputs=outputs,
        attrs={
            "pooling_type": 'max',
            "ksize": kernel_size,
            "global_pooling": False,
            "strides": stride,
            "paddings": padding,
            "padding_algorithm": padding_algorithm,
            "use_cudnn": True,
            "ceil_mode": ceil_mode,
            "use_mkldnn": False,
            "exclusive": False,
            "data_format": data_format,
        })

    return (pool_out, mask) if return_mask else pool_out


def adaptive_avg_pool1d(x, output_size, name=None):
    """
    This API implements adaptive average pooling 1d operation.
    See more details in :ref:`api_nn_pooling_AdaptiveAvgPool1d` .

    Args:
        x (Tensor): The input tensor of pooling operator, which is a 3-D tensor
                              with shape [N, C, L].  The format of input tensor is NCL,
                              where N is batch size, C is the number of channels, L is the
                              length of the feature. The data type is float32 or float64.
        output_size (int): The target output size. It must be an integer.
        name(str, optional): For detailed information, please refer
                                 to :ref:`api_guide_Name`. Usually name is no need to set and
                                 None by default.
    Returns:
            Tensor: The output tensor of adaptive average pooling result. The data type is same
                      as input tensor.
    Raises:
            ValueError: 'output_size' should be an integer.
    Examples:
        .. code-block:: python

              # average adaptive pool1d
              # suppose input data in shape of [N, C, L], `output_size` is m or [m],
              # output shape is [N, C, m], adaptive pool divide L dimension
              # of input data into m grids averagely and performs poolings in each
              # grid to get output.
              # adaptive max pool performs calculations as follow:
              #
              #     for i in range(m):
              #         lstart = floor(i * L / m)
              #         lend = ceil((i + 1) * L / m)
              #         output[:, :, i] = sum(input[:, :, lstart: lend])/(lstart - lend)
              #
              import paddle
              import paddle.nn.functional as F
              import numpy as np

              data = paddle.to_tensor(np.random.uniform(-1, 1, [1, 3, 32]).astype(np.float32))
              pool_out = F.adaptive_average_pool1d(data, output_size=16)
              # pool_out shape: [1, 3, 16])
    """
    pool_type = 'avg'
    if not in_dygraph_mode():
        check_variable_and_dtype(x, 'x', ['float16', 'float32', 'float64'],
                                 'adaptive_pool2d')
        check_type(output_size, 'pool_size', (int), 'adaptive_pool1d')
    _check_input(x, 3)
    pool_size = [1] + utils.convert_to_list(output_size, 1, 'pool_size')

    x = unsqueeze(x, [2])
    if in_dygraph_mode():
        pool_out = _C_ops.pool2d(x, 'pooling_type', pool_type, 'ksize',
                                 pool_size, 'adaptive', True)
        return squeeze(pool_out, [2])

    l_type = "pool2d"

    helper = LayerHelper(l_type, **locals())
    dtype = helper.input_dtype(input_param_name='x')
    pool_out = helper.create_variable_for_type_inference(dtype)

    outputs = {"Out": pool_out}
    helper.append_op(
        type=l_type,
        inputs={"X": x},
        outputs=outputs,
        attrs={
            "pooling_type": pool_type,
            "ksize": pool_size,
            "adaptive": True,
        })

    return squeeze(pool_out, [2])


def adaptive_avg_pool2d(x, output_size, data_format='NCHW', name=None):
    """
    This API implements adaptive average pooling 2d operation.
    See more details in :ref:`api_nn_pooling_AdaptiveAvgPool2d` .

    Args:
        x (Tensor): The input tensor of adaptive avg pool2d operator, which is a 4-D tensor.
                          The data type can be float32 or float64.
        output_size (int|list|tuple): The pool kernel size. If pool kernel size is a tuple or list,
            it must contain two element, (H, W). H and W can be either a int, or None which means
            the size will be the same as that of the input.
        data_format (str): The data format of the input and output data. An optional string
            from: "NCHW", "NHWC". The default is "NCHW". When it is "NCHW", the data is stored in
            the order of: [batch_size, input_channels, input_height, input_width].
        name(str, optional): For detailed information, please refer
                             to :ref:`api_guide_Name`. Usually name is no need to set and
                             None by default.
    Returns:
        Tensor: The output tensor of avg adaptive pool2d result. The data type is same as input tensor.
    Raises:
        ValueError: If `data_format` is not "NCHW" or "NHWC".
    Examples:
        .. code-block:: python

            # adaptive avg pool2d
            # suppose input data in shape of [N, C, H, W], `output_size` is [m, n],
            # output shape is [N, C, m, n], adaptive pool divide H and W dimensions
            # of input data into m * n grids averagely and performs poolings in each
            # grid to get output.
            # adaptive avg pool performs calculations as follow:
            #
            #     for i in range(m):
            #         for j in range(n):
            #             hstart = floor(i * H / m)
            #             hend = ceil((i + 1) * H / m)
            #             wstart = floor(i * W / n)
            #             wend = ceil((i + 1) * W / n)
            #             output[:, :, i, j] = avg(input[:, :, hstart: hend, wstart: wend])
            #
            import paddle
            import numpy as np

            input_data = np.random.rand(2, 3, 32, 32)
            x = paddle.to_tensor(input_data)
            # x.shape is [2, 3, 32, 32]
            out = paddle.nn.functional.adaptive_avg_pool2d(
                            x = x,
                            output_size=[3, 3])
            # out.shape is [2, 3, 3, 3]
    """
    if not in_dygraph_mode():
        check_variable_and_dtype(x, 'x', ['float16', 'float32', 'float64'],
                                 'adaptive_avg_pool2d')
        check_type(data_format, 'data_format', str, 'adaptive_avg_pool2d')

    if data_format not in ["NCHW", "NHWC"]:
        raise ValueError(
            "Attr(data_format) should be 'NCHW' or 'NHWC'. Received "
            "Attr(data_format): %s." % str(data_format))

    if data_format == "NCHW":
        in_h, in_w = x.shape[2:4]
    else:
        in_h, in_w = x.shape[1:3]

    if isinstance(output_size, int):
        output_size = utils.convert_to_list(output_size, 2, 'output_size')
    else:
        output_size = list(output_size)
        if output_size[0] == None:
            output_size[0] = in_h
        if output_size[1] == None:
            output_size[1] = in_w

    if in_dygraph_mode():
        output = _C_ops.pool2d(x, 'pooling_type', 'avg', 'ksize', output_size,
                               'global_pooling', False, 'adaptive', True,
                               'data_format', data_format)
        return output

    l_type = 'pool2d'

    helper = LayerHelper(l_type, **locals())
    dtype = helper.input_dtype(input_param_name='x')
    pool_out = helper.create_variable_for_type_inference(dtype)

    outputs = {"Out": pool_out}

    helper.append_op(
        type=l_type,
        inputs={"X": x},
        outputs=outputs,
        attrs={
            "pooling_type": "avg",
            "ksize": output_size,
            "adaptive": True,
            "data_format": data_format,
        })

    return pool_out


def adaptive_avg_pool3d(x, output_size, data_format='NCDHW', name=None):
    """
    This API implements adaptive average pooling 3d operation.
    See more details in :ref:`api_nn_pooling_AdaptiveAvgPool3d` .

    Args:
        x (Tensor): The input tensor of adaptive avg pool3d operator, which is a 5-D tensor.
                          The data type can be float32, float64.
        output_size (int|list|tuple): The pool kernel size. If pool kernel size is a tuple or list,
            it must contain three elements, (D, H, W). D, H and W can be either a int, or None which means
            the size will be the same as that of the input.
        data_format (str): The data format of the input and output data. An optional string
            from: "NCDHW", "NDHWC". The default is "NCDHW". When it is "NCDHW", the data is stored in
            the order of: [batch_size, input_channels, input_depth, input_height, input_width].
        name(str, optional): For detailed information, please refer
                             to :ref:`api_guide_Name`. Usually name is no need to set and
                             None by default.
    Returns:
        Tensor: The output tensor of avg adaptive pool3d result. The data type is same as input tensor.
    Raises:
        ValueError: If `data_format` is not "NCDHW" or "NDHWC".
    Examples:
        .. code-block:: python

            # adaptive avg pool3d
            # suppose input data in shape of [N, C, D, H, W], `output_size` is [l, m, n],
            # output shape is [N, C, l, m, n], adaptive pool divide D, H and W dimensions
            # of input data into l * m * n grids averagely and performs poolings in each
            # grid to get output.
            # adaptive avg pool performs calculations as follow:
            #
            #     for i in range(l):
            #         for j in range(m):
            #             for k in range(n):
            #                 dstart = floor(i * D / l)
            #                 dend = ceil((i + 1) * D / l)
            #                 hstart = floor(j * H / m)
            #                 hend = ceil((j + 1) * H / m)
            #                 wstart = floor(k * W / n)
            #                 wend = ceil((k + 1) * W / n)
            #                 output[:, :, i, j, k] =
            #                     avg(input[:, :, dstart:dend, hstart: hend, wstart: wend])
            import paddle
            import numpy as np
            input_data = np.random.rand(2, 3, 8, 32, 32)
            x = paddle.to_tensor(input_data)
            # x.shape is [2, 3, 8, 32, 32]
            out = paddle.nn.functional.adaptive_avg_pool3d(
                            x = x,
                            output_size=[3, 3, 3])
            # out.shape is [2, 3, 3, 3, 3]
    """
    if not in_dygraph_mode():
        check_variable_and_dtype(x, 'x', ['float32', 'float64'],
                                 'adaptive_avg_pool3d')
        check_type(data_format, 'data_format', str, 'adaptive_avg_pool3d')

    if data_format not in ["NCDHW", "NDHWC"]:
        raise ValueError(
            "Attr(data_format) should be 'NCDHW' or 'NDHWC'. Received "
            "Attr(data_format): %s." % str(data_format))

    if data_format == "NCDHW":
        in_l, in_h, in_w = x.shape[2:5]
    else:
        in_l, in_h, in_w = x.shape[1:4]

    if isinstance(output_size, int):
        output_size = utils.convert_to_list(output_size, 3, 'output_size')
    else:
        output_size = list(output_size)
        if output_size[0] == None:
            output_size[0] = in_l
        if output_size[1] == None:
            output_size[1] = in_h
        if output_size[2] == None:
            output_size[2] = in_w

    if in_dygraph_mode():
        output = _C_ops.pool3d(x, 'pooling_type', 'avg', 'ksize', output_size,
                               'global_pooling', False, 'adaptive', True,
                               'data_format', data_format)
        return output

    l_type = 'pool3d'

    helper = LayerHelper(l_type, **locals())
    dtype = helper.input_dtype(input_param_name='x')
    pool_out = helper.create_variable_for_type_inference(dtype)
    outputs = {"Out": pool_out}

    helper.append_op(
        type=l_type,
        inputs={"X": x},
        outputs=outputs,
        attrs={
            "pooling_type": "avg",
            "ksize": output_size,
            "adaptive": True,
            "data_format": data_format,
        })

    return pool_out


def adaptive_max_pool1d(x, output_size, return_mask=False, name=None):
    """
    This API implements adaptive max pooling 1d operation.
    See more details in :ref:`api_nn_pooling_AdaptiveMaxPool1d` .

    Args:
        x (Tensor): The input tensor of pooling operator, which is a 3-D tensor
                              with shape [N, C, L].  The format of input tensor is NCL,
                              where N is batch size, C is the number of channels, L is the
                              length of the feature. The data type is float32 or float64.
        output_size (int): The pool kernel size. The value should be an integer.
        return_mask (bool): If true, the index of max pooling point will be returned along
                with outputs. It cannot be set in average pooling type. Default False.
        name(str, optional): For detailed information, please refer
                                 to :ref:`api_guide_Name`. Usually name is no need to set and
                                 None by default.
    Returns:
            Tensor: The output tensor of adaptive pooling result. The data type is same
                      as input tensor.
    Raises:
            ValueError: 'output_size' should be an integer.
    Examples:
        .. code-block:: python

              # max adaptive pool1d
              # suppose input data in shape of [N, C, L], `output_size` is m or [m],
              # output shape is [N, C, m], adaptive pool divide L dimension
              # of input data into m grids averagely and performs poolings in each
              # grid to get output.
              # adaptive max pool performs calculations as follow:
              #
              #     for i in range(m):
              #         lstart = floor(i * L / m)
              #         lend = ceil((i + 1) * L / m)
              #         output[:, :, i] = max(input[:, :, lstart: lend])
              #
              import paddle
              import paddle.nn.functional as F
              import numpy as np

              data = paddle.to_tensor(np.random.uniform(-1, 1, [1, 3, 32]).astype(np.float32))
              pool_out = F.adaptive_max_pool1d(data, output_size=16)
              # pool_out shape: [1, 3, 16])
              pool_out, indices = F.adaptive_max_pool1d(data, output_size=16, return_mask=True)
              # pool_out shape: [1, 3, 16] indices  shape: [1, 3, 16]
    """
    pool_type = 'max'
    if not in_dygraph_mode():
        check_variable_and_dtype(x, 'x', ['float32', 'float64'],
                                 'adaptive_max_pool1d')
        check_type(output_size, 'pool_size', int, 'adaptive_max_pool1d')
        check_type(return_mask, 'return_mask', bool, 'adaptive_max_pool1d')
    _check_input(x, 3)

    pool_size = [1] + utils.convert_to_list(output_size, 1, 'pool_size')

    x = unsqueeze(x, [2])
    if in_dygraph_mode():
        pool_out = _C_ops.max_pool2d_with_index(
            x, 'pooling_type', pool_type, 'ksize', pool_size, 'adaptive', True)
        return (squeeze(pool_out[0], [2]), squeeze(
            pool_out[1], [2])) if return_mask else squeeze(pool_out[0], [2])

    l_type = 'max_pool2d_with_index'

    helper = LayerHelper(l_type, **locals())
    dtype = helper.input_dtype(input_param_name='x')
    pool_out = helper.create_variable_for_type_inference(dtype)

    mask = helper.create_variable_for_type_inference(dtype)
    outputs = {"Out": pool_out, "Mask": mask}

    helper.append_op(
        type=l_type,
        inputs={"X": x},
        outputs=outputs,
        attrs={
            "pooling_type": pool_type,
            "ksize": pool_size,
            "adaptive": True,
        })

    return (squeeze(pool_out, [2]),
            squeeze(mask, [2])) if return_mask else squeeze(pool_out, [2])


def adaptive_max_pool2d(x, output_size, return_mask=False, name=None):
    """
        This operation applies a 2D adaptive max pooling on input tensor.
        See more details in :ref:`api_nn_pooling_AdaptiveMaxPool2d` .

        Args:
            x (Tensor): The input tensor of adaptive max pool2d operator, which is a 4-D tensor. The data type can be float16, float32, float64, int32 or int64.
            output_size (int|list|tuple): The pool kernel size. If pool kernel size is a tuple or list, it must contain two elements, (H, W). H and W can be either a int, or None which means the size will be the same as that of the input.
            return_mask (bool): If true, the index of max pooling point will be returned along with outputs. Default False.
            name(str, optional): For detailed information, please refer to :ref:`api_guide_Name`. Usually name is no need to set and None by default.

        Returns:
            Tensor: The output tensor of adaptive max pool2d result. The data type is same as input tensor.

        Examples:
            .. code-block:: python

              # max adaptive pool2d
              # suppose input data in the shape of [N, C, H, W], `output_size` is [m, n]
              # output shape is [N, C, m, n], adaptive pool divide H and W dimensions
              # of input data into m*n grids averagely and performs poolings in each
              # grid to get output.
              # adaptive max pool performs calculations as follow:
              #
              #     for i in range(m):
              #         for j in range(n):
              #             hstart = floor(i * H / m)
              #             hend = ceil((i + 1) * H / m)
              #             wstart = floor(i * W / n)
              #             wend = ceil((i + 1) * W / n)
              #             output[:, :, i, j] = max(input[:, :, hstart: hend, wstart: wend])
              #
              import paddle
              import numpy as np

              input_data = np.random.rand(2, 3, 32, 32)
              x = paddle.to_tensor(input_data)
              # x.shape is [2, 3, 32, 32]
              out = paddle.nn.functional.adaptive_max_pool2d(
                            x = x,
                            output_size=[3, 3])
              # out.shape is [2, 3, 3, 3]
    """
    if not in_dygraph_mode():
        check_variable_and_dtype(x, 'x', ['float32', 'float64'],
                                 'adaptive_max_pool2d')
        check_type(return_mask, 'return_mask', bool, 'adaptive_max_pool2d')
        #check_type(output_size, 'pool_size', (int), 'adaptive_max_pool2d')
    _check_input(x, 4)

    in_h, in_w = x.shape[2:4]
    if isinstance(output_size, int):
        output_size = utils.convert_to_list(output_size, 2, 'output_size')
    else:
        output_size = list(output_size)
        if output_size[0] == None:
            output_size[0] = in_h
        if output_size[1] == None:
            output_size[1] = in_w

    if in_dygraph_mode():
        pool_out = _C_ops.max_pool2d_with_index(
            x, 'pooling_type', 'max', 'ksize', output_size, 'adaptive', True)
        return pool_out if return_mask else pool_out[0]

    l_type = 'max_pool2d_with_index'

    helper = LayerHelper(l_type, **locals())
    dtype = helper.input_dtype(input_param_name='x')
    pool_out = helper.create_variable_for_type_inference(dtype)

    mask = helper.create_variable_for_type_inference(dtype)
    outputs = {"Out": pool_out, "Mask": mask}

    helper.append_op(
        type=l_type,
        inputs={"X": x},
        outputs=outputs,
        attrs={
            "pooling_type": 'max',
            "ksize": output_size,
            "adaptive": True,
        })
    #return (pool_out, mask) if return_mask else pool_out
    return pool_out


def adaptive_max_pool3d(x, output_size, return_mask=False, name=None):
    """
        This operation applies a 3D adaptive max pooling on input tensor.
        See more details in :ref:`api_nn_pooling_AdaptiveMaxPool3d` .

        Args:
            x (Tensor): The input tensor of adaptive max pool3d operator, which is a 5-D tensor. The data type can be float32, float64.
            output_size (int|list|tuple): The pool kernel size. If pool kernel size is a tuple or list, it must contain three elements, (D, H, W). D, H and W can be either a int, or None which means the size will be the same as that of the input.
            return_mask (bool): If true, the index of max pooling point will be returned along with outputs. Default False.
            name(str, optional): For detailed information, please refer to :ref:`api_guide_Name`. Usually name is no need to set and None by default.

        Returns:
            Tensor: The output tensor of adaptive max pool3d result. The data type is same as input tensor.

        Examples:
            .. code-block:: python

              # adaptive max pool3d
              # suppose input data in the shape of [N, C, D, H, W], `output_size` is [l, m, n]
              # output shape is [N, C, l, m, n], adaptive pool divide D, H and W dimensions
              # of input data into m*n grids averagely and performs poolings in each
              # grid to get output.
              # adaptive max pool performs calculations as follow:
              #
              #     for i in range(l):
              #         for j in range(m):
              #             for k in range(n):
              #                 dstart = floor(i * D / l)
              #                 dend = ceil((i + 1) * D / l)
              #                 hstart = floor(i * H / m)
              #                 hend = ceil((i + 1) * H / m)
              #                 wstart = floor(i * W / n)
              #                 wend = ceil((i + 1) * W / n)
              #             output[:, :, i, j, k] = max(input[:, :, dstart: dend, hstart: hend, wstart: wend])
              #
              import paddle
              import numpy as np

              input_data = np.random.rand(2, 3, 8, 32, 32)
              x = paddle.to_tensor(input_data)
              # x.shape is [2, 3, 8, 32, 32]
              out = paddle.nn.functional.adaptive_max_pool3d(
                            x = x,
                            output_size=[3, 3, 3])
              # out.shape is [2, 3, 3, 3, 3]
    """

    if not in_dygraph_mode():
        check_variable_and_dtype(x, 'x', ['float32', 'float64'],
                                 'adaptive_max_pool3d')
        check_type(return_mask, 'return_mask', bool, 'adaptive_max_pool3d')
        #check_type(output_size, 'pool_size', (int), 'adaptive_max_pool3d')
    _check_input(x, 5)

    in_l, in_h, in_w = x.shape[2:5]
    if isinstance(output_size, int):
        output_size = utils.convert_to_list(output_size, 3, 'output_size')
    else:
        output_size = list(output_size)
        if output_size[0] == None:
            output_size[0] = in_l
        if output_size[1] == None:
            output_size[1] = in_h
        if output_size[2] == None:
            output_size[2] = in_w

    if in_dygraph_mode():
        pool_out = _C_ops.max_pool3d_with_index(
            x, 'pooling_type', 'max', 'ksize', output_size, 'adaptive', True)
        return pool_out if return_mask else pool_out[0]

    l_type = 'max_pool3d_with_index'

    helper = LayerHelper(l_type, **locals())
    dtype = helper.input_dtype(input_param_name='x')
    pool_out = helper.create_variable_for_type_inference(dtype)

    mask = helper.create_variable_for_type_inference(dtype)
    outputs = {"Out": pool_out, "Mask": mask}

    helper.append_op(
        type=l_type,
        inputs={"X": x},
        outputs=outputs,
        attrs={
            "pooling_type": 'max',
            "ksize": output_size,
            "adaptive": True,
        })

    return (pool_out, mask) if return_mask else pool_out
