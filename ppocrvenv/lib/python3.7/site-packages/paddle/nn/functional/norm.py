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

# TODO: define normalization api
import paddle
import paddle.fluid as fluid
from ...fluid.data_feeder import check_variable_and_dtype, check_type
from ...fluid.layer_helper import LayerHelper
from ...fluid.framework import in_dygraph_mode, core
from ...framework import create_parameter
from ..initializer import Constant
from ...framework import ParamAttr
from ...fluid import core, dygraph_utils
import numbers
from paddle import _C_ops

__all__ = []


def normalize(x, p=2, axis=1, epsilon=1e-12, name=None):
    r"""
    This op normalizes ``x`` along dimension ``axis`` using :math:`L_p` norm. This layer computes

    .. math::

        y = \frac{x}{ \max\left( \lvert \lvert x \rvert \rvert_p, epsilon\right) }

    .. math::
        \lvert \lvert x \rvert \rvert_p = \left( \sum_i {\lvert x_i \rvert^p}  \right)^{1/p}

    where, :math:`\sum_i{\lvert x_i \rvert^p}` is calculated along the ``axis`` dimension.


    Parameters:
        x (Tensor): The input tensor could be N-D tensor, and the input data type could be float32 or float64.
        p (float|int, optional): The exponent value in the norm formulation. Default: 2
        axis (int, optional): The axis on which to apply normalization. If `axis < 0`, the dimension to normalization is `x.ndim + axis`. -1 is the last dimension.
        epsilon (float, optional): Small float added to denominator to avoid dividing by zero. Default is 1e-12.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, the output has the same shape and data type with ``x``.

    Examples:

        .. code-block:: python

            import numpy as np
            import paddle
            import paddle.nn.functional as F

            paddle.disable_static()
            x = np.arange(6, dtype=np.float32).reshape(2,3)
            x = paddle.to_tensor(x)
            y = F.normalize(x)
            print(y.numpy())
            # [[0.         0.4472136  0.8944272 ]
            # [0.42426404 0.5656854  0.7071067 ]]

            y = F.normalize(x, p=1.5)
            print(y.numpy())
            # [[0.         0.40862012 0.81724024]
            # [0.35684016 0.4757869  0.5947336 ]]

            y = F.normalize(x, axis=0)
            print(y.numpy())
            # [[0.         0.24253564 0.37139067]
            # [1.         0.97014254 0.9284767 ]]
    """
    if in_dygraph_mode():
        eps = fluid.dygraph.base.to_variable([epsilon], dtype=x.dtype)
        out = _C_ops.p_norm(x, 'axis', axis, 'porder',
                            float(p), 'keepdim', True, 'epsilon', epsilon)
        return x / _C_ops.elementwise_max(out, eps)

    check_type(p, 'p', (float, int), 'normalize')
    check_type(axis, 'axis', (int), 'normalize')
    check_variable_and_dtype(x, 'x', ['float16', 'float32', 'float64'],
                             'normalize')
    if len(x.shape) == 1 and axis != 0 and axis != -1:
        raise ValueError(
            "Axis must be 0 or -1 when x is a 1-D tensor, but received axis = {}".
            format(axis))

    attrs = {
        'axis': axis,
        'porder': float(p),
        'keepdim': True,
        'epsilon': epsilon,
    }
    helper = LayerHelper('p_norm', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type='p_norm', inputs={'X': x}, outputs={'Out': out}, attrs=attrs)
    eps = out.block.create_var(dtype=out.dtype)
    paddle.fluid.layers.fill_constant([1], out.dtype, epsilon, out=eps)
    return paddle.divide(x, paddle.maximum(out, eps), name=name)


def batch_norm(x,
               running_mean,
               running_var,
               weight,
               bias,
               training=False,
               momentum=0.9,
               epsilon=1e-05,
               data_format="NCHW",
               use_global_stats=None,
               name=None):
    """
    Applies Batch Normalization as described in the paper Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift .

    nn.functional.batch_norm is uesd for nn.BatchNorm1D, nn.BatchNorm2D, nn.BatchNorm3D. Please use above API for BatchNorm.

    Parameters:
        x(Tesnor): input value. It's data type should be float32, float64.
        running_mean(Tensor): running mean.
        running_var(Tensor): running variance.
        weight(Tensor): The weight tensor of batch_norm, can not be None.
        bias(Tensor): The bias tensor of batch_norm can not be None.
        epsilon(float, optional): The small value added to the variance to prevent division by zero. Default: 1e-5.
        momentum(float, optional): The value used for the moving_mean and moving_var computation. Default: 0.9.
        training(bool, optional): True means train mode which compute by batch data and track global mean and var during train period. False means inference mode which compute by global mean and var which calculated by train period. Defalut False.
        data_format(str, optional): Specify the input data format, may be "NC", "NCL", "NCHW", "NCDHW", "NLC", "NHWC" or "NDHWC". Defalut "NCHW".
        use_global_stats(bool|None, optional): Whether to use global mean and variance. If set to False, use the statistics of one mini-batch, if set to True, use the global statistics, if set to None, use global statistics in the test phase and use the statistics of one mini-batch in the training phase. Default: None.
        name(str, optional): Name for the BatchNorm, default is None. For more information, please refer to :ref:`api_guide_Name`..

    Returns:
        None

    Examples:
        .. code-block:: python

          import paddle
          import numpy as np

          x = np.random.seed(123)
          x = np.random.random(size=(2, 1, 2, 3)).astype('float32')
          running_mean = np.random.random(size=1).astype('float32')
          running_variance = np.random.random(size=1).astype('float32')
          weight_data = np.random.random(size=1).astype('float32')
          bias_data = np.random.random(size=1).astype('float32')
          x = paddle.to_tensor(x)
          rm = paddle.to_tensor(running_mean)
          rv = paddle.to_tensor(running_variance)
          w = paddle.to_tensor(weight_data)
          b = paddle.to_tensor(bias_data)
          batch_norm_out = paddle.nn.functional.batch_norm(x, rm, rv, w, b)
          print(batch_norm_out)
    """
    assert len(x.shape) >= 2, "input dim must be larger than 1"

    # input ad out must share the memory
    mean_out = running_mean
    variance_out = running_var

    true_data_format = ['NC', 'NCL', 'NCHW', 'NCDHW', 'NLC', 'NHWC', 'NDHWC']
    if data_format not in true_data_format:
        raise ValueError(
            "data_format must be one of 'NC', 'NCL', 'NCHW', 'NCDHW', "
            "'NLC', 'NHWC', 'NDHWC' but receive {}".format(data_format))

    data_format = 'NCHW' if data_format[1] == 'C' else 'NHWC'

    if use_global_stats == None:
        use_global_stats = not training
        trainable_statistics = False
    else:
        trainable_statistics = not use_global_stats

    if in_dygraph_mode():
        # for dygraph need tuple
        attrs = ("momentum", momentum, "epsilon", epsilon, "is_test",
                 not training, "data_layout", data_format, "use_mkldnn", False,
                 "fuse_with_relu", False, "use_global_stats", use_global_stats,
                 "trainable_statistics", trainable_statistics)
        batch_norm_out, _, _, _, _, _ = _C_ops.batch_norm(
            x, weight, bias, running_mean, running_var, mean_out, variance_out,
            *attrs)
        return dygraph_utils._append_activation_in_dygraph(
            batch_norm_out, act=None)

    check_variable_and_dtype(x, 'input', ['float16', 'float32', 'float64'],
                             'BatchNorm')

    # for static need dict
    attrs = {
        "momentum": momentum,
        "epsilon": epsilon,
        "is_test": not training,
        "data_layout": data_format,
        "use_mkldnn": False,
        "fuse_with_relu": False,
        "use_global_stats": use_global_stats,
        "trainable_statistics": trainable_statistics,
    }

    inputs = {
        "X": [x],
        "Scale": [weight],
        "Bias": [bias],
        "Mean": [running_mean],
        "Variance": [running_var]
    }

    helper = LayerHelper('batch_norm', **locals())

    param_dtype = x.dtype if x.dtype is not 'float16' else 'float32'
    saved_mean = helper.create_variable_for_type_inference(
        dtype=param_dtype, stop_gradient=True)
    saved_variance = helper.create_variable_for_type_inference(
        dtype=param_dtype, stop_gradient=True)
    batch_norm_out = helper.create_variable_for_type_inference(x.dtype)

    outputs = {
        "Y": [batch_norm_out],
        "MeanOut": [running_mean],
        "VarianceOut": [running_var],
        "SavedMean": [saved_mean],
        "SavedVariance": [saved_variance]
    }

    if training or trainable_statistics:
        # reserve_space is only used for training.
        reserve_space = helper.create_variable_for_type_inference(
            dtype=x.dtype, stop_gradient=True)
        outputs["ReserveSpace"] = [reserve_space]

    helper.append_op(
        type="batch_norm", inputs=inputs, outputs=outputs, attrs=attrs)

    return helper.append_activation(batch_norm_out)


def layer_norm(x,
               normalized_shape,
               weight=None,
               bias=None,
               epsilon=1e-05,
               name=None):
    """
    see more detail in paddle.nn.LayerNorm

    Parameters:
        x(Tensor): Input Tensor. It's data type should be float32, float64.
        normalized_shape(int|list|tuple): Input shape from an expected input of
            size :math:`[*, normalized_shape[0], normalized_shape[1], ..., normalized_shape[-1]]`.
            If it is a single integer, this module will normalize over the last dimension
            which is expected to be of that specific size.
        epsilon(float, optional): The small value added to the variance to prevent
            division by zero. Default: 1e-05.
        weight(Tensor, optional): The weight tensor of batch_norm. Default: None.
        bias(Tensor, optional): The bias tensor of batch_norm. Default: None.
        name(str, optional): Name for the LayerNorm, default is None. For more information, please refer to :ref:`api_guide_Name`..

    Returns:
        None

    Examples:

        .. code-block:: python

          import paddle
          import numpy as np

          np.random.seed(123)
          x_data = np.random.random(size=(2, 2, 2, 3)).astype('float32')
          x = paddle.to_tensor(x_data)
          layer_norm_out = paddle.nn.functional.layer_norm(x, x.shape[1:])
          print(layer_norm_out)
    """
    input_shape = list(x.shape)
    input_ndim = len(input_shape)
    if isinstance(normalized_shape, numbers.Integral):
        normalized_shape = [normalized_shape]
    elif isinstance(normalized_shape, tuple):
        normalized_shape = list(normalized_shape)
    elif not isinstance(normalized_shape, list):
        raise ValueError(
            "`normalized_shape` should be int, list of ints or tuple of ints.")

    normalized_ndim = len(normalized_shape)
    begin_norm_axis = input_ndim - normalized_ndim
    if input_ndim < normalized_ndim or input_shape[
            begin_norm_axis:] != normalized_shape:
        str_normalized_shape = str(normalized_shape)
        raise ValueError('Given normalized_shape is ' + str_normalized_shape +
                         ', expected input with shape [*, ' +
                         str_normalized_shape[
                             1:] + ', but got input shape ' + str(input_shape))

    if in_dygraph_mode():
        pre_act, _, _ = _C_ops.layer_norm(x, weight, bias, 'epsilon', epsilon,
                                          'begin_norm_axis', begin_norm_axis)
        return dygraph_utils._append_activation_in_dygraph(pre_act, act=None)

    check_variable_and_dtype(x, 'input', ['float16', 'float32', 'float64'],
                             'LayerNorm')

    inputs = dict()
    inputs['X'] = [x]
    if weight:
        inputs['Scale'] = [weight]
    if bias:
        inputs['Bias'] = [bias]
    attrs = {"epsilon": epsilon, "begin_norm_axis": begin_norm_axis}

    # create output
    helper = LayerHelper('layer_norm', **locals())

    dtype = x.dtype
    mean_out = helper.create_variable_for_type_inference(
        dtype=dtype, stop_gradient=True)
    variance_out = helper.create_variable_for_type_inference(
        dtype=dtype, stop_gradient=True)
    layer_norm_out = helper.create_variable_for_type_inference(dtype)

    helper.append_op(
        type="layer_norm",
        inputs=inputs,
        outputs={
            "Y": layer_norm_out,
            "Mean": mean_out,
            "Variance": variance_out,
        },
        attrs={"epsilon": epsilon,
               "begin_norm_axis": begin_norm_axis})

    return helper.append_activation(layer_norm_out)


def instance_norm(x,
                  running_mean=None,
                  running_var=None,
                  weight=None,
                  bias=None,
                  use_input_stats=True,
                  momentum=0.9,
                  eps=1e-05,
                  data_format="NCHW",
                  name=None):
    """
    See more detail in nn.layer.InstanceNorm2D.

    Parameters:
        x(Tensor): Input Tensor. It's data type should be float32, float64.
        running_mean(Tensor): running mean. Default None.
        running_var(Tensor): running variance. Default None.
        weight(Tensor, optional): The weight tensor of instance_norm. Default: None.
        bias(Tensor, optional): The bias tensor of instance_norm. Default: None.
        eps(float, optional): A value added to the denominator for numerical stability. Default is 1e-5.
        momentum(float, optional): The value used for the moving_mean and moving_var computation. Default: 0.9.
        use_input_stats(bool): Default True.
        data_format(str, optional): Specify the input data format, may be "NC", "NCL", "NCHW" or "NCDHW". Defalut "NCHW".
        name(str, optional): Name for the InstanceNorm, default is None. For more information, please refer to :ref:`api_guide_Name`..

    Returns:
        None.

    Examples:

        .. code-block:: python

          import paddle
          import numpy as np

          np.random.seed(123)
          x_data = np.random.random(size=(2, 2, 2, 3)).astype('float32')
          x = paddle.to_tensor(x_data)
          instance_norm_out = paddle.nn.functional.instance_norm(x)

          print(instance_norm_out)

    """

    if in_dygraph_mode():
        out, _, _ = _C_ops.instance_norm(x, weight, bias, "epsilon", eps,
                                         "momentum", momentum, "data_format",
                                         data_format)
        return out

    check_variable_and_dtype(x, 'input', ['float32', 'float64'], "InstanceNorm")

    attrs = {"epsilon": eps, "momentum": momentum, "data_format": data_format}

    if weight and bias:
        inputs = {"X": [x], "Scale": [weight], "Bias": [bias]}
    else:
        inputs = {"X": [x]}

    helper = LayerHelper('instance_norm', **locals())
    saved_mean = helper.create_variable_for_type_inference(
        dtype=x.dtype, stop_gradient=True)
    saved_variance = helper.create_variable_for_type_inference(
        dtype=x.dtype, stop_gradient=True)
    instance_norm_out = helper.create_variable_for_type_inference(x.dtype)

    outputs = {
        "Y": [instance_norm_out],
        "SavedMean": [saved_mean],
        "SavedVariance": [saved_variance]
    }

    helper.append_op(
        type="instance_norm", inputs=inputs, outputs=outputs, attrs=attrs)
    return instance_norm_out


def local_response_norm(x,
                        size,
                        alpha=1e-4,
                        beta=0.75,
                        k=1.,
                        data_format="NCHW",
                        name=None):
    r"""
        Local Response Normalization performs a type of "lateral inhibition" by normalizing over local input regions.
        For more information, please refer to `ImageNet Classification with Deep Convolutional Neural Networks <https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf>`_

        The formula is as follows:

        .. math::

            Output(i, x, y) = Input(i, x, y) / \left(k + \alpha \sum\limits^{\min(C-1, i + size/2)}_{j = \max(0, i - size/2)}(Input(j, x, y))^2\right)^{\beta}

        In the above equation:

        - :math:`size` : The number of channels to sum over.
        - :math:`k` : The offset (avoid being divided by 0).
        - :math:`\\alpha` : The scaling parameter.
        - :math:`\\beta` : The exponent parameter.


        Args:
            x (Tensor): The input 3-D/4-D/5-D tensor. The data type is float32.
            size (int): The number of channels to sum over.
            alpha (float, optional): The scaling parameter, positive. Default:1e-4
            beta (float, optional): The exponent, positive. Default:0.75
            k (float, optional): An offset, positive. Default: 1.0
            data_format (str, optional): Specify the data format of the input, and the data format of the output
                will be consistent with that of the input. An optional string from:
                If x is 3-D Tensor, the string could be `"NCL"` or `"NLC"` . When it is `"NCL"`,
                the data is stored in the order of: `[batch_size, input_channels, feature_length]`.
                If x is 4-D Tensor, the string could be  `"NCHW"`, `"NHWC"`. When it is `"NCHW"`,
                the data is stored in the order of: `[batch_size, input_channels, input_height, input_width]`.
                If x is 5-D Tensor, the string could be  `"NCDHW"`, `"NDHWC"` . When it is `"NCDHW"`,
                the data is stored in the order of: `[batch_size, input_channels, input_depth, input_height, input_width]`.
            name (str, optional): Name for the operation (optional, default is None). For more information,
                please refer to :ref:`api_guide_Name`.

        Returns:
            A tensor storing the transformation result with the same shape and data type as input.


        Examples:

        .. code-block:: python

            import paddle

            x = paddle.rand(shape=(3, 3, 112, 112), dtype="float32")
            y = paddle.nn.functional.local_response_norm(x, size=5)
            print(y.shape)  # [3, 3, 112, 112]
        """
    if not in_dygraph_mode():
        check_variable_and_dtype(x, 'x', ['float32'], 'local_response_norm')
    if data_format not in ['NCL', 'NLC', 'NCHW', 'NHWC', 'NCDHW', 'NDHWC']:
        raise ValueError(
            "data_format should be in one of [NCL, NCHW, NCDHW, NLC, NHWC, NDHWC], " \
            "but got {}".format(data_format))

    sizes = x.shape
    dim = len(sizes)
    if dim < 3:
        raise ValueError(
            'Expected 3D or higher dimensionality input, but got {} dimensions'.
            format(dim))

    for i, sz in enumerate(sizes):
        if not sz > 0:
            raise ValueError("Expected every dim's size to be larger than 0, "
                             "but the size of the {}-th dim is {}".format(i,
                                                                          sz))

    channel_last = True if data_format[-1] == "C" else False

    from functools import reduce
    sum_sizes = reduce(lambda x, y: x * y, sizes[1:])

    div = paddle.unsqueeze(paddle.multiply(x, x), axis=1)
    if not channel_last:
        pad4d_shape = [0, 0, size // 2, (size - 1) // 2]
        pool2d_shape = (size, 1)
        reshape_shape = [
            sizes[0], 1, sizes[1], sizes[2],
            int(sum_sizes / (sizes[1] * sizes[2]))
        ]
        pad5d_shape = [0, 0, 0, 0, size // 2, (size - 1) // 2]
        pool3d_shape = (size, 1, 1)
    else:
        pad4d_shape = [size // 2, (size - 1) // 2, 0, 0]
        pool2d_shape = (1, size)
        reshape_shape = [
            sizes[0], 1, sizes[1], int(sum_sizes / (sizes[1] * sizes[-1])),
            sizes[-1]
        ]
        pad5d_shape = [size // 2, (size - 1) // 2, 0, 0, 0, 0]
        pool3d_shape = (1, 1, size)

    if dim == 3:
        div = paddle.nn.functional.pad(div, pad=pad4d_shape)
        div = paddle.nn.functional.avg_pool2d(
            div, kernel_size=pool2d_shape, stride=1)
        div = paddle.squeeze(div, axis=1)
    else:
        div = paddle.reshape(div, shape=reshape_shape)
        div = paddle.nn.functional.pad(div,
                                       pad=pad5d_shape,
                                       data_format='NCDHW')
        div = paddle.nn.functional.avg_pool3d(
            div, kernel_size=pool3d_shape, stride=1)
        div = paddle.reshape(paddle.squeeze(div, axis=1), sizes)

    div = paddle.scale(div, scale=alpha, bias=k)
    div = paddle.pow(div, beta)
    res = paddle.divide(x, div, name=name)
    return res
