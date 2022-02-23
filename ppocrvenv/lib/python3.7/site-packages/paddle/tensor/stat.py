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

# TODO: define statistical functions of a tensor  

import numpy as np
from ..fluid.framework import Variable
from ..fluid.layer_helper import LayerHelper
from ..fluid.framework import core, in_dygraph_mode
from ..fluid import layers
from .search import where
from ..fluid.data_feeder import convert_dtype, check_variable_and_dtype, check_type, check_dtype
import paddle
from paddle import _C_ops

__all__ = []


def mean(x, axis=None, keepdim=False, name=None):
    """
    Computes the mean of the input tensor's elements along ``axis``.

    Args:
        x (Tensor): The input Tensor with data type float32, float64.
        axis (int|list|tuple, optional): The axis along which to perform mean
            calculations. ``axis`` should be int, list(int) or tuple(int). If
            ``axis`` is a list/tuple of dimension(s), mean is calculated along
            all element(s) of ``axis`` . ``axis`` or element(s) of ``axis``
            should be in range [-D, D), where D is the dimensions of ``x`` . If
            ``axis`` or element(s) of ``axis`` is less than 0, it works the
            same way as :math:`axis + D` . If ``axis`` is None, mean is
            calculated over all elements of ``x``. Default is None.
        keepdim (bool, optional): Whether to reserve the reduced dimension(s)
            in the output Tensor. If ``keepdim`` is True, the dimensions of
            the output Tensor is the same as ``x`` except in the reduced
            dimensions(it is of size 1 in this case). Otherwise, the shape of
            the output Tensor is squeezed in ``axis`` . Default is False.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, results of average along ``axis`` of ``x``, with the same data
        type as ``x``.

    Examples:
        .. code-block:: python

            import paddle

            x = paddle.to_tensor([[[1., 2., 3., 4.],
                                   [5., 6., 7., 8.],
                                   [9., 10., 11., 12.]],
                                  [[13., 14., 15., 16.],
                                   [17., 18., 19., 20.],
                                   [21., 22., 23., 24.]]])
            out1 = paddle.mean(x)
            # [12.5]
            out2 = paddle.mean(x, axis=-1)
            # [[ 2.5  6.5 10.5]
            #  [14.5 18.5 22.5]]
            out3 = paddle.mean(x, axis=-1, keepdim=True)
            # [[[ 2.5]
            #   [ 6.5]
            #   [10.5]]
            #  [[14.5]
            #   [18.5]
            #   [22.5]]]
            out4 = paddle.mean(x, axis=[0, 2])
            # [ 8.5 12.5 16.5]
    """

    if isinstance(axis, int):
        axis = [axis]
    reduce_all = True if axis is None \
        or len(axis)==0 \
        or len(axis) == len(x.shape) else False
    if axis is None or len(axis) == 0:
        axis = [0]

    if in_dygraph_mode():
        return _C_ops.reduce_mean(x, 'dim', axis, 'keep_dim', keepdim,
                                  'reduce_all', reduce_all)

    check_variable_and_dtype(x, 'x/input', ['uint16', 'float32', 'float64'],
                             'mean/reduce_mean')
    check_type(axis, 'axis/dim', (int, list, tuple), 'mean/reduce_mean')
    if isinstance(axis, (list, tuple)):
        for item in axis:
            check_type(item, 'elements of axis/dim', (int), 'mean/reduce_mean')

    helper = LayerHelper('mean', **locals())
    attrs = {'dim': axis, 'keep_dim': keepdim, 'reduce_all': reduce_all}
    out = helper.create_variable_for_type_inference(x.dtype)
    helper.append_op(
        type='reduce_mean', inputs={'X': x}, outputs={'Out': out}, attrs=attrs)
    return out


def var(x, axis=None, unbiased=True, keepdim=False, name=None):
    """
    Computes the variance of ``x`` along ``axis`` .

    Args:
        x (Tensor): The input Tensor with data type float32, float64.
        axis (int|list|tuple, optional): The axis along which to perform
            variance calculations. ``axis`` should be int, list(int) or
            tuple(int). If ``axis`` is a list/tuple of dimension(s), variance
            is calculated along all element(s) of ``axis`` . ``axis`` or
            element(s) of ``axis`` should be in range [-D, D), where D is the
            dimensions of ``x`` . If ``axis`` or element(s) of ``axis`` is less
            than 0, it works the same way as :math:`axis + D` . If ``axis`` is
            None, variance is calculated over all elements of ``x``. Default
            is None.
        unbiased (bool, optional): Whether to use the unbiased estimation. If
            ``unbiased`` is True, the divisor used in the computation is
            :math:`N - 1`, where :math:`N` represents the number of elements
            along ``axis`` , otherwise the divisor is :math:`N`. Default is True.
        keepdim (bool, optional): Whether to reserve the reduced dimension(s)
            in the output Tensor. If ``keepdim`` is True, the dimensions of
            the output Tensor is the same as ``x`` except in the reduced
            dimensions(it is of size 1 in this case). Otherwise, the shape of
            the output Tensor is squeezed in ``axis`` . Default is False.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, results of variance along ``axis`` of ``x``, with the same data
        type as ``x``.

    Examples:
        .. code-block:: python

            import paddle

            x = paddle.to_tensor([[1.0, 2.0, 3.0], [1.0, 4.0, 5.0]])
            out1 = paddle.var(x)
            # [2.66666667]
            out2 = paddle.var(x, axis=1)
            # [1.         4.33333333]
    """
    if not in_dygraph_mode():
        check_variable_and_dtype(x, 'x', ['float32', 'float64'], 'var')

    u = mean(x, axis, True, name)
    out = paddle.sum((x - u)**2, axis, keepdim=keepdim, name=name)

    n = paddle.cast(paddle.numel(x), x.dtype) \
        / paddle.cast(paddle.numel(out), x.dtype)
    if unbiased:
        one_const = paddle.ones([1], x.dtype)
        n = where(n > one_const, n - 1., one_const)
    out /= n
    return out


def std(x, axis=None, unbiased=True, keepdim=False, name=None):
    """
    Computes the standard-deviation of ``x`` along ``axis`` .

    Args:
        x (Tensor): The input Tensor with data type float32, float64.
        axis (int|list|tuple, optional): The axis along which to perform
            standard-deviation calculations. ``axis`` should be int, list(int)
            or tuple(int). If ``axis`` is a list/tuple of dimension(s),
            standard-deviation is calculated along all element(s) of ``axis`` .
            ``axis`` or element(s) of ``axis`` should be in range [-D, D),
            where D is the dimensions of ``x`` . If ``axis`` or element(s) of
            ``axis`` is less than 0, it works the same way as :math:`axis + D` .
            If ``axis`` is None, standard-deviation is calculated over all
            elements of ``x``. Default is None.
        unbiased (bool, optional): Whether to use the unbiased estimation. If
            ``unbiased`` is True, the standard-deviation is calculated via the
            unbiased estimator. If ``unbiased`` is True,  the divisor used in
            the computation is :math:`N - 1`, where :math:`N` represents the
            number of elements along ``axis`` , otherwise the divisor is
            :math:`N`. Default is True.
        keepdim (bool, optional): Whether to reserve the reduced dimension(s)
            in the output Tensor. If ``keepdim`` is True, the dimensions of
            the output Tensor is the same as ``x`` except in the reduced
            dimensions(it is of size 1 in this case). Otherwise, the shape of
            the output Tensor is squeezed in ``axis`` . Default is False.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, results of standard-deviation along ``axis`` of ``x``, with the
        same data type as ``x``.

    Examples:
        .. code-block:: python

            import paddle

            x = paddle.to_tensor([[1.0, 2.0, 3.0], [1.0, 4.0, 5.0]])
            out1 = paddle.std(x)
            # [1.63299316]
            out2 = paddle.std(x, axis=1)
            # [1.       2.081666]
    """
    if not in_dygraph_mode():
        check_variable_and_dtype(x, 'x', ['float32', 'float64'], 'std')

    out = var(**locals())
    return paddle.sqrt(out)


def numel(x, name=None):
    """
    Returns the number of elements for a tensor, which is a int64 Tensor with shape [1] in static mode
    or a scalar value in imperative mode

    Args:
        x (Tensor): The input Tensor, it's data type can be bool, float16, float32, float64, int32, int64.

    Returns:
        Tensor: The number of elements for the input Tensor.

    Examples:
        .. code-block:: python

            import paddle
            
            x = paddle.full(shape=[4, 5, 7], fill_value=0, dtype='int32')
            numel = paddle.numel(x) # 140


    """
    if in_dygraph_mode():
        return _C_ops.size(x)

    if not isinstance(x, Variable):
        raise TypeError("x must be a Tensor in numel")
    helper = LayerHelper('numel', **locals())
    out = helper.create_variable_for_type_inference(
        dtype=core.VarDesc.VarType.INT64)
    helper.append_op(type='size', inputs={'Input': x}, outputs={'Out': out})
    return out


def median(x, axis=None, keepdim=False, name=None):
    """
    Compute the median along the specified axis.

    Args:
        x (Tensor): The input Tensor, it's data type can be bool, float16, float32, float64, int32, int64.
        axis (int, optional): The axis along which to perform median calculations ``axis`` should be int.
            ``axis`` should be in range [-D, D), where D is the dimensions of ``x`` .
            If ``axis`` is less than 0, it works the same way as :math:`axis + D`.
            If ``axis`` is None, median is calculated over all elements of ``x``. Default is None.
        keepdim (bool, optional): Whether to reserve the reduced dimension(s)
            in the output Tensor. If ``keepdim`` is True, the dimensions of
            the output Tensor is the same as ``x`` except in the reduced
            dimensions(it is of size 1 in this case). Otherwise, the shape of
            the output Tensor is squeezed in ``axis`` . Default is False.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, results of median along ``axis`` of ``x``. If data type of ``x`` is float64, data type of results will be float64, otherwise data type will be float32.

    Examples:
        .. code-block:: python

            import paddle

            x = paddle.arange(12).reshape([3, 4])
            # x is [[0 , 1 , 2 , 3 ],
            #       [4 , 5 , 6 , 7 ],
            #       [8 , 9 , 10, 11]]

            y1 = paddle.median(x)
            # y1 is [5.5]

            y2 = paddle.median(x, axis=0)
            # y2 is [4., 5., 6., 7.]

            y3 = paddle.median(x, axis=1)
            # y3 is [1.5, 5.5, 9.5]

            y4 = paddle.median(x, axis=0, keepdim=True)
            # y4 is [[4., 5., 6., 7.]]

    """
    if not isinstance(x, Variable):
        raise TypeError("In median, the input x should be a Tensor.")
    is_flatten = axis is None
    dims = len(x.shape)
    if is_flatten:
        x = paddle.flatten(x)
        axis = 0
    else:
        if not isinstance(axis, int) or not (axis < dims and axis >= -dims):
            raise ValueError(
                "In median, axis should be none or an integer in range [-rank(x), rank(x))."
            )
        if axis < 0:
            axis += dims
    sz = x.shape[axis]
    kth = sz >> 1
    tensor_topk, idx = paddle.topk(x, kth + 1, axis=axis, largest=False)
    dtype = 'float64' if x.dtype == core.VarDesc.VarType.FP64 else 'float32'
    if sz & 1 == 0:
        out_tensor = paddle.slice(
            tensor_topk, axes=[axis], starts=[kth - 1],
            ends=[kth]) + paddle.slice(
                tensor_topk, axes=[axis], starts=[kth], ends=[kth + 1])
        out_tensor = paddle.cast(out_tensor, dtype=dtype) / 2
    else:
        out_tensor = paddle.cast(
            paddle.slice(
                tensor_topk, axes=[axis], starts=[kth], ends=[kth + 1]),
            dtype=dtype)
    if not keepdim or is_flatten:
        if not is_flatten:
            newshape = x.shape[:axis] + x.shape[axis + 1:]
        elif not keepdim:
            newshape = [1]
        else:
            newshape = [1] * dims
    else:
        newshape = out_tensor.shape
    out_tensor = out_tensor.reshape(newshape, name=name)
    return out_tensor
