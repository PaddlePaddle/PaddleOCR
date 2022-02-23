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

import numpy as np
from ..fluid.layer_helper import LayerHelper
from ..fluid.data_feeder import check_variable_and_dtype, check_type, check_dtype
from ..fluid.framework import in_dygraph_mode, _varbase_creator, Variable

from ..fluid.layers import transpose, cast  # noqa: F401
from ..fluid import layers
import paddle
from paddle.common_ops_import import core
from paddle.common_ops_import import VarDesc
from paddle import _C_ops
import paddle

__all__ = []


def matmul(x, y, transpose_x=False, transpose_y=False, name=None):
    """
    Applies matrix multiplication to two tensors. `matmul` follows
    the complete broadcast rules,
    and its behavior is consistent with `np.matmul`.

    Currently, the input tensors' number of dimensions can be any, `matmul` can be used to
    achieve the `dot`, `matmul` and `batchmatmul`.

    The actual behavior depends on the shapes of :math:`x`, :math:`y` and the
    flag values of :attr:`transpose_x`, :attr:`transpose_y`. Specifically:

    - If a transpose flag is specified, the last two dimensions of the tensor
      are transposed. If the tensor is ndim-1 of shape, the transpose is invalid. If the tensor
      is ndim-1 of shape :math:`[D]`, then for :math:`x` it is treated as :math:`[1, D]`, whereas
      for :math:`y` it is the opposite: It is treated as :math:`[D, 1]`.

    The multiplication behavior depends on the dimensions of `x` and `y`. Specifically:

    - If both tensors are 1-dimensional, the dot product result is obtained.

    - If both tensors are 2-dimensional, the matrix-matrix product is obtained.

    - If the `x` is 1-dimensional and the `y` is 2-dimensional,
      a `1` is prepended to its dimension in order to conduct the matrix multiply.
      After the matrix multiply, the prepended dimension is removed.

    - If the `x` is 2-dimensional and `y` is 1-dimensional,
      the matrix-vector product is obtained.

    - If both arguments are at least 1-dimensional and at least one argument
      is N-dimensional (where N > 2), then a batched matrix multiply is obtained.
      If the first argument is 1-dimensional, a 1 is prepended to its dimension
      in order to conduct the batched matrix multiply and removed after.
      If the second argument is 1-dimensional, a 1 is appended to its
      dimension for the purpose of the batched matrix multiple and removed after.
      The non-matrix (exclude the last two dimensions) dimensions are
      broadcasted according the broadcast rule.
      For example, if input is a (j, 1, n, m) tensor and the other is a (k, m, p) tensor,
      out will be a (j, k, n, p) tensor.

    Args:
        x (Tensor): The input tensor which is a Tensor.
        y (Tensor): The input tensor which is a Tensor.
        transpose_x (bool): Whether to transpose :math:`x` before multiplication.
        transpose_y (bool): Whether to transpose :math:`y` before multiplication.
        name(str|None): A name for this layer(optional). If set None, the layer
            will be named automatically.

    Returns:
        Tensor: The output Tensor.

    Examples:

    .. code-block:: python

        import paddle
        import numpy as np

        # vector * vector
        x_data = np.random.random([10]).astype(np.float32)
        y_data = np.random.random([10]).astype(np.float32)
        x = paddle.to_tensor(x_data)
        y = paddle.to_tensor(y_data)
        z = paddle.matmul(x, y)
        print(z.numpy().shape)
        # [1]

        # matrix * vector
        x_data = np.random.random([10, 5]).astype(np.float32)
        y_data = np.random.random([5]).astype(np.float32)
        x = paddle.to_tensor(x_data)
        y = paddle.to_tensor(y_data)
        z = paddle.matmul(x, y)
        print(z.numpy().shape)
        # [10]

        # batched matrix * broadcasted vector
        x_data = np.random.random([10, 5, 2]).astype(np.float32)
        y_data = np.random.random([2]).astype(np.float32)
        x = paddle.to_tensor(x_data)
        y = paddle.to_tensor(y_data)
        z = paddle.matmul(x, y)
        print(z.numpy().shape)
        # [10, 5]

        # batched matrix * batched matrix
        x_data = np.random.random([10, 5, 2]).astype(np.float32)
        y_data = np.random.random([10, 2, 5]).astype(np.float32)
        x = paddle.to_tensor(x_data)
        y = paddle.to_tensor(y_data)
        z = paddle.matmul(x, y)
        print(z.numpy().shape)
        # [10, 5, 5]

        # batched matrix * broadcasted matrix
        x_data = np.random.random([10, 1, 5, 2]).astype(np.float32)
        y_data = np.random.random([1, 3, 2, 5]).astype(np.float32)
        x = paddle.to_tensor(x_data)
        y = paddle.to_tensor(y_data)
        z = paddle.matmul(x, y)
        print(z.numpy().shape)
        # [10, 3, 5, 5]

    """
    op_type = 'matmul_v2'
    if in_dygraph_mode():
        op = getattr(_C_ops, op_type)
        return op(x, y, 'trans_x', transpose_x, 'trans_y', transpose_y)

    attrs = {
        'trans_x': transpose_x,
        'trans_y': transpose_y,
    }

    def __check_input(x, y):
        var_names = {'x': x, 'y': y}
        for name, val in var_names.items():
            check_variable_and_dtype(
                val, name, ['float16', 'float32', 'float64'], 'matmul')

    __check_input(x, y)

    helper = LayerHelper('matmul_v2', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type='matmul_v2',
        inputs={'X': x,
                'Y': y},
        outputs={'Out': out},
        attrs=attrs)
    return out


def norm(x, p='fro', axis=None, keepdim=False, name=None):
    """

    Returns the matrix norm (Frobenius) or vector norm (the 1-norm, the Euclidean
    or 2-norm, and in general the p-norm for p > 0) of a given tensor.

    .. note::
        This norm API is different from `numpy.linalg.norm`.
        This api supports high-order input tensors (rank >= 3), and certain axis need to be pointed out to calculate the norm.
        But `numpy.linalg.norm` only supports 1-D vector or 2-D matrix as input tensor.
        For p-order matrix norm, this api actually treats matrix as a flattened vector to calculate the vector norm, NOT REAL MATRIX NORM.

    Args:
        x (Tensor): The input tensor could be N-D tensor, and the input data
            type could be float32 or float64.
        p (float|string, optional): Order of the norm. Supported values are `fro`, `0`, `1`, `2`,
            `inf`, `-inf` and any positive real number yielding the corresponding p-norm. Not supported: ord < 0 and nuclear norm.
            Default value is `fro`.
        axis (int|list|tuple, optional): The axis on which to apply norm operation. If axis is int
            or list(int)/tuple(int)  with only one element, the vector norm is computed over the axis.
            If `axis < 0`, the dimension to norm operation is rank(input) + axis.
            If axis is a list(int)/tuple(int) with two elements, the matrix norm is computed over the axis.
            Defalut value is `None`.
        keepdim (bool, optional): Whether to reserve the reduced dimension in the
            output Tensor. The result tensor will have fewer dimension
            than the :attr:`input` unless :attr:`keepdim` is true, default
            value is False.
        name (str, optional): The default value is None. Normally there is no need for
            user to set this property. For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: results of norm operation on the specified axis of input tensor,
        it's data type is the same as input's Tensor.

    Examples:
        .. code-block:: python

            import paddle
            import numpy as np
            shape=[2, 3, 4]
            np_input = np.arange(24).astype('float32') - 12
            np_input = np_input.reshape(shape)
            x = paddle.to_tensor(np_input)
            #[[[-12. -11. -10.  -9.] [ -8.  -7.  -6.  -5.] [ -4.  -3.  -2.  -1.]]
            # [[  0.   1.   2.   3.] [  4.   5.   6.   7.] [  8.   9.  10.  11.]]]

            # compute frobenius norm along last two dimensions.
            out_fro = paddle.norm(x, p='fro', axis=[0,1])
            # out_fro.numpy() [17.435596 16.911535 16.7332   16.911535]

            # compute 2-order vector norm along last dimension.
            out_pnorm = paddle.norm(x, p=2, axis=-1)
            #out_pnorm.numpy(): [[21.118711  13.190906   5.477226]
            #                    [ 3.7416575 11.224972  19.131126]]

            # compute 2-order  norm along [0,1] dimension.
            out_pnorm = paddle.norm(x, p=2, axis=[0,1])
            #out_pnorm.numpy(): [17.435596 16.911535 16.7332   16.911535]

            # compute inf-order  norm
            out_pnorm = paddle.norm(x, p=np.inf)
            #out_pnorm.numpy()  = [12.]
            out_pnorm = paddle.norm(x, p=np.inf, axis=0)
            #out_pnorm.numpy(): [[12. 11. 10. 9.] [8. 7. 6. 7.] [8. 9. 10. 11.]]

            # compute -inf-order  norm
            out_pnorm = paddle.norm(x, p=-np.inf)
            #out_pnorm.numpy(): [0.]
            out_pnorm = paddle.norm(x, p=-np.inf, axis=0)
            #out_pnorm.numpy(): [[0. 1. 2. 3.] [4. 5. 6. 5.] [4. 3. 2. 1.]]
    """

    def frobenius_norm(input, dim=None, keepdim=False, name=None):
        """
        The frobenius norm OP is to calculate the frobenius norm of certain two dimensions of Tensor `input`.
        Args:
          input (Variable): Tensor, data type float32, float64.
          dim (list, optional): None for last two dimensions.
          keepdim (bool, optional): Whether keep the dimensions as the `input`, Default False.
        """
        if dim is not None and not (isinstance(dim, list) and len(dim) == 2):
            raise ValueError(
                "The dim of frobenius norm op should be None or two elements list!"
            )
        if in_dygraph_mode():
            if dim is None:
                return _C_ops.frobenius_norm(input, 'keep_dim', keepdim,
                                             'reduce_all', True)
            return _C_ops.frobenius_norm(input, 'dim', dim, 'keep_dim', keepdim,
                                         'reduce_all', False)
        attrs = {'dim': dim, 'keep_dim': keepdim, 'reduce_all': False}
        if dim is None:
            attrs['reduce_all'] = True
        check_variable_and_dtype(input, 'input', ['float32', 'float64'],
                                 'frobenius_norm')

        helper = LayerHelper('frobenius_norm', **locals())
        out = helper.create_variable_for_type_inference(
            dtype=helper.input_dtype())

        helper.append_op(
            type='frobenius_norm',
            inputs={'X': input},
            outputs={'Out': out},
            attrs=attrs)
        return out

    def vector_norm(input,
                    porder=None,
                    axis=None,
                    keepdim=False,
                    asvector=False,
                    name=None):
        """
        Calculate the p-order vector norm for certain  dimension of Tensor `input`.
        Args:
          input (Variable): Tensor, data type float32, float64.
          porder (float, optional): None for porder=2.0.
          axis (int, optional): None for last dimension.
          keepdim (bool, optional): Whether keep the dimensions as the `input`, Default False.
        """
        if in_dygraph_mode():
            if axis is None: axis = -1
            return _C_ops.p_norm(input, 'porder', porder, 'axis', axis,
                                 'keepdim', keepdim, 'asvector', asvector)
        if porder is not None:
            check_type(porder, 'porder', (float, int), 'p_norm')
        if axis is not None:
            check_type(axis, 'axis', (int), 'p_norm')
        check_variable_and_dtype(input, 'input', ['float32', 'float64'],
                                 'p_norm')

        attrs = {
            'axis': axis if axis is not None else -1,
            'porder': float(porder) if porder is not None else 2.0,
            'keepdim': keepdim,
            'asvector': asvector,
            'epsilon': 1e-12,
        }
        helper = LayerHelper('p_norm', **locals())
        out = helper.create_variable_for_type_inference(
            dtype=helper.input_dtype())

        helper.append_op(
            type='p_norm',
            inputs={'X': input},
            outputs={'Out': out},
            attrs=attrs)
        return out

    def inf_norm(input,
                 porder=None,
                 axis=axis,
                 keepdim=False,
                 asvector=False,
                 name=None):
        helper = LayerHelper('frobenius_norm', **locals())
        out = helper.create_variable_for_type_inference(
            dtype=helper.input_dtype())
        helper.append_op(type='abs', inputs={'X': input}, outputs={'Out': out})
        reduce_out = helper.create_variable_for_type_inference(
            dtype=helper.input_dtype())

        reduce_all = True if axis == None or axis == [] or asvector == True else False
        axis = axis if axis != None and axis != [] else [0]

        reduce_type = 'reduce_max' if porder == np.float(
            'inf') else 'reduce_min'
        helper.append_op(
            type=reduce_type,
            inputs={'X': out},
            outputs={'Out': reduce_out},
            attrs={'dim': axis,
                   'keep_dim': keepdim,
                   'reduce_all': reduce_all})

        return reduce_out

    def p_matrix_norm(input, porder=1., axis=axis, keepdim=False, name=None):
        """
        NOTE:
            This function actually treats the matrix as flattened vector to calculate vector norm instead of matrix norm.
        """
        block = LayerHelper('norm', **locals())
        out = block.create_variable_for_type_inference(
            dtype=block.input_dtype())
        abs_out = block.create_variable_for_type_inference(
            dtype=block.input_dtype())
        block.append_op(
            type='abs', inputs={'X': input}, outputs={'Out': abs_out})
        pow_out = block.create_variable_for_type_inference(
            dtype=block.input_dtype())

        block.append_op(
            type='pow',
            inputs={'X': abs_out},
            outputs={'Out': pow_out},
            attrs={'factor': porder})
        sum_out = block.create_variable_for_type_inference(
            dtype=block.input_dtype())
        block.append_op(
            type='reduce_sum',
            inputs={'X': pow_out},
            outputs={'Out': sum_out},
            attrs={
                'dim': axis,
                'keep_dim': keepdim,
                'reduce_all': True if axis is None else False
            })
        porder
        block.append_op(
            type='pow',
            inputs={'X': sum_out},
            outputs={'Out': out},
            attrs={'factor': float(1. / porder)})
        return out

    if axis is None and p is not None:
        if isinstance(p, str):
            if p == "fro":
                return frobenius_norm(x, dim=axis, keepdim=keepdim, name=name)
            else:
                raise ValueError(
                    "only valid string values are 'fro', found {}".format(p))
        elif isinstance(p, (int, float)):
            return vector_norm(
                x,
                porder=p,
                axis=axis,
                keepdim=keepdim,
                asvector=True,
                name=name)
        else:
            raise ValueError("only valid p type is string or float, found {}".
                             format(type(p)))

    if isinstance(axis, tuple):
        axis = list(axis)
    if isinstance(axis, list) and len(axis) == 1:
        axis = axis[0]

    #calculate vector norm, where axis is int or list with only one integer
    if isinstance(axis, int):
        if isinstance(p, str):
            if p == "fro":
                return vector_norm(
                    x,
                    porder=2,
                    axis=axis,
                    keepdim=keepdim,
                    asvector=False,
                    name=name)

            else:
                raise ValueError(
                    "only valid string values are 'fro', found {}".format(p))
        elif isinstance(p, (int, float)):
            return vector_norm(
                x,
                axis=axis,
                porder=p,
                keepdim=keepdim,
                asvector=False,
                name=name)
        else:
            raise ValueError(
                "unspport p for p-order vector norm. except float, found {}".
                format(p))
    #calculate matrix norm, where axis is list with two integers
    elif isinstance(axis, list) and len(axis) == 2:
        if p == "fro":
            return frobenius_norm(x, dim=axis, keepdim=keepdim, name=name)
        elif p == np.inf or p == -np.inf:
            return inf_norm(x, porder=p, axis=axis, keepdim=keepdim, name=name)
        elif p == 0:
            raise ValueError(
                "just suport axis type int or list (length of list <=1) if p = 0, found {}".
                format(axis))
        else:
            return p_matrix_norm(
                x, porder=p, axis=axis, keepdim=keepdim, name=name)
    else:
        raise ValueError(
            "except axis type int or list (length of list <=2), found {}".
            format(axis))


def dist(x, y, p=2, name=None):
    r"""

    This OP returns the p-norm of (x - y). It is not a norm in a strict sense, only as a measure
    of distance. The shapes of x and y must be broadcastable. The definition is as follows, for
    details, please refer to the `numpy's broadcasting <https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html>`_:

    - Each input has at least one dimension.
    - Match the two input dimensions from back to front, the dimension sizes must either be equal, one of them is 1, or one of them does not exist.

    Where, z = x - y, the shapes of x and y are broadcastable, then the shape of z can be
    obtained as follows:

    1. If the number of dimensions of x and y are not equal, prepend 1 to the dimensions of the
    tensor with fewer dimensions.

    For example, The shape of x is [8, 1, 6, 1], the shape of y is [7, 1, 5], prepend 1 to the
    dimension of y.

    x (4-D Tensor):  8 x 1 x 6 x 1

    y (4-D Tensor):  1 x 7 x 1 x 5

    2. Determine the size of each dimension of the output z: choose the maximum value from the
    two input dimensions.

    z (4-D Tensor):  8 x 7 x 6 x 5

    If the number of dimensions of the two inputs are the same, the size of the output can be
    directly determined in step 2. When p takes different values, the norm formula is as follows:

    When p = 0, defining $0^0=0$, the zero-norm of z is simply the number of non-zero elements of z.

    .. math::

        ||z||_{0}=\lim_{p \\rightarrow 0}\sum_{i=1}^{m}|z_i|^{p}

    When p = inf, the inf-norm of z is the maximum element of z.

    .. math::

        ||z||_\infty=\max_i |z_i|

    When p = -inf, the negative-inf-norm of z is the minimum element of z.

    .. math::

        ||z||_{-\infty}=\min_i |z_i|

    Otherwise, the p-norm of z follows the formula,

    .. math::

        ||z||_{p}=(\sum_{i=1}^{m}|z_i|^p)^{\\frac{1}{p}}

    Args:
        x (Tensor): 1-D to 6-D Tensor, its data type is float32 or float64.
        y (Tensor): 1-D to 6-D Tensor, its data type is float32 or float64.
        p (float, optional): The norm to be computed, its data type is float32 or float64. Default: 2.

    Returns:
        Tensor: Tensor that is the p-norm of (x - y).

    Examples:
        .. code-block:: python

            import paddle
            import numpy as np

            x = paddle.to_tensor(np.array([[3, 3],[3, 3]]), "float32")
            y = paddle.to_tensor(np.array([[3, 3],[3, 1]]), "float32")
            out = paddle.dist(x, y, 0)
            print(out) # out = [1.]

            out = paddle.dist(x, y, 2)
            print(out) # out = [2.]

            out = paddle.dist(x, y, float("inf"))
            print(out) # out = [2.]

            out = paddle.dist(x, y, float("-inf"))
            print(out) # out = [0.]
    """
    check_variable_and_dtype(x, 'dtype', ['float32', 'float64'], 'dist')
    check_variable_and_dtype(y, 'dtype', ['float32', 'float64'], 'dist')
    check_type(p, 'p', (float, int), 'dist')
    helper = LayerHelper("dist", **locals())
    out = helper.create_variable_for_type_inference(x.dtype)

    inputs = {"X": [x], "Y": [y]}
    outputs = {'Out': [out]}
    attrs = {"p": float(p)}
    helper.append_op(
        type='dist', inputs=inputs, outputs={'Out': out}, attrs=attrs)
    return out


def cond(x, p=None, name=None):
    """

    Computes the condition number of a matrix or batches of matrices with respect to a matrix norm ``p``.

    Args:
        x (Tensor): The input tensor could be tensor of shape ``(*, m, n)`` where ``*`` is zero or more batch dimensions
            for ``p`` in ``(2, -2)``, or of shape ``(*, n, n)`` where every matrix is invertible for any supported ``p``.
            And the input data type could be ``float32`` or ``float64``.
        p (float|string, optional): Order of the norm. Supported values are `fro`, `nuc`, `1`, `-1`, `2`, `-2`,
            `inf`, `-inf`. Default value is `None`, meaning that the order of the norm is `2`.
        name (str, optional): The default value is `None`. Normally there is no need for
            user to set this property. For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: computing results of condition number, its data type is the same as input Tensor ``x``.

    Examples:
        .. code-block:: python

            import paddle
            import numpy as np

            x = paddle.to_tensor([[1., 0, -1], [0, 1, 0], [1, 0, 1]])

            # compute conditional number when p is None
            out = paddle.linalg.cond(x)
            # out.numpy() [1.4142135]

            # compute conditional number when order of the norm is 'fro'
            out_fro = paddle.linalg.cond(x, p='fro')
            # out_fro.numpy() [3.1622777]

            # compute conditional number when order of the norm is 'nuc'
            out_nuc = paddle.linalg.cond(x, p='nuc')
            # out_nuc.numpy() [9.2426405]

            # compute conditional number when order of the norm is 1
            out_1 = paddle.linalg.cond(x, p=1)
            # out_1.numpy() [2.]

            # compute conditional number when order of the norm is -1
            out_minus_1 = paddle.linalg.cond(x, p=-1)
            # out_minus_1.numpy() [1.]

            # compute conditional number when order of the norm is 2
            out_2 = paddle.linalg.cond(x, p=2)
            # out_2.numpy() [1.4142135]

            # compute conditional number when order of the norm is -1
            out_minus_2 = paddle.linalg.cond(x, p=-2)
            # out_minus_2.numpy() [0.70710677]

            # compute conditional number when order of the norm is inf
            out_inf = paddle.linalg.cond(x, p=np.inf)
            # out_inf.numpy() [2.]

            # compute conditional number when order of the norm is -inf
            out_minus_inf = paddle.linalg.cond(x, p=-np.inf)
            # out_minus_inf.numpy() [1.]

            a = paddle.to_tensor(np.random.randn(2, 4, 4).astype('float32'))
            # a.numpy()
            # [[[ 0.14063153 -0.996288    0.7996131  -0.02571543]
            #   [-0.16303636  1.5534962  -0.49919784 -0.04402903]
            #   [-1.1341571  -0.6022629   0.5445269   0.29154757]
            #   [-0.16816919 -0.30972657  1.7521842  -0.5402487 ]]
            #  [[-0.58081484  0.12402827  0.7229862  -0.55046535]
            #   [-0.15178485 -1.1604939   0.75810957  0.30971205]
            #   [-0.9669573   1.0940945  -0.27363303 -0.35416734]
            #   [-1.216529    2.0018666  -0.7773689  -0.17556527]]]
            a_cond_fro = paddle.linalg.cond(a, p='fro')
            # a_cond_fro.numpy()  [31.572273 28.120834]

            b = paddle.to_tensor(np.random.randn(2, 3, 4).astype('float64'))
            # b.numpy()
            # [[[ 1.61707487  0.46829144  0.38130416  0.82546736]
            #   [-1.72710298  0.08866375 -0.62518804  0.16128892]
            #   [-0.02822879 -1.67764516  0.11141444  0.3220113 ]]
            #  [[ 0.22524372  0.62474921 -0.85503233 -1.03960523]
            #   [-0.76620689  0.56673047  0.85064753 -0.45158196]
            #   [ 1.47595418  2.23646462  1.5701758   0.10497519]]]
            b_cond_2 = paddle.linalg.cond(b, p=2)
            # b_cond_2.numpy()  [3.30064451 2.51976252]

    """

    def mat_norm(input, porder=1., axis=None):
        """
        NOTE:
            Calculate the matrix norm of a square matrix or batches of square matrices,
            when porder is in (1, -1, inf, -inf)
        """
        reduce_all = True if axis is None or axis == [] else False
        axis = axis if axis != None and axis != [] else [0]
        keepdim = False

        if in_dygraph_mode():
            abs_out = _C_ops.abs(input)
            sum_out = _C_ops.reduce_sum(abs_out, 'dim', axis, 'keepdim',
                                        keepdim, 'reduce_all', reduce_all)
            if porder == 1 or porder == np.inf:
                return _C_ops.reduce_max(sum_out, 'dim', [-1], 'keepdim',
                                         keepdim, 'reduce_all', reduce_all)
            if porder == -1 or porder == -np.inf:
                return _C_ops.reduce_min(sum_out, 'dim', [-1], 'keepdim',
                                         keepdim, 'reduce_all', reduce_all)

        block = LayerHelper('norm', **locals())
        abs_out = block.create_variable_for_type_inference(
            dtype=block.input_dtype())
        sum_out = block.create_variable_for_type_inference(
            dtype=block.input_dtype())
        out = block.create_variable_for_type_inference(
            dtype=block.input_dtype())
        block.append_op(
            type='abs', inputs={'X': input}, outputs={'Out': abs_out})
        block.append_op(
            type='reduce_sum',
            inputs={'X': abs_out},
            outputs={'Out': sum_out},
            attrs={'dim': axis,
                   'keep_dim': keepdim,
                   'reduce_all': reduce_all})
        if porder == 1 or porder == np.inf:
            block.append_op(
                type='reduce_max',
                inputs={'X': sum_out},
                outputs={'Out': out},
                attrs={
                    'dim': [-1],
                    'keep_dim': keepdim,
                    'reduce_all': reduce_all
                })
        if porder == -1 or porder == -np.inf:
            block.append_op(
                type='reduce_min',
                inputs={'X': sum_out},
                outputs={'Out': out},
                attrs={
                    'dim': [-1],
                    'keep_dim': keepdim,
                    'reduce_all': reduce_all
                })
        return out

    def fro_norm(input, porder=2, axis=[-1]):
        """
        NOTE:
            Calculate the frobenius norm of a square matrix or batches of square matrices.
        """
        reduce_all = True if axis is None or axis == [] else False
        keepdim = False

        if in_dygraph_mode():
            pow_out = _C_ops.pow(input, 'factor', porder)
            sum_out_1 = _C_ops.reduce_sum(pow_out, 'dim', axis, 'keepdim',
                                          keepdim, 'reduce_all', reduce_all)
            sum_out_2 = _C_ops.reduce_sum(sum_out_1, 'dim', axis, 'keepdim',
                                          keepdim, 'reduce_all', reduce_all)
            return _C_ops.pow(sum_out_2, 'factor', float(1. / porder))

        block = LayerHelper('norm', **locals())
        pow_out = block.create_variable_for_type_inference(
            dtype=block.input_dtype())
        sum_out_1 = block.create_variable_for_type_inference(
            dtype=block.input_dtype())
        sum_out_2 = block.create_variable_for_type_inference(
            dtype=block.input_dtype())
        out = block.create_variable_for_type_inference(
            dtype=block.input_dtype())
        block.append_op(
            type='pow',
            inputs={'X': input},
            outputs={'Out': pow_out},
            attrs={'factor': porder})
        block.append_op(
            type='reduce_sum',
            inputs={'X': pow_out},
            outputs={'Out': sum_out_1},
            attrs={'dim': axis,
                   'keep_dim': keepdim,
                   'reduce_all': reduce_all})
        block.append_op(
            type='reduce_sum',
            inputs={'X': sum_out_1},
            outputs={'Out': sum_out_2},
            attrs={'dim': axis,
                   'keep_dim': keepdim,
                   'reduce_all': reduce_all})
        block.append_op(
            type='pow',
            inputs={'X': sum_out_2},
            outputs={'Out': out},
            attrs={'factor': float(1. / porder)})
        return out

    def svd_norm(input, porder, axis=[-1]):
        """
        NOTE:
            Calculate the matrix norm, which is related to singular values, of a matrix
            or batches of matrices, including nuclear norm, 2-norm and (-2)-norm.
        """
        reduce_all = True if axis is None or axis == [] else False
        keepdim = False

        u, s, vh = svd(input, full_matrices=False)

        if in_dygraph_mode():
            if porder == "nuc":
                return _C_ops.reduce_sum(s, 'dim', axis, 'keepdim', keepdim,
                                         'reduce_all', reduce_all)
            max_out = _C_ops.reduce_max(s, 'dim', axis, 'keepdim', keepdim,
                                        'reduce_all', reduce_all)
            min_out = _C_ops.reduce_min(s, 'dim', axis, 'keepdim', keepdim,
                                        'reduce_all', reduce_all)
            if porder == 2:
                return _C_ops.elementwise_div(max_out, min_out, 'aixs', axis,
                                              'use_mkldnn', False)
            if porder == -2:
                return _C_ops.elementwise_div(min_out, max_out, 'aixs', axis,
                                              'use_mkldnn', False)

        block = LayerHelper('norm', **locals())
        out = block.create_variable_for_type_inference(
            dtype=block.input_dtype())
        if porder == "nuc":
            block.append_op(
                type='reduce_sum',
                inputs={'X': s},
                outputs={'Out': out},
                attrs={
                    'dim': axis,
                    'keep_dim': keepdim,
                    'reduce_all': reduce_all
                })
            return out
        max_out = block.create_variable_for_type_inference(
            dtype=block.input_dtype())
        min_out = block.create_variable_for_type_inference(
            dtype=block.input_dtype())
        block.append_op(
            type='reduce_max',
            inputs={'X': s},
            outputs={'Out': max_out},
            attrs={'dim': axis,
                   'keep_dim': keepdim,
                   'reduce_all': reduce_all})
        block.append_op(
            type='reduce_min',
            inputs={'X': s},
            outputs={'Out': min_out},
            attrs={'dim': axis,
                   'keep_dim': keepdim,
                   'reduce_all': reduce_all})
        if porder == 2:
            block.append_op(
                type='elementwise_div',
                inputs={'X': max_out,
                        'Y': min_out},
                outputs={'Out': out},
                attrs={'aixs': axis,
                       'use_mkldnn': False})
            return out
        if porder == -2:
            block.append_op(
                type='elementwise_div',
                inputs={'X': min_out,
                        'Y': max_out},
                outputs={'Out': out},
                attrs={'aixs': axis,
                       'use_mkldnn': False})
            return out

    def empty_tensor(input, shape):
        if in_dygraph_mode():
            return input.reshape(shape)
        raise ValueError("only support x is nonempty tensor in static mode")

    x_shape = list(x.shape)
    if not len(x_shape) >= 2:
        raise ValueError("input should be a matrix or batches of matrices, " +
                         "but the dimention of received input is {}".format(
                             len(x_shape)))
    if p == None:
        p = 2
    x_size = 0 if (0 in x_shape) else 1
    if p in ("fro", "nuc", 1, -1, np.inf, -np.inf):
        if x_shape[len(x_shape) - 1] == x_shape[len(x_shape) - 2]:
            if x_size == 0:
                return empty_tensor(x, x_shape[:-2])
            x_inv = x.inverse()
            if p == "fro":
                return fro_norm(x) * fro_norm(x_inv)
            if p == "nuc":
                return svd_norm(x, p) * svd_norm(x_inv, p)
            if p in (1, -1):
                return mat_norm(
                    x, porder=p, axis=[-2]) * mat_norm(
                        x_inv, porder=p, axis=[-2])
            if p in (np.inf, -np.inf):
                return mat_norm(
                    x, porder=p, axis=[-1]) * mat_norm(
                        x_inv, porder=p, axis=[-1])
        else:
            raise ValueError("only support p is {} when input is a ".format(p) +
                             "square matrix or batches of square matrices")
    elif p in (2, -2):
        if x_size == 0:
            return empty_tensor(x, x_shape[:-2])
        return svd_norm(x, porder=p)
    else:
        raise ValueError(
            "unsupported {} for p, only supporting ('fro', 'nuc', ".format(
                p) + "1, -1, 2, -2, inf, -inf) or none")


def dot(x, y, name=None):
    """
    This operator calculates inner product for vectors.

    .. note::
       Support 1-d and 2-d Tensor. When it is 2d, the first dimension of this matrix
       is the batch dimension, which means that the vectors of multiple batches are dotted.

    Parameters:
        x(Tensor): 1-D or 2-D ``Tensor``. Its dtype should be ``float32``, ``float64``, ``int32``, ``int64``
        y(Tensor): 1-D or 2-D ``Tensor``. Its dtype soulde be ``float32``, ``float64``, ``int32``, ``int64``
        name(str, optional): Name of the output. Default is None. It's used to print debug info for developers. Details: :ref:`api_guide_Name`

    Returns:
        Tensor: the calculated result Tensor.

    Examples:

    .. code-block:: python

        import paddle
        import numpy as np

        x_data = np.random.uniform(0.1, 1, [10]).astype(np.float32)
        y_data = np.random.uniform(1, 3, [10]).astype(np.float32)
        x = paddle.to_tensor(x_data)
        y = paddle.to_tensor(y_data)
        z = paddle.dot(x, y)
        print(z)

    """
    op_type = 'dot'
    # skip var type check in dygraph mode to improve efficiency
    if in_dygraph_mode():
        op = getattr(_C_ops, op_type)
        return op(x, y)

    assert x is not None, 'x cannot be None in {}'.format(op_type)
    assert y is not None, 'y cannot be None in {}'.format(op_type)

    check_variable_and_dtype(x, 'x', ['float32', 'float64', 'int32', 'int64'],
                             op_type)
    check_variable_and_dtype(y, 'y', ['float32', 'float64', 'int32', 'int64'],
                             op_type)

    helper = LayerHelper(op_type, **locals())
    if name is None:
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
    else:
        out = helper.create_variable(
            name=name, dtype=x.dtype, persistable=False)
    helper.append_op(
        type="dot", inputs={'X': x,
                            'Y': y}, attrs={}, outputs={"Out": out})
    return out


def t(input, name=None):
    """
    Transpose <=2-D tensor.
    0-D and 1-D tensors are returned as it is and 2-D tensor is equal to
    the paddle.transpose function which perm dimensions set 0 and 1.

    Args:
        input (Tensor): The input Tensor. It is a N-D (N<=2) Tensor of data types float16, float32, float64, int32.
        name(str, optional): The default value is None.  Normally there is no need for
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`
    Returns:
        Tensor: A transposed n-D Tensor, with data type being float16, float32, float64, int32, int64.

    For Example:

        .. code-block:: text

             # Example 1 (0-D tensor)
             x = tensor([0.79])
             paddle.t(x) = tensor([0.79])

             # Example 2 (1-D tensor)
             x = tensor([0.79, 0.84, 0.32])
             paddle.t(x) = tensor([0.79, 0.84, 0.32])

             # Example 3 (2-D tensor)
             x = tensor([0.79, 0.84, 0.32],
                        [0.64, 0.14, 0.57])
             paddle.t(x) = tensor([0.79, 0.64],
                                  [0.84, 0.14],
                                  [0.32, 0.57])

     Examples:

        .. code-block:: python

            import paddle
            x = paddle.ones(shape=[2, 3], dtype='int32')
            x_transposed = paddle.t(x)
            print(x_transposed.shape)
            # [3, 2]
    """
    if len(input.shape) > 2:
        raise ValueError(
            "Input(input) only support N-D (N<=2) tensor, but received "
            "length of Input(input) is %s. Perhaps you can use paddle."
            "tensor.transpose() instead." % len(input.shape))
    if in_dygraph_mode():
        if len(input.shape) == 1:
            return input
        # 2-D tensor
        perm = [1, 0]
        out, _ = _C_ops.transpose2(input, 'axis', perm)
        return out

    check_variable_and_dtype(
        input, 'input', ['float16', 'float32', 'float64', 'int32',
                         'int64'], 'transpose')

    helper = LayerHelper('t', **locals())
    out = helper.create_variable_for_type_inference(input.dtype)
    input_shape = helper.create_variable_for_type_inference(input.dtype)
    if len(input.shape) == 1:
        out = input
    else:
        helper.append_op(
            type='transpose2',
            inputs={'X': [input]},
            outputs={'Out': [out],
                     'XShape': [input_shape]},
            attrs={'axis': [1, 0]})
    return out


def cross(x, y, axis=None, name=None):
    """
    Computes the cross product between two tensors along an axis.

    Inputs must have the same shape, and the length of their axes should be equal to 3.
    If `axis` is not given, it defaults to the first axis found with the length 3.

    Args:
        x (Tensor): The first input tensor.
        y (Tensor): The second input tensor.
        axis (int, optional): The axis along which to compute the cross product. It defaults to the first axis found with the length 3.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor. A Tensor with same data type as `x`.

    Examples:
        .. code-block:: python

            import paddle

            x = paddle.to_tensor([[1.0, 1.0, 1.0],
                                  [2.0, 2.0, 2.0],
                                  [3.0, 3.0, 3.0]])
            y = paddle.to_tensor([[1.0, 1.0, 1.0],
                                  [1.0, 1.0, 1.0],
                                  [1.0, 1.0, 1.0]])

            z1 = paddle.cross(x, y)
            # [[-1. -1. -1.]
            #  [ 2.  2.  2.]
            #  [-1. -1. -1.]]

            z2 = paddle.cross(x, y, axis=1)
            # [[0. 0. 0.]
            #  [0. 0. 0.]
            #  [0. 0. 0.]]
    """
    if in_dygraph_mode():
        if axis is not None:
            return _C_ops.cross(x, y, 'dim', axis)
        else:
            return _C_ops.cross(x, y)

    helper = LayerHelper("cross", **locals())
    out = helper.create_variable_for_type_inference(x.dtype)
    attrs = dict()
    attrs['dim'] = axis

    helper.append_op(
        type='cross',
        inputs={'X': x,
                'Y': y},
        outputs={'Out': out},
        attrs=attrs)
    return out


def cholesky(x, upper=False, name=None):
    r"""
    Computes the Cholesky decomposition of one symmetric positive-definite
    matrix or batches of symmetric positive-definite matrice.

    If `upper` is `True`, the decomposition has the form :math:`A = U^{T}U` ,
    and the returned matrix :math:`U` is upper-triangular. Otherwise, the
    decomposition has the form  :math:`A = LL^{T}` , and the returned matrix
    :math:`L` is lower-triangular.

    Args:
        x (Tensor): The input tensor. Its shape should be `[*, M, M]`,
            where * is zero or more batch dimensions, and matrices on the
            inner-most 2 dimensions all should be symmetric positive-definite.
            Its data type should be float32 or float64.
        upper (bool): The flag indicating whether to return upper or lower
            triangular matrices. Default: False.

    Returns:
        Tensor: A Tensor with same shape and data type as `x`. It represents \
            triangular matrices generated by Cholesky decomposition.

    Examples:
        .. code-block:: python

            import paddle
            import numpy as np

            a = np.random.rand(3, 3)
            a_t = np.transpose(a, [1, 0])
            x_data = np.matmul(a, a_t) + 1e-03
            x = paddle.to_tensor(x_data)
            out = paddle.cholesky(x, upper=False)
            print(out)
            # [[1.190523   0.         0.        ]
            #  [0.9906703  0.27676893 0.        ]
            #  [1.25450498 0.05600871 0.06400121]]

    """
    if in_dygraph_mode():
        return _C_ops.cholesky(x, "upper", upper)
    check_variable_and_dtype(x, 'dtype', ['float32', 'float64'], 'cholesky')
    check_type(upper, 'upper', bool, 'cholesky')
    helper = LayerHelper('cholesky', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type='cholesky',
        inputs={'X': [x]},
        outputs={'Out': out},
        attrs={'upper': upper})
    return out


def matrix_rank(x, tol=None, hermitian=False, name=None):
    r"""
    Computes the rank of a matrix.

    The rank of a matrix is the number of singular values that are greater than the specified `tol` threshold when hermitian=False,
    or the number of eigenvalues in absolute value that are greater than the specified `tol` threshold when hermitian=True.

    Args:
        x (Tensor): The input tensor. Its shape should be `[..., m, n]`, where `...` is zero or more batch dimensions. If `x` is a batch
            of matrices then the output has the same batch dimensions. The data type of `x` should be float32 or float64.
        tol (float,Tensor,optional): the tolerance value. Default: None. If `tol` is not specified, and `sigma` is the largest
            singular value (or eigenvalues in absolute value), and `eps` is the epsilon value for the dtype of `x`, then `tol` is computed
            with formula `tol=sigma * max(m,n) * eps`. Note that if `x` is a batch of matrices, `tol` is computed this way for every batch.
        hermitian (bool,optional): indicates whether `x` is Hermitian. Default: False. When hermitian=True, `x` is assumed to be Hermitian,
            enabling a more efficient method for finding eigenvalues, but `x` is not checked inside the function. Instead, We just use
            the lower triangular of the matrix to compute.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: Rank of tensor x.

    Examples:
        .. code-block:: python

            import paddle

            a = paddle.eye(10)
            b = paddle.linalg.matrix_rank(a)
            print(b)
            # b = [10]

            c = paddle.ones(shape=[3, 4, 5, 5])
            d = paddle.linalg.matrix_rank(c, tol=0.01, hermitian=True)
            print(d)
            # d = [[1, 1, 1, 1],
            #      [1, 1, 1, 1],
            #      [1, 1, 1, 1]]

    """

    if in_dygraph_mode():
        if tol is None:
            tol_tensor = None
            tol_attr = 0.0
            use_default_tol = True
        elif isinstance(tol, Variable):
            if tol.dtype != x.dtype:
                tol_tensor = cast(tol, x.dtype)
            else:
                tol_tensor = tol
            tol_attr = 0.0
            use_default_tol = False
        else:
            tol_tensor = None
            tol_attr = float(tol)
            use_default_tol = False
        return _C_ops.matrix_rank(x, tol_tensor, "tol", tol_attr, 'hermitian',
                                  hermitian, 'use_default_tol', use_default_tol)

    inputs = {}
    attrs = {}
    check_variable_and_dtype(x, 'x', ['float32', 'float64'], 'matrix_rank')
    inputs['X'] = x
    if tol is None:
        attrs['use_default_tol'] = True
    elif isinstance(tol, Variable):
        check_variable_and_dtype(tol, 'tol', ['float32'], 'matrix_rank')
        attrs['use_default_tol'] = False
        if tol.dtype != x.dtype:
            inputs['TolTensor'] = cast(tol, x.dtype)
        else:
            inputs['TolTensor'] = tol
    else:
        check_type(tol, 'tol', float, 'matrix_rank')
        attrs['use_default_tol'] = False
        attrs['tol'] = tol
    check_type(hermitian, 'hermitian', bool, 'matrix_rank')
    attrs['hermitian'] = hermitian

    helper = LayerHelper('matrix_rank', **locals())
    out = helper.create_variable_for_type_inference(dtype='int32')
    helper.append_op(
        type='matrix_rank', inputs=inputs, outputs={'Out': out}, attrs=attrs)
    return out


def bmm(x, y, name=None):
    """
    Applies batched matrix multiplication to two tensors.

    Both of the two input tensors must be three-dementional and share the same batch size.

    if x is a (b, m, k) tensor, y is a (b, k, n) tensor, the output will be a (b, m, n) tensor.

    Args:
        x (Tensor): The input Tensor.
        y (Tensor): The input Tensor.
        name(str|None): A name for this layer(optional). If set None, the layer
            will be named automatically.

    Returns:
        Tensor: The product Tensor.

    Examples:
        .. code-block:: python

            import paddle

            # In imperative mode:
            # size x: (2, 2, 3) and y: (2, 3, 2)
            x = paddle.to_tensor([[[1.0, 1.0, 1.0],
                                [2.0, 2.0, 2.0]],
                                [[3.0, 3.0, 3.0],
                                [4.0, 4.0, 4.0]]])
            y = paddle.to_tensor([[[1.0, 1.0],[2.0, 2.0],[3.0, 3.0]],
                                [[4.0, 4.0],[5.0, 5.0],[6.0, 6.0]]])
            out = paddle.bmm(x, y)
            #output size: (2, 2, 2)
            #output value:
            #[[[6.0, 6.0],[12.0, 12.0]],[[45.0, 45.0],[60.0, 60.0]]]
            out_np = out.numpy()

    """
    x_shape = x.shape
    y_shape = y.shape
    if not len(x_shape) == len(y_shape) == 3:
        raise ValueError(
            "x and y should be 3-dimensional. But received x's dimention: {}, y's dimention: {}".
            format(x_shape, y_shape))
    if x_shape[2] != y_shape[1]:
        raise ValueError(
            "x's width must be equal with y's height. But received x's shape: {}, y's shape: {}".
            format(x_shape, y_shape))
    if x_shape[0] != y_shape[0]:
        raise ValueError(
            "x's batch (shape[0]) must be equal with y's batch (shape[0]). But received x's shape: {}, y's shape: {}".
            format(x_shape, y_shape))

    if in_dygraph_mode():
        return _C_ops.bmm(x, y)

    helper = LayerHelper('bmm', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(type='bmm', inputs={'X': x, 'Y': y}, outputs={'Out': out})
    return out


def histogram(input, bins=100, min=0, max=0, name=None):
    """
    Computes the histogram of a tensor. The elements are sorted into equal width bins between min and max.
    If min and max are both zero, the minimum and maximum values of the data are used.

    Args:
        input (Tensor): A Tensor(or LoDTensor) with shape :math:`[N_1, N_2,..., N_k]` . The data type of the input Tensor
            should be float32, float64, int32, int64.
        bins (int): number of histogram bins
        min (int): lower end of the range (inclusive)
        max (int): upper end of the range (inclusive)

    Returns:
        Tensor: data type is int64, shape is (nbins,).

    Examples:
        .. code-block:: python

            import paddle

            inputs = paddle.to_tensor([1, 2, 1])
            result = paddle.histogram(inputs, bins=4, min=0, max=3)
            print(result) # [0, 2, 1, 0]
    """
    if in_dygraph_mode():
        return _C_ops.histogram(input, "bins", bins, "min", min, "max", max)

    helper = LayerHelper('histogram', **locals())
    check_variable_and_dtype(
        input, 'X', ['int32', 'int64', 'float32', 'float64'], 'histogram')
    out = helper.create_variable_for_type_inference(VarDesc.VarType.INT64)
    helper.append_op(
        type='histogram',
        inputs={'X': input},
        outputs={'Out': out},
        attrs={'bins': bins,
               'min': min,
               'max': max})
    return out


def bincount(x, weights=None, minlength=0, name=None):
    """
    Computes frequency of each value in the input tensor. 

    Args:
        x (Tensor): A Tensor with non-negative integer. Should be 1-D tensor.
        weights (Tensor, optional): Weight for each value in the input tensor. Should have the same shape as input. Default is None.
        minlength (int, optional): Minimum number of bins. Should be non-negative integer. Default is 0.
        name(str, optional): The default value is None.  Normally there is no need for user to set this
            property.  For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: The tensor of frequency.

    Examples:
        .. code-block:: python

            import paddle

            x = paddle.to_tensor([1, 2, 1, 4, 5])
            result1 = paddle.bincount(x)
            print(result1) # [0, 2, 1, 0, 1, 1]

            w = paddle.to_tensor([2.1, 0.4, 0.1, 0.5, 0.5])
            result2 = paddle.bincount(x, weights=w)
            print(result2) # [0., 2.19999981, 0.40000001, 0., 0.50000000, 0.50000000]
    """
    if x.dtype not in [paddle.int32, paddle.int64]:
        raise TypeError("Elements in Input(x) should all be integers")

    if in_dygraph_mode():
        return _C_ops.bincount(x, weights, "minlength", minlength)

    helper = LayerHelper('bincount', **locals())

    check_variable_and_dtype(x, 'X', ['int32', 'int64'], 'bincount')

    if weights is not None:
        check_variable_and_dtype(weights, 'Weights',
                                 ['int32', 'int64', 'float32', 'float64'],
                                 'bincount')
        out = helper.create_variable_for_type_inference(dtype=weights.dtype)
    else:
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type='bincount',
        inputs={'X': x,
                'Weights': weights},
        outputs={'Out': out},
        attrs={'minlength': minlength})
    return out


def mv(x, vec, name=None):
    """
    Performs a matrix-vector product of the matrix x and the vector vec.

    Args:
        x (Tensor): A tensor with shape :math:`[M, N]` , The data type of the input Tensor x
            should be one of float32, float64.
        vec (Tensor): A tensor with shape :math:`[N]` , The data type of the input Tensor x
            should be one of float32, float64.
        name(str, optional): The default value is None.  Normally there is no need for user to set this
            property.  For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: The tensor which is producted by x and vec.

    Examples:
        .. code-block:: python

            # x: [M, N], vec: [N]
            # paddle.mv(x, vec)  # out: [M]

            import numpy as np
            import paddle

            x_data = np.array([[2, 1, 3], [3, 0, 1]]).astype("float64")
            x = paddle.to_tensor(x_data)
            vec_data = np.array([3, 5, 1])
            vec = paddle.to_tensor(vec_data).astype("float64")
            out = paddle.mv(x, vec)
    """
    if in_dygraph_mode():
        out = _C_ops.mv(x, vec)
        return out

    def __check_input(x, vec):
        var_names = {'x': x, 'vec': vec}
        for name, val in var_names.items():
            check_variable_and_dtype(val, name, ['float32', 'float64'], 'mv')
        x_shape = list(x.shape)
        vec_shape = list(vec.shape)
        if len(x_shape) != 2:
            raise ValueError(
                "x should be 2-dimensional. But received x's dimention: {}".
                format(x_shape))
        if len(vec_shape) != 1:
            raise ValueError(
                "vec should be 1-dimensional. But received vec's dimention: {}".
                format(vec_shape))

    __check_input(x, vec)

    helper = LayerHelper('mv', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type='mv', inputs={'X': x,
                           'Vec': vec}, outputs={'Out': out})
    return out


def det(x, name=None):
    """
    Calculates determinant value of a square matrix or batches of square matrices.
    Args:
        x (Tensor): input (Tensor): the input matrix of size `(n, n)` or the batch of matrices of size
                    `(*, n, n)` where `*` is one or more batch dimensions.
    Returns:
        y (Tensor):the determinant value of a square matrix or batches of square matrices.

    Examples:
        .. code-block:: python

        import paddle

        x =  paddle.randn([3,3,3])

        A = paddle.linalg.det(x)

        print(A)

        # [ 0.02547996,  2.52317095, -6.15900707])


    """
    if in_dygraph_mode():
        return core.ops.determinant(x)

    check_dtype(x.dtype, 'Input', ['float32', 'float64'], 'det')

    input_shape = list(x.shape)
    assert len(input_shape) >= 2,                     \
            "The x must be at least 2-dimensional, "   \
            "but received Input x's dimensional: %s.\n" %  \
            len(input_shape)

    assert (input_shape[-1] == input_shape[-2]),    \
            "Expect squared input," \
            "but received %s by %s matrix.\n" \
            %(input_shape[-2], input_shape[-1]) \

    helper = LayerHelper('determinant', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)

    helper.append_op(
        type='determinant', inputs={'Input': [x]}, outputs={'Out': [out]})
    return out


def slogdet(x, name=None):
    """
    Calculates the sign and natural logarithm of the absolute value of a square matrix's or batches square matrices' determinant.
    The determinant can be computed with ``sign * exp(logabsdet)

    Supports input of float, double

    Note that for matrices that have zero determinant, this returns ``(0, -inf)``
    Args:
        x (Tensor): the batch of matrices of size :math:`(*, n, n)`
            where math:`*` is one or more batch dimensions.

    Returns:
        y (Tensor): A tensor containing the sign of the determinant and the natural logarithm
        of the absolute value of determinant, respectively.

    Examples:
    .. code-block:: python

        import paddle

        x =  paddle.randn([3,3,3])

        A = paddle.linalg.slogdet(x)

        print(A)

        # [[ 1.        ,  1.        , -1.        ],
        # [-0.98610914, -0.43010661, -0.10872950]])

    """
    if in_dygraph_mode():
        return core.ops.slogdeterminant(x)

    check_dtype(x.dtype, 'Input', ['float32', 'float64'], 'slogdet')

    input_shape = list(x.shape)
    assert len(input_shape) >= 2,                     \
            "The x must be at least 2-dimensional, "   \
            "but received Input x's dimensional: %s.\n" %  \
            len(input_shape)

    assert (input_shape[-1] == input_shape[-2]),    \
            "Expect squared input," \
            "but received %s by %s matrix.\n" \
            %(input_shape[-2], input_shape[-1]) \

    helper = LayerHelper('slogdeterminant', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)

    helper.append_op(
        type='slogdeterminant', inputs={'Input': [x]}, outputs={'Out': [out]})
    return out


def svd(x, full_matrices=False, name=None):
    r"""
    Computes the singular value decomposition of one matrix or a batch of regular matrices.

    Let :math:`X` be the input matrix or a batch of input matrices, the output should satisfies:

    .. math::
        X = U * diag(S) * VT

    Args:
        x (Tensor): The input tensor. Its shape should be `[..., N, M]`,
            where `...` is zero or more batch dimensions. N and M can be arbitraty
            positive number. Note that if x is sigular matrices, the grad is numerical
            instable. The data type of x should be float32 or float64.
        full_matrices (bool): A flag to control the behavor of svd.
            If full_matrices = True, svd op will compute full U and V matrics,
            which means shape of U is `[..., N, N]`, shape of V is `[..., M, M]`. K = min(M, N).
            If full_matrices = False, svd op will use a economic method to store U and V.
            which means shape of U is `[..., N, K]`, shape of V is `[..., M, K]`. K = min(M, N).
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tuple of 3 tensors: (U, S, VH). VH is the conjugate transpose of V. S is the singlar value vectors of matrics with shape `[..., K]`

    Examples:
        .. code-block:: python

            import paddle

            x = paddle.to_tensor([[1.0, 2.0], [1.0, 3.0], [4.0, 6.0]]).astype('float64')
            x = x.reshape([3, 2])
            u, s, vh = paddle.linalg.svd(x)
            print (u)
            #U = [[ 0.27364809, -0.21695147  ],
            #      [ 0.37892198, -0.87112408 ],
            #      [ 0.8840446 ,  0.44053933 ]]

            print (s)
            #S = [8.14753743, 0.78589688]
            print (vh)
            #VT= [[ 0.51411221,  0.85772294],
            #     [ 0.85772294, -0.51411221]]

            # one can verify : U * S * VT == X
            #                  U * UH == I
            #                  V * VH == I
    """

    if in_dygraph_mode():
        return _C_ops.svd(x, 'full_matrices', full_matrices)
    check_variable_and_dtype(x, 'dtype', ['float32', 'float64'], 'svd')
    check_type(full_matrices, 'full_matrices', bool, 'svd')
    helper = LayerHelper('svd', **locals())
    u = helper.create_variable_for_type_inference(dtype=x.dtype)
    vh = helper.create_variable_for_type_inference(dtype=x.dtype)
    s = helper.create_variable_for_type_inference(dtype=x.dtype)
    attrs = dict()
    attrs['full_matrices'] = full_matrices
    helper.append_op(
        type='svd',
        inputs={'X': [x]},
        outputs={'U': u,
                 'VH': vh,
                 'S': s},
        attrs=attrs, )
    return u, s, vh


def matrix_power(x, n, name=None):
    r"""
    Computes the n-th power of a square matrix or a batch of square matrices.

    Let :math:`X` be a sqaure matrix or a batch of square matrices, :math:`n` be
    an exponent, the equation should be:

    .. math::
        Out = X ^ {n}

    Specifically,

    - If `n > 0`, it returns the matrix or a batch of matrices raised to the power
    of `n`.

    - If `n = 0`, it returns the identity matrix or a batch of identity matrices.

    - If `n < 0`, it returns the inverse of each matrix (if invertible) raised to
    the power of `abs(n)`.

    Args:
        x (Tensor): A square matrix or a batch of square matrices to be raised
            to power `n`. Its shape should be `[*, M, M]`, where `*` is zero or
            more batch dimensions. Its data type should be float32 or float64.
        n (int): The exponent. It can be any positive, negative integer or zero.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: The n-th power of the matrix (or the batch of matrices) `x`. Its
            data type should be the same as that of `x`.

    Examples:
        .. code-block:: python

            import paddle

            x = paddle.to_tensor([[1, 2, 3],
                                  [1, 4, 9],
                                  [1, 8, 27]], dtype='float64')
            print(paddle.linalg.matrix_power(x, 2))
            # [[6.  , 34. , 102.],
            #  [14. , 90. , 282.],
            #  [36. , 250., 804.]]

            print(paddle.linalg.matrix_power(x, 0))
            # [[1., 0., 0.],
            #  [0., 1., 0.],
            #  [0., 0., 1.]]

            print(paddle.linalg.matrix_power(x, -2))
            # [[ 12.91666667, -12.75000000,  2.83333333 ],
            #  [-7.66666667 ,  8.         , -1.83333333 ],
            #  [ 1.80555556 , -1.91666667 ,  0.44444444 ]]
    """
    if in_dygraph_mode():
        return core.ops.matrix_power(x, "n", n)

    check_variable_and_dtype(x, 'dtype', ['float32', 'float64'], 'matrix_power')
    check_type(n, 'n', int, 'matrix_power')
    helper = LayerHelper('matrix_power', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type='matrix_power',
        inputs={'X': x},
        outputs={'Out': out},
        attrs={'n': n})
    return out


def qr(x, mode="reduced", name=None):
    r"""
    Computes the QR decomposition of one matrix or batches of matrice (backward is unsupported now).

    Args:
        x (Tensor): The input tensor. Its shape should be `[..., M, N]`,
            where ... is zero or more batch dimensions. M and N can be arbitrary
            positive number. The data type of x should be float32 or float64. 
        mode (str, optional): A flag to control the behavior of qr, the default is "reduced". 
            Suppose x's shape is `[..., M, N]` and denoting `K = min(M, N)`:
            If mode = "reduced", qr op will return reduced Q and R matrices, 
            which means Q's shape is `[..., M, K]` and R's shape is `[..., K, N]`.
            If mode = "complete", qr op will return complete Q and R matrices, 
            which means Q's shape is `[..., M, M]` and R's shape is `[..., M, N]`.
            If mode = "r", qr op will only return reduced R matrix, which means
            R's shape is `[..., K, N]`.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.
            
    Returns:
        If mode = "reduced" or mode = "complete", qr will return a two tensor-tuple, which represents Q and R. 
        If mode = "r", qr will return a tensor which represents R.
        
    Examples:            
        .. code-block:: python

            import paddle 

            x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).astype('float64')
            q, r = paddle.linalg.qr(x)
            print (q)
            print (r)

            # Q = [[-0.16903085,  0.89708523],
            #      [-0.50709255,  0.27602622],
            #      [-0.84515425, -0.34503278]])

            # R = [[-5.91607978, -7.43735744],
            #      [ 0.        ,  0.82807867]])
            
            # one can verify : X = Q * R ;     
    """
    if in_dygraph_mode():
        q, r = _C_ops.qr(x, 'mode', mode)
        if mode == "r":
            return r
        else:
            return q, r
    check_variable_and_dtype(x, 'dtype', ['float32', 'float64'], 'qr')
    check_type(mode, 'mode', str, 'qr')
    helper = LayerHelper('qr', **locals())
    q = helper.create_variable_for_type_inference(dtype=x.dtype)
    r = helper.create_variable_for_type_inference(dtype=x.dtype)
    attrs = dict()
    attrs['mode'] = mode
    helper.append_op(
        type='qr', inputs={'X': [x]}, outputs={'Q': q,
                                               'R': r}, attrs=attrs)
    if mode == "r":
        return r
    else:
        return q, r


def eig(x, name=None):
    """
    This API performs the eigenvalue decomposition of a square matrix or a batch of square matrices.

    .. note::
        If the matrix is a Hermitian or a real symmetric matrix, please use :ref:`paddle.linalg.eigh` instead, which is much faster.
        If only eigenvalues is needed, please use :ref:`paddle.linalg.eigvals` instead.
        If the matrix is of any shape, please use :ref:`paddle.linalg.svd`.
        This API is only supported on CPU device.
        The output datatype is always complex for both real and complex input.

    Args:
        x (Tensor): A tensor with shape math:`[*, N, N]`, The data type of the x should be one of ``float32``,
            ``float64``, ``compplex64`` or ``complex128``.
        name (str, optional): The default value is `None`. Normally there is no need for user to set 
            this property. For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Eigenvalues(Tensors): A tensor with shape math:`[*, N]` refers to the eigen values.
        Eigenvectors(Tensors): A tensor with shape math:`[*, N, N]` refers to the eigen vectors.

    Examples:
        .. code-block:: python

            import paddle
            import numpy as np

            paddle.device.set_device("cpu")

            x_data = np.array([[1.6707249, 7.2249975, 6.5045543],
                               [9.956216,  8.749598,  6.066444 ],
                               [4.4251957, 1.7983172, 0.370647 ]]).astype("float32")
            x = paddle.to_tensor(x_data)
            w, v = paddle.linalg.eig(x)
            print(w)
            # Tensor(shape=[3, 3], dtype=complex128, place=CPUPlace, stop_gradient=False,
            #       [[(-0.5061363550800655+0j) , (-0.7971760990842826+0j) ,
            #         (0.18518077798279986+0j)],
            #        [(-0.8308237755993192+0j) ,  (0.3463813401919749+0j) ,
            #         (-0.6837005269141947+0j) ],
            #        [(-0.23142567697893396+0j),  (0.4944999840400175+0j) ,
            #         (0.7058765252952796+0j) ]])

            print(v)
            # Tensor(shape=[3], dtype=complex128, place=CPUPlace, stop_gradient=False,
            #       [ (16.50471283351188+0j)  , (-5.5034820550763515+0j) ,
            #         (-0.21026087843552282+0j)])
    """
    if in_dygraph_mode():
        w, v = _C_ops.eig(x)
        return w, v

    check_variable_and_dtype(
        x, 'X', ['float32', 'float64', 'complex64', 'complex128'], 'eig')
    helper = LayerHelper('eig', **locals())

    w = helper.create_variable_for_type_inference(x.dtype)
    v = helper.create_variable_for_type_inference(x.dtype)

    inputs = {'X': x}
    outputs = {'Eigenvalues': w, 'Eigenvectors': v}
    helper.append_op(type='eig', inputs=inputs, outputs=outputs)

    return w, v


def eigvals(x, name=None):
    """
    Compute the eigenvalues of one or more general matrices.

    Warning:
        The gradient kernel of this operator does not yet developed.
        If you need back propagation through this operator, please replace it with paddle.linalg.eig.

    Args:
        x (Tensor): A square matrix or a batch of square matrices whose eigenvalues will be computed.
            Its shape should be `[*, M, M]`, where `*` is zero or more batch dimensions.
            Its data type should be float32, float64, complex64, or complex128.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.
            
    Returns:
        Tensor: A tensor containing the unsorted eigenvalues which has the same batch dimensions with `x`.
            The eigenvalues are complex-valued even when `x` is real.

    Examples:
        .. code-block:: python

            import paddle

            paddle.set_device("cpu")
            paddle.seed(1234)

            x = paddle.rand(shape=[3, 3], dtype='float64')
            # [[0.02773777, 0.93004224, 0.06911496],
            #  [0.24831591, 0.45733623, 0.07717843],
            #  [0.48016702, 0.14235102, 0.42620817]])

            print(paddle.linalg.eigvals(x))
            # [(-0.27078833542132674+0j), (0.29962280156230725+0j), (0.8824477020120244+0j)] #complex128
    """

    check_variable_and_dtype(x, 'dtype',
                             ['float32', 'float64', 'complex64',
                              'complex128'], 'eigvals')

    x_shape = list(x.shape)
    if len(x_shape) < 2:
        raise ValueError(
            "The dimension of Input(x) should be at least 2, but received x's dimention = {}, x's shape = {}".
            format(len(x_shape), x_shape))

    if x_shape[-1] != x_shape[-2]:
        raise ValueError(
            "The last two dimensions of Input(x) should be equal, but received x's shape = {}".
            format(x_shape))

    if in_dygraph_mode():
        return _C_ops.eigvals(x)

    helper = LayerHelper('eigvals', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(type='eigvals', inputs={'X': x}, outputs={'Out': out})
    return out


def multi_dot(x, name=None):
    """
    Multi_dot is an operator that calculates multiple matrix multiplications.

    Supports inputs of float16(only GPU support), float32 and float64 dtypes. This function does not
    support batched inputs.

    The input tensor in [x] must be 2-D except for the first and last can be 1-D.
    If the first tensor is a 1-D vector of shape(n, ) it is treated as row vector
    of shape(1, n), similarly if the last tensor is a 1D vector of shape(n, ), it
    is treated as a column vector of shape(n, 1).

    If the first and last tensor are 2-D matrix, then the output is also 2-D matrix,
    otherwise the output is a 1-D vector.

    Multi_dot will select the lowest cost multiplication order for calculation. The
    cost of multiplying two matrices with shapes (a, b) and (b, c) is a * b * c.
    Given matrices A, B, C with shapes (20, 5), (5, 100), (100, 10) respectively,
    we can calculate the cost of different multiplication orders as follows:
    - Cost((AB)C) = 20x5x100 + 20x100x10 = 30000
    - Cost(A(BC)) = 5x100x10 + 20x5x10 = 6000

    In this case, multiplying B and C first, then multiply A, which is 5 times faster
    than sequential calculation.

    Args:
        x ([Tensor]): The input tensors which is a list Tensor.
        name(str|None): A name for this layer(optional). If set None, the layer
            will be named automatically.

    Returns:
        Tensor: The output Tensor.


    Examples:

    .. code-block:: python

        import paddle
        import numpy as np

        # A * B
        A_data = np.random.random([3, 4]).astype(np.float32)
        B_data = np.random.random([4, 5]).astype(np.float32)
        A = paddle.to_tensor(A_data)
        B = paddle.to_tensor(B_data)
        out = paddle.linalg.multi_dot([A, B])
        print(out.numpy().shape)
        # [3, 5]

        # A * B * C
        A_data = np.random.random([10, 5]).astype(np.float32)
        B_data = np.random.random([5, 8]).astype(np.float32)
        C_data = np.random.random([8, 7]).astype(np.float32)
        A = paddle.to_tensor(A_data)
        B = paddle.to_tensor(B_data)
        C = paddle.to_tensor(C_data)
        out = paddle.linalg.multi_dot([A, B, C])
        print(out.numpy().shape)
        # [10, 7]

    """
    if in_dygraph_mode():
        return _C_ops.multi_dot(x)

    check_type(x, 'x', (list, tuple), 'multi_dot')
    for id, item in enumerate(x):
        check_variable_and_dtype(item, 'x[' + str(id) + ']',
                                 ['float16', 'float32', 'float64'], 'multi_dot')
        if item.dtype != x[0].dtype:
            raise TypeError(
                "All the Tensors in the input must have the same data type.")

    helper = LayerHelper('multi_dot', **locals())
    dtype = helper.input_dtype(input_param_name='x')
    out = helper.create_variable_for_type_inference(dtype)
    helper.append_op(type='multi_dot', inputs={"X": x}, outputs={"Out": out})
    return out


def eigh(x, UPLO='L', name=None):
    """
    Compute the eigenvalues and eigenvectors of a
    complex Hermitian (conjugate symmetric) or a real symmetric matrix.

    Args:
        x (Tensor): A tensor with shape :math:`[*, N, N]` , The data type of the input Tensor x
            should be one of float32, float64, complex64, complex128.
        UPLO(str, optional): (string, default 'L'), 'L' represents the lower triangular matrix,
                        "'U' represents the upper triangular matrix.".
        name(str, optional): The default value is None.  Normally there is no need for user to set this
            property.  For more information, please refer to :ref:`api_guide_Name`.

    Returns:

        out_value(Tensor):  A Tensor with shape [*, N] and data type of float32 and float64. The eigenvalues of eigh op.
        out_vector(Tensor): A Tensor with shape [*, N, N] and data type of float32,float64,complex64 and complex128. The eigenvectors of eigh op.

    Examples:
        .. code-block:: python

            import numpy as np
            import paddle

            x_data = np.array([[1, -2j], [2j, 5]])
            x = paddle.to_tensor(x_data)
            out_value, out_vector = paddle.linalg.eigh(x, UPLO='L')
            print(out_value)
            #[0.17157288, 5.82842712]
            print(out_vector)
            #[(-0.9238795325112867+0j), (-0.3826834323650898+0j)],
            #[ 0.3826834323650898j    , -0.9238795325112867j    ]]

    """
    if in_dygraph_mode():
        return _C_ops.eigh(x, 'UPLO', UPLO)

    def __check_input(x, UPLO):
        x_shape = list(x.shape)
        if len(x.shape) < 2:
            raise ValueError(
                "Input(input) only support >=2 tensor, but received "
                "length of Input(input) is %s." % len(x.shape))
        if x_shape[-1] != x_shape[-2]:
            raise ValueError(
                "The input matrix must be batches of square matrices. But received x's dimention: {}".
                format(x_shape))
        if UPLO != 'L' and UPLO != 'U':
            raise ValueError(
                "UPLO must be L or U. But received UPLO is: {}".format(UPLO))

    __check_input(x, UPLO)

    helper = LayerHelper('eigh', **locals())
    check_variable_and_dtype(
        x, 'dtype', ['float32', 'float64', 'complex64', 'complex128'], 'eigh')

    out_value = helper.create_variable_for_type_inference(dtype=x.dtype)
    out_vector = helper.create_variable_for_type_inference(dtype=x.dtype)

    helper.append_op(
        type='eigh',
        inputs={'X': x},
        outputs={'Eigenvalues': out_value,
                 'Eigenvectors': out_vector},
        attrs={'UPLO': UPLO})
    return out_value, out_vector


def pinv(x, rcond=1e-15, hermitian=False, name=None):
    r"""
    Calculate pseudo inverse via SVD(singular value decomposition)
    of one matrix or batches of regular matrix.

    .. math::

        if hermitian == False:
            x = u * s * vt  (SVD)
            out = v * 1/s * ut
        else:
            x = u * s * ut  (eigh)
            out = u * 1/s * u.conj().transpose(-2,-1)

    If x is hermitian or symmetric matrix, svd will be replaced with eigh.

    Args:
        x(Tensor): The input tensor. Its shape should be (*, m, n)
            where * is zero or more batch dimensions. m and n can be
            arbitraty positive number. The data type of x should be
            float32 or float64 or complex64 or complex128. When data
            type is complex64 or cpmplex128, hermitian should be set
            True.

        rcond(Tensor, optional): the tolerance value to determine
            when is a singular value zero. Defalut:1e-15.

        hermitian(bool, optional): indicates whether x is Hermitian
            if complex or symmetric if real. Default: False.

        name(str|None): A name for this layer(optional). If set None,
            the layer will be named automatically.

    Returns:
        Tensor: The tensor with same data type with x. it represents
        pseudo inverse of x. Its shape should be (*, n, m).

    Examples:
        .. code-block:: python

            import paddle

            x = paddle.arange(15).reshape((3, 5)).astype('float64')
            input = paddle.to_tensor(x)
            out = paddle.linalg.pinv(input)
            print(input)
            print(out)

            # input:
            # [[0. , 1. , 2. , 3. , 4. ],
            # [5. , 6. , 7. , 8. , 9. ],
            # [10., 11., 12., 13., 14.]]

            # out:
            # [[-0.22666667, -0.06666667,  0.09333333],
            # [-0.12333333, -0.03333333,  0.05666667],
            # [-0.02000000,  0.00000000,  0.02000000],
            # [ 0.08333333,  0.03333333, -0.01666667],
            # [ 0.18666667,  0.06666667, -0.05333333]]

            # one can verify : x * out * x = x ;
            # or              out * x * out = x ;
    """

    if in_dygraph_mode():
        if not hermitian:
            # combine svd and matmul op
            u, s, vt = _C_ops.svd(x, 'full_matrices', False)
            max_singular_val = _C_ops.reduce_max(s, 'dim', [-1], 'keep_dim', True, \
                'reduce_all', False)
            rcond = paddle.to_tensor(rcond, dtype=x.dtype)
            cutoff = rcond * max_singular_val
            y = float('inf')
            y = paddle.to_tensor(y, dtype=x.dtype)

            condition = s > cutoff
            cond_int = layers.cast(condition, s.dtype)
            cond_not_int = layers.cast(layers.logical_not(condition), s.dtype)
            out1 = layers.elementwise_mul(1 / s, cond_int)
            out2 = layers.elementwise_mul(1 / y, cond_not_int)
            singular = layers.elementwise_add(out1, out2)
            st, _ = _C_ops.unsqueeze2(singular, 'axes', [-2])

            dims = list(range(len(vt.shape)))
            perm = dims[:-2] + [dims[-1]] + [dims[-2]]
            v, _ = _C_ops.transpose2(vt, 'axis', perm)

            out_1 = v * st
            out_2 = _C_ops.matmul_v2(out_1, u, 'trans_x', False, 'trans_y',
                                     True)
            return out_2
        else:
            # combine eigh and matmul op
            s, u = _C_ops.eigh(x, 'UPLO', 'L')
            s_abs = paddle.abs(s)
            max_singular_val = _C_ops.reduce_max(s_abs, 'dim', [-1], 'keep_dim', True, \
                'reduce_all', False)
            rcond = paddle.to_tensor(rcond, dtype=s.dtype)
            cutoff = rcond * max_singular_val
            y = float('inf')
            y = paddle.to_tensor(y, dtype=s.dtype)

            condition = s_abs > cutoff
            cond_int = layers.cast(condition, s.dtype)
            cond_not_int = layers.cast(layers.logical_not(condition), s.dtype)
            out1 = layers.elementwise_mul(1 / s, cond_int)
            out2 = layers.elementwise_mul(1 / y, cond_not_int)
            singular = layers.elementwise_add(out1, out2)
            st, _ = _C_ops.unsqueeze2(singular, 'axes', [-2])

            out_1 = u * st
            u_conj = _C_ops.conj(u)
            out_2 = _C_ops.matmul_v2(out_1, u_conj, 'trans_x', False, 'trans_y',
                                     True)
            return out_2
    else:
        if not hermitian:
            helper = LayerHelper('pinv', **locals())
            dtype = x.dtype
            check_variable_and_dtype(x, 'x', ['float32', 'float64'], 'pinv')

            u = helper.create_variable_for_type_inference(dtype)
            s = helper.create_variable_for_type_inference(dtype)
            vt = helper.create_variable_for_type_inference(dtype)
            helper.append_op(
                type='svd',
                inputs={'X': [x]},
                outputs={'U': u,
                         'VH': vt,
                         'S': s},
                attrs={'full_matrices': False}, )

            max_singular_val = helper.create_variable_for_type_inference(dtype)
            helper.append_op(
                type='reduce_max',
                inputs={'X': s},
                outputs={'Out': max_singular_val},
                attrs={'dim': [-1],
                       'keep_dim': True,
                       'reduce_all': False})

            rcond = layers.fill_constant(shape=[1], value=rcond, dtype=dtype)
            cutoff = rcond * max_singular_val
            y = float('inf')
            y = layers.fill_constant(shape=[1], value=y, dtype=dtype)

            condition = s > cutoff
            cond_int = layers.cast(condition, dtype)
            cond_not_int = layers.cast(layers.logical_not(condition), dtype)
            out1 = layers.elementwise_mul(1 / s, cond_int)
            out2 = layers.elementwise_mul(1 / y, cond_not_int)
            singular = layers.elementwise_add(out1, out2)

            st = helper.create_variable_for_type_inference(dtype=dtype)
            st_shape = helper.create_variable_for_type_inference(dtype=dtype)
            helper.append_op(
                type='unsqueeze2',
                inputs={'X': singular},
                attrs={'axes': [-2]},
                outputs={'Out': st,
                         'XShape': st_shape})

            dims = list(range(len(vt.shape)))
            perm = dims[:-2] + [dims[-1]] + [dims[-2]]
            v = helper.create_variable_for_type_inference(dtype)
            v_shape = helper.create_variable_for_type_inference(dtype)
            helper.append_op(
                type='transpose2',
                inputs={'X': [vt]},
                outputs={'Out': [v],
                         'XShape': [v_shape]},
                attrs={'axis': perm})

            out_1 = helper.create_variable_for_type_inference(dtype)
            helper.append_op(
                type='elementwise_mul',
                inputs={'X': v,
                        'Y': st},
                outputs={'Out': out_1},
                attrs={'axis': -1,
                       'use_mkldnn': False})
            out_1 = helper.append_activation(out_1)

            out_2 = helper.create_variable_for_type_inference(dtype)
            helper.append_op(
                type='matmul_v2',
                inputs={'X': out_1,
                        'Y': u},
                outputs={'Out': out_2},
                attrs={'trans_x': False,
                       'trans_y': True}, )
            return out_2
        else:
            helper = LayerHelper('pinv', **locals())
            dtype = x.dtype
            check_variable_and_dtype(
                x, 'dtype', ['float32', 'float64', 'complex64',
                             'complex128'], 'pinv')

            if dtype == paddle.complex128:
                s_type = 'float64'
            elif dtype == paddle.complex64:
                s_type = 'float32'
            else:
                s_type = dtype

            u = helper.create_variable_for_type_inference(dtype)
            s = helper.create_variable_for_type_inference(s_type)
            helper.append_op(
                type='eigh',
                inputs={'X': x},
                outputs={'Eigenvalues': s,
                         'Eigenvectors': u},
                attrs={'UPLO': 'L'})
            s_abs = helper.create_variable_for_type_inference(s_type)
            helper.append_op(
                type='abs', inputs={'X': s}, outputs={'Out': s_abs})
            max_singular_val = helper.create_variable_for_type_inference(s_type)
            helper.append_op(
                type='reduce_max',
                inputs={'X': s_abs},
                outputs={'Out': max_singular_val},
                attrs={'dim': [-1],
                       'keep_dim': True,
                       'reduce_all': False})

            rcond = layers.fill_constant(shape=[1], value=rcond, dtype=s_type)
            cutoff = rcond * max_singular_val
            y = float('inf')
            y = layers.fill_constant(shape=[1], value=y, dtype=s_type)

            condition = s_abs > cutoff
            cond_int = layers.cast(condition, s_type)
            cond_not_int = layers.cast(layers.logical_not(condition), s_type)
            out1 = layers.elementwise_mul(1 / s, cond_int)
            out2 = layers.elementwise_mul(1 / y, cond_not_int)
            singular = layers.elementwise_add(out1, out2)

            st = helper.create_variable_for_type_inference(dtype=s_type)
            st_shape = helper.create_variable_for_type_inference(dtype=s_type)
            helper.append_op(
                type='unsqueeze2',
                inputs={'X': singular},
                attrs={'axes': [-2]},
                outputs={'Out': st,
                         'XShape': st_shape})

            out_1 = helper.create_variable_for_type_inference(dtype)
            helper.append_op(
                type='elementwise_mul',
                inputs={'X': u,
                        'Y': st},
                outputs={'Out': out_1},
                attrs={'axis': -1,
                       'use_mkldnn': False})
            out_1 = helper.append_activation(out_1)

            u_conj = helper.create_variable_for_type_inference(dtype)
            helper.append_op(
                type='conj', inputs={'X': u}, outputs={'Out': [u_conj]})

            out_2 = helper.create_variable_for_type_inference(dtype)
            helper.append_op(
                type='matmul_v2',
                inputs={'X': out_1,
                        'Y': u_conj},
                outputs={'Out': out_2},
                attrs={'trans_x': False,
                       'trans_y': True}, )
            return out_2


def solve(x, y, name=None):
    r"""
    Computes the solution of a square system of linear equations with a unique solution for input 'X' and 'Y'.
    Let :math: `X` be a sqaure matrix or a batch of square matrices, :math:`Y` be
    a vector/matrix or a batch of vectors/matrices, the equation should be:

    .. math::
        Out = X^-1 * Y
    Specifically,
    - This system of linear equations has one solution if and only if input 'X' is invertible.

    Args:
        x (Tensor): A square matrix or a batch of square matrices. Its shape should be `[*, M, M]`, where `*` is zero or
            more batch dimensions. Its data type should be float32 or float64.
        y (Tensor): A vector/matrix or a batch of vectors/matrices. Its shape should be `[*, M, K]`, where `*` is zero or
            more batch dimensions. Its data type should be float32 or float64.
        name(str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: The solution of a square system of linear equations with a unique solution for input 'x' and 'y'.
        Its data type should be the same as that of `x`.

    Examples:
    .. code-block:: python

        # a square system of linear equations:
        # 2*X0 + X1 = 9
        # X0 + 2*X1 = 8

        import paddle
        import numpy as np

        np_x = np.array([[3, 1],[1, 2]])
        np_y = np.array([9, 8])
        x = paddle.to_tensor(np_x, dtype="float64")
        y = paddle.to_tensor(np_y, dtype="float64")
        out = paddle.linalg.solve(x, y)

        print(out)
        # [2., 3.])
    """
    if in_dygraph_mode():
        return _C_ops.solve(x, y)

    inputs = {"X": [x], "Y": [y]}
    helper = LayerHelper("solve", **locals())
    check_variable_and_dtype(x, 'x', ['float32', 'float64'], 'solve')
    check_variable_and_dtype(y, 'y', ['float32', 'float64'], 'solve')
    out = helper.create_variable_for_type_inference(dtype=x.dtype)

    helper.append_op(
        type="solve", inputs={"X": x,
                              "Y": y}, outputs={"Out": out})
    return out


def triangular_solve(x,
                     y,
                     upper=True,
                     transpose=False,
                     unitriangular=False,
                     name=None):
    r"""
    Computes the solution of a system of equations with a triangular coefficient matrix `x` and
    multiple right-hand sides `y` .

    Input `x` and `y` is 2D matrices or batches of 2D matrices. If the inputs are batches, the outputs
    is also batches.

    Args:
        x (Tensor): The input triangular coefficient matrix. Its shape should be `[*, M, M]`, where `*` is zero or
            more batch dimensions. Its data type should be float32 or float64.
        y (Tensor): Multiple right-hand sides of system of equations. Its shape should be `[*, M, K]`, where `*` is 
            zero or more batch dimensions. Its data type should be float32 or float64.
        upper (bool, optional): Whether to solve the upper-triangular system of equations (default) or the lower-triangular 
            system of equations. Default: True.
        transpose (bool, optional): whether `x` should be transposed before calculation. Default: False.
        unitriangular (bool, optional): whether `x` is unit triangular. If True, the diagonal elements of `x` are assumed 
            to be 1 and not referenced from `x` . Default: False.
        name(str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: The solution of the system of equations. Its data type should be the same as that of `x`.

    Examples:
    .. code-block:: python

        # a square system of linear equations:
        # x1 +   x2  +   x3 = 0
        #      2*x2  +   x3 = -9
        #               -x3 = 5

        import paddle
        import numpy as np

        x = paddle.to_tensor([[1, 1, 1], 
                              [0, 2, 1],
                              [0, 0,-1]], dtype="float64")
        y = paddle.to_tensor([[0], [-9], [5]], dtype="float64")
        out = paddle.linalg.triangular_solve(x, y, upper=True)

        print(out)
        # [7, -2, -5]
    """
    if in_dygraph_mode():
        return _C_ops.triangular_solve(x, y, 'upper', upper, 'transpose',
                                       transpose, 'unitriangular',
                                       unitriangular)

    inputs = {"X": [x], "Y": [y]}
    helper = LayerHelper("triangular_solve", **locals())
    check_variable_and_dtype(x, 'x', ['float32', 'float64'], 'triangular_solve')
    check_variable_and_dtype(y, 'y', ['float32', 'float64'], 'triangular_solve')
    out = helper.create_variable_for_type_inference(dtype=x.dtype)

    helper.append_op(
        type='triangular_solve',
        inputs={'X': x,
                'Y': y},
        outputs={'Out': out},
        attrs={
            'upper': upper,
            'transpose': transpose,
            'unitriangular': unitriangular
        })
    return out


def eigvalsh(x, UPLO='L', name=None):
    """
    Computes the eigenvalues of a 
    complex Hermitian (conjugate symmetric) or a real symmetric matrix.

    Args:
        x (Tensor): A tensor with shape :math:`[_, M, M]` , The data type of the input Tensor x
            should be one of float32, float64, complex64, complex128.
        UPLO(str, optional): Lower triangular part of a (‘L’, default) or the upper triangular part (‘U’).
        name(str, optional): The default value is None.  Normally there is no need for user to set this
            property.  For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: The tensor eigenvalues in ascending order.

    Examples:
        .. code-block:: python

            import numpy as np
            import paddle

            x_data = np.array([[1, -2j], [2j, 5]])
            x = paddle.to_tensor(x_data)
            out_value = paddle.eigvalsh(x, UPLO='L')
            print(out_value)
            #[0.17157288, 5.82842712]
    """
    if in_dygraph_mode():
        is_test = x.stop_gradient
        values, _ = _C_ops.eigvalsh(x, 'UPLO', UPLO, 'is_test', is_test)
        return values

    def __check_input(x, UPLO):
        x_shape = list(x.shape)
        if len(x.shape) < 2:
            raise ValueError(
                "Input(input) only support >=2 tensor, but received "
                "length of Input(input) is %s." % len(x.shape))
        if x_shape[-1] != x_shape[-2]:
            raise ValueError(
                "The input matrix must be batches of square matrices. But received x's dimention: {}".
                format(x_shape))
        if UPLO is not 'L' and UPLO is not 'U':
            raise ValueError(
                "UPLO must be L or U. But received UPLO is: {}".format(UPLO))

    __check_input(x, UPLO)

    helper = LayerHelper('eigvalsh', **locals())
    check_variable_and_dtype(x, 'dtype',
                             ['float32', 'float64', 'complex64', 'complex128'],
                             'eigvalsh')

    out_value = helper.create_variable_for_type_inference(dtype=x.dtype)
    out_vector = helper.create_variable_for_type_inference(dtype=x.dtype)

    is_test = x.stop_gradient
    helper.append_op(
        type='eigvalsh',
        inputs={'X': x},
        outputs={'Eigenvalues': out_value,
                 'Eigenvectors': out_vector},
        attrs={'UPLO': UPLO,
               'is_test': is_test})
    return out_value
