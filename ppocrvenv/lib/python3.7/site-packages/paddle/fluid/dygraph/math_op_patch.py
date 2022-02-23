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

from __future__ import print_function

from .. import core
from ..framework import Variable, convert_np_dtype_to_dtype_, _varbase_creator
from ..layers.layer_function_generator import OpProtoHolder
from . import no_grad

import numpy as np
import warnings
from paddle import _C_ops

_supported_int_dtype_ = [
    core.VarDesc.VarType.UINT8,
    core.VarDesc.VarType.INT8,
    core.VarDesc.VarType.INT16,
    core.VarDesc.VarType.INT32,
    core.VarDesc.VarType.INT64,
    core.VarDesc.VarType.BOOL,
]

# NOTE(chenweihang): We currently do not fully support the type promotion 
# between tensors. Parting support here is because the interoperation of 
# real and complex numbers in paddle quantum is very frequent, such as the 
# binary operation between `float` and `complex64`, so we must support the 
# correct type promotion on the APIs paddle quantum used.
# Now only check in dygraph (paddle quantum based dygraph)
# Full type promotion support will need to be fully verified later.
_supported_promote_complex_types_ = [
    '__add__',
    '__radd__',
    '__sub__',
    '__rsub__',
    '__mul__',
    '__rmul__',
    '__div__',
    '__truediv__',
    '__rdiv__',
    '__rtruediv__',
    '__matmul__',
]

_complex_dtypes = [
    core.VarDesc.VarType.COMPLEX64,
    core.VarDesc.VarType.COMPLEX128,
]

_already_patch_varbase = False


def monkey_patch_math_varbase():
    """
    Similar to monkey_patch_variable.
    The difference is, in dygraph mode, use auto-generated op functions for better performance.
    """

    @no_grad
    def create_tensor(value, dtype, shape):
        out = _varbase_creator(dtype=dtype)
        out = _C_ops.fill_constant(out, 'dtype', dtype, 'shape', shape, 'value',
                                   value, 'force_cpu', False)
        out.stop_gradient = True
        return out

    def create_scalar(value, dtype):
        return create_tensor(value, dtype, shape=[1])

    def astype(self, dtype):
        """

        Cast a Tensor to a specified data type.

        Args:
            dtype: The target data type.

        Returns:
            Tensor: a new Tensor with target dtype

        Examples:
            .. code-block:: python

                import paddle
                import numpy as np

                original_tensor = paddle.ones([2, 2])
                print("original tensor's dtype is: {}".format(original_tensor.dtype))
                new_tensor = original_tensor.astype('float32')
                print("new tensor's dtype is: {}".format(new_tensor.dtype))

        """
        if not isinstance(dtype, core.VarDesc.VarType):
            dtype = convert_np_dtype_to_dtype_(dtype)
        return _C_ops.cast(self, 'in_dtype', self.dtype, 'out_dtype', dtype)

    def _scalar_elementwise_op_(var, scale, bias):
        return _C_ops.scale(var, 'scale', scale, 'bias', bias)

    def _neg_(var):
        return _scalar_elementwise_op_(var, -1.0, 0.0)

    def _float_(var):
        numel = np.prod(var.shape)
        assert numel == 1, "only one element variable can be converted to float."
        tensor = var.value().get_tensor()
        assert tensor._is_initialized(), "variable's tensor is not initialized"
        return float(var.numpy().flatten()[0])

    def _long_(var):
        numel = np.prod(var.shape)
        assert numel == 1, "only one element variable can be converted to long."
        tensor = var.value().get_tensor()
        assert tensor._is_initialized(), "variable's tensor is not initialized"
        return int(var.numpy().flatten()[0])

    def _int_(var):
        numel = np.prod(var.shape)
        assert numel == 1, "only one element variable can be converted to int."
        tensor = var.value().get_tensor()
        assert tensor._is_initialized(), "variable's tensor is not initialized"
        return int(var.numpy().flatten()[0])

    def _len_(var):
        if var.type == core.VarDesc.VarType.VOCAB:
            return len(var.value().get_map_tensor())
        elif var.type == core.VarDesc.VarType.STRINGS:
            return len(var.value().get_string_tensor())
        else:
            return var.shape[0]

    def _index_(var):
        numel = np.prod(var.shape)
        assert numel == 1, "only one element variable can be converted to python index."
        tensor = var.value().get_tensor()
        assert tensor._is_initialized(), "variable's tensor is not initialized"
        return int(var.numpy().flatten()[0])

    @property
    def _ndim_(var):
        return len(var.shape)

    @property
    def _size_(var):
        return np.prod(var.shape)

    @property
    def _T_(var):
        if len(var.shape) == 1:
            return var
        perm = []
        for i in range(len(var.shape)):
            perm.insert(0, i)
        out, _ = _C_ops.transpose2(var, 'axis', perm)
        return out

    def _scalar_add_(var, value):
        return _scalar_elementwise_op_(var, 1.0, value)

    def _scalar_sub_(var, value):
        return _scalar_elementwise_op_(var, 1.0, -value)

    def _scalar_rsub_(var, value):
        return _scalar_elementwise_op_(var, -1.0, value)

    def _scalar_mul_(var, value):
        return _scalar_elementwise_op_(var, value, 0.0)

    def _scalar_div_(var, value):
        return _scalar_elementwise_op_(var, 1.0 / value, 0.0)

    # for binary operator such as elementwise, compare
    def _binary_creator_(method_name,
                         op_type,
                         reverse=False,
                         scalar_method=None):
        def __impl__(self, other_var):
            # 1. scalar exists cases
            # we need combine the tensor.dtype and scalar.dtype, cast correct object
            if isinstance(other_var, float):
                # in all cases(+, -, *, /, **, //, %), we need cast tensor.dtype to float
                if self.dtype in _supported_int_dtype_:
                    self = astype(self, 'float32')
                # here use `scale` replace `elementwise` to get better performance
                # but only +, -, *, / can use this method
                if scalar_method is not None:
                    return scalar_method(self, other_var)
            elif isinstance(other_var, int):
                # in all cases(+, -, *, /, **, //, %), we can cast it to float
                # because the output tensor.dtype depend on the type of input tensor
                other_var = float(other_var)
                # division is a special case
                # NOTE(chenweihang): because we cast tensor to float32 instead float64,
                # the division result can only guarantee the numerical accuracy of 6 digits 
                # after the decimal point. The result of numpy calculation is of float64 type, 
                # so the calculation result here and the calculation result of numpy are 
                # different after 6 decimal point. If necessary, we can also use float64 here.
                # torch's behavior here is consistent with ours
                if op_type == 'elementwise_div' and self.dtype in _supported_int_dtype_:
                    self = astype(self, 'float32')
                # here use `scale` replace `elementwise` to get better performance
                # but only +, -, *, / can use this method
                if scalar_method is not None:
                    return scalar_method(self, other_var)
            else:
                # do nothing
                pass

            # 2. create varbase for scalar
            lhs_dtype = self.dtype
            if not isinstance(other_var, core.VarBase):
                if isinstance(other_var, complex):
                    import paddle
                    other_var = paddle.to_tensor(other_var, dtype='complex64')
                else:
                    if reverse:
                        other_var = create_tensor(
                            other_var, dtype=lhs_dtype, shape=self.shape)
                    else:
                        # add fill_op
                        other_var = create_scalar(
                            value=other_var, dtype=lhs_dtype)

            # 3. promote types or unify right var type to left var
            rhs_dtype = other_var.dtype
            if lhs_dtype != rhs_dtype:
                if method_name in _supported_promote_complex_types_ and (
                        lhs_dtype in _complex_dtypes or
                        rhs_dtype in _complex_dtypes):
                    # only when lhs_dtype or rhs_dtype is complex type,
                    # the dtype will promote, in other cases, directly
                    # use lhs_dtype, this is consistent will original rule
                    promote_dtype = core._promote_types_if_complex_exists(
                        lhs_dtype, rhs_dtype)
                    self = self if lhs_dtype == promote_dtype else astype(
                        self, promote_dtype)
                    other_var = other_var if rhs_dtype == promote_dtype else astype(
                        other_var, promote_dtype)
                else:
                    warnings.warn(
                        'The dtype of left and right variables are not the same, left dtype is {}, but right dtype is {}, the right dtype will convert to {}'.
                        format(lhs_dtype, rhs_dtype, lhs_dtype))
                    other_var = astype(other_var, lhs_dtype)

            if reverse:
                tmp = self
                self = other_var
                other_var = tmp

            # 4. calculation
            axis = -1
            math_op = getattr(_C_ops, op_type)
            return math_op(self, other_var, 'axis', axis)

        comment = OpProtoHolder.instance().get_op_proto(op_type).comment

        __impl__.__doc__ = """
        {0}
        Args:
            other_var(Tensor|float|int): right hand Tensor

        Returns:
            Tensor
        """.format(comment)
        __impl__.__name__ = method_name
        return __impl__

    varbase_methods = [
        ('__neg__', _neg_),
        ('__float__', _float_),
        ('__long__', _long_),
        ('__int__', _int_),
        ('__len__', _len_),
        ('__index__', _index_),
        ('astype', astype),
        ('dim', lambda x: len(x.shape)),
        ('ndimension', lambda x: len(x.shape)),
        ('ndim', _ndim_),
        ('size', _size_),
        ('T', _T_),
        ('__add__',
         _binary_creator_('__add__', 'elementwise_add', False, _scalar_add_)),
        ##  a+b == b+a. Do not need to reverse explicitly
        ('__radd__',
         _binary_creator_('__radd__', 'elementwise_add', False, _scalar_add_)),
        ('__sub__', _binary_creator_('__sub__', 'elementwise_sub', False,
                                     _scalar_sub_)),
        ('__rsub__', _binary_creator_('__rsub__', 'elementwise_sub', True,
                                      _scalar_rsub_)),
        ('__mul__', _binary_creator_('__mul__', 'elementwise_mul', False,
                                     _scalar_mul_)),
        ## a*b == b*a. Do not need to reverse explicitly
        ('__rmul__',
         _binary_creator_('__rmul__', 'elementwise_mul', False, _scalar_mul_)),
        ('__div__', _binary_creator_('__div__', 'elementwise_div', False,
                                     _scalar_div_)),
        ('__truediv__', _binary_creator_('__truediv__', 'elementwise_div',
                                         False, _scalar_div_)),
        ('__rdiv__', _binary_creator_('__rdiv__', 'elementwise_div', True,
                                      None)),
        ('__rtruediv__', _binary_creator_('rtruediv__', 'elementwise_div', True,
                                          None)),
        ('__pow__', _binary_creator_('__pow__', 'elementwise_pow', False,
                                     None)),
        ('__rpow__', _binary_creator_('__rpow__', 'elementwise_pow', True,
                                      None)),
        ('__floordiv__', _binary_creator_('__floordiv__',
                                          'elementwise_floordiv', False, None)),
        ('__mod__', _binary_creator_('__mod__', 'elementwise_mod', False,
                                     None)),
        ('__matmul__', _binary_creator_('__matmul__', "matmul_v2", False,
                                        None)),
        ## for logical compare
        ('__eq__', _binary_creator_('__eq__', 'equal', False, None)),
        ('__ne__', _binary_creator_('__ne__', 'not_equal', False, None)),
        ('__lt__', _binary_creator_('__lt__', 'less_than', False, None)),
        ('__le__', _binary_creator_('__le__', 'less_equal', False, None)),
        ('__gt__', _binary_creator_('__gt__', 'greater_than', False, None)),
        ('__ge__', _binary_creator_('__ge__', 'greater_equal', False, None)),
        ('__array_ufunc__', None)
    ]

    global _already_patch_varbase
    if not _already_patch_varbase:
        for method in varbase_methods:
            method_name = method[0]
            method_impl = method[1]
            setattr(core.VarBase, method_name, method_impl)
    else:
        import paddle.tensor
        # Tensor method from module paddle.tensor
        for method_name in paddle.tensor.tensor_method_func:
            if hasattr(core.VarBase, method_name): continue
            method_impl = getattr(paddle.tensor, method_name, None)
            if method_impl: setattr(core.VarBase, method_name, method_impl)

        for magic_method, origin_method in paddle.tensor.magic_method_func:
            impl = getattr(paddle.tensor, origin_method, None)
            if impl: setattr(core.VarBase, magic_method, impl)

    _already_patch_varbase = True
