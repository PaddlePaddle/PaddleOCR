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

import warnings
import inspect

from .. import core
from ..framework import Variable, unique_name, static_only
from .layer_function_generator import OpProtoHolder
from .control_flow import array_write, array_length

_supported_int_dtype_ = [
    core.VarDesc.VarType.BOOL,
    core.VarDesc.VarType.UINT8,
    core.VarDesc.VarType.INT8,
    core.VarDesc.VarType.INT16,
    core.VarDesc.VarType.INT32,
    core.VarDesc.VarType.INT64,
]

compare_ops = ['__eq__', '__ne__', '__lt__', '__le__', '__gt__', '__ge__']

EXPRESSION_MAP = {
    "__add__": "A + B",
    "__radd__": "A += B",
    "__sub__": "A - B",
    "__rsub__": "A -= B",
    "__mul__": "A * B",
    "__rmul__": "A *= B",
    "__div__": "A / B",
    "__truediv__": "A / B",
    "__rdiv__": "A /= B",
    "__rtruediv__": "A /= B",
    "__pow__": "A ** B",
    "__rpow__": "A **= B",
    "__floordiv__": "A //B",
    "__mod__": "A % B",
    "__matmul__": "A @ B",
    "__eq__": "A == B",
    "__ne__": "A != B",
    "__lt__": "A < B",
    "__le__": "A <= B",
    "__gt__": "A > B",
    "__ge__": "A >= B"
}

_already_patch_variable = False


def monkey_patch_variable():
    def unique_tmp_name():
        return unique_name.generate("tmp")

    def safe_get_dtype(var):
        try:
            dtype = var.dtype
        except:
            raise ValueError("Cannot get data type from %s", var.name)
        return dtype

    def current_block(var):
        return var.block.program.current_block()

    def create_new_tmp_var(block, dtype):
        tmp_name = unique_tmp_name()
        return block.create_var(name=tmp_name, dtype=dtype)

    def create_tensor(block, value, dtype, shape):
        value = float(value)
        var = create_new_tmp_var(block, dtype)
        block.append_op(
            type="fill_constant",
            outputs={'Out': [var]},
            attrs={
                'dtype': var.dtype,
                'shape': shape,
                'value': value,
                'force_cpu': False
            },
            stop_gradient=True)
        var.stop_gradient = True
        return var

    def create_scalar(block, value, dtype):
        return create_tensor(block, value, dtype, shape=[1])

    def create_tensor_with_batchsize(ref_var, value, dtype):
        assert isinstance(ref_var, Variable)
        value = float(value)
        block = current_block(ref_var)
        var = create_new_tmp_var(block, dtype)
        batch_dim = -1
        out_shape = []
        for i, d in enumerate(ref_var.shape):
            if d < 0:
                if batch_dim < 0:
                    batch_dim = i
                    out_shape.append(d)
                else:
                    out_shape.append(1)
            else:
                out_shape.append(d)
        assert batch_dim != -1
        block.append_op(
            type='fill_constant_batch_size_like',
            outputs={'Out': [var]},
            inputs={'Input': [ref_var]},
            attrs={
                'shape': out_shape,
                'value': value,
                'input_dim_idx': batch_dim,
                'output_dim_idx': batch_dim
            },
            stop_gradient=True)

        var.stop_gradient = True
        return var

    def astype(self, dtype):
        """
        **Notes**:
            **The variable must be a** :ref:`api_fluid_Tensor`

        Cast a variable to a specified data type.

        Args:

            self(Variable): The source variable

            dtype: The target data type

        Returns:
            Variable: Variable with new dtype

        Examples:
            In Static Graph Mode:

            .. code-block:: python

                import paddle.fluid as fluid

                startup_prog = fluid.Program()
                main_prog = fluid.Program()
                with fluid.program_guard(startup_prog, main_prog):
                    original_variable = fluid.data(name = "new_variable", shape=[2,2], dtype='float32')
                    new_variable = original_variable.astype('int64')
                    print("new var's dtype is: {}".format(new_variable.dtype))

            In Dygraph Mode:

            .. code-block:: python

                import paddle.fluid as fluid
                import numpy as np

                x = np.ones([2, 2], np.float32)
                with fluid.dygraph.guard():
                    original_variable = fluid.dygraph.to_variable(x)
                    print("original var's dtype is: {}, numpy dtype is {}".format(original_variable.dtype, original_variable.numpy().dtype))
                    new_variable = original_variable.astype('int64')
                    print("new var's dtype is: {}, numpy dtype is {}".format(new_variable.dtype, new_variable.numpy().dtype))

        """
        block = current_block(self)
        out = create_new_tmp_var(block, dtype)
        block.append_op(
            type="cast",
            inputs={"X": [self]},
            outputs={"Out": [out]},
            attrs={"in_dtype": self.dtype,
                   "out_dtype": out.dtype})
        out.stop_gradient = self.stop_gradient
        return out

    @static_only
    def append(self, var):
        """
         **Notes**:
            **The type variable must be LoD Tensor Array.
        
        """
        if not isinstance(var, Variable):
            raise TypeError(
                "Required input var should be Variable, but received {}".format(
                    type(var)))
        if self.type != core.VarDesc.VarType.LOD_TENSOR_ARRAY:
            raise TypeError(
                "Only Variable with VarType.LOD_TENSOR_ARRAY support `append` method, but received type: {}".
                format(self.type))

        array_write(x=var, i=array_length(self), array=self)

    def _scalar_op_(var, scale, bias):
        block = current_block(var)
        out = create_new_tmp_var(block, var.dtype)
        block.append_op(
            type="scale",
            inputs={"X": [var]},
            outputs={"Out": [out]},
            attrs={"scale": scale,
                   "bias": bias})
        return out

    def _neg_(var):
        return _scalar_op_(var, -1.0, 0.0)

    @property
    def _ndim_(self):
        """
        Returns the dimension of current Variable

        Returns:
            the dimension

        Examples:
            .. code-block:: python

                import paddle

                paddle.enable_static()

                # create a static Variable
                x = paddle.static.data(name='x', shape=[3, 2, 1])
                # print the dimension of the Variable
                print(x.ndim)
        """
        return len(self.shape)

    def _scalar_add_(var, value):
        return _scalar_op_(var, 1.0, value)

    def _scalar_sub_(var, value):
        return _scalar_op_(var, 1.0, -value)

    def _scalar_rsub_(var, value):
        return _scalar_op_(var, -1.0, value)

    def _scalar_mul_(var, value):
        return _scalar_op_(var, value, 0.0)

    def _scalar_div_(var, value):
        return _scalar_op_(var, 1.0 / value, 0.0)

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

            # 2. create variable for scalar
            lhs_dtype = safe_get_dtype(self)
            if not isinstance(other_var, Variable):
                if reverse:
                    has_batch_size = False
                    for elem in self.shape:
                        if elem < 0:
                            has_batch_size = True
                            break
                    if not has_batch_size:
                        other_var = create_tensor(
                            current_block(self),
                            other_var,
                            dtype=lhs_dtype,
                            shape=self.shape)
                    else:
                        other_var = create_tensor_with_batchsize(
                            self, other_var, lhs_dtype)
                else:
                    # add fill_op to current_block
                    other_var = create_scalar(
                        current_block(self), value=other_var, dtype=lhs_dtype)

            # 3. unify right var type to left var
            rhs_dtype = safe_get_dtype(other_var)
            if lhs_dtype != rhs_dtype:
                other_var = astype(other_var, lhs_dtype)
            if reverse:
                tmp = self
                self = other_var
                other_var = tmp

            # NOTE(zhiqiu): the output of compare operator should be bool.
            if method_name in compare_ops:
                out = create_new_tmp_var(current_block(self), dtype="bool")
            else:
                out = create_new_tmp_var(current_block(self), dtype=lhs_dtype)

            axis = -1
            if other_var.shape[0] == -1:
                stack = inspect.stack()[1]
                file_name = stack[1]
                line_num = stack[2]
                warnings.warn(
                    "%s:%s\nThe behavior of expression %s has been unified with %s(X, Y, axis=-1) from Paddle 2.0. "
                    "If your code works well in the older versions but crashes in this version, try to use "
                    "%s(X, Y, axis=0) instead of %s. This transitional warning will be dropped in the future."
                    % (file_name, line_num, EXPRESSION_MAP[method_name],
                       op_type, op_type, EXPRESSION_MAP[method_name]))
            current_block(self).append_op(
                type=op_type,
                inputs={'X': [self],
                        'Y': [other_var]},
                outputs={'Out': out},
                attrs={'axis': axis})
            return out

        comment = OpProtoHolder.instance().get_op_proto(op_type).comment

        __impl__.__doc__ = """
        {0}
        Args:
            self(Variable): left hand variable
            other_var(Variable|float|int): right hand variable

        Returns:
            Variable
        """.format(comment)
        __impl__.__name__ = method_name
        return __impl__

    variable_methods = [
        #   b=-a
        ('__neg__', _neg_),
        ('astype', astype),
        ('append', append),
        ('dim', lambda x: len(x.shape)),
        ('ndimension', lambda x: len(x.shape)),
        ('ndim', _ndim_),
        ('__add__', _binary_creator_('__add__', 'elementwise_add', False,
                                     _scalar_add_)),
        #  a+b == b+a. Do not need to reverse explicitly
        ('__radd__',
         _binary_creator_('__radd__', 'elementwise_add', False, _scalar_add_)),
        ('__sub__', _binary_creator_('__sub__', 'elementwise_sub', False,
                                     _scalar_sub_)),
        ('__rsub__', _binary_creator_('__rsub__', 'elementwise_sub', True,
                                      _scalar_rsub_)),
        ('__mul__', _binary_creator_('__mul__', 'elementwise_mul', False,
                                     _scalar_mul_)),
        #  a*b == b*a. Do not need to reverse explicitly
        ('__rmul__',
         _binary_creator_('__rmul__', 'elementwise_mul', False, _scalar_mul_)),
        ('__div__', _binary_creator_('__div__', 'elementwise_div', False,
                                     _scalar_div_)),
        ('__truediv__', _binary_creator_('__truediv__', 'elementwise_div',
                                         False, _scalar_div_)),
        ('__rdiv__', _binary_creator_('__rdiv__', 'elementwise_div', True,
                                      None)),
        ('__rtruediv__', _binary_creator_('__rtruediv__', 'elementwise_div',
                                          True, None)),
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
        #  for logical compare
        ('__eq__', _binary_creator_('__eq__', 'equal', False, None)),
        ('__ne__', _binary_creator_('__ne__', 'not_equal', False, None)),
        ('__lt__', _binary_creator_('__lt__', 'less_than', False, None)),
        ('__le__', _binary_creator_('__le__', 'less_equal', False, None)),
        ('__gt__', _binary_creator_('__gt__', 'greater_than', False, None)),
        ('__ge__', _binary_creator_('__ge__', 'greater_equal', False, None))
    ]

    global _already_patch_variable
    if not _already_patch_variable:
        for method in variable_methods:
            method_name = method[0]
            method_impl = method[1]
            setattr(Variable, method_name, method_impl)
    else:
        import paddle.tensor
        for method_name in paddle.tensor.tensor_method_func:
            if hasattr(Variable, method_name): continue
            method_impl = getattr(paddle.tensor, method_name, None)
            if method_impl: setattr(Variable, method_name, method_impl)

        for magic_method, origin_method in paddle.tensor.magic_method_func:
            impl = getattr(paddle.tensor, origin_method, None)
            if impl: setattr(Variable, magic_method, impl)

    _already_patch_variable = True
