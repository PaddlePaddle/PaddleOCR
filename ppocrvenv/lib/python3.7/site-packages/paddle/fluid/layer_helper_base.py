#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import copy
import numpy as np

from .framework import Variable, default_main_program, default_startup_program, in_dygraph_mode, _current_expected_place
from . import unique_name
from .param_attr import ParamAttr, WeightNormParamAttr
from . import core
from .initializer import _global_weight_initializer, _global_bias_initializer

__all__ = ['LayerHelperBase']


class LayerHelperBase(object):
    # global dtype
    __dtype = "float32"

    def __init__(self, name, layer_type):
        self._layer_type = layer_type
        self._name = name

    @property
    def name(self):
        return self._name

    @property
    def layer_type(self):
        return self._layer_type

    @property
    def main_program(self):
        return default_main_program()

    @property
    def startup_program(self):
        return default_startup_program()

    @classmethod
    def set_default_dtype(cls, dtype):
        cls.__dtype = dtype

    @classmethod
    def get_default_dtype(cls):
        return cls.__dtype

    def to_variable(self, value, name=None):
        r"""
        The API will create a ``Variable`` object from numpy\.ndarray or Variable object.

        Parameters:
            value(ndarray): The numpy\.ndarray object that needs to be converted, it can be multi-dimension, and the data type is one of numpy\.{float16, float32, float64, int16, int32, int64, uint8, uint16}.
            name(str, optional): The default value is None. Normally there is no need for user to set this property. For more information, please refer to :ref:`api_guide_Name`

        Returns:
            Variable: ``Tensor`` created from the specified numpy\.ndarray object, data type and shape is the same as ``value`` .

        Examples:

         .. code-block:: python

            import numpy as np
            import paddle.fluid as fluid

            with fluid.dygraph.guard():
                x = np.ones([2, 2], np.float32)
                y = fluid.dygraph.to_variable(x)

        """
        if isinstance(value, np.ndarray):
            assert in_dygraph_mode(
            ), "to_variable could only be called in dygraph mode"
            py_var = core.VarBase(
                value=value,
                name=name if name else '',
                persistable=False,
                place=_current_expected_place(),
                zero_copy=False)
            return py_var
        elif isinstance(value, (core.VarBase, Variable)):
            return value
        else:
            raise TypeError(
                "The type of input value is invalid, expected type is 'ndarray' or 'Variable', but received %s"
                % type(value))

    def _create_weight_normalize(self, attr, shape, dtype):
        from .layers import elementwise_mul, elementwise_div, reshape

        # Remove these ops when LayerHelper and layers support indicating
        # program and block.
        def __norm_op(x,
                      out=None,
                      p=2,
                      dim=None,
                      keep_dim=False,
                      block=self.startup_program.global_block()):
            if out is None:
                out = block.create_var(
                    name=unique_name.generate_with_ignorable_key(".".join(
                        [self.name, 'weight_norm_norm'])),
                    dtype=dtype,
                    persistable=False)
            abs_out = block.create_var(
                name=unique_name.generate_with_ignorable_key(".".join(
                    [self.name, 'weight_norm_abs'])),
                dtype=dtype,
                persistable=False)
            block.append_op(
                type='abs', inputs={'X': x}, outputs={'Out': abs_out})
            pow_out = block.create_var(
                name=unique_name.generate_with_ignorable_key(".".join(
                    [self.name, 'weight_norm_pow'])),
                dtype=dtype,
                persistable=False)
            block.append_op(
                type='pow',
                inputs={'X': abs_out},
                outputs={'Out': pow_out},
                attrs={'factor': float(p)})
            sum_out = block.create_var(
                name=unique_name.generate_with_ignorable_key(".".join(
                    [self.name, 'weight_norm_sum'])),
                dtype=dtype,
                persistable=False)
            block.append_op(
                type='reduce_sum',
                inputs={'X': pow_out},
                outputs={'Out': sum_out},
                attrs={
                    'dim': dim,
                    'keep_dim': keep_dim,
                    'reduce_all': True if dim is None else False
                })
            block.append_op(
                type='pow',
                inputs={'X': sum_out},
                outputs={'Out': out},
                attrs={'factor': 1. / p})
            return out

        def __reshape_op(x,
                         shape,
                         out=None,
                         block=self.startup_program.global_block()):
            if out is None:
                out = block.create_var(
                    name=unique_name.generate_with_ignorable_key(".".join(
                        [self.name, 'weight_norm_reshape'])),
                    dtype=dtype,
                    persistable=False)
            block.append_op(
                type='reshape',
                inputs={'X': x},
                outputs={'Out': out},
                attrs={'shape': shape})
            return out

        def __transpose_op(x,
                           axis,
                           out=None,
                           block=self.startup_program.global_block()):
            if out is None:
                out = block.create_var(
                    name=unique_name.generate_with_ignorable_key(".".join(
                        [self.name, 'weight_norm_transpose'])),
                    dtype=dtype,
                    persistable=False)
            block.append_op(
                type='transpose',
                inputs={'X': x},
                outputs={'Out': out},
                attrs={'axis': axis})
            return out

        def __norm_except_dim(x,
                              out=None,
                              dim=None,
                              block=self.startup_program.global_block()):
            """Computes the norm over all dimensions except dim"""
            if out is None:
                out = block.create_var(
                    name=unique_name.generate_with_ignorable_key(".".join(
                        [self.name, 'weight_norm_norm'])),
                    dtype=dtype,
                    persistable=False)
            if dim is None:
                __norm_op(x, out, dim=dim, block=block)
            elif dim == 0:
                out_shape = [x.shape[0]] + [1] * (len(x.shape) - 1)
                reshape = __reshape_op(x, shape=[x.shape[0], -1], block=block)
                norm = __norm_op(reshape, dim=[1], block=block)
                __reshape_op(norm, out=out, shape=out_shape, block=block)
            elif dim == len(x.shape) - 1:
                out_shape = [1] * (len(x.shape) - 1) + [x.shape[-1]]
                reshape = __reshape_op(x, shape=[-1, x.shape[-1]], block=block)
                norm = __norm_op(reshape, dim=[0], block=block)
                __reshape_op(norm, out=out, shape=out_shape, block=block)
            else:
                perm = list(range(len(x.shape)))
                perm[0], perm[dim] = dim, 0
                transpose = __transpose_op(x, perm, block=block)
                out_shape = [transpose.shape[0]] + [1] * (len(transpose.shape) -
                                                          1)
                reshape = __reshape_op(
                    transpose, shape=[transpose.shape[0], -1], block=block)
                norm = __norm_op(reshape, dim=[1], block=block)
                reshape2 = __reshape_op(norm, shape=out_shape, block=block)
                __transpose_op(reshape2, perm, out=out, block=block)
            return out

        def __weight_normalize(g, v, dim):
            """Calculations for weight normalization"""
            norm = __norm_except_dim(
                v, dim=dim, block=self.main_program.current_block())
            scale = elementwise_div(
                x=g, y=norm)  # The shapes of g and norm are the same.
            # Currently, elementwise_mul only support broadcast when the shape
            # of y is a subset of the shape of x. Thus, we reshape y to squeeze
            # to achive the subset.
            w = elementwise_mul(
                x=v,
                y=scale if dim is None else reshape(
                    x=scale, shape=[v.shape[dim]]),
                axis=-1 if dim is None else dim)
            # To serialize the original parameter for inference, maybe a
            # parameter rather than a variable should be returned.
            return w

        g_param_attr = copy.deepcopy(attr)
        g_param_attr.name = attr.name + '_g'
        g_param_shape = [1] * len(shape)
        if attr.dim is not None:
            g_param_shape[attr.dim] = shape[attr.dim]
        v_param_attr = copy.deepcopy(attr)
        v_param_attr.name = attr.name + '_v'
        v_param_shape = shape

        # Add to startup_program to initialize g and v.
        # Try to reconstruct the initializer of w by initializing g and v.
        # Set the initializers of g and v as below, then the distribution
        # of w is the same as initializing w with the given initializer.
        # For Data-Dependent Initialization, please compute the init-values
        # of g and v in external and then feed the values to g and v by
        # executing an extra program.
        g_param = self.startup_program.global_block().create_parameter(
            dtype=dtype,
            shape=g_param_shape,
            **g_param_attr._to_kwargs(with_initializer=False))
        v_param = self.startup_program.global_block().create_parameter(
            dtype=dtype,
            shape=v_param_shape,
            **v_param_attr._to_kwargs(with_initializer=True))
        __norm_except_dim(
            x=v_param,
            out=g_param,
            dim=attr.dim,
            block=self.startup_program.global_block())

        # keep g_param shape to be consistent with that in main_program
        __reshape_op(
            g_param,
            g_param_shape,
            out=g_param,
            block=self.startup_program.global_block())

        # Add weight normalization to main_program
        g_param = self.main_program.global_block().create_parameter(
            dtype=dtype, shape=g_param_shape, **g_param_attr._to_kwargs())
        v_param = self.main_program.global_block().create_parameter(
            dtype=dtype, shape=v_param_shape, **v_param_attr._to_kwargs())
        w_param = __weight_normalize(g_param, v_param, dim=attr.dim)
        return w_param

    # TODO: hide the func after we move the layers to Layers
    def create_parameter(self,
                         attr,
                         shape,
                         dtype=None,
                         is_bias=False,
                         default_initializer=None,
                         stop_gradient=False,
                         type=core.VarDesc.VarType.LOD_TENSOR):
        """Create parameters for this layers.

           Args:
               attr: [ParamAttr] should be the parameter attribute for this parameter
               shape: shape of the parameter
               dtype: data type of this parameter
               is_bias: if this is a bias parameter
               default_initializer: set the default initializer for this parameter

        Returns created parameter Variable.
        """
        # Deepcopy the attr so that parameters can be shared in program
        attr = copy.deepcopy(attr)
        attr = ParamAttr._to_attr(attr)
        if not attr:
            return None
        assert isinstance(attr, ParamAttr)
        for i, size in enumerate(shape):
            assert size > 0, (
                "Expected every dim's size to be larger than 0, "
                "but the size of the {}-th dim is {}".format(i, size))
        # set global dtype
        if not dtype:
            dtype = self.__dtype
        if is_bias:
            suffix = 'b'
            default_initializer = _global_bias_initializer(
            ) if _global_bias_initializer() is not None else default_initializer
        else:
            suffix = 'w'
            default_initializer = _global_weight_initializer(
            ) if _global_weight_initializer(
            ) is not None else default_initializer
        if attr.name is None:
            attr.name = unique_name.generate(".".join([self.name, suffix]))

        if default_initializer is None and attr.initializer is None:
            if isinstance(dtype, core.VarDesc.VarType):
                if dtype != core.VarDesc.VarType.FP32 and \
                        dtype != core.VarDesc.VarType.FP64 and \
                        dtype != core.VarDesc.VarType.FP16 and \
                        dtype != core.VarDesc.VarType.BF16:
                    raise TypeError(
                        "Can not create parameter with default initializer when dtype is not float type. Set default_initializer to fit the parameter dtype!"
                    )
            else:
                if not (dtype.startswith("float") or
                        dtype in ["double", "uint16"]):
                    raise TypeError(
                        "Can not create parameter with default initializer when dtype is not float type. Set default_initializer to fit the parameter dtype!"
                    )
            if is_bias:
                attr._set_default_bias_initializer()
            else:
                attr._set_default_param_initializer()
        else:
            attr._set_default_initializer(default_initializer)

        # If weight normalization is set, insert extra parameters and ops.
        # Refer to https://arxiv.org/pdf/1602.07868.pdf
        if isinstance(attr, WeightNormParamAttr):
            param = self._create_weight_normalize(attr, shape, dtype)
            WeightNormParamAttr.params_with_weight_norm.append(param)
            return param
        if in_dygraph_mode():
            # In dygraph mode, we want the returned parameter to be
            # initialized so that it can be used imperatively.
            # check parameter name
            is_used = unique_name.dygraph_parameter_name_checker(attr.name)
            if is_used:
                raise ValueError(
                    "parameter name [{}] have be been used. "
                    "In dygraph mode, the name of parameter can't be same."
                    "Please check the parameter attr value passed to self.create_parameter or "
                    "constructor of dygraph Layers".format(attr.name))
            return self.main_program.global_block().create_parameter(
                dtype=dtype,
                shape=shape,
                type=type,
                stop_gradient=stop_gradient,
                **attr._to_kwargs(with_initializer=True))
        else:
            self.startup_program.global_block().create_parameter(
                dtype=dtype,
                shape=shape,
                type=type,
                **attr._to_kwargs(with_initializer=True))
            return self.main_program.global_block().create_parameter(
                dtype=dtype, shape=shape, type=type, **attr._to_kwargs())

    def create_variable_for_type_inference(self,
                                           dtype,
                                           stop_gradient=False,
                                           shape=None):
        """Create a temporary variable that should be type inferred layer.

        Note:
            The default type will be set to LOD_TENSOR. However, when
            the var is used as operator output, its type will be updated
            based on operator's `VarTypeInference` implementation in
            infer_var_type.
        """
        # set global dtype
        if not dtype:
            dtype = self.__dtype
        return self.main_program.current_block().create_var(
            name=unique_name.generate_with_ignorable_key(".".join(
                [self.name, 'tmp'])),
            dtype=dtype,
            shape=shape,
            type=core.VarDesc.VarType.LOD_TENSOR,
            persistable=False,
            stop_gradient=stop_gradient)

    def create_variable(self, *args, **kwargs):
        """Create Variable for this layers.
        Returns created Variable.
        """
        return self.main_program.current_block().create_var(*args, **kwargs)

    def create_global_variable(self, persistable=False, *args, **kwargs):
        """
        create global variable, note that there is no initializer for this global variable.
        Args:
            persistable(bool): True if it is a checkpoint value.
            *args: See create_var's documentation
            **kwargs: See create_var's documentation

        Returns(Variable): the created variable.
        """
        return self.main_program.global_block().create_var(
            *args, persistable=persistable, **kwargs)

    def create_or_get_global_variable(self, name, *args, **kwargs):
        """
        Creates a global variable if not exists and returns the variable and
        a boolean flag which is true when it is a new variable.
        """
        if self.main_program.global_block().has_var(name):
            return self.main_program.global_block().var(name), False
        else:
            return self.create_global_variable(name=name, *args, **kwargs), True

    def set_variable_initializer(self, var, initializer):
        """Set target Variable's initializer

           Args:
               var: target Variable
               initializer: initializer to use
        """
        assert isinstance(var, Variable)
        if in_dygraph_mode():
            initializer(var, self.main_program.global_block())
        else:
            self.startup_program.global_block().create_var(
                name=var.name,
                type=var.type,
                dtype=var.dtype,
                shape=var.shape,
                persistable=True,
                initializer=initializer)
