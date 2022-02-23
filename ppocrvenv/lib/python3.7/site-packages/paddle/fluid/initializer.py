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

from . import framework
from . import core
from .framework import in_dygraph_mode, default_main_program
import numpy as np
from .core import VarDesc
from . import unique_name
from .data_feeder import check_variable_and_dtype, check_type, check_dtype

__all__ = [
    'Constant', 'Uniform', 'Normal', 'TruncatedNormal', 'Xavier', 'Bilinear',
    'MSRA', 'ConstantInitializer', 'UniformInitializer', 'NormalInitializer',
    'TruncatedNormalInitializer', 'XavierInitializer', 'BilinearInitializer',
    'MSRAInitializer', 'NumpyArrayInitializer', 'set_global_initializer'
]

_global_weight_initializer_ = None
_global_bias_initializer_ = None


class Initializer(object):
    """Base class for variable initializers

    Defines the common interface of variable initializers.
    They add operations to the init program that are used
    to initialize variables. Users should not use this class
    directly, but need to use one of its implementations.
    """

    def __init__(self):
        pass

    def __call__(self, param, block=None):
        """Add corresponding initialization operations to the network
        """
        raise NotImplementedError()

    def _check_block(self, block):
        if block is None:
            block = default_main_program().global_block()

        return block

    def _compute_fans(self, var):
        """Compute the fan_in and the fan_out for layers

        This method computes the fan_in and the fan_out
        for neural network layers, if not specified. It is
        not possible to perfectly estimate fan_in and fan_out.
        This method will estimate it correctly for matrix multiply and
        convolutions.

        Args:
            var: variable for which fan_in and fan_out have to be computed

        Returns:
            tuple of two integers (fan_in, fan_out)
        """
        shape = var.shape
        if not shape or len(shape) == 0:
            fan_in = fan_out = 1
        elif len(shape) == 1:
            fan_in = fan_out = shape[0]
        elif len(shape) == 2:
            # This is the case for simple matrix multiply
            fan_in = shape[0]
            fan_out = shape[1]
        else:
            # Assume this to be a convolutional kernel
            # In PaddlePaddle, the shape of the kernel is like:
            # [num_filters, num_filter_channels, ...] where the remaining
            # dimensions are the filter_size
            receptive_field_size = np.prod(shape[2:])
            fan_in = shape[1] * receptive_field_size
            fan_out = shape[0] * receptive_field_size

        return (fan_in, fan_out)


class ConstantInitializer(Initializer):
    """Implements the constant initializer

    Args:
        value (float32): constant value to initialize the variable 

    Examples:
        .. code-block:: python

            import paddle
            import paddle.fluid as fluid
            paddle.enable_static()
            x = fluid.data(name="data", shape=[8, 32, 32], dtype="float32")
            fc = fluid.layers.fc(
                input=x,
                size=10,
                param_attr=fluid.initializer.Constant(value=2.0))

    """

    def __init__(self, value=0.0, force_cpu=False):
        assert value is not None
        super(ConstantInitializer, self).__init__()
        self._value = value
        self._force_cpu = force_cpu

    def __call__(self, var, block=None):
        """Initialize the input tensor with constant.

        Args:
            var(Tensor): Tensor that needs to be initialized.
            block(Block, optional): The block in which initialization ops
                   should be added. Used in static graph only, default None.

        Returns:
            The initialization op
        """
        block = self._check_block(block)

        assert isinstance(var, framework.Variable)
        assert isinstance(block, framework.Block)

        # to be compatible of fp16 initializers
        if var.dtype == VarDesc.VarType.FP16:
            out_dtype = VarDesc.VarType.FP32
            out_var = block.create_var(
                name=unique_name.generate(".".join(
                    ['constant_init', var.name, 'tmp'])),
                shape=var.shape,
                dtype=out_dtype,
                type=VarDesc.VarType.LOD_TENSOR,
                persistable=False)
        else:
            out_dtype = var.dtype
            out_var = var

        # fill constant should set the "str_value" to preserve precision
        op = block.append_op(
            type="fill_constant",
            outputs={"Out": out_var},
            attrs={
                "shape": var.shape,
                "dtype": int(out_dtype),
                "value": float(self._value),
                'str_value': str(float(self._value)),
                'force_cpu': self._force_cpu
            },
            stop_gradient=True)

        if var.dtype == VarDesc.VarType.FP16:
            block.append_op(
                type="cast",
                inputs={"X": out_var},
                outputs={"Out": var},
                attrs={"in_dtype": out_var.dtype,
                       "out_dtype": var.dtype})

        if not framework.in_dygraph_mode():
            var.op = op
        return op


class UniformInitializer(Initializer):
    """Implements the random uniform distribution initializer

    Args:
        low (float): lower boundary of the uniform distribution
        high (float): upper boundary of the uniform distribution
        seed (int): random seed
        diag_num (int): the number of diagonal elements to initialize.
            If set to 0, diagonal initialization will be not performed.
        diag_step (int): Step size between two diagonal elements,
            which is generally the width of the square matrix.
        diag_val (float): the value of the diagonal element to be initialized,
            default 1.0. It takes effect only if the diag_num is greater than 0.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            x = fluid.data(name='x', shape=[None, 1], dtype='float32')
            fc = fluid.layers.fc(input=x, size=10,
    		param_attr=fluid.initializer.Uniform(low=-0.5, high=0.5))
    """

    def __init__(self,
                 low=-1.0,
                 high=1.0,
                 seed=0,
                 diag_num=0,
                 diag_step=0,
                 diag_val=1.0):
        assert low is not None
        assert high is not None
        assert high >= low
        assert seed is not None
        assert diag_num is not None
        assert diag_step is not None
        assert diag_val is not None
        if diag_num > 0 or diag_step > 0:
            assert (diag_num > 0 and diag_step > 0)
        super(UniformInitializer, self).__init__()
        self._low = low
        self._high = high
        self._seed = seed
        self._diag_num = diag_num
        self._diag_step = diag_step
        self._diag_val = diag_val

    def __call__(self, var, block=None):
        """Initialize the input tensor with Uniform distribution.

        Args:
            var(Tensor): Tensor that needs to be initialized.
            block(Block, optional): The block in which initialization ops
                   should be added. Used in static graph only, default None.

        Returns:
            The initialization op
        """
        block = self._check_block(block)

        assert isinstance(block, framework.Block)
        check_variable_and_dtype(var, "Out",
                                 ["uint16", "float16", "float32", "float64"],
                                 "uniform_random")

        if self._seed == 0:
            self._seed = block.program.random_seed

        # to be compatible of fp16 initializers
        if var.dtype == VarDesc.VarType.FP16:
            out_dtype = VarDesc.VarType.FP32
            out_var = block.create_var(
                name=unique_name.generate(".".join(
                    ['uniform_random', var.name, 'tmp'])),
                shape=var.shape,
                dtype=out_dtype,
                type=VarDesc.VarType.LOD_TENSOR,
                persistable=False)
        else:
            out_dtype = var.dtype
            out_var = var

        op = block.append_op(
            type="uniform_random",
            inputs={},
            outputs={"Out": out_var},
            attrs={
                "shape": var.shape,
                "dtype": out_dtype,
                "min": self._low,
                "max": self._high,
                "seed": self._seed,
                "diag_num": self._diag_num,
                "diag_step": self._diag_step,
                "diag_val": self._diag_val
            },
            stop_gradient=True)

        if var.dtype == VarDesc.VarType.FP16:
            block.append_op(
                type="cast",
                inputs={"X": out_var},
                outputs={"Out": var},
                attrs={"in_dtype": out_var.dtype,
                       "out_dtype": var.dtype})

        if not framework.in_dygraph_mode():
            var.op = op
        return op


class NormalInitializer(Initializer):
    """Implements the Random Normal(Gaussian) distribution initializer

    Args:
        loc (float): mean of the normal distribution
        scale (float): standard deviation of the normal distribution
        seed (int): random seed

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            x = fluid.data(name="data", shape=[None, 32, 32], dtype="float32")
            fc = fluid.layers.fc(input=x, size=10,
                param_attr=fluid.initializer.Normal(loc=0.0, scale=2.0))

    """

    def __init__(self, loc=0.0, scale=1.0, seed=0):
        assert loc is not None
        assert scale is not None
        assert seed is not None
        super(NormalInitializer, self).__init__()
        self._mean = loc
        self._std_dev = scale
        self._seed = seed

    def __call__(self, var, block=None):
        """Initialize the input tensor with Normal distribution.

        Args:
            var(Tensor): Tensor that needs to be initialized.
            block(Block, optional): The block in which initialization ops
                   should be added. Used in static graph only, default None.

        Returns:
            The initialization op
        """
        block = self._check_block(block)

        assert isinstance(block, framework.Block)

        check_variable_and_dtype(var, "Out",
                                 ["uint16", "float16", "float32", "float64"],
                                 "guassian_random")

        if self._seed == 0:
            self._seed = block.program.random_seed

        # to be compatible of fp16 initalizers
        if var.dtype in [VarDesc.VarType.FP16, VarDesc.VarType.BF16]:
            out_dtype = VarDesc.VarType.FP32
            out_var = block.create_var(
                name=unique_name.generate(".".join(
                    ['gaussian_random', var.name, 'tmp'])),
                shape=var.shape,
                dtype=out_dtype,
                type=VarDesc.VarType.LOD_TENSOR,
                persistable=False)
        else:
            out_dtype = var.dtype
            out_var = var

        op = block.append_op(
            type="gaussian_random",
            outputs={"Out": out_var},
            attrs={
                "shape": var.shape,
                "dtype": out_dtype,
                "mean": self._mean,
                "std": self._std_dev,
                "seed": self._seed,
                "use_mkldnn": False
            },
            stop_gradient=True)

        if var.dtype in [VarDesc.VarType.FP16, VarDesc.VarType.BF16]:
            block.append_op(
                type="cast",
                inputs={"X": out_var},
                outputs={"Out": var},
                attrs={"in_dtype": out_var.dtype,
                       "out_dtype": var.dtype})
        if not framework.in_dygraph_mode():
            var.op = op
        return op


class TruncatedNormalInitializer(Initializer):
    """Implements the Random TruncatedNormal(Gaussian) distribution initializer

    Args:
        loc (float): mean of the normal distribution
        scale (float): standard deviation of the normal distribution
        seed (int): random seed

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            x = fluid.data(name='x', shape=[None, 1], dtype='float32')
            fc = fluid.layers.fc(input=x, size=10,
                param_attr=fluid.initializer.TruncatedNormal(loc=0.0, scale=2.0))
    """

    def __init__(self, loc=0.0, scale=1.0, seed=0):
        assert loc is not None
        assert scale is not None
        assert seed is not None
        super(TruncatedNormalInitializer, self).__init__()
        self._mean = loc
        self._std_dev = scale
        self._seed = seed

    def __call__(self, var, block=None):
        """Initialize the input tensor with TruncatedNormal distribution.

        Args:
            var(Tensor): Tensor that needs to be initialized.
            block(Block, optional): The block in which initialization ops
                   should be added. Used in static graph only, default None.

        Returns:
            The initialization op
        """
        block = self._check_block(block)

        assert isinstance(var, framework.Variable)
        assert isinstance(block, framework.Block)

        if self._seed == 0:
            self._seed = block.program.random_seed

        # to be compatible of fp16 initalizers
        if var.dtype in [VarDesc.VarType.FP16, VarDesc.VarType.BF16]:
            out_dtype = VarDesc.VarType.FP32
            out_var = block.create_var(
                name=unique_name.generate(".".join(
                    ['truncated_gaussian_random', var.name, 'tmp'])),
                shape=var.shape,
                dtype=out_dtype,
                type=VarDesc.VarType.LOD_TENSOR,
                persistable=False)
        else:
            out_dtype = var.dtype
            out_var = var

        op = block.append_op(
            type="truncated_gaussian_random",
            outputs={"Out": out_var},
            attrs={
                "shape": var.shape,
                "dtype": out_dtype,
                "mean": self._mean,
                "std": self._std_dev,
                "seed": self._seed
            },
            stop_gradient=True)

        if var.dtype in [VarDesc.VarType.FP16, VarDesc.VarType.BF16]:
            block.append_op(
                type="cast",
                inputs={"X": out_var},
                outputs={"Out": var},
                attrs={"in_dtype": out_var.dtype,
                       "out_dtype": var.dtype})
        if not framework.in_dygraph_mode():
            var.op = op
        return op


class XavierInitializer(Initializer):
    r"""
    This class implements the Xavier weight initializer from the paper
    `Understanding the difficulty of training deep feedforward neural
    networks <http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf>`_
    by Xavier Glorot and Yoshua Bengio.

    This initializer is designed to keep the scale of the gradients
    approximately same in all the layers. In case of Uniform distribution,
    the range is [-x, x], where

    .. math::

        x = \sqrt{\\frac{6.0}{fan\_in + fan\_out}}

    In case of Normal distribution, the mean is 0 and the standard deviation
    is

    .. math::

        \sqrt{\\frac{2.0}{fan\_in + fan\_out}}


    Args:
        uniform (bool,default True): whether to use uniform ,if False use normal distribution
        fan_in (float,default None): fan_in for Xavier initialization. If None, it is
                inferred from the variable.
        fan_out (float,default None): fan_out for Xavier initialization. If None, it is
                 inferred from the variable.
        seed (int): random seed

    Note:
        It is recommended to set fan_in and fan_out to None for most cases.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            queries = fluid.data(name='x', shape=[None,1], dtype='float32')
            fc = fluid.layers.fc(
                input=queries, size=10,
                param_attr=fluid.initializer.Xavier(uniform=False))

    """

    def __init__(self, uniform=True, fan_in=None, fan_out=None, seed=0):
        assert uniform is not None
        assert seed is not None
        super(XavierInitializer, self).__init__()
        self._uniform = uniform
        self._fan_in = fan_in
        self._fan_out = fan_out
        self._seed = seed

    def __call__(self, var, block=None):
        """Initialize the input tensor with Xavier initialization.

        Args:
            var(Tensor): Tensor that needs to be initialized.
            block(Block, optional): The block in which initialization ops
                   should be added. Used in static graph only, default None.

        Returns:
            The initialization op
        """
        block = self._check_block(block)

        assert isinstance(block, framework.Block)
        check_variable_and_dtype(var, "Out",
                                 ["uint16", "float16", "float32", "float64"],
                                 "xavier_init")

        f_in, f_out = self._compute_fans(var)

        # If fan_in and fan_out are passed, use them
        fan_in = f_in if self._fan_in is None else self._fan_in
        fan_out = f_out if self._fan_out is None else self._fan_out

        if self._seed == 0:
            self._seed = block.program.random_seed

        # to be compatible of fp16 initalizers
        if var.dtype == VarDesc.VarType.FP16 or (
                var.dtype == VarDesc.VarType.BF16 and not self._uniform):
            out_dtype = VarDesc.VarType.FP32
            out_var = block.create_var(
                name=unique_name.generate(".".join(
                    ['xavier_init', var.name, 'tmp'])),
                shape=var.shape,
                dtype=out_dtype,
                type=VarDesc.VarType.LOD_TENSOR,
                persistable=False)
        else:
            out_dtype = var.dtype
            out_var = var

        if self._uniform:
            limit = np.sqrt(6.0 / float(fan_in + fan_out))
            op = block.append_op(
                type="uniform_random",
                inputs={},
                outputs={"Out": out_var},
                attrs={
                    "shape": out_var.shape,
                    "dtype": out_dtype,
                    "min": -limit,
                    "max": limit,
                    "seed": self._seed
                },
                stop_gradient=True)

        else:
            std = np.sqrt(2.0 / float(fan_in + fan_out))
            op = block.append_op(
                type="gaussian_random",
                outputs={"Out": out_var},
                attrs={
                    "shape": out_var.shape,
                    "dtype": out_dtype,
                    "mean": 0.0,
                    "std": std,
                    "seed": self._seed
                },
                stop_gradient=True)

        if var.dtype == VarDesc.VarType.FP16 or (
                var.dtype == VarDesc.VarType.BF16 and not self._uniform):
            block.append_op(
                type="cast",
                inputs={"X": out_var},
                outputs={"Out": var},
                attrs={"in_dtype": out_var.dtype,
                       "out_dtype": var.dtype})

        if not framework.in_dygraph_mode():
            var.op = op
        return op


class MSRAInitializer(Initializer):
    r"""Implements the MSRA initializer a.k.a. Kaiming Initializer

    This class implements the weight initialization from the paper
    `Delving Deep into Rectifiers: Surpassing Human-Level Performance on
    ImageNet Classification <https://arxiv.org/abs/1502.01852>`_
    by Kaiming He, Xiangyu Zhang, Shaoqing Ren and Jian Sun. This is a
    robust initialization method that particularly considers the rectifier
    nonlinearities. In case of Uniform distribution, the range is [-x, x], where

    .. math::

        x = \sqrt{\\frac{6.0}{fan\_in}}

    In case of Normal distribution, the mean is 0 and the standard deviation
    is

    .. math::

        \sqrt{\\frac{2.0}{fan\_in}}

    Args:
        uniform (bool): whether to use uniform or normal distribution
        fan_in (float32|None): fan_in for MSRAInitializer. If None, it is\
        inferred from the variable. default is None.
        seed (int32): random seed

    Note:
        It is recommended to set fan_in to None for most cases.

    Examples:
        .. code-block:: python

            import paddle
            import paddle.fluid as fluid
            paddle.enable_static()
            x = fluid.data(name="data", shape=[8, 32, 32], dtype="float32")
            fc = fluid.layers.fc(input=x, size=10,
                param_attr=fluid.initializer.MSRA(uniform=False))

    """

    def __init__(self, uniform=True, fan_in=None, seed=0):
        """Constructor for MSRAInitializer
        """
        assert uniform is not None
        assert seed is not None
        super(MSRAInitializer, self).__init__()
        self._uniform = uniform
        self._fan_in = fan_in
        self._seed = seed

    def __call__(self, var, block=None):
        """Initialize the input tensor with MSRA initialization.

        Args:
            var(Tensor): Tensor that needs to be initialized.
            block(Block, optional): The block in which initialization ops
                   should be added. Used in static graph only, default None.

        Returns:
            The initialization op
        """
        block = self._check_block(block)

        assert isinstance(var, framework.Variable)
        assert isinstance(block, framework.Block)
        f_in, f_out = self._compute_fans(var)

        # If fan_in is passed, use it
        fan_in = f_in if self._fan_in is None else self._fan_in

        if self._seed == 0:
            self._seed = block.program.random_seed

        # to be compatible of fp16 initalizers
        if var.dtype == VarDesc.VarType.FP16 or (
                var.dtype == VarDesc.VarType.BF16 and not self._uniform):
            out_dtype = VarDesc.VarType.FP32
            out_var = block.create_var(
                name=unique_name.generate(".".join(
                    ['masra_init', var.name, 'tmp'])),
                shape=var.shape,
                dtype=out_dtype,
                type=VarDesc.VarType.LOD_TENSOR,
                persistable=False)
        else:
            out_dtype = var.dtype
            out_var = var

        if self._uniform:
            limit = np.sqrt(6.0 / float(fan_in))
            op = block.append_op(
                type="uniform_random",
                inputs={},
                outputs={"Out": out_var},
                attrs={
                    "shape": out_var.shape,
                    "dtype": int(out_dtype),
                    "min": -limit,
                    "max": limit,
                    "seed": self._seed
                },
                stop_gradient=True)

        else:
            std = np.sqrt(2.0 / float(fan_in))
            op = block.append_op(
                type="gaussian_random",
                outputs={"Out": out_var},
                attrs={
                    "shape": out_var.shape,
                    "dtype": int(out_dtype),
                    "mean": 0.0,
                    "std": std,
                    "seed": self._seed
                },
                stop_gradient=True)

        if var.dtype == VarDesc.VarType.FP16 or (
                var.dtype == VarDesc.VarType.BF16 and not self._uniform):
            block.append_op(
                type="cast",
                inputs={"X": out_var},
                outputs={"Out": var},
                attrs={"in_dtype": out_var.dtype,
                       "out_dtype": var.dtype})

        if not framework.in_dygraph_mode():
            var.op = op
        return op


class BilinearInitializer(Initializer):
    """
    This initializer can be used in transposed convolution operator to
    act as upsampling. Users can upsample a feature map with shape of
    (B, C, H, W) by any integer factor. The usage is:

    Examples:

        .. code-block:: python

            import math

            import paddle
            import paddle.nn as nn
            from paddle.regularizer import L2Decay

            factor = 2
            C = 2
            B = 8
            H = W = 32
            w_attr = paddle.ParamAttr(learning_rate=0.,
                                      regularizer=L2Decay(0.),
                                      initializer=nn.initializer.Bilinear())
            data = paddle.rand([B, 3, H, W], dtype='float32')
            conv_up = nn.Conv2DTranspose(3,
                                         out_channels=C,
                                         kernel_size=2 * factor - factor % 2,
                                         padding=int(
                                             math.ceil((factor - 1) / 2.)),
                                         stride=factor,
                                         weight_attr=w_attr,
                                         bias_attr=False)
            x = conv_up(data)

    Where, `out_channels=C` and `groups=C` means this is channel-wise transposed
    convolution. The filter shape will be (C, 1, K, K) where K is `kernel_size`,
    This initializer will set a (K, K) interpolation kernel for every channel
    of the filter identically. The resulting shape of the output feature map
    will be (B, C, factor * H, factor * W). Note that the learning rate and the
    weight decay are set to 0 in order to keep coefficient values of bilinear
    interpolation unchanged during training.

    """

    def __init__(self):
        """Constructor for BilinearInitializer.
        """
        super(BilinearInitializer, self).__init__()

    def __call__(self, var, block=None):
        """Initialize the input tensor with Bilinear initialization.

        Args:
            var(Tensor): Tensor that needs to be initialized.
            block(Block, optional): The block in which initialization ops
                   should be added. Used in static graph only, default None.

        Returns:
            The initialization op
        """
        block = self._check_block(block)

        if not isinstance(var, framework.Variable):
            raise ValueError("var must be framework.Variable.")

        if not isinstance(block, framework.Block):
            raise ValueError("block must be framework.Block.")

        shape = var.shape
        if len(shape) != 4:
            raise ValueError("the length of shape must be 4.")
        if shape[2] != shape[3]:
            raise ValueError("shape[2] must be equal to shape[3].")

        weight = np.zeros(np.prod(var.shape), dtype='float32')
        size = shape[3]
        # factor
        f = np.ceil(size / 2.)
        # center
        c = (2 * f - 1 - f % 2) / (2. * f)
        for i in range(np.prod(shape)):
            x = i % size
            y = (i / size) % size
            weight[i] = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
        weight = np.reshape(weight, shape)

        # to be compatible of fp16 initalizers
        if var.dtype in [
                VarDesc.VarType.FP16, VarDesc.VarType.BF16, VarDesc.VarType.FP64
        ]:
            out_dtype = VarDesc.VarType.FP32
            out_var = block.create_var(
                name=unique_name.generate(".".join(
                    ['bilinear_init', var.name, 'tmp'])),
                shape=var.shape,
                dtype=out_dtype,
                type=VarDesc.VarType.LOD_TENSOR,
                persistable=False)
        else:
            out_dtype = var.dtype
            out_var = var

        if out_dtype == VarDesc.VarType.FP32:
            value_name = "fp32_values"
            values = [float(v) for v in weight.flat]
        else:
            raise TypeError("Unsupported dtype %s", var.dtype)

        if np.prod(shape) > 1024 * 1024:
            raise ValueError("The size of input is too big. ")
        op = block.append_op(
            type='assign_value',
            outputs={'Out': [out_var]},
            attrs={
                'dtype': out_dtype,
                'shape': list(shape),
                value_name: values
            })

        if var.dtype in [
                VarDesc.VarType.FP16, VarDesc.VarType.BF16, VarDesc.VarType.FP64
        ]:
            block.append_op(
                type="cast",
                inputs={"X": out_var},
                outputs={"Out": var},
                attrs={"in_dtype": out_var.dtype,
                       "out_dtype": var.dtype})

        if not framework.in_dygraph_mode():
            var.op = op
        return op


class NumpyArrayInitializer(Initializer):
    """Init an parameter with an numpy array
    This op initialize the variable by numpy array.

    Args:
        value (numpy): numpy array to initialize the variable

    Returns:
        A Tensor variable initialized by numpy.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import numpy
            x = fluid.data(name="x", shape=[2, 1], dtype='float32')
            fc = fluid.layers.fc(input=x, size=10,
                param_attr=fluid.initializer.NumpyArrayInitializer(numpy.array([1,2])))
    """

    def __init__(self, value):
        import numpy
        assert isinstance(value, numpy.ndarray)
        super(NumpyArrayInitializer, self).__init__()
        self._value = value

    def __call__(self, var, block=None):
        """Initialize the input tensor with Numpy array.

        Args:
            var(Tensor): Tensor that needs to be initialized.
            block(Block, optional): The block in which initialization ops
                   should be added. Used in static graph only, default None.

        Returns:
            The initialization op
        """
        block = self._check_block(block)

        assert isinstance(var, framework.Variable)
        assert isinstance(block, framework.Block)

        # to be compatible of fp16 initalizers
        if var.dtype in [VarDesc.VarType.FP16, VarDesc.VarType.BF16]:
            out_dtype = VarDesc.VarType.FP32
            np_value = self._value.astype("float32")
            out_var = block.create_var(
                name=unique_name.generate(".".join(
                    ['numpy_array_init', var.name, 'tmp'])),
                shape=var.shape,
                dtype=out_dtype,
                type=VarDesc.VarType.LOD_TENSOR,
                persistable=False)
        else:
            out_var = var
            out_dtype = var.dtype
            np_value = self._value

        if out_dtype == VarDesc.VarType.FP32:
            value_name = "fp32_values"
            values = [float(v) for v in np_value.flat]
        elif out_dtype == VarDesc.VarType.INT32:
            value_name = "int32_values"
            values = [int(v) for v in np_value.flat]
        else:
            raise ValueError("Unsupported dtype %s", self._value.dtype)
        if self._value.size > 1024 * 1024 * 1024:
            raise ValueError("The size of input is too big. Please consider "
                             "saving it to file and 'load_op' to load it")
        op = block.append_op(
            type='assign_value',
            outputs={'Out': out_var},
            attrs={
                'dtype': out_dtype,
                'shape': list(self._value.shape),
                value_name: values
            },
            stop_gradient=True)

        if var.dtype in [VarDesc.VarType.FP16, VarDesc.VarType.BF16]:
            block.append_op(
                type="cast",
                inputs={"X": out_var},
                outputs={"Out": var},
                attrs={"in_dtype": out_var.dtype,
                       "out_dtype": var.dtype})

        if not framework.in_dygraph_mode():
            var.op = op
        return op


def set_global_initializer(weight_init, bias_init=None):
    """
    This API is used to set up global model parameter initializer in framework.

    After this API is invoked, the global initializer will takes effect in subsequent code.

    The model parameters include ``weight`` and ``bias`` . In the framework, they correspond 
    to ``paddle.ParamAttr`` , which is inherited from ``paddle.Tensor`` , and is a persistable Variable.
    This API only takes effect for model parameters, not for variables created through apis such as 
    :ref:`api_fluid_layers_create_global_var` , :ref:`api_fluid_layers_create_tensor`.
    
    If the initializer is also set up by ``param_attr`` or ``bias_attr`` when creating a network layer,
    the global initializer setting here will not take effect because it has a lower priority.

    If you want to cancel the global initializer in framework, please set global initializer to ``None`` .

    Args:
        weight_init (Initializer): set the global initializer for ``weight`` of model parameters.
        bias_init (Initializer, optional): set the global initializer for ``bias`` of model parameters. 
            Default: None.

    Returns:
        None

    Examples:
        .. code-block:: python

            import paddle
            import paddle.nn as nn

            nn.initializer.set_global_initializer(nn.initializer.Uniform(), nn.initializer.Constant())
            x_var = paddle.uniform((2, 4, 8, 8), dtype='float32', min=-1., max=1.)

            # The weight of conv1 is initialized by Uniform
            # The bias of conv1 is initialized by Constant
            conv1 = nn.Conv2D(4, 6, (3, 3))
            y_var1 = conv1(x_var)

            # If set param_attr/bias_attr too, global initializer will not take effect
            # The weight of conv2 is initialized by Xavier
            # The bias of conv2 is initialized by Normal
            conv2 = nn.Conv2D(4, 6, (3, 3), 
                weight_attr=nn.initializer.XavierUniform(),
                bias_attr=nn.initializer.Normal())
            y_var2 = conv2(x_var)

            # Cancel the global initializer in framework, it will takes effect in subsequent code
            nn.initializer.set_global_initializer(None)
    """

    check_type(weight_init, 'weight_init', (Initializer, type(None)),
               'set_global_initializer')
    global _global_weight_initializer_
    _global_weight_initializer_ = weight_init

    check_type(bias_init, 'bias_init', (Initializer, type(None)),
               'set_global_initializer')
    global _global_bias_initializer_
    _global_bias_initializer_ = bias_init


def _global_weight_initializer():
    """
    Return the global weight initializer, The user doesn't need to use it.
    """
    return _global_weight_initializer_


def _global_bias_initializer():
    """
    Return the global weight initializer, The user doesn't need to use it.
    """
    return _global_bias_initializer_


# We short the class name, since users will use the initializer with the package
# name. The sample code:
#
# import paddle.fluid as fluid
#
# hidden = fluid.layers.fc(...,
#                          param_attr=ParamAttr(fluid.initializer.Xavier()))
#
# It is no need to add an `Initializer` as the class suffix
Constant = ConstantInitializer
Uniform = UniformInitializer
Normal = NormalInitializer
TruncatedNormal = TruncatedNormalInitializer
Xavier = XavierInitializer
MSRA = MSRAInitializer
Bilinear = BilinearInitializer
