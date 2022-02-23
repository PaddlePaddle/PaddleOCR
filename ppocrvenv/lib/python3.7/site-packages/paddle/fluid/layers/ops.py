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
import os
from .layer_function_generator import generate_layer_fn, generate_activation_fn, generate_inplace_fn, add_sample_code
from .. import core
from ..framework import convert_np_dtype_to_dtype_, Variable
from ..data_feeder import convert_dtype, check_variable_and_dtype, check_type, check_dtype
from paddle.utils import deprecated

__deprecated_func_name__ = {
    'tanh_shrink': 'tanhshrink',
    'logsigmoid': 'log_sigmoid'
}

__activations_noattr__ = [
    'sigmoid',
    'silu',
    'logsigmoid',
    'tanh_shrink',
    'softplus',
    'softsign',
    'tanh',
]

__unary_func__ = [
    'exp',
    'expm1',
    'atan',
    'sqrt',
    'rsqrt',
    'abs',
    'ceil',
    'floor',
    'cos',
    'tan',
    'acos',
    'sin',
    'sinh',
    'asin',
    'cosh',
    'round',
    'reciprocal',
    'square',
    'lgamma',
]

__inplace_unary_func__ = [
    'exp_',
    'sqrt_',
    'rsqrt_',
    'ceil_',
    'floor_',
    'round_',
    'reciprocal_',
]

__all__ = []

for _OP in set(__all__):
    globals()[_OP] = generate_layer_fn(_OP)

# It is a hot fix in some unittest using:
#   fluid.layers.scale(x=x, scale=10.0, out=out_var)
# e.g.: test_program_code.py, test_dist_train.py
globals()['_scale'] = generate_layer_fn('scale')

globals()['_elementwise_div'] = generate_layer_fn('elementwise_div')

__all__ += __activations_noattr__
__all__ += __unary_func__
__all__ += __inplace_unary_func__

for _OP in set(__activations_noattr__):
    _new_OP = _OP
    if _OP in __deprecated_func_name__:
        _new_OP = __deprecated_func_name__[_OP]
    _func = generate_activation_fn(_OP)
    _func = deprecated(
        since="2.0.0", update_to="paddle.nn.functional.%s" % (_new_OP))(_func)
    globals()[_OP] = _func

for _OP in set(__unary_func__):
    _new_OP = _OP
    if _OP in __deprecated_func_name__:
        _new_OP = __deprecated_func_name__[_OP]
    _func = generate_activation_fn(_OP)
    _func = deprecated(since="2.0.0", update_to="paddle.%s" % (_new_OP))(_func)
    globals()[_OP] = _func

for _OP in set(__inplace_unary_func__):
    _new_OP = _OP
    if _OP in __deprecated_func_name__:
        _new_OP = __deprecated_func_name__[_OP]
    _func = generate_inplace_fn(_OP)
    _func = deprecated(since="2.0.0", update_to="paddle.%s" % (_new_OP))(_func)
    globals()[_OP] = _func

add_sample_code(globals()["sigmoid"], r"""
Examples:
    .. code-block:: python

        import paddle
        import paddle.nn.functional as F

        x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
        out = F.sigmoid(x)
        print(out)
        # [0.40131234 0.450166   0.52497919 0.57444252]

""")

add_sample_code(globals()["silu"], r"""
Examples:
    .. code-block:: python

        import paddle
        import paddle.nn.functional as F

        x = paddle.to_tensor([1.0, 2.0, 3.0, 4.0])
        out = F.silu(x)
        print(out)
        # [ 0.7310586 1.7615942 2.8577224, 3.9280552 ]

""")

add_sample_code(globals()["logsigmoid"], r"""
Examples:
    .. code-block:: python

        import paddle
        import paddle.nn.functional as F

        x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
        out = F.log_sigmoid(x)
        print(out)
        # [-0.91301525 -0.79813887 -0.64439666 -0.55435524]

""")

add_sample_code(globals()["exp"], r"""
Examples:
    .. code-block:: python

        import paddle

        x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
        out = paddle.exp(x)
        print(out)
        # [0.67032005 0.81873075 1.10517092 1.34985881]

""")

add_sample_code(globals()["expm1"], r"""
Examples:
    .. code-block:: python

        import paddle

        x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
        out = paddle.expm1(x)
        print(out)
        # [-0.32967997, -0.18126924,  0.10517092,  0.34985882]

""")

add_sample_code(globals()["tanh"], r"""
Examples:
    .. code-block:: python

        import paddle

        x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
        out = paddle.tanh(x)
        print(out)
        # [-0.37994896 -0.19737532  0.09966799  0.29131261]

""")

add_sample_code(globals()["atan"], r"""
Examples:
    .. code-block:: python

        import paddle

        x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
        out = paddle.atan(x)
        print(out)
        # [-0.38050638 -0.19739556  0.09966865  0.29145679]

""")

add_sample_code(globals()["tanh_shrink"], r"""
Examples:
    .. code-block:: python

        import paddle
        import paddle.nn.functional as F

        x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
        out = F.tanhshrink(x) 
        print(out)
        # [-0.020051, -0.00262468, 0.000332005, 0.00868739]

""")

add_sample_code(globals()["sqrt"], r"""
Examples:
    .. code-block:: python

        import paddle

        x = paddle.to_tensor([0.1, 0.2, 0.3, 0.4])
        out = paddle.sqrt(x)
        print(out)
        # [0.31622777 0.4472136  0.54772256 0.63245553]

""")

add_sample_code(globals()["rsqrt"], r"""
Examples:
    .. code-block:: python

        import paddle

        x = paddle.to_tensor([0.1, 0.2, 0.3, 0.4])
        out = paddle.rsqrt(x)
        print(out)
        # [3.16227766 2.23606798 1.82574186 1.58113883]

""")

add_sample_code(globals()["abs"], r"""
Examples:
    .. code-block:: python

        import paddle

        x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
        out = paddle.abs(x)
        print(out)
        # [0.4 0.2 0.1 0.3]

""")

add_sample_code(globals()["ceil"], r"""
Examples:
    .. code-block:: python

        import paddle

        x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
        out = paddle.ceil(x)
        print(out)
        # [-0. -0.  1.  1.]

""")

add_sample_code(globals()["floor"], r"""
Examples:
    .. code-block:: python

        import paddle

        x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
        out = paddle.floor(x)
        print(out)
        # [-1. -1.  0.  0.]

""")

add_sample_code(globals()["cos"], r"""
Examples:
    .. code-block:: python

        import paddle

        x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
        out = paddle.cos(x)
        print(out)
        # [0.92106099 0.98006658 0.99500417 0.95533649]

""")

add_sample_code(globals()["tan"], r"""
Examples:
    .. code-block:: python

        import paddle

        x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
        out = paddle.tan(x)
        print(out)
        # [-0.42279324, -0.20271005, 0.10033467, 0.30933627]

""")

add_sample_code(globals()["acos"], r"""
Examples:
    .. code-block:: python

        import paddle

        x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
        out = paddle.acos(x)
        print(out)
        # [1.98231317 1.77215425 1.47062891 1.26610367]

""")

add_sample_code(globals()["sin"], r"""
Examples:
    .. code-block:: python

        import paddle

        x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
        out = paddle.sin(x)
        print(out)
        # [-0.38941834 -0.19866933  0.09983342  0.29552021]

""")

add_sample_code(globals()["asin"], r"""
Examples:
    .. code-block:: python

        import paddle

        x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
        out = paddle.asin(x)
        print(out)
        # [-0.41151685 -0.20135792  0.10016742  0.30469265]

""")

add_sample_code(globals()["cosh"], r"""
Examples:
    .. code-block:: python

        import paddle

        x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
        out = paddle.cosh(x)
        print(out)
        # [1.08107237 1.02006676 1.00500417 1.04533851]

""")

add_sample_code(globals()["sinh"], r"""
Examples:
    .. code-block:: python

        import paddle

        x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
        out = paddle.sinh(x)
        print(out)
        # [-0.41075233 -0.201336    0.10016675  0.30452029]

""")

add_sample_code(globals()["round"], r"""
Examples:
    .. code-block:: python

        import paddle

        x = paddle.to_tensor([-0.5, -0.2, 0.6, 1.5])
        out = paddle.round(x)
        print(out)
        # [-1. -0.  1.  2.]

""")

add_sample_code(globals()["reciprocal"], r"""
Examples:
    .. code-block:: python

        import paddle

        x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
        out = paddle.reciprocal(x)
        print(out)
        # [-2.5        -5.         10.          3.33333333]

""")

add_sample_code(globals()["square"], r"""
Examples:
    .. code-block:: python

        import paddle

        x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
        out = paddle.square(x)
        print(out)
        # [0.16 0.04 0.01 0.09]

""")

add_sample_code(globals()["lgamma"], r"""
Examples:
    .. code-block:: python

        import paddle

        x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
        out = paddle.lgamma(x)
        print(out)
        # [1.31452441, 1.76149750, 2.25271273, 1.09579802]

""")

add_sample_code(globals()["softplus"], r"""
Examples:
    .. code-block:: python

        import paddle
        import paddle.nn.functional as F

        x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
        out = F.softplus(x) 
        print(out)
        # [0.513015, 0.598139, 0.744397, 0.854355]

""")

add_sample_code(globals()["softsign"], r"""
Examples:
    .. code-block:: python

        import paddle
        import paddle.nn.functional as F

        x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
        out = F.softsign(x) 
        print(out)
        # [-0.285714, -0.166667, 0.0909091, 0.230769]

""")

__all__ += ['softshrink']

_softshrink_ = generate_layer_fn('softshrink')


def softshrink(x, alpha=None):
    check_variable_and_dtype(x, 'x', ['float16', 'float32', 'float64'],
                             'softshrink')

    locals_var = locals().copy()
    kwargs = dict()
    for name, val in locals_var.items():
        if val is not None:
            if name == 'alpha':
                kwargs['lambda'] = val
            else:
                kwargs[name] = val
    return _softshrink_(**kwargs)


softshrink.__doc__ = r"""
	:alias_main: paddle.nn.functional.softshrink
	:alias: paddle.nn.functional.softshrink,paddle.nn.functional.activation.softshrink
	:old_api: paddle.fluid.layers.softshrink

:strong:`Softshrink Activation Operator`

..  math::
    out = \\begin{cases}
            x - \\alpha, \\text{if } x > \\alpha \\\\
            x + \\alpha, \\text{if } x < -\\alpha \\\\
            0,  \\text{otherwise}
          \\end{cases}


Args:
    x: Input of Softshrink operator, an N-D Tensor, with data type float32, float64 or float16.
    alpha (float): non-negative offset
    
Returns:
    Output of Softshrink operator with the same type of input.

Examples:
    .. code-block:: python
    
        import paddle.fluid as fluid
        data = fluid.data(name="input", shape=[None, 784])
        result = fluid.layers.softshrink(x=data, alpha=0.3)
"""

__all__ += ['hard_shrink']

_hard_shrink_ = generate_layer_fn('hard_shrink')


@deprecated(since="2.0.0", update_to="paddle.nn.functional.hardshrink")
def hard_shrink(x, threshold=None):
    check_variable_and_dtype(x, 'x', ['float16', 'float32', 'float64'],
                             'hard_shrink')

    locals_var = locals().copy()
    kwargs = dict()
    for name, val in locals_var.items():
        if val is not None:
            kwargs[name] = val
    return _hard_shrink_(**kwargs)


hard_shrink.__doc__ = _hard_shrink_.__doc__ + """
Examples:

    >>> import paddle.fluid as fluid
    >>> data = fluid.layers.data(name="input", shape=[784])
    >>> result = fluid.layers.hard_shrink(x=data, threshold=0.3)
"""

__all__ += ['cumsum']

_cum_sum_ = generate_layer_fn('cumsum')


@deprecated(
    since="2.0.0",
    update_to="paddle.cumsum",
    reason="New APIs for Paddle 2.0 are coming.")
def cumsum(x, axis=None, exclusive=None, reverse=None):
    check_type(x, 'x', (Variable), 'cumsum')
    locals_var = locals().copy()
    kwargs = dict()
    for name, val in locals_var.items():
        if val is not None:
            kwargs[name] = val
    return _cum_sum_(**kwargs)


cumsum.__doc__ = """
	:alias_main: paddle.cumsum
	:alias: paddle.cumsum,paddle.tensor.cumsum,paddle.tensor.math.cumsum
	:old_api: paddle.fluid.layers.cumsum

The cumulative sum of the elements along a given axis. By default, the first element of the result is the same of the first element of the input. If exlusive is true, the first element of the result is 0.

Args:
    x (Variable): Input of cumsum operator, the Tensor/LoDTensor needed to be cumsumed. 
    axis (int, optional): The dimension to accumulate along. -1 means the last dimension. Default is -1.
    exclusive (bool, optional): Whether to perform exclusive cumsum. Default is False.
    reverse (bool, optional): If true, the cumsum is performed in the reversed direction. Default is False.

Returns:
    Variable(Tensor/LoDTensor): The result of cumsum operator, output of cumsum operator. 

Examples:
    .. code-block:: python
        
        import paddle.fluid as fluid
        data = fluid.layers.data(name="input", shape=[32, 784])
        result = fluid.layers.cumsum(data, axis=0)
"""

__all__ += ['thresholded_relu']

_thresholded_relu_ = generate_layer_fn('thresholded_relu')


def thresholded_relu(x, threshold=None):
    check_variable_and_dtype(x, 'x', ['float16', 'float32', 'float64'],
                             'thresholded_relu')

    locals_var = locals().copy()
    kwargs = dict()
    for name, val in locals_var.items():
        if val is not None:
            kwargs[name] = val

    return _thresholded_relu_(**kwargs)


thresholded_relu.__doc__ = r"""
	:alias_main: paddle.nn.functional.thresholded_relu
	:alias: paddle.nn.functional.thresholded_relu,paddle.nn.functional.activation.thresholded_relu
	:old_api: paddle.fluid.layers.thresholded_relu

:strong:`Thresholded ReLU Activation Operator`

Equation:
    ..  math::
        out = \\begin{cases}
            x, &if x > threshold \\\\
            0, &otherwise
            \\end{cases}

Args:
    x(Variable): The input of Thresholded ReLU op, Tensor or LoDTensor, dtype: float32 or float64.
        
    threshold(float, optional): The threshold value. Note that if the arg `threshold` is not set, the threshold in the equation is 1.0.

Returns:

    Variable: The output of Thresholded ReLU op, Tensor or LoDTensor, dtype: float32 or float64, the same as the input, shape: the same as the input.

Examples:
    
    .. code-block:: python
    
        # declarative mode
        import numpy as np
        from paddle import fluid
        
        x = fluid.data(name="x", shape=(-1, 3), dtype="float32")
        y = fluid.layers.thresholded_relu(x, threshold=0.1)
        
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        start = fluid.default_startup_program()
        main = fluid.default_main_program()
        
        data = np.random.randn(2, 3).astype("float32")
        exe.run(start)
        
        y_np, = exe.run(main, feed={"x": data}, fetch_list=[y])
        
        data
        # array([[ 0.21134382, -1.1805999 ,  0.32876605],
        #        [-1.2210793 , -0.7365624 ,  1.0013918 ]], dtype=float32)
        y_np
        # array([[ 0.21134382, -0.        ,  0.32876605],
        #        [-0.        , -0.        ,  1.0013918 ]], dtype=float32)

    .. code-block:: python
    
        # imperative mode
        import numpy as np
        from paddle import fluid
        import paddle.fluid.dygraph as dg
        
        data = np.random.randn(2, 3).astype("float32")
        place = fluid.CPUPlace()
        with dg.guard(place) as g:
            x = dg.to_variable(data)
            y = fluid.layers.thresholded_relu(x, threshold=0.1)
            y_np = y.numpy()
        data
        # array([[ 0.21134382, -1.1805999 ,  0.32876605],
        #        [-1.2210793 , -0.7365624 ,  1.0013918 ]], dtype=float32)
        y_np
        # array([[ 0.21134382, -0.        ,  0.32876605],
        #        [-0.        , -0.        ,  1.0013918 ]], dtype=float32)
"""

__all__ += ['gelu']

_gelu_ = generate_layer_fn('gelu')


@deprecated(since="2.0.0", update_to="paddle.nn.functional.gelu")
def gelu(x, approximate=False):
    locals_var = locals().copy()
    kwargs = dict()
    for name, val in locals_var.items():
        if val is not None:
            kwargs[name] = val
    return _gelu_(**kwargs)


gelu.__doc__ = r"""
:strong:`GeLU Activation Operator`
For more details, see [Gaussian Error Linear Units](https://arxiv.org/abs/1606.08415).

Equation:
    if approximate is True
    ..  math::
        out = 0.5 * x * (1 + tanh(\\sqrt{\\frac{2}{\\pi}} * (x + 0.044715x^{3})))

    else
    ..  math::
        out = 0.5 * x * (1 + erf(\\frac{x}{\\sqrt{2}}))

Args:

    x(Variable): The input of GeLU op, Tensor or LoDTensor, dtype: float32 or float64.

Returns:

    Variable: The output of GeLU op, Tensor or LoDTensor, dtype: float32 or float64, the same as the input, shape: the same as the input.

Examples:
    
    .. code-block:: python
    
        # declarative mode
        import numpy as np
        from paddle import fluid
        
        x = fluid.data(name="x", shape=(-1, 3), dtype="float32")
        y = fluid.layers.gelu(x)
        
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        start = fluid.default_startup_program()
        main = fluid.default_main_program()
        
        data = np.random.randn(2, 3).astype("float32")
        exe.run(start)
        
        y_np, = exe.run(main, feed={"x": data}, fetch_list=[y])
        
        data
        # array([[ 0.87165993, -1.0541513 , -0.37214822],
        #         [ 0.15647964,  0.32496083,  0.33045998]], dtype=float32)
        y_np
        # array([[ 0.70456535, -0.15380788, -0.13207214],
        #        [ 0.08796856,  0.20387867,  0.2080159 ]], dtype=float32)

    .. code-block:: python
    
        # imperative mode
        import numpy as np
        from paddle import fluid
        import paddle.fluid.dygraph as dg
        
        data = np.random.randn(2, 3).astype("float32")
        place = fluid.CPUPlace()
        with dg.guard(place) as g:
            x = dg.to_variable(data)
            y = fluid.layers.gelu(x)
            y_np = y.numpy()
        data
        # array([[ 0.87165993, -1.0541513 , -0.37214822],
        #        [ 0.15647964,  0.32496083,  0.33045998]], dtype=float32)
        y_np
        # array([[ 0.70456535, -0.15380788, -0.13207214],
        #        [ 0.08796856,  0.20387867,  0.2080159 ]], dtype=float32)
"""

__all__ += ['erf']

_erf_ = generate_layer_fn('erf')


def erf(x, name=None):
    locals_var = locals().copy()
    kwargs = dict()
    for name, val in locals_var.items():
        if val is not None:
            kwargs[name] = val
    return _erf_(**kwargs)


erf.__doc__ = r"""
:strong:`Erf Operator`
For more details, see [Error function](https://en.wikipedia.org/wiki/Error_function).

Equation:
    ..  math::
        out = \\frac{2}{\\sqrt{\\pi}} \\int_{0}^{x}e^{- \\eta^{2}}d\\eta

Args:

    x (Tensor): The input tensor, it's data type should be float32, float64.

Returns:

    Tensor: The output of Erf op, dtype: float32 or float64, the same as the input, shape: the same as the input.

Examples:
    
    .. code-block:: python
    
        import paddle
        x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
        out = paddle.erf(x)
        print(out)
        # [-0.42839236 -0.22270259  0.11246292  0.32862676]
"""
