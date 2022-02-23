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

# TODO: define activation functions of neural network

from ...fluid import core
from ...fluid.framework import in_dygraph_mode
from ...framework import ParamAttr
from ..initializer import Constant
from paddle.framework import get_default_dtype
from .. import functional as F
from paddle.nn import Layer

__all__ = []


class ELU(Layer):
    r"""
    ELU Activation.

    .. math::

        ELU(x)=
            \left\{
                \begin{array}{lcl}
                x,& &\text{if } \ x > 0 \\
                alpha * (e^{x} - 1),& &\text{if } \ x <= 0
                \end{array}
            \right.

    Parameters:
        alpha (float, optional): The 'alpha' value of the ELU formulation. Default is 1.0.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        - input: Tensor with any shape.
        - output: Tensor with the same shape as input.

    Examples:
        .. code-block:: python

            import paddle

            x = paddle.to_tensor([[-1. ,6.], [1., 15.6]])
            m = paddle.nn.ELU(0.2)
            out = m(x)
            # [[-0.12642411  6.        ]
            #  [ 1.          15.6      ]]
    """

    def __init__(self, alpha=1.0, name=None):
        super(ELU, self).__init__()
        self._alpha = alpha
        self._name = name

    def forward(self, x):
        return F.elu(x, self._alpha, self._name)

    def extra_repr(self):
        name_str = ', name={}'.format(self._name) if self._name else ''
        return 'alpha={}{}'.format(self._alpha, name_str)


class GELU(Layer):
    r"""
    GELU Activation.

    If approximate is True

    .. math::

        GELU(x) = 0.5 * x * (1 + tanh(\sqrt{\frac{2}{\pi}} * (x + 0.044715x^{3})))

    else

    .. math::

        GELU(x) = 0.5 * x * (1 + erf(\frac{x}{\sqrt{2}}))

    Parameters:
        approximate (bool, optional): Wether to enable approximation. Default is False.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        - input: Tensor with any shape.
        - output: Tensor with the same shape as input.

    Examples:
        .. code-block:: python

            import paddle
            import numpy as np

            x = paddle.to_tensor(np.array([[-1, 0.5],[1, 1.5]]))

            m = paddle.nn.GELU()
            out = m(x) # [-0.158655 0.345731 0.841345 1.39979]

            m = paddle.nn.GELU(True)
            out = m(x) # [-0.158808 0.345714 0.841192 1.39957]
    """

    def __init__(self, approximate=False, name=None):
        super(GELU, self).__init__()
        self._approximate = approximate
        self._name = name

    def forward(self, x):
        return F.gelu(x, self._approximate, self._name)

    def extra_repr(self):
        name_str = ', name={}'.format(self._name) if self._name else ''
        return 'approximate={}{}'.format(self._approximate, name_str)


class Hardshrink(Layer):
    r"""
    Hardshrink Activation

    .. math::

        hardshrink(x)=
            \left\{
                \begin{array}{rcl}
                    x, & & if \ x > threshold \\
                    x, & & if \ x < -threshold \\
                    0, & & if \ others
            \end{array}
            \right.

    Parameters:
        threshold (float, optional): The value of threshold for hardthrink. Default is 0.5
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        - input: Tensor with any shape.
        - output: Tensor with the same shape as input.

    Examples:

        .. code-block:: python

            import paddle

            x = paddle.to_tensor([-1, 0.3, 2.5])
            m = paddle.nn.Hardshrink()
            out = m(x) # [-1., 0., 2.5]
    """

    def __init__(self, threshold=0.5, name=None):
        super(Hardshrink, self).__init__()
        self._threshold = threshold
        self._name = name

    def forward(self, x):
        return F.hardshrink(x, self._threshold, self._name)

    def extra_repr(self):
        name_str = ', name={}'.format(self._name) if self._name else ''
        return 'threshold={}{}'.format(self._threshold, name_str)


class Hardswish(Layer):
    r"""
    Hardswish activation

    Hardswish is proposed in MobileNetV3, and performs better in computational stability
    and efficiency compared to swish function. For more details please refer
    to: https://arxiv.org/pdf/1905.02244.pdf

    .. math::

        Hardswish(x)=
            \left\{
                \begin{array}{cll}
                0 &, & \text{if } x \leq -3 \\
                x &, & \text{if } x \geq 3 \\
                \frac{x(x+3)}{6} &, & \text{otherwise}
                \end{array}
            \right.
            

    Parameters:
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        - input: Tensor with any shape.
        - output: Tensor with the same shape as input.

    Examples:

        .. code-block:: python

            import paddle

            x = paddle.to_tensor([-4., 5., 1.])
            m = paddle.nn.Hardswish()
            out = m(x) # [0., 5., 0.666667]
    """

    def __init__(self, name=None):
        super(Hardswish, self).__init__()
        self._name = name

    def forward(self, x):
        return F.hardswish(x, self._name)

    def extra_repr(self):
        name_str = 'name={}'.format(self._name) if self._name else ''
        return name_str


class Tanh(Layer):
    r"""
    Tanh Activation.

    .. math::
        Tanh(x) = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}

    Parameters:
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        - input: Tensor with any shape.
        - output: Tensor with the same shape as input.

    Examples:

        .. code-block:: python

            import paddle
            import numpy as np

            x = paddle.to_tensor(np.array([-0.4, -0.2, 0.1, 0.3]))
            m = paddle.nn.Tanh()
            out = m(x)
            print(out)
            # [-0.37994896 -0.19737532  0.09966799  0.29131261]
    """

    def __init__(self, name=None):
        super(Tanh, self).__init__()
        self._name = name

    def forward(self, x):
        return F.tanh(x, self._name)

    def extra_repr(self):
        name_str = 'name={}'.format(self._name) if self._name else ''
        return name_str


class Hardtanh(Layer):
    r"""
    Hardtanh Activation

    .. math::

        Hardtanh(x)=
            \left\{
                \begin{array}{cll}
                    max,& & \text{if } x > max \\
                    min,& & \text{if } x < min \\
                    x,& & \text{otherwise}
                \end{array}
            \right.


    Parameters:
        min (float, optional): The value of min for Hardtanh. Default is -1.
        max (float, optional): The value of max for Hardtanh. Default is 1.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        - input: Tensor with any shape.
        - output: Tensor with the same shape as input.

    Examples:
        .. code-block:: python

            import paddle

            x = paddle.to_tensor([-1.5, 0.3, 2.5])
            m = paddle.nn.Hardtanh()
            out = m(x) # [-1., 0.3, 1.]
    """

    def __init__(self, min=-1.0, max=1.0, name=None):
        super(Hardtanh, self).__init__()
        self._min = min
        self._max = max
        self._name = name

    def forward(self, x):
        return F.hardtanh(x, self._min, self._max, self._name)

    def extra_repr(self):
        name_str = ', name={}'.format(self._name) if self._name else ''
        return 'min={}, max={}{}'.format(self._min, self._max, name_str)


class PReLU(Layer):
    """
    PReLU Activation.

    .. math::

        PReLU(x) = max(0, x) + weight * min(0, x)

    Parameters:
        num_parameters (int, optional): Number of `weight` to learn. The supported values are:
            1 - a single parameter `alpha` is used for all input channels;
            Number of channels - a seperate `alpha` is used for each input channel.
            Default is 1.
        init (float, optional): Init value of learnable `weight`. Default is 0.25.
        weight_attr(ParamAttr, optional): The parameter attribute for the learnable `weight`.
            Default is None. For more information, please refer to :ref:`api_paddle_ParamAttr`.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.
        data_format(str, optional): Data format that specifies the layout of input.
            It may be "NC", "NCL", "NCHW", "NCDHW", "NLC", "NHWC" or "NDHWC". Default: "NCHW".

    Shape:
        - input: Tensor with any shape. Default dtype is float32.
        - output: Tensor with the same shape as input.

    Examples:
        .. code-block:: python

            import paddle
            import numpy as np

            paddle.set_default_dtype("float64")

            data = np.array([[[[-2.0,  3.0, -4.0,  5.0],
                            [ 3.0, -4.0,  5.0, -6.0],
                            [-7.0, -8.0,  8.0,  9.0]],
                            [[ 1.0, -2.0, -3.0,  4.0],
                            [-5.0,  6.0,  7.0, -8.0],
                            [ 6.0,  7.0,  8.0,  9.0]]]], 'float64')
            x = paddle.to_tensor(data)
            m = paddle.nn.PReLU(1, 0.25)
            out = m(x)
            # [[[[-0.5 ,  3.  , -1.  ,  5.  ],
            #    [ 3.  , -1.  ,  5.  , -1.5 ],
            #    [-1.75, -2.  ,  8.  ,  9.  ]],
            #   [[ 1.  , -0.5 , -0.75,  4.  ],
            #    [-1.25,  6.  ,  7.  , -2.  ],
            #    [ 6.  ,  7.  ,  8.  ,  9.  ]]]]
    """

    def __init__(self,
                 num_parameters=1,
                 init=0.25,
                 weight_attr=None,
                 data_format="NCHW",
                 name=None):
        super(PReLU, self).__init__()
        self._num_parameters = num_parameters
        self._init = init
        self._weight_attr = weight_attr
        self._name = name
        self._data_format = data_format

        self._weight = self.create_parameter(
            attr=self._weight_attr,
            shape=[self._num_parameters],
            dtype=get_default_dtype(),
            is_bias=False,
            default_initializer=Constant(self._init))

    def forward(self, x):
        return F.prelu(x, self._weight, data_format=self._data_format)

    def extra_repr(self):
        name_str = ', name={}'.format(self._name) if self._name else ''
        return 'num_parameters={}, data_format={}, init={}, dtype={}{}'.format(
            self._num_parameters, self._data_format, self._init, self._dtype,
            name_str)


class ReLU(Layer):
    """
    ReLU Activation.

    .. math::

        ReLU(x) = max(x, 0)

    Parameters:
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        - input: Tensor with any shape.
        - output: Tensor with the same shape as input.

    Examples:
        .. code-block:: python

            import paddle

            x = paddle.to_tensor([-2., 0., 1.])
            m = paddle.nn.ReLU()
            out = m(x) # [0., 0., 1.]
    """

    def __init__(self, name=None):
        super(ReLU, self).__init__()
        self._name = name

    def forward(self, x):
        return F.relu(x, self._name)

    def extra_repr(self):
        name_str = 'name={}'.format(self._name) if self._name else ''
        return name_str


class ReLU6(Layer):
    """
    ReLU6 Activation

    .. math::

        ReLU6(x) = min(max(0,x), 6)

    Parameters:
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        - input: Tensor with any shape.
        - output: Tensor with the same shape as input.

    Examples:
        .. code-block:: python

            import paddle
            import numpy as np

            x = paddle.to_tensor(np.array([-1, 0.3, 6.5]))
            m = paddle.nn.ReLU6()
            out = m(x) # [0, 0.3, 6]
    """

    def __init__(self, name=None):
        super(ReLU6, self).__init__()
        self._name = name

    def forward(self, x):
        return F.relu6(x, self._name)

    def extra_repr(self):
        name_str = 'name={}'.format(self._name) if self._name else ''
        return name_str


class SELU(Layer):
    r"""
    SELU Activation

    .. math::

        SELU(x)= scale *
            \left\{
                \begin{array}{lcl}
                x,& &\text{if } \ x > 0 \\
                alpha * e^{x} - alpha,& &\text{if } \ x <= 0
                \end{array}
            \right.

    Parameters:
        scale (float, optional): The value of scale(must be greater than 1.0) for SELU. Default is 1.0507009873554804934193349852946
        alpha (float, optional): The value of alpha(must be no less than zero) for SELU. Default is 1.6732632423543772848170429916717
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        - input: Tensor with any shape.
        - output: Tensor with the same shape as input.

    Examples:
        .. code-block:: python

            import paddle
            import numpy as np

            x = paddle.to_tensor(np.array([[0.0, 1.0],[2.0, 3.0]]))
            m = paddle.nn.SELU()
            out = m(x) # [[0, 1.050701],[2.101402, 3.152103]]
    """

    def __init__(self,
                 scale=1.0507009873554804934193349852946,
                 alpha=1.6732632423543772848170429916717,
                 name=None):
        super(SELU, self).__init__()
        self._scale = scale
        self._alpha = alpha
        self._name = name

    def forward(self, x):
        return F.selu(x, self._scale, self._alpha, self._name)

    def extra_repr(self):
        name_str = ', name={}'.format(self._name) if self._name else ''
        return 'scale={:.16f}, alpha={:.16f}{}'.format(self._scale, self._alpha,
                                                       name_str)


class LeakyReLU(Layer):
    r"""
    Leaky ReLU Activation.

    .. math::

        LeakyReLU(x)=
            \left\{
                \begin{array}{rcl}
                    x, & & if \ x >= 0 \\
                    negative\_slope * x, & & otherwise \\
                \end{array}
            \right.


    Parameters:
        negative_slope (float, optional): Slope of the activation function at
            :math:`x < 0` . Default is 0.01.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        - input: Tensor with any shape.
        - output: Tensor with the same shape as input.

    Examples:
        .. code-block:: python

            import paddle
            import numpy as np

            m = paddle.nn.LeakyReLU()
            x = paddle.to_tensor(np.array([-2, 0, 1], 'float32'))
            out = m(x)  # [-0.02, 0., 1.]
    """

    def __init__(self, negative_slope=0.01, name=None):
        super(LeakyReLU, self).__init__()
        self._negative_slope = negative_slope
        self._name = name

    def forward(self, x):
        return F.leaky_relu(x, self._negative_slope, self._name)

    def extra_repr(self):
        name_str = ', name={}'.format(self._name) if self._name else ''
        return 'negative_slope={}{}'.format(self._negative_slope, name_str)


class Sigmoid(Layer):
    """
    this interface is used to construct a callable object of the ``Sigmoid`` class. This layer calcluate the `sigmoid` of input x.

    .. math::

        Sigmoid(x) = \\frac{1}{1 + e^{-x}}

    Parameters:
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        x: N-D tensor, available dtype is float16, float32, float64.

    Returns:
        A callable object of Sigmoid.

    Examples:

        .. code-block:: python

          import paddle

          m = paddle.nn.Sigmoid()
          x = paddle.to_tensor([1.0, 2.0, 3.0, 4.0])
          out = m(x) # [0.7310586, 0.880797, 0.95257413, 0.98201376]
    """

    def __init__(self, name=None):
        super(Sigmoid, self).__init__()
        self.name = name

    def forward(self, x):
        return F.sigmoid(x, self.name)

    def extra_repr(self):
        name_str = 'name={}'.format(self.name) if self.name else ''
        return name_str


class Hardsigmoid(Layer):
    r"""
    This interface is used to construct a callable object of the ``Hardsigmoid`` class.
    This layer calcluate the `hardsigmoid` of input x.

    A 3-part piecewise linear approximation of sigmoid(https://arxiv.org/abs/1603.00391),
    which is much faster than sigmoid.

    .. math::

        Hardsigmoid(x)=
            \left\{
                \begin{array}{rcl}
            0, & & \text{if } \ x \leq -3 \\
            1, & & \text{if } \ x \geq 3 \\
            x/6 + 1/2, & & \text{otherwise}
                \end{array}
            \right.


    Parameters:
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        x: N-D tensor, available dtype is float32, float64.

    Returns:
        A callable object of Hardsigmoid.

    Examples:

        .. code-block:: python

          import paddle

          m = paddle.nn.Hardsigmoid()
          x = paddle.to_tensor([-4., 5., 1.])
          out = m(x) # [0., 1, 0.666667]
    """

    def __init__(self, name=None):
        super(Hardsigmoid, self).__init__()
        self.name = name

    def forward(self, x):
        return F.hardsigmoid(x, name=self.name)

    def extra_repr(self):
        name_str = 'name={}'.format(self.name) if self.name else ''
        return name_str


class Softplus(Layer):
    r"""
    Softplus Activation

    .. math::

        Softplus(x) = \frac{1}{beta} * \log(1 + e^{beta * x}) \\
        \text{For numerical stability, the implementation reverts to the linear function when: beta * x > threshold.}

    Parameters:
        beta (float, optional): The value of beta for Softplus. Default is 1
        threshold (float, optional): The value of threshold for Softplus. Default is 20
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        - input: Tensor with any shape.
        - output: Tensor with the same shape as input.

    Examples:
        .. code-block:: python

            import paddle
            import numpy as np

            x = paddle.to_tensor(np.array([-0.4, -0.2, 0.1, 0.3]))
            m = paddle.nn.Softplus()
            out = m(x) # [0.513015, 0.598139, 0.744397, 0.854355]
    """

    def __init__(self, beta=1, threshold=20, name=None):
        super(Softplus, self).__init__()
        self._beta = beta
        self._threshold = threshold
        self._name = name

    def forward(self, x):
        return F.softplus(x, self._beta, self._threshold, self._name)

    def extra_repr(self):
        name_str = ', name={}'.format(self._name) if self._name else ''
        return 'beta={}, threshold={}{}'.format(self._beta, self._threshold,
                                                name_str)


class Softshrink(Layer):
    r"""
    Softshrink Activation

    .. math::

        Softshrink(x)=
            \left\{
                \begin{array}{rcl}
                x - threshold,& & \text{if } x > threshold \\
                x + threshold,& & \text{if } x < -threshold \\
                0,& &  \text{otherwise}
            \end{array}
            \right.


    Parameters:
        threshold (float, optional): The value of threshold(must be no less than zero) for softplus. Default is 0.5
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        - input: Tensor with any shape.
        - output: Tensor with the same shape as input.

    Examples:
        .. code-block:: python

            import paddle
            import numpy as np

            x = paddle.to_tensor(np.array([-0.9, -0.2, 0.1, 0.8]))
            m = paddle.nn.Softshrink()
            out = m(x) # [-0.4, 0, 0, 0.3]
    """

    def __init__(self, threshold=0.5, name=None):
        super(Softshrink, self).__init__()
        self._threshold = threshold
        self._name = name

    def forward(self, x):
        return F.softshrink(x, self._threshold, self._name)

    def extra_repr(self):
        name_str = ', name={}'.format(self._name) if self._name else ''
        return 'threshold={}{}'.format(self._threshold, name_str)


class Softsign(Layer):
    r"""
    Softsign Activation

    .. math::

        Softsign(x) = \frac{x}{1 + |x|}

    Parameters:
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        - input: Tensor with any shape.
        - output: Tensor with the same shape as input.

    Examples:
        .. code-block:: python

            import paddle
            import numpy as np

            x = paddle.to_tensor(np.array([-0.4, -0.2, 0.1, 0.3]))
            m = paddle.nn.Softsign()
            out = m(x) # [-0.285714, -0.166667, 0.0909091, 0.230769]
    """

    def __init__(self, name=None):
        super(Softsign, self).__init__()
        self._name = name

    def forward(self, x):
        return F.softsign(x, self._name)

    def extra_repr(self):
        name_str = 'name={}'.format(self._name) if self._name else ''
        return name_str


class Swish(Layer):
    r"""
    Swish Activation.

    .. math::

        Swish(x) = \frac{x}{1 + e^{-x}}

    Parameters:
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        - input: Tensor with any shape.
        - output: Tensor with the same shape as input.

    Examples:
        .. code-block:: python

            import paddle
            import numpy as np

            x = paddle.to_tensor(np.array([-2., 0., 1.]))
            m = paddle.nn.Swish()
            out = m(x) # [-0.238406, 0., 0.731059]
    """

    def __init__(self, name=None):
        super(Swish, self).__init__()
        self._name = name

    def forward(self, x):
        return F.swish(x, self._name)

    def extra_repr(self):
        name_str = 'name={}'.format(self._name) if self._name else ''
        return name_str


class Mish(Layer):
    r"""
    Mish Activation.

    ..  math::

        softplus(x) = \begin{cases}
                x, \text{if } x > \text{threshold} \\
                \ln(1 + e^{x}),  \text{otherwise}
            \end{cases}

        Mish(x) = x * \tanh(softplus(x))
    
    Parameters:
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        - input: Tensor with any shape.
        - output: Tensor with the same shape as input.
    
    Examples:

        .. code-block:: python

            import paddle

            x = paddle.to_tensor([-5., 0., 5.])
            m = paddle.nn.Mish()
            out = m(x) # [-0.03357624, 0., 4.99955208]

    """

    def __init__(self, name=None):
        super(Mish, self).__init__()
        self._name = name

    def forward(self, x):
        return F.mish(x, self._name)

    def extra_repr(self):
        name_str = 'name={}'.format(self._name) if self._name else ''
        return name_str


class Tanhshrink(Layer):
    """
    Tanhshrink Activation

    .. math::

        Tanhshrink(x) = x - tanh(x)

    Parameters:
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        - input: Tensor with any shape.
        - output: Tensor with the same shape as input.

    Examples:
        .. code-block:: python

            import paddle
            import numpy as np

            x = paddle.to_tensor(np.array([-0.4, -0.2, 0.1, 0.3]))
            m = paddle.nn.Tanhshrink()
            out = m(x) # [-0.020051, -0.00262468, 0.000332005, 0.00868739]
    """

    def __init__(self, name=None):
        super(Tanhshrink, self).__init__()
        self._name = name

    def forward(self, x):
        return F.tanhshrink(x, self._name)

    def extra_repr(self):
        name_str = 'name={}'.format(self._name) if self._name else ''
        return name_str


class ThresholdedReLU(Layer):
    r"""
    Thresholded ReLU Activation

    .. math::

        ThresholdedReLU(x) =
            \left\{
                \begin{array}{rl}
                x,& \text{if } \ x > threshold \\
                0,& \text{otherwise}
                \end{array}
            \right.


    Parameters:
        threshold (float, optional): The value of threshold for ThresholdedReLU. Default is 1.0
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        - input: Tensor with any shape.
        - output: Tensor with the same shape as input.

    Examples:
        .. code-block:: python

            import paddle
            import numpy as np

            x = paddle.to_tensor(np.array([2., 0., 1.]))
            m = paddle.nn.ThresholdedReLU()
            out = m(x) # [2., 0., 0.]
    """

    def __init__(self, threshold=1.0, name=None):
        super(ThresholdedReLU, self).__init__()
        self._threshold = threshold
        self._name = name

    def forward(self, x):
        return F.thresholded_relu(x, self._threshold, self._name)

    def extra_repr(self):
        name_str = ', name={}'.format(self._name) if self._name else ''
        return 'threshold={}{}'.format(self._threshold, name_str)


class Silu(Layer):
    """
    Silu Activation.
    .. math::

        Silu(x) = \frac{x}{1 + e^{-x}}

    Parameters:
        x (Tensor): The input Tensor with data type float32, or float64.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        - input: Tensor with any shape.
        - output: Tensor with the same shape as input.

    Examples:
        .. code-block:: python

            import paddle

            x = paddle.to_tensor([1.0, 2.0, 3.0, 4.0])
            m = paddle.nn.Silu()
            out = m(x) # [ 0.731059, 1.761594, 2.857722, 3.928055 ]
    """

    def __init__(self, name=None):
        super(Silu, self).__init__()
        self._name = name

    def forward(self, x):
        return F.silu(x, self._name)

    def extra_repr(self):
        name_str = 'name={}'.format(self._name) if self._name else ''
        return name_str


class LogSigmoid(Layer):
    r"""
    LogSigmoid Activation.

    .. math::

        LogSigmoid(x) = log \frac{1}{1 + e^{-x}}

    Parameters:
        x (Tensor): The input Tensor with data type float32, or float64.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        - input: Tensor with any shape.
        - output: Tensor with the same shape as input.

    Examples:
        .. code-block:: python

            import paddle

            x = paddle.to_tensor([1.0, 2.0, 3.0, 4.0])
            m = paddle.nn.LogSigmoid()
            out = m(x) # [-0.313262 -0.126928 -0.0485874 -0.0181499]
    """

    def __init__(self, name=None):
        super(LogSigmoid, self).__init__()
        self._name = name

    def forward(self, x):
        return F.log_sigmoid(x, self._name)

    def extra_repr(self):
        name_str = 'name={}'.format(self._name) if self._name else ''
        return name_str


class Softmax(Layer):
    r"""
    Softmax Activation.

    This operator implements the softmax layer. The calculation process is as follows:

    1. The dimension :attr:`axis` of ``x`` will be permuted to the last.

    2. Then ``x`` will be logically flattened to a 2-D matrix. The matrix's second
    dimension(row length) is the same as the dimension :attr:`axis` of ``x``,
    and the first dimension(column length) is the product of all other dimensions
    of ``x``. For each row of the matrix, the softmax operator squashes the
    K-dimensional(K is the width of the matrix, which is also the size of ``x``'s
    dimension :attr:`axis`) vector of arbitrary real values to a K-dimensional
    vector of real values in the range [0, 1] that add up to 1.

    3. After the softmax operation is completed, the inverse operations of steps 1 and 2
    are performed to restore the two-dimensional matrix to the same dimension as the ``x`` .

    It computes the exponential of the given dimension and the sum of exponential
    values of all the other dimensions in the K-dimensional vector input.
    Then the ratio of the exponential of the given dimension and the sum of
    exponential values of all the other dimensions is the output of the softmax
    operator.

    For each row :math:`i` and each column :math:`j` in the matrix, we have:

    .. math::

        Softmax[i, j] = \frac{\exp(x[i, j])}{\sum_j(exp(x[i, j])}

    Example:

    .. code-block:: text

        Case 1:
          Input:
            x.shape = [2, 3, 4]
            x.data = [[[2.0, 3.0, 4.0, 5.0],
                       [3.0, 4.0, 5.0, 6.0],
                       [7.0, 8.0, 8.0, 9.0]],
                      [[1.0, 2.0, 3.0, 4.0],
                       [5.0, 6.0, 7.0, 8.0],
                       [6.0, 7.0, 8.0, 9.0]]]

          Attrs:
            axis = -1

          Output:
            out.shape = [2, 3, 4]
            out.data = [[[0.0320586 , 0.08714432, 0.23688282, 0.64391426],
                         [0.0320586 , 0.08714432, 0.23688282, 0.64391426],
                         [0.07232949, 0.19661193, 0.19661193, 0.53444665]],
                        [[0.0320586 , 0.08714432, 0.23688282, 0.64391426],
                         [0.0320586 , 0.08714432, 0.23688282, 0.64391426],
                         [0.0320586 , 0.08714432, 0.23688282, 0.64391426]]]

        Case 2:
          Input:
            x.shape = [2, 3, 4]
            x.data = [[[2.0, 3.0, 4.0, 5.0],
                       [3.0, 4.0, 5.0, 6.0],
                       [7.0, 8.0, 8.0, 9.0]],
                      [[1.0, 2.0, 3.0, 4.0],
                       [5.0, 6.0, 7.0, 8.0],
                       [6.0, 7.0, 8.0, 9.0]]]
          Attrs:
            axis = 1

          Output:
            out.shape = [2, 3, 4]
            out.data = [[[0.00657326, 0.00657326, 0.01714783, 0.01714783],
                         [0.01786798, 0.01786798, 0.04661262, 0.04661262],
                         [0.97555875, 0.97555875, 0.93623955, 0.93623955]],
                        [[0.00490169, 0.00490169, 0.00490169, 0.00490169],
                         [0.26762315, 0.26762315, 0.26762315, 0.26762315],
                         [0.72747516, 0.72747516, 0.72747516, 0.72747516]]]

    Parameters:
        axis (int, optional): The axis along which to perform log_softmax
            calculations. It should be in range [-D, D), where D is the
            dimensions of ``x`` . If ``axis`` < 0, it works the same way as
            :math:`axis + D` . Default is -1.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        - input: Tensor with any shape.
        - output: Tensor with the same shape as input.

    Examples:
        .. code-block:: python

            import paddle
            import numpy as np

            x = np.array([[[2.0, 3.0, 4.0, 5.0],
                        [3.0, 4.0, 5.0, 6.0],
                        [7.0, 8.0, 8.0, 9.0]],
                        [[1.0, 2.0, 3.0, 4.0],
                        [5.0, 6.0, 7.0, 8.0],
                        [6.0, 7.0, 8.0, 9.0]]], 'float32')
            x = paddle.to_tensor(x)
            m = paddle.nn.Softmax()
            out = m(x)
            # [[[0.0320586 , 0.08714432, 0.23688282, 0.64391426],
            #   [0.0320586 , 0.08714432, 0.23688282, 0.64391426],
            #   [0.07232949, 0.19661193, 0.19661193, 0.53444665]],
            # [[0.0320586 , 0.08714432, 0.23688282, 0.64391426],
            #   [0.0320586 , 0.08714432, 0.23688282, 0.64391426],
            #   [0.0320586 , 0.08714432, 0.23688282, 0.64391426]]]
    """

    def __init__(self, axis=-1, name=None):
        super(Softmax, self).__init__()
        self._axis = axis
        self._dtype = None
        self._name = name

    def forward(self, x):
        return F.softmax(x, self._axis, self._dtype, self._name)

    def extra_repr(self):
        name_str = ', name={}'.format(self._name) if self._name else ''
        return 'axis={}{}'.format(self._axis, name_str)


class LogSoftmax(Layer):
    r"""
    This operator implements the log_softmax layer. The calculation process is as follows:

    .. math::

        \begin{array} {rcl}
            Out[i, j] &= &log(softmax(x)) \\
            &= &log(\frac{\exp(X[i, j])}{\sum_j(\exp(X[i, j])})
        \end{array}

    Parameters:
        axis (int, optional): The axis along which to perform log_softmax
            calculations. It should be in range [-D, D), where D is the
            dimensions of the input Tensor . If ``axis`` < 0, it works the
            same way as :math:`axis + D` . Default is -1.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        - input: Tensor with any shape.
        - output: Tensor with the same shape as input.

    Examples:
        .. code-block:: python

            import paddle

            x = [[[-2.0, 3.0, -4.0, 5.0],
                  [3.0, -4.0, 5.0, -6.0],
                  [-7.0, -8.0, 8.0, 9.0]],
                 [[1.0, -2.0, -3.0, 4.0],
                  [-5.0, 6.0, 7.0, -8.0],
                  [6.0, 7.0, 8.0, 9.0]]]
            m = paddle.nn.LogSoftmax()
            x = paddle.to_tensor(x)
            out = m(x)
            # [[[ -7.1278396   -2.1278396   -9.127839    -0.12783948]
            #   [ -2.1270514   -9.127051    -0.12705144 -11.127051  ]
            #   [-16.313261   -17.313261    -1.3132617   -0.31326184]]
            #  [[ -3.0518122   -6.051812    -7.051812    -0.051812  ]
            #   [-12.313267    -1.3132664   -0.3132665  -15.313267  ]
            #   [ -3.4401896   -2.4401896   -1.4401896   -0.44018966]]]
    """

    def __init__(self, axis=-1, name=None):
        super(LogSoftmax, self).__init__()
        self._axis = axis
        self._name = name

    def forward(self, x):
        return F.log_softmax(x, self._axis)

    def extra_repr(self):
        name_str = ', name={}'.format(self._name) if self._name else ''
        return 'axis={}{}'.format(self._axis, name_str)


class Maxout(Layer):
    r"""
    Maxout Activation.

    Assumed the input shape is (N, Ci, H, W).
    The output shape is (N, Co, H, W).
    Then Co = Ci/groups and the operator formula is as follows:

    .. math::

        \begin{array}{l}
            &out_{si+j} = \max_{k} x_{gsi + sk + j} \\
            &g = groups \\
            &s = \frac{input.size}{num\_channels} \\
            &0 \le i < \frac{num\_channels}{groups} \\
            &0 \le j < s \\
            &0 \le k < groups
        \end{array}

    Parameters:
        groups (int, optional): The groups number of maxout. `groups` specifies the
            index of channel dimension where maxout will be performed. This must be
            a factor of number of features. Default is 1.
        axis (int, optional): The axis along which to perform maxout calculations.
            It should be 1 when data format is NCHW, be -1 or 3 when data format
            is NHWC. If ``axis`` < 0, it works the same way as :math:`axis + D` ,
            where D is the dimensions of ``x`` . Default is 1.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        - input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - output: :math:`(N, C_{out}, H_{out}, W_{out})`

    Examples:
        .. code-block:: python

            import paddle

            x = paddle.rand([1, 2, 3, 4])
            # [[[[0.5002636  0.22272532 0.17402348 0.2874594 ]
            #    [0.95313174 0.6228939  0.7129065  0.7087491 ]
            #    [0.02879342 0.88725346 0.61093384 0.38833922]]
            #   [[0.5231306  0.03807496 0.91661984 0.15602879]
            #    [0.666127   0.616567   0.30741522 0.24044901]
            #    [0.7142536  0.7351477  0.31588817 0.23782359]]]]
            m = paddle.nn.Maxout(groups=2)
            out = m(x)
            # [[[[0.5231306  0.22272532 0.91661984 0.2874594 ]
            #    [0.95313174 0.6228939  0.7129065  0.7087491 ]
            #    [0.7142536  0.88725346 0.61093384 0.38833922]]]]
    """

    def __init__(self, groups, axis=1, name=None):
        super(Maxout, self).__init__()
        self._groups = groups
        self._axis = axis
        self._name = name

    def forward(self, x):
        return F.maxout(x, self._groups, self._axis, self._name)

    def extra_repr(self):
        name_str = ', name={}'.format(self._name) if self._name else ''
        return 'groups={}, axis={}{}'.format(self._groups, self._axis, name_str)
