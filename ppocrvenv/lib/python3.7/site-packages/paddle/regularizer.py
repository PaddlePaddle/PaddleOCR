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

__all__ = ['L1Decay', 'L2Decay']

import paddle.fluid as fluid


class L1Decay(fluid.regularizer.L1Decay):
    r"""
    Implement the L1 Weight Decay Regularization, which encourages the weights to be sparse.
    
    It can be set in :ref:`api_paddle_ParamAttr` or ``optimizer`` (such as :ref:`api_paddle_optimizer_Momentum` ). 
    When set in ``ParamAttr`` , it only takes effect for trainable parameters in this layer. When set in 
    ``optimizer`` , it takes effect for all trainable parameters. When set together, ``ParamAttr`` has 
    higher priority than ``optimizer`` , which means that for a trainable parameter, if regularizer is defined 
    in its ParamAttr, then the regularizer in Optimizer will be ignored. Otherwise the  regularizer
    in Optimizer will be used.
    
    In the implementation, the loss function of L1 Weight Decay Regularization is as follows:
	
    .. math::

        loss = coeff * reduce\_sum(abs(x))

    Args:
        coeff(float, optional): regularization coeff. Default:0.0.
	
    Examples:
        .. code-block:: python

            # Example1: set Regularizer in optimizer
            import paddle
            from paddle.regularizer import L1Decay
            import numpy as np
            linear = paddle.nn.Linear(10, 10)
            inp = paddle.rand(shape=[10, 10], dtype="float32")
            out = linear(inp)
            loss = paddle.mean(out)
            beta1 = paddle.to_tensor([0.9], dtype="float32")
            beta2 = paddle.to_tensor([0.99], dtype="float32")
            momentum = paddle.optimizer.Momentum(
                learning_rate=0.1,
                parameters=linear.parameters(),
                weight_decay=L1Decay(0.0001))
            back = out.backward()
            momentum.step()
            momentum.clear_grad()

            # Example2: set Regularizer in parameters
            # Set L1 regularization in parameters.
            # Global regularizer does not take effect on my_conv2d for this case.
            from paddle.nn import Conv2D
            from paddle import ParamAttr
            from paddle.regularizer import L2Decay

            my_conv2d = Conv2D(
                    in_channels=10,
                    out_channels=10,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    weight_attr=ParamAttr(regularizer=L2Decay(coeff=0.01)),
                    bias_attr=False)
    """

    def __init__(self, coeff=0.0):
        super(L1Decay, self).__init__(coeff)


class L2Decay(fluid.regularizer.L2Decay):
    r"""
    Implement the L2 Weight Decay Regularization, which helps to prevent the model over-fitting.
    
    It can be set in :ref:`api_paddle_ParamAttr` or ``optimizer`` (such as :ref:`api_paddle_optimizer_Momentum` ). 
    When set in ``ParamAttr`` , it only takes effect for trainable parameters in this layer. When set in 
    ``optimizer`` , it takes effect for all trainable parameters. When set together, ``ParamAttr`` has 
    higher priority than ``optimizer`` , which means that for a trainable parameter, if regularizer is defined 
    in its ParamAttr, then the regularizer in Optimizer will be ignored. Otherwise the  regularizer
    in Optimizer will be used.
    
    In the implementation, the loss function of L2 Weight Decay Regularization is as follows:

    .. math::

        loss = 0.5 * coeff * reduce\_sum(square(x))

    Args:
        regularization_coeff(float, optional): regularization coeff. Default:0.0
	
    Examples:
        .. code-block:: python

            # Example1: set Regularizer in optimizer
            import paddle
            from paddle.regularizer import L2Decay
            import numpy as np
            linear = paddle.nn.Linear(10, 10)
            inp = paddle.rand(shape=[10, 10], dtype="float32")
            out = linear(inp)
            loss = paddle.mean(out)
            beta1 = paddle.to_tensor([0.9], dtype="float32")
            beta2 = paddle.to_tensor([0.99], dtype="float32")
            momentum = paddle.optimizer.Momentum(
                learning_rate=0.1,
                parameters=linear.parameters(),
                weight_decay=L2Decay(0.0001))
            back = out.backward()
            momentum.step()
            momentum.clear_grad()

            # Example2: set Regularizer in parameters
            # Set L2 regularization in parameters.
            # Global regularizer does not take effect on my_conv2d for this case.
            from paddle.nn import Conv2D
            from paddle import ParamAttr
            from paddle.regularizer import L2Decay

            my_conv2d = Conv2D(
                    in_channels=10,
                    out_channels=10,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    weight_attr=ParamAttr(regularizer=L2Decay(coeff=0.01)),
                    bias_attr=False)
    """

    def __init__(self, coeff=0.0):
        super(L2Decay, self).__init__(coeff)
