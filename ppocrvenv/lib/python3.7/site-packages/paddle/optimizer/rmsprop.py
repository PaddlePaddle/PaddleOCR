# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from .optimizer import Optimizer
from ..fluid import core
from ..fluid import framework
from ..fluid.framework import Variable

__all__ = []


class RMSProp(Optimizer):
    r"""
    Root Mean Squared Propagation (RMSProp) is an unpublished, adaptive learning
    rate method. The original slides proposed RMSProp: Slide 29 of
    http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf .

    The original equation is as follows:

    ..  math::

        r(w, t) & = \rho r(w, t-1) + (1 - \rho)(\nabla Q_{i}(w))^2

        w & = w - \frac{\eta} {\sqrt{r(w,t) + \epsilon}} \nabla Q_{i}(w)

    The first equation calculates moving average of the squared gradient for
    each weight. Then dividing the gradient by :math:`sqrt{v(w,t)}`.

    In some cases, adding a momentum term :math: `\\beta` is beneficial.
    In our implementation, Nesterov momentum is used:

    ..  math::

        r(w, t) & = \rho r(w, t-1) + (1 - \rho)(\nabla Q_{i}(w))^2

        v(w, t) & = \beta v(w, t-1) + \frac{\eta} {\sqrt{r(w,t) +
            \epsilon}} \nabla Q_{i}(w)

        w & = w - v(w, t)

    if centered is True:

    ..  math::

        r(w, t) & = \rho r(w, t-1) + (1 - \rho)(\nabla Q_{i}(w))^2

        g(w, t) & = \rho g(w, t-1) + (1 - \rho)\nabla Q_{i}(w)

        v(w, t) & = \beta v(w, t-1) + \frac{\eta} {\sqrt{r(w,t) - (g(w, t))^2 +
            \epsilon}} \nabla Q_{i}(w)

        w & = w - v(w, t)

    where, :math:`\rho` is a hyperparameter and typical values are 0.9, 0.95
    and so on. :math:`\beta` is the momentum term. :math:`\epsilon` is a
    smoothing term to avoid division by zero, usually set somewhere in range
    from 1e-4 to 1e-8.


    Parameters:
        learning_rate (float|LRScheduler): The learning rate used to update ``Parameter``.
          It can be a float value or a LRScheduler.
        rho(float): rho is :math:`\rho` in equation, default is 0.95.
        epsilon(float): :math:`\epsilon` in equation is smoothing term to
          avoid division by zero, default is 1e-6.
        momentum(float): :math:`\beta` in equation is the momentum term,
          default is 0.0.
        centered(bool): If True, gradients are normalized by the estimated variance of
          the gradient; if False, by the uncentered second moment. Setting this to
          True may help with training, but is slightly more expensive in terms of
          computation and memory. Defaults to False.
        parameters (list|tuple, optional): List/Tuple of ``Tensor`` to update to minimize ``loss``. 
          This parameter is required in dygraph mode. And you can specify different options for 
          different parameter groups such as the learning rate, weight decay, etc, 
          then the parameters are list of dict. Note that the learning_rate in paramter groups 
          represents the scale of base learning_rate. 
          The default value is None in static mode, at this time all parameters will be updated.
        weight_decay (float|WeightDecayRegularizer, optional): The strategy of regularization. 
          It canbe a float value as coeff of L2 regularization or \
          :ref:`api_fluid_regularizer_L1Decay`, :ref:`api_fluid_regularizer_L2Decay`.
          If a parameter has set regularizer using :ref:`api_fluid_ParamAttr` already, 
          the regularization setting here in optimizer will be ignored for this parameter. 
          Otherwise, the regularization setting here in optimizer will take effect. 
          Default None, meaning there is no regularization.
        grad_clip (GradientClipBase, optional): Gradient cliping strategy, it's an instance of
          some derived class of ``GradientClipBase`` . There are three cliping strategies
          ( :ref:`api_fluid_clip_GradientClipByGlobalNorm` , :ref:`api_fluid_clip_GradientClipByNorm` ,
          :ref:`api_fluid_clip_GradientClipByValue` ). Default None, meaning there is no gradient clipping.
        name (str, optional): This parameter is used by developers to print debugging information. 
          For details, please refer to :ref:`api_guide_Name`. Default is None.

    Raises:
        ValueError: If learning_rate, rho, epsilon, momentum are None.

    Examples:
          .. code-block:: python

            import paddle

            inp = paddle.rand([10,10], dtype="float32")
            linear = paddle.nn.Linear(10, 10)
            out = linear(inp)
            loss = paddle.mean(out)

            rmsprop = paddle.optimizer.RMSProp(learning_rate=0.1,
                             parameters=linear.parameters(),
                                       weight_decay=0.01)
            out.backward()
            rmsprop.step()
            rmsprop.clear_grad()

            #Note that the learning_rate of linear_2 is 0.01.
            linear_1 = paddle.nn.Linear(10, 10)
            linear_2 = paddle.nn.Linear(10, 10)
            inp = paddle.uniform(shape=[10, 10], min=-0.1, max=0.1)
            out = linear_1(inp)
            out = linear_2(out)
            loss = paddle.mean(out)
            rmsprop = paddle.optimizer.RMSProp(
                learning_rate=0.1,
                parameters=[{
                    'params': linear_1.parameters()
                }, {
                    'params': linear_2.parameters(),
                    'weight_decay': 0.001,
                    'learning_rate': 0.1
                }],
                weight_decay=0.01)                   
            out.backward()
            rmsprop.step()
            rmsprop.clear_grad()
    """

    _momentum_acc_str = "momentum"
    _mean_square_acc_str = "mean_square"
    _mean_grad_acc_str = "mean_grad"

    def __init__(self,
                 learning_rate,
                 rho=0.95,
                 epsilon=1.0e-6,
                 momentum=0.0,
                 centered=False,
                 parameters=None,
                 weight_decay=None,
                 grad_clip=None,
                 name=None):
        if learning_rate is None:
            raise ValueError("learning_rate is not set.")
        if rho is None:
            raise ValueError("rho is not set.")
        if epsilon is None:
            raise ValueError("epsilon is not set.")
        if momentum is None:
            raise ValueError("momentum is not set.")
        if not 0.0 <= epsilon:
            raise ValueError("Invalid value of epsilon, expect epsilon >= 0.")
        if not 0.0 <= momentum:
            raise ValueError("Invalid value of momentum, expect momentum >= 0.")
        if not 0.0 <= rho:
            raise ValueError("Invalid value of rho, expect rho >= 0.")

        super(RMSProp, self).__init__(
            learning_rate=learning_rate,
            parameters=parameters,
            weight_decay=weight_decay,
            grad_clip=grad_clip,
            name=name)

        self.type = "rmsprop"
        self._rho = rho
        self._epsilon = epsilon
        self._momentum = momentum
        self._centered = centered
        self._default_dict = {
            'rho': rho,
            'epsilon': epsilon,
            'momentum': momentum,
            'centered': centered,
        }

    def _create_accumulators(self, block, parameters):
        if not isinstance(block, framework.Block):
            raise TypeError("block is not instance of framework.Block.")

        if isinstance(parameters, dict):
            parameters = parameters.get('params')

        for p in parameters:
            self._add_accumulator(self._momentum_acc_str, p)
            self._add_accumulator(self._mean_square_acc_str, p)
            self._add_accumulator(self._mean_grad_acc_str, p)

    def _append_optimize_op(self, block, param_and_grad):
        if not isinstance(block, framework.Block):
            raise TypeError("block is not instance of framework.Block.")

        if isinstance(param_and_grad, dict):
            param_and_grad = self._update_param_group(param_and_grad)

        momentum_acc = self._get_accumulator(self._momentum_acc_str,
                                             param_and_grad[0])
        mean_square_acc = self._get_accumulator(self._mean_square_acc_str,
                                                param_and_grad[0])
        mean_grad_acc = self._get_accumulator(self._mean_grad_acc_str,
                                              param_and_grad[0])
        rmsprop_op = block.append_op(
            type=self.type,
            inputs={
                "Param": param_and_grad[0],
                "Grad": param_and_grad[1],
                "Moment": momentum_acc,
                "MeanSquare": mean_square_acc,
                "MeanGrad": mean_grad_acc,
                "LearningRate": self._create_param_lr(param_and_grad),
            },
            outputs={
                "ParamOut": param_and_grad[0],
                "MomentOut": momentum_acc,
                "MeanSquareOut": mean_square_acc,
                "MeanGradOut": mean_grad_acc
            },
            attrs={
                "epsilon": self._epsilon,
                "decay": self._rho,
                "momentum": self._momentum,
                "centered": self._centered
            },
            stop_gradient=True)

        return rmsprop_op

    def _update_param_group(self, parameters):
        self._epsilon = parameters.get('epsilon', self._default_dict['epsilon'])
        self._rho = parameters.get('rho', self._default_dict['rho'])
        self._momentum = parameters.get('momentum',
                                        self._default_dict['momentum'])
        self._centered = parameters.get('centered',
                                        self._default_dict['centered'])
        parameters = parameters.get('params')
        return parameters
