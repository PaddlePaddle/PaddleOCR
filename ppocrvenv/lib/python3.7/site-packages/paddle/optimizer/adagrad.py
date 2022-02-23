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


class Adagrad(Optimizer):
    r"""
    The Adaptive Gradient optimizer (Adagrad for short) use an optimization described 
    in paper: `Adaptive Subgradient Methods for Online Learning and
    Stochastic Optimization <http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf>`_.

    The parameter ``param_out`` update rule with gradient ``grad``:

    .. math::

        moment\_out &= moment + grad * grad

        param\_out &= param - \frac{learning\_rate * grad}{\sqrt{moment\_out} + \epsilon}


    The original paper does not have the ``epsilon`` attribute. It is added here
    in our implementation as also proposed `Per-parameter adaptive learning rate
    methods <http://cs231n.github.io/neural-networks-3/#ada>`_
    for numerical stability to avoid the division by zero error.

    Args:
        learning_rate (float|Tensor): The learning rate used to update ``Parameter``.
            It can be a float value or a ``Variable`` with a float type.
        epsilon (float, optional): A small float value for numerical stability.
            The default value is 1e-06.
	parameters (list|tuple, optional): List/Tuple of ``Tensor`` to update to minimize ``loss``. \
	    This parameter is required in dygraph mode. And you can specify different options for \
            different parameter groups such as the learning rate, weight decay, etc, \
            then the parameters are list of dict. Note that the learning_rate in paramter groups \
            represents the scale of base learning_rate. \
	    The default value is None in static mode, at this time all parameters will be updated.
	weight_decay (float|WeightDecayRegularizer, optional): The strategy of regularization. \
	    It canbe a float value as coeff of L2 regularization or \
	    :ref:`api_paddle_regularizer_L1Decay`, :ref:`api_paddle_regularizer_L2Decay`.
	    If a parameter has set regularizer using :ref:`api_paddle_fluid_param_attr_aramAttr` already, \
	    the regularization setting here in optimizer will be ignored for this parameter. \
	    Otherwise, the regularization setting here in optimizer will take effect. \
	    Default None, meaning there is no regularization.
        grad_clip (GradientClipBase, optional): Gradient cliping strategy, it's an instance of 
            some derived class of ``GradientClipBase`` . There are three cliping strategies, 
            ClipGradByGlobalNorm, ClipGradByNorm and ClipGradByValue. Default None, 
            meaning there is no gradient clipping.
        name (str, optional): Normally there is no need for user to set this property.
            For more information, please refer to :ref:`api_guide_Name`.
            The default value is None.
        initial_accumulator_value (float, optional): Initial value for moment accumulator.
            The default value is 0.0.

    Examples:
        .. code-block:: python

            import paddle
            import numpy as np

            inp = paddle.rand(shape=[10, 10])
            linear = paddle.nn.Linear(10, 10)
            out = linear(inp)
            loss = paddle.mean(out)
            adagrad = paddle.optimizer.Adagrad(learning_rate=0.1,
                    parameters=linear.parameters())
            out.backward()
            adagrad.step()
            adagrad.clear_grad()

            #Note that the learning_rate of linear_2 is 0.01.
            linear_1 = paddle.nn.Linear(10, 10)
            linear_2 = paddle.nn.Linear(10, 10)
            inp = paddle.uniform(shape=[10, 10], min=-0.1, max=0.1)
            out = linear_1(inp)
            out = linear_2(out)
            loss = paddle.mean(out)
            adagrad = paddle.optimizer.Adagrad(
                learning_rate=0.1,
                parameters=[{
                    'params': linear_1.parameters()
                }, {
                    'params': linear_2.parameters(),
                    'weight_decay': 0.001,
                    'learning_rate': 0.1,
                }],
                weight_decay=0.01)                   
            out.backward()
            adagrad.step()
            adagrad.clear_grad()

    """
    _moment_acc_str = "moment"

    def __init__(self,
                 learning_rate,
                 epsilon=1.0e-6,
                 parameters=None,
                 weight_decay=None,
                 grad_clip=None,
                 name=None,
                 initial_accumulator_value=0.0):
        assert learning_rate is not None
        assert epsilon is not None
        super(Adagrad, self).__init__(
            learning_rate=learning_rate,
            parameters=parameters,
            weight_decay=weight_decay,
            grad_clip=grad_clip,
            name=name)
        self.type = "adagrad"
        self._epsilon = epsilon
        self.initial_accumulator_value = initial_accumulator_value
        self._default_dict = {
            'epsilon': epsilon,
            'initial_accumulator_value': initial_accumulator_value,
        }

    def _create_accumulators(self, block, parameters):
        assert isinstance(block, framework.Block)

        if isinstance(parameters, dict):
            parameters = self._update_param_group(parameters)

        for p in parameters:
            self._add_accumulator(
                self._moment_acc_str,
                p,
                fill_value=self.initial_accumulator_value)

    def _append_optimize_op(self, block, param_and_grad):
        assert isinstance(block, framework.Block)

        if isinstance(param_and_grad, dict):
            param_and_grad = self._update_param_group(param_and_grad)

        moment_acc = self._get_accumulator(self._moment_acc_str,
                                           param_and_grad[0])
        # Create the adagrad optimizer op
        adagrad_op = block.append_op(
            type=self.type,
            inputs={
                "Param": param_and_grad[0],
                "Grad": param_and_grad[1],
                "Moment": moment_acc,
                "LearningRate": self._create_param_lr(param_and_grad)
            },
            outputs={"ParamOut": param_and_grad[0],
                     "MomentOut": moment_acc},
            attrs={"epsilon": self._epsilon},
            stop_gradient=True)

        return adagrad_op

    def _update_param_group(self, parameters):
        self._epsilon = parameters.get('epsilon', self._default_dict['epsilon'])
        self.initial_accumulator_value = parameters.get(
            'initial_accumulator_value',
            self._default_dict['initial_accumulator_value'])
        parameters = parameters.get('params')
        return parameters
