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
from ..fluid.framework import Variable, name_scope

__all__ = []


class Adamax(Optimizer):
    r"""
    The Adamax optimizer is implemented based on the Adamax Optimization 
    in Section 7 of `Adam paper <https://arxiv.org/abs/1412.6980>`_.
    The Adamax algorithm is a variant of the Adam algorithm based on the infinite norm,
    which makes the learning rate update algorithm more stable and simple.

    The parameter ``param_out`` update rule with gradient ``grad``:

    .. math::

        t & = t + 1

        moment\_out & = {\beta}_1 * moment + (1 - {\beta}_1) * grad

        inf\_norm\_out & = max({\beta}_2 * inf\_norm + \epsilon, |grad|)

        learning\_rate & = \frac{learning\_rate}{1 - {\beta}_1^t}

        param\_out & = param - learning\_rate * \frac{moment\_out}{inf\_norm\_out}

    Related paper: `Adam: A Method for Stochastic Optimization <https://arxiv.org/abs/1412.6980>`_

    The original paper does not have an ``epsilon`` attribute,
    it is added here for numerical stability to prevent the division by 0 error.

    Args:
        learning_rate (float|LRScheduler, optional): The learning rate used to update ``Parameter``.
            It can be a float value or a LRScheduler. The default value is 0.001.
        beta1 (float, optional): The exponential decay rate for the 1st moment estimates.
            The default value is 0.9.
        beta2 (float, optional): The exponential decay rate for the 2nd moment estimates.
            The default value is 0.999.
        epsilon (float, optional): A small float value for numerical stability.
            The default value is 1e-08.
	parameters (list|tuple, optional): List/Tuple of ``Tensor`` to update to minimize ``loss``. \
	    This parameter is required in dygraph mode. And you can specify different options for \
            different parameter groups such as the learning rate, weight decay, etc, \
            then the parameters are list of dict. Note that the learning_rate in paramter groups \
            represents the scale of base learning_rate. \
	    The default value is None in static mode, at this time all parameters will be updated.
	weight_decay (float|WeightDecayRegularizer, optional): The strategy of regularization. \
	    It canbe a float value as coeff of L2 regularization or \
	    :ref:`api_fluid_regularizer_L1Decay`, :ref:`api_fluid_regularizer_L2Decay`.
	    If a parameter has set regularizer using :ref:`api_fluid_ParamAttr` already, \
	    the regularization setting here in optimizer will be ignored for this parameter. \
	    Otherwise, the regularization setting here in optimizer will take effect. \
	    Default None, meaning there is no regularization.
        grad_clip (GradientClipBase, optional): Gradient cliping strategy, it's an instance of 
            some derived class of ``GradientClipBase`` . There are three cliping strategies 
            ( :ref:`api_fluid_clip_GradientClipByGlobalNorm` , :ref:`api_fluid_clip_GradientClipByNorm` , 
            :ref:`api_fluid_clip_GradientClipByValue` ). Default None, meaning there is no gradient clipping.
        name (str, optional): Normally there is no need for user to set this property.
            For more information, please refer to :ref:`api_guide_Name`.
            The default value is None.

    **Notes**:
        **Currently, Adamax doesn't support sparse parameter optimization.**

    Examples:
        .. code-block:: python
            
            import paddle
            import numpy as np

            inp = np.random.uniform(-0.1, 0.1, [10, 10]).astype("float32")
            linear = paddle.nn.Linear(10, 10)
            inp = paddle.to_tensor(inp)
            out = linear(inp)
            loss = paddle.mean(out)

            beta1 = paddle.to_tensor([0.9], dtype="float32")
            beta2 = paddle.to_tensor([0.99], dtype="float32")

            adam = paddle.optimizer.Adamax(learning_rate=0.1,
                    parameters=linear.parameters(),
                    beta1=beta1,
                    beta2=beta2,
                    weight_decay=0.01)
            out.backward()
            adam.step()
            adam.clear_grad()


            #Note that the learning_rate of linear_2 is 0.01.
            linear_1 = paddle.nn.Linear(10, 10)
            linear_2 = paddle.nn.Linear(10, 10)
            inp = paddle.uniform(shape=[10, 10], min=-0.1, max=0.1)
            out = linear_1(inp)
            out = linear_2(out)
            loss = paddle.mean(out)
            adam = paddle.optimizer.Adamax(
                learning_rate=0.1,
                parameters=[{
                    'params': linear_1.parameters()
                }, {
                    'params': linear_2.parameters(),
                    'weight_decay': 0.001,
                    'learning_rate': 0.1,
                    'beta1': 0.8
                }],
                weight_decay=0.01,
                beta1=0.9)                   
            out.backward()
            adam.step()
            adam.clear_grad()
    """
    _moment_acc_str = "moment"
    _inf_norm_acc_str = "inf_norm"
    _beta1_pow_acc_str = "beta1_pow_acc"

    def __init__(self,
                 learning_rate=0.001,
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-8,
                 parameters=None,
                 weight_decay=None,
                 grad_clip=None,
                 name=None):
        assert learning_rate is not None
        assert beta1 is not None
        assert beta2 is not None
        assert epsilon is not None
        if not 0 <= beta1 < 1:
            raise ValueError("Invaild value of beta1, expect beta1 in [0,1).")
        if not 0 <= beta2 < 1:
            raise ValueError("Invaild value of beta2, expect beta2 in [0,1).")
        if not 0 <= epsilon:
            raise ValueError("Invaild value of epsilon, expect epsilon >= 0.")
        super(Adamax, self).__init__(
            learning_rate=learning_rate,
            parameters=parameters,
            weight_decay=weight_decay,
            grad_clip=grad_clip,
            name=name)
        self.type = "adamax"
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon
        self._default_dict = {
            'beta1': beta1,
            'beta2': beta2,
            'epsilon': epsilon
        }

    def _create_accumulators(self, block, parameters):
        if isinstance(parameters, dict):
            parameters = self._update_param_group(parameters)

        # Create accumulator tensors for first moment and infinity norm
        for p in parameters:
            self._add_accumulator(self._moment_acc_str, p)
            self._add_accumulator(self._inf_norm_acc_str, p)
            self._add_accumulator(
                name=self._beta1_pow_acc_str,
                param=p,
                fill_value=self._beta1,
                shape=[1])

    def _append_optimize_op(self, block, param_and_grad):
        assert isinstance(block, framework.Block)
        if isinstance(param_and_grad, dict):
            param_and_grad = self._update_param_group(param_and_grad)

        moment = self._get_accumulator(self._moment_acc_str, param_and_grad[0])
        inf_norm = self._get_accumulator(self._inf_norm_acc_str,
                                         param_and_grad[0])
        beta1_pow_acc = self._get_accumulator(self._beta1_pow_acc_str,
                                              param_and_grad[0])
        # create the adamax optimize op
        adamax_op = block.append_op(
            type=self.type,
            inputs={
                "Param": param_and_grad[0],
                "Grad": param_and_grad[1],
                "LearningRate": self._create_param_lr(param_and_grad),
                "Moment": moment,
                "InfNorm": inf_norm,
                "Beta1Pow": beta1_pow_acc
            },
            outputs={
                "ParamOut": param_and_grad[0],
                "MomentOut": moment,
                "InfNormOut": inf_norm
            },
            attrs={
                "beta1": self._beta1,
                "beta2": self._beta2,
                "epsilon": self._epsilon
            },
            stop_gradient=True)

        return adamax_op

    def _finish_update(self, block, parameters_and_grads):
        """Update Beta1 Power accumulator
        """
        assert isinstance(block, framework.Block)
        if isinstance(parameters_and_grads, list):
            for param, grad in parameters_and_grads:
                if grad is None or param.stop_gradient is True:
                    continue
                with param.block.program._optimized_guard(
                    [param, grad]), name_scope('adamax'):
                    beta1_pow_acc = self._get_accumulator(
                        self._beta1_pow_acc_str, param)
                    block.append_op(
                        type="scale",
                        inputs={"X": beta1_pow_acc},
                        outputs={"Out": beta1_pow_acc},
                        attrs={"scale": self._beta1},
                        stop_gradient=True)
        else:
            for param, grad in parameters_and_grads['params']:
                if grad is None or param.stop_gradient is True:
                    continue
                with param.block.program._optimized_guard(
                    [param, grad]), name_scope('adamax'):
                    beta1_pow_acc = self._get_accumulator(
                        self._beta1_pow_acc_str, param)
                    self._beta1 = parameters_and_grads.get(
                        'beta1', self._default_dict['beta1'])
                    block.append_op(
                        type="scale",
                        inputs={"X": beta1_pow_acc},
                        outputs={"Out": beta1_pow_acc},
                        attrs={"scale": self._beta1},
                        stop_gradient=True)

    def _update_param_group(self, parameters):
        self._beta1 = parameters.get('beta1', self._default_dict['beta1'])
        self._beta2 = parameters.get('beta2', self._default_dict['beta2'])
        self._epsilon = parameters.get('epsilon', self._default_dict['epsilon'])
        parameters = parameters.get('params')
        return parameters
