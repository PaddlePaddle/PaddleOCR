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
import paddle.fluid
from paddle.fluid import framework as framework

__all__ = ["extend_with_decoupled_weight_decay"]


class DecoupledWeightDecay(object):
    def __init__(self, coeff=0.0, apply_decay_param_fun=None, **kwargs):
        if not isinstance(coeff, float) and \
                not isinstance(coeff, framework.Variable):
            raise TypeError("coeff should be float or Variable.")
        self._params_name = set()
        self._apply_decay_param_fun = apply_decay_param_fun
        self._coeff = coeff
        super(DecoupledWeightDecay, self).__init__(**kwargs)

    def _scale_parameters(self, params_and_grads):
        """
        Adds weight decay ops.
            scaled_parameter = parameter * coeff

        Args:
            params_and_grads: A list of (parameters, gradients) pairs,
                the parameters need to decay.
        Raises:
            Exception: The type of coeff and parameter is not consistent.
        """
        if isinstance(self._coeff, float) and self._coeff == 0.0:
            return

        scaled_params = []
        for param, grad in params_and_grads:
            # If no gradient then we don't need to do anything
            if grad is None:
                continue
            if self._apply_decay_param_fun is not None \
                    and not self._apply_decay_param_fun(param.name):
                continue

            if isinstance(self._coeff, float):
                assert param.dtype is not paddle.fluid.core.VarDesc.VarType.FP32, \
                    "the type of coeff(float) and parameter(%s) is not consistent."%(self._coeff.dtype)
            else:
                assert self._coeff.dtype == param.dtype, \
                    "the type of coeff(%s) and parameter(%s) is not consistent."%(self._coeff.dtype, param.dtype)

            with param.block.program._optimized_guard(
                [param, grad]), framework.name_scope('weight decay'):
                assert param.name not in self._params_name
                scaled_params.append((param, grad, param * self._coeff))
                self._params_name.add(param.name)
        return scaled_params

    def backward(self, **kargs):
        return super(DecoupledWeightDecay, self).backward(**kargs)

    def apply_optimize(self, **kargs):
        return super(DecoupledWeightDecay, self).apply_optimize(**kargs)

    def minimize(self,
                 loss,
                 startup_program=None,
                 parameter_list=None,
                 no_grad_set=None):
        params_grads = self.backward(
            loss=loss,
            startup_program=startup_program,
            parameter_list=parameter_list,
            no_grad_set=no_grad_set)
        scaled_params = self._scale_parameters(params_grads)
        for p_grad_sgrad in scaled_params:
            param, grad, scaled_param = p_grad_sgrad
            with param.block.program._optimized_guard(
                [param, grad]), framework.name_scope('weight decay'):
                updated_param = paddle.fluid.layers.elementwise_sub(
                    x=param, y=scaled_param)
                paddle.fluid.layers.assign(input=updated_param, output=param)

        optimize_ops = self.apply_optimize(
            loss=loss,
            params_grads=params_grads,
            startup_program=startup_program)
        return optimize_ops, params_grads

    def __str__(self):
        return " ".join(["Weight Decay, params:", ",".join(self._params_name)])


def extend_with_decoupled_weight_decay(base_optimizer):
    """
    extend_with_decoupled_weight_decay is a decorator function, it returns an
    optimizer class with decoupled weight decay. The returned optimizer will
    apply weight decay on the optimized parameters with the parameters before
    optimization, i.e: new_parameter = optimized_parameter - parameter * coeff.
    The details of decoupled weight decay yplease refer to this
    `DECOUPLED WEIGHT DECAY REGULARIZATION <https://arxiv.org/pdf/1711.05101.pdf>`_.

    Args:
        base_optimizer (Optimizer): The base_optimizer should be a derived class of Optimizer.

    Returns:
        OptimizerWithDecoupledWeightDecay: the optimizer with decouple weight decay.

    Examples:

      .. code-block:: python

        AdamW = fluid.contrib.extend_with_decoupled_weight_decay(
            fluid.optimizer.Adam)
        optimizer = AdamW(learning_rate=0.1,
                          weight_decay=0.01)

        optimizer.minimize(cost)
    """
    if not issubclass(base_optimizer, paddle.fluid.optimizer.Optimizer):
        raise TypeError(
            "The input(base_optimizer) should be a derived class of Optimizer.")

    class OptimizerWithDecoupledWeightDecay(DecoupledWeightDecay,
                                            base_optimizer):
        """
        OptimizerWithDecoupledWeightDecay is used to update the optimized parameters
        with the parameters before optimization. For more information, please refer:
        https://arxiv.org/pdf/1711.05101.pdf.

        Args:
            weight_decay (float|Variable): The weight decay coefficient, it can be
                float or Variable.
            apply_decay_param_fun (function|None): If it is not None,
                only variables that makes apply_decay_param_fun(variable)==True
                will be updated. It only works when we want to specify variables.
                Default: None.
        """

        def __init__(self, weight_decay, apply_decay_param_fun=None, **kwargs):
            super(OptimizerWithDecoupledWeightDecay, self).__init__(
                weight_decay, apply_decay_param_fun, **kwargs)

    return OptimizerWithDecoupledWeightDecay
