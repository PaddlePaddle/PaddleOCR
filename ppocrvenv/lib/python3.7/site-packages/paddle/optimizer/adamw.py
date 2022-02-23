# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from .adam import Adam
from ..fluid import core
from ..fluid import framework
from ..fluid.framework import Variable
from ..fluid.dygraph import base as imperative_base
from collections.abc import Callable
import paddle

_C_ops = core.ops

__all__ = []


class AdamW(Adam):
    r"""
    The AdamW optimizer is implemented based on the AdamW Optimization
    in paper `DECOUPLED WEIGHT DECAY REGULARIZATION <https://arxiv.org/pdf/1711.05101.pdf>`_.
    it can resolves the problem of L2 regularization failure in the Adam optimizer.

    .. math::

        t & = t + 1

        moment\_1\_out & = {\beta}_1 * moment\_1 + (1 - {\beta}_1) * grad

        moemnt\_2\_out & = {\beta}_2 * moment\_2 + (1 - {\beta}_2) * grad * grad

        learning\_rate & = learning\_rate * 
            \frac{\sqrt{1 - {\beta}_2^t}}{1 - {beta}_1^t}

        param\_out & = param - learning\_rate * (\frac{moment\_1}{\sqrt{moment\_2} + \epsilon} + \lambda * param)


    Args:
        learning_rate (float|LRScheduler, optional): The learning rate used to update ``Parameter``.
            It can be a float value or a LRScheduler. The default value is 0.001.
        parameters (list|tuple, optional): List/Tuple of ``Tensor`` names to update to minimize ``loss``. \
            This parameter is required in dygraph mode. And you can specify different options for \
            different parameter groups such as the learning rate, weight decay, etc, \
            then the parameters are list of dict. Note that the learning_rate in paramter groups \
            represents the scale of base learning_rate. \
	    The default value is None in static mode, at this time all parameters will be updated.
        beta1 (float|Tensor, optional): The exponential decay rate for the 1st moment estimates.
            It should be a float number or a Tensor with shape [1] and data type as float32.
            The default value is 0.9.
        beta2 (float|Tensor, optional): The exponential decay rate for the 2nd moment estimates.
            It should be a float number or a Tensor with shape [1] and data type as float32.
            The default value is 0.999.
        epsilon (float, optional): A small float value for numerical stability.
            The default value is 1e-08.
        weight_decay (float|Tensor, optional): The weight decay coefficient, it can be float or Tensor. The default value is 0.01.
        lr_ratio (function|None, optional): If it is not None, 
            the learning rate will be updated with layerwise learning rate ratio.
            Otherwise, the learning rate is the original.
            Default: None.
        apply_decay_param_fun (function|None, optional): If it is not None,
            only tensors that makes apply_decay_param_fun(Tensor.name)==True
            will be updated with weight decay. It only works when we want to specify tensors.
            Default: None.
        grad_clip (GradientClipBase, optional): Gradient cliping strategy, it's an instance of
            some derived class of ``GradientClipBase`` . There are three cliping strategies
            ( :ref:`api_fluid_clip_GradientClipByGlobalNorm` , :ref:`api_fluid_clip_GradientClipByNorm` ,
            :ref:`api_fluid_clip_GradientClipByValue` ). Default None, meaning there is no gradient clipping.
        lazy_mode (bool, optional): The official Adam algorithm has two moving-average accumulators.
            The accumulators are updated at every step. Every element of the two moving-average
            is updated in both dense mode and sparse mode. If the size of parameter is very large,
            then the update may be very slow. The lazy mode only update the element that has
            gradient in current mini-batch, so it will be much more faster. But this mode has
            different semantics with the original Adam algorithm and may lead to different result.
            The default value is False.
        multi_precision (bool, optional): Whether to use multi-precision during weight updating. Default is false.
        name (str, optional): Normally there is no need for user to set this property.
            For more information, please refer to :ref:`api_guide_Name`.
            The default value is None.
    **Notes**:
        **Currently, AdamW doesn't support sparse parameter optimization.**

    Examples:
        .. code-block:: python
            
            import paddle

            linear = paddle.nn.Linear(10, 10)
            inp = paddle.rand([10,10], dtype="float32")
            out = linear(inp)
            loss = paddle.mean(out)

            beta1 = paddle.to_tensor([0.9], dtype="float32")
            beta2 = paddle.to_tensor([0.99], dtype="float32")

            adam = paddle.optimizer.AdamW(learning_rate=0.1,
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
            adam = paddle.optimizer.AdamW(
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

    def __init__(self,
                 learning_rate=0.001,
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-8,
                 parameters=None,
                 weight_decay=0.01,
                 lr_ratio=None,
                 apply_decay_param_fun=None,
                 grad_clip=None,
                 lazy_mode=False,
                 multi_precision=False,
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
        coeff = weight_decay
        if not isinstance(coeff, float) and \
                not isinstance(coeff, framework.Variable):
            raise TypeError("coeff should be float or Tensor.")
        self._params_name = set()
        self._apply_decay_param_fun = apply_decay_param_fun
        self._coeff = coeff
        self._lr_to_coeff = dict()
        if lr_ratio is not None:
            assert isinstance(lr_ratio, Callable)
            if not core.is_compiled_with_cuda():
                raise NotImplementedError(
                    "'lr_ratio' is unimplemented in CPU, XPU and NPU")
        self._lr_ratio = lr_ratio

        super(AdamW, self).__init__(
            learning_rate=learning_rate,
            parameters=parameters,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
            grad_clip=grad_clip,
            name=name,
            lazy_mode=lazy_mode,
            multi_precision=multi_precision)
        self._default_dict = {'coeff': coeff}

        self.type = "adamw"

        if core.is_compiled_with_xpu():
            self.type = "adam"

        # Use _auxiliary_vars together with _set_auxiliary_var/_get_auxiliary_var to achieve that.
        self._auxiliary_vars = dict()

    def _set_auxiliary_var(self, key, val):
        self._auxiliary_vars[key] = val

    def _get_auxiliary_var(self, key):
        if key in self._auxiliary_vars:
            return self._auxiliary_vars[key]
        else:
            return None

    def _append_decoupled_weight_decay(self, block, param_and_grad):
        """
        Add decoupled weight decay op.
            parameter = parameter - parameter * coeff * lr
        Args:
            block: block in which variable is to be created
            param_and_grad: (parameters, gradients) pairs,
                the parameters need to decay.
        Raises:
            Exception: The type of coeff and parameter is not consistent.
        """
        if isinstance(param_and_grad, dict):
            param_and_grad = self._update_param_group(param_and_grad)
        param, grad = param_and_grad

        if self._apply_decay_param_fun is not None \
                and not self._apply_decay_param_fun(param.name):
            return

        if isinstance(self._learning_rate, float):
            learning_rate = self._learning_rate
        else:
            # NOTE. We add this function to the _append_optimize_op(),
            # for we must make sure _create_param_lr() be called after
            # optimizer._create_global_learning_rate().
            learning_rate = self._create_param_lr(param_and_grad)

        with block.program._optimized_guard(
            [param, grad]), framework.name_scope('weight decay'):
            self._params_name.add(param.name)

            # If it has been calculated, the result will be reused.
            # NOTE(wangxi): In dygraph mode, apply_gradient will be executed
            # every step, so need clear _lr_to_coeff every step,
            # we do this in _create_optimization_pass
            decay_coeff = self._lr_to_coeff.get(learning_rate, None)
            if decay_coeff is None:
                # NOTE(wangxi): for pipeline to set device:all
                with paddle.static.device_guard(None):
                    decay_coeff = 1.0 - learning_rate * self._coeff
                self._lr_to_coeff[learning_rate] = decay_coeff

            find_master = (self._multi_precision and
                           param.dtype == core.VarDesc.VarType.FP16)
            if find_master:
                master_weight = self._master_weights[param.name]
                scaled_param = master_weight * decay_coeff
                paddle.fluid.layers.assign(
                    input=scaled_param, output=master_weight)
            else:
                scaled_param = param * decay_coeff
                paddle.fluid.layers.assign(input=scaled_param, output=param)

    def _append_optimize_op(self, block, param_and_grad):
        if paddle.is_compiled_with_xpu():
            self._append_decoupled_weight_decay(block, param_and_grad)
            return super(AdamW, self)._append_optimize_op(block, param_and_grad)

        assert isinstance(block, framework.Block)
        if isinstance(param_and_grad, dict):
            param_and_grad = self._update_param_group(param_and_grad)
        param, grad = param_and_grad

        # Whether we should do weight decay for the parameter.
        with_decay = True
        if self._apply_decay_param_fun is not None \
                and not self._apply_decay_param_fun(param.name):
            with_decay = False

        moment1 = self._get_accumulator(self._moment1_acc_str,
                                        param_and_grad[0])
        moment2 = self._get_accumulator(self._moment2_acc_str,
                                        param_and_grad[0])
        beta1_pow_acc = self._get_accumulator(self._beta1_pow_acc_str,
                                              param_and_grad[0])
        beta2_pow_acc = self._get_accumulator(self._beta2_pow_acc_str,
                                              param_and_grad[0])
        find_master = self._multi_precision and param_and_grad[
            0].dtype == core.VarDesc.VarType.FP16
        master_weight = (self._master_weights[param_and_grad[0].name]
                         if find_master else None)
        lr = self._create_param_lr(param_and_grad)

        # create the adamw optimize op
        if framework.in_dygraph_mode():
            lr_ratio_ = 1. if self._lr_ratio is None else self._lr_ratio(
                param_and_grad[0])

            _beta1 = self._beta1 if not isinstance(
                self._beta1, Variable) else self._beta1.numpy().item(0)
            _beta2 = self._beta2 if not isinstance(
                self._beta2, Variable) else self._beta2.numpy().item(0)

            _, _, _, _, _, _ = _C_ops.adamw(
                param_and_grad[0], param_and_grad[1], lr, moment1, moment2,
                beta1_pow_acc, beta2_pow_acc, master_weight, param_and_grad[0],
                moment1, moment2, beta1_pow_acc, beta2_pow_acc, master_weight,
                'epsilon', self._epsilon, 'lazy_mode', self._lazy_mode,
                'min_row_size_to_use_multithread', 1000, 'beta1', _beta1,
                'beta2', _beta2, "with_decay", with_decay, 'coeff', self._coeff,
                'multi_precision', find_master, 'lr_ratio', lr_ratio_)
            return None

        inputs = {
            "Param": [param_and_grad[0]],
            "Grad": [param_and_grad[1]],
            "LearningRate": [lr],
            "Moment1": [moment1],
            "Moment2": [moment2],
            "Beta1Pow": [beta1_pow_acc],
            "Beta2Pow": [beta2_pow_acc],
        }

        # Pass found_inf to adamw, to skip update for not only param, but also momentum and beta_pow
        found_inf = self._get_auxiliary_var('found_inf')

        if found_inf:
            inputs['SkipUpdate'] = found_inf

        outputs = {
            "ParamOut": [param_and_grad[0]],
            "Moment1Out": [moment1],
            "Moment2Out": [moment2],
            "Beta1PowOut": [beta1_pow_acc],
            "Beta2PowOut": [beta2_pow_acc],
        }
        attrs = {
            "lazy_mode": self._lazy_mode,
            "min_row_size_to_use_multithread": 1000,
            "multi_precision": find_master,
            "with_decay": with_decay,
            "coeff": self._coeff,
            "lr_ratio": 1.
            if self._lr_ratio is None else self._lr_ratio(param_and_grad[0])
        }

        if isinstance(self._beta1, Variable):
            inputs['Beta1Tensor'] = self._beta1
        else:
            attrs['beta1'] = self._beta1
        if isinstance(self._beta2, Variable):
            inputs['Beta2Tensor'] = self._beta2
        else:
            attrs['beta2'] = self._beta2
        if isinstance(self._epsilon, Variable):
            inputs['EpsilonTensor'] = self._epsilon
        else:
            attrs['epsilon'] = self._epsilon

        if find_master:
            inputs["MasterParam"] = master_weight
            outputs["MasterParamOut"] = master_weight

        adamw_op = block.append_op(
            type=self.type,
            inputs=inputs,
            outputs=outputs,
            attrs=attrs,
            stop_gradient=True)

        return adamw_op

    def _create_optimization_pass(self, parameters_and_grads):
        optimize_ops = super(
            AdamW, self)._create_optimization_pass(parameters_and_grads)
        # In dygraph mode, clear _lr_to_coeff after applied gradient
        self._lr_to_coeff = dict()
        return optimize_ops

    def __str__(self):
        return " ".join(["Weight Decay, params:", ",".join(self._params_name)])

    def _update_param_group(self, parameters):
        self._coeff = parameters.get('coeff', self._default_dict['coeff'])
        parameters = parameters.get('params')
        return parameters
