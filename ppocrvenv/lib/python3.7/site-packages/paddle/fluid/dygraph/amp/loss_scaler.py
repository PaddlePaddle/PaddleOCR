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

from __future__ import print_function
from paddle.fluid import core
from paddle.fluid.dygraph import to_variable
from paddle.fluid.framework import _varbase_creator, _dygraph_tracer, dygraph_only
from paddle.fluid.data_feeder import check_type
from ...wrapped_decorator import signature_safe_contextmanager, wrap_decorator
import warnings
import numpy as np
from paddle import _C_ops
from collections import defaultdict
from enum import Enum

__all__ = ['AmpScaler', 'OptimizerState']


class OptimizerState(Enum):
    INIT = 0
    UNSCALED = 1
    STEPPED = 2


def _refresh_optimizer_state():
    return {"state": OptimizerState.INIT}


class AmpScaler(object):
    """
    :api_attr: imperative

    AmpScaler is used for Auto-Mixed-Precision training/inferring in imperative
    mode. It controls the scaling of loss, helps avoiding numerical overflow.
    The object of this class has seventeen methods `scale()`, `unscale_()`, `minimize()` and `get`/`set` api of parameters.

    `scale()` is used to multiply the loss by a scale ratio.
    `unscale_()` is used to unscale the gradients of parameters, multiplies the gradients of parameters by 1/(scale ratio)
    `minimize()` is similar as `optimizer.minimize()`, performs parameters updating, and it will update the loss_scaling.

    Commonly, it is used together with `amp_guard` to achieve Auto-Mixed-Precision in 
    imperative mode.

    Args:
        enable(bool, optional): Enable loss scaling or not. Default is True.
        init_loss_scaling (float, optional): The initial loss scaling factor. Default is 2**15.
        incr_ratio(float, optional): The multiplier to use when increasing the loss 
                        scaling. Default is 2.0.
        decr_ratio(float, optional): The less-than-one-multiplier to use when decreasing 
                        the loss scaling. Default is 0.5.
        incr_every_n_steps(int, optional): Increases loss scaling every n consecutive 
                                steps with finite gradients. Default is 1000.
        decr_every_n_nan_or_inf(int, optional): Decreases loss scaling every n 
                                    accumulated steps with nan or inf gradients. Default is 2.
        use_dynamic_loss_scaling(bool, optional): Whether to use dynamic loss scaling. If False, fixed loss_scaling is used. If True, the loss scaling is updated dynamicly. Default is True.
    Returns:
        An AmpScaler object.

    Examples:

     .. code-block:: python

        import numpy as np
        import paddle.fluid as fluid

        data = np.random.uniform(-1, 1, [10, 3, 32, 32]).astype('float32')
        with fluid.dygraph.guard():
            model = fluid.dygraph.Conv2D(3, 2, 3)
            optimizer = fluid.optimizer.SGDOptimizer(
                    learning_rate=0.01, parameter_list=model.parameters())
            scaler = fluid.dygraph.AmpScaler(init_loss_scaling=1024)
            data = fluid.dygraph.to_variable(data)
            with fluid.dygraph.amp_guard():
                conv = model(data)
                loss = fluid.layers.reduce_mean(conv)
                scaled = scaler.scale(loss)
                scaled.backward()
                scaler.minimize(optimizer, scaled)         
    """

    @dygraph_only
    def __init__(self,
                 enable=True,
                 init_loss_scaling=2.**15,
                 incr_ratio=2.0,
                 decr_ratio=0.5,
                 incr_every_n_steps=1000,
                 decr_every_n_nan_or_inf=1,
                 use_dynamic_loss_scaling=True):

        tracer = _dygraph_tracer()
        if not tracer:
            raise ValueError(
                "current_tracer is None, maybe it is not in imperative mode.")

        if enable and not (tracer._expected_place.is_gpu_place() or
                           tracer._expected_place.is_xpu_place()):
            warnings.warn(
                'AmpScaler can only be enabled on CUDAPlace and XPUPlace, current place is %s, so it makes no effect.'
                % tracer._expected_place)
            enable = False

        self._enable = enable

        if self._enable:
            assert incr_ratio > 1.0, "The incr_ratio must be > 1.0."
            assert decr_ratio < 1.0, "The decr_ratio must be < 1.0."

            self._init_loss_scaling = init_loss_scaling
            self._incr_ratio = incr_ratio
            self._decr_ratio = decr_ratio
            self._incr_every_n_steps = incr_every_n_steps
            self._decr_every_n_nan_or_inf = decr_every_n_nan_or_inf
            self._incr_count = 0
            self._decr_count = 0
            self._use_dynamic_loss_scaling = use_dynamic_loss_scaling

            self._found_inf = to_variable(np.array([0]).astype(np.bool))
            self._scale = to_variable(
                np.array([self._init_loss_scaling]).astype(np.float32))
            self._cache_founf_inf = None
            self._optimizer_states = defaultdict(_refresh_optimizer_state)

    def scale(self, var):
        """
        Multiplies a variable(Tensor) by the scale factor and returns scaled outputs.  
        If this instance of :class:`AmpScaler` is not enabled, output are returned unmodified.

        Args:
            var (Variable):  The variable to scale.
        Returns:
            The scaled variable or original variable.
        
        Examples:

            .. code-block:: python

                import numpy as np
                import paddle.fluid as fluid

                data = np.random.uniform(-1, 1, [10, 3, 32, 32]).astype('float32')
                with fluid.dygraph.guard():
                    model = fluid.dygraph.Conv2D(3, 2, 3)
                    optimizer = fluid.optimizer.SGDOptimizer(
                            learning_rate=0.01, parameter_list=model.parameters())
                    scaler = fluid.dygraph.AmpScaler(init_loss_scaling=1024)
                    data = fluid.dygraph.to_variable(data)
                    with fluid.dygraph.amp_guard():
                        conv = model(data)
                        loss = fluid.layers.reduce_mean(conv)
                        scaled = scaler.scale(loss)
                        scaled.backward()
                        scaler.minimize(optimizer, scaled) 
        """
        check_type(var, "var", core.VarBase, 'AmpScaler.scale()')

        if not self._enable:
            return var

        return var * self._scale

    def minimize(self, optimizer, *args, **kwargs):
        """
        This function is similar as `Optimizer.minimize()`, which performs parameters updating.
        
        If the scaled gradients of parameters contains NAN or INF, the parameters updating is skipped.
        Otherwise, if `unscale_()` has not been called, it first unscales the scaled gradients of parameters, then updates the parameters.

        Finally, the loss scaling ratio is updated.

        Args:
            optimizer(Optimizer):  The optimizer used to update parameters.
            args:  Arguments, which will be forward to `optimizer.minimize()`.
            kwargs: Keyword arguments, which will be forward to `Optimizer.minimize()`.

        Examples:

            .. code-block:: python

                import numpy as np
                import paddle.fluid as fluid

                data = np.random.uniform(-1, 1, [10, 3, 32, 32]).astype('float32')
                with fluid.dygraph.guard():
                    model = fluid.dygraph.Conv2D(3, 2, 3)
                    optimizer = fluid.optimizer.SGDOptimizer(
                            learning_rate=0.01, parameter_list=model.parameters())
                    scaler = fluid.dygraph.AmpScaler(init_loss_scaling=1024)
                    data = fluid.dygraph.to_variable(data)
                    with fluid.dygraph.amp_guard():
                        conv = model(data)
                        loss = fluid.layers.reduce_mean(conv)
                        scaled = scaler.scale(loss)
                        scaled.backward()
                        scaler.minimize(optimizer, scaled) 
        """
        if not self._enable:
            return optimizer.minimize(*args, **kwargs)

        optimizer_state = self._optimizer_states[id(optimizer)]

        #  unscale the grad
        if optimizer_state["state"] is OptimizerState.INIT:
            self._unscale(optimizer)

        optimize_ops, params_grads = (None, None)

        if self._found_inf:
            self._cache_founf_inf = True
        else:
            optimize_ops, params_grads = optimizer.minimize(*args, **kwargs)
            self._cache_founf_inf = False

        if self._use_dynamic_loss_scaling:
            # uopdate the scale
            self._update()

        self._optimizer_states = defaultdict(_refresh_optimizer_state)

        return optimize_ops, params_grads

    def _unscale(self, optimizer):
        """
        Unscale the gradients of parameters, multiplies the gradients of parameters by 1/(loss scaling ratio).  
        If this instance of :class:`GradScaler` is not enabled, output are returned unmodified.
        Args:
            optimizer(Optimizer):  The optimizer used to update parameters.
        Returns:
            The unscaled parameters or original parameters.
        """
        if not self._enable:
            return

        optimizer_state = self._optimizer_states[id(optimizer)]

        if optimizer_state["state"] is OptimizerState.UNSCALED:
            raise RuntimeError(
                "unscale_() has already been called on this optimizer since the last update()."
            )
        elif optimizer_state["state"] is OptimizerState.STEPPED:
            raise RuntimeError("unscale_() is being called after step().")

        if getattr(optimizer, '_param_groups', None) and isinstance(
                optimizer._param_groups[0], dict):
            param_grads = []
            param_grads_fp16 = []
            param_grads_fp32 = []
            for group in optimizer._param_groups:
                for param in group['params']:
                    if param._grad_ivar() is not None:
                        param_grads.append(param._grad_ivar())
                        if param._grad_ivar(
                        ).dtype == core.VarDesc.VarType.FP16:
                            param_grads_fp16.append(param._grad_ivar())
                        else:
                            param_grads_fp32.append(param._grad_ivar())
        else:
            param_grads = [
                param._grad_ivar() for param in optimizer._parameter_list
                if param._grad_ivar() is not None
            ]
            param_grads_fp16 = [
                param._grad_ivar() for param in optimizer._parameter_list
                if (param._grad_ivar() is not None
                    ) and (param._grad_ivar().dtype == core.VarDesc.VarType.FP16
                           )
            ]
            param_grads_fp32 = [
                param._grad_ivar() for param in optimizer._parameter_list
                if (param._grad_ivar() is not None
                    ) and (param._grad_ivar().dtype == core.VarDesc.VarType.FP32
                           )
            ]
        temp_found_inf_fp16 = to_variable(np.array([0]).astype(np.bool))
        temp_found_inf_fp32 = to_variable(np.array([0]).astype(np.bool))
        if len(param_grads_fp16):
            _C_ops.check_finite_and_unscale(param_grads_fp16, self._scale,
                                            param_grads_fp16,
                                            temp_found_inf_fp16)
        if len(param_grads_fp32):
            _C_ops.check_finite_and_unscale(param_grads_fp32, self._scale,
                                            param_grads_fp32,
                                            temp_found_inf_fp32)
        self._found_inf = temp_found_inf_fp16 or temp_found_inf_fp32

        optimizer_state["state"] = OptimizerState.UNSCALED

    def _update(self):
        """
        Updates the loss_scaling.
        """
        if not self._enable:
            return

        if self._cache_founf_inf:
            self._incr_count = 0
            self._decr_count = self._decr_count + 1
            if self._decr_count == self._decr_every_n_nan_or_inf:
                print(
                    'Found inf or nan, current scale is: {}, decrease to: {}*{}'.
                    format(
                        float(self._scale),
                        float(self._scale), float(self._decr_ratio)))
                self._scale = self._scale * self._decr_ratio
                self._decr_count = 0
        else:
            self._decr_count = 0
            self._incr_count = self._incr_count + 1
            if self._incr_count == self._incr_every_n_steps:
                self._scale = self._scale * self._incr_ratio
                self._incr_count = 0

        return

    def is_enable(self):
        """
        Enable loss scaling or not.

        Returns:
            bool: enable loss scaling return True else return False.
        """
        return self._enable

    def is_use_dynamic_loss_scaling(self):
        """
        Whether to use dynamic loss scaling.

        Returns:
            bool: if fixed loss_scaling is used return False, if the loss scaling is updated dynamicly return true.
        """
        return self._use_dynamic_loss_scaling

    def get_init_loss_scaling(self):
        """
        Return the initial loss scaling factor.

        Reurns:
            float:  the initial loss scaling factor.
        """
        return self._init_loss_scaling

    def set_init_loss_scaling(self, new_init_loss_scaling):
        """
        Set the initial loss scaling factor by `new_init_loss_scaling`.

        Args:
            new_init_loss_scaling(int):  The new_init_loss_scaling used to update initial loss scaling factor.s
        """
        self._init_loss_scaling = new_init_loss_scaling
        self._scale = to_variable(
            np.array([self._init_loss_scaling]).astype(np.float32))

    def get_incr_ratio(self):
        """
        Return the multiplier to use when increasing the loss scaling.

        Reurns:
            float:  the multiplier to use when increasing the loss scaling.
        """
        return self._incr_ratio

    def set_incr_ratio(self, new_incr_ratio):
        """
        Set the multiplier to use when increasing the loss scaling by `new_incr_ratio`, `new_incr_ratio` should > 1.0.

        Args:
            new_incr_ratio(float):  The new_incr_ratio used to update the multiplier to use when increasing the loss scaling.
        """
        assert new_incr_ratio > 1.0, "The new_incr_ratio must be > 1.0."
        self._incr_ratio = new_incr_ratio

    def get_decr_ratio(self):
        """
        Get the less-than-one-multiplier to use when decreasing the loss scaling.

        Reurns:
            float:  the less-than-one-multiplier to use when decreasing the loss scaling.
        """
        return self._decr_ratio

    def set_decr_ratio(self, new_decr_ratio):
        """
        Set the less-than-one-multiplier to use when decreasing the loss scaling by `new_incr_ratio`, `new_decr_ratio` should < 1.0.

        Args:
            new_decr_ratio(float):  The new_decr_ratio used to update the less-than-one-multiplier to use when decreasing the loss scaling.
        """
        assert new_decr_ratio < 1.0, "The new_decr_ratio must be < 1.0."
        self._decr_ratio = new_decr_ratio

    def get_incr_every_n_steps(self):
        """
        Return the num `n`, `n` represent increases loss scaling every `n` consecutive steps with finite gradients.

        Reurns:
            int:  the num `n`, `n` represent increases loss scaling every `n` consecutive steps with finite gradients.
        """
        return self._incr_every_n_steps

    def set_incr_every_n_steps(self, new_incr_every_n_steps):
        """
        Set the num `n` by `new_incr_every_n_steps`, `n` represent increases loss scaling every `n` consecutive steps with finite gradients.

        Args:
            new_incr_every_n_steps(int):  The new_incr_every_n_steps used to update the num `n`, `n` represent increases loss scaling every `n` consecutive steps with finite gradients.
        """
        self._incr_every_n_steps = new_incr_every_n_steps

    def get_decr_every_n_nan_or_inf(self):
        """
        Return the num `n`, `n` represent decreases loss scaling every `n` accumulated steps with nan or inf gradients.

        Reurns:
            int:  the num `n`, `n` represent decreases loss scaling every `n` accumulated steps with nan or inf gradients.
        """
        return self._decr_every_n_nan_or_inf

    def set_decr_every_n_nan_or_inf(self, new_decr_every_n_nan_or_inf):
        """
        Set the num `n` by `new_decr_every_n_nan_or_inf`, `n` represent decreases loss scaling every `n` accumulated steps with nan or inf gradients.

        Args:
            new_decr_every_n_nan_or_inf(int):  The new_decr_every_n_nan_or_inf used to update the num `n`, `n` represent decreases loss scaling every `n` accumulated steps with nan or inf gradients.
        """
        self._decr_every_n_nan_or_inf = new_decr_every_n_nan_or_inf

    def state_dict(self):
        """
        Returns the state of the scaler as a `dict`, If this instance is not enabled, returns an empty dict.

        Reurns:
            A dict of scaler includes:
            scale (tensor): The loss scaling factor.
            incr_ratio(float): The multiplier to use when increasing the loss scaling.
            decr_ratio(float): The less-than-one-multiplier to use when decreasing the loss scaling.
            incr_every_n_steps(int): Increases loss scaling every n consecutive steps with finite gradients.
            decr_every_n_nan_or_inf(int): Decreases loss scaling every n accumulated steps with nan or inf gradients.
            incr_count(int): The number of recent consecutive unskipped steps.
            decr_count(int): The number of recent consecutive skipped steps.
            use_dynamic_loss_scaling(bool): Whether to use dynamic loss scaling. If False, fixed loss_scaling is used. If True, the loss scaling is updated dynamicly. Default is True.
        """
        return {
            "scale": self._scale.numpy(),
            "incr_ratio": self._incr_ratio,
            "decr_ratio": self._decr_ratio,
            "incr_every_n_steps": self._incr_every_n_steps,
            "decr_every_n_nan_or_inf": self._decr_every_n_nan_or_inf,
            "incr_count": self._incr_count,
            "decr_count": self._decr_count,
            "use_dynamic_loss_scaling": self._use_dynamic_loss_scaling
        } if self._enable else {}

    def load_state_dict(self, state_dict):
        """
        Loads the scaler state.
        
        Args:
           state_dict(dict): scaler state.  Should be an object returned from a call to `AmpScaler.state_dict()`.
        """
        if not self._enable:
            return

        if len(state_dict) == 0:
            raise RuntimeError(
                "The input state dict is empty, possibly because it was saved "
                "from a disabled instance of GradScaler.")

        self._init_loss_scaling = state_dict["scale"][0]
        self._scale = to_variable(
            np.array([self._init_loss_scaling]).astype(np.float32))
        self._incr_ratio = state_dict["incr_ratio"]
        self._decr_ratio = state_dict["decr_ratio"]
        self._incr_every_n_steps = state_dict["incr_every_n_steps"]
        self._decr_every_n_nan_or_inf = state_dict["decr_every_n_nan_or_inf"]
        self._incr_count = state_dict["incr_count"]
        self._decr_count = state_dict["decr_count"]
        self._use_dynamic_loss_scaling = state_dict["use_dynamic_loss_scaling"]
