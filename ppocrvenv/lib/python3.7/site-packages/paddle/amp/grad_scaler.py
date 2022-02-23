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

from paddle.fluid.dygraph.amp import AmpScaler
from paddle.fluid.dygraph.amp import OptimizerState
from collections import defaultdict

__all__ = []


def _refresh_optimizer_state():
    return {"state": OptimizerState.INIT}


class GradScaler(AmpScaler):
    """
    GradScaler is used for Auto-Mixed-Precision training in dynamic graph mode. 
    It controls the scaling of loss, helps avoiding numerical overflow.
    The object of this class has nineteen methods `scale()`, `unscale_()`, `minimize()`, `step()`, `update()` and `get`/`set` api of parameters.

    `scale()` is used to multiply the loss by a scale ratio.
    `unscale_()` is used to unscale the gradients of parameters, multiplies the gradients of parameters by 1/(scale ratio)
    `minimize()` is similar as `optimizer.minimize()`, performs parameters updating, and it will update the loss_scaling, it equal to `step()` + `update()`.
    `step()` is similar as `optimizer.step()`, which performs parameters updating.
    `update` is used to update the loss_scaling.


    Commonly, it is used together with `paddle.amp.auto_cast` to achieve Auto-Mixed-Precision in 
    dynamic graph mode.

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
        An GradScaler object.

    Examples:

        .. code-block:: python
            
            import paddle

            model = paddle.nn.Conv2D(3, 2, 3, bias_attr=True)
            optimizer = paddle.optimizer.SGD(learning_rate=0.01, parameters=model.parameters())
            scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
            data = paddle.rand([10, 3, 32, 32])

            with paddle.amp.auto_cast():
                conv = model(data)
                loss = paddle.mean(conv)
                
            scaled = scaler.scale(loss)  # scale the loss 
            scaled.backward()            # do backward
            scaler.minimize(optimizer, scaled)  # update parameters     
            optimizer.clear_grad()
    """

    def __init__(self,
                 enable=True,
                 init_loss_scaling=2.**15,
                 incr_ratio=2.0,
                 decr_ratio=0.5,
                 incr_every_n_steps=1000,
                 decr_every_n_nan_or_inf=2,
                 use_dynamic_loss_scaling=True):
        super(GradScaler, self).__init__(enable, init_loss_scaling, incr_ratio,
                                         decr_ratio, incr_every_n_steps,
                                         decr_every_n_nan_or_inf,
                                         use_dynamic_loss_scaling)

    def scale(self, var):
        """
        Multiplies a Tensor by the scale factor and returns scaled outputs.  
        If this instance of :class:`GradScaler` is not enabled, output are returned unmodified.

        Args:
            var (Tensor):  The tensor to scale.
        Returns:
            The scaled tensor or original tensor.
        
        Examples:

            .. code-block:: python
                
                import paddle

                model = paddle.nn.Conv2D(3, 2, 3, bias_attr=True)
                optimizer = paddle.optimizer.SGD(learning_rate=0.01, parameters=model.parameters())
                scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
                data = paddle.rand([10, 3, 32, 32])

                with paddle.amp.auto_cast():
                    conv = model(data)
                    loss = paddle.mean(conv)

                scaled = scaler.scale(loss)  # scale the loss 
                scaled.backward()            # do backward
                scaler.minimize(optimizer, scaled)  # update parameters  
                optimizer.clear_grad()
        """
        return super(GradScaler, self).scale(var)

    def minimize(self, optimizer, *args, **kwargs):
        """
        This function is similar as `optimizer.minimize()`, which performs parameters updating.
        
        If the scaled gradients of parameters contains NAN or INF, the parameters updating is skipped.
        Otherwise, if `unscale_()` has not been called, it first unscales the scaled gradients of parameters, then updates the parameters.

        Finally, the loss scaling ratio is updated.

        Args:
            optimizer(Optimizer):  The optimizer used to update parameters.
            args:  Arguments, which will be forward to `optimizer.minimize()`.
            kwargs: Keyword arguments, which will be forward to `optimizer.minimize()`.

        Examples:

            .. code-block:: python

                import paddle

                model = paddle.nn.Conv2D(3, 2, 3, bias_attr=True)
                optimizer = paddle.optimizer.SGD(learning_rate=0.01, parameters=model.parameters())
                scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
                data = paddle.rand([10, 3, 32, 32])

                with paddle.amp.auto_cast():
                    conv = model(data)
                    loss = paddle.mean(conv)

                scaled = scaler.scale(loss)  # scale the loss 
                scaled.backward()            # do backward
                scaler.minimize(optimizer, scaled)  # update parameters  
                optimizer.clear_grad()
        """
        return super(GradScaler, self).minimize(optimizer, *args, **kwargs)

    def step(self, optimizer):
        """
        This function is similar as `optimizer.step()`, which performs parameters updating.
        
        If the scaled gradients of parameters contains NAN or INF, the parameters updating is skipped.
        Otherwise, if `unscale_()` has not been called, it first unscales the scaled gradients of parameters, then updates the parameters.

        Args:
            optimizer(Optimizer):  The optimizer used to update parameters.

        Examples:

            .. code-block:: python
            
                # required: gpu
                import paddle

                model = paddle.nn.Conv2D(3, 2, 3, bias_attr=True)
                optimizer = paddle.optimizer.SGD(learning_rate=0.01, parameters=model.parameters())
                scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
                data = paddle.rand([10, 3, 32, 32])
                with paddle.amp.auto_cast():
                    conv = model(data)
                    loss = paddle.mean(conv)
                scaled = scaler.scale(loss)  # scale the loss 
                scaled.backward()            # do backward
                scaler.step(optimizer)       # update parameters
                scaler.update()              # update the loss scaling ratio
                optimizer.clear_grad()
        """
        if not self._enable:
            return optimizer.step()

        optimizer_state = self._optimizer_states[id(optimizer)]
        if optimizer_state["state"] is OptimizerState.STEPPED:
            raise RuntimeError(
                "step() has already been called since the last update().")

        #  unscale the grad
        if optimizer_state["state"] is OptimizerState.INIT:
            self._unscale(optimizer)

        if self._found_inf:
            self._cache_founf_inf = True
        else:
            optimizer.step()
            self._cache_founf_inf = False

        optimizer_state["state"] = OptimizerState.STEPPED

        if not self._use_dynamic_loss_scaling:
            self._optimizer_states = defaultdict(_refresh_optimizer_state)

    def update(self):
        """
        Updates the loss_scaling.
        
        Examples:

            .. code-block:: python
            
                # required: gpu
                import paddle

                model = paddle.nn.Conv2D(3, 2, 3, bias_attr=True)
                optimizer = paddle.optimizer.SGD(learning_rate=0.01, parameters=model.parameters())
                scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
                data = paddle.rand([10, 3, 32, 32])
                with paddle.amp.auto_cast():
                    conv = model(data)
                    loss = paddle.mean(conv)
                scaled = scaler.scale(loss)     # scale the loss 
                scaled.backward()               # do backward
                scaler.step(optimizer)          # update parameters
                scaler.update()                 # update the loss scaling ratio
                optimizer.clear_grad() 
        """
        if not self._enable:
            return
        if self._use_dynamic_loss_scaling:
            self._update()
            self._optimizer_states = defaultdict(_refresh_optimizer_state)
        return

    def unscale_(self, optimizer):
        """
        Unscale the gradients of parameters, multiplies the gradients of parameters by 1/(loss scaling ratio).  
        If this instance of :class:`GradScaler` is not enabled, output are returned unmodified.

        Args:
            optimizer(Optimizer):  The optimizer used to update parameters.

        Returns:
            The unscaled parameters or original parameters.
        
        Examples:

            .. code-block:: python

                # required: gpu
                import paddle

                model = paddle.nn.Conv2D(3, 2, 3, bias_attr=True)
                optimizer = paddle.optimizer.SGD(learning_rate=0.01, parameters=model.parameters())
                scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
                data = paddle.rand([10, 3, 32, 32])
                with paddle.amp.auto_cast():
                    conv = model(data)
                    loss = paddle.mean(conv)
                scaled = scaler.scale(loss)  # scale the loss 
                scaled.backward()            # do backward
                scaler.unscale_(optimizer)    # unscale the parameter
                scaler.step(optimizer)
                scaler.update()  
                optimizer.clear_grad() 
        """
        return super(GradScaler, self)._unscale(optimizer)

    def is_enable(self):
        """
        Enable loss scaling or not.

        Returns:
            bool: enable loss scaling return True else return False.
        
        Examples:
            .. code-block:: python

                # required: gpu,xpu
                import paddle
                scaler = paddle.amp.GradScaler(enable=True,
                                               init_loss_scaling=1024,
                                               incr_ratio=2.0,
                                               decr_ratio=0.5,
                                               incr_every_n_steps=1000,
                                               decr_every_n_nan_or_inf=2,
                                               use_dynamic_loss_scaling=True)
                enable = scaler.is_enable()
                print(enable) # True
        """
        return super(GradScaler, self).is_enable()

    def is_use_dynamic_loss_scaling(self):
        """
        Whether to use dynamic loss scaling.

        Returns:
            bool: if fixed loss_scaling is used return False, if the loss scaling is updated dynamicly return true.
        
        Examples:
            .. code-block:: python

                # required: gpu,xpu         
                import paddle
                scaler = paddle.amp.GradScaler(enable=True,
                                               init_loss_scaling=1024,
                                               incr_ratio=2.0,
                                               decr_ratio=0.5,
                                               incr_every_n_steps=1000,
                                               decr_every_n_nan_or_inf=2,
                                               use_dynamic_loss_scaling=True)
                use_dynamic_loss_scaling = scaler.is_use_dynamic_loss_scaling()
                print(use_dynamic_loss_scaling) # True
        """
        return super(GradScaler, self).is_use_dynamic_loss_scaling()

    def get_init_loss_scaling(self):
        """
        Return the initial loss scaling factor.

        Reurns:
            float:  the initial loss scaling factor.
        
        Examples:
            .. code-block:: python

                # required: gpu,xpu
                import paddle
                scaler = paddle.amp.GradScaler(enable=True,
                                               init_loss_scaling=1024,
                                               incr_ratio=2.0,
                                               decr_ratio=0.5,
                                               incr_every_n_steps=1000,
                                               decr_every_n_nan_or_inf=2,
                                               use_dynamic_loss_scaling=True)
                init_loss_scaling = scaler.get_init_loss_scaling()
                print(init_loss_scaling) # 1024
        """
        return super(GradScaler, self).get_init_loss_scaling()

    def set_init_loss_scaling(self, new_init_loss_scaling):
        """
        Set the initial loss scaling factor by `new_init_loss_scaling`.

        Args:
            new_init_loss_scaling(float):  The new_init_loss_scaling used to update initial loss scaling factor.
        
        Examples:
            .. code-block:: python
                
                # required: gpu,xpu
                import paddle
                scaler = paddle.amp.GradScaler(enable=True,
                                               init_loss_scaling=1024,
                                               incr_ratio=2.0,
                                               decr_ratio=0.5,
                                               incr_every_n_steps=1000,
                                               decr_every_n_nan_or_inf=2,
                                               use_dynamic_loss_scaling=True)
                print(scaler.get_init_loss_scaling()) # 1024
                new_init_loss_scaling = 1000
                scaler.set_init_loss_scaling(new_init_loss_scaling)
                print(scaler.get_init_loss_scaling()) # 1000
        """
        super(GradScaler, self).set_init_loss_scaling(new_init_loss_scaling)

    def get_incr_ratio(self):
        """
        Return the multiplier to use when increasing the loss scaling.

        Reurns:
            float:  the multiplier to use when increasing the loss scaling.
        
        Examples:
            .. code-block:: python

                # required: gpu,xpu
                import paddle
                scaler = paddle.amp.GradScaler(enable=True,
                                               init_loss_scaling=1024,
                                               incr_ratio=2.0,
                                               decr_ratio=0.5,
                                               incr_every_n_steps=1000,
                                               decr_every_n_nan_or_inf=2,
                                               use_dynamic_loss_scaling=True)
                incr_ratio = scaler.get_incr_ratio()
                print(incr_ratio) # 2.0
        """
        return super(GradScaler, self).get_incr_ratio()

    def set_incr_ratio(self, new_incr_ratio):
        """
        Set the multiplier to use when increasing the loss scaling by `new_incr_ratio`, `new_incr_ratio` should > 1.0.

        Args:
            new_incr_ratio(float):  The new_incr_ratio used to update the multiplier to use when increasing the loss scaling.
        
        Examples:
            .. code-block:: python

                # required: gpu,xpu
                import paddle
                scaler = paddle.amp.GradScaler(enable=True,
                                               init_loss_scaling=1024,
                                               incr_ratio=2.0,
                                               decr_ratio=0.5,
                                               incr_every_n_steps=1000,
                                               decr_every_n_nan_or_inf=2,
                                               use_dynamic_loss_scaling=True)
                print(scaler.get_incr_ratio()) # 2.0
                new_incr_ratio = 3.0
                scaler.set_incr_ratio(new_incr_ratio)
                print(scaler.get_incr_ratio()) # 3.0
        """
        super(GradScaler, self).set_incr_ratio(new_incr_ratio)

    def get_decr_ratio(self):
        """
        Get the less-than-one-multiplier to use when decreasing the loss scaling.

        Reurns:
            float:  the less-than-one-multiplier to use when decreasing the loss scaling.
        
        Examples:
            .. code-block:: python

                # required: gpu,xpu
                import paddle
                scaler = paddle.amp.GradScaler(enable=True,
                                               init_loss_scaling=1024,
                                               incr_ratio=2.0,
                                               decr_ratio=0.5,
                                               incr_every_n_steps=1000,
                                               decr_every_n_nan_or_inf=2,
                                               use_dynamic_loss_scaling=True)
                decr_ratio = scaler.get_decr_ratio()
                print(decr_ratio) # 0.5
        """
        return super(GradScaler, self).get_decr_ratio()

    def set_decr_ratio(self, new_decr_ratio):
        """
        Set the less-than-one-multiplier to use when decreasing the loss scaling by `new_incr_ratio`, `new_decr_ratio` should < 1.0.

        Args:
            new_decr_ratio(float):  The new_decr_ratio used to update the less-than-one-multiplier to use when decreasing the loss scaling.
        
        Examples:
            .. code-block:: python

                # required: gpu,xpu
                import paddle
                scaler = paddle.amp.GradScaler(enable=True,
                                               init_loss_scaling=1024,
                                               incr_ratio=2.0,
                                               decr_ratio=0.5,
                                               incr_every_n_steps=1000,
                                               decr_every_n_nan_or_inf=2,
                                               use_dynamic_loss_scaling=True)
                print(scaler.get_decr_ratio()) # 0.5
                new_decr_ratio = 0.1
                scaler.set_decr_ratio(new_decr_ratio)
                print(scaler.get_decr_ratio()) # 0.1
        """
        super(GradScaler, self).set_decr_ratio(new_decr_ratio)

    def get_incr_every_n_steps(self):
        """
        Return the num `n`, `n` represent increases loss scaling every `n` consecutive steps with finite gradients.

        Reurns:
            int:  the num `n`, `n` represent increases loss scaling every `n` consecutive steps with finite gradients.
        
        Examples:
            .. code-block:: python

                # required: gpu,xpu
                import paddle
                scaler = paddle.amp.GradScaler(enable=True,
                                               init_loss_scaling=1024,
                                               incr_ratio=2.0,
                                               decr_ratio=0.5,
                                               incr_every_n_steps=1000,
                                               decr_every_n_nan_or_inf=2,
                                               use_dynamic_loss_scaling=True)
                incr_every_n_steps = scaler.get_incr_every_n_steps()
                print(incr_every_n_steps) # 1000
        """
        return super(GradScaler, self).get_incr_every_n_steps()

    def set_incr_every_n_steps(self, new_incr_every_n_steps):
        """
        Set the num `n` by `new_incr_every_n_steps`, `n` represent increases loss scaling every `n` consecutive steps with finite gradients.

        Args:
            new_incr_every_n_steps(int):  The new_incr_every_n_steps used to update the num `n`, `n` represent increases loss scaling every `n` consecutive steps with finite gradients.
        
        Examples:
            .. code-block:: python

                # required: gpu,xpu
                import paddle
                scaler = paddle.amp.GradScaler(enable=True,
                                               init_loss_scaling=1024,
                                               incr_ratio=2.0,
                                               decr_ratio=0.5,
                                               incr_every_n_steps=1000,
                                               decr_every_n_nan_or_inf=2,
                                               use_dynamic_loss_scaling=True)
                print(scaler.get_incr_every_n_steps()) # 1000
                new_incr_every_n_steps = 2000
                scaler.set_incr_every_n_steps(new_incr_every_n_steps)
                print(scaler.get_incr_every_n_steps()) # 2000
        """
        super(GradScaler, self).set_incr_every_n_steps(new_incr_every_n_steps)

    def get_decr_every_n_nan_or_inf(self):
        """
        Return the num `n`, `n` represent decreases loss scaling every `n` accumulated steps with nan or inf gradients.

        Reurns:
            int:  the num `n`, `n` represent decreases loss scaling every `n` accumulated steps with nan or inf gradients.
        
        Examples:
            .. code-block:: python

                # required: gpu,xpu
                import paddle
                scaler = paddle.amp.GradScaler(enable=True,
                                               init_loss_scaling=1024,
                                               incr_ratio=2.0,
                                               decr_ratio=0.5,
                                               incr_every_n_steps=1000,
                                               decr_every_n_nan_or_inf=2,
                                               use_dynamic_loss_scaling=True)
                decr_every_n_nan_or_inf = scaler.get_decr_every_n_nan_or_inf()
                print(decr_every_n_nan_or_inf) # 2
        """
        return super(GradScaler, self).get_decr_every_n_nan_or_inf()

    def set_decr_every_n_nan_or_inf(self, new_decr_every_n_nan_or_inf):
        """
        Set the num `n` by `new_decr_every_n_nan_or_inf`, `n` represent decreases loss scaling every `n` accumulated steps with nan or inf gradients.

        Args:
            new_decr_every_n_nan_or_inf(int):  The new_decr_every_n_nan_or_inf used to update the num `n`, `n` represent decreases loss scaling every `n` accumulated steps with nan or inf gradients.
        
        Examples:
            .. code-block:: python

                # required: gpu,xpu
                import paddle
                scaler = paddle.amp.GradScaler(enable=True,
                                               init_loss_scaling=1024,
                                               incr_ratio=2.0,
                                               decr_ratio=0.5,
                                               incr_every_n_steps=1000,
                                               decr_every_n_nan_or_inf=2,
                                               use_dynamic_loss_scaling=True)
                print(scaler.get_decr_every_n_nan_or_inf()) # 2
                new_decr_every_n_nan_or_inf = 3
                scaler.set_decr_every_n_nan_or_inf(new_decr_every_n_nan_or_inf)
                print(scaler.get_decr_every_n_nan_or_inf()) # 3
        """
        super(GradScaler,
              self).set_decr_every_n_nan_or_inf(new_decr_every_n_nan_or_inf)

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

        
        Examples:

            .. code-block:: python

                # required: gpu,xpu
                import paddle

                scaler = paddle.amp.GradScaler(enable=True,
                                               init_loss_scaling=1024,
                                               incr_ratio=2.0,
                                               decr_ratio=0.5,
                                               incr_every_n_steps=1000,
                                               decr_every_n_nan_or_inf=2,
                                               use_dynamic_loss_scaling=True)
                scaler_state = scaler.state_dict()
        """
        return super(GradScaler, self).state_dict()

    def load_state_dict(self, state_dict):
        """
        Loads the scaler state.
        
        Args:
           state_dict(dict): scaler state.  Should be an object returned from a call to `GradScaler.state_dict()`.
                
        Examples:

            .. code-block:: python

                # required: gpu,xpu
                import paddle

                scaler = paddle.amp.GradScaler(enable=True,
                                               init_loss_scaling=1024,
                                               incr_ratio=2.0,
                                               decr_ratio=0.5,
                                               incr_every_n_steps=1000,
                                               decr_every_n_nan_or_inf=2,
                                               use_dynamic_loss_scaling=True)
                scaler_state = scaler.state_dict()
                scaler.load_state_dict(scaler_state)
        """
        super(GradScaler, self).load_state_dict(state_dict)
