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

from paddle.fluid import (core, default_main_program, layers, program_guard,
                          unique_name)
from .amp_utils import (rewrite_program_bf16, cast_model_to_bf16,
                        cast_parameters_to_bf16)
from .amp_lists import AutoMixedPrecisionListsBF16
import types
import warnings

__all__ = ["decorate_bf16"]


class OptimizerWithMixedPrecision(object):
    """
    Optimizer with mixed-precision (MP) training. This is a wrapper of a common 
    optimizer, plus the support of mixed-precision pre-training. The object
    of this class almost has the same behavior as the common optimizer, with the 
    methods `minimize()`, `backward()`, `apply_gradients()` implemented. 
    Additionally, it enables the MP training automatically, i.e, the creation 
    and maintenance of master parameters, scaling of loss, etc.

    Args:
        optimizer (Optimizer): A common Optimizer object.
        amp_lists (CustomOpLists): An CustomOpLists object.
        use_pure_bf16(bool): Whether to use the pure bf16 training.
        use_bf16_guard(bool): Whether to use `bf16_guard` when constructing the program.

    """

    def __init__(self, optimizer, amp_lists, use_pure_bf16, use_bf16_guard):
        self._optimizer = optimizer
        self._amp_lists = amp_lists
        self._param_grads = None
        self._train_program = None

        self._learning_rate = optimizer._learning_rate
        self._learning_rate_map = optimizer._learning_rate_map
        self._use_pure_bf16 = use_pure_bf16
        self._use_bf16_guard = use_bf16_guard
        self._to_bf16_var_names = None

    def _init_amp_var(self):
        # Ensure the data type of learning rate vars is float32 (same as the
        # master parameter dtype)
        if isinstance(self._optimizer._learning_rate, float):
            self._optimizer._learning_rate_map[default_main_program()] = \
                    layers.create_global_var(
                    name=unique_name.generate("learning_rate"),
                    shape=[1],
                    value=float(self._optimizer._learning_rate),
                    dtype='float32',
                    persistable=True)

    def backward(self,
                 loss,
                 startup_program=None,
                 parameter_list=None,
                 no_grad_set=None,
                 callbacks=None):
        """
        Backward propagation or auto differentiation for gradients' computation.

        Args:
            loss (Variable): The loss Variable to minimize.
            startup_program (Program|None): The startup Program for initializing 
                                       parameters in `parameter_list`.
            parameter_list (list|None): A list of Variables to update.
            no_grad_set (set|None): A set of Variables should be ignored.
            callbacks (list|None): A list of callable objects to run when appending
                                   backward operator for one parameter.

        Returns:
            A list of (param, grad), which is a tuple of a parameter and its 
            gradient respectively, and the scaled loss.
        """
        train_program = loss.block.program
        self._train_program = train_program

        with program_guard(self._train_program, startup_program):
            self._init_amp_var()

            if self._use_pure_bf16:
                self._to_bf16_var_names = cast_model_to_bf16(
                    self._train_program, startup_program, self._amp_lists,
                    self._use_bf16_guard)
            else:
                rewrite_program_bf16(self._train_program, self._amp_lists)

            if loss.dtype != core.VarDesc.VarType.FP32:
                loss = loss.astype('float32')

            params_grads = self._optimizer.backward(
                loss, startup_program, parameter_list, no_grad_set, callbacks)
        return params_grads

    def amp_init(self,
                 place,
                 scope=None,
                 test_program=None,
                 use_bf16_test=False):
        """
        Init the amp training, such as cast fp32 parameters to bf16 type.
  
        Args:
            place(CPUPlace): place is used to initialize 
                bf16 parameters with fp32 values.
            scope(Scope): The scope is used to find fp32 parameters.
            test_program(Program): The program is used for testing.
            use_bf16_test(bool): Whether to use bf16 testing.

        Examples:
            .. code-block:: python

                import numpy as np
                import paddle
                import paddle.nn.functional as F
                paddle.enable_static()

                def run_example_code():
                    place = paddle.CPUPlace(0)
                    exe = paddle.static.Executor(place)
                    data = paddle.static.data(name='X', shape=[None, 1, 28, 28], dtype='float32')
                    conv2d = paddle.static.nn.conv2d(input=data, num_filters=6, filter_size=3)
                    # 1) Use bf16_guard to control the range of bf16 kernels used.
                    with paddle.static.amp.bf16_guard():
                        bn = paddle.static.nn.batch_norm(input=conv2d, act="relu")
                        pool = F.max_pool2d(bn, kernel_size=2, stride=2)
                        hidden = paddle.static.nn.fc(pool, size=10)
                        loss = paddle.mean(hidden)
                    # 2) Create the optimizer and set `multi_precision` to True.
                    # Setting `multi_precision` to True can avoid the poor accuracy
                    # or the slow convergence in a way. 
                    optimizer = paddle.optimizer.Momentum(learning_rate=0.01, multi_precision=True)
                    # 3) These ops in `custom_fp32_list` will keep in the float32 computation type.
                    amp_list = paddle.static.amp.CustomOpLists(
                        custom_fp32_list=['pool2d'])
                    # 4) The entry of Paddle AMP.
                    # Enable pure bf16 training by setting `use_pure_bf16` to True.
                    optimizer = paddle.static.amp.bf16.decorate_bf16(
                        optimizer,
                        amp_list,
                        use_pure_bf16=True)
                    # If you don't use the default_startup_program(), you sholud pass
                    # your defined `startup_program` into `minimize`.
                    optimizer.minimize(loss)
                    exe.run(paddle.static.default_startup_program())
                    # 5) Use `amp_init` after FP32 parameters initialization(such as `exe.run(startup_program)`).
                    # If you want to perform the testing process, you should pass `test_program` into `amp_init`.
                    optimizer.amp_init(place, scope=paddle.static.global_scope())
                    
        """
        assert self._train_program is not None, \
            "Please call the minimize method first."
        if self._use_pure_bf16:
            cast_parameters_to_bf16(place, self._train_program, scope,
                                    self._to_bf16_var_names)
        if test_program is not None:
            if self._use_pure_bf16:
                cast_model_to_bf16(
                    test_program,
                    amp_lists=self._amp_lists,
                    use_bf16_guard=self._use_bf16_guard)
            elif use_bf16_test:
                rewrite_program_bf16(test_program, amp_lists=self._amp_lists)

    def apply_gradients(self, params_grads):
        """
        Apply gradients.
  
        Args:
            params_grads (list): A list of params.
    
        Returns:
            A list of optimize operators.
        """

        return self._optimizer.apply_gradients(params_grads)

    def apply_optimize(self, loss, startup_program, params_grads):
        program = loss.block.program
        with program_guard(program, startup_program):
            optimize_ops = self.apply_gradients(params_grads)
        return optimize_ops

    def minimize(self,
                 loss,
                 startup_program=None,
                 parameter_list=None,
                 no_grad_set=None):
        """
        Perform optimization by minimizing the given loss.

        Args:
            loss (Variable): The loss Variable.
            startup_program (Program): startup_program for initializing parameters
                in `parameter_list`.
            parameter_list (list): list of Variables to update.
            no_grad_set (set|None): set of Variables should be ignored.

        Returns:
            The scaled loss by scaling factor, the list of optimize ops, and a
            list of scaled parameters and gradients.
        """
        opt_dict = self._optimizer.__class__.__dict__
        if 'minimize' in opt_dict and isinstance(opt_dict['minimize'],
                                                 types.FunctionType):
            warnings.warn(
                "The decorated optimizer has its own `minimize` method, but it will not be executed."
            )

        params_grads = self.backward(
            loss,
            startup_program=startup_program,
            parameter_list=parameter_list,
            no_grad_set=no_grad_set)

        optimize_ops = self.apply_optimize(loss, startup_program, params_grads)

        return optimize_ops, params_grads


def decorate_bf16(optimizer,
                  amp_lists=None,
                  use_pure_bf16=False,
                  use_bf16_guard=None):
    """ 
    Decorate the given optimizer to adapt to the mixed-precision training.

    Args:
        optimizer(Optimizer): A common Optimizer.
        amp_lists (CustomOpLists): An CustomOpLists object.
        use_pure_bf16(bool): Whether to use the pure bf16 training. Default False.
        use_bf16_guard(bool): Whether to use `bf16_guard` when constructing the program.
                           Default None, which means that its value equals to `use_pure_bf16`.

    Returns:
        An optimizer acting like a normal one but with mixed-precision training 
        enabled.

    Examples 1:
	    .. code-block:: python

            # fp32&bf16 list based strategy example
            import paddle
            import paddle.static as static

            paddle.enable_static()

            data = static.data(name='X', shape=[None, 1], dtype='float32')
            hidden = static.nn.fc(x=data, size=10)
            loss = paddle.mean(hidden)
            optimizer = paddle.optimizer.Adam(learning_rate=0.001)

            mp_optimizer = static.amp.decorate_bf16(optimizer=optimizer)

            ops, param_grads = mp_optimizer.minimize(loss)

    Examples 2:
        .. code-block:: python

            # pure bf16 training example
            import numpy as np
            import paddle
            import paddle.nn.functional as F

            def run_example_code():
                place = paddle.CPUPlace(0)
                exe = paddle.static.Executor(place)
                data = paddle.static.data(name='X', shape=[None, 1, 28, 28], dtype='float32')
                conv2d = paddle.static.nn.conv2d(input=data, num_filters=6, filter_size=3)
                # 1) Use bf16_guard to control the range of bf16 kernels used.
                with paddle.static.amp.bf16_guard():
                    bn = paddle.static.nn.batch_norm(input=conv2d, act="relu")
                    pool = F.max_pool2d(bn, kernel_size=2, stride=2)
                    hidden = paddle.static.nn.fc(pool, size=10)
                    loss = paddle.mean(hidden)
                # 2) Create the optimizer and set `multi_precision` to True.
                # Setting `multi_precision` to True can avoid the poor accuracy
                # or the slow convergence in a way. 
                optimizer = paddle.optimizer.Momentum(learning_rate=0.01, multi_precision=True)
                # 3) These ops in `custom_fp32_list` will keep in the float32 computation type.
                amp_list = paddle.static.amp.CustomOpLists(
                    custom_fp32_list=['pool2d'])
                # 4) The entry of Paddle AMP.
                # Enable pure bf16 training by setting `use_pure_bf16` to True.
                optimizer = paddle.static.amp.decorate_bf16(
                    optimizer,
                    amp_list,
                    use_pure_bf16=True)
                # If you don't use the default_startup_program(), you sholud pass
                # your defined `startup_program` into `minimize`.
                optimizer.minimize(loss)
                exe.run(paddle.static.default_startup_program())
                # 5) Use `amp_init` after FP32 parameters initialization(such as `exe.run(startup_program)`).
                # If you want to perform the testing process, you should pass `test_program` into `amp_init`.
                optimizer.amp_init(place, scope=paddle.static.global_scope())
                
    """
    if amp_lists is None:
        amp_lists = AutoMixedPrecisionListsBF16()

    if use_bf16_guard is None:
        use_bf16_guard = use_pure_bf16

    mp_optimizer = OptimizerWithMixedPrecision(optimizer, amp_lists,
                                               use_pure_bf16, use_bf16_guard)

    return mp_optimizer
