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

from paddle.optimizer import Optimizer
from paddle.fluid import core, framework, layers
from paddle.fluid.framework import Program, Variable
from paddle.fluid.layer_helper import LayerHelper
import paddle
import numpy as np
from paddle.fluid.dygraph import base as imperative_base
from paddle.fluid.wrapped_decorator import signature_safe_contextmanager
from paddle import _C_ops

__all__ = []


class ModelAverage(Optimizer):
    r"""
    The ModelAverage optimizer accumulates specific continuous historical
    parameters during training. The accumulated historical range can be controlled
    by the passed ``average_window_rate`` argument. The averaged ``Parameter`` are
    used in the prediction, which usually can improve the accuracy of the prediction.

    Accumulate the average of the ``Parameter`` in the sliding window, the result will be saved
    in a temporary variable, can be applied to the current model's ``Parameter`` by calling
    the ``apply()`` method, and the current model ``Parameter`` can be restored by calling
    the ``restore()`` method.

    The window size for calculating the average is determined by ``average_window_rate``,
    ``min_average_window``, ``max_average_window`` and the current ``Parameter`` update times (num_updates).

    When the cumulative times (num_accumulates) is greater than the specific window
    threshold (average_window), the accumulated ``Parameter`` temporary variable is set to 0.0.
    The following example will help to understand the role of these arguments:

    ::

        if num_accumulates >= min_average_window and num_accumulates >= min(max_average_window, num_updates * average_window_rate):
            num_accumulates = 0

    In the above conditional judgment statement, ``num_accumulates`` indicates the current
    accumulated number, which can be abstractly understood as the length of the cumulative window.
    The length of the window must be at least the length set by the ``min_average_window`` argument,
    and cannot exceed the length specified by the ``max_average_window`` argument or
    ``num_updates * average_window_rate``, where ``num_updates`` indicates the current ``Parameter``
    update times, ``average_window_rate`` is a coefficient that calculates the length of the window.

    Args:
        average_window_rate (float): The calculate ratio of the window length relative to ``Parameter`` update times.
        parameters (list, optional): List of ``Tensor`` names to update to minimize ``loss``. \
            This parameter is required in dygraph mode. \
            The default value is None in static mode, at this time all parameters will be updated.
        min_average_window (int, optional): the minimum size of average window length. The default value is 10000.
        max_average_window (int, optional): The maximum size of average window length. The default value is 10000.
        name (str, optional): Normally there is no need for user to set this property.
            For more information, please refer to :ref:`api_guide_Name`.
            The default value is None.

    Examples:

      .. code-block:: python

        import numpy as np
        import paddle
        import paddle.nn as nn
        import paddle.optimizer as opt

        BATCH_SIZE = 16
        BATCH_NUM = 4
        EPOCH_NUM = 4

        IMAGE_SIZE = 784
        CLASS_NUM = 10

        # define a random dataset
        class RandomDataset(paddle.io.Dataset):
            def __init__(self, num_samples):
                self.num_samples = num_samples

            def __getitem__(self, idx):
                image = np.random.random([IMAGE_SIZE]).astype('float32')
                label = np.random.randint(0, CLASS_NUM - 1, (1, )).astype('int64')
                return image, label

            def __len__(self):
                return self.num_samples

        class LinearNet(nn.Layer):
            def __init__(self):
                super(LinearNet, self).__init__()
                self._linear = nn.Linear(IMAGE_SIZE, CLASS_NUM)
                self.bias = self._linear.bias

            @paddle.jit.to_static
            def forward(self, x):
                return self._linear(x)

        def train(layer, loader, loss_fn, opt, model_average):
            for epoch_id in range(EPOCH_NUM):
                for batch_id, (image, label) in enumerate(loader()):
                    out = layer(image)
                    loss = loss_fn(out, label)
                    loss.backward()
                    opt.step()
                    model_average.step()
                    opt.clear_grad()
                    model_average.clear_grad()
                    print("Train Epoch {} batch {}: loss = {}, bias = {}".format(
                        epoch_id, batch_id, np.mean(loss.numpy()), layer.bias.numpy()))
        def evaluate(layer, loader, loss_fn):
            for batch_id, (image, label) in enumerate(loader()):
                out = layer(image)
                loss = loss_fn(out, label)
                loss.backward()
                print("Evaluate batch {}: loss = {}, bias = {}".format(
                    batch_id, np.mean(loss.numpy()), layer.bias.numpy()))

        # create network
        layer = LinearNet()
        loss_fn = nn.CrossEntropyLoss()
        optimizer = opt.Momentum(learning_rate=0.2, momentum=0.1, parameters=layer.parameters())
        model_average = paddle.incubate.ModelAverage(0.15,
                                                    parameters=layer.parameters(),
                                                    min_average_window=2,
                                                    max_average_window=10)

        # create data loader
        dataset = RandomDataset(BATCH_NUM * BATCH_SIZE)
        loader = paddle.io.DataLoader(dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            drop_last=True,
            num_workers=2)
        # create data loader
        eval_loader = paddle.io.DataLoader(dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            drop_last=True,
            num_workers=1)

        # train
        train(layer, loader, loss_fn, optimizer, model_average)

        print("\nEvaluate With ModelAverage")
        with model_average.apply(need_restore=False):
            evaluate(layer, eval_loader, loss_fn)

        print("\nEvaluate With Restored Paramters")
        model_average.restore()
        evaluate(layer, eval_loader, loss_fn)
  
    """

    def __init__(self,
                 average_window_rate,
                 parameters=None,
                 min_average_window=10000,
                 max_average_window=10000,
                 name=None):
        super(ModelAverage, self).__init__(
            learning_rate=0.0,
            parameters=parameters,
            weight_decay=None,
            grad_clip=None,
            name=name)

        self.helper = LayerHelper(self.__class__.__name__)
        self.average_window = average_window_rate
        self.min_average_window = min_average_window
        self.max_average_window = max_average_window
        self.type = "average_accumulates"

        if not framework.in_dygraph_mode():
            global_block = framework.default_main_program().global_block()
            all_parameters = parameters if parameters else global_block.all_parameters(
            )

            self._create_accumulators(global_block, all_parameters)
            for param in all_parameters:
                self._append_optimize_op(global_block, [param, None])
            self.apply_program = Program()
            block = self.apply_program.global_block()
            with framework.program_guard(main_program=self.apply_program):
                for param in all_parameters:
                    self._add_average_apply_op(block, param)
            self.restore_program = Program()
            block = self.restore_program.global_block()
            with framework.program_guard(main_program=self.restore_program):
                for param in all_parameters:
                    self._add_average_restore_op(block, param)

    def _create_accumulators(self, block, parameters):
        assert isinstance(block, framework.Block)

        for param in parameters:
            self._add_accumulator('sum_1', param)
            self._add_accumulator('sum_2', param)
            self._add_accumulator('sum_3', param)
            self._add_accumulator('restore', param)
            self._add_accumulator(
                'num_accumulates', param, dtype='int64', shape=[1])
            self._add_accumulator(
                'old_num_accumulates', param, dtype='int64', shape=[1])
            self._add_accumulator(
                'num_updates', param, dtype='int64', shape=[1])

    def _append_optimize_op(self, block, param_and_grad):
        assert isinstance(block, framework.Block)

        sum_1 = self._get_accumulator('sum_1', param_and_grad[0])
        sum_2 = self._get_accumulator('sum_2', param_and_grad[0])
        sum_3 = self._get_accumulator('sum_3', param_and_grad[0])
        num_accumulates = self._get_accumulator('num_accumulates',
                                                param_and_grad[0])
        old_num_accumulates = self._get_accumulator('old_num_accumulates',
                                                    param_and_grad[0])
        num_updates = self._get_accumulator('num_updates', param_and_grad[0])
        if framework.in_dygraph_mode():
            _, _, _, _, _, _ = _C_ops.average_accumulates(
                param_and_grad[0], sum_1, sum_2, sum_3, num_accumulates,
                old_num_accumulates, num_updates, sum_1, sum_2, sum_3,
                num_accumulates, old_num_accumulates, num_updates,
                'average_window', self.average_window, 'min_average_window',
                self.min_average_window, 'max_average_window',
                self.max_average_window)
            return None

        block = framework.default_main_program().global_block()
        attrs = {
            "average_window": self.average_window,
            "min_average_window": self.min_average_window,
            "max_average_window": self.max_average_window,
        }

        inputs = {
            "param": param_and_grad[0],
            "in_sum_1": sum_1,
            "in_sum_2": sum_2,
            "in_sum_3": sum_3,
            "in_num_accumulates": num_accumulates,
            "in_old_num_accumulates": old_num_accumulates,
            "in_num_updates": num_updates
        }

        outputs = {
            "out_sum_1": sum_1,
            "out_sum_2": sum_2,
            "out_sum_3": sum_3,
            "out_num_accumulates": num_accumulates,
            "out_old_num_accumulates": old_num_accumulates,
            "out_num_updates": num_updates,
        }

        average_accumulates_op = block.append_op(
            type=self.type,
            inputs=inputs,
            outputs=outputs,
            attrs=attrs,
            stop_gradient=True)

        return average_accumulates_op

    @imperative_base.no_grad
    def minimize(self,
                 loss,
                 startup_program=None,
                 parameters=None,
                 no_grad_set=None):
        """
        Add operations to minimize ``loss`` by updating ``parameters``.
        
        Args:
            loss (Tensor): A ``Tensor`` containing the value to minimize.
            startup_program (Program, optional): :ref:`api_fluid_Program` for
                initializing parameters in ``parameters``. The default value
                is None, at this time :ref:`api_fluid_default_startup_program` will be used.
            parameters (list, optional): List of ``Tensor`` or ``Tensor.name`` to update
                to minimize ``loss``. The default value is None, at this time all parameters
                will be updated.
            no_grad_set (set, optional): Set of ``Tensor``  or ``Tensor.name`` that don't need
                to be updated. The default value is None.
        
        Returns:
            tuple: tuple (optimize_ops, params_grads), A list of operators appended
            by minimize and a list of (param, grad) tensor pairs, param is
            ``Parameter``, grad is the gradient value corresponding to the parameter.
            In static graph mode, the returned tuple can be passed to ``fetch_list`` in ``Executor.run()`` to 
            indicate program pruning. If so, the program will be pruned by ``feed`` and 
            ``fetch_list`` before run, see details in ``Executor``.
        
        Examples:
        
            .. code-block:: python

                import paddle
                import numpy as np
                inp = paddle.to_tensor(np.random.random([1, 10]).astype('float32'))
                linear = paddle.nn.Linear(10, 1)
                out = linear(inp)
                loss = paddle.mean(out)
                loss.backward()

                sgd = paddle.optimizer.SGD(learning_rate=0.1,parameters=linear.parameters())
                sgd.minimize(loss)

                modelaverage = paddle.incubate.ModelAverage(0.15,
                                                            parameters=linear.parameters(),
                                                            min_average_window=2,
                                                            max_average_window=4)
                modelaverage.minimize(loss)
                sgd.clear_grad()
                modelaverage.clear_grad()

        """
        if framework.in_dygraph_mode():
            self.step()

    @framework.dygraph_only
    @imperative_base.no_grad
    def step(self):
        """
        Execute the optimizer and update parameters once.
        
        Returns:
            None

        Examples:

            .. code-block:: python

                import paddle
                import numpy as np
                inp = paddle.to_tensor(np.random.random([1, 10]).astype('float32'))
                linear = paddle.nn.Linear(10, 1)
                out = linear(inp)
                loss = paddle.mean(out)
                sgd = paddle.optimizer.SGD(learning_rate=0.1,parameters=linear.parameters())
                modelaverage = paddle.incubate.ModelAverage(0.15,
                                                            parameters=linear.parameters(),
                                                            min_average_window=2,
                                                            max_average_window=4)
                loss.backward()
                sgd.step()
                modelaverage.step()
                sgd.clear_grad()
                modelaverage.clear_grad()
        """

        params_grads = []
        for param in self._parameter_list:
            if not param.trainable:
                continue
            if param._grad_ivar() is not None:
                grad_var = param._grad_ivar()
                params_grads.append((param, grad_var))

        block = framework.default_main_program().global_block()
        self._create_accumulators(block, self._parameter_list)
        for param_and_grad in params_grads:
            self._append_optimize_op(block, param_and_grad)

    @signature_safe_contextmanager
    @imperative_base.no_grad
    def apply(self, executor=None, need_restore=True):
        """
        Apply the average of the cumulative ``Parameter`` to the parameters of the current model.

        Args:
            executor(Executor): The network executor in static-graph mode. The default value is None in dygraph mode.
            need_restore(bool): Restore flag variable, if set to True, the network will restore
                the parameters of the network to the default value, if set to False,
                it will not be restored. The default value is True.

        Examples:

            .. code-block:: python

                import paddle
                import numpy as np
                inp = paddle.to_tensor(np.random.random([1, 10]).astype('float32'))
                linear = paddle.nn.Linear(10, 1)
                out = linear(inp)
                loss = paddle.mean(out)
                loss.backward()

                sgd = paddle.optimizer.SGD(learning_rate=0.1,parameters=linear.parameters())

                modelaverage = paddle.incubate.ModelAverage(0.15,
                                                            parameters=linear.parameters(),
                                                            min_average_window=2,
                                                            max_average_window=4)
                sgd.step()
                modelaverage.step()
                
                with modelaverage.apply():
                    for param in linear.parameters():
                        print(param)

                for param in linear.parameters():
                    print(param)
        """
        if framework.in_dygraph_mode():
            for param in self._parameter_list:
                num_accumulates = self._get_accumulator('num_accumulates',
                                                        param)
                old_num_accumulates = self._get_accumulator(
                    'old_num_accumulates', param)
                sum_1 = self._get_accumulator('sum_1', param)
                sum_2 = self._get_accumulator('sum_2', param)
                sum_3 = self._get_accumulator('sum_3', param)
                param_restore = self._get_accumulator('restore', param)

                paddle.assign(param, param_restore)
                total_param = sum_1 + sum_2 + sum_3
                total_accumulates = num_accumulates + old_num_accumulates
                total_param = paddle.cast(total_param, dtype='float32')
                total_accumulates = paddle.cast(
                    total_accumulates, dtype='float32')
                average_param = total_param / total_accumulates
                paddle.assign(average_param, param)
            try:
                yield
            finally:
                if need_restore:
                    self.restore()
            return
        if executor is None:
            raise RuntimeError(
                "Executor should not be None in static graph mode.")
        executor.run(self.apply_program)
        try:
            yield
        finally:
            if need_restore:
                self.restore(executor)

    @imperative_base.no_grad
    def restore(self, executor=None):
        """
        Restore ``Parameter`` values of current model.
        
        Args:
            executor(Executor): The network executor in static-graph mode. The default value is None in dygraph mode

        Examples:

            .. code-block:: python

                import paddle
                import numpy as np
                inp = paddle.to_tensor(np.random.random([1, 10]).astype('float32'))
                linear = paddle.nn.Linear(10, 1)
                out = linear(inp)
                loss = paddle.mean(out)
                loss.backward()

                sgd = paddle.optimizer.SGD(learning_rate=0.1,parameters=linear.parameters())

                modelaverage = paddle.incubate.ModelAverage(0.15,
                                                            parameters=linear.parameters(),
                                                            min_average_window=2,
                                                            max_average_window=4)
                sgd.step()
                modelaverage.step()
                
                with modelaverage.apply(need_restore=False):
                    for param in linear.parameters():
                        print(param)

                for param in linear.parameters():
                    print(param)

                modelaverage.restore()

                for param in linear.parameters():
                    print(param)
        """
        if framework.in_dygraph_mode():
            for param in self._parameter_list:
                param_restore = self._get_accumulator('restore', param)
                paddle.assign(param_restore, param)
            return
        if executor is None:
            raise RuntimeError(
                "Executor should not be None in static graph mode.")
        executor.run(self.restore_program)

    def _add_average_apply_op(self, block, param):
        param = block._clone_variable(param)
        grad = block._clone_variable(self._get_accumulator('restore', param))
        sum_1 = block._clone_variable(self._get_accumulator('sum_1', param))
        sum_2 = block._clone_variable(self._get_accumulator('sum_2', param))
        sum_3 = block._clone_variable(self._get_accumulator('sum_3', param))
        num_accumulates = block._clone_variable(
            self._get_accumulator('num_accumulates', param))
        old_num_accumulates = block._clone_variable(
            self._get_accumulator('old_num_accumulates', param))
        # backup param value to grad
        layers.assign(input=param, output=grad)
        # param = (sum_1 + sum_2 + sum_3) / (num_accumulates + old_num_accumulates)
        tmp = layers.sum(x=[num_accumulates, old_num_accumulates])
        sum = layers.sum(x=[sum_1, sum_2, sum_3])
        tmp = layers.cast(
            x=tmp, dtype='float32' if self._dtype is None else self._dtype)
        sum = layers.cast(
            x=sum, dtype='float32' if self._dtype is None else self._dtype)
        layers.ops._elementwise_div(x=sum, y=tmp, out=param)

    def _add_average_restore_op(self, block, param):
        param = block._clone_variable(param)
        grad = block._clone_variable(self._get_accumulator('restore', param))
        layers.assign(input=grad, output=param)
