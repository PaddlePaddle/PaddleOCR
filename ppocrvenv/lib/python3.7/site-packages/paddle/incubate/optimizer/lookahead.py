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
from paddle.fluid import core, framework, layers, unique_name
from paddle.fluid.framework import Program, Variable, name_scope, default_main_program, default_startup_program, device_guard
from paddle.fluid.layer_helper import LayerHelper
import paddle
import numpy as np
from paddle.fluid.dygraph import base as imperative_base

__all__ = []


class LookAhead(Optimizer):
    r"""
    This implements the Lookahead optimizer of the
    paper : https://arxiv.org/abs/1907.08610.

    Lookahead keeps two sets of params: the fast_params and
    the slow_params. inner_optimizer update fast_params every 
    training step. Lookahead updates the slow_params and fast_params 
    every k training steps as follows:

    .. math::
        
        slow\_param_t &= slow\_param_{t-1} + \\alpha * (fast\_param_{t-1} - slow\_param_{t-1})
	    
        fast\_param_t &=  slow\_param_t

    Args:
        inner_optimizer (Optimizer): The optimizer that update fast params step by step. 
        alpha (float, optinal): The learning rate of Lookahead. The default value is 0.5.
        k (int, optinal): The slow params is updated every k steps. The default value is 5.
        name (str, optional): Normally there is no need for user to set this property.
            For more information, please refer to :ref:`api_guide_Name`.
            The default value is None.

    Examples:

        .. code-block:: python
        
            import numpy as np
            import paddle
            import paddle.nn as nn

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
                    label = np.random.randint(0, CLASS_NUM - 1,
                                            (1, )).astype('int64')
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

            def train(layer, loader, loss_fn, opt):
                for epoch_id in range(EPOCH_NUM):
                    for batch_id, (image, label) in enumerate(loader()):
                        out = layer(image)
                        loss = loss_fn(out, label)
                        loss.backward()
                        opt.step()
                        opt.clear_grad()
                        print("Train Epoch {} batch {}: loss = {}".format(
                            epoch_id, batch_id, np.mean(loss.numpy())))

            layer = LinearNet()
            loss_fn = nn.CrossEntropyLoss()
            optimizer = paddle.optimizer.SGD(learning_rate=0.1, parameters=layer.parameters())
            lookahead = paddle.incubate.LookAhead(optimizer, alpha=0.2, k=5)

            # create data loader
            dataset = RandomDataset(BATCH_NUM * BATCH_SIZE)
            loader = paddle.io.DataLoader(
                dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                drop_last=True,
                num_workers=2)
            
            train(layer, loader, loss_fn, lookahead)

    """
    _slow_str = "slow"

    def __init__(self, inner_optimizer, alpha=0.5, k=5, name=None):
        assert (inner_optimizer is not None), "inner optimizer can not be None"
        assert (
            0.0 <= alpha <= 1.0
        ), "alpha should be larger or equal to 0.0, and less or equal than 1.0"
        assert (isinstance(k, int) and k > 0), "k should be a positive integer"

        self.inner_optimizer = inner_optimizer
        if self.inner_optimizer._parameter_list is None:
            parameters = framework.default_main_program().global_block(
            ).all_parameters()
        else:
            parameters = self.inner_optimizer._parameter_list

        super(LookAhead, self).__init__(
            learning_rate=alpha,
            parameters=parameters,
            weight_decay=None,
            grad_clip=None,
            name=name)

        self.alpha = alpha
        self.k = k
        self.type = "lookahead"
        self.helper = LayerHelper(self.__class__.__name__)
        self._global_step_var = None
        self._k_var = None

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
                lookahead = paddle.incubate.LookAhead(sgd, alpha=0.2, k=5)
                loss.backward()
                lookahead.step()
                lookahead.clear_grad()

        """
        self.inner_optimizer.step()

        self._increment_global_var()
        params_grads = []
        for param in self._parameter_list:
            if not param.trainable:
                continue
            if param._grad_ivar() is not None:
                grad_var = param._grad_ivar()
                params_grads.append((param, grad_var))

        self._apply_optimize(
            loss=None, startup_program=None, params_grads=params_grads)

    def _create_accumulators(self, block, parameters):
        assert isinstance(block, framework.Block)

        for p in parameters:
            self._add_accumulator(self._slow_str, p)

    def _increment_global_var(self):
        if self._global_step_var is None:
            self._global_step_var = layers.create_global_var(
                name=unique_name.generate("lookahead_step"),
                shape=[1],
                value=0,
                dtype='int32',
                persistable=True)

        self.helper.append_op(
            type='increment',
            inputs={'X': [self._global_step_var]},
            outputs={'Out': [self._global_step_var]},
            attrs={'step': 1.0})

    def _append_optimize_op(self, block, param_and_grad):
        one_var = paddle.ones(shape=[1], dtype='int32', name='lookahead_ones')
        zero_var = paddle.zeros(
            shape=[1], dtype='int32', name='lookahead_zeros')
        k_var = layers.create_global_var(
            name=unique_name.generate("lookahead_k"),
            shape=[1],
            value=self.k,
            dtype='int32',
            persistable=True)

        mod = paddle.remainder(self._global_step_var, k_var)

        cond_1 = paddle.equal(self._global_step_var, one_var)
        cond_1 = paddle.cast(cond_1, dtype='float32')

        cond_2 = paddle.equal(mod, zero_var)
        cond_2 = paddle.cast(cond_2, dtype='float32')

        slow_var = self._get_accumulator(self._slow_str, param_and_grad[0])

        tmp_var = cond_1 * param_and_grad[0] + (1 - cond_1) * slow_var
        paddle.assign(tmp_var, slow_var)

        tmp_var = self.alpha * param_and_grad[0] + (1.0 - self.alpha) * slow_var
        tmp_var_1 = cond_2 * tmp_var + (1 - cond_2) * param_and_grad[0]
        paddle.assign(tmp_var_1, param_and_grad[0])

        tmp_var_1 = cond_2 * tmp_var + (1 - cond_2) * slow_var
        paddle.assign(tmp_var_1, slow_var)

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
                sgd = paddle.optimizer.SGD(learning_rate=0.1,parameters=linear.parameters())
                lookahead = paddle.incubate.LookAhead(sgd, alpha=0.2, k=5)
                loss.backward()
                lookahead.minimize(loss)
                lookahead.clear_grad()

        """
        assert isinstance(loss, Variable), "The loss should be an Tensor."

        # Apply inner optimizer to the main_program
        optimize_ops, params_grads = self.inner_optimizer.minimize(
            loss,
            startup_program=startup_program,
            parameters=parameters,
            no_grad_set=no_grad_set)

        self._increment_global_var()

        _ = self._apply_optimize(
            loss, startup_program=startup_program, params_grads=params_grads)

        return optimize_ops, params_grads
