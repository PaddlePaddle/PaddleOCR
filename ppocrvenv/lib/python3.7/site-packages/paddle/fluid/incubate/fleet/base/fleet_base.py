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

from __future__ import print_function

import abc

import paddle.fluid as fluid
from paddle.fluid.executor import Executor
from paddle.fluid.optimizer import SGD
from paddle.optimizer import SGD as SGD_v2

from paddle.fluid.incubate.fleet.base.mode import Mode
from paddle.distributed.fleet.base.role_maker import RoleMakerBase
from paddle.fluid.contrib.mixed_precision.decorator import OptimizerWithMixedPrecision
from . import mode

__all__ = ['Fleet', 'DistributedOptimizer']
__all__ += mode.__all__


class Fleet(object):
    """
    Fleet is the base class, transpiler and pslib are implementation of Fleet.

    Args:
        mode(Mode): the implementation of Fleet's mode.

    Returns:
        None
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, mode):
        self._is_initialized = False
        self._mode = mode
        self._optimizer = None
        self._role_maker = None
        self._executor = None

    def is_first_worker(self):
        """
        Check whether the node is the first instance of worker.

        Returns:
            bool: True if this is the first node of worker,
                  False if not.
        """
        return self._role_maker.is_first_worker()

    def worker_index(self):
        """
        Get current worker index.

        Returns:
            int: node id
        """
        return self._role_maker.worker_index()

    def worker_num(self):
        """
        Get current total worker number.

        Returns:
            int: worker numbers
        """
        return self._role_maker.worker_num()

    def is_worker(self):
        """
        Check whether the node is an instance of worker.

        Returns:
            bool: True if this is a node of worker,
                  False if not.
        """
        return self._role_maker.is_worker()

    def worker_endpoints(self, to_string=False):
        """
        Get current server endpoints, such as ["127.0.0.1:1001", "127.0.0.1:1002"].

        Returns:
            list/string: server endpoints
        """

        if to_string:
            return ",".join(self._role_maker.get_trainer_endpoints())
        else:
            return self._role_maker.get_trainer_endpoints()

    def server_num(self):
        """
        Get current total worker number.

        Returns:
            int: server number
        """
        return len(self._role_maker.get_pserver_endpoints())

    def server_index(self):
        """
        Get current server index.

        Returns:
            int: node id
        """
        return self._role_maker.server_index()

    def server_endpoints(self, to_string=False):
        """
        Get current server endpoints, such as ["127.0.0.1:1001", "127.0.0.1:1002"].

        Returns:
            list/string: server endpoints
        """

        if to_string:
            return ",".join(self._role_maker.get_pserver_endpoints())
        else:
            return self._role_maker.get_pserver_endpoints()

    def is_server(self):
        """
        Check whether the node is an instance of server.

        Returns:
            bool: True if this is a node of server,
                  False if not
        """
        return self._role_maker.is_server()

    def is_xpu(self):
        """
        Check whether the node is an instance of server.

        Returns:
            bool: True if this is a node of server,
                  False if not.
        """
        return self._role_maker.is_xpu()

    def split_files(self, files):
        """
        split files before distributed training,
        example 1: files is [a, b, c ,d, e]  and trainer_num = 2, then trainer
                   0 gets [a, b, c] and trainer 1 gets [d, e].
        example 2: files is [a, b], and trainer_num = 3, then trainer 0 gets
                   [a], trainer 1 gets [b],  trainer 2 gets []

        Args:
            files(list): file list need to be read.

        Returns:
            list: files belongs to this worker.
        """
        if not isinstance(files, list):
            raise TypeError("files should be a list of file need to be read.")

        trainer_id = self.worker_index()
        trainers = self.worker_num()

        remainder = len(files) % trainers
        blocksize = len(files) // trainers

        blocks = [blocksize] * trainers
        for i in range(remainder):
            blocks[i] += 1

        trainer_files = [[]] * trainers
        begin = 0
        for i in range(trainers):
            trainer_files[i] = files[begin:begin + blocks[i]]
            begin += blocks[i]

        return trainer_files[trainer_id]

    def init(self, role_maker=None):
        """
        should be called only once in user's python scripts,
        init() will initialize RoleMaker which is used for identifying
            current node's role, e.g. worker, server, etc.

        Args:
            role_maker(RoleMakerBase): subclass of RoleMakerBase.

        Returns:
            None
        """
        self._executor = Executor(fluid.CPUPlace())

        if role_maker and not isinstance(role_maker, RoleMakerBase):
            from paddle.fluid.incubate.fleet.base.role_maker import RoleMakerBase as RoleMakerBaseIncubate
            if role_maker and not isinstance(role_maker, RoleMakerBaseIncubate):
                raise TypeError(
                    "role_maker must be an instance of RoleMakerBase")

        self._role_maker = role_maker
        self._role_maker.generate_role()
        self._is_initialized = True

    def all_reduce_worker(self, input, output):
        """
        all reduce between workers, only support array of one dim.

        Args:
            input(list|numpy.array): array of one dim
            output(list|numpy.array): array of one dim
        """
        self._role_maker.all_reduce_worker(input, output)

    def barrier_worker(self):
        """
        barrier between workers
        """
        self._role_maker.barrier_worker()

    @abc.abstractmethod
    def init_worker(self):
        pass

    @abc.abstractmethod
    def init_server(self, model_dir=None, **kwargs):
        pass

    @abc.abstractmethod
    def run_server(self):
        pass

    @abc.abstractmethod
    def stop_worker(self):
        pass

    @abc.abstractmethod
    def distributed_optimizer(self, optimizer, strategy=None):
        pass

    @abc.abstractmethod
    def save_inference_model(self,
                             executor,
                             dirname,
                             feeded_var_names,
                             target_vars,
                             main_program=None,
                             export_for_deployment=True):
        pass

    @abc.abstractmethod
    def save_persistables(self, executor, dirname, main_program=None):
        pass


class DistributedOptimizer(object):
    """
    DistributedOptimizer is a wrapper for paddle.fluid.optimizer
    A user should pass a paddle.fluid.optimizer to DistributedOptimizer
    minimize() function is implemented.
    DistributedOptimizer is the starting point for a user who wants to
    run distributed training. The optimized information will be stored in
    Fleet() instance who holds the global information about current distributed
    training.

    Args:
        optimizer(Optimizer): subclass of Optimizer.
        strategy(any): the user define config for Optimizer.

    Returns:
        None

    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, optimizer, strategy=None):
        if not isinstance(optimizer, SGD.__bases__) \
                and not isinstance(optimizer, OptimizerWithMixedPrecision) \
                and not isinstance(optimizer, SGD_v2.__base__):
            raise TypeError("optimizer must be an instance of Optimizer")

        self._optimizer = optimizer
        self._strategy = strategy

    @abc.abstractmethod
    def backward(self,
                 loss,
                 startup_program=None,
                 parameter_list=None,
                 no_grad_set=None,
                 callbacks=None):
        """
        First part of `minimize`, do auto-diff to append backward ops for
        the current program.

        Args:
            loss (Variable): loss variable to run optimizations.
            startup_program (Program): startup_program for initializing parameters
                in `parameter_list`.
            parameter_list (list): list of Variables to update.
            no_grad_set (set|None): set of Variables should be ignored.
            callbacks (list|None): list of callables to run when appending backward
                operator for one parameter.

        Return:
            list: list of (param, grad) pair, grad is the output of backward.

        Examples:
            See examples in `apply_gradients`.
        """
        pass

    @abc.abstractmethod
    def apply_gradients(self, params_grads):
        """
        Second part of `minimize`, appending optimization operators for
        given `params_grads` pairs.

        Args:
            params_grads (list): list of (param, grad) pair to do optimization.

        Returns:
            list: A list of operators appended to the current program.

        Examples:
            .. code-block:: python

                loss = network()
                optimizer = fluid.optimizer.SGD(learning_rate=0.1)
                params_grads = optimizer.backward(loss)
                # you may append operations for params_grads here
                # ...
                optimizer.apply_gradients(params_grads)
        """
        pass

    @abc.abstractmethod
    def minimize(self,
                 losses,
                 scopes=None,
                 startup_programs=None,
                 parameter_list=None,
                 no_grad_set=None):
        """
        Add operations to minimize `loss` by updating `parameter_list`.

        This method combines interface `backward()` and
        `apply_gradients()` into one.

        Args:
            losses (Variable|Variable List): loss variable to run optimizations.
            scopes (Scope| Scope List): scope instance.
            startup_programs (Program|Program List): startup_program for initializing parameters
                in `parameter_list`.
            parameter_list (list): list of Variables to update.
            no_grad_set (set|None): set of Variables should be ignored.

        Returns:
            tuple: (optimize_ops, params_grads) which are, list of operators appended;
            and list of (param, grad) Variables pair for optimization.
        """
        pass
