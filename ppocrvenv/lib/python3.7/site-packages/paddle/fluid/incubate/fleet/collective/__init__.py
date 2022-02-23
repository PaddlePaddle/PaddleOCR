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

import logging

import paddle.fluid as fluid
import paddle.fluid.io as io
import paddle.fluid.transpiler.distribute_transpiler as dist_transpiler
from paddle.fluid.executor import Executor
from paddle.fluid.parallel_executor import ParallelExecutor
from paddle.fluid.compiler import CompiledProgram
from paddle.fluid.framework import Program

from paddle.fluid.incubate.fleet.base.fleet_base import Fleet
from paddle.fluid.incubate.fleet.base.fleet_base import Mode
from paddle.fluid.incubate.fleet.base.fleet_base import DistributedOptimizer

from paddle.fluid import compiler
from paddle.fluid.incubate.checkpoint.checkpoint_saver import PaddleModel, CheckpointSaver

import paddle

import os
import sys
import six
import json
import re
import shutil


class LambConfig(object):
    def __init__(self):
        pass


class DistFCConfig(object):
    def __init__(self):
        pass


class Collective(Fleet):
    def __init__(self):
        super(Collective, self).__init__(Mode.COLLECTIVE)
        self._local_ip = 0

        self.startup_program = None
        self._origin_program = None
        self._transpiled_program = None
        self.main_program = None
        self._checkpoint_prefix = "__paddle_fleet_checkpoint__"
        self._param_file_name = "_paddle_fleet_param__"

    def init_worker(self):
        logging.warn(
            "You should not call 'init_worker' method for collective mode.")

    def run_worker(self, main_programs=None, scopes=None):
        logging.warn(
            "You should not call 'run_worker' method for collective mode.")

    def init_server(self, model_dir=None):
        logging.warn(
            "You should not call 'init_server' method for collective mode.")

    def run_server(self):
        logging.warn(
            "You should not call 'run_server' method for collective mode.")

    def stop_worker(self):
        logging.warn(
            "You should not call 'stop_worker' method for collective mode.")

    def distributed_optimizer(self, optimizer, strategy=None):
        self._optimizer = \
            CollectiveOptimizer(optimizer, strategy)
        return self._optimizer

    def save_inference_model(self,
                             executor,
                             dirname,
                             feeded_var_names=None,
                             target_vars=None,
                             main_program=None,
                             export_for_deployment=True):
        """
        Prune the given `main_program` to build a new program especially for
        inference, and then save it and all related parameters to given
        `dirname` by the `executor`.
        """
        assert isinstance(executor, Executor), \
            "In fleet.save_inference_model() function, executor must be as" \
            " Executor type."

        if main_program is None:
            main_program = self._origin_program
        assert isinstance(main_program, Program), \
            "In fleet.save_inference_model() function, main_program " \
            "must be as Program type."

        io.save_inference_model(dirname, feeded_var_names, target_vars,
                                executor, main_program, None, None,
                                export_for_deployment)

    def save_persistables(self,
                          executor,
                          dirname,
                          main_program=None,
                          filename=None):
        """
        This function filters out all variables with `persistable==True` from
        the give `main_program` and then saves these variables to the folder
        `dirname` or file `filename`.

        The `dirname` is used to specify the folder where persistable variables
        are going to be saved. If you would like to save variables in separate
        files, set `filename` None; if you would like to save all variables in a
        single file, use `filename` to specify the file name.
        """
        assert isinstance(executor, Executor), \
            "In fleet.save_inference_model() function, executor must be as" \
            " Executor type."

        if main_program is None:
            main_program = self._origin_program

        assert isinstance(main_program, Program), \
            "In fleet.save_inference_model() function, main_program " \
            "must be as Program type."

        io.save_persistables(executor, dirname, main_program, filename=filename)

    def save_checkpoint(self,
                        executor,
                        path,
                        trainer_id,
                        train_status,
                        fs,
                        main_program=None,
                        local_cache_path=".cache",
                        remain_all_checkpoint=True):
        """
        This function save persistables and current epoch num to path.
        """
        if main_program == None:
            main_program = self._transpiled_program

        m = PaddleModel(executor, main_program)
        t = train_status
        c = CheckpointSaver(fs)
        real_path, checkpoint_no = c.save_checkpoint(
            path=path,
            slists=[m, t],
            trainer_id=trainer_id,
            local_cache_path=local_cache_path)

        if not remain_all_checkpoint:
            c.clean_redundant_checkpoints(path)

        return real_path, checkpoint_no

    def load_checkpoint(self,
                        executor,
                        path,
                        trainer_id,
                        train_status,
                        fs,
                        main_program=None,
                        local_cache_path=".cache",
                        ignore_empty=True):
        """
        This function load persistables and current epoch num from path.
        """

        if main_program == None:
            main_program = self._transpiled_program

        m = PaddleModel(executor, main_program)
        c = CheckpointSaver(fs)
        return c.load_checkpoint(
            path, [m, train_status],
            trainer_id=trainer_id,
            ignore_empty=ignore_empty,
            local_cache_path=local_cache_path)


fleet = Collective()


class DistributedStrategy(fluid.BuildStrategy):
    """
    Init function of DistributedStrategy
    """

    def __init__(self):
        super(DistributedStrategy, self).__init__()
        self.use_local_sgd = False
        self.use_dist_fc = False

        self.dist_fc_config = None  # DistFCConfig
        self.mode = "nccl2"  # or collective
        self.collective_mode = None  # local_sgd or grad_allreduce
        self.nccl_comm_num = 1
        self.forward_recompute = False  # use RecomputeOptimizer
        self.recompute_checkpoints = []
        self.use_amp = False  # use mixed precision optimizer
        self.amp_loss_scaling = 2**15

        self.exec_strategy = fluid.ExecutionStrategy()

        # configurations below are used for unit test
        self._ut4grad_allreduce = False


class CollectiveOpBasedOptimizer(DistributedOptimizer):
    """
    Collective Operator Base Class For Distributed Optimizer
    The class is invisible to a user
    """

    def __init__(self, optimizer, strategy=None):
        assert isinstance(
            strategy,
            DistributedStrategy), "strategy must be DistributedStrategy"
        super(CollectiveOpBasedOptimizer, self).__init__(optimizer, strategy)

    def backward(self,
                 loss,
                 startup_program=None,
                 parameter_list=None,
                 no_grad_set=None,
                 callbacks=None):
        return self._optimizer.backward(loss, startup_program, parameter_list,
                                        no_grad_set, callbacks)

    def apply_gradients(self, params_grads):
        return self._optimizer.apply_gradients(params_grads)


class CollectiveOptimizer(DistributedOptimizer):
    """
    DistributedOptimizer is a wrapper for paddle.fluid.optimizer
    A user should pass a paddle.fluid.optimizer to DistributedOptimizer
    minimize() function is implemented.
    DistributedOptimizer is the starting point for a user who wants to
    run distributed training. The optimized information will be stored in
    Fleet() instance who holds the global information about current distributed
    training.
    """

    def __init__(self, optimizer, strategy=DistributedStrategy()):
        if strategy is None:
            strategy = DistributedStrategy()
        super(CollectiveOptimizer, self).__init__(optimizer, strategy)
        self._forward_recompute = strategy.forward_recompute
        if (not isinstance(strategy.recompute_checkpoints, list)):
            raise ValueError("DistStrategy.recompute_checkpoints should"
                             "be a List")
        self._recompute_checkpoints = strategy.recompute_checkpoints
        self._use_amp = strategy.use_amp
        self._amp_loss_scaling = strategy.amp_loss_scaling
        self.print_config = False

    def backward(self,
                 loss,
                 startup_program=None,
                 parameter_list=None,
                 no_grad_set=None,
                 callbacks=None):
        return self._optimizer.backward(loss, startup_program, parameter_list,
                                        no_grad_set, callbacks)

    def apply_gradients(self, params_grads):
        return self._optimizer.apply_gradients(params_grads)

    def _check_condition(self, name, **kwargs):
        for k, v in six.iteritems(kwargs):
            if v is True:
                assert False, "you can't use %s and %s together" % (name, k)

    def _check_collective_mode(self, main_program, optimizer, strategy):
        """
        Check the conflict conditions.
        """
        if strategy.use_local_sgd:
            strategy.mode = "collective"
            strategy.collective_mode = "local_sgd"
            self._check_condition(
                "use_local_sgd",
                use_dgc=main_program._enable_dgc,
                use_dist_fc=strategy.use_dist_fc,
                use_lamb=main_program._use_lamb)

        if strategy.use_dist_fc:
            self._check_condition(
                "use_dist_fc",
                use_dgc=main_program._enable_dgc,
                use_local_sgd=strategy.use_local_sgd,
                use_lamb=main_program._use_lamb)
            assert strategy.dist_fc_config is not None, "DistributedStrategy.dist_fc_config should be set"

        if strategy._ut4grad_allreduce:
            strategy.mode = "collective"
            strategy.collective_mode = "grad_allreduce"
            self._check_condition(
                "_ut4grad_allreduce",
                use_dgc=main_program._enable_dgc,
                use_lamb=main_program._use_lamb)

        if self._strategy.collective_mode=="local_sgd" \
                or self._strategy.collective_mode == "grad_allreduce":
            assert self._strategy.mode == "collective", \
                "local_sgd and grad_allreduce can be used under collective mode"

    def _transpile(self, startup_program, main_program):
        """
        Transpile the programs to distributed programs. And add the variables.
        """
        worker_endpoints = fleet.worker_endpoints()
        trainer_id = fleet.worker_index()
        current_endpoint = fleet.worker_endpoints()[trainer_id]
        worker_endpoints_env = ','.join(worker_endpoints)
        trainers_num = fleet.worker_num()

        if self.print_config:
            print("worker_endpoints:{} trainers_num:{} current_endpoint:{} \
                  trainer_id:{}".format(worker_endpoints, trainers_num,
                                        current_endpoint, trainer_id))

        # call transpiler
        config = dist_transpiler.DistributeTranspilerConfig()
        config.mode = self._strategy.mode
        config.collective_mode = self._strategy.collective_mode

        config.nccl_comm_num = self._strategy.nccl_comm_num
        config.use_hierarchical_allreduce = self._strategy.use_hierarchical_allreduce
        config.hierarchical_allreduce_inter_nranks = self._strategy.hierarchical_allreduce_inter_nranks

        t = dist_transpiler.DistributeTranspiler(config=config)
        t.transpile(
            trainer_id=trainer_id,
            trainers=worker_endpoints_env,
            startup_program=startup_program,
            program=main_program,
            current_endpoint=current_endpoint)

    def _get_node_ips_from_endpoints(self, endpoints):
        ss = set()
        ips = []
        for ep in endpoints:
            ip = ep.split(":")[0].strip()
            if ip not in ss:
                ss.add(ip)
                ips.append(ip)
            else:
                continue

        return ips

    def _node_num(self):
        worker_endpoints = fleet.worker_endpoints()
        current_endpoint = fleet.worker_endpoints()[fleet.worker_index()]
        worker_endpoints_env = ','.join(worker_endpoints)

        node_ips = self._get_node_ips_from_endpoints(worker_endpoints)
        node_ip = current_endpoint.split(":")[0].strip()

        node_num = len(node_ips)

        return node_num

    def _try_to_compile(self, startup_program, main_program):
        node_num = self._node_num()
        assert node_num >= 1, "nccl2 node_num must >= 1, now:{}" % node_num

        exec_strategy = self._strategy.exec_strategy

        if node_num <= 1:
            if self._strategy.nccl_comm_num > 1:
                logging.warn("set nccl_comm_num=1 since you only have 1 node.")
            self._strategy.nccl_comm_num = 1

            if self._strategy.use_hierarchical_allreduce:
                logging.warn(
                    "set use_hierarchical_allreduce=False since you only have 1 node."
                )
            self._strategy.use_hierarchical_allreduce = False

        sync_allreduce = os.getenv("FLAGS_sync_nccl_allreduce")
        if sync_allreduce is None or sync_allreduce == "1":
            exec_strategy.num_threads = self._strategy.nccl_comm_num + 1
            if self._strategy.use_hierarchical_allreduce:
                exec_strategy.num_threads = 2 * self._strategy.nccl_comm_num + 1
            if exec_strategy.num_threads > 4:
                logging.warn(
                    "if you use use_hierarchical_allreduce or "
                    "with multi nccl comm, please export FLAGS_sync_nccl_allreduce = 0"
                )

        # NOTE. open sync_batch_norm will hang when use multi num_threads
        sync_batch_norm = self._strategy.sync_batch_norm
        if sync_batch_norm is not None and sync_batch_norm is True:
            self._strategy.nccl_comm_num = 1
            self._strategy.use_hierarchical_allreduce = False
            exec_strategy.num_threads = 1
            logging.warn(
                "use sync_batch_norm will hang when set num_threads > 1, so "
                "set num_threads=1, nccl_comm_num=1, use_hierarchical_allreduce=False."
            )

        if self.print_config:
            print("node_num:", node_num, "num_threads:",
                  exec_strategy.num_threads, "use_hierarchical_allreduce:",
                  self._strategy.use_hierarchical_allreduce, "nccl_comm_num:",
                  self._strategy.nccl_comm_num, "FLAGS_sync_nccl_allreduce:",
                  sync_allreduce)

        self._transpile(startup_program, main_program)

        if self._strategy.mode == "collective":
            return main_program

        self._strategy.num_trainers = fleet.worker_num()
        self._strategy.trainer_id = fleet.worker_index()
        self._strategy.trainers_endpoints = fleet.worker_endpoints()
        self._strategy.enable_backward_optimizer_op_deps = True

        self._compiled_program = compiler.CompiledProgram(main_program)

        self._compiled_program.with_data_parallel(
            loss_name=self._loss.name,
            build_strategy=self._strategy,
            exec_strategy=self._strategy.exec_strategy,
            share_vars_from=None)

        return self._compiled_program

    def raiseOptimizeError(self, strategy_name, optimize_name):
        raise ValueError("can not use {0} when you set DistStrategy.{1} "
                         "as True".format(optimize_name, strategy_name))

    def minimize(self,
                 loss,
                 startup_program=None,
                 parameter_list=None,
                 no_grad_set=None):
        """
        minimize a program through loss
        Args:
            loss (Variable|Variable List): loss variable or loss variable list to run optimization.
            startup_program (Program): startup_program for initializing parameters
                in `parameter_list`.
            parameter_list (list): list of Variables to update.
            no_grad_set (set|None): set of Variables should be ignored.
        Returns:
            tuple: (optimize_ops, params_grads) which are, list of operators appended;
            and list of (param, grad) Variables pair for optimization.
        Note that in parameter server mode, a worker will not get anything about optimize_os
        Because optimizer algorithms run on pserver side. We will make this usable in pserver
        process, but currently the optimization part is written into Fleet(). A user does not
        need to care about how to startup a pserver node.
        """

        # check optimizer conflicts
        if self._forward_recompute:
            if self._recompute_checkpoints == []:
                raise ValueError("please set strategy.recompute_checkpoints"
                                 "when set strategy.forward_recompute as True")
            if self._optimizer.__class__.__name__ in [
                    "RecomputeOptimizer", "OptimizerWithMixedPrecision"
            ]:
                self.raiseOptimizeError("forward_recompute",
                                        self._optimizer.__class__.__name__)

            self._optimizer = \
                fluid.optimizer.RecomputeOptimizer(self._optimizer)
            self._optimizer._set_checkpoints(self._recompute_checkpoints)

        if self._use_amp:
            if self._optimizer.__class__.__name__ in [
                    "OptimizerWithMixedPrecision", "DGCMomentumOptimizer"
            ]:
                self.raiseOptimizeError("mixed_precision",
                                        self._optimizer.__class__.__name__)
            self._optimizer = fluid.contrib.mixed_precision.decorate(
                self._optimizer,
                init_loss_scaling=self._amp_loss_scaling,
                use_dynamic_loss_scaling=True)

        main_program = loss.block.program
        if startup_program is None:
            startup_program = fluid.default_startup_program()
        fleet.startup_program = startup_program

        self._loss = loss

        self._check_collective_mode(main_program, self._optimizer,
                                    self._strategy)

        optimize_ops, param_grads = self._optimizer.minimize(
            loss, startup_program, parameter_list, no_grad_set=no_grad_set)

        fleet._origin_program = main_program.clone(for_test=False)
        fleet._transpiled_program = main_program
        fleet.main_program = self._try_to_compile(startup_program, main_program)

        return optimize_ops, param_grads
