#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

from .node import DownpourServer
from .node import DownpourWorker
from ..backward import append_backward
import ps_pb2 as pslib
from paddle.fluid.distribute_lookup_table import find_distributed_lookup_table
from paddle.fluid.distribute_lookup_table import find_distributed_lookup_table_inputs
from paddle.fluid.distribute_lookup_table import find_distributed_lookup_table_outputs
from google.protobuf import text_format


class DownpourSGD(object):
    r"""
    Distributed optimizer of downpour stochastic gradient descent
    Standard implementation of Google's Downpour SGD
    in Large Scale Distributed Deep Networks

    Args:
        learning_rate (float): the learning rate used to update parameters. \
        Can be a float value
    Examples:
        .. code-block:: python
    
             opt = fluid.DistributedOptimizer(sgd_opt)
             opt.minimize()

             downpour_sgd = fluid.distributed.DownpourSGD(learning_rate=0.2)
             downpour_sgd.minimize(cost)
    """

    def __init__(self, learning_rate=0.001, window=1):
        # todo(guru4elephant): add more optimizers here as argument
        # todo(guru4elephant): make learning_rate as a variable
        self.learning_rate_ = learning_rate
        self.window_ = window
        self.type = "downpour"
        self.data_norm_name = [
            ".batch_size", ".batch_square_sum", ".batch_sum",
            ".batch_size@GRAD", ".batch_square_sum@GRAD", ".batch_sum@GRAD"
        ]

    def minimize(self,
                 losses,
                 startup_program=None,
                 parameter_list=None,
                 no_grad_set=None):
        """
        DownpounSGD is a distributed optimizer so
        that user can call minimize to generate backward
        operators and optimization operators within minimize function
        Args:
            loss(Variable): loss variable defined by user
            startup_program(Program): startup program that defined by user
            parameter_list(str list): parameter names defined by users
            no_grad_set(set): a set of variables that is defined by users
            so that these variables do not need gradient computation
        Returns:
            [ps_param, worker_skipped_ops]
            ps_param: parameter server protobuf desc
            worker_skipped_ops: operator names that need
            to be skipped during execution
        """
        if not isinstance(losses, list):
            raise ValueError('losses is a list, just lick [model.cost]')
        table_name = find_distributed_lookup_table(losses[0].block.program)
        prefetch_slots = find_distributed_lookup_table_inputs(
            losses[0].block.program, table_name)
        prefetch_slots_emb = find_distributed_lookup_table_outputs(
            losses[0].block.program, table_name)

        ps_param = pslib.PSParameter()
        server = DownpourServer()
        worker = DownpourWorker(self.window_)
        sparse_table_index = 0
        server.add_sparse_table(sparse_table_index, self.learning_rate_,
                                prefetch_slots, prefetch_slots_emb)
        worker.add_sparse_table(sparse_table_index, self.learning_rate_,
                                prefetch_slots, prefetch_slots_emb)
        dense_table_index = 1
        program_configs = []
        param_grads_list = []
        for loss_index in range(len(losses)):
            program_config = ps_param.trainer_param.program_config.add()
            program_config.program_id = str(
                id(losses[loss_index].block.program))
            program_config.pull_sparse_table_id.extend([sparse_table_index])
            program_config.push_sparse_table_id.extend([sparse_table_index])
            params_grads = sorted(
                append_backward(losses[loss_index], parameter_list,
                                no_grad_set),
                key=lambda x: x[0].name)
            param_grads_list.append(params_grads)
            params = []
            grads = []
            data_norm_params = []
            data_norm_grads = []
            for i in params_grads:
                is_data_norm_data = False
                for data_norm_name in self.data_norm_name:
                    if i[0].name.endswith(data_norm_name):
                        is_data_norm_data = True
                        data_norm_params.append(i[0])
                if not is_data_norm_data:
                    params.append(i[0])
            for i in params_grads:
                is_data_norm_data = False
                for data_norm_grad in self.data_norm_name:
                    if i[0].name.endswith(data_norm_grad):
                        is_data_norm_data = True
                        data_norm_grads.append(i[1])
                if not is_data_norm_data:
                    grads.append(i[1])
            server.add_dense_table(dense_table_index, self.learning_rate_,
                                   params, grads)
            worker.add_dense_table(dense_table_index, self.learning_rate_,
                                   params, grads)
            program_config.pull_dense_table_id.extend([dense_table_index])
            program_config.push_dense_table_id.extend([dense_table_index])
            if len(data_norm_params) != 0 and len(data_norm_grads) != 0:
                dense_table_index += 1
                server.add_data_norm_table(dense_table_index,
                                           self.learning_rate_,
                                           data_norm_params, data_norm_grads)
                worker.add_dense_table(dense_table_index, self.learning_rate_,
                                       data_norm_params, data_norm_grads)
                program_config.pull_dense_table_id.extend([dense_table_index])
                program_config.push_dense_table_id.extend([dense_table_index])
            dense_table_index += 1
            program_configs.append(program_config)
        ps_param.server_param.CopyFrom(server.get_desc())
        ps_param.trainer_param.CopyFrom(worker.get_desc())
        for program_config in program_configs:
            ps_param.trainer_param.program_config.extend([program_config])
        # Todo(guru4elephant): figure out how to support more sparse parameters
        # currently only support lookup_table
        worker_skipped_ops = ["lookup_table", "lookup_table_grad"]
        ps_param.trainer_param.skip_op.extend(worker_skipped_ops)

        # all fleet operations should be defined in operators in the future
        # we want to return an object here containing:
        # 1) worker execution strategy
        # 2) pserver execution strategy
        # 3) fleet configurations
        # 4) skipped operators in runtime
        # 5) distributed optimization
        opt_info = {}
        opt_info["trainer"] = "DistMultiTrainer"
        opt_info["device_worker"] = "DownpourSGD"
        opt_info["optimizer"] = "DownpourSGD"
        opt_info["fleet_desc"] = ps_param
        opt_info["worker_skipped_ops"] = worker_skipped_ops

        for loss in losses:
            loss.block.program._fleet_opt = opt_info

        return None, param_grads_list
