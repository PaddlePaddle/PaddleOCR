# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function
import os

import paddle.fluid as fluid
from paddle.fluid import core, unique_name
from ..base.private_helper_function import wait_server_ready

__all__ = []

OpRole = core.op_proto_and_checker_maker.OpRole

OP_ROLE_KEY = core.op_proto_and_checker_maker.kOpRoleAttrName()
OP_ROLE_VAR_KEY = core.op_proto_and_checker_maker.kOpRoleVarAttrName()


def is_update_op(op):
    return 'Param' in op.input_names and 'Grad' in op.input_names and \
            "LearningRate" in op.input_names


def is_loss_grad_op(op):
    if OP_ROLE_KEY not in op.attr_names:
        return False
    op_role = int(op.all_attrs()[OP_ROLE_KEY])
    return op_role & int(OpRole.Backward) and op_role & int(OpRole.Loss)


def is_backward_op(op):
    return OP_ROLE_KEY in op.attr_names and \
            int(op.all_attrs()[OP_ROLE_KEY]) & int(OpRole.Backward)


def is_optimizer_op(op):
    return OP_ROLE_KEY in op.attr_names and \
            int(op.all_attrs()[OP_ROLE_KEY]) & int(OpRole.Optimize)


class CollectiveHelper(object):
    def __init__(self, role_maker, nrings=1, wait_port=True):
        self.nrings = nrings
        self.wait_port = wait_port
        self.role_maker = role_maker

    def update_startup_program(self, startup_program=None):
        self.startup_program = startup_program
        if startup_program is None:
            self.startup_program = fluid.default_startup_program()

        endpoints = self.role_maker._get_trainer_endpoints()
        current_endpoint = endpoints[self.role_maker._worker_index()]
        for ring_id in range(self.nrings):
            self._init_communicator(
                self.startup_program, current_endpoint, endpoints,
                self.role_maker._worker_index(), ring_id, self.wait_port)
        self._broadcast_params()

    def _init_communicator(self,
                           program,
                           current_endpoint,
                           endpoints,
                           rank,
                           ring_id,
                           wait_port,
                           global_ring_id=None,
                           sync=True):
        # if current_endpoint is None, it means just for sync,
        # no group is created.
        if current_endpoint:
            nranks = len(endpoints)
            other_endpoints = endpoints[:]
            other_endpoints.remove(current_endpoint)

        if rank == 0 and wait_port:
            wait_server_ready(other_endpoints)

        def _add_sync_by_allreduce(block):
            sync_var = block.create_var(
                name=unique_name.generate('sync_var'),
                dtype=core.VarDesc.VarType.INT32,
                persistable=False,
                stop_gradient=True)
            block.append_op(
                type='fill_constant',
                inputs={},
                outputs={'Out': [sync_var]},
                attrs={
                    'shape': [1],
                    'dtype': sync_var.dtype,
                    'value': 1,
                    'force_cpu': False,
                    OP_ROLE_KEY: OpRole.Forward
                })
            block.append_op(
                type='c_allreduce_sum',
                inputs={'X': [sync_var]},
                outputs={'Out': [sync_var]},
                attrs={
                    'ring_id': global_ring_id,
                    'use_calc_stream': True,
                    OP_ROLE_KEY: OpRole.Forward
                })
            block.append_op(
                type='c_sync_calc_stream',
                inputs={'X': sync_var},
                outputs={'Out': sync_var},
                attrs={OP_ROLE_KEY: OpRole.Forward})

        block = program.global_block()
        if current_endpoint is None:
            assert endpoints is None
            assert sync
            _add_sync_by_allreduce(block)
            return

        comm_id_var = block.create_var(
            name=unique_name.generate('comm_id'),
            persistable=True,
            type=core.VarDesc.VarType.RAW)
        if core.is_compiled_with_cuda():
            block.append_op(
                type='c_gen_nccl_id',
                inputs={},
                outputs={'Out': comm_id_var},
                attrs={
                    'rank': rank,
                    'endpoint': current_endpoint,
                    'other_endpoints': other_endpoints,
                    'ring_id': ring_id,
                    OP_ROLE_KEY: OpRole.Forward
                })
            block.append_op(
                type='c_comm_init',
                inputs={'X': comm_id_var},
                outputs={},
                attrs={
                    'nranks': nranks,
                    'rank': rank,
                    'ring_id': ring_id,
                    OP_ROLE_KEY: OpRole.Forward
                })
        elif core.is_compiled_with_xpu():
            block.append_op(
                type='c_gen_bkcl_id',
                inputs={},
                outputs={'Out': comm_id_var},
                attrs={
                    'rank': rank,
                    'endpoint': current_endpoint,
                    'other_endpoints': other_endpoints,
                    'ring_id': ring_id,
                    OP_ROLE_KEY: OpRole.Forward
                })
            block.append_op(
                type='c_comm_init',
                inputs={'X': comm_id_var},
                outputs={},
                attrs={
                    'nranks': nranks,
                    'rank': rank,
                    'ring_id': ring_id,
                    OP_ROLE_KEY: OpRole.Forward
                })
        elif core.is_compiled_with_npu():
            block.append_op(
                type='c_gen_hccl_id',
                inputs={},
                outputs={'Out': comm_id_var},
                attrs={
                    'rank': rank,
                    'endpoint': current_endpoint,
                    'other_endpoints': other_endpoints,
                    'ring_id': ring_id,
                    OP_ROLE_KEY: OpRole.Forward
                })
            block.append_op(
                type='c_comm_init_hccl',
                inputs={'X': comm_id_var},
                outputs={},
                attrs={
                    'rank': rank,
                    'ring_id': ring_id,
                    'device_id': int(os.getenv("FLAGS_selected_npus")),
                    'rank_ids': nranks,
                    OP_ROLE_KEY: OpRole.Forward
                })
        else:
            raise ValueError(
                "comm_id must be generated in paddlepaddle-xpu or paddlepaddle-xpu."
            )
        if sync: _add_sync_by_allreduce(block)

    def _wait(self, current_endpoint, endpoints):
        assert (self.wait_port)
        other_endpoints = endpoints[:]
        other_endpoints.remove(current_endpoint)
        wait_server_ready(other_endpoints)

    def _broadcast_params(self):
        block = self.startup_program.global_block()
        ring_id = -1
        for param in block.iter_parameters():
            if param.is_distributed:
                continue

            ring_id = (ring_id + 1) % self.nrings
            block.append_op(
                type='c_broadcast',
                inputs={'X': param},
                outputs={'Out': param},
                attrs={
                    'ring_id': ring_id,
                    'root': 0,
                    OP_ROLE_KEY: OpRole.Forward
                })

        for ring_id in range(self.nrings):
            block.append_op(
                type='c_sync_comm_stream',
                inputs={'X': param},
                outputs={'Out': param},
                attrs={'ring_id': ring_id,
                       OP_ROLE_KEY: OpRole.Forward})
