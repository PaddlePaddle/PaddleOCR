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

from . import collective
from .. import core
OpRole = core.op_proto_and_checker_maker.OpRole
from paddle.distributed import fleet


class AscendTranspiler(collective.Collective):
    def __init__(self, startup_program, main_program):
        self.nrings = 1
        super(AscendTranspiler, self).__init__(self.nrings)
        self._startup_program = startup_program
        self._main_program = main_program

    def _insert_allreduce_ops(self):
        block = self._main_program.global_block()
        ring_id = -1
        grad = None
        for idx, op in reversed(list(enumerate(block.ops))):
            if self._is_backward_op(op) and \
                    self.op_role_var_key in op.attr_names:
                op_role_var = op.all_attrs()[self.op_role_var_key]

                if len(op_role_var) == 0:
                    continue
                assert len(op_role_var) % 2 == 0

                offset = idx
                for i in range(0, len(op_role_var), 2):
                    param = block.vars[op_role_var[i]]
                    grad = block.vars[op_role_var[i + 1]]
                    if param.is_distributed:
                        continue

                    # As we search ops reversedly, we should insert c_allreduce_sum
                    # op in the same way to keep the ring_id alternate
                    ring_id = (ring_id + 1) % self.nrings
                    block._insert_op(
                        offset + 1,
                        type='c_allreduce_sum',
                        inputs={'X': grad},
                        outputs={'Out': grad},
                        attrs={
                            'ring_id': ring_id,
                            self.op_role_key: OpRole.Backward
                        })
                    block._insert_op(
                        offset + 2,
                        type='scale',
                        inputs={'X': grad},
                        outputs={'Out': grad},
                        attrs={
                            'scale': 1.0 / fleet.worker_num(),
                            self.op_role_key: OpRole.Backward
                        })

        if grad is None:
            return

    def transpile(self):
        self._insert_allreduce_ops()
