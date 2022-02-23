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
"""Parameter Server utils"""

import numpy as np
import os
import paddle
import warnings

__all__ = []


class DistributedInfer:
    """
    Utility class for distributed infer of PaddlePaddle.
    """

    def __init__(self, main_program=None, startup_program=None):
        if main_program:
            self.origin_main_program = main_program.clone()
        else:
            self.origin_main_program = paddle.static.default_main_program(
            ).clone()

        if startup_program:
            self.origin_startup_program = startup_program
        else:
            self.origin_startup_program = paddle.static.default_startup_program(
            )
        self.sparse_table_maps = None

    def init_distributed_infer_env(self,
                                   exe,
                                   loss,
                                   role_maker=None,
                                   dirname=None):
        import paddle.distributed.fleet as fleet

        if fleet.fleet._runtime_handle is None:
            fleet.init(role_maker=role_maker)

            fake_optimizer = paddle.optimizer.SGD()
            strategy = fleet.DistributedStrategy()
            strategy.a_sync = True
            optimizer = fleet.distributed_optimizer(
                fake_optimizer, strategy=strategy)
            optimizer.minimize(
                loss, startup_program=self.origin_startup_program)

            if fleet.is_server():
                fleet.init_server(dirname=dirname)
                fleet.run_server()
            else:
                exe.run(paddle.static.default_startup_program())
                fleet.init_worker()
                self._init_dense_params(exe, dirname)
            global_startup_program = paddle.static.default_startup_program()
            global_startup_program = self.origin_startup_program
            global_main_program = paddle.static.default_main_program()
            global_main_program = self.origin_main_program

    def _get_sparse_table_map(self):
        import paddle.distributed.fleet as fleet

        if self.sparse_table_maps is None:
            self.sparse_table_maps = {}
            send_ctx = fleet.fleet._runtime_handle._communicator.send_ctx_
            for gradname, ctx in send_ctx.items():
                if ctx.is_sparse:
                    param = gradname.strip("@GRAD")
                    self.sparse_table_maps[param] = ctx.table_id()
                else:
                    continue
        return self.sparse_table_maps

    def _init_dense_params(self, exe=None, dirname=None):
        import paddle.distributed.fleet as fleet

        sparse_table_maps = self._get_sparse_table_map()

        if dirname is not None and exe is not None:
            all_persist_vars = [
                v for v in self.origin_main_program.list_vars()
                if paddle.static.io.is_persistable(v)
            ]
            dense_persist_vars = [(v.name, v) for v in all_persist_vars
                                  if v.name not in sparse_table_maps]
            need_load_vars = [
                v[1] for v in dense_persist_vars
                if os.path.isfile(os.path.join(dirname, v[0]))
            ]
            paddle.static.load_vars(
                exe,
                dirname,
                main_program=self.origin_main_program,
                vars=need_load_vars)

    def get_dist_infer_program(self):
        varname2tables = self._get_sparse_table_map()
        convert_program = self._convert_program(self.origin_main_program,
                                                varname2tables)
        return convert_program

    def _convert_program(self, main_program, varname2tables):
        def distributed_ops_pass(program):
            SPARSE_OP_TYPE_DICT = {"lookup_table": "W", "lookup_table_v2": "W"}

            def _get_pull_sparse_ops(_program):
                pull_sparse_ops = {}
                for op in _program.global_block().ops:
                    if op.type in SPARSE_OP_TYPE_DICT.keys() \
                            and op.attr('remote_prefetch') is True:
                        param_name = op.input(SPARSE_OP_TYPE_DICT[op.type])[0]
                        ops = pull_sparse_ops.get(param_name, [])
                        ops.append(op)
                        pull_sparse_ops[param_name] = ops
                return pull_sparse_ops

            def _pull_sparse_fuse(_program, pull_sparse_ops):
                for param, ops in pull_sparse_ops.items():
                    all_ops = program.global_block().ops
                    op_idxs = [all_ops.index(op) for op in ops]

                    inputs = [
                        program.global_block().vars[op.input("Ids")[0]]
                        for op in ops
                    ]

                    w = program.global_block().vars[ops[0].input("W")[0]]

                    if w.name not in varname2tables.keys():
                        raise ValueError(
                            "can not find variable {}, please check your configuration".
                            format(w.name))

                    table_id = varname2tables[w.name]

                    padding_idx = ops[0].attr("padding_idx")
                    is_distributed = ops[0].attr("is_distributed")
                    op_type = ops[0].type

                    outputs = [
                        program.global_block().vars[op.output("Out")[0]]
                        for op in ops
                    ]

                    for idx in op_idxs[::-1]:
                        program.global_block()._remove_op(idx)

                    inputs_idxs = [-1] * len(inputs)
                    outputs_idxs = [-1] * len(outputs)

                    for idx, op in enumerate(program.global_block().ops):
                        for i in range(0, len(op.output_names)):
                            outs = op.output(op.output_names[i])
                            for in_id, in_var in enumerate(inputs):
                                if in_var.name in outs:
                                    inputs_idxs[in_id] = idx
                        for i in range(0, len(op.input_names)):
                            ins = op.input(op.input_names[i])
                            for out_id, out_var in enumerate(outputs):
                                if out_var.name in ins:
                                    outputs_idxs[out_id] = idx

                    if min(outputs_idxs) - max(inputs_idxs) >= 1:
                        distributed_idx = max(inputs_idxs) + 1

                        program.global_block()._insert_op(
                            index=distributed_idx,
                            type="distributed_lookup_table",
                            inputs={"Ids": inputs,
                                    'W': w},
                            outputs={"Outputs": outputs},
                            attrs={
                                "is_distributed": is_distributed,
                                "padding_idx": padding_idx,
                                "table_id": table_id,
                                "is_test": True,
                                "lookup_table_version": op_type
                            })
                    else:
                        raise ValueError(
                            "something wrong with Fleet, submit a issue is recommended"
                        )

            pull_sparse_ops = _get_pull_sparse_ops(program)
            warnings.warn(
                "lookup_table will be forced to test mode when use DistributedInfer"
            )
            _pull_sparse_fuse(program, pull_sparse_ops)
            return program

        covert_program = distributed_ops_pass(main_program)
        return covert_program
