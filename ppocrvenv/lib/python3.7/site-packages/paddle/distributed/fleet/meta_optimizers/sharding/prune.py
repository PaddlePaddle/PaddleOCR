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

__all__ = []


class ProgramDeps(object):
    def __init__(self, block, start_vars, end_vars):
        self._block = block
        # vars where to start to build the deps
        self._start_vars = start_vars
        # vars where to stop to build the deps
        self._end_vars = end_vars
        # var name -> op idxs which depends on this var
        self._var_to_use_op = {}
        # sub block deps which is a subset of this topo
        self._sub_block_deps = {}
        # var name -> op idxs which generate var
        self._var_to_generate_op = {}
        self._should_removed_var = set()
        self._father_block_deps = None
        self._build_deps()

    def get_sub_block_deps(self, idx):
        if idx in self._sub_block_deps:
            return self._sub_block_deps[idx]
        else:
            return None

    def get_var_deps(self, var_name):
        if var_name in self._var_to_use_op:
            return self._var_to_use_op[var_name]
        else:
            return None

    def _build_deps(self, ):

        for var_name in self._start_vars:
            self._var_to_use_op[var_name] = []
            self._var_to_generate_op[var_name] = []

        for idx, op in enumerate(self._block.ops):
            if op.type in [
                    "c_allreduce_sum", "c_sync_comm_stream",
                    "c_calc_comm_stream"
            ]:
                continue
            input_vars = op.desc.input_arg_names()
            output_vars = op.desc.output_arg_names()
            deps_reduce = False
            for input_name in input_vars:
                if input_name in self._var_to_use_op:
                    deps_reduce = True
            if not deps_reduce:
                continue
            for input_name in input_vars:
                if input_name in self._var_to_use_op:
                    self._var_to_use_op[input_name].append(idx)
            for output_name in output_vars:
                if output_name not in self._var_to_use_op:
                    self._var_to_use_op[output_name] = []
                if output_name not in self._var_to_generate_op:
                    self._var_to_generate_op[output_name] = [idx]
                else:
                    self._var_to_generate_op[output_name].append(idx)
            if op.type == "conditional_block":
                # subblock
                assert (op.desc.has_attr("sub_block"))
                subblock_idx = op.desc.attr("sub_block").id
                subblock_deps = ProgramDeps(
                    self._block.program.block(subblock_idx),
                    op.desc.input_arg_names(), op.desc.output_arg_names())
                self._sub_block_deps[subblock_idx] = subblock_deps
                subblock_deps._father_block_deps = self

    def crop_input_var_from_op(self, op_idx, var_name):
        if var_name in self._var_to_use_op:
            # update var -> dep_var_op
            if self._var_to_use_op[var_name] != []:
                if op_idx not in self._var_to_use_op[var_name]:
                    raise ValueError(
                        "op_idx: {} is not in self._var_to_use_op[{}], "
                        "self._var_to_use_op[{}] is {}".format(
                            op_idx, var_name, var_name, self._var_to_use_op[
                                var_name]))
                self._var_to_use_op[var_name].remove(op_idx)
            # update _should_removed_var
            if var_name in self._start_vars:
                self._should_removed_var.discard(var_name)
            elif self._var_to_use_op[
                    var_name] == []:  # no more deps of this var
                self._should_removed_var.add(var_name)
            elif self._var_to_generate_op[var_name][-1] >= self._var_to_use_op[
                    var_name][-1]:
                # there are circle in the graph
                self._should_removed_var.add(var_name)
            else:  # input_name should not be deleted
                self._should_removed_var.discard(var_name)

    def crop_output_var_from_op(self, op_idx, var_name):
        if var_name in self._var_to_generate_op:
            assert (op_idx in self._var_to_generate_op[var_name])
            self._var_to_generate_op[var_name].remove(op_idx)
        if self._block.has_var(var_name):
            if var_name not in self._var_to_generate_op or self._var_to_generate_op[
                    var_name] == []:
                self._block._remove_var(var_name, sync=False)

    def remove_op(self, op_idx, reserved_vars=None):
        # update deps
        op = self._block.ops[op_idx]
        for input_name in op.desc.input_arg_names():
            if reserved_vars is not None and input_name in reserved_vars:
                continue
            self.crop_input_var_from_op(op_idx, input_name)
        for output_name in op.desc.output_arg_names():
            if reserved_vars is not None and output_name in reserved_vars:
                continue
            self.crop_output_var_from_op(op_idx, output_name)
        self._block._remove_op(op_idx, sync=False)

    def should_remove_op(self, op_idx):
        op = self._block.ops[op_idx]

        # NOTE: At present, it is found that the OP without output is
        # only send_v2 and partial_send op, which will be used in
        # all device
        if len(op.desc.output_arg_names()) == 0:
            return False

        for output_name in op.desc.output_arg_names():
            if output_name not in self._should_removed_var:
                return False
        return True
