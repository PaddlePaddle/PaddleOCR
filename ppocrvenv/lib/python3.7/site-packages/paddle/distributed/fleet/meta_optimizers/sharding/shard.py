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

from paddle.distributed.fleet.meta_optimizers.common import is_optimizer_op
from paddle.distributed.fleet.meta_optimizers.sharding.utils import *
from paddle.distributed.fleet.meta_optimizers.sharding.fp16_helper import FP16Utils

__all__ = []


class Shard(object):
    def __init__(self, ):
        self.global_params = set([])
        self.worker_idx = -1
        self.worker_num = -1
        self.global_param2device = dict()
        self.device2global_params = dict()

    def setup(self, params_grads, worker_idx, worker_num):
        # param names of all devices
        self.global_params = set([x[0].name for x in params_grads])
        # _param(str) -> device_id(int) 
        self.worker_idx = worker_idx
        self.worker_num = worker_num
        # global_param2device contains fp32 params and fp16 params
        # device2global_params only contains fp32 params
        self.global_param2device, self.device2global_params \
            = self._split_params(params_grads, worker_idx, worker_num)

    def has_param(self, var_name):
        return var_name in self.global_param2device and \
            self._var_device_id(var_name) == self.worker_idx

    def has_opt_var(self, var_name):
        return self._var_device_id(var_name) == self.worker_idx

    def has_var(self, var_name):
        return self._var_device_id(var_name) == -1 or \
            self._var_device_id(var_name) == self.worker_idx

    def _split_params(self, params_grads, worker_idx, worker_num):
        param2device = {}
        total_param_mem = 0.0
        param2mem = []
        for param in [x[0] for x in params_grads]:
            mem = get_var_size(param)
            total_param_mem += mem
            param2mem.append((param.name, mem))
        device2params = {x: [] for x in range(worker_num)}
        device_idx = 0
        mem_accu = 0.0
        for param_name, mem in param2mem:
            if mem_accu > total_param_mem * 1.0 * (device_idx + 1) / worker_num:
                device_idx += 1
            device2params[device_idx].append(param_name)
            param2device[param_name] = device_idx
            mem_accu += mem
        return param2device, device2params

    def _var_device_id(self, var_name):
        if var_name in self.global_param2device:
            return self.global_param2device[var_name]
        for suffix in [
                "_moment1_0", "_moment2_0", "_beta1_pow_acc_0",
                "_beta2_pow_acc_0", "_velocity_0"
        ]:
            base_name = re.sub(suffix, '', var_name)
            if base_name in self.global_param2device:
                return self.global_param2device[base_name]
        return -1

    def find_broadcast_params(self, block):
        broadcast_vars = set([])
        fp16_params = set([])
        fp16_to_fp32 = {}

        param_usage = {x: 0 for x in self.global_params}
        for op in block.ops:
            if is_optimizer_op(op):
                continue
            for input_name in op.desc.input_arg_names():
                if input_name in self.global_params:
                    param_usage[input_name] += 1

        for op in block.ops:
            if not FP16Utils.is_fp16_cast_op(block, op, self.global_params):
                continue
            input_name = op.input_arg_names[0]
            output_name = op.output_arg_names[0]
            broadcast_vars.add(output_name)
            fp16_params.add(output_name)
            fp16_to_fp32[output_name] = input_name
            param_usage[input_name] -= 1
            self.global_param2device[output_name] = self.global_param2device[
                input_name]

        for param, usage in param_usage.items():
            if usage > 0:
                broadcast_vars.add(param)
        return broadcast_vars

    def device(self, var_name):
        return self._var_device_id(var_name)

    def is_param(self, var_name):
        return var_name in self.global_params

    def is_opti_var(self, var_name):
        if var_name in self.global_params:
            return True
        for suffix in [
                "_moment1_0", "_moment2_0", "_beta1_pow_acc_0",
                "_beta2_pow_acc_0", "_velocity_0"
        ]:
            base_name = re.sub(suffix, '', var_name)
            if base_name in self.global_params:
                return True
        return False

    def filter_grads(self, grads):
        grads_in_shard = []
        for grad in grads:
            param = grad.split("@")[0]
            if self.has_param(param):
                grads_in_shard.append(grad)
        return grads_in_shard


class ProgramSegment(object):
    def __init__(self, block):
        self._block = block
        self._allreduce_vars = []
        # sub program start idx
        self._start_idx = -1
        # sub program end idx
        self._end_idx = -1
        # param name to broadcast name
        self._param2broadcast = {}
        self._broadcast_vars = []
        # cast op pairs, fp16 name (str) -> fp32 name (str)
        self._cast_ops = {}
        # fill constant vars
        self._fill_constant_vars = []
        # parameter mems
        self._param_mem = 0.0
