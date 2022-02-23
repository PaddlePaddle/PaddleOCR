#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
# limitations under the License

import paddle
import paddle.fluid.core as core
from ..collective import _get_global_env
from ..collective import _new_ring_id
from ...fluid.framework import in_dygraph_mode
from ...fluid.layers.tensor import fill_constant

LOGICAL_PROCESS_TO_PHYSICAL_PROCESS_MAP = None
PROCESSOR_TO_PHYSICAL_PROCESS_MAP = None


def get_all_logical_process_set():
    from .interface import _g_process_mesh_map
    all_logical_process_set = set(_g_process_mesh_map[0].process_group)
    return all_logical_process_set


def get_logical_process_to_physical_process_map():
    global LOGICAL_PROCESS_TO_PHYSICAL_PROCESS_MAP
    return LOGICAL_PROCESS_TO_PHYSICAL_PROCESS_MAP


def set_logical_process_to_physical_process_map(mapping):
    global LOGICAL_PROCESS_TO_PHYSICAL_PROCESS_MAP
    LOGICAL_PROCESS_TO_PHYSICAL_PROCESS_MAP = mapping


def get_processor_to_physical_process_map():
    global PROCESSOR_TO_PHYSICAL_PROCESS_MAP
    return PROCESSOR_TO_PHYSICAL_PROCESS_MAP


def set_processor_to_physical_process_map(mapping):
    global PROCESSOR_TO_PHYSICAL_PROCESS_MAP
    PROCESSOR_TO_PHYSICAL_PROCESS_MAP = mapping


PROCESS_GROUP_MAP = {}


def get_all_process_groups():
    global PROCESS_GROUP_MAP
    return PROCESS_GROUP_MAP.values()


def new_process_group(ranks):
    global PROCESS_GROUP_MAP
    if not PROCESS_GROUP_MAP:
        genv = _get_global_env()
        PROCESS_GROUP_MAP["global_group"] = ProcessGroup(
            0, list(range(genv.world_size)))
    # A key constructed from ranks is used in the global process group map
    key = ''.join(map(str, sorted(ranks)))
    if key not in PROCESS_GROUP_MAP:
        num_groups = len(PROCESS_GROUP_MAP)
        # Note: our process group may interfere with the original implementation
        # so the created group id should start from the original _new_ring_id()
        group_id = _new_ring_id() + num_groups + 1
        pg = ProcessGroup(group_id, ranks)
        PROCESS_GROUP_MAP[key] = pg
        return pg
    else:
        pg = PROCESS_GROUP_MAP[key]
        return pg


# This implementation refers to lots of Paddle/python/paddle/distributed/collective.py,
# Fleet also has a collective helper which uses ops to initialize communication in 
# Paddle/python/paddle/distributed/fleet/meta_optimizers/common.py. We use the first one
# because it seems simple. This should be enhanced to manage the process membership and 
# the instantiation process in a more general way. In the future, the process group may 
# handle the communication implementation choice.
class ProcessGroup:
    def __init__(self, group_id, ranks):
        self._group_id = group_id
        self._ranks = sorted(ranks)
        self._nranks = len(self._ranks)
        self._is_instantiate = False

    @property
    def id(self):
        return self._group_id

    # @property
    # def key(self):
    #     return ''.join(map(str, sorted(self._ranks)))

    def local_rank(self, global_rank):
        if global_rank in self._ranks:
            return self._ranks.index(global_rank)
        else:
            assert False, \
                "Rank {} doesn't belong to this group".format(global_rank)

    def is_instantiate(self):
        return self._is_instantiate

    def instantiate(self):
        if self._is_instantiate:
            return
        ring_id = self.id
        genv = _get_global_env()
        global_rank = genv.rank

        if self._nranks >= 2:
            strategy = core.ParallelStrategy()
            strategy.nranks = self._nranks
            strategy.local_rank = self.local_rank(global_rank)
            strategy.trainer_endpoints = [
                genv.trainer_endpoints[i] for i in self._ranks
            ]
            strategy.current_endpoint = genv.current_endpoint
            strategy.nrings = 1

            if core.is_compiled_with_cuda():
                place = core.CUDAPlace(genv.device_id)
                core.NCCLParallelContext(strategy,
                                         place).init_with_ring_id(ring_id)
            else:
                assert False, ("No CUDA device found")

        # TODO(shenliang03): This is a temporary solution to solve the problem of 
        # hang caused by cross-creation of new_group
        tmp = paddle.to_tensor(
            [1], dtype="int32") if in_dygraph_mode() else fill_constant(
                [0], dtype="int32", value="1")
        paddle.distributed.all_reduce(tmp, use_calc_stream=True)
        paddle.distributed.wait(tmp)

        self._is_instantiate = True

    def __str__(self):
        string = "id: {}, nranks: {}, ranks: {}.".format(
            self.id, self._nranks, ", ".join(map(str, self._ranks)))
        return string
