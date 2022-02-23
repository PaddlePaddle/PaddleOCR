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

# TODO: define distributed api under this directory,
from .base.role_maker import Role  # noqa: F401
from .base.role_maker import UserDefinedRoleMaker  # noqa: F401
from .base.role_maker import PaddleCloudRoleMaker  # noqa: F401
from .base.distributed_strategy import DistributedStrategy  # noqa: F401
from .base.fleet_base import Fleet  # noqa: F401
from .base.util_factory import UtilBase  # noqa: F401
from .dataset import DatasetBase  # noqa: F401
from .dataset import InMemoryDataset  # noqa: F401
from .dataset import QueueDataset  # noqa: F401
from .dataset import FileInstantDataset  # noqa: F401
from .dataset import BoxPSDataset  # noqa: F401
from .data_generator.data_generator import MultiSlotDataGenerator  # noqa: F401
from .data_generator.data_generator import MultiSlotStringDataGenerator  # noqa: F401
from . import metrics  # noqa: F401
from .base.topology import CommunicateTopology
from .base.topology import HybridCommunicateGroup  # noqa: F401

__all__ = [ #noqa
      "CommunicateTopology",
      "UtilBase",
      "HybridCommunicateGroup",
      "MultiSlotStringDataGenerator",
      "UserDefinedRoleMaker",
      "DistributedStrategy",
      "Role",
      "MultiSlotDataGenerator",
      "PaddleCloudRoleMaker",
      "Fleet"
]

fleet = Fleet()
_final_strategy = fleet._final_strategy
_get_applied_meta_list = fleet._get_applied_meta_list
_get_applied_graph_list = fleet._get_applied_graph_list
init = fleet.init
is_first_worker = fleet.is_first_worker
worker_index = fleet.worker_index
worker_num = fleet.worker_num
node_num = fleet.node_num
rank = fleet.worker_index
nranks = fleet.worker_num
world_size = fleet.worker_num
# device id in current trainer
local_device_ids = fleet.local_device_ids
# device ids in world
world_device_ids = fleet.world_device_ids
# rank in node
local_rank = fleet.local_rank
rank_in_node = local_rank
is_worker = fleet.is_worker
worker_endpoints = fleet.worker_endpoints
server_num = fleet.server_num
server_index = fleet.server_index
server_endpoints = fleet.server_endpoints
is_server = fleet.is_server
util = UtilBase()
barrier_worker = fleet.barrier_worker
init_worker = fleet.init_worker
init_server = fleet.init_server
run_server = fleet.run_server
stop_worker = fleet.stop_worker
distributed_optimizer = fleet.distributed_optimizer
save_inference_model = fleet.save_inference_model
save_persistables = fleet.save_persistables
load_model = fleet.load_model
minimize = fleet.minimize
distributed_model = fleet.distributed_model
step = fleet.step
clear_grad = fleet.clear_grad
set_lr = fleet.set_lr
get_lr = fleet.get_lr
state_dict = fleet.state_dict
set_state_dict = fleet.set_state_dict
shrink = fleet.shrink
get_hybrid_communicate_group = fleet.get_hybrid_communicate_group
distributed_scaler = fleet.distributed_scaler
