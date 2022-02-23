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

import numpy as np
from collections import namedtuple

GroupInfo = namedtuple('GroupInfo', ['size', 'rank', 'world'])


class Topology:
    def __init__(self,
                 device_rank,
                 world_size,
                 dp_degree=None,
                 pp_degree=1,
                 sharding_degree=1,
                 mp_degree=1):
        arr = np.arange(0, dp_degree * pp_degree * sharding_degree *
                        mp_degree).reshape(
                            [dp_degree, pp_degree, sharding_degree, mp_degree])

        dp_rank, pp_rank, sharding_rank, mp_rank = np.where(arr == device_rank)
        dp_rank = dp_rank[0]
        pp_rank = pp_rank[0]
        sharding_rank = sharding_rank[0]
        mp_rank = mp_rank[0]

        self.world = GroupInfo(
            size=world_size, rank=device_rank,
            world=list(range(0, world_size)))

        mp_world = arr[dp_rank, pp_rank, sharding_rank, :]
        self.mp_info = GroupInfo(
            size=len(mp_world), rank=mp_rank, world=mp_world.tolist())

        sharding_world = arr[dp_rank, pp_rank, :, mp_rank]
        self.sharding_info = GroupInfo(
            size=len(sharding_world),
            rank=sharding_rank,
            world=sharding_world.tolist())

        pp_world = arr[dp_rank, :, sharding_rank, mp_rank]
        self.pp_info = GroupInfo(
            size=len(pp_world), rank=pp_rank, world=pp_world.tolist())

        dp_world = arr[:, pp_rank, sharding_rank, mp_rank]
        self.dp_info = GroupInfo(
            size=len(dp_world), rank=dp_rank, world=dp_world.tolist())

        self.is_last = self.pp_info.rank == self.pp_info.size - 1

        data_arr = np.arange(0, dp_degree * sharding_degree).reshape(
            [dp_degree, sharding_degree])
        data_arr = np.expand_dims(data_arr, axis=1).repeat(pp_degree, axis=1)
        data_arr = np.expand_dims(data_arr, axis=3).repeat(mp_degree, axis=3)

        self.data_info = GroupInfo(
            size=int(self.dp_info.size * self.sharding_info.size),
            rank=int(self.dp_info.rank * self.sharding_info.size +
                     self.sharding_info.rank),
            world=data_arr.reshape(-1).tolist())

        assert self.data_info.world[
            device_rank] == self.data_info.rank, "Data rank caculate error!"
        self.data_inner_times = self.world.size // self.data_info.size

    def __repr__(self):
        return f'dp_info:\n\t {self.dp_info}, \npp_info:\n\t {self.pp_info}, \nsharding_info:\n\t {self.sharding_info}, \nmp_info:\n\t {self.mp_info}\ndata_info:\n\t {self.data_info}'
