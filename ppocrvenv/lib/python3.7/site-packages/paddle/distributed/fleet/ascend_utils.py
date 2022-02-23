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

import os
import json
import paddle
from paddle.distributed.fleet.launch_utils import get_cluster, logger, get_host_name_ip, DeviceMode

__all__ = []


def _get_ascend_rankfile(rank_table_file_path):
    """
    Args:
    rank_table_file_path: ascend npu rank file json
    {
        "status": "completed",
        "version": "1.0",
        "server_count": "2",
        "server_list": [
            {
                "server_id": "192.168.24.217",
                "device": [
                    {
                        "device_id": "0",
                        "device_ip": "192.1.184.23",
                        "rank_id": "0"
                    },
                    {
                        "device_id": "1",
                        "device_ip": "192.2.21.93",
                        "rank_id": "1"
                    }
                ]
            },
            {
                "server_id": "192.168.26.177",
                "device": [
                    {
                        "device_id": "0",
                        "device_ip": "192.1.94.132",
                        "rank_id": "2"
                    },
                    {
                        "device_id": "1",
                        "device_ip": "192.2.94.30",
                        "rank_id": "3"
                    }
                ]
            }
        ]
    }

    Returns:
        node_ips: node ip list
        device_count: number of npu per machine
    """
    json_data = None
    with open(rank_table_file_path) as json_file:
        json_data = json.load(json_file)

    node_ips = []
    device_count = 0
    server_list = json_data['server_list']
    for server in server_list:
        device_list = server['device']
        device_count = len(device_list)
        if os.getenv("FLAGS_MODELARTS", None):
            nodes = os.getenv("DLS_TASK_NUMBER", None)
            assert nodes is not None, "DLS_TASK_NUMBER didn't set!"
            for node in range(int(nodes)):
                node_ip = os.getenv("VC_CUSTOM{}_HOSTS".format(node), None)
                assert node_ip is not None, "VC_CUSTOM{}_HOSTS didn't set!".format(
                    node)
                node_ips.append(node_ip)
            return node_ips, device_count
        node_ips.append(server['server_id'])
    return node_ips, device_count


def get_cloud_cluster(rank_table_file=None,
                      device_mode=DeviceMode.ASCEND_NPU,
                      start_port=6070):
    """
    Args:
    rank_table_file: string, ascend npu rank file path
    device_mode: DeviceMode(Int)
    start_port: the start port of current runtime env
    """
    if rank_table_file:
        # multi trainers
        node_ips, device_count = _get_ascend_rankfile(rank_table_file)
        if len(node_ips) == 1:
            node_ip = node_ips[0]
        else:
            node_index = os.environ.get("PADDLE_TRAINER_ID")
            node_ip = None
            if node_index:
                node_ip = node_ips[int(node_index)]
            else:
                _, node_ip = get_host_name_ip()

        assert node_ip in node_ips, "Can't find your local ip {%s} in node_ips: {%s}" \
            % (node_ip, node_ips)
    else:
        # single trainer (single ascend card)
        node_ips = ["127.0.0.1"]
        node_ip = node_ips[0]
        device_count = 1

    devices_per_proc = [str(x) for x in range(device_count)]
    free_ports = [
        x for x in range(start_port, start_port + len(devices_per_proc))
    ]

    trainer_endpoints = []
    for ip in node_ips:
        trainer_endpoints.append(["%s:%d" % (ip, port) for port in free_ports])

    return get_cluster(node_ips, node_ip, trainer_endpoints, device_mode,
                       devices_per_proc)
