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

import os
import paddle
from paddle.distributed.fleet.launch_utils import get_cluster, logger

__all__ = []


def get_cloud_cluster(args_node_ips,
                      device_mode,
                      devices_per_proc,
                      args_port=6170):
    """
    args_node_ips:string, device_mode:DeviceMode(Int), device_per_proc:list, args_port: int
    """
    #you can automatically get ip info while using paddlecloud multi nodes mode.
    node_ips = os.getenv("PADDLE_TRAINERS")
    assert node_ips is not None, "PADDLE_TRAINERS should not be None"

    node_ip = os.getenv("POD_IP")
    assert node_ip is not None, "POD_IP should not be None"

    node_rank = os.getenv("PADDLE_TRAINER_ID")
    assert node_rank is not None, "PADDLE_TRAINER_ID should not be None"

    paddle_ports_num = int(os.getenv("TRAINER_PORTS_NUM"))
    assert paddle_ports_num is not None, "TRAINER_PORTS_NUM should not be None"

    node_ips = node_ips.split(",")
    num_nodes = len(node_ips)
    node_rank = int(node_rank)

    if args_node_ips != "127.0.0.1" and args_node_ips != ",".join(node_ips):
        logger.warning(
            "Please NOTE: When using paddlecloud, cluster_node_ips is \
automatically got from PADDLE_TRAINERS(multi nodes) or POD_IP(single node).\
Your input cluster_node_ips: {} doesn't equals to IPs: {} from \
paddlecloud environment.".format(args_node_ips, node_ips))

    # DISTRIBUTED_TRAINER_ENDPOINTS: new environment since paddlecloud 1.8.4
    # e.g: DISTRIBUTED_TRAINER_ENDPOINTS="ip1:port1,ip1:port2,ip1:port3,ip1:port4,ip2:port5,ip2:port6,ip2:port7,ip2:port8"
    trainer_endpoints = os.getenv("DISTRIBUTED_TRAINER_ENDPOINTS")
    if trainer_endpoints is None:
        started_port = args_port
        if num_nodes > 1:
            try:
                paddle_port = int(os.getenv("PADDLE_PORT", ""))

                if paddle_ports_num >= len(
                        devices_per_proc) and paddle_port != args_port:
                    logger.warning("Use Cloud specified port:{}.".format(
                        paddle_port))
                    started_port = paddle_port

            except Exception as e:
                print(e)
                pass

        if started_port is None:
            started_port = 6170
        ports = [
            x for x in range(started_port, started_port + len(devices_per_proc))
        ]
        trainer_endpoints = []
        for ip in node_ips:
            trainer_endpoints.append(["%s:%d" % (ip, port) for port in ports])
    else:
        trainer_endpoints_ori = trainer_endpoints.split(",")
        trainer_endpoints = []
        assert num_nodes * paddle_ports_num == len(trainer_endpoints_ori)
        for i in range(num_nodes):
            trainer_endpoints.append(trainer_endpoints_ori[
                i * paddle_ports_num:(i + 1) * paddle_ports_num])

    logger.debug("parsed from args: node_ips:{} \
        node_ip:{} node_rank:{} trainer_endpoints:{}"
                 .format(node_ips, node_ip, node_rank, trainer_endpoints))

    cluster, pod = get_cluster(node_ips, node_ip, trainer_endpoints,
                               device_mode, devices_per_proc)
    return cluster, cluster.pods[node_rank]


def use_paddlecloud():
    node_ips = os.getenv("PADDLE_TRAINERS")
    node_ip = os.getenv("POD_IP")
    node_rank = os.getenv("PADDLE_TRAINER_ID")
    paddle_ports_num = os.getenv("TRAINER_PORTS_NUM")
    if node_ips is None or node_ip is None or node_rank is None or paddle_ports_num is None:
        return False
    else:
        return True


def get_trainers_num():
    return int(os.getenv("PADDLE_TRAINERS_NUM", "1"))
