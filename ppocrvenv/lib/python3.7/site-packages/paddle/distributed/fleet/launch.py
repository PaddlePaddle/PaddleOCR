# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
r"""
fleetrun is a module that spawns multiple distributed
process on each training node for gpu training and cpu training.
Usage:
    In both of single node training or multiple node training, this module
launch a process on each of the given gpu card or cpu machine.
    GPU training:
    1. for single node training with all visible gpu cards:
       fleetrun your_training_py (arg1 arg2 and all others)
    2. for single node training with [0,4) cards
       fleetrun --gpus="0,1,2,3" your_training_py (arg1 arg2 and all others)
    3. for multiple node training such as two node:192.168.0.16, 192.168.0.17
        on 192.168.0.16:
            fleetrun --ips="192.168.0.16,192.168.0.17" \
                your_training_py (arg1 arg2 and all others)
        on 192.168.0.17:
            fleetrun --ips="192.168.0.16,192.168.0.17" \
                your_training_py (arg1 arg2 and all others)
    CPU training:
    1. for single node training with multi servers and workers:
        fleetrun --server_num=2 --worker_num=2 your_training_py (arg1 arg2 and all others)
    2. for multiple node training such as two node:192.168.0.16, 192.168.0.17 \
        with 2 servers and 4 workers.
        on 192.168.0.16:
            fleetrun --servers="192.168.0.16:6170,192.168.0.17:6170" \
                --workers="192.168.0.16,192.168.0.17,192.168.0.16,192.168.0.17" \
                your_training_py (arg1 arg2 and all others)
        on 192.168.0.17:
            fleetrun --servers="192.168.0.16:6170,192.168.0.17:6171" \
                --workers="192.168.0.16,192.168.0.17,192.168.0.16,192.168.0.17" \
                your_training_py (arg1 arg2 and all others)
    3. use gloo backend for multiple node training such as two node:192.168.0.16, 192.168.0.17 \
        with 2 servers and 4 workers. (workers should set port)
        on 192.168.0.16:
            fleetrun --servers="192.168.0.16:6170,192.168.0.17:6170" \
                --workers="192.168.0.16:6171,192.168.0.17:6171,192.168.0.16:6172,192.168.0.17:6172" \
                your_training_py (arg1 arg2 and all others)
        on 192.168.0.17:
            fleetrun --servers="192.168.0.16:6170,192.168.0.17:6170" \
                --workers="192.168.0.16:6171,192.168.0.17:6171,192.168.0.16:6172,192.168.0.17:6172" \
                your_training_py (arg1 arg2 and all others)
"""

from __future__ import print_function

import shutil
import sys
import tempfile
from sys import version
import subprocess
import os
import time
import six
import copy
from argparse import ArgumentParser, REMAINDER
import paddle
import paddle.fluid as fluid
from paddle.distributed.fleet import launch_utils

# TODO(danleifeng): Don't import * from a module
from paddle.distributed.fleet.launch_utils import *
from paddle.distributed.fleet import cloud_utils
from paddle.distributed.fleet import ascend_utils

from paddle.distributed.fleet.elastic import enable_elastic, launch_elastic

__all__ = []


def _print_arguments(args):
    print("-----------  Configuration Arguments -----------")
    for arg, value in sorted(six.iteritems(vars(args))):
        print("%s: %s" % (arg, value))
    print("------------------------------------------------")


def _parse_args():
    """
    Helper function parsing the command line options
    @retval ArgumentParser
    """
    parser = ArgumentParser(
        description='''start paddle training using multi-process mode.
see: http://www.paddlepaddle.org/documentation/docs/zh/1.6/user_guides/howto/training/cluster_howto.html#permalink-8--nccl2-
''')
    base_group = parser.add_argument_group("Base Parameters")

    base_group.add_argument(
        "--log_dir",
        type=str,
        default="log",
        help="The path for each process's log. Default --log_dir=log/")
    base_group.add_argument(
        "--backend",
        type=str,
        default="auto",
        help="Specifize the backend, can be gloo|nccl|bkcl|auto. Default value is auto which perfers nccl or bkcl."
    )
    base_group.add_argument(
        "--nproc_per_node",
        type=int,
        default=None,
        help="The number of processes to launch on a node."
        "In gpu training, it should be less or equal to the gpus number of you system(or you set by --gpus). And so each process can"
        " bound to one or average number of gpus.")

    base_group.add_argument(
        "--run_mode",
        type=str,
        default=None,
        help="run mode of job, can be:collective/ps/ps-heter")

    if fluid.core.is_compiled_with_cuda():
        base_group.add_argument(
            "--gpus",
            type=str,
            default=None,
            help="It's for gpu training."
            "For example:"
            "--gpus=\"0,1,2,3\" will launch four training processes each bound to one gpu."
        )
        base_group.add_argument("--selected_gpus", dest="gpus")

    if fluid.core.is_compiled_with_xpu():
        base_group.add_argument(
            "--xpus",
            type=str,
            default=None,
            help="It's for xpu training. For example: "
            "--xpus=\"0,1,2,3\" will launch four training processes each bound to one xpu."
        )
        base_group.add_argument("--selected_xpus", dest="xpus")

    base_group.add_argument(
        "training_script",
        type=str,
        help="The full path to the single GPU training "
        "program/script to be launched in parallel, "
        "followed by all the arguments for the "
        "training script")

    base_group.add_argument('training_script_args', nargs=REMAINDER)

    # Optional arguments for the launch helper
    # for collective
    collective_group = parser.add_argument_group("Collective Parameters")
    collective_group.add_argument(
        "--ips",
        type=str,
        default="127.0.0.1",
        help="Paddle cluster nodes ips, such as 192.168.0.16,192.168.0.17..")

    ps_group = parser.add_argument_group("Parameter-Server Parameters")
    # for parameter server
    ps_group.add_argument(
        "--servers", type=str, default="", help="User defined servers ip:port")
    ps_group.add_argument(
        "--workers", type=str, default="", help="User defined workers ip:port")
    ps_group.add_argument(
        "--heter_workers",
        type=str,
        default="",
        help="User defined heter workers in each stage ip1:port1;ip2:port2")
    ps_group.add_argument(
        "--heter_devices",
        type=str,
        default="",
        help="User defined heter devices in each stage cpu;gpu;cpu")

    ps_group.add_argument("--worker_num", type=int, help="number of workers")
    ps_group.add_argument("--server_num", type=int, help="number of servers")
    ps_group.add_argument(
        "--heter_worker_num",
        type=str,
        help="number of heter_workers in each stage 1;2;3")
    ps_group.add_argument("--http_port", type=int, help="Gloo http Port")

    # parameter elastic mode
    elastic_group = parser.add_argument_group("Elastic Parameters")
    elastic_group.add_argument(
        "--elastic_server", type=str, help="etcd server host:port")
    elastic_group.add_argument("--job_id", type=str, help="job unique id")
    elastic_group.add_argument("--np", type=int, help="job pod/node number")
    elastic_group.add_argument("--scale", type=int, default=0, help="scale np")
    elastic_group.add_argument(
        "--host", type=str, help="bind host, default to POD_IP env")
    elastic_group.add_argument(
        "--force", type=bool, default=False, help="update np force")

    return parser.parse_args()


def get_cluster_from_args(args, device_mode, devices_per_proc):
    node_ips = [x.strip() for x in args.ips.split(',')]
    if len(node_ips) == 1:
        node_ip = node_ips[0]
    else:
        if args.host:
            node_ip = args.host
        else:
            _, node_ip = get_host_name_ip()

    assert node_ip in node_ips, "Can't find your local ip {%s} in node_ips: {%s}" \
        % (node_ip, node_ips)
    node_rank = node_ips.index(node_ip)

    logger.debug("parsed from args: node_ips:{} node_ip:{} node_rank:{}".format(
        node_ips, node_ip, node_rank))

    free_ports = None
    if not cloud_utils.use_paddlecloud() and len(
            node_ips) <= 1 and os.environ.get('FLAGS_START_PORT') is None:
        free_ports = find_free_ports(len(devices_per_proc))
        if free_ports is not None:
            free_ports = list(free_ports)
    else:
        start_port = 6070
        if os.environ.get('FLAGS_START_PORT') is not None:
            start_port = int(os.environ.get('FLAGS_START_PORT'))

        free_ports = [
            x for x in range(start_port, start_port + len(devices_per_proc))
        ]

    trainer_endpoints = []
    for ip in node_ips:
        trainer_endpoints.append(["%s:%d" % (ip, port) for port in free_ports])
    return get_cluster(node_ips, node_ip, trainer_endpoints, device_mode,
                       devices_per_proc)


def cpuonly_check(args):
    if args.ips and len(args.ips.split(',')) > 1:
        raise RuntimeError(
            "CPUONLY launch only support single trainer, that is len(ips)=1, but got %s."
            % args.ips)
    if args.run_mode:
        assert args.run_mode == 'cpuonly', "CPUONLY launch only support run mode is CPUONLY"
    if args.servers:
        raise RuntimeError("CPUONLY launch can't have --servers as arguments.")
    return True


def launch_collective(args):
    # parse arguments, used for cloud-single-machine and local
    if args.backend == 'gloo': cpuonly_check(args)
    (device_mode, devices_per_proc) = launch_utils.get_device_proc_info(args)
    trainers_num = cloud_utils.get_trainers_num()
    logger.debug("parsed from args trainerss_num:{} mode:{} devices:{}".format(
        trainers_num, device_mode, devices_per_proc))

    cluster = None
    pod = None

    start_port = 6170
    if os.environ.get('FLAGS_START_PORT') is not None:
        start_port = os.environ.get('FLAGS_START_PORT')
    if cloud_utils.use_paddlecloud() and trainers_num != 1:
        cluster, pod = cloud_utils.get_cloud_cluster(
            args.ips, device_mode, devices_per_proc, start_port)
        logger.debug("get cluster from cloud:{}".format(cluster))
    elif device_mode == DeviceMode.ASCEND_NPU:
        # for ascend
        cluster, pod = ascend_utils.get_cloud_cluster(
            rank_table_file=os.getenv("RANK_TABLE_FILE", None),
            device_mode=device_mode,
            start_port=start_port)
    else:
        # trainers_num = 1 or not use paddlecloud ips="a,b"
        cluster, pod = get_cluster_from_args(args, device_mode,
                                             devices_per_proc)
        logger.debug("get cluster from args:{}".format(cluster))

    global_envs = copy.copy(os.environ.copy())
    gloo_rendezvous_dir = tempfile.mkdtemp()
    # add gloo env
    global_envs["PADDLE_WITH_GLOO"] = str(os.getenv("PADDLE_WITH_GLOO", "0"))
    global_envs["PADDLE_GLOO_RENDEZVOUS"] = "3"
    global_envs["PADDLE_GLOO_FS_PATH"] = gloo_rendezvous_dir
    global_envs["PADDLE_DISTRI_BACKEND"] = args.backend

    procs = start_local_trainers(
        cluster,
        pod,
        training_script=args.training_script,
        training_script_args=args.training_script_args,
        log_dir=args.log_dir,
        envs=global_envs)

    for idx, proc in enumerate(procs):
        print("launch proc_id:{} idx:{}".format(proc.proc.pid, idx))

    while True:
        try:
            alive = watch_local_trainers(procs, cluster.trainers_nranks())

            if not alive:
                logger.info("Local processes completed.")
                logger.debug("POD info:{}".format(pod))
                break

            time.sleep(3)

        except:
            logger.warning("Terminating... exit")
            terminate_local_procs(procs)
            exit(1)

    if os.path.exists(gloo_rendezvous_dir):
        shutil.rmtree(gloo_rendezvous_dir)


def launch_ps(args, distribute_mode):
    cloud_flag = cloud_utils.use_paddlecloud()

    # for ps-cpu on paddlecloud
    if cloud_flag and distribute_mode == DistributeMode.PS:
        direct_start(args)
        return
    #elif cloud_flag and distribute_mode == DistributeMode.PS_HETER:
    #    cloud_ps_heter_env_set(args)
    #    args.workers = os.getenv("PADDLE_TRAINER_ENDPOINTS")
    #    args.servers = os.getenv("PADDLE_PSERVERS_IP_PORT_LIST")
    #    args.heter_workers = os.getenv("PADDLE_HETER_TRAINER_IP_PORT_LIST")

    ps_launcher = ParameterServerLauncher(args, distribute_mode)
    ps_launcher.start_ps()
    return


def infer_backend(args):
    if args.backend != "auto": return
    if fluid.core.is_compiled_with_cuda():
        args.backend = 'nccl'
    elif fluid.core.is_compiled_with_npu():
        args.backend = 'unknown'
    elif fluid.core.is_compiled_with_xpu():
        args.backend = 'bkcl'
    else:
        args.backend = 'gloo'


def which_distributed_mode(args):
    infer_backend(args)  # modify the args.backend
    if args.run_mode is not None:
        assert args.run_mode in ["collective", "ps", "ps-heter"]

    if args.run_mode == "collective":
        return DistributeMode.COLLECTIVE
    elif args.run_mode == "ps":
        return DistributeMode.PS
    elif args.run_mode == "ps-heter":
        return DistributeMode.PS_HETER

    ps_args = [
        '--worker_num', '--server_num', '--heter_worker_num', '--servers',
        '--workers', '--heter_workers', '--heter_devices', '--http_port'
    ]
    collective_args = ['--ips']

    ps_heter_args = ["--heter_worker_num", "--heter_workers", "--heter_devices"]

    has_ps_args = [
        ps_arg for ps_arg in ps_args if ps_arg in " ".join(sys.argv[1:-1])
    ]
    has_collective_args = [
        co_arg for co_arg in collective_args
        if co_arg in " ".join(sys.argv[1:-1])
    ]

    if len(has_ps_args) > 1 and len(has_collective_args) > 1:
        raise ValueError(
            "Only one mode(Collective or Parameter-Server) can be selected at the same time, but more than one configuration was received."
        )

    if fluid.core.is_compiled_with_cuda():
        accelerators = fluid.core.get_cuda_device_count()
    elif fluid.core.is_compiled_with_npu():
        accelerators = fluid.core.get_npu_device_count()
    elif fluid.core.is_compiled_with_xpu():
        accelerators = fluid.core.get_xpu_device_count()
    else:
        accelerators = 0

    if len(has_ps_args) > 0:
        logger.info(
            "Run parameter-sever mode. pserver arguments:{}, accelerators count:{}".
            format(has_ps_args, accelerators))
        has_ps_heter_args = list(set(has_ps_args) & set(ps_heter_args))
        if len(has_ps_heter_args) > 0:
            return DistributeMode.PS_HETER
        else:
            return DistributeMode.PS
    elif len(has_collective_args) > 0:
        logger.info("Run collective mode. gpu arguments:{}, cuda count:{}".
                    format(has_collective_args, accelerators))
        return DistributeMode.COLLECTIVE
    else:
        if not fluid.core.is_compiled_with_cuda(
        ) and not fluid.core.is_compiled_with_xpu():
            if args.servers:
                logger.warning(
                    "Not found distinct arguments and not compiled with cuda or xpu. \
But found args.servers not empty, default use ps mode")
                return DistributeMode.PS
            else:
                return DistributeMode.COLLECTIVE
        else:
            logger.warning(
                "Not found distinct arguments and compiled with cuda or xpu. Default use collective mode"
            )
            return DistributeMode.COLLECTIVE


def launch():
    """
    Paddle distribution training entry ``python -m paddle.distributed.launch``.
    
    Usage:
        .. code-block:: bash
            :name: code-block-bash1

            python -m paddle.distributed.launch [-h] [--log_dir LOG_DIR] [--nproc_per_node NPROC_PER_NODE] [--run_mode RUN_MODE] [--gpus GPUS]
                             [--selected_gpus GPUS] [--ips IPS] [--servers SERVERS] [--workers WORKERS] [--heter_workers HETER_WORKERS]
                             [--worker_num WORKER_NUM] [--server_num SERVER_NUM] [--heter_worker_num HETER_WORKER_NUM]
                             [--http_port HTTP_PORT] [--elastic_server ELASTIC_SERVER] [--job_id JOB_ID] [--np NP] [--scale SCALE]
                             [--host HOST] [--force FORCE]
                             training_script ...    


    Base Parameters:
        - ``--log_dir``: The path for each process's log. e.g., ``--log_dir=output_dir``. Default ``--log_dir=log``.

        - ``--nproc_per_node``: The number of processes to launch on a node. In gpu training, it should be less or equal to the gpus number of you system(or you set by --gpus).  e.g., ``--nproc_per_node=8``

        - ``--run_mode``: run mode of job, can be:collective/ps/ps-heter. e.g., ``--run_mode=ps``. Default ``--run_mode=collective``.

        - ``--gpus``: It's for gpu training. e.g., ``--gpus=0,1,2,3`` will launch four training processes each bound to one gpu.

        - ``--selected_gpus``: gpus aliases, recommend to use ``--gpus``.
        
        - ``--xpus``: It's for xpu training if xpu is available. e.g., ``--xpus=0,1,2,3``.
        
        - ``--selected_xpus``: xpus aliases, recommend to use ``--xpus``.

        - ``training_script``: The full path to the single GPU training program/script to be launched in parallel, followed by all the arguments for the training script. e.g., ``traing.py``

        - ``training_script_args``: The args of training_script. e.g., ``--lr=0.1``

    Collective Parameters:
        - ``--ips``: Paddle cluster nodes ips, e.g., ``--ips=192.168.0.16,192.168.0.17``. Default ``--ips=127.0.0.1``.

    Parameter-Server Parameters:
        - ``--servers``: User defined servers ip:port, e.g., ``--servers="192.168.0.16:6170,192.168.0.17:6170"``

        - ``--workers``: User defined workers ip:port, e.g., ``--workers="192.168.0.16:6171,192.168.0.16:6172,192.168.0.17:6171,192.168.0.17:6172"``

        - ``--heter_workers``: User defined heter workers ip1:port1;ip2:port2, e.g., ``--heter_workers="192.168.0.16:6172;192.168.0.17:6172"``

        - ``--worker_num``: Number of workers (It recommend to set when in the emulated distributed environment using single node)

        - ``--server_num``: Number of servers (It recommend to set when in the emulated distributed environment using single node)

        - ``--heter_worker_num``: Number of heter_workers in each stage (It recommend to set when in the emulated distributed environment using single node)
        
        - ``--heter_devices``: Type of heter_device in each stage

        - ``--http_port``: Gloo http Port

    Elastic Parameters:
        - ``--elastic_server``: etcd server host:port, e.g., ``--elastic_server=127.0.0.1:2379``

        - ``--job_id``: job unique id, e.g., ``--job_id=job1``

        - ``--np``: job pod/node number, e.g., ``--np=2``

        - ``--host``: bind host, default to POD_IP env.


    Returns:
        ``None``

    Examples 1 (collective, single node):
        .. code-block:: bash
            :name: code-block-example-bash1
            
            # For training on single node using 4 gpus.

            python -m paddle.distributed.launch --gpus=0,1,2,3 train.py --lr=0.01
        
    Examples 2 (collective, multi node):
        .. code-block:: bash
            :name: code-block-example-bash2

            # The parameters of --gpus and --ips must be consistent in each node.

            # For training on multiple nodes, e.g., 192.168.0.16, 192.168.0.17 

            # On 192.168.0.16:

            python -m paddle.distributed.launch --gpus=0,1,2,3 --ips=192.168.0.16,192.168.0.17 train.py --lr=0.01

            # On 192.168.0.17:
            python -m paddle.distributed.launch --gpus=0,1,2,3 --ips=192.168.0.16,192.168.0.17 train.py --lr=0.01
        
    Examples 3 (ps, cpu, single node):
        .. code-block:: bash
            :name: code-block-example-bash3

            # To simulate distributed environment using single node, e.g., 2 servers and 4 workers.
            
            python -m paddle.distributed.launch --server_num=2 --worker_num=4 train.py --lr=0.01
        
    Examples 4 (ps, cpu, multi node):
        .. code-block:: bash
            :name: code-block-example-bash4

            # For training on multiple nodes, e.g., 192.168.0.16, 192.168.0.17 where each node with 1 server and 2 workers.

            # On 192.168.0.16:

            python -m paddle.distributed.launch --servers="192.168.0.16:6170,192.168.0.17:6170" --workers="192.168.0.16:6171,192.168.0.16:6172,192.168.0.17:6171,192.168.0.17:6172" train.py --lr=0.01

            # On 192.168.0.17:

            python -m paddle.distributed.launch --servers="192.168.0.16:6170,192.168.0.17:6170" --workers="192.168.0.16:6171,192.168.0.16:6172,192.168.0.17:6171,192.168.0.17:6172" train.py --lr=0.01

    Examples 5 (ps, gpu, single node):
        .. code-block:: bash
            :name: code-block-example-bash5

           # To simulate distributed environment using single node, e.g., 2 servers and 4 workers, each worker use single gpu.
            
            export CUDA_VISIBLE_DEVICES=0,1,2,3
            python -m paddle.distributed.launch --server_num=2 --worker_num=4 train.py --lr=0.01
            
    Examples 6 (ps, gpu, multi node):
        .. code-block:: bash
            :name: code-block-example-bash6

            # For training on multiple nodes, e.g., 192.168.0.16, 192.168.0.17 where each node with 1 server and 2 workers.

            # On 192.168.0.16:

            export CUDA_VISIBLE_DEVICES=0,1
            python -m paddle.distributed.launch --servers="192.168.0.16:6170,192.168.0.17:6170" --workers="192.168.0.16:6171,192.168.0.16:6172,192.168.0.17:6171,192.168.0.17:6172" train.py --lr=0.01

            # On 192.168.0.17:

            export CUDA_VISIBLE_DEVICES=0,1
            python -m paddle.distributed.launch --servers="192.168.0.16:6170,192.168.0.17:6170" --workers="192.168.0.16:6171,192.168.0.16:6172,192.168.0.17:6171,192.168.0.17:6172" train.py --lr=0.01

    Examples 7 (ps-heter, cpu + gpu, single node):
        .. code-block:: bash
            :name: code-block-example-bash7

            # To simulate distributed environment using single node, e.g., 2 servers and 4 workers, two workers use gpu, two workers use cpu.
            
            export CUDA_VISIBLE_DEVICES=0,1
            python -m paddle.distributed.launch --server_num=2 --worker_num=2 --heter_worker_num=2 train.py --lr=0.01
            
    Examples 8 (ps-heter, cpu + gpu, multi node):
        .. code-block:: bash
            :name: code-block-example-bash8

            # For training on multiple nodes, e.g., 192.168.0.16, 192.168.0.17 where each node with 1 server, 1 gpu worker, 1 cpu worker.

            # On 192.168.0.16:

            export CUDA_VISIBLE_DEVICES=0
            python -m paddle.distributed.launch --servers="192.168.0.16:6170,192.168.0.17:6170" --workers="192.168.0.16:6171,192.168.0.17:6171" --heter_workers="192.168.0.16:6172,192.168.0.17:6172" train.py --lr=0.01

            # On 192.168.0.17:

            export CUDA_VISIBLE_DEVICES=0
            python -m paddle.distributed.launch --servers="192.168.0.16:6170,192.168.0.17:6170" --workers="192.168.0.16:6171,192.168.0.17:6171" --heter_workers="192.168.0.16:6172,192.168.0.17:6172" train.py --lr=0.01

    Examples 9 (elastic):
        .. code-block:: bash
            :name: code-block-example-bash9

            python -m paddle.distributed.launch --elastic_server=127.0.0.1:2379 --np=2 --job_id=job1  --gpus=0,1,2,3 train.py
        
    """

    args = _parse_args()
    logger = get_logger()
    _print_arguments(args)

    if args.backend == 'auto':
        distribute_mode = which_distributed_mode(
            args)  # which_distributed_mode must modify args.backend
    else:
        assert args.run_mode == 'collective' or args.run_mode == None, "When backend is not 'auto', run mode must be collective"
        check_backend(args.backend)
        distribute_mode = DistributeMode.COLLECTIVE

    assert args.backend in ['gloo', 'nccl', 'bkcl', 'unknown']

    if args.backend == 'gloo':
        logger.warning("launch start with CPUONLY mode")

    block_windows_and_macos(
        args.backend)  # raise error when using gloo on windows or macos

    if enable_elastic(args, distribute_mode):
        launch_elastic(args, distribute_mode)
        return

    if distribute_mode == DistributeMode.COLLECTIVE:
        launch_collective(args)
    else:
        launch_ps(args, distribute_mode)


if __name__ == "__main__":
    launch()
