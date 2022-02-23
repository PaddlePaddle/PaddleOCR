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

import functools
import logging
import socket
import time
import os
import signal
import copy
import sys
import six
import subprocess
from contextlib import closing
import socket
from paddle.fluid import core
from paddle.distributed.fleet.launch_utils import get_backend_by_compile_flag
from distutils.util import strtobool

from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.framework import in_dygraph_mode
from paddle.fluid.data_feeder import check_variable_and_dtype


__all__ = [     #noqa
           'get_host_name_ip',
           'Trainer',
           'get_cluster',
           'start_local_trainers',
           'watch_local_trainers',
           'find_free_ports',
           'JobServer',
           'Cluster',
           'Pod',
           'Hdfs',
           'add_arguments',
           'terminate_local_procs',
           'TrainerProc',
           'get_logger',
           'pull_worker_log',
           'global_scatter',
           'global_gather',
]


def global_scatter(x,
                   local_count,
                   global_count,
                   group=None,
                   use_calc_stream=True):
    """
    Scatter data in x which has been put together belong to one expert 
    to n_expert * world_size exeperts according to local_count and receive tensors 
    from n_expert * world_size experts according
    to global_count.
    
    Args:
        x (Tensor): Tensor. Every element in the list must be a Tensor whose data type
            should be float16, float32, float64, int32 or int64.
        local_count (Tensor): Tensor which have n_expert * world_size elements that indicates
            how many data needed to be sent. Every element in the list must be a Tensor whose 
            data type should be int64.
        global_count (Tensor): Tensor which have n_expert * world_size elements that indicates
            how many data needed to be received. Every element in the list must be a Tensor whose 
            data type should be int64.
        group (Group, optional): The group instance return by new_group or None for global default group. Default: None.
        use_calc_stream (bool, optional): Wether to use calculation stream (True) or communication stream. Default: True.
    
    Returns:
        out (Tensor): The data received from all experts. 
    
    Examples:
        .. code-block:: python

            # required: distributed
            import numpy as np
            import paddle
            from paddle.distributed import init_parallel_env
            init_parallel_env()
            n_expert = 2
            world_size = 2
            d_model = 2
            in_feat = d_model
            local_input_buf = np.array([[1, 2],[3, 4],[5, 6],[7, 8],[9, 10]], \
            dtype=np.float32)
            if paddle.distributed.ParallelEnv().local_rank == 0:
                local_count = np.array([2, 1, 1, 1]) 
                global_count = np.array([2, 1, 1, 1])
            else:
                local_count = np.array([1, 1, 2, 1])
                global_count = np.array([1, 1, 2, 1])
            local_input_buf = paddle.to_tensor(local_input_buf, dtype="float32", stop_gradient=False)
            local_count = paddle.to_tensor(local_count, dtype="int64")
            global_count = paddle.to_tensor(global_count, dtype="int64")
            a = paddle.distributed.utils.global_scatter(local_input_buf, \
            local_count, global_count)
            a.stop_gradient = False
            print(a)
            # out for rank 0: [[1, 2], [3, 4], [1, 2], [5, 6], [3, 4]]
            # out for rank 1: [[7, 8], [5, 6], [7, 8], [9, 10], [9, 10]]
            # backward test
            c = a * a
            c.backward()
            print("local_input_buf.grad: ", local_input_buf.grad)
            # out for rank 0: [[2, 4], [6, 8], [10, 12], [14, 16], [18, 20]]
            # out for rank 1: [[2, 4], [6, 8], [10, 12], [14, 16], [18, 20]]
    """
    if group is not None and not group.is_member():
        return

    ring_id = 0 if group is None else group.id
    if in_dygraph_mode():
        return core.ops.global_scatter(x, local_count, \
                                    global_count,  \
                                    'use_calc_stream', use_calc_stream, \
                                    'ring_id', ring_id)
    else:
        op_type = 'global_scatter'
        check_variable_and_dtype(
            x, 'x', ['float16', 'float32', 'float64', 'int32', 'int64'],
            'global_scatter')
        check_variable_and_dtype(local_count, 'local_count', ['int64'],
                                 'global_scatter')
        check_variable_and_dtype(global_count, 'global_count', ['int64'],
                                 'global_scatter')

        helper = LayerHelper(op_type, **locals())
        out = helper.create_variable_for_type_inference(dtype=x.dtype)

        helper.append_op(
            type=op_type,
            inputs={
                'X': [x],
                'local_count': [local_count],
                'global_count': [global_count],
            },
            outputs={'Out': [out]},
            attrs={'ring_id': ring_id,
                   'use_calc_stream': use_calc_stream})
        return out


def global_gather(x,
                  local_count,
                  global_count,
                  group=None,
                  use_calc_stream=True):
    """
    Gather data in x to n_expert * world_size exeperts according to
    local_count and receive tensors from n_expert * world_size experts according
    to global_count.

    Args:
        x (Tensor): Tensor. Every element in the list must be a Tensor whose data type
            should be float16, float32, float64, int32 or int64.
        local_count (Tensor): Tensor which have n_expert * world_size elements that indicates
            how many data needed to be received. Every element in the list must be a Tensor whose 
            data type should be int64.
        global_count (Tensor): Tensor which have n_expert * world_size elements that indicates
            how many data needed to be sent. Every element in the list must be a Tensor whose 
            data type should be int64.
        group (Group, optional): The group instance return by new_group or None for global default group. Default: None.
        use_calc_stream (bool, optional): Wether to use calculation stream (True) or communication stream. Default: True.
    
    Returns:
        None.
    
    Examples:
        .. code-block:: python

            # required: distributed
            import numpy as np
            import paddle
            from paddle.distributed import init_parallel_env
            init_parallel_env()
            n_expert = 2
            world_size = 2
            d_model = 2
            in_feat = d_model
            local_input_buf = np.array([[1, 2],[3, 4],[5, 6],[7, 8],[9, 10]],\
                                        dtype=np.float32)
            if paddle.distributed.ParallelEnv().local_rank == 0:
                local_count = np.array([2, 1, 1, 1])
                global_count = np.array([2, 1, 1, 1])
            else:
                local_count = np.array([1, 1, 2, 1])
                global_count = np.array([1, 1, 2, 1])
            local_input_buf = paddle.to_tensor(local_input_buf, dtype="float32", stop_gradient=False)
            local_count = paddle.to_tensor(local_count, dtype="int64")
            global_count = paddle.to_tensor(global_count, dtype="int64")
            a = paddle.distributed.utils.global_gather(local_input_buf, local_count, global_count)
            print(a)
            # out for rank 0: [[1, 2], [3, 4], [7, 8], [1, 2], [7, 8]]
            # out for rank 1: [[5, 6], [9, 10], [3, 4], [5, 6], [9, 10]]
            a.stop_gradient = False
            c = a * a
            c.backward()
            print("local_input_buf.grad", local_input_buf.grad)
            # out for rank 0: [[2, 4], [6, 8], [10, 12], [14, 16], [18, 20]]
            # out for rank 1: [[2, 4], [6, 8], [10, 12], [14, 16], [18, 20]]
    """
    if group is not None and not group.is_member():
        return

    ring_id = 0 if group is None else group.id
    if in_dygraph_mode():
        return core.ops.global_gather(x, local_count, \
                                    global_count, \
                                    'use_calc_stream', use_calc_stream, \
                                    'ring_id', ring_id)
    else:
        op_type = 'global_gather'
        check_variable_and_dtype(
            x, 'x', ['float16', 'float32', 'float64', 'int32', 'int64'],
            'global_gather')

        check_variable_and_dtype(local_count, 'local_count', ['int64'],
                                 'global_gather')

        check_variable_and_dtype(global_count, 'global_count', ['int64'],
                                 'global_gather')
        helper = LayerHelper(op_type, **locals())
        out = helper.create_variable_for_type_inference(dtype=x.dtype)

        helper.append_op(
            type=op_type,
            inputs={
                'X': [x],
                'local_count': [local_count],
                'global_count': [global_count]
            },
            outputs={'Out': [out]},
            attrs={
                'ring_id': group,
                'use_calc_stream': use_calc_stream,
            })
        return out


logger = logging.getLogger("root")
logger.propagate = False


def get_cluster_from_args(args, selected_gpus):
    node_ips = [x.strip() for x in args.cluster_node_ips.split(',')]
    node_ip = args.node_ip
    node_rank = node_ips.index(node_ip)

    logger.debug("parsed from args:node_ips:{} node_ip:{} node_rank:{}".format(
        node_ips, node_ip, node_rank))

    free_ports = None
    if not args.use_paddlecloud and len(
            node_ips) <= 1 and args.started_port is None:
        free_ports = find_free_ports(len(selected_gpus))
        if free_ports is not None:
            free_ports = list(free_ports)
    else:
        started_port = 6070
        if args.started_port is not None:
            started_port = args.started_port

        free_ports = [
            x for x in range(started_port, started_port + len(selected_gpus))
        ]

    trainer_endpoints = []
    for ip in node_ips:
        trainer_endpoints.append(["%s:%d" % (ip, port) for port in free_ports])
    return get_cluster(node_ips, node_ip, trainer_endpoints, selected_gpus)


def get_gpus(selected_gpus):
    if selected_gpus is None:
        from paddle.fluid import core
        gpus_num = core.get_cuda_device_count()
        gpus = [str(x) for x in range(0, gpus_num)]
    else:
        cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES")
        if cuda_visible_devices is None or cuda_visible_devices == "":
            gpus = [x.strip() for x in selected_gpus.split(',')]
        else:
            # change selected_gpus into relative values
            # e.g. CUDA_VISIBLE_DEVICES=4,5,6,7; args.selected_gpus=4,5,6,7;
            # therefore selected_gpus=0,1,2,3
            cuda_visible_devices_list = cuda_visible_devices.split(',')
            for x in selected_gpus.split(','):
                assert x in cuda_visible_devices_list, "Can't find "\
                "your selected_gpus %s in CUDA_VISIBLE_DEVICES[%s]."\
                % (x, cuda_visible_devices)
            gpus = [
                cuda_visible_devices_list.index(x.strip())
                for x in selected_gpus.split(',')
            ]
            logger.info("Change selected_gpus into reletive values. --ips:{} "
                        "will change into relative_ips:{} according to your "
                        "CUDA_VISIBLE_DEVICES:{}".format(
                            selected_gpus, gpus, cuda_visible_devices_list))

    return gpus


def _print_arguments(args):
    print("-----------  Configuration Arguments -----------")
    for arg, value in sorted(six.iteritems(vars(args))):
        print("%s: %s" % (arg, value))
    print("------------------------------------------------")


class Hdfs(object):
    def __init__(self):
        self.hdfs_ugi = None
        self.hdfs_name = None
        self.hdfs_path = None

    def is_valid(self):
        return self.hdfs_ugi is not None and \
            self.hdfs_name is not None and \
            self.hdfs_path is not None

    def __str__(self):
        return "hdfs_ugi:{} hdfs_name:{} hdfs_path{}".format(
            self.hdfs_ugi, self.hdfs_name, self.hdfs_path)

    def __eq__(self, n):
        return self.hdfs_ugi == n.hdfs_ugi and \
            self.hdfs_name == n.hdfs_name and \
            self.hdfs_path == n.hdfs_path

    def __ne__(self, n):
        return not self == n


class Cluster(object):
    def __init__(self, hdfs):
        self.job_server = None
        self.pods = []
        self.hdfs = None
        self.job_stage_flag = None

    def __str__(self):
        return "job_server:{} pods:{} job_stage_flag:{} hdfs:{}".format(
            self.job_server, [str(pod) for pod in self.pods],
            self.job_stage_flag, self.hdfs)

    def __eq__(self, cluster):
        if len(self.pods) != len(cluster.pods):
            return False

        for a, b in zip(self.pods, cluster.pods):
            if a != b:
                return False

        if self.job_stage_flag != cluster.job_stage_flag:
            return False

        return True

    def __ne__(self, cluster):
        return not self.__eq__(cluster)

    def update_pods(self, cluster):
        self.pods = copy.copy(cluster.pods)

    def trainers_nranks(self):
        return len(self.trainers_endpoints())

    def pods_nranks(self):
        return len(self.pods)

    def trainers_endpoints(self):
        r = []
        for pod in self.pods:
            for t in pod.trainers:
                r.append(t.endpoint)
        return r

    def pods_endpoints(self):
        r = []
        for pod in self.pods:
            ep = "{}:{}".format(pod.addr, pod.port)
            assert pod.port != None and pod.addr != None, "{} not a valid endpoint".format(
                ep)
            r.append(ep)

        return r

    def get_pod_by_id(self, pod_id):
        for pod in self.pods:
            if str(pod_id) == str(pod.id):
                return pod

        return None


class JobServer(object):
    def __init__(self):
        self.endpoint = None

    def __str__(self):
        return "{}".format(self.endpoint)

    def __eq__(self, j):
        return self.endpint == j.endpoint

    def __ne__(self, j):
        return not self == j


class Trainer(object):
    def __init__(self):
        self.gpus = []
        self.endpoint = None
        self.rank = None

    def __str__(self):
        return "gpu:{} endpoint:{} rank:{}".format(self.gpus, self.endpoint,
                                                   self.rank)

    def __eq__(self, t):
        if len(self.gpus) != len(t.gpus):
            return False

        if self.endpoint != t.endpoint or \
                self.rank != t.rank:
            return False

        for a, b in zip(self.gpus, t.gpus):
            if a != b:
                return False

        return True

    def __ne__(self, t):
        return not self == t

    def get_rank(self):
        return self.rank


class Pod(object):
    def __init__(self):
        self.rank = None
        self.id = None
        self.addr = None
        self.port = None
        self.trainers = []
        self.gpus = []

    def __str__(self):
        return "rank:{} id:{} addr:{} port:{} visible_gpu:{} trainers:{}".format(
            self.rank, self.id, self.addr, self.port, self.gpus,
            [str(t) for t in self.trainers])

    def __eq__(self, pod):
        if self.rank != pod.rank or \
                self.id != pod.id or \
                self.addr != pod.addr or \
                self.port != pod.port:
            logger.debug("pod {} != {}".format(self, pod))
            return False

        if len(self.trainers) != len(pod.trainers):
            logger.debug("trainers {} != {}".format(self.trainers,
                                                    pod.trainers))
            return False

        for i in range(len(self.trainers)):
            if self.trainers[i] != pod.trainers[i]:
                logger.debug("trainer {} != {}".format(self.trainers[i],
                                                       pod.trainers[i]))
                return False

        return True

    def __ne__(self, pod):
        return not self == pod

    def parse_response(self, res_pods):
        pass

    def rank(self):
        return self.rank

    def get_visible_gpus(self):
        r = ""
        for g in self.gpus:
            r += "{},".format(g)

        assert r != "", "this pod {} can't see any gpus".format(self)

        r = r[:-1]
        return r


def get_logger(log_level, name="root"):
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    log_handler = logging.StreamHandler()
    log_format = logging.Formatter(
        '%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s')
    log_handler.setFormatter(log_format)
    logger.addHandler(log_handler)

    return logger


def get_cluster(node_ips, node_ip, trainer_endpoints, selected_gpus):
    assert type(trainer_endpoints) is list, "trainer_endpoints must be list"
    cluster = Cluster(hdfs=None)
    trainer_rank = 0
    for node_rank, ip in enumerate(node_ips):
        pod = Pod()
        pod.rank = node_rank
        pod.addr = ip
        cur_node_endpoints = trainer_endpoints[node_rank]
        # when use paddlecloud, endpoints may > selected_gpus(user_defined)
        assert len(cur_node_endpoints) >= len(
            selected_gpus
        ), "current trainer_endpoints size should be greater equal than selected_gpus size."
        for i in range(len(selected_gpus)):
            trainer = Trainer()
            trainer.gpus.append(selected_gpus[i])
            trainer.endpoint = "%s" % (cur_node_endpoints[i])
            trainer.rank = trainer_rank
            trainer_rank += 1

            pod.trainers.append(trainer)
        cluster.pods.append(pod)

    pod_rank = node_ips.index(node_ip)
    return cluster, cluster.pods[pod_rank]


def terminate_local_procs(procs):
    for p in procs:
        if p.proc.poll() is None:
            p.proc.terminate()
            if p.log_fn:
                p.log_fn.close()
            logger.debug("terminate process id:{}".format(p.proc.pid))

    #wait all process terminiated
    time.sleep(3)
    for step in range(0, 50):
        alive = False
        for p in procs:
            if p.proc.poll() is None:  # not termniate
                os.kill(p.proc.pid, signal.SIGKILL)
                alive = True

        if not alive:
            logger.info("terminate all the procs")
            return

        time.sleep(3)

    logger.fatal("can't kill all process and exit")
    exit(1)


def get_host_name_ip():
    try:
        host_name = socket.gethostname()
        host_ip = socket.gethostbyname(host_name)
        return host_name, host_ip
    except:
        return None


def add_arguments(argname, type, default, help, argparser, **kwargs):
    """Add argparse's argument.
    Usage:
    .. code-block:: python
        parser = argparse.ArgumentParser()
        add_argument("name", str, "Jonh", "User name.", parser)
        args = parser.parse_args()
    """
    type = strtobool if type == bool else type
    argparser.add_argument(
        "--" + argname,
        default=default,
        type=type,
        help=help + ' Default: %(default)s.',
        **kwargs)


def find_free_ports(num):
    def __free_port():
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.bind(('', 0))
            return s.getsockname()[1]

    port_set = set()
    step = 0
    while True:
        port = __free_port()
        if port not in port_set:
            port_set.add(port)

        if len(port_set) >= num:
            return port_set

        step += 1
        if step > 100:
            print(
                "can't find avilable port and use the specified static port now!"
            )
            return None

    return None


def _prepare_trainer_env(cluster, trainer, backend=None):
    if backend is None:
        backend = get_backend_by_compile_flag()  # for compatibility
    if backend == 'bkcl':
        proc_env = {
            "FLAGS_selected_xpus":
            "%s" % ",".join([str(g) for g in trainer.gpus]),
            "PADDLE_TRAINER_ID": "%d" % trainer.rank,
            "PADDLE_CURRENT_ENDPOINT": "%s" % trainer.endpoint,
            "PADDLE_TRAINERS_NUM": "%d" % cluster.trainers_nranks(),
            "PADDLE_TRAINER_ENDPOINTS": ",".join(cluster.trainers_endpoints())
        }
    elif backend == 'nccl':
        proc_env = {
            "FLAGS_selected_gpus":
            "%s" % ",".join([str(g) for g in trainer.gpus]),
            "PADDLE_TRAINER_ID": "%d" % trainer.rank,
            "PADDLE_CURRENT_ENDPOINT": "%s" % trainer.endpoint,
            "PADDLE_TRAINERS_NUM": "%d" % cluster.trainers_nranks(),
            "PADDLE_TRAINER_ENDPOINTS": ",".join(cluster.trainers_endpoints())
        }
    elif backend == 'gloo':
        # NOTE (xiongkun) default fall back into cpu only
        proc_env = {
            "PADDLE_TRAINER_ID": "%d" % trainer.rank,
            "PADDLE_CURRENT_ENDPOINT": "%s" % trainer.endpoint,
            "PADDLE_TRAINERS_NUM": "%d" % cluster.trainers_nranks(),
            "PADDLE_TRAINER_ENDPOINTS": ",".join(cluster.trainers_endpoints()),
            "PADDLE_DISTRI_BACKEND":
            backend,  # only add here, other will be auto
        }
    else:
        raise ValueError("backend must be one of 'gloo, nccl, bkcl'")

    return proc_env


class TrainerProc(object):
    def __init__(self):
        self.proc = None
        self.log_fn = None
        self.log_offset = None
        self.rank = None
        self.local_rank = None
        self.cmd = None


def start_local_trainers(cluster,
                         pod,
                         training_script,
                         training_script_args,
                         log_dir=None):
    current_env = copy.copy(os.environ.copy())
    #paddle broadcast ncclUniqueId use socket, and
    #proxy maybe make trainers unreachable, so delete them.
    #if we set them to "", grpc will log error message "bad uri"
    #so just delete them.
    current_env.pop("http_proxy", None)
    current_env.pop("https_proxy", None)

    procs = []
    for idx, t in enumerate(pod.trainers):
        proc_env = _prepare_trainer_env(cluster, t)
        current_env.update(proc_env)

        logger.debug("trainer proc env:{}".format(current_env))

        cmd = [sys.executable, "-u", training_script] + training_script_args

        logger.info("start trainer proc:{} env:{}".format(cmd, proc_env))

        fn = None
        if log_dir is not None:
            os.system("mkdir -p {}".format(log_dir))
            fn = open("%s/workerlog.%d" % (log_dir, idx), "a")
            proc = subprocess.Popen(cmd, env=current_env, stdout=fn, stderr=fn)
        else:
            proc = subprocess.Popen(cmd, env=current_env)

        tp = TrainerProc()
        tp.proc = proc
        tp.rank = t.rank
        tp.local_rank = idx
        tp.log_fn = fn
        tp.log_offset = fn.tell() if fn else None
        tp.cmd = cmd

        procs.append(tp)

    return procs


def pull_worker_log(tp):
    if tp.log_fn:
        with open(tp.log_fn.name, 'r') as fin:
            fin.seek(tp.log_offset, 0)
            for line in fin:
                try:
                    sys.stdout.write(line)
                except UnicodeEncodeError:
                    sys.stdout.write(
                        'UnicodeEncodeError occurs at this line. '
                        'Please refer to the original log file "%s"\n' %
                        tp.log_fn.name)
            tp.log_offset = fin.tell()


def watch_local_trainers(procs, nranks):
    try:
        error = False
        error_rank = []
        # wait all process finish or one error
        alive = False
        for p in procs:
            if p.log_fn and p.local_rank == 0:
                pull_worker_log(p)

            ret = p.proc.poll()
            if ret is None:
                alive = True
            elif ret != 0:
                error = True
                error_rank.append(p.rank)

        if error:
            terminate_local_procs(procs)
            exit(1)

    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt, exit")
        terminate_local_procs(procs)
        raise
    except SystemExit:
        logger.error(
            "ABORT!!! Out of all {} trainers, the trainer process with rank={} was aborted. Please check its log.".
            format(nranks, error_rank))
        terminate_local_procs(procs)
        raise
    except:
        logger.error(
            "ABORT!!! Out of all {} trainers, the trainer process with rank={} was aborted. Please check its log.".
            format(nranks, error_rank))
        terminate_local_procs(procs)
        raise

    return alive
