# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except jin compliance with the License.
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
import sys
import time
import warnings
from multiprocessing import Process, Manager

# deprecated module import
from paddle.fluid import core
from paddle.distributed.fleet.base.private_helper_function import wait_server_ready

__all__ = []

_global_gloo_ctx = None


def _start_kv_server(port, http_server_d, size):
    from paddle.distributed.fleet.utils.http_server import KVServer
    http_server = KVServer(int(port), size=size)
    http_server.start()
    wait_seconds = 3
    while http_server_d.get("running", False) or not http_server.should_stop():
        time.sleep(wait_seconds)
    http_server.stop()


def gloo_init_parallel_env(rank_id, rank_num, server_endpoint):
    """
    Initialize parallel environment with gloo for cpu only.

    Args:
        - rank_id（int, required) - the index of current rank;
        - rank_num (int, required) - the number of ranks in this parallel env;
        - server_endpoint (str, required) - endpoint of server to init gloo context in ip:port format;

    Returns:
        None

    Examples:
        .. code-block:: python

            import paddle
            import multiprocessing
            from contextlib import closing
            import socket

            port_set = set()

            def find_free_port():
                def _free_port():
                    with closing(socket.socket(socket.AF_INET,
                        socket.SOCK_STREAM)) as s:
                        s.bind(('', 0))
                        return s.getsockname()[1]
                while True:
                    port = _free_port()
                    if port not in port_set:
                        port_set.add(port)
                        return port

            def test_gloo_init(id, rank_num, server_endpoint):
                paddle.distributed.gloo_init_parallel_env(
                    id, rank_num, server_endpoint)

            def test_gloo_init_with_multiprocess(num_of_ranks):
                jobs = []
                server_endpoint = "127.0.0.1:%s" % (find_free_port())
                for id in range(num_of_ranks):
                    p = multiprocessing.Process(
                        target=test_gloo_init,
                        args=(id, num_of_ranks, server_endpoint))
                    jobs.append(p)
                    p.start()
                for proc in jobs:
                    proc.join()

            if __name__ == '__main__':
                # Arg: number of ranks (processes)
                test_gloo_init_with_multiprocess(2)
    """

    assert (rank_num < 2) is False, \
        "rank_num should greater than or equal to 2 for parallel environment initialzation."

    # init gloo context
    manager = Manager()
    # global dict to store status
    http_server_status = manager.dict()
    http_server_status["running"] = False
    if rank_id == 0:
        # The scope for worker used by http server is '_worker'
        size = {'_worker': rank_num}
        http_server_proc = Process(
            target=_start_kv_server,
            args=(int(server_endpoint.split(":")[1]), http_server_status, size))
        http_server_proc.daemon = True
        http_server_status["running"] = True
        http_server_proc.start()

    # all processes in this parallel environment should wait until server is ready
    wait_server_ready([server_endpoint])

    gloo_strategy = core.GlooParallelStrategy()
    gloo_strategy.rank = rank_id
    gloo_strategy.rank_num = rank_num
    gloo_strategy.ip_address = server_endpoint.split(":")[0]
    gloo_strategy.ip_port = int(server_endpoint.split(":")[1])
    # default_init_timeout_seconds
    gloo_strategy.init_seconds = 3600
    # default_run_timeout_seconds
    gloo_strategy.run_seconds = 9999999

    global _global_gloo_ctx
    _global_gloo_ctx = core.GlooParallelContext(gloo_strategy)
    _global_gloo_ctx.init()

    if rank_id == 0:
        http_server_status["running"] = False
        http_server_proc.join()


def gloo_barrier():
    """
    Call barrier function with initialized gloo context.

    Args:
        None

    Returns:
        None

    Examples:
        .. code-block:: python

            import paddle
            import multiprocessing
            from contextlib import closing
            import socket

            port_set = set()

            def find_free_port():
                def _free_port():
                    with closing(socket.socket(socket.AF_INET,
                        socket.SOCK_STREAM)) as s:
                        s.bind(('', 0))
                        return s.getsockname()[1]
                while True:
                    port = _free_port()
                    if port not in port_set:
                        port_set.add(port)
                        return port

            def test_gloo_barrier(id, rank_num, server_endpoint):
                paddle.distributed.gloo_init_parallel_env(
                    id, rank_num, server_endpoint)
                paddle.distributed.gloo_barrier()

            def test_gloo_barrier_with_multiprocess(num_of_ranks):
                jobs = []
                server_endpoint = "127.0.0.1:%s" % (find_free_port())
                for id in range(num_of_ranks):
                    p = multiprocessing.Process(
                        target=test_gloo_barrier,
                        args=(id, num_of_ranks, server_endpoint))
                    jobs.append(p)
                    p.start()
                for proc in jobs:
                    proc.join()

            if __name__ == '__main__':
                # Arg: number of ranks (processes)
                test_gloo_barrier_with_multiprocess(2)
    """

    assert _global_gloo_ctx is not None, "gloo context is not initialzed."
    _global_gloo_ctx.barrier()


def gloo_release():
    """
    Release the parallel environment initialized by gloo

    Args:
        None

    Returns:
        None

    Examples:
        .. code-block:: python

            import paddle
            import multiprocessing
            from contextlib import closing
            import socket

            port_set = set()

            def find_free_port():
                def _free_port():
                    with closing(socket.socket(socket.AF_INET,
                        socket.SOCK_STREAM)) as s:
                        s.bind(('', 0))
                        return s.getsockname()[1]
                while True:
                    port = _free_port()
                    if port not in port_set:
                        port_set.add(port)
                        return port

            def test_gloo_release(id, rank_num, server_endpoint):
                paddle.distributed.gloo_init_parallel_env(
                    id, rank_num, server_endpoint)
                paddle.distributed.gloo_barrier()
                paddle.distributed.gloo_release()

            def test_gloo_release_with_multiprocess(num_of_ranks):
                jobs = []
                server_endpoint = "127.0.0.1:%s" % (find_free_port())
                for id in range(num_of_ranks):
                    p = multiprocessing.Process(
                        target=test_gloo_release,
                        args=(id, num_of_ranks, server_endpoint))
                    jobs.append(p)
                    p.start()
                for proc in jobs:
                    proc.join()

            if __name__ == '__main__':
                # Arg: number of ranks (processes)
                test_gloo_release_with_multiprocess(2)
    """

    if _global_gloo_ctx is not None:
        _global_gloo_ctx.release()
