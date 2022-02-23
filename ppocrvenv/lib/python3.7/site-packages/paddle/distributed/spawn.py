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

from __future__ import print_function, division

import multiprocessing
import os
import signal
import six
import sys
import warnings

from paddle.distributed.utils import _print_arguments
from paddle.distributed.utils import _prepare_trainer_env
from paddle.distributed.utils import get_host_name_ip
from paddle.distributed.cloud_utils import get_cluster_and_pod, _get_trainers_num
from paddle.distributed.fleet.launch import get_cluster_from_args
from paddle.distributed.fleet.cloud_utils import use_paddlecloud
from paddle.distributed.fleet.launch_utils import DeviceMode, check_backend, block_windows_and_macos
from paddle.device import get_device

# deprecated module import
from paddle.fluid import core
from paddle.fluid.framework import _cpu_num, set_flags

__all__ = []


class ParallelEnvArgs(object):
    def __init__(self):
        # Paddle cluster nodes ips, such as 192.168.0.16,192.168.0.17..
        self.cluster_node_ips = None

        # The current node ip.
        self.node_ip = None

        # whether to use paddlecloud platform to run your multi-process job.
        # If false, no need to set this argument.
        self.use_paddlecloud = None

        # The trainer's started port on a single node
        self.started_port = None

        # Print the config or not
        self.print_config = True

        # It's for gpu training and the training process will run 
        # on the selected_devices, each process is bound to a single GPU. 
        # And if it's not set, this module will use all the gpu cards 
        # for training.
        self.selected_devices = None


def _py_supported_check():
    if not sys.version_info >= (3, 4):
        raise RuntimeError(
            "Use `paddle.distributed.spawn` to start parallel training "
            "requires python version greater than 3.4, if your python "
            "is lower than this version, please use "
            "`paddle.distributed.launch` instead.")


def _options_valid_check(options):
    # `print_config` keeped as a debug options, not show to users
    supported_options = [
        'start_method', 'ips', 'gpus', 'xpus', 'print_config', 'backend'
    ]
    deprecated_options = [
        'selected_devices', 'started_port', 'cluster_node_ips', 'node_ip',
        'use_paddlecloud'
    ]
    for key in options:
        if key not in supported_options:
            if key in deprecated_options:
                warnings.warn(
                    "The config option (%s) of `paddle.distributed.spawn` is deprecated. "
                    "Please use the latest config options stated in the `spawn` API documentation."
                    % key, DeprecationWarning)
            else:
                raise ValueError(
                    "The config option (%s) of `paddle.distributed.spawn` is not supported."
                    % key)


def _get_default_nprocs():
    device = get_device()
    if 'gpu' in device:
        return core.get_cuda_device_count()
    elif 'xpu' in device:
        return core.get_xpu_device_count()
    elif 'cpu' in device:
        return multiprocessing.cpu_count()
    else:
        raise RuntimeError(
            "`paddle.distributed.spawn` does not support parallel training on device `{}` now.".
            format(device))


def _get_default_backend():
    device = get_device()
    if 'gpu' in device:
        return 'nccl'
    elif 'xpu' in device:
        return 'bkcl'
    elif 'cpu' in device:
        return 'gloo'
    else:
        raise RuntimeError(
            "`paddle.distributed.spawn` does not support parallel training on device `{}` now.".
            format(device))


def _get_node_ip(ips):
    node_ip = None
    node_ips = [x.strip() for x in ips.split(',')]
    if len(node_ips) == 1:
        node_ip = node_ips[0]
    else:
        _, node_ip = get_host_name_ip()
    return node_ip


def _get_subprocess_env_list(nprocs, options):
    # NOTE (xiongkun03) Why put backend deduction  here ? 
    # Becase _get_subprocess_env_list is used by many testcases. 
    # So for campability, we put backend deduction here 

    # logic for handle backend option
    if 'backend' not in options or options['backend'] == 'auto':
        options['backend'] = _get_default_backend()
    check_backend(options['backend'])
    block_windows_and_macos(options['backend'])

    # contruct processes env list
    processes_env_list = []

    # get args from kwargs
    args = ParallelEnvArgs()

    # deal with `ips`
    args.cluster_node_ips = options.get('ips', None)
    if args.cluster_node_ips is None:
        args.cluster_node_ips = options.get('cluster_node_ips', None)
        if args.cluster_node_ips is None:
            args.cluster_node_ips = "127.0.0.1"

    # deal with `gpus` or `xpus`
    # set default selected devices(gpus or xpus)
    # e.g. if the nprocs is 4, the selected gpus is "0,1,2,3"
    # NOTE(chenweihang): [ why not use FLAGS_selected_gpus or FLAGS_selected_xpus directly? ]
    # because the FLAGS_selected_gpus or FLAGS_selected_xpus may be used in other place,
    # if we set FLAGS_selected_gpus or FLAGS_selected_xpus to be `0,1,2,3`, it may cause error
    # when using `ParallelEnv`
    # NOTE(chenweihang): use absolute gpu or xpu card id
    if options['backend'] == 'nccl':
        args.selected_devices = options.get('gpus', None)
        if args.selected_devices is None:
            args.selected_devices = options.get('selected_devices', None)
        env_devices = os.getenv("CUDA_VISIBLE_DEVICES", None)
        if env_devices is None or env_devices == "":
            env_devices_list = [
                str(x) for x in six.moves.range(core.get_cuda_device_count())
            ]
        else:
            env_devices_list = env_devices.split(',')
        if args.selected_devices is None:
            if len(env_devices_list) < nprocs:
                raise RuntimeError(
                    "the number of visible devices(%d) is less than the number "
                    "of spawn processes(%d), please ensure that the correct "
                    "`nprocs` argument is passed or the environment variable "
                    "`CUDA_VISIBLE_DEVICES` is correctly configured." %
                    (len(env_devices_list), nprocs))
            args.selected_devices = ",".join(
                [str(env_devices_list[x]) for x in range(0, nprocs)])
        else:
            selected_device_list = args.selected_devices.split(',')
            if len(selected_device_list) != nprocs:
                raise ValueError(
                    "The number of selected devices(%s) is not equal to "
                    "the number of spawn processes(%d), please ensure that the "
                    "correct `nprocs` and `gpus` arguments are passed." %
                    (len(selected_device_list), nprocs))
            for card_id in selected_device_list:
                if card_id not in env_devices_list:
                    raise ValueError("The selected gpu card %s cannot found in "
                                     "CUDA_VISIBLE_DEVICES (%s)." %
                                     (card_id, ",".join(env_devices_list)))

    elif options['backend'] == 'bkcl':
        args.selected_devices = options.get('xpus', None)
        if args.selected_devices is None:
            args.selected_devices = options.get('selected_devices', None)
        env_devices = os.getenv("XPU_VISIBLE_DEVICES", None)
        if env_devices is None or env_devices == "":
            env_devices_list = [
                str(x) for x in six.moves.range(core.get_xpu_device_count())
            ]
        else:
            env_devices_list = env_devices.split(',')
        if args.selected_devices is None:
            if len(env_devices_list) < nprocs:
                raise RuntimeError(
                    "the number of visible devices(%d) is less than the number "
                    "of spawn processes(%d), please ensure that the correct "
                    "`nprocs` argument is passed or the environment variable "
                    "`XPU_VISIBLE_DEVICES` is correctly configured." %
                    (len(env_devices_list), nprocs))
            args.selected_devices = ",".join(
                [str(env_devices_list[x]) for x in range(0, nprocs)])
        else:
            selected_device_list = args.selected_devices.split(',')
            if len(selected_device_list) != nprocs:
                raise ValueError(
                    "The number of selected devices(%s) is not equal to "
                    "the number of spawn processes(%d), please ensure that the "
                    "correct `nprocs` and `xpus` arguments are passed." %
                    (len(selected_device_list), nprocs))
            for card_id in selected_device_list:
                if card_id not in env_devices_list:
                    raise ValueError("The selected xpu card %s cannot found in "
                                     "XPU_VISIBLE_DEVICES (%s)." %
                                     (card_id, ",".join(env_devices_list)))
    elif options['backend'] == 'gloo':
        # TODO check gpu / xpu flag must not exist
        warnings.warn(
            "Your model will be trained under CPUONLY mode by using GLOO,"
            "because CPUPlace is specified manually or your installed PaddlePaddle only support CPU Device."
        )
        args.paddle_cpuonly = True
        args.selected_devices = None
        args.ips = args.cluster_node_ips
        assert options.get(
            'use_paddlecloud',
            None) is None, "CPUONLY spawn doesn't support use paddle cloud"
        assert len(
            args.cluster_node_ips.split(',')
        ) <= 1, "CPUONLY spawn only support single trainer, that is len(ips)=1, but got %s."
        assert _get_trainers_num(
        ) == 1, "CPUONLY spawn doesn't support multi-trainer"

    # set other inner args
    args.node_ip = options.get('node_ip', None)
    if args.node_ip is None:
        args.node_ip = _get_node_ip(args.cluster_node_ips)

    args.started_port = options.get('started_port', None)

    args.use_paddlecloud = options.get('use_paddlecloud', None)
    if args.use_paddlecloud is None:
        args.use_paddlecloud = use_paddlecloud()

    # get cluster and pod config
    if options['backend'] == 'gloo':
        devices_per_proc = [x for x in range(0, nprocs)]
        cluster, pod = get_cluster_from_args(args, DeviceMode.CPU,
                                             devices_per_proc)
    else:
        cluster, pod = get_cluster_and_pod(args)

    # prepare subprocess env list
    for trainer in pod.trainers:
        processes_env_list.append(
            _prepare_trainer_env(cluster, trainer, options['backend']))

    # [Debug] print config
    args.print_config = options.get('print_config', False)
    if args.print_config:
        _print_arguments(args)

    return processes_env_list


def _remove_risky_env():
    # remove useless env vars
    # no copy, each process will hold env vars itself
    os.environ.pop("http_proxy", None)
    os.environ.pop("https_proxy", None)


def _set_trainer_env(env_dict, backend):
    # NOTE(chenweihang): [ Why need set FLAGS_selected_gpus or FLAGS_selected_xpus here? ]
    # When the child process starts, it will inherit the configuration of the 
    # main process and set the FLAGS once, but the environment variable has 
    # not been set at this time, which leads to the FLAGS_selected_gpus or FLAGS_selected_xpus
    # is keep same with mainprocess(usually empty), so manually update the flags here

    # NOTE(xiongkun): why put backend here?  because if gloo, we shouldn't set FLAGS_selectedXXX
    #

    if backend == 'nccl':
        set_flags({'FLAGS_selected_gpus': env_dict['FLAGS_selected_gpus']})
    elif backend == 'bkcl':
        set_flags({'FLAGS_selected_xpus': env_dict['FLAGS_selected_xpus']})
    else:
        #NOTE(xiongkun) why not raise Error ? 
        # So far, we added support for CPU parallel, and will be applied when paddle is not 
        # compiled with cuda or xp. just do nothing.
        pass

    for var_name in env_dict:
        os.environ[var_name] = env_dict[var_name]


def _func_wrapper(func, args, error_queue, return_queue, env_dict, backend):
    try:
        # config subprocess environment variables
        _remove_risky_env()
        _set_trainer_env(env_dict, backend)
        # execute function
        result = func(*args)
        # record function return value
        return_queue.put(result)
    except KeyboardInterrupt:
        pass
    except Exception:
        import traceback
        error_queue.put(traceback.format_exc())
        sys.exit(1)


class MultiprocessContext(object):
    def __init__(self, processes, error_queues, return_queues):
        _py_supported_check()
        self.error_queues = error_queues
        # NOTE(chenweihang): The `spawn` method is mainly used 
        # to wrap the outermost execution function of the program for 
        # parallel execution. Generally, the return value is not concerned, 
        # but if the user needs to obtain the return value, users can get  
        # the return result of each process from context.return_queues
        self.return_queues = return_queues
        self.processes = processes
        self.sentinels = {
            process.sentinel: index
            for index, process in enumerate(processes)
        }

    def join(self, timeout=None):
        if len(self.sentinels) == 0:
            return True

        ready = multiprocessing.connection.wait(
            self.sentinels.keys(), timeout=timeout)

        error_index = None
        for sentinel in ready:
            index = self.sentinels.pop(sentinel)
            process = self.processes[index]
            process.join()
            if process.exitcode != 0:
                error_index = index
                break

        if error_index is None:
            return len(self.sentinels) == 0

        for process in self.processes:
            if process.is_alive():
                process.terminate()
            process.join()

        self._throw_exception(error_index)

    def _throw_exception(self, error_index):
        if self.error_queues[error_index].empty():
            exitcode = self.processes[error_index].exitcode
            if exitcode < 0:
                name = signal.Signals(-exitcode).name
                raise Exception("Process %d terminated with signal %s." %
                                (error_index, name))
            else:
                raise Exception("Process %d terminated with exit code %d." %
                                (error_index, exitcode))

        original_trace = self.error_queues[error_index].get()
        msg = "\n\n----------------------------------------------\n" \
              "Process %d terminated with the following error:\n" \
              "----------------------------------------------\n\n" % error_index
        msg += original_trace
        raise Exception(msg)


def spawn(func, args=(), nprocs=-1, join=True, daemon=False, **options):
    """
    Start multiple processes with ``spawn`` method for parallel training.

    .. note::
        ``spawn`` now only supports GPU or XPU collective mode. The collective mode
        of GPU and XPU cannot be started at the same time, so the option `gpus` and
        `xpus` cannot be configured at the same time.

    Args:
        func (function): The target function is called by spawned process.
            This function need to be able to pickled, so it must be defined
            at the top level of a module.
        args (list|tuple, optional): Arguments passed to ``func``.
        nprocs (int, optional): Number of processed to start. Default: -1.
            when nprocs is -1, the available device will be obtained from
            the environment variable when the model is executed: If use GPU,
            the currently available device ID is obtained from the environment
            variable CUDA_VISIBLE_DEVICES; If use XPU, the currently available
            device ID is obtained from the environment variable XPU_VISIBLE_DEVICES.
        join (bool, optional): Perform a blocking join on all spawned processes.
            Default: True.
        daemon (bool, optional): The spawned processes' daemon flag. Default: False.
        **options(dict, optional): Other initial parallel execution environment
            configuration options. The following options are currently supported:
            (1) start_method (string): the way to start a process.
            The start method can be ``spawn`` , ``fork`` , ``forkserver`` .
            Because the CUDA runtime does not support the ``fork`` start method,
            when use CUDA in subprocesses, we should start process by ``spawn``
            or ``forkserver`` method. Default: "spawn" ;
            (2) gpus (string): The training process will run on the
            selected gpus, such as "0,1,2,3". Default: None;
            (3) xpus (string): The training process will run on the
            selected xpus, such as "0,1,2,3". Default: None;
            (4) ips (string): Paddle cluster nodes ips, such as
            "192.168.0.16,192.168.0.17". Default: "127.0.0.1" .

    Returns:
        ``MultiprocessContext`` object, it hold the spawned processes.

    Examples:
        .. code-block:: python

            from __future__ import print_function

            import paddle
            import paddle.nn as nn
            import paddle.optimizer as opt
            import paddle.distributed as dist

            class LinearNet(nn.Layer):
                def __init__(self):
                    super(LinearNet, self).__init__()
                    self._linear1 = nn.Linear(10, 10)
                    self._linear2 = nn.Linear(10, 1)

                def forward(self, x):
                    return self._linear2(self._linear1(x))

            def train(print_result=False):
                # 1. initialize parallel environment
                dist.init_parallel_env()

                # 2. create data parallel layer & optimizer
                layer = LinearNet()
                dp_layer = paddle.DataParallel(layer)

                loss_fn = nn.MSELoss()
                adam = opt.Adam(
                    learning_rate=0.001, parameters=dp_layer.parameters())

                # 3. run layer
                inputs = paddle.randn([10, 10], 'float32')
                outputs = dp_layer(inputs)
                labels = paddle.randn([10, 1], 'float32')
                loss = loss_fn(outputs, labels)

                if print_result is True:
                    print("loss:", loss.numpy())

                loss.backward()

                adam.step()
                adam.clear_grad()

            # Usage 1: only pass function.
            # If your training method no need any argument, and
            # use all visible devices for parallel training.
            if __name__ == '__main__':
                dist.spawn(train)

            # Usage 2: pass function and arguments.
            # If your training method need some arguments, and
            # use all visible devices for parallel training.
            if __name__ == '__main__':
                dist.spawn(train, args=(True,))

            # Usage 3: pass function, arguments and nprocs.
            # If your training method need some arguments, and
            # only use part of visible devices for parallel training.
            # If your machine hold 8 cards {0,1,2,3,4,5,6,7},
            # this case will use cards {0,1}; If you set
            # CUDA_VISIBLE_DEVICES=4,5,6,7, this case will use
            # cards {4,5}
            if __name__ == '__main__':
                dist.spawn(train, args=(True,), nprocs=2)

            # Usage 4: pass function, arguments, nprocs and gpus.
            # If your training method need some arguments, and
            # only use part of visible devices for parallel training,
            # but you can't set your machine's environment variable
            # CUDA_VISIBLE_DEVICES, such as it is None or all cards
            # {0,1,2,3,4,5,6,7}, you can pass `gpus` to
            # select the GPU cards you want to use. For example,
            # this case will use cards {4,5} if your machine hold 8 cards.
            if __name__ == '__main__':
                dist.spawn(train, args=(True,), nprocs=2, gpus='4,5')
    """
    # NOTE(chenweihang): [ why only supports python3.4+ ? ]
    # Python supported setting the child process startup method
    # since 3.4. The previous version can only use the default startup 
    # method, while the default startup method of Unix is fork, which 
    # cannot support CUDA runtime multi-process
    _py_supported_check()

    # Give an error hint when the users enter a configuration option 
    # that does not exist
    _options_valid_check(options)

    # get default nprocs
    if nprocs == -1:
        nprocs = _get_default_nprocs()

    # NOTE(chenweihang): [ why need get cluster info before run? ]
    # when using `paddle.distributed.spawn` start parallel training, 
    # we should get cluster info before starting subprocess, and pass 
    # correct info to each subprocess
    procs_env_list = _get_subprocess_env_list(nprocs, options)

    # start processes
    # NOTE(chenweihang): [ why default start method is spawn? ]
    # The CUDA runtime does not support the fork start method, 
    # either the spawn or forkserver start method are required 
    # to use CUDA in subprocesses.
    start_method = options.get('start_method', None)
    if start_method is None:
        start_method = 'spawn'
    mp = multiprocessing.get_context(start_method)

    error_queues = []
    return_queues = []
    processes = []
    for i in range(nprocs):
        error_queue = mp.SimpleQueue()
        return_queue = mp.SimpleQueue()
        process = mp.Process(
            target=_func_wrapper,
            args=(func, args, error_queue, return_queue, procs_env_list[i],
                  options['backend']))
        process.daemon = daemon
        process.start()
        error_queues.append(error_queue)
        return_queues.append(return_queue)
        processes.append(process)

    context = MultiprocessContext(processes, error_queues, return_queues)
    if not join:
        return context

    # loop until all process end
    while not context.join():
        pass

    # finally return context
    return context
