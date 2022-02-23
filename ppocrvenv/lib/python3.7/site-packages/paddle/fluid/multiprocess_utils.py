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

import sys
import signal
import atexit

from . import core

# NOTE: queue has a different name in python2 and python3
import queue

# multi-process worker check indices queue interval, avoid
# hanging in subprocess data loading
MP_STATUS_CHECK_INTERVAL = 5.

# NOTE: [ mmap files clear ] If there is still data in the multiprocess queue when the main process finishes reading,
# the data in the queue needs to be popped. Then the LoDTensor read by the main process
# from the child process will automatically clear the memory-mapped file.
multiprocess_queue_set = set()


def _clear_multiprocess_queue_set():
    global multiprocess_queue_set
    for data_queue in multiprocess_queue_set:
        while True:
            try:
                data_queue.get_nowait()
            except queue.Empty:
                break


# NOTE: main process clear function at exit
def _cleanup():
    # NOTE: inter-process Queue shared memory objects clear function
    _clear_multiprocess_queue_set()
    # NOTE: main process memory map files clear funciton
    core._cleanup_mmap_fds()


# NOTE: for child process clear function at exit
def _cleanup_mmap():
    # clear memory map files in child process
    core._cleanup_mmap_fds()


# NOTE used for register a function to be executed at interpreter exit.
class CleanupFuncRegistrar():
    # Record the cleanup functions that have been executed
    _executed_func_set = set()
    # Record the cleanup functions that have been registered
    _registered_func_set = set()

    @classmethod
    def register(cls, function, signals=[]):
        def _func_exectuor():
            if function not in cls._executed_func_set:
                try:
                    function()
                finally:
                    cls._executed_func_set.add(function)

        def _func_register(function):
            if not callable(function):
                raise TypeError("%s is not callable object." % (function))
            # check function object whether hash-able
            set([function])
            if function not in cls._registered_func_set:
                atexit.register(_func_exectuor)
                cls._registered_func_set.add(function)

        def _signal_handler(signum=None, frame=None):
            _func_exectuor()
            if signum is not None:
                if signum == signal.SIGINT:
                    raise KeyboardInterrupt
                sys.exit(signum)

        def _signal_register(signals):
            signals = set(signals)
            for sig in signals:
                orig_handler = signal.signal(sig, _signal_handler)
                if orig_handler not in (signal.SIG_DFL, signal.SIG_IGN):
                    if (sig == signal.SIGINT and
                            orig_handler is signal.default_int_handler):
                        continue
                    if orig_handler not in cls._registered_func_set:
                        atexit.register(orig_handler)
                        cls._registered_func_set.add(orig_handler)

        # deal with signals
        _signal_register(signals)
        # deal with function
        _func_register(function)


# NOTE: [ mmap files clear ] When the main process exits unexpectedly, the remaining
# shared memory objects in the inter-process Queue and the main process (mostly in the
# BlockingQueue) may not be completely released, resulting in the corresponding
# memory-mapped file remaining on the disk (/dev/shm), so register this function
# to clean up shared memory objects in these two queues before the python interpreter exits.
# NOTE: Currently multi-process DataLoader only supports Linux platform
if not (sys.platform == 'darwin' or sys.platform == 'win32'):
    CleanupFuncRegistrar.register(_cleanup)

# ------------ SIGCHLD handler setting --------------
_SIGCHLD_handler_set = False


def _set_SIGCHLD_handler():
    global _SIGCHLD_handler_set
    if _SIGCHLD_handler_set:
        return

    current_handler = signal.getsignal(signal.SIGCHLD)
    if not callable(current_handler):
        current_handler = None

    def __handler__(signum, frame):
        # NOTE: Here the signum is SIGCHLD, when the child process exits,
        # this handler will be called whenever the child process exits
        # normally or abnormally.
        core._throw_error_if_process_failed()
        if current_handler is not None:
            current_handler(signum, frame)

    signal.signal(signal.SIGCHLD, __handler__)
    _SIGCHLD_handler_set = True
