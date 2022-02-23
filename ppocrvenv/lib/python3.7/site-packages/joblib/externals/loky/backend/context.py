###############################################################################
# Basic context management with LokyContext and  provides
# compat for UNIX 2.7 and 3.3
#
# author: Thomas Moreau and Olivier Grisel
#
# adapted from multiprocessing/context.py
#  * Create a context ensuring loky uses only objects that are compatible
#  * Add LokyContext to the list of context of multiprocessing so loky can be
#    used with multiprocessing.set_start_method
#  * Add some compat function for python2.7 and 3.3.
#
from __future__ import division

import os
import sys
import subprocess
import traceback
import warnings
import multiprocessing as mp


from .process import LokyProcess, LokyInitMainProcess

START_METHODS = ['loky', 'loky_init_main']
_DEFAULT_START_METHOD = None

# Cache for the number of physical cores to avoid repeating subprocess calls.
# It should not change during the lifetime of the program.
physical_cores_cache = None

if sys.version_info[:2] >= (3, 4):
    from multiprocessing import get_context as mp_get_context
    from multiprocessing.context import assert_spawning, set_spawning_popen
    from multiprocessing.context import get_spawning_popen, BaseContext

    START_METHODS += ['spawn']
    if sys.platform != 'win32':
        START_METHODS += ['fork', 'forkserver']

    def get_context(method=None):
        # Try to overload the default context
        method = method or _DEFAULT_START_METHOD or "loky"
        if method == "fork":
            # If 'fork' is explicitly requested, warn user about potential
            # issues.
            warnings.warn("`fork` start method should not be used with "
                          "`loky` as it does not respect POSIX. Try using "
                          "`spawn` or `loky` instead.", UserWarning)
        try:
            context = mp_get_context(method)
        except ValueError:
            raise ValueError("Unknown context '{}'. Value should be in {}."
                             .format(method, START_METHODS))

        return context

else:
    if sys.platform != 'win32':
        import threading
        # Mechanism to check that the current thread is spawning a process
        _tls = threading.local()
        popen_attr = 'spawning_popen'
    else:
        from multiprocessing.forking import Popen
        _tls = Popen._tls
        popen_attr = 'process_handle'

    BaseContext = object

    def get_spawning_popen():
        return getattr(_tls, popen_attr, None)

    def set_spawning_popen(popen):
        setattr(_tls, popen_attr, popen)

    def assert_spawning(obj):
        if get_spawning_popen() is None:
            raise RuntimeError(
                '%s objects should only be shared between processes'
                ' through inheritance' % type(obj).__name__
            )

    def get_context(method=None):
        method = method or _DEFAULT_START_METHOD or 'loky'
        if method == "loky":
            return LokyContext()
        elif method == "loky_init_main":
            return LokyInitMainContext()
        else:
            raise ValueError("Unknown context '{}'. Value should be in {}."
                             .format(method, START_METHODS))


def set_start_method(method, force=False):
    global _DEFAULT_START_METHOD
    if _DEFAULT_START_METHOD is not None and not force:
        raise RuntimeError('context has already been set')
    assert method is None or method in START_METHODS, (
        "'{}' is not a valid start_method. It should be in {}"
        .format(method, START_METHODS))

    _DEFAULT_START_METHOD = method


def get_start_method():
    return _DEFAULT_START_METHOD


def cpu_count(only_physical_cores=False):
    """Return the number of CPUs the current process can use.

    The returned number of CPUs accounts for:
     * the number of CPUs in the system, as given by
       ``multiprocessing.cpu_count``;
     * the CPU affinity settings of the current process
       (available with Python 3.4+ on some Unix systems);
     * CFS scheduler CPU bandwidth limit (available on Linux only, typically
       set by docker and similar container orchestration systems);
     * the value of the LOKY_MAX_CPU_COUNT environment variable if defined.
    and is given as the minimum of these constraints.

    If ``only_physical_cores`` is True, return the number of physical cores
    instead of the number of logical cores (hyperthreading / SMT). Note that
    this option is not enforced if the number of usable cores is controlled in
    any other way such as: process affinity, restricting CFS scheduler policy
    or the LOKY_MAX_CPU_COUNT environment variable. If the number of physical
    cores is not found, return the number of logical cores.
 
    It is also always larger or equal to 1.
    """
    # TODO: use os.cpu_count when dropping python 2 support
    try:
        cpu_count_mp = mp.cpu_count()
    except NotImplementedError:
        cpu_count_mp = 1

    cpu_count_user = _cpu_count_user(cpu_count_mp)
    aggregate_cpu_count = min(cpu_count_mp, cpu_count_user)

    if only_physical_cores:
        cpu_count_physical, exception = _count_physical_cores()
        if cpu_count_user < cpu_count_mp:
            # Respect user setting
            cpu_count = max(cpu_count_user, 1)
        elif cpu_count_physical == "not found":
            # Fallback to default behavior
            if exception is not None:
                # warns only the first time
                warnings.warn(
                    "Could not find the number of physical cores for the "
                    "following reason:\n" + str(exception) + "\n"
                    "Returning the number of logical cores instead. You can "
                    "silence this warning by setting LOKY_MAX_CPU_COUNT to "
                    "the number of cores you want to use.")
                if sys.version_info >= (3, 5):
                    # TODO remove the version check when dropping py2 support
                    traceback.print_tb(exception.__traceback__)

            cpu_count = max(aggregate_cpu_count, 1)
        else:
            return cpu_count_physical
    else:
        cpu_count = max(aggregate_cpu_count, 1)

    return cpu_count


def _cpu_count_user(cpu_count_mp):
    """Number of user defined available CPUs"""
    import math

    # Number of available CPUs given affinity settings
    cpu_count_affinity = cpu_count_mp
    if hasattr(os, 'sched_getaffinity'):
        try:
            cpu_count_affinity = len(os.sched_getaffinity(0))
        except NotImplementedError:
            pass

    # CFS scheduler CPU bandwidth limit
    # available in Linux since 2.6 kernel
    cpu_count_cfs = cpu_count_mp
    cfs_quota_fname = "/sys/fs/cgroup/cpu/cpu.cfs_quota_us"
    cfs_period_fname = "/sys/fs/cgroup/cpu/cpu.cfs_period_us"
    if os.path.exists(cfs_quota_fname) and os.path.exists(cfs_period_fname):
        with open(cfs_quota_fname, 'r') as fh:
            cfs_quota_us = int(fh.read())
        with open(cfs_period_fname, 'r') as fh:
            cfs_period_us = int(fh.read())

        if cfs_quota_us > 0 and cfs_period_us > 0:
            # Make sure this quantity is an int as math.ceil returns a
            # float in python2.7. (See issue #165)
            cpu_count_cfs = int(math.ceil(cfs_quota_us / cfs_period_us))

    # User defined soft-limit passed as a loky specific environment variable.
    cpu_count_loky = int(os.environ.get('LOKY_MAX_CPU_COUNT', cpu_count_mp))

    return min(cpu_count_affinity, cpu_count_cfs, cpu_count_loky)


def _count_physical_cores():
    """Return a tuple (number of physical cores, exception)

    If the number of physical cores is found, exception is set to None.
    If it has not been found, return ("not found", exception).

    The number of physical cores is cached to avoid repeating subprocess calls.
    """
    exception = None

    # First check if the value is cached
    global physical_cores_cache
    if physical_cores_cache is not None:
        return physical_cores_cache, exception

    # Not cached yet, find it
    try:
        if sys.platform == "linux":
            cpu_info = subprocess.run(
                "lscpu --parse=core".split(" "), capture_output=True)
            cpu_info = cpu_info.stdout.decode("utf-8").splitlines()
            cpu_info = {line for line in cpu_info if not line.startswith("#")}
            cpu_count_physical = len(cpu_info)
        elif sys.platform == "win32":
            cpu_info = subprocess.run(
                "wmic CPU Get NumberOfCores /Format:csv".split(" "),
                capture_output=True)
            cpu_info = cpu_info.stdout.decode('utf-8').splitlines()
            cpu_info = [l.split(",")[1] for l in cpu_info
                        if (l and l != "Node,NumberOfCores")]
            cpu_count_physical = sum(map(int, cpu_info))
        elif sys.platform == "darwin":
            cpu_info = subprocess.run(
                "sysctl -n hw.physicalcpu".split(" "), capture_output=True)
            cpu_info = cpu_info.stdout.decode('utf-8')
            cpu_count_physical = int(cpu_info)
        else:
            raise NotImplementedError(
                "unsupported platform: {}".format(sys.platform))

        # if cpu_count_physical < 1, we did not find a valid value
        if cpu_count_physical < 1:
            raise ValueError(
                "found {} physical cores < 1".format(cpu_count_physical))
        
    except Exception as e:
        exception = e
        cpu_count_physical = "not found"

    # Put the result in cache
    physical_cores_cache = cpu_count_physical
    
    return cpu_count_physical, exception


class LokyContext(BaseContext):
    """Context relying on the LokyProcess."""
    _name = 'loky'
    Process = LokyProcess
    cpu_count = staticmethod(cpu_count)

    def Queue(self, maxsize=0, reducers=None):
        '''Returns a queue object'''
        from .queues import Queue
        return Queue(maxsize, reducers=reducers,
                     ctx=self.get_context())

    def SimpleQueue(self, reducers=None):
        '''Returns a queue object'''
        from .queues import SimpleQueue
        return SimpleQueue(reducers=reducers, ctx=self.get_context())

    if sys.version_info[:2] < (3, 4):
        """Compat for python2.7/3.3 for necessary methods in Context"""
        def get_context(self):
            return self

        def get_start_method(self):
            return self._name

        def Pipe(self, duplex=True):
            '''Returns two connection object connected by a pipe'''
            return mp.Pipe(duplex)

        if sys.platform != "win32":
            """Use the compat Manager for python2.7/3.3 on UNIX to avoid
            relying on fork processes
            """
            def Manager(self):
                """Returns a manager object"""
                from .managers import LokyManager
                m = LokyManager()
                m.start()
                return m
        else:
            """Compat for context on Windows and python2.7/3.3. Using regular
            multiprocessing objects as it does not rely on fork.
            """
            from multiprocessing import synchronize
            Semaphore = staticmethod(synchronize.Semaphore)
            BoundedSemaphore = staticmethod(synchronize.BoundedSemaphore)
            Lock = staticmethod(synchronize.Lock)
            RLock = staticmethod(synchronize.RLock)
            Condition = staticmethod(synchronize.Condition)
            Event = staticmethod(synchronize.Event)
            Manager = staticmethod(mp.Manager)

    if sys.platform != "win32":
        """For Unix platform, use our custom implementation of synchronize
        relying on ctypes to interface with pthread semaphores.
        """
        def Semaphore(self, value=1):
            """Returns a semaphore object"""
            from .synchronize import Semaphore
            return Semaphore(value=value)

        def BoundedSemaphore(self, value):
            """Returns a bounded semaphore object"""
            from .synchronize import BoundedSemaphore
            return BoundedSemaphore(value)

        def Lock(self):
            """Returns a lock object"""
            from .synchronize import Lock
            return Lock()

        def RLock(self):
            """Returns a recurrent lock object"""
            from .synchronize import RLock
            return RLock()

        def Condition(self, lock=None):
            """Returns a condition object"""
            from .synchronize import Condition
            return Condition(lock)

        def Event(self):
            """Returns an event object"""
            from .synchronize import Event
            return Event()


class LokyInitMainContext(LokyContext):
    """Extra context with LokyProcess, which does load the main module

    This context is used for compatibility in the case ``cloudpickle`` is not
    present on the running system. This permits to load functions defined in
    the ``main`` module, using proper safeguards. The declaration of the
    ``executor`` should be protected by ``if __name__ == "__main__":`` and the
    functions and variable used from main should be out of this block.

    This mimics the default behavior of multiprocessing under Windows and the
    behavior of the ``spawn`` start method on a posix system for python3.4+.
    For more details, see the end of the following section of python doc
    https://docs.python.org/3/library/multiprocessing.html#multiprocessing-programming
    """
    _name = 'loky_init_main'
    Process = LokyInitMainProcess


if sys.version_info > (3, 4):
    """Register loky context so it works with multiprocessing.get_context"""
    ctx_loky = LokyContext()
    mp.context._concrete_contexts['loky'] = ctx_loky
    mp.context._concrete_contexts['loky_init_main'] = LokyInitMainContext()
