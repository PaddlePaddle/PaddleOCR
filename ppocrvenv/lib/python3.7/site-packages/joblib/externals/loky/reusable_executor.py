###############################################################################
# Reusable ProcessPoolExecutor
#
# author: Thomas Moreau and Olivier Grisel
#
import time
import warnings
import threading
import multiprocessing as mp

from .process_executor import ProcessPoolExecutor, EXTRA_QUEUED_CALLS
from .backend.context import cpu_count
from .backend import get_context

__all__ = ['get_reusable_executor']

# Python 2 compat helper
STRING_TYPE = type("")

# Singleton executor and id management
_executor_lock = threading.RLock()
_next_executor_id = 0
_executor = None
_executor_kwargs = None


def _get_next_executor_id():
    """Ensure that each successive executor instance has a unique, monotonic id.

    The purpose of this monotonic id is to help debug and test automated
    instance creation.
    """
    global _next_executor_id
    with _executor_lock:
        executor_id = _next_executor_id
        _next_executor_id += 1
        return executor_id


def get_reusable_executor(max_workers=None, context=None, timeout=10,
                          kill_workers=False, reuse="auto",
                          job_reducers=None, result_reducers=None,
                          initializer=None, initargs=(), env=None):
    """Return the current ReusableExectutor instance.

    Start a new instance if it has not been started already or if the previous
    instance was left in a broken state.

    If the previous instance does not have the requested number of workers, the
    executor is dynamically resized to adjust the number of workers prior to
    returning.

    Reusing a singleton instance spares the overhead of starting new worker
    processes and importing common python packages each time.

    ``max_workers`` controls the maximum number of tasks that can be running in
    parallel in worker processes. By default this is set to the number of
    CPUs on the host.

    Setting ``timeout`` (in seconds) makes idle workers automatically shutdown
    so as to release system resources. New workers are respawn upon submission
    of new tasks so that ``max_workers`` are available to accept the newly
    submitted tasks. Setting ``timeout`` to around 100 times the time required
    to spawn new processes and import packages in them (on the order of 100ms)
    ensures that the overhead of spawning workers is negligible.

    Setting ``kill_workers=True`` makes it possible to forcibly interrupt
    previously spawned jobs to get a new instance of the reusable executor
    with new constructor argument values.

    The ``job_reducers`` and ``result_reducers`` are used to customize the
    pickling of tasks and results send to the executor.

    When provided, the ``initializer`` is run first in newly spawned
    processes with argument ``initargs``.

    The environment variable in the child process are a copy of the values in
    the main process. One can provide a dict ``{ENV: VAL}`` where ``ENV`` and
    ``VAR`` are string literals to overwrite the environment variable ``ENV``
    in the child processes to value ``VAL``. The environment variables are set
    in the children before any module is loaded. This only works with with the
    ``loky`` context and it is unreliable on Windows with Python < 3.6.
    """
    _executor, _ = _ReusablePoolExecutor.get_reusable_executor(
        max_workers=max_workers, context=context, timeout=timeout,
        kill_workers=kill_workers, reuse=reuse, job_reducers=job_reducers,
        result_reducers=result_reducers, initializer=initializer,
        initargs=initargs, env=env
    )
    return _executor


class _ReusablePoolExecutor(ProcessPoolExecutor):
    def __init__(self, submit_resize_lock, max_workers=None, context=None,
                 timeout=None, executor_id=0, job_reducers=None,
                 result_reducers=None, initializer=None, initargs=(),
                 env=None):
        super(_ReusablePoolExecutor, self).__init__(
            max_workers=max_workers, context=context, timeout=timeout,
            job_reducers=job_reducers, result_reducers=result_reducers,
            initializer=initializer, initargs=initargs, env=env)
        self.executor_id = executor_id
        self._submit_resize_lock = submit_resize_lock

    @classmethod
    def get_reusable_executor(cls, max_workers=None, context=None, timeout=10,
                              kill_workers=False, reuse="auto",
                              job_reducers=None, result_reducers=None,
                              initializer=None, initargs=(), env=None):
        with _executor_lock:
            global _executor, _executor_kwargs
            executor = _executor

            if max_workers is None:
                if reuse is True and executor is not None:
                    max_workers = executor._max_workers
                else:
                    max_workers = cpu_count()
            elif max_workers <= 0:
                raise ValueError(
                    "max_workers must be greater than 0, got {}."
                    .format(max_workers))

            if isinstance(context, STRING_TYPE):
                context = get_context(context)
            if context is not None and context.get_start_method() == "fork":
                raise ValueError(
                    "Cannot use reusable executor with the 'fork' context"
                )

            kwargs = dict(context=context, timeout=timeout,
                          job_reducers=job_reducers,
                          result_reducers=result_reducers,
                          initializer=initializer, initargs=initargs,
                          env=env)
            if executor is None:
                is_reused = False
                mp.util.debug("Create a executor with max_workers={}."
                              .format(max_workers))
                executor_id = _get_next_executor_id()
                _executor_kwargs = kwargs
                _executor = executor = cls(
                    _executor_lock, max_workers=max_workers,
                    executor_id=executor_id, **kwargs)
            else:
                if reuse == 'auto':
                    reuse = kwargs == _executor_kwargs
                if (executor._flags.broken or executor._flags.shutdown
                        or not reuse):
                    if executor._flags.broken:
                        reason = "broken"
                    elif executor._flags.shutdown:
                        reason = "shutdown"
                    else:
                        reason = "arguments have changed"
                    mp.util.debug(
                        "Creating a new executor with max_workers={} as the "
                        "previous instance cannot be reused ({})."
                        .format(max_workers, reason))
                    executor.shutdown(wait=True, kill_workers=kill_workers)
                    _executor = executor = _executor_kwargs = None
                    # Recursive call to build a new instance
                    return cls.get_reusable_executor(max_workers=max_workers,
                                                     **kwargs)
                else:
                    mp.util.debug(
                        "Reusing existing executor with max_workers={}."
                        .format(executor._max_workers)
                    )
                    is_reused = True
                    executor._resize(max_workers)

        return executor, is_reused

    def submit(self, fn, *args, **kwargs):
        with self._submit_resize_lock:
            return super(_ReusablePoolExecutor, self).submit(
                fn, *args, **kwargs)

    def _resize(self, max_workers):
        with self._submit_resize_lock:
            if max_workers is None:
                raise ValueError("Trying to resize with max_workers=None")
            elif max_workers == self._max_workers:
                return

            if self._executor_manager_thread is None:
                # If the executor_manager_thread has not been started
                # then no processes have been spawned and we can just
                # update _max_workers and return
                self._max_workers = max_workers
                return

            self._wait_job_completion()

            # Some process might have returned due to timeout so check how many
            # children are still alive. Use the _process_management_lock to
            # ensure that no process are spawned or timeout during the resize.
            with self._processes_management_lock:
                processes = list(self._processes.values())
                nb_children_alive = sum(p.is_alive() for p in processes)
                self._max_workers = max_workers
                for _ in range(max_workers, nb_children_alive):
                    self._call_queue.put(None)
            while (len(self._processes) > max_workers
                   and not self._flags.broken):
                time.sleep(1e-3)

            self._adjust_process_count()
            processes = list(self._processes.values())
            while not all([p.is_alive() for p in processes]):
                time.sleep(1e-3)

    def _wait_job_completion(self):
        """Wait for the cache to be empty before resizing the pool."""
        # Issue a warning to the user about the bad effect of this usage.
        if len(self._pending_work_items) > 0:
            warnings.warn("Trying to resize an executor with running jobs: "
                          "waiting for jobs completion before resizing.",
                          UserWarning)
            mp.util.debug("Executor {} waiting for jobs completion before"
                          " resizing".format(self.executor_id))
        # Wait for the completion of the jobs
        while len(self._pending_work_items) > 0:
            time.sleep(1e-3)

    def _setup_queues(self, job_reducers, result_reducers):
        # As this executor can be resized, use a large queue size to avoid
        # underestimating capacity and introducing overhead
        queue_size = 2 * cpu_count() + EXTRA_QUEUED_CALLS
        super(_ReusablePoolExecutor, self)._setup_queues(
            job_reducers, result_reducers, queue_size=queue_size)
