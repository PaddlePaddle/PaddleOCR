"""
Helpers for embarrassingly parallel code.
"""
# Author: Gael Varoquaux < gael dot varoquaux at normalesup dot org >
# Copyright: 2010, Gael Varoquaux
# License: BSD 3 clause

from __future__ import division

import os
import sys
from math import sqrt
import functools
import time
import threading
import itertools
from uuid import uuid4
from numbers import Integral
import warnings
import queue

from ._multiprocessing_helpers import mp

from .logger import Logger, short_format_time
from .disk import memstr_to_bytes
from ._parallel_backends import (FallbackToBackend, MultiprocessingBackend,
                                 ThreadingBackend, SequentialBackend,
                                 LokyBackend)
from .externals.cloudpickle import dumps, loads
from .externals import loky

# Make sure that those two classes are part of the public joblib.parallel API
# so that 3rd party backend implementers can import them from here.
from ._parallel_backends import AutoBatchingMixin  # noqa
from ._parallel_backends import ParallelBackendBase  # noqa


BACKENDS = {
    'multiprocessing': MultiprocessingBackend,
    'threading': ThreadingBackend,
    'sequential': SequentialBackend,
    'loky': LokyBackend,
}
# name of the backend used by default by Parallel outside of any context
# managed by ``parallel_backend``.
DEFAULT_BACKEND = 'loky'
DEFAULT_N_JOBS = 1
DEFAULT_THREAD_BACKEND = 'threading'

# Thread local value that can be overridden by the ``parallel_backend`` context
# manager
_backend = threading.local()

VALID_BACKEND_HINTS = ('processes', 'threads', None)
VALID_BACKEND_CONSTRAINTS = ('sharedmem', None)


def _register_dask():
    """ Register Dask Backend if called with parallel_backend("dask") """
    try:
        from ._dask import DaskDistributedBackend
        register_parallel_backend('dask', DaskDistributedBackend)
    except ImportError as e:
        msg = ("To use the dask.distributed backend you must install both "
               "the `dask` and distributed modules.\n\n"
               "See https://dask.pydata.org/en/latest/install.html for more "
               "information.")
        raise ImportError(msg) from e


EXTERNAL_BACKENDS = {
    'dask': _register_dask,
}


def get_active_backend(prefer=None, require=None, verbose=0):
    """Return the active default backend"""
    if prefer not in VALID_BACKEND_HINTS:
        raise ValueError("prefer=%r is not a valid backend hint, "
                         "expected one of %r" % (prefer, VALID_BACKEND_HINTS))
    if require not in VALID_BACKEND_CONSTRAINTS:
        raise ValueError("require=%r is not a valid backend constraint, "
                         "expected one of %r"
                         % (require, VALID_BACKEND_CONSTRAINTS))

    if prefer == 'processes' and require == 'sharedmem':
        raise ValueError("prefer == 'processes' and require == 'sharedmem'"
                         " are inconsistent settings")
    backend_and_jobs = getattr(_backend, 'backend_and_jobs', None)
    if backend_and_jobs is not None:
        # Try to use the backend set by the user with the context manager.
        backend, n_jobs = backend_and_jobs
        nesting_level = backend.nesting_level
        supports_sharedmem = getattr(backend, 'supports_sharedmem', False)
        if require == 'sharedmem' and not supports_sharedmem:
            # This backend does not match the shared memory constraint:
            # fallback to the default thead-based backend.
            sharedmem_backend = BACKENDS[DEFAULT_THREAD_BACKEND](
                nesting_level=nesting_level)
            if verbose >= 10:
                print("Using %s as joblib.Parallel backend instead of %s "
                      "as the latter does not provide shared memory semantics."
                      % (sharedmem_backend.__class__.__name__,
                         backend.__class__.__name__))
            return sharedmem_backend, DEFAULT_N_JOBS
        else:
            return backend_and_jobs

    # We are outside of the scope of any parallel_backend context manager,
    # create the default backend instance now.
    backend = BACKENDS[DEFAULT_BACKEND](nesting_level=0)
    supports_sharedmem = getattr(backend, 'supports_sharedmem', False)
    uses_threads = getattr(backend, 'uses_threads', False)
    if ((require == 'sharedmem' and not supports_sharedmem) or
            (prefer == 'threads' and not uses_threads)):
        # Make sure the selected default backend match the soft hints and
        # hard constraints:
        backend = BACKENDS[DEFAULT_THREAD_BACKEND](nesting_level=0)
    return backend, DEFAULT_N_JOBS


class parallel_backend(object):
    """Change the default backend used by Parallel inside a with block.

    If ``backend`` is a string it must match a previously registered
    implementation using the ``register_parallel_backend`` function.

    By default the following backends are available:

    - 'loky': single-host, process-based parallelism (used by default),
    - 'threading': single-host, thread-based parallelism,
    - 'multiprocessing': legacy single-host, process-based parallelism.

    'loky' is recommended to run functions that manipulate Python objects.
    'threading' is a low-overhead alternative that is most efficient for
    functions that release the Global Interpreter Lock: e.g. I/O-bound code or
    CPU-bound code in a few calls to native code that explicitly releases the
    GIL.

    In addition, if the `dask` and `distributed` Python packages are installed,
    it is possible to use the 'dask' backend for better scheduling of nested
    parallel calls without over-subscription and potentially distribute
    parallel calls over a networked cluster of several hosts.

    It is also possible to use the distributed 'ray' backend for distributing
    the workload to a cluster of nodes. To use the 'ray' joblib backend add
    the following lines::

     >>> from ray.util.joblib import register_ray  # doctest: +SKIP
     >>> register_ray()  # doctest: +SKIP
     >>> with parallel_backend("ray"):  # doctest: +SKIP
     ...     print(Parallel()(delayed(neg)(i + 1) for i in range(5)))
     [-1, -2, -3, -4, -5]

    Alternatively the backend can be passed directly as an instance.

    By default all available workers will be used (``n_jobs=-1``) unless the
    caller passes an explicit value for the ``n_jobs`` parameter.

    This is an alternative to passing a ``backend='backend_name'`` argument to
    the ``Parallel`` class constructor. It is particularly useful when calling
    into library code that uses joblib internally but does not expose the
    backend argument in its own API.

    >>> from operator import neg
    >>> with parallel_backend('threading'):
    ...     print(Parallel()(delayed(neg)(i + 1) for i in range(5)))
    ...
    [-1, -2, -3, -4, -5]

    Warning: this function is experimental and subject to change in a future
    version of joblib.

    Joblib also tries to limit the oversubscription by limiting the number of
    threads usable in some third-party library threadpools like OpenBLAS, MKL
    or OpenMP. The default limit in each worker is set to
    ``max(cpu_count() // effective_n_jobs, 1)`` but this limit can be
    overwritten with the ``inner_max_num_threads`` argument which will be used
    to set this limit in the child processes.

    .. versionadded:: 0.10

    """
    def __init__(self, backend, n_jobs=-1, inner_max_num_threads=None,
                 **backend_params):
        if isinstance(backend, str):
            if backend not in BACKENDS and backend in EXTERNAL_BACKENDS:
                register = EXTERNAL_BACKENDS[backend]
                register()

            backend = BACKENDS[backend](**backend_params)

        if inner_max_num_threads is not None:
            msg = ("{} does not accept setting the inner_max_num_threads "
                   "argument.".format(backend.__class__.__name__))
            assert backend.supports_inner_max_num_threads, msg
            backend.inner_max_num_threads = inner_max_num_threads

        # If the nesting_level of the backend is not set previously, use the
        # nesting level from the previous active_backend to set it
        current_backend_and_jobs = getattr(_backend, 'backend_and_jobs', None)
        if backend.nesting_level is None:
            if current_backend_and_jobs is None:
                nesting_level = 0
            else:
                nesting_level = current_backend_and_jobs[0].nesting_level

            backend.nesting_level = nesting_level

        # Save the backends info and set the active backend
        self.old_backend_and_jobs = current_backend_and_jobs
        self.new_backend_and_jobs = (backend, n_jobs)

        _backend.backend_and_jobs = (backend, n_jobs)

    def __enter__(self):
        return self.new_backend_and_jobs

    def __exit__(self, type, value, traceback):
        self.unregister()

    def unregister(self):
        if self.old_backend_and_jobs is None:
            if getattr(_backend, 'backend_and_jobs', None) is not None:
                del _backend.backend_and_jobs
        else:
            _backend.backend_and_jobs = self.old_backend_and_jobs


# Under Linux or OS X the default start method of multiprocessing
# can cause third party libraries to crash. Under Python 3.4+ it is possible
# to set an environment variable to switch the default start method from
# 'fork' to 'forkserver' or 'spawn' to avoid this issue albeit at the cost
# of causing semantic changes and some additional pool instantiation overhead.
DEFAULT_MP_CONTEXT = None
if hasattr(mp, 'get_context'):
    method = os.environ.get('JOBLIB_START_METHOD', '').strip() or None
    if method is not None:
        DEFAULT_MP_CONTEXT = mp.get_context(method=method)


class BatchedCalls(object):
    """Wrap a sequence of (func, args, kwargs) tuples as a single callable"""

    def __init__(self, iterator_slice, backend_and_jobs, reducer_callback=None,
                 pickle_cache=None):
        self.items = list(iterator_slice)
        self._size = len(self.items)
        self._reducer_callback = reducer_callback
        if isinstance(backend_and_jobs, tuple):
            self._backend, self._n_jobs = backend_and_jobs
        else:
            # this is for backward compatibility purposes. Before 0.12.6,
            # nested backends were returned without n_jobs indications.
            self._backend, self._n_jobs = backend_and_jobs, None
        self._pickle_cache = pickle_cache if pickle_cache is not None else {}

    def __call__(self):
        # Set the default nested backend to self._backend but do not set the
        # change the default number of processes to -1
        with parallel_backend(self._backend, n_jobs=self._n_jobs):
            return [func(*args, **kwargs)
                    for func, args, kwargs in self.items]

    def __reduce__(self):
        if self._reducer_callback is not None:
            self._reducer_callback()
        # no need pickle the callback.
        return (
            BatchedCalls,
            (self.items, (self._backend, self._n_jobs), None,
             self._pickle_cache)
        )

    def __len__(self):
        return self._size


###############################################################################
# CPU count that works also when multiprocessing has been disabled via
# the JOBLIB_MULTIPROCESSING environment variable
def cpu_count(only_physical_cores=False):
    """Return the number of CPUs.

    This delegates to loky.cpu_count that takes into account additional
    constraints such as Linux CFS scheduler quotas (typically set by container
    runtimes such as docker) and CPU affinity (for instance using the taskset
    command on Linux).

    If only_physical_cores is True, do not take hyperthreading / SMT logical
    cores into account.
    """
    if mp is None:
        return 1

    return loky.cpu_count(only_physical_cores=only_physical_cores)


###############################################################################
# For verbosity

def _verbosity_filter(index, verbose):
    """ Returns False for indices increasingly apart, the distance
        depending on the value of verbose.

        We use a lag increasing as the square of index
    """
    if not verbose:
        return True
    elif verbose > 10:
        return False
    if index == 0:
        return False
    verbose = .5 * (11 - verbose) ** 2
    scale = sqrt(index / verbose)
    next_scale = sqrt((index + 1) / verbose)
    return (int(next_scale) == int(scale))


###############################################################################
def delayed(function):
    """Decorator used to capture the arguments of a function."""

    def delayed_function(*args, **kwargs):
        return function, args, kwargs
    try:
        delayed_function = functools.wraps(function)(delayed_function)
    except AttributeError:
        " functools.wraps fails on some callable objects "
    return delayed_function


###############################################################################
class BatchCompletionCallBack(object):
    """Callback used by joblib.Parallel's multiprocessing backend.

    This callable is executed by the parent process whenever a worker process
    has returned the results of a batch of tasks.

    It is used for progress reporting, to update estimate of the batch
    processing duration and to schedule the next batch of tasks to be
    processed.

    """
    def __init__(self, dispatch_timestamp, batch_size, parallel):
        self.dispatch_timestamp = dispatch_timestamp
        self.batch_size = batch_size
        self.parallel = parallel

    def __call__(self, out):
        self.parallel.n_completed_tasks += self.batch_size
        this_batch_duration = time.time() - self.dispatch_timestamp

        self.parallel._backend.batch_completed(self.batch_size,
                                               this_batch_duration)
        self.parallel.print_progress()
        with self.parallel._lock:
            if self.parallel._original_iterator is not None:
                self.parallel.dispatch_next()


###############################################################################
def register_parallel_backend(name, factory, make_default=False):
    """Register a new Parallel backend factory.

    The new backend can then be selected by passing its name as the backend
    argument to the Parallel class. Moreover, the default backend can be
    overwritten globally by setting make_default=True.

    The factory can be any callable that takes no argument and return an
    instance of ``ParallelBackendBase``.

    Warning: this function is experimental and subject to change in a future
    version of joblib.

    .. versionadded:: 0.10

    """
    BACKENDS[name] = factory
    if make_default:
        global DEFAULT_BACKEND
        DEFAULT_BACKEND = name


def effective_n_jobs(n_jobs=-1):
    """Determine the number of jobs that can actually run in parallel

    n_jobs is the number of workers requested by the callers. Passing n_jobs=-1
    means requesting all available workers for instance matching the number of
    CPU cores on the worker host(s).

    This method should return a guesstimate of the number of workers that can
    actually perform work concurrently with the currently enabled default
    backend. The primary use case is to make it possible for the caller to know
    in how many chunks to slice the work.

    In general working on larger data chunks is more efficient (less scheduling
    overhead and better use of CPU cache prefetching heuristics) as long as all
    the workers have enough work to do.

    Warning: this function is experimental and subject to change in a future
    version of joblib.

    .. versionadded:: 0.10

    """
    backend, backend_n_jobs = get_active_backend()
    if n_jobs is None:
        n_jobs = backend_n_jobs
    return backend.effective_n_jobs(n_jobs=n_jobs)


###############################################################################
class Parallel(Logger):
    ''' Helper class for readable parallel mapping.

        Read more in the :ref:`User Guide <parallel>`.

        Parameters
        -----------
        n_jobs: int, default: None
            The maximum number of concurrently running jobs, such as the number
            of Python worker processes when backend="multiprocessing"
            or the size of the thread-pool when backend="threading".
            If -1 all CPUs are used. If 1 is given, no parallel computing code
            is used at all, which is useful for debugging. For n_jobs below -1,
            (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all
            CPUs but one are used.
            None is a marker for 'unset' that will be interpreted as n_jobs=1
            (sequential execution) unless the call is performed under a
            parallel_backend context manager that sets another value for
            n_jobs.
        backend: str, ParallelBackendBase instance or None, default: 'loky'
            Specify the parallelization backend implementation.
            Supported backends are:

            - "loky" used by default, can induce some
              communication and memory overhead when exchanging input and
              output data with the worker Python processes.
            - "multiprocessing" previous process-based backend based on
              `multiprocessing.Pool`. Less robust than `loky`.
            - "threading" is a very low-overhead backend but it suffers
              from the Python Global Interpreter Lock if the called function
              relies a lot on Python objects. "threading" is mostly useful
              when the execution bottleneck is a compiled extension that
              explicitly releases the GIL (for instance a Cython loop wrapped
              in a "with nogil" block or an expensive call to a library such
              as NumPy).
            - finally, you can register backends by calling
              register_parallel_backend. This will allow you to implement
              a backend of your liking.

            It is not recommended to hard-code the backend name in a call to
            Parallel in a library. Instead it is recommended to set soft hints
            (prefer) or hard constraints (require) so as to make it possible
            for library users to change the backend from the outside using the
            parallel_backend context manager.
        prefer: str in {'processes', 'threads'} or None, default: None
            Soft hint to choose the default backend if no specific backend
            was selected with the parallel_backend context manager. The
            default process-based backend is 'loky' and the default
            thread-based backend is 'threading'. Ignored if the ``backend``
            parameter is specified.
        require: 'sharedmem' or None, default None
            Hard constraint to select the backend. If set to 'sharedmem',
            the selected backend will be single-host and thread-based even
            if the user asked for a non-thread based backend with
            parallel_backend.
        verbose: int, optional
            The verbosity level: if non zero, progress messages are
            printed. Above 50, the output is sent to stdout.
            The frequency of the messages increases with the verbosity level.
            If it more than 10, all iterations are reported.
        timeout: float, optional
            Timeout limit for each task to complete.  If any task takes longer
            a TimeOutError will be raised. Only applied when n_jobs != 1
        pre_dispatch: {'all', integer, or expression, as in '3*n_jobs'}
            The number of batches (of tasks) to be pre-dispatched.
            Default is '2*n_jobs'. When batch_size="auto" this is reasonable
            default and the workers should never starve.
        batch_size: int or 'auto', default: 'auto'
            The number of atomic tasks to dispatch at once to each
            worker. When individual evaluations are very fast, dispatching
            calls to workers can be slower than sequential computation because
            of the overhead. Batching fast computations together can mitigate
            this.
            The ``'auto'`` strategy keeps track of the time it takes for a batch
            to complete, and dynamically adjusts the batch size to keep the time
            on the order of half a second, using a heuristic. The initial batch
            size is 1.
            ``batch_size="auto"`` with ``backend="threading"`` will dispatch
            batches of a single task at a time as the threading backend has
            very little overhead and using larger batch size has not proved to
            bring any gain in that case.
        temp_folder: str, optional
            Folder to be used by the pool for memmapping large arrays
            for sharing memory with worker processes. If None, this will try in
            order:

            - a folder pointed by the JOBLIB_TEMP_FOLDER environment
              variable,
            - /dev/shm if the folder exists and is writable: this is a
              RAM disk filesystem available by default on modern Linux
              distributions,
            - the default system temporary folder that can be
              overridden with TMP, TMPDIR or TEMP environment
              variables, typically /tmp under Unix operating systems.

            Only active when backend="loky" or "multiprocessing".
        max_nbytes int, str, or None, optional, 1M by default
            Threshold on the size of arrays passed to the workers that
            triggers automated memory mapping in temp_folder. Can be an int
            in Bytes, or a human-readable string, e.g., '1M' for 1 megabyte.
            Use None to disable memmapping of large arrays.
            Only active when backend="loky" or "multiprocessing".
        mmap_mode: {None, 'r+', 'r', 'w+', 'c'}, default: 'r'
            Memmapping mode for numpy arrays passed to workers. None will 
            disable memmapping, other modes defined in the numpy.memmap doc:
            https://numpy.org/doc/stable/reference/generated/numpy.memmap.html
            Also, see 'max_nbytes' parameter documentation for more details.

        Notes
        -----

        This object uses workers to compute in parallel the application of a
        function to many different arguments. The main functionality it brings
        in addition to using the raw multiprocessing or concurrent.futures API
        are (see examples for details):

        * More readable code, in particular since it avoids
          constructing list of arguments.

        * Easier debugging:
            - informative tracebacks even when the error happens on
              the client side
            - using 'n_jobs=1' enables to turn off parallel computing
              for debugging without changing the codepath
            - early capture of pickling errors

        * An optional progress meter.

        * Interruption of multiprocesses jobs with 'Ctrl-C'

        * Flexible pickling control for the communication to and from
          the worker processes.

        * Ability to use shared memory efficiently with worker
          processes for large numpy-based datastructures.

        Examples
        --------

        A simple example:

        >>> from math import sqrt
        >>> from joblib import Parallel, delayed
        >>> Parallel(n_jobs=1)(delayed(sqrt)(i**2) for i in range(10))
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]

        Reshaping the output when the function has several return
        values:

        >>> from math import modf
        >>> from joblib import Parallel, delayed
        >>> r = Parallel(n_jobs=1)(delayed(modf)(i/2.) for i in range(10))
        >>> res, i = zip(*r)
        >>> res
        (0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5)
        >>> i
        (0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0)

        The progress meter: the higher the value of `verbose`, the more
        messages:

        >>> from time import sleep
        >>> from joblib import Parallel, delayed
        >>> r = Parallel(n_jobs=2, verbose=10)(delayed(sleep)(.2) for _ in range(10)) #doctest: +SKIP
        [Parallel(n_jobs=2)]: Done   1 tasks      | elapsed:    0.6s
        [Parallel(n_jobs=2)]: Done   4 tasks      | elapsed:    0.8s
        [Parallel(n_jobs=2)]: Done  10 out of  10 | elapsed:    1.4s finished

        Traceback example, note how the line of the error is indicated
        as well as the values of the parameter passed to the function that
        triggered the exception, even though the traceback happens in the
        child process:

        >>> from heapq import nlargest
        >>> from joblib import Parallel, delayed
        >>> Parallel(n_jobs=2)(delayed(nlargest)(2, n) for n in (range(4), 'abcde', 3)) #doctest: +SKIP
        #...
        ---------------------------------------------------------------------------
        Sub-process traceback:
        ---------------------------------------------------------------------------
        TypeError                                          Mon Nov 12 11:37:46 2012
        PID: 12934                                    Python 2.7.3: /usr/bin/python
        ...........................................................................
        /usr/lib/python2.7/heapq.pyc in nlargest(n=2, iterable=3, key=None)
            419         if n >= size:
            420             return sorted(iterable, key=key, reverse=True)[:n]
            421
            422     # When key is none, use simpler decoration
            423     if key is None:
        --> 424         it = izip(iterable, count(0,-1))                    # decorate
            425         result = _nlargest(n, it)
            426         return map(itemgetter(0), result)                   # undecorate
            427
            428     # General case, slowest method
         TypeError: izip argument #1 must support iteration
        ___________________________________________________________________________


        Using pre_dispatch in a producer/consumer situation, where the
        data is generated on the fly. Note how the producer is first
        called 3 times before the parallel loop is initiated, and then
        called to generate new data on the fly:

        >>> from math import sqrt
        >>> from joblib import Parallel, delayed
        >>> def producer():
        ...     for i in range(6):
        ...         print('Produced %s' % i)
        ...         yield i
        >>> out = Parallel(n_jobs=2, verbose=100, pre_dispatch='1.5*n_jobs')(
        ...                delayed(sqrt)(i) for i in producer()) #doctest: +SKIP
        Produced 0
        Produced 1
        Produced 2
        [Parallel(n_jobs=2)]: Done 1 jobs     | elapsed:  0.0s
        Produced 3
        [Parallel(n_jobs=2)]: Done 2 jobs     | elapsed:  0.0s
        Produced 4
        [Parallel(n_jobs=2)]: Done 3 jobs     | elapsed:  0.0s
        Produced 5
        [Parallel(n_jobs=2)]: Done 4 jobs     | elapsed:  0.0s
        [Parallel(n_jobs=2)]: Done 6 out of 6 | elapsed:  0.0s remaining: 0.0s
        [Parallel(n_jobs=2)]: Done 6 out of 6 | elapsed:  0.0s finished

    '''
    def __init__(self, n_jobs=None, backend=None, verbose=0, timeout=None,
                 pre_dispatch='2 * n_jobs', batch_size='auto',
                 temp_folder=None, max_nbytes='1M', mmap_mode='r',
                 prefer=None, require=None):
        active_backend, context_n_jobs = get_active_backend(
            prefer=prefer, require=require, verbose=verbose)
        nesting_level = active_backend.nesting_level
        if backend is None and n_jobs is None:
            # If we are under a parallel_backend context manager, look up
            # the default number of jobs and use that instead:
            n_jobs = context_n_jobs
        if n_jobs is None:
            # No specific context override and no specific value request:
            # default to 1.
            n_jobs = 1
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.timeout = timeout
        self.pre_dispatch = pre_dispatch
        self._ready_batches = queue.Queue()
        self._id = uuid4().hex
        self._reducer_callback = None

        if isinstance(max_nbytes, str):
            max_nbytes = memstr_to_bytes(max_nbytes)

        self._backend_args = dict(
            max_nbytes=max_nbytes,
            mmap_mode=mmap_mode,
            temp_folder=temp_folder,
            prefer=prefer,
            require=require,
            verbose=max(0, self.verbose - 50),
        )
        if DEFAULT_MP_CONTEXT is not None:
            self._backend_args['context'] = DEFAULT_MP_CONTEXT
        elif hasattr(mp, "get_context"):
            self._backend_args['context'] = mp.get_context()

        if backend is None:
            backend = active_backend

        elif isinstance(backend, ParallelBackendBase):
            # Use provided backend as is, with the current nesting_level if it
            # is not set yet.
            if backend.nesting_level is None:
                backend.nesting_level = nesting_level

        elif hasattr(backend, 'Pool') and hasattr(backend, 'Lock'):
            # Make it possible to pass a custom multiprocessing context as
            # backend to change the start method to forkserver or spawn or
            # preload modules on the forkserver helper process.
            self._backend_args['context'] = backend
            backend = MultiprocessingBackend(nesting_level=nesting_level)
        else:
            try:
                backend_factory = BACKENDS[backend]
            except KeyError as e:
                raise ValueError("Invalid backend: %s, expected one of %r"
                                 % (backend, sorted(BACKENDS.keys()))) from e
            backend = backend_factory(nesting_level=nesting_level)

        if (require == 'sharedmem' and
                not getattr(backend, 'supports_sharedmem', False)):
            raise ValueError("Backend %s does not support shared memory"
                             % backend)

        if (batch_size == 'auto' or isinstance(batch_size, Integral) and
                batch_size > 0):
            self.batch_size = batch_size
        else:
            raise ValueError(
                "batch_size must be 'auto' or a positive integer, got: %r"
                % batch_size)

        self._backend = backend
        self._output = None
        self._jobs = list()
        self._managed_backend = False

        # This lock is used coordinate the main thread of this process with
        # the async callback thread of our the pool.
        self._lock = threading.RLock()

    def __enter__(self):
        self._managed_backend = True
        self._initialize_backend()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._terminate_backend()
        self._managed_backend = False

    def _initialize_backend(self):
        """Build a process or thread pool and return the number of workers"""
        try:
            n_jobs = self._backend.configure(n_jobs=self.n_jobs, parallel=self,
                                             **self._backend_args)
            if self.timeout is not None and not self._backend.supports_timeout:
                warnings.warn(
                    'The backend class {!r} does not support timeout. '
                    "You have set 'timeout={}' in Parallel but "
                    "the 'timeout' parameter will not be used.".format(
                        self._backend.__class__.__name__,
                        self.timeout))

        except FallbackToBackend as e:
            # Recursively initialize the backend in case of requested fallback.
            self._backend = e.backend
            n_jobs = self._initialize_backend()

        return n_jobs

    def _effective_n_jobs(self):
        if self._backend:
            return self._backend.effective_n_jobs(self.n_jobs)
        return 1

    def _terminate_backend(self):
        if self._backend is not None:
            self._backend.terminate()

    def _dispatch(self, batch):
        """Queue the batch for computing, with or without multiprocessing

        WARNING: this method is not thread-safe: it should be only called
        indirectly via dispatch_one_batch.

        """
        # If job.get() catches an exception, it closes the queue:
        if self._aborting:
            return

        self.n_dispatched_tasks += len(batch)
        self.n_dispatched_batches += 1

        dispatch_timestamp = time.time()
        cb = BatchCompletionCallBack(dispatch_timestamp, len(batch), self)
        with self._lock:
            job_idx = len(self._jobs)
            job = self._backend.apply_async(batch, callback=cb)
            # A job can complete so quickly than its callback is
            # called before we get here, causing self._jobs to
            # grow. To ensure correct results ordering, .insert is
            # used (rather than .append) in the following line
            self._jobs.insert(job_idx, job)

    def dispatch_next(self):
        """Dispatch more data for parallel processing

        This method is meant to be called concurrently by the multiprocessing
        callback. We rely on the thread-safety of dispatch_one_batch to protect
        against concurrent consumption of the unprotected iterator.

        """
        if not self.dispatch_one_batch(self._original_iterator):
            self._iterating = False
            self._original_iterator = None

    def dispatch_one_batch(self, iterator):
        """Prefetch the tasks for the next batch and dispatch them.

        The effective size of the batch is computed here.
        If there are no more jobs to dispatch, return False, else return True.

        The iterator consumption and dispatching is protected by the same
        lock so calling this function should be thread safe.

        """
        if self.batch_size == 'auto':
            batch_size = self._backend.compute_batch_size()
        else:
            # Fixed batch size strategy
            batch_size = self.batch_size

        with self._lock:
            # to ensure an even distribution of the workolad between workers,
            # we look ahead in the original iterators more than batch_size
            # tasks - However, we keep consuming only one batch at each
            # dispatch_one_batch call. The extra tasks are stored in a local
            # queue, _ready_batches, that is looked-up prior to re-consuming
            # tasks from the origal iterator.
            try:
                tasks = self._ready_batches.get(block=False)
            except queue.Empty:
                # slice the iterator n_jobs * batchsize items at a time. If the
                # slice returns less than that, then the current batchsize puts
                # too much weight on a subset of workers, while other may end
                # up starving. So in this case, re-scale the batch size
                # accordingly to distribute evenly the last items between all
                # workers.
                n_jobs = self._cached_effective_n_jobs
                big_batch_size = batch_size * n_jobs

                islice = list(itertools.islice(iterator, big_batch_size))
                if len(islice) == 0:
                    return False
                elif (iterator is self._original_iterator
                      and len(islice) < big_batch_size):
                    # We reached the end of the original iterator (unless
                    # iterator is the ``pre_dispatch``-long initial slice of
                    # the original iterator) -- decrease the batch size to
                    # account for potential variance in the batches running
                    # time.
                    final_batch_size = max(1, len(islice) // (10 * n_jobs))
                else:
                    final_batch_size = max(1, len(islice) // n_jobs)

                # enqueue n_jobs batches in a local queue
                for i in range(0, len(islice), final_batch_size):
                    tasks = BatchedCalls(islice[i:i + final_batch_size],
                                         self._backend.get_nested_backend(),
                                         self._reducer_callback,
                                         self._pickle_cache)
                    self._ready_batches.put(tasks)

                # finally, get one task.
                tasks = self._ready_batches.get(block=False)
            if len(tasks) == 0:
                # No more tasks available in the iterator: tell caller to stop.
                return False
            else:
                self._dispatch(tasks)
                return True

    def _print(self, msg, msg_args):
        """Display the message on stout or stderr depending on verbosity"""
        # XXX: Not using the logger framework: need to
        # learn to use logger better.
        if not self.verbose:
            return
        if self.verbose < 50:
            writer = sys.stderr.write
        else:
            writer = sys.stdout.write
        msg = msg % msg_args
        writer('[%s]: %s\n' % (self, msg))

    def print_progress(self):
        """Display the process of the parallel execution only a fraction
           of time, controlled by self.verbose.
        """
        if not self.verbose:
            return
        elapsed_time = time.time() - self._start_time

        # Original job iterator becomes None once it has been fully
        # consumed : at this point we know the total number of jobs and we are
        # able to display an estimation of the remaining time based on already
        # completed jobs. Otherwise, we simply display the number of completed
        # tasks.
        if self._original_iterator is not None:
            if _verbosity_filter(self.n_dispatched_batches, self.verbose):
                return
            self._print('Done %3i tasks      | elapsed: %s',
                        (self.n_completed_tasks,
                         short_format_time(elapsed_time), ))
        else:
            index = self.n_completed_tasks
            # We are finished dispatching
            total_tasks = self.n_dispatched_tasks
            # We always display the first loop
            if not index == 0:
                # Display depending on the number of remaining items
                # A message as soon as we finish dispatching, cursor is 0
                cursor = (total_tasks - index + 1 -
                          self._pre_dispatch_amount)
                frequency = (total_tasks // self.verbose) + 1
                is_last_item = (index + 1 == total_tasks)
                if (is_last_item or cursor % frequency):
                    return
            remaining_time = (elapsed_time / index) * \
                             (self.n_dispatched_tasks - index * 1.0)
            # only display status if remaining time is greater or equal to 0
            self._print('Done %3i out of %3i | elapsed: %s remaining: %s',
                        (index,
                         total_tasks,
                         short_format_time(elapsed_time),
                         short_format_time(remaining_time),
                         ))

    def retrieve(self):
        self._output = list()
        while self._iterating or len(self._jobs) > 0:
            if len(self._jobs) == 0:
                # Wait for an async callback to dispatch new jobs
                time.sleep(0.01)
                continue
            # We need to be careful: the job list can be filling up as
            # we empty it and Python list are not thread-safe by default hence
            # the use of the lock
            with self._lock:
                job = self._jobs.pop(0)

            try:
                if getattr(self._backend, 'supports_timeout', False):
                    self._output.extend(job.get(timeout=self.timeout))
                else:
                    self._output.extend(job.get())

            except BaseException as exception:
                # Note: we catch any BaseException instead of just Exception
                # instances to also include KeyboardInterrupt.

                # Stop dispatching any new job in the async callback thread
                self._aborting = True

                # If the backend allows it, cancel or kill remaining running
                # tasks without waiting for the results as we will raise
                # the exception we got back to the caller instead of returning
                # any result.
                backend = self._backend
                if (backend is not None and
                        hasattr(backend, 'abort_everything')):
                    # If the backend is managed externally we need to make sure
                    # to leave it in a working state to allow for future jobs
                    # scheduling.
                    ensure_ready = self._managed_backend
                    backend.abort_everything(ensure_ready=ensure_ready)
                raise

    def __call__(self, iterable):
        if self._jobs:
            raise ValueError('This Parallel instance is already running')
        # A flag used to abort the dispatching of jobs in case an
        # exception is found
        self._aborting = False

        if not self._managed_backend:
            n_jobs = self._initialize_backend()
        else:
            n_jobs = self._effective_n_jobs()

        if isinstance(self._backend, LokyBackend):
            # For the loky backend, we add a callback executed when reducing
            # BatchCalls, that makes the loky executor use a temporary folder
            # specific to this Parallel object when pickling temporary memmaps.
            # This callback is necessary to ensure that several Parallel
            # objects using the same resuable executor don't use the same
            # temporary resources.

            def _batched_calls_reducer_callback():
                # Relevant implementation detail: the following lines, called
                # when reducing BatchedCalls, are called in a thread-safe
                # situation, meaning that the context of the temporary folder
                # manager will not be changed in between the callback execution
                # and the end of the BatchedCalls pickling. The reason is that
                # pickling (the only place where set_current_context is used)
                # is done from a single thread (the queue_feeder_thread).
                self._backend._workers._temp_folder_manager.set_current_context(  # noqa
                    self._id
                )
            self._reducer_callback = _batched_calls_reducer_callback

        # self._effective_n_jobs should be called in the Parallel.__call__
        # thread only -- store its value in an attribute for further queries.
        self._cached_effective_n_jobs = n_jobs

        backend_name = self._backend.__class__.__name__
        if n_jobs == 0:
            raise RuntimeError("%s has no active worker." % backend_name)

        self._print("Using backend %s with %d concurrent workers.",
                    (backend_name, n_jobs))
        if hasattr(self._backend, 'start_call'):
            self._backend.start_call()
        iterator = iter(iterable)
        pre_dispatch = self.pre_dispatch

        if pre_dispatch == 'all' or n_jobs == 1:
            # prevent further dispatch via multiprocessing callback thread
            self._original_iterator = None
            self._pre_dispatch_amount = 0
        else:
            self._original_iterator = iterator
            if hasattr(pre_dispatch, 'endswith'):
                pre_dispatch = eval(pre_dispatch)
            self._pre_dispatch_amount = pre_dispatch = int(pre_dispatch)

            # The main thread will consume the first pre_dispatch items and
            # the remaining items will later be lazily dispatched by async
            # callbacks upon task completions.

            # TODO: this iterator should be batch_size * n_jobs
            iterator = itertools.islice(iterator, self._pre_dispatch_amount)

        self._start_time = time.time()
        self.n_dispatched_batches = 0
        self.n_dispatched_tasks = 0
        self.n_completed_tasks = 0
        # Use a caching dict for callables that are pickled with cloudpickle to
        # improve performances. This cache is used only in the case of
        # functions that are defined in the __main__ module, functions that are
        # defined locally (inside another function) and lambda expressions.
        self._pickle_cache = dict()
        try:
            # Only set self._iterating to True if at least a batch
            # was dispatched. In particular this covers the edge
            # case of Parallel used with an exhausted iterator. If
            # self._original_iterator is None, then this means either
            # that pre_dispatch == "all", n_jobs == 1 or that the first batch
            # was very quick and its callback already dispatched all the
            # remaining jobs.
            self._iterating = False
            if self.dispatch_one_batch(iterator):
                self._iterating = self._original_iterator is not None

            while self.dispatch_one_batch(iterator):
                pass

            if pre_dispatch == "all" or n_jobs == 1:
                # The iterable was consumed all at once by the above for loop.
                # No need to wait for async callbacks to trigger to
                # consumption.
                self._iterating = False

            with self._backend.retrieval_context():
                self.retrieve()
            # Make sure that we get a last message telling us we are done
            elapsed_time = time.time() - self._start_time
            self._print('Done %3i out of %3i | elapsed: %s finished',
                        (len(self._output), len(self._output),
                         short_format_time(elapsed_time)))
        finally:
            if hasattr(self._backend, 'stop_call'):
                self._backend.stop_call()
            if not self._managed_backend:
                self._terminate_backend()
            self._jobs = list()
            self._pickle_cache = None
        output = self._output
        self._output = None
        return output

    def __repr__(self):
        return '%s(n_jobs=%s)' % (self.__class__.__name__, self.n_jobs)
