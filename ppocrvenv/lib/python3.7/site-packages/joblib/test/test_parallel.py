"""
Test the parallel module.
"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# Copyright (c) 2010-2011 Gael Varoquaux
# License: BSD Style, 3 clauses.

import os
import sys
import time
import mmap
import threading
from traceback import format_exception
from math import sqrt
from time import sleep
from pickle import PicklingError
from multiprocessing import TimeoutError
import pickle
import pytest

from importlib import reload

import joblib
from joblib import parallel
from joblib import dump, load
from joblib.externals.loky import get_reusable_executor

from joblib.test.common import np, with_numpy
from joblib.test.common import with_multiprocessing
from joblib.testing import (parametrize, raises, check_subprocess_call,
                            skipif, SkipTest, warns)

from joblib.externals.loky.process_executor import TerminatedWorkerError

from queue import Queue

try:
    import posix
except ImportError:
    posix = None

try:
    from ._openmp_test_helper.parallel_sum import parallel_sum
except ImportError:
    parallel_sum = None

try:
    import distributed
except ImportError:
    distributed = None

from joblib._parallel_backends import SequentialBackend
from joblib._parallel_backends import ThreadingBackend
from joblib._parallel_backends import MultiprocessingBackend
from joblib._parallel_backends import ParallelBackendBase
from joblib._parallel_backends import LokyBackend
from joblib._parallel_backends import SafeFunction

from joblib.parallel import Parallel, delayed
from joblib.parallel import register_parallel_backend, parallel_backend
from joblib.parallel import effective_n_jobs, cpu_count

from joblib.parallel import mp, BACKENDS, DEFAULT_BACKEND, EXTERNAL_BACKENDS
from joblib.my_exceptions import JoblibException
from joblib.my_exceptions import WorkerInterrupt


ALL_VALID_BACKENDS = [None] + sorted(BACKENDS.keys())
# Add instances of backend classes deriving from ParallelBackendBase
ALL_VALID_BACKENDS += [BACKENDS[backend_str]() for backend_str in BACKENDS]
PROCESS_BACKENDS = ['multiprocessing', 'loky']
PARALLEL_BACKENDS = PROCESS_BACKENDS + ['threading']

if hasattr(mp, 'get_context'):
    # Custom multiprocessing context in Python 3.4+
    ALL_VALID_BACKENDS.append(mp.get_context('spawn'))

DefaultBackend = BACKENDS[DEFAULT_BACKEND]


def get_workers(backend):
    return getattr(backend, '_pool', getattr(backend, '_workers', None))


def division(x, y):
    return x / y


def square(x):
    return x ** 2


class MyExceptionWithFinickyInit(Exception):
    """An exception class with non trivial __init__
    """
    def __init__(self, a, b, c, d):
        pass


def exception_raiser(x, custom_exception=False):
    if x == 7:
        raise (MyExceptionWithFinickyInit('a', 'b', 'c', 'd')
               if custom_exception else ValueError)
    return x


def interrupt_raiser(x):
    time.sleep(.05)
    raise KeyboardInterrupt


def f(x, y=0, z=0):
    """ A module-level function so that it can be spawn with
    multiprocessing.
    """
    return x ** 2 + y + z


def _active_backend_type():
    return type(parallel.get_active_backend()[0])


def parallel_func(inner_n_jobs, backend):
    return Parallel(n_jobs=inner_n_jobs, backend=backend)(
        delayed(square)(i) for i in range(3))


###############################################################################
def test_cpu_count():
    assert cpu_count() > 0


def test_effective_n_jobs():
    assert effective_n_jobs() > 0


@pytest.mark.parametrize(
    "backend_n_jobs, expected_n_jobs",
    [(3, 3), (-1, effective_n_jobs(n_jobs=-1)), (None, 1)],
    ids=["positive-int", "negative-int", "None"]
)
@with_multiprocessing
def test_effective_n_jobs_None(backend_n_jobs, expected_n_jobs):
    # check the number of effective jobs when `n_jobs=None`
    # non-regression test for https://github.com/joblib/joblib/issues/984
    with parallel_backend("threading", n_jobs=backend_n_jobs):
        # when using a backend, the default of number jobs will be the one set
        # in the backend
        assert effective_n_jobs(n_jobs=None) == expected_n_jobs
    # without any backend, None will default to a single job
    assert effective_n_jobs(n_jobs=None) == 1


###############################################################################
# Test parallel

@parametrize('backend', ALL_VALID_BACKENDS)
@parametrize('n_jobs', [1, 2, -1, -2])
@parametrize('verbose', [2, 11, 100])
def test_simple_parallel(backend, n_jobs, verbose):
    assert ([square(x) for x in range(5)] ==
            Parallel(n_jobs=n_jobs, backend=backend,
                     verbose=verbose)(
                delayed(square)(x) for x in range(5)))


@parametrize('backend', ALL_VALID_BACKENDS)
def test_main_thread_renamed_no_warning(backend, monkeypatch):
    # Check that no default backend relies on the name of the main thread:
    # https://github.com/joblib/joblib/issues/180#issuecomment-253266247
    # Some programs use a different name for the main thread. This is the case
    # for uWSGI apps for instance.
    monkeypatch.setattr(target=threading.current_thread(), name='name',
                        value='some_new_name_for_the_main_thread')

    with warns(None) as warninfo:
        results = Parallel(n_jobs=2, backend=backend)(
            delayed(square)(x) for x in range(3))
        assert results == [0, 1, 4]

    # Due to the default parameters of LokyBackend, there is a chance that
    # warninfo catches Warnings from worker timeouts. We remove it if it exists
    warninfo = [w for w in warninfo if "worker timeout" not in str(w.message)]

    # The multiprocessing backend will raise a warning when detecting that is
    # started from the non-main thread. Let's check that there is no false
    # positive because of the name change.
    assert len(warninfo) == 0


def _assert_warning_nested(backend, inner_n_jobs, expected):
    with warns(None) as records:
        parallel_func(backend=backend, inner_n_jobs=inner_n_jobs)

    if expected:
        # with threading, we might see more that one records
        if len(records) > 0:
            return 'backed parallel loops cannot' in records[0].message.args[0]
        return False
    else:
        assert len(records) == 0
        return True


@with_multiprocessing
@parametrize('parent_backend,child_backend,expected', [
    ('loky', 'multiprocessing', True), ('loky', 'loky', False),
    ('multiprocessing', 'multiprocessing', True),
    ('multiprocessing', 'loky', True),
    ('threading', 'multiprocessing', True),
    ('threading', 'loky', True),
])
def test_nested_parallel_warnings(parent_backend, child_backend, expected):

    # no warnings if inner_n_jobs=1
    Parallel(n_jobs=2, backend=parent_backend)(
        delayed(_assert_warning_nested)(
            backend=child_backend, inner_n_jobs=1,
            expected=False)
        for _ in range(5))

    #  warnings if inner_n_jobs != 1 and expected
    res = Parallel(n_jobs=2, backend=parent_backend)(
        delayed(_assert_warning_nested)(
            backend=child_backend, inner_n_jobs=2,
            expected=expected)
        for _ in range(5))

    # warning handling is not thread safe. One thread might see multiple
    # warning or no warning at all.
    if parent_backend == "threading":
        assert any(res)
    else:
        assert all(res)


@with_multiprocessing
@parametrize('backend', ['loky', 'multiprocessing', 'threading'])
def test_background_thread_parallelism(backend):
    is_run_parallel = [False]

    def background_thread(is_run_parallel):
        with warns(None) as records:
            Parallel(n_jobs=2)(
                delayed(sleep)(.1) for _ in range(4))
        print(len(records))
        is_run_parallel[0] = len(records) == 0

    t = threading.Thread(target=background_thread, args=(is_run_parallel,))
    t.start()
    t.join()
    assert is_run_parallel[0]


def nested_loop(backend):
    Parallel(n_jobs=2, backend=backend)(
        delayed(square)(.01) for _ in range(2))


@parametrize('child_backend', BACKENDS)
@parametrize('parent_backend', BACKENDS)
def test_nested_loop(parent_backend, child_backend):
    Parallel(n_jobs=2, backend=parent_backend)(
        delayed(nested_loop)(child_backend) for _ in range(2))


def raise_exception(backend):
    raise ValueError


def test_nested_loop_with_exception_with_loky():
    with raises(ValueError):
        with Parallel(n_jobs=2, backend="loky") as parallel:
            parallel([delayed(nested_loop)("loky"),
                      delayed(raise_exception)("loky")])


def test_mutate_input_with_threads():
    """Input is mutable when using the threading backend"""
    q = Queue(maxsize=5)
    Parallel(n_jobs=2, backend="threading")(
        delayed(q.put)(1) for _ in range(5))
    assert q.full()


@parametrize('n_jobs', [1, 2, 3])
def test_parallel_kwargs(n_jobs):
    """Check the keyword argument processing of pmap."""
    lst = range(10)
    assert ([f(x, y=1) for x in lst] ==
            Parallel(n_jobs=n_jobs)(delayed(f)(x, y=1) for x in lst))


@parametrize('backend', PARALLEL_BACKENDS)
def test_parallel_as_context_manager(backend):
    lst = range(10)
    expected = [f(x, y=1) for x in lst]

    with Parallel(n_jobs=4, backend=backend) as p:
        # Internally a pool instance has been eagerly created and is managed
        # via the context manager protocol
        managed_backend = p._backend

        # We make call with the managed parallel object several times inside
        # the managed block:
        assert expected == p(delayed(f)(x, y=1) for x in lst)
        assert expected == p(delayed(f)(x, y=1) for x in lst)

        # Those calls have all used the same pool instance:
        if mp is not None:
            assert get_workers(managed_backend) is get_workers(p._backend)

    # As soon as we exit the context manager block, the pool is terminated and
    # no longer referenced from the parallel object:
    if mp is not None:
        assert get_workers(p._backend) is None

    # It's still possible to use the parallel instance in non-managed mode:
    assert expected == p(delayed(f)(x, y=1) for x in lst)
    if mp is not None:
        assert get_workers(p._backend) is None


@with_multiprocessing
def test_parallel_pickling():
    """ Check that pmap captures the errors when it is passed an object
        that cannot be pickled.
    """
    class UnpicklableObject(object):
        def __reduce__(self):
            raise RuntimeError('123')

    with raises(PicklingError, match=r"the task to send"):
        Parallel(n_jobs=2)(delayed(id)(UnpicklableObject()) for _ in range(10))


@parametrize('backend', PARALLEL_BACKENDS)
def test_parallel_timeout_success(backend):
    # Check that timeout isn't thrown when function is fast enough
    assert len(Parallel(n_jobs=2, backend=backend, timeout=10)(
        delayed(sleep)(0.001) for x in range(10))) == 10


@with_multiprocessing
@parametrize('backend', PARALLEL_BACKENDS)
def test_parallel_timeout_fail(backend):
    # Check that timeout properly fails when function is too slow
    with raises(TimeoutError):
        Parallel(n_jobs=2, backend=backend, timeout=0.01)(
            delayed(sleep)(10) for x in range(10))


@with_multiprocessing
@parametrize('backend', PROCESS_BACKENDS)
def test_error_capture(backend):
    # Check that error are captured, and that correct exceptions
    # are raised.
    if mp is not None:
        with raises(ZeroDivisionError):
            Parallel(n_jobs=2, backend=backend)(
                [delayed(division)(x, y)
                    for x, y in zip((0, 1), (1, 0))])
        with raises(WorkerInterrupt):
            Parallel(n_jobs=2, backend=backend)(
                [delayed(interrupt_raiser)(x) for x in (1, 0)])

        # Try again with the context manager API
        with Parallel(n_jobs=2, backend=backend) as parallel:
            assert get_workers(parallel._backend) is not None
            original_workers = get_workers(parallel._backend)

            with raises(ZeroDivisionError):
                parallel([delayed(division)(x, y)
                          for x, y in zip((0, 1), (1, 0))])

            # The managed pool should still be available and be in a working
            # state despite the previously raised (and caught) exception
            assert get_workers(parallel._backend) is not None

            # The pool should have been interrupted and restarted:
            assert get_workers(parallel._backend) is not original_workers

            assert ([f(x, y=1) for x in range(10)] ==
                    parallel(delayed(f)(x, y=1) for x in range(10)))

            original_workers = get_workers(parallel._backend)
            with raises(WorkerInterrupt):
                parallel([delayed(interrupt_raiser)(x) for x in (1, 0)])

            # The pool should still be available despite the exception
            assert get_workers(parallel._backend) is not None

            # The pool should have been interrupted and restarted:
            assert get_workers(parallel._backend) is not original_workers

            assert ([f(x, y=1) for x in range(10)] ==
                    parallel(delayed(f)(x, y=1) for x in range(10)))

        # Check that the inner pool has been terminated when exiting the
        # context manager
        assert get_workers(parallel._backend) is None
    else:
        with raises(KeyboardInterrupt):
            Parallel(n_jobs=2)(
                [delayed(interrupt_raiser)(x) for x in (1, 0)])

    # wrapped exceptions should inherit from the class of the original
    # exception to make it easy to catch them
    with raises(ZeroDivisionError):
        Parallel(n_jobs=2)(
            [delayed(division)(x, y) for x, y in zip((0, 1), (1, 0))])

    with raises(MyExceptionWithFinickyInit):
        Parallel(n_jobs=2, verbose=0)(
            (delayed(exception_raiser)(i, custom_exception=True)
             for i in range(30)))

    try:
        # JoblibException wrapping is disabled in sequential mode:
        Parallel(n_jobs=1)(
            delayed(division)(x, y) for x, y in zip((0, 1), (1, 0)))
    except Exception as ex:
        assert not isinstance(ex, JoblibException)
    else:
        raise ValueError("The excepted error has not been raised.")


def consumer(queue, item):
    queue.append('Consumed %s' % item)


@parametrize('backend', BACKENDS)
@parametrize('batch_size, expected_queue',
             [(1, ['Produced 0', 'Consumed 0',
                   'Produced 1', 'Consumed 1',
                   'Produced 2', 'Consumed 2',
                   'Produced 3', 'Consumed 3',
                   'Produced 4', 'Consumed 4',
                   'Produced 5', 'Consumed 5']),
              (4, [  # First Batch
                  'Produced 0', 'Produced 1', 'Produced 2', 'Produced 3',
                  'Consumed 0', 'Consumed 1', 'Consumed 2', 'Consumed 3',
                     # Second batch
                  'Produced 4', 'Produced 5', 'Consumed 4', 'Consumed 5'])])
def test_dispatch_one_job(backend, batch_size, expected_queue):
    """ Test that with only one job, Parallel does act as a iterator.
    """
    queue = list()

    def producer():
        for i in range(6):
            queue.append('Produced %i' % i)
            yield i

    Parallel(n_jobs=1, batch_size=batch_size, backend=backend)(
        delayed(consumer)(queue, x) for x in producer())
    assert queue == expected_queue
    assert len(queue) == 12


@with_multiprocessing
@parametrize('backend', PARALLEL_BACKENDS)
def test_dispatch_multiprocessing(backend):
    """ Check that using pre_dispatch Parallel does indeed dispatch items
        lazily.
    """
    manager = mp.Manager()
    queue = manager.list()

    def producer():
        for i in range(6):
            queue.append('Produced %i' % i)
            yield i

    Parallel(n_jobs=2, batch_size=1, pre_dispatch=3, backend=backend)(
        delayed(consumer)(queue, 'any') for _ in producer())

    queue_contents = list(queue)
    assert queue_contents[0] == 'Produced 0'

    # Only 3 tasks are pre-dispatched out of 6. The 4th task is dispatched only
    # after any of the first 3 jobs have completed.
    first_consumption_index = queue_contents[:4].index('Consumed any')
    assert first_consumption_index > -1

    produced_3_index = queue_contents.index('Produced 3')  # 4th task produced
    assert produced_3_index > first_consumption_index

    assert len(queue) == 12


def test_batching_auto_threading():
    # batching='auto' with the threading backend leaves the effective batch
    # size to 1 (no batching) as it has been found to never be beneficial with
    # this low-overhead backend.

    with Parallel(n_jobs=2, batch_size='auto', backend='threading') as p:
        p(delayed(id)(i) for i in range(5000))  # many very fast tasks
        assert p._backend.compute_batch_size() == 1


@with_multiprocessing
@parametrize('backend', PROCESS_BACKENDS)
def test_batching_auto_subprocesses(backend):
    with Parallel(n_jobs=2, batch_size='auto', backend=backend) as p:
        p(delayed(id)(i) for i in range(5000))  # many very fast tasks

        # It should be strictly larger than 1 but as we don't want heisen
        # failures on clogged CI worker environment be safe and only check that
        # it's a strictly positive number.
        assert p._backend.compute_batch_size() > 0


def test_exception_dispatch():
    """Make sure that exception raised during dispatch are indeed captured"""
    with raises(ValueError):
        Parallel(n_jobs=2, pre_dispatch=16, verbose=0)(
            delayed(exception_raiser)(i) for i in range(30))


def nested_function_inner(i):
    Parallel(n_jobs=2)(
        delayed(exception_raiser)(j) for j in range(30))


def nested_function_outer(i):
    Parallel(n_jobs=2)(
        delayed(nested_function_inner)(j) for j in range(30))


@with_multiprocessing
@parametrize('backend', PARALLEL_BACKENDS)
@pytest.mark.xfail(reason="https://github.com/joblib/loky/pull/255")
def test_nested_exception_dispatch(backend):
    """Ensure errors for nested joblib cases gets propagated

    We rely on the Python 3 built-in __cause__ system that already
    report this kind of information to the user.
    """
    with raises(ValueError) as excinfo:
        Parallel(n_jobs=2, backend=backend)(
            delayed(nested_function_outer)(i) for i in range(30))

    # Check that important information such as function names are visible
    # in the final error message reported to the user
    report_lines = format_exception(excinfo.type, excinfo.value, excinfo.tb)
    report = "".join(report_lines)
    assert 'nested_function_outer' in report
    assert 'nested_function_inner' in report
    assert 'exception_raiser' in report

    assert type(excinfo.value) is ValueError


class FakeParallelBackend(SequentialBackend):
    """Pretends to run concurrently while running sequentially."""

    def configure(self, n_jobs=1, parallel=None, **backend_args):
        self.n_jobs = self.effective_n_jobs(n_jobs)
        self.parallel = parallel
        return n_jobs

    def effective_n_jobs(self, n_jobs=1):
        if n_jobs < 0:
            n_jobs = max(mp.cpu_count() + 1 + n_jobs, 1)
        return n_jobs


def test_invalid_backend():
    with raises(ValueError):
        Parallel(backend='unit-testing')


@parametrize('backend', ALL_VALID_BACKENDS)
def test_invalid_njobs(backend):
    with raises(ValueError) as excinfo:
        Parallel(n_jobs=0, backend=backend)._initialize_backend()
    assert "n_jobs == 0 in Parallel has no meaning" in str(excinfo.value)


def test_register_parallel_backend():
    try:
        register_parallel_backend("test_backend", FakeParallelBackend)
        assert "test_backend" in BACKENDS
        assert BACKENDS["test_backend"] == FakeParallelBackend
    finally:
        del BACKENDS["test_backend"]


def test_overwrite_default_backend():
    assert _active_backend_type() == DefaultBackend
    try:
        register_parallel_backend("threading", BACKENDS["threading"],
                                  make_default=True)
        assert _active_backend_type() == ThreadingBackend
    finally:
        # Restore the global default manually
        parallel.DEFAULT_BACKEND = DEFAULT_BACKEND
    assert _active_backend_type() == DefaultBackend


def check_backend_context_manager(backend_name):
    with parallel_backend(backend_name, n_jobs=3):
        active_backend, active_n_jobs = parallel.get_active_backend()
        assert active_n_jobs == 3
        assert effective_n_jobs(3) == 3
        p = Parallel()
        assert p.n_jobs == 3
        if backend_name == 'multiprocessing':
            assert type(active_backend) == MultiprocessingBackend
            assert type(p._backend) == MultiprocessingBackend
        elif backend_name == 'loky':
            assert type(active_backend) == LokyBackend
            assert type(p._backend) == LokyBackend
        elif backend_name == 'threading':
            assert type(active_backend) == ThreadingBackend
            assert type(p._backend) == ThreadingBackend
        elif backend_name.startswith('test_'):
            assert type(active_backend) == FakeParallelBackend
            assert type(p._backend) == FakeParallelBackend


all_backends_for_context_manager = PARALLEL_BACKENDS[:]
all_backends_for_context_manager.extend(
    ['test_backend_%d' % i for i in range(3)]
)


@with_multiprocessing
@parametrize('backend', all_backends_for_context_manager)
def test_backend_context_manager(monkeypatch, backend):
    if backend not in BACKENDS:
        monkeypatch.setitem(BACKENDS, backend, FakeParallelBackend)

    assert _active_backend_type() == DefaultBackend
    # check that this possible to switch parallel backends sequentially
    check_backend_context_manager(backend)

    # The default backend is restored
    assert _active_backend_type() == DefaultBackend

    # Check that context manager switching is thread safe:
    Parallel(n_jobs=2, backend='threading')(
        delayed(check_backend_context_manager)(b)
        for b in all_backends_for_context_manager if not b)

    # The default backend is again restored
    assert _active_backend_type() == DefaultBackend


class ParameterizedParallelBackend(SequentialBackend):
    """Pretends to run conncurrently while running sequentially."""

    def __init__(self, param=None):
        if param is None:
            raise ValueError('param should not be None')
        self.param = param


def test_parameterized_backend_context_manager(monkeypatch):
    monkeypatch.setitem(BACKENDS, 'param_backend',
                        ParameterizedParallelBackend)
    assert _active_backend_type() == DefaultBackend

    with parallel_backend('param_backend', param=42, n_jobs=3):
        active_backend, active_n_jobs = parallel.get_active_backend()
        assert type(active_backend) == ParameterizedParallelBackend
        assert active_backend.param == 42
        assert active_n_jobs == 3
        p = Parallel()
        assert p.n_jobs == 3
        assert p._backend is active_backend
        results = p(delayed(sqrt)(i) for i in range(5))
    assert results == [sqrt(i) for i in range(5)]

    # The default backend is again restored
    assert _active_backend_type() == DefaultBackend


def test_directly_parameterized_backend_context_manager():
    assert _active_backend_type() == DefaultBackend

    # Check that it's possible to pass a backend instance directly,
    # without registration
    with parallel_backend(ParameterizedParallelBackend(param=43), n_jobs=5):
        active_backend, active_n_jobs = parallel.get_active_backend()
        assert type(active_backend) == ParameterizedParallelBackend
        assert active_backend.param == 43
        assert active_n_jobs == 5
        p = Parallel()
        assert p.n_jobs == 5
        assert p._backend is active_backend
        results = p(delayed(sqrt)(i) for i in range(5))
    assert results == [sqrt(i) for i in range(5)]

    # The default backend is again restored
    assert _active_backend_type() == DefaultBackend


def sleep_and_return_pid():
    sleep(.1)
    return os.getpid()


def get_nested_pids():
    assert _active_backend_type() == ThreadingBackend
    # Assert that the nested backend does not change the default number of
    # jobs used in Parallel
    assert Parallel()._effective_n_jobs() == 1

    # Assert that the tasks are running only on one process
    return Parallel(n_jobs=2)(delayed(sleep_and_return_pid)()
                              for _ in range(2))


class MyBackend(joblib._parallel_backends.LokyBackend):
    """Backend to test backward compatibility with older backends"""
    def get_nested_backend(self, ):
        # Older backends only return a backend, without n_jobs indications.
        return super(MyBackend, self).get_nested_backend()[0]


register_parallel_backend('back_compat_backend', MyBackend)


@with_multiprocessing
@parametrize('backend', ['threading', 'loky', 'multiprocessing',
                         'back_compat_backend'])
def test_nested_backend_context_manager(backend):
    # Check that by default, nested parallel calls will always use the
    # ThreadingBackend

    with parallel_backend(backend):
        pid_groups = Parallel(n_jobs=2)(
            delayed(get_nested_pids)()
            for _ in range(10)
        )
        for pid_group in pid_groups:
            assert len(set(pid_group)) == 1


@with_multiprocessing
@parametrize('n_jobs', [2, -1, None])
@parametrize('backend', PARALLEL_BACKENDS)
def test_nested_backend_in_sequential(backend, n_jobs):
    # Check that by default, nested parallel calls will always use the
    # ThreadingBackend

    def check_nested_backend(expected_backend_type, expected_n_job):
        # Assert that the sequential backend at top level, does not change the
        # backend for nested calls.
        assert _active_backend_type() == BACKENDS[expected_backend_type]

        # Assert that the nested backend in SequentialBackend does not change
        # the default number of jobs used in Parallel
        expected_n_job = effective_n_jobs(expected_n_job)
        assert Parallel()._effective_n_jobs() == expected_n_job

    Parallel(n_jobs=1)(
        delayed(check_nested_backend)('loky', 1)
        for _ in range(10)
    )

    with parallel_backend(backend, n_jobs=n_jobs):
        Parallel(n_jobs=1)(
            delayed(check_nested_backend)(backend, n_jobs)
            for _ in range(10)
        )


def check_nesting_level(inner_backend, expected_level):
    with parallel_backend(inner_backend) as (backend, n_jobs):
        assert backend.nesting_level == expected_level


@with_multiprocessing
@parametrize('outer_backend', PARALLEL_BACKENDS)
@parametrize('inner_backend', PARALLEL_BACKENDS)
def test_backend_nesting_level(outer_backend, inner_backend):
    # Check that the nesting level for the backend is correctly set
    check_nesting_level(outer_backend, 0)

    Parallel(n_jobs=2, backend=outer_backend)(
        delayed(check_nesting_level)(inner_backend, 1)
        for _ in range(10)
    )

    with parallel_backend(inner_backend, n_jobs=2):
        Parallel()(delayed(check_nesting_level)(inner_backend, 1)
                   for _ in range(10))


@with_multiprocessing
def test_retrieval_context():
    import contextlib

    class MyBackend(ThreadingBackend):
        i = 0

        @contextlib.contextmanager
        def retrieval_context(self):
            self.i += 1
            yield

    register_parallel_backend("retrieval", MyBackend)

    def nested_call(n):
        return Parallel(n_jobs=2)(delayed(id)(i) for i in range(n))

    with parallel_backend("retrieval") as (ba, _):
        Parallel(n_jobs=2)(
            delayed(nested_call)(i)
            for i in range(5)
        )
        assert ba.i == 1


###############################################################################
# Test helpers
def test_joblib_exception():
    # Smoke-test the custom exception
    e = JoblibException('foobar')
    # Test the repr
    repr(e)
    # Test the pickle
    pickle.dumps(e)


def test_safe_function():
    safe_division = SafeFunction(division)
    with raises(ZeroDivisionError):
        safe_division(1, 0)

    safe_interrupt = SafeFunction(interrupt_raiser)
    with raises(WorkerInterrupt):
        safe_interrupt('x')


@parametrize('batch_size', [0, -1, 1.42])
def test_invalid_batch_size(batch_size):
    with raises(ValueError):
        Parallel(batch_size=batch_size)


@parametrize('n_tasks, n_jobs, pre_dispatch, batch_size',
             [(2, 2, 'all', 'auto'),
              (2, 2, 'n_jobs', 'auto'),
              (10, 2, 'n_jobs', 'auto'),
              (517, 2, 'n_jobs', 'auto'),
              (10, 2, 'n_jobs', 'auto'),
              (10, 4, 'n_jobs', 'auto'),
              (200, 12, 'n_jobs', 'auto'),
              (25, 12, '2 * n_jobs', 1),
              (250, 12, 'all', 1),
              (250, 12, '2 * n_jobs', 7),
              (200, 12, '2 * n_jobs', 'auto')])
def test_dispatch_race_condition(n_tasks, n_jobs, pre_dispatch, batch_size):
    # Check that using (async-)dispatch does not yield a race condition on the
    # iterable generator that is not thread-safe natively.
    # This is a non-regression test for the "Pool seems closed" class of error
    params = {'n_jobs': n_jobs, 'pre_dispatch': pre_dispatch,
              'batch_size': batch_size}
    expected = [square(i) for i in range(n_tasks)]
    results = Parallel(**params)(delayed(square)(i) for i in range(n_tasks))
    assert results == expected


@with_multiprocessing
def test_default_mp_context():
    mp_start_method = mp.get_start_method()
    p = Parallel(n_jobs=2, backend='multiprocessing')
    context = p._backend_args.get('context')
    start_method = context.get_start_method()
    assert start_method == mp_start_method


@with_numpy
@with_multiprocessing
@parametrize('backend', PROCESS_BACKENDS)
def test_no_blas_crash_or_freeze_with_subprocesses(backend):
    if backend == 'multiprocessing':
        # Use the spawn backend that is both robust and available on all
        # platforms
        backend = mp.get_context('spawn')

    # Check that on recent Python version, the 'spawn' start method can make
    # it possible to use multiprocessing in conjunction of any BLAS
    # implementation that happens to be used by numpy with causing a freeze or
    # a crash
    rng = np.random.RandomState(42)

    # call BLAS DGEMM to force the initialization of the internal thread-pool
    # in the main process
    a = rng.randn(1000, 1000)
    np.dot(a, a.T)

    # check that the internal BLAS thread-pool is not in an inconsistent state
    # in the worker processes managed by multiprocessing
    Parallel(n_jobs=2, backend=backend)(
        delayed(np.dot)(a, a.T) for i in range(2))


UNPICKLABLE_CALLABLE_SCRIPT_TEMPLATE_NO_MAIN = """\
from joblib import Parallel, delayed

def square(x):
    return x ** 2

backend = "{}"
if backend == "spawn":
    from multiprocessing import get_context
    backend = get_context(backend)

print(Parallel(n_jobs=2, backend=backend)(
      delayed(square)(i) for i in range(5)))
"""


@with_multiprocessing
@parametrize('backend', PROCESS_BACKENDS)
def test_parallel_with_interactively_defined_functions(backend):
    # When using the "-c" flag, interactive functions defined in __main__
    # should work with any backend.
    if backend == "multiprocessing" and mp.get_start_method() != "fork":
        pytest.skip("Require fork start method to use interactively defined "
                    "functions with multiprocessing.")
    code = UNPICKLABLE_CALLABLE_SCRIPT_TEMPLATE_NO_MAIN.format(backend)
    check_subprocess_call(
        [sys.executable, '-c', code], timeout=10,
        stdout_regex=r'\[0, 1, 4, 9, 16\]')


UNPICKLABLE_CALLABLE_SCRIPT_TEMPLATE_MAIN = """\
import sys
# Make sure that joblib is importable in the subprocess launching this
# script. This is needed in case we run the tests from the joblib root
# folder without having installed joblib
sys.path.insert(0, {joblib_root_folder!r})

from joblib import Parallel, delayed

def run(f, x):
    return f(x)

{define_func}

if __name__ == "__main__":
    backend = "{backend}"
    if backend == "spawn":
        from multiprocessing import get_context
        backend = get_context(backend)

    callable_position = "{callable_position}"
    if callable_position == "delayed":
        print(Parallel(n_jobs=2, backend=backend)(
                delayed(square)(i) for i in range(5)))
    elif callable_position == "args":
        print(Parallel(n_jobs=2, backend=backend)(
                delayed(run)(square, i) for i in range(5)))
    else:
        print(Parallel(n_jobs=2, backend=backend)(
                delayed(run)(f=square, x=i) for i in range(5)))
"""

SQUARE_MAIN = """\
def square(x):
    return x ** 2
"""
SQUARE_LOCAL = """\
def gen_square():
    def square(x):
        return x ** 2
    return square
square = gen_square()
"""
SQUARE_LAMBDA = """\
square = lambda x: x ** 2
"""


@with_multiprocessing
@parametrize('backend', PROCESS_BACKENDS + ([] if mp is None else ['spawn']))
@parametrize('define_func', [SQUARE_MAIN, SQUARE_LOCAL, SQUARE_LAMBDA])
@parametrize('callable_position', ['delayed', 'args', 'kwargs'])
def test_parallel_with_unpicklable_functions_in_args(
        backend, define_func, callable_position, tmpdir):
    if backend in ['multiprocessing', 'spawn'] and (
            define_func != SQUARE_MAIN or sys.platform == "win32"):
        pytest.skip("Not picklable with pickle")
    code = UNPICKLABLE_CALLABLE_SCRIPT_TEMPLATE_MAIN.format(
        define_func=define_func, backend=backend,
        callable_position=callable_position,
        joblib_root_folder=os.path.dirname(os.path.dirname(joblib.__file__)))
    code_file = tmpdir.join("unpicklable_func_script.py")
    code_file.write(code)
    check_subprocess_call(
        [sys.executable, code_file.strpath], timeout=10,
        stdout_regex=r'\[0, 1, 4, 9, 16\]')


INTERACTIVE_DEFINED_FUNCTION_AND_CLASS_SCRIPT_CONTENT = """\
import sys
# Make sure that joblib is importable in the subprocess launching this
# script. This is needed in case we run the tests from the joblib root
# folder without having installed joblib
sys.path.insert(0, {joblib_root_folder!r})

from joblib import Parallel, delayed
from functools import partial

class MyClass:
    '''Class defined in the __main__ namespace'''
    def __init__(self, value):
        self.value = value


def square(x, ignored=None, ignored2=None):
    '''Function defined in the __main__ namespace'''
    return x.value ** 2


square2 = partial(square, ignored2='something')

# Here, we do not need the `if __name__ == "__main__":` safeguard when
# using the default `loky` backend (even on Windows).

# The following baroque function call is meant to check that joblib
# introspection rightfully uses cloudpickle instead of the (faster) pickle
# module of the standard library when necessary. In particular cloudpickle is
# necessary for functions and instances of classes interactively defined in the
# __main__ module.

print(Parallel(n_jobs=2)(
    delayed(square2)(MyClass(i), ignored=[dict(a=MyClass(1))])
    for i in range(5)
))
""".format(joblib_root_folder=os.path.dirname(
    os.path.dirname(joblib.__file__)))


@with_multiprocessing
def test_parallel_with_interactively_defined_functions_default_backend(tmpdir):
    # The default backend (loky) accepts interactive functions defined in
    # __main__ and does not require if __name__ == '__main__' even when
    # the __main__ module is defined by the result of the execution of a
    # filesystem script.
    script = tmpdir.join('joblib_interactively_defined_function.py')
    script.write(INTERACTIVE_DEFINED_FUNCTION_AND_CLASS_SCRIPT_CONTENT)
    check_subprocess_call([sys.executable, script.strpath],
                          stdout_regex=r'\[0, 1, 4, 9, 16\]',
                          timeout=5)


INTERACTIVELY_DEFINED_SUBCLASS_WITH_METHOD_SCRIPT_CONTENT = """\
import sys
# Make sure that joblib is importable in the subprocess launching this
# script. This is needed in case we run the tests from the joblib root
# folder without having installed joblib
sys.path.insert(0, {joblib_root_folder!r})

from joblib import Parallel, delayed, hash
import multiprocessing as mp
mp.util.log_to_stderr(5)

class MyList(list):
    '''MyList is interactively defined by MyList.append is a built-in'''
    def __hash__(self):
        # XXX: workaround limitation in cloudpickle
        return hash(self).__hash__()

l = MyList()

print(Parallel(n_jobs=2)(
    delayed(l.append)(i) for i in range(3)
))
""".format(joblib_root_folder=os.path.dirname(
    os.path.dirname(joblib.__file__)))


@with_multiprocessing
def test_parallel_with_interactively_defined_bound_method(tmpdir):
    script = tmpdir.join('joblib_interactive_bound_method_script.py')
    script.write(INTERACTIVELY_DEFINED_SUBCLASS_WITH_METHOD_SCRIPT_CONTENT)
    check_subprocess_call([sys.executable, script.strpath],
                          stdout_regex=r'\[None, None, None\]',
                          stderr_regex=r'LokyProcess',
                          timeout=15)


def test_parallel_with_exhausted_iterator():
    exhausted_iterator = iter([])
    assert Parallel(n_jobs=2)(exhausted_iterator) == []


def check_memmap(a):
    if not isinstance(a, np.memmap):
        raise TypeError('Expected np.memmap instance, got %r',
                        type(a))
    return a.copy()  # return a regular array instead of a memmap


@with_numpy
@with_multiprocessing
@parametrize('backend', PROCESS_BACKENDS)
def test_auto_memmap_on_arrays_from_generator(backend):
    # Non-regression test for a problem with a bad interaction between the
    # GC collecting arrays recently created during iteration inside the
    # parallel dispatch loop and the auto-memmap feature of Parallel.
    # See: https://github.com/joblib/joblib/pull/294
    def generate_arrays(n):
        for i in range(n):
            yield np.ones(10, dtype=np.float32) * i
    # Use max_nbytes=1 to force the use of memory-mapping even for small
    # arrays
    results = Parallel(n_jobs=2, max_nbytes=1, backend=backend)(
        delayed(check_memmap)(a) for a in generate_arrays(100))
    for result, expected in zip(results, generate_arrays(len(results))):
        np.testing.assert_array_equal(expected, result)

    # Second call to force loky to adapt the executor by growing the number
    # of worker processes. This is a non-regression test for:
    # https://github.com/joblib/joblib/issues/629.
    results = Parallel(n_jobs=4, max_nbytes=1, backend=backend)(
        delayed(check_memmap)(a) for a in generate_arrays(100))
    for result, expected in zip(results, generate_arrays(len(results))):
        np.testing.assert_array_equal(expected, result)


def identity(arg):
    return arg


@with_numpy
@with_multiprocessing
def test_memmap_with_big_offset(tmpdir):
    fname = tmpdir.join('test.mmap').strpath
    size = mmap.ALLOCATIONGRANULARITY
    obj = [np.zeros(size, dtype='uint8'), np.ones(size, dtype='uint8')]
    dump(obj, fname)
    memmap = load(fname, mmap_mode='r')
    result, = Parallel(n_jobs=2)(delayed(identity)(memmap) for _ in [0])
    assert isinstance(memmap[1], np.memmap)
    assert memmap[1].offset > size
    np.testing.assert_array_equal(obj, result)


def test_warning_about_timeout_not_supported_by_backend():
    with warns(None) as warninfo:
        Parallel(timeout=1)(delayed(square)(i) for i in range(50))
    assert len(warninfo) == 1
    w = warninfo[0]
    assert isinstance(w.message, UserWarning)
    assert str(w.message) == (
        "The backend class 'SequentialBackend' does not support timeout. "
        "You have set 'timeout=1' in Parallel but the 'timeout' parameter "
        "will not be used.")


@parametrize('backend', ALL_VALID_BACKENDS)
@parametrize('n_jobs', [1, 2, -2, -1])
def test_abort_backend(n_jobs, backend):
    delays = ["a"] + [10] * 100
    with raises(TypeError):
        t_start = time.time()
        Parallel(n_jobs=n_jobs, backend=backend)(
            delayed(time.sleep)(i) for i in delays)
    dt = time.time() - t_start
    assert dt < 20


@with_numpy
@with_multiprocessing
@parametrize('backend', PROCESS_BACKENDS)
def test_memmapping_leaks(backend, tmpdir):
    # Non-regression test for memmapping backends. Ensure that the data
    # does not stay too long in memory
    tmpdir = tmpdir.strpath

    # Use max_nbytes=1 to force the use of memory-mapping even for small
    # arrays
    with Parallel(n_jobs=2, max_nbytes=1, backend=backend,
                  temp_folder=tmpdir) as p:
        p(delayed(check_memmap)(a) for a in [np.random.random(10)] * 2)

        # The memmap folder should not be clean in the context scope
        assert len(os.listdir(tmpdir)) > 0

    # Make sure that the shared memory is cleaned at the end when we exit
    # the context
    for _ in range(100):
        if not os.listdir(tmpdir):
            break
        sleep(.1)
    else:
        raise AssertionError('temporary directory of Parallel was not removed')

    # Make sure that the shared memory is cleaned at the end of a call
    p = Parallel(n_jobs=2, max_nbytes=1, backend=backend)
    p(delayed(check_memmap)(a) for a in [np.random.random(10)] * 2)

    for _ in range(100):
        if not os.listdir(tmpdir):
            break
        sleep(.1)
    else:
        raise AssertionError('temporary directory of Parallel was not removed')


@parametrize('backend', [None, 'loky', 'threading'])
def test_lambda_expression(backend):
    # cloudpickle is used to pickle delayed callables
    results = Parallel(n_jobs=2, backend=backend)(
        delayed(lambda x: x ** 2)(i) for i in range(10))
    assert results == [i ** 2 for i in range(10)]


@with_multiprocessing
@parametrize('backend', PROCESS_BACKENDS)
def test_backend_batch_statistics_reset(backend):
    """Test that a parallel backend correctly resets its batch statistics."""
    n_jobs = 2
    n_inputs = 500
    task_time = 2. / n_inputs

    p = Parallel(verbose=10, n_jobs=n_jobs, backend=backend)
    p(delayed(time.sleep)(task_time) for i in range(n_inputs))
    assert (p._backend._effective_batch_size ==
            p._backend._DEFAULT_EFFECTIVE_BATCH_SIZE)
    assert (p._backend._smoothed_batch_duration ==
            p._backend._DEFAULT_SMOOTHED_BATCH_DURATION)

    p(delayed(time.sleep)(task_time) for i in range(n_inputs))
    assert (p._backend._effective_batch_size ==
            p._backend._DEFAULT_EFFECTIVE_BATCH_SIZE)
    assert (p._backend._smoothed_batch_duration ==
            p._backend._DEFAULT_SMOOTHED_BATCH_DURATION)


def test_backend_hinting_and_constraints():
    for n_jobs in [1, 2, -1]:
        assert type(Parallel(n_jobs=n_jobs)._backend) == LokyBackend

        p = Parallel(n_jobs=n_jobs, prefer='threads')
        assert type(p._backend) == ThreadingBackend

        p = Parallel(n_jobs=n_jobs, prefer='processes')
        assert type(p._backend) == LokyBackend

        p = Parallel(n_jobs=n_jobs, require='sharedmem')
        assert type(p._backend) == ThreadingBackend

    # Explicit backend selection can override backend hinting although it
    # is useless to pass a hint when selecting a backend.
    p = Parallel(n_jobs=2, backend='loky', prefer='threads')
    assert type(p._backend) == LokyBackend

    with parallel_backend('loky', n_jobs=2):
        # Explicit backend selection by the user with the context manager
        # should be respected when combined with backend hints only.
        p = Parallel(prefer='threads')
        assert type(p._backend) == LokyBackend
        assert p.n_jobs == 2

    with parallel_backend('loky', n_jobs=2):
        # Locally hard-coded n_jobs value is respected.
        p = Parallel(n_jobs=3, prefer='threads')
        assert type(p._backend) == LokyBackend
        assert p.n_jobs == 3

    with parallel_backend('loky', n_jobs=2):
        # Explicit backend selection by the user with the context manager
        # should be ignored when the Parallel call has hard constraints.
        # In this case, the default backend that supports shared mem is
        # used an the default number of processes is used.
        p = Parallel(require='sharedmem')
        assert type(p._backend) == ThreadingBackend
        assert p.n_jobs == 1

    with parallel_backend('loky', n_jobs=2):
        p = Parallel(n_jobs=3, require='sharedmem')
        assert type(p._backend) == ThreadingBackend
        assert p.n_jobs == 3


def test_backend_hinting_and_constraints_with_custom_backends(capsys):
    # Custom backends can declare that they use threads and have shared memory
    # semantics:
    class MyCustomThreadingBackend(ParallelBackendBase):
        supports_sharedmem = True
        use_threads = True

        def apply_async(self):
            pass

        def effective_n_jobs(self, n_jobs):
            return n_jobs

    with parallel_backend(MyCustomThreadingBackend()):
        p = Parallel(n_jobs=2, prefer='processes')  # ignored
        assert type(p._backend) == MyCustomThreadingBackend

        p = Parallel(n_jobs=2, require='sharedmem')
        assert type(p._backend) == MyCustomThreadingBackend

    class MyCustomProcessingBackend(ParallelBackendBase):
        supports_sharedmem = False
        use_threads = False

        def apply_async(self):
            pass

        def effective_n_jobs(self, n_jobs):
            return n_jobs

    with parallel_backend(MyCustomProcessingBackend()):
        p = Parallel(n_jobs=2, prefer='processes')
        assert type(p._backend) == MyCustomProcessingBackend

        out, err = capsys.readouterr()
        assert out == ""
        assert err == ""

        p = Parallel(n_jobs=2, require='sharedmem', verbose=10)
        assert type(p._backend) == ThreadingBackend

        out, err = capsys.readouterr()
        expected = ("Using ThreadingBackend as joblib.Parallel backend "
                    "instead of MyCustomProcessingBackend as the latter "
                    "does not provide shared memory semantics.")
        assert out.strip() == expected
        assert err == ""

    with raises(ValueError):
        Parallel(backend=MyCustomProcessingBackend(), require='sharedmem')


def test_invalid_backend_hinting_and_constraints():
    with raises(ValueError):
        Parallel(prefer='invalid')

    with raises(ValueError):
        Parallel(require='invalid')

    with raises(ValueError):
        # It is inconsistent to prefer process-based parallelism while
        # requiring shared memory semantics.
        Parallel(prefer='processes', require='sharedmem')

    # It is inconsistent to ask explicitly for a process-based parallelism
    # while requiring shared memory semantics.
    with raises(ValueError):
        Parallel(backend='loky', require='sharedmem')
    with raises(ValueError):
        Parallel(backend='multiprocessing', require='sharedmem')


def test_global_parallel_backend():
    default = Parallel()._backend

    pb = parallel_backend('threading')
    assert isinstance(Parallel()._backend, ThreadingBackend)

    pb.unregister()
    assert type(Parallel()._backend) is type(default)


def test_external_backends():
    def register_foo():
        BACKENDS['foo'] = ThreadingBackend

    EXTERNAL_BACKENDS['foo'] = register_foo

    with parallel_backend('foo'):
        assert isinstance(Parallel()._backend, ThreadingBackend)


def _recursive_backend_info(limit=3, **kwargs):
    """Perform nested parallel calls and introspect the backend on the way"""

    with Parallel(n_jobs=2) as p:
        this_level = [(type(p._backend).__name__, p._backend.nesting_level)]
        if limit == 0:
            return this_level
        results = p(delayed(_recursive_backend_info)(limit=limit - 1, **kwargs)
                    for i in range(1))
        return this_level + results[0]


@with_multiprocessing
@parametrize('backend', ['loky', 'threading'])
def test_nested_parallelism_limit(backend):
    with parallel_backend(backend, n_jobs=2):
        backend_types_and_levels = _recursive_backend_info()

    if cpu_count() == 1:
        second_level_backend_type = 'SequentialBackend'
        max_level = 1
    else:
        second_level_backend_type = 'ThreadingBackend'
        max_level = 2

    top_level_backend_type = backend.title() + 'Backend'
    expected_types_and_levels = [
        (top_level_backend_type, 0),
        (second_level_backend_type, 1),
        ('SequentialBackend', max_level),
        ('SequentialBackend', max_level)
    ]
    assert backend_types_and_levels == expected_types_and_levels


@with_numpy
@skipif(distributed is None, reason='This test requires dask')
def test_nested_parallelism_with_dask():
    client = distributed.Client(n_workers=2, threads_per_worker=2)  # noqa

    # 10 MB of data as argument to trigger implicit scattering
    data = np.ones(int(1e7), dtype=np.uint8)
    for i in range(2):
        with parallel_backend('dask'):
            backend_types_and_levels = _recursive_backend_info(data=data)
        assert len(backend_types_and_levels) == 4
        assert all(name == 'DaskDistributedBackend'
                   for name, _ in backend_types_and_levels)

    # No argument
    with parallel_backend('dask'):
        backend_types_and_levels = _recursive_backend_info()
    assert len(backend_types_and_levels) == 4
    assert all(name == 'DaskDistributedBackend'
               for name, _ in backend_types_and_levels)


def _recursive_parallel(nesting_limit=None):
    """A horrible function that does recursive parallel calls"""
    return Parallel()(delayed(_recursive_parallel)() for i in range(2))


@parametrize('backend', ['loky', 'threading'])
def test_thread_bomb_mitigation(backend):
    # Test that recursive parallelism raises a recursion rather than
    # saturating the operating system resources by creating a unbounded number
    # of threads.
    with parallel_backend(backend, n_jobs=2):
        with raises(BaseException) as excinfo:
            _recursive_parallel()
    exc = excinfo.value
    if backend == "loky" and isinstance(exc, TerminatedWorkerError):
        # The recursion exception can itself cause an error when pickling it to
        # be send back to the parent process. In this case the worker crashes
        # but the original traceback is still printed on stderr. This could be
        # improved but does not seem simple to do and this is is not critical
        # for users (as long as there is no process or thread bomb happening).
        pytest.xfail("Loky worker crash when serializing RecursionError")
    else:
        assert isinstance(exc, RecursionError)


def _run_parallel_sum():
    env_vars = {}
    for var in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS',
                'NUMBA_NUM_THREADS', 'ENABLE_IPC']:
        env_vars[var] = os.environ.get(var)
    return env_vars, parallel_sum(100)


@parametrize("backend", [None, 'loky'])
@skipif(parallel_sum is None, reason="Need OpenMP helper compiled")
def test_parallel_thread_limit(backend):
    results = Parallel(n_jobs=2, backend=backend)(
        delayed(_run_parallel_sum)() for _ in range(2)
    )
    expected_num_threads = max(cpu_count() // 2, 1)
    for worker_env_vars, omp_num_threads in results:
        assert omp_num_threads == expected_num_threads
        for name, value in worker_env_vars.items():
            if name.endswith("_THREADS"):
                assert value == str(expected_num_threads)
            else:
                assert name == "ENABLE_IPC"
                assert value == "1"


@skipif(distributed is not None,
        reason='This test requires dask NOT installed')
def test_dask_backend_when_dask_not_installed():
    with raises(ValueError, match='Please install dask'):
        parallel_backend('dask')


def test_zero_worker_backend():
    # joblib.Parallel should reject with an explicit error message parallel
    # backends that have no worker.
    class ZeroWorkerBackend(ThreadingBackend):
        def configure(self, *args, **kwargs):
            return 0

        def apply_async(self, func, callback=None):   # pragma: no cover
            raise TimeoutError("No worker available")

        def effective_n_jobs(self, n_jobs):   # pragma: no cover
            return 0

    expected_msg = "ZeroWorkerBackend has no active worker"
    with parallel_backend(ZeroWorkerBackend()):
        with pytest.raises(RuntimeError, match=expected_msg):
            Parallel(n_jobs=2)(delayed(id)(i) for i in range(2))


def test_globals_update_at_each_parallel_call():
    # This is a non-regression test related to joblib issues #836 and #833.
    # Cloudpickle versions between 0.5.4 and 0.7 introduced a bug where global
    # variables changes in a parent process between two calls to
    # joblib.Parallel would not be propagated into the workers.
    global MY_GLOBAL_VARIABLE
    MY_GLOBAL_VARIABLE = "original value"

    def check_globals():
        global MY_GLOBAL_VARIABLE
        return MY_GLOBAL_VARIABLE

    assert check_globals() == "original value"

    workers_global_variable = Parallel(n_jobs=2)(
        delayed(check_globals)() for i in range(2))
    assert set(workers_global_variable) == {"original value"}

    # Change the value of MY_GLOBAL_VARIABLE, and make sure this change gets
    # propagated into the workers environment
    MY_GLOBAL_VARIABLE = "changed value"
    assert check_globals() == "changed value"

    workers_global_variable = Parallel(n_jobs=2)(
        delayed(check_globals)() for i in range(2))
    assert set(workers_global_variable) == {"changed value"}


##############################################################################
# Test environment variable in child env, in particular for limiting
# the maximal number of threads in C-library threadpools.
#

def _check_numpy_threadpool_limits():
    import numpy as np
    # Let's call BLAS on a Matrix Matrix multiplication with dimensions large
    # enough to ensure that the threadpool managed by the underlying BLAS
    # implementation is actually used so as to force its initialization.
    a = np.random.randn(100, 100)
    np.dot(a, a)
    from threadpoolctl import threadpool_info
    return threadpool_info()


def _parent_max_num_threads_for(child_module, parent_info):
    for parent_module in parent_info:
        if parent_module['filepath'] == child_module['filepath']:
            return parent_module['num_threads']
    raise ValueError("An unexpected module was loaded in child:\n{}"
                     .format(child_module))


def check_child_num_threads(workers_info, parent_info, num_threads):
    # Check that the number of threads reported in workers_info is consistent
    # with the expectation. We need to be careful to handle the cases where
    # the requested number of threads is below max_num_thread for the library.
    for child_threadpool_info in workers_info:
        for child_module in child_threadpool_info:
            parent_max_num_threads = _parent_max_num_threads_for(
                child_module, parent_info)
            expected = {min(num_threads, parent_max_num_threads), num_threads}
            assert child_module['num_threads'] in expected


@with_numpy
@with_multiprocessing
@parametrize('n_jobs', [2, 4, -2, -1])
def test_threadpool_limitation_in_child(n_jobs):
    # Check that the protection against oversubscription in workers is working
    # using threadpoolctl functionalities.

    # Skip this test if numpy is not linked to a BLAS library
    parent_info = _check_numpy_threadpool_limits()
    if len(parent_info) == 0:
        pytest.skip(msg="Need a version of numpy linked to BLAS")

    workers_threadpool_infos = Parallel(n_jobs=n_jobs)(
        delayed(_check_numpy_threadpool_limits)() for i in range(2))

    n_jobs = effective_n_jobs(n_jobs)
    expected_child_num_threads = max(cpu_count() // n_jobs, 1)

    check_child_num_threads(workers_threadpool_infos, parent_info,
                            expected_child_num_threads)


@with_numpy
@with_multiprocessing
@parametrize('inner_max_num_threads', [1, 2, 4, None])
@parametrize('n_jobs', [2, -1])
def test_threadpool_limitation_in_child_context(n_jobs, inner_max_num_threads):
    # Check that the protection against oversubscription in workers is working
    # using threadpoolctl functionalities.

    # Skip this test if numpy is not linked to a BLAS library
    parent_info = _check_numpy_threadpool_limits()
    if len(parent_info) == 0:
        pytest.skip(msg="Need a version of numpy linked to BLAS")

    with parallel_backend('loky', inner_max_num_threads=inner_max_num_threads):
        workers_threadpool_infos = Parallel(n_jobs=n_jobs)(
            delayed(_check_numpy_threadpool_limits)() for i in range(2))

    n_jobs = effective_n_jobs(n_jobs)
    if inner_max_num_threads is None:
        expected_child_num_threads = max(cpu_count() // n_jobs, 1)
    else:
        expected_child_num_threads = inner_max_num_threads

    check_child_num_threads(workers_threadpool_infos, parent_info,
                            expected_child_num_threads)


@with_multiprocessing
@parametrize('n_jobs', [2, -1])
@parametrize('var_name', ["OPENBLAS_NUM_THREADS",
                          "MKL_NUM_THREADS",
                          "OMP_NUM_THREADS"])
def test_threadpool_limitation_in_child_override(n_jobs, var_name):
    # Check that environment variables set by the user on the main process
    # always have the priority.

    # Clean up the existing executor because we change the environment of the
    # parent at runtime and it is not detected in loky intentionally.
    get_reusable_executor(reuse=True).shutdown()

    def _get_env(var_name):
        return os.environ.get(var_name)

    original_var_value = os.environ.get(var_name)
    try:
        os.environ[var_name] = "4"
        # Skip this test if numpy is not linked to a BLAS library
        results = Parallel(n_jobs=n_jobs)(
            delayed(_get_env)(var_name) for i in range(2))
        assert results == ["4", "4"]

        with parallel_backend('loky', inner_max_num_threads=1):
            results = Parallel(n_jobs=n_jobs)(
                delayed(_get_env)(var_name) for i in range(2))
        assert results == ["1", "1"]

    finally:
        if original_var_value is None:
            del os.environ[var_name]
        else:
            os.environ[var_name] = original_var_value


@with_numpy
@with_multiprocessing
@parametrize('backend', ['multiprocessing', 'threading',
                         MultiprocessingBackend(), ThreadingBackend()])
def test_threadpool_limitation_in_child_context_error(backend):

    with raises(AssertionError, match=r"does not acc.*inner_max_num_threads"):
        parallel_backend(backend, inner_max_num_threads=1)


@with_multiprocessing
@parametrize('n_jobs', [2, 4, -1])
def test_loky_reuse_workers(n_jobs):
    # Non-regression test for issue #967 where the workers are not reused when
    # calling multiple Parallel loops.

    def parallel_call(n_jobs):
        x = range(10)
        Parallel(n_jobs=n_jobs)(delayed(sum)(x) for i in range(10))

    # Run a parallel loop and get the workers used for computations
    parallel_call(n_jobs)
    first_executor = get_reusable_executor(reuse=True)

    # Ensure that the workers are reused for the next calls, as the executor is
    # not restarted.
    for _ in range(10):
        parallel_call(n_jobs)
        executor = get_reusable_executor(reuse=True)
        assert executor == first_executor
