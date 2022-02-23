from __future__ import print_function, division, absolute_import
import os

import pytest
from random import random
from uuid import uuid4
from time import sleep

from .. import Parallel, delayed, parallel_backend
from ..parallel import ThreadingBackend, AutoBatchingMixin
from .._dask import DaskDistributedBackend

distributed = pytest.importorskip('distributed')
from distributed import Client, LocalCluster, get_client
from distributed.metrics import time
from distributed.utils_test import cluster, inc


def noop(*args, **kwargs):
    pass


def slow_raise_value_error(condition, duration=0.05):
    sleep(duration)
    if condition:
        raise ValueError("condition evaluated to True")


def count_events(event_name, client):
    worker_events = client.run(lambda dask_worker: dask_worker.log)
    event_counts = {}
    for w, events in worker_events.items():
        event_counts[w] = len([event for event in list(events)
                               if event[1] == event_name])
    return event_counts


def test_simple(loop):
    with cluster() as (s, [a, b]):
        with Client(s['address'], loop=loop) as client:  # noqa: F841
            with parallel_backend('dask') as (ba, _):
                seq = Parallel()(delayed(inc)(i) for i in range(10))
                assert seq == [inc(i) for i in range(10)]

                with pytest.raises(ValueError):
                    Parallel()(delayed(slow_raise_value_error)(i == 3)
                               for i in range(10))

                seq = Parallel()(delayed(inc)(i) for i in range(10))
                assert seq == [inc(i) for i in range(10)]


def test_dask_backend_uses_autobatching(loop):
    assert (DaskDistributedBackend.compute_batch_size
            is AutoBatchingMixin.compute_batch_size)

    with cluster() as (s, [a, b]):
        with Client(s['address'], loop=loop) as client:  # noqa: F841
            with parallel_backend('dask') as (ba, _):
                with Parallel() as parallel:
                    # The backend should be initialized with a default
                    # batch size of 1:
                    backend = parallel._backend
                    assert isinstance(backend, DaskDistributedBackend)
                    assert backend.parallel is parallel
                    assert backend._effective_batch_size == 1

                    # Launch many short tasks that should trigger
                    # auto-batching:
                    parallel(
                        delayed(lambda: None)()
                        for _ in range(int(1e4))
                    )
                    assert backend._effective_batch_size > 10


def random2():
    return random()


def test_dont_assume_function_purity(loop):
    with cluster() as (s, [a, b]):
        with Client(s['address'], loop=loop) as client:  # noqa: F841
            with parallel_backend('dask') as (ba, _):
                x, y = Parallel()(delayed(random2)() for i in range(2))
                assert x != y


@pytest.mark.parametrize("mixed", [True, False])
def test_dask_funcname(loop, mixed):
    from joblib._dask import Batch
    if not mixed:
        tasks = [delayed(inc)(i) for i in range(4)]
        batch_repr = 'batch_of_inc_4_calls'
    else:
        tasks = [
            delayed(abs)(i) if i % 2 else delayed(inc)(i) for i in range(4)
        ]
        batch_repr = 'mixed_batch_of_inc_4_calls'

    assert repr(Batch(tasks)) == batch_repr

    with cluster() as (s, [a, b]):
        with Client(s['address'], loop=loop) as client:
            with parallel_backend('dask') as (ba, _):
                _ = Parallel(batch_size=2, pre_dispatch='all')(tasks)

            def f(dask_scheduler):
                return list(dask_scheduler.transition_log)
            batch_repr = batch_repr.replace('4', '2')
            log = client.run_on_scheduler(f)
            assert all('batch_of_inc' in tup[0] for tup in log)


def test_no_undesired_distributed_cache_hit(loop):
    # Dask has a pickle cache for callables that are called many times. Because
    # the dask backends used to wrapp both the functions and the arguments
    # under instances of the Batch callable class this caching mechanism could
    # lead to bugs as described in: https://github.com/joblib/joblib/pull/1055
    # The joblib-dask backend has been refactored to avoid bundling the
    # arguments as an attribute of the Batch instance to avoid this problem.
    # This test serves as non-regression problem.

    # Use a large number of input arguments to give the AutoBatchingMixin
    # enough tasks to kick-in.
    lists = [[] for _ in range(100)]
    np = pytest.importorskip('numpy')
    X = np.arange(int(1e6))

    def isolated_operation(list_, data=None):
        if data is not None:
            np.testing.assert_array_equal(data, X)
        list_.append(uuid4().hex)
        return list_

    cluster = LocalCluster(n_workers=1, threads_per_worker=2)
    client = Client(cluster)
    try:
        with parallel_backend('dask') as (ba, _):
            # dispatches joblib.parallel.BatchedCalls
            res = Parallel()(
                delayed(isolated_operation)(list_) for list_ in lists
            )

        # The original arguments should not have been mutated as the mutation
        # happens in the dask worker process.
        assert lists == [[] for _ in range(100)]

        # Here we did not pass any large numpy array as argument to
        # isolated_operation so no scattering event should happen under the
        # hood.
        counts = count_events('receive-from-scatter', client)
        assert sum(counts.values()) == 0
        assert all([len(r) == 1 for r in res])

        with parallel_backend('dask') as (ba, _):
            # Append a large array which will be scattered by dask, and
            # dispatch joblib._dask.Batch
            res = Parallel()(
                delayed(isolated_operation)(list_, data=X) for list_ in lists
            )

        # This time, auto-scattering should have kicked it.
        counts = count_events('receive-from-scatter', client)
        assert sum(counts.values()) > 0
        assert all([len(r) == 1 for r in res])
    finally:
        client.close()
        cluster.close()


class CountSerialized(object):
    def __init__(self, x):
        self.x = x
        self.count = 0

    def __add__(self, other):
        return self.x + getattr(other, 'x', other)

    __radd__ = __add__

    def __reduce__(self):
        self.count += 1
        return (CountSerialized, (self.x,))


def add5(a, b, c, d=0, e=0):
    return a + b + c + d + e


def test_manual_scatter(loop):
    x = CountSerialized(1)
    y = CountSerialized(2)
    z = CountSerialized(3)

    with cluster() as (s, [a, b]):
        with Client(s['address'], loop=loop) as client:  # noqa: F841
            with parallel_backend('dask', scatter=[x, y]) as (ba, _):
                f = delayed(add5)
                tasks = [f(x, y, z, d=4, e=5),
                         f(x, z, y, d=5, e=4),
                         f(y, x, z, d=x, e=5),
                         f(z, z, x, d=z, e=y)]
                expected = [func(*args, **kwargs)
                            for func, args, kwargs in tasks]
                results = Parallel()(tasks)

            # Scatter must take a list/tuple
            with pytest.raises(TypeError):
                with parallel_backend('dask', loop=loop, scatter=1):
                    pass

    assert results == expected

    # Scattered variables only serialized once
    assert x.count == 1
    assert y.count == 1
    # Depending on the version of distributed, the unscattered z variable
    # is either pickled 4 or 6 times, possibly because of the memoization
    # of objects that appear several times in the arguments of a delayed
    # task.
    assert z.count in (4, 6)


# When the same IOLoop is used for multiple clients in a row, use
# loop_in_thread instead of loop to prevent the Client from closing it.  See
# dask/distributed #4112
def test_auto_scatter(loop_in_thread):
    np = pytest.importorskip('numpy')
    data1 = np.ones(int(1e4), dtype=np.uint8)
    data2 = np.ones(int(1e4), dtype=np.uint8)
    data_to_process = ([data1] * 3) + ([data2] * 3)

    with cluster() as (s, [a, b]):
        with Client(s['address'], loop=loop_in_thread) as client:
            with parallel_backend('dask') as (ba, _):
                # Passing the same data as arg and kwarg triggers a single
                # scatter operation whose result is reused.
                Parallel()(delayed(noop)(data, data, i, opt=data)
                           for i, data in enumerate(data_to_process))
            # By default large array are automatically scattered with
            # broadcast=1 which means that one worker must directly receive
            # the data from the scatter operation once.
            counts = count_events('receive-from-scatter', client)
            assert counts[a['address']] + counts[b['address']] == 2

    with cluster() as (s, [a, b]):
        with Client(s['address'], loop=loop_in_thread) as client:
            with parallel_backend('dask') as (ba, _):
                Parallel()(delayed(noop)(data1[:3], i) for i in range(5))
            # Small arrays are passed within the task definition without going
            # through a scatter operation.
            counts = count_events('receive-from-scatter', client)
            assert counts[a['address']] == 0
            assert counts[b['address']] == 0


@pytest.mark.parametrize("retry_no", list(range(2)))
def test_nested_scatter(loop, retry_no):

    np = pytest.importorskip('numpy')

    NUM_INNER_TASKS = 10
    NUM_OUTER_TASKS = 10

    def my_sum(x, i, j):
        return np.sum(x)

    def outer_function_joblib(array, i):
        client = get_client()  # noqa
        with parallel_backend("dask"):
            results = Parallel()(
                delayed(my_sum)(array[j:], i, j) for j in range(
                    NUM_INNER_TASKS)
            )
        return sum(results)

    with cluster() as (s, [a, b]):
        with Client(s['address'], loop=loop) as _:
            with parallel_backend("dask"):
                my_array = np.ones(10000)
                _ = Parallel()(
                    delayed(outer_function_joblib)(
                        my_array[i:], i) for i in range(NUM_OUTER_TASKS)
                )


def test_nested_backend_context_manager(loop_in_thread):
    def get_nested_pids():
        pids = set(Parallel(n_jobs=2)(delayed(os.getpid)() for _ in range(2)))
        pids |= set(Parallel(n_jobs=2)(delayed(os.getpid)() for _ in range(2)))
        return pids

    with cluster() as (s, [a, b]):
        with Client(s['address'], loop=loop_in_thread) as client:
            with parallel_backend('dask') as (ba, _):
                pid_groups = Parallel(n_jobs=2)(
                    delayed(get_nested_pids)()
                    for _ in range(10)
                )
                for pid_group in pid_groups:
                    assert len(set(pid_group)) <= 2

        # No deadlocks
        with Client(s['address'], loop=loop_in_thread) as client:  # noqa: F841
            with parallel_backend('dask') as (ba, _):
                pid_groups = Parallel(n_jobs=2)(
                    delayed(get_nested_pids)()
                    for _ in range(10)
                )
                for pid_group in pid_groups:
                    assert len(set(pid_group)) <= 2


def test_nested_backend_context_manager_implicit_n_jobs(loop):
    # Check that Parallel with no explicit n_jobs value automatically selects
    # all the dask workers, including in nested calls.

    def _backend_type(p):
        return p._backend.__class__.__name__

    def get_nested_implicit_n_jobs():
        with Parallel() as p:
            return _backend_type(p), p.n_jobs

    with cluster() as (s, [a, b]):
        with Client(s['address'], loop=loop) as client:  # noqa: F841
            with parallel_backend('dask') as (ba, _):
                with Parallel() as p:
                    assert _backend_type(p) == "DaskDistributedBackend"
                    assert p.n_jobs == -1
                    all_nested_n_jobs = p(
                        delayed(get_nested_implicit_n_jobs)()
                        for _ in range(2)
                    )
                for backend_type, nested_n_jobs in all_nested_n_jobs:
                    assert backend_type == "DaskDistributedBackend"
                    assert nested_n_jobs == -1


def test_errors(loop):
    with pytest.raises(ValueError) as info:
        with parallel_backend('dask'):
            pass

    assert "create a dask client" in str(info.value).lower()


def test_correct_nested_backend(loop):
    with cluster() as (s, [a, b]):
        with Client(s['address'], loop=loop) as client:  # noqa: F841
            # No requirement, should be us
            with parallel_backend('dask') as (ba, _):
                result = Parallel(n_jobs=2)(
                    delayed(outer)(nested_require=None) for _ in range(1))
                assert isinstance(result[0][0][0], DaskDistributedBackend)

            # Require threads, should be threading
            with parallel_backend('dask') as (ba, _):
                result = Parallel(n_jobs=2)(
                    delayed(outer)(nested_require='sharedmem')
                    for _ in range(1))
                assert isinstance(result[0][0][0], ThreadingBackend)


def outer(nested_require):
    return Parallel(n_jobs=2, prefer='threads')(
        delayed(middle)(nested_require) for _ in range(1)
    )


def middle(require):
    return Parallel(n_jobs=2, require=require)(
        delayed(inner)() for _ in range(1)
    )


def inner():
    return Parallel()._backend


def test_secede_with_no_processes(loop):
    # https://github.com/dask/distributed/issues/1775
    with Client(loop=loop, processes=False, set_as_default=True):
        with parallel_backend('dask'):
            Parallel(n_jobs=4)(delayed(id)(i) for i in range(2))


def _worker_address(_):
    from distributed import get_worker
    return get_worker().address


def test_dask_backend_keywords(loop):
    with cluster() as (s, [a, b]):
        with Client(s['address'], loop=loop) as client:  # noqa: F841
            with parallel_backend('dask', workers=a['address']) as (ba, _):
                seq = Parallel()(
                    delayed(_worker_address)(i) for i in range(10))
                assert seq == [a['address']] * 10

            with parallel_backend('dask', workers=b['address']) as (ba, _):
                seq = Parallel()(
                    delayed(_worker_address)(i) for i in range(10))
                assert seq == [b['address']] * 10


def test_cleanup(loop):
    with Client(processes=False, loop=loop) as client:
        with parallel_backend('dask'):
            Parallel()(delayed(inc)(i) for i in range(10))

        start = time()
        while client.cluster.scheduler.tasks:
            sleep(0.01)
            assert time() < start + 5

        assert not client.futures


@pytest.mark.parametrize("cluster_strategy", ["adaptive", "late_scaling"])
@pytest.mark.skipif(
    distributed.__version__ <= '2.1.1' and distributed.__version__ >= '1.28.0',
    reason="distributed bug - https://github.com/dask/distributed/pull/2841")
def test_wait_for_workers(cluster_strategy):
    cluster = LocalCluster(n_workers=0, processes=False, threads_per_worker=2)
    client = Client(cluster)
    if cluster_strategy == "adaptive":
        cluster.adapt(minimum=0, maximum=2)
    elif cluster_strategy == "late_scaling":
        # Tell the cluster to start workers but this is a non-blocking call
        # and new workers might take time to connect. In this case the Parallel
        # call should wait for at least one worker to come up before starting
        # to schedule work.
        cluster.scale(2)
    try:
        with parallel_backend('dask'):
            # The following should wait a bit for at least one worker to
            # become available.
            Parallel()(delayed(inc)(i) for i in range(10))
    finally:
        client.close()
        cluster.close()


def test_wait_for_workers_timeout():
    # Start a cluster with 0 worker:
    cluster = LocalCluster(n_workers=0, processes=False, threads_per_worker=2)
    client = Client(cluster)
    try:
        with parallel_backend('dask', wait_for_workers_timeout=0.1):
            # Short timeout: DaskDistributedBackend
            msg = "DaskDistributedBackend has no worker after 0.1 seconds."
            with pytest.raises(TimeoutError, match=msg):
                Parallel()(delayed(inc)(i) for i in range(10))

        with parallel_backend('dask', wait_for_workers_timeout=0):
            # No timeout: fallback to generic joblib failure:
            msg = "DaskDistributedBackend has no active worker"
            with pytest.raises(RuntimeError, match=msg):
                Parallel()(delayed(inc)(i) for i in range(10))
    finally:
        client.close()
        cluster.close()
