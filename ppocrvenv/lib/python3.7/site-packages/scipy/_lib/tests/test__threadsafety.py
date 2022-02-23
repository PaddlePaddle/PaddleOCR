import threading
import time
import traceback

from numpy.testing import assert_
from pytest import raises as assert_raises

from scipy._lib._threadsafety import ReentrancyLock, non_reentrant, ReentrancyError


def test_parallel_threads():
    # Check that ReentrancyLock serializes work in parallel threads.
    #
    # The test is not fully deterministic, and may succeed falsely if
    # the timings go wrong.

    lock = ReentrancyLock("failure")

    failflag = [False]
    exceptions_raised = []

    def worker(k):
        try:
            with lock:
                assert_(not failflag[0])
                failflag[0] = True
                time.sleep(0.1 * k)
                assert_(failflag[0])
                failflag[0] = False
        except Exception:
            exceptions_raised.append(traceback.format_exc(2))

    threads = [threading.Thread(target=lambda k=k: worker(k))
               for k in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    exceptions_raised = "\n".join(exceptions_raised)
    assert_(not exceptions_raised, exceptions_raised)


def test_reentering():
    # Check that ReentrancyLock prevents re-entering from the same thread.

    @non_reentrant()
    def func(x):
        return func(x)

    assert_raises(ReentrancyError, func, 0)
