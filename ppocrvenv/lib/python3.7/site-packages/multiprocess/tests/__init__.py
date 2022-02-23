#
# Unit tests for the multiprocess package
#

import unittest
import queue as pyqueue
import contextlib
import time
import io
import itertools
import sys
import os
import gc
import errno
import signal
import array
import socket
import random
import logging
import struct
import operator
import weakref
import test.support
import test.support.script_helper
from test import support


# Skip tests if _multiprocessing wasn't built.
_multiprocessing = test.support.import_module('_multiprocessing')
# Skip tests if sem_open implementation is broken.
test.support.import_module('multiprocess.synchronize')
# import threading after _multiprocessing to raise a more relevant error
# message: "No module named _multiprocessing". _multiprocessing is not compiled
# without thread support.
import threading

import multiprocess.connection
import multiprocess.dummy
import multiprocess.heap
import multiprocess.managers
import multiprocess.pool
import multiprocess.queues

from multiprocess import util

try:
    from multiprocess import reduction
    HAS_REDUCTION = reduction.HAVE_SEND_HANDLE
except ImportError:
    HAS_REDUCTION = False

try:
    from multiprocess.sharedctypes import Value, copy
    HAS_SHAREDCTYPES = True
except ImportError:
    HAS_SHAREDCTYPES = False

try:
    import msvcrt
except ImportError:
    msvcrt = None

#
#
#

# Timeout to wait until a process completes
TIMEOUT = 60.0 # seconds

def latin(s):
    return s.encode('latin')


def close_queue(queue):
    if isinstance(queue, multiprocess.queues.Queue):
        queue.close()
        queue.join_thread()


def join_process(process):
    # Since multiprocess.Process has the same API than threading.Thread
    # (join() and is_alive(), the support function can be reused
    support.join_thread(process, timeout=TIMEOUT)


#
# Constants
#

LOG_LEVEL = util.SUBWARNING
#LOG_LEVEL = logging.DEBUG

DELTA = 0.1
CHECK_TIMINGS = False     # making true makes tests take a lot longer
                          # and can sometimes cause some non-serious
                          # failures because some calls block a bit
                          # longer than expected
if CHECK_TIMINGS:
    TIMEOUT1, TIMEOUT2, TIMEOUT3 = 0.82, 0.35, 1.4
else:
    TIMEOUT1, TIMEOUT2, TIMEOUT3 = 0.1, 0.1, 0.1

HAVE_GETVALUE = not getattr(_multiprocessing,
                            'HAVE_BROKEN_SEM_GETVALUE', False)

WIN32 = (sys.platform == "win32")

from multiprocess.connection import wait

def wait_for_handle(handle, timeout):
    if timeout is not None and timeout < 0.0:
        timeout = None
    return wait([handle], timeout)

try:
    MAXFD = os.sysconf("SC_OPEN_MAX")
except:
    MAXFD = 256

# To speed up tests when using the forkserver, we can preload these:
PRELOAD = ['__main__', 'test_multiprocessing_forkserver']

#
# Some tests require ctypes
#

try:
    from ctypes import Structure, c_int, c_double, c_longlong
except ImportError:
    Structure = object
    c_int = c_double = c_longlong = None


def check_enough_semaphores():
    """Check that the system supports enough semaphores to run the test."""
    # minimum number of semaphores available according to POSIX
    nsems_min = 256
    try:
        nsems = os.sysconf("SC_SEM_NSEMS_MAX")
    except (AttributeError, ValueError):
        # sysconf not available or setting not available
        return
    if nsems == -1 or nsems >= nsems_min:
        return
    raise unittest.SkipTest("The OS doesn't support enough semaphores "
                            "to run the test (required: %d)." % nsems_min)


#
# Creates a wrapper for a function which records the time it takes to finish
#

class TimingWrapper(object):

    def __init__(self, func):
        self.func = func
        self.elapsed = None

    def __call__(self, *args, **kwds):
        t = getattr(time,'monotonic',time.time)()
        try:
            return self.func(*args, **kwds)
        finally:
            self.elapsed = getattr(time,'monotonic',time.time)() - t

#
# Base class for test cases
#

class BaseTestCase(object):

    ALLOWED_TYPES = ('processes', 'manager', 'threads')

    def assertTimingAlmostEqual(self, a, b):
        if CHECK_TIMINGS:
            self.assertAlmostEqual(a, b, 1)

    def assertReturnsIfImplemented(self, value, func, *args):
        try:
            res = func(*args)
        except NotImplementedError:
            pass
        else:
            return self.assertEqual(value, res)

    # For the sanity of Windows users, rather than crashing or freezing in
    # multiple ways.
    def __reduce__(self, *args):
        raise NotImplementedError("shouldn't try to pickle a test case")

    __reduce_ex__ = __reduce__

#
# Return the value of a semaphore
#

def get_value(self):
    try:
        return self.get_value()
    except AttributeError:
        try:
            return self._Semaphore__value
        except AttributeError:
            try:
                return self._value
            except AttributeError:
                raise NotImplementedError

#
# Testcases
#

class DummyCallable:
    def __call__(self, q, c):
        assert isinstance(c, DummyCallable)
        q.put(5)


class _TestProcess(BaseTestCase):

    ALLOWED_TYPES = ('processes', 'threads')

    def test_current(self):
        if self.TYPE == 'threads':
            self.skipTest('test not appropriate for {}'.format(self.TYPE))

        current = self.current_process()
        authkey = current.authkey

        self.assertTrue(current.is_alive())
        self.assertTrue(not current.daemon)
        self.assertIsInstance(authkey, bytes)
        self.assertTrue(len(authkey) > 0)
        self.assertEqual(current.ident, os.getpid())
        self.assertEqual(current.exitcode, None)

    def test_daemon_argument(self):
        if self.TYPE == "threads":
            self.skipTest('test not appropriate for {}'.format(self.TYPE))

        # By default uses the current process's daemon flag.
        proc0 = self.Process(target=self._test)
        self.assertEqual(proc0.daemon, self.current_process().daemon)
        proc1 = self.Process(target=self._test, daemon=True)
        self.assertTrue(proc1.daemon)
        proc2 = self.Process(target=self._test, daemon=False)
        self.assertFalse(proc2.daemon)

    @classmethod
    def _test(cls, q, *args, **kwds):
        current = cls.current_process()
        q.put(args)
        q.put(kwds)
        q.put(current.name)
        if cls.TYPE != 'threads':
            q.put(bytes(current.authkey))
            q.put(current.pid)

    def test_process(self):
        q = self.Queue(1)
        e = self.Event()
        args = (q, 1, 2)
        kwargs = {'hello':23, 'bye':2.54}
        name = 'SomeProcess'
        p = self.Process(
            target=self._test, args=args, kwargs=kwargs, name=name
            )
        p.daemon = True
        current = self.current_process()

        if self.TYPE != 'threads':
            self.assertEqual(p.authkey, current.authkey)
        self.assertEqual(p.is_alive(), False)
        self.assertEqual(p.daemon, True)
        self.assertNotIn(p, self.active_children())
        self.assertTrue(type(self.active_children()) is list)
        self.assertEqual(p.exitcode, None)

        p.start()

        self.assertEqual(p.exitcode, None)
        self.assertEqual(p.is_alive(), True)
        self.assertIn(p, self.active_children())

        self.assertEqual(q.get(), args[1:])
        self.assertEqual(q.get(), kwargs)
        self.assertEqual(q.get(), p.name)
        if self.TYPE != 'threads':
            self.assertEqual(q.get(), current.authkey)
            self.assertEqual(q.get(), p.pid)

        p.join()

        self.assertEqual(p.exitcode, 0)
        self.assertEqual(p.is_alive(), False)
        self.assertNotIn(p, self.active_children())
        close_queue(q)

    @classmethod
    def _sleep_some(cls):
        time.sleep(100)

    @classmethod
    def _test_sleep(cls, delay):
        time.sleep(delay)

    def _kill_process(self, meth):
        if self.TYPE == 'threads':
            self.skipTest('test not appropriate for {}'.format(self.TYPE))

        p = self.Process(target=self._sleep_some)
        p.daemon = True
        p.start()

        self.assertEqual(p.is_alive(), True)
        self.assertIn(p, self.active_children())
        self.assertEqual(p.exitcode, None)

        join = TimingWrapper(p.join)

        self.assertEqual(join(0), None)
        self.assertTimingAlmostEqual(join.elapsed, 0.0)
        self.assertEqual(p.is_alive(), True)

        self.assertEqual(join(-1), None)
        self.assertTimingAlmostEqual(join.elapsed, 0.0)
        self.assertEqual(p.is_alive(), True)

        # XXX maybe terminating too soon causes the problems on Gentoo...
        time.sleep(1)

        meth(p)

        if hasattr(signal, 'alarm'):
            # On the Gentoo buildbot waitpid() often seems to block forever.
            # We use alarm() to interrupt it if it blocks for too long.
            def handler(*args):
                raise RuntimeError('join took too long: %s' % p)
            old_handler = signal.signal(signal.SIGALRM, handler)
            try:
                signal.alarm(10)
                self.assertEqual(join(), None)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        else:
            self.assertEqual(join(), None)

        self.assertTimingAlmostEqual(join.elapsed, 0.0)

        self.assertEqual(p.is_alive(), False)
        self.assertNotIn(p, self.active_children())

        p.join()

        return p.exitcode

    def test_terminate(self):
        exitcode = self._kill_process(multiprocess.Process.terminate)
        if os.name != 'nt':
            self.assertEqual(exitcode, -signal.SIGTERM)

    def test_kill(self):
        exitcode = self._kill_process(multiprocess.Process.kill)
        if os.name != 'nt':
            self.assertEqual(exitcode, -signal.SIGKILL)

    def test_cpu_count(self):
        try:
            cpus = multiprocess.cpu_count()
        except NotImplementedError:
            cpus = 1
        self.assertTrue(type(cpus) is int)
        self.assertTrue(cpus >= 1)

    def test_active_children(self):
        self.assertEqual(type(self.active_children()), list)

        p = self.Process(target=time.sleep, args=(DELTA,))
        self.assertNotIn(p, self.active_children())

        p.daemon = True
        p.start()
        self.assertIn(p, self.active_children())

        p.join()
        self.assertNotIn(p, self.active_children())

    @classmethod
    def _test_recursion(cls, wconn, id):
        wconn.send(id)
        if len(id) < 2:
            for i in range(2):
                p = cls.Process(
                    target=cls._test_recursion, args=(wconn, id+[i])
                    )
                p.start()
                p.join()

    @unittest.skipIf(True, "fails with is_dill(obj, child=True)")
    def test_recursion(self):
        rconn, wconn = self.Pipe(duplex=False)
        self._test_recursion(wconn, [])

        time.sleep(DELTA)
        result = []
        while rconn.poll():
            result.append(rconn.recv())

        expected = [
            [],
              [0],
                [0, 0],
                [0, 1],
              [1],
                [1, 0],
                [1, 1]
            ]
        self.assertEqual(result, expected)

    @classmethod
    def _test_sentinel(cls, event):
        event.wait(10.0)

    def test_sentinel(self):
        if self.TYPE == "threads":
            self.skipTest('test not appropriate for {}'.format(self.TYPE))
        event = self.Event()
        p = self.Process(target=self._test_sentinel, args=(event,))
        with self.assertRaises(ValueError):
            p.sentinel
        p.start()
        self.addCleanup(p.join)
        sentinel = p.sentinel
        self.assertIsInstance(sentinel, int)
        self.assertFalse(wait_for_handle(sentinel, timeout=0.0))
        event.set()
        p.join()
        self.assertTrue(wait_for_handle(sentinel, timeout=1))

    @classmethod
    def _test_close(cls, rc=0, q=None):
        if q is not None:
            q.get()
        sys.exit(rc)

    def test_close(self):
        if self.TYPE == "threads":
            self.skipTest('test not appropriate for {}'.format(self.TYPE))
        q = self.Queue()
        p = self.Process(target=self._test_close, kwargs={'q': q})
        p.daemon = True
        p.start()
        self.assertEqual(p.is_alive(), True)
        # Child is still alive, cannot close
        with self.assertRaises(ValueError):
            p.close()

        q.put(None)
        p.join()
        self.assertEqual(p.is_alive(), False)
        self.assertEqual(p.exitcode, 0)
        p.close()
        with self.assertRaises(ValueError):
            p.is_alive()
        with self.assertRaises(ValueError):
            p.join()
        with self.assertRaises(ValueError):
            p.terminate()
        p.close()

        wr = weakref.ref(p)
        del p
        gc.collect()
        self.assertIs(wr(), None)

        close_queue(q)

    def test_many_processes(self):
        if self.TYPE == 'threads':
            self.skipTest('test not appropriate for {}'.format(self.TYPE))

        sm = multiprocess.get_start_method()
        N = 5 if sm == 'spawn' else 100

        # Try to overwhelm the forkserver loop with events
        procs = [self.Process(target=self._test_sleep, args=(0.01,))
                 for i in range(N)]
        for p in procs:
            p.start()
        for p in procs:
            join_process(p)
        for p in procs:
            self.assertEqual(p.exitcode, 0)

        procs = [self.Process(target=self._sleep_some)
                 for i in range(N)]
        for p in procs:
            p.start()
        time.sleep(0.001)  # let the children start...
        for p in procs:
            p.terminate()
        for p in procs:
            join_process(p)
        if os.name != 'nt':
            exitcodes = [-signal.SIGTERM]
            if sys.platform == 'darwin':
                # bpo-31510: On macOS, killing a freshly started process with
                # SIGTERM sometimes kills the process with SIGKILL.
                exitcodes.append(-signal.SIGKILL)
            for p in procs:
                self.assertIn(p.exitcode, exitcodes)

    def test_lose_target_ref(self):
        c = DummyCallable()
        wr = weakref.ref(c)
        q = self.Queue()
        p = self.Process(target=c, args=(q, c))
        del c
        p.start()
        p.join()
        self.assertIs(wr(), None)
        self.assertEqual(q.get(), 5)
        close_queue(q)

    @classmethod
    def _test_child_fd_inflation(self, evt, q):
        q.put(test.support.fd_count())
        evt.wait()

    def test_child_fd_inflation(self):
        # Number of fds in child processes should not grow with the
        # number of running children.
        if self.TYPE == 'threads':
            self.skipTest('test not appropriate for {}'.format(self.TYPE))

        sm = multiprocess.get_start_method()
        if sm == 'fork':
            # The fork method by design inherits all fds from the parent,
            # trying to go against it is a lost battle
            self.skipTest('test not appropriate for {}'.format(sm))

        N = 5
        evt = self.Event()
        q = self.Queue()

        procs = [self.Process(target=self._test_child_fd_inflation, args=(evt, q))
                 for i in range(N)]
        for p in procs:
            p.start()

        try:
            fd_counts = [q.get() for i in range(N)]
            self.assertEqual(len(set(fd_counts)), 1, fd_counts)

        finally:
            evt.set()
            for p in procs:
                p.join()
            close_queue(q)

    @classmethod
    def _test_wait_for_threads(self, evt):
        def func1():
            time.sleep(0.5)
            evt.set()

        def func2():
            time.sleep(20)
            evt.clear()

        threading.Thread(target=func1).start()
        threading.Thread(target=func2, daemon=True).start()

    def test_wait_for_threads(self):
        # A child process should wait for non-daemonic threads to end
        # before exiting
        if self.TYPE == 'threads':
            self.skipTest('test not appropriate for {}'.format(self.TYPE))

        evt = self.Event()
        proc = self.Process(target=self._test_wait_for_threads, args=(evt,))
        proc.start()
        proc.join()
        self.assertTrue(evt.is_set())

    @classmethod
    def _test_error_on_stdio_flush(self, evt, break_std_streams={}):
        for stream_name, action in break_std_streams.items():
            if action == 'close':
                stream = io.StringIO()
                stream.close()
            else:
                assert action == 'remove'
                stream = None
            setattr(sys, stream_name, None)
        evt.set()

    def test_error_on_stdio_flush_1(self):
        # Check that Process works with broken standard streams
        streams = [io.StringIO(), None]
        streams[0].close()
        for stream_name in ('stdout', 'stderr'):
            for stream in streams:
                old_stream = getattr(sys, stream_name)
                setattr(sys, stream_name, stream)
                try:
                    evt = self.Event()
                    proc = self.Process(target=self._test_error_on_stdio_flush,
                                        args=(evt,))
                    proc.start()
                    proc.join()
                    self.assertTrue(evt.is_set())
                    self.assertEqual(proc.exitcode, 0)
                finally:
                    setattr(sys, stream_name, old_stream)

    def test_error_on_stdio_flush_2(self):
        # Same as test_error_on_stdio_flush_1(), but standard streams are
        # broken by the child process
        for stream_name in ('stdout', 'stderr'):
            for action in ('close', 'remove'):
                old_stream = getattr(sys, stream_name)
                try:
                    evt = self.Event()
                    proc = self.Process(target=self._test_error_on_stdio_flush,
                                        args=(evt, {stream_name: action}))
                    proc.start()
                    proc.join()
                    self.assertTrue(evt.is_set())
                    self.assertEqual(proc.exitcode, 0)
                finally:
                    setattr(sys, stream_name, old_stream)

    @classmethod
    def _sleep_and_set_event(self, evt, delay=0.0):
        time.sleep(delay)
        evt.set()

    def check_forkserver_death(self, signum):
        # bpo-31308: if the forkserver process has died, we should still
        # be able to create and run new Process instances (the forkserver
        # is implicitly restarted).
        if self.TYPE == 'threads':
            self.skipTest('test not appropriate for {}'.format(self.TYPE))
        sm = multiprocess.get_start_method()
        if sm != 'forkserver':
            # The fork method by design inherits all fds from the parent,
            # trying to go against it is a lost battle
            self.skipTest('test not appropriate for {}'.format(sm))

        from multiprocess.forkserver import _forkserver
        _forkserver.ensure_running()

        # First process sleeps 500 ms
        delay = 0.5

        evt = self.Event()
        proc = self.Process(target=self._sleep_and_set_event, args=(evt, delay))
        proc.start()

        pid = _forkserver._forkserver_pid
        os.kill(pid, signum)
        # give time to the fork server to die and time to proc to complete
        time.sleep(delay * 2.0)

        evt2 = self.Event()
        proc2 = self.Process(target=self._sleep_and_set_event, args=(evt2,))
        proc2.start()
        proc2.join()
        self.assertTrue(evt2.is_set())
        self.assertEqual(proc2.exitcode, 0)

        proc.join()
        self.assertTrue(evt.is_set())
        self.assertIn(proc.exitcode, (0, 255))

    def test_forkserver_sigint(self):
        # Catchable signal
        self.check_forkserver_death(signal.SIGINT)

    def test_forkserver_sigkill(self):
        # Uncatchable signal
        if os.name != 'nt':
            self.check_forkserver_death(signal.SIGKILL)


#
#
#

class _UpperCaser(multiprocess.Process):

    def __init__(self):
        multiprocess.Process.__init__(self)
        self.child_conn, self.parent_conn = multiprocess.Pipe()

    def run(self):
        self.parent_conn.close()
        for s in iter(self.child_conn.recv, None):
            self.child_conn.send(s.upper())
        self.child_conn.close()

    def submit(self, s):
        assert type(s) is str
        self.parent_conn.send(s)
        return self.parent_conn.recv()

    def stop(self):
        self.parent_conn.send(None)
        self.parent_conn.close()
        self.child_conn.close()

class _TestSubclassingProcess(BaseTestCase):

    ALLOWED_TYPES = ('processes',)

    def test_subclassing(self):
        uppercaser = _UpperCaser()
        uppercaser.daemon = True
        uppercaser.start()
        self.assertEqual(uppercaser.submit('hello'), 'HELLO')
        self.assertEqual(uppercaser.submit('world'), 'WORLD')
        uppercaser.stop()
        uppercaser.join()

    def test_stderr_flush(self):
        # sys.stderr is flushed at process shutdown (issue #13812)
        if self.TYPE == "threads":
            self.skipTest('test not appropriate for {}'.format(self.TYPE))

        testfn = test.support.TESTFN
        self.addCleanup(test.support.unlink, testfn)
        proc = self.Process(target=self._test_stderr_flush, args=(testfn,))
        proc.start()
        proc.join()
        with open(testfn, 'r') as f:
            err = f.read()
            # The whole traceback was printed
            self.assertIn("ZeroDivisionError", err)
            self.assertIn("__init__.py", err)
            self.assertIn("1/0 # MARKER", err)

    @classmethod
    def _test_stderr_flush(cls, testfn):
        fd = os.open(testfn, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        sys.stderr = open(fd, 'w', closefd=False)
        1/0 # MARKER


    @classmethod
    def _test_sys_exit(cls, reason, testfn):
        fd = os.open(testfn, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        sys.stderr = open(fd, 'w', closefd=False)
        sys.exit(reason)

    def test_sys_exit(self):
        # See Issue 13854
        if self.TYPE == 'threads':
            self.skipTest('test not appropriate for {}'.format(self.TYPE))

        testfn = test.support.TESTFN
        self.addCleanup(test.support.unlink, testfn)

        for reason in (
            [1, 2, 3],
            'ignore this',
        ):
            p = self.Process(target=self._test_sys_exit, args=(reason, testfn))
            p.daemon = True
            p.start()
            join_process(p)
            self.assertEqual(p.exitcode, 1)

            with open(testfn, 'r') as f:
                content = f.read()
            self.assertEqual(content.rstrip(), str(reason))

            os.unlink(testfn)

        for reason in (True, False, 8):
            p = self.Process(target=sys.exit, args=(reason,))
            p.daemon = True
            p.start()
            join_process(p)
            self.assertEqual(p.exitcode, reason)

#
#
#

def queue_empty(q):
    if hasattr(q, 'empty'):
        return q.empty()
    else:
        return q.qsize() == 0

def queue_full(q, maxsize):
    if hasattr(q, 'full'):
        return q.full()
    else:
        return q.qsize() == maxsize


class _TestQueue(BaseTestCase):


    @classmethod
    def _test_put(cls, queue, child_can_start, parent_can_continue):
        child_can_start.wait()
        for i in range(6):
            queue.get()
        parent_can_continue.set()

    def test_put(self):
        MAXSIZE = 6
        queue = self.Queue(maxsize=MAXSIZE)
        child_can_start = self.Event()
        parent_can_continue = self.Event()

        proc = self.Process(
            target=self._test_put,
            args=(queue, child_can_start, parent_can_continue)
            )
        proc.daemon = True
        proc.start()

        self.assertEqual(queue_empty(queue), True)
        self.assertEqual(queue_full(queue, MAXSIZE), False)

        queue.put(1)
        queue.put(2, True)
        queue.put(3, True, None)
        queue.put(4, False)
        queue.put(5, False, None)
        queue.put_nowait(6)

        # the values may be in buffer but not yet in pipe so sleep a bit
        time.sleep(DELTA)

        self.assertEqual(queue_empty(queue), False)
        self.assertEqual(queue_full(queue, MAXSIZE), True)

        put = TimingWrapper(queue.put)
        put_nowait = TimingWrapper(queue.put_nowait)

        self.assertRaises(pyqueue.Full, put, 7, False)
        self.assertTimingAlmostEqual(put.elapsed, 0)

        self.assertRaises(pyqueue.Full, put, 7, False, None)
        self.assertTimingAlmostEqual(put.elapsed, 0)

        self.assertRaises(pyqueue.Full, put_nowait, 7)
        self.assertTimingAlmostEqual(put_nowait.elapsed, 0)

        self.assertRaises(pyqueue.Full, put, 7, True, TIMEOUT1)
        self.assertTimingAlmostEqual(put.elapsed, TIMEOUT1)

        self.assertRaises(pyqueue.Full, put, 7, False, TIMEOUT2)
        self.assertTimingAlmostEqual(put.elapsed, 0)

        self.assertRaises(pyqueue.Full, put, 7, True, timeout=TIMEOUT3)
        self.assertTimingAlmostEqual(put.elapsed, TIMEOUT3)

        child_can_start.set()
        parent_can_continue.wait()

        self.assertEqual(queue_empty(queue), True)
        self.assertEqual(queue_full(queue, MAXSIZE), False)

        proc.join()
        close_queue(queue)

    @classmethod
    def _test_get(cls, queue, child_can_start, parent_can_continue):
        child_can_start.wait()
        #queue.put(1)
        queue.put(2)
        queue.put(3)
        queue.put(4)
        queue.put(5)
        parent_can_continue.set()

    def test_get(self):
        queue = self.Queue()
        child_can_start = self.Event()
        parent_can_continue = self.Event()

        proc = self.Process(
            target=self._test_get,
            args=(queue, child_can_start, parent_can_continue)
            )
        proc.daemon = True
        proc.start()

        self.assertEqual(queue_empty(queue), True)

        child_can_start.set()
        parent_can_continue.wait()

        time.sleep(DELTA)
        self.assertEqual(queue_empty(queue), False)

        # Hangs unexpectedly, remove for now
        #self.assertEqual(queue.get(), 1)
        self.assertEqual(queue.get(True, None), 2)
        self.assertEqual(queue.get(True), 3)
        self.assertEqual(queue.get(timeout=1), 4)
        self.assertEqual(queue.get_nowait(), 5)

        self.assertEqual(queue_empty(queue), True)

        get = TimingWrapper(queue.get)
        get_nowait = TimingWrapper(queue.get_nowait)

        self.assertRaises(pyqueue.Empty, get, False)
        self.assertTimingAlmostEqual(get.elapsed, 0)

        self.assertRaises(pyqueue.Empty, get, False, None)
        self.assertTimingAlmostEqual(get.elapsed, 0)

        self.assertRaises(pyqueue.Empty, get_nowait)
        self.assertTimingAlmostEqual(get_nowait.elapsed, 0)

        self.assertRaises(pyqueue.Empty, get, True, TIMEOUT1)
        self.assertTimingAlmostEqual(get.elapsed, TIMEOUT1)

        self.assertRaises(pyqueue.Empty, get, False, TIMEOUT2)
        self.assertTimingAlmostEqual(get.elapsed, 0)

        self.assertRaises(pyqueue.Empty, get, timeout=TIMEOUT3)
        self.assertTimingAlmostEqual(get.elapsed, TIMEOUT3)

        proc.join()
        close_queue(queue)

    @classmethod
    def _test_fork(cls, queue):
        for i in range(10, 20):
            queue.put(i)
        # note that at this point the items may only be buffered, so the
        # process cannot shutdown until the feeder thread has finished
        # pushing items onto the pipe.

    def test_fork(self):
        # Old versions of Queue would fail to create a new feeder
        # thread for a forked process if the original process had its
        # own feeder thread.  This test checks that this no longer
        # happens.

        queue = self.Queue()

        # put items on queue so that main process starts a feeder thread
        for i in range(10):
            queue.put(i)

        # wait to make sure thread starts before we fork a new process
        time.sleep(DELTA)

        # fork process
        p = self.Process(target=self._test_fork, args=(queue,))
        p.daemon = True
        p.start()

        # check that all expected items are in the queue
        for i in range(20):
            self.assertEqual(queue.get(), i)
        self.assertRaises(pyqueue.Empty, queue.get, False)

        p.join()
        close_queue(queue)

    def test_qsize(self):
        q = self.Queue()
        try:
            self.assertEqual(q.qsize(), 0)
        except NotImplementedError:
            self.skipTest('qsize method not implemented')
        q.put(1)
        self.assertEqual(q.qsize(), 1)
        q.put(5)
        self.assertEqual(q.qsize(), 2)
        q.get()
        self.assertEqual(q.qsize(), 1)
        q.get()
        self.assertEqual(q.qsize(), 0)
        close_queue(q)

    @classmethod
    def _test_task_done(cls, q):
        for obj in iter(q.get, None):
            time.sleep(DELTA)
            q.task_done()

    def test_task_done(self):
        queue = self.JoinableQueue()

        workers = [self.Process(target=self._test_task_done, args=(queue,))
                   for i in range(4)]

        for p in workers:
            p.daemon = True
            p.start()

        for i in range(10):
            queue.put(i)

        queue.join()

        for p in workers:
            queue.put(None)

        for p in workers:
            p.join()
        close_queue(queue)

    def test_no_import_lock_contention(self):
        with test.support.temp_cwd():
            module_name = 'imported_by_an_imported_module'
            with open(module_name + '.py', 'w') as f:
                f.write("""if 1:
                    import multiprocess as multiprocessing

                    q = multiprocessing.Queue()
                    q.put('knock knock')
                    q.get(timeout=3)
                    q.close()
                    del q
                """)

            with test.support.DirsOnSysPath(os.getcwd()):
                try:
                    __import__(module_name)
                except pyqueue.Empty:
                    self.fail("Probable regression on import lock contention;"
                              " see Issue #22853")

    def test_timeout(self):
        q = multiprocess.Queue()
        start = getattr(time,'monotonic',time.time)()
        self.assertRaises(pyqueue.Empty, q.get, True, 0.200)
        delta = getattr(time,'monotonic',time.time)() - start
        # bpo-30317: Tolerate a delta of 100 ms because of the bad clock
        # resolution on Windows (usually 15.6 ms). x86 Windows7 3.x once
        # failed because the delta was only 135.8 ms.
        self.assertGreaterEqual(delta, 0.100)
        close_queue(q)

    def test_queue_feeder_donot_stop_onexc(self):
        # bpo-30414: verify feeder handles exceptions correctly
        if self.TYPE != 'processes':
            self.skipTest('test not appropriate for {}'.format(self.TYPE))

        class NotSerializable(object):
            def __reduce__(self):
                raise AttributeError
        with test.support.captured_stderr():
            q = self.Queue()
            q.put(NotSerializable())
            q.put(True)
            self.assertTrue(q.get(timeout=TIMEOUT))
            close_queue(q)

        with test.support.captured_stderr():
            # bpo-33078: verify that the queue size is correctly handled
            # on errors.
            q = self.Queue(maxsize=1)
            q.put(NotSerializable())
            q.put(True)
            try:
                self.assertEqual(q.qsize(), 1)
            except NotImplementedError:
                # qsize is not available on all platform as it
                # relies on sem_getvalue
                pass
            # bpo-30595: use a timeout of 1 second for slow buildbots
            self.assertTrue(q.get(timeout=1.0))
            # Check that the size of the queue is correct
            self.assertTrue(q.empty())
            close_queue(q)

    def test_queue_feeder_on_queue_feeder_error(self):
        # bpo-30006: verify feeder handles exceptions using the
        # _on_queue_feeder_error hook.
        if self.TYPE != 'processes':
            self.skipTest('test not appropriate for {}'.format(self.TYPE))

        class NotSerializable(object):
            """Mock unserializable object"""
            def __init__(self):
                self.reduce_was_called = False
                self.on_queue_feeder_error_was_called = False

            def __reduce__(self):
                self.reduce_was_called = True
                raise AttributeError

        class SafeQueue(multiprocess.queues.Queue):
            """Queue with overloaded _on_queue_feeder_error hook"""
            @staticmethod
            def _on_queue_feeder_error(e, obj):
                if (isinstance(e, AttributeError) and
                        isinstance(obj, NotSerializable)):
                    obj.on_queue_feeder_error_was_called = True

        not_serializable_obj = NotSerializable()
        # The captured_stderr reduces the noise in the test report
        with test.support.captured_stderr():
            q = SafeQueue(ctx=multiprocess.get_context())
            q.put(not_serializable_obj)

            # Verify that q is still functioning correctly
            q.put(True)
            self.assertTrue(q.get(timeout=1.0))

        # Assert that the serialization and the hook have been called correctly
        self.assertTrue(not_serializable_obj.reduce_was_called)
        self.assertTrue(not_serializable_obj.on_queue_feeder_error_was_called)
#
#
#

class _TestLock(BaseTestCase):

    def test_lock(self):
        lock = self.Lock()
        self.assertEqual(lock.acquire(), True)
        self.assertEqual(lock.acquire(False), False)
        self.assertEqual(lock.release(), None)
        self.assertRaises((ValueError, threading.ThreadError), lock.release)

    def test_rlock(self):
        lock = self.RLock()
        self.assertEqual(lock.acquire(), True)
        self.assertEqual(lock.acquire(), True)
        self.assertEqual(lock.acquire(), True)
        self.assertEqual(lock.release(), None)
        self.assertEqual(lock.release(), None)
        self.assertEqual(lock.release(), None)
        self.assertRaises((AssertionError, RuntimeError), lock.release)

    def test_lock_context(self):
        with self.Lock():
            pass


class _TestSemaphore(BaseTestCase):

    def _test_semaphore(self, sem):
        self.assertReturnsIfImplemented(2, get_value, sem)
        self.assertEqual(sem.acquire(), True)
        self.assertReturnsIfImplemented(1, get_value, sem)
        self.assertEqual(sem.acquire(), True)
        self.assertReturnsIfImplemented(0, get_value, sem)
        self.assertEqual(sem.acquire(False), False)
        self.assertReturnsIfImplemented(0, get_value, sem)
        self.assertEqual(sem.release(), None)
        self.assertReturnsIfImplemented(1, get_value, sem)
        self.assertEqual(sem.release(), None)
        self.assertReturnsIfImplemented(2, get_value, sem)

    def test_semaphore(self):
        sem = self.Semaphore(2)
        self._test_semaphore(sem)
        self.assertEqual(sem.release(), None)
        self.assertReturnsIfImplemented(3, get_value, sem)
        self.assertEqual(sem.release(), None)
        self.assertReturnsIfImplemented(4, get_value, sem)

    def test_bounded_semaphore(self):
        sem = self.BoundedSemaphore(2)
        self._test_semaphore(sem)
        # Currently fails on OS/X
        #if HAVE_GETVALUE:
        #    self.assertRaises(ValueError, sem.release)
        #    self.assertReturnsIfImplemented(2, get_value, sem)

    def test_timeout(self):
        if self.TYPE != 'processes':
            self.skipTest('test not appropriate for {}'.format(self.TYPE))

        sem = self.Semaphore(0)
        acquire = TimingWrapper(sem.acquire)

        self.assertEqual(acquire(False), False)
        self.assertTimingAlmostEqual(acquire.elapsed, 0.0)

        self.assertEqual(acquire(False, None), False)
        self.assertTimingAlmostEqual(acquire.elapsed, 0.0)

        self.assertEqual(acquire(False, TIMEOUT1), False)
        self.assertTimingAlmostEqual(acquire.elapsed, 0)

        self.assertEqual(acquire(True, TIMEOUT2), False)
        self.assertTimingAlmostEqual(acquire.elapsed, TIMEOUT2)

        self.assertEqual(acquire(timeout=TIMEOUT3), False)
        self.assertTimingAlmostEqual(acquire.elapsed, TIMEOUT3)


class _TestCondition(BaseTestCase):

    @classmethod
    def f(cls, cond, sleeping, woken, timeout=None):
        cond.acquire()
        sleeping.release()
        cond.wait(timeout)
        woken.release()
        cond.release()

    def assertReachesEventually(self, func, value):
        for i in range(10):
            try:
                if func() == value:
                    break
            except NotImplementedError:
                break
            time.sleep(DELTA)
        time.sleep(DELTA)
        self.assertReturnsIfImplemented(value, func)

    def check_invariant(self, cond):
        # this is only supposed to succeed when there are no sleepers
        if self.TYPE == 'processes':
            try:
                sleepers = (cond._sleeping_count.get_value() -
                            cond._woken_count.get_value())
                self.assertEqual(sleepers, 0)
                self.assertEqual(cond._wait_semaphore.get_value(), 0)
            except NotImplementedError:
                pass

    def test_notify(self):
        cond = self.Condition()
        sleeping = self.Semaphore(0)
        woken = self.Semaphore(0)

        p = self.Process(target=self.f, args=(cond, sleeping, woken))
        p.daemon = True
        p.start()
        self.addCleanup(p.join)

        p = threading.Thread(target=self.f, args=(cond, sleeping, woken))
        p.daemon = True
        p.start()
        self.addCleanup(p.join)

        # wait for both children to start sleeping
        sleeping.acquire()
        sleeping.acquire()

        # check no process/thread has woken up
        time.sleep(DELTA)
        self.assertReturnsIfImplemented(0, get_value, woken)

        # wake up one process/thread
        cond.acquire()
        cond.notify()
        cond.release()

        # check one process/thread has woken up
        time.sleep(DELTA)
        self.assertReturnsIfImplemented(1, get_value, woken)

        # wake up another
        cond.acquire()
        cond.notify()
        cond.release()

        # check other has woken up
        time.sleep(DELTA)
        self.assertReturnsIfImplemented(2, get_value, woken)

        # check state is not mucked up
        self.check_invariant(cond)
        p.join()

    def test_notify_all(self):
        cond = self.Condition()
        sleeping = self.Semaphore(0)
        woken = self.Semaphore(0)

        # start some threads/processes which will timeout
        for i in range(3):
            p = self.Process(target=self.f,
                             args=(cond, sleeping, woken, TIMEOUT1))
            p.daemon = True
            p.start()
            self.addCleanup(p.join)

            t = threading.Thread(target=self.f,
                                 args=(cond, sleeping, woken, TIMEOUT1))
            t.daemon = True
            t.start()
            self.addCleanup(t.join)

        # wait for them all to sleep
        for i in range(6):
            sleeping.acquire()

        # check they have all timed out
        for i in range(6):
            woken.acquire()
        self.assertReturnsIfImplemented(0, get_value, woken)

        # check state is not mucked up
        self.check_invariant(cond)

        # start some more threads/processes
        for i in range(3):
            p = self.Process(target=self.f, args=(cond, sleeping, woken))
            p.daemon = True
            p.start()
            self.addCleanup(p.join)

            t = threading.Thread(target=self.f, args=(cond, sleeping, woken))
            t.daemon = True
            t.start()
            self.addCleanup(t.join)

        # wait for them to all sleep
        for i in range(6):
            sleeping.acquire()

        # check no process/thread has woken up
        time.sleep(DELTA)
        self.assertReturnsIfImplemented(0, get_value, woken)

        # wake them all up
        cond.acquire()
        cond.notify_all()
        cond.release()

        # check they have all woken
        self.assertReachesEventually(lambda: get_value(woken), 6)

        # check state is not mucked up
        self.check_invariant(cond)

    def test_notify_n(self):
        cond = self.Condition()
        sleeping = self.Semaphore(0)
        woken = self.Semaphore(0)

        # start some threads/processes
        for i in range(3):
            p = self.Process(target=self.f, args=(cond, sleeping, woken))
            p.daemon = True
            p.start()
            self.addCleanup(p.join)

            t = threading.Thread(target=self.f, args=(cond, sleeping, woken))
            t.daemon = True
            t.start()
            self.addCleanup(t.join)

        # wait for them to all sleep
        for i in range(6):
            sleeping.acquire()

        # check no process/thread has woken up
        time.sleep(DELTA)
        self.assertReturnsIfImplemented(0, get_value, woken)

        # wake some of them up
        cond.acquire()
        cond.notify(n=2)
        cond.release()

        # check 2 have woken
        self.assertReachesEventually(lambda: get_value(woken), 2)

        # wake the rest of them
        cond.acquire()
        cond.notify(n=4)
        cond.release()

        self.assertReachesEventually(lambda: get_value(woken), 6)

        # doesn't do anything more
        cond.acquire()
        cond.notify(n=3)
        cond.release()

        self.assertReturnsIfImplemented(6, get_value, woken)

        # check state is not mucked up
        self.check_invariant(cond)

    def test_timeout(self):
        cond = self.Condition()
        wait = TimingWrapper(cond.wait)
        cond.acquire()
        res = wait(TIMEOUT1)
        cond.release()
        self.assertEqual(res, False)
        self.assertTimingAlmostEqual(wait.elapsed, TIMEOUT1)

    @classmethod
    def _test_waitfor_f(cls, cond, state):
        with cond:
            state.value = 0
            cond.notify()
            result = cond.wait_for(lambda : state.value==4)
            if not result or state.value != 4:
                sys.exit(1)

    @unittest.skipUnless(HAS_SHAREDCTYPES, 'needs sharedctypes')
    def test_waitfor(self):
        # based on test in test/lock_tests.py
        cond = self.Condition()
        state = self.Value('i', -1)

        p = self.Process(target=self._test_waitfor_f, args=(cond, state))
        p.daemon = True
        p.start()

        with cond:
            result = cond.wait_for(lambda : state.value==0)
            self.assertTrue(result)
            self.assertEqual(state.value, 0)

        for i in range(4):
            time.sleep(0.01)
            with cond:
                state.value += 1
                cond.notify()

        join_process(p)
        self.assertEqual(p.exitcode, 0)

    @classmethod
    def _test_waitfor_timeout_f(cls, cond, state, success, sem):
        sem.release()
        with cond:
            expected = 0.1
            dt = getattr(time,'monotonic',time.time)()
            result = cond.wait_for(lambda : state.value==4, timeout=expected)
            dt = getattr(time,'monotonic',time.time)() - dt
            # borrow logic in assertTimeout() from test/lock_tests.py
            if not result and expected * 0.6 < dt < expected * 10.0:
                success.value = True

    @unittest.skipUnless(HAS_SHAREDCTYPES, 'needs sharedctypes')
    def test_waitfor_timeout(self):
        # based on test in test/lock_tests.py
        cond = self.Condition()
        state = self.Value('i', 0)
        success = self.Value('i', False)
        sem = self.Semaphore(0)

        p = self.Process(target=self._test_waitfor_timeout_f,
                         args=(cond, state, success, sem))
        p.daemon = True
        p.start()
        self.assertTrue(sem.acquire(timeout=TIMEOUT))

        # Only increment 3 times, so state == 4 is never reached.
        for i in range(3):
            time.sleep(0.01)
            with cond:
                state.value += 1
                cond.notify()

        join_process(p)
        self.assertTrue(success.value)

    @classmethod
    def _test_wait_result(cls, c, pid):
        with c:
            c.notify()
        time.sleep(1)
        if pid is not None:
            os.kill(pid, signal.SIGINT)

    def test_wait_result(self):
        if isinstance(self, ProcessesMixin) and sys.platform != 'win32':
            pid = os.getpid()
        else:
            pid = None

        c = self.Condition()
        with c:
            self.assertFalse(c.wait(0))
            self.assertFalse(c.wait(0.1))

            p = self.Process(target=self._test_wait_result, args=(c, pid))
            p.start()

            self.assertTrue(c.wait(60))
            if pid is not None:
                self.assertRaises(KeyboardInterrupt, c.wait, 60)

            p.join()


class _TestEvent(BaseTestCase):

    @classmethod
    def _test_event(cls, event):
        time.sleep(TIMEOUT2)
        event.set()

    def test_event(self):
        event = self.Event()
        wait = TimingWrapper(event.wait)

        # Removed temporarily, due to API shear, this does not
        # work with threading._Event objects. is_set == isSet
        self.assertEqual(event.is_set(), False)

        # Removed, threading.Event.wait() will return the value of the __flag
        # instead of None. API Shear with the semaphore backed mp.Event
        self.assertEqual(wait(0.0), False)
        self.assertTimingAlmostEqual(wait.elapsed, 0.0)
        self.assertEqual(wait(TIMEOUT1), False)
        self.assertTimingAlmostEqual(wait.elapsed, TIMEOUT1)

        event.set()

        # See note above on the API differences
        self.assertEqual(event.is_set(), True)
        self.assertEqual(wait(), True)
        self.assertTimingAlmostEqual(wait.elapsed, 0.0)
        self.assertEqual(wait(TIMEOUT1), True)
        self.assertTimingAlmostEqual(wait.elapsed, 0.0)
        # self.assertEqual(event.is_set(), True)

        event.clear()

        #self.assertEqual(event.is_set(), False)

        p = self.Process(target=self._test_event, args=(event,))
        p.daemon = True
        p.start()
        self.assertEqual(wait(), True)
        p.join()

#
# Tests for Barrier - adapted from tests in test/lock_tests.py
#

# Many of the tests for threading.Barrier use a list as an atomic
# counter: a value is appended to increment the counter, and the
# length of the list gives the value.  We use the class DummyList
# for the same purpose.

class _DummyList(object):

    def __init__(self):
        wrapper = multiprocess.heap.BufferWrapper(struct.calcsize('i'))
        lock = multiprocess.Lock()
        self.__setstate__((wrapper, lock))
        self._lengthbuf[0] = 0

    def __setstate__(self, state):
        (self._wrapper, self._lock) = state
        self._lengthbuf = self._wrapper.create_memoryview().cast('i')

    def __getstate__(self):
        return (self._wrapper, self._lock)

    def append(self, _):
        with self._lock:
            self._lengthbuf[0] += 1

    def __len__(self):
        with self._lock:
            return self._lengthbuf[0]

def _wait():
    # A crude wait/yield function not relying on synchronization primitives.
    time.sleep(0.01)


class Bunch(object):
    """
    A bunch of threads.
    """
    def __init__(self, namespace, f, args, n, wait_before_exit=False):
        """
        Construct a bunch of `n` threads running the same function `f`.
        If `wait_before_exit` is True, the threads won't terminate until
        do_finish() is called.
        """
        self.f = f
        self.args = args
        self.n = n
        self.started = namespace.DummyList()
        self.finished = namespace.DummyList()
        self._can_exit = namespace.Event()
        if not wait_before_exit:
            self._can_exit.set()

        threads = []
        for i in range(n):
            p = namespace.Process(target=self.task)
            p.daemon = True
            p.start()
            threads.append(p)

        def finalize(threads):
            for p in threads:
                p.join()

        self._finalizer = weakref.finalize(self, finalize, threads)

    def task(self):
        pid = os.getpid()
        self.started.append(pid)
        try:
            self.f(*self.args)
        finally:
            self.finished.append(pid)
            self._can_exit.wait(30)
            assert self._can_exit.is_set()

    def wait_for_started(self):
        while len(self.started) < self.n:
            _wait()

    def wait_for_finished(self):
        while len(self.finished) < self.n:
            _wait()

    def do_finish(self):
        self._can_exit.set()

    def close(self):
        self._finalizer()


class AppendTrue(object):
    def __init__(self, obj):
        self.obj = obj
    def __call__(self):
        self.obj.append(True)


class _TestBarrier(BaseTestCase):
    """
    Tests for Barrier objects.
    """
    N = 5
    defaultTimeout = 30.0  # XXX Slow Windows buildbots need generous timeout

    def setUp(self):
        self.barrier = self.Barrier(self.N, timeout=self.defaultTimeout)

    def tearDown(self):
        self.barrier.abort()
        self.barrier = None

    def DummyList(self):
        if self.TYPE == 'threads':
            return []
        elif self.TYPE == 'manager':
            return self.manager.list()
        else:
            return _DummyList()

    def run_threads(self, f, args):
        b = Bunch(self, f, args, self.N-1)
        try:
            f(*args)
            b.wait_for_finished()
        finally:
            b.close()

    @classmethod
    def multipass(cls, barrier, results, n):
        m = barrier.parties
        assert m == cls.N
        for i in range(n):
            results[0].append(True)
            assert len(results[1]) == i * m
            barrier.wait()
            results[1].append(True)
            assert len(results[0]) == (i + 1) * m
            barrier.wait()
        try:
            assert barrier.n_waiting == 0
        except NotImplementedError:
            pass
        assert not barrier.broken

    def test_barrier(self, passes=1):
        """
        Test that a barrier is passed in lockstep
        """
        results = [self.DummyList(), self.DummyList()]
        self.run_threads(self.multipass, (self.barrier, results, passes))

    def test_barrier_10(self):
        """
        Test that a barrier works for 10 consecutive runs
        """
        return self.test_barrier(10)

    @classmethod
    def _test_wait_return_f(cls, barrier, queue):
        res = barrier.wait()
        queue.put(res)

    def test_wait_return(self):
        """
        test the return value from barrier.wait
        """
        queue = self.Queue()
        self.run_threads(self._test_wait_return_f, (self.barrier, queue))
        results = [queue.get() for i in range(self.N)]
        self.assertEqual(results.count(0), 1)
        close_queue(queue)

    @classmethod
    def _test_action_f(cls, barrier, results):
        barrier.wait()
        if len(results) != 1:
            raise RuntimeError

    def test_action(self):
        """
        Test the 'action' callback
        """
        results = self.DummyList()
        barrier = self.Barrier(self.N, action=AppendTrue(results))
        self.run_threads(self._test_action_f, (barrier, results))
        self.assertEqual(len(results), 1)

    @classmethod
    def _test_abort_f(cls, barrier, results1, results2):
        try:
            i = barrier.wait()
            if i == cls.N//2:
                raise RuntimeError
            barrier.wait()
            results1.append(True)
        except threading.BrokenBarrierError:
            results2.append(True)
        except RuntimeError:
            barrier.abort()

    def test_abort(self):
        """
        Test that an abort will put the barrier in a broken state
        """
        results1 = self.DummyList()
        results2 = self.DummyList()
        self.run_threads(self._test_abort_f,
                         (self.barrier, results1, results2))
        self.assertEqual(len(results1), 0)
        self.assertEqual(len(results2), self.N-1)
        self.assertTrue(self.barrier.broken)

    @classmethod
    def _test_reset_f(cls, barrier, results1, results2, results3):
        i = barrier.wait()
        if i == cls.N//2:
            # Wait until the other threads are all in the barrier.
            while barrier.n_waiting < cls.N-1:
                time.sleep(0.001)
            barrier.reset()
        else:
            try:
                barrier.wait()
                results1.append(True)
            except threading.BrokenBarrierError:
                results2.append(True)
        # Now, pass the barrier again
        barrier.wait()
        results3.append(True)

    def test_reset(self):
        """
        Test that a 'reset' on a barrier frees the waiting threads
        """
        results1 = self.DummyList()
        results2 = self.DummyList()
        results3 = self.DummyList()
        self.run_threads(self._test_reset_f,
                         (self.barrier, results1, results2, results3))
        self.assertEqual(len(results1), 0)
        self.assertEqual(len(results2), self.N-1)
        self.assertEqual(len(results3), self.N)

    @classmethod
    def _test_abort_and_reset_f(cls, barrier, barrier2,
                                results1, results2, results3):
        try:
            i = barrier.wait()
            if i == cls.N//2:
                raise RuntimeError
            barrier.wait()
            results1.append(True)
        except threading.BrokenBarrierError:
            results2.append(True)
        except RuntimeError:
            barrier.abort()
        # Synchronize and reset the barrier.  Must synchronize first so
        # that everyone has left it when we reset, and after so that no
        # one enters it before the reset.
        if barrier2.wait() == cls.N//2:
            barrier.reset()
        barrier2.wait()
        barrier.wait()
        results3.append(True)

    def test_abort_and_reset(self):
        """
        Test that a barrier can be reset after being broken.
        """
        results1 = self.DummyList()
        results2 = self.DummyList()
        results3 = self.DummyList()
        barrier2 = self.Barrier(self.N)

        self.run_threads(self._test_abort_and_reset_f,
                         (self.barrier, barrier2, results1, results2, results3))
        self.assertEqual(len(results1), 0)
        self.assertEqual(len(results2), self.N-1)
        self.assertEqual(len(results3), self.N)

    @classmethod
    def _test_timeout_f(cls, barrier, results):
        i = barrier.wait()
        if i == cls.N//2:
            # One thread is late!
            time.sleep(1.0)
        try:
            barrier.wait(0.5)
        except threading.BrokenBarrierError:
            results.append(True)

    def test_timeout(self):
        """
        Test wait(timeout)
        """
        results = self.DummyList()
        self.run_threads(self._test_timeout_f, (self.barrier, results))
        self.assertEqual(len(results), self.barrier.parties)

    @classmethod
    def _test_default_timeout_f(cls, barrier, results):
        i = barrier.wait(cls.defaultTimeout)
        if i == cls.N//2:
            # One thread is later than the default timeout
            time.sleep(1.0)
        try:
            barrier.wait()
        except threading.BrokenBarrierError:
            results.append(True)

    def test_default_timeout(self):
        """
        Test the barrier's default timeout
        """
        barrier = self.Barrier(self.N, timeout=0.5)
        results = self.DummyList()
        self.run_threads(self._test_default_timeout_f, (barrier, results))
        self.assertEqual(len(results), barrier.parties)

    def test_single_thread(self):
        b = self.Barrier(1)
        b.wait()
        b.wait()

    @classmethod
    def _test_thousand_f(cls, barrier, passes, conn, lock):
        for i in range(passes):
            barrier.wait()
            with lock:
                conn.send(i)

    def test_thousand(self):
        if self.TYPE == 'manager':
            self.skipTest('test not appropriate for {}'.format(self.TYPE))
        passes = 1000
        lock = self.Lock()
        conn, child_conn = self.Pipe(False)
        for j in range(self.N):
            p = self.Process(target=self._test_thousand_f,
                           args=(self.barrier, passes, child_conn, lock))
            p.start()
            self.addCleanup(p.join)

        for i in range(passes):
            for j in range(self.N):
                self.assertEqual(conn.recv(), i)

#
#
#

class _TestValue(BaseTestCase):

    ALLOWED_TYPES = ('processes',)

    codes_values = [
        ('i', 4343, 24234),
        ('d', 3.625, -4.25),
        ('h', -232, 234),
        ('q', 2 ** 33, 2 ** 34),
        ('c', latin('x'), latin('y'))
        ]

    def setUp(self):
        if not HAS_SHAREDCTYPES:
            self.skipTest("requires multiprocess.sharedctypes")

    @classmethod
    def _test(cls, values):
        for sv, cv in zip(values, cls.codes_values):
            sv.value = cv[2]


    def test_value(self, raw=False):
        if raw:
            values = [self.RawValue(code, value)
                      for code, value, _ in self.codes_values]
        else:
            values = [self.Value(code, value)
                      for code, value, _ in self.codes_values]

        for sv, cv in zip(values, self.codes_values):
            self.assertEqual(sv.value, cv[1])

        proc = self.Process(target=self._test, args=(values,))
        proc.daemon = True
        proc.start()
        proc.join()

        for sv, cv in zip(values, self.codes_values):
            self.assertEqual(sv.value, cv[2])

    def test_rawvalue(self):
        self.test_value(raw=True)

    def test_getobj_getlock(self):
        val1 = self.Value('i', 5)
        lock1 = val1.get_lock()
        obj1 = val1.get_obj()

        val2 = self.Value('i', 5, lock=None)
        lock2 = val2.get_lock()
        obj2 = val2.get_obj()

        lock = self.Lock()
        val3 = self.Value('i', 5, lock=lock)
        lock3 = val3.get_lock()
        obj3 = val3.get_obj()
        self.assertEqual(lock, lock3)

        arr4 = self.Value('i', 5, lock=False)
        self.assertFalse(hasattr(arr4, 'get_lock'))
        self.assertFalse(hasattr(arr4, 'get_obj'))

        self.assertRaises(AttributeError, self.Value, 'i', 5, lock='navalue')

        arr5 = self.RawValue('i', 5)
        self.assertFalse(hasattr(arr5, 'get_lock'))
        self.assertFalse(hasattr(arr5, 'get_obj'))


class _TestArray(BaseTestCase):

    ALLOWED_TYPES = ('processes',)

    @classmethod
    def f(cls, seq):
        for i in range(1, len(seq)):
            seq[i] += seq[i-1]

    @unittest.skipIf(c_int is None, "requires _ctypes")
    def test_array(self, raw=False):
        seq = [680, 626, 934, 821, 150, 233, 548, 982, 714, 831]
        if raw:
            arr = self.RawArray('i', seq)
        else:
            arr = self.Array('i', seq)

        self.assertEqual(len(arr), len(seq))
        self.assertEqual(arr[3], seq[3])
        self.assertEqual(list(arr[2:7]), list(seq[2:7]))

        arr[4:8] = seq[4:8] = array.array('i', [1, 2, 3, 4])

        self.assertEqual(list(arr[:]), seq)

        self.f(seq)

        p = self.Process(target=self.f, args=(arr,))
        p.daemon = True
        p.start()
        p.join()

        self.assertEqual(list(arr[:]), seq)

    @unittest.skipIf(c_int is None, "requires _ctypes")
    def test_array_from_size(self):
        size = 10
        # Test for zeroing (see issue #11675).
        # The repetition below strengthens the test by increasing the chances
        # of previously allocated non-zero memory being used for the new array
        # on the 2nd and 3rd loops.
        for _ in range(3):
            arr = self.Array('i', size)
            self.assertEqual(len(arr), size)
            self.assertEqual(list(arr), [0] * size)
            arr[:] = range(10)
            self.assertEqual(list(arr), list(range(10)))
            del arr

    @unittest.skipIf(c_int is None, "requires _ctypes")
    def test_rawarray(self):
        self.test_array(raw=True)

    @unittest.skipIf(c_int is None, "requires _ctypes")
    def test_getobj_getlock_obj(self):
        arr1 = self.Array('i', list(range(10)))
        lock1 = arr1.get_lock()
        obj1 = arr1.get_obj()

        arr2 = self.Array('i', list(range(10)), lock=None)
        lock2 = arr2.get_lock()
        obj2 = arr2.get_obj()

        lock = self.Lock()
        arr3 = self.Array('i', list(range(10)), lock=lock)
        lock3 = arr3.get_lock()
        obj3 = arr3.get_obj()
        self.assertEqual(lock, lock3)

        arr4 = self.Array('i', range(10), lock=False)
        self.assertFalse(hasattr(arr4, 'get_lock'))
        self.assertFalse(hasattr(arr4, 'get_obj'))
        self.assertRaises(AttributeError,
                          self.Array, 'i', range(10), lock='notalock')

        arr5 = self.RawArray('i', range(10))
        self.assertFalse(hasattr(arr5, 'get_lock'))
        self.assertFalse(hasattr(arr5, 'get_obj'))

#
#
#

class _TestContainers(BaseTestCase):

    ALLOWED_TYPES = ('manager',)

    def test_list(self):
        a = self.list(list(range(10)))
        self.assertEqual(a[:], list(range(10)))

        b = self.list()
        self.assertEqual(b[:], [])

        b.extend(list(range(5)))
        self.assertEqual(b[:], list(range(5)))

        self.assertEqual(b[2], 2)
        self.assertEqual(b[2:10], [2,3,4])

        b *= 2
        self.assertEqual(b[:], [0, 1, 2, 3, 4, 0, 1, 2, 3, 4])

        self.assertEqual(b + [5, 6], [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6])

        self.assertEqual(a[:], list(range(10)))

        d = [a, b]
        e = self.list(d)
        self.assertEqual(
            [element[:] for element in e],
            [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]]
            )

        f = self.list([a])
        a.append('hello')
        self.assertEqual(f[0][:], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 'hello'])

    def test_list_iter(self):
        a = self.list(list(range(10)))
        it = iter(a)
        self.assertEqual(list(it), list(range(10)))
        self.assertEqual(list(it), [])  # exhausted
        # list modified during iteration
        it = iter(a)
        a[0] = 100
        self.assertEqual(next(it), 100)

    def test_list_proxy_in_list(self):
        a = self.list([self.list(range(3)) for _i in range(3)])
        self.assertEqual([inner[:] for inner in a], [[0, 1, 2]] * 3)

        a[0][-1] = 55
        self.assertEqual(a[0][:], [0, 1, 55])
        for i in range(1, 3):
            self.assertEqual(a[i][:], [0, 1, 2])

        self.assertEqual(a[1].pop(), 2)
        self.assertEqual(len(a[1]), 2)
        for i in range(0, 3, 2):
            self.assertEqual(len(a[i]), 3)

        del a

        b = self.list()
        b.append(b)
        del b

    def test_dict(self):
        d = self.dict()
        indices = list(range(65, 70))
        for i in indices:
            d[i] = chr(i)
        self.assertEqual(d.copy(), dict((i, chr(i)) for i in indices))
        self.assertEqual(sorted(d.keys()), indices)
        self.assertEqual(sorted(d.values()), [chr(i) for i in indices])
        self.assertEqual(sorted(d.items()), [(i, chr(i)) for i in indices])

    def test_dict_iter(self):
        d = self.dict()
        indices = list(range(65, 70))
        for i in indices:
            d[i] = chr(i)
        it = iter(d)
        self.assertEqual(list(it), indices)
        self.assertEqual(list(it), [])  # exhausted
        # dictionary changed size during iteration
        it = iter(d)
        d.clear()
        self.assertRaises(RuntimeError, next, it)

    def test_dict_proxy_nested(self):
        pets = self.dict(ferrets=2, hamsters=4)
        supplies = self.dict(water=10, feed=3)
        d = self.dict(pets=pets, supplies=supplies)

        self.assertEqual(supplies['water'], 10)
        self.assertEqual(d['supplies']['water'], 10)

        d['supplies']['blankets'] = 5
        self.assertEqual(supplies['blankets'], 5)
        self.assertEqual(d['supplies']['blankets'], 5)

        d['supplies']['water'] = 7
        self.assertEqual(supplies['water'], 7)
        self.assertEqual(d['supplies']['water'], 7)

        del pets
        del supplies
        self.assertEqual(d['pets']['ferrets'], 2)
        d['supplies']['blankets'] = 11
        self.assertEqual(d['supplies']['blankets'], 11)

        pets = d['pets']
        supplies = d['supplies']
        supplies['water'] = 7
        self.assertEqual(supplies['water'], 7)
        self.assertEqual(d['supplies']['water'], 7)

        d.clear()
        self.assertEqual(len(d), 0)
        self.assertEqual(supplies['water'], 7)
        self.assertEqual(pets['hamsters'], 4)

        l = self.list([pets, supplies])
        l[0]['marmots'] = 1
        self.assertEqual(pets['marmots'], 1)
        self.assertEqual(l[0]['marmots'], 1)

        del pets
        del supplies
        self.assertEqual(l[0]['marmots'], 1)

        outer = self.list([[88, 99], l])
        self.assertIsInstance(outer[0], list)  # Not a ListProxy
        self.assertEqual(outer[-1][-1]['feed'], 3)

    def test_namespace(self):
        n = self.Namespace()
        n.name = 'Bob'
        n.job = 'Builder'
        n._hidden = 'hidden'
        self.assertEqual((n.name, n.job), ('Bob', 'Builder'))
        del n.job
        self.assertEqual(str(n), "Namespace(name='Bob')")
        self.assertTrue(hasattr(n, 'name'))
        self.assertTrue(not hasattr(n, 'job'))

#
#
#

def sqr(x, wait=0.0):
    time.sleep(wait)
    return x*x

def mul(x, y):
    return x*y

def raise_large_valuerror(wait):
    time.sleep(wait)
    raise ValueError("x" * 1024**2)

def identity(x):
    return x

class CountedObject(object):
    n_instances = 0

    def __new__(cls):
        cls.n_instances += 1
        return object.__new__(cls)

    def __del__(self):
        type(self).n_instances -= 1

class SayWhenError(ValueError): pass

def exception_throwing_generator(total, when):
    if when == -1:
        raise SayWhenError("Somebody said when")
    for i in range(total):
        if i == when:
            raise SayWhenError("Somebody said when")
        yield i


class _TestPool(BaseTestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.pool = cls.Pool(4)

    @classmethod
    def tearDownClass(cls):
        cls.pool.terminate()
        cls.pool.join()
        cls.pool = None
        super().tearDownClass()

    def test_apply(self):
        papply = self.pool.apply
        self.assertEqual(papply(sqr, (5,)), sqr(5))
        self.assertEqual(papply(sqr, (), {'x':3}), sqr(x=3))

    def test_map(self):
        pmap = self.pool.map
        self.assertEqual(pmap(sqr, list(range(10))), list(map(sqr, list(range(10)))))
        self.assertEqual(pmap(sqr, list(range(100)), chunksize=20),
                         list(map(sqr, list(range(100)))))

    def test_starmap(self):
        psmap = self.pool.starmap
        tuples = list(zip(range(10), range(9,-1, -1)))
        self.assertEqual(psmap(mul, tuples),
                         list(itertools.starmap(mul, tuples)))
        tuples = list(zip(range(100), range(99,-1, -1)))
        self.assertEqual(psmap(mul, tuples, chunksize=20),
                         list(itertools.starmap(mul, tuples)))

    def test_starmap_async(self):
        tuples = list(zip(range(100), range(99,-1, -1)))
        self.assertEqual(self.pool.starmap_async(mul, tuples).get(),
                         list(itertools.starmap(mul, tuples)))

    def test_map_async(self):
        self.assertEqual(self.pool.map_async(sqr, list(range(10))).get(),
                         list(map(sqr, list(range(10)))))

    def test_map_async_callbacks(self):
        call_args = self.manager.list() if self.TYPE == 'manager' else []
        self.pool.map_async(int, ['1'],
                            callback=call_args.append,
                            error_callback=call_args.append).wait()
        self.assertEqual(1, len(call_args))
        self.assertEqual([1], call_args[0])
        self.pool.map_async(int, ['a'],
                            callback=call_args.append,
                            error_callback=call_args.append).wait()
        self.assertEqual(2, len(call_args))
        self.assertIsInstance(call_args[1], ValueError)

    def test_map_unplicklable(self):
        # Issue #19425 -- failure to pickle should not cause a hang
        if self.TYPE == 'threads':
            self.skipTest('test not appropriate for {}'.format(self.TYPE))
        class A(object):
            def __reduce__(self):
                raise RuntimeError('cannot pickle')
        with self.assertRaises(RuntimeError):
            self.pool.map(sqr, [A()]*10)

    def test_map_chunksize(self):
        try:
            self.pool.map_async(sqr, [], chunksize=1).get(timeout=TIMEOUT1)
        except multiprocess.TimeoutError:
            self.fail("pool.map_async with chunksize stalled on null list")

    def test_map_handle_iterable_exception(self):
        if self.TYPE == 'manager':
            self.skipTest('test not appropriate for {}'.format(self.TYPE))

        # SayWhenError seen at the very first of the iterable
        with self.assertRaises(SayWhenError):
            self.pool.map(sqr, exception_throwing_generator(1, -1), 1)
        # again, make sure it's reentrant
        with self.assertRaises(SayWhenError):
            self.pool.map(sqr, exception_throwing_generator(1, -1), 1)

        with self.assertRaises(SayWhenError):
            self.pool.map(sqr, exception_throwing_generator(10, 3), 1)

        class SpecialIterable:
            def __iter__(self):
                return self
            def __next__(self):
                raise SayWhenError
            def __len__(self):
                return 1
        with self.assertRaises(SayWhenError):
            self.pool.map(sqr, SpecialIterable(), 1)
        with self.assertRaises(SayWhenError):
            self.pool.map(sqr, SpecialIterable(), 1)

    def test_async(self):
        res = self.pool.apply_async(sqr, (7, TIMEOUT1,))
        get = TimingWrapper(res.get)
        self.assertEqual(get(), 49)
        self.assertTimingAlmostEqual(get.elapsed, TIMEOUT1)

    def test_async_timeout(self):
        res = self.pool.apply_async(sqr, (6, TIMEOUT2 + 1.0))
        get = TimingWrapper(res.get)
        self.assertRaises(multiprocess.TimeoutError, get, timeout=TIMEOUT2)
        self.assertTimingAlmostEqual(get.elapsed, TIMEOUT2)

    def test_imap(self):
        it = self.pool.imap(sqr, list(range(10)))
        self.assertEqual(list(it), list(map(sqr, list(range(10)))))

        it = self.pool.imap(sqr, list(range(10)))
        for i in range(10):
            self.assertEqual(next(it), i*i)
        self.assertRaises(StopIteration, it.__next__)

        it = self.pool.imap(sqr, list(range(1000)), chunksize=100)
        for i in range(1000):
            self.assertEqual(next(it), i*i)
        self.assertRaises(StopIteration, it.__next__)

    def test_imap_handle_iterable_exception(self):
        if self.TYPE == 'manager':
            self.skipTest('test not appropriate for {}'.format(self.TYPE))

        # SayWhenError seen at the very first of the iterable
        it = self.pool.imap(sqr, exception_throwing_generator(1, -1), 1)
        self.assertRaises(SayWhenError, it.__next__)
        # again, make sure it's reentrant
        it = self.pool.imap(sqr, exception_throwing_generator(1, -1), 1)
        self.assertRaises(SayWhenError, it.__next__)

        it = self.pool.imap(sqr, exception_throwing_generator(10, 3), 1)
        for i in range(3):
            self.assertEqual(next(it), i*i)
        self.assertRaises(SayWhenError, it.__next__)

        # SayWhenError seen at start of problematic chunk's results
        it = self.pool.imap(sqr, exception_throwing_generator(20, 7), 2)
        for i in range(6):
            self.assertEqual(next(it), i*i)
        self.assertRaises(SayWhenError, it.__next__)
        it = self.pool.imap(sqr, exception_throwing_generator(20, 7), 4)
        for i in range(4):
            self.assertEqual(next(it), i*i)
        self.assertRaises(SayWhenError, it.__next__)

    def test_imap_unordered(self):
        it = self.pool.imap_unordered(sqr, list(range(10)))
        self.assertEqual(sorted(it), list(map(sqr, list(range(10)))))

        it = self.pool.imap_unordered(sqr, list(range(1000)), chunksize=100)
        self.assertEqual(sorted(it), list(map(sqr, list(range(1000)))))

    def test_imap_unordered_handle_iterable_exception(self):
        if self.TYPE == 'manager':
            self.skipTest('test not appropriate for {}'.format(self.TYPE))

        # SayWhenError seen at the very first of the iterable
        it = self.pool.imap_unordered(sqr,
                                      exception_throwing_generator(1, -1),
                                      1)
        self.assertRaises(SayWhenError, it.__next__)
        # again, make sure it's reentrant
        it = self.pool.imap_unordered(sqr,
                                      exception_throwing_generator(1, -1),
                                      1)
        self.assertRaises(SayWhenError, it.__next__)

        it = self.pool.imap_unordered(sqr,
                                      exception_throwing_generator(10, 3),
                                      1)
        expected_values = list(map(sqr, list(range(10))))
        with self.assertRaises(SayWhenError):
            # imap_unordered makes it difficult to anticipate the SayWhenError
            for i in range(10):
                value = next(it)
                self.assertIn(value, expected_values)
                expected_values.remove(value)

        it = self.pool.imap_unordered(sqr,
                                      exception_throwing_generator(20, 7),
                                      2)
        expected_values = list(map(sqr, list(range(20))))
        with self.assertRaises(SayWhenError):
            for i in range(20):
                value = next(it)
                self.assertIn(value, expected_values)
                expected_values.remove(value)

    def test_make_pool(self):
        expected_error = (RemoteError if self.TYPE == 'manager'
                          else ValueError)

        self.assertRaises(expected_error, self.Pool, -1)
        self.assertRaises(expected_error, self.Pool, 0)

        if self.TYPE != 'manager':
            p = self.Pool(3)
            try:
                self.assertEqual(3, len(p._pool))
            finally:
                p.close()
                p.join()

    def test_terminate(self):
        result = self.pool.map_async(
            time.sleep, [0.1 for i in range(10000)], chunksize=1
            )
        self.pool.terminate()
        join = TimingWrapper(self.pool.join)
        join()
        # Sanity check the pool didn't wait for all tasks to finish
        self.assertLess(join.elapsed, 2.0)

    def test_empty_iterable(self):
        # See Issue 12157
        p = self.Pool(1)

        self.assertEqual(p.map(sqr, []), [])
        self.assertEqual(list(p.imap(sqr, [])), [])
        self.assertEqual(list(p.imap_unordered(sqr, [])), [])
        self.assertEqual(p.map_async(sqr, []).get(), [])

        p.close()
        p.join()

    def test_context(self):
        if self.TYPE == 'processes':
            L = list(range(10))
            expected = [sqr(i) for i in L]
            with self.Pool(2) as p:
                r = p.map_async(sqr, L)
                self.assertEqual(r.get(), expected)
            p.join()
            self.assertRaises(ValueError, p.map_async, sqr, L)

    @classmethod
    def _test_traceback(cls):
        raise RuntimeError(123) # some comment

    @unittest.skipIf(True, "fails with is_dill(obj, child=True)")
    def test_traceback(self):
        # We want ensure that the traceback from the child process is
        # contained in the traceback raised in the main process.
        if self.TYPE == 'processes':
            with self.Pool(1) as p:
                try:
                    p.apply(self._test_traceback)
                except Exception as e:
                    exc = e
                else:
                    self.fail('expected RuntimeError')
            p.join()
            self.assertIs(type(exc), RuntimeError)
            self.assertEqual(exc.args, (123,))
            cause = exc.__cause__
            self.assertIs(type(cause), multiprocess.pool.RemoteTraceback)
            self.assertIn('raise RuntimeError(123) # some comment', cause.tb)

            with test.support.captured_stderr() as f1:
                try:
                    raise exc
                except RuntimeError:
                    sys.excepthook(*sys.exc_info())
            self.assertIn('raise RuntimeError(123) # some comment',
                          f1.getvalue())
            # _helper_reraises_exception should not make the error
            # a remote exception
            with self.Pool(1) as p:
                try:
                    p.map(sqr, exception_throwing_generator(1, -1), 1)
                except Exception as e:
                    exc = e
                else:
                    self.fail('expected SayWhenError')
                self.assertIs(type(exc), SayWhenError)
                self.assertIs(exc.__cause__, None)
            p.join()

    @classmethod
    def _test_wrapped_exception(cls):
        raise RuntimeError('foo')

    @unittest.skipIf(True, "fails with is_dill(obj, child=True)")
    def test_wrapped_exception(self):
        # Issue #20980: Should not wrap exception when using thread pool
        with self.Pool(1) as p:
            with self.assertRaises(RuntimeError):
                p.apply(self._test_wrapped_exception)
        p.join()

    def test_map_no_failfast(self):
        # Issue #23992: the fail-fast behaviour when an exception is raised
        # during map() would make Pool.join() deadlock, because a worker
        # process would fill the result queue (after the result handler thread
        # terminated, hence not draining it anymore).

        t_start = getattr(time,'monotonic',time.time)()

        with self.assertRaises(ValueError):
            with self.Pool(2) as p:
                try:
                    p.map(raise_large_valuerror, [0, 1])
                finally:
                    time.sleep(0.5)
                    p.close()
                    p.join()

        # check that we indeed waited for all jobs
        self.assertGreater(getattr(time,'monotonic',time.time)() - t_start, 0.9)

    def test_release_task_refs(self):
        # Issue #29861: task arguments and results should not be kept
        # alive after we are done with them.
        objs = [CountedObject() for i in range(10)]
        refs = [weakref.ref(o) for o in objs]
        self.pool.map(identity, objs)

        del objs
        time.sleep(DELTA)  # let threaded cleanup code run
        self.assertEqual(set(wr() for wr in refs), {None})
        # With a process pool, copies of the objects are returned, check
        # they were released too.
        self.assertEqual(CountedObject.n_instances, 0)


def raising():
    raise KeyError("key")

def unpickleable_result():
    return lambda: 42

class _TestPoolWorkerErrors(BaseTestCase):
    ALLOWED_TYPES = ('processes', )

    def test_async_error_callback(self):
        p = multiprocess.Pool(2)

        scratchpad = [None]
        def errback(exc):
            scratchpad[0] = exc

        res = p.apply_async(raising, error_callback=errback)
        self.assertRaises(KeyError, res.get)
        self.assertTrue(scratchpad[0])
        self.assertIsInstance(scratchpad[0], KeyError)

        p.close()
        p.join()

    def _test_unpickleable_result(self):
        from multiprocess.pool import MaybeEncodingError
        p = multiprocess.Pool(2)

        # Make sure we don't lose pool processes because of encoding errors.
        for iteration in range(20):

            scratchpad = [None]
            def errback(exc):
                scratchpad[0] = exc

            res = p.apply_async(unpickleable_result, error_callback=errback)
            self.assertRaises(MaybeEncodingError, res.get)
            wrapped = scratchpad[0]
            self.assertTrue(wrapped)
            self.assertIsInstance(scratchpad[0], MaybeEncodingError)
            self.assertIsNotNone(wrapped.exc)
            self.assertIsNotNone(wrapped.value)

        p.close()
        p.join()

class _TestPoolWorkerLifetime(BaseTestCase):
    ALLOWED_TYPES = ('processes', )

    def test_pool_worker_lifetime(self):
        p = multiprocess.Pool(3, maxtasksperchild=10)
        self.assertEqual(3, len(p._pool))
        origworkerpids = [w.pid for w in p._pool]
        # Run many tasks so each worker gets replaced (hopefully)
        results = []
        for i in range(100):
            results.append(p.apply_async(sqr, (i, )))
        # Fetch the results and verify we got the right answers,
        # also ensuring all the tasks have completed.
        for (j, res) in enumerate(results):
            self.assertEqual(res.get(), sqr(j))
        # Refill the pool
        p._repopulate_pool()
        # Wait until all workers are alive
        # (countdown * DELTA = 5 seconds max startup process time)
        countdown = 50
        while countdown and not all(w.is_alive() for w in p._pool):
            countdown -= 1
            time.sleep(DELTA)
        finalworkerpids = [w.pid for w in p._pool]
        # All pids should be assigned.  See issue #7805.
        self.assertNotIn(None, origworkerpids)
        self.assertNotIn(None, finalworkerpids)
        # Finally, check that the worker pids have changed
        self.assertNotEqual(sorted(origworkerpids), sorted(finalworkerpids))
        p.close()
        p.join()

    def test_pool_worker_lifetime_early_close(self):
        # Issue #10332: closing a pool whose workers have limited lifetimes
        # before all the tasks completed would make join() hang.
        p = multiprocess.Pool(3, maxtasksperchild=1)
        results = []
        for i in range(6):
            results.append(p.apply_async(sqr, (i, 0.3)))
        p.close()
        p.join()
        # check the results
        for (j, res) in enumerate(results):
            self.assertEqual(res.get(), sqr(j))

#
# Test of creating a customized manager class
#

from multiprocess.managers import BaseManager, BaseProxy, RemoteError

class FooBar(object):
    def f(self):
        return 'f()'
    def g(self):
        raise ValueError
    def _h(self):
        return '_h()'

def baz():
    for i in range(10):
        yield i*i

class IteratorProxy(BaseProxy):
    _exposed_ = ('__next__',)
    def __iter__(self):
        return self
    def __next__(self):
        return self._callmethod('__next__')

class MyManager(BaseManager):
    pass

MyManager.register('Foo', callable=FooBar)
MyManager.register('Bar', callable=FooBar, exposed=('f', '_h'))
MyManager.register('baz', callable=baz, proxytype=IteratorProxy)


class _TestMyManager(BaseTestCase):

    ALLOWED_TYPES = ('manager',)

    def test_mymanager(self):
        manager = MyManager()
        manager.start()
        self.common(manager)
        manager.shutdown()

        # bpo-30356: BaseManager._finalize_manager() sends SIGTERM
        # to the manager process if it takes longer than 1 second to stop,
        # which happens on slow buildbots.
        self.assertIn(manager._process.exitcode, (0, -signal.SIGTERM))

    def test_mymanager_context(self):
        with MyManager() as manager:
            self.common(manager)
        # bpo-30356: BaseManager._finalize_manager() sends SIGTERM
        # to the manager process if it takes longer than 1 second to stop,
        # which happens on slow buildbots.
        self.assertIn(manager._process.exitcode, (0, -signal.SIGTERM))

    def test_mymanager_context_prestarted(self):
        manager = MyManager()
        manager.start()
        with manager:
            self.common(manager)
        self.assertEqual(manager._process.exitcode, 0)

    def common(self, manager):
        foo = manager.Foo()
        bar = manager.Bar()
        baz = manager.baz()

        foo_methods = [name for name in ('f', 'g', '_h') if hasattr(foo, name)]
        bar_methods = [name for name in ('f', 'g', '_h') if hasattr(bar, name)]

        self.assertEqual(foo_methods, ['f', 'g'])
        self.assertEqual(bar_methods, ['f', '_h'])

        self.assertEqual(foo.f(), 'f()')
        self.assertRaises(ValueError, foo.g)
        self.assertEqual(foo._callmethod('f'), 'f()')
        self.assertRaises(RemoteError, foo._callmethod, '_h')

        self.assertEqual(bar.f(), 'f()')
        self.assertEqual(bar._h(), '_h()')
        self.assertEqual(bar._callmethod('f'), 'f()')
        self.assertEqual(bar._callmethod('_h'), '_h()')

        self.assertEqual(list(baz), [i*i for i in range(10)])


#
# Test of connecting to a remote server and using xmlrpclib for serialization
#

_queue = pyqueue.Queue()
def get_queue():
    return _queue

class QueueManager(BaseManager):
    '''manager class used by server process'''
QueueManager.register('get_queue', callable=get_queue)

class QueueManager2(BaseManager):
    '''manager class which specifies the same interface as QueueManager'''
QueueManager2.register('get_queue')


SERIALIZER = 'xmlrpclib'

class _TestRemoteManager(BaseTestCase):

    ALLOWED_TYPES = ('manager',)
    values = ['hello world', None, True, 2.25,
              'hall\xe5 v\xe4rlden',
              '\u043f\u0440\u0438\u0432\u0456\u0442 \u0441\u0432\u0456\u0442',
              b'hall\xe5 v\xe4rlden',
             ]
    result = values[:]

    @classmethod
    def _putter(cls, address, authkey):
        manager = QueueManager2(
            address=address, authkey=authkey, serializer=SERIALIZER
            )
        manager.connect()
        queue = manager.get_queue()
        # Note that xmlrpclib will deserialize object as a list not a tuple
        queue.put(tuple(cls.values))

    def test_remote(self):
        authkey = os.urandom(32)

        manager = QueueManager(
            address=(test.support.HOST, 0), authkey=authkey, serializer=SERIALIZER
            )
        manager.start()
        self.addCleanup(manager.shutdown)

        p = self.Process(target=self._putter, args=(manager.address, authkey))
        p.daemon = True
        p.start()

        manager2 = QueueManager2(
            address=manager.address, authkey=authkey, serializer=SERIALIZER
            )
        manager2.connect()
        queue = manager2.get_queue()

        self.assertEqual(queue.get(), self.result)

        # Because we are using xmlrpclib for serialization instead of
        # pickle this will cause a serialization error.
        self.assertRaises(Exception, queue.put, time.sleep)

        # Make queue finalizer run before the server is stopped
        del queue

class _TestManagerRestart(BaseTestCase):

    @classmethod
    def _putter(cls, address, authkey):
        manager = QueueManager(
            address=address, authkey=authkey, serializer=SERIALIZER)
        manager.connect()
        queue = manager.get_queue()
        queue.put('hello world')

    def test_rapid_restart(self):
        authkey = os.urandom(32)
        manager = QueueManager(
            address=(test.support.HOST, 0), authkey=authkey, serializer=SERIALIZER)
        try:
            srvr = manager.get_server()
            addr = srvr.address
            # Close the connection.Listener socket which gets opened as a part
            # of manager.get_server(). It's not needed for the test.
            srvr.listener.close()
            manager.start()

            p = self.Process(target=self._putter, args=(manager.address, authkey))
            p.start()
            p.join()
            queue = manager.get_queue()
            self.assertEqual(queue.get(), 'hello world')
            del queue
        finally:
            if hasattr(manager, "shutdown"):
                manager.shutdown()

        manager = QueueManager(
            address=addr, authkey=authkey, serializer=SERIALIZER)
        try:
            manager.start()
            self.addCleanup(manager.shutdown)
        except OSError as e:
            if e.errno != errno.EADDRINUSE:
                raise
            # Retry after some time, in case the old socket was lingering
            # (sporadic failure on buildbots)
            time.sleep(1.0)
            manager = QueueManager(
                address=addr, authkey=authkey, serializer=SERIALIZER)
            if hasattr(manager, "shutdown"):
                self.addCleanup(manager.shutdown)

#
#
#

SENTINEL = latin('')

class _TestConnection(BaseTestCase):

    ALLOWED_TYPES = ('processes', 'threads')

    @classmethod
    def _echo(cls, conn):
        for msg in iter(conn.recv_bytes, SENTINEL):
            conn.send_bytes(msg)
        conn.close()

    def test_connection(self):
        conn, child_conn = self.Pipe()

        p = self.Process(target=self._echo, args=(child_conn,))
        p.daemon = True
        p.start()

        seq = [1, 2.25, None]
        msg = latin('hello world')
        longmsg = msg * 10
        arr = array.array('i', list(range(4)))

        if self.TYPE == 'processes':
            self.assertEqual(type(conn.fileno()), int)

        self.assertEqual(conn.send(seq), None)
        self.assertEqual(conn.recv(), seq)

        self.assertEqual(conn.send_bytes(msg), None)
        self.assertEqual(conn.recv_bytes(), msg)

        if self.TYPE == 'processes':
            buffer = array.array('i', [0]*10)
            expected = list(arr) + [0] * (10 - len(arr))
            self.assertEqual(conn.send_bytes(arr), None)
            self.assertEqual(conn.recv_bytes_into(buffer),
                             len(arr) * buffer.itemsize)
            self.assertEqual(list(buffer), expected)

            buffer = array.array('i', [0]*10)
            expected = [0] * 3 + list(arr) + [0] * (10 - 3 - len(arr))
            self.assertEqual(conn.send_bytes(arr), None)
            self.assertEqual(conn.recv_bytes_into(buffer, 3 * buffer.itemsize),
                             len(arr) * buffer.itemsize)
            self.assertEqual(list(buffer), expected)

            buffer = bytearray(latin(' ' * 40))
            self.assertEqual(conn.send_bytes(longmsg), None)
            try:
                res = conn.recv_bytes_into(buffer)
            except multiprocess.BufferTooShort as e:
                self.assertEqual(e.args, (longmsg,))
            else:
                self.fail('expected BufferTooShort, got %s' % res)

        poll = TimingWrapper(conn.poll)

        self.assertEqual(poll(), False)
        self.assertTimingAlmostEqual(poll.elapsed, 0)

        self.assertEqual(poll(-1), False)
        self.assertTimingAlmostEqual(poll.elapsed, 0)

        self.assertEqual(poll(TIMEOUT1), False)
        self.assertTimingAlmostEqual(poll.elapsed, TIMEOUT1)

        conn.send(None)
        time.sleep(.1)

        self.assertEqual(poll(TIMEOUT1), True)
        self.assertTimingAlmostEqual(poll.elapsed, 0)

        self.assertEqual(conn.recv(), None)

        really_big_msg = latin('X') * (1024 * 1024 * 16)   # 16Mb
        conn.send_bytes(really_big_msg)
        self.assertEqual(conn.recv_bytes(), really_big_msg)

        conn.send_bytes(SENTINEL)                          # tell child to quit
        child_conn.close()

        if self.TYPE == 'processes':
            self.assertEqual(conn.readable, True)
            self.assertEqual(conn.writable, True)
            self.assertRaises(EOFError, conn.recv)
            self.assertRaises(EOFError, conn.recv_bytes)

        p.join()

    def test_duplex_false(self):
        reader, writer = self.Pipe(duplex=False)
        self.assertEqual(writer.send(1), None)
        self.assertEqual(reader.recv(), 1)
        if self.TYPE == 'processes':
            self.assertEqual(reader.readable, True)
            self.assertEqual(reader.writable, False)
            self.assertEqual(writer.readable, False)
            self.assertEqual(writer.writable, True)
            self.assertRaises(OSError, reader.send, 2)
            self.assertRaises(OSError, writer.recv)
            self.assertRaises(OSError, writer.poll)

    def test_spawn_close(self):
        # We test that a pipe connection can be closed by parent
        # process immediately after child is spawned.  On Windows this
        # would have sometimes failed on old versions because
        # child_conn would be closed before the child got a chance to
        # duplicate it.
        conn, child_conn = self.Pipe()

        p = self.Process(target=self._echo, args=(child_conn,))
        p.daemon = True
        p.start()
        child_conn.close()    # this might complete before child initializes

        msg = latin('hello')
        conn.send_bytes(msg)
        self.assertEqual(conn.recv_bytes(), msg)

        conn.send_bytes(SENTINEL)
        conn.close()
        p.join()

    def test_sendbytes(self):
        if self.TYPE != 'processes':
            self.skipTest('test not appropriate for {}'.format(self.TYPE))

        msg = latin('abcdefghijklmnopqrstuvwxyz')
        a, b = self.Pipe()

        a.send_bytes(msg)
        self.assertEqual(b.recv_bytes(), msg)

        a.send_bytes(msg, 5)
        self.assertEqual(b.recv_bytes(), msg[5:])

        a.send_bytes(msg, 7, 8)
        self.assertEqual(b.recv_bytes(), msg[7:7+8])

        a.send_bytes(msg, 26)
        self.assertEqual(b.recv_bytes(), latin(''))

        a.send_bytes(msg, 26, 0)
        self.assertEqual(b.recv_bytes(), latin(''))

        self.assertRaises(ValueError, a.send_bytes, msg, 27)

        self.assertRaises(ValueError, a.send_bytes, msg, 22, 5)

        self.assertRaises(ValueError, a.send_bytes, msg, 26, 1)

        self.assertRaises(ValueError, a.send_bytes, msg, -1)

        self.assertRaises(ValueError, a.send_bytes, msg, 4, -1)

    @classmethod
    def _is_fd_assigned(cls, fd):
        try:
            os.fstat(fd)
        except OSError as e:
            if e.errno == errno.EBADF:
                return False
            raise
        else:
            return True

    @classmethod
    def _writefd(cls, conn, data, create_dummy_fds=False):
        if create_dummy_fds:
            for i in range(0, 256):
                if not cls._is_fd_assigned(i):
                    os.dup2(conn.fileno(), i)
        fd = reduction.recv_handle(conn)
        if msvcrt:
            fd = msvcrt.open_osfhandle(fd, os.O_WRONLY)
        os.write(fd, data)
        os.close(fd)

    @unittest.skipUnless(HAS_REDUCTION, "test needs multiprocess.reduction")
    def test_fd_transfer(self):
        if self.TYPE != 'processes':
            self.skipTest("only makes sense with processes")
        conn, child_conn = self.Pipe(duplex=True)

        p = self.Process(target=self._writefd, args=(child_conn, b"foo"))
        p.daemon = True
        p.start()
        self.addCleanup(test.support.unlink, test.support.TESTFN)
        with open(test.support.TESTFN, "wb") as f:
            fd = f.fileno()
            if msvcrt:
                fd = msvcrt.get_osfhandle(fd)
            reduction.send_handle(conn, fd, p.pid)
        p.join()
        with open(test.support.TESTFN, "rb") as f:
            self.assertEqual(f.read(), b"foo")

    @unittest.skipUnless(HAS_REDUCTION, "test needs multiprocess.reduction")
    @unittest.skipIf(sys.platform == "win32",
                     "test semantics don't make sense on Windows")
    @unittest.skipIf(MAXFD <= 256,
                     "largest assignable fd number is too small")
    @unittest.skipUnless(hasattr(os, "dup2"),
                         "test needs os.dup2()")
    def test_large_fd_transfer(self):
        # With fd > 256 (issue #11657)
        if self.TYPE != 'processes':
            self.skipTest("only makes sense with processes")
        conn, child_conn = self.Pipe(duplex=True)

        p = self.Process(target=self._writefd, args=(child_conn, b"bar", True))
        p.daemon = True
        p.start()
        self.addCleanup(test.support.unlink, test.support.TESTFN)
        with open(test.support.TESTFN, "wb") as f:
            fd = f.fileno()
            for newfd in range(256, MAXFD):
                if not self._is_fd_assigned(newfd):
                    break
            else:
                self.fail("could not find an unassigned large file descriptor")
            os.dup2(fd, newfd)
            try:
                reduction.send_handle(conn, newfd, p.pid)
            finally:
                os.close(newfd)
        p.join()
        with open(test.support.TESTFN, "rb") as f:
            self.assertEqual(f.read(), b"bar")

    @classmethod
    def _send_data_without_fd(self, conn):
        os.write(conn.fileno(), b"\0")

    @unittest.skipUnless(HAS_REDUCTION, "test needs multiprocess.reduction")
    @unittest.skipIf(sys.platform == "win32", "doesn't make sense on Windows")
    def test_missing_fd_transfer(self):
        # Check that exception is raised when received data is not
        # accompanied by a file descriptor in ancillary data.
        if self.TYPE != 'processes':
            self.skipTest("only makes sense with processes")
        conn, child_conn = self.Pipe(duplex=True)

        p = self.Process(target=self._send_data_without_fd, args=(child_conn,))
        p.daemon = True
        p.start()
        self.assertRaises(RuntimeError, reduction.recv_handle, conn)
        p.join()

    def test_context(self):
        a, b = self.Pipe()

        with a, b:
            a.send(1729)
            self.assertEqual(b.recv(), 1729)
            if self.TYPE == 'processes':
                self.assertFalse(a.closed)
                self.assertFalse(b.closed)

        if self.TYPE == 'processes':
            self.assertTrue(a.closed)
            self.assertTrue(b.closed)
            self.assertRaises(OSError, a.recv)
            self.assertRaises(OSError, b.recv)

class _TestListener(BaseTestCase):

    ALLOWED_TYPES = ('processes',)

    def test_multiple_bind(self):
        for family in self.connection.families:
            l = self.connection.Listener(family=family)
            self.addCleanup(l.close)
            self.assertRaises(OSError, self.connection.Listener,
                              l.address, family)

    def test_context(self):
        with self.connection.Listener() as l:
            with self.connection.Client(l.address) as c:
                with l.accept() as d:
                    c.send(1729)
                    self.assertEqual(d.recv(), 1729)

        if self.TYPE == 'processes':
            self.assertRaises(OSError, l.accept)

    @unittest.skipUnless(util.abstract_sockets_supported,
                         "test needs abstract socket support")
    def test_abstract_socket(self):
        with self.connection.Listener("\0something") as listener:
            with self.connection.Client(listener.address) as client:
                with listener.accept() as d:
                    client.send(1729)
                    self.assertEqual(d.recv(), 1729)

        if self.TYPE == 'processes':
            self.assertRaises(OSError, listener.accept)


class _TestListenerClient(BaseTestCase):

    ALLOWED_TYPES = ('processes', 'threads')

    @classmethod
    def _test(cls, address):
        conn = cls.connection.Client(address)
        conn.send('hello')
        conn.close()

    def test_listener_client(self):
        for family in self.connection.families:
            l = self.connection.Listener(family=family)
            p = self.Process(target=self._test, args=(l.address,))
            p.daemon = True
            p.start()
            conn = l.accept()
            self.assertEqual(conn.recv(), 'hello')
            p.join()
            l.close()

    def test_issue14725(self):
        l = self.connection.Listener()
        p = self.Process(target=self._test, args=(l.address,))
        p.daemon = True
        p.start()
        time.sleep(1)
        # On Windows the client process should by now have connected,
        # written data and closed the pipe handle by now.  This causes
        # ConnectNamdedPipe() to fail with ERROR_NO_DATA.  See Issue
        # 14725.
        conn = l.accept()
        self.assertEqual(conn.recv(), 'hello')
        conn.close()
        p.join()
        l.close()

    def test_issue16955(self):
        for fam in self.connection.families:
            l = self.connection.Listener(family=fam)
            c = self.connection.Client(l.address)
            a = l.accept()
            a.send_bytes(b"hello")
            self.assertTrue(c.poll(1))
            a.close()
            c.close()
            l.close()

class _TestPoll(BaseTestCase):

    ALLOWED_TYPES = ('processes', 'threads')

    def test_empty_string(self):
        a, b = self.Pipe()
        self.assertEqual(a.poll(), False)
        b.send_bytes(b'')
        self.assertEqual(a.poll(), True)
        self.assertEqual(a.poll(), True)

    @classmethod
    def _child_strings(cls, conn, strings):
        for s in strings:
            time.sleep(0.1)
            conn.send_bytes(s)
        conn.close()

    def test_strings(self):
        strings = (b'hello', b'', b'a', b'b', b'', b'bye', b'', b'lop')
        a, b = self.Pipe()
        p = self.Process(target=self._child_strings, args=(b, strings))
        p.start()

        for s in strings:
            for i in range(200):
                if a.poll(0.01):
                    break
            x = a.recv_bytes()
            self.assertEqual(s, x)

        p.join()

    @classmethod
    def _child_boundaries(cls, r):
        # Polling may "pull" a message in to the child process, but we
        # don't want it to pull only part of a message, as that would
        # corrupt the pipe for any other processes which might later
        # read from it.
        r.poll(5)

    def test_boundaries(self):
        r, w = self.Pipe(False)
        p = self.Process(target=self._child_boundaries, args=(r,))
        p.start()
        time.sleep(2)
        L = [b"first", b"second"]
        for obj in L:
            w.send_bytes(obj)
        w.close()
        p.join()
        self.assertIn(r.recv_bytes(), L)

    @classmethod
    def _child_dont_merge(cls, b):
        b.send_bytes(b'a')
        b.send_bytes(b'b')
        b.send_bytes(b'cd')

    def test_dont_merge(self):
        a, b = self.Pipe()
        self.assertEqual(a.poll(0.0), False)
        self.assertEqual(a.poll(0.1), False)

        p = self.Process(target=self._child_dont_merge, args=(b,))
        p.start()

        self.assertEqual(a.recv_bytes(), b'a')
        self.assertEqual(a.poll(1.0), True)
        self.assertEqual(a.poll(1.0), True)
        self.assertEqual(a.recv_bytes(), b'b')
        self.assertEqual(a.poll(1.0), True)
        self.assertEqual(a.poll(1.0), True)
        self.assertEqual(a.poll(0.0), True)
        self.assertEqual(a.recv_bytes(), b'cd')

        p.join()

#
# Test of sending connection and socket objects between processes
#

@unittest.skipUnless(HAS_REDUCTION, "test needs multiprocess.reduction")
class _TestPicklingConnections(BaseTestCase):

    ALLOWED_TYPES = ('processes',)

    @classmethod
    def tearDownClass(cls):
        from multiprocess import resource_sharer
        resource_sharer.stop(timeout=TIMEOUT)

    @classmethod
    def _listener(cls, conn, families):
        for fam in families:
            l = cls.connection.Listener(family=fam)
            conn.send(l.address)
            new_conn = l.accept()
            conn.send(new_conn)
            new_conn.close()
            l.close()

        l = socket.socket()
        l.bind((test.support.HOST, 0))
        l.listen()
        conn.send(l.getsockname())
        new_conn, addr = l.accept()
        conn.send(new_conn)
        new_conn.close()
        l.close()

        conn.recv()

    @classmethod
    def _remote(cls, conn):
        for (address, msg) in iter(conn.recv, None):
            client = cls.connection.Client(address)
            client.send(msg.upper())
            client.close()

        address, msg = conn.recv()
        client = socket.socket()
        client.connect(address)
        client.sendall(msg.upper())
        client.close()

        conn.close()

    def test_pickling(self):
        families = self.connection.families

        lconn, lconn0 = self.Pipe()
        lp = self.Process(target=self._listener, args=(lconn0, families))
        lp.daemon = True
        lp.start()
        lconn0.close()

        rconn, rconn0 = self.Pipe()
        rp = self.Process(target=self._remote, args=(rconn0,))
        rp.daemon = True
        rp.start()
        rconn0.close()

        for fam in families:
            msg = ('This connection uses family %s' % fam).encode('ascii')
            address = lconn.recv()
            rconn.send((address, msg))
            new_conn = lconn.recv()
            self.assertEqual(new_conn.recv(), msg.upper())

        rconn.send(None)

        msg = latin('This connection uses a normal socket')
        address = lconn.recv()
        rconn.send((address, msg))
        new_conn = lconn.recv()
        buf = []
        while True:
            s = new_conn.recv(100)
            if not s:
                break
            buf.append(s)
        buf = b''.join(buf)
        self.assertEqual(buf, msg.upper())
        new_conn.close()

        lconn.send(None)

        rconn.close()
        lconn.close()

        lp.join()
        rp.join()

    @classmethod
    def child_access(cls, conn):
        w = conn.recv()
        w.send('all is well')
        w.close()

        r = conn.recv()
        msg = r.recv()
        conn.send(msg*2)

        conn.close()

    def test_access(self):
        # On Windows, if we do not specify a destination pid when
        # using DupHandle then we need to be careful to use the
        # correct access flags for DuplicateHandle(), or else
        # DupHandle.detach() will raise PermissionError.  For example,
        # for a read only pipe handle we should use
        # access=FILE_GENERIC_READ.  (Unfortunately
        # DUPLICATE_SAME_ACCESS does not work.)
        conn, child_conn = self.Pipe()
        p = self.Process(target=self.child_access, args=(child_conn,))
        p.daemon = True
        p.start()
        child_conn.close()

        r, w = self.Pipe(duplex=False)
        conn.send(w)
        w.close()
        self.assertEqual(r.recv(), 'all is well')
        r.close()

        r, w = self.Pipe(duplex=False)
        conn.send(r)
        r.close()
        w.send('foobar')
        w.close()
        self.assertEqual(conn.recv(), 'foobar'*2)

        p.join()

#
#
#

class _TestHeap(BaseTestCase):

    ALLOWED_TYPES = ('processes',)

    def setUp(self):
        super().setUp()
        # Make pristine heap for these tests
        self.old_heap = multiprocess.heap.BufferWrapper._heap
        multiprocess.heap.BufferWrapper._heap = multiprocess.heap.Heap()

    def tearDown(self):
        multiprocess.heap.BufferWrapper._heap = self.old_heap
        super().tearDown()

    def test_heap(self):
        iterations = 5000
        maxblocks = 50
        blocks = []

        # create and destroy lots of blocks of different sizes
        for i in range(iterations):
            size = int(random.lognormvariate(0, 1) * 1000)
            b = multiprocess.heap.BufferWrapper(size)
            blocks.append(b)
            if len(blocks) > maxblocks:
                i = random.randrange(maxblocks)
                del blocks[i]

        # get the heap object
        heap = multiprocess.heap.BufferWrapper._heap

        # verify the state of the heap
        all = []
        occupied = 0
        heap._lock.acquire()
        self.addCleanup(heap._lock.release)
        for L in list(heap._len_to_seq.values()):
            for arena, start, stop in L:
                all.append((heap._arenas.index(arena), start, stop,
                            stop-start, 'free'))
        for arena, start, stop in heap._allocated_blocks:
            all.append((heap._arenas.index(arena), start, stop,
                        stop-start, 'occupied'))
            occupied += (stop-start)

        all.sort()

        for i in range(len(all)-1):
            (arena, start, stop) = all[i][:3]
            (narena, nstart, nstop) = all[i+1][:3]
            self.assertTrue((arena != narena and nstart == 0) or
                            (stop == nstart))

    def test_free_from_gc(self):
        # Check that freeing of blocks by the garbage collector doesn't deadlock
        # (issue #12352).
        # Make sure the GC is enabled, and set lower collection thresholds to
        # make collections more frequent (and increase the probability of
        # deadlock).
        if not gc.isenabled():
            gc.enable()
            self.addCleanup(gc.disable)
        thresholds = gc.get_threshold()
        self.addCleanup(gc.set_threshold, *thresholds)
        gc.set_threshold(10)

        # perform numerous block allocations, with cyclic references to make
        # sure objects are collected asynchronously by the gc
        for i in range(5000):
            a = multiprocess.heap.BufferWrapper(1)
            b = multiprocess.heap.BufferWrapper(1)
            # circular references
            a.buddy = b
            b.buddy = a

#
#
#

class _Foo(Structure):
    _fields_ = [
        ('x', c_int),
        ('y', c_double),
        ('z', c_longlong,)
        ]

class _TestSharedCTypes(BaseTestCase):

    ALLOWED_TYPES = ('processes',)

    def setUp(self):
        if not HAS_SHAREDCTYPES:
            self.skipTest("requires multiprocess.sharedctypes")

    @classmethod
    def _double(cls, x, y, z, foo, arr, string):
        x.value *= 2
        y.value *= 2
        z.value *= 2
        foo.x *= 2
        foo.y *= 2
        string.value *= 2
        for i in range(len(arr)):
            arr[i] *= 2

    def test_sharedctypes(self, lock=False):
        x = Value('i', 7, lock=lock)
        y = Value(c_double, 1.0/3.0, lock=lock)
        z = Value(c_longlong, 2 ** 33, lock=lock)
        foo = Value(_Foo, 3, 2, lock=lock)
        arr = self.Array('d', list(range(10)), lock=lock)
        string = self.Array('c', 20, lock=lock)
        string.value = latin('hello')

        p = self.Process(target=self._double, args=(x, y, z, foo, arr, string))
        p.daemon = True
        p.start()
        p.join()

        self.assertEqual(x.value, 14)
        self.assertAlmostEqual(y.value, 2.0/3.0)
        self.assertEqual(z.value, 2 ** 34)
        self.assertEqual(foo.x, 6)
        self.assertAlmostEqual(foo.y, 4.0)
        for i in range(10):
            self.assertAlmostEqual(arr[i], i*2)
        self.assertEqual(string.value, latin('hellohello'))

    def test_synchronize(self):
        self.test_sharedctypes(lock=True)

    def test_copy(self):
        foo = _Foo(2, 5.0, 2 ** 33)
        bar = copy(foo)
        foo.x = 0
        foo.y = 0
        foo.z = 0
        self.assertEqual(bar.x, 2)
        self.assertAlmostEqual(bar.y, 5.0)
        self.assertEqual(bar.z, 2 ** 33)

#
#
#

class _TestFinalize(BaseTestCase):

    ALLOWED_TYPES = ('processes',)

    def setUp(self):
        self.registry_backup = util._finalizer_registry.copy()
        util._finalizer_registry.clear()

    def tearDown(self):
        self.assertFalse(util._finalizer_registry)
        util._finalizer_registry.update(self.registry_backup)

    @classmethod
    def _test_finalize(cls, conn):
        class Foo(object):
            pass

        a = Foo()
        util.Finalize(a, conn.send, args=('a',))
        del a           # triggers callback for a

        b = Foo()
        close_b = util.Finalize(b, conn.send, args=('b',))
        close_b()       # triggers callback for b
        close_b()       # does nothing because callback has already been called
        del b           # does nothing because callback has already been called

        c = Foo()
        util.Finalize(c, conn.send, args=('c',))

        d10 = Foo()
        util.Finalize(d10, conn.send, args=('d10',), exitpriority=1)

        d01 = Foo()
        util.Finalize(d01, conn.send, args=('d01',), exitpriority=0)
        d02 = Foo()
        util.Finalize(d02, conn.send, args=('d02',), exitpriority=0)
        d03 = Foo()
        util.Finalize(d03, conn.send, args=('d03',), exitpriority=0)

        util.Finalize(None, conn.send, args=('e',), exitpriority=-10)

        util.Finalize(None, conn.send, args=('STOP',), exitpriority=-100)

        # call multiprocess's cleanup function then exit process without
        # garbage collecting locals
        util._exit_function()
        conn.close()
        os._exit(0)

    def test_finalize(self):
        conn, child_conn = self.Pipe()

        p = self.Process(target=self._test_finalize, args=(child_conn,))
        p.daemon = True
        p.start()
        p.join()

        result = [obj for obj in iter(conn.recv, 'STOP')]
        self.assertEqual(result, ['a', 'b', 'd10', 'd03', 'd02', 'd01', 'e'])

    def test_thread_safety(self):
        # bpo-24484: _run_finalizers() should be thread-safe
        def cb():
            pass

        class Foo(object):
            def __init__(self):
                self.ref = self  # create reference cycle
                # insert finalizer at random key
                util.Finalize(self, cb, exitpriority=random.randint(1, 100))

        finish = False
        exc = None

        def run_finalizers():
            nonlocal exc
            while not finish:
                time.sleep(random.random() * 1e-1)
                try:
                    # A GC run will eventually happen during this,
                    # collecting stale Foo's and mutating the registry
                    util._run_finalizers()
                except Exception as e:
                    exc = e

        def make_finalizers():
            nonlocal exc
            d = {}
            while not finish:
                try:
                    # Old Foo's get gradually replaced and later
                    # collected by the GC (because of the cyclic ref)
                    d[random.getrandbits(5)] = {Foo() for i in range(10)}
                except Exception as e:
                    exc = e
                    d.clear()

        old_interval = sys.getswitchinterval()
        old_threshold = gc.get_threshold()
        try:
            sys.setswitchinterval(1e-6)
            gc.set_threshold(5, 5, 5)
            threads = [threading.Thread(target=run_finalizers),
                       threading.Thread(target=make_finalizers)]
            with test.support.start_threads(threads):
                time.sleep(4.0)  # Wait a bit to trigger race condition
                finish = True
            if exc is not None:
                raise exc
        finally:
            sys.setswitchinterval(old_interval)
            gc.set_threshold(*old_threshold)
            gc.collect()  # Collect remaining Foo's


#
# Test that from ... import * works for each module
#

class _TestImportStar(unittest.TestCase):

    def get_module_names(self):
        import glob
        folder = os.path.dirname(multiprocess.__file__)
        pattern = os.path.join(folder, '*.py')
        files = glob.glob(pattern)
        modules = [os.path.splitext(os.path.split(f)[1])[0] for f in files]
        modules = ['multiprocess.' + m for m in modules]
        modules.remove('multiprocess.__init__')
        modules.append('multiprocess')
        return modules

    def test_import(self):
        modules = self.get_module_names()
        if sys.platform == 'win32':
            modules.remove('multiprocess.popen_fork')
            modules.remove('multiprocess.popen_forkserver')
            modules.remove('multiprocess.popen_spawn_posix')
        else:
            modules.remove('multiprocess.popen_spawn_win32')
            if not HAS_REDUCTION:
                modules.remove('multiprocess.popen_forkserver')

        if c_int is None:
            # This module requires _ctypes
            modules.remove('multiprocess.sharedctypes')

        for name in modules:
            __import__(name)
            mod = sys.modules[name]
            self.assertTrue(hasattr(mod, '__all__'), name)

            for attr in mod.__all__:
                self.assertTrue(
                    hasattr(mod, attr),
                    '%r does not have attribute %r' % (mod, attr)
                    )

#
# Quick test that logging works -- does not test logging output
#

class _TestLogging(BaseTestCase):

    ALLOWED_TYPES = ('processes',)

    def test_enable_logging(self):
        logger = multiprocess.get_logger()
        logger.setLevel(util.SUBWARNING)
        self.assertTrue(logger is not None)
        logger.debug('this will not be printed')
        logger.info('nor will this')
        logger.setLevel(LOG_LEVEL)

    @classmethod
    def _test_level(cls, conn):
        logger = multiprocess.get_logger()
        conn.send(logger.getEffectiveLevel())

    def test_level(self):
        LEVEL1 = 32
        LEVEL2 = 37

        logger = multiprocess.get_logger()
        root_logger = logging.getLogger()
        root_level = root_logger.level

        reader, writer = multiprocess.Pipe(duplex=False)

        logger.setLevel(LEVEL1)
        p = self.Process(target=self._test_level, args=(writer,))
        p.start()
        self.assertEqual(LEVEL1, reader.recv())
        p.join()
        p.close()

        logger.setLevel(logging.NOTSET)
        root_logger.setLevel(LEVEL2)
        p = self.Process(target=self._test_level, args=(writer,))
        p.start()
        self.assertEqual(LEVEL2, reader.recv())
        p.join()
        p.close()

        root_logger.setLevel(root_level)
        logger.setLevel(level=LOG_LEVEL)


# class _TestLoggingProcessName(BaseTestCase):
#
#     def handle(self, record):
#         assert record.processName == multiprocess.current_process().name
#         self.__handled = True
#
#     def test_logging(self):
#         handler = logging.Handler()
#         handler.handle = self.handle
#         self.__handled = False
#         # Bypass getLogger() and side-effects
#         logger = logging.getLoggerClass()(
#                 'multiprocess.test.TestLoggingProcessName')
#         logger.addHandler(handler)
#         logger.propagate = False
#
#         logger.warn('foo')
#         assert self.__handled

#
# Check that Process.join() retries if os.waitpid() fails with EINTR
#

class _TestPollEintr(BaseTestCase):

    ALLOWED_TYPES = ('processes',)

    @classmethod
    def _killer(cls, pid):
        time.sleep(0.1)
        os.kill(pid, signal.SIGUSR1)

    @unittest.skipUnless(hasattr(signal, 'SIGUSR1'), 'requires SIGUSR1')
    def test_poll_eintr(self):
        got_signal = [False]
        def record(*args):
            got_signal[0] = True
        pid = os.getpid()
        oldhandler = signal.signal(signal.SIGUSR1, record)
        try:
            killer = self.Process(target=self._killer, args=(pid,))
            killer.start()
            try:
                p = self.Process(target=time.sleep, args=(2,))
                p.start()
                p.join()
            finally:
                killer.join()
            self.assertTrue(got_signal[0])
            self.assertEqual(p.exitcode, 0)
        finally:
            signal.signal(signal.SIGUSR1, oldhandler)

#
# Test to verify handle verification, see issue 3321
#

class TestInvalidHandle(unittest.TestCase):

    @unittest.skipIf(WIN32, "skipped on Windows")
    def test_invalid_handles(self):
        conn = multiprocess.connection.Connection(44977608)
        # check that poll() doesn't crash
        try:
            conn.poll()
        except (ValueError, OSError):
            pass
        finally:
            # Hack private attribute _handle to avoid printing an error
            # in conn.__del__
            conn._handle = None
        self.assertRaises((ValueError, OSError),
                          multiprocess.connection.Connection, -1)



class OtherTest(unittest.TestCase):
    # TODO: add more tests for deliver/answer challenge.
    def test_deliver_challenge_auth_failure(self):
        class _FakeConnection(object):
            def recv_bytes(self, size):
                return b'something bogus'
            def send_bytes(self, data):
                pass
        self.assertRaises(multiprocess.AuthenticationError,
                          multiprocess.connection.deliver_challenge,
                          _FakeConnection(), b'abc')

    def test_answer_challenge_auth_failure(self):
        class _FakeConnection(object):
            def __init__(self):
                self.count = 0
            def recv_bytes(self, size):
                self.count += 1
                if self.count == 1:
                    return multiprocess.connection.CHALLENGE
                elif self.count == 2:
                    return b'something bogus'
                return b''
            def send_bytes(self, data):
                pass
        self.assertRaises(multiprocess.AuthenticationError,
                          multiprocess.connection.answer_challenge,
                          _FakeConnection(), b'abc')

#
# Test Manager.start()/Pool.__init__() initializer feature - see issue 5585
#

def initializer(ns):
    ns.test += 1

class TestInitializers(unittest.TestCase):
    def setUp(self):
        self.mgr = multiprocess.Manager()
        self.ns = self.mgr.Namespace()
        self.ns.test = 0

    def tearDown(self):
        self.mgr.shutdown()
        self.mgr.join()

    def test_manager_initializer(self):
        m = multiprocess.managers.SyncManager()
        self.assertRaises(TypeError, m.start, 1)
        m.start(initializer, (self.ns,))
        self.assertEqual(self.ns.test, 1)
        m.shutdown()
        m.join()

    def test_pool_initializer(self):
        self.assertRaises(TypeError, multiprocess.Pool, initializer=1)
        p = multiprocess.Pool(1, initializer, (self.ns,))
        p.close()
        p.join()
        self.assertEqual(self.ns.test, 1)

#
# Issue 5155, 5313, 5331: Test process in processes
# Verifies os.close(sys.stdin.fileno) vs. sys.stdin.close() behavior
#

def _this_sub_process(q):
    try:
        item = q.get(block=False)
    except pyqueue.Empty:
        pass

def _test_process():
    queue = multiprocess.Queue()
    subProc = multiprocess.Process(target=_this_sub_process, args=(queue,))
    subProc.daemon = True
    subProc.start()
    subProc.join()

def _afunc(x):
    return x*x

def pool_in_process():
    pool = multiprocess.Pool(processes=4)
    x = pool.map(_afunc, [1, 2, 3, 4, 5, 6, 7])
    pool.close()
    pool.join()

class _file_like(object):
    def __init__(self, delegate):
        self._delegate = delegate
        self._pid = None

    @property
    def cache(self):
        pid = os.getpid()
        # There are no race conditions since fork keeps only the running thread
        if pid != self._pid:
            self._pid = pid
            self._cache = []
        return self._cache

    def write(self, data):
        self.cache.append(data)

    def flush(self):
        self._delegate.write(''.join(self.cache))
        self._cache = []

class TestStdinBadfiledescriptor(unittest.TestCase):

    def test_queue_in_process(self):
        proc = multiprocess.Process(target=_test_process)
        proc.start()
        proc.join()

    def test_pool_in_process(self):
        p = multiprocess.Process(target=pool_in_process)
        p.start()
        p.join()

    def test_flushing(self):
        sio = io.StringIO()
        flike = _file_like(sio)
        flike.write('foo')
        proc = multiprocess.Process(target=lambda: flike.flush())
        flike.flush()
        assert sio.getvalue() == 'foo'


class TestWait(unittest.TestCase):

    @classmethod
    def _child_test_wait(cls, w, slow):
        for i in range(10):
            if slow:
                time.sleep(random.random()*0.1)
            w.send((i, os.getpid()))
        w.close()

    def test_wait(self, slow=False):
        from multiprocess.connection import wait
        readers = []
        procs = []
        messages = []

        for i in range(4):
            r, w = multiprocess.Pipe(duplex=False)
            p = multiprocess.Process(target=self._child_test_wait, args=(w, slow))
            p.daemon = True
            p.start()
            w.close()
            readers.append(r)
            procs.append(p)
            self.addCleanup(p.join)

        while readers:
            for r in wait(readers):
                try:
                    msg = r.recv()
                except EOFError:
                    readers.remove(r)
                    r.close()
                else:
                    messages.append(msg)

        messages.sort()
        expected = sorted((i, p.pid) for i in range(10) for p in procs)
        self.assertEqual(messages, expected)

    @classmethod
    def _child_test_wait_socket(cls, address, slow):
        s = socket.socket()
        s.connect(address)
        for i in range(10):
            if slow:
                time.sleep(random.random()*0.1)
            s.sendall(('%s\n' % i).encode('ascii'))
        s.close()

    def test_wait_socket(self, slow=False):
        from multiprocess.connection import wait
        l = socket.socket()
        l.bind((test.support.HOST, 0))
        l.listen()
        addr = l.getsockname()
        readers = []
        procs = []
        dic = {}

        for i in range(4):
            p = multiprocess.Process(target=self._child_test_wait_socket,
                                        args=(addr, slow))
            p.daemon = True
            p.start()
            procs.append(p)
            self.addCleanup(p.join)

        for i in range(4):
            r, _ = l.accept()
            readers.append(r)
            dic[r] = []
        l.close()

        while readers:
            for r in wait(readers):
                msg = r.recv(32)
                if not msg:
                    readers.remove(r)
                    r.close()
                else:
                    dic[r].append(msg)

        expected = ''.join('%s\n' % i for i in range(10)).encode('ascii')
        for v in dic.values():
            self.assertEqual(b''.join(v), expected)

    def test_wait_slow(self):
        self.test_wait(True)

    def test_wait_socket_slow(self):
        self.test_wait_socket(True)

    def test_wait_timeout(self):
        from multiprocess.connection import wait

        expected = 5
        a, b = multiprocess.Pipe()

        start = getattr(time,'monotonic',time.time)()
        res = wait([a, b], expected)
        delta = getattr(time,'monotonic',time.time)() - start

        self.assertEqual(res, [])
        self.assertLess(delta, expected * 2)
        self.assertGreater(delta, expected * 0.5)

        b.send(None)

        start = getattr(time,'monotonic',time.time)()
        res = wait([a, b], 20)
        delta = getattr(time,'monotonic',time.time)() - start

        self.assertEqual(res, [a])
        self.assertLess(delta, 0.4)

    @classmethod
    def signal_and_sleep(cls, sem, period):
        sem.release()
        time.sleep(period)

    def test_wait_integer(self):
        from multiprocess.connection import wait

        expected = 3
        sorted_ = lambda l: sorted(l, key=lambda x: id(x))
        sem = multiprocess.Semaphore(0)
        a, b = multiprocess.Pipe()
        p = multiprocess.Process(target=self.signal_and_sleep,
                                    args=(sem, expected))

        p.start()
        self.assertIsInstance(p.sentinel, int)
        self.assertTrue(sem.acquire(timeout=20))

        start = getattr(time,'monotonic',time.time)()
        res = wait([a, p.sentinel, b], expected + 20)
        delta = getattr(time,'monotonic',time.time)() - start

        self.assertEqual(res, [p.sentinel])
        self.assertLess(delta, expected + 2)
        self.assertGreater(delta, expected - 2)

        a.send(None)

        start = getattr(time,'monotonic',time.time)()
        res = wait([a, p.sentinel, b], 20)
        delta = getattr(time,'monotonic',time.time)() - start

        self.assertEqual(sorted_(res), sorted_([p.sentinel, b]))
        self.assertLess(delta, 0.4)

        b.send(None)

        start = getattr(time,'monotonic',time.time)()
        res = wait([a, p.sentinel, b], 20)
        delta = getattr(time,'monotonic',time.time)() - start

        self.assertEqual(sorted_(res), sorted_([a, p.sentinel, b]))
        self.assertLess(delta, 0.4)

        p.terminate()
        p.join()

    def test_neg_timeout(self):
        from multiprocess.connection import wait
        a, b = multiprocess.Pipe()
        t = getattr(time,'monotonic',time.time)()
        res = wait([a], timeout=-1)
        t = getattr(time,'monotonic',time.time)() - t
        self.assertEqual(res, [])
        self.assertLess(t, 1)
        a.close()
        b.close()

#
# Issue 14151: Test invalid family on invalid environment
#

class TestInvalidFamily(unittest.TestCase):

    @unittest.skipIf(WIN32, "skipped on Windows")
    def test_invalid_family(self):
        with self.assertRaises(ValueError):
            multiprocess.connection.Listener(r'\\.\test')

    @unittest.skipUnless(WIN32, "skipped on non-Windows platforms")
    def test_invalid_family_win32(self):
        with self.assertRaises(ValueError):
            multiprocess.connection.Listener('/var/test.pipe')

#
# Issue 12098: check sys.flags of child matches that for parent
#

class TestFlags(unittest.TestCase):
    @classmethod
    def run_in_grandchild(cls, conn):
        conn.send(tuple(sys.flags))

    @classmethod
    def run_in_child(cls):
        import json
        r, w = multiprocess.Pipe(duplex=False)
        p = multiprocess.Process(target=cls.run_in_grandchild, args=(w,))
        p.start()
        grandchild_flags = r.recv()
        p.join()
        r.close()
        w.close()
        flags = (tuple(sys.flags), grandchild_flags)
        print(json.dumps(flags))

    def _test_flags(self):
        import json, subprocess
        # start child process using unusual flags
        prog = ('from multiprocess.tests import TestFlags; ' +
                'TestFlags.run_in_child()')
        data = subprocess.check_output(
            [sys.executable, '-E', '-S', '-O', '-c', prog])
        child_flags, grandchild_flags = json.loads(data.decode('ascii'))
        self.assertEqual(child_flags, grandchild_flags)

#
# Test interaction with socket timeouts - see Issue #6056
#

class TestTimeouts(unittest.TestCase):
    @classmethod
    def _test_timeout(cls, child, address):
        time.sleep(1)
        child.send(123)
        child.close()
        conn = multiprocess.connection.Client(address)
        conn.send(456)
        conn.close()

    def test_timeout(self):
        old_timeout = socket.getdefaulttimeout()
        try:
            socket.setdefaulttimeout(0.1)
            parent, child = multiprocess.Pipe(duplex=True)
            l = multiprocess.connection.Listener(family='AF_INET')
            p = multiprocess.Process(target=self._test_timeout,
                                        args=(child, l.address))
            p.start()
            child.close()
            self.assertEqual(parent.recv(), 123)
            parent.close()
            conn = l.accept()
            self.assertEqual(conn.recv(), 456)
            conn.close()
            l.close()
            join_process(p)
        finally:
            socket.setdefaulttimeout(old_timeout)

#
# Test what happens with no "if __name__ == '__main__'"
#

class TestNoForkBomb(unittest.TestCase):
    def test_noforkbomb(self):
        sm = multiprocess.get_start_method()
        name = os.path.join(os.path.dirname(__file__), 'mp_fork_bomb.py')
        if sm != 'fork':
            rc, out, err = test.support.script_helper.assert_python_failure(name, sm)
            self.assertEqual(out, b'')
            self.assertIn(b'RuntimeError', err)
        else:
            rc, out, err = test.support.script_helper.assert_python_ok(name, sm)
            self.assertEqual(out.rstrip(), b'123')
            self.assertEqual(err, b'')

#
# Issue #17555: ForkAwareThreadLock
#

class TestForkAwareThreadLock(unittest.TestCase):
    # We recursively start processes.  Issue #17555 meant that the
    # after fork registry would get duplicate entries for the same
    # lock.  The size of the registry at generation n was ~2**n.

    @classmethod
    def child(cls, n, conn):
        if n > 1:
            p = multiprocess.Process(target=cls.child, args=(n-1, conn))
            p.start()
            conn.close()
            join_process(p)
        else:
            conn.send(len(util._afterfork_registry))
        conn.close()

    def test_lock(self):
        r, w = multiprocess.Pipe(False)
        l = util.ForkAwareThreadLock()
        old_size = len(util._afterfork_registry)
        p = multiprocess.Process(target=self.child, args=(5, w))
        p.start()
        w.close()
        new_size = r.recv()
        join_process(p)
        self.assertLessEqual(new_size, old_size)

#
# Check that non-forked child processes do not inherit unneeded fds/handles
#

class TestCloseFds(unittest.TestCase):

    def get_high_socket_fd(self):
        if WIN32:
            # The child process will not have any socket handles, so
            # calling socket.fromfd() should produce WSAENOTSOCK even
            # if there is a handle of the same number.
            return socket.socket().detach()
        else:
            # We want to produce a socket with an fd high enough that a
            # freshly created child process will not have any fds as high.
            fd = socket.socket().detach()
            to_close = []
            while fd < 50:
                to_close.append(fd)
                fd = os.dup(fd)
            for x in to_close:
                os.close(x)
            return fd

    def close(self, fd):
        if WIN32:
            socket.socket(socket.AF_INET, socket.SOCK_STREAM, fileno=fd).close()
        else:
            os.close(fd)

    @classmethod
    def _test_closefds(cls, conn, fd):
        try:
            s = socket.fromfd(fd, socket.AF_INET, socket.SOCK_STREAM)
        except Exception as e:
            conn.send(e)
        else:
            s.close()
            conn.send(None)

    def test_closefd(self):
        if not HAS_REDUCTION:
            raise unittest.SkipTest('requires fd pickling')

        reader, writer = multiprocess.Pipe()
        fd = self.get_high_socket_fd()
        try:
            p = multiprocess.Process(target=self._test_closefds,
                                        args=(writer, fd))
            p.start()
            writer.close()
            e = reader.recv()
            join_process(p)
        finally:
            self.close(fd)
            writer.close()
            reader.close()

        if multiprocess.get_start_method() == 'fork':
            self.assertIs(e, None)
        else:
            WSAENOTSOCK = 10038
            self.assertIsInstance(e, OSError)
            self.assertTrue(e.errno == errno.EBADF or
                            e.winerror == WSAENOTSOCK, e)

#
# Issue #17097: EINTR should be ignored by recv(), send(), accept() etc
#

class TestIgnoreEINTR(unittest.TestCase):

    # Sending CONN_MAX_SIZE bytes into a multiprocess pipe must block
    CONN_MAX_SIZE = max(support.PIPE_MAX_SIZE, support.SOCK_MAX_SIZE)

    @classmethod
    def _test_ignore(cls, conn):
        def handler(signum, frame):
            pass
        signal.signal(signal.SIGUSR1, handler)
        conn.send('ready')
        x = conn.recv()
        conn.send(x)
        conn.send_bytes(b'x' * cls.CONN_MAX_SIZE)

    @unittest.skipUnless(hasattr(signal, 'SIGUSR1'), 'requires SIGUSR1')
    def test_ignore(self):
        conn, child_conn = multiprocess.Pipe()
        try:
            p = multiprocess.Process(target=self._test_ignore,
                                        args=(child_conn,))
            p.daemon = True
            p.start()
            child_conn.close()
            self.assertEqual(conn.recv(), 'ready')
            time.sleep(0.1)
            os.kill(p.pid, signal.SIGUSR1)
            time.sleep(0.1)
            conn.send(1234)
            self.assertEqual(conn.recv(), 1234)
            time.sleep(0.1)
            os.kill(p.pid, signal.SIGUSR1)
            self.assertEqual(conn.recv_bytes(), b'x' * self.CONN_MAX_SIZE)
            time.sleep(0.1)
            p.join()
        finally:
            conn.close()

    @classmethod
    def _test_ignore_listener(cls, conn):
        def handler(signum, frame):
            pass
        signal.signal(signal.SIGUSR1, handler)
        with multiprocess.connection.Listener() as l:
            conn.send(l.address)
            a = l.accept()
            a.send('welcome')

    @unittest.skipUnless(hasattr(signal, 'SIGUSR1'), 'requires SIGUSR1')
    def test_ignore_listener(self):
        conn, child_conn = multiprocess.Pipe()
        try:
            p = multiprocess.Process(target=self._test_ignore_listener,
                                        args=(child_conn,))
            p.daemon = True
            p.start()
            child_conn.close()
            address = conn.recv()
            time.sleep(0.1)
            os.kill(p.pid, signal.SIGUSR1)
            time.sleep(0.1)
            client = multiprocess.connection.Client(address)
            self.assertEqual(client.recv(), 'welcome')
            p.join()
        finally:
            conn.close()

class TestStartMethod(unittest.TestCase):
    @classmethod
    def _check_context(cls, conn):
        conn.send(multiprocess.get_start_method())

    def check_context(self, ctx):
        r, w = ctx.Pipe(duplex=False)
        p = ctx.Process(target=self._check_context, args=(w,))
        p.start()
        w.close()
        child_method = r.recv()
        r.close()
        p.join()
        self.assertEqual(child_method, ctx.get_start_method())

    def test_context(self):
        for method in ('fork', 'spawn', 'forkserver'):
            try:
                ctx = multiprocess.get_context(method)
            except ValueError:
                continue
            self.assertEqual(ctx.get_start_method(), method)
            self.assertIs(ctx.get_context(), ctx)
            self.assertRaises(ValueError, ctx.set_start_method, 'spawn')
            self.assertRaises(ValueError, ctx.set_start_method, None)
            self.check_context(ctx)

    def test_set_get(self):
        multiprocess.set_forkserver_preload(PRELOAD)
        count = 0
        old_method = multiprocess.get_start_method()
        try:
            for method in ('fork', 'spawn', 'forkserver'):
                try:
                    multiprocess.set_start_method(method, force=True)
                except ValueError:
                    continue
                self.assertEqual(multiprocess.get_start_method(), method)
                ctx = multiprocess.get_context()
                self.assertEqual(ctx.get_start_method(), method)
                self.assertTrue(type(ctx).__name__.lower().startswith(method))
                self.assertTrue(
                    ctx.Process.__name__.lower().startswith(method))
                self.check_context(multiprocess)
                count += 1
        finally:
            multiprocess.set_start_method(old_method, force=True)
        self.assertGreaterEqual(count, 1)

    def test_get_all(self):
        methods = multiprocess.get_all_start_methods()
        if sys.platform == 'win32':
            self.assertEqual(methods, ['spawn'])
        else:
            self.assertTrue(methods == ['fork', 'spawn'] or
                            methods == ['fork', 'spawn', 'forkserver'])

    def test_preload_resources(self):
        if multiprocess.get_start_method() != 'forkserver':
            self.skipTest("test only relevant for 'forkserver' method")
        name = os.path.join(os.path.dirname(__file__), 'mp_preload.py')
        rc, out, err = test.support.script_helper.assert_python_ok(name)
        out = out.decode()
        err = err.decode()
        if out.rstrip() != 'ok' or err != '':
            print(out)
            print(err)
            self.fail("failed spawning forkserver or grandchild")


@unittest.skipIf(sys.platform == "win32",
                 "test semantics don't make sense on Windows")
class TestSemaphoreTracker(unittest.TestCase):

    def _test_semaphore_tracker(self):
        #
        # Check that killing process does not leak named semaphores
        #
        import subprocess
        cmd = '''if 1:
            import multiprocess as mp, time, os
            mp.set_start_method("spawn")
            lock1 = mp.Lock()
            lock2 = mp.Lock()
            os.write(%d, lock1._semlock.name.encode("ascii") + b"\\n")
            os.write(%d, lock2._semlock.name.encode("ascii") + b"\\n")
            time.sleep(10)
        '''
        r, w = os.pipe()
        p = subprocess.Popen([sys.executable,
                             '-E', '-c', cmd % (w, w)],
                             pass_fds=[w],
                             stderr=subprocess.PIPE)
        os.close(w)
        with open(r, 'rb', closefd=True) as f:
            name1 = f.readline().rstrip().decode('ascii')
            name2 = f.readline().rstrip().decode('ascii')
        _multiprocessing.sem_unlink(name1)
        p.terminate()
        p.wait()
        time.sleep(2.0)
        with self.assertRaises(OSError) as ctx:
            _multiprocessing.sem_unlink(name2)
        # docs say it should be ENOENT, but OSX seems to give EINVAL
        self.assertIn(ctx.exception.errno, (errno.ENOENT, errno.EINVAL))
        err = p.stderr.read().decode('utf-8')
        p.stderr.close()
        expected = 'semaphore_tracker: There appear to be 2 leaked semaphores'
        self.assertRegex(err, expected)
        self.assertRegex(err, r'semaphore_tracker: %r: \[Errno' % name1)

    def check_semaphore_tracker_death(self, signum, should_die):
        # bpo-31310: if the semaphore tracker process has died, it should
        # be restarted implicitly.
        from multiprocess.semaphore_tracker import _semaphore_tracker
        _semaphore_tracker.ensure_running()
        pid = _semaphore_tracker._pid
        os.kill(pid, signum)
        time.sleep(1.0)  # give it time to die

        ctx = multiprocess.get_context("spawn")
        with contextlib.ExitStack() as stack:
            if should_die:
                stack.enter_context(self.assertWarnsRegex(
                    UserWarning,
                    "semaphore_tracker: process died"))
            sem = ctx.Semaphore()
            sem.acquire()
            sem.release()
            wr = weakref.ref(sem)
            # ensure `sem` gets collected, which triggers communication with
            # the semaphore tracker
            del sem
            gc.collect()
            self.assertIsNone(wr())

    def test_semaphore_tracker_sigint(self):
        # Catchable signal (ignored by semaphore tracker)
        self.check_semaphore_tracker_death(signal.SIGINT, False)

    def test_semaphore_tracker_sigkill(self):
        # Uncatchable signal.
        self.check_semaphore_tracker_death(signal.SIGKILL, True)


class TestSimpleQueue(unittest.TestCase):

    @classmethod
    def _test_empty(cls, queue, child_can_start, parent_can_continue):
        child_can_start.wait()
        # issue 30301, could fail under spawn and forkserver
        try:
            queue.put(queue.empty())
            queue.put(queue.empty())
        finally:
            parent_can_continue.set()

    def test_empty(self):
        queue = multiprocess.SimpleQueue()
        child_can_start = multiprocess.Event()
        parent_can_continue = multiprocess.Event()

        proc = multiprocess.Process(
            target=self._test_empty,
            args=(queue, child_can_start, parent_can_continue)
        )
        proc.daemon = True
        proc.start()

        self.assertTrue(queue.empty())

        child_can_start.set()
        parent_can_continue.wait()

        self.assertFalse(queue.empty())
        self.assertEqual(queue.get(), True)
        self.assertEqual(queue.get(), False)
        self.assertTrue(queue.empty())

        proc.join()


class TestSyncManagerTypes(unittest.TestCase):
    """Test all the types which can be shared between a parent and a
    child process by using a manager which acts as an intermediary
    between them.

    In the following unit-tests the base type is created in the parent
    process, the @classmethod represents the worker process and the
    shared object is readable and editable between the two.

    # The child.
    @classmethod
    def _test_list(cls, obj):
        assert obj[0] == 5
        assert obj.append(6)

    # The parent.
    def test_list(self):
        o = self.manager.list()
        o.append(5)
        self.run_worker(self._test_list, o)
        assert o[1] == 6
    """
    manager_class = multiprocess.managers.SyncManager

    def setUp(self):
        self.manager = self.manager_class()
        self.manager.start()
        self.proc = None

    def tearDown(self):
        if self.proc is not None and self.proc.is_alive():
            self.proc.terminate()
            self.proc.join()
        self.manager.shutdown()
        self.manager = None
        self.proc = None

    @classmethod
    def setUpClass(cls):
        support.reap_children()

    tearDownClass = setUpClass

    def wait_proc_exit(self):
        # Only the manager process should be returned by active_children()
        # but this can take a bit on slow machines, so wait a few seconds
        # if there are other children too (see #17395).
        join_process(self.proc)
        start_time = getattr(time,'monotonic',time.time)()
        t = 0.01
        while len(multiprocess.active_children()) > 1:
            time.sleep(t)
            t *= 2
            dt = getattr(time,'monotonic',time.time)() - start_time
            if dt >= 5.0:
                test.support.environment_altered = True
                print("Warning -- multiprocess.Manager still has %s active "
                      "children after %s seconds"
                      % (multiprocess.active_children(), dt),
                      file=sys.stderr)
                break

    def run_worker(self, worker, obj):
        self.proc = multiprocess.Process(target=worker, args=(obj, ))
        self.proc.daemon = True
        self.proc.start()
        self.wait_proc_exit()
        self.assertEqual(self.proc.exitcode, 0)

    @classmethod
    def _test_queue(cls, obj):
        assert obj.qsize() == 2
        assert obj.full()
        assert not obj.empty()
        assert obj.get() == 5
        assert not obj.empty()
        assert obj.get() == 6
        assert obj.empty()

    def test_queue(self, qname="Queue"):
        o = getattr(self.manager, qname)(2)
        o.put(5)
        o.put(6)
        self.run_worker(self._test_queue, o)
        assert o.empty()
        assert not o.full()

    def test_joinable_queue(self):
        self.test_queue("JoinableQueue")

    @classmethod
    def _test_event(cls, obj):
        assert obj.is_set()
        obj.wait()
        obj.clear()
        obj.wait(0.001)

    def test_event(self):
        o = self.manager.Event()
        o.set()
        self.run_worker(self._test_event, o)
        assert not o.is_set()
        o.wait(0.001)

    @classmethod
    def _test_lock(cls, obj):
        obj.acquire()

    def test_lock(self, lname="Lock"):
        o = getattr(self.manager, lname)()
        self.run_worker(self._test_lock, o)
        o.release()
        self.assertRaises(RuntimeError, o.release)  # already released

    @classmethod
    def _test_rlock(cls, obj):
        obj.acquire()
        obj.release()

    def test_rlock(self, lname="Lock"):
        o = getattr(self.manager, lname)()
        self.run_worker(self._test_rlock, o)

    @classmethod
    def _test_semaphore(cls, obj):
        obj.acquire()

    def test_semaphore(self, sname="Semaphore"):
        o = getattr(self.manager, sname)()
        self.run_worker(self._test_semaphore, o)
        o.release()

    def test_bounded_semaphore(self):
        self.test_semaphore(sname="BoundedSemaphore")

    @classmethod
    def _test_condition(cls, obj):
        obj.acquire()
        obj.release()

    def test_condition(self):
        o = self.manager.Condition()
        self.run_worker(self._test_condition, o)

    @classmethod
    def _test_barrier(cls, obj):
        assert obj.parties == 5
        obj.reset()

    def test_barrier(self):
        o = self.manager.Barrier(5)
        self.run_worker(self._test_barrier, o)

    @classmethod
    def _test_pool(cls, obj):
        # TODO: fix https://bugs.python.org/issue35919
        with obj:
            pass

    def test_pool(self):
        o = self.manager.Pool(processes=4)
        self.run_worker(self._test_pool, o)

    @classmethod
    def _test_list(cls, obj):
        assert obj[0] == 5
        assert obj.count(5) == 1
        assert obj.index(5) == 0
        obj.sort()
        obj.reverse()
        for x in obj:
            pass
        assert len(obj) == 1
        assert obj.pop(0) == 5

    def test_list(self):
        o = self.manager.list()
        o.append(5)
        self.run_worker(self._test_list, o)
        assert not o
        self.assertEqual(len(o), 0)

    @classmethod
    def _test_dict(cls, obj):
        assert len(obj) == 1
        assert obj['foo'] == 5
        assert obj.get('foo') == 5
        assert list(obj.items()) == [('foo', 5)]
        assert list(obj.keys()) == ['foo']
        assert list(obj.values()) == [5]
        assert obj.copy() == {'foo': 5}
        assert obj.popitem() == ('foo', 5)

    def test_dict(self):
        o = self.manager.dict()
        o['foo'] = 5
        self.run_worker(self._test_dict, o)
        assert not o
        self.assertEqual(len(o), 0)

    @classmethod
    def _test_value(cls, obj):
        assert obj.value == 1
        assert obj.get() == 1
        obj.set(2)

    def test_value(self):
        o = self.manager.Value('i', 1)
        self.run_worker(self._test_value, o)
        self.assertEqual(o.value, 2)
        self.assertEqual(o.get(), 2)

    @classmethod
    def _test_array(cls, obj):
        assert obj[0] == 0
        assert obj[1] == 1
        assert len(obj) == 2
        assert list(obj) == [0, 1]

    def test_array(self):
        o = self.manager.Array('i', [0, 1])
        self.run_worker(self._test_array, o)

    @classmethod
    def _test_namespace(cls, obj):
        assert obj.x == 0
        assert obj.y == 1

    def test_namespace(self):
        o = self.manager.Namespace()
        o.x = 0
        o.y = 1
        self.run_worker(self._test_namespace, o)


#
# Mixins
#

class BaseMixin(object):
    @classmethod
    def setUpClass(cls):
        cls.dangling = (multiprocess.process._dangling.copy(),
                        threading._dangling.copy())

    @classmethod
    def tearDownClass(cls):
        # bpo-26762: Some multiprocess objects like Pool create reference
        # cycles. Trigger a garbage collection to break these cycles.
        test.support.gc_collect()

        processes = set(multiprocess.process._dangling) - set(cls.dangling[0])
        if processes:
            test.support.environment_altered = True
            print('Warning -- Dangling processes: %s' % processes,
                  file=sys.stderr)
        processes = None

        threads = set(threading._dangling) - set(cls.dangling[1])
        if threads:
            test.support.environment_altered = True
            print('Warning -- Dangling threads: %s' % threads,
                  file=sys.stderr)
        threads = None


class ProcessesMixin(BaseMixin):
    TYPE = 'processes'
    Process = multiprocess.Process
    connection = multiprocess.connection
    current_process = staticmethod(multiprocess.current_process)
    active_children = staticmethod(multiprocess.active_children)
    Pool = staticmethod(multiprocess.Pool)
    Pipe = staticmethod(multiprocess.Pipe)
    Queue = staticmethod(multiprocess.Queue)
    JoinableQueue = staticmethod(multiprocess.JoinableQueue)
    Lock = staticmethod(multiprocess.Lock)
    RLock = staticmethod(multiprocess.RLock)
    Semaphore = staticmethod(multiprocess.Semaphore)
    BoundedSemaphore = staticmethod(multiprocess.BoundedSemaphore)
    Condition = staticmethod(multiprocess.Condition)
    Event = staticmethod(multiprocess.Event)
    Barrier = staticmethod(multiprocess.Barrier)
    Value = staticmethod(multiprocess.Value)
    Array = staticmethod(multiprocess.Array)
    RawValue = staticmethod(multiprocess.RawValue)
    RawArray = staticmethod(multiprocess.RawArray)


class ManagerMixin(BaseMixin):
    TYPE = 'manager'
    Process = multiprocess.Process
    Queue = property(operator.attrgetter('manager.Queue'))
    JoinableQueue = property(operator.attrgetter('manager.JoinableQueue'))
    Lock = property(operator.attrgetter('manager.Lock'))
    RLock = property(operator.attrgetter('manager.RLock'))
    Semaphore = property(operator.attrgetter('manager.Semaphore'))
    BoundedSemaphore = property(operator.attrgetter('manager.BoundedSemaphore'))
    Condition = property(operator.attrgetter('manager.Condition'))
    Event = property(operator.attrgetter('manager.Event'))
    Barrier = property(operator.attrgetter('manager.Barrier'))
    Value = property(operator.attrgetter('manager.Value'))
    Array = property(operator.attrgetter('manager.Array'))
    list = property(operator.attrgetter('manager.list'))
    dict = property(operator.attrgetter('manager.dict'))
    Namespace = property(operator.attrgetter('manager.Namespace'))

    @classmethod
    def Pool(cls, *args, **kwds):
        return cls.manager.Pool(*args, **kwds)

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.manager = multiprocess.Manager()

    @classmethod
    def tearDownClass(cls):
        # only the manager process should be returned by active_children()
        # but this can take a bit on slow machines, so wait a few seconds
        # if there are other children too (see #17395)
        start_time = getattr(time,'monotonic',time.time)()
        t = 0.01
        while len(multiprocess.active_children()) > 1:
            time.sleep(t)
            t *= 2
            dt = getattr(time,'monotonic',time.time)() - start_time
            if dt >= 5.0:
                test.support.environment_altered = True
                print("Warning -- multiprocess.Manager still has %s active "
                      "children after %s seconds"
                      % (multiprocess.active_children(), dt),
                      file=sys.stderr)
                break

        gc.collect()                       # do garbage collection
        if cls.manager._number_of_objects() != 0:
            # This is not really an error since some tests do not
            # ensure that all processes which hold a reference to a
            # managed object have been joined.
            test.support.environment_altered = True
            print('Warning -- Shared objects which still exist at manager '
                  'shutdown:')
            print(cls.manager._debug_info())
        cls.manager.shutdown()
        cls.manager.join()
        cls.manager = None

        super().tearDownClass()


class ThreadsMixin(BaseMixin):
    TYPE = 'threads'
    Process = multiprocess.dummy.Process
    connection = multiprocess.dummy.connection
    current_process = staticmethod(multiprocess.dummy.current_process)
    active_children = staticmethod(multiprocess.dummy.active_children)
    Pool = staticmethod(multiprocess.dummy.Pool)
    Pipe = staticmethod(multiprocess.dummy.Pipe)
    Queue = staticmethod(multiprocess.dummy.Queue)
    JoinableQueue = staticmethod(multiprocess.dummy.JoinableQueue)
    Lock = staticmethod(multiprocess.dummy.Lock)
    RLock = staticmethod(multiprocess.dummy.RLock)
    Semaphore = staticmethod(multiprocess.dummy.Semaphore)
    BoundedSemaphore = staticmethod(multiprocess.dummy.BoundedSemaphore)
    Condition = staticmethod(multiprocess.dummy.Condition)
    Event = staticmethod(multiprocess.dummy.Event)
    Barrier = staticmethod(multiprocess.dummy.Barrier)
    Value = staticmethod(multiprocess.dummy.Value)
    Array = staticmethod(multiprocess.dummy.Array)

#
# Functions used to create test cases from the base ones in this module
#

def install_tests_in_module_dict(remote_globs, start_method):
    __module__ = remote_globs['__name__']
    local_globs = globals()
    ALL_TYPES = {'processes', 'threads', 'manager'}

    for name, base in local_globs.items():
        if not isinstance(base, type):
            continue
        if issubclass(base, BaseTestCase):
            if base is BaseTestCase:
                continue
            assert set(base.ALLOWED_TYPES) <= ALL_TYPES, base.ALLOWED_TYPES
            for type_ in base.ALLOWED_TYPES:
                newname = 'With' + type_.capitalize() + name[1:]
                Mixin = local_globs[type_.capitalize() + 'Mixin']
                class Temp(base, Mixin, unittest.TestCase):
                    pass
                Temp.__name__ = Temp.__qualname__ = newname
                Temp.__module__ = __module__
                remote_globs[newname] = Temp
        elif issubclass(base, unittest.TestCase):
            class Temp(base, object):
                pass
            Temp.__name__ = Temp.__qualname__ = name
            Temp.__module__ = __module__
            remote_globs[name] = Temp

    dangling = [None, None]
    old_start_method = [None]

    def setUpModule():
        multiprocess.set_forkserver_preload(PRELOAD)
        multiprocess.process._cleanup()
        dangling[0] = multiprocess.process._dangling.copy()
        dangling[1] = threading._dangling.copy()
        old_start_method[0] = multiprocess.get_start_method(allow_none=True)
        try:
            multiprocess.set_start_method(start_method, force=True)
        except ValueError:
            raise unittest.SkipTest(start_method +
                                    ' start method not supported')

        if sys.platform.startswith("linux"):
            try:
                lock = multiprocess.RLock()
            except OSError:
                raise unittest.SkipTest("OSError raises on RLock creation, "
                                        "see issue 3111!")
        check_enough_semaphores()
        util.get_temp_dir()     # creates temp directory
        multiprocess.get_logger().setLevel(LOG_LEVEL)

    def tearDownModule():
        need_sleep = False

        # bpo-26762: Some multiprocess objects like Pool create reference
        # cycles. Trigger a garbage collection to break these cycles.
        test.support.gc_collect()

        multiprocess.set_start_method(old_start_method[0], force=True)
        # pause a bit so we don't get warning about dangling threads/processes
        processes = set(multiprocess.process._dangling) - set(dangling[0])
        if processes:
            need_sleep = True
            test.support.environment_altered = True
            print('Warning -- Dangling processes: %s' % processes,
                  file=sys.stderr)
        processes = None

        threads = set(threading._dangling) - set(dangling[1])
        if threads:
            need_sleep = True
            test.support.environment_altered = True
            print('Warning -- Dangling threads: %s' % threads,
                  file=sys.stderr)
        threads = None

        # Sleep 500 ms to give time to child processes to complete.
        if need_sleep:
            time.sleep(0.5)

        multiprocess.util._cleanup_tests()

    remote_globs['setUpModule'] = setUpModule
    remote_globs['tearDownModule'] = tearDownModule
