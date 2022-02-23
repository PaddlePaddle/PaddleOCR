###############################################################################
# Queue and SimpleQueue implementation for loky
#
# authors: Thomas Moreau, Olivier Grisel
#
# based on multiprocessing/queues.py (16/02/2017)
# * Add some compatibility function for python2.7 and 3.3 and makes sure
#   it uses the right synchronization primitive.
# * Add some custom reducers for the Queues/SimpleQueue to tweak the
#   pickling process. (overload Queue._feed/SimpleQueue.put)
#
import os
import sys
import errno
import weakref
import threading

from multiprocessing import util
from multiprocessing import connection
from multiprocessing.synchronize import SEM_VALUE_MAX
from multiprocessing.queues import Full
from multiprocessing.queues import _sentinel, Queue as mp_Queue
from multiprocessing.queues import SimpleQueue as mp_SimpleQueue

from .reduction import loads, dumps
from .context import assert_spawning, get_context


__all__ = ['Queue', 'SimpleQueue', 'Full']


class Queue(mp_Queue):

    def __init__(self, maxsize=0, reducers=None, ctx=None):

        if sys.version_info[:2] >= (3, 4):
            super().__init__(maxsize=maxsize, ctx=ctx)
        else:
            if maxsize <= 0:
                # Can raise ImportError (see issues #3770 and #23400)
                maxsize = SEM_VALUE_MAX
            if ctx is None:
                ctx = get_context()
            self._maxsize = maxsize
            self._reader, self._writer = connection.Pipe(duplex=False)
            self._rlock = ctx.Lock()
            self._opid = os.getpid()
            if sys.platform == 'win32':
                self._wlock = None
            else:
                self._wlock = ctx.Lock()
            self._sem = ctx.BoundedSemaphore(maxsize)

            # For use by concurrent.futures
            self._ignore_epipe = False

            self._after_fork()

            if sys.platform != 'win32':
                util.register_after_fork(self, Queue._after_fork)

        self._reducers = reducers

    # Use custom queue set/get state to be able to reduce the custom reducers
    def __getstate__(self):
        assert_spawning(self)
        return (self._ignore_epipe, self._maxsize, self._reader, self._writer,
                self._reducers, self._rlock, self._wlock, self._sem,
                self._opid)

    def __setstate__(self, state):
        (self._ignore_epipe, self._maxsize, self._reader, self._writer,
         self._reducers, self._rlock, self._wlock, self._sem,
         self._opid) = state
        if sys.version_info >= (3, 9):
            self._reset()
        else:
            self._after_fork()

    # Overload _start_thread to correctly call our custom _feed
    def _start_thread(self):
        util.debug('Queue._start_thread()')

        # Start thread which transfers data from buffer to pipe
        self._buffer.clear()
        self._thread = threading.Thread(
            target=Queue._feed,
            args=(self._buffer, self._notempty, self._send_bytes,
                  self._wlock, self._writer.close, self._reducers,
                  self._ignore_epipe, self._on_queue_feeder_error, self._sem),
            name='QueueFeederThread'
        )
        self._thread.daemon = True

        util.debug('doing self._thread.start()')
        self._thread.start()
        util.debug('... done self._thread.start()')

        # On process exit we will wait for data to be flushed to pipe.
        #
        # However, if this process created the queue then all
        # processes which use the queue will be descendants of this
        # process.  Therefore waiting for the queue to be flushed
        # is pointless once all the child processes have been joined.
        created_by_this_process = (self._opid == os.getpid())
        if not self._joincancelled and not created_by_this_process:
            self._jointhread = util.Finalize(
                self._thread, Queue._finalize_join,
                [weakref.ref(self._thread)],
                exitpriority=-5
            )

        # Send sentinel to the thread queue object when garbage collected
        self._close = util.Finalize(
            self, Queue._finalize_close,
            [self._buffer, self._notempty],
            exitpriority=10
        )

    # Overload the _feed methods to use our custom pickling strategy.
    @staticmethod
    def _feed(buffer, notempty, send_bytes, writelock, close, reducers,
              ignore_epipe, onerror, queue_sem):
        util.debug('starting thread to feed data to pipe')
        nacquire = notempty.acquire
        nrelease = notempty.release
        nwait = notempty.wait
        bpopleft = buffer.popleft
        sentinel = _sentinel
        if sys.platform != 'win32':
            wacquire = writelock.acquire
            wrelease = writelock.release
        else:
            wacquire = None

        while 1:
            try:
                nacquire()
                try:
                    if not buffer:
                        nwait()
                finally:
                    nrelease()
                try:
                    while 1:
                        obj = bpopleft()
                        if obj is sentinel:
                            util.debug('feeder thread got sentinel -- exiting')
                            close()
                            return

                        # serialize the data before acquiring the lock
                        obj_ = dumps(obj, reducers=reducers)
                        if wacquire is None:
                            send_bytes(obj_)
                        else:
                            wacquire()
                            try:
                                send_bytes(obj_)
                            finally:
                                wrelease()
                        # Remove references early to avoid leaking memory
                        del obj, obj_
                except IndexError:
                    pass
            except BaseException as e:
                if ignore_epipe and getattr(e, 'errno', 0) == errno.EPIPE:
                    return
                # Since this runs in a daemon thread the resources it uses
                # may be become unusable while the process is cleaning up.
                # We ignore errors which happen after the process has
                # started to cleanup.
                if util.is_exiting():
                    util.info('error in queue thread: %s', e)
                    return
                else:
                    queue_sem.release()
                    onerror(e, obj)

    def _on_queue_feeder_error(self, e, obj):
        """
        Private API hook called when feeding data in the background thread
        raises an exception.  For overriding by concurrent.futures.
        """
        import traceback
        traceback.print_exc()

    if sys.version_info[:2] < (3, 4):
        # Compat for python2.7/3.3 that use _send instead of _send_bytes
        def _after_fork(self):
            super(Queue, self)._after_fork()
            self._send_bytes = self._writer.send_bytes


class SimpleQueue(mp_SimpleQueue):

    def __init__(self, reducers=None, ctx=None):
        if sys.version_info[:2] >= (3, 4):
            super().__init__(ctx=ctx)
        else:
            # Use the context to create the sync objects for python2.7/3.3
            if ctx is None:
                ctx = get_context()
            self._reader, self._writer = connection.Pipe(duplex=False)
            self._rlock = ctx.Lock()
            self._poll = self._reader.poll
            if sys.platform == 'win32':
                self._wlock = None
            else:
                self._wlock = ctx.Lock()

        # Add possiblity to use custom reducers
        self._reducers = reducers

    def close(self):
        self._reader.close()
        self._writer.close()

    # Use custom queue set/get state to be able to reduce the custom reducers
    def __getstate__(self):
        assert_spawning(self)
        return (self._reader, self._writer, self._reducers, self._rlock,
                self._wlock)

    def __setstate__(self, state):
        (self._reader, self._writer, self._reducers, self._rlock,
         self._wlock) = state

    if sys.version_info[:2] < (3, 4):
        # For python2.7/3.3, overload get to avoid creating deadlocks with
        # unpickling errors.
        def get(self):
            with self._rlock:
                res = self._reader.recv_bytes()
            # unserialize the data after having released the lock
            return loads(res)

    # Overload put to use our customizable reducer
    def put(self, obj):
        # serialize the data before acquiring the lock
        obj = dumps(obj, reducers=self._reducers)
        if self._wlock is None:
            # writes to a message oriented win32 pipe are atomic
            self._writer.send_bytes(obj)
        else:
            with self._wlock:
                self._writer.send_bytes(obj)
