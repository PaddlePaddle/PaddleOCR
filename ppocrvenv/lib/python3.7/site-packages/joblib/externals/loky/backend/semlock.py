###############################################################################
# Ctypes implementation for posix semaphore.
#
# author: Thomas Moreau and Olivier Grisel
#
# adapted from cpython/Modules/_multiprocessing/semaphore.c (17/02/2017)
#  * use ctypes to access pthread semaphores and provide a full python
#    semaphore management.
#  * For OSX, as no sem_getvalue is not implemented, Semaphore with value > 1
#    are not guaranteed to work.
#  * Only work with LokyProcess on posix
#
import os
import sys
import time
import errno
import ctypes
import tempfile
import threading
from ctypes.util import find_library

# As we need to use ctypes return types for semlock object, failure value
# needs to be cast to proper python value. Unix failure convention is to
# return 0, whereas OSX returns -1
SEM_FAILURE = ctypes.c_void_p(0).value
if sys.platform == 'darwin':
    SEM_FAILURE = ctypes.c_void_p(-1).value

# Semaphore types
RECURSIVE_MUTEX = 0
SEMAPHORE = 1

# Semaphore constants
SEM_OFLAG = ctypes.c_int(os.O_CREAT | os.O_EXCL)
SEM_PERM = ctypes.c_int(384)


class timespec(ctypes.Structure):
    _fields_ = [("tv_sec", ctypes.c_long), ("tv_nsec", ctypes.c_long)]


if sys.platform != 'win32':
    pthread = ctypes.CDLL(find_library('pthread'), use_errno=True)
    pthread.sem_open.restype = ctypes.c_void_p
    pthread.sem_close.argtypes = [ctypes.c_void_p]
    pthread.sem_wait.argtypes = [ctypes.c_void_p]
    pthread.sem_trywait.argtypes = [ctypes.c_void_p]
    pthread.sem_post.argtypes = [ctypes.c_void_p]
    pthread.sem_getvalue.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    pthread.sem_unlink.argtypes = [ctypes.c_char_p]
    if sys.platform != "darwin":
        pthread.sem_timedwait.argtypes = [ctypes.c_void_p,
                                          ctypes.POINTER(timespec)]

try:
    from threading import get_ident
except ImportError:
    def get_ident():
        return threading.current_thread().ident


if sys.version_info[:2] < (3, 3):
    class FileExistsError(OSError):
        pass

    class FileNotFoundError(OSError):
        pass


def sem_unlink(name):
    if pthread.sem_unlink(name.encode('ascii')) < 0:
        raiseFromErrno()


def _sem_open(name, value=None):
    """ Construct or retrieve a semaphore with the given name

    If value is None, try to retrieve an existing named semaphore.
    Else create a new semaphore with the given value
    """
    if value is None:
        handle = pthread.sem_open(ctypes.c_char_p(name), 0)
    else:
        handle = pthread.sem_open(ctypes.c_char_p(name), SEM_OFLAG, SEM_PERM,
                                  ctypes.c_int(value))

    if handle == SEM_FAILURE:
        e = ctypes.get_errno()
        if e == errno.EEXIST:
            raise FileExistsError("a semaphore named %s already exists" % name)
        elif e == errno.ENOENT:
            raise FileNotFoundError('cannot find semaphore named %s' % name)
        elif e == errno.ENOSYS:
            raise NotImplementedError('No semaphore implementation on this '
                                      'system')
        else:
            raiseFromErrno()

    return handle


def _sem_timedwait(handle, timeout):
    t_start = time.time()
    if sys.platform != "darwin":
        sec = int(timeout)
        tv_sec = int(t_start)
        nsec = int(1e9 * (timeout - sec) + .5)
        tv_nsec = int(1e9 * (t_start - tv_sec) + .5)
        deadline = timespec(sec+tv_sec, nsec+tv_nsec)
        deadline.tv_sec += int(deadline.tv_nsec / 1000000000)
        deadline.tv_nsec %= 1000000000
        return pthread.sem_timedwait(handle, ctypes.pointer(deadline))

    # PERFORMANCE WARNING
    # No sem_timedwait on OSX so we implement our own method. This method can
    # degrade performances has the wait can have a latency up to 20 msecs
    deadline = t_start + timeout
    delay = 0
    now = time.time()
    while True:
        # Poll the sem file
        res = pthread.sem_trywait(handle)
        if res == 0:
            return 0
        else:
            e = ctypes.get_errno()
            if e != errno.EAGAIN:
                raiseFromErrno()

        # check for timeout
        now = time.time()
        if now > deadline:
            ctypes.set_errno(errno.ETIMEDOUT)
            return -1

        # calculate how much time left and check the delay is not too long
        # -- maximum is 20 msecs
        difference = (deadline - now)
        delay = min(delay, 20e-3, difference)

        # Sleep and increase delay
        time.sleep(delay)
        delay += 1e-3


class SemLock(object):
    """ctypes wrapper to the unix semaphore"""

    _rand = tempfile._RandomNameSequence()

    def __init__(self, kind, value, maxvalue, name=None, unlink_now=False):
        self.count = 0
        self.ident = 0
        self.kind = kind
        self.maxvalue = maxvalue
        self.name = name
        self.handle = _sem_open(self.name.encode('ascii'), value)

    def __del__(self):
        try:
            res = pthread.sem_close(self.handle)
            assert res == 0, "Issue while closing semaphores"
        except AttributeError:
            pass

    def _is_mine(self):
        return self.count > 0 and get_ident() == self.ident

    def acquire(self, block=True, timeout=None):
        if self.kind == RECURSIVE_MUTEX and self._is_mine():
            self.count += 1
            return True

        if block and timeout is None:
            res = pthread.sem_wait(self.handle)
        elif not block or timeout <= 0:
            res = pthread.sem_trywait(self.handle)
        else:
            res = _sem_timedwait(self.handle, timeout)
        if res < 0:
            e = ctypes.get_errno()
            if e == errno.EINTR:
                return None
            elif e in [errno.EAGAIN, errno.ETIMEDOUT]:
                return False
            raiseFromErrno()
        self.count += 1
        self.ident = get_ident()
        return True

    def release(self):
        if self.kind == RECURSIVE_MUTEX:
            assert self._is_mine(), (
                "attempt to release recursive lock not owned by thread")
            if self.count > 1:
                self.count -= 1
                return
            assert self.count == 1
        else:
            if sys.platform == 'darwin':
                # Handle broken get_value for mac ==> only Lock will work
                # as sem_get_value do not work properly
                if self.maxvalue == 1:
                    if pthread.sem_trywait(self.handle) < 0:
                        e = ctypes.get_errno()
                        if e != errno.EAGAIN:
                            raise OSError(e, errno.errorcode[e])
                    else:
                        if pthread.sem_post(self.handle) < 0:
                            raiseFromErrno()
                        else:
                            raise ValueError(
                                "semaphore or lock released too many times")
                else:
                    import warnings
                    warnings.warn("semaphore are broken on OSX, release might "
                                  "increase its maximal value", RuntimeWarning)
            else:
                value = self._get_value()
                if value >= self.maxvalue:
                    raise ValueError(
                        "semaphore or lock released too many times")

        if pthread.sem_post(self.handle) < 0:
            raiseFromErrno()

        self.count -= 1

    def _get_value(self):
        value = ctypes.pointer(ctypes.c_int(-1))
        if pthread.sem_getvalue(self.handle, value) < 0:
            raiseFromErrno()
        return value.contents.value

    def _count(self):
        return self.count

    def _is_zero(self):
        if sys.platform == 'darwin':
            # Handle broken get_value for mac ==> only Lock will work
            # as sem_get_value do not work properly
            if pthread.sem_trywait(self.handle) < 0:
                e = ctypes.get_errno()
                if e == errno.EAGAIN:
                    return True
                raise OSError(e, errno.errorcode[e])
            else:
                if pthread.sem_post(self.handle) < 0:
                    raiseFromErrno()
                return False
        else:
            value = ctypes.pointer(ctypes.c_int(-1))
            if pthread.sem_getvalue(self.handle, value) < 0:
                raiseFromErrno()
            return value.contents.value == 0

    def _after_fork(self):
        self.count = 0

    @staticmethod
    def _rebuild(handle, kind, maxvalue, name):
        self = SemLock.__new__(SemLock)
        self.count = 0
        self.ident = 0
        self.kind = kind
        self.maxvalue = maxvalue
        self.name = name
        self.handle = _sem_open(name.encode('ascii'))
        return self


def raiseFromErrno():
    e = ctypes.get_errno()
    raise OSError(e, errno.errorcode[e])
