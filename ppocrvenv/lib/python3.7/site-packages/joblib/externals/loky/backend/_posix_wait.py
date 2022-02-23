###############################################################################
# Compat for wait function on UNIX based system
#
# author: Thomas Moreau and Olivier Grisel
#
# adapted from multiprocessing/connection.py (17/02/2017)
#  * Backport wait function to python2.7
#

import platform
import select
import socket
import errno
SYSTEM = platform.system()

try:
    import ctypes
except ImportError:  # pragma: no cover
    ctypes = None  # noqa

if SYSTEM == 'Darwin' and ctypes is not None:
    from ctypes.util import find_library
    libSystem = ctypes.CDLL(find_library('libSystem.dylib'))
    CoreServices = ctypes.CDLL(find_library('CoreServices'),
                               use_errno=True)
    mach_absolute_time = libSystem.mach_absolute_time
    mach_absolute_time.restype = ctypes.c_uint64
    absolute_to_nanoseconds = CoreServices.AbsoluteToNanoseconds
    absolute_to_nanoseconds.restype = ctypes.c_uint64
    absolute_to_nanoseconds.argtypes = [ctypes.c_uint64]

    def monotonic():
        return absolute_to_nanoseconds(mach_absolute_time()) * 1e-9

elif SYSTEM == 'Linux' and ctypes is not None:
    # from stackoverflow:
    # questions/1205722/how-do-i-get-monotonic-time-durations-in-python
    import ctypes
    import os

    CLOCK_MONOTONIC = 1  # see <linux/time.h>

    class timespec(ctypes.Structure):
        _fields_ = [
            ('tv_sec', ctypes.c_long),
            ('tv_nsec', ctypes.c_long),
        ]

    librt = ctypes.CDLL('librt.so.1', use_errno=True)
    clock_gettime = librt.clock_gettime
    clock_gettime.argtypes = [
        ctypes.c_int, ctypes.POINTER(timespec),
    ]

    def monotonic():  # noqa
        t = timespec()
        if clock_gettime(CLOCK_MONOTONIC, ctypes.pointer(t)) != 0:
            errno_ = ctypes.get_errno()
            raise OSError(errno_, os.strerror(errno_))
        return t.tv_sec + t.tv_nsec * 1e-9
else:  # pragma: no cover
    from time import time as monotonic


if hasattr(select, 'poll'):
    def _poll(fds, timeout):
        if timeout is not None:
            timeout = int(timeout * 1000)  # timeout is in milliseconds
        fd_map = {}
        pollster = select.poll()
        for fd in fds:
            pollster.register(fd, select.POLLIN)
            if hasattr(fd, 'fileno'):
                fd_map[fd.fileno()] = fd
            else:
                fd_map[fd] = fd
        ls = []
        for fd, event in pollster.poll(timeout):
            if event & select.POLLNVAL:  # pragma: no cover
                raise ValueError('invalid file descriptor %i' % fd)
            ls.append(fd_map[fd])
        return ls
else:
    def _poll(fds, timeout):
        return select.select(fds, [], [], timeout)[0]


def wait(object_list, timeout=None):
    '''
    Wait till an object in object_list is ready/readable.
    Returns list of those objects which are ready/readable.
    '''
    if timeout is not None:
        if timeout <= 0:
            return _poll(object_list, 0)
        else:
            deadline = monotonic() + timeout
    while True:
        try:
            return _poll(object_list, timeout)
        except (OSError, IOError, socket.error) as e:  # pragma: no cover
            if e.errno != errno.EINTR:
                raise
        if timeout is not None:
            timeout = deadline - monotonic()
