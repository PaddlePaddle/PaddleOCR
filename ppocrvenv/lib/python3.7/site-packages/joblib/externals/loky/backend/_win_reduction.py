###############################################################################
# Extra reducers for Windows system and connections objects
#
# author: Thomas Moreau and Olivier Grisel
#
# adapted from multiprocessing/reduction.py (17/02/2017)
#  * Add adapted reduction for LokyProcesses and socket/PipeConnection
#
import os
import sys
import socket
from .reduction import register


if sys.platform == 'win32':
    if sys.version_info[:2] < (3, 3):
        from _multiprocessing import PipeConnection
    else:
        import _winapi
        from multiprocessing.connection import PipeConnection


if sys.version_info[:2] >= (3, 4) and sys.platform == 'win32':
    class DupHandle(object):
        def __init__(self, handle, access, pid=None):
            # duplicate handle for process with given pid
            if pid is None:
                pid = os.getpid()
            proc = _winapi.OpenProcess(_winapi.PROCESS_DUP_HANDLE, False, pid)
            try:
                self._handle = _winapi.DuplicateHandle(
                    _winapi.GetCurrentProcess(),
                    handle, proc, access, False, 0)
            finally:
                _winapi.CloseHandle(proc)
            self._access = access
            self._pid = pid

        def detach(self):
            # retrieve handle from process which currently owns it
            if self._pid == os.getpid():
                return self._handle
            proc = _winapi.OpenProcess(_winapi.PROCESS_DUP_HANDLE, False,
                                       self._pid)
            try:
                return _winapi.DuplicateHandle(
                    proc, self._handle, _winapi.GetCurrentProcess(),
                    self._access, False, _winapi.DUPLICATE_CLOSE_SOURCE)
            finally:
                _winapi.CloseHandle(proc)

    def reduce_pipe_connection(conn):
        access = ((_winapi.FILE_GENERIC_READ if conn.readable else 0) |
                  (_winapi.FILE_GENERIC_WRITE if conn.writable else 0))
        dh = DupHandle(conn.fileno(), access)
        return rebuild_pipe_connection, (dh, conn.readable, conn.writable)

    def rebuild_pipe_connection(dh, readable, writable):
        from multiprocessing.connection import PipeConnection
        handle = dh.detach()
        return PipeConnection(handle, readable, writable)
    register(PipeConnection, reduce_pipe_connection)

elif sys.platform == 'win32':
    # Older Python versions
    from multiprocessing.reduction import reduce_pipe_connection
    register(PipeConnection, reduce_pipe_connection)


if sys.version_info[:2] < (3, 3) and sys.platform == 'win32':
    from _multiprocessing import win32
    from multiprocessing.reduction import reduce_handle, rebuild_handle
    close = win32.CloseHandle

    def fromfd(handle, family, type_, proto=0):
        s = socket.socket(family, type_, proto, fileno=handle)
        if s.__class__ is not socket.socket:
            s = socket.socket(_sock=s)
        return s

    def reduce_socket(s):
        if not hasattr(socket, "fromfd"):
            raise TypeError("sockets cannot be pickled on this system.")
        reduced_handle = reduce_handle(s.fileno())
        return _rebuild_socket, (reduced_handle, s.family, s.type, s.proto)

    def _rebuild_socket(reduced_handle, family, type_, proto):
        handle = rebuild_handle(reduced_handle)
        s = fromfd(handle, family, type_, proto)
        close(handle)
        return s

    register(socket.socket, reduce_socket)
elif sys.version_info[:2] < (3, 4):
    from multiprocessing.reduction import reduce_socket
    register(socket.socket, reduce_socket)
else:
    from multiprocessing.reduction import _reduce_socket
    register(socket.socket, _reduce_socket)
