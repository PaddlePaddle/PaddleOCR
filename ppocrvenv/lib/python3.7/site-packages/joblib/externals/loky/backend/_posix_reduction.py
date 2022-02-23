###############################################################################
# Extra reducers for Unix based system and connections objects
#
# author: Thomas Moreau and Olivier Grisel
#
# adapted from multiprocessing/reduction.py (17/02/2017)
#  * Add adapted reduction for LokyProcesses and socket/Connection
#
import os
import sys
import socket
import _socket

from .reduction import register
from .context import get_spawning_popen

if sys.version_info >= (3, 3):
    from multiprocessing.connection import Connection
else:
    from _multiprocessing import Connection


HAVE_SEND_HANDLE = (hasattr(socket, 'CMSG_LEN') and
                    hasattr(socket, 'SCM_RIGHTS') and
                    hasattr(socket.socket, 'sendmsg'))


def _mk_inheritable(fd):
    if sys.version_info[:2] > (3, 3):
        os.set_inheritable(fd, True)
    return fd


def DupFd(fd):
    '''Return a wrapper for an fd.'''
    popen_obj = get_spawning_popen()
    if popen_obj is not None:
        return popen_obj.DupFd(popen_obj.duplicate_for_child(fd))
    elif HAVE_SEND_HANDLE and sys.version_info[:2] > (3, 3):
        from multiprocessing import resource_sharer
        return resource_sharer.DupFd(fd)
    else:
        raise TypeError(
            'Cannot pickle connection object. This object can only be '
            'passed when spawning a new process'
        )


if sys.version_info[:2] != (3, 3):
    def _reduce_socket(s):
        df = DupFd(s.fileno())
        return _rebuild_socket, (df, s.family, s.type, s.proto)

    def _rebuild_socket(df, family, type, proto):
        fd = df.detach()
        return socket.fromfd(fd, family, type, proto)
else:
    from multiprocessing.reduction import reduce_socket as _reduce_socket


register(socket.socket, _reduce_socket)
register(_socket.socket, _reduce_socket)


if sys.version_info[:2] != (3, 3):
    def reduce_connection(conn):
        df = DupFd(conn.fileno())
        return rebuild_connection, (df, conn.readable, conn.writable)

    def rebuild_connection(df, readable, writable):
        fd = df.detach()
        return Connection(fd, readable, writable)
else:
    from multiprocessing.reduction import reduce_connection

register(Connection, reduce_connection)
