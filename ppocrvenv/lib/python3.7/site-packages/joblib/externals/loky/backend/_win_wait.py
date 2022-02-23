###############################################################################
# Compat for wait function on Windows system
#
# author: Thomas Moreau and Olivier Grisel
#
# adapted from multiprocessing/connection.py (17/02/2017)
#  * Backport wait function to python2.7
#

import ctypes
import sys
from time import sleep


if sys.platform == 'win32' and sys.version_info[:2] < (3, 3):
    from _subprocess import WaitForSingleObject, WAIT_OBJECT_0

    try:
        from time import monotonic
    except ImportError:
        # Backward old for crappy old Python that did not have cross-platform
        # monotonic clock by default.

        # TODO: do we want to add support for cygwin at some point? See:
        # https://github.com/atdt/monotonic/blob/master/monotonic.py
        GetTickCount64 = ctypes.windll.kernel32.GetTickCount64
        GetTickCount64.restype = ctypes.c_ulonglong

        def monotonic():
            """Monotonic clock, cannot go backward."""
            return GetTickCount64() / 1000.0

    def wait(handles, timeout=None):
        """Backward compat for python2.7

        This function wait for either:
        * one connection is ready for read,
        * one process handle has exited or got killed,
        * timeout is reached. Note that this function has a precision of 2
          msec.
        """
        if timeout is not None:
            deadline = monotonic() + timeout

        while True:
            # We cannot use select as in windows it only support sockets
            ready = []
            for h in handles:
                if type(h) in [int, long]:
                    if WaitForSingleObject(h, 0) == WAIT_OBJECT_0:
                        ready += [h]
                elif h.poll(0):
                    ready.append(h)
            if len(ready) > 0:
                return ready
            sleep(.001)
            if timeout is not None and deadline - monotonic() <= 0:
                return []
