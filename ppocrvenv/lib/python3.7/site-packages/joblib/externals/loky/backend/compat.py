###############################################################################
# Compat file to import the correct modules for each platform and python
# version.
#
# author: Thomas Moreau and Olivier grisel
#
import sys

PY3 = sys.version_info[:2] >= (3, 3)

if PY3:
    import queue
else:
    import Queue as queue

if sys.version_info >= (3, 4):
    from multiprocessing.process import BaseProcess
else:
    from multiprocessing.process import Process as BaseProcess

# Platform specific compat
if sys.platform == "win32":
    from .compat_win32 import wait
else:
    from .compat_posix import wait


def set_cause(exc, cause):
    exc.__cause__ = cause

    if not PY3:
        # Preformat message here.
        if exc.__cause__ is not None:
            exc.args = ("{}\n\nThis was caused directly by {}".format(
                exc.args if len(exc.args) != 1 else exc.args[0],
                str(exc.__cause__)),)

    return exc


__all__ = ["queue", "BaseProcess", "set_cause", "wait"]
