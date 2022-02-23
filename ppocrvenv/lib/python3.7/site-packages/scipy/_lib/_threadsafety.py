import threading

import scipy._lib.decorator


__all__ = ['ReentrancyError', 'ReentrancyLock', 'non_reentrant']


class ReentrancyError(RuntimeError):
    pass


class ReentrancyLock:
    """
    Threading lock that raises an exception for reentrant calls.

    Calls from different threads are serialized, and nested calls from the
    same thread result to an error.

    The object can be used as a context manager or to decorate functions
    via the decorate() method.

    """

    def __init__(self, err_msg):
        self._rlock = threading.RLock()
        self._entered = False
        self._err_msg = err_msg

    def __enter__(self):
        self._rlock.acquire()
        if self._entered:
            self._rlock.release()
            raise ReentrancyError(self._err_msg)
        self._entered = True

    def __exit__(self, type, value, traceback):
        self._entered = False
        self._rlock.release()

    def decorate(self, func):
        def caller(func, *a, **kw):
            with self:
                return func(*a, **kw)
        return scipy._lib.decorator.decorate(func, caller)


def non_reentrant(err_msg=None):
    """
    Decorate a function with a threading lock and prevent reentrant calls.
    """
    def decorator(func):
        msg = err_msg
        if msg is None:
            msg = "%s is not re-entrant" % func.__name__
        lock = ReentrancyLock(msg)
        return lock.decorate(func)
    return decorator
