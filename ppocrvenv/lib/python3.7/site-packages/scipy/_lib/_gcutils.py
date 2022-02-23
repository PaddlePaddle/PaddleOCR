"""
Module for testing automatic garbage collection of objects

.. autosummary::
   :toctree: generated/

   set_gc_state - enable or disable garbage collection
   gc_state - context manager for given state of garbage collector
   assert_deallocated - context manager to check for circular references on object

"""
import weakref
import gc

from contextlib import contextmanager
from platform import python_implementation

__all__ = ['set_gc_state', 'gc_state', 'assert_deallocated']


IS_PYPY = python_implementation() == 'PyPy'


class ReferenceError(AssertionError):
    pass


def set_gc_state(state):
    """ Set status of garbage collector """
    if gc.isenabled() == state:
        return
    if state:
        gc.enable()
    else:
        gc.disable()


@contextmanager
def gc_state(state):
    """ Context manager to set state of garbage collector to `state`

    Parameters
    ----------
    state : bool
        True for gc enabled, False for disabled

    Examples
    --------
    >>> with gc_state(False):
    ...     assert not gc.isenabled()
    >>> with gc_state(True):
    ...     assert gc.isenabled()
    """
    orig_state = gc.isenabled()
    set_gc_state(state)
    yield
    set_gc_state(orig_state)


@contextmanager
def assert_deallocated(func, *args, **kwargs):
    """Context manager to check that object is deallocated

    This is useful for checking that an object can be freed directly by
    reference counting, without requiring gc to break reference cycles.
    GC is disabled inside the context manager.

    This check is not available on PyPy.

    Parameters
    ----------
    func : callable
        Callable to create object to check
    \\*args : sequence
        positional arguments to `func` in order to create object to check
    \\*\\*kwargs : dict
        keyword arguments to `func` in order to create object to check

    Examples
    --------
    >>> class C: pass
    >>> with assert_deallocated(C) as c:
    ...     # do something
    ...     del c

    >>> class C:
    ...     def __init__(self):
    ...         self._circular = self # Make circular reference
    >>> with assert_deallocated(C) as c: #doctest: +IGNORE_EXCEPTION_DETAIL
    ...     # do something
    ...     del c
    Traceback (most recent call last):
        ...
    ReferenceError: Remaining reference(s) to object
    """
    if IS_PYPY:
        raise RuntimeError("assert_deallocated is unavailable on PyPy")

    with gc_state(False):
        obj = func(*args, **kwargs)
        ref = weakref.ref(obj)
        yield obj
        del obj
        if ref() is not None:
            raise ReferenceError("Remaining reference(s) to object")
