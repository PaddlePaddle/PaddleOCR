""" Test for assert_deallocated context manager and gc utilities
"""
import gc

from scipy._lib._gcutils import (set_gc_state, gc_state, assert_deallocated,
                                 ReferenceError, IS_PYPY)

from numpy.testing import assert_equal

import pytest


def test_set_gc_state():
    gc_status = gc.isenabled()
    try:
        for state in (True, False):
            gc.enable()
            set_gc_state(state)
            assert_equal(gc.isenabled(), state)
            gc.disable()
            set_gc_state(state)
            assert_equal(gc.isenabled(), state)
    finally:
        if gc_status:
            gc.enable()


def test_gc_state():
    # Test gc_state context manager
    gc_status = gc.isenabled()
    try:
        for pre_state in (True, False):
            set_gc_state(pre_state)
            for with_state in (True, False):
                # Check the gc state is with_state in with block
                with gc_state(with_state):
                    assert_equal(gc.isenabled(), with_state)
                # And returns to previous state outside block
                assert_equal(gc.isenabled(), pre_state)
                # Even if the gc state is set explicitly within the block
                with gc_state(with_state):
                    assert_equal(gc.isenabled(), with_state)
                    set_gc_state(not with_state)
                assert_equal(gc.isenabled(), pre_state)
    finally:
        if gc_status:
            gc.enable()


@pytest.mark.skipif(IS_PYPY, reason="Test not meaningful on PyPy")
def test_assert_deallocated():
    # Ordinary use
    class C:
        def __init__(self, arg0, arg1, name='myname'):
            self.name = name
    for gc_current in (True, False):
        with gc_state(gc_current):
            # We are deleting from with-block context, so that's OK
            with assert_deallocated(C, 0, 2, 'another name') as c:
                assert_equal(c.name, 'another name')
                del c
            # Or not using the thing in with-block context, also OK
            with assert_deallocated(C, 0, 2, name='third name'):
                pass
            assert_equal(gc.isenabled(), gc_current)


@pytest.mark.skipif(IS_PYPY, reason="Test not meaningful on PyPy")
def test_assert_deallocated_nodel():
    class C:
        pass
    with pytest.raises(ReferenceError):
        # Need to delete after using if in with-block context
        # Note: assert_deallocated(C) needs to be assigned for the test
        # to function correctly.  It is assigned to c, but c itself is
        # not referenced in the body of the with, it is only there for
        # the refcount.
        with assert_deallocated(C) as c:
            pass


@pytest.mark.skipif(IS_PYPY, reason="Test not meaningful on PyPy")
def test_assert_deallocated_circular():
    class C:
        def __init__(self):
            self._circular = self
    with pytest.raises(ReferenceError):
        # Circular reference, no automatic garbage collection
        with assert_deallocated(C) as c:
            del c


@pytest.mark.skipif(IS_PYPY, reason="Test not meaningful on PyPy")
def test_assert_deallocated_circular2():
    class C:
        def __init__(self):
            self._circular = self
    with pytest.raises(ReferenceError):
        # Still circular reference, no automatic garbage collection
        with assert_deallocated(C):
            pass
