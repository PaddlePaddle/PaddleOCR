from numpy.testing import assert_equal, assert_
from pytest import raises as assert_raises

import time
import pytest
import ctypes
import threading
from scipy._lib import _ccallback_c as _test_ccallback_cython
from scipy._lib import _test_ccallback
from scipy._lib._ccallback import LowLevelCallable

try:
    import cffi
    HAVE_CFFI = True
except ImportError:
    HAVE_CFFI = False


ERROR_VALUE = 2.0


def callback_python(a, user_data=None):
    if a == ERROR_VALUE:
        raise ValueError("bad value")

    if user_data is None:
        return a + 1
    else:
        return a + user_data

def _get_cffi_func(base, signature):
    if not HAVE_CFFI:
        pytest.skip("cffi not installed")

    # Get function address
    voidp = ctypes.cast(base, ctypes.c_void_p)
    address = voidp.value

    # Create corresponding cffi handle
    ffi = cffi.FFI()
    func = ffi.cast(signature, address)
    return func


def _get_ctypes_data():
    value = ctypes.c_double(2.0)
    return ctypes.cast(ctypes.pointer(value), ctypes.c_voidp)


def _get_cffi_data():
    if not HAVE_CFFI:
        pytest.skip("cffi not installed")
    ffi = cffi.FFI()
    return ffi.new('double *', 2.0)


CALLERS = {
    'simple': _test_ccallback.test_call_simple,
    'nodata': _test_ccallback.test_call_nodata,
    'nonlocal': _test_ccallback.test_call_nonlocal,
    'cython': _test_ccallback_cython.test_call_cython,
}

# These functions have signatures known to the callers
FUNCS = {
    'python': lambda: callback_python,
    'capsule': lambda: _test_ccallback.test_get_plus1_capsule(),
    'cython': lambda: LowLevelCallable.from_cython(_test_ccallback_cython, "plus1_cython"),
    'ctypes': lambda: _test_ccallback_cython.plus1_ctypes,
    'cffi': lambda: _get_cffi_func(_test_ccallback_cython.plus1_ctypes,
                                   'double (*)(double, int *, void *)'),
    'capsule_b': lambda: _test_ccallback.test_get_plus1b_capsule(),
    'cython_b': lambda: LowLevelCallable.from_cython(_test_ccallback_cython, "plus1b_cython"),
    'ctypes_b': lambda: _test_ccallback_cython.plus1b_ctypes,
    'cffi_b': lambda: _get_cffi_func(_test_ccallback_cython.plus1b_ctypes,
                                     'double (*)(double, double, int *, void *)'),
}

# These functions have signatures the callers don't know
BAD_FUNCS = {
    'capsule_bc': lambda: _test_ccallback.test_get_plus1bc_capsule(),
    'cython_bc': lambda: LowLevelCallable.from_cython(_test_ccallback_cython, "plus1bc_cython"),
    'ctypes_bc': lambda: _test_ccallback_cython.plus1bc_ctypes,
    'cffi_bc': lambda: _get_cffi_func(_test_ccallback_cython.plus1bc_ctypes,
                                      'double (*)(double, double, double, int *, void *)'),
}

USER_DATAS = {
    'ctypes': _get_ctypes_data,
    'cffi': _get_cffi_data,
    'capsule': _test_ccallback.test_get_data_capsule,
}


def test_callbacks():
    def check(caller, func, user_data):
        caller = CALLERS[caller]
        func = FUNCS[func]()
        user_data = USER_DATAS[user_data]()

        if func is callback_python:
            func2 = lambda x: func(x, 2.0)
        else:
            func2 = LowLevelCallable(func, user_data)
            func = LowLevelCallable(func)

        # Test basic call
        assert_equal(caller(func, 1.0), 2.0)

        # Test 'bad' value resulting to an error
        assert_raises(ValueError, caller, func, ERROR_VALUE)

        # Test passing in user_data
        assert_equal(caller(func2, 1.0), 3.0)

    for caller in sorted(CALLERS.keys()):
        for func in sorted(FUNCS.keys()):
            for user_data in sorted(USER_DATAS.keys()):
                check(caller, func, user_data)


def test_bad_callbacks():
    def check(caller, func, user_data):
        caller = CALLERS[caller]
        user_data = USER_DATAS[user_data]()
        func = BAD_FUNCS[func]()

        if func is callback_python:
            func2 = lambda x: func(x, 2.0)
        else:
            func2 = LowLevelCallable(func, user_data)
            func = LowLevelCallable(func)

        # Test that basic call fails
        assert_raises(ValueError, caller, LowLevelCallable(func), 1.0)

        # Test that passing in user_data also fails
        assert_raises(ValueError, caller, func2, 1.0)

        # Test error message
        llfunc = LowLevelCallable(func)
        try:
            caller(llfunc, 1.0)
        except ValueError as err:
            msg = str(err)
            assert_(llfunc.signature in msg, msg)
            assert_('double (double, double, int *, void *)' in msg, msg)

    for caller in sorted(CALLERS.keys()):
        for func in sorted(BAD_FUNCS.keys()):
            for user_data in sorted(USER_DATAS.keys()):
                check(caller, func, user_data)


def test_signature_override():
    caller = _test_ccallback.test_call_simple
    func = _test_ccallback.test_get_plus1_capsule()

    llcallable = LowLevelCallable(func, signature="bad signature")
    assert_equal(llcallable.signature, "bad signature")
    assert_raises(ValueError, caller, llcallable, 3)

    llcallable = LowLevelCallable(func, signature="double (double, int *, void *)")
    assert_equal(llcallable.signature, "double (double, int *, void *)")
    assert_equal(caller(llcallable, 3), 4)


def test_threadsafety():
    def callback(a, caller):
        if a <= 0:
            return 1
        else:
            res = caller(lambda x: callback(x, caller), a - 1)
            return 2*res

    def check(caller):
        caller = CALLERS[caller]

        results = []

        count = 10

        def run():
            time.sleep(0.01)
            r = caller(lambda x: callback(x, caller), count)
            results.append(r)

        threads = [threading.Thread(target=run) for j in range(20)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert_equal(results, [2.0**count]*len(threads))

    for caller in CALLERS.keys():
        check(caller)
