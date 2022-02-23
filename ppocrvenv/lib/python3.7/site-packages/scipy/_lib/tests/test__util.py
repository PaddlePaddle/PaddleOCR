from multiprocessing import Pool
from multiprocessing.pool import Pool as PWL
import os
import math
from fractions import Fraction

import numpy as np
from numpy.testing import assert_equal, assert_
import pytest
from pytest import raises as assert_raises, deprecated_call

import scipy
from scipy._lib._util import (_aligned_zeros, check_random_state, MapWrapper,
                              getfullargspec_no_self, FullArgSpec,
                              rng_integers, _validate_int)


def test__aligned_zeros():
    niter = 10

    def check(shape, dtype, order, align):
        err_msg = repr((shape, dtype, order, align))
        x = _aligned_zeros(shape, dtype, order, align=align)
        if align is None:
            align = np.dtype(dtype).alignment
        assert_equal(x.__array_interface__['data'][0] % align, 0)
        if hasattr(shape, '__len__'):
            assert_equal(x.shape, shape, err_msg)
        else:
            assert_equal(x.shape, (shape,), err_msg)
        assert_equal(x.dtype, dtype)
        if order == "C":
            assert_(x.flags.c_contiguous, err_msg)
        elif order == "F":
            if x.size > 0:
                # Size-0 arrays get invalid flags on NumPy 1.5
                assert_(x.flags.f_contiguous, err_msg)
        elif order is None:
            assert_(x.flags.c_contiguous, err_msg)
        else:
            raise ValueError()

    # try various alignments
    for align in [1, 2, 3, 4, 8, 16, 32, 64, None]:
        for n in [0, 1, 3, 11]:
            for order in ["C", "F", None]:
                for dtype in [np.uint8, np.float64]:
                    for shape in [n, (1, 2, 3, n)]:
                        for j in range(niter):
                            check(shape, dtype, order, align)


def test_check_random_state():
    # If seed is None, return the RandomState singleton used by np.random.
    # If seed is an int, return a new RandomState instance seeded with seed.
    # If seed is already a RandomState instance, return it.
    # Otherwise raise ValueError.
    rsi = check_random_state(1)
    assert_equal(type(rsi), np.random.RandomState)
    rsi = check_random_state(rsi)
    assert_equal(type(rsi), np.random.RandomState)
    rsi = check_random_state(None)
    assert_equal(type(rsi), np.random.RandomState)
    assert_raises(ValueError, check_random_state, 'a')
    if hasattr(np.random, 'Generator'):
        # np.random.Generator is only available in NumPy >= 1.17
        rg = np.random.Generator(np.random.PCG64())
        rsi = check_random_state(rg)
        assert_equal(type(rsi), np.random.Generator)


def test_getfullargspec_no_self():
    p = MapWrapper(1)
    argspec = getfullargspec_no_self(p.__init__)
    assert_equal(argspec, FullArgSpec(['pool'], None, None, (1,), [],
                                      None, {}))
    argspec = getfullargspec_no_self(p.__call__)
    assert_equal(argspec, FullArgSpec(['func', 'iterable'], None, None, None,
                                      [], None, {}))

    class _rv_generic:
        def _rvs(self, a, b=2, c=3, *args, size=None, **kwargs):
            return None

    rv_obj = _rv_generic()
    argspec = getfullargspec_no_self(rv_obj._rvs)
    assert_equal(argspec, FullArgSpec(['a', 'b', 'c'], 'args', 'kwargs',
                                      (2, 3), ['size'], {'size': None}, {}))


def test_mapwrapper_serial():
    in_arg = np.arange(10.)
    out_arg = np.sin(in_arg)

    p = MapWrapper(1)
    assert_(p._mapfunc is map)
    assert_(p.pool is None)
    assert_(p._own_pool is False)
    out = list(p(np.sin, in_arg))
    assert_equal(out, out_arg)

    with assert_raises(RuntimeError):
        p = MapWrapper(0)


def test_pool():
    with Pool(2) as p:
        p.map(math.sin, [1, 2, 3, 4])


def test_mapwrapper_parallel():
    in_arg = np.arange(10.)
    out_arg = np.sin(in_arg)

    with MapWrapper(2) as p:
        out = p(np.sin, in_arg)
        assert_equal(list(out), out_arg)

        assert_(p._own_pool is True)
        assert_(isinstance(p.pool, PWL))
        assert_(p._mapfunc is not None)

    # the context manager should've closed the internal pool
    # check that it has by asking it to calculate again.
    with assert_raises(Exception) as excinfo:
        p(np.sin, in_arg)

    assert_(excinfo.type is ValueError)

    # can also set a PoolWrapper up with a map-like callable instance
    with Pool(2) as p:
        q = MapWrapper(p.map)

        assert_(q._own_pool is False)
        q.close()

        # closing the PoolWrapper shouldn't close the internal pool
        # because it didn't create it
        out = p.map(np.sin, in_arg)
        assert_equal(list(out), out_arg)


# get our custom ones and a few from the "import *" cases
@pytest.mark.parametrize(
    'key', ('ifft', 'diag', 'arccos', 'randn', 'rand', 'array'))
def test_numpy_deprecation(key):
    """Test that 'from numpy import *' functions are deprecated."""
    if key in ('ifft', 'diag', 'arccos'):
        arg = [1.0, 0.]
    elif key == 'finfo':
        arg = float
    else:
        arg = 2
    func = getattr(scipy, key)
    match = r'scipy\.%s is deprecated.*2\.0\.0' % key
    with deprecated_call(match=match) as dep:
        func(arg)  # deprecated
    # in case we catch more than one dep warning
    fnames = [os.path.splitext(d.filename)[0] for d in dep.list]
    basenames = [os.path.basename(fname) for fname in fnames]
    assert 'test__util' in basenames
    if key in ('rand', 'randn'):
        root = np.random
    elif key == 'ifft':
        root = np.fft
    else:
        root = np
    func_np = getattr(root, key)
    func_np(arg)  # not deprecated
    assert func_np is not func
    # classes should remain classes
    if isinstance(func_np, type):
        assert isinstance(func, type)


def test_numpy_deprecation_functionality():
    # Check that the deprecation wrappers don't break basic NumPy
    # functionality
    with deprecated_call():
        x = scipy.array([1, 2, 3], dtype=scipy.float64)
        assert x.dtype == scipy.float64
        assert x.dtype == np.float64

        x = scipy.finfo(scipy.float32)
        assert x.eps == np.finfo(np.float32).eps

        assert scipy.float64 == np.float64
        assert issubclass(np.float64, scipy.float64)


def test_rng_integers():
    rng = np.random.RandomState()

    # test that numbers are inclusive of high point
    arr = rng_integers(rng, low=2, high=5, size=100, endpoint=True)
    assert np.max(arr) == 5
    assert np.min(arr) == 2
    assert arr.shape == (100, )

    # test that numbers are inclusive of high point
    arr = rng_integers(rng, low=5, size=100, endpoint=True)
    assert np.max(arr) == 5
    assert np.min(arr) == 0
    assert arr.shape == (100, )

    # test that numbers are exclusive of high point
    arr = rng_integers(rng, low=2, high=5, size=100, endpoint=False)
    assert np.max(arr) == 4
    assert np.min(arr) == 2
    assert arr.shape == (100, )

    # test that numbers are exclusive of high point
    arr = rng_integers(rng, low=5, size=100, endpoint=False)
    assert np.max(arr) == 4
    assert np.min(arr) == 0
    assert arr.shape == (100, )

    # now try with np.random.Generator
    try:
        rng = np.random.default_rng()
    except AttributeError:
        return

    # test that numbers are inclusive of high point
    arr = rng_integers(rng, low=2, high=5, size=100, endpoint=True)
    assert np.max(arr) == 5
    assert np.min(arr) == 2
    assert arr.shape == (100, )

    # test that numbers are inclusive of high point
    arr = rng_integers(rng, low=5, size=100, endpoint=True)
    assert np.max(arr) == 5
    assert np.min(arr) == 0
    assert arr.shape == (100, )

    # test that numbers are exclusive of high point
    arr = rng_integers(rng, low=2, high=5, size=100, endpoint=False)
    assert np.max(arr) == 4
    assert np.min(arr) == 2
    assert arr.shape == (100, )

    # test that numbers are exclusive of high point
    arr = rng_integers(rng, low=5, size=100, endpoint=False)
    assert np.max(arr) == 4
    assert np.min(arr) == 0
    assert arr.shape == (100, )


class TestValidateInt:

    @pytest.mark.parametrize('n', [4, np.uint8(4), np.int16(4), np.array(4)])
    def test_validate_int(self, n):
        n = _validate_int(n, 'n')
        assert n == 4

    @pytest.mark.parametrize('n', [4.0, np.array([4]), Fraction(4, 1)])
    def test_validate_int_bad(self, n):
        with pytest.raises(TypeError, match='n must be an integer'):
            _validate_int(n, 'n')

    def test_validate_int_below_min(self):
        with pytest.raises(ValueError, match='n must be an integer not '
                                             'less than 0'):
            _validate_int(-1, 'n', 0)
