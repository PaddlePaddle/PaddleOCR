import cmath
import math

import numpy as np

from numpy cimport import_array

import_array()

from pandas._libs.lib import is_complex

from pandas._libs.util cimport (
    is_array,
    is_real_number_object,
)

from pandas.core.dtypes.common import is_dtype_equal
from pandas.core.dtypes.missing import (
    array_equivalent,
    isna,
)


cdef bint isiterable(obj):
    return hasattr(obj, '__iter__')


cdef bint has_length(obj):
    return hasattr(obj, '__len__')


cdef bint is_dictlike(obj):
    return hasattr(obj, 'keys') and hasattr(obj, '__getitem__')


cpdef assert_dict_equal(a, b, bint compare_keys=True):
    assert is_dictlike(a) and is_dictlike(b), (
        "Cannot compare dict objects, one or both is not dict-like"
    )

    a_keys = frozenset(a.keys())
    b_keys = frozenset(b.keys())

    if compare_keys:
        assert a_keys == b_keys

    for k in a_keys:
        assert_almost_equal(a[k], b[k])

    return True


cpdef assert_almost_equal(a, b,
                          rtol=1.e-5, atol=1.e-8,
                          bint check_dtype=True,
                          obj=None, lobj=None, robj=None, index_values=None):
    """
    Check that left and right objects are almost equal.

    Parameters
    ----------
    a : object
    b : object
    rtol : float, default 1e-5
        Relative tolerance.

        .. versionadded:: 1.1.0
    atol : float, default 1e-8
        Absolute tolerance.

        .. versionadded:: 1.1.0
    check_dtype: bool, default True
        check dtype if both a and b are np.ndarray.
    obj : str, default None
        Specify object name being compared, internally used to show
        appropriate assertion message.
    lobj : str, default None
        Specify left object name being compared, internally used to show
        appropriate assertion message.
    robj : str, default None
        Specify right object name being compared, internally used to show
        appropriate assertion message.
    index_values : ndarray, default None
        Specify shared index values of objects being compared, internally used
        to show appropriate assertion message.

        .. versionadded:: 1.1.0

    """
    cdef:
        double diff = 0.0
        Py_ssize_t i, na, nb
        double fa, fb
        bint is_unequal = False, a_is_ndarray, b_is_ndarray

    if lobj is None:
        lobj = a
    if robj is None:
        robj = b

    if isinstance(a, dict) or isinstance(b, dict):
        return assert_dict_equal(a, b)

    if isinstance(a, str) or isinstance(b, str):
        assert a == b, f"{a} != {b}"
        return True

    a_is_ndarray = is_array(a)
    b_is_ndarray = is_array(b)

    if obj is None:
        if a_is_ndarray or b_is_ndarray:
            obj = 'numpy array'
        else:
            obj = 'Iterable'

    if isiterable(a):

        if not isiterable(b):
            from pandas._testing import assert_class_equal

            # classes can't be the same, to raise error
            assert_class_equal(a, b, obj=obj)

        assert has_length(a) and has_length(b), (
            f"Can't compare objects without length, one or both is invalid: ({a}, {b})"
        )

        if a_is_ndarray and b_is_ndarray:
            na, nb = a.size, b.size
            if a.shape != b.shape:
                from pandas._testing import raise_assert_detail
                raise_assert_detail(
                    obj, f'{obj} shapes are different', a.shape, b.shape)

            if check_dtype and not is_dtype_equal(a.dtype, b.dtype):
                from pandas._testing import assert_attr_equal
                assert_attr_equal('dtype', a, b, obj=obj)

            if array_equivalent(a, b, strict_nan=True):
                return True

        else:
            na, nb = len(a), len(b)

        if na != nb:
            from pandas._testing import raise_assert_detail

            # if we have a small diff set, print it
            if abs(na - nb) < 10:
                r = list(set(a) ^ set(b))
            else:
                r = None

            raise_assert_detail(obj, f"{obj} length are different", na, nb, r)

        for i in range(len(a)):
            try:
                assert_almost_equal(a[i], b[i], rtol=rtol, atol=atol)
            except AssertionError:
                is_unequal = True
                diff += 1

        if is_unequal:
            from pandas._testing import raise_assert_detail
            msg = (f"{obj} values are different "
                   f"({np.round(diff * 100.0 / na, 5)} %)")
            raise_assert_detail(obj, msg, lobj, robj, index_values=index_values)

        return True

    elif isiterable(b):
        from pandas._testing import assert_class_equal

        # classes can't be the same, to raise error
        assert_class_equal(a, b, obj=obj)

    if isna(a) and isna(b):
        # TODO: Should require same-dtype NA?
        # nan / None comparison
        return True

    if a == b:
        # object comparison
        return True

    if is_real_number_object(a) and is_real_number_object(b):
        if array_equivalent(a, b, strict_nan=True):
            # inf comparison
            return True

        fa, fb = a, b

        if not math.isclose(fa, fb, rel_tol=rtol, abs_tol=atol):
            assert False, (f"expected {fb:.5f} but got {fa:.5f}, "
                           f"with rtol={rtol}, atol={atol}")
        return True

    if is_complex(a) and is_complex(b):
        if array_equivalent(a, b, strict_nan=True):
            # inf comparison
            return True

        if not cmath.isclose(a, b, rel_tol=rtol, abs_tol=atol):
            assert False, (f"expected {b:.5f} but got {a:.5f}, "
                           f"with rtol={rtol}, atol={atol}")
        return True

    raise AssertionError(f"{a} != {b}")
