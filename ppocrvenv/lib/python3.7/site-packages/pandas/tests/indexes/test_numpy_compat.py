import numpy as np
import pytest

from pandas.compat import np_version_under1p18

from pandas import (
    DatetimeIndex,
    Float64Index,
    Index,
    Int64Index,
    PeriodIndex,
    RangeIndex,
    TimedeltaIndex,
    UInt64Index,
)
import pandas._testing as tm
from pandas.core.indexes.datetimelike import DatetimeIndexOpsMixin


@pytest.mark.parametrize(
    "func",
    [
        np.exp,
        np.exp2,
        np.expm1,
        np.log,
        np.log2,
        np.log10,
        np.log1p,
        np.sqrt,
        np.sin,
        np.cos,
        np.tan,
        np.arcsin,
        np.arccos,
        np.arctan,
        np.sinh,
        np.cosh,
        np.tanh,
        np.arcsinh,
        np.arccosh,
        np.arctanh,
        np.deg2rad,
        np.rad2deg,
    ],
    ids=lambda x: x.__name__,
)
def test_numpy_ufuncs_basic(index, func):
    # test ufuncs of numpy, see:
    # https://numpy.org/doc/stable/reference/ufuncs.html

    if isinstance(index, DatetimeIndexOpsMixin):
        with tm.external_error_raised((TypeError, AttributeError)):
            with np.errstate(all="ignore"):
                func(index)
    elif isinstance(index, (Float64Index, Int64Index, UInt64Index, RangeIndex)):
        # coerces to float (e.g. np.sin)
        with np.errstate(all="ignore"):
            result = func(index)
            exp = Index(func(index.values), name=index.name)

        tm.assert_index_equal(result, exp)
        assert isinstance(result, Float64Index)
    else:
        # raise AttributeError or TypeError
        if len(index) == 0:
            pass
        else:
            with tm.external_error_raised((TypeError, AttributeError)):
                with np.errstate(all="ignore"):
                    func(index)


@pytest.mark.parametrize(
    "func", [np.isfinite, np.isinf, np.isnan, np.signbit], ids=lambda x: x.__name__
)
def test_numpy_ufuncs_other(index, func, request):
    # test ufuncs of numpy, see:
    # https://numpy.org/doc/stable/reference/ufuncs.html
    if isinstance(index, (DatetimeIndex, TimedeltaIndex)):
        if (
            isinstance(index, DatetimeIndex)
            and index.tz is not None
            and func in [np.isfinite, np.isnan, np.isinf]
            and (
                not np_version_under1p18
                or (np_version_under1p18 and func is np.isfinite)
            )
        ):
            mark = pytest.mark.xfail(reason="__array_ufunc__ is not defined")
            request.node.add_marker(mark)

        if not np_version_under1p18 and func in [np.isfinite, np.isinf, np.isnan]:
            # numpy 1.18(dev) changed isinf and isnan to not raise on dt64/tfd64
            result = func(index)
            assert isinstance(result, np.ndarray)

        elif func is np.isfinite:
            # ok under numpy >= 1.17
            # Results in bool array
            result = func(index)
            assert isinstance(result, np.ndarray)
        else:
            with tm.external_error_raised(TypeError):
                func(index)

    elif isinstance(index, PeriodIndex):
        with tm.external_error_raised(TypeError):
            func(index)

    elif isinstance(index, (Float64Index, Int64Index, UInt64Index, RangeIndex)):
        # Results in bool array
        result = func(index)
        assert isinstance(result, np.ndarray)
        assert not isinstance(result, Index)
    else:
        if len(index) == 0:
            pass
        else:
            with tm.external_error_raised(TypeError):
                func(index)
