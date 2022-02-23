import numpy as np
import pytest

import pandas as pd
from pandas import (
    Float64Index,
    Int64Index,
    RangeIndex,
    UInt64Index,
)
import pandas._testing as tm
from pandas.core.computation import expressions as expr


@pytest.fixture(
    autouse=True, scope="module", params=[0, 1000000], ids=["numexpr", "python"]
)
def switch_numexpr_min_elements(request):
    _MIN_ELEMENTS = expr._MIN_ELEMENTS
    expr._MIN_ELEMENTS = request.param
    yield request.param
    expr._MIN_ELEMENTS = _MIN_ELEMENTS


# ------------------------------------------------------------------
# Helper Functions


def id_func(x):
    if isinstance(x, tuple):
        assert len(x) == 2
        return x[0].__name__ + "-" + str(x[1])
    else:
        return x.__name__


# ------------------------------------------------------------------


@pytest.fixture(params=[1, np.array(1, dtype=np.int64)])
def one(request):
    """
    Several variants of integer value 1. The zero-dim integer array
    behaves like an integer.

    This fixture can be used to check that datetimelike indexes handle
    addition and subtraction of integers and zero-dimensional arrays
    of integers.

    Examples
    --------
    >>> dti = pd.date_range('2016-01-01', periods=2, freq='H')
    >>> dti
    DatetimeIndex(['2016-01-01 00:00:00', '2016-01-01 01:00:00'],
    dtype='datetime64[ns]', freq='H')
    >>> dti + one
    DatetimeIndex(['2016-01-01 01:00:00', '2016-01-01 02:00:00'],
    dtype='datetime64[ns]', freq='H')
    """
    return request.param


zeros = [
    box_cls([0] * 5, dtype=dtype)
    for box_cls in [pd.Index, np.array, pd.array]
    for dtype in [np.int64, np.uint64, np.float64]
]
zeros.extend(
    [box_cls([-0.0] * 5, dtype=np.float64) for box_cls in [pd.Index, np.array]]
)
zeros.extend([np.array(0, dtype=dtype) for dtype in [np.int64, np.uint64, np.float64]])
zeros.extend([np.array(-0.0, dtype=np.float64)])
zeros.extend([0, 0.0, -0.0])


@pytest.fixture(params=zeros)
def zero(request):
    """
    Several types of scalar zeros and length 5 vectors of zeros.

    This fixture can be used to check that numeric-dtype indexes handle
    division by any zero numeric-dtype.

    Uses vector of length 5 for broadcasting with `numeric_idx` fixture,
    which creates numeric-dtype vectors also of length 5.

    Examples
    --------
    >>> arr = RangeIndex(5)
    >>> arr / zeros
    Float64Index([nan, inf, inf, inf, inf], dtype='float64')
    """
    return request.param


# ------------------------------------------------------------------
# Vector Fixtures


@pytest.fixture(
    params=[
        Float64Index(np.arange(5, dtype="float64")),
        Int64Index(np.arange(5, dtype="int64")),
        UInt64Index(np.arange(5, dtype="uint64")),
        RangeIndex(5),
    ],
    ids=lambda x: type(x).__name__,
)
def numeric_idx(request):
    """
    Several types of numeric-dtypes Index objects
    """
    return request.param


# ------------------------------------------------------------------
# Scalar Fixtures


@pytest.fixture(
    params=[
        pd.Timedelta("5m4s").to_pytimedelta(),
        pd.Timedelta("5m4s"),
        pd.Timedelta("5m4s").to_timedelta64(),
    ],
    ids=lambda x: type(x).__name__,
)
def scalar_td(request):
    """
    Several variants of Timedelta scalars representing 5 minutes and 4 seconds
    """
    return request.param


@pytest.fixture(
    params=[
        pd.offsets.Day(3),
        pd.offsets.Hour(72),
        pd.Timedelta(days=3).to_pytimedelta(),
        pd.Timedelta("72:00:00"),
        np.timedelta64(3, "D"),
        np.timedelta64(72, "h"),
    ],
    ids=lambda x: type(x).__name__,
)
def three_days(request):
    """
    Several timedelta-like and DateOffset objects that each represent
    a 3-day timedelta
    """
    return request.param


@pytest.fixture(
    params=[
        pd.offsets.Hour(2),
        pd.offsets.Minute(120),
        pd.Timedelta(hours=2).to_pytimedelta(),
        pd.Timedelta(seconds=2 * 3600),
        np.timedelta64(2, "h"),
        np.timedelta64(120, "m"),
    ],
    ids=lambda x: type(x).__name__,
)
def two_hours(request):
    """
    Several timedelta-like and DateOffset objects that each represent
    a 2-hour timedelta
    """
    return request.param


_common_mismatch = [
    pd.offsets.YearBegin(2),
    pd.offsets.MonthBegin(1),
    pd.offsets.Minute(),
]


@pytest.fixture(
    params=[
        pd.Timedelta(minutes=30).to_pytimedelta(),
        np.timedelta64(30, "s"),
        pd.Timedelta(seconds=30),
    ]
    + _common_mismatch
)
def not_hourly(request):
    """
    Several timedelta-like and DateOffset instances that are _not_
    compatible with Hourly frequencies.
    """
    return request.param


@pytest.fixture(
    params=[
        np.timedelta64(4, "h"),
        pd.Timedelta(hours=23).to_pytimedelta(),
        pd.Timedelta("23:00:00"),
    ]
    + _common_mismatch
)
def not_daily(request):
    """
    Several timedelta-like and DateOffset instances that are _not_
    compatible with Daily frequencies.
    """
    return request.param


@pytest.fixture(
    params=[
        np.timedelta64(365, "D"),
        pd.Timedelta(days=365).to_pytimedelta(),
        pd.Timedelta(days=365),
    ]
    + _common_mismatch
)
def mismatched_freq(request):
    """
    Several timedelta-like and DateOffset instances that are _not_
    compatible with Monthly or Annual frequencies.
    """
    return request.param


# ------------------------------------------------------------------


@pytest.fixture(params=[pd.Index, pd.Series, pd.DataFrame, pd.array], ids=id_func)
def box_with_array(request):
    """
    Fixture to test behavior for Index, Series, DataFrame, and pandas Array
    classes
    """
    return request.param


@pytest.fixture(params=[pd.Index, pd.Series, tm.to_array, np.array, list], ids=id_func)
def box_1d_array(request):
    """
    Fixture to test behavior for Index, Series, tm.to_array, numpy Array and list
    classes
    """
    return request.param


# alias so we can use the same fixture for multiple parameters in a test
box_with_array2 = box_with_array
