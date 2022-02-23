from datetime import datetime

import numpy as np
import pytest
from pytz import UTC

from pandas._libs.tslibs import (
    OutOfBoundsTimedelta,
    conversion,
    iNaT,
    timezones,
    tzconversion,
)

from pandas import (
    Timestamp,
    date_range,
)
import pandas._testing as tm


def _compare_utc_to_local(tz_didx):
    def f(x):
        return tzconversion.tz_convert_from_utc_single(x, tz_didx.tz)

    result = tzconversion.tz_convert_from_utc(tz_didx.asi8, tz_didx.tz)
    expected = np.vectorize(f)(tz_didx.asi8)

    tm.assert_numpy_array_equal(result, expected)


def _compare_local_to_utc(tz_didx, naive_didx):
    # Check that tz_localize behaves the same vectorized and pointwise.
    err1 = err2 = None
    try:
        result = tzconversion.tz_localize_to_utc(naive_didx.asi8, tz_didx.tz)
        err1 = None
    except Exception as err:
        err1 = err

    try:
        expected = naive_didx.map(lambda x: x.tz_localize(tz_didx.tz)).asi8
    except Exception as err:
        err2 = err

    if err1 is not None:
        assert type(err1) == type(err2)
    else:
        assert err2 is None
        tm.assert_numpy_array_equal(result, expected)


def test_tz_convert_single_matches_tz_convert_hourly(tz_aware_fixture):
    tz = tz_aware_fixture
    tz_didx = date_range("2014-03-01", "2015-01-10", freq="H", tz=tz)
    naive_didx = date_range("2014-03-01", "2015-01-10", freq="H")

    _compare_utc_to_local(tz_didx)
    _compare_local_to_utc(tz_didx, naive_didx)


@pytest.mark.parametrize("freq", ["D", "A"])
def test_tz_convert_single_matches_tz_convert(tz_aware_fixture, freq):
    tz = tz_aware_fixture
    tz_didx = date_range("2000-01-01", "2020-01-01", freq=freq, tz=tz)
    naive_didx = date_range("2000-01-01", "2020-01-01", freq=freq)

    _compare_utc_to_local(tz_didx)
    _compare_local_to_utc(tz_didx, naive_didx)


@pytest.mark.parametrize(
    "arr",
    [
        pytest.param(np.array([], dtype=np.int64), id="empty"),
        pytest.param(np.array([iNaT], dtype=np.int64), id="all_nat"),
    ],
)
def test_tz_convert_corner(arr):
    result = tzconversion.tz_convert_from_utc(arr, timezones.maybe_get_tz("Asia/Tokyo"))
    tm.assert_numpy_array_equal(result, arr)


def test_tz_convert_readonly():
    # GH#35530
    arr = np.array([0], dtype=np.int64)
    arr.setflags(write=False)
    result = tzconversion.tz_convert_from_utc(arr, UTC)
    tm.assert_numpy_array_equal(result, arr)


@pytest.mark.parametrize("copy", [True, False])
@pytest.mark.parametrize("dtype", ["M8[ns]", "M8[s]"])
def test_length_zero_copy(dtype, copy):
    arr = np.array([], dtype=dtype)
    result = conversion.ensure_datetime64ns(arr, copy=copy)
    assert result.base is (None if copy else arr)


def test_ensure_datetime64ns_bigendian():
    # GH#29684
    arr = np.array([np.datetime64(1, "ms")], dtype=">M8[ms]")
    result = conversion.ensure_datetime64ns(arr)

    expected = np.array([np.datetime64(1, "ms")], dtype="M8[ns]")
    tm.assert_numpy_array_equal(result, expected)


def test_ensure_timedelta64ns_overflows():
    arr = np.arange(10).astype("m8[Y]") * 100
    msg = r"Out of bounds for nanosecond timedelta64\[Y\] 900"
    with pytest.raises(OutOfBoundsTimedelta, match=msg):
        conversion.ensure_timedelta64ns(arr)


class SubDatetime(datetime):
    pass


@pytest.mark.parametrize(
    "dt, expected",
    [
        pytest.param(
            Timestamp("2000-01-01"), Timestamp("2000-01-01", tz=UTC), id="timestamp"
        ),
        pytest.param(
            datetime(2000, 1, 1), datetime(2000, 1, 1, tzinfo=UTC), id="datetime"
        ),
        pytest.param(
            SubDatetime(2000, 1, 1),
            SubDatetime(2000, 1, 1, tzinfo=UTC),
            id="subclassed_datetime",
        ),
    ],
)
def test_localize_pydatetime_dt_types(dt, expected):
    # GH 25851
    # ensure that subclassed datetime works with
    # localize_pydatetime
    result = conversion.localize_pydatetime(dt, UTC)
    assert result == expected
