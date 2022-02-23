import numpy as np
import pytest

from pandas import (
    NA,
    Series,
    StringDtype,
)
import pandas._testing as tm


def test_mask():
    # compare with tested results in test_where
    s = Series(np.random.randn(5))
    cond = s > 0

    rs = s.where(~cond, np.nan)
    tm.assert_series_equal(rs, s.mask(cond))

    rs = s.where(~cond)
    rs2 = s.mask(cond)
    tm.assert_series_equal(rs, rs2)

    rs = s.where(~cond, -s)
    rs2 = s.mask(cond, -s)
    tm.assert_series_equal(rs, rs2)

    cond = Series([True, False, False, True, False], index=s.index)
    s2 = -(s.abs())
    rs = s2.where(~cond[:3])
    rs2 = s2.mask(cond[:3])
    tm.assert_series_equal(rs, rs2)

    rs = s2.where(~cond[:3], -s2)
    rs2 = s2.mask(cond[:3], -s2)
    tm.assert_series_equal(rs, rs2)

    msg = "Array conditional must be same shape as self"
    with pytest.raises(ValueError, match=msg):
        s.mask(1)
    with pytest.raises(ValueError, match=msg):
        s.mask(cond[:3].values, -s)

    # dtype changes
    s = Series([1, 2, 3, 4])
    result = s.mask(s > 2, np.nan)
    expected = Series([1, 2, np.nan, np.nan])
    tm.assert_series_equal(result, expected)

    # see gh-21891
    s = Series([1, 2])
    res = s.mask([True, False])

    exp = Series([np.nan, 2])
    tm.assert_series_equal(res, exp)


def test_mask_inplace():
    s = Series(np.random.randn(5))
    cond = s > 0

    rs = s.copy()
    rs.mask(cond, inplace=True)
    tm.assert_series_equal(rs.dropna(), s[~cond])
    tm.assert_series_equal(rs, s.mask(cond))

    rs = s.copy()
    rs.mask(cond, -s, inplace=True)
    tm.assert_series_equal(rs, s.mask(cond, -s))


def test_mask_stringdtype():
    # GH 40824
    ser = Series(
        ["foo", "bar", "baz", NA],
        index=["id1", "id2", "id3", "id4"],
        dtype=StringDtype(),
    )
    filtered_ser = Series(["this", "that"], index=["id2", "id3"], dtype=StringDtype())
    filter_ser = Series([False, True, True, False])
    result = ser.mask(filter_ser, filtered_ser)

    expected = Series(
        [NA, "this", "that", NA],
        index=["id1", "id2", "id3", "id4"],
        dtype=StringDtype(),
    )
    tm.assert_series_equal(result, expected)


def test_mask_pos_args_deprecation():
    # https://github.com/pandas-dev/pandas/issues/41485
    s = Series(range(5))
    expected = Series([-1, 1, -1, 3, -1])
    cond = s % 2 == 0
    msg = (
        r"In a future version of pandas all arguments of Series.mask except for "
        r"the arguments 'cond' and 'other' will be keyword-only"
    )
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = s.mask(cond, -1, False)
    tm.assert_series_equal(result, expected)
