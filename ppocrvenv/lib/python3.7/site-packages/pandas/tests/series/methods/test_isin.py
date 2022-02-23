import numpy as np
import pytest

import pandas as pd
from pandas import (
    Series,
    date_range,
)
import pandas._testing as tm
from pandas.core.arrays import PeriodArray


class TestSeriesIsIn:
    def test_isin(self):
        s = Series(["A", "B", "C", "a", "B", "B", "A", "C"])

        result = s.isin(["A", "C"])
        expected = Series([True, False, True, False, False, False, True, True])
        tm.assert_series_equal(result, expected)

        # GH#16012
        # This specific issue has to have a series over 1e6 in len, but the
        # comparison array (in_list) must be large enough so that numpy doesn't
        # do a manual masking trick that will avoid this issue altogether
        s = Series(list("abcdefghijk" * 10 ** 5))
        # If numpy doesn't do the manual comparison/mask, these
        # unorderable mixed types are what cause the exception in numpy
        in_list = [-1, "a", "b", "G", "Y", "Z", "E", "K", "E", "S", "I", "R", "R"] * 6

        assert s.isin(in_list).sum() == 200000

    def test_isin_with_string_scalar(self):
        # GH#4763
        s = Series(["A", "B", "C", "a", "B", "B", "A", "C"])
        msg = (
            r"only list-like objects are allowed to be passed to isin\(\), "
            r"you passed a \[str\]"
        )
        with pytest.raises(TypeError, match=msg):
            s.isin("a")

        s = Series(["aaa", "b", "c"])
        with pytest.raises(TypeError, match=msg):
            s.isin("aaa")

    def test_isin_with_i8(self):
        # GH#5021

        expected = Series([True, True, False, False, False])
        expected2 = Series([False, True, False, False, False])

        # datetime64[ns]
        s = Series(date_range("jan-01-2013", "jan-05-2013"))

        result = s.isin(s[0:2])
        tm.assert_series_equal(result, expected)

        result = s.isin(s[0:2].values)
        tm.assert_series_equal(result, expected)

        # fails on dtype conversion in the first place
        result = s.isin(np.asarray(s[0:2].values).astype("datetime64[D]"))
        tm.assert_series_equal(result, expected)

        result = s.isin([s[1]])
        tm.assert_series_equal(result, expected2)

        result = s.isin([np.datetime64(s[1])])
        tm.assert_series_equal(result, expected2)

        result = s.isin(set(s[0:2]))
        tm.assert_series_equal(result, expected)

        # timedelta64[ns]
        s = Series(pd.to_timedelta(range(5), unit="d"))
        result = s.isin(s[0:2])
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("empty", [[], Series(dtype=object), np.array([])])
    def test_isin_empty(self, empty):
        # see GH#16991
        s = Series(["a", "b"])
        expected = Series([False, False])

        result = s.isin(empty)
        tm.assert_series_equal(expected, result)

    def test_isin_read_only(self):
        # https://github.com/pandas-dev/pandas/issues/37174
        arr = np.array([1, 2, 3])
        arr.setflags(write=False)
        s = Series([1, 2, 3])
        result = s.isin(arr)
        expected = Series([True, True, True])
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("dtype", [object, None])
    def test_isin_dt64_values_vs_ints(self, dtype):
        # GH#36621 dont cast integers to datetimes for isin
        dti = date_range("2013-01-01", "2013-01-05")
        ser = Series(dti)

        comps = np.asarray([1356998400000000000], dtype=dtype)

        res = dti.isin(comps)
        expected = np.array([False] * len(dti), dtype=bool)
        tm.assert_numpy_array_equal(res, expected)

        res = ser.isin(comps)
        tm.assert_series_equal(res, Series(expected))

        res = pd.core.algorithms.isin(ser, comps)
        tm.assert_numpy_array_equal(res, expected)

    def test_isin_tzawareness_mismatch(self):
        dti = date_range("2013-01-01", "2013-01-05")
        ser = Series(dti)

        other = dti.tz_localize("UTC")

        res = dti.isin(other)
        expected = np.array([False] * len(dti), dtype=bool)
        tm.assert_numpy_array_equal(res, expected)

        res = ser.isin(other)
        tm.assert_series_equal(res, Series(expected))

        res = pd.core.algorithms.isin(ser, other)
        tm.assert_numpy_array_equal(res, expected)

    def test_isin_period_freq_mismatch(self):
        dti = date_range("2013-01-01", "2013-01-05")
        pi = dti.to_period("M")
        ser = Series(pi)

        # We construct another PeriodIndex with the same i8 values
        #  but different dtype
        dtype = dti.to_period("Y").dtype
        other = PeriodArray._simple_new(pi.asi8, dtype=dtype)

        res = pi.isin(other)
        expected = np.array([False] * len(pi), dtype=bool)
        tm.assert_numpy_array_equal(res, expected)

        res = ser.isin(other)
        tm.assert_series_equal(res, Series(expected))

        res = pd.core.algorithms.isin(ser, other)
        tm.assert_numpy_array_equal(res, expected)

    @pytest.mark.parametrize("values", [[-9.0, 0.0], [-9, 0]])
    def test_isin_float_in_int_series(self, values):
        # GH#19356 GH#21804
        ser = Series(values)
        result = ser.isin([-9, -0.5])
        expected = Series([True, False])
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("dtype", ["boolean", "Int64", "Float64"])
    @pytest.mark.parametrize(
        "data,values,expected",
        [
            ([0, 1, 0], [1], [False, True, False]),
            ([0, 1, 0], [1, pd.NA], [False, True, False]),
            ([0, pd.NA, 0], [1, 0], [True, False, True]),
            ([0, 1, pd.NA], [1, pd.NA], [False, True, True]),
            ([0, 1, pd.NA], [1, np.nan], [False, True, False]),
            ([0, pd.NA, pd.NA], [np.nan, pd.NaT, None], [False, False, False]),
        ],
    )
    def test_isin_masked_types(self, dtype, data, values, expected):
        # GH#42405
        ser = Series(data, dtype=dtype)

        result = ser.isin(values)
        expected = Series(expected, dtype="boolean")

        tm.assert_series_equal(result, expected)


@pytest.mark.slow
def test_isin_large_series_mixed_dtypes_and_nan():
    # https://github.com/pandas-dev/pandas/issues/37094
    # combination of object dtype for the values and > 1_000_000 elements
    ser = Series([1, 2, np.nan] * 1_000_000)
    result = ser.isin({"foo", "bar"})
    expected = Series([False] * 3 * 1_000_000)
    tm.assert_series_equal(result, expected)
