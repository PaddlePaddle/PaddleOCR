from datetime import (
    time,
    timedelta,
)

import numpy as np
import pytest

from pandas.errors import OutOfBoundsTimedelta

import pandas as pd
from pandas import (
    Series,
    TimedeltaIndex,
    isna,
    to_timedelta,
)
import pandas._testing as tm
from pandas.core.arrays import TimedeltaArray


class TestTimedeltas:
    @pytest.mark.parametrize("readonly", [True, False])
    def test_to_timedelta_readonly(self, readonly):
        # GH#34857
        arr = np.array([], dtype=object)
        if readonly:
            arr.setflags(write=False)
        result = to_timedelta(arr)
        expected = to_timedelta([])
        tm.assert_index_equal(result, expected)

    def test_to_timedelta(self):

        result = to_timedelta(["", ""])
        assert isna(result).all()

        # pass thru
        result = to_timedelta(np.array([np.timedelta64(1, "s")]))
        expected = pd.Index(np.array([np.timedelta64(1, "s")]))
        tm.assert_index_equal(result, expected)

        # Series
        expected = Series([timedelta(days=1), timedelta(days=1, seconds=1)])
        result = to_timedelta(Series(["1d", "1days 00:00:01"]))
        tm.assert_series_equal(result, expected)

        # with units
        result = TimedeltaIndex(
            [np.timedelta64(0, "ns"), np.timedelta64(10, "s").astype("m8[ns]")]
        )
        expected = to_timedelta([0, 10], unit="s")
        tm.assert_index_equal(result, expected)

        # arrays of various dtypes
        arr = np.array([1] * 5, dtype="int64")
        result = to_timedelta(arr, unit="s")
        expected = TimedeltaIndex([np.timedelta64(1, "s")] * 5)
        tm.assert_index_equal(result, expected)

        arr = np.array([1] * 5, dtype="int64")
        result = to_timedelta(arr, unit="m")
        expected = TimedeltaIndex([np.timedelta64(1, "m")] * 5)
        tm.assert_index_equal(result, expected)

        arr = np.array([1] * 5, dtype="int64")
        result = to_timedelta(arr, unit="h")
        expected = TimedeltaIndex([np.timedelta64(1, "h")] * 5)
        tm.assert_index_equal(result, expected)

        arr = np.array([1] * 5, dtype="timedelta64[s]")
        result = to_timedelta(arr)
        expected = TimedeltaIndex([np.timedelta64(1, "s")] * 5)
        tm.assert_index_equal(result, expected)

        arr = np.array([1] * 5, dtype="timedelta64[D]")
        result = to_timedelta(arr)
        expected = TimedeltaIndex([np.timedelta64(1, "D")] * 5)
        tm.assert_index_equal(result, expected)

    def test_to_timedelta_oob_non_nano(self):
        arr = np.array([pd.NaT.value + 1], dtype="timedelta64[s]")

        msg = r"Out of bounds for nanosecond timedelta64\[s\] -9223372036854775807"
        with pytest.raises(OutOfBoundsTimedelta, match=msg):
            to_timedelta(arr)

        with pytest.raises(OutOfBoundsTimedelta, match=msg):
            TimedeltaIndex(arr)

        with pytest.raises(OutOfBoundsTimedelta, match=msg):
            TimedeltaArray._from_sequence(arr)

    def test_to_timedelta_dataframe(self):
        # GH 11776
        arr = np.arange(10).reshape(2, 5)
        df = pd.DataFrame(np.arange(10).reshape(2, 5))
        for arg in (arr, df):
            with pytest.raises(TypeError, match="1-d array"):
                to_timedelta(arg)
            for errors in ["ignore", "raise", "coerce"]:
                with pytest.raises(TypeError, match="1-d array"):
                    to_timedelta(arg, errors=errors)

    def test_to_timedelta_invalid(self):

        # bad value for errors parameter
        msg = "errors must be one of"
        with pytest.raises(ValueError, match=msg):
            to_timedelta(["foo"], errors="never")

        # these will error
        msg = "invalid unit abbreviation: foo"
        with pytest.raises(ValueError, match=msg):
            to_timedelta([1, 2], unit="foo")
        with pytest.raises(ValueError, match=msg):
            to_timedelta(1, unit="foo")

        # time not supported ATM
        msg = (
            "Value must be Timedelta, string, integer, float, timedelta or convertible"
        )
        with pytest.raises(ValueError, match=msg):
            to_timedelta(time(second=1))
        assert to_timedelta(time(second=1), errors="coerce") is pd.NaT

        msg = "Could not convert 'foo' to NumPy timedelta"
        with pytest.raises(ValueError, match=msg):
            to_timedelta(["foo", "bar"])

        tm.assert_index_equal(
            TimedeltaIndex([pd.NaT, pd.NaT]),
            to_timedelta(["foo", "bar"], errors="coerce"),
        )

        tm.assert_index_equal(
            TimedeltaIndex(["1 day", pd.NaT, "1 min"]),
            to_timedelta(["1 day", "bar", "1 min"], errors="coerce"),
        )

        # gh-13613: these should not error because errors='ignore'
        invalid_data = "apple"
        assert invalid_data == to_timedelta(invalid_data, errors="ignore")

        invalid_data = ["apple", "1 days"]
        tm.assert_numpy_array_equal(
            np.array(invalid_data, dtype=object),
            to_timedelta(invalid_data, errors="ignore"),
        )

        invalid_data = pd.Index(["apple", "1 days"])
        tm.assert_index_equal(invalid_data, to_timedelta(invalid_data, errors="ignore"))

        invalid_data = Series(["apple", "1 days"])
        tm.assert_series_equal(
            invalid_data, to_timedelta(invalid_data, errors="ignore")
        )

    @pytest.mark.parametrize(
        "val, warning",
        [
            ("1M", FutureWarning),
            ("1 M", FutureWarning),
            ("1Y", FutureWarning),
            ("1 Y", FutureWarning),
            ("1y", FutureWarning),
            ("1 y", FutureWarning),
            ("1m", None),
            ("1 m", None),
            ("1 day", None),
            ("2day", None),
        ],
    )
    def test_unambiguous_timedelta_values(self, val, warning):
        # GH36666 Deprecate use of strings denoting units with 'M', 'Y', 'm' or 'y'
        # in pd.to_timedelta
        msg = "Units 'M', 'Y' and 'y' do not represent unambiguous timedelta"
        with tm.assert_produces_warning(warning, match=msg, check_stacklevel=False):
            to_timedelta(val)

    def test_to_timedelta_via_apply(self):
        # GH 5458
        expected = Series([np.timedelta64(1, "s")])
        result = Series(["00:00:01"]).apply(to_timedelta)
        tm.assert_series_equal(result, expected)

        result = Series([to_timedelta("00:00:01")])
        tm.assert_series_equal(result, expected)

    def test_to_timedelta_inference_without_warning(self):
        # GH#41731 inference produces a warning in the Series constructor,
        #  but _not_ in to_timedelta
        vals = ["00:00:01", pd.NaT]
        with tm.assert_produces_warning(None):
            result = to_timedelta(vals)

        expected = TimedeltaIndex([pd.Timedelta(seconds=1), pd.NaT])
        tm.assert_index_equal(result, expected)

    def test_to_timedelta_on_missing_values(self):
        # GH5438
        timedelta_NaT = np.timedelta64("NaT")

        actual = to_timedelta(Series(["00:00:01", np.nan]))
        expected = Series(
            [np.timedelta64(1000000000, "ns"), timedelta_NaT], dtype="<m8[ns]"
        )
        tm.assert_series_equal(actual, expected)

        with tm.assert_produces_warning(FutureWarning, match="Inferring timedelta64"):
            ser = Series(["00:00:01", pd.NaT])
        assert ser.dtype == "m8[ns]"
        actual = to_timedelta(ser)
        tm.assert_series_equal(actual, expected)

        actual = to_timedelta(np.nan)
        assert actual.value == timedelta_NaT.astype("int64")

        actual = to_timedelta(pd.NaT)
        assert actual.value == timedelta_NaT.astype("int64")

    def test_to_timedelta_float(self):
        # https://github.com/pandas-dev/pandas/issues/25077
        arr = np.arange(0, 1, 1e-6)[-10:]
        result = to_timedelta(arr, unit="s")
        expected_asi8 = np.arange(999990000, 10 ** 9, 1000, dtype="int64")
        tm.assert_numpy_array_equal(result.asi8, expected_asi8)

    def test_to_timedelta_coerce_strings_unit(self):
        arr = np.array([1, 2, "error"], dtype=object)
        result = to_timedelta(arr, unit="ns", errors="coerce")
        expected = to_timedelta([1, 2, pd.NaT], unit="ns")
        tm.assert_index_equal(result, expected)

    def test_to_timedelta_ignore_strings_unit(self):
        arr = np.array([1, 2, "error"], dtype=object)
        result = to_timedelta(arr, unit="ns", errors="ignore")
        tm.assert_numpy_array_equal(result, arr)

    def test_to_timedelta_nullable_int64_dtype(self):
        # GH 35574
        expected = Series([timedelta(days=1), timedelta(days=2)])
        result = to_timedelta(Series([1, 2], dtype="Int64"), unit="days")

        tm.assert_series_equal(result, expected)

        # IntegerArray Series with nulls
        expected = Series([timedelta(days=1), None])
        result = to_timedelta(Series([1, None], dtype="Int64"), unit="days")

        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        ("input", "expected"),
        [
            ("8:53:08.71800000001", "8:53:08.718"),
            ("8:53:08.718001", "8:53:08.718001"),
            ("8:53:08.7180000001", "8:53:08.7180000001"),
            ("-8:53:08.71800000001", "-8:53:08.718"),
            ("8:53:08.7180000089", "8:53:08.718000008"),
        ],
    )
    @pytest.mark.parametrize("func", [pd.Timedelta, to_timedelta])
    def test_to_timedelta_precision_over_nanos(self, input, expected, func):
        # GH: 36738
        expected = pd.Timedelta(expected)
        result = func(input)
        assert result == expected

    def test_to_timedelta_zerodim(self):
        # ndarray.item() incorrectly returns int for dt64[ns] and td64[ns]
        dt64 = pd.Timestamp.now().to_datetime64()
        arg = np.array(dt64)

        msg = (
            "Value must be Timedelta, string, integer, float, timedelta "
            "or convertible, not datetime64"
        )
        with pytest.raises(ValueError, match=msg):
            to_timedelta(arg)

        arg2 = arg.view("m8[ns]")
        result = to_timedelta(arg2)
        assert isinstance(result, pd.Timedelta)
        assert result.value == dt64.view("i8")
