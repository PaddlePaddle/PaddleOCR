from datetime import datetime

import numpy as np
import pytest

from pandas import (
    Series,
    Timestamp,
)
import pandas._testing as tm


class TestConvert:
    def test_convert(self):
        # GH#10265
        dt = datetime(2001, 1, 1, 0, 0)
        td = dt - datetime(2000, 1, 1, 0, 0)

        # Test coercion with mixed types
        ser = Series(["a", "3.1415", dt, td])

        results = ser._convert(numeric=True)
        expected = Series([np.nan, 3.1415, np.nan, np.nan])
        tm.assert_series_equal(results, expected)

        # Test standard conversion returns original
        results = ser._convert(datetime=True)
        tm.assert_series_equal(results, ser)
        results = ser._convert(numeric=True)
        expected = Series([np.nan, 3.1415, np.nan, np.nan])
        tm.assert_series_equal(results, expected)
        results = ser._convert(timedelta=True)
        tm.assert_series_equal(results, ser)

        # test pass-through and non-conversion when other types selected
        ser = Series(["1.0", "2.0", "3.0"])
        results = ser._convert(datetime=True, numeric=True, timedelta=True)
        expected = Series([1.0, 2.0, 3.0])
        tm.assert_series_equal(results, expected)
        results = ser._convert(True, False, True)
        tm.assert_series_equal(results, ser)

        ser = Series(
            [datetime(2001, 1, 1, 0, 0), datetime(2001, 1, 1, 0, 0)], dtype="O"
        )
        results = ser._convert(datetime=True, numeric=True, timedelta=True)
        expected = Series([datetime(2001, 1, 1, 0, 0), datetime(2001, 1, 1, 0, 0)])
        tm.assert_series_equal(results, expected)
        results = ser._convert(datetime=False, numeric=True, timedelta=True)
        tm.assert_series_equal(results, ser)

        td = datetime(2001, 1, 1, 0, 0) - datetime(2000, 1, 1, 0, 0)
        ser = Series([td, td], dtype="O")
        results = ser._convert(datetime=True, numeric=True, timedelta=True)
        expected = Series([td, td])
        tm.assert_series_equal(results, expected)
        results = ser._convert(True, True, False)
        tm.assert_series_equal(results, ser)

        ser = Series([1.0, 2, 3], index=["a", "b", "c"])
        result = ser._convert(numeric=True)
        tm.assert_series_equal(result, ser)

        # force numeric conversion
        res = ser.copy().astype("O")
        res["a"] = "1"
        result = res._convert(numeric=True)
        tm.assert_series_equal(result, ser)

        res = ser.copy().astype("O")
        res["a"] = "1."
        result = res._convert(numeric=True)
        tm.assert_series_equal(result, ser)

        res = ser.copy().astype("O")
        res["a"] = "garbled"
        result = res._convert(numeric=True)
        expected = ser.copy()
        expected["a"] = np.nan
        tm.assert_series_equal(result, expected)

        # GH 4119, not converting a mixed type (e.g.floats and object)
        ser = Series([1, "na", 3, 4])
        result = ser._convert(datetime=True, numeric=True)
        expected = Series([1, np.nan, 3, 4])
        tm.assert_series_equal(result, expected)

        ser = Series([1, "", 3, 4])
        result = ser._convert(datetime=True, numeric=True)
        tm.assert_series_equal(result, expected)

        # dates
        ser = Series(
            [
                datetime(2001, 1, 1, 0, 0),
                datetime(2001, 1, 2, 0, 0),
                datetime(2001, 1, 3, 0, 0),
            ]
        )

        result = ser._convert(datetime=True)
        expected = Series(
            [Timestamp("20010101"), Timestamp("20010102"), Timestamp("20010103")],
            dtype="M8[ns]",
        )
        tm.assert_series_equal(result, expected)

        result = ser._convert(datetime=True)
        tm.assert_series_equal(result, expected)

        # preserver if non-object
        ser = Series([1], dtype="float32")
        result = ser._convert(datetime=True)
        tm.assert_series_equal(result, ser)

        # FIXME: dont leave commented-out
        # res = ser.copy()
        # r[0] = np.nan
        # result = res._convert(convert_dates=True,convert_numeric=False)
        # assert result.dtype == 'M8[ns]'

    def test_convert_no_arg_error(self):
        ser = Series(["1.0", "2"])
        msg = r"At least one of datetime, numeric or timedelta must be True\."
        with pytest.raises(ValueError, match=msg):
            ser._convert()

    def test_convert_preserve_bool(self):
        ser = Series([1, True, 3, 5], dtype=object)
        res = ser._convert(datetime=True, numeric=True)
        expected = Series([1, 1, 3, 5], dtype="i8")
        tm.assert_series_equal(res, expected)

    def test_convert_preserve_all_bool(self):
        ser = Series([False, True, False, False], dtype=object)
        res = ser._convert(datetime=True, numeric=True)
        expected = Series([False, True, False, False], dtype=bool)
        tm.assert_series_equal(res, expected)
