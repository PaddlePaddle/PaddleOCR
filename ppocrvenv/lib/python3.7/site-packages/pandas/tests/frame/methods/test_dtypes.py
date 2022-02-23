from datetime import timedelta

import numpy as np

from pandas.core.dtypes.dtypes import DatetimeTZDtype

import pandas as pd
from pandas import (
    DataFrame,
    Series,
    date_range,
    option_context,
)
import pandas._testing as tm


def _check_cast(df, v):
    """
    Check if all dtypes of df are equal to v
    """
    assert all(s.dtype.name == v for _, s in df.items())


class TestDataFrameDataTypes:
    def test_empty_frame_dtypes(self):
        empty_df = DataFrame()
        tm.assert_series_equal(empty_df.dtypes, Series(dtype=object))

        nocols_df = DataFrame(index=[1, 2, 3])
        tm.assert_series_equal(nocols_df.dtypes, Series(dtype=object))

        norows_df = DataFrame(columns=list("abc"))
        tm.assert_series_equal(norows_df.dtypes, Series(object, index=list("abc")))

        norows_int_df = DataFrame(columns=list("abc")).astype(np.int32)
        tm.assert_series_equal(
            norows_int_df.dtypes, Series(np.dtype("int32"), index=list("abc"))
        )

        df = DataFrame({"a": 1, "b": True, "c": 1.0}, index=[1, 2, 3])
        ex_dtypes = Series({"a": np.int64, "b": np.bool_, "c": np.float64})
        tm.assert_series_equal(df.dtypes, ex_dtypes)

        # same but for empty slice of df
        tm.assert_series_equal(df[:0].dtypes, ex_dtypes)

    def test_datetime_with_tz_dtypes(self):
        tzframe = DataFrame(
            {
                "A": date_range("20130101", periods=3),
                "B": date_range("20130101", periods=3, tz="US/Eastern"),
                "C": date_range("20130101", periods=3, tz="CET"),
            }
        )
        tzframe.iloc[1, 1] = pd.NaT
        tzframe.iloc[1, 2] = pd.NaT
        result = tzframe.dtypes.sort_index()
        expected = Series(
            [
                np.dtype("datetime64[ns]"),
                DatetimeTZDtype("ns", "US/Eastern"),
                DatetimeTZDtype("ns", "CET"),
            ],
            ["A", "B", "C"],
        )

        tm.assert_series_equal(result, expected)

    def test_dtypes_are_correct_after_column_slice(self):
        # GH6525
        df = DataFrame(index=range(5), columns=list("abc"), dtype=np.float_)
        tm.assert_series_equal(
            df.dtypes,
            Series({"a": np.float_, "b": np.float_, "c": np.float_}),
        )
        tm.assert_series_equal(df.iloc[:, 2:].dtypes, Series({"c": np.float_}))
        tm.assert_series_equal(
            df.dtypes,
            Series({"a": np.float_, "b": np.float_, "c": np.float_}),
        )

    def test_dtypes_gh8722(self, float_string_frame):
        float_string_frame["bool"] = float_string_frame["A"] > 0
        result = float_string_frame.dtypes
        expected = Series(
            {k: v.dtype for k, v in float_string_frame.items()}, index=result.index
        )
        tm.assert_series_equal(result, expected)

        # compat, GH 8722
        with option_context("use_inf_as_na", True):
            df = DataFrame([[1]])
            result = df.dtypes
            tm.assert_series_equal(result, Series({0: np.dtype("int64")}))

    def test_dtypes_timedeltas(self):
        df = DataFrame(
            {
                "A": Series(date_range("2012-1-1", periods=3, freq="D")),
                "B": Series([timedelta(days=i) for i in range(3)]),
            }
        )
        result = df.dtypes
        expected = Series(
            [np.dtype("datetime64[ns]"), np.dtype("timedelta64[ns]")], index=list("AB")
        )
        tm.assert_series_equal(result, expected)

        df["C"] = df["A"] + df["B"]
        result = df.dtypes
        expected = Series(
            [
                np.dtype("datetime64[ns]"),
                np.dtype("timedelta64[ns]"),
                np.dtype("datetime64[ns]"),
            ],
            index=list("ABC"),
        )
        tm.assert_series_equal(result, expected)

        # mixed int types
        df["D"] = 1
        result = df.dtypes
        expected = Series(
            [
                np.dtype("datetime64[ns]"),
                np.dtype("timedelta64[ns]"),
                np.dtype("datetime64[ns]"),
                np.dtype("int64"),
            ],
            index=list("ABCD"),
        )
        tm.assert_series_equal(result, expected)
