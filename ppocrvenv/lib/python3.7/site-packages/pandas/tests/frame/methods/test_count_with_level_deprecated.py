import numpy as np
import pytest

from pandas import (
    DataFrame,
    Index,
    Series,
)
import pandas._testing as tm


class TestDataFrameCount:
    def test_count_multiindex(self, multiindex_dataframe_random_data):
        frame = multiindex_dataframe_random_data

        frame = frame.copy()
        frame.index.names = ["a", "b"]

        with tm.assert_produces_warning(FutureWarning):
            result = frame.count(level="b")
        with tm.assert_produces_warning(FutureWarning):
            expected = frame.count(level=1)
        tm.assert_frame_equal(result, expected, check_names=False)

        with tm.assert_produces_warning(FutureWarning):
            result = frame.count(level="a")
        with tm.assert_produces_warning(FutureWarning):
            expected = frame.count(level=0)
        tm.assert_frame_equal(result, expected, check_names=False)

        msg = "Level x not found"
        with pytest.raises(KeyError, match=msg):
            with tm.assert_produces_warning(FutureWarning):
                frame.count(level="x")

    def test_count_level_corner(self, multiindex_dataframe_random_data):
        frame = multiindex_dataframe_random_data

        ser = frame["A"][:0]
        with tm.assert_produces_warning(FutureWarning):
            result = ser.count(level=0)
        expected = Series(0, index=ser.index.levels[0], name="A")
        tm.assert_series_equal(result, expected)

        df = frame[:0]
        with tm.assert_produces_warning(FutureWarning):
            result = df.count(level=0)
        expected = (
            DataFrame(
                index=ser.index.levels[0].set_names(["first"]), columns=df.columns
            )
            .fillna(0)
            .astype(np.int64)
        )
        tm.assert_frame_equal(result, expected)

    def test_count_index_with_nan(self):
        # https://github.com/pandas-dev/pandas/issues/21824
        df = DataFrame(
            {
                "Person": ["John", "Myla", None, "John", "Myla"],
                "Age": [24.0, 5, 21.0, 33, 26],
                "Single": [False, True, True, True, False],
            }
        )

        # count on row labels
        with tm.assert_produces_warning(FutureWarning):
            res = df.set_index(["Person", "Single"]).count(level="Person")
        expected = DataFrame(
            index=Index(["John", "Myla"], name="Person"),
            columns=Index(["Age"]),
            data=[2, 2],
        )
        tm.assert_frame_equal(res, expected)

        # count on column labels
        with tm.assert_produces_warning(FutureWarning):
            res = df.set_index(["Person", "Single"]).T.count(level="Person", axis=1)
        expected = DataFrame(
            columns=Index(["John", "Myla"], name="Person"),
            index=Index(["Age"]),
            data=[[2, 2]],
        )
        tm.assert_frame_equal(res, expected)

    def test_count_level(
        self,
        multiindex_year_month_day_dataframe_random_data,
        multiindex_dataframe_random_data,
    ):
        ymd = multiindex_year_month_day_dataframe_random_data
        frame = multiindex_dataframe_random_data

        def _check_counts(frame, axis=0):
            index = frame._get_axis(axis)
            for i in range(index.nlevels):
                with tm.assert_produces_warning(FutureWarning):
                    result = frame.count(axis=axis, level=i)
                expected = frame.groupby(axis=axis, level=i).count()
                expected = expected.reindex_like(result).astype("i8")
                tm.assert_frame_equal(result, expected)

        frame.iloc[1, [1, 2]] = np.nan
        frame.iloc[7, [0, 1]] = np.nan
        ymd.iloc[1, [1, 2]] = np.nan
        ymd.iloc[7, [0, 1]] = np.nan

        _check_counts(frame)
        _check_counts(ymd)
        _check_counts(frame.T, axis=1)
        _check_counts(ymd.T, axis=1)

        # can't call with level on regular DataFrame
        df = tm.makeTimeDataFrame()
        with pytest.raises(TypeError, match="hierarchical"):
            with tm.assert_produces_warning(FutureWarning):
                df.count(level=0)

        frame["D"] = "foo"
        with tm.assert_produces_warning(FutureWarning):
            result = frame.count(level=0, numeric_only=True)
        tm.assert_index_equal(result.columns, Index(list("ABC"), name="exp"))
