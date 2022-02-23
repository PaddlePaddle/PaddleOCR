import datetime

import dateutil
import numpy as np
import pytest

import pandas as pd
from pandas import (
    DataFrame,
    Series,
)
import pandas._testing as tm


class TestDataFrameMissingData:
    def test_dropEmptyRows(self, float_frame):
        N = len(float_frame.index)
        mat = np.random.randn(N)
        mat[:5] = np.nan

        frame = DataFrame({"foo": mat}, index=float_frame.index)
        original = Series(mat, index=float_frame.index, name="foo")
        expected = original.dropna()
        inplace_frame1, inplace_frame2 = frame.copy(), frame.copy()

        smaller_frame = frame.dropna(how="all")
        # check that original was preserved
        tm.assert_series_equal(frame["foo"], original)
        return_value = inplace_frame1.dropna(how="all", inplace=True)
        tm.assert_series_equal(smaller_frame["foo"], expected)
        tm.assert_series_equal(inplace_frame1["foo"], expected)
        assert return_value is None

        smaller_frame = frame.dropna(how="all", subset=["foo"])
        return_value = inplace_frame2.dropna(how="all", subset=["foo"], inplace=True)
        tm.assert_series_equal(smaller_frame["foo"], expected)
        tm.assert_series_equal(inplace_frame2["foo"], expected)
        assert return_value is None

    def test_dropIncompleteRows(self, float_frame):
        N = len(float_frame.index)
        mat = np.random.randn(N)
        mat[:5] = np.nan

        frame = DataFrame({"foo": mat}, index=float_frame.index)
        frame["bar"] = 5
        original = Series(mat, index=float_frame.index, name="foo")
        inp_frame1, inp_frame2 = frame.copy(), frame.copy()

        smaller_frame = frame.dropna()
        tm.assert_series_equal(frame["foo"], original)
        return_value = inp_frame1.dropna(inplace=True)

        exp = Series(mat[5:], index=float_frame.index[5:], name="foo")
        tm.assert_series_equal(smaller_frame["foo"], exp)
        tm.assert_series_equal(inp_frame1["foo"], exp)
        assert return_value is None

        samesize_frame = frame.dropna(subset=["bar"])
        tm.assert_series_equal(frame["foo"], original)
        assert (frame["bar"] == 5).all()
        return_value = inp_frame2.dropna(subset=["bar"], inplace=True)
        tm.assert_index_equal(samesize_frame.index, float_frame.index)
        tm.assert_index_equal(inp_frame2.index, float_frame.index)
        assert return_value is None

    def test_dropna(self):
        df = DataFrame(np.random.randn(6, 4))
        df[2][:2] = np.nan

        dropped = df.dropna(axis=1)
        expected = df.loc[:, [0, 1, 3]]
        inp = df.copy()
        return_value = inp.dropna(axis=1, inplace=True)
        tm.assert_frame_equal(dropped, expected)
        tm.assert_frame_equal(inp, expected)
        assert return_value is None

        dropped = df.dropna(axis=0)
        expected = df.loc[list(range(2, 6))]
        inp = df.copy()
        return_value = inp.dropna(axis=0, inplace=True)
        tm.assert_frame_equal(dropped, expected)
        tm.assert_frame_equal(inp, expected)
        assert return_value is None

        # threshold
        dropped = df.dropna(axis=1, thresh=5)
        expected = df.loc[:, [0, 1, 3]]
        inp = df.copy()
        return_value = inp.dropna(axis=1, thresh=5, inplace=True)
        tm.assert_frame_equal(dropped, expected)
        tm.assert_frame_equal(inp, expected)
        assert return_value is None

        dropped = df.dropna(axis=0, thresh=4)
        expected = df.loc[range(2, 6)]
        inp = df.copy()
        return_value = inp.dropna(axis=0, thresh=4, inplace=True)
        tm.assert_frame_equal(dropped, expected)
        tm.assert_frame_equal(inp, expected)
        assert return_value is None

        dropped = df.dropna(axis=1, thresh=4)
        tm.assert_frame_equal(dropped, df)

        dropped = df.dropna(axis=1, thresh=3)
        tm.assert_frame_equal(dropped, df)

        # subset
        dropped = df.dropna(axis=0, subset=[0, 1, 3])
        inp = df.copy()
        return_value = inp.dropna(axis=0, subset=[0, 1, 3], inplace=True)
        tm.assert_frame_equal(dropped, df)
        tm.assert_frame_equal(inp, df)
        assert return_value is None

        # all
        dropped = df.dropna(axis=1, how="all")
        tm.assert_frame_equal(dropped, df)

        df[2] = np.nan
        dropped = df.dropna(axis=1, how="all")
        expected = df.loc[:, [0, 1, 3]]
        tm.assert_frame_equal(dropped, expected)

        # bad input
        msg = "No axis named 3 for object type DataFrame"
        with pytest.raises(ValueError, match=msg):
            df.dropna(axis=3)

    def test_drop_and_dropna_caching(self):
        # tst that cacher updates
        original = Series([1, 2, np.nan], name="A")
        expected = Series([1, 2], dtype=original.dtype, name="A")
        df = DataFrame({"A": original.values.copy()})
        df2 = df.copy()
        df["A"].dropna()
        tm.assert_series_equal(df["A"], original)

        ser = df["A"]
        return_value = ser.dropna(inplace=True)
        tm.assert_series_equal(ser, expected)
        tm.assert_series_equal(df["A"], original)
        assert return_value is None

        df2["A"].drop([1])
        tm.assert_series_equal(df2["A"], original)

        ser = df2["A"]
        return_value = ser.drop([1], inplace=True)
        tm.assert_series_equal(ser, original.drop([1]))
        tm.assert_series_equal(df2["A"], original)
        assert return_value is None

    def test_dropna_corner(self, float_frame):
        # bad input
        msg = "invalid how option: foo"
        with pytest.raises(ValueError, match=msg):
            float_frame.dropna(how="foo")
        msg = "must specify how or thresh"
        with pytest.raises(TypeError, match=msg):
            float_frame.dropna(how=None)
        # non-existent column - 8303
        with pytest.raises(KeyError, match=r"^\['X'\]$"):
            float_frame.dropna(subset=["A", "X"])

    def test_dropna_multiple_axes(self):
        df = DataFrame(
            [
                [1, np.nan, 2, 3],
                [4, np.nan, 5, 6],
                [np.nan, np.nan, np.nan, np.nan],
                [7, np.nan, 8, 9],
            ]
        )

        # GH20987
        with pytest.raises(TypeError, match="supplying multiple axes"):
            df.dropna(how="all", axis=[0, 1])
        with pytest.raises(TypeError, match="supplying multiple axes"):
            df.dropna(how="all", axis=(0, 1))

        inp = df.copy()
        with pytest.raises(TypeError, match="supplying multiple axes"):
            inp.dropna(how="all", axis=(0, 1), inplace=True)

    def test_dropna_tz_aware_datetime(self):
        # GH13407
        df = DataFrame()
        dt1 = datetime.datetime(2015, 1, 1, tzinfo=dateutil.tz.tzutc())
        dt2 = datetime.datetime(2015, 2, 2, tzinfo=dateutil.tz.tzutc())
        df["Time"] = [dt1]
        result = df.dropna(axis=0)
        expected = DataFrame({"Time": [dt1]})
        tm.assert_frame_equal(result, expected)

        # Ex2
        df = DataFrame({"Time": [dt1, None, np.nan, dt2]})
        result = df.dropna(axis=0)
        expected = DataFrame([dt1, dt2], columns=["Time"], index=[0, 3])
        tm.assert_frame_equal(result, expected)

    def test_dropna_categorical_interval_index(self):
        # GH 25087
        ii = pd.IntervalIndex.from_breaks([0, 2.78, 3.14, 6.28])
        ci = pd.CategoricalIndex(ii)
        df = DataFrame({"A": list("abc")}, index=ci)

        expected = df
        result = df.dropna()
        tm.assert_frame_equal(result, expected)

    def test_dropna_with_duplicate_columns(self):
        df = DataFrame(
            {
                "A": np.random.randn(5),
                "B": np.random.randn(5),
                "C": np.random.randn(5),
                "D": ["a", "b", "c", "d", "e"],
            }
        )
        df.iloc[2, [0, 1, 2]] = np.nan
        df.iloc[0, 0] = np.nan
        df.iloc[1, 1] = np.nan
        df.iloc[:, 3] = np.nan
        expected = df.dropna(subset=["A", "B", "C"], how="all")
        expected.columns = ["A", "A", "B", "C"]

        df.columns = ["A", "A", "B", "C"]

        result = df.dropna(subset=["A", "C"], how="all")
        tm.assert_frame_equal(result, expected)

    def test_dropna_pos_args_deprecation(self):
        # https://github.com/pandas-dev/pandas/issues/41485
        df = DataFrame({"a": [1, 2, 3]})
        msg = (
            r"In a future version of pandas all arguments of DataFrame\.dropna "
            r"will be keyword-only"
        )
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = df.dropna(1)
        expected = DataFrame({"a": [1, 2, 3]})
        tm.assert_frame_equal(result, expected)
