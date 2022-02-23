from datetime import datetime

import numpy as np
import pytest

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    RangeIndex,
    Series,
    date_range,
)
import pandas._testing as tm


class TestResetIndex:
    def test_reset_index_dti_round_trip(self):
        dti = date_range(start="1/1/2001", end="6/1/2001", freq="D")._with_freq(None)
        d1 = DataFrame({"v": np.random.rand(len(dti))}, index=dti)
        d2 = d1.reset_index()
        assert d2.dtypes[0] == np.dtype("M8[ns]")
        d3 = d2.set_index("index")
        tm.assert_frame_equal(d1, d3, check_names=False)

        # GH#2329
        stamp = datetime(2012, 11, 22)
        df = DataFrame([[stamp, 12.1]], columns=["Date", "Value"])
        df = df.set_index("Date")

        assert df.index[0] == stamp
        assert df.reset_index()["Date"][0] == stamp

    def test_reset_index(self):
        df = tm.makeDataFrame()[:5]
        ser = df.stack()
        ser.index.names = ["hash", "category"]

        ser.name = "value"
        df = ser.reset_index()
        assert "value" in df

        df = ser.reset_index(name="value2")
        assert "value2" in df

        # check inplace
        s = ser.reset_index(drop=True)
        s2 = ser
        return_value = s2.reset_index(drop=True, inplace=True)
        assert return_value is None
        tm.assert_series_equal(s, s2)

        # level
        index = MultiIndex(
            levels=[["bar"], ["one", "two", "three"], [0, 1]],
            codes=[[0, 0, 0, 0, 0, 0], [0, 1, 2, 0, 1, 2], [0, 1, 0, 1, 0, 1]],
        )
        s = Series(np.random.randn(6), index=index)
        rs = s.reset_index(level=1)
        assert len(rs.columns) == 2

        rs = s.reset_index(level=[0, 2], drop=True)
        tm.assert_index_equal(rs.index, Index(index.get_level_values(1)))
        assert isinstance(rs, Series)

    def test_reset_index_name(self):
        s = Series([1, 2, 3], index=Index(range(3), name="x"))
        assert s.reset_index().index.name is None
        assert s.reset_index(drop=True).index.name is None

    def test_reset_index_level(self):
        df = DataFrame([[1, 2, 3], [4, 5, 6]], columns=["A", "B", "C"])

        for levels in ["A", "B"], [0, 1]:
            # With MultiIndex
            s = df.set_index(["A", "B"])["C"]

            result = s.reset_index(level=levels[0])
            tm.assert_frame_equal(result, df.set_index("B"))

            result = s.reset_index(level=levels[:1])
            tm.assert_frame_equal(result, df.set_index("B"))

            result = s.reset_index(level=levels)
            tm.assert_frame_equal(result, df)

            result = df.set_index(["A", "B"]).reset_index(level=levels, drop=True)
            tm.assert_frame_equal(result, df[["C"]])

            with pytest.raises(KeyError, match="Level E "):
                s.reset_index(level=["A", "E"])

            # With single-level Index
            s = df.set_index("A")["B"]

            result = s.reset_index(level=levels[0])
            tm.assert_frame_equal(result, df[["A", "B"]])

            result = s.reset_index(level=levels[:1])
            tm.assert_frame_equal(result, df[["A", "B"]])

            result = s.reset_index(level=levels[0], drop=True)
            tm.assert_series_equal(result, df["B"])

            with pytest.raises(IndexError, match="Too many levels"):
                s.reset_index(level=[0, 1, 2])

        # Check that .reset_index([],drop=True) doesn't fail
        result = Series(range(4)).reset_index([], drop=True)
        expected = Series(range(4))
        tm.assert_series_equal(result, expected)

    def test_reset_index_range(self):
        # GH 12071
        s = Series(range(2), name="A", dtype="int64")
        series_result = s.reset_index()
        assert isinstance(series_result.index, RangeIndex)
        series_expected = DataFrame(
            [[0, 0], [1, 1]], columns=["index", "A"], index=RangeIndex(stop=2)
        )
        tm.assert_frame_equal(series_result, series_expected)

    def test_reset_index_drop_errors(self):
        #  GH 20925

        # KeyError raised for series index when passed level name is missing
        s = Series(range(4))
        with pytest.raises(KeyError, match="does not match index name"):
            s.reset_index("wrong", drop=True)
        with pytest.raises(KeyError, match="does not match index name"):
            s.reset_index("wrong")

        # KeyError raised for series when level to be dropped is missing
        s = Series(range(4), index=MultiIndex.from_product([[1, 2]] * 2))
        with pytest.raises(KeyError, match="not found"):
            s.reset_index("wrong", drop=True)

    def test_reset_index_with_drop(self, series_with_multilevel_index):
        ser = series_with_multilevel_index

        deleveled = ser.reset_index()
        assert isinstance(deleveled, DataFrame)
        assert len(deleveled.columns) == len(ser.index.levels) + 1
        assert deleveled.index.name == ser.index.name

        deleveled = ser.reset_index(drop=True)
        assert isinstance(deleveled, Series)
        assert deleveled.index.name == ser.index.name

    def test_drop_pos_args_deprecation(self):
        # https://github.com/pandas-dev/pandas/issues/41485
        ser = Series([1, 2, 3], index=Index([1, 2, 3], name="a"))
        msg = (
            r"In a future version of pandas all arguments of Series\.reset_index "
            r"except for the argument 'level' will be keyword-only"
        )
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = ser.reset_index("a", False)
        expected = DataFrame({"a": [1, 2, 3], 0: [1, 2, 3]})
        tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "array, dtype",
    [
        (["a", "b"], object),
        (
            pd.period_range("12-1-2000", periods=2, freq="Q-DEC"),
            pd.PeriodDtype(freq="Q-DEC"),
        ),
    ],
)
def test_reset_index_dtypes_on_empty_series_with_multiindex(array, dtype):
    # GH 19602 - Preserve dtype on empty Series with MultiIndex
    idx = MultiIndex.from_product([[0, 1], [0.5, 1.0], array])
    result = Series(dtype=object, index=idx)[:0].reset_index().dtypes
    expected = Series(
        {"level_0": np.int64, "level_1": np.float64, "level_2": dtype, 0: object}
    )
    tm.assert_series_equal(result, expected)
