import numpy as np
import pytest

import pandas as pd
from pandas import (
    Categorical,
    MultiIndex,
    Series,
)
import pandas._testing as tm


class TestSeriesCount:
    def test_count_level_series(self):
        index = MultiIndex(
            levels=[["foo", "bar", "baz"], ["one", "two", "three", "four"]],
            codes=[[0, 0, 0, 2, 2], [2, 0, 1, 1, 2]],
        )

        ser = Series(np.random.randn(len(index)), index=index)

        with tm.assert_produces_warning(FutureWarning):
            result = ser.count(level=0)
        expected = ser.groupby(level=0).count()
        tm.assert_series_equal(
            result.astype("f8"), expected.reindex(result.index).fillna(0)
        )

        with tm.assert_produces_warning(FutureWarning):
            result = ser.count(level=1)
        expected = ser.groupby(level=1).count()
        tm.assert_series_equal(
            result.astype("f8"), expected.reindex(result.index).fillna(0)
        )

    def test_count_multiindex(self, series_with_multilevel_index):
        ser = series_with_multilevel_index

        series = ser.copy()
        series.index.names = ["a", "b"]

        with tm.assert_produces_warning(FutureWarning):
            result = series.count(level="b")
        with tm.assert_produces_warning(FutureWarning):
            expect = ser.count(level=1).rename_axis("b")
        tm.assert_series_equal(result, expect)

        with tm.assert_produces_warning(FutureWarning):
            result = series.count(level="a")
        with tm.assert_produces_warning(FutureWarning):
            expect = ser.count(level=0).rename_axis("a")
        tm.assert_series_equal(result, expect)

        msg = "Level x not found"
        with pytest.raises(KeyError, match=msg):
            with tm.assert_produces_warning(FutureWarning):
                series.count("x")

    def test_count_level_without_multiindex(self):
        ser = Series(range(3))

        msg = "Series.count level is only valid with a MultiIndex"
        with pytest.raises(ValueError, match=msg):
            with tm.assert_produces_warning(FutureWarning):
                ser.count(level=1)

    def test_count(self, datetime_series):
        assert datetime_series.count() == len(datetime_series)

        datetime_series[::2] = np.NaN

        assert datetime_series.count() == np.isfinite(datetime_series).sum()

        mi = MultiIndex.from_arrays([list("aabbcc"), [1, 2, 2, np.nan, 1, 2]])
        ts = Series(np.arange(len(mi)), index=mi)

        with tm.assert_produces_warning(FutureWarning):
            left = ts.count(level=1)
        right = Series([2, 3, 1], index=[1, 2, np.nan])
        tm.assert_series_equal(left, right)

        ts.iloc[[0, 3, 5]] = np.nan
        with tm.assert_produces_warning(FutureWarning):
            tm.assert_series_equal(ts.count(level=1), right - 1)

        # GH#29478
        with pd.option_context("use_inf_as_na", True):
            assert Series([pd.Timestamp("1990/1/1")]).count() == 1

    def test_count_categorical(self):

        ser = Series(
            Categorical(
                [np.nan, 1, 2, np.nan], categories=[5, 4, 3, 2, 1], ordered=True
            )
        )
        result = ser.count()
        assert result == 2
