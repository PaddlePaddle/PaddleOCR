import numpy as np

import pandas as pd
import pandas._testing as tm


class TestSeriesSubclassing:
    def test_indexing_sliced(self):
        s = tm.SubclassedSeries([1, 2, 3, 4], index=list("abcd"))
        res = s.loc[["a", "b"]]
        exp = tm.SubclassedSeries([1, 2], index=list("ab"))
        tm.assert_series_equal(res, exp)

        res = s.iloc[[2, 3]]
        exp = tm.SubclassedSeries([3, 4], index=list("cd"))
        tm.assert_series_equal(res, exp)

        res = s.loc[["a", "b"]]
        exp = tm.SubclassedSeries([1, 2], index=list("ab"))
        tm.assert_series_equal(res, exp)

    def test_to_frame(self):
        s = tm.SubclassedSeries([1, 2, 3, 4], index=list("abcd"), name="xxx")
        res = s.to_frame()
        exp = tm.SubclassedDataFrame({"xxx": [1, 2, 3, 4]}, index=list("abcd"))
        tm.assert_frame_equal(res, exp)

    def test_subclass_unstack(self):
        # GH 15564
        s = tm.SubclassedSeries([1, 2, 3, 4], index=[list("aabb"), list("xyxy")])

        res = s.unstack()
        exp = tm.SubclassedDataFrame({"x": [1, 3], "y": [2, 4]}, index=["a", "b"])

        tm.assert_frame_equal(res, exp)

    def test_subclass_empty_repr(self):
        with tm.assert_produces_warning(DeprecationWarning):
            sub_series = tm.SubclassedSeries()
        assert "SubclassedSeries" in repr(sub_series)

    def test_asof(self):
        N = 3
        rng = pd.date_range("1/1/1990", periods=N, freq="53s")
        s = tm.SubclassedSeries({"A": [np.nan, np.nan, np.nan]}, index=rng)

        result = s.asof(rng[-2:])
        assert isinstance(result, tm.SubclassedSeries)

    def test_explode(self):
        s = tm.SubclassedSeries([[1, 2, 3], "foo", [], [3, 4]])
        result = s.explode()
        assert isinstance(result, tm.SubclassedSeries)

    def test_equals(self):
        # https://github.com/pandas-dev/pandas/pull/34402
        # allow subclass in both directions
        s1 = pd.Series([1, 2, 3])
        s2 = tm.SubclassedSeries([1, 2, 3])
        assert s1.equals(s2)
        assert s2.equals(s1)
