import numpy as np
import pytest

import pandas as pd
from pandas import (
    DataFrame,
    IntervalIndex,
    Series,
)
import pandas._testing as tm


class TestIntervalIndex:
    @pytest.fixture
    def series_with_interval_index(self):
        return Series(np.arange(5), IntervalIndex.from_breaks(np.arange(6)))

    def test_getitem_with_scalar(self, series_with_interval_index, indexer_sl):

        ser = series_with_interval_index.copy()

        expected = ser.iloc[:3]
        tm.assert_series_equal(expected, indexer_sl(ser)[:3])
        tm.assert_series_equal(expected, indexer_sl(ser)[:2.5])
        tm.assert_series_equal(expected, indexer_sl(ser)[0.1:2.5])
        if indexer_sl is tm.loc:
            tm.assert_series_equal(expected, ser.loc[-1:3])

        expected = ser.iloc[1:4]
        tm.assert_series_equal(expected, indexer_sl(ser)[[1.5, 2.5, 3.5]])
        tm.assert_series_equal(expected, indexer_sl(ser)[[2, 3, 4]])
        tm.assert_series_equal(expected, indexer_sl(ser)[[1.5, 3, 4]])

        expected = ser.iloc[2:5]
        tm.assert_series_equal(expected, indexer_sl(ser)[ser >= 2])

    @pytest.mark.parametrize("direction", ["increasing", "decreasing"])
    def test_getitem_nonoverlapping_monotonic(self, direction, closed, indexer_sl):
        tpls = [(0, 1), (2, 3), (4, 5)]
        if direction == "decreasing":
            tpls = tpls[::-1]

        idx = IntervalIndex.from_tuples(tpls, closed=closed)
        ser = Series(list("abc"), idx)

        for key, expected in zip(idx.left, ser):
            if idx.closed_left:
                assert indexer_sl(ser)[key] == expected
            else:
                with pytest.raises(KeyError, match=str(key)):
                    indexer_sl(ser)[key]

        for key, expected in zip(idx.right, ser):
            if idx.closed_right:
                assert indexer_sl(ser)[key] == expected
            else:
                with pytest.raises(KeyError, match=str(key)):
                    indexer_sl(ser)[key]

        for key, expected in zip(idx.mid, ser):
            assert indexer_sl(ser)[key] == expected

    def test_getitem_non_matching(self, series_with_interval_index, indexer_sl):
        ser = series_with_interval_index.copy()

        # this is a departure from our current
        # indexing scheme, but simpler
        with pytest.raises(KeyError, match=r"\[-1\] not in index"):
            indexer_sl(ser)[[-1, 3, 4, 5]]

        with pytest.raises(KeyError, match=r"\[-1\] not in index"):
            indexer_sl(ser)[[-1, 3]]

    @pytest.mark.slow
    def test_loc_getitem_large_series(self):
        ser = Series(
            np.arange(1000000), index=IntervalIndex.from_breaks(np.arange(1000001))
        )

        result1 = ser.loc[:80000]
        result2 = ser.loc[0:80000]
        result3 = ser.loc[0:80000:1]
        tm.assert_series_equal(result1, result2)
        tm.assert_series_equal(result1, result3)

    def test_loc_getitem_frame(self):
        # CategoricalIndex with IntervalIndex categories
        df = DataFrame({"A": range(10)})
        ser = pd.cut(df.A, 5)
        df["B"] = ser
        df = df.set_index("B")

        result = df.loc[4]
        expected = df.iloc[4:6]
        tm.assert_frame_equal(result, expected)

        with pytest.raises(KeyError, match="10"):
            df.loc[10]

        # single list-like
        result = df.loc[[4]]
        expected = df.iloc[4:6]
        tm.assert_frame_equal(result, expected)

        # non-unique
        result = df.loc[[4, 5]]
        expected = df.take([4, 5, 4, 5])
        tm.assert_frame_equal(result, expected)

        with pytest.raises(KeyError, match=r"None of \[\[10\]\] are"):
            df.loc[[10]]

        # partial missing
        with pytest.raises(KeyError, match=r"\[10\] not in index"):
            df.loc[[10, 4]]

    def test_getitem_interval_with_nans(self, frame_or_series, indexer_sl):
        # GH#41831

        index = IntervalIndex([np.nan, np.nan])
        key = index[:-1]

        obj = frame_or_series(range(2), index=index)
        if frame_or_series is DataFrame and indexer_sl is tm.setitem:
            obj = obj.T

        result = indexer_sl(obj)[key]
        expected = obj

        tm.assert_equal(result, expected)


class TestIntervalIndexInsideMultiIndex:
    def test_mi_intervalindex_slicing_with_scalar(self):
        # GH#27456
        idx = pd.MultiIndex.from_arrays(
            [
                pd.Index(["FC", "FC", "FC", "FC", "OWNER", "OWNER", "OWNER", "OWNER"]),
                pd.Index(
                    ["RID1", "RID1", "RID2", "RID2", "RID1", "RID1", "RID2", "RID2"]
                ),
                IntervalIndex.from_arrays(
                    [0, 1, 10, 11, 0, 1, 10, 11], [1, 2, 11, 12, 1, 2, 11, 12]
                ),
            ]
        )

        idx.names = ["Item", "RID", "MP"]
        df = DataFrame({"value": [1, 2, 3, 4, 5, 6, 7, 8]})
        df.index = idx
        query_df = DataFrame(
            {
                "Item": ["FC", "OWNER", "FC", "OWNER", "OWNER"],
                "RID": ["RID1", "RID1", "RID1", "RID2", "RID2"],
                "MP": [0.2, 1.5, 1.6, 11.1, 10.9],
            }
        )

        query_df = query_df.sort_index()

        idx = pd.MultiIndex.from_arrays([query_df.Item, query_df.RID, query_df.MP])
        query_df.index = idx
        result = df.value.loc[query_df.index]
        expected = Series([1, 6, 2, 8, 7], index=idx, name="value")
        tm.assert_series_equal(result, expected)
