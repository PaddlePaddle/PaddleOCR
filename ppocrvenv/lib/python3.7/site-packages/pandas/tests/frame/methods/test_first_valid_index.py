"""
Includes test for last_valid_index.
"""
import numpy as np
import pytest

from pandas import (
    DataFrame,
    Series,
    date_range,
)
import pandas._testing as tm


class TestFirstValidIndex:
    def test_first_valid_index_single_nan(self, frame_or_series):
        # GH#9752 Series/DataFrame should both return None, not raise
        obj = frame_or_series([np.nan])

        assert obj.first_valid_index() is None
        assert obj.iloc[:0].first_valid_index() is None

    @pytest.mark.parametrize(
        "empty", [DataFrame(), Series(dtype=object), Series([], index=[], dtype=object)]
    )
    def test_first_valid_index_empty(self, empty):
        # GH#12800
        assert empty.last_valid_index() is None
        assert empty.first_valid_index() is None

    @pytest.mark.parametrize(
        "data,idx,expected_first,expected_last",
        [
            ({"A": [1, 2, 3]}, [1, 1, 2], 1, 2),
            ({"A": [1, 2, 3]}, [1, 2, 2], 1, 2),
            ({"A": [1, 2, 3, 4]}, ["d", "d", "d", "d"], "d", "d"),
            ({"A": [1, np.nan, 3]}, [1, 1, 2], 1, 2),
            ({"A": [np.nan, np.nan, 3]}, [1, 1, 2], 2, 2),
            ({"A": [1, np.nan, 3]}, [1, 2, 2], 1, 2),
        ],
    )
    def test_first_last_valid_frame(self, data, idx, expected_first, expected_last):
        # GH#21441
        df = DataFrame(data, index=idx)
        assert expected_first == df.first_valid_index()
        assert expected_last == df.last_valid_index()

    @pytest.mark.parametrize("index_func", [tm.makeStringIndex, tm.makeDateIndex])
    def test_first_last_valid(self, index_func):
        N = 30
        index = index_func(N)
        mat = np.random.randn(N)
        mat[:5] = np.nan
        mat[-5:] = np.nan

        frame = DataFrame({"foo": mat}, index=index)
        assert frame.first_valid_index() == frame.index[5]
        assert frame.last_valid_index() == frame.index[-6]

        ser = frame["foo"]
        assert ser.first_valid_index() == frame.index[5]
        assert ser.last_valid_index() == frame.index[-6]

    @pytest.mark.parametrize("index_func", [tm.makeStringIndex, tm.makeDateIndex])
    def test_first_last_valid_all_nan(self, index_func):
        # GH#17400: no valid entries
        index = index_func(30)
        frame = DataFrame(np.nan, columns=["foo"], index=index)

        assert frame.last_valid_index() is None
        assert frame.first_valid_index() is None

        ser = frame["foo"]
        assert ser.first_valid_index() is None
        assert ser.last_valid_index() is None

    @pytest.mark.filterwarnings("ignore:Timestamp.freq is deprecated:FutureWarning")
    def test_first_last_valid_preserves_freq(self):
        # GH#20499: its preserves freq with holes
        index = date_range("20110101", periods=30, freq="B")
        frame = DataFrame(np.nan, columns=["foo"], index=index)

        frame.iloc[1] = 1
        frame.iloc[-2] = 1
        assert frame.first_valid_index() == frame.index[1]
        assert frame.last_valid_index() == frame.index[-2]
        assert frame.first_valid_index().freq == frame.index.freq
        assert frame.last_valid_index().freq == frame.index.freq

        ts = frame["foo"]
        assert ts.first_valid_index() == ts.index[1]
        assert ts.last_valid_index() == ts.index[-2]
        assert ts.first_valid_index().freq == ts.index.freq
        assert ts.last_valid_index().freq == ts.index.freq
