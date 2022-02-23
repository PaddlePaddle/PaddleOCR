from datetime import timedelta

import numpy as np
import pytest

import pandas as pd
from pandas import (
    Float64Index,
    Index,
    Int64Index,
    NaT,
    Timedelta,
    TimedeltaIndex,
    timedelta_range,
)
import pandas._testing as tm


class TestTimedeltaIndex:
    def test_astype_object(self):
        idx = timedelta_range(start="1 days", periods=4, freq="D", name="idx")
        expected_list = [
            Timedelta("1 days"),
            Timedelta("2 days"),
            Timedelta("3 days"),
            Timedelta("4 days"),
        ]
        result = idx.astype(object)
        expected = Index(expected_list, dtype=object, name="idx")
        tm.assert_index_equal(result, expected)
        assert idx.tolist() == expected_list

    def test_astype_object_with_nat(self):
        idx = TimedeltaIndex(
            [timedelta(days=1), timedelta(days=2), NaT, timedelta(days=4)], name="idx"
        )
        expected_list = [
            Timedelta("1 days"),
            Timedelta("2 days"),
            NaT,
            Timedelta("4 days"),
        ]
        result = idx.astype(object)
        expected = Index(expected_list, dtype=object, name="idx")
        tm.assert_index_equal(result, expected)
        assert idx.tolist() == expected_list

    def test_astype(self):
        # GH 13149, GH 13209
        idx = TimedeltaIndex([1e14, "NaT", NaT, np.NaN], name="idx")

        result = idx.astype(object)
        expected = Index(
            [Timedelta("1 days 03:46:40")] + [NaT] * 3, dtype=object, name="idx"
        )
        tm.assert_index_equal(result, expected)

        with tm.assert_produces_warning(FutureWarning):
            result = idx.astype(int)
        expected = Int64Index(
            [100000000000000] + [-9223372036854775808] * 3, dtype=np.int64, name="idx"
        )
        tm.assert_index_equal(result, expected)

        result = idx.astype(str)
        expected = Index([str(x) for x in idx], name="idx")
        tm.assert_index_equal(result, expected)

        rng = timedelta_range("1 days", periods=10)
        with tm.assert_produces_warning(FutureWarning):
            result = rng.astype("i8")
        tm.assert_index_equal(result, Index(rng.asi8))
        tm.assert_numpy_array_equal(rng.asi8, result.values)

    def test_astype_uint(self):
        arr = timedelta_range("1H", periods=2)
        expected = pd.UInt64Index(
            np.array([3600000000000, 90000000000000], dtype="uint64")
        )
        with tm.assert_produces_warning(FutureWarning):
            tm.assert_index_equal(arr.astype("uint64"), expected)
            tm.assert_index_equal(arr.astype("uint32"), expected)

    def test_astype_timedelta64(self):
        # GH 13149, GH 13209
        idx = TimedeltaIndex([1e14, "NaT", NaT, np.NaN])

        result = idx.astype("timedelta64")
        expected = Float64Index([1e14] + [np.NaN] * 3, dtype="float64")
        tm.assert_index_equal(result, expected)

        result = idx.astype("timedelta64[ns]")
        tm.assert_index_equal(result, idx)
        assert result is not idx

        result = idx.astype("timedelta64[ns]", copy=False)
        tm.assert_index_equal(result, idx)
        assert result is idx

    @pytest.mark.parametrize("dtype", [float, "datetime64", "datetime64[ns]"])
    def test_astype_raises(self, dtype):
        # GH 13149, GH 13209
        idx = TimedeltaIndex([1e14, "NaT", NaT, np.NaN])
        msg = "Cannot cast TimedeltaIndex to dtype"
        with pytest.raises(TypeError, match=msg):
            idx.astype(dtype)

    def test_astype_category(self):
        obj = timedelta_range("1H", periods=2, freq="H")

        result = obj.astype("category")
        expected = pd.CategoricalIndex([Timedelta("1H"), Timedelta("2H")])
        tm.assert_index_equal(result, expected)

        result = obj._data.astype("category")
        expected = expected.values
        tm.assert_categorical_equal(result, expected)

    def test_astype_array_fallback(self):
        obj = timedelta_range("1H", periods=2)
        result = obj.astype(bool)
        expected = Index(np.array([True, True]))
        tm.assert_index_equal(result, expected)

        result = obj._data.astype(bool)
        expected = np.array([True, True])
        tm.assert_numpy_array_equal(result, expected)
