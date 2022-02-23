import numpy as np
import pytest

from pandas import (
    Series,
    TimedeltaIndex,
    Timestamp,
    array,
)
import pandas._testing as tm


class TestSearchSorted:
    @pytest.mark.parametrize("klass", [list, np.array, array, Series])
    def test_searchsorted_different_argument_classes(self, klass):
        idx = TimedeltaIndex(["1 day", "2 days", "3 days"])
        result = idx.searchsorted(klass(idx))
        expected = np.arange(len(idx), dtype=result.dtype)
        tm.assert_numpy_array_equal(result, expected)

        result = idx._data.searchsorted(klass(idx))
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize(
        "arg", [[1, 2], ["a", "b"], [Timestamp("2020-01-01", tz="Europe/London")] * 2]
    )
    def test_searchsorted_invalid_argument_dtype(self, arg):
        idx = TimedeltaIndex(["1 day", "2 days", "3 days"])
        msg = "value should be a 'Timedelta', 'NaT', or array of those. Got"
        with pytest.raises(TypeError, match=msg):
            idx.searchsorted(arg)
