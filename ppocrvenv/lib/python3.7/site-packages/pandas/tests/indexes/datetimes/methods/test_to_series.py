import numpy as np
import pytest

from pandas import (
    DatetimeIndex,
    Series,
)
import pandas._testing as tm


class TestToSeries:
    @pytest.fixture
    def idx_expected(self):
        naive = DatetimeIndex(["2013-1-1 13:00", "2013-1-2 14:00"], name="B")
        idx = naive.tz_localize("US/Pacific")

        expected = Series(np.array(idx.tolist(), dtype="object"), name="B")

        assert expected.dtype == idx.dtype
        return idx, expected

    def test_to_series_keep_tz_deprecated_true(self, idx_expected):
        # convert to series while keeping the timezone
        idx, expected = idx_expected

        msg = "stop passing 'keep_tz'"
        with tm.assert_produces_warning(FutureWarning) as m:
            result = idx.to_series(keep_tz=True, index=[0, 1])
        assert msg in str(m[0].message)

        tm.assert_series_equal(result, expected)

    def test_to_series_keep_tz_deprecated_false(self, idx_expected):
        idx, expected = idx_expected

        with tm.assert_produces_warning(FutureWarning) as m:
            result = idx.to_series(keep_tz=False, index=[0, 1])
        tm.assert_series_equal(result, expected.dt.tz_convert(None))
        msg = "do 'idx.tz_convert(None)' before calling"
        assert msg in str(m[0].message)
