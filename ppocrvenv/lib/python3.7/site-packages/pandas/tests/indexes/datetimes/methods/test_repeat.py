import numpy as np
import pytest

from pandas import (
    DatetimeIndex,
    Timestamp,
    date_range,
)
import pandas._testing as tm


class TestRepeat:
    def test_repeat_range(self, tz_naive_fixture):
        tz = tz_naive_fixture
        rng = date_range("1/1/2000", "1/1/2001")

        result = rng.repeat(5)
        assert result.freq is None
        assert len(result) == 5 * len(rng)

        index = date_range("2001-01-01", periods=2, freq="D", tz=tz)
        exp = DatetimeIndex(
            ["2001-01-01", "2001-01-01", "2001-01-02", "2001-01-02"], tz=tz
        )
        for res in [index.repeat(2), np.repeat(index, 2)]:
            tm.assert_index_equal(res, exp)
            assert res.freq is None

        index = date_range("2001-01-01", periods=2, freq="2D", tz=tz)
        exp = DatetimeIndex(
            ["2001-01-01", "2001-01-01", "2001-01-03", "2001-01-03"], tz=tz
        )
        for res in [index.repeat(2), np.repeat(index, 2)]:
            tm.assert_index_equal(res, exp)
            assert res.freq is None

        index = DatetimeIndex(["2001-01-01", "NaT", "2003-01-01"], tz=tz)
        exp = DatetimeIndex(
            [
                "2001-01-01",
                "2001-01-01",
                "2001-01-01",
                "NaT",
                "NaT",
                "NaT",
                "2003-01-01",
                "2003-01-01",
                "2003-01-01",
            ],
            tz=tz,
        )
        for res in [index.repeat(3), np.repeat(index, 3)]:
            tm.assert_index_equal(res, exp)
            assert res.freq is None

    def test_repeat(self, tz_naive_fixture):
        tz = tz_naive_fixture
        reps = 2
        msg = "the 'axis' parameter is not supported"

        rng = date_range(start="2016-01-01", periods=2, freq="30Min", tz=tz)

        expected_rng = DatetimeIndex(
            [
                Timestamp("2016-01-01 00:00:00", tz=tz),
                Timestamp("2016-01-01 00:00:00", tz=tz),
                Timestamp("2016-01-01 00:30:00", tz=tz),
                Timestamp("2016-01-01 00:30:00", tz=tz),
            ]
        )

        res = rng.repeat(reps)
        tm.assert_index_equal(res, expected_rng)
        assert res.freq is None

        tm.assert_index_equal(np.repeat(rng, reps), expected_rng)
        with pytest.raises(ValueError, match=msg):
            np.repeat(rng, reps, axis=1)
