import numpy as np
import pytest

from pandas._libs.tslibs import IncompatibleFrequency

from pandas import (
    DataFrame,
    Period,
    Series,
    Timestamp,
    date_range,
    period_range,
    to_datetime,
)
import pandas._testing as tm


@pytest.fixture
def date_range_frame():
    """
    Fixture for DataFrame of ints with date_range index

    Columns are ['A', 'B'].
    """
    N = 50
    rng = date_range("1/1/1990", periods=N, freq="53s")
    return DataFrame({"A": np.arange(N), "B": np.arange(N)}, index=rng)


class TestFrameAsof:
    def test_basic(self, date_range_frame):
        df = date_range_frame
        N = 50
        df.loc[df.index[15:30], "A"] = np.nan
        dates = date_range("1/1/1990", periods=N * 3, freq="25s")

        result = df.asof(dates)
        assert result.notna().all(1).all()
        lb = df.index[14]
        ub = df.index[30]

        dates = list(dates)

        result = df.asof(dates)
        assert result.notna().all(1).all()

        mask = (result.index >= lb) & (result.index < ub)
        rs = result[mask]
        assert (rs == 14).all(1).all()

    def test_subset(self, date_range_frame):
        N = 10
        df = date_range_frame.iloc[:N].copy()
        df.loc[df.index[4:8], "A"] = np.nan
        dates = date_range("1/1/1990", periods=N * 3, freq="25s")

        # with a subset of A should be the same
        result = df.asof(dates, subset="A")
        expected = df.asof(dates)
        tm.assert_frame_equal(result, expected)

        # same with A/B
        result = df.asof(dates, subset=["A", "B"])
        expected = df.asof(dates)
        tm.assert_frame_equal(result, expected)

        # B gives df.asof
        result = df.asof(dates, subset="B")
        expected = df.resample("25s", closed="right").ffill().reindex(dates)
        expected.iloc[20:] = 9

        tm.assert_frame_equal(result, expected)

    def test_missing(self, date_range_frame):
        # GH 15118
        # no match found - `where` value before earliest date in index
        N = 10
        df = date_range_frame.iloc[:N].copy()

        result = df.asof("1989-12-31")

        expected = Series(
            index=["A", "B"], name=Timestamp("1989-12-31"), dtype=np.float64
        )
        tm.assert_series_equal(result, expected)

        result = df.asof(to_datetime(["1989-12-31"]))
        expected = DataFrame(
            index=to_datetime(["1989-12-31"]), columns=["A", "B"], dtype="float64"
        )
        tm.assert_frame_equal(result, expected)

        # Check that we handle PeriodIndex correctly, dont end up with
        #  period.ordinal for series name
        df = df.to_period("D")
        result = df.asof("1989-12-31")
        assert isinstance(result.name, Period)

    def test_asof_all_nans(self, frame_or_series):
        # GH 15713
        # DataFrame/Series is all nans
        result = frame_or_series([np.nan]).asof([0])
        expected = frame_or_series([np.nan])
        tm.assert_equal(result, expected)

    def test_all_nans(self, date_range_frame):
        # GH 15713
        # DataFrame is all nans

        # testing non-default indexes, multiple inputs
        N = 150
        rng = date_range_frame.index
        dates = date_range("1/1/1990", periods=N, freq="25s")
        result = DataFrame(np.nan, index=rng, columns=["A"]).asof(dates)
        expected = DataFrame(np.nan, index=dates, columns=["A"])
        tm.assert_frame_equal(result, expected)

        # testing multiple columns
        dates = date_range("1/1/1990", periods=N, freq="25s")
        result = DataFrame(np.nan, index=rng, columns=["A", "B", "C"]).asof(dates)
        expected = DataFrame(np.nan, index=dates, columns=["A", "B", "C"])
        tm.assert_frame_equal(result, expected)

        # testing scalar input
        result = DataFrame(np.nan, index=[1, 2], columns=["A", "B"]).asof([3])
        expected = DataFrame(np.nan, index=[3], columns=["A", "B"])
        tm.assert_frame_equal(result, expected)

        result = DataFrame(np.nan, index=[1, 2], columns=["A", "B"]).asof(3)
        expected = Series(np.nan, index=["A", "B"], name=3)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "stamp,expected",
        [
            (
                Timestamp("2018-01-01 23:22:43.325+00:00"),
                Series(2.0, name=Timestamp("2018-01-01 23:22:43.325+00:00")),
            ),
            (
                Timestamp("2018-01-01 22:33:20.682+01:00"),
                Series(1.0, name=Timestamp("2018-01-01 22:33:20.682+01:00")),
            ),
        ],
    )
    def test_time_zone_aware_index(self, stamp, expected):
        # GH21194
        # Testing awareness of DataFrame index considering different
        # UTC and timezone
        df = DataFrame(
            data=[1, 2],
            index=[
                Timestamp("2018-01-01 21:00:05.001+00:00"),
                Timestamp("2018-01-01 22:35:10.550+00:00"),
            ],
        )

        result = df.asof(stamp)
        tm.assert_series_equal(result, expected)

    def test_is_copy(self, date_range_frame):
        # GH-27357, GH-30784: ensure the result of asof is an actual copy and
        # doesn't track the parent dataframe / doesn't give SettingWithCopy warnings
        df = date_range_frame
        N = 50
        df.loc[df.index[15:30], "A"] = np.nan
        dates = date_range("1/1/1990", periods=N * 3, freq="25s")

        result = df.asof(dates)

        with tm.assert_produces_warning(None):
            result["C"] = 1

    def test_asof_periodindex_mismatched_freq(self):
        N = 50
        rng = period_range("1/1/1990", periods=N, freq="H")
        df = DataFrame(np.random.randn(N), index=rng)

        # Mismatched freq
        msg = "Input has different freq"
        with pytest.raises(IncompatibleFrequency, match=msg):
            df.asof(rng.asfreq("D"))
