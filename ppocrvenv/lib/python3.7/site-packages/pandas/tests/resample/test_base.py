from datetime import datetime

import numpy as np
import pytest

from pandas import (
    DataFrame,
    NaT,
    PeriodIndex,
    Series,
)
import pandas._testing as tm
from pandas.core.groupby.groupby import DataError
from pandas.core.groupby.grouper import Grouper
from pandas.core.indexes.datetimes import date_range
from pandas.core.indexes.period import period_range
from pandas.core.indexes.timedeltas import timedelta_range
from pandas.core.resample import _asfreq_compat

# a fixture value can be overridden by the test parameter value. Note that the
# value of the fixture can be overridden this way even if the test doesn't use
# it directly (doesn't mention it in the function prototype).
# see https://docs.pytest.org/en/latest/fixture.html#override-a-fixture-with-direct-test-parametrization  # noqa
# in this module we override the fixture values defined in conftest.py
# tuples of '_index_factory,_series_name,_index_start,_index_end'
DATE_RANGE = (date_range, "dti", datetime(2005, 1, 1), datetime(2005, 1, 10))
PERIOD_RANGE = (period_range, "pi", datetime(2005, 1, 1), datetime(2005, 1, 10))
TIMEDELTA_RANGE = (timedelta_range, "tdi", "1 day", "10 day")

all_ts = pytest.mark.parametrize(
    "_index_factory,_series_name,_index_start,_index_end",
    [DATE_RANGE, PERIOD_RANGE, TIMEDELTA_RANGE],
)


@pytest.fixture
def create_index(_index_factory):
    def _create_index(*args, **kwargs):
        """return the _index_factory created using the args, kwargs"""
        return _index_factory(*args, **kwargs)

    return _create_index


@pytest.mark.parametrize("freq", ["2D", "1H"])
@pytest.mark.parametrize(
    "_index_factory,_series_name,_index_start,_index_end", [DATE_RANGE, TIMEDELTA_RANGE]
)
def test_asfreq(series_and_frame, freq, create_index):
    obj = series_and_frame

    result = obj.resample(freq).asfreq()
    new_index = create_index(obj.index[0], obj.index[-1], freq=freq)
    expected = obj.reindex(new_index)
    tm.assert_almost_equal(result, expected)


@pytest.mark.parametrize(
    "_index_factory,_series_name,_index_start,_index_end", [DATE_RANGE, TIMEDELTA_RANGE]
)
def test_asfreq_fill_value(series, create_index):
    # test for fill value during resampling, issue 3715

    s = series

    result = s.resample("1H").asfreq()
    new_index = create_index(s.index[0], s.index[-1], freq="1H")
    expected = s.reindex(new_index)
    tm.assert_series_equal(result, expected)

    frame = s.to_frame("value")
    frame.iloc[1] = None
    result = frame.resample("1H").asfreq(fill_value=4.0)
    new_index = create_index(frame.index[0], frame.index[-1], freq="1H")
    expected = frame.reindex(new_index, fill_value=4.0)
    tm.assert_frame_equal(result, expected)


@all_ts
def test_resample_interpolate(frame):
    # # 12925
    df = frame
    tm.assert_frame_equal(
        df.resample("1T").asfreq().interpolate(), df.resample("1T").interpolate()
    )


def test_raises_on_non_datetimelike_index():
    # this is a non datetimelike index
    xp = DataFrame()
    msg = (
        "Only valid with DatetimeIndex, TimedeltaIndex or PeriodIndex, "
        "but got an instance of 'Index'"
    )
    with pytest.raises(TypeError, match=msg):
        xp.resample("A").mean()


@all_ts
@pytest.mark.parametrize("freq", ["M", "D", "H"])
def test_resample_empty_series(freq, empty_series_dti, resample_method):
    # GH12771 & GH12868

    if resample_method == "ohlc":
        pytest.skip("need to test for ohlc from GH13083")

    s = empty_series_dti
    result = getattr(s.resample(freq), resample_method)()

    expected = s.copy()
    expected.index = _asfreq_compat(s.index, freq)

    tm.assert_index_equal(result.index, expected.index)
    assert result.index.freq == expected.index.freq
    tm.assert_series_equal(result, expected, check_dtype=False)


@all_ts
@pytest.mark.parametrize("freq", ["M", "D", "H"])
def test_resample_nat_index_series(request, freq, series, resample_method):
    # GH39227

    if freq == "M":
        request.node.add_marker(pytest.mark.xfail(reason="Don't know why this fails"))

    s = series.copy()
    s.index = PeriodIndex([NaT] * len(s), freq=freq)
    result = getattr(s.resample(freq), resample_method)()

    if resample_method == "ohlc":
        expected = DataFrame(
            [], index=s.index[:0].copy(), columns=["open", "high", "low", "close"]
        )
        tm.assert_frame_equal(result, expected, check_dtype=False)
    else:
        expected = s[:0].copy()
        tm.assert_series_equal(result, expected, check_dtype=False)
    tm.assert_index_equal(result.index, expected.index)
    assert result.index.freq == expected.index.freq


@all_ts
@pytest.mark.parametrize("freq", ["M", "D", "H"])
@pytest.mark.parametrize("resample_method", ["count", "size"])
def test_resample_count_empty_series(freq, empty_series_dti, resample_method):
    # GH28427
    result = getattr(empty_series_dti.resample(freq), resample_method)()

    index = _asfreq_compat(empty_series_dti.index, freq)

    expected = Series([], dtype="int64", index=index, name=empty_series_dti.name)

    tm.assert_series_equal(result, expected)


@all_ts
@pytest.mark.parametrize("freq", ["M", "D", "H"])
def test_resample_empty_dataframe(empty_frame_dti, freq, resample_method):
    # GH13212
    df = empty_frame_dti
    # count retains dimensions too
    result = getattr(df.resample(freq), resample_method)()
    if resample_method != "size":
        expected = df.copy()
    else:
        # GH14962
        expected = Series([], dtype=object)

    expected.index = _asfreq_compat(df.index, freq)

    tm.assert_index_equal(result.index, expected.index)
    assert result.index.freq == expected.index.freq
    tm.assert_almost_equal(result, expected, check_dtype=False)

    # test size for GH13212 (currently stays as df)


@all_ts
@pytest.mark.parametrize("freq", ["M", "D", "H"])
def test_resample_count_empty_dataframe(freq, empty_frame_dti):
    # GH28427

    empty_frame_dti["a"] = []

    result = empty_frame_dti.resample(freq).count()

    index = _asfreq_compat(empty_frame_dti.index, freq)

    expected = DataFrame({"a": []}, dtype="int64", index=index)

    tm.assert_frame_equal(result, expected)


@all_ts
@pytest.mark.parametrize("freq", ["M", "D", "H"])
def test_resample_size_empty_dataframe(freq, empty_frame_dti):
    # GH28427

    empty_frame_dti["a"] = []

    result = empty_frame_dti.resample(freq).size()

    index = _asfreq_compat(empty_frame_dti.index, freq)

    expected = Series([], dtype="int64", index=index)

    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("index", tm.all_timeseries_index_generator(0))
@pytest.mark.parametrize("dtype", [float, int, object, "datetime64[ns]"])
def test_resample_empty_dtypes(index, dtype, resample_method):
    # Empty series were sometimes causing a segfault (for the functions
    # with Cython bounds-checking disabled) or an IndexError.  We just run
    # them to ensure they no longer do.  (GH #10228)
    empty_series_dti = Series([], index, dtype)
    try:
        getattr(empty_series_dti.resample("d"), resample_method)()
    except DataError:
        # Ignore these since some combinations are invalid
        # (ex: doing mean with dtype of np.object_)
        pass


@all_ts
def test_apply_to_empty_series(empty_series_dti):
    # GH 14313
    s = empty_series_dti
    for freq in ["M", "D", "H"]:
        result = s.resample(freq).apply(lambda x: 1)
        expected = s.resample(freq).apply(np.sum)

        tm.assert_series_equal(result, expected, check_dtype=False)


@all_ts
def test_resampler_is_iterable(series):
    # GH 15314
    freq = "H"
    tg = Grouper(freq=freq, convention="start")
    grouped = series.groupby(tg)
    resampled = series.resample(freq)
    for (rk, rv), (gk, gv) in zip(resampled, grouped):
        assert rk == gk
        tm.assert_series_equal(rv, gv)


@all_ts
def test_resample_quantile(series):
    # GH 15023
    s = series
    q = 0.75
    freq = "H"
    result = s.resample(freq).quantile(q)
    expected = s.resample(freq).agg(lambda x: x.quantile(q)).rename(s.name)
    tm.assert_series_equal(result, expected)
