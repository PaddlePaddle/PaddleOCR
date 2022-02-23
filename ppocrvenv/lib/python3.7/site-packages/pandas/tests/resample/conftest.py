from datetime import datetime

import numpy as np
import pytest

from pandas import (
    DataFrame,
    Series,
)
from pandas.core.indexes.datetimes import date_range
from pandas.core.indexes.period import period_range

# The various methods we support
downsample_methods = [
    "min",
    "max",
    "first",
    "last",
    "sum",
    "mean",
    "sem",
    "median",
    "prod",
    "var",
    "std",
    "ohlc",
    "quantile",
]
upsample_methods = ["count", "size"]
series_methods = ["nunique"]
resample_methods = downsample_methods + upsample_methods + series_methods


@pytest.fixture(params=downsample_methods)
def downsample_method(request):
    """Fixture for parametrization of Grouper downsample methods."""
    return request.param


@pytest.fixture(params=resample_methods)
def resample_method(request):
    """Fixture for parametrization of Grouper resample methods."""
    return request.param


@pytest.fixture
def simple_date_range_series():
    """
    Series with date range index and random data for test purposes.
    """

    def _simple_date_range_series(start, end, freq="D"):
        rng = date_range(start, end, freq=freq)
        return Series(np.random.randn(len(rng)), index=rng)

    return _simple_date_range_series


@pytest.fixture
def simple_period_range_series():
    """
    Series with period range index and random data for test purposes.
    """

    def _simple_period_range_series(start, end, freq="D"):
        rng = period_range(start, end, freq=freq)
        return Series(np.random.randn(len(rng)), index=rng)

    return _simple_period_range_series


@pytest.fixture
def _index_start():
    """Fixture for parametrization of index, series and frame."""
    return datetime(2005, 1, 1)


@pytest.fixture
def _index_end():
    """Fixture for parametrization of index, series and frame."""
    return datetime(2005, 1, 10)


@pytest.fixture
def _index_freq():
    """Fixture for parametrization of index, series and frame."""
    return "D"


@pytest.fixture
def _index_name():
    """Fixture for parametrization of index, series and frame."""
    return None


@pytest.fixture
def index(_index_factory, _index_start, _index_end, _index_freq, _index_name):
    """
    Fixture for parametrization of date_range, period_range and
    timedelta_range indexes
    """
    return _index_factory(_index_start, _index_end, freq=_index_freq, name=_index_name)


@pytest.fixture
def _static_values(index):
    """
    Fixture for parametrization of values used in parametrization of
    Series and DataFrames with date_range, period_range and
    timedelta_range indexes
    """
    return np.arange(len(index))


@pytest.fixture
def _series_name():
    """
    Fixture for parametrization of Series name for Series used with
    date_range, period_range and timedelta_range indexes
    """
    return None


@pytest.fixture
def series(index, _series_name, _static_values):
    """
    Fixture for parametrization of Series with date_range, period_range and
    timedelta_range indexes
    """
    return Series(_static_values, index=index, name=_series_name)


@pytest.fixture
def empty_series_dti(series):
    """
    Fixture for parametrization of empty Series with date_range,
    period_range and timedelta_range indexes
    """
    return series[:0]


@pytest.fixture
def frame(index, _series_name, _static_values):
    """
    Fixture for parametrization of DataFrame with date_range, period_range
    and timedelta_range indexes
    """
    # _series_name is intentionally unused
    return DataFrame({"value": _static_values}, index=index)


@pytest.fixture
def empty_frame_dti(series):
    """
    Fixture for parametrization of empty DataFrame with date_range,
    period_range and timedelta_range indexes
    """
    index = series.index[:0]
    return DataFrame(index=index)


@pytest.fixture(params=[Series, DataFrame])
def series_and_frame(request, series, frame):
    """
    Fixture for parametrization of Series and DataFrame with date_range,
    period_range and timedelta_range indexes
    """
    if request.param == Series:
        return series
    if request.param == DataFrame:
        return frame
