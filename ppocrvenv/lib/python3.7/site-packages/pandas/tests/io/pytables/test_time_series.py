import datetime

import numpy as np
import pytest

from pandas import (
    DataFrame,
    Series,
    _testing as tm,
)
from pandas.tests.io.pytables.common import ensure_clean_store

pytestmark = pytest.mark.single


def test_store_datetime_fractional_secs(setup_path):

    with ensure_clean_store(setup_path) as store:
        dt = datetime.datetime(2012, 1, 2, 3, 4, 5, 123456)
        series = Series([0], [dt])
        store["a"] = series
        assert store["a"].index[0] == dt


def test_tseries_indices_series(setup_path):

    with ensure_clean_store(setup_path) as store:
        idx = tm.makeDateIndex(10)
        ser = Series(np.random.randn(len(idx)), idx)
        store["a"] = ser
        result = store["a"]

        tm.assert_series_equal(result, ser)
        assert result.index.freq == ser.index.freq
        tm.assert_class_equal(result.index, ser.index, obj="series index")

        idx = tm.makePeriodIndex(10)
        ser = Series(np.random.randn(len(idx)), idx)
        store["a"] = ser
        result = store["a"]

        tm.assert_series_equal(result, ser)
        assert result.index.freq == ser.index.freq
        tm.assert_class_equal(result.index, ser.index, obj="series index")


def test_tseries_indices_frame(setup_path):

    with ensure_clean_store(setup_path) as store:
        idx = tm.makeDateIndex(10)
        df = DataFrame(np.random.randn(len(idx), 3), index=idx)
        store["a"] = df
        result = store["a"]

        tm.assert_frame_equal(result, df)
        assert result.index.freq == df.index.freq
        tm.assert_class_equal(result.index, df.index, obj="dataframe index")

        idx = tm.makePeriodIndex(10)
        df = DataFrame(np.random.randn(len(idx), 3), idx)
        store["a"] = df
        result = store["a"]

        tm.assert_frame_equal(result, df)
        assert result.index.freq == df.index.freq
        tm.assert_class_equal(result.index, df.index, obj="dataframe index")
