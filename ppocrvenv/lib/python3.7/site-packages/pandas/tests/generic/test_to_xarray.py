import numpy as np
import pytest

import pandas.util._test_decorators as td

from pandas import (
    Categorical,
    DataFrame,
    MultiIndex,
    Series,
    date_range,
)
import pandas._testing as tm


@td.skip_if_no("xarray")
class TestDataFrameToXArray:
    @pytest.fixture
    def df(self):
        return DataFrame(
            {
                "a": list("abc"),
                "b": list(range(1, 4)),
                "c": np.arange(3, 6).astype("u1"),
                "d": np.arange(4.0, 7.0, dtype="float64"),
                "e": [True, False, True],
                "f": Categorical(list("abc")),
                "g": date_range("20130101", periods=3),
                "h": date_range("20130101", periods=3, tz="US/Eastern"),
            }
        )

    def test_to_xarray_index_types(self, index, df):
        if isinstance(index, MultiIndex):
            pytest.skip("MultiIndex is tested separately")
        if len(index) == 0:
            pytest.skip("Test doesn't make sense for empty index")

        from xarray import Dataset

        df.index = index[:3]
        df.index.name = "foo"
        df.columns.name = "bar"
        result = df.to_xarray()
        assert result.dims["foo"] == 3
        assert len(result.coords) == 1
        assert len(result.data_vars) == 8
        tm.assert_almost_equal(list(result.coords.keys()), ["foo"])
        assert isinstance(result, Dataset)

        # idempotency
        # datetimes w/tz are preserved
        # column names are lost
        expected = df.copy()
        expected["f"] = expected["f"].astype(object)
        expected.columns.name = None
        tm.assert_frame_equal(result.to_dataframe(), expected)

    def test_to_xarray_empty(self, df):
        from xarray import Dataset

        df.index.name = "foo"
        result = df[0:0].to_xarray()
        assert result.dims["foo"] == 0
        assert isinstance(result, Dataset)

    def test_to_xarray_with_multiindex(self, df):
        from xarray import Dataset

        # MultiIndex
        df.index = MultiIndex.from_product([["a"], range(3)], names=["one", "two"])
        result = df.to_xarray()
        assert result.dims["one"] == 1
        assert result.dims["two"] == 3
        assert len(result.coords) == 2
        assert len(result.data_vars) == 8
        tm.assert_almost_equal(list(result.coords.keys()), ["one", "two"])
        assert isinstance(result, Dataset)

        result = result.to_dataframe()
        expected = df.copy()
        expected["f"] = expected["f"].astype(object)
        expected.columns.name = None
        tm.assert_frame_equal(result, expected)


@td.skip_if_no("xarray")
class TestSeriesToXArray:
    def test_to_xarray_index_types(self, index):
        if isinstance(index, MultiIndex):
            pytest.skip("MultiIndex is tested separately")

        from xarray import DataArray

        ser = Series(range(len(index)), index=index, dtype="int64")
        ser.index.name = "foo"
        result = ser.to_xarray()
        repr(result)
        assert len(result) == len(index)
        assert len(result.coords) == 1
        tm.assert_almost_equal(list(result.coords.keys()), ["foo"])
        assert isinstance(result, DataArray)

        # idempotency
        tm.assert_series_equal(result.to_series(), ser)

    def test_to_xarray_empty(self):
        from xarray import DataArray

        ser = Series([], dtype=object)
        ser.index.name = "foo"
        result = ser.to_xarray()
        assert len(result) == 0
        assert len(result.coords) == 1
        tm.assert_almost_equal(list(result.coords.keys()), ["foo"])
        assert isinstance(result, DataArray)

    def test_to_xarray_with_multiindex(self):
        from xarray import DataArray

        mi = MultiIndex.from_product([["a", "b"], range(3)], names=["one", "two"])
        ser = Series(range(6), dtype="int64", index=mi)
        result = ser.to_xarray()
        assert len(result) == 2
        tm.assert_almost_equal(list(result.coords.keys()), ["one", "two"])
        assert isinstance(result, DataArray)
        res = result.to_series()
        tm.assert_series_equal(res, ser)
