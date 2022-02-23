import numpy as np
import pytest

import pandas.util._test_decorators as td

import pandas as pd
from pandas.core.internals import ObjectBlock
from pandas.tests.extension.base.base import BaseExtensionTests


class BaseCastingTests(BaseExtensionTests):
    """Casting to and from ExtensionDtypes"""

    def test_astype_object_series(self, all_data):
        ser = pd.Series(all_data, name="A")
        result = ser.astype(object)
        assert result.dtype == np.dtype(object)
        if hasattr(result._mgr, "blocks"):
            assert isinstance(result._mgr.blocks[0], ObjectBlock)
        assert isinstance(result._mgr.array, np.ndarray)
        assert result._mgr.array.dtype == np.dtype(object)

    def test_astype_object_frame(self, all_data):
        df = pd.DataFrame({"A": all_data})

        result = df.astype(object)
        if hasattr(result._mgr, "blocks"):
            blk = result._data.blocks[0]
            assert isinstance(blk, ObjectBlock), type(blk)
        assert isinstance(result._mgr.arrays[0], np.ndarray)
        assert result._mgr.arrays[0].dtype == np.dtype(object)

        # FIXME: these currently fail; dont leave commented-out
        # check that we can compare the dtypes
        # cmp = result.dtypes.equals(df.dtypes)
        # assert not cmp.any()

    def test_tolist(self, data):
        result = pd.Series(data).tolist()
        expected = list(data)
        assert result == expected

    def test_astype_str(self, data):
        result = pd.Series(data[:5]).astype(str)
        expected = pd.Series([str(x) for x in data[:5]], dtype=str)
        self.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "nullable_string_dtype",
        [
            "string[python]",
            pytest.param(
                "string[pyarrow]", marks=td.skip_if_no("pyarrow", min_version="1.0.0")
            ),
        ],
    )
    def test_astype_string(self, data, nullable_string_dtype):
        # GH-33465
        result = pd.Series(data[:5]).astype(nullable_string_dtype)
        expected = pd.Series([str(x) for x in data[:5]], dtype=nullable_string_dtype)
        self.assert_series_equal(result, expected)

    def test_to_numpy(self, data):
        expected = np.asarray(data)

        result = data.to_numpy()
        self.assert_equal(result, expected)

        result = pd.Series(data).to_numpy()
        self.assert_equal(result, expected)

    def test_astype_empty_dataframe(self, dtype):
        # https://github.com/pandas-dev/pandas/issues/33113
        df = pd.DataFrame()
        result = df.astype(dtype)
        self.assert_frame_equal(result, df)

    @pytest.mark.parametrize("copy", [True, False])
    def test_astype_own_type(self, data, copy):
        # ensure that astype returns the original object for equal dtype and copy=False
        # https://github.com/pandas-dev/pandas/issues/28488
        result = data.astype(data.dtype, copy=copy)
        assert (result is data) is (not copy)
        self.assert_extension_array_equal(result, data)
