import numpy as np
import pytest

import pandas as pd
from pandas.api.extensions import ExtensionArray
from pandas.core.internals.blocks import (
    DatetimeTZBlock,
    ExtensionBlock,
)
from pandas.tests.extension.base.base import BaseExtensionTests


class BaseConstructorsTests(BaseExtensionTests):
    def test_from_sequence_from_cls(self, data):
        result = type(data)._from_sequence(data, dtype=data.dtype)
        self.assert_extension_array_equal(result, data)

        data = data[:0]
        result = type(data)._from_sequence(data, dtype=data.dtype)
        self.assert_extension_array_equal(result, data)

    def test_array_from_scalars(self, data):
        scalars = [data[0], data[1], data[2]]
        result = data._from_sequence(scalars)
        assert isinstance(result, type(data))

    def test_series_constructor(self, data):
        result = pd.Series(data)
        assert result.dtype == data.dtype
        assert len(result) == len(data)
        if hasattr(result._mgr, "blocks"):
            assert isinstance(result._mgr.blocks[0], (ExtensionBlock, DatetimeTZBlock))
        assert result._mgr.array is data

        # Series[EA] is unboxed / boxed correctly
        result2 = pd.Series(result)
        assert result2.dtype == data.dtype
        if hasattr(result._mgr, "blocks"):
            assert isinstance(result2._mgr.blocks[0], (ExtensionBlock, DatetimeTZBlock))

    def test_series_constructor_no_data_with_index(self, dtype, na_value):
        result = pd.Series(index=[1, 2, 3], dtype=dtype)
        expected = pd.Series([na_value] * 3, index=[1, 2, 3], dtype=dtype)
        self.assert_series_equal(result, expected)

        # GH 33559 - empty index
        result = pd.Series(index=[], dtype=dtype)
        expected = pd.Series([], index=pd.Index([], dtype="object"), dtype=dtype)
        self.assert_series_equal(result, expected)

    def test_series_constructor_scalar_na_with_index(self, dtype, na_value):
        result = pd.Series(na_value, index=[1, 2, 3], dtype=dtype)
        expected = pd.Series([na_value] * 3, index=[1, 2, 3], dtype=dtype)
        self.assert_series_equal(result, expected)

    def test_series_constructor_scalar_with_index(self, data, dtype):
        scalar = data[0]
        result = pd.Series(scalar, index=[1, 2, 3], dtype=dtype)
        expected = pd.Series([scalar] * 3, index=[1, 2, 3], dtype=dtype)
        self.assert_series_equal(result, expected)

        result = pd.Series(scalar, index=["foo"], dtype=dtype)
        expected = pd.Series([scalar], index=["foo"], dtype=dtype)
        self.assert_series_equal(result, expected)

    @pytest.mark.parametrize("from_series", [True, False])
    def test_dataframe_constructor_from_dict(self, data, from_series):
        if from_series:
            data = pd.Series(data)
        result = pd.DataFrame({"A": data})
        assert result.dtypes["A"] == data.dtype
        assert result.shape == (len(data), 1)
        if hasattr(result._mgr, "blocks"):
            assert isinstance(result._mgr.blocks[0], (ExtensionBlock, DatetimeTZBlock))
        assert isinstance(result._mgr.arrays[0], ExtensionArray)

    def test_dataframe_from_series(self, data):
        result = pd.DataFrame(pd.Series(data))
        assert result.dtypes[0] == data.dtype
        assert result.shape == (len(data), 1)
        if hasattr(result._mgr, "blocks"):
            assert isinstance(result._mgr.blocks[0], (ExtensionBlock, DatetimeTZBlock))
        assert isinstance(result._mgr.arrays[0], ExtensionArray)

    def test_series_given_mismatched_index_raises(self, data):
        msg = r"Length of values \(3\) does not match length of index \(5\)"
        with pytest.raises(ValueError, match=msg):
            pd.Series(data[:3], index=[0, 1, 2, 3, 4])

    def test_from_dtype(self, data):
        # construct from our dtype & string dtype
        dtype = data.dtype

        expected = pd.Series(data)
        result = pd.Series(list(data), dtype=dtype)
        self.assert_series_equal(result, expected)

        result = pd.Series(list(data), dtype=str(dtype))
        self.assert_series_equal(result, expected)

        # gh-30280

        expected = pd.DataFrame(data).astype(dtype)
        result = pd.DataFrame(list(data), dtype=dtype)
        self.assert_frame_equal(result, expected)

        result = pd.DataFrame(list(data), dtype=str(dtype))
        self.assert_frame_equal(result, expected)

    def test_pandas_array(self, data):
        # pd.array(extension_array) should be idempotent...
        result = pd.array(data)
        self.assert_extension_array_equal(result, data)

    def test_pandas_array_dtype(self, data):
        # ... but specifying dtype will override idempotency
        result = pd.array(data, dtype=np.dtype(object))
        expected = pd.arrays.PandasArray(np.asarray(data, dtype=object))
        self.assert_equal(result, expected)

    def test_construct_empty_dataframe(self, dtype):
        # GH 33623
        result = pd.DataFrame(columns=["a"], dtype=dtype)
        expected = pd.DataFrame(
            {"a": pd.array([], dtype=dtype)}, index=pd.Index([], dtype="object")
        )
        self.assert_frame_equal(result, expected)

    def test_empty(self, dtype):
        cls = dtype.construct_array_type()
        result = cls._empty((4,), dtype=dtype)

        assert isinstance(result, cls)
        assert result.dtype == dtype
