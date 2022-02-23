import re

import numpy as np
import pytest

from pandas.core.dtypes.common import pandas_dtype

from pandas import (
    Float64Index,
    Index,
    Int64Index,
)
import pandas._testing as tm


class TestAstype:
    def test_astype_float64_to_object(self):
        float_index = Float64Index([0.0, 2.5, 5.0, 7.5, 10.0])
        result = float_index.astype(object)
        assert result.equals(float_index)
        assert float_index.equals(result)
        assert isinstance(result, Index) and not isinstance(result, Float64Index)

    def test_astype_float64_mixed_to_object(self):
        # mixed int-float
        idx = Float64Index([1.5, 2, 3, 4, 5])
        idx.name = "foo"
        result = idx.astype(object)
        assert result.equals(idx)
        assert idx.equals(result)
        assert isinstance(result, Index) and not isinstance(result, Float64Index)

    @pytest.mark.parametrize("dtype", ["int16", "int32", "int64"])
    def test_astype_float64_to_int_dtype(self, dtype):
        # GH#12881
        # a float astype int
        idx = Float64Index([0, 1, 2])
        result = idx.astype(dtype)
        expected = Int64Index([0, 1, 2])
        tm.assert_index_equal(result, expected)

        idx = Float64Index([0, 1.1, 2])
        result = idx.astype(dtype)
        expected = Int64Index([0, 1, 2])
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("dtype", ["float32", "float64"])
    def test_astype_float64_to_float_dtype(self, dtype):
        # GH#12881
        # a float astype int
        idx = Float64Index([0, 1, 2])
        result = idx.astype(dtype)
        expected = idx
        tm.assert_index_equal(result, expected)

        idx = Float64Index([0, 1.1, 2])
        result = idx.astype(dtype)
        expected = Index(idx.values.astype(dtype))
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("dtype", ["M8[ns]", "m8[ns]"])
    def test_cannot_cast_to_datetimelike(self, dtype):
        idx = Float64Index([0, 1.1, 2])

        msg = (
            f"Cannot convert Float64Index to dtype {pandas_dtype(dtype)}; "
            f"integer values are required for conversion"
        )
        with pytest.raises(TypeError, match=re.escape(msg)):
            idx.astype(dtype)

    @pytest.mark.parametrize("dtype", [int, "int16", "int32", "int64"])
    @pytest.mark.parametrize("non_finite", [np.inf, np.nan])
    def test_cannot_cast_inf_to_int(self, non_finite, dtype):
        # GH#13149
        idx = Float64Index([1, 2, non_finite])

        msg = r"Cannot convert non-finite values \(NA or inf\) to integer"
        with pytest.raises(ValueError, match=msg):
            idx.astype(dtype)

    def test_astype_from_object(self):
        index = Index([1.0, np.nan, 0.2], dtype="object")
        result = index.astype(float)
        expected = Float64Index([1.0, np.nan, 0.2])
        assert result.dtype == expected.dtype
        tm.assert_index_equal(result, expected)
