import numpy as np
import pytest

from pandas.core.dtypes.common import is_integer_dtype

import pandas as pd
import pandas._testing as tm

arrays = [pd.array([1, 2, 3, None], dtype=dtype) for dtype in tm.ALL_EA_INT_DTYPES]
arrays += [
    pd.array([0.141, -0.268, 5.895, None], dtype=dtype) for dtype in tm.FLOAT_EA_DTYPES
]


@pytest.fixture(params=arrays, ids=[a.dtype.name for a in arrays])
def data(request):
    return request.param


@pytest.fixture()
def numpy_dtype(data):
    # For integer dtype, the numpy conversion must be done to float
    if is_integer_dtype(data):
        numpy_dtype = float
    else:
        numpy_dtype = data.dtype.type
    return numpy_dtype


def test_round(data, numpy_dtype):
    # No arguments
    result = data.round()
    expected = pd.array(
        np.round(data.to_numpy(dtype=numpy_dtype, na_value=None)), dtype=data.dtype
    )
    tm.assert_extension_array_equal(result, expected)

    # Decimals argument
    result = data.round(decimals=2)
    expected = pd.array(
        np.round(data.to_numpy(dtype=numpy_dtype, na_value=None), decimals=2),
        dtype=data.dtype,
    )
    tm.assert_extension_array_equal(result, expected)
