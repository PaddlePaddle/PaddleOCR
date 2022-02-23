import numpy as np
import pytest

import pandas as pd
from pandas.core.arrays.floating import (
    Float32Dtype,
    Float64Dtype,
)


@pytest.fixture(params=[Float32Dtype, Float64Dtype])
def dtype(request):
    return request.param()


@pytest.fixture
def data(dtype):
    return pd.array(
        list(np.arange(0.1, 0.9, 0.1))
        + [pd.NA]
        + list(np.arange(1, 9.8, 0.1))
        + [pd.NA]
        + [9.9, 10.0],
        dtype=dtype,
    )


@pytest.fixture
def data_missing(dtype):
    return pd.array([np.nan, 0.1], dtype=dtype)


@pytest.fixture(params=["data", "data_missing"])
def all_data(request, data, data_missing):
    """Parametrized fixture giving 'data' and 'data_missing'"""
    if request.param == "data":
        return data
    elif request.param == "data_missing":
        return data_missing
