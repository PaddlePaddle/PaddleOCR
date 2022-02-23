import numpy as np
import pytest

from pandas.core.dtypes.dtypes import PeriodDtype

import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import period_array


@pytest.mark.parametrize("dtype", [int, np.int32, np.int64, "uint32", "uint64"])
def test_astype(dtype):
    # We choose to ignore the sign and size of integers for
    # Period/Datetime/Timedelta astype
    arr = period_array(["2000", "2001", None], freq="D")
    with tm.assert_produces_warning(FutureWarning):
        # astype(int..) deprecated
        result = arr.astype(dtype)

    if np.dtype(dtype).kind == "u":
        expected_dtype = np.dtype("uint64")
    else:
        expected_dtype = np.dtype("int64")

    with tm.assert_produces_warning(FutureWarning):
        # astype(int..) deprecated
        expected = arr.astype(expected_dtype)

    assert result.dtype == expected_dtype
    tm.assert_numpy_array_equal(result, expected)


def test_astype_copies():
    arr = period_array(["2000", "2001", None], freq="D")
    with tm.assert_produces_warning(FutureWarning):
        # astype(int..) deprecated
        result = arr.astype(np.int64, copy=False)

    # Add the `.base`, since we now use `.asi8` which returns a view.
    # We could maybe override it in PeriodArray to return ._data directly.
    assert result.base is arr._data

    with tm.assert_produces_warning(FutureWarning):
        # astype(int..) deprecated
        result = arr.astype(np.int64, copy=True)
    assert result is not arr._data
    tm.assert_numpy_array_equal(result, arr._data.view("i8"))


def test_astype_categorical():
    arr = period_array(["2000", "2001", "2001", None], freq="D")
    result = arr.astype("category")
    categories = pd.PeriodIndex(["2000", "2001"], freq="D")
    expected = pd.Categorical.from_codes([0, 1, 1, -1], categories=categories)
    tm.assert_categorical_equal(result, expected)


def test_astype_period():
    arr = period_array(["2000", "2001", None], freq="D")
    result = arr.astype(PeriodDtype("M"))
    expected = period_array(["2000", "2001", None], freq="M")
    tm.assert_period_array_equal(result, expected)


@pytest.mark.parametrize("other", ["datetime64[ns]", "timedelta64[ns]"])
def test_astype_datetime(other):
    arr = period_array(["2000", "2001", None], freq="D")
    # slice off the [ns] so that the regex matches.
    with pytest.raises(TypeError, match=other[:-4]):
        arr.astype(other)
