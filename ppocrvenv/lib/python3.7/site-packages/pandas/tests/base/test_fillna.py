"""
Though Index.fillna and Series.fillna has separate impl,
test here to confirm these works as the same
"""

import numpy as np
import pytest

from pandas._libs import iNaT

from pandas.core.dtypes.common import needs_i8_conversion
from pandas.core.dtypes.generic import ABCMultiIndex

from pandas import Index
import pandas._testing as tm
from pandas.tests.base.common import allow_na_ops


def test_fillna(index_or_series_obj):
    # GH 11343
    obj = index_or_series_obj
    if isinstance(obj, ABCMultiIndex):
        pytest.skip("MultiIndex doesn't support isna")

    # values will not be changed
    fill_value = obj.values[0] if len(obj) > 0 else 0
    result = obj.fillna(fill_value)
    if isinstance(obj, Index):
        tm.assert_index_equal(obj, result)
    else:
        tm.assert_series_equal(obj, result)

    # check shallow_copied
    assert obj is not result


@pytest.mark.parametrize("null_obj", [np.nan, None])
def test_fillna_null(null_obj, index_or_series_obj):
    # GH 11343
    obj = index_or_series_obj
    klass = type(obj)

    if not allow_na_ops(obj):
        pytest.skip(f"{klass} doesn't allow for NA operations")
    elif len(obj) < 1:
        pytest.skip("Test doesn't make sense on empty data")
    elif isinstance(obj, ABCMultiIndex):
        pytest.skip(f"MultiIndex can't hold '{null_obj}'")

    values = obj.values
    fill_value = values[0]
    expected = values.copy()
    if needs_i8_conversion(obj.dtype):
        values[0:2] = iNaT
        expected[0:2] = fill_value
    else:
        values[0:2] = null_obj
        expected[0:2] = fill_value

    expected = klass(expected)
    obj = klass(values)

    result = obj.fillna(fill_value)
    if isinstance(obj, Index):
        tm.assert_index_equal(result, expected)
    else:
        tm.assert_series_equal(result, expected)

    # check shallow_copied
    assert obj is not result
