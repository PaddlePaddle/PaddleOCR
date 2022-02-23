import re

import numpy as np
import pytest

from pandas.compat import pa_version_under1p0

import pandas as pd
import pandas._testing as tm
from pandas.core.arrays.string_ import (
    StringArray,
    StringDtype,
)
from pandas.core.arrays.string_arrow import ArrowStringArray

skip_if_no_pyarrow = pytest.mark.skipif(
    pa_version_under1p0,
    reason="pyarrow>=1.0.0 is required for PyArrow backed StringArray",
)


@skip_if_no_pyarrow
def test_eq_all_na():
    a = pd.array([pd.NA, pd.NA], dtype=StringDtype("pyarrow"))
    result = a == a
    expected = pd.array([pd.NA, pd.NA], dtype="boolean")
    tm.assert_extension_array_equal(result, expected)


def test_config(string_storage):
    with pd.option_context("string_storage", string_storage):
        assert StringDtype().storage == string_storage
        result = pd.array(["a", "b"])
        assert result.dtype.storage == string_storage

    expected = (
        StringDtype(string_storage).construct_array_type()._from_sequence(["a", "b"])
    )
    tm.assert_equal(result, expected)


def test_config_bad_storage_raises():
    msg = re.escape("Value must be one of python|pyarrow")
    with pytest.raises(ValueError, match=msg):
        pd.options.mode.string_storage = "foo"


@skip_if_no_pyarrow
@pytest.mark.parametrize("chunked", [True, False])
@pytest.mark.parametrize("array", ["numpy", "pyarrow"])
def test_constructor_not_string_type_raises(array, chunked):
    import pyarrow as pa

    array = pa if array == "pyarrow" else np

    arr = array.array([1, 2, 3])
    if chunked:
        if array is np:
            pytest.skip("chunked not applicable to numpy array")
        arr = pa.chunked_array(arr)
    if array is np:
        msg = "Unsupported type '<class 'numpy.ndarray'>' for ArrowStringArray"
    else:
        msg = re.escape(
            "ArrowStringArray requires a PyArrow (chunked) array of string type"
        )
    with pytest.raises(ValueError, match=msg):
        ArrowStringArray(arr)


@skip_if_no_pyarrow
def test_from_sequence_wrong_dtype_raises():
    with pd.option_context("string_storage", "python"):
        ArrowStringArray._from_sequence(["a", None, "c"], dtype="string")

    with pd.option_context("string_storage", "pyarrow"):
        ArrowStringArray._from_sequence(["a", None, "c"], dtype="string")

    with pytest.raises(AssertionError, match=None):
        ArrowStringArray._from_sequence(["a", None, "c"], dtype="string[python]")

    ArrowStringArray._from_sequence(["a", None, "c"], dtype="string[pyarrow]")

    with pytest.raises(AssertionError, match=None):
        with pd.option_context("string_storage", "python"):
            ArrowStringArray._from_sequence(["a", None, "c"], dtype=StringDtype())

    with pd.option_context("string_storage", "pyarrow"):
        ArrowStringArray._from_sequence(["a", None, "c"], dtype=StringDtype())

    with pytest.raises(AssertionError, match=None):
        ArrowStringArray._from_sequence(["a", None, "c"], dtype=StringDtype("python"))

    ArrowStringArray._from_sequence(["a", None, "c"], dtype=StringDtype("pyarrow"))

    with pd.option_context("string_storage", "python"):
        StringArray._from_sequence(["a", None, "c"], dtype="string")

    with pd.option_context("string_storage", "pyarrow"):
        StringArray._from_sequence(["a", None, "c"], dtype="string")

    StringArray._from_sequence(["a", None, "c"], dtype="string[python]")

    with pytest.raises(AssertionError, match=None):
        StringArray._from_sequence(["a", None, "c"], dtype="string[pyarrow]")

    with pd.option_context("string_storage", "python"):
        StringArray._from_sequence(["a", None, "c"], dtype=StringDtype())

    with pytest.raises(AssertionError, match=None):
        with pd.option_context("string_storage", "pyarrow"):
            StringArray._from_sequence(["a", None, "c"], dtype=StringDtype())

    StringArray._from_sequence(["a", None, "c"], dtype=StringDtype("python"))

    with pytest.raises(AssertionError, match=None):
        StringArray._from_sequence(["a", None, "c"], dtype=StringDtype("pyarrow"))


@pytest.mark.skipif(
    not pa_version_under1p0,
    reason="pyarrow is installed",
)
def test_pyarrow_not_installed_raises():
    msg = re.escape("pyarrow>=1.0.0 is required for PyArrow backed StringArray")

    with pytest.raises(ImportError, match=msg):
        StringDtype(storage="pyarrow")

    with pytest.raises(ImportError, match=msg):
        ArrowStringArray([])

    with pytest.raises(ImportError, match=msg):
        ArrowStringArray._from_sequence(["a", None, "b"])
