"""
This file contains a minimal set of tests for compliance with the extension
array interface test suite, and should contain no other tests.
The test suite for the full functionality of the array is located in
`pandas/tests/arrays/`.

The tests in this file are inherited from the BaseExtensionTests, and only
minimal tweaks should be applied to get the tests passing (by overwriting a
parent method).

Additional tests should either be added to one of the BaseExtensionTests
classes (if they are relevant for the extension interface for all dtypes), or
be added to the array-specific tests in `pandas/tests/arrays/`.

"""
import string

import numpy as np
import pytest

import pandas as pd
from pandas.core.arrays.string_ import StringDtype
from pandas.tests.extension import base


def split_array(arr):
    if arr.dtype.storage != "pyarrow":
        pytest.skip("chunked array n/a")

    def _split_array(arr):
        import pyarrow as pa

        arrow_array = arr._data
        split = len(arrow_array) // 2
        arrow_array = pa.chunked_array(
            [*arrow_array[:split].chunks, *arrow_array[split:].chunks]
        )
        assert arrow_array.num_chunks == 2
        return type(arr)(arrow_array)

    return _split_array(arr)


@pytest.fixture(params=[True, False])
def chunked(request):
    return request.param


@pytest.fixture
def dtype(string_storage):
    return StringDtype(storage=string_storage)


@pytest.fixture
def data(dtype, chunked):
    strings = np.random.choice(list(string.ascii_letters), size=100)
    while strings[0] == strings[1]:
        strings = np.random.choice(list(string.ascii_letters), size=100)

    arr = dtype.construct_array_type()._from_sequence(strings)
    return split_array(arr) if chunked else arr


@pytest.fixture
def data_missing(dtype, chunked):
    """Length 2 array with [NA, Valid]"""
    arr = dtype.construct_array_type()._from_sequence([pd.NA, "A"])
    return split_array(arr) if chunked else arr


@pytest.fixture
def data_for_sorting(dtype, chunked):
    arr = dtype.construct_array_type()._from_sequence(["B", "C", "A"])
    return split_array(arr) if chunked else arr


@pytest.fixture
def data_missing_for_sorting(dtype, chunked):
    arr = dtype.construct_array_type()._from_sequence(["B", pd.NA, "A"])
    return split_array(arr) if chunked else arr


@pytest.fixture
def na_value():
    return pd.NA


@pytest.fixture
def data_for_grouping(dtype, chunked):
    arr = dtype.construct_array_type()._from_sequence(
        ["B", "B", pd.NA, pd.NA, "A", "A", "B", "C"]
    )
    return split_array(arr) if chunked else arr


class TestDtype(base.BaseDtypeTests):
    def test_eq_with_str(self, dtype):
        assert dtype == f"string[{dtype.storage}]"
        super().test_eq_with_str(dtype)


class TestInterface(base.BaseInterfaceTests):
    def test_view(self, data, request):
        if data.dtype.storage == "pyarrow":
            mark = pytest.mark.xfail(reason="not implemented")
            request.node.add_marker(mark)
        super().test_view(data)


class TestConstructors(base.BaseConstructorsTests):
    def test_from_dtype(self, data):
        # base test uses string representation of dtype
        pass


class TestReshaping(base.BaseReshapingTests):
    def test_transpose(self, data, request):
        if data.dtype.storage == "pyarrow":
            mark = pytest.mark.xfail(reason="not implemented")
            request.node.add_marker(mark)
        super().test_transpose(data)


class TestGetitem(base.BaseGetitemTests):
    pass


class TestSetitem(base.BaseSetitemTests):
    def test_setitem_preserves_views(self, data, request):
        if data.dtype.storage == "pyarrow":
            mark = pytest.mark.xfail(reason="not implemented")
            request.node.add_marker(mark)
        super().test_setitem_preserves_views(data)


class TestMissing(base.BaseMissingTests):
    pass


class TestNoReduce(base.BaseNoReduceTests):
    @pytest.mark.parametrize("skipna", [True, False])
    def test_reduce_series_numeric(self, data, all_numeric_reductions, skipna):
        op_name = all_numeric_reductions

        if op_name in ["min", "max"]:
            return None

        s = pd.Series(data)
        with pytest.raises(TypeError):
            getattr(s, op_name)(skipna=skipna)


class TestMethods(base.BaseMethodsTests):
    @pytest.mark.skip(reason="returns nullable")
    def test_value_counts(self, all_data, dropna):
        return super().test_value_counts(all_data, dropna)

    @pytest.mark.skip(reason="returns nullable")
    def test_value_counts_with_normalize(self, data):
        pass


class TestCasting(base.BaseCastingTests):
    pass


class TestComparisonOps(base.BaseComparisonOpsTests):
    def _compare_other(self, s, data, op_name, other):
        result = getattr(s, op_name)(other)
        expected = getattr(s.astype(object), op_name)(other).astype("boolean")
        self.assert_series_equal(result, expected)

    def test_compare_scalar(self, data, all_compare_operators):
        op_name = all_compare_operators
        s = pd.Series(data)
        self._compare_other(s, data, op_name, "abc")


class TestParsing(base.BaseParsingTests):
    pass


class TestPrinting(base.BasePrintingTests):
    pass


class TestGroupBy(base.BaseGroupbyTests):
    pass
