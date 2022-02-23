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
import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm
from pandas.core.arrays.boolean import BooleanDtype
from pandas.tests.extension import base


def make_data():
    return [True, False] * 4 + [np.nan] + [True, False] * 44 + [np.nan] + [True, False]


@pytest.fixture
def dtype():
    return BooleanDtype()


@pytest.fixture
def data(dtype):
    return pd.array(make_data(), dtype=dtype)


@pytest.fixture
def data_for_twos(dtype):
    return pd.array(np.ones(100), dtype=dtype)


@pytest.fixture
def data_missing(dtype):
    return pd.array([np.nan, True], dtype=dtype)


@pytest.fixture
def data_for_sorting(dtype):
    return pd.array([True, True, False], dtype=dtype)


@pytest.fixture
def data_missing_for_sorting(dtype):
    return pd.array([True, np.nan, False], dtype=dtype)


@pytest.fixture
def na_cmp():
    # we are pd.NA
    return lambda x, y: x is pd.NA and y is pd.NA


@pytest.fixture
def na_value():
    return pd.NA


@pytest.fixture
def data_for_grouping(dtype):
    b = True
    a = False
    na = np.nan
    return pd.array([b, b, na, na, a, a, b], dtype=dtype)


class TestDtype(base.BaseDtypeTests):
    pass


class TestInterface(base.BaseInterfaceTests):
    pass


class TestConstructors(base.BaseConstructorsTests):
    pass


class TestGetitem(base.BaseGetitemTests):
    pass


class TestSetitem(base.BaseSetitemTests):
    pass


class TestMissing(base.BaseMissingTests):
    pass


class TestArithmeticOps(base.BaseArithmeticOpsTests):
    implements = {"__sub__", "__rsub__"}

    def check_opname(self, s, op_name, other, exc=None):
        # overwriting to indicate ops don't raise an error
        super().check_opname(s, op_name, other, exc=None)

    def _check_op(self, obj, op, other, op_name, exc=NotImplementedError):
        if exc is None:
            if op_name in self.implements:
                msg = r"numpy boolean subtract"
                with pytest.raises(TypeError, match=msg):
                    op(obj, other)
                return

            result = op(obj, other)
            expected = self._combine(obj, other, op)

            if op_name in (
                "__floordiv__",
                "__rfloordiv__",
                "__pow__",
                "__rpow__",
                "__mod__",
                "__rmod__",
            ):
                # combine keeps boolean type
                expected = expected.astype("Int8")
            elif op_name in ("__truediv__", "__rtruediv__"):
                # combine with bools does not generate the correct result
                #  (numpy behaviour for div is to regard the bools as numeric)
                expected = self._combine(obj.astype(float), other, op)
                expected = expected.astype("Float64")
            if op_name == "__rpow__":
                # for rpow, combine does not propagate NaN
                expected[result.isna()] = np.nan
            self.assert_equal(result, expected)
        else:
            with pytest.raises(exc):
                op(obj, other)

    def _check_divmod_op(self, s, op, other, exc=None):
        # override to not raise an error
        super()._check_divmod_op(s, op, other, None)


class TestComparisonOps(base.BaseComparisonOpsTests):
    def check_opname(self, s, op_name, other, exc=None):
        # overwriting to indicate ops don't raise an error
        super().check_opname(s, op_name, other, exc=None)

    def _compare_other(self, s, data, op_name, other):
        self.check_opname(s, op_name, other)

    @pytest.mark.skip(reason="Tested in tests/arrays/test_boolean.py")
    def test_compare_scalar(self, data, all_compare_operators):
        pass

    @pytest.mark.skip(reason="Tested in tests/arrays/test_boolean.py")
    def test_compare_array(self, data, all_compare_operators):
        pass


class TestReshaping(base.BaseReshapingTests):
    pass


class TestMethods(base.BaseMethodsTests):
    @pytest.mark.parametrize("na_sentinel", [-1, -2])
    def test_factorize(self, data_for_grouping, na_sentinel):
        # override because we only have 2 unique values
        labels, uniques = pd.factorize(data_for_grouping, na_sentinel=na_sentinel)
        expected_labels = np.array(
            [0, 0, na_sentinel, na_sentinel, 1, 1, 0], dtype=np.intp
        )
        expected_uniques = data_for_grouping.take([0, 4])

        tm.assert_numpy_array_equal(labels, expected_labels)
        self.assert_extension_array_equal(uniques, expected_uniques)

    def test_combine_le(self, data_repeated):
        # override because expected needs to be boolean instead of bool dtype
        orig_data1, orig_data2 = data_repeated(2)
        s1 = pd.Series(orig_data1)
        s2 = pd.Series(orig_data2)
        result = s1.combine(s2, lambda x1, x2: x1 <= x2)
        expected = pd.Series(
            [a <= b for (a, b) in zip(list(orig_data1), list(orig_data2))],
            dtype="boolean",
        )
        self.assert_series_equal(result, expected)

        val = s1.iloc[0]
        result = s1.combine(val, lambda x1, x2: x1 <= x2)
        expected = pd.Series([a <= val for a in list(orig_data1)], dtype="boolean")
        self.assert_series_equal(result, expected)

    def test_searchsorted(self, data_for_sorting, as_series):
        # override because we only have 2 unique values
        data_for_sorting = pd.array([True, False], dtype="boolean")
        b, a = data_for_sorting
        arr = type(data_for_sorting)._from_sequence([a, b])

        if as_series:
            arr = pd.Series(arr)
        assert arr.searchsorted(a) == 0
        assert arr.searchsorted(a, side="right") == 1

        assert arr.searchsorted(b) == 1
        assert arr.searchsorted(b, side="right") == 2

        result = arr.searchsorted(arr.take([0, 1]))
        expected = np.array([0, 1], dtype=np.intp)

        tm.assert_numpy_array_equal(result, expected)

        # sorter
        sorter = np.array([1, 0])
        assert data_for_sorting.searchsorted(a, sorter=sorter) == 0

    @pytest.mark.skip(reason="uses nullable integer")
    def test_value_counts(self, all_data, dropna):
        return super().test_value_counts(all_data, dropna)

    @pytest.mark.skip(reason="uses nullable integer")
    def test_value_counts_with_normalize(self, data):
        pass

    def test_argmin_argmax(self, data_for_sorting, data_missing_for_sorting):
        # override because there are only 2 unique values

        # data_for_sorting -> [B, C, A] with A < B < C -> here True, True, False
        assert data_for_sorting.argmax() == 0
        assert data_for_sorting.argmin() == 2

        # with repeated values -> first occurrence
        data = data_for_sorting.take([2, 0, 0, 1, 1, 2])
        assert data.argmax() == 1
        assert data.argmin() == 0

        # with missing values
        # data_missing_for_sorting -> [B, NA, A] with A < B and NA missing.
        assert data_missing_for_sorting.argmax() == 0
        assert data_missing_for_sorting.argmin() == 2


class TestCasting(base.BaseCastingTests):
    pass


class TestGroupby(base.BaseGroupbyTests):
    """
    Groupby-specific tests are overridden because boolean only has 2
    unique values, base tests uses 3 groups.
    """

    def test_grouping_grouper(self, data_for_grouping):
        df = pd.DataFrame(
            {"A": ["B", "B", None, None, "A", "A", "B"], "B": data_for_grouping}
        )
        gr1 = df.groupby("A").grouper.groupings[0]
        gr2 = df.groupby("B").grouper.groupings[0]

        tm.assert_numpy_array_equal(gr1.grouping_vector, df.A.values)
        tm.assert_extension_array_equal(gr2.grouping_vector, data_for_grouping)

    @pytest.mark.parametrize("as_index", [True, False])
    def test_groupby_extension_agg(self, as_index, data_for_grouping):
        df = pd.DataFrame({"A": [1, 1, 2, 2, 3, 3, 1], "B": data_for_grouping})
        result = df.groupby("B", as_index=as_index).A.mean()
        _, index = pd.factorize(data_for_grouping, sort=True)

        index = pd.Index(index, name="B")
        expected = pd.Series([3.0, 1.0], index=index, name="A")
        if as_index:
            self.assert_series_equal(result, expected)
        else:
            expected = expected.reset_index()
            self.assert_frame_equal(result, expected)

    def test_groupby_agg_extension(self, data_for_grouping):
        # GH#38980 groupby agg on extension type fails for non-numeric types
        df = pd.DataFrame({"A": [1, 1, 2, 2, 3, 3, 1], "B": data_for_grouping})

        expected = df.iloc[[0, 2, 4]]
        expected = expected.set_index("A")

        result = df.groupby("A").agg({"B": "first"})
        self.assert_frame_equal(result, expected)

        result = df.groupby("A").agg("first")
        self.assert_frame_equal(result, expected)

        result = df.groupby("A").first()
        self.assert_frame_equal(result, expected)

    def test_groupby_extension_no_sort(self, data_for_grouping):
        df = pd.DataFrame({"A": [1, 1, 2, 2, 3, 3, 1], "B": data_for_grouping})
        result = df.groupby("B", sort=False).A.mean()
        _, index = pd.factorize(data_for_grouping, sort=False)

        index = pd.Index(index, name="B")
        expected = pd.Series([1.0, 3.0], index=index, name="A")
        self.assert_series_equal(result, expected)

    def test_groupby_extension_transform(self, data_for_grouping):
        valid = data_for_grouping[~data_for_grouping.isna()]
        df = pd.DataFrame({"A": [1, 1, 3, 3, 1], "B": valid})

        result = df.groupby("B").A.transform(len)
        expected = pd.Series([3, 3, 2, 2, 3], name="A")

        self.assert_series_equal(result, expected)

    def test_groupby_extension_apply(self, data_for_grouping, groupby_apply_op):
        df = pd.DataFrame({"A": [1, 1, 2, 2, 3, 3, 1], "B": data_for_grouping})
        df.groupby("B").apply(groupby_apply_op)
        df.groupby("B").A.apply(groupby_apply_op)
        df.groupby("A").apply(groupby_apply_op)
        df.groupby("A").B.apply(groupby_apply_op)

    def test_groupby_apply_identity(self, data_for_grouping):
        df = pd.DataFrame({"A": [1, 1, 2, 2, 3, 3, 1], "B": data_for_grouping})
        result = df.groupby("A").B.apply(lambda x: x.array)
        expected = pd.Series(
            [
                df.B.iloc[[0, 1, 6]].array,
                df.B.iloc[[2, 3]].array,
                df.B.iloc[[4, 5]].array,
            ],
            index=pd.Index([1, 2, 3], name="A"),
            name="B",
        )
        self.assert_series_equal(result, expected)

    def test_in_numeric_groupby(self, data_for_grouping):
        df = pd.DataFrame(
            {
                "A": [1, 1, 2, 2, 3, 3, 1],
                "B": data_for_grouping,
                "C": [1, 1, 1, 1, 1, 1, 1],
            }
        )
        result = df.groupby("A").sum().columns

        if data_for_grouping.dtype._is_numeric:
            expected = pd.Index(["B", "C"])
        else:
            expected = pd.Index(["C"])

        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("min_count", [0, 10])
    def test_groupby_sum_mincount(self, data_for_grouping, min_count):
        df = pd.DataFrame({"A": [1, 1, 2, 2, 3, 3, 1], "B": data_for_grouping})
        result = df.groupby("A").sum(min_count=min_count)
        if min_count == 0:
            expected = pd.DataFrame(
                {"B": pd.array([3, 0, 0], dtype="Int64")},
                index=pd.Index([1, 2, 3], name="A"),
            )
            tm.assert_frame_equal(result, expected)
        else:
            expected = pd.DataFrame(
                {"B": pd.array([pd.NA] * 3, dtype="Int64")},
                index=pd.Index([1, 2, 3], name="A"),
            )
            tm.assert_frame_equal(result, expected)


class TestNumericReduce(base.BaseNumericReduceTests):
    def check_reduce(self, s, op_name, skipna):
        result = getattr(s, op_name)(skipna=skipna)
        expected = getattr(s.astype("float64"), op_name)(skipna=skipna)
        # override parent function to cast to bool for min/max
        if np.isnan(expected):
            expected = pd.NA
        elif op_name in ("min", "max"):
            expected = bool(expected)
        tm.assert_almost_equal(result, expected)


class TestBooleanReduce(base.BaseBooleanReduceTests):
    pass


class TestPrinting(base.BasePrintingTests):
    pass


class TestUnaryOps(base.BaseUnaryOpsTests):
    pass


class TestParsing(base.BaseParsingTests):
    pass
