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

from pandas.errors import PerformanceWarning

from pandas.core.dtypes.common import is_object_dtype

import pandas as pd
from pandas import SparseDtype
import pandas._testing as tm
from pandas.arrays import SparseArray
from pandas.tests.extension import base


def make_data(fill_value):
    if np.isnan(fill_value):
        data = np.random.uniform(size=100)
    else:
        data = np.random.randint(1, 100, size=100)
        if data[0] == data[1]:
            data[0] += 1

    data[2::3] = fill_value
    return data


@pytest.fixture
def dtype():
    return SparseDtype()


@pytest.fixture(params=[0, np.nan])
def data(request):
    """Length-100 PeriodArray for semantics test."""
    res = SparseArray(make_data(request.param), fill_value=request.param)
    return res


@pytest.fixture
def data_for_twos(request):
    return SparseArray(np.ones(100) * 2)


@pytest.fixture(params=[0, np.nan])
def data_missing(request):
    """Length 2 array with [NA, Valid]"""
    return SparseArray([np.nan, 1], fill_value=request.param)


@pytest.fixture(params=[0, np.nan])
def data_repeated(request):
    """Return different versions of data for count times"""

    def gen(count):
        for _ in range(count):
            yield SparseArray(make_data(request.param), fill_value=request.param)

    yield gen


@pytest.fixture(params=[0, np.nan])
def data_for_sorting(request):
    return SparseArray([2, 3, 1], fill_value=request.param)


@pytest.fixture(params=[0, np.nan])
def data_missing_for_sorting(request):
    return SparseArray([2, np.nan, 1], fill_value=request.param)


@pytest.fixture
def na_value():
    return np.nan


@pytest.fixture
def na_cmp():
    return lambda left, right: pd.isna(left) and pd.isna(right)


@pytest.fixture(params=[0, np.nan])
def data_for_grouping(request):
    return SparseArray([1, 1, np.nan, np.nan, 2, 2, 1, 3], fill_value=request.param)


class BaseSparseTests:
    def _check_unsupported(self, data):
        if data.dtype == SparseDtype(int, 0):
            pytest.skip("Can't store nan in int array.")

    @pytest.mark.xfail(reason="SparseArray does not support setitem")
    def test_ravel(self, data):
        super().test_ravel(data)


class TestDtype(BaseSparseTests, base.BaseDtypeTests):
    def test_array_type_with_arg(self, data, dtype):
        assert dtype.construct_array_type() is SparseArray


class TestInterface(BaseSparseTests, base.BaseInterfaceTests):
    def test_no_values_attribute(self, data):
        pytest.skip("We have values")

    def test_copy(self, data):
        # __setitem__ does not work, so we only have a smoke-test
        data.copy()

    def test_view(self, data):
        # __setitem__ does not work, so we only have a smoke-test
        data.view()


class TestConstructors(BaseSparseTests, base.BaseConstructorsTests):
    pass


class TestReshaping(BaseSparseTests, base.BaseReshapingTests):
    def test_concat_mixed_dtypes(self, data):
        # https://github.com/pandas-dev/pandas/issues/20762
        # This should be the same, aside from concat([sparse, float])
        df1 = pd.DataFrame({"A": data[:3]})
        df2 = pd.DataFrame({"A": [1, 2, 3]})
        df3 = pd.DataFrame({"A": ["a", "b", "c"]}).astype("category")
        dfs = [df1, df2, df3]

        # dataframes
        result = pd.concat(dfs)
        expected = pd.concat(
            [x.apply(lambda s: np.asarray(s).astype(object)) for x in dfs]
        )
        self.assert_frame_equal(result, expected)

    def test_concat_columns(self, data, na_value):
        self._check_unsupported(data)
        super().test_concat_columns(data, na_value)

    def test_concat_extension_arrays_copy_false(self, data, na_value):
        self._check_unsupported(data)
        super().test_concat_extension_arrays_copy_false(data, na_value)

    def test_align(self, data, na_value):
        self._check_unsupported(data)
        super().test_align(data, na_value)

    def test_align_frame(self, data, na_value):
        self._check_unsupported(data)
        super().test_align_frame(data, na_value)

    def test_align_series_frame(self, data, na_value):
        self._check_unsupported(data)
        super().test_align_series_frame(data, na_value)

    def test_merge(self, data, na_value):
        self._check_unsupported(data)
        super().test_merge(data, na_value)

    @pytest.mark.xfail(reason="SparseArray does not support setitem")
    def test_transpose(self, data):
        super().test_transpose(data)


class TestGetitem(BaseSparseTests, base.BaseGetitemTests):
    def test_get(self, data):
        s = pd.Series(data, index=[2 * i for i in range(len(data))])
        if np.isnan(s.values.fill_value):
            assert np.isnan(s.get(4)) and np.isnan(s.iloc[2])
        else:
            assert s.get(4) == s.iloc[2]
        assert s.get(2) == s.iloc[1]

    def test_reindex(self, data, na_value):
        self._check_unsupported(data)
        super().test_reindex(data, na_value)


# Skipping TestSetitem, since we don't implement it.


class TestMissing(BaseSparseTests, base.BaseMissingTests):
    def test_isna(self, data_missing):
        expected_dtype = SparseDtype(bool, pd.isna(data_missing.dtype.fill_value))
        expected = SparseArray([True, False], dtype=expected_dtype)

        result = pd.isna(data_missing)
        self.assert_equal(result, expected)

        result = pd.Series(data_missing).isna()
        expected = pd.Series(expected)
        self.assert_series_equal(result, expected)

        # GH 21189
        result = pd.Series(data_missing).drop([0, 1]).isna()
        expected = pd.Series([], dtype=expected_dtype)
        self.assert_series_equal(result, expected)

    def test_fillna_limit_pad(self, data_missing):
        with tm.assert_produces_warning(PerformanceWarning):
            super().test_fillna_limit_pad(data_missing)

    def test_fillna_limit_backfill(self, data_missing):
        with tm.assert_produces_warning(PerformanceWarning):
            super().test_fillna_limit_backfill(data_missing)

    def test_fillna_no_op_returns_copy(self, data, request):
        if np.isnan(data.fill_value):
            request.node.add_marker(
                pytest.mark.xfail(reason="returns array with different fill value")
            )
        with tm.assert_produces_warning(PerformanceWarning):
            super().test_fillna_no_op_returns_copy(data)

    def test_fillna_series_method(self, data_missing):
        with tm.assert_produces_warning(PerformanceWarning):
            super().test_fillna_limit_backfill(data_missing)

    @pytest.mark.skip(reason="Unsupported")
    def test_fillna_series(self):
        # this one looks doable.
        pass

    def test_fillna_frame(self, data_missing):
        # Have to override to specify that fill_value will change.
        fill_value = data_missing[1]

        result = pd.DataFrame({"A": data_missing, "B": [1, 2]}).fillna(fill_value)

        if pd.isna(data_missing.fill_value):
            dtype = SparseDtype(data_missing.dtype, fill_value)
        else:
            dtype = data_missing.dtype

        expected = pd.DataFrame(
            {
                "A": data_missing._from_sequence([fill_value, fill_value], dtype=dtype),
                "B": [1, 2],
            }
        )

        self.assert_frame_equal(result, expected)


class TestMethods(BaseSparseTests, base.BaseMethodsTests):
    def test_combine_le(self, data_repeated):
        # We return a Series[SparseArray].__le__ returns a
        # Series[Sparse[bool]]
        # rather than Series[bool]
        orig_data1, orig_data2 = data_repeated(2)
        s1 = pd.Series(orig_data1)
        s2 = pd.Series(orig_data2)
        result = s1.combine(s2, lambda x1, x2: x1 <= x2)
        expected = pd.Series(
            SparseArray(
                [a <= b for (a, b) in zip(list(orig_data1), list(orig_data2))],
                fill_value=False,
            )
        )
        self.assert_series_equal(result, expected)

        val = s1.iloc[0]
        result = s1.combine(val, lambda x1, x2: x1 <= x2)
        expected = pd.Series(
            SparseArray([a <= val for a in list(orig_data1)], fill_value=False)
        )
        self.assert_series_equal(result, expected)

    def test_fillna_copy_frame(self, data_missing):
        arr = data_missing.take([1, 1])
        df = pd.DataFrame({"A": arr}, copy=False)

        filled_val = df.iloc[0, 0]
        result = df.fillna(filled_val)

        if hasattr(df._mgr, "blocks"):
            assert df.values.base is not result.values.base
        assert df.A._values.to_dense() is arr.to_dense()

    def test_fillna_copy_series(self, data_missing):
        arr = data_missing.take([1, 1])
        ser = pd.Series(arr)

        filled_val = ser[0]
        result = ser.fillna(filled_val)

        assert ser._values is not result._values
        assert ser._values.to_dense() is arr.to_dense()

    @pytest.mark.skip(reason="Not Applicable")
    def test_fillna_length_mismatch(self, data_missing):
        pass

    def test_where_series(self, data, na_value):
        assert data[0] != data[1]
        cls = type(data)
        a, b = data[:2]

        ser = pd.Series(cls._from_sequence([a, a, b, b], dtype=data.dtype))

        cond = np.array([True, True, False, False])
        result = ser.where(cond)

        new_dtype = SparseDtype("float", 0.0)
        expected = pd.Series(
            cls._from_sequence([a, a, na_value, na_value], dtype=new_dtype)
        )
        self.assert_series_equal(result, expected)

        other = cls._from_sequence([a, b, a, b], dtype=data.dtype)
        cond = np.array([True, False, True, True])
        result = ser.where(cond, other)
        expected = pd.Series(cls._from_sequence([a, b, b, b], dtype=data.dtype))
        self.assert_series_equal(result, expected)

    def test_combine_first(self, data):
        if data.dtype.subtype == "int":
            # Right now this is upcasted to float, just like combine_first
            # for Series[int]
            pytest.skip("TODO(SparseArray.__setitem__ will preserve dtype.")
        super().test_combine_first(data)

    def test_searchsorted(self, data_for_sorting, as_series):
        with tm.assert_produces_warning(PerformanceWarning):
            super().test_searchsorted(data_for_sorting, as_series)

    def test_shift_0_periods(self, data):
        # GH#33856 shifting with periods=0 should return a copy, not same obj
        result = data.shift(0)

        data._sparse_values[0] = data._sparse_values[1]
        assert result._sparse_values[0] != result._sparse_values[1]

    @pytest.mark.parametrize("method", ["argmax", "argmin"])
    def test_argmin_argmax_all_na(self, method, data, na_value):
        # overriding because Sparse[int64, 0] cannot handle na_value
        self._check_unsupported(data)
        super().test_argmin_argmax_all_na(method, data, na_value)

    @pytest.mark.parametrize("box", [pd.array, pd.Series, pd.DataFrame])
    def test_equals(self, data, na_value, as_series, box):
        self._check_unsupported(data)
        super().test_equals(data, na_value, as_series, box)


class TestCasting(BaseSparseTests, base.BaseCastingTests):
    def test_astype_object_series(self, all_data):
        # Unlike the base class, we do not expect the resulting Block
        #  to be ObjectBlock / resulting array to be np.dtype("object")
        ser = pd.Series(all_data, name="A")
        result = ser.astype(object)
        assert is_object_dtype(result.dtype)
        assert is_object_dtype(result._mgr.array.dtype)

    def test_astype_object_frame(self, all_data):
        # Unlike the base class, we do not expect the resulting Block
        #  to be ObjectBlock / resulting array to be np.dtype("object")
        df = pd.DataFrame({"A": all_data})

        result = df.astype(object)
        assert is_object_dtype(result._mgr.arrays[0].dtype)

        # FIXME: these currently fail; dont leave commented-out
        # check that we can compare the dtypes
        # comp = result.dtypes.equals(df.dtypes)
        # assert not comp.any()

    def test_astype_str(self, data):
        result = pd.Series(data[:5]).astype(str)
        expected_dtype = SparseDtype(str, str(data.fill_value))
        expected = pd.Series([str(x) for x in data[:5]], dtype=expected_dtype)
        self.assert_series_equal(result, expected)

    @pytest.mark.xfail(raises=TypeError, reason="no sparse StringDtype")
    def test_astype_string(self, data):
        super().test_astype_string(data)


class TestArithmeticOps(BaseSparseTests, base.BaseArithmeticOpsTests):
    series_scalar_exc = None
    frame_scalar_exc = None
    divmod_exc = None
    series_array_exc = None

    def _skip_if_different_combine(self, data):
        if data.fill_value == 0:
            # arith ops call on dtype.fill_value so that the sparsity
            # is maintained. Combine can't be called on a dtype in
            # general, so we can't make the expected. This is tested elsewhere
            raise pytest.skip("Incorrected expected from Series.combine")

    def test_arith_series_with_scalar(self, data, all_arithmetic_operators):
        self._skip_if_different_combine(data)
        super().test_arith_series_with_scalar(data, all_arithmetic_operators)

    def test_arith_series_with_array(self, data, all_arithmetic_operators):
        self._skip_if_different_combine(data)
        super().test_arith_series_with_array(data, all_arithmetic_operators)

    def test_arith_frame_with_scalar(self, data, all_arithmetic_operators, request):
        if data.dtype.fill_value != 0:
            pass
        elif all_arithmetic_operators.strip("_") not in [
            "mul",
            "rmul",
            "floordiv",
            "rfloordiv",
            "pow",
            "mod",
            "rmod",
        ]:
            mark = pytest.mark.xfail(reason="result dtype.fill_value mismatch")
            request.node.add_marker(mark)
        super().test_arith_frame_with_scalar(data, all_arithmetic_operators)


class TestComparisonOps(BaseSparseTests, base.BaseComparisonOpsTests):
    def _compare_other(self, s, data, op_name, other):
        op = self.get_op_from_name(op_name)

        # array
        result = pd.Series(op(data, other))
        # hard to test the fill value, since we don't know what expected
        # is in general.
        # Rely on tests in `tests/sparse` to validate that.
        assert isinstance(result.dtype, SparseDtype)
        assert result.dtype.subtype == np.dtype("bool")

        with np.errstate(all="ignore"):
            expected = pd.Series(
                SparseArray(
                    op(np.asarray(data), np.asarray(other)),
                    fill_value=result.values.fill_value,
                )
            )

        tm.assert_series_equal(result, expected)

        # series
        s = pd.Series(data)
        result = op(s, other)
        tm.assert_series_equal(result, expected)


class TestPrinting(BaseSparseTests, base.BasePrintingTests):
    @pytest.mark.xfail(reason="Different repr")
    def test_array_repr(self, data, size):
        super().test_array_repr(data, size)


class TestParsing(BaseSparseTests, base.BaseParsingTests):
    @pytest.mark.parametrize("engine", ["c", "python"])
    def test_EA_types(self, engine, data):
        expected_msg = r".*must implement _from_sequence_of_strings.*"
        with pytest.raises(NotImplementedError, match=expected_msg):
            super().test_EA_types(engine, data)
