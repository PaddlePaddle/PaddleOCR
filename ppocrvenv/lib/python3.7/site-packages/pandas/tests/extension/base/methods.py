import inspect
import operator

import numpy as np
import pytest

from pandas.core.dtypes.common import is_bool_dtype

import pandas as pd
import pandas._testing as tm
from pandas.core.sorting import nargsort
from pandas.tests.extension.base.base import BaseExtensionTests


class BaseMethodsTests(BaseExtensionTests):
    """Various Series and DataFrame methods."""

    def test_value_counts_default_dropna(self, data):
        # make sure we have consistent default dropna kwarg
        if not hasattr(data, "value_counts"):
            pytest.skip("value_counts is not implemented")
        sig = inspect.signature(data.value_counts)
        kwarg = sig.parameters["dropna"]
        assert kwarg.default is True

    @pytest.mark.parametrize("dropna", [True, False])
    def test_value_counts(self, all_data, dropna):
        all_data = all_data[:10]
        if dropna:
            other = np.array(all_data[~all_data.isna()])
        else:
            other = all_data

        result = pd.Series(all_data).value_counts(dropna=dropna).sort_index()
        expected = pd.Series(other).value_counts(dropna=dropna).sort_index()

        self.assert_series_equal(result, expected)

    def test_value_counts_with_normalize(self, data):
        # GH 33172
        data = data[:10].unique()
        values = np.array(data[~data.isna()])
        ser = pd.Series(data, dtype=data.dtype)

        result = ser.value_counts(normalize=True).sort_index()

        if not isinstance(data, pd.Categorical):
            expected = pd.Series([1 / len(values)] * len(values), index=result.index)
        else:
            expected = pd.Series(0.0, index=result.index)
            expected[result > 0] = 1 / len(values)

        self.assert_series_equal(result, expected)

    def test_count(self, data_missing):
        df = pd.DataFrame({"A": data_missing})
        result = df.count(axis="columns")
        expected = pd.Series([0, 1])
        self.assert_series_equal(result, expected)

    def test_series_count(self, data_missing):
        # GH#26835
        ser = pd.Series(data_missing)
        result = ser.count()
        expected = 1
        assert result == expected

    def test_apply_simple_series(self, data):
        result = pd.Series(data).apply(id)
        assert isinstance(result, pd.Series)

    def test_argsort(self, data_for_sorting):
        result = pd.Series(data_for_sorting).argsort()
        expected = pd.Series(np.array([2, 0, 1], dtype=np.int64))
        self.assert_series_equal(result, expected)

    def test_argsort_missing_array(self, data_missing_for_sorting):
        result = data_missing_for_sorting.argsort()
        expected = np.array([2, 0, 1], dtype=np.dtype("int"))
        # we don't care whether it's int32 or int64
        result = result.astype("int64", casting="safe")
        expected = expected.astype("int64", casting="safe")
        tm.assert_numpy_array_equal(result, expected)

    def test_argsort_missing(self, data_missing_for_sorting):
        result = pd.Series(data_missing_for_sorting).argsort()
        expected = pd.Series(np.array([1, -1, 0], dtype=np.int64))
        self.assert_series_equal(result, expected)

    def test_argmin_argmax(self, data_for_sorting, data_missing_for_sorting, na_value):
        # GH 24382

        # data_for_sorting -> [B, C, A] with A < B < C
        assert data_for_sorting.argmax() == 1
        assert data_for_sorting.argmin() == 2

        # with repeated values -> first occurrence
        data = data_for_sorting.take([2, 0, 0, 1, 1, 2])
        assert data.argmax() == 3
        assert data.argmin() == 0

        # with missing values
        # data_missing_for_sorting -> [B, NA, A] with A < B and NA missing.
        assert data_missing_for_sorting.argmax() == 0
        assert data_missing_for_sorting.argmin() == 2

    @pytest.mark.parametrize("method", ["argmax", "argmin"])
    def test_argmin_argmax_empty_array(self, method, data):
        # GH 24382
        err_msg = "attempt to get"
        with pytest.raises(ValueError, match=err_msg):
            getattr(data[:0], method)()

    @pytest.mark.parametrize("method", ["argmax", "argmin"])
    def test_argmin_argmax_all_na(self, method, data, na_value):
        # all missing with skipna=True is the same as empty
        err_msg = "attempt to get"
        data_na = type(data)._from_sequence([na_value, na_value], dtype=data.dtype)
        with pytest.raises(ValueError, match=err_msg):
            getattr(data_na, method)()

    @pytest.mark.parametrize(
        "op_name, skipna, expected",
        [
            ("idxmax", True, 0),
            ("idxmin", True, 2),
            ("argmax", True, 0),
            ("argmin", True, 2),
            ("idxmax", False, np.nan),
            ("idxmin", False, np.nan),
            ("argmax", False, -1),
            ("argmin", False, -1),
        ],
    )
    def test_argreduce_series(
        self, data_missing_for_sorting, op_name, skipna, expected
    ):
        # data_missing_for_sorting -> [B, NA, A] with A < B and NA missing.
        ser = pd.Series(data_missing_for_sorting)
        result = getattr(ser, op_name)(skipna=skipna)
        tm.assert_almost_equal(result, expected)

    def test_argmax_argmin_no_skipna_notimplemented(self, data_missing_for_sorting):
        # GH#38733
        data = data_missing_for_sorting

        with pytest.raises(NotImplementedError, match=""):
            data.argmin(skipna=False)

        with pytest.raises(NotImplementedError, match=""):
            data.argmax(skipna=False)

    @pytest.mark.parametrize(
        "na_position, expected",
        [
            ("last", np.array([2, 0, 1], dtype=np.dtype("intp"))),
            ("first", np.array([1, 2, 0], dtype=np.dtype("intp"))),
        ],
    )
    def test_nargsort(self, data_missing_for_sorting, na_position, expected):
        # GH 25439
        result = nargsort(data_missing_for_sorting, na_position=na_position)
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize("ascending", [True, False])
    def test_sort_values(self, data_for_sorting, ascending, sort_by_key):
        ser = pd.Series(data_for_sorting)
        result = ser.sort_values(ascending=ascending, key=sort_by_key)
        expected = ser.iloc[[2, 0, 1]]
        if not ascending:
            # GH 35922. Expect stable sort
            if ser.nunique() == 2:
                expected = ser.iloc[[0, 1, 2]]
            else:
                expected = ser.iloc[[1, 0, 2]]

        self.assert_series_equal(result, expected)

    @pytest.mark.parametrize("ascending", [True, False])
    def test_sort_values_missing(
        self, data_missing_for_sorting, ascending, sort_by_key
    ):
        ser = pd.Series(data_missing_for_sorting)
        result = ser.sort_values(ascending=ascending, key=sort_by_key)
        if ascending:
            expected = ser.iloc[[2, 0, 1]]
        else:
            expected = ser.iloc[[0, 2, 1]]
        self.assert_series_equal(result, expected)

    @pytest.mark.parametrize("ascending", [True, False])
    def test_sort_values_frame(self, data_for_sorting, ascending):
        df = pd.DataFrame({"A": [1, 2, 1], "B": data_for_sorting})
        result = df.sort_values(["A", "B"])
        expected = pd.DataFrame(
            {"A": [1, 1, 2], "B": data_for_sorting.take([2, 0, 1])}, index=[2, 0, 1]
        )
        self.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("box", [pd.Series, lambda x: x])
    @pytest.mark.parametrize("method", [lambda x: x.unique(), pd.unique])
    def test_unique(self, data, box, method):
        duplicated = box(data._from_sequence([data[0], data[0]]))

        result = method(duplicated)

        assert len(result) == 1
        assert isinstance(result, type(data))
        assert result[0] == duplicated[0]

    @pytest.mark.parametrize("na_sentinel", [-1, -2])
    def test_factorize(self, data_for_grouping, na_sentinel):
        codes, uniques = pd.factorize(data_for_grouping, na_sentinel=na_sentinel)
        expected_codes = np.array(
            [0, 0, na_sentinel, na_sentinel, 1, 1, 0, 2], dtype=np.intp
        )
        expected_uniques = data_for_grouping.take([0, 4, 7])

        tm.assert_numpy_array_equal(codes, expected_codes)
        self.assert_extension_array_equal(uniques, expected_uniques)

    @pytest.mark.parametrize("na_sentinel", [-1, -2])
    def test_factorize_equivalence(self, data_for_grouping, na_sentinel):
        codes_1, uniques_1 = pd.factorize(data_for_grouping, na_sentinel=na_sentinel)
        codes_2, uniques_2 = data_for_grouping.factorize(na_sentinel=na_sentinel)

        tm.assert_numpy_array_equal(codes_1, codes_2)
        self.assert_extension_array_equal(uniques_1, uniques_2)
        assert len(uniques_1) == len(pd.unique(uniques_1))
        assert uniques_1.dtype == data_for_grouping.dtype

    def test_factorize_empty(self, data):
        codes, uniques = pd.factorize(data[:0])
        expected_codes = np.array([], dtype=np.intp)
        expected_uniques = type(data)._from_sequence([], dtype=data[:0].dtype)

        tm.assert_numpy_array_equal(codes, expected_codes)
        self.assert_extension_array_equal(uniques, expected_uniques)

    def test_fillna_copy_frame(self, data_missing):
        arr = data_missing.take([1, 1])
        df = pd.DataFrame({"A": arr})

        filled_val = df.iloc[0, 0]
        result = df.fillna(filled_val)

        assert df.A.values is not result.A.values

    def test_fillna_copy_series(self, data_missing):
        arr = data_missing.take([1, 1])
        ser = pd.Series(arr)

        filled_val = ser[0]
        result = ser.fillna(filled_val)

        assert ser._values is not result._values
        assert ser._values is arr

    def test_fillna_length_mismatch(self, data_missing):
        msg = "Length of 'value' does not match."
        with pytest.raises(ValueError, match=msg):
            data_missing.fillna(data_missing.take([1]))

    def test_combine_le(self, data_repeated):
        # GH 20825
        # Test that combine works when doing a <= (le) comparison
        orig_data1, orig_data2 = data_repeated(2)
        s1 = pd.Series(orig_data1)
        s2 = pd.Series(orig_data2)
        result = s1.combine(s2, lambda x1, x2: x1 <= x2)
        expected = pd.Series(
            [a <= b for (a, b) in zip(list(orig_data1), list(orig_data2))]
        )
        self.assert_series_equal(result, expected)

        val = s1.iloc[0]
        result = s1.combine(val, lambda x1, x2: x1 <= x2)
        expected = pd.Series([a <= val for a in list(orig_data1)])
        self.assert_series_equal(result, expected)

    def test_combine_add(self, data_repeated):
        # GH 20825
        orig_data1, orig_data2 = data_repeated(2)
        s1 = pd.Series(orig_data1)
        s2 = pd.Series(orig_data2)
        result = s1.combine(s2, lambda x1, x2: x1 + x2)
        with np.errstate(over="ignore"):
            expected = pd.Series(
                orig_data1._from_sequence(
                    [a + b for (a, b) in zip(list(orig_data1), list(orig_data2))]
                )
            )
        self.assert_series_equal(result, expected)

        val = s1.iloc[0]
        result = s1.combine(val, lambda x1, x2: x1 + x2)
        expected = pd.Series(
            orig_data1._from_sequence([a + val for a in list(orig_data1)])
        )
        self.assert_series_equal(result, expected)

    def test_combine_first(self, data):
        # https://github.com/pandas-dev/pandas/issues/24147
        a = pd.Series(data[:3])
        b = pd.Series(data[2:5], index=[2, 3, 4])
        result = a.combine_first(b)
        expected = pd.Series(data[:5])
        self.assert_series_equal(result, expected)

    @pytest.mark.parametrize("frame", [True, False])
    @pytest.mark.parametrize(
        "periods, indices",
        [(-2, [2, 3, 4, -1, -1]), (0, [0, 1, 2, 3, 4]), (2, [-1, -1, 0, 1, 2])],
    )
    def test_container_shift(self, data, frame, periods, indices):
        # https://github.com/pandas-dev/pandas/issues/22386
        subset = data[:5]
        data = pd.Series(subset, name="A")
        expected = pd.Series(subset.take(indices, allow_fill=True), name="A")

        if frame:
            result = data.to_frame(name="A").assign(B=1).shift(periods)
            expected = pd.concat(
                [expected, pd.Series([1] * 5, name="B").shift(periods)], axis=1
            )
            compare = self.assert_frame_equal
        else:
            result = data.shift(periods)
            compare = self.assert_series_equal

        compare(result, expected)

    def test_shift_0_periods(self, data):
        # GH#33856 shifting with periods=0 should return a copy, not same obj
        result = data.shift(0)
        assert data[0] != data[1]  # otherwise below is invalid
        data[0] = data[1]
        assert result[0] != result[1]  # i.e. not the same object/view

    @pytest.mark.parametrize("periods", [1, -2])
    def test_diff(self, data, periods):
        data = data[:5]
        if is_bool_dtype(data.dtype):
            op = operator.xor
        else:
            op = operator.sub
        try:
            # does this array implement ops?
            op(data, data)
        except Exception:
            pytest.skip(f"{type(data)} does not support diff")
        s = pd.Series(data)
        result = s.diff(periods)
        expected = pd.Series(op(data, data.shift(periods)))
        self.assert_series_equal(result, expected)

        df = pd.DataFrame({"A": data, "B": [1.0] * 5})
        result = df.diff(periods)
        if periods == 1:
            b = [np.nan, 0, 0, 0, 0]
        else:
            b = [0, 0, 0, np.nan, np.nan]
        expected = pd.DataFrame({"A": expected, "B": b})
        self.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "periods, indices",
        [[-4, [-1, -1]], [-1, [1, -1]], [0, [0, 1]], [1, [-1, 0]], [4, [-1, -1]]],
    )
    def test_shift_non_empty_array(self, data, periods, indices):
        # https://github.com/pandas-dev/pandas/issues/23911
        subset = data[:2]
        result = subset.shift(periods)
        expected = subset.take(indices, allow_fill=True)
        self.assert_extension_array_equal(result, expected)

    @pytest.mark.parametrize("periods", [-4, -1, 0, 1, 4])
    def test_shift_empty_array(self, data, periods):
        # https://github.com/pandas-dev/pandas/issues/23911
        empty = data[:0]
        result = empty.shift(periods)
        expected = empty
        self.assert_extension_array_equal(result, expected)

    def test_shift_zero_copies(self, data):
        result = data.shift(0)
        assert result is not data

        result = data[:0].shift(2)
        assert result is not data

    def test_shift_fill_value(self, data):
        arr = data[:4]
        fill_value = data[0]
        result = arr.shift(1, fill_value=fill_value)
        expected = data.take([0, 0, 1, 2])
        self.assert_extension_array_equal(result, expected)

        result = arr.shift(-2, fill_value=fill_value)
        expected = data.take([2, 3, 0, 0])
        self.assert_extension_array_equal(result, expected)

    def test_not_hashable(self, data):
        # We are in general mutable, so not hashable
        with pytest.raises(TypeError, match="unhashable type"):
            hash(data)

    def test_hash_pandas_object_works(self, data, as_frame):
        # https://github.com/pandas-dev/pandas/issues/23066
        data = pd.Series(data)
        if as_frame:
            data = data.to_frame()
        a = pd.util.hash_pandas_object(data)
        b = pd.util.hash_pandas_object(data)
        self.assert_equal(a, b)

    def test_searchsorted(self, data_for_sorting, as_series):
        b, c, a = data_for_sorting
        arr = data_for_sorting.take([2, 0, 1])  # to get [a, b, c]

        if as_series:
            arr = pd.Series(arr)
        assert arr.searchsorted(a) == 0
        assert arr.searchsorted(a, side="right") == 1

        assert arr.searchsorted(b) == 1
        assert arr.searchsorted(b, side="right") == 2

        assert arr.searchsorted(c) == 2
        assert arr.searchsorted(c, side="right") == 3

        result = arr.searchsorted(arr.take([0, 2]))
        expected = np.array([0, 2], dtype=np.intp)

        tm.assert_numpy_array_equal(result, expected)

        # sorter
        sorter = np.array([1, 2, 0])
        assert data_for_sorting.searchsorted(a, sorter=sorter) == 0

    def test_where_series(self, data, na_value, as_frame):
        assert data[0] != data[1]
        cls = type(data)
        a, b = data[:2]

        ser = pd.Series(cls._from_sequence([a, a, b, b], dtype=data.dtype))
        cond = np.array([True, True, False, False])

        if as_frame:
            ser = ser.to_frame(name="a")
            cond = cond.reshape(-1, 1)

        result = ser.where(cond)
        expected = pd.Series(
            cls._from_sequence([a, a, na_value, na_value], dtype=data.dtype)
        )

        if as_frame:
            expected = expected.to_frame(name="a")
        self.assert_equal(result, expected)

        # array other
        cond = np.array([True, False, True, True])
        other = cls._from_sequence([a, b, a, b], dtype=data.dtype)
        if as_frame:
            other = pd.DataFrame({"a": other})
            cond = pd.DataFrame({"a": cond})
        result = ser.where(cond, other)
        expected = pd.Series(cls._from_sequence([a, b, b, b], dtype=data.dtype))
        if as_frame:
            expected = expected.to_frame(name="a")
        self.assert_equal(result, expected)

    @pytest.mark.parametrize("repeats", [0, 1, 2, [1, 2, 3]])
    def test_repeat(self, data, repeats, as_series, use_numpy):
        arr = type(data)._from_sequence(data[:3], dtype=data.dtype)
        if as_series:
            arr = pd.Series(arr)

        result = np.repeat(arr, repeats) if use_numpy else arr.repeat(repeats)

        repeats = [repeats] * 3 if isinstance(repeats, int) else repeats
        expected = [x for x, n in zip(arr, repeats) for _ in range(n)]
        expected = type(data)._from_sequence(expected, dtype=data.dtype)
        if as_series:
            expected = pd.Series(expected, index=arr.index.repeat(repeats))

        self.assert_equal(result, expected)

    @pytest.mark.parametrize(
        "repeats, kwargs, error, msg",
        [
            (2, {"axis": 1}, ValueError, "axis"),
            (-1, {}, ValueError, "negative"),
            ([1, 2], {}, ValueError, "shape"),
            (2, {"foo": "bar"}, TypeError, "'foo'"),
        ],
    )
    def test_repeat_raises(self, data, repeats, kwargs, error, msg, use_numpy):
        with pytest.raises(error, match=msg):
            if use_numpy:
                np.repeat(data, repeats, **kwargs)
            else:
                data.repeat(repeats, **kwargs)

    def test_delete(self, data):
        result = data.delete(0)
        expected = data[1:]
        self.assert_extension_array_equal(result, expected)

        result = data.delete([1, 3])
        expected = data._concat_same_type([data[[0]], data[[2]], data[4:]])
        self.assert_extension_array_equal(result, expected)

    @pytest.mark.parametrize("box", [pd.array, pd.Series, pd.DataFrame])
    def test_equals(self, data, na_value, as_series, box):
        data2 = type(data)._from_sequence([data[0]] * len(data), dtype=data.dtype)
        data_na = type(data)._from_sequence([na_value] * len(data), dtype=data.dtype)

        data = tm.box_expected(data, box, transpose=False)
        data2 = tm.box_expected(data2, box, transpose=False)
        data_na = tm.box_expected(data_na, box, transpose=False)

        # we are asserting with `is True/False` explicitly, to test that the
        # result is an actual Python bool, and not something "truthy"

        assert data.equals(data) is True
        assert data.equals(data.copy()) is True

        # unequal other data
        assert data.equals(data2) is False
        assert data.equals(data_na) is False

        # different length
        assert data[:2].equals(data[:3]) is False

        # empty are equal
        assert data[:0].equals(data[:0]) is True

        # other types
        assert data.equals(None) is False
        assert data[[0]].equals(data[0]) is False
