"""
The tests in this package are to ensure the proper resultant dtypes of
set operations.
"""
from datetime import datetime
import operator

import numpy as np
import pytest

from pandas.core.dtypes.cast import find_common_type

from pandas import (
    CategoricalIndex,
    DatetimeIndex,
    Float64Index,
    Index,
    Int64Index,
    MultiIndex,
    Series,
    TimedeltaIndex,
    Timestamp,
    UInt64Index,
)
import pandas._testing as tm
from pandas.api.types import (
    is_datetime64tz_dtype,
    is_signed_integer_dtype,
    pandas_dtype,
)

COMPATIBLE_INCONSISTENT_PAIRS = [
    (np.float64, np.int64),
    (np.float64, np.uint64),
]


def test_union_same_types(index):
    # Union with a non-unique, non-monotonic index raises error
    # Only needed for bool index factory
    idx1 = index.sort_values()
    idx2 = index.sort_values()
    assert idx1.union(idx2).dtype == idx1.dtype


def test_union_different_types(index_flat, index_flat2):
    # This test only considers combinations of indices
    # GH 23525
    idx1 = index_flat
    idx2 = index_flat2

    common_dtype = find_common_type([idx1.dtype, idx2.dtype])

    any_uint64 = idx1.dtype == np.uint64 or idx2.dtype == np.uint64
    idx1_signed = is_signed_integer_dtype(idx1.dtype)
    idx2_signed = is_signed_integer_dtype(idx2.dtype)

    # Union with a non-unique, non-monotonic index raises error
    # This applies to the boolean index
    idx1 = idx1.sort_values()
    idx2 = idx2.sort_values()

    res1 = idx1.union(idx2)
    res2 = idx2.union(idx1)

    if any_uint64 and (idx1_signed or idx2_signed):
        assert res1.dtype == np.dtype("O")
        assert res2.dtype == np.dtype("O")
    else:
        assert res1.dtype == common_dtype
        assert res2.dtype == common_dtype


@pytest.mark.parametrize(
    "idx_fact1,idx_fact2",
    [
        (tm.makeIntIndex, tm.makeRangeIndex),
        (tm.makeFloatIndex, tm.makeIntIndex),
        (tm.makeFloatIndex, tm.makeRangeIndex),
        (tm.makeFloatIndex, tm.makeUIntIndex),
    ],
)
def test_compatible_inconsistent_pairs(idx_fact1, idx_fact2):
    # GH 23525
    idx1 = idx_fact1(10)
    idx2 = idx_fact2(20)

    res1 = idx1.union(idx2)
    res2 = idx2.union(idx1)

    assert res1.dtype in (idx1.dtype, idx2.dtype)
    assert res2.dtype in (idx1.dtype, idx2.dtype)


@pytest.mark.parametrize(
    "left, right, expected",
    [
        ("int64", "int64", "int64"),
        ("int64", "uint64", "object"),
        ("int64", "float64", "float64"),
        ("uint64", "float64", "float64"),
        ("uint64", "uint64", "uint64"),
        ("float64", "float64", "float64"),
        ("datetime64[ns]", "int64", "object"),
        ("datetime64[ns]", "uint64", "object"),
        ("datetime64[ns]", "float64", "object"),
        ("datetime64[ns, CET]", "int64", "object"),
        ("datetime64[ns, CET]", "uint64", "object"),
        ("datetime64[ns, CET]", "float64", "object"),
        ("Period[D]", "int64", "object"),
        ("Period[D]", "uint64", "object"),
        ("Period[D]", "float64", "object"),
    ],
)
@pytest.mark.parametrize("names", [("foo", "foo", "foo"), ("foo", "bar", None)])
def test_union_dtypes(left, right, expected, names):
    left = pandas_dtype(left)
    right = pandas_dtype(right)
    a = Index([], dtype=left, name=names[0])
    b = Index([], dtype=right, name=names[1])
    result = a.union(b)
    assert result.dtype == expected
    assert result.name == names[2]

    # Testing name retention
    # TODO: pin down desired dtype; do we want it to be commutative?
    result = a.intersection(b)
    assert result.name == names[2]


def test_dunder_inplace_setops_deprecated(index):
    # GH#37374 these will become logical ops, not setops

    with tm.assert_produces_warning(FutureWarning):
        index |= index

    with tm.assert_produces_warning(FutureWarning):
        index &= index

    with tm.assert_produces_warning(FutureWarning):
        index ^= index


@pytest.mark.parametrize("values", [[1, 2, 2, 3], [3, 3]])
def test_intersection_duplicates(values):
    # GH#31326
    a = Index(values)
    b = Index([3, 3])
    result = a.intersection(b)
    expected = Index([3])
    tm.assert_index_equal(result, expected)


class TestSetOps:
    # Set operation tests shared by all indexes in the `index` fixture
    @pytest.mark.parametrize("case", [0.5, "xxx"])
    @pytest.mark.parametrize(
        "method", ["intersection", "union", "difference", "symmetric_difference"]
    )
    def test_set_ops_error_cases(self, case, method, index):
        # non-iterable input
        msg = "Input must be Index or array-like"
        with pytest.raises(TypeError, match=msg):
            getattr(index, method)(case)

    def test_intersection_base(self, index):
        if isinstance(index, CategoricalIndex):
            return

        first = index[:5]
        second = index[:3]
        intersect = first.intersection(second)
        assert tm.equalContents(intersect, second)

        if is_datetime64tz_dtype(index.dtype):
            # The second.values below will drop tz, so the rest of this test
            #  is not applicable.
            return

        # GH#10149
        cases = [klass(second.values) for klass in [np.array, Series, list]]
        for case in cases:
            result = first.intersection(case)
            assert tm.equalContents(result, second)

        if isinstance(index, MultiIndex):
            msg = "other must be a MultiIndex or a list of tuples"
            with pytest.raises(TypeError, match=msg):
                first.intersection([1, 2, 3])

    def test_union_base(self, index):
        first = index[3:]
        second = index[:5]
        everything = index
        union = first.union(second)
        assert tm.equalContents(union, everything)

        if is_datetime64tz_dtype(index.dtype):
            # The second.values below will drop tz, so the rest of this test
            #  is not applicable.
            return

        # GH#10149
        cases = [klass(second.values) for klass in [np.array, Series, list]]
        for case in cases:
            if not isinstance(index, CategoricalIndex):
                result = first.union(case)
                assert tm.equalContents(result, everything), (
                    result,
                    everything,
                    type(case),
                )

        if isinstance(index, MultiIndex):
            msg = "other must be a MultiIndex or a list of tuples"
            with pytest.raises(TypeError, match=msg):
                first.union([1, 2, 3])

    def test_difference_base(self, sort, index):
        first = index[2:]
        second = index[:4]
        if isinstance(index, CategoricalIndex) or index.is_boolean():
            answer = []
        else:
            answer = index[4:]
        result = first.difference(second, sort)
        assert tm.equalContents(result, answer)

        # GH#10149
        cases = [klass(second.values) for klass in [np.array, Series, list]]
        for case in cases:
            if isinstance(index, (DatetimeIndex, TimedeltaIndex)):
                assert type(result) == type(answer)
                tm.assert_numpy_array_equal(
                    result.sort_values().asi8, answer.sort_values().asi8
                )
            else:
                result = first.difference(case, sort)
                assert tm.equalContents(result, answer)

        if isinstance(index, MultiIndex):
            msg = "other must be a MultiIndex or a list of tuples"
            with pytest.raises(TypeError, match=msg):
                first.difference([1, 2, 3], sort)

    def test_symmetric_difference(self, index):
        if isinstance(index, CategoricalIndex):
            return
        if len(index) < 2:
            return
        if index[0] in index[1:] or index[-1] in index[:-1]:
            # index fixture has e.g. an index of bools that does not satisfy this,
            #  another with [0, 0, 1, 1, 2, 2]
            return

        first = index[1:]
        second = index[:-1]
        answer = index[[0, -1]]
        result = first.symmetric_difference(second)
        assert tm.equalContents(result, answer)

        # GH#10149
        cases = [klass(second.values) for klass in [np.array, Series, list]]
        for case in cases:
            result = first.symmetric_difference(case)

            if is_datetime64tz_dtype(first):
                # second.values casts to tznaive
                expected = first.union(case)
                tm.assert_index_equal(result, expected)
                continue

            assert tm.equalContents(result, answer)

        if isinstance(index, MultiIndex):
            msg = "other must be a MultiIndex or a list of tuples"
            with pytest.raises(TypeError, match=msg):
                first.symmetric_difference([1, 2, 3])

    @pytest.mark.parametrize(
        "fname, sname, expected_name",
        [
            ("A", "A", "A"),
            ("A", "B", None),
            ("A", None, None),
            (None, "B", None),
            (None, None, None),
        ],
    )
    def test_corner_union(self, index_flat, fname, sname, expected_name):
        # GH#9943, GH#9862
        # Test unions with various name combinations
        # Do not test MultiIndex or repeats
        index = index_flat
        if not index.is_unique:
            pytest.skip("Not for MultiIndex or repeated indices")

        # Test copy.union(copy)
        first = index.copy().set_names(fname)
        second = index.copy().set_names(sname)
        union = first.union(second)
        expected = index.copy().set_names(expected_name)
        tm.assert_index_equal(union, expected)

        # Test copy.union(empty)
        first = index.copy().set_names(fname)
        second = index.drop(index).set_names(sname)
        union = first.union(second)
        expected = index.copy().set_names(expected_name)
        tm.assert_index_equal(union, expected)

        # Test empty.union(copy)
        first = index.drop(index).set_names(fname)
        second = index.copy().set_names(sname)
        union = first.union(second)
        expected = index.copy().set_names(expected_name)
        tm.assert_index_equal(union, expected)

        # Test empty.union(empty)
        first = index.drop(index).set_names(fname)
        second = index.drop(index).set_names(sname)
        union = first.union(second)
        expected = index.drop(index).set_names(expected_name)
        tm.assert_index_equal(union, expected)

    @pytest.mark.parametrize(
        "fname, sname, expected_name",
        [
            ("A", "A", "A"),
            ("A", "B", None),
            ("A", None, None),
            (None, "B", None),
            (None, None, None),
        ],
    )
    def test_union_unequal(self, index_flat, fname, sname, expected_name):
        index = index_flat
        if not index.is_unique:
            pytest.skip("Not for MultiIndex or repeated indices")

        # test copy.union(subset) - need sort for unicode and string
        first = index.copy().set_names(fname)
        second = index[1:].set_names(sname)
        union = first.union(second).sort_values()
        expected = index.set_names(expected_name).sort_values()
        tm.assert_index_equal(union, expected)

    @pytest.mark.parametrize(
        "fname, sname, expected_name",
        [
            ("A", "A", "A"),
            ("A", "B", None),
            ("A", None, None),
            (None, "B", None),
            (None, None, None),
        ],
    )
    def test_corner_intersect(self, index_flat, fname, sname, expected_name):
        # GH#35847
        # Test intersections with various name combinations
        index = index_flat
        if not index.is_unique:
            pytest.skip("Not for MultiIndex or repeated indices")

        # Test copy.intersection(copy)
        first = index.copy().set_names(fname)
        second = index.copy().set_names(sname)
        intersect = first.intersection(second)
        expected = index.copy().set_names(expected_name)
        tm.assert_index_equal(intersect, expected)

        # Test copy.intersection(empty)
        first = index.copy().set_names(fname)
        second = index.drop(index).set_names(sname)
        intersect = first.intersection(second)
        expected = index.drop(index).set_names(expected_name)
        tm.assert_index_equal(intersect, expected)

        # Test empty.intersection(copy)
        first = index.drop(index).set_names(fname)
        second = index.copy().set_names(sname)
        intersect = first.intersection(second)
        expected = index.drop(index).set_names(expected_name)
        tm.assert_index_equal(intersect, expected)

        # Test empty.intersection(empty)
        first = index.drop(index).set_names(fname)
        second = index.drop(index).set_names(sname)
        intersect = first.intersection(second)
        expected = index.drop(index).set_names(expected_name)
        tm.assert_index_equal(intersect, expected)

    @pytest.mark.parametrize(
        "fname, sname, expected_name",
        [
            ("A", "A", "A"),
            ("A", "B", None),
            ("A", None, None),
            (None, "B", None),
            (None, None, None),
        ],
    )
    def test_intersect_unequal(self, index_flat, fname, sname, expected_name):
        index = index_flat
        if not index.is_unique:
            pytest.skip("Not for MultiIndex or repeated indices")

        # test copy.intersection(subset) - need sort for unicode and string
        first = index.copy().set_names(fname)
        second = index[1:].set_names(sname)
        intersect = first.intersection(second).sort_values()
        expected = index[1:].set_names(expected_name).sort_values()
        tm.assert_index_equal(intersect, expected)

    def test_intersection_name_retention_with_nameless(self, index):
        if isinstance(index, MultiIndex):
            index = index.rename(list(range(index.nlevels)))
        else:
            index = index.rename("foo")

        other = np.asarray(index)

        result = index.intersection(other)
        assert result.name == index.name

        # empty other, same dtype
        result = index.intersection(other[:0])
        assert result.name == index.name

        # empty `self`
        result = index[:0].intersection(other)
        assert result.name == index.name

    def test_difference_preserves_type_empty(self, index, sort):
        # GH#20040
        # If taking difference of a set and itself, it
        # needs to preserve the type of the index
        if not index.is_unique:
            return
        result = index.difference(index, sort=sort)
        expected = index[:0]
        tm.assert_index_equal(result, expected, exact=True)

    def test_difference_name_retention_equals(self, index, sort, names):
        if isinstance(index, MultiIndex):
            names = [[x] * index.nlevels for x in names]
        index = index.rename(names[0])
        other = index.rename(names[1])

        assert index.equals(other)

        result = index.difference(other)
        expected = index[:0].rename(names[2])
        tm.assert_index_equal(result, expected)

    def test_intersection_difference_match_empty(self, index, sort):
        # GH#20040
        # Test that the intersection of an index with an
        # empty index produces the same index as the difference
        # of an index with itself.  Test for all types
        if not index.is_unique:
            return
        inter = index.intersection(index[:0])
        diff = index.difference(index, sort=sort)
        tm.assert_index_equal(inter, diff, exact=True)


@pytest.mark.parametrize(
    "method", ["intersection", "union", "difference", "symmetric_difference"]
)
def test_setop_with_categorical(index, sort, method):
    if isinstance(index, MultiIndex):
        # tested separately in tests.indexes.multi.test_setops
        return

    other = index.astype("category")

    result = getattr(index, method)(other, sort=sort)
    expected = getattr(index, method)(index, sort=sort)
    tm.assert_index_equal(result, expected)

    result = getattr(index, method)(other[:5], sort=sort)
    expected = getattr(index, method)(index[:5], sort=sort)
    tm.assert_index_equal(result, expected)


def test_intersection_duplicates_all_indexes(index):
    # GH#38743
    if index.empty:
        # No duplicates in empty indexes
        return

    def check_intersection_commutative(left, right):
        assert left.intersection(right).equals(right.intersection(left))

    idx = index
    idx_non_unique = idx[[0, 0, 1, 2]]

    check_intersection_commutative(idx, idx_non_unique)
    assert idx.intersection(idx_non_unique).is_unique


@pytest.mark.parametrize(
    "cls",
    [
        Int64Index,
        Float64Index,
        DatetimeIndex,
        CategoricalIndex,
        lambda x: CategoricalIndex(x, categories=set(x)),
        TimedeltaIndex,
        lambda x: Index(x, dtype=object),
        UInt64Index,
    ],
)
def test_union_duplicate_index_subsets_of_each_other(cls):
    # GH#31326
    a = cls([1, 2, 2, 3])
    b = cls([3, 3, 4])
    expected = cls([1, 2, 2, 3, 3, 4])
    if isinstance(a, CategoricalIndex):
        expected = Index([1, 2, 2, 3, 3, 4])
    result = a.union(b)
    tm.assert_index_equal(result, expected)
    result = a.union(b, sort=False)
    tm.assert_index_equal(result, expected)


@pytest.mark.parametrize(
    "cls",
    [
        Int64Index,
        Float64Index,
        DatetimeIndex,
        CategoricalIndex,
        TimedeltaIndex,
        lambda x: Index(x, dtype=object),
    ],
)
def test_union_with_duplicate_index_and_non_monotonic(cls):
    # GH#36289
    a = cls([1, 0, 0])
    b = cls([0, 1])
    expected = cls([0, 0, 1])

    result = a.union(b)
    tm.assert_index_equal(result, expected)

    result = b.union(a)
    tm.assert_index_equal(result, expected)


def test_union_duplicate_index_different_dtypes():
    # GH#36289
    a = Index([1, 2, 2, 3])
    b = Index(["1", "0", "0"])
    expected = Index([1, 2, 2, 3, "1", "0", "0"])
    result = a.union(b, sort=False)
    tm.assert_index_equal(result, expected)


def test_union_same_value_duplicated_in_both():
    # GH#36289
    a = Index([0, 0, 1])
    b = Index([0, 0, 1, 2])
    result = a.union(b)
    expected = Index([0, 0, 1, 2])
    tm.assert_index_equal(result, expected)


@pytest.mark.parametrize("dup", [1, np.nan])
def test_union_nan_in_both(dup):
    # GH#36289
    a = Index([np.nan, 1, 2, 2])
    b = Index([np.nan, dup, 1, 2])
    result = a.union(b, sort=False)
    expected = Index([np.nan, dup, 1.0, 2.0, 2.0])
    tm.assert_index_equal(result, expected)


@pytest.mark.parametrize(
    "cls",
    [
        Int64Index,
        Float64Index,
        DatetimeIndex,
        TimedeltaIndex,
        lambda x: Index(x, dtype=object),
    ],
)
def test_union_with_duplicate_index_not_subset_and_non_monotonic(cls):
    # GH#36289
    a = cls([1, 0, 2])
    b = cls([0, 0, 1])
    expected = cls([0, 0, 1, 2])

    result = a.union(b)
    tm.assert_index_equal(result, expected)

    result = b.union(a)
    tm.assert_index_equal(result, expected)


class TestSetOpsUnsorted:
    # These may eventually belong in a dtype-specific test_setops, or
    #  parametrized over a more general fixture
    def test_intersect_str_dates(self):
        dt_dates = [datetime(2012, 2, 9), datetime(2012, 2, 22)]

        index1 = Index(dt_dates, dtype=object)
        index2 = Index(["aa"], dtype=object)
        result = index2.intersection(index1)

        expected = Index([], dtype=object)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("index", ["string"], indirect=True)
    def test_intersection(self, index, sort):
        first = index[:20]
        second = index[:10]
        intersect = first.intersection(second, sort=sort)
        if sort is None:
            tm.assert_index_equal(intersect, second.sort_values())
        assert tm.equalContents(intersect, second)

        # Corner cases
        inter = first.intersection(first, sort=sort)
        assert inter is first

    @pytest.mark.parametrize(
        "index2,keeps_name",
        [
            (Index([3, 4, 5, 6, 7], name="index"), True),  # preserve same name
            (Index([3, 4, 5, 6, 7], name="other"), False),  # drop diff names
            (Index([3, 4, 5, 6, 7]), False),
        ],
    )
    def test_intersection_name_preservation(self, index2, keeps_name, sort):
        index1 = Index([1, 2, 3, 4, 5], name="index")
        expected = Index([3, 4, 5])
        result = index1.intersection(index2, sort)

        if keeps_name:
            expected.name = "index"

        assert result.name == expected.name
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("index", ["string"], indirect=True)
    @pytest.mark.parametrize(
        "first_name,second_name,expected_name",
        [("A", "A", "A"), ("A", "B", None), (None, "B", None)],
    )
    def test_intersection_name_preservation2(
        self, index, first_name, second_name, expected_name, sort
    ):
        first = index[5:20]
        second = index[:10]
        first.name = first_name
        second.name = second_name
        intersect = first.intersection(second, sort=sort)
        assert intersect.name == expected_name

    def test_chained_union(self, sort):
        # Chained unions handles names correctly
        i1 = Index([1, 2], name="i1")
        i2 = Index([5, 6], name="i2")
        i3 = Index([3, 4], name="i3")
        union = i1.union(i2.union(i3, sort=sort), sort=sort)
        expected = i1.union(i2, sort=sort).union(i3, sort=sort)
        tm.assert_index_equal(union, expected)

        j1 = Index([1, 2], name="j1")
        j2 = Index([], name="j2")
        j3 = Index([], name="j3")
        union = j1.union(j2.union(j3, sort=sort), sort=sort)
        expected = j1.union(j2, sort=sort).union(j3, sort=sort)
        tm.assert_index_equal(union, expected)

    @pytest.mark.parametrize("index", ["string"], indirect=True)
    def test_union(self, index, sort):
        first = index[5:20]
        second = index[:10]
        everything = index[:20]

        union = first.union(second, sort=sort)
        if sort is None:
            tm.assert_index_equal(union, everything.sort_values())
        assert tm.equalContents(union, everything)

    @pytest.mark.parametrize("klass", [np.array, Series, list])
    @pytest.mark.parametrize("index", ["string"], indirect=True)
    def test_union_from_iterables(self, index, klass, sort):
        # GH#10149
        first = index[5:20]
        second = index[:10]
        everything = index[:20]

        case = klass(second.values)
        result = first.union(case, sort=sort)
        if sort is None:
            tm.assert_index_equal(result, everything.sort_values())
        assert tm.equalContents(result, everything)

    @pytest.mark.parametrize("index", ["string"], indirect=True)
    def test_union_identity(self, index, sort):
        first = index[5:20]

        union = first.union(first, sort=sort)
        # i.e. identity is not preserved when sort is True
        assert (union is first) is (not sort)

        # This should no longer be the same object, since [] is not consistent,
        # both objects will be recast to dtype('O')
        union = first.union([], sort=sort)
        assert (union is first) is (not sort)

        union = Index([]).union(first, sort=sort)
        assert (union is first) is (not sort)

    @pytest.mark.parametrize("index", ["string"], indirect=True)
    @pytest.mark.parametrize("second_name,expected", [(None, None), ("name", "name")])
    def test_difference_name_preservation(self, index, second_name, expected, sort):
        first = index[5:20]
        second = index[:10]
        answer = index[10:20]

        first.name = "name"
        second.name = second_name
        result = first.difference(second, sort=sort)

        assert tm.equalContents(result, answer)

        if expected is None:
            assert result.name is None
        else:
            assert result.name == expected

    def test_difference_empty_arg(self, index, sort):
        first = index[5:20]
        first.name = "name"
        result = first.difference([], sort)

        tm.assert_index_equal(result, first)

    @pytest.mark.parametrize("index", ["string"], indirect=True)
    def test_difference_identity(self, index, sort):
        first = index[5:20]
        first.name = "name"
        result = first.difference(first, sort)

        assert len(result) == 0
        assert result.name == first.name

    @pytest.mark.parametrize("index", ["string"], indirect=True)
    def test_difference_sort(self, index, sort):
        first = index[5:20]
        second = index[:10]

        result = first.difference(second, sort)
        expected = index[10:20]

        if sort is None:
            expected = expected.sort_values()

        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("opname", ["difference", "symmetric_difference"])
    def test_difference_incomparable(self, opname):
        a = Index([3, Timestamp("2000"), 1])
        b = Index([2, Timestamp("1999"), 1])
        op = operator.methodcaller(opname, b)

        with tm.assert_produces_warning(RuntimeWarning):
            # sort=None, the default
            result = op(a)
        expected = Index([3, Timestamp("2000"), 2, Timestamp("1999")])
        if opname == "difference":
            expected = expected[:2]
        tm.assert_index_equal(result, expected)

        # sort=False
        op = operator.methodcaller(opname, b, sort=False)
        result = op(a)
        tm.assert_index_equal(result, expected)

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.parametrize("opname", ["difference", "symmetric_difference"])
    def test_difference_incomparable_true(self, opname):
        # TODO: decide on True behaviour
        # # sort=True, raises
        a = Index([3, Timestamp("2000"), 1])
        b = Index([2, Timestamp("1999"), 1])
        op = operator.methodcaller(opname, b, sort=True)

        with pytest.raises(TypeError, match="Cannot compare"):
            op(a)

    def test_symmetric_difference_mi(self, sort):
        index1 = MultiIndex.from_tuples(zip(["foo", "bar", "baz"], [1, 2, 3]))
        index2 = MultiIndex.from_tuples([("foo", 1), ("bar", 3)])
        result = index1.symmetric_difference(index2, sort=sort)
        expected = MultiIndex.from_tuples([("bar", 2), ("baz", 3), ("bar", 3)])
        if sort is None:
            expected = expected.sort_values()
        tm.assert_index_equal(result, expected)
        assert tm.equalContents(result, expected)

    @pytest.mark.parametrize(
        "index2,expected",
        [
            (Index([0, 1, np.nan]), Index([2.0, 3.0, 0.0])),
            (Index([0, 1]), Index([np.nan, 2.0, 3.0, 0.0])),
        ],
    )
    def test_symmetric_difference_missing(self, index2, expected, sort):
        # GH#13514 change: {nan} - {nan} == {}
        # (GH#6444, sorting of nans, is no longer an issue)
        index1 = Index([1, np.nan, 2, 3])

        result = index1.symmetric_difference(index2, sort=sort)
        if sort is None:
            expected = expected.sort_values()
        tm.assert_index_equal(result, expected)

    def test_symmetric_difference_non_index(self, sort):
        index1 = Index([1, 2, 3, 4], name="index1")
        index2 = np.array([2, 3, 4, 5])
        expected = Index([1, 5])
        result = index1.symmetric_difference(index2, sort=sort)
        assert tm.equalContents(result, expected)
        assert result.name == "index1"

        result = index1.symmetric_difference(index2, result_name="new_name", sort=sort)
        assert tm.equalContents(result, expected)
        assert result.name == "new_name"
