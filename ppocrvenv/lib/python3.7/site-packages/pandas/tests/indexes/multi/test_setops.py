import numpy as np
import pytest

import pandas as pd
from pandas import (
    CategoricalIndex,
    Index,
    IntervalIndex,
    MultiIndex,
    Series,
)
import pandas._testing as tm


@pytest.mark.parametrize("case", [0.5, "xxx"])
@pytest.mark.parametrize(
    "method", ["intersection", "union", "difference", "symmetric_difference"]
)
def test_set_ops_error_cases(idx, case, sort, method):
    # non-iterable input
    msg = "Input must be Index or array-like"
    with pytest.raises(TypeError, match=msg):
        getattr(idx, method)(case, sort=sort)


@pytest.mark.parametrize("klass", [MultiIndex, np.array, Series, list])
def test_intersection_base(idx, sort, klass):
    first = idx[2::-1]  # first 3 elements reversed
    second = idx[:5]

    if klass is not MultiIndex:
        second = klass(second.values)

    intersect = first.intersection(second, sort=sort)
    if sort is None:
        expected = first.sort_values()
    else:
        expected = first
    tm.assert_index_equal(intersect, expected)

    msg = "other must be a MultiIndex or a list of tuples"
    with pytest.raises(TypeError, match=msg):
        first.intersection([1, 2, 3], sort=sort)


@pytest.mark.arm_slow
@pytest.mark.parametrize("klass", [MultiIndex, np.array, Series, list])
def test_union_base(idx, sort, klass):
    first = idx[::-1]
    second = idx[:5]

    if klass is not MultiIndex:
        second = klass(second.values)

    union = first.union(second, sort=sort)
    if sort is None:
        expected = first.sort_values()
    else:
        expected = first
    tm.assert_index_equal(union, expected)

    msg = "other must be a MultiIndex or a list of tuples"
    with pytest.raises(TypeError, match=msg):
        first.union([1, 2, 3], sort=sort)


def test_difference_base(idx, sort):
    second = idx[4:]
    answer = idx[:4]
    result = idx.difference(second, sort=sort)

    if sort is None:
        answer = answer.sort_values()

    assert result.equals(answer)
    tm.assert_index_equal(result, answer)

    # GH 10149
    cases = [klass(second.values) for klass in [np.array, Series, list]]
    for case in cases:
        result = idx.difference(case, sort=sort)
        tm.assert_index_equal(result, answer)

    msg = "other must be a MultiIndex or a list of tuples"
    with pytest.raises(TypeError, match=msg):
        idx.difference([1, 2, 3], sort=sort)


def test_symmetric_difference(idx, sort):
    first = idx[1:]
    second = idx[:-1]
    answer = idx[[-1, 0]]
    result = first.symmetric_difference(second, sort=sort)

    if sort is None:
        answer = answer.sort_values()

    tm.assert_index_equal(result, answer)

    # GH 10149
    cases = [klass(second.values) for klass in [np.array, Series, list]]
    for case in cases:
        result = first.symmetric_difference(case, sort=sort)
        tm.assert_index_equal(result, answer)

    msg = "other must be a MultiIndex or a list of tuples"
    with pytest.raises(TypeError, match=msg):
        first.symmetric_difference([1, 2, 3], sort=sort)


def test_multiindex_symmetric_difference():
    # GH 13490
    idx = MultiIndex.from_product([["a", "b"], ["A", "B"]], names=["a", "b"])
    with tm.assert_produces_warning(FutureWarning):
        result = idx ^ idx
    assert result.names == idx.names

    idx2 = idx.copy().rename(["A", "B"])
    with tm.assert_produces_warning(FutureWarning):
        result = idx ^ idx2
    assert result.names == [None, None]


def test_empty(idx):
    # GH 15270
    assert not idx.empty
    assert idx[:0].empty


def test_difference(idx, sort):

    first = idx
    result = first.difference(idx[-3:], sort=sort)
    vals = idx[:-3].values

    if sort is None:
        vals = sorted(vals)

    expected = MultiIndex.from_tuples(vals, sortorder=0, names=idx.names)

    assert isinstance(result, MultiIndex)
    assert result.equals(expected)
    assert result.names == idx.names
    tm.assert_index_equal(result, expected)

    # empty difference: reflexive
    result = idx.difference(idx, sort=sort)
    expected = idx[:0]
    assert result.equals(expected)
    assert result.names == idx.names

    # empty difference: superset
    result = idx[-3:].difference(idx, sort=sort)
    expected = idx[:0]
    assert result.equals(expected)
    assert result.names == idx.names

    # empty difference: degenerate
    result = idx[:0].difference(idx, sort=sort)
    expected = idx[:0]
    assert result.equals(expected)
    assert result.names == idx.names

    # names not the same
    chunklet = idx[-3:]
    chunklet.names = ["foo", "baz"]
    result = first.difference(chunklet, sort=sort)
    assert result.names == (None, None)

    # empty, but non-equal
    result = idx.difference(idx.sortlevel(1)[0], sort=sort)
    assert len(result) == 0

    # raise Exception called with non-MultiIndex
    result = first.difference(first.values, sort=sort)
    assert result.equals(first[:0])

    # name from empty array
    result = first.difference([], sort=sort)
    assert first.equals(result)
    assert first.names == result.names

    # name from non-empty array
    result = first.difference([("foo", "one")], sort=sort)
    expected = MultiIndex.from_tuples(
        [("bar", "one"), ("baz", "two"), ("foo", "two"), ("qux", "one"), ("qux", "two")]
    )
    expected.names = first.names
    assert first.names == result.names

    msg = "other must be a MultiIndex or a list of tuples"
    with pytest.raises(TypeError, match=msg):
        first.difference([1, 2, 3, 4, 5], sort=sort)


def test_difference_sort_special():
    # GH-24959
    idx = MultiIndex.from_product([[1, 0], ["a", "b"]])
    # sort=None, the default
    result = idx.difference([])
    tm.assert_index_equal(result, idx)


@pytest.mark.xfail(reason="Not implemented.")
def test_difference_sort_special_true():
    # TODO decide on True behaviour
    idx = MultiIndex.from_product([[1, 0], ["a", "b"]])
    result = idx.difference([], sort=True)
    expected = MultiIndex.from_product([[0, 1], ["a", "b"]])
    tm.assert_index_equal(result, expected)


def test_difference_sort_incomparable():
    # GH-24959
    idx = MultiIndex.from_product([[1, pd.Timestamp("2000"), 2], ["a", "b"]])

    other = MultiIndex.from_product([[3, pd.Timestamp("2000"), 4], ["c", "d"]])
    # sort=None, the default
    msg = "sort order is undefined for incomparable objects"
    with tm.assert_produces_warning(RuntimeWarning, match=msg):
        result = idx.difference(other)
    tm.assert_index_equal(result, idx)

    # sort=False
    result = idx.difference(other, sort=False)
    tm.assert_index_equal(result, idx)


def test_difference_sort_incomparable_true():
    idx = MultiIndex.from_product([[1, pd.Timestamp("2000"), 2], ["a", "b"]])
    other = MultiIndex.from_product([[3, pd.Timestamp("2000"), 4], ["c", "d"]])

    msg = "The 'sort' keyword only takes the values of None or False; True was passed."
    with pytest.raises(ValueError, match=msg):
        idx.difference(other, sort=True)


def test_union(idx, sort):
    piece1 = idx[:5][::-1]
    piece2 = idx[3:]

    the_union = piece1.union(piece2, sort=sort)

    if sort is None:
        tm.assert_index_equal(the_union, idx.sort_values())

    assert tm.equalContents(the_union, idx)

    # corner case, pass self or empty thing:
    the_union = idx.union(idx, sort=sort)
    tm.assert_index_equal(the_union, idx)

    the_union = idx.union(idx[:0], sort=sort)
    tm.assert_index_equal(the_union, idx)

    # FIXME: dont leave commented-out
    # won't work in python 3
    # tuples = _index.values
    # result = _index[:4] | tuples[4:]
    # assert result.equals(tuples)

    # not valid for python 3
    # def test_union_with_regular_index(self):
    #     other = Index(['A', 'B', 'C'])

    #     result = other.union(idx)
    #     assert ('foo', 'one') in result
    #     assert 'B' in result

    #     result2 = _index.union(other)
    #     assert result.equals(result2)


def test_intersection(idx, sort):
    piece1 = idx[:5][::-1]
    piece2 = idx[3:]

    the_int = piece1.intersection(piece2, sort=sort)

    if sort is None:
        tm.assert_index_equal(the_int, idx[3:5])
    assert tm.equalContents(the_int, idx[3:5])

    # corner case, pass self
    the_int = idx.intersection(idx, sort=sort)
    tm.assert_index_equal(the_int, idx)

    # empty intersection: disjoint
    empty = idx[:2].intersection(idx[2:], sort=sort)
    expected = idx[:0]
    assert empty.equals(expected)

    # FIXME: dont leave commented-out
    # can't do in python 3
    # tuples = _index.values
    # result = _index & tuples
    # assert result.equals(tuples)


@pytest.mark.parametrize(
    "method", ["intersection", "union", "difference", "symmetric_difference"]
)
def test_setop_with_categorical(idx, sort, method):
    other = idx.to_flat_index().astype("category")
    res_names = [None] * idx.nlevels

    result = getattr(idx, method)(other, sort=sort)
    expected = getattr(idx, method)(idx, sort=sort).rename(res_names)
    tm.assert_index_equal(result, expected)

    result = getattr(idx, method)(other[:5], sort=sort)
    expected = getattr(idx, method)(idx[:5], sort=sort).rename(res_names)
    tm.assert_index_equal(result, expected)


def test_intersection_non_object(idx, sort):
    other = Index(range(3), name="foo")

    result = idx.intersection(other, sort=sort)
    expected = MultiIndex(levels=idx.levels, codes=[[]] * idx.nlevels, names=None)
    tm.assert_index_equal(result, expected, exact=True)

    # if we pass a length-0 ndarray (i.e. no name, we retain our idx.name)
    result = idx.intersection(np.asarray(other)[:0], sort=sort)
    expected = MultiIndex(levels=idx.levels, codes=[[]] * idx.nlevels, names=idx.names)
    tm.assert_index_equal(result, expected, exact=True)

    msg = "other must be a MultiIndex or a list of tuples"
    with pytest.raises(TypeError, match=msg):
        # With non-zero length non-index, we try and fail to convert to tuples
        idx.intersection(np.asarray(other), sort=sort)


def test_intersect_equal_sort():
    # GH-24959
    idx = MultiIndex.from_product([[1, 0], ["a", "b"]])
    tm.assert_index_equal(idx.intersection(idx, sort=False), idx)
    tm.assert_index_equal(idx.intersection(idx, sort=None), idx)


@pytest.mark.xfail(reason="Not implemented.")
def test_intersect_equal_sort_true():
    # TODO decide on True behaviour
    idx = MultiIndex.from_product([[1, 0], ["a", "b"]])
    sorted_ = MultiIndex.from_product([[0, 1], ["a", "b"]])
    tm.assert_index_equal(idx.intersection(idx, sort=True), sorted_)


@pytest.mark.parametrize("slice_", [slice(None), slice(0)])
def test_union_sort_other_empty(slice_):
    # https://github.com/pandas-dev/pandas/issues/24959
    idx = MultiIndex.from_product([[1, 0], ["a", "b"]])

    # default, sort=None
    other = idx[slice_]
    tm.assert_index_equal(idx.union(other), idx)
    # MultiIndex does not special case empty.union(idx)
    # tm.assert_index_equal(other.union(idx), idx)

    # sort=False
    tm.assert_index_equal(idx.union(other, sort=False), idx)


@pytest.mark.xfail(reason="Not implemented.")
def test_union_sort_other_empty_sort(slice_):
    # TODO decide on True behaviour
    # # sort=True
    idx = MultiIndex.from_product([[1, 0], ["a", "b"]])
    other = idx[:0]
    result = idx.union(other, sort=True)
    expected = MultiIndex.from_product([[0, 1], ["a", "b"]])
    tm.assert_index_equal(result, expected)


def test_union_sort_other_incomparable():
    # https://github.com/pandas-dev/pandas/issues/24959
    idx = MultiIndex.from_product([[1, pd.Timestamp("2000")], ["a", "b"]])

    # default, sort=None
    with tm.assert_produces_warning(RuntimeWarning):
        result = idx.union(idx[:1])
    tm.assert_index_equal(result, idx)

    # sort=False
    result = idx.union(idx[:1], sort=False)
    tm.assert_index_equal(result, idx)


@pytest.mark.xfail(reason="Not implemented.")
def test_union_sort_other_incomparable_sort():
    # TODO decide on True behaviour
    # # sort=True
    idx = MultiIndex.from_product([[1, pd.Timestamp("2000")], ["a", "b"]])
    with pytest.raises(TypeError, match="Cannot compare"):
        idx.union(idx[:1], sort=True)


def test_union_non_object_dtype_raises():
    # GH#32646 raise NotImplementedError instead of less-informative error
    mi = MultiIndex.from_product([["a", "b"], [1, 2]])

    idx = mi.levels[1]

    msg = "Can only union MultiIndex with MultiIndex or Index of tuples"
    with pytest.raises(NotImplementedError, match=msg):
        mi.union(idx)


def test_union_empty_self_different_names():
    # GH#38423
    mi = MultiIndex.from_arrays([[]])
    mi2 = MultiIndex.from_arrays([[1, 2], [3, 4]], names=["a", "b"])
    result = mi.union(mi2)
    expected = MultiIndex.from_arrays([[1, 2], [3, 4]])
    tm.assert_index_equal(result, expected)


def test_union_multiindex_empty_rangeindex():
    # GH#41234
    mi = MultiIndex.from_arrays([[1, 2], [3, 4]], names=["a", "b"])
    ri = pd.RangeIndex(0)

    result_left = mi.union(ri)
    tm.assert_index_equal(mi, result_left, check_names=False)

    result_right = ri.union(mi)
    tm.assert_index_equal(mi, result_right, check_names=False)


@pytest.mark.parametrize(
    "method", ["union", "intersection", "difference", "symmetric_difference"]
)
def test_setops_disallow_true(method):
    idx1 = MultiIndex.from_product([["a", "b"], [1, 2]])
    idx2 = MultiIndex.from_product([["b", "c"], [1, 2]])

    with pytest.raises(ValueError, match="The 'sort' keyword only takes"):
        getattr(idx1, method)(idx2, sort=True)


@pytest.mark.parametrize(
    ("tuples", "exp_tuples"),
    [
        ([("val1", "test1")], [("val1", "test1")]),
        ([("val1", "test1"), ("val1", "test1")], [("val1", "test1")]),
        (
            [("val2", "test2"), ("val1", "test1")],
            [("val2", "test2"), ("val1", "test1")],
        ),
    ],
)
def test_intersect_with_duplicates(tuples, exp_tuples):
    # GH#36915
    left = MultiIndex.from_tuples(tuples, names=["first", "second"])
    right = MultiIndex.from_tuples(
        [("val1", "test1"), ("val1", "test1"), ("val2", "test2")],
        names=["first", "second"],
    )
    result = left.intersection(right)
    expected = MultiIndex.from_tuples(exp_tuples, names=["first", "second"])
    tm.assert_index_equal(result, expected)


@pytest.mark.parametrize(
    "data, names, expected",
    [
        ((1,), None, [None, None]),
        ((1,), ["a"], [None, None]),
        ((1,), ["b"], [None, None]),
        ((1, 2), ["c", "d"], [None, None]),
        ((1, 2), ["b", "a"], [None, None]),
        ((1, 2, 3), ["a", "b", "c"], [None, None]),
        ((1, 2), ["a", "c"], ["a", None]),
        ((1, 2), ["c", "b"], [None, "b"]),
        ((1, 2), ["a", "b"], ["a", "b"]),
        ((1, 2), [None, "b"], [None, "b"]),
    ],
)
def test_maybe_match_names(data, names, expected):
    # GH#38323
    mi = MultiIndex.from_tuples([], names=["a", "b"])
    mi2 = MultiIndex.from_tuples([data], names=names)
    result = mi._maybe_match_names(mi2)
    assert result == expected


def test_intersection_equal_different_names():
    # GH#30302
    mi1 = MultiIndex.from_arrays([[1, 2], [3, 4]], names=["c", "b"])
    mi2 = MultiIndex.from_arrays([[1, 2], [3, 4]], names=["a", "b"])

    result = mi1.intersection(mi2)
    expected = MultiIndex.from_arrays([[1, 2], [3, 4]], names=[None, "b"])
    tm.assert_index_equal(result, expected)


def test_intersection_different_names():
    # GH#38323
    mi = MultiIndex.from_arrays([[1], [3]], names=["c", "b"])
    mi2 = MultiIndex.from_arrays([[1], [3]])
    result = mi.intersection(mi2)
    tm.assert_index_equal(result, mi2)


def test_intersection_with_missing_values_on_both_sides(nulls_fixture):
    # GH#38623
    mi1 = MultiIndex.from_arrays([[3, nulls_fixture, 4, nulls_fixture], [1, 2, 4, 2]])
    mi2 = MultiIndex.from_arrays([[3, nulls_fixture, 3], [1, 2, 4]])
    result = mi1.intersection(mi2)
    expected = MultiIndex.from_arrays([[3.0, nulls_fixture], [1, 2]])
    tm.assert_index_equal(result, expected)


def test_union_nan_got_duplicated():
    # GH#38977
    mi1 = MultiIndex.from_arrays([[1.0, np.nan], [2, 3]])
    mi2 = MultiIndex.from_arrays([[1.0, np.nan, 3.0], [2, 3, 4]])
    result = mi1.union(mi2)
    tm.assert_index_equal(result, mi2)


def test_union_duplicates(index):
    # GH#38977
    if index.empty or isinstance(index, (IntervalIndex, CategoricalIndex)):
        # No duplicates in empty indexes
        return
    values = index.unique().values.tolist()
    mi1 = MultiIndex.from_arrays([values, [1] * len(values)])
    mi2 = MultiIndex.from_arrays([[values[0]] + values, [1] * (len(values) + 1)])
    result = mi1.union(mi2)
    tm.assert_index_equal(result, mi2.sort_values())

    result = mi2.union(mi1)
    tm.assert_index_equal(result, mi2.sort_values())
