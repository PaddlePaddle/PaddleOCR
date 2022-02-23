from datetime import (
    datetime,
    timedelta,
)

import numpy as np
import pytest

from pandas import (
    Index,
    Int64Index,
    RangeIndex,
    UInt64Index,
)
import pandas._testing as tm


class TestRangeIndexSetOps:
    @pytest.mark.parametrize("klass", [RangeIndex, Int64Index, UInt64Index])
    def test_intersection_mismatched_dtype(self, klass):
        # check that we cast to float, not object
        index = RangeIndex(start=0, stop=20, step=2, name="foo")
        index = klass(index)

        flt = index.astype(np.float64)

        # bc index.equals(flt), we go through fastpath and get RangeIndex back
        result = index.intersection(flt)
        tm.assert_index_equal(result, index, exact=True)

        result = flt.intersection(index)
        tm.assert_index_equal(result, flt, exact=True)

        # neither empty, not-equals
        result = index.intersection(flt[1:])
        tm.assert_index_equal(result, flt[1:], exact=True)

        result = flt[1:].intersection(index)
        tm.assert_index_equal(result, flt[1:], exact=True)

        # empty other
        result = index.intersection(flt[:0])
        tm.assert_index_equal(result, flt[:0], exact=True)

        result = flt[:0].intersection(index)
        tm.assert_index_equal(result, flt[:0], exact=True)

    def test_intersection_empty(self, sort, names):
        # name retention on empty intersections
        index = RangeIndex(start=0, stop=20, step=2, name=names[0])

        # empty other
        result = index.intersection(index[:0].rename(names[1]), sort=sort)
        tm.assert_index_equal(result, index[:0].rename(names[2]), exact=True)

        # empty self
        result = index[:0].intersection(index.rename(names[1]), sort=sort)
        tm.assert_index_equal(result, index[:0].rename(names[2]), exact=True)

    def test_intersection(self, sort):
        # intersect with Int64Index
        index = RangeIndex(start=0, stop=20, step=2)
        other = Index(np.arange(1, 6))
        result = index.intersection(other, sort=sort)
        expected = Index(np.sort(np.intersect1d(index.values, other.values)))
        tm.assert_index_equal(result, expected)

        result = other.intersection(index, sort=sort)
        expected = Index(
            np.sort(np.asarray(np.intersect1d(index.values, other.values)))
        )
        tm.assert_index_equal(result, expected)

        # intersect with increasing RangeIndex
        other = RangeIndex(1, 6)
        result = index.intersection(other, sort=sort)
        expected = Index(np.sort(np.intersect1d(index.values, other.values)))
        tm.assert_index_equal(result, expected)

        # intersect with decreasing RangeIndex
        other = RangeIndex(5, 0, -1)
        result = index.intersection(other, sort=sort)
        expected = Index(np.sort(np.intersect1d(index.values, other.values)))
        tm.assert_index_equal(result, expected)

        # reversed (GH 17296)
        result = other.intersection(index, sort=sort)
        tm.assert_index_equal(result, expected)

        # GH 17296: intersect two decreasing RangeIndexes
        first = RangeIndex(10, -2, -2)
        other = RangeIndex(5, -4, -1)
        expected = first.astype(int).intersection(other.astype(int), sort=sort)
        result = first.intersection(other, sort=sort).astype(int)
        tm.assert_index_equal(result, expected)

        # reversed
        result = other.intersection(first, sort=sort).astype(int)
        tm.assert_index_equal(result, expected)

        index = RangeIndex(5, name="foo")

        # intersect of non-overlapping indices
        other = RangeIndex(5, 10, 1, name="foo")
        result = index.intersection(other, sort=sort)
        expected = RangeIndex(0, 0, 1, name="foo")
        tm.assert_index_equal(result, expected)

        other = RangeIndex(-1, -5, -1)
        result = index.intersection(other, sort=sort)
        expected = RangeIndex(0, 0, 1)
        tm.assert_index_equal(result, expected)

        # intersection of empty indices
        other = RangeIndex(0, 0, 1)
        result = index.intersection(other, sort=sort)
        expected = RangeIndex(0, 0, 1)
        tm.assert_index_equal(result, expected)

        result = other.intersection(index, sort=sort)
        tm.assert_index_equal(result, expected)

    def test_intersection_non_overlapping_gcd(self, sort, names):
        # intersection of non-overlapping values based on start value and gcd
        index = RangeIndex(1, 10, 2, name=names[0])
        other = RangeIndex(0, 10, 4, name=names[1])
        result = index.intersection(other, sort=sort)
        expected = RangeIndex(0, 0, 1, name=names[2])
        tm.assert_index_equal(result, expected)

    def test_union_noncomparable(self, sort):
        # corner case, non-Int64Index
        index = RangeIndex(start=0, stop=20, step=2)
        other = Index([datetime.now() + timedelta(i) for i in range(4)], dtype=object)
        result = index.union(other, sort=sort)
        expected = Index(np.concatenate((index, other)))
        tm.assert_index_equal(result, expected)

        result = other.union(index, sort=sort)
        expected = Index(np.concatenate((other, index)))
        tm.assert_index_equal(result, expected)

    @pytest.fixture(
        params=[
            (
                RangeIndex(0, 10, 1),
                RangeIndex(0, 10, 1),
                RangeIndex(0, 10, 1),
                RangeIndex(0, 10, 1),
            ),
            (
                RangeIndex(0, 10, 1),
                RangeIndex(5, 20, 1),
                RangeIndex(0, 20, 1),
                Int64Index(range(20)),
            ),
            (
                RangeIndex(0, 10, 1),
                RangeIndex(10, 20, 1),
                RangeIndex(0, 20, 1),
                Int64Index(range(20)),
            ),
            (
                RangeIndex(0, -10, -1),
                RangeIndex(0, -10, -1),
                RangeIndex(0, -10, -1),
                RangeIndex(0, -10, -1),
            ),
            (
                RangeIndex(0, -10, -1),
                RangeIndex(-10, -20, -1),
                RangeIndex(-19, 1, 1),
                Int64Index(range(0, -20, -1)),
            ),
            (
                RangeIndex(0, 10, 2),
                RangeIndex(1, 10, 2),
                RangeIndex(0, 10, 1),
                Int64Index(list(range(0, 10, 2)) + list(range(1, 10, 2))),
            ),
            (
                RangeIndex(0, 11, 2),
                RangeIndex(1, 12, 2),
                RangeIndex(0, 12, 1),
                Int64Index(list(range(0, 11, 2)) + list(range(1, 12, 2))),
            ),
            (
                RangeIndex(0, 21, 4),
                RangeIndex(-2, 24, 4),
                RangeIndex(-2, 24, 2),
                Int64Index(list(range(0, 21, 4)) + list(range(-2, 24, 4))),
            ),
            (
                RangeIndex(0, -20, -2),
                RangeIndex(-1, -21, -2),
                RangeIndex(-19, 1, 1),
                Int64Index(list(range(0, -20, -2)) + list(range(-1, -21, -2))),
            ),
            (
                RangeIndex(0, 100, 5),
                RangeIndex(0, 100, 20),
                RangeIndex(0, 100, 5),
                Int64Index(range(0, 100, 5)),
            ),
            (
                RangeIndex(0, -100, -5),
                RangeIndex(5, -100, -20),
                RangeIndex(-95, 10, 5),
                Int64Index(list(range(0, -100, -5)) + [5]),
            ),
            (
                RangeIndex(0, -11, -1),
                RangeIndex(1, -12, -4),
                RangeIndex(-11, 2, 1),
                Int64Index(list(range(0, -11, -1)) + [1, -11]),
            ),
            (RangeIndex(0), RangeIndex(0), RangeIndex(0), RangeIndex(0)),
            (
                RangeIndex(0, -10, -2),
                RangeIndex(0),
                RangeIndex(0, -10, -2),
                RangeIndex(0, -10, -2),
            ),
            (
                RangeIndex(0, 100, 2),
                RangeIndex(100, 150, 200),
                RangeIndex(0, 102, 2),
                Int64Index(range(0, 102, 2)),
            ),
            (
                RangeIndex(0, -100, -2),
                RangeIndex(-100, 50, 102),
                RangeIndex(-100, 4, 2),
                Int64Index(list(range(0, -100, -2)) + [-100, 2]),
            ),
            (
                RangeIndex(0, -100, -1),
                RangeIndex(0, -50, -3),
                RangeIndex(-99, 1, 1),
                Int64Index(list(range(0, -100, -1))),
            ),
            (
                RangeIndex(0, 1, 1),
                RangeIndex(5, 6, 10),
                RangeIndex(0, 6, 5),
                Int64Index([0, 5]),
            ),
            (
                RangeIndex(0, 10, 5),
                RangeIndex(-5, -6, -20),
                RangeIndex(-5, 10, 5),
                Int64Index([0, 5, -5]),
            ),
            (
                RangeIndex(0, 3, 1),
                RangeIndex(4, 5, 1),
                Int64Index([0, 1, 2, 4]),
                Int64Index([0, 1, 2, 4]),
            ),
            (
                RangeIndex(0, 10, 1),
                Int64Index([]),
                RangeIndex(0, 10, 1),
                RangeIndex(0, 10, 1),
            ),
            (
                RangeIndex(0),
                Int64Index([1, 5, 6]),
                Int64Index([1, 5, 6]),
                Int64Index([1, 5, 6]),
            ),
        ]
    )
    def unions(self, request):
        """Inputs and expected outputs for RangeIndex.union tests"""
        return request.param

    def test_union_sorted(self, unions):

        idx1, idx2, expected_sorted, expected_notsorted = unions

        res1 = idx1.union(idx2, sort=None)
        tm.assert_index_equal(res1, expected_sorted, exact=True)

        res1 = idx1.union(idx2, sort=False)
        tm.assert_index_equal(res1, expected_notsorted, exact=True)

        res2 = idx2.union(idx1, sort=None)
        res3 = idx1._int64index.union(idx2, sort=None)
        tm.assert_index_equal(res2, expected_sorted, exact=True)
        tm.assert_index_equal(res3, expected_sorted)

    def test_difference(self):
        # GH#12034 Cases where we operate against another RangeIndex and may
        #  get back another RangeIndex
        obj = RangeIndex.from_range(range(1, 10), name="foo")

        result = obj.difference(obj)
        expected = RangeIndex.from_range(range(0), name="foo")
        tm.assert_index_equal(result, expected, exact=True)

        result = obj.difference(expected.rename("bar"))
        tm.assert_index_equal(result, obj.rename(None), exact=True)

        result = obj.difference(obj[:3])
        tm.assert_index_equal(result, obj[3:], exact=True)

        result = obj.difference(obj[-3:])
        tm.assert_index_equal(result, obj[:-3], exact=True)

        result = obj[::-1].difference(obj[-3:])
        tm.assert_index_equal(result, obj[:-3][::-1], exact=True)

        result = obj[::-1].difference(obj[-3:][::-1])
        tm.assert_index_equal(result, obj[:-3][::-1], exact=True)

        result = obj.difference(obj[2:6])
        expected = Int64Index([1, 2, 7, 8, 9], name="foo")
        tm.assert_index_equal(result, expected)

    def test_difference_mismatched_step(self):
        obj = RangeIndex.from_range(range(1, 10), name="foo")

        result = obj.difference(obj[::2])
        expected = obj[1::2]._int64index
        tm.assert_index_equal(result, expected, exact=True)

        result = obj.difference(obj[1::2])
        expected = obj[::2]._int64index
        tm.assert_index_equal(result, expected, exact=True)

    def test_symmetric_difference(self):
        # GH#12034 Cases where we operate against another RangeIndex and may
        #  get back another RangeIndex
        left = RangeIndex.from_range(range(1, 10), name="foo")

        result = left.symmetric_difference(left)
        expected = RangeIndex.from_range(range(0), name="foo")
        tm.assert_index_equal(result, expected)

        result = left.symmetric_difference(expected.rename("bar"))
        tm.assert_index_equal(result, left.rename(None))

        result = left[:-2].symmetric_difference(left[2:])
        expected = Int64Index([1, 2, 8, 9], name="foo")
        tm.assert_index_equal(result, expected)

        right = RangeIndex.from_range(range(10, 15))

        result = left.symmetric_difference(right)
        expected = RangeIndex.from_range(range(1, 15))
        tm.assert_index_equal(result, expected)

        result = left.symmetric_difference(right[1:])
        expected = Int64Index([1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14])
        tm.assert_index_equal(result, expected)

    def test_putmask_range_cast(self):
        # GH#43240
        idx = RangeIndex(0, 5, name="test")
        result = idx.putmask(np.array([True, True, False, False, False]), 10)
        expected = Index([10, 10, 2, 3, 4], name="test")
        tm.assert_index_equal(result, expected)

    def test_where_range_cast(self):
        # GH#43240
        idx = RangeIndex(0, 5, name="test")
        result = idx.where(np.array([False, False, True, True, True]), 10)
        expected = Index([10, 10, 2, 3, 4], name="test")
        tm.assert_index_equal(result, expected)
