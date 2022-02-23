import numpy as np
import pytest

from pandas import (
    Float64Index,
    Index,
    Int64Index,
    RangeIndex,
    Series,
    Timestamp,
    UInt64Index,
)
import pandas._testing as tm


@pytest.fixture
def index_large():
    # large values used in UInt64Index tests where no compat needed with Int64/Float64
    large = [2 ** 63, 2 ** 63 + 10, 2 ** 63 + 15, 2 ** 63 + 20, 2 ** 63 + 25]
    return UInt64Index(large)


class TestGetLoc:
    @pytest.mark.parametrize("method", [None, "pad", "backfill", "nearest"])
    def test_get_loc(self, method):
        index = Index([0, 1, 2])
        assert index.get_loc(1, method=method) == 1

        if method:
            assert index.get_loc(1, method=method, tolerance=0) == 1

    @pytest.mark.parametrize("method", [None, "pad", "backfill", "nearest"])
    def test_get_loc_raises_bad_label(self, method):
        index = Index([0, 1, 2])
        if method:
            msg = "not supported between"
        else:
            msg = "invalid key"

        with pytest.raises(TypeError, match=msg):
            index.get_loc([1, 2], method=method)

    @pytest.mark.parametrize(
        "method,loc", [("pad", 1), ("backfill", 2), ("nearest", 1)]
    )
    def test_get_loc_tolerance(self, method, loc):
        index = Index([0, 1, 2])
        assert index.get_loc(1.1, method) == loc
        assert index.get_loc(1.1, method, tolerance=1) == loc

    @pytest.mark.parametrize("method", ["pad", "backfill", "nearest"])
    def test_get_loc_outside_tolerance_raises(self, method):
        index = Index([0, 1, 2])
        with pytest.raises(KeyError, match="1.1"):
            index.get_loc(1.1, method, tolerance=0.05)

    def test_get_loc_bad_tolerance_raises(self):
        index = Index([0, 1, 2])
        with pytest.raises(ValueError, match="must be numeric"):
            index.get_loc(1.1, "nearest", tolerance="invalid")

    def test_get_loc_tolerance_no_method_raises(self):
        index = Index([0, 1, 2])
        with pytest.raises(ValueError, match="tolerance .* valid if"):
            index.get_loc(1.1, tolerance=1)

    def test_get_loc_raises_missized_tolerance(self):
        index = Index([0, 1, 2])
        with pytest.raises(ValueError, match="tolerance size must match"):
            index.get_loc(1.1, "nearest", tolerance=[1, 1])

    def test_get_loc_float64(self):
        idx = Float64Index([0.0, 1.0, 2.0])
        for method in [None, "pad", "backfill", "nearest"]:
            assert idx.get_loc(1, method) == 1
            if method is not None:
                assert idx.get_loc(1, method, tolerance=0) == 1

        for method, loc in [("pad", 1), ("backfill", 2), ("nearest", 1)]:
            assert idx.get_loc(1.1, method) == loc
            assert idx.get_loc(1.1, method, tolerance=0.9) == loc

        with pytest.raises(KeyError, match="^'foo'$"):
            idx.get_loc("foo")
        with pytest.raises(KeyError, match=r"^1\.5$"):
            idx.get_loc(1.5)
        with pytest.raises(KeyError, match=r"^1\.5$"):
            idx.get_loc(1.5, method="pad", tolerance=0.1)
        with pytest.raises(KeyError, match="^True$"):
            idx.get_loc(True)
        with pytest.raises(KeyError, match="^False$"):
            idx.get_loc(False)

        with pytest.raises(ValueError, match="must be numeric"):
            idx.get_loc(1.4, method="nearest", tolerance="foo")

        with pytest.raises(ValueError, match="must contain numeric elements"):
            idx.get_loc(1.4, method="nearest", tolerance=np.array(["foo"]))

        with pytest.raises(
            ValueError, match="tolerance size must match target index size"
        ):
            idx.get_loc(1.4, method="nearest", tolerance=np.array([1, 2]))

    def test_get_loc_na(self):
        idx = Float64Index([np.nan, 1, 2])
        assert idx.get_loc(1) == 1
        assert idx.get_loc(np.nan) == 0

        idx = Float64Index([np.nan, 1, np.nan])
        assert idx.get_loc(1) == 1

        # representable by slice [0:2:2]
        msg = "'Cannot get left slice bound for non-unique label: nan'"
        with pytest.raises(KeyError, match=msg):
            idx.slice_locs(np.nan)
        # not representable by slice
        idx = Float64Index([np.nan, 1, np.nan, np.nan])
        assert idx.get_loc(1) == 1
        msg = "'Cannot get left slice bound for non-unique label: nan"
        with pytest.raises(KeyError, match=msg):
            idx.slice_locs(np.nan)

    def test_get_loc_missing_nan(self):
        # GH#8569
        idx = Float64Index([1, 2])
        assert idx.get_loc(1) == 0
        with pytest.raises(KeyError, match=r"^3$"):
            idx.get_loc(3)
        with pytest.raises(KeyError, match="^nan$"):
            idx.get_loc(np.nan)
        with pytest.raises(TypeError, match=r"'\[nan\]' is an invalid key"):
            # listlike/non-hashable raises TypeError
            idx.get_loc([np.nan])

    @pytest.mark.parametrize("vals", [[1], [1.0], [Timestamp("2019-12-31")], ["test"]])
    @pytest.mark.parametrize("method", ["nearest", "pad", "backfill"])
    def test_get_loc_float_index_nan_with_method(self, vals, method):
        # GH#39382
        idx = Index(vals)
        with pytest.raises(KeyError, match="nan"):
            idx.get_loc(np.nan, method=method)


class TestGetIndexer:
    def test_get_indexer(self):
        index1 = Index([1, 2, 3, 4, 5])
        index2 = Index([2, 4, 6])

        r1 = index1.get_indexer(index2)
        e1 = np.array([1, 3, -1], dtype=np.intp)
        tm.assert_almost_equal(r1, e1)

    @pytest.mark.parametrize("reverse", [True, False])
    @pytest.mark.parametrize(
        "expected,method",
        [
            (np.array([-1, 0, 0, 1, 1], dtype=np.intp), "pad"),
            (np.array([-1, 0, 0, 1, 1], dtype=np.intp), "ffill"),
            (np.array([0, 0, 1, 1, 2], dtype=np.intp), "backfill"),
            (np.array([0, 0, 1, 1, 2], dtype=np.intp), "bfill"),
        ],
    )
    def test_get_indexer_methods(self, reverse, expected, method):
        index1 = Index([1, 2, 3, 4, 5])
        index2 = Index([2, 4, 6])

        if reverse:
            index1 = index1[::-1]
            expected = expected[::-1]

        result = index2.get_indexer(index1, method=method)
        tm.assert_almost_equal(result, expected)

    def test_get_indexer_invalid(self):
        # GH10411
        index = Index(np.arange(10))

        with pytest.raises(ValueError, match="tolerance argument"):
            index.get_indexer([1, 0], tolerance=1)

        with pytest.raises(ValueError, match="limit argument"):
            index.get_indexer([1, 0], limit=1)

    @pytest.mark.parametrize(
        "method, tolerance, indexer, expected",
        [
            ("pad", None, [0, 5, 9], [0, 5, 9]),
            ("backfill", None, [0, 5, 9], [0, 5, 9]),
            ("nearest", None, [0, 5, 9], [0, 5, 9]),
            ("pad", 0, [0, 5, 9], [0, 5, 9]),
            ("backfill", 0, [0, 5, 9], [0, 5, 9]),
            ("nearest", 0, [0, 5, 9], [0, 5, 9]),
            ("pad", None, [0.2, 1.8, 8.5], [0, 1, 8]),
            ("backfill", None, [0.2, 1.8, 8.5], [1, 2, 9]),
            ("nearest", None, [0.2, 1.8, 8.5], [0, 2, 9]),
            ("pad", 1, [0.2, 1.8, 8.5], [0, 1, 8]),
            ("backfill", 1, [0.2, 1.8, 8.5], [1, 2, 9]),
            ("nearest", 1, [0.2, 1.8, 8.5], [0, 2, 9]),
            ("pad", 0.2, [0.2, 1.8, 8.5], [0, -1, -1]),
            ("backfill", 0.2, [0.2, 1.8, 8.5], [-1, 2, -1]),
            ("nearest", 0.2, [0.2, 1.8, 8.5], [0, 2, -1]),
        ],
    )
    def test_get_indexer_nearest(self, method, tolerance, indexer, expected):
        index = Index(np.arange(10))

        actual = index.get_indexer(indexer, method=method, tolerance=tolerance)
        tm.assert_numpy_array_equal(actual, np.array(expected, dtype=np.intp))

    @pytest.mark.parametrize("listtype", [list, tuple, Series, np.array])
    @pytest.mark.parametrize(
        "tolerance, expected",
        list(
            zip(
                [[0.3, 0.3, 0.1], [0.2, 0.1, 0.1], [0.1, 0.5, 0.5]],
                [[0, 2, -1], [0, -1, -1], [-1, 2, 9]],
            )
        ),
    )
    def test_get_indexer_nearest_listlike_tolerance(
        self, tolerance, expected, listtype
    ):
        index = Index(np.arange(10))

        actual = index.get_indexer(
            [0.2, 1.8, 8.5], method="nearest", tolerance=listtype(tolerance)
        )
        tm.assert_numpy_array_equal(actual, np.array(expected, dtype=np.intp))

    def test_get_indexer_nearest_error(self):
        index = Index(np.arange(10))
        with pytest.raises(ValueError, match="limit argument"):
            index.get_indexer([1, 0], method="nearest", limit=1)

        with pytest.raises(ValueError, match="tolerance size must match"):
            index.get_indexer([1, 0], method="nearest", tolerance=[1, 2, 3])

    @pytest.mark.parametrize(
        "method,expected",
        [("pad", [8, 7, 0]), ("backfill", [9, 8, 1]), ("nearest", [9, 7, 0])],
    )
    def test_get_indexer_nearest_decreasing(self, method, expected):
        index = Index(np.arange(10))[::-1]

        actual = index.get_indexer([0, 5, 9], method=method)
        tm.assert_numpy_array_equal(actual, np.array([9, 4, 0], dtype=np.intp))

        actual = index.get_indexer([0.2, 1.8, 8.5], method=method)
        tm.assert_numpy_array_equal(actual, np.array(expected, dtype=np.intp))

    @pytest.mark.parametrize(
        "idx_class", [Int64Index, RangeIndex, Float64Index, UInt64Index]
    )
    @pytest.mark.parametrize("method", ["get_indexer", "get_indexer_non_unique"])
    def test_get_indexer_numeric_index_boolean_target(self, method, idx_class):
        # GH 16877

        numeric_index = idx_class(RangeIndex(4))
        other = Index([True, False, True])

        result = getattr(numeric_index, method)(other)
        expected = np.array([-1, -1, -1], dtype=np.intp)
        if method == "get_indexer":
            tm.assert_numpy_array_equal(result, expected)
        else:
            missing = np.arange(3, dtype=np.intp)
            tm.assert_numpy_array_equal(result[0], expected)
            tm.assert_numpy_array_equal(result[1], missing)

    @pytest.mark.parametrize("method", ["pad", "backfill", "nearest"])
    def test_get_indexer_with_method_numeric_vs_bool(self, method):
        left = Index([1, 2, 3])
        right = Index([True, False])

        with pytest.raises(TypeError, match="Cannot compare"):
            left.get_indexer(right, method=method)

        with pytest.raises(TypeError, match="Cannot compare"):
            right.get_indexer(left, method=method)

    def test_get_indexer_numeric_vs_bool(self):
        left = Index([1, 2, 3])
        right = Index([True, False])

        res = left.get_indexer(right)
        expected = -1 * np.ones(len(right), dtype=np.intp)
        tm.assert_numpy_array_equal(res, expected)

        res = right.get_indexer(left)
        expected = -1 * np.ones(len(left), dtype=np.intp)
        tm.assert_numpy_array_equal(res, expected)

        res = left.get_indexer_non_unique(right)[0]
        expected = -1 * np.ones(len(right), dtype=np.intp)
        tm.assert_numpy_array_equal(res, expected)

        res = right.get_indexer_non_unique(left)[0]
        expected = -1 * np.ones(len(left), dtype=np.intp)
        tm.assert_numpy_array_equal(res, expected)

    def test_get_indexer_float64(self):
        idx = Float64Index([0.0, 1.0, 2.0])
        tm.assert_numpy_array_equal(
            idx.get_indexer(idx), np.array([0, 1, 2], dtype=np.intp)
        )

        target = [-0.1, 0.5, 1.1]
        tm.assert_numpy_array_equal(
            idx.get_indexer(target, "pad"), np.array([-1, 0, 1], dtype=np.intp)
        )
        tm.assert_numpy_array_equal(
            idx.get_indexer(target, "backfill"), np.array([0, 1, 2], dtype=np.intp)
        )
        tm.assert_numpy_array_equal(
            idx.get_indexer(target, "nearest"), np.array([0, 1, 1], dtype=np.intp)
        )

    def test_get_indexer_nan(self):
        # GH#7820
        result = Float64Index([1, 2, np.nan]).get_indexer([np.nan])
        expected = np.array([2], dtype=np.intp)
        tm.assert_numpy_array_equal(result, expected)

    def test_get_indexer_int64(self):
        index = Int64Index(range(0, 20, 2))
        target = Int64Index(np.arange(10))
        indexer = index.get_indexer(target)
        expected = np.array([0, -1, 1, -1, 2, -1, 3, -1, 4, -1], dtype=np.intp)
        tm.assert_numpy_array_equal(indexer, expected)

        target = Int64Index(np.arange(10))
        indexer = index.get_indexer(target, method="pad")
        expected = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4], dtype=np.intp)
        tm.assert_numpy_array_equal(indexer, expected)

        target = Int64Index(np.arange(10))
        indexer = index.get_indexer(target, method="backfill")
        expected = np.array([0, 1, 1, 2, 2, 3, 3, 4, 4, 5], dtype=np.intp)
        tm.assert_numpy_array_equal(indexer, expected)

    def test_get_indexer_uint64(self, index_large):
        target = UInt64Index(np.arange(10).astype("uint64") * 5 + 2 ** 63)
        indexer = index_large.get_indexer(target)
        expected = np.array([0, -1, 1, 2, 3, 4, -1, -1, -1, -1], dtype=np.intp)
        tm.assert_numpy_array_equal(indexer, expected)

        target = UInt64Index(np.arange(10).astype("uint64") * 5 + 2 ** 63)
        indexer = index_large.get_indexer(target, method="pad")
        expected = np.array([0, 0, 1, 2, 3, 4, 4, 4, 4, 4], dtype=np.intp)
        tm.assert_numpy_array_equal(indexer, expected)

        target = UInt64Index(np.arange(10).astype("uint64") * 5 + 2 ** 63)
        indexer = index_large.get_indexer(target, method="backfill")
        expected = np.array([0, 1, 1, 2, 3, 4, -1, -1, -1, -1], dtype=np.intp)
        tm.assert_numpy_array_equal(indexer, expected)


class TestWhere:
    @pytest.mark.parametrize(
        "index",
        [
            Float64Index(np.arange(5, dtype="float64")),
            Int64Index(range(0, 20, 2)),
            UInt64Index(np.arange(5, dtype="uint64")),
        ],
    )
    @pytest.mark.parametrize("klass", [list, tuple, np.array, Series])
    def test_where(self, klass, index):
        cond = [True] * len(index)
        expected = index
        result = index.where(klass(cond))

        cond = [False] + [True] * (len(index) - 1)
        expected = Float64Index([index._na_value] + index[1:].tolist())
        result = index.where(klass(cond))
        tm.assert_index_equal(result, expected)

    def test_where_uin64(self):
        idx = UInt64Index([0, 6, 2])
        mask = np.array([False, True, False])
        other = np.array([1], dtype=np.int64)

        expected = UInt64Index([1, 6, 1])

        result = idx.where(mask, other)
        tm.assert_index_equal(result, expected)

        result = idx.putmask(~mask, other)
        tm.assert_index_equal(result, expected)


class TestTake:
    @pytest.mark.parametrize("klass", [Float64Index, Int64Index, UInt64Index])
    def test_take_preserve_name(self, klass):
        index = klass([1, 2, 3, 4], name="foo")
        taken = index.take([3, 0, 1])
        assert index.name == taken.name

    def test_take_fill_value_float64(self):
        # GH 12631
        idx = Float64Index([1.0, 2.0, 3.0], name="xxx")
        result = idx.take(np.array([1, 0, -1]))
        expected = Float64Index([2.0, 1.0, 3.0], name="xxx")
        tm.assert_index_equal(result, expected)

        # fill_value
        result = idx.take(np.array([1, 0, -1]), fill_value=True)
        expected = Float64Index([2.0, 1.0, np.nan], name="xxx")
        tm.assert_index_equal(result, expected)

        # allow_fill=False
        result = idx.take(np.array([1, 0, -1]), allow_fill=False, fill_value=True)
        expected = Float64Index([2.0, 1.0, 3.0], name="xxx")
        tm.assert_index_equal(result, expected)

        msg = (
            "When allow_fill=True and fill_value is not None, "
            "all indices must be >= -1"
        )
        with pytest.raises(ValueError, match=msg):
            idx.take(np.array([1, 0, -2]), fill_value=True)
        with pytest.raises(ValueError, match=msg):
            idx.take(np.array([1, 0, -5]), fill_value=True)

        msg = "index -5 is out of bounds for (axis 0 with )?size 3"
        with pytest.raises(IndexError, match=msg):
            idx.take(np.array([1, -5]))

    @pytest.mark.parametrize("klass", [Int64Index, UInt64Index])
    def test_take_fill_value_ints(self, klass):
        # see gh-12631
        idx = klass([1, 2, 3], name="xxx")
        result = idx.take(np.array([1, 0, -1]))
        expected = klass([2, 1, 3], name="xxx")
        tm.assert_index_equal(result, expected)

        name = klass.__name__
        msg = f"Unable to fill values because {name} cannot contain NA"

        # fill_value=True
        with pytest.raises(ValueError, match=msg):
            idx.take(np.array([1, 0, -1]), fill_value=True)

        # allow_fill=False
        result = idx.take(np.array([1, 0, -1]), allow_fill=False, fill_value=True)
        expected = klass([2, 1, 3], name="xxx")
        tm.assert_index_equal(result, expected)

        with pytest.raises(ValueError, match=msg):
            idx.take(np.array([1, 0, -2]), fill_value=True)
        with pytest.raises(ValueError, match=msg):
            idx.take(np.array([1, 0, -5]), fill_value=True)

        msg = "index -5 is out of bounds for (axis 0 with )?size 3"
        with pytest.raises(IndexError, match=msg):
            idx.take(np.array([1, -5]))


class TestContains:
    @pytest.mark.parametrize("klass", [Float64Index, Int64Index, UInt64Index])
    def test_contains_none(self, klass):
        # GH#35788 should return False, not raise TypeError
        index = klass([0, 1, 2, 3, 4])
        assert None not in index

    def test_contains_float64_nans(self):
        index = Float64Index([1.0, 2.0, np.nan])
        assert np.nan in index

    def test_contains_float64_not_nans(self):
        index = Float64Index([1.0, 2.0, np.nan])
        assert 1.0 in index


class TestSliceLocs:
    @pytest.mark.parametrize("dtype", [int, float])
    def test_slice_locs(self, dtype):
        index = Index(np.array([0, 1, 2, 5, 6, 7, 9, 10], dtype=dtype))
        n = len(index)

        assert index.slice_locs(start=2) == (2, n)
        assert index.slice_locs(start=3) == (3, n)
        assert index.slice_locs(3, 8) == (3, 6)
        assert index.slice_locs(5, 10) == (3, n)
        assert index.slice_locs(end=8) == (0, 6)
        assert index.slice_locs(end=9) == (0, 7)

        # reversed
        index2 = index[::-1]
        assert index2.slice_locs(8, 2) == (2, 6)
        assert index2.slice_locs(7, 3) == (2, 5)

    @pytest.mark.parametrize("dtype", [int, float])
    def test_slice_locs_float_locs(self, dtype):
        index = Index(np.array([0, 1, 2, 5, 6, 7, 9, 10], dtype=dtype))
        n = len(index)
        assert index.slice_locs(5.0, 10.0) == (3, n)
        assert index.slice_locs(4.5, 10.5) == (3, 8)

        index2 = index[::-1]
        assert index2.slice_locs(8.5, 1.5) == (2, 6)
        assert index2.slice_locs(10.5, -1) == (0, n)

    @pytest.mark.parametrize("dtype", [int, float])
    def test_slice_locs_dup_numeric(self, dtype):
        index = Index(np.array([10, 12, 12, 14], dtype=dtype))
        assert index.slice_locs(12, 12) == (1, 3)
        assert index.slice_locs(11, 13) == (1, 3)

        index2 = index[::-1]
        assert index2.slice_locs(12, 12) == (1, 3)
        assert index2.slice_locs(13, 11) == (1, 3)

    def test_slice_locs_na(self):
        index = Index([np.nan, 1, 2])
        assert index.slice_locs(1) == (1, 3)
        assert index.slice_locs(np.nan) == (0, 3)

        index = Index([0, np.nan, np.nan, 1, 2])
        assert index.slice_locs(np.nan) == (1, 5)

    def test_slice_locs_na_raises(self):
        index = Index([np.nan, 1, 2])
        with pytest.raises(KeyError, match=""):
            index.slice_locs(start=1.5)

        with pytest.raises(KeyError, match=""):
            index.slice_locs(end=1.5)


class TestGetSliceBounds:
    @pytest.mark.parametrize("kind", ["getitem", "loc", None])
    @pytest.mark.parametrize("side, expected", [("left", 4), ("right", 5)])
    def test_get_slice_bounds_within(self, kind, side, expected):
        index = Index(range(6))
        result = index.get_slice_bound(4, kind=kind, side=side)
        assert result == expected

    @pytest.mark.parametrize("kind", ["getitem", "loc", None])
    @pytest.mark.parametrize("side", ["left", "right"])
    @pytest.mark.parametrize("bound, expected", [(-1, 0), (10, 6)])
    def test_get_slice_bounds_outside(self, kind, side, expected, bound):
        index = Index(range(6))
        result = index.get_slice_bound(bound, kind=kind, side=side)
        assert result == expected
