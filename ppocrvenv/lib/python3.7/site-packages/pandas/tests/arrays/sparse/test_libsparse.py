import operator

import numpy as np
import pytest

import pandas._libs.sparse as splib
import pandas.util._test_decorators as td

from pandas import Series
import pandas._testing as tm
from pandas.core.arrays.sparse import (
    BlockIndex,
    IntIndex,
    make_sparse_index,
)

TEST_LENGTH = 20

plain_case = {
    "xloc": [0, 7, 15],
    "xlen": [3, 5, 5],
    "yloc": [2, 9, 14],
    "ylen": [2, 3, 5],
    "intersect_loc": [2, 9, 15],
    "intersect_len": [1, 3, 4],
}
delete_blocks = {
    "xloc": [0, 5],
    "xlen": [4, 4],
    "yloc": [1],
    "ylen": [4],
    "intersect_loc": [1],
    "intersect_len": [3],
}
split_blocks = {
    "xloc": [0],
    "xlen": [10],
    "yloc": [0, 5],
    "ylen": [3, 7],
    "intersect_loc": [0, 5],
    "intersect_len": [3, 5],
}
skip_block = {
    "xloc": [10],
    "xlen": [5],
    "yloc": [0, 12],
    "ylen": [5, 3],
    "intersect_loc": [12],
    "intersect_len": [3],
}

no_intersect = {
    "xloc": [0, 10],
    "xlen": [4, 6],
    "yloc": [5, 17],
    "ylen": [4, 2],
    "intersect_loc": [],
    "intersect_len": [],
}


def check_cases(_check_case):
    def _check_case_dict(case):
        _check_case(
            case["xloc"],
            case["xlen"],
            case["yloc"],
            case["ylen"],
            case["intersect_loc"],
            case["intersect_len"],
        )

    _check_case_dict(plain_case)
    _check_case_dict(delete_blocks)
    _check_case_dict(split_blocks)
    _check_case_dict(skip_block)
    _check_case_dict(no_intersect)

    # one or both is empty
    _check_case([0], [5], [], [], [], [])
    _check_case([], [], [], [], [], [])


class TestSparseIndexUnion:
    def test_index_make_union(self):
        def _check_case(xloc, xlen, yloc, ylen, eloc, elen):
            xindex = BlockIndex(TEST_LENGTH, xloc, xlen)
            yindex = BlockIndex(TEST_LENGTH, yloc, ylen)
            bresult = xindex.make_union(yindex)
            assert isinstance(bresult, BlockIndex)
            tm.assert_numpy_array_equal(bresult.blocs, np.array(eloc, dtype=np.int32))
            tm.assert_numpy_array_equal(
                bresult.blengths, np.array(elen, dtype=np.int32)
            )

            ixindex = xindex.to_int_index()
            iyindex = yindex.to_int_index()
            iresult = ixindex.make_union(iyindex)
            assert isinstance(iresult, IntIndex)
            tm.assert_numpy_array_equal(iresult.indices, bresult.to_int_index().indices)

        """
        x: ----
        y:     ----
        r: --------
        """
        xloc = [0]
        xlen = [5]
        yloc = [5]
        ylen = [4]
        eloc = [0]
        elen = [9]
        _check_case(xloc, xlen, yloc, ylen, eloc, elen)
        """
        x: -----     -----
        y:   -----          --
        """
        xloc = [0, 10]
        xlen = [5, 5]
        yloc = [2, 17]
        ylen = [5, 2]
        eloc = [0, 10, 17]
        elen = [7, 5, 2]
        _check_case(xloc, xlen, yloc, ylen, eloc, elen)
        """
        x: ------
        y:    -------
        r: ----------
        """
        xloc = [1]
        xlen = [5]
        yloc = [3]
        ylen = [5]
        eloc = [1]
        elen = [7]
        _check_case(xloc, xlen, yloc, ylen, eloc, elen)
        """
        x: ------  -----
        y:    -------
        r: -------------
        """
        xloc = [2, 10]
        xlen = [4, 4]
        yloc = [4]
        ylen = [8]
        eloc = [2]
        elen = [12]
        _check_case(xloc, xlen, yloc, ylen, eloc, elen)
        """
        x: ---  -----
        y: -------
        r: -------------
        """
        xloc = [0, 5]
        xlen = [3, 5]
        yloc = [0]
        ylen = [7]
        eloc = [0]
        elen = [10]
        _check_case(xloc, xlen, yloc, ylen, eloc, elen)
        """
        x: ------  -----
        y:    -------  ---
        r: -------------
        """
        xloc = [2, 10]
        xlen = [4, 4]
        yloc = [4, 13]
        ylen = [8, 4]
        eloc = [2]
        elen = [15]
        _check_case(xloc, xlen, yloc, ylen, eloc, elen)
        """
        x: ----------------------
        y:   ----  ----   ---
        r: ----------------------
        """
        xloc = [2]
        xlen = [15]
        yloc = [4, 9, 14]
        ylen = [3, 2, 2]
        eloc = [2]
        elen = [15]
        _check_case(xloc, xlen, yloc, ylen, eloc, elen)
        """
        x: ----       ---
        y:       ---       ---
        """
        xloc = [0, 10]
        xlen = [3, 3]
        yloc = [5, 15]
        ylen = [2, 2]
        eloc = [0, 5, 10, 15]
        elen = [3, 2, 3, 2]
        _check_case(xloc, xlen, yloc, ylen, eloc, elen)

    def test_int_index_make_union(self):
        a = IntIndex(5, np.array([0, 3, 4], dtype=np.int32))
        b = IntIndex(5, np.array([0, 2], dtype=np.int32))
        res = a.make_union(b)
        exp = IntIndex(5, np.array([0, 2, 3, 4], np.int32))
        assert res.equals(exp)

        a = IntIndex(5, np.array([], dtype=np.int32))
        b = IntIndex(5, np.array([0, 2], dtype=np.int32))
        res = a.make_union(b)
        exp = IntIndex(5, np.array([0, 2], np.int32))
        assert res.equals(exp)

        a = IntIndex(5, np.array([], dtype=np.int32))
        b = IntIndex(5, np.array([], dtype=np.int32))
        res = a.make_union(b)
        exp = IntIndex(5, np.array([], np.int32))
        assert res.equals(exp)

        a = IntIndex(5, np.array([0, 1, 2, 3, 4], dtype=np.int32))
        b = IntIndex(5, np.array([0, 1, 2, 3, 4], dtype=np.int32))
        res = a.make_union(b)
        exp = IntIndex(5, np.array([0, 1, 2, 3, 4], np.int32))
        assert res.equals(exp)

        a = IntIndex(5, np.array([0, 1], dtype=np.int32))
        b = IntIndex(4, np.array([0, 1], dtype=np.int32))

        msg = "Indices must reference same underlying length"
        with pytest.raises(ValueError, match=msg):
            a.make_union(b)


class TestSparseIndexIntersect:
    @td.skip_if_windows
    def test_intersect(self):
        def _check_correct(a, b, expected):
            result = a.intersect(b)
            assert result.equals(expected)

        def _check_length_exc(a, longer):
            msg = "Indices must reference same underlying length"
            with pytest.raises(Exception, match=msg):
                a.intersect(longer)

        def _check_case(xloc, xlen, yloc, ylen, eloc, elen):
            xindex = BlockIndex(TEST_LENGTH, xloc, xlen)
            yindex = BlockIndex(TEST_LENGTH, yloc, ylen)
            expected = BlockIndex(TEST_LENGTH, eloc, elen)
            longer_index = BlockIndex(TEST_LENGTH + 1, yloc, ylen)

            _check_correct(xindex, yindex, expected)
            _check_correct(
                xindex.to_int_index(), yindex.to_int_index(), expected.to_int_index()
            )

            _check_length_exc(xindex, longer_index)
            _check_length_exc(xindex.to_int_index(), longer_index.to_int_index())

        check_cases(_check_case)

    def test_intersect_empty(self):
        xindex = IntIndex(4, np.array([], dtype=np.int32))
        yindex = IntIndex(4, np.array([2, 3], dtype=np.int32))
        assert xindex.intersect(yindex).equals(xindex)
        assert yindex.intersect(xindex).equals(xindex)

        xindex = xindex.to_block_index()
        yindex = yindex.to_block_index()
        assert xindex.intersect(yindex).equals(xindex)
        assert yindex.intersect(xindex).equals(xindex)

    def test_intersect_identical(self):
        cases = [
            IntIndex(5, np.array([1, 2], dtype=np.int32)),
            IntIndex(5, np.array([0, 2, 4], dtype=np.int32)),
            IntIndex(0, np.array([], dtype=np.int32)),
            IntIndex(5, np.array([], dtype=np.int32)),
        ]

        for case in cases:
            assert case.intersect(case).equals(case)
            case = case.to_block_index()
            assert case.intersect(case).equals(case)


class TestSparseIndexCommon:
    def test_int_internal(self):
        idx = make_sparse_index(4, np.array([2, 3], dtype=np.int32), kind="integer")
        assert isinstance(idx, IntIndex)
        assert idx.npoints == 2
        tm.assert_numpy_array_equal(idx.indices, np.array([2, 3], dtype=np.int32))

        idx = make_sparse_index(4, np.array([], dtype=np.int32), kind="integer")
        assert isinstance(idx, IntIndex)
        assert idx.npoints == 0
        tm.assert_numpy_array_equal(idx.indices, np.array([], dtype=np.int32))

        idx = make_sparse_index(
            4, np.array([0, 1, 2, 3], dtype=np.int32), kind="integer"
        )
        assert isinstance(idx, IntIndex)
        assert idx.npoints == 4
        tm.assert_numpy_array_equal(idx.indices, np.array([0, 1, 2, 3], dtype=np.int32))

    def test_block_internal(self):
        idx = make_sparse_index(4, np.array([2, 3], dtype=np.int32), kind="block")
        assert isinstance(idx, BlockIndex)
        assert idx.npoints == 2
        tm.assert_numpy_array_equal(idx.blocs, np.array([2], dtype=np.int32))
        tm.assert_numpy_array_equal(idx.blengths, np.array([2], dtype=np.int32))

        idx = make_sparse_index(4, np.array([], dtype=np.int32), kind="block")
        assert isinstance(idx, BlockIndex)
        assert idx.npoints == 0
        tm.assert_numpy_array_equal(idx.blocs, np.array([], dtype=np.int32))
        tm.assert_numpy_array_equal(idx.blengths, np.array([], dtype=np.int32))

        idx = make_sparse_index(4, np.array([0, 1, 2, 3], dtype=np.int32), kind="block")
        assert isinstance(idx, BlockIndex)
        assert idx.npoints == 4
        tm.assert_numpy_array_equal(idx.blocs, np.array([0], dtype=np.int32))
        tm.assert_numpy_array_equal(idx.blengths, np.array([4], dtype=np.int32))

        idx = make_sparse_index(4, np.array([0, 2, 3], dtype=np.int32), kind="block")
        assert isinstance(idx, BlockIndex)
        assert idx.npoints == 3
        tm.assert_numpy_array_equal(idx.blocs, np.array([0, 2], dtype=np.int32))
        tm.assert_numpy_array_equal(idx.blengths, np.array([1, 2], dtype=np.int32))

    def test_lookup(self):
        for kind in ["integer", "block"]:
            idx = make_sparse_index(4, np.array([2, 3], dtype=np.int32), kind=kind)
            assert idx.lookup(-1) == -1
            assert idx.lookup(0) == -1
            assert idx.lookup(1) == -1
            assert idx.lookup(2) == 0
            assert idx.lookup(3) == 1
            assert idx.lookup(4) == -1

            idx = make_sparse_index(4, np.array([], dtype=np.int32), kind=kind)

            for i in range(-1, 5):
                assert idx.lookup(i) == -1

            idx = make_sparse_index(
                4, np.array([0, 1, 2, 3], dtype=np.int32), kind=kind
            )
            assert idx.lookup(-1) == -1
            assert idx.lookup(0) == 0
            assert idx.lookup(1) == 1
            assert idx.lookup(2) == 2
            assert idx.lookup(3) == 3
            assert idx.lookup(4) == -1

            idx = make_sparse_index(4, np.array([0, 2, 3], dtype=np.int32), kind=kind)
            assert idx.lookup(-1) == -1
            assert idx.lookup(0) == 0
            assert idx.lookup(1) == -1
            assert idx.lookup(2) == 1
            assert idx.lookup(3) == 2
            assert idx.lookup(4) == -1

    def test_lookup_array(self):
        for kind in ["integer", "block"]:
            idx = make_sparse_index(4, np.array([2, 3], dtype=np.int32), kind=kind)

            res = idx.lookup_array(np.array([-1, 0, 2], dtype=np.int32))
            exp = np.array([-1, -1, 0], dtype=np.int32)
            tm.assert_numpy_array_equal(res, exp)

            res = idx.lookup_array(np.array([4, 2, 1, 3], dtype=np.int32))
            exp = np.array([-1, 0, -1, 1], dtype=np.int32)
            tm.assert_numpy_array_equal(res, exp)

            idx = make_sparse_index(4, np.array([], dtype=np.int32), kind=kind)
            res = idx.lookup_array(np.array([-1, 0, 2, 4], dtype=np.int32))
            exp = np.array([-1, -1, -1, -1], dtype=np.int32)

            idx = make_sparse_index(
                4, np.array([0, 1, 2, 3], dtype=np.int32), kind=kind
            )
            res = idx.lookup_array(np.array([-1, 0, 2], dtype=np.int32))
            exp = np.array([-1, 0, 2], dtype=np.int32)
            tm.assert_numpy_array_equal(res, exp)

            res = idx.lookup_array(np.array([4, 2, 1, 3], dtype=np.int32))
            exp = np.array([-1, 2, 1, 3], dtype=np.int32)
            tm.assert_numpy_array_equal(res, exp)

            idx = make_sparse_index(4, np.array([0, 2, 3], dtype=np.int32), kind=kind)
            res = idx.lookup_array(np.array([2, 1, 3, 0], dtype=np.int32))
            exp = np.array([1, -1, 2, 0], dtype=np.int32)
            tm.assert_numpy_array_equal(res, exp)

            res = idx.lookup_array(np.array([1, 4, 2, 5], dtype=np.int32))
            exp = np.array([-1, -1, 1, -1], dtype=np.int32)
            tm.assert_numpy_array_equal(res, exp)

    def test_lookup_basics(self):
        def _check(index):
            assert index.lookup(0) == -1
            assert index.lookup(5) == 0
            assert index.lookup(7) == 2
            assert index.lookup(8) == -1
            assert index.lookup(9) == -1
            assert index.lookup(10) == -1
            assert index.lookup(11) == -1
            assert index.lookup(12) == 3
            assert index.lookup(17) == 8
            assert index.lookup(18) == -1

        bindex = BlockIndex(20, [5, 12], [3, 6])
        iindex = bindex.to_int_index()

        _check(bindex)
        _check(iindex)

        # corner cases


class TestBlockIndex:
    def test_block_internal(self):
        idx = make_sparse_index(4, np.array([2, 3], dtype=np.int32), kind="block")
        assert isinstance(idx, BlockIndex)
        assert idx.npoints == 2
        tm.assert_numpy_array_equal(idx.blocs, np.array([2], dtype=np.int32))
        tm.assert_numpy_array_equal(idx.blengths, np.array([2], dtype=np.int32))

        idx = make_sparse_index(4, np.array([], dtype=np.int32), kind="block")
        assert isinstance(idx, BlockIndex)
        assert idx.npoints == 0
        tm.assert_numpy_array_equal(idx.blocs, np.array([], dtype=np.int32))
        tm.assert_numpy_array_equal(idx.blengths, np.array([], dtype=np.int32))

        idx = make_sparse_index(4, np.array([0, 1, 2, 3], dtype=np.int32), kind="block")
        assert isinstance(idx, BlockIndex)
        assert idx.npoints == 4
        tm.assert_numpy_array_equal(idx.blocs, np.array([0], dtype=np.int32))
        tm.assert_numpy_array_equal(idx.blengths, np.array([4], dtype=np.int32))

        idx = make_sparse_index(4, np.array([0, 2, 3], dtype=np.int32), kind="block")
        assert isinstance(idx, BlockIndex)
        assert idx.npoints == 3
        tm.assert_numpy_array_equal(idx.blocs, np.array([0, 2], dtype=np.int32))
        tm.assert_numpy_array_equal(idx.blengths, np.array([1, 2], dtype=np.int32))

    def test_make_block_boundary(self):
        for i in [5, 10, 100, 101]:
            idx = make_sparse_index(i, np.arange(0, i, 2, dtype=np.int32), kind="block")

            exp = np.arange(0, i, 2, dtype=np.int32)
            tm.assert_numpy_array_equal(idx.blocs, exp)
            tm.assert_numpy_array_equal(idx.blengths, np.ones(len(exp), dtype=np.int32))

    def test_equals(self):
        index = BlockIndex(10, [0, 4], [2, 5])

        assert index.equals(index)
        assert not index.equals(BlockIndex(10, [0, 4], [2, 6]))

    def test_check_integrity(self):
        locs = []
        lengths = []

        # 0-length OK
        # TODO: index variables are not used...is that right?
        index = BlockIndex(0, locs, lengths)

        # also OK even though empty
        index = BlockIndex(1, locs, lengths)  # noqa

        msg = "Block 0 extends beyond end"
        with pytest.raises(ValueError, match=msg):
            BlockIndex(10, [5], [10])

        msg = "Block 0 overlaps"
        with pytest.raises(ValueError, match=msg):
            BlockIndex(10, [2, 5], [5, 3])

    def test_to_int_index(self):
        locs = [0, 10]
        lengths = [4, 6]
        exp_inds = [0, 1, 2, 3, 10, 11, 12, 13, 14, 15]

        block = BlockIndex(20, locs, lengths)
        dense = block.to_int_index()

        tm.assert_numpy_array_equal(dense.indices, np.array(exp_inds, dtype=np.int32))

    def test_to_block_index(self):
        index = BlockIndex(10, [0, 5], [4, 5])
        assert index.to_block_index() is index


class TestIntIndex:
    def test_check_integrity(self):

        # Too many indices than specified in self.length
        msg = "Too many indices"

        with pytest.raises(ValueError, match=msg):
            IntIndex(length=1, indices=[1, 2, 3])

        # No index can be negative.
        msg = "No index can be less than zero"

        with pytest.raises(ValueError, match=msg):
            IntIndex(length=5, indices=[1, -2, 3])

        # No index can be negative.
        msg = "No index can be less than zero"

        with pytest.raises(ValueError, match=msg):
            IntIndex(length=5, indices=[1, -2, 3])

        # All indices must be less than the length.
        msg = "All indices must be less than the length"

        with pytest.raises(ValueError, match=msg):
            IntIndex(length=5, indices=[1, 2, 5])

        with pytest.raises(ValueError, match=msg):
            IntIndex(length=5, indices=[1, 2, 6])

        # Indices must be strictly ascending.
        msg = "Indices must be strictly increasing"

        with pytest.raises(ValueError, match=msg):
            IntIndex(length=5, indices=[1, 3, 2])

        with pytest.raises(ValueError, match=msg):
            IntIndex(length=5, indices=[1, 3, 3])

    def test_int_internal(self):
        idx = make_sparse_index(4, np.array([2, 3], dtype=np.int32), kind="integer")
        assert isinstance(idx, IntIndex)
        assert idx.npoints == 2
        tm.assert_numpy_array_equal(idx.indices, np.array([2, 3], dtype=np.int32))

        idx = make_sparse_index(4, np.array([], dtype=np.int32), kind="integer")
        assert isinstance(idx, IntIndex)
        assert idx.npoints == 0
        tm.assert_numpy_array_equal(idx.indices, np.array([], dtype=np.int32))

        idx = make_sparse_index(
            4, np.array([0, 1, 2, 3], dtype=np.int32), kind="integer"
        )
        assert isinstance(idx, IntIndex)
        assert idx.npoints == 4
        tm.assert_numpy_array_equal(idx.indices, np.array([0, 1, 2, 3], dtype=np.int32))

    def test_equals(self):
        index = IntIndex(10, [0, 1, 2, 3, 4])
        assert index.equals(index)
        assert not index.equals(IntIndex(10, [0, 1, 2, 3]))

    def test_to_block_index(self):
        def _check_case(xloc, xlen, yloc, ylen, eloc, elen):
            xindex = BlockIndex(TEST_LENGTH, xloc, xlen)
            yindex = BlockIndex(TEST_LENGTH, yloc, ylen)

            # see if survive the round trip
            xbindex = xindex.to_int_index().to_block_index()
            ybindex = yindex.to_int_index().to_block_index()
            assert isinstance(xbindex, BlockIndex)
            assert xbindex.equals(xindex)
            assert ybindex.equals(yindex)

        check_cases(_check_case)

    def test_to_int_index(self):
        index = IntIndex(10, [2, 3, 4, 5, 6])
        assert index.to_int_index() is index


class TestSparseOperators:
    def _op_tests(self, sparse_op, python_op):
        def _check_case(xloc, xlen, yloc, ylen, eloc, elen):
            xindex = BlockIndex(TEST_LENGTH, xloc, xlen)
            yindex = BlockIndex(TEST_LENGTH, yloc, ylen)

            xdindex = xindex.to_int_index()
            ydindex = yindex.to_int_index()

            x = np.arange(xindex.npoints) * 10.0 + 1
            y = np.arange(yindex.npoints) * 100.0 + 1

            xfill = 0
            yfill = 2

            result_block_vals, rb_index, bfill = sparse_op(
                x, xindex, xfill, y, yindex, yfill
            )
            result_int_vals, ri_index, ifill = sparse_op(
                x, xdindex, xfill, y, ydindex, yfill
            )

            assert rb_index.to_int_index().equals(ri_index)
            tm.assert_numpy_array_equal(result_block_vals, result_int_vals)
            assert bfill == ifill

            # check versus Series...
            xseries = Series(x, xdindex.indices)
            xseries = xseries.reindex(np.arange(TEST_LENGTH)).fillna(xfill)

            yseries = Series(y, ydindex.indices)
            yseries = yseries.reindex(np.arange(TEST_LENGTH)).fillna(yfill)

            series_result = python_op(xseries, yseries)
            series_result = series_result.reindex(ri_index.indices)

            tm.assert_numpy_array_equal(result_block_vals, series_result.values)
            tm.assert_numpy_array_equal(result_int_vals, series_result.values)

        check_cases(_check_case)

    @pytest.mark.parametrize("opname", ["add", "sub", "mul", "truediv", "floordiv"])
    def test_op(self, opname):
        sparse_op = getattr(splib, f"sparse_{opname}_float64")
        python_op = getattr(operator, opname)
        self._op_tests(sparse_op, python_op)
