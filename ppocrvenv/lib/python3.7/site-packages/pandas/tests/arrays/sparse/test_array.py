import operator
import re
import warnings

import numpy as np
import pytest

from pandas._libs.sparse import IntIndex
import pandas.util._test_decorators as td

import pandas as pd
from pandas import isna
import pandas._testing as tm
from pandas.core.arrays.sparse import (
    SparseArray,
    SparseDtype,
)


class TestSparseArray:
    def setup_method(self, method):
        self.arr_data = np.array([np.nan, np.nan, 1, 2, 3, np.nan, 4, 5, np.nan, 6])
        self.arr = SparseArray(self.arr_data)
        self.zarr = SparseArray([0, 0, 1, 2, 3, 0, 4, 5, 0, 6], fill_value=0)

    def test_constructor_dtype(self):
        arr = SparseArray([np.nan, 1, 2, np.nan])
        assert arr.dtype == SparseDtype(np.float64, np.nan)
        assert arr.dtype.subtype == np.float64
        assert np.isnan(arr.fill_value)

        arr = SparseArray([np.nan, 1, 2, np.nan], fill_value=0)
        assert arr.dtype == SparseDtype(np.float64, 0)
        assert arr.fill_value == 0

        arr = SparseArray([0, 1, 2, 4], dtype=np.float64)
        assert arr.dtype == SparseDtype(np.float64, np.nan)
        assert np.isnan(arr.fill_value)

        arr = SparseArray([0, 1, 2, 4], dtype=np.int64)
        assert arr.dtype == SparseDtype(np.int64, 0)
        assert arr.fill_value == 0

        arr = SparseArray([0, 1, 2, 4], fill_value=0, dtype=np.int64)
        assert arr.dtype == SparseDtype(np.int64, 0)
        assert arr.fill_value == 0

        arr = SparseArray([0, 1, 2, 4], dtype=None)
        assert arr.dtype == SparseDtype(np.int64, 0)
        assert arr.fill_value == 0

        arr = SparseArray([0, 1, 2, 4], fill_value=0, dtype=None)
        assert arr.dtype == SparseDtype(np.int64, 0)
        assert arr.fill_value == 0

    def test_constructor_dtype_str(self):
        result = SparseArray([1, 2, 3], dtype="int")
        expected = SparseArray([1, 2, 3], dtype=int)
        tm.assert_sp_array_equal(result, expected)

    def test_constructor_sparse_dtype(self):
        result = SparseArray([1, 0, 0, 1], dtype=SparseDtype("int64", -1))
        expected = SparseArray([1, 0, 0, 1], fill_value=-1, dtype=np.int64)
        tm.assert_sp_array_equal(result, expected)
        assert result.sp_values.dtype == np.dtype("int64")

    def test_constructor_sparse_dtype_str(self):
        result = SparseArray([1, 0, 0, 1], dtype="Sparse[int32]")
        expected = SparseArray([1, 0, 0, 1], dtype=np.int32)
        tm.assert_sp_array_equal(result, expected)
        assert result.sp_values.dtype == np.dtype("int32")

    def test_constructor_object_dtype(self):
        # GH 11856
        arr = SparseArray(["A", "A", np.nan, "B"], dtype=object)
        assert arr.dtype == SparseDtype(object)
        assert np.isnan(arr.fill_value)

        arr = SparseArray(["A", "A", np.nan, "B"], dtype=object, fill_value="A")
        assert arr.dtype == SparseDtype(object, "A")
        assert arr.fill_value == "A"

        # GH 17574
        data = [False, 0, 100.0, 0.0]
        arr = SparseArray(data, dtype=object, fill_value=False)
        assert arr.dtype == SparseDtype(object, False)
        assert arr.fill_value is False
        arr_expected = np.array(data, dtype=object)
        it = (type(x) == type(y) and x == y for x, y in zip(arr, arr_expected))
        assert np.fromiter(it, dtype=np.bool_).all()

    @pytest.mark.parametrize("dtype", [SparseDtype(int, 0), int])
    def test_constructor_na_dtype(self, dtype):
        with pytest.raises(ValueError, match="Cannot convert"):
            SparseArray([0, 1, np.nan], dtype=dtype)

    def test_constructor_warns_when_losing_timezone(self):
        # GH#32501 warn when losing timezone information
        dti = pd.date_range("2016-01-01", periods=3, tz="US/Pacific")

        expected = SparseArray(np.asarray(dti, dtype="datetime64[ns]"))

        with tm.assert_produces_warning(UserWarning):
            result = SparseArray(dti)

        tm.assert_sp_array_equal(result, expected)

        with tm.assert_produces_warning(UserWarning):
            result = SparseArray(pd.Series(dti))

        tm.assert_sp_array_equal(result, expected)

    def test_constructor_spindex_dtype(self):
        arr = SparseArray(data=[1, 2], sparse_index=IntIndex(4, [1, 2]))
        # XXX: Behavior change: specifying SparseIndex no longer changes the
        # fill_value
        expected = SparseArray([0, 1, 2, 0], kind="integer")
        tm.assert_sp_array_equal(arr, expected)
        assert arr.dtype == SparseDtype(np.int64)
        assert arr.fill_value == 0

        arr = SparseArray(
            data=[1, 2, 3],
            sparse_index=IntIndex(4, [1, 2, 3]),
            dtype=np.int64,
            fill_value=0,
        )
        exp = SparseArray([0, 1, 2, 3], dtype=np.int64, fill_value=0)
        tm.assert_sp_array_equal(arr, exp)
        assert arr.dtype == SparseDtype(np.int64)
        assert arr.fill_value == 0

        arr = SparseArray(
            data=[1, 2], sparse_index=IntIndex(4, [1, 2]), fill_value=0, dtype=np.int64
        )
        exp = SparseArray([0, 1, 2, 0], fill_value=0, dtype=np.int64)
        tm.assert_sp_array_equal(arr, exp)
        assert arr.dtype == SparseDtype(np.int64)
        assert arr.fill_value == 0

        arr = SparseArray(
            data=[1, 2, 3],
            sparse_index=IntIndex(4, [1, 2, 3]),
            dtype=None,
            fill_value=0,
        )
        exp = SparseArray([0, 1, 2, 3], dtype=None)
        tm.assert_sp_array_equal(arr, exp)
        assert arr.dtype == SparseDtype(np.int64)
        assert arr.fill_value == 0

    @pytest.mark.parametrize("sparse_index", [None, IntIndex(1, [0])])
    def test_constructor_spindex_dtype_scalar(self, sparse_index):
        # scalar input
        arr = SparseArray(data=1, sparse_index=sparse_index, dtype=None)
        exp = SparseArray([1], dtype=None)
        tm.assert_sp_array_equal(arr, exp)
        assert arr.dtype == SparseDtype(np.int64)
        assert arr.fill_value == 0

        arr = SparseArray(data=1, sparse_index=IntIndex(1, [0]), dtype=None)
        exp = SparseArray([1], dtype=None)
        tm.assert_sp_array_equal(arr, exp)
        assert arr.dtype == SparseDtype(np.int64)
        assert arr.fill_value == 0

    def test_constructor_spindex_dtype_scalar_broadcasts(self):
        arr = SparseArray(
            data=[1, 2], sparse_index=IntIndex(4, [1, 2]), fill_value=0, dtype=None
        )
        exp = SparseArray([0, 1, 2, 0], fill_value=0, dtype=None)
        tm.assert_sp_array_equal(arr, exp)
        assert arr.dtype == SparseDtype(np.int64)
        assert arr.fill_value == 0

    @pytest.mark.parametrize(
        "data, fill_value",
        [
            (np.array([1, 2]), 0),
            (np.array([1.0, 2.0]), np.nan),
            ([True, False], False),
            ([pd.Timestamp("2017-01-01")], pd.NaT),
        ],
    )
    def test_constructor_inferred_fill_value(self, data, fill_value):
        result = SparseArray(data).fill_value

        if isna(fill_value):
            assert isna(result)
        else:
            assert result == fill_value

    @pytest.mark.parametrize("format", ["coo", "csc", "csr"])
    @pytest.mark.parametrize("size", [0, 10])
    @td.skip_if_no_scipy
    def test_from_spmatrix(self, size, format):
        import scipy.sparse

        mat = scipy.sparse.random(size, 1, density=0.5, format=format)
        result = SparseArray.from_spmatrix(mat)

        result = np.asarray(result)
        expected = mat.toarray().ravel()
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize("format", ["coo", "csc", "csr"])
    @td.skip_if_no_scipy
    def test_from_spmatrix_including_explicit_zero(self, format):
        import scipy.sparse

        mat = scipy.sparse.random(10, 1, density=0.5, format=format)
        mat.data[0] = 0
        result = SparseArray.from_spmatrix(mat)

        result = np.asarray(result)
        expected = mat.toarray().ravel()
        tm.assert_numpy_array_equal(result, expected)

    @td.skip_if_no_scipy
    def test_from_spmatrix_raises(self):
        import scipy.sparse

        mat = scipy.sparse.eye(5, 4, format="csc")

        with pytest.raises(ValueError, match="not '4'"):
            SparseArray.from_spmatrix(mat)

    @pytest.mark.parametrize(
        "scalar,dtype",
        [
            (False, SparseDtype(bool, False)),
            (0.0, SparseDtype("float64", 0)),
            (1, SparseDtype("int64", 1)),
            ("z", SparseDtype("object", "z")),
        ],
    )
    def test_scalar_with_index_infer_dtype(self, scalar, dtype):
        # GH 19163
        arr = SparseArray(scalar, index=[1, 2, 3], fill_value=scalar)
        exp = SparseArray([scalar, scalar, scalar], fill_value=scalar)

        tm.assert_sp_array_equal(arr, exp)

        assert arr.dtype == dtype
        assert exp.dtype == dtype

    def test_get_item(self):

        assert np.isnan(self.arr[1])
        assert self.arr[2] == 1
        assert self.arr[7] == 5

        assert self.zarr[0] == 0
        assert self.zarr[2] == 1
        assert self.zarr[7] == 5

        errmsg = re.compile("bounds")

        with pytest.raises(IndexError, match=errmsg):
            self.arr[11]

        with pytest.raises(IndexError, match=errmsg):
            self.arr[-11]

        assert self.arr[-1] == self.arr[len(self.arr) - 1]

    def test_take_scalar_raises(self):
        msg = "'indices' must be an array, not a scalar '2'."
        with pytest.raises(ValueError, match=msg):
            self.arr.take(2)

    def test_take(self):
        exp = SparseArray(np.take(self.arr_data, [2, 3]))
        tm.assert_sp_array_equal(self.arr.take([2, 3]), exp)

        exp = SparseArray(np.take(self.arr_data, [0, 1, 2]))
        tm.assert_sp_array_equal(self.arr.take([0, 1, 2]), exp)

    def test_take_all_empty(self):
        a = pd.array([0, 0], dtype=SparseDtype("int64"))
        result = a.take([0, 1], allow_fill=True, fill_value=np.nan)
        tm.assert_sp_array_equal(a, result)

    def test_take_fill_value(self):
        data = np.array([1, np.nan, 0, 3, 0])
        sparse = SparseArray(data, fill_value=0)

        exp = SparseArray(np.take(data, [0]), fill_value=0)
        tm.assert_sp_array_equal(sparse.take([0]), exp)

        exp = SparseArray(np.take(data, [1, 3, 4]), fill_value=0)
        tm.assert_sp_array_equal(sparse.take([1, 3, 4]), exp)

    def test_take_negative(self):
        exp = SparseArray(np.take(self.arr_data, [-1]))
        tm.assert_sp_array_equal(self.arr.take([-1]), exp)

        exp = SparseArray(np.take(self.arr_data, [-4, -3, -2]))
        tm.assert_sp_array_equal(self.arr.take([-4, -3, -2]), exp)

    @pytest.mark.parametrize("fill_value", [0, None, np.nan])
    def test_shift_fill_value(self, fill_value):
        # GH #24128
        sparse = SparseArray(np.array([1, 0, 0, 3, 0]), fill_value=8.0)
        res = sparse.shift(1, fill_value=fill_value)
        if isna(fill_value):
            fill_value = res.dtype.na_value
        exp = SparseArray(np.array([fill_value, 1, 0, 0, 3]), fill_value=8.0)
        tm.assert_sp_array_equal(res, exp)

    def test_bad_take(self):
        with pytest.raises(IndexError, match="bounds"):
            self.arr.take([11])

    def test_take_filling(self):
        # similar tests as GH 12631
        sparse = SparseArray([np.nan, np.nan, 1, np.nan, 4])
        result = sparse.take(np.array([1, 0, -1]))
        expected = SparseArray([np.nan, np.nan, 4])
        tm.assert_sp_array_equal(result, expected)

        # XXX: test change: fill_value=True -> allow_fill=True
        result = sparse.take(np.array([1, 0, -1]), allow_fill=True)
        expected = SparseArray([np.nan, np.nan, np.nan])
        tm.assert_sp_array_equal(result, expected)

        # allow_fill=False
        result = sparse.take(np.array([1, 0, -1]), allow_fill=False, fill_value=True)
        expected = SparseArray([np.nan, np.nan, 4])
        tm.assert_sp_array_equal(result, expected)

        msg = "Invalid value in 'indices'"
        with pytest.raises(ValueError, match=msg):
            sparse.take(np.array([1, 0, -2]), allow_fill=True)

        with pytest.raises(ValueError, match=msg):
            sparse.take(np.array([1, 0, -5]), allow_fill=True)

        msg = "out of bounds value in 'indices'"
        with pytest.raises(IndexError, match=msg):
            sparse.take(np.array([1, -6]))
        with pytest.raises(IndexError, match=msg):
            sparse.take(np.array([1, 5]))
        with pytest.raises(IndexError, match=msg):
            sparse.take(np.array([1, 5]), allow_fill=True)

    def test_take_filling_fill_value(self):
        # same tests as GH 12631
        sparse = SparseArray([np.nan, 0, 1, 0, 4], fill_value=0)
        result = sparse.take(np.array([1, 0, -1]))
        expected = SparseArray([0, np.nan, 4], fill_value=0)
        tm.assert_sp_array_equal(result, expected)

        # fill_value
        result = sparse.take(np.array([1, 0, -1]), allow_fill=True)
        # XXX: behavior change.
        # the old way of filling self.fill_value doesn't follow EA rules.
        # It's supposed to be self.dtype.na_value (nan in this case)
        expected = SparseArray([0, np.nan, np.nan], fill_value=0)
        tm.assert_sp_array_equal(result, expected)

        # allow_fill=False
        result = sparse.take(np.array([1, 0, -1]), allow_fill=False, fill_value=True)
        expected = SparseArray([0, np.nan, 4], fill_value=0)
        tm.assert_sp_array_equal(result, expected)

        msg = "Invalid value in 'indices'."
        with pytest.raises(ValueError, match=msg):
            sparse.take(np.array([1, 0, -2]), allow_fill=True)
        with pytest.raises(ValueError, match=msg):
            sparse.take(np.array([1, 0, -5]), allow_fill=True)

        msg = "out of bounds value in 'indices'"
        with pytest.raises(IndexError, match=msg):
            sparse.take(np.array([1, -6]))
        with pytest.raises(IndexError, match=msg):
            sparse.take(np.array([1, 5]))
        with pytest.raises(IndexError, match=msg):
            sparse.take(np.array([1, 5]), fill_value=True)

    def test_take_filling_all_nan(self):
        sparse = SparseArray([np.nan, np.nan, np.nan, np.nan, np.nan])
        # XXX: did the default kind from take change?
        result = sparse.take(np.array([1, 0, -1]))
        expected = SparseArray([np.nan, np.nan, np.nan], kind="block")
        tm.assert_sp_array_equal(result, expected)

        result = sparse.take(np.array([1, 0, -1]), fill_value=True)
        expected = SparseArray([np.nan, np.nan, np.nan], kind="block")
        tm.assert_sp_array_equal(result, expected)

        msg = "out of bounds value in 'indices'"
        with pytest.raises(IndexError, match=msg):
            sparse.take(np.array([1, -6]))
        with pytest.raises(IndexError, match=msg):
            sparse.take(np.array([1, 5]))
        with pytest.raises(IndexError, match=msg):
            sparse.take(np.array([1, 5]), fill_value=True)

    def test_set_item(self):
        def setitem():
            self.arr[5] = 3

        def setslice():
            self.arr[1:5] = 2

        with pytest.raises(TypeError, match="assignment via setitem"):
            setitem()

        with pytest.raises(TypeError, match="assignment via setitem"):
            setslice()

    def test_constructor_from_too_large_array(self):
        with pytest.raises(TypeError, match="expected dimension <= 1 data"):
            SparseArray(np.arange(10).reshape((2, 5)))

    def test_constructor_from_sparse(self):
        res = SparseArray(self.zarr)
        assert res.fill_value == 0
        tm.assert_almost_equal(res.sp_values, self.zarr.sp_values)

    def test_constructor_copy(self):
        cp = SparseArray(self.arr, copy=True)
        cp.sp_values[:3] = 0
        assert not (self.arr.sp_values[:3] == 0).any()

        not_copy = SparseArray(self.arr)
        not_copy.sp_values[:3] = 0
        assert (self.arr.sp_values[:3] == 0).all()

    def test_constructor_bool(self):
        # GH 10648
        data = np.array([False, False, True, True, False, False])
        arr = SparseArray(data, fill_value=False, dtype=bool)

        assert arr.dtype == SparseDtype(bool)
        tm.assert_numpy_array_equal(arr.sp_values, np.array([True, True]))
        # Behavior change: np.asarray densifies.
        # tm.assert_numpy_array_equal(arr.sp_values, np.asarray(arr))
        tm.assert_numpy_array_equal(arr.sp_index.indices, np.array([2, 3], np.int32))

        dense = arr.to_dense()
        assert dense.dtype == bool
        tm.assert_numpy_array_equal(dense, data)

    def test_constructor_bool_fill_value(self):
        arr = SparseArray([True, False, True], dtype=None)
        assert arr.dtype == SparseDtype(np.bool_)
        assert not arr.fill_value

        arr = SparseArray([True, False, True], dtype=np.bool_)
        assert arr.dtype == SparseDtype(np.bool_)
        assert not arr.fill_value

        arr = SparseArray([True, False, True], dtype=np.bool_, fill_value=True)
        assert arr.dtype == SparseDtype(np.bool_, True)
        assert arr.fill_value

    def test_constructor_float32(self):
        # GH 10648
        data = np.array([1.0, np.nan, 3], dtype=np.float32)
        arr = SparseArray(data, dtype=np.float32)

        assert arr.dtype == SparseDtype(np.float32)
        tm.assert_numpy_array_equal(arr.sp_values, np.array([1, 3], dtype=np.float32))
        # Behavior change: np.asarray densifies.
        # tm.assert_numpy_array_equal(arr.sp_values, np.asarray(arr))
        tm.assert_numpy_array_equal(
            arr.sp_index.indices, np.array([0, 2], dtype=np.int32)
        )

        dense = arr.to_dense()
        assert dense.dtype == np.float32
        tm.assert_numpy_array_equal(dense, data)

    def test_astype(self):
        # float -> float
        arr = SparseArray([None, None, 0, 2])
        result = arr.astype("Sparse[float32]")
        expected = SparseArray([None, None, 0, 2], dtype=np.dtype("float32"))
        tm.assert_sp_array_equal(result, expected)

        dtype = SparseDtype("float64", fill_value=0)
        result = arr.astype(dtype)
        expected = SparseArray._simple_new(
            np.array([0.0, 2.0], dtype=dtype.subtype), IntIndex(4, [2, 3]), dtype
        )
        tm.assert_sp_array_equal(result, expected)

        dtype = SparseDtype("int64", 0)
        result = arr.astype(dtype)
        expected = SparseArray._simple_new(
            np.array([0, 2], dtype=np.int64), IntIndex(4, [2, 3]), dtype
        )
        tm.assert_sp_array_equal(result, expected)

        arr = SparseArray([0, np.nan, 0, 1], fill_value=0)
        with pytest.raises(ValueError, match="NA"):
            arr.astype("Sparse[i8]")

    def test_astype_bool(self):
        a = SparseArray([1, 0, 0, 1], dtype=SparseDtype(int, 0))
        result = a.astype(bool)
        expected = SparseArray([True, 0, 0, True], dtype=SparseDtype(bool, 0))
        tm.assert_sp_array_equal(result, expected)

        # update fill value
        result = a.astype(SparseDtype(bool, False))
        expected = SparseArray(
            [True, False, False, True], dtype=SparseDtype(bool, False)
        )
        tm.assert_sp_array_equal(result, expected)

    def test_astype_all(self, any_real_dtype):
        vals = np.array([1, 2, 3])
        arr = SparseArray(vals, fill_value=1)
        typ = np.dtype(any_real_dtype)
        res = arr.astype(typ)
        assert res.dtype == SparseDtype(typ, 1)
        assert res.sp_values.dtype == typ

        tm.assert_numpy_array_equal(np.asarray(res.to_dense()), vals.astype(typ))

    @pytest.mark.parametrize(
        "arr, dtype, expected",
        [
            (
                SparseArray([0, 1]),
                "float",
                SparseArray([0.0, 1.0], dtype=SparseDtype(float, 0.0)),
            ),
            (SparseArray([0, 1]), bool, SparseArray([False, True])),
            (
                SparseArray([0, 1], fill_value=1),
                bool,
                SparseArray([False, True], dtype=SparseDtype(bool, True)),
            ),
            pytest.param(
                SparseArray([0, 1]),
                "datetime64[ns]",
                SparseArray(
                    np.array([0, 1], dtype="datetime64[ns]"),
                    dtype=SparseDtype("datetime64[ns]", pd.Timestamp("1970")),
                ),
                marks=[pytest.mark.xfail(reason="NumPy-7619")],
            ),
            (
                SparseArray([0, 1, 10]),
                str,
                SparseArray(["0", "1", "10"], dtype=SparseDtype(str, "0")),
            ),
            (SparseArray(["10", "20"]), float, SparseArray([10.0, 20.0])),
            (
                SparseArray([0, 1, 0]),
                object,
                SparseArray([0, 1, 0], dtype=SparseDtype(object, 0)),
            ),
        ],
    )
    def test_astype_more(self, arr, dtype, expected):
        result = arr.astype(dtype)
        tm.assert_sp_array_equal(result, expected)

    def test_astype_nan_raises(self):
        arr = SparseArray([1.0, np.nan])
        with pytest.raises(ValueError, match="Cannot convert non-finite"):
            arr.astype(int)

    def test_astype_copy_false(self):
        # GH#34456 bug caused by using .view instead of .astype in astype_nansafe
        arr = SparseArray([1, 2, 3])

        result = arr.astype(float, copy=False)
        expected = SparseArray([1.0, 2.0, 3.0], fill_value=0.0)
        tm.assert_sp_array_equal(result, expected)

    def test_set_fill_value(self):
        arr = SparseArray([1.0, np.nan, 2.0], fill_value=np.nan)
        arr.fill_value = 2
        assert arr.fill_value == 2

        arr = SparseArray([1, 0, 2], fill_value=0, dtype=np.int64)
        arr.fill_value = 2
        assert arr.fill_value == 2

        # XXX: this seems fine? You can construct an integer
        # sparsearray with NaN fill value, why not update one?
        # coerces to int
        # msg = "unable to set fill_value 3\\.1 to int64 dtype"
        # with pytest.raises(ValueError, match=msg):
        arr.fill_value = 3.1
        assert arr.fill_value == 3.1

        # msg = "unable to set fill_value nan to int64 dtype"
        # with pytest.raises(ValueError, match=msg):
        arr.fill_value = np.nan
        assert np.isnan(arr.fill_value)

        arr = SparseArray([True, False, True], fill_value=False, dtype=np.bool_)
        arr.fill_value = True
        assert arr.fill_value

        # coerces to bool
        # msg = "unable to set fill_value 0 to bool dtype"
        # with pytest.raises(ValueError, match=msg):
        arr.fill_value = 0
        assert arr.fill_value == 0

        # msg = "unable to set fill_value nan to bool dtype"
        # with pytest.raises(ValueError, match=msg):
        arr.fill_value = np.nan
        assert np.isnan(arr.fill_value)

    @pytest.mark.parametrize("val", [[1, 2, 3], np.array([1, 2]), (1, 2, 3)])
    def test_set_fill_invalid_non_scalar(self, val):
        arr = SparseArray([True, False, True], fill_value=False, dtype=np.bool_)
        msg = "fill_value must be a scalar"

        with pytest.raises(ValueError, match=msg):
            arr.fill_value = val

    def test_copy(self):
        arr2 = self.arr.copy()
        assert arr2.sp_values is not self.arr.sp_values
        assert arr2.sp_index is self.arr.sp_index

    def test_values_asarray(self):
        tm.assert_almost_equal(self.arr.to_dense(), self.arr_data)

    @pytest.mark.parametrize(
        "data,shape,dtype",
        [
            ([0, 0, 0, 0, 0], (5,), None),
            ([], (0,), None),
            ([0], (1,), None),
            (["A", "A", np.nan, "B"], (4,), object),
        ],
    )
    def test_shape(self, data, shape, dtype):
        # GH 21126
        out = SparseArray(data, dtype=dtype)
        assert out.shape == shape

    @pytest.mark.parametrize(
        "vals",
        [
            [np.nan, np.nan, np.nan, np.nan, np.nan],
            [1, np.nan, np.nan, 3, np.nan],
            [1, np.nan, 0, 3, 0],
        ],
    )
    @pytest.mark.parametrize("fill_value", [None, 0])
    def test_dense_repr(self, vals, fill_value):
        vals = np.array(vals)
        arr = SparseArray(vals, fill_value=fill_value)

        res = arr.to_dense()
        tm.assert_numpy_array_equal(res, vals)

        res2 = arr._internal_get_values()

        tm.assert_numpy_array_equal(res2, vals)

    def test_getitem(self):
        def _checkit(i):
            tm.assert_almost_equal(self.arr[i], self.arr.to_dense()[i])

        for i in range(len(self.arr)):
            _checkit(i)
            _checkit(-i)

    def test_getitem_arraylike_mask(self):
        arr = SparseArray([0, 1, 2])
        result = arr[[True, False, True]]
        expected = SparseArray([0, 2])
        tm.assert_sp_array_equal(result, expected)

    def test_getslice(self):
        result = self.arr[:-3]
        exp = SparseArray(self.arr.to_dense()[:-3])
        tm.assert_sp_array_equal(result, exp)

        result = self.arr[-4:]
        exp = SparseArray(self.arr.to_dense()[-4:])
        tm.assert_sp_array_equal(result, exp)

        # two corner cases from Series
        result = self.arr[-12:]
        exp = SparseArray(self.arr)
        tm.assert_sp_array_equal(result, exp)

        result = self.arr[:-12]
        exp = SparseArray(self.arr.to_dense()[:0])
        tm.assert_sp_array_equal(result, exp)

    def test_getslice_tuple(self):
        dense = np.array([np.nan, 0, 3, 4, 0, 5, np.nan, np.nan, 0])

        sparse = SparseArray(dense)
        res = sparse[(slice(4, None),)]
        exp = SparseArray(dense[4:])
        tm.assert_sp_array_equal(res, exp)

        sparse = SparseArray(dense, fill_value=0)
        res = sparse[(slice(4, None),)]
        exp = SparseArray(dense[4:], fill_value=0)
        tm.assert_sp_array_equal(res, exp)

        msg = "too many indices for array"
        with pytest.raises(IndexError, match=msg):
            sparse[4:, :]

        with pytest.raises(IndexError, match=msg):
            # check numpy compat
            dense[4:, :]

    def test_boolean_slice_empty(self):
        arr = SparseArray([0, 1, 2])
        res = arr[[False, False, False]]
        assert res.dtype == arr.dtype

    @pytest.mark.parametrize("op", ["add", "sub", "mul", "truediv", "floordiv", "pow"])
    def test_binary_operators(self, op):
        op = getattr(operator, op)
        data1 = np.random.randn(20)
        data2 = np.random.randn(20)

        data1[::2] = np.nan
        data2[::3] = np.nan

        arr1 = SparseArray(data1)
        arr2 = SparseArray(data2)

        data1[::2] = 3
        data2[::3] = 3
        farr1 = SparseArray(data1, fill_value=3)
        farr2 = SparseArray(data2, fill_value=3)

        def _check_op(op, first, second):
            res = op(first, second)
            exp = SparseArray(
                op(first.to_dense(), second.to_dense()), fill_value=first.fill_value
            )
            assert isinstance(res, SparseArray)
            tm.assert_almost_equal(res.to_dense(), exp.to_dense())

            res2 = op(first, second.to_dense())
            assert isinstance(res2, SparseArray)
            tm.assert_sp_array_equal(res, res2)

            res3 = op(first.to_dense(), second)
            assert isinstance(res3, SparseArray)
            tm.assert_sp_array_equal(res, res3)

            res4 = op(first, 4)
            assert isinstance(res4, SparseArray)

            # Ignore this if the actual op raises (e.g. pow).
            try:
                exp = op(first.to_dense(), 4)
                exp_fv = op(first.fill_value, 4)
            except ValueError:
                pass
            else:
                tm.assert_almost_equal(res4.fill_value, exp_fv)
                tm.assert_almost_equal(res4.to_dense(), exp)

        with np.errstate(all="ignore"):
            for first_arr, second_arr in [(arr1, arr2), (farr1, farr2)]:
                _check_op(op, first_arr, second_arr)

    def test_pickle(self):
        def _check_roundtrip(obj):
            unpickled = tm.round_trip_pickle(obj)
            tm.assert_sp_array_equal(unpickled, obj)

        _check_roundtrip(self.arr)
        _check_roundtrip(self.zarr)

    def test_generator_warnings(self):
        sp_arr = SparseArray([1, 2, 3])
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings(action="always", category=DeprecationWarning)
            warnings.filterwarnings(action="always", category=PendingDeprecationWarning)
            for _ in sp_arr:
                pass
            assert len(w) == 0

    def test_fillna(self):
        s = SparseArray([1, np.nan, np.nan, 3, np.nan])
        res = s.fillna(-1)
        exp = SparseArray([1, -1, -1, 3, -1], fill_value=-1, dtype=np.float64)
        tm.assert_sp_array_equal(res, exp)

        s = SparseArray([1, np.nan, np.nan, 3, np.nan], fill_value=0)
        res = s.fillna(-1)
        exp = SparseArray([1, -1, -1, 3, -1], fill_value=0, dtype=np.float64)
        tm.assert_sp_array_equal(res, exp)

        s = SparseArray([1, np.nan, 0, 3, 0])
        res = s.fillna(-1)
        exp = SparseArray([1, -1, 0, 3, 0], fill_value=-1, dtype=np.float64)
        tm.assert_sp_array_equal(res, exp)

        s = SparseArray([1, np.nan, 0, 3, 0], fill_value=0)
        res = s.fillna(-1)
        exp = SparseArray([1, -1, 0, 3, 0], fill_value=0, dtype=np.float64)
        tm.assert_sp_array_equal(res, exp)

        s = SparseArray([np.nan, np.nan, np.nan, np.nan])
        res = s.fillna(-1)
        exp = SparseArray([-1, -1, -1, -1], fill_value=-1, dtype=np.float64)
        tm.assert_sp_array_equal(res, exp)

        s = SparseArray([np.nan, np.nan, np.nan, np.nan], fill_value=0)
        res = s.fillna(-1)
        exp = SparseArray([-1, -1, -1, -1], fill_value=0, dtype=np.float64)
        tm.assert_sp_array_equal(res, exp)

        # float dtype's fill_value is np.nan, replaced by -1
        s = SparseArray([0.0, 0.0, 0.0, 0.0])
        res = s.fillna(-1)
        exp = SparseArray([0.0, 0.0, 0.0, 0.0], fill_value=-1)
        tm.assert_sp_array_equal(res, exp)

        # int dtype shouldn't have missing. No changes.
        s = SparseArray([0, 0, 0, 0])
        assert s.dtype == SparseDtype(np.int64)
        assert s.fill_value == 0
        res = s.fillna(-1)
        tm.assert_sp_array_equal(res, s)

        s = SparseArray([0, 0, 0, 0], fill_value=0)
        assert s.dtype == SparseDtype(np.int64)
        assert s.fill_value == 0
        res = s.fillna(-1)
        exp = SparseArray([0, 0, 0, 0], fill_value=0)
        tm.assert_sp_array_equal(res, exp)

        # fill_value can be nan if there is no missing hole.
        # only fill_value will be changed
        s = SparseArray([0, 0, 0, 0], fill_value=np.nan)
        assert s.dtype == SparseDtype(np.int64, fill_value=np.nan)
        assert np.isnan(s.fill_value)
        res = s.fillna(-1)
        exp = SparseArray([0, 0, 0, 0], fill_value=-1)
        tm.assert_sp_array_equal(res, exp)

    def test_fillna_overlap(self):
        s = SparseArray([1, np.nan, np.nan, 3, np.nan])
        # filling with existing value doesn't replace existing value with
        # fill_value, i.e. existing 3 remains in sp_values
        res = s.fillna(3)
        exp = np.array([1, 3, 3, 3, 3], dtype=np.float64)
        tm.assert_numpy_array_equal(res.to_dense(), exp)

        s = SparseArray([1, np.nan, np.nan, 3, np.nan], fill_value=0)
        res = s.fillna(3)
        exp = SparseArray([1, 3, 3, 3, 3], fill_value=0, dtype=np.float64)
        tm.assert_sp_array_equal(res, exp)

    def test_nonzero(self):
        # Tests regression #21172.
        sa = SparseArray([float("nan"), float("nan"), 1, 0, 0, 2, 0, 0, 0, 3, 0, 0])
        expected = np.array([2, 5, 9], dtype=np.int32)
        (result,) = sa.nonzero()
        tm.assert_numpy_array_equal(expected, result)

        sa = SparseArray([0, 0, 1, 0, 0, 2, 0, 0, 0, 3, 0, 0])
        (result,) = sa.nonzero()
        tm.assert_numpy_array_equal(expected, result)


class TestSparseArrayAnalytics:
    @pytest.mark.parametrize(
        "data,pos,neg",
        [
            ([True, True, True], True, False),
            ([1, 2, 1], 1, 0),
            ([1.0, 2.0, 1.0], 1.0, 0.0),
        ],
    )
    def test_all(self, data, pos, neg):
        # GH 17570
        out = SparseArray(data).all()
        assert out

        out = SparseArray(data, fill_value=pos).all()
        assert out

        data[1] = neg
        out = SparseArray(data).all()
        assert not out

        out = SparseArray(data, fill_value=pos).all()
        assert not out

    @pytest.mark.parametrize(
        "data,pos,neg",
        [
            ([True, True, True], True, False),
            ([1, 2, 1], 1, 0),
            ([1.0, 2.0, 1.0], 1.0, 0.0),
        ],
    )
    def test_numpy_all(self, data, pos, neg):
        # GH 17570
        out = np.all(SparseArray(data))
        assert out

        out = np.all(SparseArray(data, fill_value=pos))
        assert out

        data[1] = neg
        out = np.all(SparseArray(data))
        assert not out

        out = np.all(SparseArray(data, fill_value=pos))
        assert not out

        # raises with a different message on py2.
        msg = "the 'out' parameter is not supported"
        with pytest.raises(ValueError, match=msg):
            np.all(SparseArray(data), out=np.array([]))

    @pytest.mark.parametrize(
        "data,pos,neg",
        [
            ([False, True, False], True, False),
            ([0, 2, 0], 2, 0),
            ([0.0, 2.0, 0.0], 2.0, 0.0),
        ],
    )
    def test_any(self, data, pos, neg):
        # GH 17570
        out = SparseArray(data).any()
        assert out

        out = SparseArray(data, fill_value=pos).any()
        assert out

        data[1] = neg
        out = SparseArray(data).any()
        assert not out

        out = SparseArray(data, fill_value=pos).any()
        assert not out

    @pytest.mark.parametrize(
        "data,pos,neg",
        [
            ([False, True, False], True, False),
            ([0, 2, 0], 2, 0),
            ([0.0, 2.0, 0.0], 2.0, 0.0),
        ],
    )
    def test_numpy_any(self, data, pos, neg):
        # GH 17570
        out = np.any(SparseArray(data))
        assert out

        out = np.any(SparseArray(data, fill_value=pos))
        assert out

        data[1] = neg
        out = np.any(SparseArray(data))
        assert not out

        out = np.any(SparseArray(data, fill_value=pos))
        assert not out

        msg = "the 'out' parameter is not supported"
        with pytest.raises(ValueError, match=msg):
            np.any(SparseArray(data), out=out)

    def test_sum(self):
        data = np.arange(10).astype(float)
        out = SparseArray(data).sum()
        assert out == 45.0

        data[5] = np.nan
        out = SparseArray(data, fill_value=2).sum()
        assert out == 40.0

        out = SparseArray(data, fill_value=np.nan).sum()
        assert out == 40.0

    @pytest.mark.parametrize(
        "arr",
        [
            np.array([0, 1, np.nan, 1]),
            np.array([0, 1, 1]),
            np.array([True, True, False]),
        ],
    )
    @pytest.mark.parametrize("fill_value", [0, 1, np.nan, True, False])
    @pytest.mark.parametrize("min_count, expected", [(3, 2), (4, np.nan)])
    def test_sum_min_count(self, arr, fill_value, min_count, expected):
        # https://github.com/pandas-dev/pandas/issues/25777
        sparray = SparseArray(arr, fill_value=fill_value)
        result = sparray.sum(min_count=min_count)
        if np.isnan(expected):
            assert np.isnan(result)
        else:
            assert result == expected

    def test_numpy_sum(self):
        data = np.arange(10).astype(float)
        out = np.sum(SparseArray(data))
        assert out == 45.0

        data[5] = np.nan
        out = np.sum(SparseArray(data, fill_value=2))
        assert out == 40.0

        out = np.sum(SparseArray(data, fill_value=np.nan))
        assert out == 40.0

        msg = "the 'dtype' parameter is not supported"
        with pytest.raises(ValueError, match=msg):
            np.sum(SparseArray(data), dtype=np.int64)

        msg = "the 'out' parameter is not supported"
        with pytest.raises(ValueError, match=msg):
            np.sum(SparseArray(data), out=out)

    @pytest.mark.parametrize(
        "data,expected",
        [
            (
                np.array([1, 2, 3, 4, 5], dtype=float),  # non-null data
                SparseArray(np.array([1.0, 3.0, 6.0, 10.0, 15.0])),
            ),
            (
                np.array([1, 2, np.nan, 4, 5], dtype=float),  # null data
                SparseArray(np.array([1.0, 3.0, np.nan, 7.0, 12.0])),
            ),
        ],
    )
    @pytest.mark.parametrize("numpy", [True, False])
    def test_cumsum(self, data, expected, numpy):
        cumsum = np.cumsum if numpy else lambda s: s.cumsum()

        out = cumsum(SparseArray(data))
        tm.assert_sp_array_equal(out, expected)

        out = cumsum(SparseArray(data, fill_value=np.nan))
        tm.assert_sp_array_equal(out, expected)

        out = cumsum(SparseArray(data, fill_value=2))
        tm.assert_sp_array_equal(out, expected)

        if numpy:  # numpy compatibility checks.
            msg = "the 'dtype' parameter is not supported"
            with pytest.raises(ValueError, match=msg):
                np.cumsum(SparseArray(data), dtype=np.int64)

            msg = "the 'out' parameter is not supported"
            with pytest.raises(ValueError, match=msg):
                np.cumsum(SparseArray(data), out=out)
        else:
            axis = 1  # SparseArray currently 1-D, so only axis = 0 is valid.
            msg = re.escape(f"axis(={axis}) out of bounds")
            with pytest.raises(ValueError, match=msg):
                SparseArray(data).cumsum(axis=axis)

    def test_mean(self):
        data = np.arange(10).astype(float)
        out = SparseArray(data).mean()
        assert out == 4.5

        data[5] = np.nan
        out = SparseArray(data).mean()
        assert out == 40.0 / 9

    def test_numpy_mean(self):
        data = np.arange(10).astype(float)
        out = np.mean(SparseArray(data))
        assert out == 4.5

        data[5] = np.nan
        out = np.mean(SparseArray(data))
        assert out == 40.0 / 9

        msg = "the 'dtype' parameter is not supported"
        with pytest.raises(ValueError, match=msg):
            np.mean(SparseArray(data), dtype=np.int64)

        msg = "the 'out' parameter is not supported"
        with pytest.raises(ValueError, match=msg):
            np.mean(SparseArray(data), out=out)

    def test_ufunc(self):
        # GH 13853 make sure ufunc is applied to fill_value
        sparse = SparseArray([1, np.nan, 2, np.nan, -2])
        result = SparseArray([1, np.nan, 2, np.nan, 2])
        tm.assert_sp_array_equal(abs(sparse), result)
        tm.assert_sp_array_equal(np.abs(sparse), result)

        sparse = SparseArray([1, -1, 2, -2], fill_value=1)
        result = SparseArray([1, 2, 2], sparse_index=sparse.sp_index, fill_value=1)
        tm.assert_sp_array_equal(abs(sparse), result)
        tm.assert_sp_array_equal(np.abs(sparse), result)

        sparse = SparseArray([1, -1, 2, -2], fill_value=-1)
        result = SparseArray([1, 2, 2], sparse_index=sparse.sp_index, fill_value=1)
        tm.assert_sp_array_equal(abs(sparse), result)
        tm.assert_sp_array_equal(np.abs(sparse), result)

        sparse = SparseArray([1, np.nan, 2, np.nan, -2])
        result = SparseArray(np.sin([1, np.nan, 2, np.nan, -2]))
        tm.assert_sp_array_equal(np.sin(sparse), result)

        sparse = SparseArray([1, -1, 2, -2], fill_value=1)
        result = SparseArray(np.sin([1, -1, 2, -2]), fill_value=np.sin(1))
        tm.assert_sp_array_equal(np.sin(sparse), result)

        sparse = SparseArray([1, -1, 0, -2], fill_value=0)
        result = SparseArray(np.sin([1, -1, 0, -2]), fill_value=np.sin(0))
        tm.assert_sp_array_equal(np.sin(sparse), result)

    def test_ufunc_args(self):
        # GH 13853 make sure ufunc is applied to fill_value, including its arg
        sparse = SparseArray([1, np.nan, 2, np.nan, -2])
        result = SparseArray([2, np.nan, 3, np.nan, -1])
        tm.assert_sp_array_equal(np.add(sparse, 1), result)

        sparse = SparseArray([1, -1, 2, -2], fill_value=1)
        result = SparseArray([2, 0, 3, -1], fill_value=2)
        tm.assert_sp_array_equal(np.add(sparse, 1), result)

        sparse = SparseArray([1, -1, 0, -2], fill_value=0)
        result = SparseArray([2, 0, 1, -1], fill_value=1)
        tm.assert_sp_array_equal(np.add(sparse, 1), result)

    @pytest.mark.parametrize("fill_value", [0.0, np.nan])
    def test_modf(self, fill_value):
        # https://github.com/pandas-dev/pandas/issues/26946
        sparse = SparseArray([fill_value] * 10 + [1.1, 2.2], fill_value=fill_value)
        r1, r2 = np.modf(sparse)
        e1, e2 = np.modf(np.asarray(sparse))
        tm.assert_sp_array_equal(r1, SparseArray(e1, fill_value=fill_value))
        tm.assert_sp_array_equal(r2, SparseArray(e2, fill_value=fill_value))

    def test_nbytes_integer(self):
        arr = SparseArray([1, 0, 0, 0, 2], kind="integer")
        result = arr.nbytes
        # (2 * 8) + 2 * 4
        assert result == 24

    def test_nbytes_block(self):
        arr = SparseArray([1, 2, 0, 0, 0], kind="block")
        result = arr.nbytes
        # (2 * 8) + 4 + 4
        # sp_values, blocs, blengths
        assert result == 24

    def test_asarray_datetime64(self):
        s = SparseArray(pd.to_datetime(["2012", None, None, "2013"]))
        np.asarray(s)

    def test_density(self):
        arr = SparseArray([0, 1])
        assert arr.density == 0.5

    def test_npoints(self):
        arr = SparseArray([0, 1])
        assert arr.npoints == 1


class TestAccessor:
    @pytest.mark.parametrize("attr", ["npoints", "density", "fill_value", "sp_values"])
    def test_get_attributes(self, attr):
        arr = SparseArray([0, 1])
        ser = pd.Series(arr)

        result = getattr(ser.sparse, attr)
        expected = getattr(arr, attr)
        assert result == expected

    @td.skip_if_no_scipy
    def test_from_coo(self):
        import scipy.sparse

        row = [0, 3, 1, 0]
        col = [0, 3, 1, 2]
        data = [4, 5, 7, 9]
        # TODO: Remove dtype when scipy is fixed
        # https://github.com/scipy/scipy/issues/13585
        sp_array = scipy.sparse.coo_matrix((data, (row, col)), dtype="int")
        result = pd.Series.sparse.from_coo(sp_array)

        index = pd.MultiIndex.from_arrays([[0, 0, 1, 3], [0, 2, 1, 3]])
        expected = pd.Series([4, 9, 7, 5], index=index, dtype="Sparse[int]")
        tm.assert_series_equal(result, expected)

    @td.skip_if_no_scipy
    def test_to_coo(self):
        import scipy.sparse

        ser = pd.Series(
            [1, 2, 3],
            index=pd.MultiIndex.from_product([[0], [1, 2, 3]], names=["a", "b"]),
            dtype="Sparse[int]",
        )
        A, _, _ = ser.sparse.to_coo()
        assert isinstance(A, scipy.sparse.coo_matrix)

    def test_non_sparse_raises(self):
        ser = pd.Series([1, 2, 3])
        with pytest.raises(AttributeError, match=".sparse"):
            ser.sparse.density


def test_setting_fill_value_fillna_still_works():
    # This is why letting users update fill_value / dtype is bad
    # astype has the same problem.
    arr = SparseArray([1.0, np.nan, 1.0], fill_value=0.0)
    arr.fill_value = np.nan
    result = arr.isna()
    # Can't do direct comparison, since the sp_index will be different
    # So let's convert to ndarray and check there.
    result = np.asarray(result)

    expected = np.array([False, True, False])
    tm.assert_numpy_array_equal(result, expected)


def test_setting_fill_value_updates():
    arr = SparseArray([0.0, np.nan], fill_value=0)
    arr.fill_value = np.nan
    # use private constructor to get the index right
    # otherwise both nans would be un-stored.
    expected = SparseArray._simple_new(
        sparse_array=np.array([np.nan]),
        sparse_index=IntIndex(2, [1]),
        dtype=SparseDtype(float, np.nan),
    )
    tm.assert_sp_array_equal(arr, expected)


@pytest.mark.parametrize(
    "arr, loc",
    [
        ([None, 1, 2], 0),
        ([0, None, 2], 1),
        ([0, 1, None], 2),
        ([0, 1, 1, None, None], 3),
        ([1, 1, 1, 2], -1),
        ([], -1),
    ],
)
def test_first_fill_value_loc(arr, loc):
    result = SparseArray(arr)._first_fill_value_loc()
    assert result == loc


@pytest.mark.parametrize(
    "arr", [[1, 2, np.nan, np.nan], [1, np.nan, 2, np.nan], [1, 2, np.nan]]
)
@pytest.mark.parametrize("fill_value", [np.nan, 0, 1])
def test_unique_na_fill(arr, fill_value):
    a = SparseArray(arr, fill_value=fill_value).unique()
    b = pd.Series(arr).unique()
    assert isinstance(a, SparseArray)
    a = np.asarray(a)
    tm.assert_numpy_array_equal(a, b)


def test_unique_all_sparse():
    # https://github.com/pandas-dev/pandas/issues/23168
    arr = SparseArray([0, 0])
    result = arr.unique()
    expected = SparseArray([0])
    tm.assert_sp_array_equal(result, expected)


def test_map():
    arr = SparseArray([0, 1, 2])
    expected = SparseArray([10, 11, 12], fill_value=10)

    # dict
    result = arr.map({0: 10, 1: 11, 2: 12})
    tm.assert_sp_array_equal(result, expected)

    # series
    result = arr.map(pd.Series({0: 10, 1: 11, 2: 12}))
    tm.assert_sp_array_equal(result, expected)

    # function
    result = arr.map(pd.Series({0: 10, 1: 11, 2: 12}))
    expected = SparseArray([10, 11, 12], fill_value=10)
    tm.assert_sp_array_equal(result, expected)


def test_map_missing():
    arr = SparseArray([0, 1, 2])
    expected = SparseArray([10, 11, None], fill_value=10)

    result = arr.map({0: 10, 1: 11})
    tm.assert_sp_array_equal(result, expected)


@pytest.mark.parametrize("fill_value", [np.nan, 1])
def test_dropna(fill_value):
    # GH-28287
    arr = SparseArray([np.nan, 1], fill_value=fill_value)
    exp = SparseArray([1.0], fill_value=fill_value)
    tm.assert_sp_array_equal(arr.dropna(), exp)

    df = pd.DataFrame({"a": [0, 1], "b": arr})
    expected_df = pd.DataFrame({"a": [1], "b": exp}, index=pd.Int64Index([1]))
    tm.assert_equal(df.dropna(), expected_df)


def test_drop_duplicates_fill_value():
    # GH 11726
    df = pd.DataFrame(np.zeros((5, 5))).apply(lambda x: SparseArray(x, fill_value=0))
    result = df.drop_duplicates()
    expected = pd.DataFrame({i: SparseArray([0.0], fill_value=0) for i in range(5)})
    tm.assert_frame_equal(result, expected)


class TestMinMax:
    plain_data = np.arange(5).astype(float)
    data_neg = plain_data * (-1)
    data_NaN = SparseArray(np.array([0, 1, 2, np.nan, 4]))
    data_all_NaN = SparseArray(np.array([np.nan, np.nan, np.nan, np.nan, np.nan]))
    data_NA_filled = SparseArray(
        np.array([np.nan, np.nan, np.nan, np.nan, np.nan]), fill_value=5
    )

    @pytest.mark.parametrize(
        "raw_data,max_expected,min_expected",
        [
            (plain_data, [4], [0]),
            (data_neg, [0], [-4]),
            (data_NaN, [4], [0]),
            (data_all_NaN, [np.nan], [np.nan]),
            (data_NA_filled, [5], [5]),
        ],
    )
    def test_maxmin(self, raw_data, max_expected, min_expected):
        max_result = SparseArray(raw_data).max()
        min_result = SparseArray(raw_data).min()
        assert max_result in max_expected
        assert min_result in min_expected
