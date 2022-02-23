from datetime import datetime
from itertools import permutations
import struct

import numpy as np
import pytest

from pandas._libs import (
    algos as libalgos,
    hashtable as ht,
)
from pandas.compat import np_array_datetime64_compat
import pandas.util._test_decorators as td

from pandas.core.dtypes.common import (
    is_bool_dtype,
    is_complex_dtype,
    is_float_dtype,
    is_integer_dtype,
    is_object_dtype,
)
from pandas.core.dtypes.dtypes import CategoricalDtype as CDT

import pandas as pd
from pandas import (
    Categorical,
    CategoricalIndex,
    DataFrame,
    DatetimeIndex,
    Index,
    IntervalIndex,
    MultiIndex,
    NaT,
    Period,
    PeriodIndex,
    Series,
    Timedelta,
    Timestamp,
    date_range,
    timedelta_range,
    to_datetime,
    to_timedelta,
)
import pandas._testing as tm
import pandas.core.algorithms as algos
from pandas.core.arrays import DatetimeArray
import pandas.core.common as com


class TestFactorize:
    @pytest.mark.parametrize("sort", [True, False])
    def test_factorize(self, index_or_series_obj, sort):
        obj = index_or_series_obj
        result_codes, result_uniques = obj.factorize(sort=sort)

        constructor = Index
        if isinstance(obj, MultiIndex):
            constructor = MultiIndex.from_tuples
        expected_uniques = constructor(obj.unique())

        if sort:
            expected_uniques = expected_uniques.sort_values()

        # construct an integer ndarray so that
        # `expected_uniques.take(expected_codes)` is equal to `obj`
        expected_uniques_list = list(expected_uniques)
        expected_codes = [expected_uniques_list.index(val) for val in obj]
        expected_codes = np.asarray(expected_codes, dtype=np.intp)

        tm.assert_numpy_array_equal(result_codes, expected_codes)
        tm.assert_index_equal(result_uniques, expected_uniques)

    def test_series_factorize_na_sentinel_none(self):
        # GH#35667
        values = np.array([1, 2, 1, np.nan])
        ser = Series(values)
        codes, uniques = ser.factorize(na_sentinel=None)

        expected_codes = np.array([0, 1, 0, 2], dtype=np.intp)
        expected_uniques = Index([1.0, 2.0, np.nan])

        tm.assert_numpy_array_equal(codes, expected_codes)
        tm.assert_index_equal(uniques, expected_uniques)

    def test_basic(self):

        codes, uniques = algos.factorize(["a", "b", "b", "a", "a", "c", "c", "c"])
        tm.assert_numpy_array_equal(uniques, np.array(["a", "b", "c"], dtype=object))

        codes, uniques = algos.factorize(
            ["a", "b", "b", "a", "a", "c", "c", "c"], sort=True
        )
        exp = np.array([0, 1, 1, 0, 0, 2, 2, 2], dtype=np.intp)
        tm.assert_numpy_array_equal(codes, exp)
        exp = np.array(["a", "b", "c"], dtype=object)
        tm.assert_numpy_array_equal(uniques, exp)

        arr = np.arange(5, dtype=np.intp)[::-1]

        codes, uniques = algos.factorize(arr)
        exp = np.array([0, 1, 2, 3, 4], dtype=np.intp)
        tm.assert_numpy_array_equal(codes, exp)
        exp = np.array([4, 3, 2, 1, 0], dtype=arr.dtype)
        tm.assert_numpy_array_equal(uniques, exp)

        codes, uniques = algos.factorize(arr, sort=True)
        exp = np.array([4, 3, 2, 1, 0], dtype=np.intp)
        tm.assert_numpy_array_equal(codes, exp)
        exp = np.array([0, 1, 2, 3, 4], dtype=arr.dtype)
        tm.assert_numpy_array_equal(uniques, exp)

        arr = np.arange(5.0)[::-1]

        codes, uniques = algos.factorize(arr)
        exp = np.array([0, 1, 2, 3, 4], dtype=np.intp)
        tm.assert_numpy_array_equal(codes, exp)
        exp = np.array([4.0, 3.0, 2.0, 1.0, 0.0], dtype=arr.dtype)
        tm.assert_numpy_array_equal(uniques, exp)

        codes, uniques = algos.factorize(arr, sort=True)
        exp = np.array([4, 3, 2, 1, 0], dtype=np.intp)
        tm.assert_numpy_array_equal(codes, exp)
        exp = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=arr.dtype)
        tm.assert_numpy_array_equal(uniques, exp)

    def test_mixed(self):

        # doc example reshaping.rst
        x = Series(["A", "A", np.nan, "B", 3.14, np.inf])
        codes, uniques = algos.factorize(x)

        exp = np.array([0, 0, -1, 1, 2, 3], dtype=np.intp)
        tm.assert_numpy_array_equal(codes, exp)
        exp = Index(["A", "B", 3.14, np.inf])
        tm.assert_index_equal(uniques, exp)

        codes, uniques = algos.factorize(x, sort=True)
        exp = np.array([2, 2, -1, 3, 0, 1], dtype=np.intp)
        tm.assert_numpy_array_equal(codes, exp)
        exp = Index([3.14, np.inf, "A", "B"])
        tm.assert_index_equal(uniques, exp)

    def test_datelike(self):

        # M8
        v1 = Timestamp("20130101 09:00:00.00004")
        v2 = Timestamp("20130101")
        x = Series([v1, v1, v1, v2, v2, v1])
        codes, uniques = algos.factorize(x)

        exp = np.array([0, 0, 0, 1, 1, 0], dtype=np.intp)
        tm.assert_numpy_array_equal(codes, exp)
        exp = DatetimeIndex([v1, v2])
        tm.assert_index_equal(uniques, exp)

        codes, uniques = algos.factorize(x, sort=True)
        exp = np.array([1, 1, 1, 0, 0, 1], dtype=np.intp)
        tm.assert_numpy_array_equal(codes, exp)
        exp = DatetimeIndex([v2, v1])
        tm.assert_index_equal(uniques, exp)

        # period
        v1 = Period("201302", freq="M")
        v2 = Period("201303", freq="M")
        x = Series([v1, v1, v1, v2, v2, v1])

        # periods are not 'sorted' as they are converted back into an index
        codes, uniques = algos.factorize(x)
        exp = np.array([0, 0, 0, 1, 1, 0], dtype=np.intp)
        tm.assert_numpy_array_equal(codes, exp)
        tm.assert_index_equal(uniques, PeriodIndex([v1, v2]))

        codes, uniques = algos.factorize(x, sort=True)
        exp = np.array([0, 0, 0, 1, 1, 0], dtype=np.intp)
        tm.assert_numpy_array_equal(codes, exp)
        tm.assert_index_equal(uniques, PeriodIndex([v1, v2]))

        # GH 5986
        v1 = to_timedelta("1 day 1 min")
        v2 = to_timedelta("1 day")
        x = Series([v1, v2, v1, v1, v2, v2, v1])
        codes, uniques = algos.factorize(x)
        exp = np.array([0, 1, 0, 0, 1, 1, 0], dtype=np.intp)
        tm.assert_numpy_array_equal(codes, exp)
        tm.assert_index_equal(uniques, to_timedelta([v1, v2]))

        codes, uniques = algos.factorize(x, sort=True)
        exp = np.array([1, 0, 1, 1, 0, 0, 1], dtype=np.intp)
        tm.assert_numpy_array_equal(codes, exp)
        tm.assert_index_equal(uniques, to_timedelta([v2, v1]))

    def test_factorize_nan(self):
        # nan should map to na_sentinel, not reverse_indexer[na_sentinel]
        # rizer.factorize should not raise an exception if na_sentinel indexes
        # outside of reverse_indexer
        key = np.array([1, 2, 1, np.nan], dtype="O")
        rizer = ht.ObjectFactorizer(len(key))
        for na_sentinel in (-1, 20):
            ids = rizer.factorize(key, sort=True, na_sentinel=na_sentinel)
            expected = np.array([0, 1, 0, na_sentinel], dtype="int32")
            assert len(set(key)) == len(set(expected))
            tm.assert_numpy_array_equal(pd.isna(key), expected == na_sentinel)

        # nan still maps to na_sentinel when sort=False
        key = np.array([0, np.nan, 1], dtype="O")
        na_sentinel = -1

        # TODO(wesm): unused?
        ids = rizer.factorize(key, sort=False, na_sentinel=na_sentinel)  # noqa

        expected = np.array([2, -1, 0], dtype="int32")
        assert len(set(key)) == len(set(expected))
        tm.assert_numpy_array_equal(pd.isna(key), expected == na_sentinel)

    @pytest.mark.parametrize(
        "data, expected_codes, expected_uniques",
        [
            (
                [(1, 1), (1, 2), (0, 0), (1, 2), "nonsense"],
                [0, 1, 2, 1, 3],
                [(1, 1), (1, 2), (0, 0), "nonsense"],
            ),
            (
                [(1, 1), (1, 2), (0, 0), (1, 2), (1, 2, 3)],
                [0, 1, 2, 1, 3],
                [(1, 1), (1, 2), (0, 0), (1, 2, 3)],
            ),
            ([(1, 1), (1, 2), (0, 0), (1, 2)], [0, 1, 2, 1], [(1, 1), (1, 2), (0, 0)]),
        ],
    )
    def test_factorize_tuple_list(self, data, expected_codes, expected_uniques):
        # GH9454
        codes, uniques = pd.factorize(data)

        tm.assert_numpy_array_equal(codes, np.array(expected_codes, dtype=np.intp))

        expected_uniques_array = com.asarray_tuplesafe(expected_uniques, dtype=object)
        tm.assert_numpy_array_equal(uniques, expected_uniques_array)

    def test_complex_sorting(self):
        # gh 12666 - check no segfault
        x17 = np.array([complex(i) for i in range(17)], dtype=object)

        msg = (
            "unorderable types: .* [<>] .*"
            "|"  # the above case happens for numpy < 1.14
            "'[<>]' not supported between instances of .*"
        )
        with pytest.raises(TypeError, match=msg):
            algos.factorize(x17[::-1], sort=True)

    def test_numeric_dtype_factorize(self, any_real_dtype):
        # GH41132
        dtype = any_real_dtype
        data = np.array([1, 2, 2, 1], dtype=dtype)
        expected_codes = np.array([0, 1, 1, 0], dtype=np.intp)
        expected_uniques = np.array([1, 2], dtype=dtype)

        codes, uniques = algos.factorize(data)
        tm.assert_numpy_array_equal(codes, expected_codes)
        tm.assert_numpy_array_equal(uniques, expected_uniques)

    def test_float64_factorize(self, writable):
        data = np.array([1.0, 1e8, 1.0, 1e-8, 1e8, 1.0], dtype=np.float64)
        data.setflags(write=writable)
        expected_codes = np.array([0, 1, 0, 2, 1, 0], dtype=np.intp)
        expected_uniques = np.array([1.0, 1e8, 1e-8], dtype=np.float64)

        codes, uniques = algos.factorize(data)
        tm.assert_numpy_array_equal(codes, expected_codes)
        tm.assert_numpy_array_equal(uniques, expected_uniques)

    def test_uint64_factorize(self, writable):
        data = np.array([2 ** 64 - 1, 1, 2 ** 64 - 1], dtype=np.uint64)
        data.setflags(write=writable)
        expected_codes = np.array([0, 1, 0], dtype=np.intp)
        expected_uniques = np.array([2 ** 64 - 1, 1], dtype=np.uint64)

        codes, uniques = algos.factorize(data)
        tm.assert_numpy_array_equal(codes, expected_codes)
        tm.assert_numpy_array_equal(uniques, expected_uniques)

    def test_int64_factorize(self, writable):
        data = np.array([2 ** 63 - 1, -(2 ** 63), 2 ** 63 - 1], dtype=np.int64)
        data.setflags(write=writable)
        expected_codes = np.array([0, 1, 0], dtype=np.intp)
        expected_uniques = np.array([2 ** 63 - 1, -(2 ** 63)], dtype=np.int64)

        codes, uniques = algos.factorize(data)
        tm.assert_numpy_array_equal(codes, expected_codes)
        tm.assert_numpy_array_equal(uniques, expected_uniques)

    def test_string_factorize(self, writable):
        data = np.array(["a", "c", "a", "b", "c"], dtype=object)
        data.setflags(write=writable)
        expected_codes = np.array([0, 1, 0, 2, 1], dtype=np.intp)
        expected_uniques = np.array(["a", "c", "b"], dtype=object)

        codes, uniques = algos.factorize(data)
        tm.assert_numpy_array_equal(codes, expected_codes)
        tm.assert_numpy_array_equal(uniques, expected_uniques)

    def test_object_factorize(self, writable):
        data = np.array(["a", "c", None, np.nan, "a", "b", NaT, "c"], dtype=object)
        data.setflags(write=writable)
        expected_codes = np.array([0, 1, -1, -1, 0, 2, -1, 1], dtype=np.intp)
        expected_uniques = np.array(["a", "c", "b"], dtype=object)

        codes, uniques = algos.factorize(data)
        tm.assert_numpy_array_equal(codes, expected_codes)
        tm.assert_numpy_array_equal(uniques, expected_uniques)

    def test_datetime64_factorize(self, writable):
        # GH35650 Verify whether read-only datetime64 array can be factorized
        data = np.array([np.datetime64("2020-01-01T00:00:00.000")])
        data.setflags(write=writable)
        expected_codes = np.array([0], dtype=np.intp)
        expected_uniques = np.array(
            ["2020-01-01T00:00:00.000000000"], dtype="datetime64[ns]"
        )

        codes, uniques = pd.factorize(data)
        tm.assert_numpy_array_equal(codes, expected_codes)
        tm.assert_numpy_array_equal(uniques, expected_uniques)

    @pytest.mark.parametrize("sort", [True, False])
    def test_factorize_rangeindex(self, sort):
        # increasing -> sort doesn't matter
        ri = pd.RangeIndex.from_range(range(10))
        expected = np.arange(10, dtype=np.intp), ri

        result = algos.factorize(ri, sort=sort)
        tm.assert_numpy_array_equal(result[0], expected[0])
        tm.assert_index_equal(result[1], expected[1], exact=True)

        result = ri.factorize(sort=sort)
        tm.assert_numpy_array_equal(result[0], expected[0])
        tm.assert_index_equal(result[1], expected[1], exact=True)

    @pytest.mark.parametrize("sort", [True, False])
    def test_factorize_rangeindex_decreasing(self, sort):
        # decreasing -> sort matters
        ri = pd.RangeIndex.from_range(range(10))
        expected = np.arange(10, dtype=np.intp), ri

        ri2 = ri[::-1]
        expected = expected[0], ri2
        if sort:
            expected = expected[0][::-1], expected[1][::-1]

        result = algos.factorize(ri2, sort=sort)
        tm.assert_numpy_array_equal(result[0], expected[0])
        tm.assert_index_equal(result[1], expected[1], exact=True)

        result = ri2.factorize(sort=sort)
        tm.assert_numpy_array_equal(result[0], expected[0])
        tm.assert_index_equal(result[1], expected[1], exact=True)

    def test_deprecate_order(self):
        # gh 19727 - check warning is raised for deprecated keyword, order.
        # Test not valid once order keyword is removed.
        data = np.array([2 ** 63, 1, 2 ** 63], dtype=np.uint64)
        with pytest.raises(TypeError, match="got an unexpected keyword"):
            algos.factorize(data, order=True)
        with tm.assert_produces_warning(False):
            algos.factorize(data)

    @pytest.mark.parametrize(
        "data",
        [
            np.array([0, 1, 0], dtype="u8"),
            np.array([-(2 ** 63), 1, -(2 ** 63)], dtype="i8"),
            np.array(["__nan__", "foo", "__nan__"], dtype="object"),
        ],
    )
    def test_parametrized_factorize_na_value_default(self, data):
        # arrays that include the NA default for that type, but isn't used.
        codes, uniques = algos.factorize(data)
        expected_uniques = data[[0, 1]]
        expected_codes = np.array([0, 1, 0], dtype=np.intp)
        tm.assert_numpy_array_equal(codes, expected_codes)
        tm.assert_numpy_array_equal(uniques, expected_uniques)

    @pytest.mark.parametrize(
        "data, na_value",
        [
            (np.array([0, 1, 0, 2], dtype="u8"), 0),
            (np.array([1, 0, 1, 2], dtype="u8"), 1),
            (np.array([-(2 ** 63), 1, -(2 ** 63), 0], dtype="i8"), -(2 ** 63)),
            (np.array([1, -(2 ** 63), 1, 0], dtype="i8"), 1),
            (np.array(["a", "", "a", "b"], dtype=object), "a"),
            (np.array([(), ("a", 1), (), ("a", 2)], dtype=object), ()),
            (np.array([("a", 1), (), ("a", 1), ("a", 2)], dtype=object), ("a", 1)),
        ],
    )
    def test_parametrized_factorize_na_value(self, data, na_value):
        codes, uniques = algos.factorize_array(data, na_value=na_value)
        expected_uniques = data[[1, 3]]
        expected_codes = np.array([-1, 0, -1, 1], dtype=np.intp)
        tm.assert_numpy_array_equal(codes, expected_codes)
        tm.assert_numpy_array_equal(uniques, expected_uniques)

    @pytest.mark.parametrize("sort", [True, False])
    @pytest.mark.parametrize("na_sentinel", [-1, -10, 100])
    @pytest.mark.parametrize(
        "data, uniques",
        [
            (
                np.array(["b", "a", None, "b"], dtype=object),
                np.array(["b", "a"], dtype=object),
            ),
            (
                pd.array([2, 1, np.nan, 2], dtype="Int64"),
                pd.array([2, 1], dtype="Int64"),
            ),
        ],
        ids=["numpy_array", "extension_array"],
    )
    def test_factorize_na_sentinel(self, sort, na_sentinel, data, uniques):
        codes, uniques = algos.factorize(data, sort=sort, na_sentinel=na_sentinel)
        if sort:
            expected_codes = np.array([1, 0, na_sentinel, 1], dtype=np.intp)
            expected_uniques = algos.safe_sort(uniques)
        else:
            expected_codes = np.array([0, 1, na_sentinel, 0], dtype=np.intp)
            expected_uniques = uniques
        tm.assert_numpy_array_equal(codes, expected_codes)
        if isinstance(data, np.ndarray):
            tm.assert_numpy_array_equal(uniques, expected_uniques)
        else:
            tm.assert_extension_array_equal(uniques, expected_uniques)

    @pytest.mark.parametrize(
        "data, expected_codes, expected_uniques",
        [
            (
                ["a", None, "b", "a"],
                np.array([0, 2, 1, 0], dtype=np.dtype("intp")),
                np.array(["a", "b", np.nan], dtype=object),
            ),
            (
                ["a", np.nan, "b", "a"],
                np.array([0, 2, 1, 0], dtype=np.dtype("intp")),
                np.array(["a", "b", np.nan], dtype=object),
            ),
        ],
    )
    def test_object_factorize_na_sentinel_none(
        self, data, expected_codes, expected_uniques
    ):
        codes, uniques = algos.factorize(data, na_sentinel=None)

        tm.assert_numpy_array_equal(uniques, expected_uniques)
        tm.assert_numpy_array_equal(codes, expected_codes)

    @pytest.mark.parametrize(
        "data, expected_codes, expected_uniques",
        [
            (
                [1, None, 1, 2],
                np.array([0, 2, 0, 1], dtype=np.dtype("intp")),
                np.array([1, 2, np.nan], dtype="O"),
            ),
            (
                [1, np.nan, 1, 2],
                np.array([0, 2, 0, 1], dtype=np.dtype("intp")),
                np.array([1, 2, np.nan], dtype=np.float64),
            ),
        ],
    )
    def test_int_factorize_na_sentinel_none(
        self, data, expected_codes, expected_uniques
    ):
        codes, uniques = algos.factorize(data, na_sentinel=None)

        tm.assert_numpy_array_equal(uniques, expected_uniques)
        tm.assert_numpy_array_equal(codes, expected_codes)


class TestUnique:
    def test_ints(self):
        arr = np.random.randint(0, 100, size=50)

        result = algos.unique(arr)
        assert isinstance(result, np.ndarray)

    def test_objects(self):
        arr = np.random.randint(0, 100, size=50).astype("O")

        result = algos.unique(arr)
        assert isinstance(result, np.ndarray)

    def test_object_refcount_bug(self):
        lst = ["A", "B", "C", "D", "E"]
        for i in range(1000):
            len(algos.unique(lst))

    def test_on_index_object(self):

        mindex = MultiIndex.from_arrays(
            [np.arange(5).repeat(5), np.tile(np.arange(5), 5)]
        )
        expected = mindex.values
        expected.sort()

        mindex = mindex.repeat(2)

        result = pd.unique(mindex)
        result.sort()

        tm.assert_almost_equal(result, expected)

    def test_dtype_preservation(self, any_numpy_dtype):
        # GH 15442
        if any_numpy_dtype in (tm.BYTES_DTYPES + tm.STRING_DTYPES):
            pytest.skip("skip string dtype")
        elif is_integer_dtype(any_numpy_dtype):
            data = [1, 2, 2]
            uniques = [1, 2]
        elif is_float_dtype(any_numpy_dtype):
            data = [1, 2, 2]
            uniques = [1.0, 2.0]
        elif is_complex_dtype(any_numpy_dtype):
            data = [complex(1, 0), complex(2, 0), complex(2, 0)]
            uniques = [complex(1, 0), complex(2, 0)]
        elif is_bool_dtype(any_numpy_dtype):
            data = [True, True, False]
            uniques = [True, False]
        elif is_object_dtype(any_numpy_dtype):
            data = ["A", "B", "B"]
            uniques = ["A", "B"]
        else:
            # datetime64[ns]/M8[ns]/timedelta64[ns]/m8[ns] tested elsewhere
            data = [1, 2, 2]
            uniques = [1, 2]

        result = Series(data, dtype=any_numpy_dtype).unique()
        expected = np.array(uniques, dtype=any_numpy_dtype)

        tm.assert_numpy_array_equal(result, expected)

    def test_datetime64_dtype_array_returned(self):
        # GH 9431
        expected = np_array_datetime64_compat(
            [
                "2015-01-03T00:00:00.000000000+0000",
                "2015-01-01T00:00:00.000000000+0000",
            ],
            dtype="M8[ns]",
        )

        dt_index = to_datetime(
            [
                "2015-01-03T00:00:00.000000000",
                "2015-01-01T00:00:00.000000000",
                "2015-01-01T00:00:00.000000000",
            ]
        )
        result = algos.unique(dt_index)
        tm.assert_numpy_array_equal(result, expected)
        assert result.dtype == expected.dtype

        s = Series(dt_index)
        result = algos.unique(s)
        tm.assert_numpy_array_equal(result, expected)
        assert result.dtype == expected.dtype

        arr = s.values
        result = algos.unique(arr)
        tm.assert_numpy_array_equal(result, expected)
        assert result.dtype == expected.dtype

    def test_datetime_non_ns(self):
        a = np.array(["2000", "2000", "2001"], dtype="datetime64[s]")
        result = pd.unique(a)
        expected = np.array(["2000", "2001"], dtype="datetime64[ns]")
        tm.assert_numpy_array_equal(result, expected)

    def test_timedelta_non_ns(self):
        a = np.array(["2000", "2000", "2001"], dtype="timedelta64[s]")
        result = pd.unique(a)
        expected = np.array([2000000000000, 2001000000000], dtype="timedelta64[ns]")
        tm.assert_numpy_array_equal(result, expected)

    def test_timedelta64_dtype_array_returned(self):
        # GH 9431
        expected = np.array([31200, 45678, 10000], dtype="m8[ns]")

        td_index = to_timedelta([31200, 45678, 31200, 10000, 45678])
        result = algos.unique(td_index)
        tm.assert_numpy_array_equal(result, expected)
        assert result.dtype == expected.dtype

        s = Series(td_index)
        result = algos.unique(s)
        tm.assert_numpy_array_equal(result, expected)
        assert result.dtype == expected.dtype

        arr = s.values
        result = algos.unique(arr)
        tm.assert_numpy_array_equal(result, expected)
        assert result.dtype == expected.dtype

    def test_uint64_overflow(self):
        s = Series([1, 2, 2 ** 63, 2 ** 63], dtype=np.uint64)
        exp = np.array([1, 2, 2 ** 63], dtype=np.uint64)
        tm.assert_numpy_array_equal(algos.unique(s), exp)

    def test_nan_in_object_array(self):
        duplicated_items = ["a", np.nan, "c", "c"]
        result = pd.unique(duplicated_items)
        expected = np.array(["a", np.nan, "c"], dtype=object)
        tm.assert_numpy_array_equal(result, expected)

    def test_categorical(self):

        # we are expecting to return in the order
        # of appearance
        expected = Categorical(list("bac"))

        # we are expecting to return in the order
        # of the categories
        expected_o = Categorical(list("bac"), categories=list("abc"), ordered=True)

        # GH 15939
        c = Categorical(list("baabc"))
        result = c.unique()
        tm.assert_categorical_equal(result, expected)

        result = algos.unique(c)
        tm.assert_categorical_equal(result, expected)

        c = Categorical(list("baabc"), ordered=True)
        result = c.unique()
        tm.assert_categorical_equal(result, expected_o)

        result = algos.unique(c)
        tm.assert_categorical_equal(result, expected_o)

        # Series of categorical dtype
        s = Series(Categorical(list("baabc")), name="foo")
        result = s.unique()
        tm.assert_categorical_equal(result, expected)

        result = pd.unique(s)
        tm.assert_categorical_equal(result, expected)

        # CI -> return CI
        ci = CategoricalIndex(Categorical(list("baabc"), categories=list("abc")))
        expected = CategoricalIndex(expected)
        result = ci.unique()
        tm.assert_index_equal(result, expected)

        result = pd.unique(ci)
        tm.assert_index_equal(result, expected)

    def test_datetime64tz_aware(self):
        # GH 15939

        result = Series(
            Index(
                [
                    Timestamp("20160101", tz="US/Eastern"),
                    Timestamp("20160101", tz="US/Eastern"),
                ]
            )
        ).unique()
        expected = DatetimeArray._from_sequence(
            np.array([Timestamp("2016-01-01 00:00:00-0500", tz="US/Eastern")])
        )
        tm.assert_extension_array_equal(result, expected)

        result = Index(
            [
                Timestamp("20160101", tz="US/Eastern"),
                Timestamp("20160101", tz="US/Eastern"),
            ]
        ).unique()
        expected = DatetimeIndex(
            ["2016-01-01 00:00:00"], dtype="datetime64[ns, US/Eastern]", freq=None
        )
        tm.assert_index_equal(result, expected)

        result = pd.unique(
            Series(
                Index(
                    [
                        Timestamp("20160101", tz="US/Eastern"),
                        Timestamp("20160101", tz="US/Eastern"),
                    ]
                )
            )
        )
        expected = DatetimeArray._from_sequence(
            np.array([Timestamp("2016-01-01", tz="US/Eastern")])
        )
        tm.assert_extension_array_equal(result, expected)

        result = pd.unique(
            Index(
                [
                    Timestamp("20160101", tz="US/Eastern"),
                    Timestamp("20160101", tz="US/Eastern"),
                ]
            )
        )
        expected = DatetimeIndex(
            ["2016-01-01 00:00:00"], dtype="datetime64[ns, US/Eastern]", freq=None
        )
        tm.assert_index_equal(result, expected)

    def test_order_of_appearance(self):
        # 9346
        # light testing of guarantee of order of appearance
        # these also are the doc-examples
        result = pd.unique(Series([2, 1, 3, 3]))
        tm.assert_numpy_array_equal(result, np.array([2, 1, 3], dtype="int64"))

        result = pd.unique(Series([2] + [1] * 5))
        tm.assert_numpy_array_equal(result, np.array([2, 1], dtype="int64"))

        result = pd.unique(Series([Timestamp("20160101"), Timestamp("20160101")]))
        expected = np.array(["2016-01-01T00:00:00.000000000"], dtype="datetime64[ns]")
        tm.assert_numpy_array_equal(result, expected)

        result = pd.unique(
            Index(
                [
                    Timestamp("20160101", tz="US/Eastern"),
                    Timestamp("20160101", tz="US/Eastern"),
                ]
            )
        )
        expected = DatetimeIndex(
            ["2016-01-01 00:00:00"], dtype="datetime64[ns, US/Eastern]", freq=None
        )
        tm.assert_index_equal(result, expected)

        result = pd.unique(list("aabc"))
        expected = np.array(["a", "b", "c"], dtype=object)
        tm.assert_numpy_array_equal(result, expected)

        result = pd.unique(Series(Categorical(list("aabc"))))
        expected = Categorical(list("abc"))
        tm.assert_categorical_equal(result, expected)

    @pytest.mark.parametrize(
        "arg ,expected",
        [
            (("1", "1", "2"), np.array(["1", "2"], dtype=object)),
            (("foo",), np.array(["foo"], dtype=object)),
        ],
    )
    def test_tuple_with_strings(self, arg, expected):
        # see GH 17108
        result = pd.unique(arg)
        tm.assert_numpy_array_equal(result, expected)

    def test_obj_none_preservation(self):
        # GH 20866
        arr = np.array(["foo", None], dtype=object)
        result = pd.unique(arr)
        expected = np.array(["foo", None], dtype=object)

        tm.assert_numpy_array_equal(result, expected, strict_nan=True)

    def test_signed_zero(self):
        # GH 21866
        a = np.array([-0.0, 0.0])
        result = pd.unique(a)
        expected = np.array([-0.0])  # 0.0 and -0.0 are equivalent
        tm.assert_numpy_array_equal(result, expected)

    def test_different_nans(self):
        # GH 21866
        # create different nans from bit-patterns:
        NAN1 = struct.unpack("d", struct.pack("=Q", 0x7FF8000000000000))[0]
        NAN2 = struct.unpack("d", struct.pack("=Q", 0x7FF8000000000001))[0]
        assert NAN1 != NAN1
        assert NAN2 != NAN2
        a = np.array([NAN1, NAN2])  # NAN1 and NAN2 are equivalent
        result = pd.unique(a)
        expected = np.array([np.nan])
        tm.assert_numpy_array_equal(result, expected)

    def test_first_nan_kept(self):
        # GH 22295
        # create different nans from bit-patterns:
        bits_for_nan1 = 0xFFF8000000000001
        bits_for_nan2 = 0x7FF8000000000001
        NAN1 = struct.unpack("d", struct.pack("=Q", bits_for_nan1))[0]
        NAN2 = struct.unpack("d", struct.pack("=Q", bits_for_nan2))[0]
        assert NAN1 != NAN1
        assert NAN2 != NAN2
        for el_type in [np.float64, object]:
            a = np.array([NAN1, NAN2], dtype=el_type)
            result = pd.unique(a)
            assert result.size == 1
            # use bit patterns to identify which nan was kept:
            result_nan_bits = struct.unpack("=Q", struct.pack("d", result[0]))[0]
            assert result_nan_bits == bits_for_nan1

    def test_do_not_mangle_na_values(self, unique_nulls_fixture, unique_nulls_fixture2):
        # GH 22295
        if unique_nulls_fixture is unique_nulls_fixture2:
            return  # skip it, values not unique
        a = np.array([unique_nulls_fixture, unique_nulls_fixture2], dtype=object)
        result = pd.unique(a)
        assert result.size == 2
        assert a[0] is unique_nulls_fixture
        assert a[1] is unique_nulls_fixture2


class TestIsin:
    def test_invalid(self):

        msg = (
            r"only list-like objects are allowed to be passed to isin\(\), "
            r"you passed a \[int\]"
        )
        with pytest.raises(TypeError, match=msg):
            algos.isin(1, 1)
        with pytest.raises(TypeError, match=msg):
            algos.isin(1, [1])
        with pytest.raises(TypeError, match=msg):
            algos.isin([1], 1)

    def test_basic(self):

        result = algos.isin([1, 2], [1])
        expected = np.array([True, False])
        tm.assert_numpy_array_equal(result, expected)

        result = algos.isin(np.array([1, 2]), [1])
        expected = np.array([True, False])
        tm.assert_numpy_array_equal(result, expected)

        result = algos.isin(Series([1, 2]), [1])
        expected = np.array([True, False])
        tm.assert_numpy_array_equal(result, expected)

        result = algos.isin(Series([1, 2]), Series([1]))
        expected = np.array([True, False])
        tm.assert_numpy_array_equal(result, expected)

        result = algos.isin(Series([1, 2]), {1})
        expected = np.array([True, False])
        tm.assert_numpy_array_equal(result, expected)

        result = algos.isin(["a", "b"], ["a"])
        expected = np.array([True, False])
        tm.assert_numpy_array_equal(result, expected)

        result = algos.isin(Series(["a", "b"]), Series(["a"]))
        expected = np.array([True, False])
        tm.assert_numpy_array_equal(result, expected)

        result = algos.isin(Series(["a", "b"]), {"a"})
        expected = np.array([True, False])
        tm.assert_numpy_array_equal(result, expected)

        result = algos.isin(["a", "b"], [1])
        expected = np.array([False, False])
        tm.assert_numpy_array_equal(result, expected)

    def test_i8(self):

        arr = date_range("20130101", periods=3).values
        result = algos.isin(arr, [arr[0]])
        expected = np.array([True, False, False])
        tm.assert_numpy_array_equal(result, expected)

        result = algos.isin(arr, arr[0:2])
        expected = np.array([True, True, False])
        tm.assert_numpy_array_equal(result, expected)

        result = algos.isin(arr, set(arr[0:2]))
        expected = np.array([True, True, False])
        tm.assert_numpy_array_equal(result, expected)

        arr = timedelta_range("1 day", periods=3).values
        result = algos.isin(arr, [arr[0]])
        expected = np.array([True, False, False])
        tm.assert_numpy_array_equal(result, expected)

        result = algos.isin(arr, arr[0:2])
        expected = np.array([True, True, False])
        tm.assert_numpy_array_equal(result, expected)

        result = algos.isin(arr, set(arr[0:2]))
        expected = np.array([True, True, False])
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize("dtype1", ["m8[ns]", "M8[ns]", "M8[ns, UTC]", "period[D]"])
    @pytest.mark.parametrize("dtype", ["i8", "f8", "u8"])
    def test_isin_datetimelike_values_numeric_comps(self, dtype, dtype1):
        # Anything but object and we get all-False shortcut

        dta = date_range("2013-01-01", periods=3)._values
        if dtype1 == "period[D]":
            # TODO: fix Series.view to get this on its own
            arr = dta.to_period("D")
        elif dtype1 == "M8[ns, UTC]":
            # TODO: fix Series.view to get this on its own
            arr = dta.tz_localize("UTC")
        else:
            arr = Series(dta.view("i8")).view(dtype1)._values

        comps = arr.view("i8").astype(dtype)

        result = algos.isin(comps, arr)
        expected = np.zeros(comps.shape, dtype=bool)
        tm.assert_numpy_array_equal(result, expected)

    def test_large(self):
        s = date_range("20000101", periods=2000000, freq="s").values
        result = algos.isin(s, s[0:2])
        expected = np.zeros(len(s), dtype=bool)
        expected[0] = True
        expected[1] = True
        tm.assert_numpy_array_equal(result, expected)

    def test_categorical_from_codes(self):
        # GH 16639
        vals = np.array([0, 1, 2, 0])
        cats = ["a", "b", "c"]
        Sd = Series(Categorical([1]).from_codes(vals, cats))
        St = Series(Categorical([1]).from_codes(np.array([0, 1]), cats))
        expected = np.array([True, True, False, True])
        result = algos.isin(Sd, St)
        tm.assert_numpy_array_equal(expected, result)

    def test_categorical_isin(self):
        vals = np.array([0, 1, 2, 0])
        cats = ["a", "b", "c"]
        cat = Categorical([1]).from_codes(vals, cats)
        other = Categorical([1]).from_codes(np.array([0, 1]), cats)

        expected = np.array([True, True, False, True])
        result = algos.isin(cat, other)
        tm.assert_numpy_array_equal(expected, result)

    def test_same_nan_is_in(self):
        # GH 22160
        # nan is special, because from " a is b" doesn't follow "a == b"
        # at least, isin() should follow python's "np.nan in [nan] == True"
        # casting to -> np.float64 -> another float-object somewhere on
        # the way could lead jepardize this behavior
        comps = [np.nan]  # could be casted to float64
        values = [np.nan]
        expected = np.array([True])
        result = algos.isin(comps, values)
        tm.assert_numpy_array_equal(expected, result)

    def test_same_nan_is_in_large(self):
        # https://github.com/pandas-dev/pandas/issues/22205
        s = np.tile(1.0, 1_000_001)
        s[0] = np.nan
        result = algos.isin(s, [np.nan, 1])
        expected = np.ones(len(s), dtype=bool)
        tm.assert_numpy_array_equal(result, expected)

    def test_same_nan_is_in_large_series(self):
        # https://github.com/pandas-dev/pandas/issues/22205
        s = np.tile(1.0, 1_000_001)
        series = Series(s)
        s[0] = np.nan
        result = series.isin([np.nan, 1])
        expected = Series(np.ones(len(s), dtype=bool))
        tm.assert_series_equal(result, expected)

    def test_same_object_is_in(self):
        # GH 22160
        # there could be special treatment for nans
        # the user however could define a custom class
        # with similar behavior, then we at least should
        # fall back to usual python's behavior: "a in [a] == True"
        class LikeNan:
            def __eq__(self, other) -> bool:
                return False

            def __hash__(self):
                return 0

        a, b = LikeNan(), LikeNan()
        # same object -> True
        tm.assert_numpy_array_equal(algos.isin([a], [a]), np.array([True]))
        # different objects -> False
        tm.assert_numpy_array_equal(algos.isin([a], [b]), np.array([False]))

    def test_different_nans(self):
        # GH 22160
        # all nans are handled as equivalent

        comps = [float("nan")]
        values = [float("nan")]
        assert comps[0] is not values[0]  # different nan-objects

        # as list of python-objects:
        result = algos.isin(comps, values)
        tm.assert_numpy_array_equal(np.array([True]), result)

        # as object-array:
        result = algos.isin(
            np.asarray(comps, dtype=object), np.asarray(values, dtype=object)
        )
        tm.assert_numpy_array_equal(np.array([True]), result)

        # as float64-array:
        result = algos.isin(
            np.asarray(comps, dtype=np.float64), np.asarray(values, dtype=np.float64)
        )
        tm.assert_numpy_array_equal(np.array([True]), result)

    def test_no_cast(self):
        # GH 22160
        # ensure 42 is not casted to a string
        comps = ["ss", 42]
        values = ["42"]
        expected = np.array([False, False])
        result = algos.isin(comps, values)
        tm.assert_numpy_array_equal(expected, result)

    @pytest.mark.parametrize("empty", [[], Series(dtype=object), np.array([])])
    def test_empty(self, empty):
        # see gh-16991
        vals = Index(["a", "b"])
        expected = np.array([False, False])

        result = algos.isin(vals, empty)
        tm.assert_numpy_array_equal(expected, result)

    def test_different_nan_objects(self):
        # GH 22119
        comps = np.array(["nan", np.nan * 1j, float("nan")], dtype=object)
        vals = np.array([float("nan")], dtype=object)
        expected = np.array([False, False, True])
        result = algos.isin(comps, vals)
        tm.assert_numpy_array_equal(expected, result)

    def test_different_nans_as_float64(self):
        # GH 21866
        # create different nans from bit-patterns,
        # these nans will land in different buckets in the hash-table
        # if no special care is taken
        NAN1 = struct.unpack("d", struct.pack("=Q", 0x7FF8000000000000))[0]
        NAN2 = struct.unpack("d", struct.pack("=Q", 0x7FF8000000000001))[0]
        assert NAN1 != NAN1
        assert NAN2 != NAN2

        # check that NAN1 and NAN2 are equivalent:
        arr = np.array([NAN1, NAN2], dtype=np.float64)
        lookup1 = np.array([NAN1], dtype=np.float64)
        result = algos.isin(arr, lookup1)
        expected = np.array([True, True])
        tm.assert_numpy_array_equal(result, expected)

        lookup2 = np.array([NAN2], dtype=np.float64)
        result = algos.isin(arr, lookup2)
        expected = np.array([True, True])
        tm.assert_numpy_array_equal(result, expected)

    def test_isin_int_df_string_search(self):
        """Comparing df with int`s (1,2) with a string at isin() ("1")
        -> should not match values because int 1 is not equal str 1"""
        df = DataFrame({"values": [1, 2]})
        result = df.isin(["1"])
        expected_false = DataFrame({"values": [False, False]})
        tm.assert_frame_equal(result, expected_false)

    def test_isin_nan_df_string_search(self):
        """Comparing df with nan value (np.nan,2) with a string at isin() ("NaN")
        -> should not match values because np.nan is not equal str NaN"""
        df = DataFrame({"values": [np.nan, 2]})
        result = df.isin(["NaN"])
        expected_false = DataFrame({"values": [False, False]})
        tm.assert_frame_equal(result, expected_false)

    def test_isin_float_df_string_search(self):
        """Comparing df with floats (1.4245,2.32441) with a string at isin() ("1.4245")
        -> should not match values because float 1.4245 is not equal str 1.4245"""
        df = DataFrame({"values": [1.4245, 2.32441]})
        result = df.isin(["1.4245"])
        expected_false = DataFrame({"values": [False, False]})
        tm.assert_frame_equal(result, expected_false)


class TestValueCounts:
    def test_value_counts(self):
        np.random.seed(1234)
        from pandas.core.reshape.tile import cut

        arr = np.random.randn(4)
        factor = cut(arr, 4)

        # assert isinstance(factor, n)
        result = algos.value_counts(factor)
        breaks = [-1.194, -0.535, 0.121, 0.777, 1.433]
        index = IntervalIndex.from_breaks(breaks).astype(CDT(ordered=True))
        expected = Series([1, 1, 1, 1], index=index)
        tm.assert_series_equal(result.sort_index(), expected.sort_index())

    def test_value_counts_bins(self):
        s = [1, 2, 3, 4]
        result = algos.value_counts(s, bins=1)
        expected = Series([4], index=IntervalIndex.from_tuples([(0.996, 4.0)]))
        tm.assert_series_equal(result, expected)

        result = algos.value_counts(s, bins=2, sort=False)
        expected = Series(
            [2, 2], index=IntervalIndex.from_tuples([(0.996, 2.5), (2.5, 4.0)])
        )
        tm.assert_series_equal(result, expected)

    def test_value_counts_dtypes(self):
        result = algos.value_counts([1, 1.0])
        assert len(result) == 1

        result = algos.value_counts([1, 1.0], bins=1)
        assert len(result) == 1

        result = algos.value_counts(Series([1, 1.0, "1"]))  # object
        assert len(result) == 2

        msg = "bins argument only works with numeric data"
        with pytest.raises(TypeError, match=msg):
            algos.value_counts(["1", 1], bins=1)

    def test_value_counts_nat(self):
        td = Series([np.timedelta64(10000), NaT], dtype="timedelta64[ns]")
        dt = to_datetime(["NaT", "2014-01-01"])

        for s in [td, dt]:
            vc = algos.value_counts(s)
            vc_with_na = algos.value_counts(s, dropna=False)
            assert len(vc) == 1
            assert len(vc_with_na) == 2

        exp_dt = Series({Timestamp("2014-01-01 00:00:00"): 1})
        tm.assert_series_equal(algos.value_counts(dt), exp_dt)
        # TODO same for (timedelta)

    def test_value_counts_datetime_outofbounds(self):
        # GH 13663
        s = Series(
            [
                datetime(3000, 1, 1),
                datetime(5000, 1, 1),
                datetime(5000, 1, 1),
                datetime(6000, 1, 1),
                datetime(3000, 1, 1),
                datetime(3000, 1, 1),
            ]
        )
        res = s.value_counts()

        exp_index = Index(
            [datetime(3000, 1, 1), datetime(5000, 1, 1), datetime(6000, 1, 1)],
            dtype=object,
        )
        exp = Series([3, 2, 1], index=exp_index)
        tm.assert_series_equal(res, exp)

        # GH 12424
        res = to_datetime(Series(["2362-01-01", np.nan]), errors="ignore")
        exp = Series(["2362-01-01", np.nan], dtype=object)
        tm.assert_series_equal(res, exp)

    def test_categorical(self):
        s = Series(Categorical(list("aaabbc")))
        result = s.value_counts()
        expected = Series([3, 2, 1], index=CategoricalIndex(["a", "b", "c"]))

        tm.assert_series_equal(result, expected, check_index_type=True)

        # preserve order?
        s = s.cat.as_ordered()
        result = s.value_counts()
        expected.index = expected.index.as_ordered()
        tm.assert_series_equal(result, expected, check_index_type=True)

    def test_categorical_nans(self):
        s = Series(Categorical(list("aaaaabbbcc")))  # 4,3,2,1 (nan)
        s.iloc[1] = np.nan
        result = s.value_counts()
        expected = Series(
            [4, 3, 2],
            index=CategoricalIndex(["a", "b", "c"], categories=["a", "b", "c"]),
        )
        tm.assert_series_equal(result, expected, check_index_type=True)
        result = s.value_counts(dropna=False)
        expected = Series([4, 3, 2, 1], index=CategoricalIndex(["a", "b", "c", np.nan]))
        tm.assert_series_equal(result, expected, check_index_type=True)

        # out of order
        s = Series(
            Categorical(list("aaaaabbbcc"), ordered=True, categories=["b", "a", "c"])
        )
        s.iloc[1] = np.nan
        result = s.value_counts()
        expected = Series(
            [4, 3, 2],
            index=CategoricalIndex(
                ["a", "b", "c"], categories=["b", "a", "c"], ordered=True
            ),
        )
        tm.assert_series_equal(result, expected, check_index_type=True)

        result = s.value_counts(dropna=False)
        expected = Series(
            [4, 3, 2, 1],
            index=CategoricalIndex(
                ["a", "b", "c", np.nan], categories=["b", "a", "c"], ordered=True
            ),
        )
        tm.assert_series_equal(result, expected, check_index_type=True)

    def test_categorical_zeroes(self):
        # keep the `d` category with 0
        s = Series(Categorical(list("bbbaac"), categories=list("abcd"), ordered=True))
        result = s.value_counts()
        expected = Series(
            [3, 2, 1, 0],
            index=Categorical(
                ["b", "a", "c", "d"], categories=list("abcd"), ordered=True
            ),
        )
        tm.assert_series_equal(result, expected, check_index_type=True)

    def test_dropna(self):
        # https://github.com/pandas-dev/pandas/issues/9443#issuecomment-73719328

        tm.assert_series_equal(
            Series([True, True, False]).value_counts(dropna=True),
            Series([2, 1], index=[True, False]),
        )
        tm.assert_series_equal(
            Series([True, True, False]).value_counts(dropna=False),
            Series([2, 1], index=[True, False]),
        )

        tm.assert_series_equal(
            Series([True] * 3 + [False] * 2 + [None] * 5).value_counts(dropna=True),
            Series([3, 2], index=[True, False]),
        )
        tm.assert_series_equal(
            Series([True] * 5 + [False] * 3 + [None] * 2).value_counts(dropna=False),
            Series([5, 3, 2], index=[True, False, np.nan]),
        )
        tm.assert_series_equal(
            Series([10.3, 5.0, 5.0]).value_counts(dropna=True),
            Series([2, 1], index=[5.0, 10.3]),
        )
        tm.assert_series_equal(
            Series([10.3, 5.0, 5.0]).value_counts(dropna=False),
            Series([2, 1], index=[5.0, 10.3]),
        )

        tm.assert_series_equal(
            Series([10.3, 5.0, 5.0, None]).value_counts(dropna=True),
            Series([2, 1], index=[5.0, 10.3]),
        )

        result = Series([10.3, 10.3, 5.0, 5.0, 5.0, None]).value_counts(dropna=False)
        expected = Series([3, 2, 1], index=[5.0, 10.3, np.nan])
        tm.assert_series_equal(result, expected)

    def test_value_counts_normalized(self):
        # GH12558
        s = Series([1] * 2 + [2] * 3 + [np.nan] * 5)
        dtypes = (np.float64, object, "M8[ns]")
        for t in dtypes:
            s_typed = s.astype(t)
            result = s_typed.value_counts(normalize=True, dropna=False)
            expected = Series(
                [0.5, 0.3, 0.2], index=Series([np.nan, 2.0, 1.0], dtype=t)
            )
            tm.assert_series_equal(result, expected)

            result = s_typed.value_counts(normalize=True, dropna=True)
            expected = Series([0.6, 0.4], index=Series([2.0, 1.0], dtype=t))
            tm.assert_series_equal(result, expected)

    def test_value_counts_uint64(self):
        arr = np.array([2 ** 63], dtype=np.uint64)
        expected = Series([1], index=[2 ** 63])
        result = algos.value_counts(arr)

        tm.assert_series_equal(result, expected)

        arr = np.array([-1, 2 ** 63], dtype=object)
        expected = Series([1, 1], index=[-1, 2 ** 63])
        result = algos.value_counts(arr)

        tm.assert_series_equal(result, expected)


class TestDuplicated:
    def test_duplicated_with_nas(self):
        keys = np.array([0, 1, np.nan, 0, 2, np.nan], dtype=object)

        result = algos.duplicated(keys)
        expected = np.array([False, False, False, True, False, True])
        tm.assert_numpy_array_equal(result, expected)

        result = algos.duplicated(keys, keep="first")
        expected = np.array([False, False, False, True, False, True])
        tm.assert_numpy_array_equal(result, expected)

        result = algos.duplicated(keys, keep="last")
        expected = np.array([True, False, True, False, False, False])
        tm.assert_numpy_array_equal(result, expected)

        result = algos.duplicated(keys, keep=False)
        expected = np.array([True, False, True, True, False, True])
        tm.assert_numpy_array_equal(result, expected)

        keys = np.empty(8, dtype=object)
        for i, t in enumerate(
            zip([0, 0, np.nan, np.nan] * 2, [0, np.nan, 0, np.nan] * 2)
        ):
            keys[i] = t

        result = algos.duplicated(keys)
        falses = [False] * 4
        trues = [True] * 4
        expected = np.array(falses + trues)
        tm.assert_numpy_array_equal(result, expected)

        result = algos.duplicated(keys, keep="last")
        expected = np.array(trues + falses)
        tm.assert_numpy_array_equal(result, expected)

        result = algos.duplicated(keys, keep=False)
        expected = np.array(trues + trues)
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize(
        "case",
        [
            np.array([1, 2, 1, 5, 3, 2, 4, 1, 5, 6]),
            np.array([1.1, 2.2, 1.1, np.nan, 3.3, 2.2, 4.4, 1.1, np.nan, 6.6]),
            np.array(
                [
                    1 + 1j,
                    2 + 2j,
                    1 + 1j,
                    5 + 5j,
                    3 + 3j,
                    2 + 2j,
                    4 + 4j,
                    1 + 1j,
                    5 + 5j,
                    6 + 6j,
                ]
            ),
            np.array(["a", "b", "a", "e", "c", "b", "d", "a", "e", "f"], dtype=object),
            np.array(
                [1, 2 ** 63, 1, 3 ** 5, 10, 2 ** 63, 39, 1, 3 ** 5, 7], dtype=np.uint64
            ),
        ],
    )
    def test_numeric_object_likes(self, case):
        exp_first = np.array(
            [False, False, True, False, False, True, False, True, True, False]
        )
        exp_last = np.array(
            [True, True, True, True, False, False, False, False, False, False]
        )
        exp_false = exp_first | exp_last

        res_first = algos.duplicated(case, keep="first")
        tm.assert_numpy_array_equal(res_first, exp_first)

        res_last = algos.duplicated(case, keep="last")
        tm.assert_numpy_array_equal(res_last, exp_last)

        res_false = algos.duplicated(case, keep=False)
        tm.assert_numpy_array_equal(res_false, exp_false)

        # index
        for idx in [Index(case), Index(case, dtype="category")]:
            res_first = idx.duplicated(keep="first")
            tm.assert_numpy_array_equal(res_first, exp_first)

            res_last = idx.duplicated(keep="last")
            tm.assert_numpy_array_equal(res_last, exp_last)

            res_false = idx.duplicated(keep=False)
            tm.assert_numpy_array_equal(res_false, exp_false)

        # series
        for s in [Series(case), Series(case, dtype="category")]:
            res_first = s.duplicated(keep="first")
            tm.assert_series_equal(res_first, Series(exp_first))

            res_last = s.duplicated(keep="last")
            tm.assert_series_equal(res_last, Series(exp_last))

            res_false = s.duplicated(keep=False)
            tm.assert_series_equal(res_false, Series(exp_false))

    def test_datetime_likes(self):

        dt = [
            "2011-01-01",
            "2011-01-02",
            "2011-01-01",
            "NaT",
            "2011-01-03",
            "2011-01-02",
            "2011-01-04",
            "2011-01-01",
            "NaT",
            "2011-01-06",
        ]
        td = [
            "1 days",
            "2 days",
            "1 days",
            "NaT",
            "3 days",
            "2 days",
            "4 days",
            "1 days",
            "NaT",
            "6 days",
        ]

        cases = [
            np.array([Timestamp(d) for d in dt]),
            np.array([Timestamp(d, tz="US/Eastern") for d in dt]),
            np.array([Period(d, freq="D") for d in dt]),
            np.array([np.datetime64(d) for d in dt]),
            np.array([Timedelta(d) for d in td]),
        ]

        exp_first = np.array(
            [False, False, True, False, False, True, False, True, True, False]
        )
        exp_last = np.array(
            [True, True, True, True, False, False, False, False, False, False]
        )
        exp_false = exp_first | exp_last

        for case in cases:
            res_first = algos.duplicated(case, keep="first")
            tm.assert_numpy_array_equal(res_first, exp_first)

            res_last = algos.duplicated(case, keep="last")
            tm.assert_numpy_array_equal(res_last, exp_last)

            res_false = algos.duplicated(case, keep=False)
            tm.assert_numpy_array_equal(res_false, exp_false)

            # index
            for idx in [
                Index(case),
                Index(case, dtype="category"),
                Index(case, dtype=object),
            ]:
                res_first = idx.duplicated(keep="first")
                tm.assert_numpy_array_equal(res_first, exp_first)

                res_last = idx.duplicated(keep="last")
                tm.assert_numpy_array_equal(res_last, exp_last)

                res_false = idx.duplicated(keep=False)
                tm.assert_numpy_array_equal(res_false, exp_false)

            # series
            for s in [
                Series(case),
                Series(case, dtype="category"),
                Series(case, dtype=object),
            ]:
                res_first = s.duplicated(keep="first")
                tm.assert_series_equal(res_first, Series(exp_first))

                res_last = s.duplicated(keep="last")
                tm.assert_series_equal(res_last, Series(exp_last))

                res_false = s.duplicated(keep=False)
                tm.assert_series_equal(res_false, Series(exp_false))

    def test_unique_index(self):
        cases = [Index([1, 2, 3]), pd.RangeIndex(0, 3)]
        for case in cases:
            assert case.is_unique is True
            tm.assert_numpy_array_equal(
                case.duplicated(), np.array([False, False, False])
            )

    @pytest.mark.parametrize(
        "arr, uniques",
        [
            (
                [(0, 0), (0, 1), (1, 0), (1, 1), (0, 0), (0, 1), (1, 0), (1, 1)],
                [(0, 0), (0, 1), (1, 0), (1, 1)],
            ),
            (
                [("b", "c"), ("a", "b"), ("a", "b"), ("b", "c")],
                [("b", "c"), ("a", "b")],
            ),
            ([("a", 1), ("b", 2), ("a", 3), ("a", 1)], [("a", 1), ("b", 2), ("a", 3)]),
        ],
    )
    def test_unique_tuples(self, arr, uniques):
        # https://github.com/pandas-dev/pandas/issues/16519
        expected = np.empty(len(uniques), dtype=object)
        expected[:] = uniques

        result = pd.unique(arr)
        tm.assert_numpy_array_equal(result, expected)


class TestHashTable:
    def test_string_hashtable_set_item_signature(self):
        # GH#30419 fix typing in StringHashTable.set_item to prevent segfault
        tbl = ht.StringHashTable()

        tbl.set_item("key", 1)
        assert tbl.get_item("key") == 1

        with pytest.raises(TypeError, match="'key' has incorrect type"):
            # key arg typed as string, not object
            tbl.set_item(4, 6)
        with pytest.raises(TypeError, match="'val' has incorrect type"):
            tbl.get_item(4)

    def test_lookup_nan(self, writable):
        xs = np.array([2.718, 3.14, np.nan, -7, 5, 2, 3])
        # GH 21688 ensure we can deal with readonly memory views
        xs.setflags(write=writable)
        m = ht.Float64HashTable()
        m.map_locations(xs)
        tm.assert_numpy_array_equal(m.lookup(xs), np.arange(len(xs), dtype=np.intp))

    def test_add_signed_zeros(self):
        # GH 21866 inconsistent hash-function for float64
        # default hash-function would lead to different hash-buckets
        # for 0.0 and -0.0 if there are more than 2^30 hash-buckets
        # but this would mean 16GB
        N = 4  # 12 * 10**8 would trigger the error, if you have enough memory
        m = ht.Float64HashTable(N)
        m.set_item(0.0, 0)
        m.set_item(-0.0, 0)
        assert len(m) == 1  # 0.0 and -0.0 are equivalent

    def test_add_different_nans(self):
        # GH 21866 inconsistent hash-function for float64
        # create different nans from bit-patterns:
        NAN1 = struct.unpack("d", struct.pack("=Q", 0x7FF8000000000000))[0]
        NAN2 = struct.unpack("d", struct.pack("=Q", 0x7FF8000000000001))[0]
        assert NAN1 != NAN1
        assert NAN2 != NAN2
        # default hash function would lead to different hash-buckets
        # for NAN1 and NAN2 even if there are only 4 buckets:
        m = ht.Float64HashTable()
        m.set_item(NAN1, 0)
        m.set_item(NAN2, 0)
        assert len(m) == 1  # NAN1 and NAN2 are equivalent

    def test_lookup_overflow(self, writable):
        xs = np.array([1, 2, 2 ** 63], dtype=np.uint64)
        # GH 21688 ensure we can deal with readonly memory views
        xs.setflags(write=writable)
        m = ht.UInt64HashTable()
        m.map_locations(xs)
        tm.assert_numpy_array_equal(m.lookup(xs), np.arange(len(xs), dtype=np.intp))

    def test_get_unique(self):
        s = Series([1, 2, 2 ** 63, 2 ** 63], dtype=np.uint64)
        exp = np.array([1, 2, 2 ** 63], dtype=np.uint64)
        tm.assert_numpy_array_equal(s.unique(), exp)

    @pytest.mark.parametrize("nvals", [0, 10])  # resizing to 0 is special case
    @pytest.mark.parametrize(
        "htable, uniques, dtype, safely_resizes",
        [
            (ht.PyObjectHashTable, ht.ObjectVector, "object", False),
            (ht.StringHashTable, ht.ObjectVector, "object", True),
            (ht.Float64HashTable, ht.Float64Vector, "float64", False),
            (ht.Int64HashTable, ht.Int64Vector, "int64", False),
            (ht.Int32HashTable, ht.Int32Vector, "int32", False),
            (ht.UInt64HashTable, ht.UInt64Vector, "uint64", False),
        ],
    )
    def test_vector_resize(
        self, writable, htable, uniques, dtype, safely_resizes, nvals
    ):
        # Test for memory errors after internal vector
        # reallocations (GH 7157)
        vals = np.array(np.random.randn(1000), dtype=dtype)

        # GH 21688 ensures we can deal with read-only memory views
        vals.setflags(write=writable)

        # initialise instances; cannot initialise in parametrization,
        # as otherwise external views would be held on the array (which is
        # one of the things this test is checking)
        htable = htable()
        uniques = uniques()

        # get_labels may append to uniques
        htable.get_labels(vals[:nvals], uniques, 0, -1)
        # to_array() sets an external_view_exists flag on uniques.
        tmp = uniques.to_array()
        oldshape = tmp.shape

        # subsequent get_labels() calls can no longer append to it
        # (except for StringHashTables + ObjectVector)
        if safely_resizes:
            htable.get_labels(vals, uniques, 0, -1)
        else:
            with pytest.raises(ValueError, match="external reference.*"):
                htable.get_labels(vals, uniques, 0, -1)

        uniques.to_array()  # should not raise here
        assert tmp.shape == oldshape

    @pytest.mark.parametrize(
        "htable, tm_dtype",
        [
            (ht.PyObjectHashTable, "String"),
            (ht.StringHashTable, "String"),
            (ht.Float64HashTable, "Float"),
            (ht.Int64HashTable, "Int"),
            (ht.UInt64HashTable, "UInt"),
        ],
    )
    def test_hashtable_unique(self, htable, tm_dtype, writable):
        # output of maker has guaranteed unique elements
        maker = getattr(tm, "make" + tm_dtype + "Index")
        s = Series(maker(1000))
        if htable == ht.Float64HashTable:
            # add NaN for float column
            s.loc[500] = np.nan
        elif htable == ht.PyObjectHashTable:
            # use different NaN types for object column
            s.loc[500:502] = [np.nan, None, NaT]

        # create duplicated selection
        s_duplicated = s.sample(frac=3, replace=True).reset_index(drop=True)
        s_duplicated.values.setflags(write=writable)

        # drop_duplicates has own cython code (hash_table_func_helper.pxi)
        # and is tested separately; keeps first occurrence like ht.unique()
        expected_unique = s_duplicated.drop_duplicates(keep="first").values
        result_unique = htable().unique(s_duplicated.values)
        tm.assert_numpy_array_equal(result_unique, expected_unique)

        # test return_inverse=True
        # reconstruction can only succeed if the inverse is correct
        result_unique, result_inverse = htable().unique(
            s_duplicated.values, return_inverse=True
        )
        tm.assert_numpy_array_equal(result_unique, expected_unique)
        reconstr = result_unique[result_inverse]
        tm.assert_numpy_array_equal(reconstr, s_duplicated.values)

    @pytest.mark.parametrize(
        "htable, tm_dtype",
        [
            (ht.PyObjectHashTable, "String"),
            (ht.StringHashTable, "String"),
            (ht.Float64HashTable, "Float"),
            (ht.Int64HashTable, "Int"),
            (ht.UInt64HashTable, "UInt"),
        ],
    )
    def test_hashtable_factorize(self, htable, tm_dtype, writable):
        # output of maker has guaranteed unique elements
        maker = getattr(tm, "make" + tm_dtype + "Index")
        s = Series(maker(1000))
        if htable == ht.Float64HashTable:
            # add NaN for float column
            s.loc[500] = np.nan
        elif htable == ht.PyObjectHashTable:
            # use different NaN types for object column
            s.loc[500:502] = [np.nan, None, NaT]

        # create duplicated selection
        s_duplicated = s.sample(frac=3, replace=True).reset_index(drop=True)
        s_duplicated.values.setflags(write=writable)
        na_mask = s_duplicated.isna().values

        result_unique, result_inverse = htable().factorize(s_duplicated.values)

        # drop_duplicates has own cython code (hash_table_func_helper.pxi)
        # and is tested separately; keeps first occurrence like ht.factorize()
        # since factorize removes all NaNs, we do the same here
        expected_unique = s_duplicated.dropna().drop_duplicates().values
        tm.assert_numpy_array_equal(result_unique, expected_unique)

        # reconstruction can only succeed if the inverse is correct. Since
        # factorize removes the NaNs, those have to be excluded here as well
        result_reconstruct = result_unique[result_inverse[~na_mask]]
        expected_reconstruct = s_duplicated.dropna().values
        tm.assert_numpy_array_equal(result_reconstruct, expected_reconstruct)

    @pytest.mark.parametrize(
        "hashtable",
        [
            ht.PyObjectHashTable,
            ht.StringHashTable,
            ht.Float64HashTable,
            ht.Int64HashTable,
            ht.Int32HashTable,
            ht.UInt64HashTable,
        ],
    )
    def test_hashtable_large_sizehint(self, hashtable):
        # GH 22729
        size_hint = np.iinfo(np.uint32).max + 1
        tbl = hashtable(size_hint=size_hint)  # noqa


def test_quantile():
    s = Series(np.random.randn(100))

    result = algos.quantile(s, [0, 0.25, 0.5, 0.75, 1.0])
    expected = algos.quantile(s.values, [0, 0.25, 0.5, 0.75, 1.0])
    tm.assert_almost_equal(result, expected)


def test_unique_label_indices():

    a = np.random.randint(1, 1 << 10, 1 << 15).astype("int64")

    left = ht.unique_label_indices(a)
    right = np.unique(a, return_index=True)[1]

    tm.assert_numpy_array_equal(left, right, check_dtype=False)

    a[np.random.choice(len(a), 10)] = -1
    left = ht.unique_label_indices(a)
    right = np.unique(a, return_index=True)[1][1:]
    tm.assert_numpy_array_equal(left, right, check_dtype=False)


class TestRank:
    @td.skip_if_no_scipy
    def test_scipy_compat(self):
        from scipy.stats import rankdata

        def _check(arr):
            mask = ~np.isfinite(arr)
            arr = arr.copy()
            result = libalgos.rank_1d(arr, labels=np.zeros(len(arr), dtype=np.intp))
            arr[mask] = np.inf
            exp = rankdata(arr)
            exp[mask] = np.nan
            tm.assert_almost_equal(result, exp)

        _check(np.array([np.nan, np.nan, 5.0, 5.0, 5.0, np.nan, 1, 2, 3, np.nan]))
        _check(np.array([4.0, np.nan, 5.0, 5.0, 5.0, np.nan, 1, 2, 4.0, np.nan]))

    @pytest.mark.parametrize("dtype", np.typecodes["AllInteger"])
    def test_basic(self, writable, dtype):
        exp = np.array([1, 2], dtype=np.float64)

        data = np.array([1, 100], dtype=dtype)
        data.setflags(write=writable)
        ser = Series(data)
        result = algos.rank(ser)
        tm.assert_numpy_array_equal(result, exp)

    def test_uint64_overflow(self):
        exp = np.array([1, 2], dtype=np.float64)

        for dtype in [np.float64, np.uint64]:
            s = Series([1, 2 ** 63], dtype=dtype)
            tm.assert_numpy_array_equal(algos.rank(s), exp)

    def test_too_many_ndims(self):
        arr = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
        msg = "Array with ndim > 2 are not supported"

        with pytest.raises(TypeError, match=msg):
            algos.rank(arr)

    @pytest.mark.single
    @pytest.mark.high_memory
    @pytest.mark.parametrize(
        "values",
        [np.arange(2 ** 24 + 1), np.arange(2 ** 25 + 2).reshape(2 ** 24 + 1, 2)],
        ids=["1d", "2d"],
    )
    def test_pct_max_many_rows(self, values):
        # GH 18271
        result = algos.rank(values, pct=True).max()
        assert result == 1


def test_pad_backfill_object_segfault():

    old = np.array([], dtype="O")
    new = np.array([datetime(2010, 12, 31)], dtype="O")

    result = libalgos.pad["object"](old, new)
    expected = np.array([-1], dtype=np.intp)
    tm.assert_numpy_array_equal(result, expected)

    result = libalgos.pad["object"](new, old)
    expected = np.array([], dtype=np.intp)
    tm.assert_numpy_array_equal(result, expected)

    result = libalgos.backfill["object"](old, new)
    expected = np.array([-1], dtype=np.intp)
    tm.assert_numpy_array_equal(result, expected)

    result = libalgos.backfill["object"](new, old)
    expected = np.array([], dtype=np.intp)
    tm.assert_numpy_array_equal(result, expected)


class TestTseriesUtil:
    def test_combineFunc(self):
        pass

    def test_reindex(self):
        pass

    def test_isna(self):
        pass

    def test_groupby(self):
        pass

    def test_groupby_withnull(self):
        pass

    def test_backfill(self):
        old = Index([1, 5, 10])
        new = Index(list(range(12)))

        filler = libalgos.backfill["int64_t"](old.values, new.values)

        expect_filler = np.array([0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, -1], dtype=np.intp)
        tm.assert_numpy_array_equal(filler, expect_filler)

        # corner case
        old = Index([1, 4])
        new = Index(list(range(5, 10)))
        filler = libalgos.backfill["int64_t"](old.values, new.values)

        expect_filler = np.array([-1, -1, -1, -1, -1], dtype=np.intp)
        tm.assert_numpy_array_equal(filler, expect_filler)

    def test_pad(self):
        old = Index([1, 5, 10])
        new = Index(list(range(12)))

        filler = libalgos.pad["int64_t"](old.values, new.values)

        expect_filler = np.array([-1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2], dtype=np.intp)
        tm.assert_numpy_array_equal(filler, expect_filler)

        # corner case
        old = Index([5, 10])
        new = Index(np.arange(5))
        filler = libalgos.pad["int64_t"](old.values, new.values)
        expect_filler = np.array([-1, -1, -1, -1, -1], dtype=np.intp)
        tm.assert_numpy_array_equal(filler, expect_filler)


def test_is_lexsorted():
    failure = [
        np.array(
            [
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            dtype="int64",
        ),
        np.array(
            [
                30,
                29,
                28,
                27,
                26,
                25,
                24,
                23,
                22,
                21,
                20,
                19,
                18,
                17,
                16,
                15,
                14,
                13,
                12,
                11,
                10,
                9,
                8,
                7,
                6,
                5,
                4,
                3,
                2,
                1,
                0,
                30,
                29,
                28,
                27,
                26,
                25,
                24,
                23,
                22,
                21,
                20,
                19,
                18,
                17,
                16,
                15,
                14,
                13,
                12,
                11,
                10,
                9,
                8,
                7,
                6,
                5,
                4,
                3,
                2,
                1,
                0,
                30,
                29,
                28,
                27,
                26,
                25,
                24,
                23,
                22,
                21,
                20,
                19,
                18,
                17,
                16,
                15,
                14,
                13,
                12,
                11,
                10,
                9,
                8,
                7,
                6,
                5,
                4,
                3,
                2,
                1,
                0,
                30,
                29,
                28,
                27,
                26,
                25,
                24,
                23,
                22,
                21,
                20,
                19,
                18,
                17,
                16,
                15,
                14,
                13,
                12,
                11,
                10,
                9,
                8,
                7,
                6,
                5,
                4,
                3,
                2,
                1,
                0,
            ],
            dtype="int64",
        ),
    ]

    assert not libalgos.is_lexsorted(failure)


def test_groupsort_indexer():
    a = np.random.randint(0, 1000, 100).astype(np.intp)
    b = np.random.randint(0, 1000, 100).astype(np.intp)

    result = libalgos.groupsort_indexer(a, 1000)[0]

    # need to use a stable sort
    # np.argsort returns int, groupsort_indexer
    # always returns intp
    expected = np.argsort(a, kind="mergesort")
    expected = expected.astype(np.intp)

    tm.assert_numpy_array_equal(result, expected)

    # compare with lexsort
    # np.lexsort returns int, groupsort_indexer
    # always returns intp
    key = a * 1000 + b
    result = libalgos.groupsort_indexer(key, 1000000)[0]
    expected = np.lexsort((b, a))
    expected = expected.astype(np.intp)

    tm.assert_numpy_array_equal(result, expected)


def test_infinity_sort():
    # GH 13445
    # numpy's argsort can be unhappy if something is less than
    # itself.  Instead, let's give our infinities a self-consistent
    # ordering, but outside the float extended real line.

    Inf = libalgos.Infinity()
    NegInf = libalgos.NegInfinity()

    ref_nums = [NegInf, float("-inf"), -1e100, 0, 1e100, float("inf"), Inf]

    assert all(Inf >= x for x in ref_nums)
    assert all(Inf > x or x is Inf for x in ref_nums)
    assert Inf >= Inf and Inf == Inf
    assert not Inf < Inf and not Inf > Inf
    assert libalgos.Infinity() == libalgos.Infinity()
    assert not libalgos.Infinity() != libalgos.Infinity()

    assert all(NegInf <= x for x in ref_nums)
    assert all(NegInf < x or x is NegInf for x in ref_nums)
    assert NegInf <= NegInf and NegInf == NegInf
    assert not NegInf < NegInf and not NegInf > NegInf
    assert libalgos.NegInfinity() == libalgos.NegInfinity()
    assert not libalgos.NegInfinity() != libalgos.NegInfinity()

    for perm in permutations(ref_nums):
        assert sorted(perm) == ref_nums

    # smoke tests
    np.array([libalgos.Infinity()] * 32).argsort()
    np.array([libalgos.NegInfinity()] * 32).argsort()


def test_infinity_against_nan():
    Inf = libalgos.Infinity()
    NegInf = libalgos.NegInfinity()

    assert not Inf > np.nan
    assert not Inf >= np.nan
    assert not Inf < np.nan
    assert not Inf <= np.nan
    assert not Inf == np.nan
    assert Inf != np.nan

    assert not NegInf > np.nan
    assert not NegInf >= np.nan
    assert not NegInf < np.nan
    assert not NegInf <= np.nan
    assert not NegInf == np.nan
    assert NegInf != np.nan


def test_ensure_platform_int():
    arr = np.arange(100, dtype=np.intp)

    result = libalgos.ensure_platform_int(arr)
    assert result is arr


def test_int64_add_overflow():
    # see gh-14068
    msg = "Overflow in int64 addition"
    m = np.iinfo(np.int64).max
    n = np.iinfo(np.int64).min

    with pytest.raises(OverflowError, match=msg):
        algos.checked_add_with_arr(np.array([m, m]), m)
    with pytest.raises(OverflowError, match=msg):
        algos.checked_add_with_arr(np.array([m, m]), np.array([m, m]))
    with pytest.raises(OverflowError, match=msg):
        algos.checked_add_with_arr(np.array([n, n]), n)
    with pytest.raises(OverflowError, match=msg):
        algos.checked_add_with_arr(np.array([n, n]), np.array([n, n]))
    with pytest.raises(OverflowError, match=msg):
        algos.checked_add_with_arr(np.array([m, n]), np.array([n, n]))
    with pytest.raises(OverflowError, match=msg):
        algos.checked_add_with_arr(
            np.array([m, m]), np.array([m, m]), arr_mask=np.array([False, True])
        )
    with pytest.raises(OverflowError, match=msg):
        algos.checked_add_with_arr(
            np.array([m, m]), np.array([m, m]), b_mask=np.array([False, True])
        )
    with pytest.raises(OverflowError, match=msg):
        algos.checked_add_with_arr(
            np.array([m, m]),
            np.array([m, m]),
            arr_mask=np.array([False, True]),
            b_mask=np.array([False, True]),
        )
    with pytest.raises(OverflowError, match=msg):
        with tm.assert_produces_warning(RuntimeWarning):
            algos.checked_add_with_arr(np.array([m, m]), np.array([np.nan, m]))

    # Check that the nan boolean arrays override whether or not
    # the addition overflows. We don't check the result but just
    # the fact that an OverflowError is not raised.
    algos.checked_add_with_arr(
        np.array([m, m]), np.array([m, m]), arr_mask=np.array([True, True])
    )
    algos.checked_add_with_arr(
        np.array([m, m]), np.array([m, m]), b_mask=np.array([True, True])
    )
    algos.checked_add_with_arr(
        np.array([m, m]),
        np.array([m, m]),
        arr_mask=np.array([True, False]),
        b_mask=np.array([False, True]),
    )


class TestMode:
    def test_no_mode(self):
        exp = Series([], dtype=np.float64, index=Index([], dtype=int))
        tm.assert_series_equal(algos.mode([]), exp)

    def test_mode_single(self):
        # GH 15714
        exp_single = [1]
        data_single = [1]

        exp_multi = [1]
        data_multi = [1, 1]

        for dt in np.typecodes["AllInteger"] + np.typecodes["Float"]:
            s = Series(data_single, dtype=dt)
            exp = Series(exp_single, dtype=dt)
            tm.assert_series_equal(algos.mode(s), exp)

            s = Series(data_multi, dtype=dt)
            exp = Series(exp_multi, dtype=dt)
            tm.assert_series_equal(algos.mode(s), exp)

        exp = Series([1], dtype=int)
        tm.assert_series_equal(algos.mode([1]), exp)

        exp = Series(["a", "b", "c"], dtype=object)
        tm.assert_series_equal(algos.mode(["a", "b", "c"]), exp)

    def test_number_mode(self):
        exp_single = [1]
        data_single = [1] * 5 + [2] * 3

        exp_multi = [1, 3]
        data_multi = [1] * 5 + [2] * 3 + [3] * 5

        for dt in np.typecodes["AllInteger"] + np.typecodes["Float"]:
            s = Series(data_single, dtype=dt)
            exp = Series(exp_single, dtype=dt)
            tm.assert_series_equal(algos.mode(s), exp)

            s = Series(data_multi, dtype=dt)
            exp = Series(exp_multi, dtype=dt)
            tm.assert_series_equal(algos.mode(s), exp)

    def test_strobj_mode(self):
        exp = ["b"]
        data = ["a"] * 2 + ["b"] * 3

        s = Series(data, dtype="c")
        exp = Series(exp, dtype="c")
        tm.assert_series_equal(algos.mode(s), exp)

        exp = ["bar"]
        data = ["foo"] * 2 + ["bar"] * 3

        for dt in [str, object]:
            s = Series(data, dtype=dt)
            exp = Series(exp, dtype=dt)
            tm.assert_series_equal(algos.mode(s), exp)

    def test_datelike_mode(self):
        exp = Series(["1900-05-03", "2011-01-03", "2013-01-02"], dtype="M8[ns]")
        s = Series(["2011-01-03", "2013-01-02", "1900-05-03"], dtype="M8[ns]")
        tm.assert_series_equal(algos.mode(s), exp)

        exp = Series(["2011-01-03", "2013-01-02"], dtype="M8[ns]")
        s = Series(
            ["2011-01-03", "2013-01-02", "1900-05-03", "2011-01-03", "2013-01-02"],
            dtype="M8[ns]",
        )
        tm.assert_series_equal(algos.mode(s), exp)

    def test_timedelta_mode(self):
        exp = Series(["-1 days", "0 days", "1 days"], dtype="timedelta64[ns]")
        s = Series(["1 days", "-1 days", "0 days"], dtype="timedelta64[ns]")
        tm.assert_series_equal(algos.mode(s), exp)

        exp = Series(["2 min", "1 day"], dtype="timedelta64[ns]")
        s = Series(
            ["1 day", "1 day", "-1 day", "-1 day 2 min", "2 min", "2 min"],
            dtype="timedelta64[ns]",
        )
        tm.assert_series_equal(algos.mode(s), exp)

    def test_mixed_dtype(self):
        exp = Series(["foo"])
        s = Series([1, "foo", "foo"])
        tm.assert_series_equal(algos.mode(s), exp)

    def test_uint64_overflow(self):
        exp = Series([2 ** 63], dtype=np.uint64)
        s = Series([1, 2 ** 63, 2 ** 63], dtype=np.uint64)
        tm.assert_series_equal(algos.mode(s), exp)

        exp = Series([1, 2 ** 63], dtype=np.uint64)
        s = Series([1, 2 ** 63], dtype=np.uint64)
        tm.assert_series_equal(algos.mode(s), exp)

    def test_categorical(self):
        c = Categorical([1, 2])
        exp = c
        tm.assert_categorical_equal(algos.mode(c), exp)
        tm.assert_categorical_equal(c.mode(), exp)

        c = Categorical([1, "a", "a"])
        exp = Categorical(["a"], categories=[1, "a"])
        tm.assert_categorical_equal(algos.mode(c), exp)
        tm.assert_categorical_equal(c.mode(), exp)

        c = Categorical([1, 1, 2, 3, 3])
        exp = Categorical([1, 3], categories=[1, 2, 3])
        tm.assert_categorical_equal(algos.mode(c), exp)
        tm.assert_categorical_equal(c.mode(), exp)

    def test_index(self):
        idx = Index([1, 2, 3])
        exp = Series([1, 2, 3], dtype=np.int64)
        tm.assert_series_equal(algos.mode(idx), exp)

        idx = Index([1, "a", "a"])
        exp = Series(["a"], dtype=object)
        tm.assert_series_equal(algos.mode(idx), exp)

        idx = Index([1, 1, 2, 3, 3])
        exp = Series([1, 3], dtype=np.int64)
        tm.assert_series_equal(algos.mode(idx), exp)

        exp = Series(["2 min", "1 day"], dtype="timedelta64[ns]")
        idx = Index(
            ["1 day", "1 day", "-1 day", "-1 day 2 min", "2 min", "2 min"],
            dtype="timedelta64[ns]",
        )
        tm.assert_series_equal(algos.mode(idx), exp)


class TestDiff:
    @pytest.mark.parametrize("dtype", ["M8[ns]", "m8[ns]"])
    def test_diff_datetimelike_nat(self, dtype):
        # NaT - NaT is NaT, not 0
        arr = np.arange(12).astype(np.int64).view(dtype).reshape(3, 4)
        arr[:, 2] = arr.dtype.type("NaT", "ns")
        result = algos.diff(arr, 1, axis=0)

        expected = np.ones(arr.shape, dtype="timedelta64[ns]") * 4
        expected[:, 2] = np.timedelta64("NaT", "ns")
        expected[0, :] = np.timedelta64("NaT", "ns")

        tm.assert_numpy_array_equal(result, expected)

        result = algos.diff(arr.T, 1, axis=1)
        tm.assert_numpy_array_equal(result, expected.T)

    def test_diff_ea_axis(self):
        dta = date_range("2016-01-01", periods=3, tz="US/Pacific")._data

        msg = "cannot diff DatetimeArray on axis=1"
        with pytest.raises(ValueError, match=msg):
            algos.diff(dta, 1, axis=1)

    @pytest.mark.parametrize("dtype", ["int8", "int16"])
    def test_diff_low_precision_int(self, dtype):
        arr = np.array([0, 1, 1, 0, 0], dtype=dtype)
        result = algos.diff(arr, 1)
        expected = np.array([np.nan, 1, 0, -1, 0], dtype="float32")
        tm.assert_numpy_array_equal(result, expected)


@pytest.mark.parametrize("op", [np.array, pd.array])
def test_union_with_duplicates(op):
    # GH#36289
    lvals = op([3, 1, 3, 4])
    rvals = op([2, 3, 1, 1])
    expected = op([3, 3, 1, 1, 4, 2])
    if isinstance(expected, np.ndarray):
        result = algos.union_with_duplicates(lvals, rvals)
        tm.assert_numpy_array_equal(result, expected)
    else:
        with tm.assert_produces_warning(RuntimeWarning):
            result = algos.union_with_duplicates(lvals, rvals)
        tm.assert_extension_array_equal(result, expected)
