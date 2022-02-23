import numpy as np
import pytest

from pandas import (
    Series,
    Timestamp,
    isna,
)
import pandas._testing as tm


class TestSeriesArgsort:
    def _check_accum_op(self, name, ser, check_dtype=True):
        func = getattr(np, name)
        tm.assert_numpy_array_equal(
            func(ser).values, func(np.array(ser)), check_dtype=check_dtype
        )

        # with missing values
        ts = ser.copy()
        ts[::2] = np.NaN

        result = func(ts)[1::2]
        expected = func(np.array(ts.dropna()))

        tm.assert_numpy_array_equal(result.values, expected, check_dtype=False)

    def test_argsort(self, datetime_series):
        self._check_accum_op("argsort", datetime_series, check_dtype=False)
        argsorted = datetime_series.argsort()
        assert issubclass(argsorted.dtype.type, np.integer)

        # GH#2967 (introduced bug in 0.11-dev I think)
        s = Series([Timestamp(f"201301{i:02d}") for i in range(1, 6)])
        assert s.dtype == "datetime64[ns]"
        shifted = s.shift(-1)
        assert shifted.dtype == "datetime64[ns]"
        assert isna(shifted[4])

        result = s.argsort()
        expected = Series(range(5), dtype="int64")
        tm.assert_series_equal(result, expected)

        result = shifted.argsort()
        expected = Series(list(range(4)) + [-1], dtype="int64")
        tm.assert_series_equal(result, expected)

    def test_argsort_stable(self):
        s = Series(np.random.randint(0, 100, size=10000))
        mindexer = s.argsort(kind="mergesort")
        qindexer = s.argsort()

        mexpected = np.argsort(s.values, kind="mergesort")
        qexpected = np.argsort(s.values, kind="quicksort")

        tm.assert_series_equal(mindexer.astype(np.intp), Series(mexpected))
        tm.assert_series_equal(qindexer.astype(np.intp), Series(qexpected))
        msg = (
            r"ndarray Expected type <class 'numpy\.ndarray'>, "
            r"found <class 'pandas\.core\.series\.Series'> instead"
        )
        with pytest.raises(AssertionError, match=msg):
            tm.assert_numpy_array_equal(qindexer, mindexer)

    def test_argsort_preserve_name(self, datetime_series):
        result = datetime_series.argsort()
        assert result.name == datetime_series.name
