from itertools import (
    chain,
    product,
)

import numpy as np
import pytest

from pandas._libs.algos import (
    Infinity,
    NegInfinity,
)
import pandas.util._test_decorators as td

from pandas import (
    NaT,
    Series,
    Timestamp,
    date_range,
)
import pandas._testing as tm
from pandas.api.types import CategoricalDtype


class TestSeriesRank:
    s = Series([1, 3, 4, 2, np.nan, 2, 1, 5, np.nan, 3])

    results = {
        "average": np.array([1.5, 5.5, 7.0, 3.5, np.nan, 3.5, 1.5, 8.0, np.nan, 5.5]),
        "min": np.array([1, 5, 7, 3, np.nan, 3, 1, 8, np.nan, 5]),
        "max": np.array([2, 6, 7, 4, np.nan, 4, 2, 8, np.nan, 6]),
        "first": np.array([1, 5, 7, 3, np.nan, 4, 2, 8, np.nan, 6]),
        "dense": np.array([1, 3, 4, 2, np.nan, 2, 1, 5, np.nan, 3]),
    }

    def test_rank(self, datetime_series):
        pytest.importorskip("scipy.stats.special")
        rankdata = pytest.importorskip("scipy.stats.rankdata")

        datetime_series[::2] = np.nan
        datetime_series[:10][::3] = 4.0

        ranks = datetime_series.rank()
        oranks = datetime_series.astype("O").rank()

        tm.assert_series_equal(ranks, oranks)

        mask = np.isnan(datetime_series)
        filled = datetime_series.fillna(np.inf)

        # rankdata returns a ndarray
        exp = Series(rankdata(filled), index=filled.index, name="ts")
        exp[mask] = np.nan

        tm.assert_series_equal(ranks, exp)

        iseries = Series(np.arange(5).repeat(2))

        iranks = iseries.rank()
        exp = iseries.astype(float).rank()
        tm.assert_series_equal(iranks, exp)
        iseries = Series(np.arange(5)) + 1.0
        exp = iseries / 5.0
        iranks = iseries.rank(pct=True)

        tm.assert_series_equal(iranks, exp)

        iseries = Series(np.repeat(1, 100))
        exp = Series(np.repeat(0.505, 100))
        iranks = iseries.rank(pct=True)
        tm.assert_series_equal(iranks, exp)

        iseries[1] = np.nan
        exp = Series(np.repeat(50.0 / 99.0, 100))
        exp[1] = np.nan
        iranks = iseries.rank(pct=True)
        tm.assert_series_equal(iranks, exp)

        iseries = Series(np.arange(5)) + 1.0
        iseries[4] = np.nan
        exp = iseries / 4.0
        iranks = iseries.rank(pct=True)
        tm.assert_series_equal(iranks, exp)

        iseries = Series(np.repeat(np.nan, 100))
        exp = iseries.copy()
        iranks = iseries.rank(pct=True)
        tm.assert_series_equal(iranks, exp)

        iseries = Series(np.arange(5)) + 1
        iseries[4] = np.nan
        exp = iseries / 4.0
        iranks = iseries.rank(pct=True)
        tm.assert_series_equal(iranks, exp)

        rng = date_range("1/1/1990", periods=5)
        iseries = Series(np.arange(5), rng) + 1
        iseries.iloc[4] = np.nan
        exp = iseries / 4.0
        iranks = iseries.rank(pct=True)
        tm.assert_series_equal(iranks, exp)

        iseries = Series([1e-50, 1e-100, 1e-20, 1e-2, 1e-20 + 1e-30, 1e-1])
        exp = Series([2, 1, 3, 5, 4, 6.0])
        iranks = iseries.rank()
        tm.assert_series_equal(iranks, exp)

        # GH 5968
        iseries = Series(["3 day", "1 day 10m", "-2 day", NaT], dtype="m8[ns]")
        exp = Series([3, 2, 1, np.nan])
        iranks = iseries.rank()
        tm.assert_series_equal(iranks, exp)

        values = np.array(
            [-50, -1, -1e-20, -1e-25, -1e-50, 0, 1e-40, 1e-20, 1e-10, 2, 40],
            dtype="float64",
        )
        random_order = np.random.permutation(len(values))
        iseries = Series(values[random_order])
        exp = Series(random_order + 1.0, dtype="float64")
        iranks = iseries.rank()
        tm.assert_series_equal(iranks, exp)

    def test_rank_categorical(self):
        # GH issue #15420 rank incorrectly orders ordered categories

        # Test ascending/descending ranking for ordered categoricals
        exp = Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        exp_desc = Series([6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
        ordered = Series(
            ["first", "second", "third", "fourth", "fifth", "sixth"]
        ).astype(
            CategoricalDtype(
                categories=["first", "second", "third", "fourth", "fifth", "sixth"],
                ordered=True,
            )
        )
        tm.assert_series_equal(ordered.rank(), exp)
        tm.assert_series_equal(ordered.rank(ascending=False), exp_desc)

        # Unordered categoricals should be ranked as objects
        unordered = Series(
            ["first", "second", "third", "fourth", "fifth", "sixth"]
        ).astype(
            CategoricalDtype(
                categories=["first", "second", "third", "fourth", "fifth", "sixth"],
                ordered=False,
            )
        )
        exp_unordered = Series([2.0, 4.0, 6.0, 3.0, 1.0, 5.0])
        res = unordered.rank()
        tm.assert_series_equal(res, exp_unordered)

        unordered1 = Series([1, 2, 3, 4, 5, 6]).astype(
            CategoricalDtype([1, 2, 3, 4, 5, 6], False)
        )
        exp_unordered1 = Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        res1 = unordered1.rank()
        tm.assert_series_equal(res1, exp_unordered1)

        # Test na_option for rank data
        na_ser = Series(
            ["first", "second", "third", "fourth", "fifth", "sixth", np.NaN]
        ).astype(
            CategoricalDtype(
                ["first", "second", "third", "fourth", "fifth", "sixth", "seventh"],
                True,
            )
        )

        exp_top = Series([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1.0])
        exp_bot = Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        exp_keep = Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, np.NaN])

        tm.assert_series_equal(na_ser.rank(na_option="top"), exp_top)
        tm.assert_series_equal(na_ser.rank(na_option="bottom"), exp_bot)
        tm.assert_series_equal(na_ser.rank(na_option="keep"), exp_keep)

        # Test na_option for rank data with ascending False
        exp_top = Series([7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
        exp_bot = Series([6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 7.0])
        exp_keep = Series([6.0, 5.0, 4.0, 3.0, 2.0, 1.0, np.NaN])

        tm.assert_series_equal(na_ser.rank(na_option="top", ascending=False), exp_top)
        tm.assert_series_equal(
            na_ser.rank(na_option="bottom", ascending=False), exp_bot
        )
        tm.assert_series_equal(na_ser.rank(na_option="keep", ascending=False), exp_keep)

        # Test invalid values for na_option
        msg = "na_option must be one of 'keep', 'top', or 'bottom'"

        with pytest.raises(ValueError, match=msg):
            na_ser.rank(na_option="bad", ascending=False)

        # invalid type
        with pytest.raises(ValueError, match=msg):
            na_ser.rank(na_option=True, ascending=False)

        # Test with pct=True
        na_ser = Series(["first", "second", "third", "fourth", np.NaN]).astype(
            CategoricalDtype(["first", "second", "third", "fourth"], True)
        )
        exp_top = Series([0.4, 0.6, 0.8, 1.0, 0.2])
        exp_bot = Series([0.2, 0.4, 0.6, 0.8, 1.0])
        exp_keep = Series([0.25, 0.5, 0.75, 1.0, np.NaN])

        tm.assert_series_equal(na_ser.rank(na_option="top", pct=True), exp_top)
        tm.assert_series_equal(na_ser.rank(na_option="bottom", pct=True), exp_bot)
        tm.assert_series_equal(na_ser.rank(na_option="keep", pct=True), exp_keep)

    def test_rank_signature(self):
        s = Series([0, 1])
        s.rank(method="average")
        msg = "No axis named average for object type Series"
        with pytest.raises(ValueError, match=msg):
            s.rank("average")

    def test_rank_tie_methods(self):
        s = self.s

        def _check(s, expected, method="average"):
            result = s.rank(method=method)
            tm.assert_series_equal(result, Series(expected))

        dtypes = [None, object]
        disabled = {(object, "first")}
        results = self.results

        for method, dtype in product(results, dtypes):
            if (dtype, method) in disabled:
                continue
            series = s if dtype is None else s.astype(dtype)
            _check(series, results[method], method=method)

    @td.skip_if_no_scipy
    @pytest.mark.parametrize("ascending", [True, False])
    @pytest.mark.parametrize("method", ["average", "min", "max", "first", "dense"])
    @pytest.mark.parametrize("na_option", ["top", "bottom", "keep"])
    def test_rank_tie_methods_on_infs_nans(self, method, na_option, ascending):
        dtypes = [
            ("object", None, Infinity(), NegInfinity()),
            ("float64", np.nan, np.inf, -np.inf),
        ]
        chunk = 3
        disabled = {("object", "first")}

        def _check(s, method, na_option, ascending):
            exp_ranks = {
                "average": ([2, 2, 2], [5, 5, 5], [8, 8, 8]),
                "min": ([1, 1, 1], [4, 4, 4], [7, 7, 7]),
                "max": ([3, 3, 3], [6, 6, 6], [9, 9, 9]),
                "first": ([1, 2, 3], [4, 5, 6], [7, 8, 9]),
                "dense": ([1, 1, 1], [2, 2, 2], [3, 3, 3]),
            }
            ranks = exp_ranks[method]
            if na_option == "top":
                order = [ranks[1], ranks[0], ranks[2]]
            elif na_option == "bottom":
                order = [ranks[0], ranks[2], ranks[1]]
            else:
                order = [ranks[0], [np.nan] * chunk, ranks[1]]
            expected = order if ascending else order[::-1]
            expected = list(chain.from_iterable(expected))
            result = s.rank(method=method, na_option=na_option, ascending=ascending)
            tm.assert_series_equal(result, Series(expected, dtype="float64"))

        for dtype, na_value, pos_inf, neg_inf in dtypes:
            in_arr = [neg_inf] * chunk + [na_value] * chunk + [pos_inf] * chunk
            iseries = Series(in_arr, dtype=dtype)
            if (dtype, method) in disabled:
                continue
            _check(iseries, method, na_option, ascending)

    def test_rank_desc_mix_nans_infs(self):
        # GH 19538
        # check descending ranking when mix nans and infs
        iseries = Series([1, np.nan, np.inf, -np.inf, 25])
        result = iseries.rank(ascending=False)
        exp = Series([3, np.nan, 1, 4, 2], dtype="float64")
        tm.assert_series_equal(result, exp)

    def test_rank_methods_series(self):
        pytest.importorskip("scipy.stats.special")
        rankdata = pytest.importorskip("scipy.stats.rankdata")

        xs = np.random.randn(9)
        xs = np.concatenate([xs[i:] for i in range(0, 9, 2)])  # add duplicates
        np.random.shuffle(xs)

        index = [chr(ord("a") + i) for i in range(len(xs))]

        for vals in [xs, xs + 1e6, xs * 1e-6]:
            ts = Series(vals, index=index)

            for m in ["average", "min", "max", "first", "dense"]:
                result = ts.rank(method=m)
                sprank = rankdata(vals, m if m != "first" else "ordinal")
                expected = Series(sprank, index=index).astype("float64")
                tm.assert_series_equal(result, expected)

    def test_rank_dense_method(self):
        dtypes = ["O", "f8", "i8"]
        in_out = [
            ([1], [1]),
            ([2], [1]),
            ([0], [1]),
            ([2, 2], [1, 1]),
            ([1, 2, 3], [1, 2, 3]),
            ([4, 2, 1], [3, 2, 1]),
            ([1, 1, 5, 5, 3], [1, 1, 3, 3, 2]),
            ([-5, -4, -3, -2, -1], [1, 2, 3, 4, 5]),
        ]

        for ser, exp in in_out:
            for dtype in dtypes:
                s = Series(ser).astype(dtype)
                result = s.rank(method="dense")
                expected = Series(exp).astype(result.dtype)
                tm.assert_series_equal(result, expected)

    def test_rank_descending(self):
        dtypes = ["O", "f8", "i8"]

        for dtype, method in product(dtypes, self.results):
            if "i" in dtype:
                s = self.s.dropna()
            else:
                s = self.s.astype(dtype)

            res = s.rank(ascending=False)
            expected = (s.max() - s).rank()
            tm.assert_series_equal(res, expected)

            if method == "first" and dtype == "O":
                continue

            expected = (s.max() - s).rank(method=method)
            res2 = s.rank(method=method, ascending=False)
            tm.assert_series_equal(res2, expected)

    def test_rank_int(self):
        s = self.s.dropna().astype("i8")

        for method, res in self.results.items():
            result = s.rank(method=method)
            expected = Series(res).dropna()
            expected.index = result.index
            tm.assert_series_equal(result, expected)

    def test_rank_object_bug(self):
        # GH 13445

        # smoke tests
        Series([np.nan] * 32).astype(object).rank(ascending=True)
        Series([np.nan] * 32).astype(object).rank(ascending=False)

    def test_rank_modify_inplace(self):
        # GH 18521
        # Check rank does not mutate series
        s = Series([Timestamp("2017-01-05 10:20:27.569000"), NaT])
        expected = s.copy()

        s.rank()
        result = s
        tm.assert_series_equal(result, expected)


# GH15630, pct should be on 100% basis when method='dense'


@pytest.mark.parametrize("dtype", ["O", "f8", "i8"])
@pytest.mark.parametrize(
    "ser, exp",
    [
        ([1], [1.0]),
        ([1, 2], [1.0 / 2, 2.0 / 2]),
        ([2, 2], [1.0, 1.0]),
        ([1, 2, 3], [1.0 / 3, 2.0 / 3, 3.0 / 3]),
        ([1, 2, 2], [1.0 / 2, 2.0 / 2, 2.0 / 2]),
        ([4, 2, 1], [3.0 / 3, 2.0 / 3, 1.0 / 3]),
        ([1, 1, 5, 5, 3], [1.0 / 3, 1.0 / 3, 3.0 / 3, 3.0 / 3, 2.0 / 3]),
        ([1, 1, 3, 3, 5, 5], [1.0 / 3, 1.0 / 3, 2.0 / 3, 2.0 / 3, 3.0 / 3, 3.0 / 3]),
        ([-5, -4, -3, -2, -1], [1.0 / 5, 2.0 / 5, 3.0 / 5, 4.0 / 5, 5.0 / 5]),
    ],
)
def test_rank_dense_pct(dtype, ser, exp):
    s = Series(ser).astype(dtype)
    result = s.rank(method="dense", pct=True)
    expected = Series(exp).astype(result.dtype)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("dtype", ["O", "f8", "i8"])
@pytest.mark.parametrize(
    "ser, exp",
    [
        ([1], [1.0]),
        ([1, 2], [1.0 / 2, 2.0 / 2]),
        ([2, 2], [1.0 / 2, 1.0 / 2]),
        ([1, 2, 3], [1.0 / 3, 2.0 / 3, 3.0 / 3]),
        ([1, 2, 2], [1.0 / 3, 2.0 / 3, 2.0 / 3]),
        ([4, 2, 1], [3.0 / 3, 2.0 / 3, 1.0 / 3]),
        ([1, 1, 5, 5, 3], [1.0 / 5, 1.0 / 5, 4.0 / 5, 4.0 / 5, 3.0 / 5]),
        ([1, 1, 3, 3, 5, 5], [1.0 / 6, 1.0 / 6, 3.0 / 6, 3.0 / 6, 5.0 / 6, 5.0 / 6]),
        ([-5, -4, -3, -2, -1], [1.0 / 5, 2.0 / 5, 3.0 / 5, 4.0 / 5, 5.0 / 5]),
    ],
)
def test_rank_min_pct(dtype, ser, exp):
    s = Series(ser).astype(dtype)
    result = s.rank(method="min", pct=True)
    expected = Series(exp).astype(result.dtype)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("dtype", ["O", "f8", "i8"])
@pytest.mark.parametrize(
    "ser, exp",
    [
        ([1], [1.0]),
        ([1, 2], [1.0 / 2, 2.0 / 2]),
        ([2, 2], [1.0, 1.0]),
        ([1, 2, 3], [1.0 / 3, 2.0 / 3, 3.0 / 3]),
        ([1, 2, 2], [1.0 / 3, 3.0 / 3, 3.0 / 3]),
        ([4, 2, 1], [3.0 / 3, 2.0 / 3, 1.0 / 3]),
        ([1, 1, 5, 5, 3], [2.0 / 5, 2.0 / 5, 5.0 / 5, 5.0 / 5, 3.0 / 5]),
        ([1, 1, 3, 3, 5, 5], [2.0 / 6, 2.0 / 6, 4.0 / 6, 4.0 / 6, 6.0 / 6, 6.0 / 6]),
        ([-5, -4, -3, -2, -1], [1.0 / 5, 2.0 / 5, 3.0 / 5, 4.0 / 5, 5.0 / 5]),
    ],
)
def test_rank_max_pct(dtype, ser, exp):
    s = Series(ser).astype(dtype)
    result = s.rank(method="max", pct=True)
    expected = Series(exp).astype(result.dtype)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("dtype", ["O", "f8", "i8"])
@pytest.mark.parametrize(
    "ser, exp",
    [
        ([1], [1.0]),
        ([1, 2], [1.0 / 2, 2.0 / 2]),
        ([2, 2], [1.5 / 2, 1.5 / 2]),
        ([1, 2, 3], [1.0 / 3, 2.0 / 3, 3.0 / 3]),
        ([1, 2, 2], [1.0 / 3, 2.5 / 3, 2.5 / 3]),
        ([4, 2, 1], [3.0 / 3, 2.0 / 3, 1.0 / 3]),
        ([1, 1, 5, 5, 3], [1.5 / 5, 1.5 / 5, 4.5 / 5, 4.5 / 5, 3.0 / 5]),
        ([1, 1, 3, 3, 5, 5], [1.5 / 6, 1.5 / 6, 3.5 / 6, 3.5 / 6, 5.5 / 6, 5.5 / 6]),
        ([-5, -4, -3, -2, -1], [1.0 / 5, 2.0 / 5, 3.0 / 5, 4.0 / 5, 5.0 / 5]),
    ],
)
def test_rank_average_pct(dtype, ser, exp):
    s = Series(ser).astype(dtype)
    result = s.rank(method="average", pct=True)
    expected = Series(exp).astype(result.dtype)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("dtype", ["f8", "i8"])
@pytest.mark.parametrize(
    "ser, exp",
    [
        ([1], [1.0]),
        ([1, 2], [1.0 / 2, 2.0 / 2]),
        ([2, 2], [1.0 / 2, 2.0 / 2.0]),
        ([1, 2, 3], [1.0 / 3, 2.0 / 3, 3.0 / 3]),
        ([1, 2, 2], [1.0 / 3, 2.0 / 3, 3.0 / 3]),
        ([4, 2, 1], [3.0 / 3, 2.0 / 3, 1.0 / 3]),
        ([1, 1, 5, 5, 3], [1.0 / 5, 2.0 / 5, 4.0 / 5, 5.0 / 5, 3.0 / 5]),
        ([1, 1, 3, 3, 5, 5], [1.0 / 6, 2.0 / 6, 3.0 / 6, 4.0 / 6, 5.0 / 6, 6.0 / 6]),
        ([-5, -4, -3, -2, -1], [1.0 / 5, 2.0 / 5, 3.0 / 5, 4.0 / 5, 5.0 / 5]),
    ],
)
def test_rank_first_pct(dtype, ser, exp):
    s = Series(ser).astype(dtype)
    result = s.rank(method="first", pct=True)
    expected = Series(exp).astype(result.dtype)
    tm.assert_series_equal(result, expected)


@pytest.mark.single
@pytest.mark.high_memory
def test_pct_max_many_rows():
    # GH 18271
    s = Series(np.arange(2 ** 24 + 1))
    result = s.rank(pct=True).max()
    assert result == 1
