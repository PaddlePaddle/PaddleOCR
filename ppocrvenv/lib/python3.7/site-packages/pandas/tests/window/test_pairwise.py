import warnings

import numpy as np
import pytest

from pandas import (
    DataFrame,
    MultiIndex,
    Series,
    date_range,
)
import pandas._testing as tm
from pandas.core.algorithms import safe_sort


class TestPairwise:

    # GH 7738
    @pytest.mark.parametrize("f", [lambda x: x.cov(), lambda x: x.corr()])
    def test_no_flex(self, pairwise_frames, pairwise_target_frame, f):

        # DataFrame methods (which do not call flex_binary_moment())

        result = f(pairwise_frames)
        tm.assert_index_equal(result.index, pairwise_frames.columns)
        tm.assert_index_equal(result.columns, pairwise_frames.columns)
        expected = f(pairwise_target_frame)
        # since we have sorted the results
        # we can only compare non-nans
        result = result.dropna().values
        expected = expected.dropna().values

        tm.assert_numpy_array_equal(result, expected, check_dtype=False)

    @pytest.mark.parametrize(
        "f",
        [
            lambda x: x.expanding().cov(pairwise=True),
            lambda x: x.expanding().corr(pairwise=True),
            lambda x: x.rolling(window=3).cov(pairwise=True),
            lambda x: x.rolling(window=3).corr(pairwise=True),
            lambda x: x.ewm(com=3).cov(pairwise=True),
            lambda x: x.ewm(com=3).corr(pairwise=True),
        ],
    )
    def test_pairwise_with_self(self, pairwise_frames, pairwise_target_frame, f):

        # DataFrame with itself, pairwise=True
        # note that we may construct the 1st level of the MI
        # in a non-monotonic way, so compare accordingly
        result = f(pairwise_frames)
        tm.assert_index_equal(
            result.index.levels[0], pairwise_frames.index, check_names=False
        )
        tm.assert_numpy_array_equal(
            safe_sort(result.index.levels[1]),
            safe_sort(pairwise_frames.columns.unique()),
        )
        tm.assert_index_equal(result.columns, pairwise_frames.columns)
        expected = f(pairwise_target_frame)
        # since we have sorted the results
        # we can only compare non-nans
        result = result.dropna().values
        expected = expected.dropna().values

        tm.assert_numpy_array_equal(result, expected, check_dtype=False)

    @pytest.mark.parametrize(
        "f",
        [
            lambda x: x.expanding().cov(pairwise=False),
            lambda x: x.expanding().corr(pairwise=False),
            lambda x: x.rolling(window=3).cov(pairwise=False),
            lambda x: x.rolling(window=3).corr(pairwise=False),
            lambda x: x.ewm(com=3).cov(pairwise=False),
            lambda x: x.ewm(com=3).corr(pairwise=False),
        ],
    )
    def test_no_pairwise_with_self(self, pairwise_frames, pairwise_target_frame, f):

        # DataFrame with itself, pairwise=False
        result = f(pairwise_frames)
        tm.assert_index_equal(result.index, pairwise_frames.index)
        tm.assert_index_equal(result.columns, pairwise_frames.columns)
        expected = f(pairwise_target_frame)
        # since we have sorted the results
        # we can only compare non-nans
        result = result.dropna().values
        expected = expected.dropna().values

        tm.assert_numpy_array_equal(result, expected, check_dtype=False)

    @pytest.mark.parametrize(
        "f",
        [
            lambda x, y: x.expanding().cov(y, pairwise=True),
            lambda x, y: x.expanding().corr(y, pairwise=True),
            lambda x, y: x.rolling(window=3).cov(y, pairwise=True),
            lambda x, y: x.rolling(window=3).corr(y, pairwise=True),
            lambda x, y: x.ewm(com=3).cov(y, pairwise=True),
            lambda x, y: x.ewm(com=3).corr(y, pairwise=True),
        ],
    )
    def test_pairwise_with_other(
        self, pairwise_frames, pairwise_target_frame, pairwise_other_frame, f
    ):

        # DataFrame with another DataFrame, pairwise=True
        result = f(pairwise_frames, pairwise_other_frame)
        tm.assert_index_equal(
            result.index.levels[0], pairwise_frames.index, check_names=False
        )
        tm.assert_numpy_array_equal(
            safe_sort(result.index.levels[1]),
            safe_sort(pairwise_other_frame.columns.unique()),
        )
        expected = f(pairwise_target_frame, pairwise_other_frame)
        # since we have sorted the results
        # we can only compare non-nans
        result = result.dropna().values
        expected = expected.dropna().values

        tm.assert_numpy_array_equal(result, expected, check_dtype=False)

    @pytest.mark.parametrize(
        "f",
        [
            lambda x, y: x.expanding().cov(y, pairwise=False),
            lambda x, y: x.expanding().corr(y, pairwise=False),
            lambda x, y: x.rolling(window=3).cov(y, pairwise=False),
            lambda x, y: x.rolling(window=3).corr(y, pairwise=False),
            lambda x, y: x.ewm(com=3).cov(y, pairwise=False),
            lambda x, y: x.ewm(com=3).corr(y, pairwise=False),
        ],
    )
    def test_no_pairwise_with_other(self, pairwise_frames, pairwise_other_frame, f):

        # DataFrame with another DataFrame, pairwise=False
        result = (
            f(pairwise_frames, pairwise_other_frame)
            if pairwise_frames.columns.is_unique
            else None
        )
        if result is not None:
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("ignore", RuntimeWarning)
                # we can have int and str columns
                expected_index = pairwise_frames.index.union(pairwise_other_frame.index)
                expected_columns = pairwise_frames.columns.union(
                    pairwise_other_frame.columns
                )
            tm.assert_index_equal(result.index, expected_index)
            tm.assert_index_equal(result.columns, expected_columns)
        else:
            with pytest.raises(ValueError, match="'arg1' columns are not unique"):
                f(pairwise_frames, pairwise_other_frame)
            with pytest.raises(ValueError, match="'arg2' columns are not unique"):
                f(pairwise_other_frame, pairwise_frames)

    @pytest.mark.parametrize(
        "f",
        [
            lambda x, y: x.expanding().cov(y),
            lambda x, y: x.expanding().corr(y),
            lambda x, y: x.rolling(window=3).cov(y),
            lambda x, y: x.rolling(window=3).corr(y),
            lambda x, y: x.ewm(com=3).cov(y),
            lambda x, y: x.ewm(com=3).corr(y),
        ],
    )
    def test_pairwise_with_series(self, pairwise_frames, pairwise_target_frame, f):

        # DataFrame with a Series
        result = f(pairwise_frames, Series([1, 1, 3, 8]))
        tm.assert_index_equal(result.index, pairwise_frames.index)
        tm.assert_index_equal(result.columns, pairwise_frames.columns)
        expected = f(pairwise_target_frame, Series([1, 1, 3, 8]))
        # since we have sorted the results
        # we can only compare non-nans
        result = result.dropna().values
        expected = expected.dropna().values
        tm.assert_numpy_array_equal(result, expected, check_dtype=False)

        result = f(Series([1, 1, 3, 8]), pairwise_frames)
        tm.assert_index_equal(result.index, pairwise_frames.index)
        tm.assert_index_equal(result.columns, pairwise_frames.columns)
        expected = f(Series([1, 1, 3, 8]), pairwise_target_frame)
        # since we have sorted the results
        # we can only compare non-nans
        result = result.dropna().values
        expected = expected.dropna().values
        tm.assert_numpy_array_equal(result, expected, check_dtype=False)

    def test_corr_freq_memory_error(self):
        # GH 31789
        s = Series(range(5), index=date_range("2020", periods=5))
        result = s.rolling("12H").corr(s)
        expected = Series([np.nan] * 5, index=date_range("2020", periods=5))
        tm.assert_series_equal(result, expected)

    def test_cov_mulittindex(self):
        # GH 34440

        columns = MultiIndex.from_product([list("ab"), list("xy"), list("AB")])
        index = range(3)
        df = DataFrame(np.arange(24).reshape(3, 8), index=index, columns=columns)

        result = df.ewm(alpha=0.1).cov()

        index = MultiIndex.from_product([range(3), list("ab"), list("xy"), list("AB")])
        columns = MultiIndex.from_product([list("ab"), list("xy"), list("AB")])
        expected = DataFrame(
            np.vstack(
                (
                    np.full((8, 8), np.NaN),
                    np.full((8, 8), 32.000000),
                    np.full((8, 8), 63.881919),
                )
            ),
            index=index,
            columns=columns,
        )

        tm.assert_frame_equal(result, expected)
