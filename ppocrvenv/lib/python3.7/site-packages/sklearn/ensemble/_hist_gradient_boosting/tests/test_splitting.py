import numpy as np
import pytest
from numpy.testing import assert_array_equal

from sklearn.ensemble._hist_gradient_boosting.common import HISTOGRAM_DTYPE
from sklearn.ensemble._hist_gradient_boosting.common import G_H_DTYPE
from sklearn.ensemble._hist_gradient_boosting.common import X_BINNED_DTYPE
from sklearn.ensemble._hist_gradient_boosting.common import MonotonicConstraint
from sklearn.ensemble._hist_gradient_boosting.splitting import (
    Splitter,
    compute_node_value,
)
from sklearn.ensemble._hist_gradient_boosting.histogram import HistogramBuilder
from sklearn.utils._testing import skip_if_32bit
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads

n_threads = _openmp_effective_n_threads()


@pytest.mark.parametrize("n_bins", [3, 32, 256])
def test_histogram_split(n_bins):
    rng = np.random.RandomState(42)
    feature_idx = 0
    l2_regularization = 0
    min_hessian_to_split = 1e-3
    min_samples_leaf = 1
    min_gain_to_split = 0.0
    X_binned = np.asfortranarray(
        rng.randint(0, n_bins - 1, size=(int(1e4), 1)), dtype=X_BINNED_DTYPE
    )
    binned_feature = X_binned.T[feature_idx]
    sample_indices = np.arange(binned_feature.shape[0], dtype=np.uint32)
    ordered_hessians = np.ones_like(binned_feature, dtype=G_H_DTYPE)
    all_hessians = ordered_hessians
    sum_hessians = all_hessians.sum()
    hessians_are_constant = False

    for true_bin in range(1, n_bins - 2):
        for sign in [-1, 1]:
            ordered_gradients = np.full_like(binned_feature, sign, dtype=G_H_DTYPE)
            ordered_gradients[binned_feature <= true_bin] *= -1
            all_gradients = ordered_gradients
            sum_gradients = all_gradients.sum()

            builder = HistogramBuilder(
                X_binned,
                n_bins,
                all_gradients,
                all_hessians,
                hessians_are_constant,
                n_threads,
            )
            n_bins_non_missing = np.array(
                [n_bins - 1] * X_binned.shape[1], dtype=np.uint32
            )
            has_missing_values = np.array([False] * X_binned.shape[1], dtype=np.uint8)
            monotonic_cst = np.array(
                [MonotonicConstraint.NO_CST] * X_binned.shape[1], dtype=np.int8
            )
            is_categorical = np.zeros_like(monotonic_cst, dtype=np.uint8)
            missing_values_bin_idx = n_bins - 1
            splitter = Splitter(
                X_binned,
                n_bins_non_missing,
                missing_values_bin_idx,
                has_missing_values,
                is_categorical,
                monotonic_cst,
                l2_regularization,
                min_hessian_to_split,
                min_samples_leaf,
                min_gain_to_split,
                hessians_are_constant,
            )

            histograms = builder.compute_histograms_brute(sample_indices)
            value = compute_node_value(
                sum_gradients, sum_hessians, -np.inf, np.inf, l2_regularization
            )
            split_info = splitter.find_node_split(
                sample_indices.shape[0], histograms, sum_gradients, sum_hessians, value
            )

            assert split_info.bin_idx == true_bin
            assert split_info.gain >= 0
            assert split_info.feature_idx == feature_idx
            assert (
                split_info.n_samples_left + split_info.n_samples_right
                == sample_indices.shape[0]
            )
            # Constant hessian: 1. per sample.
            assert split_info.n_samples_left == split_info.sum_hessian_left


@skip_if_32bit
@pytest.mark.parametrize("constant_hessian", [True, False])
def test_gradient_and_hessian_sanity(constant_hessian):
    # This test checks that the values of gradients and hessians are
    # consistent in different places:
    # - in split_info: si.sum_gradient_left + si.sum_gradient_right must be
    #   equal to the gradient at the node. Same for hessians.
    # - in the histograms: summing 'sum_gradients' over the bins must be
    #   constant across all features, and those sums must be equal to the
    #   node's gradient. Same for hessians.

    rng = np.random.RandomState(42)

    n_bins = 10
    n_features = 20
    n_samples = 500
    l2_regularization = 0.0
    min_hessian_to_split = 1e-3
    min_samples_leaf = 1
    min_gain_to_split = 0.0

    X_binned = rng.randint(
        0, n_bins, size=(n_samples, n_features), dtype=X_BINNED_DTYPE
    )
    X_binned = np.asfortranarray(X_binned)
    sample_indices = np.arange(n_samples, dtype=np.uint32)
    all_gradients = rng.randn(n_samples).astype(G_H_DTYPE)
    sum_gradients = all_gradients.sum()
    if constant_hessian:
        all_hessians = np.ones(1, dtype=G_H_DTYPE)
        sum_hessians = 1 * n_samples
    else:
        all_hessians = rng.lognormal(size=n_samples).astype(G_H_DTYPE)
        sum_hessians = all_hessians.sum()

    builder = HistogramBuilder(
        X_binned, n_bins, all_gradients, all_hessians, constant_hessian, n_threads
    )
    n_bins_non_missing = np.array([n_bins - 1] * X_binned.shape[1], dtype=np.uint32)
    has_missing_values = np.array([False] * X_binned.shape[1], dtype=np.uint8)
    monotonic_cst = np.array(
        [MonotonicConstraint.NO_CST] * X_binned.shape[1], dtype=np.int8
    )
    is_categorical = np.zeros_like(monotonic_cst, dtype=np.uint8)
    missing_values_bin_idx = n_bins - 1
    splitter = Splitter(
        X_binned,
        n_bins_non_missing,
        missing_values_bin_idx,
        has_missing_values,
        is_categorical,
        monotonic_cst,
        l2_regularization,
        min_hessian_to_split,
        min_samples_leaf,
        min_gain_to_split,
        constant_hessian,
    )

    hists_parent = builder.compute_histograms_brute(sample_indices)
    value_parent = compute_node_value(
        sum_gradients, sum_hessians, -np.inf, np.inf, l2_regularization
    )
    si_parent = splitter.find_node_split(
        n_samples, hists_parent, sum_gradients, sum_hessians, value_parent
    )
    sample_indices_left, sample_indices_right, _ = splitter.split_indices(
        si_parent, sample_indices
    )

    hists_left = builder.compute_histograms_brute(sample_indices_left)
    value_left = compute_node_value(
        si_parent.sum_gradient_left,
        si_parent.sum_hessian_left,
        -np.inf,
        np.inf,
        l2_regularization,
    )
    hists_right = builder.compute_histograms_brute(sample_indices_right)
    value_right = compute_node_value(
        si_parent.sum_gradient_right,
        si_parent.sum_hessian_right,
        -np.inf,
        np.inf,
        l2_regularization,
    )
    si_left = splitter.find_node_split(
        n_samples,
        hists_left,
        si_parent.sum_gradient_left,
        si_parent.sum_hessian_left,
        value_left,
    )
    si_right = splitter.find_node_split(
        n_samples,
        hists_right,
        si_parent.sum_gradient_right,
        si_parent.sum_hessian_right,
        value_right,
    )

    # make sure that si.sum_gradient_left + si.sum_gradient_right have their
    # expected value, same for hessians
    for si, indices in (
        (si_parent, sample_indices),
        (si_left, sample_indices_left),
        (si_right, sample_indices_right),
    ):
        gradient = si.sum_gradient_right + si.sum_gradient_left
        expected_gradient = all_gradients[indices].sum()
        hessian = si.sum_hessian_right + si.sum_hessian_left
        if constant_hessian:
            expected_hessian = indices.shape[0] * all_hessians[0]
        else:
            expected_hessian = all_hessians[indices].sum()

        assert np.isclose(gradient, expected_gradient)
        assert np.isclose(hessian, expected_hessian)

    # make sure sum of gradients in histograms are the same for all features,
    # and make sure they're equal to their expected value
    hists_parent = np.asarray(hists_parent, dtype=HISTOGRAM_DTYPE)
    hists_left = np.asarray(hists_left, dtype=HISTOGRAM_DTYPE)
    hists_right = np.asarray(hists_right, dtype=HISTOGRAM_DTYPE)
    for hists, indices in (
        (hists_parent, sample_indices),
        (hists_left, sample_indices_left),
        (hists_right, sample_indices_right),
    ):
        # note: gradients and hessians have shape (n_features,),
        # we're comparing them to *scalars*. This has the benefit of also
        # making sure that all the entries are equal across features.
        gradients = hists["sum_gradients"].sum(axis=1)  # shape = (n_features,)
        expected_gradient = all_gradients[indices].sum()  # scalar
        hessians = hists["sum_hessians"].sum(axis=1)
        if constant_hessian:
            # 0 is not the actual hessian, but it's not computed in this case
            expected_hessian = 0.0
        else:
            expected_hessian = all_hessians[indices].sum()

        assert np.allclose(gradients, expected_gradient)
        assert np.allclose(hessians, expected_hessian)


def test_split_indices():
    # Check that split_indices returns the correct splits and that
    # splitter.partition is consistent with what is returned.
    rng = np.random.RandomState(421)

    n_bins = 5
    n_samples = 10
    l2_regularization = 0.0
    min_hessian_to_split = 1e-3
    min_samples_leaf = 1
    min_gain_to_split = 0.0

    # split will happen on feature 1 and on bin 3
    X_binned = [
        [0, 0],
        [0, 3],
        [0, 4],
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 4],
        [0, 0],
        [0, 4],
    ]
    X_binned = np.asfortranarray(X_binned, dtype=X_BINNED_DTYPE)
    sample_indices = np.arange(n_samples, dtype=np.uint32)
    all_gradients = rng.randn(n_samples).astype(G_H_DTYPE)
    all_hessians = np.ones(1, dtype=G_H_DTYPE)
    sum_gradients = all_gradients.sum()
    sum_hessians = 1 * n_samples
    hessians_are_constant = True

    builder = HistogramBuilder(
        X_binned, n_bins, all_gradients, all_hessians, hessians_are_constant, n_threads
    )
    n_bins_non_missing = np.array([n_bins] * X_binned.shape[1], dtype=np.uint32)
    has_missing_values = np.array([False] * X_binned.shape[1], dtype=np.uint8)
    monotonic_cst = np.array(
        [MonotonicConstraint.NO_CST] * X_binned.shape[1], dtype=np.int8
    )
    is_categorical = np.zeros_like(monotonic_cst, dtype=np.uint8)
    missing_values_bin_idx = n_bins - 1
    splitter = Splitter(
        X_binned,
        n_bins_non_missing,
        missing_values_bin_idx,
        has_missing_values,
        is_categorical,
        monotonic_cst,
        l2_regularization,
        min_hessian_to_split,
        min_samples_leaf,
        min_gain_to_split,
        hessians_are_constant,
    )

    assert np.all(sample_indices == splitter.partition)

    histograms = builder.compute_histograms_brute(sample_indices)
    value = compute_node_value(
        sum_gradients, sum_hessians, -np.inf, np.inf, l2_regularization
    )
    si_root = splitter.find_node_split(
        n_samples, histograms, sum_gradients, sum_hessians, value
    )

    # sanity checks for best split
    assert si_root.feature_idx == 1
    assert si_root.bin_idx == 3

    samples_left, samples_right, position_right = splitter.split_indices(
        si_root, splitter.partition
    )
    assert set(samples_left) == set([0, 1, 3, 4, 5, 6, 8])
    assert set(samples_right) == set([2, 7, 9])

    assert list(samples_left) == list(splitter.partition[:position_right])
    assert list(samples_right) == list(splitter.partition[position_right:])

    # Check that the resulting split indices sizes are consistent with the
    # count statistics anticipated when looking for the best split.
    assert samples_left.shape[0] == si_root.n_samples_left
    assert samples_right.shape[0] == si_root.n_samples_right


def test_min_gain_to_split():
    # Try to split a pure node (all gradients are equal, same for hessians)
    # with min_gain_to_split = 0 and make sure that the node is not split (best
    # possible gain = -1). Note: before the strict inequality comparison, this
    # test would fail because the node would be split with a gain of 0.
    rng = np.random.RandomState(42)
    l2_regularization = 0
    min_hessian_to_split = 0
    min_samples_leaf = 1
    min_gain_to_split = 0.0
    n_bins = 255
    n_samples = 100
    X_binned = np.asfortranarray(
        rng.randint(0, n_bins, size=(n_samples, 1)), dtype=X_BINNED_DTYPE
    )
    binned_feature = X_binned[:, 0]
    sample_indices = np.arange(n_samples, dtype=np.uint32)
    all_hessians = np.ones_like(binned_feature, dtype=G_H_DTYPE)
    all_gradients = np.ones_like(binned_feature, dtype=G_H_DTYPE)
    sum_gradients = all_gradients.sum()
    sum_hessians = all_hessians.sum()
    hessians_are_constant = False

    builder = HistogramBuilder(
        X_binned, n_bins, all_gradients, all_hessians, hessians_are_constant, n_threads
    )
    n_bins_non_missing = np.array([n_bins - 1] * X_binned.shape[1], dtype=np.uint32)
    has_missing_values = np.array([False] * X_binned.shape[1], dtype=np.uint8)
    monotonic_cst = np.array(
        [MonotonicConstraint.NO_CST] * X_binned.shape[1], dtype=np.int8
    )
    is_categorical = np.zeros_like(monotonic_cst, dtype=np.uint8)
    missing_values_bin_idx = n_bins - 1
    splitter = Splitter(
        X_binned,
        n_bins_non_missing,
        missing_values_bin_idx,
        has_missing_values,
        is_categorical,
        monotonic_cst,
        l2_regularization,
        min_hessian_to_split,
        min_samples_leaf,
        min_gain_to_split,
        hessians_are_constant,
    )

    histograms = builder.compute_histograms_brute(sample_indices)
    value = compute_node_value(
        sum_gradients, sum_hessians, -np.inf, np.inf, l2_regularization
    )
    split_info = splitter.find_node_split(
        n_samples, histograms, sum_gradients, sum_hessians, value
    )
    assert split_info.gain == -1


@pytest.mark.parametrize(
    "X_binned, all_gradients, has_missing_values, n_bins_non_missing, "
    " expected_split_on_nan, expected_bin_idx, expected_go_to_left",
    [
        # basic sanity check with no missing values: given the gradient
        # values, the split must occur on bin_idx=3
        (
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],  # X_binned
            [1, 1, 1, 1, 5, 5, 5, 5, 5, 5],  # gradients
            False,  # no missing values
            10,  # n_bins_non_missing
            False,  # don't split on nans
            3,  # expected_bin_idx
            "not_applicable",
        ),
        # We replace 2 samples by NaNs (bin_idx=8)
        # These 2 samples were mapped to the left node before, so they should
        # be mapped to left node again
        # Notice how the bin_idx threshold changes from 3 to 1.
        (
            [8, 0, 1, 8, 2, 3, 4, 5, 6, 7],  # 8 <=> missing
            [1, 1, 1, 1, 5, 5, 5, 5, 5, 5],
            True,  # missing values
            8,  # n_bins_non_missing
            False,  # don't split on nans
            1,  # cut on bin_idx=1
            True,
        ),  # missing values go to left
        # same as above, but with non-consecutive missing_values_bin
        (
            [9, 0, 1, 9, 2, 3, 4, 5, 6, 7],  # 9 <=> missing
            [1, 1, 1, 1, 5, 5, 5, 5, 5, 5],
            True,  # missing values
            8,  # n_bins_non_missing
            False,  # don't split on nans
            1,  # cut on bin_idx=1
            True,
        ),  # missing values go to left
        # this time replacing 2 samples that were on the right.
        (
            [0, 1, 2, 3, 8, 4, 8, 5, 6, 7],  # 8 <=> missing
            [1, 1, 1, 1, 5, 5, 5, 5, 5, 5],
            True,  # missing values
            8,  # n_bins_non_missing
            False,  # don't split on nans
            3,  # cut on bin_idx=3 (like in first case)
            False,
        ),  # missing values go to right
        # same as above, but with non-consecutive missing_values_bin
        (
            [0, 1, 2, 3, 9, 4, 9, 5, 6, 7],  # 9 <=> missing
            [1, 1, 1, 1, 5, 5, 5, 5, 5, 5],
            True,  # missing values
            8,  # n_bins_non_missing
            False,  # don't split on nans
            3,  # cut on bin_idx=3 (like in first case)
            False,
        ),  # missing values go to right
        # For the following cases, split_on_nans is True (we replace all of
        # the samples with nans, instead of just 2).
        (
            [0, 1, 2, 3, 4, 4, 4, 4, 4, 4],  # 4 <=> missing
            [1, 1, 1, 1, 5, 5, 5, 5, 5, 5],
            True,  # missing values
            4,  # n_bins_non_missing
            True,  # split on nans
            3,  # cut on bin_idx=3
            False,
        ),  # missing values go to right
        # same as above, but with non-consecutive missing_values_bin
        (
            [0, 1, 2, 3, 9, 9, 9, 9, 9, 9],  # 9 <=> missing
            [1, 1, 1, 1, 1, 1, 5, 5, 5, 5],
            True,  # missing values
            4,  # n_bins_non_missing
            True,  # split on nans
            3,  # cut on bin_idx=3
            False,
        ),  # missing values go to right
        (
            [6, 6, 6, 6, 0, 1, 2, 3, 4, 5],  # 6 <=> missing
            [1, 1, 1, 1, 5, 5, 5, 5, 5, 5],
            True,  # missing values
            6,  # n_bins_non_missing
            True,  # split on nans
            5,  # cut on bin_idx=5
            False,
        ),  # missing values go to right
        # same as above, but with non-consecutive missing_values_bin
        (
            [9, 9, 9, 9, 0, 1, 2, 3, 4, 5],  # 9 <=> missing
            [1, 1, 1, 1, 5, 5, 5, 5, 5, 5],
            True,  # missing values
            6,  # n_bins_non_missing
            True,  # split on nans
            5,  # cut on bin_idx=5
            False,
        ),  # missing values go to right
    ],
)
def test_splitting_missing_values(
    X_binned,
    all_gradients,
    has_missing_values,
    n_bins_non_missing,
    expected_split_on_nan,
    expected_bin_idx,
    expected_go_to_left,
):
    # Make sure missing values are properly supported.
    # we build an artificial example with gradients such that the best split
    # is on bin_idx=3, when there are no missing values.
    # Then we introduce missing values and:
    #   - make sure the chosen bin is correct (find_best_bin()): it's
    #     still the same split, even though the index of the bin may change
    #   - make sure the missing values are mapped to the correct child
    #     (split_indices())

    n_bins = max(X_binned) + 1
    n_samples = len(X_binned)
    l2_regularization = 0.0
    min_hessian_to_split = 1e-3
    min_samples_leaf = 1
    min_gain_to_split = 0.0

    sample_indices = np.arange(n_samples, dtype=np.uint32)
    X_binned = np.array(X_binned, dtype=X_BINNED_DTYPE).reshape(-1, 1)
    X_binned = np.asfortranarray(X_binned)
    all_gradients = np.array(all_gradients, dtype=G_H_DTYPE)
    has_missing_values = np.array([has_missing_values], dtype=np.uint8)
    all_hessians = np.ones(1, dtype=G_H_DTYPE)
    sum_gradients = all_gradients.sum()
    sum_hessians = 1 * n_samples
    hessians_are_constant = True

    builder = HistogramBuilder(
        X_binned, n_bins, all_gradients, all_hessians, hessians_are_constant, n_threads
    )

    n_bins_non_missing = np.array([n_bins_non_missing], dtype=np.uint32)
    monotonic_cst = np.array(
        [MonotonicConstraint.NO_CST] * X_binned.shape[1], dtype=np.int8
    )
    is_categorical = np.zeros_like(monotonic_cst, dtype=np.uint8)
    missing_values_bin_idx = n_bins - 1
    splitter = Splitter(
        X_binned,
        n_bins_non_missing,
        missing_values_bin_idx,
        has_missing_values,
        is_categorical,
        monotonic_cst,
        l2_regularization,
        min_hessian_to_split,
        min_samples_leaf,
        min_gain_to_split,
        hessians_are_constant,
    )

    histograms = builder.compute_histograms_brute(sample_indices)
    value = compute_node_value(
        sum_gradients, sum_hessians, -np.inf, np.inf, l2_regularization
    )
    split_info = splitter.find_node_split(
        n_samples, histograms, sum_gradients, sum_hessians, value
    )

    assert split_info.bin_idx == expected_bin_idx
    if has_missing_values:
        assert split_info.missing_go_to_left == expected_go_to_left

    split_on_nan = split_info.bin_idx == n_bins_non_missing[0] - 1
    assert split_on_nan == expected_split_on_nan

    # Make sure the split is properly computed.
    # This also make sure missing values are properly assigned to the correct
    # child in split_indices()
    samples_left, samples_right, _ = splitter.split_indices(
        split_info, splitter.partition
    )

    if not expected_split_on_nan:
        # When we don't split on nans, the split should always be the same.
        assert set(samples_left) == set([0, 1, 2, 3])
        assert set(samples_right) == set([4, 5, 6, 7, 8, 9])
    else:
        # When we split on nans, samples with missing values are always mapped
        # to the right child.
        missing_samples_indices = np.flatnonzero(
            np.array(X_binned) == missing_values_bin_idx
        )
        non_missing_samples_indices = np.flatnonzero(
            np.array(X_binned) != missing_values_bin_idx
        )

        assert set(samples_right) == set(missing_samples_indices)
        assert set(samples_left) == set(non_missing_samples_indices)


@pytest.mark.parametrize(
    "X_binned, has_missing_values, n_bins_non_missing, ",
    [
        # one category
        ([0] * 20, False, 1),
        # all categories appear less than MIN_CAT_SUPPORT (hardcoded to 10)
        ([0] * 9 + [1] * 8, False, 2),
        # only one category appears more than MIN_CAT_SUPPORT
        ([0] * 12 + [1] * 8, False, 2),
        # missing values + category appear less than MIN_CAT_SUPPORT
        # 9 is missing
        ([0] * 9 + [1] * 8 + [9] * 4, True, 2),
        # no non-missing category
        ([9] * 11, True, 0),
    ],
)
def test_splitting_categorical_cat_smooth(
    X_binned, has_missing_values, n_bins_non_missing
):
    # Checks categorical splits are correct when the MIN_CAT_SUPPORT constraint
    # isn't respected: there are no splits

    n_bins = max(X_binned) + 1
    n_samples = len(X_binned)
    X_binned = np.array([X_binned], dtype=X_BINNED_DTYPE).T
    X_binned = np.asfortranarray(X_binned)

    l2_regularization = 0.0
    min_hessian_to_split = 1e-3
    min_samples_leaf = 1
    min_gain_to_split = 0.0

    sample_indices = np.arange(n_samples, dtype=np.uint32)
    all_gradients = np.ones(n_samples, dtype=G_H_DTYPE)
    has_missing_values = np.array([has_missing_values], dtype=np.uint8)
    all_hessians = np.ones(1, dtype=G_H_DTYPE)
    sum_gradients = all_gradients.sum()
    sum_hessians = n_samples
    hessians_are_constant = True

    builder = HistogramBuilder(
        X_binned, n_bins, all_gradients, all_hessians, hessians_are_constant, n_threads
    )

    n_bins_non_missing = np.array([n_bins_non_missing], dtype=np.uint32)
    monotonic_cst = np.array(
        [MonotonicConstraint.NO_CST] * X_binned.shape[1], dtype=np.int8
    )
    is_categorical = np.ones_like(monotonic_cst, dtype=np.uint8)
    missing_values_bin_idx = n_bins - 1

    splitter = Splitter(
        X_binned,
        n_bins_non_missing,
        missing_values_bin_idx,
        has_missing_values,
        is_categorical,
        monotonic_cst,
        l2_regularization,
        min_hessian_to_split,
        min_samples_leaf,
        min_gain_to_split,
        hessians_are_constant,
    )

    histograms = builder.compute_histograms_brute(sample_indices)
    value = compute_node_value(
        sum_gradients, sum_hessians, -np.inf, np.inf, l2_regularization
    )
    split_info = splitter.find_node_split(
        n_samples, histograms, sum_gradients, sum_hessians, value
    )

    # no split found
    assert split_info.gain == -1


def _assert_categories_equals_bitset(categories, bitset):
    # assert that the bitset exactly corresponds to the categories
    # bitset is assumed to be an array of 8 uint32 elements

    # form bitset from threshold
    expected_bitset = np.zeros(8, dtype=np.uint32)
    for cat in categories:
        idx = cat // 32
        shift = cat % 32
        expected_bitset[idx] |= 1 << shift

    # check for equality
    assert_array_equal(expected_bitset, bitset)


@pytest.mark.parametrize(
    "X_binned, all_gradients, expected_categories_left, n_bins_non_missing,"
    "missing_values_bin_idx, has_missing_values, expected_missing_go_to_left",
    [
        # 4 categories
        (
            [0, 1, 2, 3] * 11,  # X_binned
            [10, 1, 10, 10] * 11,  # all_gradients
            [1],  # expected_categories_left
            4,  # n_bins_non_missing
            4,  # missing_values_bin_idx
            False,  # has_missing_values
            None,
        ),  # expected_missing_go_to_left, unchecked
        # Make sure that the categories that are on the right (second half) of
        # the sorted categories array can still go in the left child. In this
        # case, the best split was found when scanning from right to left.
        (
            [0, 1, 2, 3] * 11,  # X_binned
            [10, 10, 10, 1] * 11,  # all_gradients
            [3],  # expected_categories_left
            4,  # n_bins_non_missing
            4,  # missing_values_bin_idx
            False,  # has_missing_values
            None,
        ),  # expected_missing_go_to_left, unchecked
        # categories that don't respect MIN_CAT_SUPPORT (cat 4) are always
        # mapped to the right child
        (
            [0, 1, 2, 3] * 11 + [4] * 5,  # X_binned
            [10, 10, 10, 1] * 11 + [10] * 5,  # all_gradients
            [3],  # expected_categories_left
            4,  # n_bins_non_missing
            4,  # missing_values_bin_idx
            False,  # has_missing_values
            None,
        ),  # expected_missing_go_to_left, unchecked
        # categories that don't respect MIN_CAT_SUPPORT are always mapped to
        # the right child: in this case a more sensible split could have been
        # 3, 4 - 0, 1, 2
        # But the split is still 3 - 0, 1, 2, 4. this is because we only scan
        # up to the middle of the sorted category array (0, 1, 2, 3), and
        # because we exclude cat 4 in this array.
        (
            [0, 1, 2, 3] * 11 + [4] * 5,  # X_binned
            [10, 10, 10, 1] * 11 + [1] * 5,  # all_gradients
            [3],  # expected_categories_left
            4,  # n_bins_non_missing
            4,  # missing_values_bin_idx
            False,  # has_missing_values
            None,
        ),  # expected_missing_go_to_left, unchecked
        # 4 categories with missing values that go to the right
        (
            [0, 1, 2] * 11 + [9] * 11,  # X_binned
            [10, 1, 10] * 11 + [10] * 11,  # all_gradients
            [1],  # expected_categories_left
            3,  # n_bins_non_missing
            9,  # missing_values_bin_idx
            True,  # has_missing_values
            False,
        ),  # expected_missing_go_to_left
        # 4 categories with missing values that go to the left
        (
            [0, 1, 2] * 11 + [9] * 11,  # X_binned
            [10, 1, 10] * 11 + [1] * 11,  # all_gradients
            [1, 9],  # expected_categories_left
            3,  # n_bins_non_missing
            9,  # missing_values_bin_idx
            True,  # has_missing_values
            True,
        ),  # expected_missing_go_to_left
        # split is on the missing value
        (
            [0, 1, 2, 3, 4] * 11 + [255] * 12,  # X_binned
            [10, 10, 10, 10, 10] * 11 + [1] * 12,  # all_gradients
            [255],  # expected_categories_left
            5,  # n_bins_non_missing
            255,  # missing_values_bin_idx
            True,  # has_missing_values
            True,
        ),  # expected_missing_go_to_left
        # split on even categories
        (
            list(range(60)) * 12,  # X_binned
            [10, 1] * 360,  # all_gradients
            list(range(1, 60, 2)),  # expected_categories_left
            59,  # n_bins_non_missing
            59,  # missing_values_bin_idx
            True,  # has_missing_values
            True,
        ),  # expected_missing_go_to_left
        # split on every 8 categories
        (
            list(range(256)) * 12,  # X_binned
            [10, 10, 10, 10, 10, 10, 10, 1] * 384,  # all_gradients
            list(range(7, 256, 8)),  # expected_categories_left
            255,  # n_bins_non_missing
            255,  # missing_values_bin_idx
            True,  # has_missing_values
            True,
        ),  # expected_missing_go_to_left
    ],
)
def test_splitting_categorical_sanity(
    X_binned,
    all_gradients,
    expected_categories_left,
    n_bins_non_missing,
    missing_values_bin_idx,
    has_missing_values,
    expected_missing_go_to_left,
):
    # Tests various combinations of categorical splits

    n_samples = len(X_binned)
    n_bins = max(X_binned) + 1

    X_binned = np.array(X_binned, dtype=X_BINNED_DTYPE).reshape(-1, 1)
    X_binned = np.asfortranarray(X_binned)

    l2_regularization = 0.0
    min_hessian_to_split = 1e-3
    min_samples_leaf = 1
    min_gain_to_split = 0.0

    sample_indices = np.arange(n_samples, dtype=np.uint32)
    all_gradients = np.array(all_gradients, dtype=G_H_DTYPE)
    all_hessians = np.ones(1, dtype=G_H_DTYPE)
    has_missing_values = np.array([has_missing_values], dtype=np.uint8)
    sum_gradients = all_gradients.sum()
    sum_hessians = n_samples
    hessians_are_constant = True

    builder = HistogramBuilder(
        X_binned, n_bins, all_gradients, all_hessians, hessians_are_constant, n_threads
    )

    n_bins_non_missing = np.array([n_bins_non_missing], dtype=np.uint32)
    monotonic_cst = np.array(
        [MonotonicConstraint.NO_CST] * X_binned.shape[1], dtype=np.int8
    )
    is_categorical = np.ones_like(monotonic_cst, dtype=np.uint8)

    splitter = Splitter(
        X_binned,
        n_bins_non_missing,
        missing_values_bin_idx,
        has_missing_values,
        is_categorical,
        monotonic_cst,
        l2_regularization,
        min_hessian_to_split,
        min_samples_leaf,
        min_gain_to_split,
        hessians_are_constant,
    )

    histograms = builder.compute_histograms_brute(sample_indices)

    value = compute_node_value(
        sum_gradients, sum_hessians, -np.inf, np.inf, l2_regularization
    )
    split_info = splitter.find_node_split(
        n_samples, histograms, sum_gradients, sum_hessians, value
    )

    assert split_info.is_categorical
    assert split_info.gain > 0
    _assert_categories_equals_bitset(
        expected_categories_left, split_info.left_cat_bitset
    )
    if has_missing_values:
        assert split_info.missing_go_to_left == expected_missing_go_to_left
    # If there is no missing value during training, the flag missing_go_to_left
    # is set later in the grower.

    # make sure samples are split correctly
    samples_left, samples_right, _ = splitter.split_indices(
        split_info, splitter.partition
    )

    left_mask = np.isin(X_binned.ravel(), expected_categories_left)
    assert_array_equal(sample_indices[left_mask], samples_left)
    assert_array_equal(sample_indices[~left_mask], samples_right)
