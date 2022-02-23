import numpy as np
from numpy.testing import assert_allclose
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pytest

from sklearn.ensemble._hist_gradient_boosting.binning import _BinMapper
from sklearn.ensemble._hist_gradient_boosting.grower import TreeGrower
from sklearn.ensemble._hist_gradient_boosting.predictor import TreePredictor
from sklearn.ensemble._hist_gradient_boosting.common import (
    G_H_DTYPE,
    PREDICTOR_RECORD_DTYPE,
    ALMOST_INF,
    X_BINNED_DTYPE,
    X_BITSET_INNER_DTYPE,
    X_DTYPE,
)
from sklearn.ensemble._hist_gradient_boosting._bitset import (
    set_bitset_memoryview,
    set_raw_bitset_from_binned_bitset,
)
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads

n_threads = _openmp_effective_n_threads()


@pytest.mark.parametrize("n_bins", [200, 256])
def test_regression_dataset(n_bins):
    X, y = make_regression(
        n_samples=500, n_features=10, n_informative=5, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    mapper = _BinMapper(n_bins=n_bins, random_state=42)
    X_train_binned = mapper.fit_transform(X_train)

    # Init gradients and hessians to that of least squares loss
    gradients = -y_train.astype(G_H_DTYPE)
    hessians = np.ones(1, dtype=G_H_DTYPE)

    min_samples_leaf = 10
    max_leaf_nodes = 30
    grower = TreeGrower(
        X_train_binned,
        gradients,
        hessians,
        min_samples_leaf=min_samples_leaf,
        max_leaf_nodes=max_leaf_nodes,
        n_bins=n_bins,
        n_bins_non_missing=mapper.n_bins_non_missing_,
    )
    grower.grow()

    predictor = grower.make_predictor(binning_thresholds=mapper.bin_thresholds_)

    known_cat_bitsets = np.zeros((0, 8), dtype=X_BITSET_INNER_DTYPE)
    f_idx_map = np.zeros(0, dtype=np.uint32)

    y_pred_train = predictor.predict(X_train, known_cat_bitsets, f_idx_map, n_threads)
    assert r2_score(y_train, y_pred_train) > 0.82

    y_pred_test = predictor.predict(X_test, known_cat_bitsets, f_idx_map, n_threads)
    assert r2_score(y_test, y_pred_test) > 0.67


@pytest.mark.parametrize(
    "num_threshold, expected_predictions",
    [
        (-np.inf, [0, 1, 1, 1]),
        (10, [0, 0, 1, 1]),
        (20, [0, 0, 0, 1]),
        (ALMOST_INF, [0, 0, 0, 1]),
        (np.inf, [0, 0, 0, 0]),
    ],
)
def test_infinite_values_and_thresholds(num_threshold, expected_predictions):
    # Make sure infinite values and infinite thresholds are handled properly.
    # In particular, if a value is +inf and the threshold is ALMOST_INF the
    # sample should go to the right child. If the threshold is inf (split on
    # nan), the +inf sample will go to the left child.

    X = np.array([-np.inf, 10, 20, np.inf]).reshape(-1, 1)
    nodes = np.zeros(3, dtype=PREDICTOR_RECORD_DTYPE)

    # We just construct a simple tree with 1 root and 2 children
    # parent node
    nodes[0]["left"] = 1
    nodes[0]["right"] = 2
    nodes[0]["feature_idx"] = 0
    nodes[0]["num_threshold"] = num_threshold

    # left child
    nodes[1]["is_leaf"] = True
    nodes[1]["value"] = 0

    # right child
    nodes[2]["is_leaf"] = True
    nodes[2]["value"] = 1

    binned_cat_bitsets = np.zeros((0, 8), dtype=X_BITSET_INNER_DTYPE)
    raw_categorical_bitsets = np.zeros((0, 8), dtype=X_BITSET_INNER_DTYPE)
    known_cat_bitset = np.zeros((0, 8), dtype=X_BITSET_INNER_DTYPE)
    f_idx_map = np.zeros(0, dtype=np.uint32)

    predictor = TreePredictor(nodes, binned_cat_bitsets, raw_categorical_bitsets)
    predictions = predictor.predict(X, known_cat_bitset, f_idx_map, n_threads)

    assert np.all(predictions == expected_predictions)


@pytest.mark.parametrize(
    "bins_go_left, expected_predictions",
    [
        ([0, 3, 4, 6], [1, 0, 0, 1, 1, 0]),
        ([0, 1, 2, 6], [1, 1, 1, 0, 0, 0]),
        ([3, 5, 6], [0, 0, 0, 1, 0, 1]),
    ],
)
def test_categorical_predictor(bins_go_left, expected_predictions):
    # Test predictor outputs are correct with categorical features

    X_binned = np.array([[0, 1, 2, 3, 4, 5]], dtype=X_BINNED_DTYPE).T
    categories = np.array([2, 5, 6, 8, 10, 15], dtype=X_DTYPE)

    bins_go_left = np.array(bins_go_left, dtype=X_BINNED_DTYPE)

    # We just construct a simple tree with 1 root and 2 children
    # parent node
    nodes = np.zeros(3, dtype=PREDICTOR_RECORD_DTYPE)
    nodes[0]["left"] = 1
    nodes[0]["right"] = 2
    nodes[0]["feature_idx"] = 0
    nodes[0]["is_categorical"] = True
    nodes[0]["missing_go_to_left"] = True

    # left child
    nodes[1]["is_leaf"] = True
    nodes[1]["value"] = 1

    # right child
    nodes[2]["is_leaf"] = True
    nodes[2]["value"] = 0

    binned_cat_bitsets = np.zeros((1, 8), dtype=X_BITSET_INNER_DTYPE)
    raw_categorical_bitsets = np.zeros((1, 8), dtype=X_BITSET_INNER_DTYPE)
    for go_left in bins_go_left:
        set_bitset_memoryview(binned_cat_bitsets[0], go_left)

    set_raw_bitset_from_binned_bitset(
        raw_categorical_bitsets[0], binned_cat_bitsets[0], categories
    )

    predictor = TreePredictor(nodes, binned_cat_bitsets, raw_categorical_bitsets)

    # Check binned data gives correct predictions
    prediction_binned = predictor.predict_binned(
        X_binned, missing_values_bin_idx=6, n_threads=n_threads
    )
    assert_allclose(prediction_binned, expected_predictions)

    # manually construct bitset
    known_cat_bitsets = np.zeros((1, 8), dtype=np.uint32)
    known_cat_bitsets[0, 0] = np.sum(2 ** categories, dtype=np.uint32)
    f_idx_map = np.array([0], dtype=np.uint32)

    # Check with un-binned data
    predictions = predictor.predict(
        categories.reshape(-1, 1), known_cat_bitsets, f_idx_map, n_threads
    )
    assert_allclose(predictions, expected_predictions)

    # Check missing goes left because missing_values_bin_idx=6
    X_binned_missing = np.array([[6]], dtype=X_BINNED_DTYPE).T
    predictions = predictor.predict_binned(
        X_binned_missing, missing_values_bin_idx=6, n_threads=n_threads
    )
    assert_allclose(predictions, [1])

    # missing and unknown go left
    predictions = predictor.predict(
        np.array([[np.nan, 17]], dtype=X_DTYPE).T,
        known_cat_bitsets,
        f_idx_map,
        n_threads,
    )
    assert_allclose(predictions, [1, 1])
