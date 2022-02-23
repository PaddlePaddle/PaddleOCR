import pytest
import numpy as np
import scipy.sparse as sp
import warnings

from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils._testing import (
    assert_array_almost_equal,
    assert_array_equal,
    assert_allclose_dense_sparse,
)

X = [[-2, 1.5, -4, -1], [-1, 2.5, -3, -0.5], [0, 3.5, -2, 0.5], [1, 4.5, -1, 2]]


@pytest.mark.parametrize(
    "strategy, expected",
    [
        ("uniform", [[0, 0, 0, 0], [1, 1, 1, 0], [2, 2, 2, 1], [2, 2, 2, 2]]),
        ("kmeans", [[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]]),
        ("quantile", [[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2], [2, 2, 2, 2]]),
    ],
)
def test_fit_transform(strategy, expected):
    est = KBinsDiscretizer(n_bins=3, encode="ordinal", strategy=strategy)
    est.fit(X)
    assert_array_equal(expected, est.transform(X))


def test_valid_n_bins():
    KBinsDiscretizer(n_bins=2).fit_transform(X)
    KBinsDiscretizer(n_bins=np.array([2])[0]).fit_transform(X)
    assert KBinsDiscretizer(n_bins=2).fit(X).n_bins_.dtype == np.dtype(int)


def test_invalid_n_bins():
    est = KBinsDiscretizer(n_bins=1)
    err_msg = (
        "KBinsDiscretizer received an invalid "
        "number of bins. Received 1, expected at least 2."
    )
    with pytest.raises(ValueError, match=err_msg):
        est.fit_transform(X)

    est = KBinsDiscretizer(n_bins=1.1)
    err_msg = (
        "KBinsDiscretizer received an invalid "
        "n_bins type. Received float, expected int."
    )
    with pytest.raises(ValueError, match=err_msg):
        est.fit_transform(X)


def test_invalid_n_bins_array():
    # Bad shape
    n_bins = np.full((2, 4), 2.0)
    est = KBinsDiscretizer(n_bins=n_bins)
    err_msg = r"n_bins must be a scalar or array of shape \(n_features,\)."
    with pytest.raises(ValueError, match=err_msg):
        est.fit_transform(X)

    # Incorrect number of features
    n_bins = [1, 2, 2]
    est = KBinsDiscretizer(n_bins=n_bins)
    err_msg = r"n_bins must be a scalar or array of shape \(n_features,\)."
    with pytest.raises(ValueError, match=err_msg):
        est.fit_transform(X)

    # Bad bin values
    n_bins = [1, 2, 2, 1]
    est = KBinsDiscretizer(n_bins=n_bins)
    err_msg = (
        "KBinsDiscretizer received an invalid number of bins "
        "at indices 0, 3. Number of bins must be at least 2, "
        "and must be an int."
    )
    with pytest.raises(ValueError, match=err_msg):
        est.fit_transform(X)

    # Float bin values
    n_bins = [2.1, 2, 2.1, 2]
    est = KBinsDiscretizer(n_bins=n_bins)
    err_msg = (
        "KBinsDiscretizer received an invalid number of bins "
        "at indices 0, 2. Number of bins must be at least 2, "
        "and must be an int."
    )
    with pytest.raises(ValueError, match=err_msg):
        est.fit_transform(X)


@pytest.mark.parametrize(
    "strategy, expected",
    [
        ("uniform", [[0, 0, 0, 0], [0, 1, 1, 0], [1, 2, 2, 1], [1, 2, 2, 2]]),
        ("kmeans", [[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1], [1, 2, 2, 2]]),
        ("quantile", [[0, 0, 0, 0], [0, 1, 1, 1], [1, 2, 2, 2], [1, 2, 2, 2]]),
    ],
)
def test_fit_transform_n_bins_array(strategy, expected):
    est = KBinsDiscretizer(
        n_bins=[2, 3, 3, 3], encode="ordinal", strategy=strategy
    ).fit(X)
    assert_array_equal(expected, est.transform(X))

    # test the shape of bin_edges_
    n_features = np.array(X).shape[1]
    assert est.bin_edges_.shape == (n_features,)
    for bin_edges, n_bins in zip(est.bin_edges_, est.n_bins_):
        assert bin_edges.shape == (n_bins + 1,)


@pytest.mark.parametrize("strategy", ["uniform", "kmeans", "quantile"])
def test_same_min_max(strategy):
    warnings.simplefilter("always")
    X = np.array([[1, -2], [1, -1], [1, 0], [1, 1]])
    est = KBinsDiscretizer(strategy=strategy, n_bins=3, encode="ordinal")
    warning_message = "Feature 0 is constant and will be replaced with 0."
    with pytest.warns(UserWarning, match=warning_message):
        est.fit(X)
    assert est.n_bins_[0] == 1
    # replace the feature with zeros
    Xt = est.transform(X)
    assert_array_equal(Xt[:, 0], np.zeros(X.shape[0]))


def test_transform_1d_behavior():
    X = np.arange(4)
    est = KBinsDiscretizer(n_bins=2)
    with pytest.raises(ValueError):
        est.fit(X)

    est = KBinsDiscretizer(n_bins=2)
    est.fit(X.reshape(-1, 1))
    with pytest.raises(ValueError):
        est.transform(X)


@pytest.mark.parametrize("i", range(1, 9))
def test_numeric_stability(i):
    X_init = np.array([2.0, 4.0, 6.0, 8.0, 10.0]).reshape(-1, 1)
    Xt_expected = np.array([0, 0, 1, 1, 1]).reshape(-1, 1)

    # Test up to discretizing nano units
    X = X_init / 10 ** i
    Xt = KBinsDiscretizer(n_bins=2, encode="ordinal").fit_transform(X)
    assert_array_equal(Xt_expected, Xt)


def test_invalid_encode_option():
    est = KBinsDiscretizer(n_bins=[2, 3, 3, 3], encode="invalid-encode")
    err_msg = (
        r"Valid options for 'encode' are "
        r"\('onehot', 'onehot-dense', 'ordinal'\). "
        r"Got encode='invalid-encode' instead."
    )
    with pytest.raises(ValueError, match=err_msg):
        est.fit(X)


def test_encode_options():
    est = KBinsDiscretizer(n_bins=[2, 3, 3, 3], encode="ordinal").fit(X)
    Xt_1 = est.transform(X)
    est = KBinsDiscretizer(n_bins=[2, 3, 3, 3], encode="onehot-dense").fit(X)
    Xt_2 = est.transform(X)
    assert not sp.issparse(Xt_2)
    assert_array_equal(
        OneHotEncoder(
            categories=[np.arange(i) for i in [2, 3, 3, 3]], sparse=False
        ).fit_transform(Xt_1),
        Xt_2,
    )
    est = KBinsDiscretizer(n_bins=[2, 3, 3, 3], encode="onehot").fit(X)
    Xt_3 = est.transform(X)
    assert sp.issparse(Xt_3)
    assert_array_equal(
        OneHotEncoder(categories=[np.arange(i) for i in [2, 3, 3, 3]], sparse=True)
        .fit_transform(Xt_1)
        .toarray(),
        Xt_3.toarray(),
    )


def test_invalid_strategy_option():
    est = KBinsDiscretizer(n_bins=[2, 3, 3, 3], strategy="invalid-strategy")
    err_msg = (
        r"Valid options for 'strategy' are "
        r"\('uniform', 'quantile', 'kmeans'\). "
        r"Got strategy='invalid-strategy' instead."
    )
    with pytest.raises(ValueError, match=err_msg):
        est.fit(X)


@pytest.mark.parametrize(
    "strategy, expected_2bins, expected_3bins, expected_5bins",
    [
        ("uniform", [0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 2, 2], [0, 0, 1, 1, 4, 4]),
        ("kmeans", [0, 0, 0, 0, 1, 1], [0, 0, 1, 1, 2, 2], [0, 0, 1, 2, 3, 4]),
        ("quantile", [0, 0, 0, 1, 1, 1], [0, 0, 1, 1, 2, 2], [0, 1, 2, 3, 4, 4]),
    ],
)
def test_nonuniform_strategies(
    strategy, expected_2bins, expected_3bins, expected_5bins
):
    X = np.array([0, 0.5, 2, 3, 9, 10]).reshape(-1, 1)

    # with 2 bins
    est = KBinsDiscretizer(n_bins=2, strategy=strategy, encode="ordinal")
    Xt = est.fit_transform(X)
    assert_array_equal(expected_2bins, Xt.ravel())

    # with 3 bins
    est = KBinsDiscretizer(n_bins=3, strategy=strategy, encode="ordinal")
    Xt = est.fit_transform(X)
    assert_array_equal(expected_3bins, Xt.ravel())

    # with 5 bins
    est = KBinsDiscretizer(n_bins=5, strategy=strategy, encode="ordinal")
    Xt = est.fit_transform(X)
    assert_array_equal(expected_5bins, Xt.ravel())


@pytest.mark.parametrize(
    "strategy, expected_inv",
    [
        (
            "uniform",
            [
                [-1.5, 2.0, -3.5, -0.5],
                [-0.5, 3.0, -2.5, -0.5],
                [0.5, 4.0, -1.5, 0.5],
                [0.5, 4.0, -1.5, 1.5],
            ],
        ),
        (
            "kmeans",
            [
                [-1.375, 2.125, -3.375, -0.5625],
                [-1.375, 2.125, -3.375, -0.5625],
                [-0.125, 3.375, -2.125, 0.5625],
                [0.75, 4.25, -1.25, 1.625],
            ],
        ),
        (
            "quantile",
            [
                [-1.5, 2.0, -3.5, -0.75],
                [-0.5, 3.0, -2.5, 0.0],
                [0.5, 4.0, -1.5, 1.25],
                [0.5, 4.0, -1.5, 1.25],
            ],
        ),
    ],
)
@pytest.mark.parametrize("encode", ["ordinal", "onehot", "onehot-dense"])
def test_inverse_transform(strategy, encode, expected_inv):
    kbd = KBinsDiscretizer(n_bins=3, strategy=strategy, encode=encode)
    Xt = kbd.fit_transform(X)
    Xinv = kbd.inverse_transform(Xt)
    assert_array_almost_equal(expected_inv, Xinv)


@pytest.mark.parametrize("strategy", ["uniform", "kmeans", "quantile"])
def test_transform_outside_fit_range(strategy):
    X = np.array([0, 1, 2, 3])[:, None]
    kbd = KBinsDiscretizer(n_bins=4, strategy=strategy, encode="ordinal")
    kbd.fit(X)

    X2 = np.array([-2, 5])[:, None]
    X2t = kbd.transform(X2)
    assert_array_equal(X2t.max(axis=0) + 1, kbd.n_bins_)
    assert_array_equal(X2t.min(axis=0), [0])


def test_overwrite():
    X = np.array([0, 1, 2, 3])[:, None]
    X_before = X.copy()

    est = KBinsDiscretizer(n_bins=3, encode="ordinal")
    Xt = est.fit_transform(X)
    assert_array_equal(X, X_before)

    Xt_before = Xt.copy()
    Xinv = est.inverse_transform(Xt)
    assert_array_equal(Xt, Xt_before)
    assert_array_equal(Xinv, np.array([[0.5], [1.5], [2.5], [2.5]]))


@pytest.mark.parametrize(
    "strategy, expected_bin_edges", [("quantile", [0, 1, 3]), ("kmeans", [0, 1.5, 3])]
)
def test_redundant_bins(strategy, expected_bin_edges):
    X = [[0], [0], [0], [0], [3], [3]]
    kbd = KBinsDiscretizer(n_bins=3, strategy=strategy)
    warning_message = "Consider decreasing the number of bins."
    with pytest.warns(UserWarning, match=warning_message):
        kbd.fit(X)
    assert_array_almost_equal(kbd.bin_edges_[0], expected_bin_edges)


def test_percentile_numeric_stability():
    X = np.array([0.05, 0.05, 0.95]).reshape(-1, 1)
    bin_edges = np.array([0.05, 0.23, 0.41, 0.59, 0.77, 0.95])
    Xt = np.array([0, 0, 4]).reshape(-1, 1)
    kbd = KBinsDiscretizer(n_bins=10, encode="ordinal", strategy="quantile")
    warning_message = "Consider decreasing the number of bins."
    with pytest.warns(UserWarning, match=warning_message):
        kbd.fit(X)

    assert_array_almost_equal(kbd.bin_edges_[0], bin_edges)
    assert_array_almost_equal(kbd.transform(X), Xt)


@pytest.mark.parametrize("in_dtype", [np.float16, np.float32, np.float64])
@pytest.mark.parametrize("out_dtype", [None, np.float16, np.float32, np.float64])
@pytest.mark.parametrize("encode", ["ordinal", "onehot", "onehot-dense"])
def test_consistent_dtype(in_dtype, out_dtype, encode):
    X_input = np.array(X, dtype=in_dtype)
    kbd = KBinsDiscretizer(n_bins=3, encode=encode, dtype=out_dtype)

    # a error is raised if a wrong dtype is define for the model
    if out_dtype not in [None, np.float32, np.float64]:
        with pytest.raises(ValueError, match="Valid options for 'dtype' are"):
            kbd.fit(X_input)
    else:
        kbd.fit(X_input)

        # test output dtype
        if out_dtype is not None:
            expected_dtype = out_dtype
        elif out_dtype is None and X_input.dtype == np.float16:
            # wrong numeric input dtype are cast in np.float64
            expected_dtype = np.float64
        else:
            expected_dtype = X_input.dtype
        Xt = kbd.transform(X_input)
        assert Xt.dtype == expected_dtype


@pytest.mark.parametrize("input_dtype", [np.float16, np.float32, np.float64])
@pytest.mark.parametrize("encode", ["ordinal", "onehot", "onehot-dense"])
def test_32_equal_64(input_dtype, encode):
    # TODO this check is redundant with common checks and can be removed
    #  once #16290 is merged
    X_input = np.array(X, dtype=input_dtype)

    # 32 bit output
    kbd_32 = KBinsDiscretizer(n_bins=3, encode=encode, dtype=np.float32)
    kbd_32.fit(X_input)
    Xt_32 = kbd_32.transform(X_input)

    # 64 bit output
    kbd_64 = KBinsDiscretizer(n_bins=3, encode=encode, dtype=np.float64)
    kbd_64.fit(X_input)
    Xt_64 = kbd_64.transform(X_input)

    assert_allclose_dense_sparse(Xt_32, Xt_64)
