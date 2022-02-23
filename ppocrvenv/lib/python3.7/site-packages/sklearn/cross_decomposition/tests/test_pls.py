import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal, assert_allclose

from sklearn.datasets import load_linnerud
from sklearn.cross_decomposition._pls import (
    _center_scale_xy,
    _get_first_singular_vectors_power_method,
    _get_first_singular_vectors_svd,
    _svd_flip_1d,
)
from sklearn.cross_decomposition import CCA
from sklearn.cross_decomposition import PLSSVD, PLSRegression, PLSCanonical
from sklearn.datasets import make_regression
from sklearn.utils import check_random_state
from sklearn.utils.extmath import svd_flip
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings


def assert_matrix_orthogonal(M):
    K = np.dot(M.T, M)
    assert_array_almost_equal(K, np.diag(np.diag(K)))


def test_pls_canonical_basics():
    # Basic checks for PLSCanonical
    d = load_linnerud()
    X = d.data
    Y = d.target

    pls = PLSCanonical(n_components=X.shape[1])
    pls.fit(X, Y)

    assert_matrix_orthogonal(pls.x_weights_)
    assert_matrix_orthogonal(pls.y_weights_)
    assert_matrix_orthogonal(pls._x_scores)
    assert_matrix_orthogonal(pls._y_scores)

    # Check X = TP' and Y = UQ'
    T = pls._x_scores
    P = pls.x_loadings_
    U = pls._y_scores
    Q = pls.y_loadings_
    # Need to scale first
    Xc, Yc, x_mean, y_mean, x_std, y_std = _center_scale_xy(
        X.copy(), Y.copy(), scale=True
    )
    assert_array_almost_equal(Xc, np.dot(T, P.T))
    assert_array_almost_equal(Yc, np.dot(U, Q.T))

    # Check that rotations on training data lead to scores
    Xt = pls.transform(X)
    assert_array_almost_equal(Xt, pls._x_scores)
    Xt, Yt = pls.transform(X, Y)
    assert_array_almost_equal(Xt, pls._x_scores)
    assert_array_almost_equal(Yt, pls._y_scores)

    # Check that inverse_transform works
    X_back = pls.inverse_transform(Xt)
    assert_array_almost_equal(X_back, X)


def test_sanity_check_pls_regression():
    # Sanity check for PLSRegression
    # The results were checked against the R-packages plspm, misOmics and pls

    d = load_linnerud()
    X = d.data
    Y = d.target

    pls = PLSRegression(n_components=X.shape[1])
    pls.fit(X, Y)

    expected_x_weights = np.array(
        [
            [-0.61330704, -0.00443647, 0.78983213],
            [-0.74697144, -0.32172099, -0.58183269],
            [-0.25668686, 0.94682413, -0.19399983],
        ]
    )

    expected_x_loadings = np.array(
        [
            [-0.61470416, -0.24574278, 0.78983213],
            [-0.65625755, -0.14396183, -0.58183269],
            [-0.51733059, 1.00609417, -0.19399983],
        ]
    )

    expected_y_weights = np.array(
        [
            [+0.32456184, 0.29892183, 0.20316322],
            [+0.42439636, 0.61970543, 0.19320542],
            [-0.13143144, -0.26348971, -0.17092916],
        ]
    )

    expected_y_loadings = np.array(
        [
            [+0.32456184, 0.29892183, 0.20316322],
            [+0.42439636, 0.61970543, 0.19320542],
            [-0.13143144, -0.26348971, -0.17092916],
        ]
    )

    assert_array_almost_equal(np.abs(pls.x_loadings_), np.abs(expected_x_loadings))
    assert_array_almost_equal(np.abs(pls.x_weights_), np.abs(expected_x_weights))
    assert_array_almost_equal(np.abs(pls.y_loadings_), np.abs(expected_y_loadings))
    assert_array_almost_equal(np.abs(pls.y_weights_), np.abs(expected_y_weights))

    # The R / Python difference in the signs should be consistent across
    # loadings, weights, etc.
    x_loadings_sign_flip = np.sign(pls.x_loadings_ / expected_x_loadings)
    x_weights_sign_flip = np.sign(pls.x_weights_ / expected_x_weights)
    y_weights_sign_flip = np.sign(pls.y_weights_ / expected_y_weights)
    y_loadings_sign_flip = np.sign(pls.y_loadings_ / expected_y_loadings)
    assert_array_almost_equal(x_loadings_sign_flip, x_weights_sign_flip)
    assert_array_almost_equal(y_loadings_sign_flip, y_weights_sign_flip)


def test_sanity_check_pls_regression_constant_column_Y():
    # Check behavior when the first column of Y is constant
    # The results are checked against a modified version of plsreg2
    # from the R-package plsdepot
    d = load_linnerud()
    X = d.data
    Y = d.target
    Y[:, 0] = 1
    pls = PLSRegression(n_components=X.shape[1])
    pls.fit(X, Y)

    expected_x_weights = np.array(
        [
            [-0.6273573, 0.007081799, 0.7786994],
            [-0.7493417, -0.277612681, -0.6011807],
            [-0.2119194, 0.960666981, -0.1794690],
        ]
    )

    expected_x_loadings = np.array(
        [
            [-0.6273512, -0.22464538, 0.7786994],
            [-0.6643156, -0.09871193, -0.6011807],
            [-0.5125877, 1.01407380, -0.1794690],
        ]
    )

    expected_y_loadings = np.array(
        [
            [0.0000000, 0.0000000, 0.0000000],
            [0.4357300, 0.5828479, 0.2174802],
            [-0.1353739, -0.2486423, -0.1810386],
        ]
    )

    assert_array_almost_equal(np.abs(expected_x_weights), np.abs(pls.x_weights_))
    assert_array_almost_equal(np.abs(expected_x_loadings), np.abs(pls.x_loadings_))
    # For the PLSRegression with default parameters, y_loadings == y_weights
    assert_array_almost_equal(np.abs(pls.y_loadings_), np.abs(expected_y_loadings))
    assert_array_almost_equal(np.abs(pls.y_weights_), np.abs(expected_y_loadings))

    x_loadings_sign_flip = np.sign(expected_x_loadings / pls.x_loadings_)
    x_weights_sign_flip = np.sign(expected_x_weights / pls.x_weights_)
    # we ignore the first full-zeros row for y
    y_loadings_sign_flip = np.sign(expected_y_loadings[1:] / pls.y_loadings_[1:])

    assert_array_equal(x_loadings_sign_flip, x_weights_sign_flip)
    assert_array_equal(x_loadings_sign_flip[1:], y_loadings_sign_flip)


def test_sanity_check_pls_canonical():
    # Sanity check for PLSCanonical
    # The results were checked against the R-package plspm

    d = load_linnerud()
    X = d.data
    Y = d.target

    pls = PLSCanonical(n_components=X.shape[1])
    pls.fit(X, Y)

    expected_x_weights = np.array(
        [
            [-0.61330704, 0.25616119, -0.74715187],
            [-0.74697144, 0.11930791, 0.65406368],
            [-0.25668686, -0.95924297, -0.11817271],
        ]
    )

    expected_x_rotations = np.array(
        [
            [-0.61330704, 0.41591889, -0.62297525],
            [-0.74697144, 0.31388326, 0.77368233],
            [-0.25668686, -0.89237972, -0.24121788],
        ]
    )

    expected_y_weights = np.array(
        [
            [+0.58989127, 0.7890047, 0.1717553],
            [+0.77134053, -0.61351791, 0.16920272],
            [-0.23887670, -0.03267062, 0.97050016],
        ]
    )

    expected_y_rotations = np.array(
        [
            [+0.58989127, 0.7168115, 0.30665872],
            [+0.77134053, -0.70791757, 0.19786539],
            [-0.23887670, -0.00343595, 0.94162826],
        ]
    )

    assert_array_almost_equal(np.abs(pls.x_rotations_), np.abs(expected_x_rotations))
    assert_array_almost_equal(np.abs(pls.x_weights_), np.abs(expected_x_weights))
    assert_array_almost_equal(np.abs(pls.y_rotations_), np.abs(expected_y_rotations))
    assert_array_almost_equal(np.abs(pls.y_weights_), np.abs(expected_y_weights))

    x_rotations_sign_flip = np.sign(pls.x_rotations_ / expected_x_rotations)
    x_weights_sign_flip = np.sign(pls.x_weights_ / expected_x_weights)
    y_rotations_sign_flip = np.sign(pls.y_rotations_ / expected_y_rotations)
    y_weights_sign_flip = np.sign(pls.y_weights_ / expected_y_weights)
    assert_array_almost_equal(x_rotations_sign_flip, x_weights_sign_flip)
    assert_array_almost_equal(y_rotations_sign_flip, y_weights_sign_flip)

    assert_matrix_orthogonal(pls.x_weights_)
    assert_matrix_orthogonal(pls.y_weights_)

    assert_matrix_orthogonal(pls._x_scores)
    assert_matrix_orthogonal(pls._y_scores)


def test_sanity_check_pls_canonical_random():
    # Sanity check for PLSCanonical on random data
    # The results were checked against the R-package plspm
    n = 500
    p_noise = 10
    q_noise = 5
    # 2 latents vars:
    rng = check_random_state(11)
    l1 = rng.normal(size=n)
    l2 = rng.normal(size=n)
    latents = np.array([l1, l1, l2, l2]).T
    X = latents + rng.normal(size=4 * n).reshape((n, 4))
    Y = latents + rng.normal(size=4 * n).reshape((n, 4))
    X = np.concatenate((X, rng.normal(size=p_noise * n).reshape(n, p_noise)), axis=1)
    Y = np.concatenate((Y, rng.normal(size=q_noise * n).reshape(n, q_noise)), axis=1)

    pls = PLSCanonical(n_components=3)
    pls.fit(X, Y)

    expected_x_weights = np.array(
        [
            [0.65803719, 0.19197924, 0.21769083],
            [0.7009113, 0.13303969, -0.15376699],
            [0.13528197, -0.68636408, 0.13856546],
            [0.16854574, -0.66788088, -0.12485304],
            [-0.03232333, -0.04189855, 0.40690153],
            [0.1148816, -0.09643158, 0.1613305],
            [0.04792138, -0.02384992, 0.17175319],
            [-0.06781, -0.01666137, -0.18556747],
            [-0.00266945, -0.00160224, 0.11893098],
            [-0.00849528, -0.07706095, 0.1570547],
            [-0.00949471, -0.02964127, 0.34657036],
            [-0.03572177, 0.0945091, 0.3414855],
            [0.05584937, -0.02028961, -0.57682568],
            [0.05744254, -0.01482333, -0.17431274],
        ]
    )

    expected_x_loadings = np.array(
        [
            [0.65649254, 0.1847647, 0.15270699],
            [0.67554234, 0.15237508, -0.09182247],
            [0.19219925, -0.67750975, 0.08673128],
            [0.2133631, -0.67034809, -0.08835483],
            [-0.03178912, -0.06668336, 0.43395268],
            [0.15684588, -0.13350241, 0.20578984],
            [0.03337736, -0.03807306, 0.09871553],
            [-0.06199844, 0.01559854, -0.1881785],
            [0.00406146, -0.00587025, 0.16413253],
            [-0.00374239, -0.05848466, 0.19140336],
            [0.00139214, -0.01033161, 0.32239136],
            [-0.05292828, 0.0953533, 0.31916881],
            [0.04031924, -0.01961045, -0.65174036],
            [0.06172484, -0.06597366, -0.1244497],
        ]
    )

    expected_y_weights = np.array(
        [
            [0.66101097, 0.18672553, 0.22826092],
            [0.69347861, 0.18463471, -0.23995597],
            [0.14462724, -0.66504085, 0.17082434],
            [0.22247955, -0.6932605, -0.09832993],
            [0.07035859, 0.00714283, 0.67810124],
            [0.07765351, -0.0105204, -0.44108074],
            [-0.00917056, 0.04322147, 0.10062478],
            [-0.01909512, 0.06182718, 0.28830475],
            [0.01756709, 0.04797666, 0.32225745],
        ]
    )

    expected_y_loadings = np.array(
        [
            [0.68568625, 0.1674376, 0.0969508],
            [0.68782064, 0.20375837, -0.1164448],
            [0.11712173, -0.68046903, 0.12001505],
            [0.17860457, -0.6798319, -0.05089681],
            [0.06265739, -0.0277703, 0.74729584],
            [0.0914178, 0.00403751, -0.5135078],
            [-0.02196918, -0.01377169, 0.09564505],
            [-0.03288952, 0.09039729, 0.31858973],
            [0.04287624, 0.05254676, 0.27836841],
        ]
    )

    assert_array_almost_equal(np.abs(pls.x_loadings_), np.abs(expected_x_loadings))
    assert_array_almost_equal(np.abs(pls.x_weights_), np.abs(expected_x_weights))
    assert_array_almost_equal(np.abs(pls.y_loadings_), np.abs(expected_y_loadings))
    assert_array_almost_equal(np.abs(pls.y_weights_), np.abs(expected_y_weights))

    x_loadings_sign_flip = np.sign(pls.x_loadings_ / expected_x_loadings)
    x_weights_sign_flip = np.sign(pls.x_weights_ / expected_x_weights)
    y_weights_sign_flip = np.sign(pls.y_weights_ / expected_y_weights)
    y_loadings_sign_flip = np.sign(pls.y_loadings_ / expected_y_loadings)
    assert_array_almost_equal(x_loadings_sign_flip, x_weights_sign_flip)
    assert_array_almost_equal(y_loadings_sign_flip, y_weights_sign_flip)

    assert_matrix_orthogonal(pls.x_weights_)
    assert_matrix_orthogonal(pls.y_weights_)

    assert_matrix_orthogonal(pls._x_scores)
    assert_matrix_orthogonal(pls._y_scores)


def test_convergence_fail():
    # Make sure ConvergenceWarning is raised if max_iter is too small
    d = load_linnerud()
    X = d.data
    Y = d.target
    pls_nipals = PLSCanonical(n_components=X.shape[1], max_iter=2)
    with pytest.warns(ConvergenceWarning):
        pls_nipals.fit(X, Y)


@pytest.mark.filterwarnings("ignore:.*`scores_` was deprecated")  # 1.1
@pytest.mark.parametrize("Est", (PLSSVD, PLSRegression, PLSCanonical))
def test_attibutes_shapes(Est):
    # Make sure attributes are of the correct shape depending on n_components
    d = load_linnerud()
    X = d.data
    Y = d.target
    n_components = 2
    pls = Est(n_components=n_components)
    pls.fit(X, Y)
    assert all(
        attr.shape[1] == n_components for attr in (pls.x_weights_, pls.y_weights_)
    )
    # TODO: remove in 1.1
    with ignore_warnings(category=FutureWarning):
        assert all(
            attr.shape[1] == n_components for attr in (pls.x_scores_, pls.y_scores_)
        )


@pytest.mark.parametrize("Est", (PLSRegression, PLSCanonical, CCA))
def test_univariate_equivalence(Est):
    # Ensure 2D Y with 1 column is equivalent to 1D Y
    d = load_linnerud()
    X = d.data
    Y = d.target

    est = Est(n_components=1)
    one_d_coeff = est.fit(X, Y[:, 0]).coef_
    two_d_coeff = est.fit(X, Y[:, :1]).coef_

    assert one_d_coeff.shape == two_d_coeff.shape
    assert_array_almost_equal(one_d_coeff, two_d_coeff)


@pytest.mark.parametrize("Est", (PLSRegression, PLSCanonical, CCA, PLSSVD))
def test_copy(Est):
    # check that the "copy" keyword works
    d = load_linnerud()
    X = d.data
    Y = d.target
    X_orig = X.copy()

    # copy=True won't modify inplace
    pls = Est(copy=True).fit(X, Y)
    assert_array_equal(X, X_orig)

    # copy=False will modify inplace
    with pytest.raises(AssertionError):
        Est(copy=False).fit(X, Y)
        assert_array_almost_equal(X, X_orig)

    if Est is PLSSVD:
        return  # PLSSVD does not support copy param in predict or transform

    X_orig = X.copy()
    with pytest.raises(AssertionError):
        pls.transform(X, Y, copy=False),
        assert_array_almost_equal(X, X_orig)

    X_orig = X.copy()
    with pytest.raises(AssertionError):
        pls.predict(X, copy=False),
        assert_array_almost_equal(X, X_orig)

    # Make sure copy=True gives same transform and predictions as predict=False
    assert_array_almost_equal(
        pls.transform(X, Y, copy=True), pls.transform(X.copy(), Y.copy(), copy=False)
    )
    assert_array_almost_equal(
        pls.predict(X, copy=True), pls.predict(X.copy(), copy=False)
    )


def _generate_test_scale_and_stability_datasets():
    """Generate dataset for test_scale_and_stability"""
    # dataset for non-regression 7818
    rng = np.random.RandomState(0)
    n_samples = 1000
    n_targets = 5
    n_features = 10
    Q = rng.randn(n_targets, n_features)
    Y = rng.randn(n_samples, n_targets)
    X = np.dot(Y, Q) + 2 * rng.randn(n_samples, n_features) + 1
    X *= 1000
    yield X, Y

    # Data set where one of the features is constraint
    X, Y = load_linnerud(return_X_y=True)
    # causes X[:, -1].std() to be zero
    X[:, -1] = 1.0
    yield X, Y

    X = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [2.0, 2.0, 2.0], [3.0, 5.0, 4.0]])
    Y = np.array([[0.1, -0.2], [0.9, 1.1], [6.2, 5.9], [11.9, 12.3]])
    yield X, Y

    # Seeds that provide a non-regression test for #18746, where CCA fails
    seeds = [530, 741]
    for seed in seeds:
        rng = np.random.RandomState(seed)
        X = rng.randn(4, 3)
        Y = rng.randn(4, 2)
        yield X, Y


@pytest.mark.parametrize("Est", (CCA, PLSCanonical, PLSRegression, PLSSVD))
@pytest.mark.parametrize("X, Y", _generate_test_scale_and_stability_datasets())
def test_scale_and_stability(Est, X, Y):
    """scale=True is equivalent to scale=False on centered/scaled data
    This allows to check numerical stability over platforms as well"""

    X_s, Y_s, *_ = _center_scale_xy(X, Y)

    X_score, Y_score = Est(scale=True).fit_transform(X, Y)
    X_s_score, Y_s_score = Est(scale=False).fit_transform(X_s, Y_s)

    assert_allclose(X_s_score, X_score, atol=1e-4)
    assert_allclose(Y_s_score, Y_score, atol=1e-4)


@pytest.mark.parametrize("Est", (PLSSVD, PLSCanonical, CCA))
@pytest.mark.parametrize("n_components", (0, 4))
def test_n_components_bounds(Est, n_components):
    # n_components should be in [1, min(n_samples, n_features, n_targets)]
    # TODO: catch error instead of warning in 1.1
    rng = np.random.RandomState(0)
    X = rng.randn(10, 5)
    Y = rng.randn(10, 3)
    est = Est(n_components=n_components)
    with pytest.warns(FutureWarning, match="n_components=3 will be used instead"):
        est.fit(X, Y)
        # make sure upper bound of rank is used as a fallback
        assert est.transform(X).shape[1] == 3


@pytest.mark.parametrize("n_components", (0, 6))
def test_n_components_bounds_pls_regression(n_components):
    # For PLSRegression, the upper bound for n_components is n_features
    # TODO: catch error instead of warning in 1.1
    rng = np.random.RandomState(0)
    X = rng.randn(10, 5)
    Y = rng.randn(10, 3)
    est = PLSRegression(n_components=n_components)
    with pytest.warns(FutureWarning, match="n_components=5 will be used instead"):
        est.fit(X, Y)
        # make sure upper bound of rank is used as a fallback
        assert est.transform(X).shape[1] == 5


@pytest.mark.parametrize("Est", (PLSSVD, CCA, PLSCanonical))
def test_scores_deprecations(Est):
    # Make sure x_scores_ and y_scores_ are deprecated.
    # It's not deprecated for PLSRegression because y_score_ is different from
    # transform(Y_train)
    # TODO: remove attributes and test in 1.1
    rng = np.random.RandomState(0)
    X = rng.randn(10, 5)
    Y = rng.randn(10, 3)
    est = Est().fit(X, Y)
    with pytest.warns(FutureWarning, match="`x_scores_` was deprecated"):
        assert_allclose(est.x_scores_, est.transform(X))
    with pytest.warns(FutureWarning, match="`y_scores_` was deprecated"):
        assert_allclose(est.y_scores_, est.transform(X, Y)[1])


@pytest.mark.parametrize("Est", (PLSRegression, PLSCanonical, CCA))
def test_norm_y_weights_deprecation(Est):
    rng = np.random.RandomState(0)
    X = rng.randn(10, 5)
    Y = rng.randn(10, 3)
    est = Est().fit(X, Y)
    with pytest.warns(FutureWarning, match="`norm_y_weights` was deprecated"):
        est.norm_y_weights


# TODO: Remove test in 1.1
@pytest.mark.parametrize("Estimator", (PLSRegression, PLSCanonical, CCA, PLSSVD))
@pytest.mark.parametrize("attribute", ("x_mean_", "y_mean_", "x_std_", "y_std_"))
def test_mean_and_std_deprecation(Estimator, attribute):
    rng = np.random.RandomState(0)
    X = rng.randn(10, 5)
    Y = rng.randn(10, 3)
    estimator = Estimator().fit(X, Y)
    with pytest.warns(FutureWarning, match=f"`{attribute}` was deprecated"):
        getattr(estimator, attribute)


@pytest.mark.parametrize("n_samples, n_features", [(100, 10), (100, 200)])
@pytest.mark.parametrize("seed", range(10))
def test_singular_value_helpers(n_samples, n_features, seed):
    # Make sure SVD and power method give approximately the same results
    X, Y = make_regression(n_samples, n_features, n_targets=5, random_state=seed)
    u1, v1, _ = _get_first_singular_vectors_power_method(X, Y, norm_y_weights=True)
    u2, v2 = _get_first_singular_vectors_svd(X, Y)

    _svd_flip_1d(u1, v1)
    _svd_flip_1d(u2, v2)

    rtol = 1e-1
    assert_allclose(u1, u2, rtol=rtol)
    assert_allclose(v1, v2, rtol=rtol)


def test_one_component_equivalence():
    # PLSSVD, PLSRegression and PLSCanonical should all be equivalent when
    # n_components is 1
    X, Y = make_regression(100, 10, n_targets=5, random_state=0)
    svd = PLSSVD(n_components=1).fit(X, Y).transform(X)
    reg = PLSRegression(n_components=1).fit(X, Y).transform(X)
    canonical = PLSCanonical(n_components=1).fit(X, Y).transform(X)

    assert_allclose(svd, reg, rtol=1e-2)
    assert_allclose(svd, canonical, rtol=1e-2)


def test_svd_flip_1d():
    # Make sure svd_flip_1d is equivalent to svd_flip
    u = np.array([1, -4, 2])
    v = np.array([1, 2, 3])

    u_expected, v_expected = svd_flip(u.reshape(-1, 1), v.reshape(1, -1))
    _svd_flip_1d(u, v)  # inplace

    assert_allclose(u, u_expected.ravel())
    assert_allclose(u, [-1, 4, -2])

    assert_allclose(v, v_expected.ravel())
    assert_allclose(v, [-1, -2, -3])


def test_loadings_converges():
    """Test that CCA converges. Non-regression test for #19549."""
    X, y = make_regression(n_samples=200, n_features=20, n_targets=20, random_state=20)

    cca = CCA(n_components=10, max_iter=500)

    with pytest.warns(None) as record:
        cca.fit(X, y)
    # ConvergenceWarning should not be raised
    if len(record) > 0:
        pytest.fail(f"Unexpected warning: {str(record[0].message)}")

    # Loadings converges to reasonable values
    assert np.all(np.abs(cca.x_loadings_) < 1)


def test_pls_constant_y():
    """Checks warning when y is constant. Non-regression test for #19831"""
    rng = np.random.RandomState(42)
    x = rng.rand(100, 3)
    y = np.zeros(100)

    pls = PLSRegression()

    msg = "Y residual is constant at iteration"
    with pytest.warns(UserWarning, match=msg):
        pls.fit(x, y)

    assert_allclose(pls.x_rotations_, 0)
