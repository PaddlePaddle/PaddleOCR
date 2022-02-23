# Author: Christian Osendorfer <osendorf@gmail.com>
#         Alexandre Gramfort <alexandre.gramfort@inria.fr>
# License: BSD3

from itertools import combinations

import numpy as np
import pytest

from sklearn.utils._testing import assert_almost_equal
from sklearn.utils._testing import assert_array_almost_equal
from sklearn.exceptions import ConvergenceWarning
from sklearn.decomposition import FactorAnalysis
from sklearn.utils._testing import ignore_warnings
from sklearn.decomposition._factor_analysis import _ortho_rotation


# Ignore warnings from switching to more power iterations in randomized_svd
@ignore_warnings
def test_factor_analysis():
    # Test FactorAnalysis ability to recover the data covariance structure
    rng = np.random.RandomState(0)
    n_samples, n_features, n_components = 20, 5, 3

    # Some random settings for the generative model
    W = rng.randn(n_components, n_features)
    # latent variable of dim 3, 20 of it
    h = rng.randn(n_samples, n_components)
    # using gamma to model different noise variance
    # per component
    noise = rng.gamma(1, size=n_features) * rng.randn(n_samples, n_features)

    # generate observations
    # wlog, mean is 0
    X = np.dot(h, W) + noise

    with pytest.raises(ValueError):
        FactorAnalysis(svd_method="foo")
    fa_fail = FactorAnalysis()
    fa_fail.svd_method = "foo"
    with pytest.raises(ValueError):
        fa_fail.fit(X)
    fas = []
    for method in ["randomized", "lapack"]:
        fa = FactorAnalysis(n_components=n_components, svd_method=method)
        fa.fit(X)
        fas.append(fa)

        X_t = fa.transform(X)
        assert X_t.shape == (n_samples, n_components)

        assert_almost_equal(fa.loglike_[-1], fa.score_samples(X).sum())
        assert_almost_equal(fa.score_samples(X).mean(), fa.score(X))

        diff = np.all(np.diff(fa.loglike_))
        assert diff > 0.0, "Log likelihood dif not increase"

        # Sample Covariance
        scov = np.cov(X, rowvar=0.0, bias=1.0)

        # Model Covariance
        mcov = fa.get_covariance()
        diff = np.sum(np.abs(scov - mcov)) / W.size
        assert diff < 0.1, "Mean absolute difference is %f" % diff
        fa = FactorAnalysis(
            n_components=n_components, noise_variance_init=np.ones(n_features)
        )
        with pytest.raises(ValueError):
            fa.fit(X[:, :2])

    def f(x, y):
        return np.abs(getattr(x, y))  # sign will not be equal

    fa1, fa2 = fas
    for attr in ["loglike_", "components_", "noise_variance_"]:
        assert_almost_equal(f(fa1, attr), f(fa2, attr))

    fa1.max_iter = 1
    fa1.verbose = True
    with pytest.warns(ConvergenceWarning):
        fa1.fit(X)

    # Test get_covariance and get_precision with n_components == n_features
    # with n_components < n_features and with n_components == 0
    for n_components in [0, 2, X.shape[1]]:
        fa.n_components = n_components
        fa.fit(X)
        cov = fa.get_covariance()
        precision = fa.get_precision()
        assert_array_almost_equal(np.dot(cov, precision), np.eye(X.shape[1]), 12)

    # test rotation
    n_components = 2

    results, projections = {}, {}
    for method in (None, "varimax", "quartimax"):
        fa_var = FactorAnalysis(n_components=n_components, rotation=method)
        results[method] = fa_var.fit_transform(X)
        projections[method] = fa_var.get_covariance()
    for rot1, rot2 in combinations([None, "varimax", "quartimax"], 2):
        assert not np.allclose(results[rot1], results[rot2])
        assert np.allclose(projections[rot1], projections[rot2], atol=3)

    with pytest.raises(ValueError):
        FactorAnalysis(rotation="not_implemented").fit_transform(X)

    # test against R's psych::principal with rotate="varimax"
    # (i.e., the values below stem from rotating the components in R)
    # R's factor analysis returns quite different values; therefore, we only
    # test the rotation itself
    factors = np.array(
        [
            [0.89421016, -0.35854928, -0.27770122, 0.03773647],
            [-0.45081822, -0.89132754, 0.0932195, -0.01787973],
            [0.99500666, -0.02031465, 0.05426497, -0.11539407],
            [0.96822861, -0.06299656, 0.24411001, 0.07540887],
        ]
    )
    r_solution = np.array(
        [[0.962, 0.052], [-0.141, 0.989], [0.949, -0.300], [0.937, -0.251]]
    )
    rotated = _ortho_rotation(factors[:, :n_components], method="varimax").T
    assert_array_almost_equal(np.abs(rotated), np.abs(r_solution), decimal=3)
