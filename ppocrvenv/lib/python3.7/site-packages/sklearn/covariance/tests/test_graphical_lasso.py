""" Test the graphical_lasso module.
"""
import sys
import pytest

import numpy as np
from scipy import linalg

from numpy.testing import assert_allclose
from sklearn.utils._testing import assert_array_almost_equal
from sklearn.utils._testing import assert_array_less

from sklearn.covariance import (
    graphical_lasso,
    GraphicalLasso,
    GraphicalLassoCV,
    empirical_covariance,
)
from sklearn.datasets import make_sparse_spd_matrix
from io import StringIO
from sklearn.utils import check_random_state
from sklearn import datasets


def test_graphical_lasso(random_state=0):
    # Sample data from a sparse multivariate normal
    dim = 20
    n_samples = 100
    random_state = check_random_state(random_state)
    prec = make_sparse_spd_matrix(dim, alpha=0.95, random_state=random_state)
    cov = linalg.inv(prec)
    X = random_state.multivariate_normal(np.zeros(dim), cov, size=n_samples)
    emp_cov = empirical_covariance(X)

    for alpha in (0.0, 0.1, 0.25):
        covs = dict()
        icovs = dict()
        for method in ("cd", "lars"):
            cov_, icov_, costs = graphical_lasso(
                emp_cov, return_costs=True, alpha=alpha, mode=method
            )
            covs[method] = cov_
            icovs[method] = icov_
            costs, dual_gap = np.array(costs).T
            # Check that the costs always decrease (doesn't hold if alpha == 0)
            if not alpha == 0:
                assert_array_less(np.diff(costs), 0)
        # Check that the 2 approaches give similar results
        assert_array_almost_equal(covs["cd"], covs["lars"], decimal=4)
        assert_array_almost_equal(icovs["cd"], icovs["lars"], decimal=4)

    # Smoke test the estimator
    model = GraphicalLasso(alpha=0.25).fit(X)
    model.score(X)
    assert_array_almost_equal(model.covariance_, covs["cd"], decimal=4)
    assert_array_almost_equal(model.covariance_, covs["lars"], decimal=4)

    # For a centered matrix, assume_centered could be chosen True or False
    # Check that this returns indeed the same result for centered data
    Z = X - X.mean(0)
    precs = list()
    for assume_centered in (False, True):
        prec_ = GraphicalLasso(assume_centered=assume_centered).fit(Z).precision_
        precs.append(prec_)
    assert_array_almost_equal(precs[0], precs[1])


def test_graphical_lasso_iris():
    # Hard-coded solution from R glasso package for alpha=1.0
    # (need to set penalize.diagonal to FALSE)
    cov_R = np.array(
        [
            [0.68112222, 0.0000000, 0.265820, 0.02464314],
            [0.00000000, 0.1887129, 0.000000, 0.00000000],
            [0.26582000, 0.0000000, 3.095503, 0.28697200],
            [0.02464314, 0.0000000, 0.286972, 0.57713289],
        ]
    )
    icov_R = np.array(
        [
            [1.5190747, 0.000000, -0.1304475, 0.0000000],
            [0.0000000, 5.299055, 0.0000000, 0.0000000],
            [-0.1304475, 0.000000, 0.3498624, -0.1683946],
            [0.0000000, 0.000000, -0.1683946, 1.8164353],
        ]
    )
    X = datasets.load_iris().data
    emp_cov = empirical_covariance(X)
    for method in ("cd", "lars"):
        cov, icov = graphical_lasso(emp_cov, alpha=1.0, return_costs=False, mode=method)
        assert_array_almost_equal(cov, cov_R)
        assert_array_almost_equal(icov, icov_R)


def test_graph_lasso_2D():
    # Hard-coded solution from Python skggm package
    # obtained by calling `quic(emp_cov, lam=.1, tol=1e-8)`
    cov_skggm = np.array([[3.09550269, 1.186972], [1.186972, 0.57713289]])

    icov_skggm = np.array([[1.52836773, -3.14334831], [-3.14334831, 8.19753385]])
    X = datasets.load_iris().data[:, 2:]
    emp_cov = empirical_covariance(X)
    for method in ("cd", "lars"):
        cov, icov = graphical_lasso(emp_cov, alpha=0.1, return_costs=False, mode=method)
        assert_array_almost_equal(cov, cov_skggm)
        assert_array_almost_equal(icov, icov_skggm)


def test_graphical_lasso_iris_singular():
    # Small subset of rows to test the rank-deficient case
    # Need to choose samples such that none of the variances are zero
    indices = np.arange(10, 13)

    # Hard-coded solution from R glasso package for alpha=0.01
    cov_R = np.array(
        [
            [0.08, 0.056666662595, 0.00229729713223, 0.00153153142149],
            [0.056666662595, 0.082222222222, 0.00333333333333, 0.00222222222222],
            [0.002297297132, 0.003333333333, 0.00666666666667, 0.00009009009009],
            [0.001531531421, 0.002222222222, 0.00009009009009, 0.00222222222222],
        ]
    )
    icov_R = np.array(
        [
            [24.42244057, -16.831679593, 0.0, 0.0],
            [-16.83168201, 24.351841681, -6.206896552, -12.5],
            [0.0, -6.206896171, 153.103448276, 0.0],
            [0.0, -12.499999143, 0.0, 462.5],
        ]
    )
    X = datasets.load_iris().data[indices, :]
    emp_cov = empirical_covariance(X)
    for method in ("cd", "lars"):
        cov, icov = graphical_lasso(
            emp_cov, alpha=0.01, return_costs=False, mode=method
        )
        assert_array_almost_equal(cov, cov_R, decimal=5)
        assert_array_almost_equal(icov, icov_R, decimal=5)


def test_graphical_lasso_cv(random_state=1):
    # Sample data from a sparse multivariate normal
    dim = 5
    n_samples = 6
    random_state = check_random_state(random_state)
    prec = make_sparse_spd_matrix(dim, alpha=0.96, random_state=random_state)
    cov = linalg.inv(prec)
    X = random_state.multivariate_normal(np.zeros(dim), cov, size=n_samples)
    # Capture stdout, to smoke test the verbose mode
    orig_stdout = sys.stdout
    try:
        sys.stdout = StringIO()
        # We need verbose very high so that Parallel prints on stdout
        GraphicalLassoCV(verbose=100, alphas=5, tol=1e-1).fit(X)
    finally:
        sys.stdout = orig_stdout

    # Smoke test with specified alphas
    GraphicalLassoCV(alphas=[0.8, 0.5], tol=1e-1, n_jobs=1).fit(X)


# TODO: Remove in 1.1 when grid_scores_ is deprecated
def test_graphical_lasso_cv_grid_scores_and_cv_alphas_deprecated():
    splits = 4
    n_alphas = 5
    n_refinements = 3
    true_cov = np.array(
        [
            [0.8, 0.0, 0.2, 0.0],
            [0.0, 0.4, 0.0, 0.0],
            [0.2, 0.0, 0.3, 0.1],
            [0.0, 0.0, 0.1, 0.7],
        ]
    )
    rng = np.random.RandomState(0)
    X = rng.multivariate_normal(mean=[0, 0, 0, 0], cov=true_cov, size=200)
    cov = GraphicalLassoCV(cv=splits, alphas=n_alphas, n_refinements=n_refinements).fit(
        X
    )

    total_alphas = n_refinements * n_alphas + 1
    msg = (
        r"The `grid_scores_` attribute is deprecated in version 0\.24 in "
        r"favor of `cv_results_` and will be removed in version 1\.1 "
        r"\(renaming of 0\.26\)."
    )
    with pytest.warns(FutureWarning, match=msg):
        assert cov.grid_scores_.shape == (total_alphas, splits)

    msg = (
        r"The `cv_alphas_` attribute is deprecated in version 0\.24 in "
        r"favor of `cv_results_\['alpha'\]` and will be removed in version "
        r"1\.1 \(renaming of 0\.26\)"
    )
    with pytest.warns(FutureWarning, match=msg):
        assert len(cov.cv_alphas_) == total_alphas


# TODO: Remove `score` and `test_score` suffix in 1.2
@pytest.mark.parametrize("suffix", ["score", "test_score"])
@pytest.mark.filterwarnings("ignore:Key*:FutureWarning:sklearn")
def test_graphical_lasso_cv_scores(suffix):
    splits = 4
    n_alphas = 5
    n_refinements = 3
    true_cov = np.array(
        [
            [0.8, 0.0, 0.2, 0.0],
            [0.0, 0.4, 0.0, 0.0],
            [0.2, 0.0, 0.3, 0.1],
            [0.0, 0.0, 0.1, 0.7],
        ]
    )
    rng = np.random.RandomState(0)
    X = rng.multivariate_normal(mean=[0, 0, 0, 0], cov=true_cov, size=200)
    cov = GraphicalLassoCV(cv=splits, alphas=n_alphas, n_refinements=n_refinements).fit(
        X
    )

    cv_results = cov.cv_results_
    # alpha and one for each split

    total_alphas = n_refinements * n_alphas + 1
    keys = ["alphas"]
    split_keys = [f"split{i}_{suffix}" for i in range(splits)]
    for key in keys + split_keys:
        assert key in cv_results
        assert len(cv_results[key]) == total_alphas

    cv_scores = np.asarray([cov.cv_results_[key] for key in split_keys])
    expected_mean = cv_scores.mean(axis=0)
    expected_std = cv_scores.std(axis=0)

    assert_allclose(cov.cv_results_[f"mean_{suffix}"], expected_mean)
    assert_allclose(cov.cv_results_[f"std_{suffix}"], expected_std)


# TODO: Remove in 1.2 when mean_score, std_score, and split(k)_score is removed.
def test_graphical_lasso_cv_scores_deprecated():
    """Check that the following keys in cv_results_ are deprecated: `mean_score`,
    `std_score`, and `split(k)_score`."""
    splits = 4
    n_alphas = 5
    n_refinements = 3
    true_cov = np.array(
        [
            [0.8, 0.0, 0.2, 0.0],
            [0.0, 0.4, 0.0, 0.0],
            [0.2, 0.0, 0.3, 0.1],
            [0.0, 0.0, 0.1, 0.7],
        ]
    )
    rng = np.random.RandomState(0)
    X = rng.multivariate_normal(mean=[0, 0, 0, 0], cov=true_cov, size=200)
    cov = GraphicalLassoCV(cv=splits, alphas=n_alphas, n_refinements=n_refinements).fit(
        X
    )
    cv_results = cov.cv_results_

    deprecated_keys = ["mean_score", "std_score"] + [
        f"split{k}_score" for k in range(splits)
    ]

    for deprecated_key in deprecated_keys:
        new_key = deprecated_key.replace("_score", "_test_score")
        msg = (
            f"Key: '{deprecated_key}', is deprecated in 1.0 and will be removed in 1.2."
            f" Use '{new_key}' instead"
        )
        with pytest.warns(FutureWarning, match=msg):
            cv_results[deprecated_key]
