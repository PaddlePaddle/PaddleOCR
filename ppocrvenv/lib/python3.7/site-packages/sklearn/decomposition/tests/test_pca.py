import numpy as np
import scipy as sp

import pytest

from sklearn.utils._testing import assert_allclose

from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.decomposition._pca import _assess_dimension
from sklearn.decomposition._pca import _infer_dimension

iris = datasets.load_iris()
PCA_SOLVERS = ["full", "arpack", "randomized", "auto"]


@pytest.mark.parametrize("svd_solver", PCA_SOLVERS)
@pytest.mark.parametrize("n_components", range(1, iris.data.shape[1]))
def test_pca(svd_solver, n_components):
    X = iris.data
    pca = PCA(n_components=n_components, svd_solver=svd_solver)

    # check the shape of fit.transform
    X_r = pca.fit(X).transform(X)
    assert X_r.shape[1] == n_components

    # check the equivalence of fit.transform and fit_transform
    X_r2 = pca.fit_transform(X)
    assert_allclose(X_r, X_r2)
    X_r = pca.transform(X)
    assert_allclose(X_r, X_r2)

    # Test get_covariance and get_precision
    cov = pca.get_covariance()
    precision = pca.get_precision()
    assert_allclose(np.dot(cov, precision), np.eye(X.shape[1]), atol=1e-12)


def test_no_empty_slice_warning():
    # test if we avoid numpy warnings for computing over empty arrays
    n_components = 10
    n_features = n_components + 2  # anything > n_comps triggered it in 0.16
    X = np.random.uniform(-1, 1, size=(n_components, n_features))
    pca = PCA(n_components=n_components)
    with pytest.warns(None) as record:
        pca.fit(X)
    assert not record.list


@pytest.mark.parametrize("copy", [True, False])
@pytest.mark.parametrize("solver", PCA_SOLVERS)
def test_whitening(solver, copy):
    # Check that PCA output has unit-variance
    rng = np.random.RandomState(0)
    n_samples = 100
    n_features = 80
    n_components = 30
    rank = 50

    # some low rank data with correlated features
    X = np.dot(
        rng.randn(n_samples, rank),
        np.dot(np.diag(np.linspace(10.0, 1.0, rank)), rng.randn(rank, n_features)),
    )
    # the component-wise variance of the first 50 features is 3 times the
    # mean component-wise variance of the remaining 30 features
    X[:, :50] *= 3

    assert X.shape == (n_samples, n_features)

    # the component-wise variance is thus highly varying:
    assert X.std(axis=0).std() > 43.8

    # whiten the data while projecting to the lower dim subspace
    X_ = X.copy()  # make sure we keep an original across iterations.
    pca = PCA(
        n_components=n_components,
        whiten=True,
        copy=copy,
        svd_solver=solver,
        random_state=0,
        iterated_power=7,
    )
    # test fit_transform
    X_whitened = pca.fit_transform(X_.copy())
    assert X_whitened.shape == (n_samples, n_components)
    X_whitened2 = pca.transform(X_)
    assert_allclose(X_whitened, X_whitened2, rtol=5e-4)

    assert_allclose(X_whitened.std(ddof=1, axis=0), np.ones(n_components))
    assert_allclose(X_whitened.mean(axis=0), np.zeros(n_components), atol=1e-12)

    X_ = X.copy()
    pca = PCA(
        n_components=n_components, whiten=False, copy=copy, svd_solver=solver
    ).fit(X_)
    X_unwhitened = pca.transform(X_)
    assert X_unwhitened.shape == (n_samples, n_components)

    # in that case the output components still have varying variances
    assert X_unwhitened.std(axis=0).std() == pytest.approx(74.1, rel=1e-1)
    # we always center, so no test for non-centering.


@pytest.mark.parametrize("svd_solver", ["arpack", "randomized"])
def test_pca_explained_variance_equivalence_solver(svd_solver):
    rng = np.random.RandomState(0)
    n_samples, n_features = 100, 80
    X = rng.randn(n_samples, n_features)

    pca_full = PCA(n_components=2, svd_solver="full")
    pca_other = PCA(n_components=2, svd_solver=svd_solver, random_state=0)

    pca_full.fit(X)
    pca_other.fit(X)

    assert_allclose(
        pca_full.explained_variance_, pca_other.explained_variance_, rtol=5e-2
    )
    assert_allclose(
        pca_full.explained_variance_ratio_,
        pca_other.explained_variance_ratio_,
        rtol=5e-2,
    )


@pytest.mark.parametrize(
    "X",
    [
        np.random.RandomState(0).randn(100, 80),
        datasets.make_classification(100, 80, n_informative=78, random_state=0)[0],
    ],
    ids=["random-data", "correlated-data"],
)
@pytest.mark.parametrize("svd_solver", PCA_SOLVERS)
def test_pca_explained_variance_empirical(X, svd_solver):
    pca = PCA(n_components=2, svd_solver=svd_solver, random_state=0)
    X_pca = pca.fit_transform(X)
    assert_allclose(pca.explained_variance_, np.var(X_pca, ddof=1, axis=0))

    expected_result = np.linalg.eig(np.cov(X, rowvar=False))[0]
    expected_result = sorted(expected_result, reverse=True)[:2]
    assert_allclose(pca.explained_variance_, expected_result, rtol=5e-3)


@pytest.mark.parametrize("svd_solver", ["arpack", "randomized"])
def test_pca_singular_values_consistency(svd_solver):
    rng = np.random.RandomState(0)
    n_samples, n_features = 100, 80
    X = rng.randn(n_samples, n_features)

    pca_full = PCA(n_components=2, svd_solver="full", random_state=rng)
    pca_other = PCA(n_components=2, svd_solver=svd_solver, random_state=rng)

    pca_full.fit(X)
    pca_other.fit(X)

    assert_allclose(pca_full.singular_values_, pca_other.singular_values_, rtol=5e-3)


@pytest.mark.parametrize("svd_solver", PCA_SOLVERS)
def test_pca_singular_values(svd_solver):
    rng = np.random.RandomState(0)
    n_samples, n_features = 100, 80
    X = rng.randn(n_samples, n_features)

    pca = PCA(n_components=2, svd_solver=svd_solver, random_state=rng)
    X_trans = pca.fit_transform(X)

    # compare to the Frobenius norm
    assert_allclose(
        np.sum(pca.singular_values_ ** 2), np.linalg.norm(X_trans, "fro") ** 2
    )
    # Compare to the 2-norms of the score vectors
    assert_allclose(pca.singular_values_, np.sqrt(np.sum(X_trans ** 2, axis=0)))

    # set the singular values and see what er get back
    n_samples, n_features = 100, 110
    X = rng.randn(n_samples, n_features)

    pca = PCA(n_components=3, svd_solver=svd_solver, random_state=rng)
    X_trans = pca.fit_transform(X)
    X_trans /= np.sqrt(np.sum(X_trans ** 2, axis=0))
    X_trans[:, 0] *= 3.142
    X_trans[:, 1] *= 2.718
    X_hat = np.dot(X_trans, pca.components_)
    pca.fit(X_hat)
    assert_allclose(pca.singular_values_, [3.142, 2.718, 1.0])


@pytest.mark.parametrize("svd_solver", PCA_SOLVERS)
def test_pca_check_projection(svd_solver):
    # Test that the projection of data is correct
    rng = np.random.RandomState(0)
    n, p = 100, 3
    X = rng.randn(n, p) * 0.1
    X[:10] += np.array([3, 4, 5])
    Xt = 0.1 * rng.randn(1, p) + np.array([3, 4, 5])

    Yt = PCA(n_components=2, svd_solver=svd_solver).fit(X).transform(Xt)
    Yt /= np.sqrt((Yt ** 2).sum())

    assert_allclose(np.abs(Yt[0][0]), 1.0, rtol=5e-3)


@pytest.mark.parametrize("svd_solver", PCA_SOLVERS)
def test_pca_check_projection_list(svd_solver):
    # Test that the projection of data is correct
    X = [[1.0, 0.0], [0.0, 1.0]]
    pca = PCA(n_components=1, svd_solver=svd_solver, random_state=0)
    X_trans = pca.fit_transform(X)
    assert X_trans.shape, (2, 1)
    assert_allclose(X_trans.mean(), 0.00, atol=1e-12)
    assert_allclose(X_trans.std(), 0.71, rtol=5e-3)


@pytest.mark.parametrize("svd_solver", ["full", "arpack", "randomized"])
@pytest.mark.parametrize("whiten", [False, True])
def test_pca_inverse(svd_solver, whiten):
    # Test that the projection of data can be inverted
    rng = np.random.RandomState(0)
    n, p = 50, 3
    X = rng.randn(n, p)  # spherical data
    X[:, 1] *= 0.00001  # make middle component relatively small
    X += [5, 4, 3]  # make a large mean

    # same check that we can find the original data from the transformed
    # signal (since the data is almost of rank n_components)
    pca = PCA(n_components=2, svd_solver=svd_solver, whiten=whiten).fit(X)
    Y = pca.transform(X)
    Y_inverse = pca.inverse_transform(Y)
    assert_allclose(X, Y_inverse, rtol=5e-6)


@pytest.mark.parametrize(
    "data", [np.array([[0, 1, 0], [1, 0, 0]]), np.array([[0, 1, 0], [1, 0, 0]]).T]
)
@pytest.mark.parametrize(
    "svd_solver, n_components, err_msg",
    [
        ("arpack", 0, r"must be between 1 and min\(n_samples, n_features\)"),
        ("randomized", 0, r"must be between 1 and min\(n_samples, n_features\)"),
        ("arpack", 2, r"must be strictly less than min"),
        (
            "auto",
            -1,
            (
                r"n_components={}L? must be between {}L? and "
                r"min\(n_samples, n_features\)={}L? with "
                r"svd_solver=\'{}\'"
            ),
        ),
        (
            "auto",
            3,
            (
                r"n_components={}L? must be between {}L? and "
                r"min\(n_samples, n_features\)={}L? with "
                r"svd_solver=\'{}\'"
            ),
        ),
        ("auto", 1.0, "must be of type int"),
    ],
)
def test_pca_validation(svd_solver, data, n_components, err_msg):
    # Ensures that solver-specific extreme inputs for the n_components
    # parameter raise errors
    smallest_d = 2  # The smallest dimension
    lower_limit = {"randomized": 1, "arpack": 1, "full": 0, "auto": 0}
    pca_fitted = PCA(n_components, svd_solver=svd_solver)

    solver_reported = "full" if svd_solver == "auto" else svd_solver
    err_msg = err_msg.format(
        n_components, lower_limit[svd_solver], smallest_d, solver_reported
    )
    with pytest.raises(ValueError, match=err_msg):
        pca_fitted.fit(data)

    # Additional case for arpack
    if svd_solver == "arpack":
        n_components = smallest_d

        err_msg = (
            "n_components={}L? must be strictly less than "
            r"min\(n_samples, n_features\)={}L? with "
            "svd_solver='arpack'".format(n_components, smallest_d)
        )
        with pytest.raises(ValueError, match=err_msg):
            PCA(n_components, svd_solver=svd_solver).fit(data)


@pytest.mark.parametrize(
    "solver, n_components_",
    [
        ("full", min(iris.data.shape)),
        ("arpack", min(iris.data.shape) - 1),
        ("randomized", min(iris.data.shape)),
    ],
)
@pytest.mark.parametrize("data", [iris.data, iris.data.T])
def test_n_components_none(data, solver, n_components_):
    pca = PCA(svd_solver=solver)
    pca.fit(data)
    assert pca.n_components_ == n_components_


@pytest.mark.parametrize("svd_solver", ["auto", "full"])
def test_n_components_mle(svd_solver):
    # Ensure that n_components == 'mle' doesn't raise error for auto/full
    rng = np.random.RandomState(0)
    n_samples, n_features = 600, 10
    X = rng.randn(n_samples, n_features)
    pca = PCA(n_components="mle", svd_solver=svd_solver)
    pca.fit(X)
    assert pca.n_components_ == 1


@pytest.mark.parametrize("svd_solver", ["arpack", "randomized"])
def test_n_components_mle_error(svd_solver):
    # Ensure that n_components == 'mle' will raise an error for unsupported
    # solvers
    rng = np.random.RandomState(0)
    n_samples, n_features = 600, 10
    X = rng.randn(n_samples, n_features)
    pca = PCA(n_components="mle", svd_solver=svd_solver)
    err_msg = "n_components='mle' cannot be a string with svd_solver='{}'".format(
        svd_solver
    )
    with pytest.raises(ValueError, match=err_msg):
        pca.fit(X)


def test_pca_dim():
    # Check automated dimensionality setting
    rng = np.random.RandomState(0)
    n, p = 100, 5
    X = rng.randn(n, p) * 0.1
    X[:10] += np.array([3, 4, 5, 1, 2])
    pca = PCA(n_components="mle", svd_solver="full").fit(X)
    assert pca.n_components == "mle"
    assert pca.n_components_ == 1


def test_infer_dim_1():
    # TODO: explain what this is testing
    # Or at least use explicit variable names...
    n, p = 1000, 5
    rng = np.random.RandomState(0)
    X = (
        rng.randn(n, p) * 0.1
        + rng.randn(n, 1) * np.array([3, 4, 5, 1, 2])
        + np.array([1, 0, 7, 4, 6])
    )
    pca = PCA(n_components=p, svd_solver="full")
    pca.fit(X)
    spect = pca.explained_variance_
    ll = np.array([_assess_dimension(spect, k, n) for k in range(1, p)])
    assert ll[1] > ll.max() - 0.01 * n


def test_infer_dim_2():
    # TODO: explain what this is testing
    # Or at least use explicit variable names...
    n, p = 1000, 5
    rng = np.random.RandomState(0)
    X = rng.randn(n, p) * 0.1
    X[:10] += np.array([3, 4, 5, 1, 2])
    X[10:20] += np.array([6, 0, 7, 2, -1])
    pca = PCA(n_components=p, svd_solver="full")
    pca.fit(X)
    spect = pca.explained_variance_
    assert _infer_dimension(spect, n) > 1


def test_infer_dim_3():
    n, p = 100, 5
    rng = np.random.RandomState(0)
    X = rng.randn(n, p) * 0.1
    X[:10] += np.array([3, 4, 5, 1, 2])
    X[10:20] += np.array([6, 0, 7, 2, -1])
    X[30:40] += 2 * np.array([-1, 1, -1, 1, -1])
    pca = PCA(n_components=p, svd_solver="full")
    pca.fit(X)
    spect = pca.explained_variance_
    assert _infer_dimension(spect, n) > 2


@pytest.mark.parametrize(
    "X, n_components, n_components_validated",
    [
        (iris.data, 0.95, 2),  # row > col
        (iris.data, 0.01, 1),  # row > col
        (np.random.RandomState(0).rand(5, 20), 0.5, 2),
    ],  # row < col
)
def test_infer_dim_by_explained_variance(X, n_components, n_components_validated):
    pca = PCA(n_components=n_components, svd_solver="full")
    pca.fit(X)
    assert pca.n_components == pytest.approx(n_components)
    assert pca.n_components_ == n_components_validated


@pytest.mark.parametrize("svd_solver", PCA_SOLVERS)
def test_pca_score(svd_solver):
    # Test that probabilistic PCA scoring yields a reasonable score
    n, p = 1000, 3
    rng = np.random.RandomState(0)
    X = rng.randn(n, p) * 0.1 + np.array([3, 4, 5])
    pca = PCA(n_components=2, svd_solver=svd_solver)
    pca.fit(X)

    ll1 = pca.score(X)
    h = -0.5 * np.log(2 * np.pi * np.exp(1) * 0.1 ** 2) * p
    assert_allclose(ll1 / h, 1, rtol=5e-2)

    ll2 = pca.score(rng.randn(n, p) * 0.2 + np.array([3, 4, 5]))
    assert ll1 > ll2

    pca = PCA(n_components=2, whiten=True, svd_solver=svd_solver)
    pca.fit(X)
    ll2 = pca.score(X)
    assert ll1 > ll2


def test_pca_score3():
    # Check that probabilistic PCA selects the right model
    n, p = 200, 3
    rng = np.random.RandomState(0)
    Xl = rng.randn(n, p) + rng.randn(n, 1) * np.array([3, 4, 5]) + np.array([1, 0, 7])
    Xt = rng.randn(n, p) + rng.randn(n, 1) * np.array([3, 4, 5]) + np.array([1, 0, 7])
    ll = np.zeros(p)
    for k in range(p):
        pca = PCA(n_components=k, svd_solver="full")
        pca.fit(Xl)
        ll[k] = pca.score(Xt)

    assert ll.argmax() == 1


@pytest.mark.parametrize("svd_solver", PCA_SOLVERS)
def test_pca_sanity_noise_variance(svd_solver):
    # Sanity check for the noise_variance_. For more details see
    # https://github.com/scikit-learn/scikit-learn/issues/7568
    # https://github.com/scikit-learn/scikit-learn/issues/8541
    # https://github.com/scikit-learn/scikit-learn/issues/8544
    X, _ = datasets.load_digits(return_X_y=True)
    pca = PCA(n_components=30, svd_solver=svd_solver, random_state=0)
    pca.fit(X)
    assert np.all((pca.explained_variance_ - pca.noise_variance_) >= 0)


@pytest.mark.parametrize("svd_solver", ["arpack", "randomized"])
def test_pca_score_consistency_solvers(svd_solver):
    # Check the consistency of score between solvers
    X, _ = datasets.load_digits(return_X_y=True)
    pca_full = PCA(n_components=30, svd_solver="full", random_state=0)
    pca_other = PCA(n_components=30, svd_solver=svd_solver, random_state=0)
    pca_full.fit(X)
    pca_other.fit(X)
    assert_allclose(pca_full.score(X), pca_other.score(X), rtol=5e-6)


# arpack raises ValueError for n_components == min(n_samples,  n_features)
@pytest.mark.parametrize("svd_solver", ["full", "randomized"])
def test_pca_zero_noise_variance_edge_cases(svd_solver):
    # ensure that noise_variance_ is 0 in edge cases
    # when n_components == min(n_samples, n_features)
    n, p = 100, 3
    rng = np.random.RandomState(0)
    X = rng.randn(n, p) * 0.1 + np.array([3, 4, 5])

    pca = PCA(n_components=p, svd_solver=svd_solver)
    pca.fit(X)
    assert pca.noise_variance_ == 0

    pca.fit(X.T)
    assert pca.noise_variance_ == 0


@pytest.mark.parametrize(
    "data, n_components, expected_solver",
    [  # case: n_components in (0,1) => 'full'
        (np.random.RandomState(0).uniform(size=(1000, 50)), 0.5, "full"),
        # case: max(X.shape) <= 500 => 'full'
        (np.random.RandomState(0).uniform(size=(10, 50)), 5, "full"),
        # case: n_components >= .8 * min(X.shape) => 'full'
        (np.random.RandomState(0).uniform(size=(1000, 50)), 50, "full"),
        # n_components >= 1 and n_components < .8*min(X.shape) => 'randomized'
        (np.random.RandomState(0).uniform(size=(1000, 50)), 10, "randomized"),
    ],
)
def test_pca_svd_solver_auto(data, n_components, expected_solver):
    pca_auto = PCA(n_components=n_components, random_state=0)
    pca_test = PCA(
        n_components=n_components, svd_solver=expected_solver, random_state=0
    )
    pca_auto.fit(data)
    pca_test.fit(data)
    assert_allclose(pca_auto.components_, pca_test.components_)


@pytest.mark.parametrize("svd_solver", PCA_SOLVERS)
def test_pca_sparse_input(svd_solver):
    X = np.random.RandomState(0).rand(5, 4)
    X = sp.sparse.csr_matrix(X)
    assert sp.sparse.issparse(X)

    pca = PCA(n_components=3, svd_solver=svd_solver)
    with pytest.raises(TypeError):
        pca.fit(X)


def test_pca_bad_solver():
    X = np.random.RandomState(0).rand(5, 4)
    pca = PCA(n_components=3, svd_solver="bad_argument")
    with pytest.raises(ValueError):
        pca.fit(X)


@pytest.mark.parametrize("svd_solver", PCA_SOLVERS)
def test_pca_deterministic_output(svd_solver):
    rng = np.random.RandomState(0)
    X = rng.rand(10, 10)

    transformed_X = np.zeros((20, 2))
    for i in range(20):
        pca = PCA(n_components=2, svd_solver=svd_solver, random_state=rng)
        transformed_X[i, :] = pca.fit_transform(X)[0]
    assert_allclose(transformed_X, np.tile(transformed_X[0, :], 20).reshape(20, 2))


@pytest.mark.parametrize("svd_solver", PCA_SOLVERS)
def test_pca_dtype_preservation(svd_solver):
    check_pca_float_dtype_preservation(svd_solver)
    check_pca_int_dtype_upcast_to_double(svd_solver)


def check_pca_float_dtype_preservation(svd_solver):
    # Ensure that PCA does not upscale the dtype when input is float32
    X_64 = np.random.RandomState(0).rand(1000, 4).astype(np.float64, copy=False)
    X_32 = X_64.astype(np.float32)

    pca_64 = PCA(n_components=3, svd_solver=svd_solver, random_state=0).fit(X_64)
    pca_32 = PCA(n_components=3, svd_solver=svd_solver, random_state=0).fit(X_32)

    assert pca_64.components_.dtype == np.float64
    assert pca_32.components_.dtype == np.float32
    assert pca_64.transform(X_64).dtype == np.float64
    assert pca_32.transform(X_32).dtype == np.float32

    # the rtol is set such that the test passes on all platforms tested on
    # conda-forge: PR#15775
    # see: https://github.com/conda-forge/scikit-learn-feedstock/pull/113
    assert_allclose(pca_64.components_, pca_32.components_, rtol=2e-4)


def check_pca_int_dtype_upcast_to_double(svd_solver):
    # Ensure that all int types will be upcast to float64
    X_i64 = np.random.RandomState(0).randint(0, 1000, (1000, 4))
    X_i64 = X_i64.astype(np.int64, copy=False)
    X_i32 = X_i64.astype(np.int32, copy=False)

    pca_64 = PCA(n_components=3, svd_solver=svd_solver, random_state=0).fit(X_i64)
    pca_32 = PCA(n_components=3, svd_solver=svd_solver, random_state=0).fit(X_i32)

    assert pca_64.components_.dtype == np.float64
    assert pca_32.components_.dtype == np.float64
    assert pca_64.transform(X_i64).dtype == np.float64
    assert pca_32.transform(X_i32).dtype == np.float64

    assert_allclose(pca_64.components_, pca_32.components_, rtol=1e-4)


def test_pca_n_components_mostly_explained_variance_ratio():
    # when n_components is the second highest cumulative sum of the
    # explained_variance_ratio_, then n_components_ should equal the
    # number of features in the dataset #15669
    X, y = load_iris(return_X_y=True)
    pca1 = PCA().fit(X, y)

    n_components = pca1.explained_variance_ratio_.cumsum()[-2]
    pca2 = PCA(n_components=n_components).fit(X, y)
    assert pca2.n_components_ == X.shape[1]


def test_assess_dimension_bad_rank():
    # Test error when tested rank not in [1, n_features - 1]
    spectrum = np.array([1, 1e-30, 1e-30, 1e-30])
    n_samples = 10
    for rank in (0, 5):
        with pytest.raises(ValueError, match=r"should be in \[1, n_features - 1\]"):
            _assess_dimension(spectrum, rank, n_samples)


def test_small_eigenvalues_mle():
    # Test rank associated with tiny eigenvalues are given a log-likelihood of
    # -inf. The inferred rank will be 1
    spectrum = np.array([1, 1e-30, 1e-30, 1e-30])

    assert _assess_dimension(spectrum, rank=1, n_samples=10) > -np.inf

    for rank in (2, 3):
        assert _assess_dimension(spectrum, rank, 10) == -np.inf

    assert _infer_dimension(spectrum, 10) == 1


def test_mle_redundant_data():
    # Test 'mle' with pathological X: only one relevant feature should give a
    # rank of 1
    X, _ = datasets.make_classification(
        n_features=20,
        n_informative=1,
        n_repeated=18,
        n_redundant=1,
        n_clusters_per_class=1,
        random_state=42,
    )
    pca = PCA(n_components="mle").fit(X)
    assert pca.n_components_ == 1


def test_fit_mle_too_few_samples():
    # Tests that an error is raised when the number of samples is smaller
    # than the number of features during an mle fit
    X, _ = datasets.make_classification(n_samples=20, n_features=21, random_state=42)

    pca = PCA(n_components="mle", svd_solver="full")
    with pytest.raises(
        ValueError,
        match="n_components='mle' is only supported if n_samples >= n_features",
    ):
        pca.fit(X)


def test_mle_simple_case():
    # non-regression test for issue
    # https://github.com/scikit-learn/scikit-learn/issues/16730
    n_samples, n_dim = 1000, 10
    X = np.random.RandomState(0).randn(n_samples, n_dim)
    X[:, -1] = np.mean(X[:, :-1], axis=-1)  # true X dim is ndim - 1
    pca_skl = PCA("mle", svd_solver="full")
    pca_skl.fit(X)
    assert pca_skl.n_components_ == n_dim - 1


def test_assess_dimesion_rank_one():
    # Make sure assess_dimension works properly on a matrix of rank 1
    n_samples, n_features = 9, 6
    X = np.ones((n_samples, n_features))  # rank 1 matrix
    _, s, _ = np.linalg.svd(X, full_matrices=True)
    # except for rank 1, all eigenvalues are 0 resp. close to 0 (FP)
    assert_allclose(s[1:], np.zeros(n_features - 1), atol=1e-12)

    assert np.isfinite(_assess_dimension(s, rank=1, n_samples=n_samples))
    for rank in range(2, n_features):
        assert _assess_dimension(s, rank, n_samples) == -np.inf
