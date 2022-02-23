"""Test truncated SVD transformer."""

import numpy as np
import scipy.sparse as sp

import pytest

from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.utils import check_random_state
from sklearn.utils._testing import assert_array_less, assert_allclose

SVD_SOLVERS = ["arpack", "randomized"]


@pytest.fixture(scope="module")
def X_sparse():
    # Make an X that looks somewhat like a small tf-idf matrix.
    rng = check_random_state(42)
    X = sp.random(60, 55, density=0.2, format="csr", random_state=rng)
    X.data[:] = 1 + np.log(X.data)
    return X


@pytest.mark.parametrize("solver", ["randomized"])
@pytest.mark.parametrize("kind", ("dense", "sparse"))
def test_solvers(X_sparse, solver, kind):
    X = X_sparse if kind == "sparse" else X_sparse.toarray()
    svd_a = TruncatedSVD(30, algorithm="arpack")
    svd = TruncatedSVD(30, algorithm=solver, random_state=42)

    Xa = svd_a.fit_transform(X)[:, :6]
    Xr = svd.fit_transform(X)[:, :6]
    assert_allclose(Xa, Xr, rtol=2e-3)

    comp_a = np.abs(svd_a.components_)
    comp = np.abs(svd.components_)
    # All elements are equal, but some elements are more equal than others.
    assert_allclose(comp_a[:9], comp[:9], rtol=1e-3)
    assert_allclose(comp_a[9:], comp[9:], atol=1e-2)


@pytest.mark.parametrize("n_components", (10, 25, 41))
def test_attributes(n_components, X_sparse):
    n_features = X_sparse.shape[1]
    tsvd = TruncatedSVD(n_components).fit(X_sparse)
    assert tsvd.n_components == n_components
    assert tsvd.components_.shape == (n_components, n_features)


@pytest.mark.parametrize("algorithm", SVD_SOLVERS)
def test_too_many_components(algorithm, X_sparse):
    n_features = X_sparse.shape[1]
    for n_components in (n_features, n_features + 1):
        tsvd = TruncatedSVD(n_components=n_components, algorithm=algorithm)
        with pytest.raises(ValueError):
            tsvd.fit(X_sparse)


@pytest.mark.parametrize("fmt", ("array", "csr", "csc", "coo", "lil"))
def test_sparse_formats(fmt, X_sparse):
    n_samples = X_sparse.shape[0]
    Xfmt = X_sparse.toarray() if fmt == "dense" else getattr(X_sparse, "to" + fmt)()
    tsvd = TruncatedSVD(n_components=11)
    Xtrans = tsvd.fit_transform(Xfmt)
    assert Xtrans.shape == (n_samples, 11)
    Xtrans = tsvd.transform(Xfmt)
    assert Xtrans.shape == (n_samples, 11)


@pytest.mark.parametrize("algo", SVD_SOLVERS)
def test_inverse_transform(algo, X_sparse):
    # We need a lot of components for the reconstruction to be "almost
    # equal" in all positions. XXX Test means or sums instead?
    tsvd = TruncatedSVD(n_components=52, random_state=42, algorithm=algo)
    Xt = tsvd.fit_transform(X_sparse)
    Xinv = tsvd.inverse_transform(Xt)
    assert_allclose(Xinv, X_sparse.toarray(), rtol=1e-1, atol=2e-1)


def test_integers(X_sparse):
    n_samples = X_sparse.shape[0]
    Xint = X_sparse.astype(np.int64)
    tsvd = TruncatedSVD(n_components=6)
    Xtrans = tsvd.fit_transform(Xint)
    assert Xtrans.shape == (n_samples, tsvd.n_components)


@pytest.mark.parametrize("kind", ("dense", "sparse"))
@pytest.mark.parametrize("n_components", [10, 20])
@pytest.mark.parametrize("solver", SVD_SOLVERS)
def test_explained_variance(X_sparse, kind, n_components, solver):
    X = X_sparse if kind == "sparse" else X_sparse.toarray()
    svd = TruncatedSVD(n_components, algorithm=solver)
    X_tr = svd.fit_transform(X)
    # Assert that all the values are greater than 0
    assert_array_less(0.0, svd.explained_variance_ratio_)

    # Assert that total explained variance is less than 1
    assert_array_less(svd.explained_variance_ratio_.sum(), 1.0)

    # Test that explained_variance is correct
    total_variance = np.var(X_sparse.toarray(), axis=0).sum()
    variances = np.var(X_tr, axis=0)
    true_explained_variance_ratio = variances / total_variance

    assert_allclose(
        svd.explained_variance_ratio_,
        true_explained_variance_ratio,
    )


@pytest.mark.parametrize("kind", ("dense", "sparse"))
@pytest.mark.parametrize("solver", SVD_SOLVERS)
def test_explained_variance_components_10_20(X_sparse, kind, solver):
    X = X_sparse if kind == "sparse" else X_sparse.toarray()
    svd_10 = TruncatedSVD(10, algorithm=solver, n_iter=10).fit(X)
    svd_20 = TruncatedSVD(20, algorithm=solver, n_iter=10).fit(X)

    # Assert the 1st component is equal
    assert_allclose(
        svd_10.explained_variance_ratio_,
        svd_20.explained_variance_ratio_[:10],
        rtol=5e-3,
    )

    # Assert that 20 components has higher explained variance than 10
    assert (
        svd_20.explained_variance_ratio_.sum() > svd_10.explained_variance_ratio_.sum()
    )


@pytest.mark.parametrize("solver", SVD_SOLVERS)
def test_singular_values_consistency(solver):
    # Check that the TruncatedSVD output has the correct singular values
    rng = np.random.RandomState(0)
    n_samples, n_features = 100, 80
    X = rng.randn(n_samples, n_features)

    pca = TruncatedSVD(n_components=2, algorithm=solver, random_state=rng).fit(X)

    # Compare to the Frobenius norm
    X_pca = pca.transform(X)
    assert_allclose(
        np.sum(pca.singular_values_ ** 2.0),
        np.linalg.norm(X_pca, "fro") ** 2.0,
        rtol=1e-2,
    )

    # Compare to the 2-norms of the score vectors
    assert_allclose(
        pca.singular_values_, np.sqrt(np.sum(X_pca ** 2.0, axis=0)), rtol=1e-2
    )


@pytest.mark.parametrize("solver", SVD_SOLVERS)
def test_singular_values_expected(solver):
    # Set the singular values and see what we get back
    rng = np.random.RandomState(0)
    n_samples = 100
    n_features = 110

    X = rng.randn(n_samples, n_features)

    pca = TruncatedSVD(n_components=3, algorithm=solver, random_state=rng)
    X_pca = pca.fit_transform(X)

    X_pca /= np.sqrt(np.sum(X_pca ** 2.0, axis=0))
    X_pca[:, 0] *= 3.142
    X_pca[:, 1] *= 2.718

    X_hat_pca = np.dot(X_pca, pca.components_)
    pca.fit(X_hat_pca)
    assert_allclose(pca.singular_values_, [3.142, 2.718, 1.0], rtol=1e-14)


def test_truncated_svd_eq_pca(X_sparse):
    # TruncatedSVD should be equal to PCA on centered data

    X_dense = X_sparse.toarray()

    X_c = X_dense - X_dense.mean(axis=0)

    params = dict(n_components=10, random_state=42)

    svd = TruncatedSVD(algorithm="arpack", **params)
    pca = PCA(svd_solver="arpack", **params)

    Xt_svd = svd.fit_transform(X_c)
    Xt_pca = pca.fit_transform(X_c)

    assert_allclose(Xt_svd, Xt_pca, rtol=1e-9)
    assert_allclose(pca.mean_, 0, atol=1e-9)
    assert_allclose(svd.components_, pca.components_)


@pytest.mark.parametrize(
    "algorithm, tol", [("randomized", 0.0), ("arpack", 1e-6), ("arpack", 0.0)]
)
@pytest.mark.parametrize("kind", ("dense", "sparse"))
def test_fit_transform(X_sparse, algorithm, tol, kind):
    # fit_transform(X) should equal fit(X).transform(X)
    X = X_sparse if kind == "sparse" else X_sparse.toarray()
    svd = TruncatedSVD(
        n_components=5, n_iter=7, random_state=42, algorithm=algorithm, tol=tol
    )
    X_transformed_1 = svd.fit_transform(X)
    X_transformed_2 = svd.fit(X).transform(X)
    assert_allclose(X_transformed_1, X_transformed_2)
