import numpy as np
import scipy.sparse as sp
import pytest

from sklearn.utils._testing import (
    assert_array_almost_equal,
    assert_array_equal,
    assert_allclose,
)

from sklearn.decomposition import PCA, KernelPCA
from sklearn.datasets import make_circles
from sklearn.datasets import make_blobs
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import Perceptron
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.utils.validation import _check_psd_eigenvalues


def test_kernel_pca():
    """Nominal test for all solvers and all known kernels + a custom one

    It tests
     - that fit_transform is equivalent to fit+transform
     - that the shapes of transforms and inverse transforms are correct
    """
    rng = np.random.RandomState(0)
    X_fit = rng.random_sample((5, 4))
    X_pred = rng.random_sample((2, 4))

    def histogram(x, y, **kwargs):
        # Histogram kernel implemented as a callable.
        assert kwargs == {}  # no kernel_params that we didn't ask for
        return np.minimum(x, y).sum()

    for eigen_solver in ("auto", "dense", "arpack", "randomized"):
        for kernel in ("linear", "rbf", "poly", histogram):
            # histogram kernel produces singular matrix inside linalg.solve
            # XXX use a least-squares approximation?
            inv = not callable(kernel)

            # transform fit data
            kpca = KernelPCA(
                4, kernel=kernel, eigen_solver=eigen_solver, fit_inverse_transform=inv
            )
            X_fit_transformed = kpca.fit_transform(X_fit)
            X_fit_transformed2 = kpca.fit(X_fit).transform(X_fit)
            assert_array_almost_equal(
                np.abs(X_fit_transformed), np.abs(X_fit_transformed2)
            )

            # non-regression test: previously, gamma would be 0 by default,
            # forcing all eigenvalues to 0 under the poly kernel
            assert X_fit_transformed.size != 0

            # transform new data
            X_pred_transformed = kpca.transform(X_pred)
            assert X_pred_transformed.shape[1] == X_fit_transformed.shape[1]

            # inverse transform
            if inv:
                X_pred2 = kpca.inverse_transform(X_pred_transformed)
                assert X_pred2.shape == X_pred.shape


def test_kernel_pca_invalid_solver():
    """Check that kPCA raises an error if the solver parameter is invalid"""
    with pytest.raises(ValueError):
        KernelPCA(eigen_solver="unknown").fit(np.random.randn(10, 10))


def test_kernel_pca_invalid_parameters():
    """Check that kPCA raises an error if the parameters are invalid

    Tests fitting inverse transform with a precomputed kernel raises a
    ValueError.
    """
    with pytest.raises(ValueError):
        KernelPCA(10, fit_inverse_transform=True, kernel="precomputed")


def test_kernel_pca_consistent_transform():
    """Check robustness to mutations in the original training array

    Test that after fitting a kPCA model, it stays independent of any
    mutation of the values of the original data object by relying on an
    internal copy.
    """
    # X_fit_ needs to retain the old, unmodified copy of X
    state = np.random.RandomState(0)
    X = state.rand(10, 10)
    kpca = KernelPCA(random_state=state).fit(X)
    transformed1 = kpca.transform(X)

    X_copy = X.copy()
    X[:, 0] = 666
    transformed2 = kpca.transform(X_copy)
    assert_array_almost_equal(transformed1, transformed2)


def test_kernel_pca_deterministic_output():
    """Test that Kernel PCA produces deterministic output

    Tests that the same inputs and random state produce the same output.
    """
    rng = np.random.RandomState(0)
    X = rng.rand(10, 10)
    eigen_solver = ("arpack", "dense")

    for solver in eigen_solver:
        transformed_X = np.zeros((20, 2))
        for i in range(20):
            kpca = KernelPCA(n_components=2, eigen_solver=solver, random_state=rng)
            transformed_X[i, :] = kpca.fit_transform(X)[0]
        assert_allclose(transformed_X, np.tile(transformed_X[0, :], 20).reshape(20, 2))


def test_kernel_pca_sparse():
    """Test that kPCA works on a sparse data input.

    Same test as ``test_kernel_pca except inverse_transform`` since it's not
    implemented for sparse matrices.
    """
    rng = np.random.RandomState(0)
    X_fit = sp.csr_matrix(rng.random_sample((5, 4)))
    X_pred = sp.csr_matrix(rng.random_sample((2, 4)))

    for eigen_solver in ("auto", "arpack", "randomized"):
        for kernel in ("linear", "rbf", "poly"):
            # transform fit data
            kpca = KernelPCA(
                4,
                kernel=kernel,
                eigen_solver=eigen_solver,
                fit_inverse_transform=False,
                random_state=0,
            )
            X_fit_transformed = kpca.fit_transform(X_fit)
            X_fit_transformed2 = kpca.fit(X_fit).transform(X_fit)
            assert_array_almost_equal(
                np.abs(X_fit_transformed), np.abs(X_fit_transformed2)
            )

            # transform new data
            X_pred_transformed = kpca.transform(X_pred)
            assert X_pred_transformed.shape[1] == X_fit_transformed.shape[1]

            # inverse transform: not available for sparse matrices
            # XXX: should we raise another exception type here? For instance:
            # NotImplementedError.
            with pytest.raises(NotFittedError):
                kpca.inverse_transform(X_pred_transformed)


@pytest.mark.parametrize("solver", ["auto", "dense", "arpack", "randomized"])
@pytest.mark.parametrize("n_features", [4, 10])
def test_kernel_pca_linear_kernel(solver, n_features):
    """Test that kPCA with linear kernel is equivalent to PCA for all solvers.

    KernelPCA with linear kernel should produce the same output as PCA.
    """
    rng = np.random.RandomState(0)
    X_fit = rng.random_sample((5, n_features))
    X_pred = rng.random_sample((2, n_features))

    # for a linear kernel, kernel PCA should find the same projection as PCA
    # modulo the sign (direction)
    # fit only the first four components: fifth is near zero eigenvalue, so
    # can be trimmed due to roundoff error
    n_comps = 3 if solver == "arpack" else 4
    assert_array_almost_equal(
        np.abs(KernelPCA(n_comps, eigen_solver=solver).fit(X_fit).transform(X_pred)),
        np.abs(
            PCA(n_comps, svd_solver=solver if solver != "dense" else "full")
            .fit(X_fit)
            .transform(X_pred)
        ),
    )


def test_kernel_pca_n_components():
    """Test that `n_components` is correctly taken into account for projections

    For all solvers this tests that the output has the correct shape depending
    on the selected number of components.
    """
    rng = np.random.RandomState(0)
    X_fit = rng.random_sample((5, 4))
    X_pred = rng.random_sample((2, 4))

    for eigen_solver in ("dense", "arpack", "randomized"):
        for c in [1, 2, 4]:
            kpca = KernelPCA(n_components=c, eigen_solver=eigen_solver)
            shape = kpca.fit(X_fit).transform(X_pred).shape

            assert shape == (2, c)


def test_remove_zero_eig():
    """Check that the ``remove_zero_eig`` parameter works correctly.

    Tests that the null-space (Zero) eigenvalues are removed when
    remove_zero_eig=True, whereas they are not by default.
    """
    X = np.array([[1 - 1e-30, 1], [1, 1], [1, 1 - 1e-20]])

    # n_components=None (default) => remove_zero_eig is True
    kpca = KernelPCA()
    Xt = kpca.fit_transform(X)
    assert Xt.shape == (3, 0)

    kpca = KernelPCA(n_components=2)
    Xt = kpca.fit_transform(X)
    assert Xt.shape == (3, 2)

    kpca = KernelPCA(n_components=2, remove_zero_eig=True)
    Xt = kpca.fit_transform(X)
    assert Xt.shape == (3, 0)


def test_leave_zero_eig():
    """Non-regression test for issue #12141 (PR #12143)

    This test checks that fit().transform() returns the same result as
    fit_transform() in case of non-removed zero eigenvalue.
    """
    X_fit = np.array([[1, 1], [0, 0]])

    # Assert that even with all np warnings on, there is no div by zero warning
    with pytest.warns(None) as record:
        with np.errstate(all="warn"):
            k = KernelPCA(n_components=2, remove_zero_eig=False, eigen_solver="dense")
            # Fit, then transform
            A = k.fit(X_fit).transform(X_fit)
            # Do both at once
            B = k.fit_transform(X_fit)
            # Compare
            assert_array_almost_equal(np.abs(A), np.abs(B))

    for w in record:
        # There might be warnings about the kernel being badly conditioned,
        # but there should not be warnings about division by zero.
        # (Numpy division by zero warning can have many message variants, but
        # at least we know that it is a RuntimeWarning so lets check only this)
        assert not issubclass(w.category, RuntimeWarning)


def test_kernel_pca_precomputed():
    """Test that kPCA works with a precomputed kernel, for all solvers"""
    rng = np.random.RandomState(0)
    X_fit = rng.random_sample((5, 4))
    X_pred = rng.random_sample((2, 4))

    for eigen_solver in ("dense", "arpack", "randomized"):
        X_kpca = (
            KernelPCA(4, eigen_solver=eigen_solver, random_state=0)
            .fit(X_fit)
            .transform(X_pred)
        )

        X_kpca2 = (
            KernelPCA(
                4, eigen_solver=eigen_solver, kernel="precomputed", random_state=0
            )
            .fit(np.dot(X_fit, X_fit.T))
            .transform(np.dot(X_pred, X_fit.T))
        )

        X_kpca_train = KernelPCA(
            4, eigen_solver=eigen_solver, kernel="precomputed", random_state=0
        ).fit_transform(np.dot(X_fit, X_fit.T))

        X_kpca_train2 = (
            KernelPCA(
                4, eigen_solver=eigen_solver, kernel="precomputed", random_state=0
            )
            .fit(np.dot(X_fit, X_fit.T))
            .transform(np.dot(X_fit, X_fit.T))
        )

        assert_array_almost_equal(np.abs(X_kpca), np.abs(X_kpca2))

        assert_array_almost_equal(np.abs(X_kpca_train), np.abs(X_kpca_train2))


@pytest.mark.parametrize("solver", ["auto", "dense", "arpack", "randomized"])
def test_kernel_pca_precomputed_non_symmetric(solver):
    """Check that the kernel centerer works.

    Tests that a non symmetric precomputed kernel is actually accepted
    because the kernel centerer does its job correctly.
    """

    # a non symmetric gram matrix
    K = [[1, 2], [3, 40]]
    kpca = KernelPCA(
        kernel="precomputed", eigen_solver=solver, n_components=1, random_state=0
    )
    kpca.fit(K)  # no error

    # same test with centered kernel
    Kc = [[9, -9], [-9, 9]]
    kpca_c = KernelPCA(
        kernel="precomputed", eigen_solver=solver, n_components=1, random_state=0
    )
    kpca_c.fit(Kc)

    # comparison between the non-centered and centered versions
    assert_array_equal(kpca.eigenvectors_, kpca_c.eigenvectors_)
    assert_array_equal(kpca.eigenvalues_, kpca_c.eigenvalues_)


def test_kernel_pca_invalid_kernel():
    """Tests that using an invalid kernel name raises a ValueError

    An invalid kernel name should raise a ValueError at fit time.
    """
    rng = np.random.RandomState(0)
    X_fit = rng.random_sample((2, 4))
    kpca = KernelPCA(kernel="tototiti")
    with pytest.raises(ValueError):
        kpca.fit(X_fit)


def test_gridsearch_pipeline():
    """Check that kPCA works as expected in a grid search pipeline

    Test if we can do a grid-search to find parameters to separate
    circles with a perceptron model.
    """
    X, y = make_circles(n_samples=400, factor=0.3, noise=0.05, random_state=0)
    kpca = KernelPCA(kernel="rbf", n_components=2)
    pipeline = Pipeline([("kernel_pca", kpca), ("Perceptron", Perceptron(max_iter=5))])
    param_grid = dict(kernel_pca__gamma=2.0 ** np.arange(-2, 2))
    grid_search = GridSearchCV(pipeline, cv=3, param_grid=param_grid)
    grid_search.fit(X, y)
    assert grid_search.best_score_ == 1


def test_gridsearch_pipeline_precomputed():
    """Check that kPCA works as expected in a grid search pipeline (2)

    Test if we can do a grid-search to find parameters to separate
    circles with a perceptron model. This test uses a precomputed kernel.
    """
    X, y = make_circles(n_samples=400, factor=0.3, noise=0.05, random_state=0)
    kpca = KernelPCA(kernel="precomputed", n_components=2)
    pipeline = Pipeline([("kernel_pca", kpca), ("Perceptron", Perceptron(max_iter=5))])
    param_grid = dict(Perceptron__max_iter=np.arange(1, 5))
    grid_search = GridSearchCV(pipeline, cv=3, param_grid=param_grid)
    X_kernel = rbf_kernel(X, gamma=2.0)
    grid_search.fit(X_kernel, y)
    assert grid_search.best_score_ == 1


def test_nested_circles():
    """Check that kPCA projects in a space where nested circles are separable

    Tests that 2D nested circles become separable with a perceptron when
    projected in the first 2 kPCA using an RBF kernel, while raw samples
    are not directly separable in the original space.
    """
    X, y = make_circles(n_samples=400, factor=0.3, noise=0.05, random_state=0)

    # 2D nested circles are not linearly separable
    train_score = Perceptron(max_iter=5).fit(X, y).score(X, y)
    assert train_score < 0.8

    # Project the circles data into the first 2 components of a RBF Kernel
    # PCA model.
    # Note that the gamma value is data dependent. If this test breaks
    # and the gamma value has to be updated, the Kernel PCA example will
    # have to be updated too.
    kpca = KernelPCA(
        kernel="rbf", n_components=2, fit_inverse_transform=True, gamma=2.0
    )
    X_kpca = kpca.fit_transform(X)

    # The data is perfectly linearly separable in that space
    train_score = Perceptron(max_iter=5).fit(X_kpca, y).score(X_kpca, y)
    assert train_score == 1.0


def test_kernel_conditioning():
    """Check that ``_check_psd_eigenvalues`` is correctly called in kPCA

    Non-regression test for issue #12140 (PR #12145).
    """

    # create a pathological X leading to small non-zero eigenvalue
    X = [[5, 1], [5 + 1e-8, 1e-8], [5 + 1e-8, 0]]
    kpca = KernelPCA(kernel="linear", n_components=2, fit_inverse_transform=True)
    kpca.fit(X)

    # check that the small non-zero eigenvalue was correctly set to zero
    assert kpca.eigenvalues_.min() == 0
    assert np.all(kpca.eigenvalues_ == _check_psd_eigenvalues(kpca.eigenvalues_))


@pytest.mark.parametrize("solver", ["auto", "dense", "arpack", "randomized"])
def test_precomputed_kernel_not_psd(solver):
    """Check how KernelPCA works with non-PSD kernels depending on n_components

    Tests for all methods what happens with a non PSD gram matrix (this
    can happen in an isomap scenario, or with custom kernel functions, or
    maybe with ill-posed datasets).

    When ``n_component`` is large enough to capture a negative eigenvalue, an
    error should be raised. Otherwise, KernelPCA should run without error
    since the negative eigenvalues are not selected.
    """

    # a non PSD kernel with large eigenvalues, already centered
    # it was captured from an isomap call and multiplied by 100 for compacity
    K = [
        [4.48, -1.0, 8.07, 2.33, 2.33, 2.33, -5.76, -12.78],
        [-1.0, -6.48, 4.5, -1.24, -1.24, -1.24, -0.81, 7.49],
        [8.07, 4.5, 15.48, 2.09, 2.09, 2.09, -11.1, -23.23],
        [2.33, -1.24, 2.09, 4.0, -3.65, -3.65, 1.02, -0.9],
        [2.33, -1.24, 2.09, -3.65, 4.0, -3.65, 1.02, -0.9],
        [2.33, -1.24, 2.09, -3.65, -3.65, 4.0, 1.02, -0.9],
        [-5.76, -0.81, -11.1, 1.02, 1.02, 1.02, 4.86, 9.75],
        [-12.78, 7.49, -23.23, -0.9, -0.9, -0.9, 9.75, 21.46],
    ]
    # this gram matrix has 5 positive eigenvalues and 3 negative ones
    # [ 52.72,   7.65,   7.65,   5.02,   0.  ,  -0.  ,  -6.13, -15.11]

    # 1. ask for enough components to get a significant negative one
    kpca = KernelPCA(kernel="precomputed", eigen_solver=solver, n_components=7)
    # make sure that the appropriate error is raised
    with pytest.raises(ValueError, match="There are significant negative eigenvalues"):
        kpca.fit(K)

    # 2. ask for a small enough n_components to get only positive ones
    kpca = KernelPCA(kernel="precomputed", eigen_solver=solver, n_components=2)
    if solver == "randomized":
        # the randomized method is still inconsistent with the others on this
        # since it selects the eigenvalues based on the largest 2 modules, not
        # on the largest 2 values.
        #
        # At least we can ensure that we return an error instead of returning
        # the wrong eigenvalues
        with pytest.raises(
            ValueError, match="There are significant negative eigenvalues"
        ):
            kpca.fit(K)
    else:
        # general case: make sure that it works
        kpca.fit(K)


@pytest.mark.parametrize("n_components", [4, 10, 20])
def test_kernel_pca_solvers_equivalence(n_components):
    """Check that 'dense' 'arpack' & 'randomized' solvers give similar results"""

    # Generate random data
    n_train, n_test = 1_000, 100
    X, _ = make_circles(
        n_samples=(n_train + n_test), factor=0.3, noise=0.05, random_state=0
    )
    X_fit, X_pred = X[:n_train, :], X[n_train:, :]

    # reference (full)
    ref_pred = (
        KernelPCA(n_components, eigen_solver="dense", random_state=0)
        .fit(X_fit)
        .transform(X_pred)
    )

    # arpack
    a_pred = (
        KernelPCA(n_components, eigen_solver="arpack", random_state=0)
        .fit(X_fit)
        .transform(X_pred)
    )
    # check that the result is still correct despite the approx
    assert_array_almost_equal(np.abs(a_pred), np.abs(ref_pred))

    # randomized
    r_pred = (
        KernelPCA(n_components, eigen_solver="randomized", random_state=0)
        .fit(X_fit)
        .transform(X_pred)
    )
    # check that the result is still correct despite the approximation
    assert_array_almost_equal(np.abs(r_pred), np.abs(ref_pred))


def test_kernel_pca_inverse_transform_reconstruction():
    """Test if the reconstruction is a good approximation.

    Note that in general it is not possible to get an arbitrarily good
    reconstruction because of kernel centering that does not
    preserve all the information of the original data.
    """
    X, *_ = make_blobs(n_samples=100, n_features=4, random_state=0)

    kpca = KernelPCA(
        n_components=20, kernel="rbf", fit_inverse_transform=True, alpha=1e-3
    )
    X_trans = kpca.fit_transform(X)
    X_reconst = kpca.inverse_transform(X_trans)
    assert np.linalg.norm(X - X_reconst) / np.linalg.norm(X) < 1e-1


def test_kernel_pca_raise_not_fitted_error():
    X = np.random.randn(15).reshape(5, 3)
    kpca = KernelPCA()
    kpca.fit(X)
    with pytest.raises(NotFittedError):
        kpca.inverse_transform(X)


def test_32_64_decomposition_shape():
    """Test that the decomposition is similar for 32 and 64 bits data

    Non regression test for
    https://github.com/scikit-learn/scikit-learn/issues/18146
    """
    X, y = make_blobs(
        n_samples=30, centers=[[0, 0, 0], [1, 1, 1]], random_state=0, cluster_std=0.1
    )
    X = StandardScaler().fit_transform(X)
    X -= X.min()

    # Compare the shapes (corresponds to the number of non-zero eigenvalues)
    kpca = KernelPCA()
    assert kpca.fit_transform(X).shape == kpca.fit_transform(X.astype(np.float32)).shape


# TODO: Remove in 1.1
def test_kernel_pcc_pairwise_is_deprecated():
    """Check that `_pairwise` is correctly marked with deprecation warning

    Tests that a `FutureWarning` is issued when `_pairwise` is accessed.
    """
    kp = KernelPCA(kernel="precomputed")
    msg = r"Attribute `_pairwise` was deprecated in version 0\.24"
    with pytest.warns(FutureWarning, match=msg):
        kp._pairwise


# TODO: Remove in 1.2
def test_kernel_pca_lambdas_deprecated():
    kp = KernelPCA()
    kp.eigenvalues_ = None
    msg = r"Attribute `lambdas_` was deprecated in version 1\.0"
    with pytest.warns(FutureWarning, match=msg):
        kp.lambdas_


# TODO: Remove in 1.2
def test_kernel_pca_alphas_deprecated():
    kp = KernelPCA(kernel="precomputed")
    kp.eigenvectors_ = None
    msg = r"Attribute `alphas_` was deprecated in version 1\.0"
    with pytest.warns(FutureWarning, match=msg):
        kp.alphas_
