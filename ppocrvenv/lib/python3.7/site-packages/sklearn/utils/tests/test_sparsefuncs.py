import pytest
import numpy as np
import scipy.sparse as sp

from scipy import linalg
from numpy.testing import assert_array_almost_equal, assert_array_equal
from numpy.random import RandomState

from sklearn.datasets import make_classification
from sklearn.utils.sparsefuncs import (
    mean_variance_axis,
    incr_mean_variance_axis,
    inplace_column_scale,
    inplace_row_scale,
    inplace_swap_row,
    inplace_swap_column,
    min_max_axis,
    count_nonzero,
    csc_median_axis_0,
)
from sklearn.utils.sparsefuncs_fast import (
    assign_rows_csr,
    inplace_csr_row_normalize_l1,
    inplace_csr_row_normalize_l2,
    csr_row_norms,
)
from sklearn.utils._testing import assert_allclose


def test_mean_variance_axis0():
    X, _ = make_classification(5, 4, random_state=0)
    # Sparsify the array a little bit
    X[0, 0] = 0
    X[2, 1] = 0
    X[4, 3] = 0
    X_lil = sp.lil_matrix(X)
    X_lil[1, 0] = 0
    X[1, 0] = 0

    with pytest.raises(TypeError):
        mean_variance_axis(X_lil, axis=0)

    X_csr = sp.csr_matrix(X_lil)
    X_csc = sp.csc_matrix(X_lil)

    expected_dtypes = [
        (np.float32, np.float32),
        (np.float64, np.float64),
        (np.int32, np.float64),
        (np.int64, np.float64),
    ]

    for input_dtype, output_dtype in expected_dtypes:
        X_test = X.astype(input_dtype)
        for X_sparse in (X_csr, X_csc):
            X_sparse = X_sparse.astype(input_dtype)
            X_means, X_vars = mean_variance_axis(X_sparse, axis=0)
            assert X_means.dtype == output_dtype
            assert X_vars.dtype == output_dtype
            assert_array_almost_equal(X_means, np.mean(X_test, axis=0))
            assert_array_almost_equal(X_vars, np.var(X_test, axis=0))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("sparse_constructor", [sp.csr_matrix, sp.csc_matrix])
def test_mean_variance_axis0_precision(dtype, sparse_constructor):
    # Check that there's no big loss of precision when the real variance is
    # exactly 0. (#19766)
    rng = np.random.RandomState(0)
    X = np.full(fill_value=100.0, shape=(1000, 1), dtype=dtype)
    # Add some missing records which should be ignored:
    missing_indices = rng.choice(np.arange(X.shape[0]), 10, replace=False)
    X[missing_indices, 0] = np.nan
    X = sparse_constructor(X)

    # Random positive weights:
    sample_weight = rng.rand(X.shape[0]).astype(dtype)

    _, var = mean_variance_axis(X, weights=sample_weight, axis=0)

    assert var < np.finfo(dtype).eps


def test_mean_variance_axis1():
    X, _ = make_classification(5, 4, random_state=0)
    # Sparsify the array a little bit
    X[0, 0] = 0
    X[2, 1] = 0
    X[4, 3] = 0
    X_lil = sp.lil_matrix(X)
    X_lil[1, 0] = 0
    X[1, 0] = 0

    with pytest.raises(TypeError):
        mean_variance_axis(X_lil, axis=1)

    X_csr = sp.csr_matrix(X_lil)
    X_csc = sp.csc_matrix(X_lil)

    expected_dtypes = [
        (np.float32, np.float32),
        (np.float64, np.float64),
        (np.int32, np.float64),
        (np.int64, np.float64),
    ]

    for input_dtype, output_dtype in expected_dtypes:
        X_test = X.astype(input_dtype)
        for X_sparse in (X_csr, X_csc):
            X_sparse = X_sparse.astype(input_dtype)
            X_means, X_vars = mean_variance_axis(X_sparse, axis=0)
            assert X_means.dtype == output_dtype
            assert X_vars.dtype == output_dtype
            assert_array_almost_equal(X_means, np.mean(X_test, axis=0))
            assert_array_almost_equal(X_vars, np.var(X_test, axis=0))


@pytest.mark.parametrize(
    ["Xw", "X", "weights"],
    [
        ([[0, 0, 1], [0, 2, 3]], [[0, 0, 1], [0, 2, 3]], [1, 1, 1]),
        ([[0, 0, 1], [0, 1, 1]], [[0, 0, 0, 1], [0, 1, 1, 1]], [1, 2, 1]),
        ([[0, 0, 1], [0, 1, 1]], [[0, 0, 1], [0, 1, 1]], None),
        (
            [[0, np.nan, 2], [0, np.nan, np.nan]],
            [[0, np.nan, 2], [0, np.nan, np.nan]],
            [1.0, 1.0, 1.0],
        ),
        (
            [[0, 0], [1, np.nan], [2, 0], [0, 3], [np.nan, np.nan], [np.nan, 2]],
            [
                [0, 0, 0],
                [1, 1, np.nan],
                [2, 2, 0],
                [0, 0, 3],
                [np.nan, np.nan, np.nan],
                [np.nan, np.nan, 2],
            ],
            [2.0, 1.0],
        ),
        (
            [[1, 0, 1], [0, 3, 1]],
            [[1, 0, 0, 0, 1], [0, 3, 3, 3, 1]],
            np.array([1, 3, 1]),
        ),
    ],
)
@pytest.mark.parametrize("sparse_constructor", [sp.csc_matrix, sp.csr_matrix])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_incr_mean_variance_axis_weighted_axis1(
    Xw, X, weights, sparse_constructor, dtype
):
    axis = 1
    Xw_sparse = sparse_constructor(Xw).astype(dtype)
    X_sparse = sparse_constructor(X).astype(dtype)

    last_mean = np.zeros(np.shape(Xw)[0], dtype=dtype)
    last_var = np.zeros_like(last_mean, dtype=dtype)
    last_n = np.zeros_like(last_mean, dtype=np.int64)
    means0, vars0, n_incr0 = incr_mean_variance_axis(
        X=X_sparse,
        axis=axis,
        last_mean=last_mean,
        last_var=last_var,
        last_n=last_n,
        weights=None,
    )

    means_w0, vars_w0, n_incr_w0 = incr_mean_variance_axis(
        X=Xw_sparse,
        axis=axis,
        last_mean=last_mean,
        last_var=last_var,
        last_n=last_n,
        weights=weights,
    )

    assert means_w0.dtype == dtype
    assert vars_w0.dtype == dtype
    assert n_incr_w0.dtype == dtype

    means_simple, vars_simple = mean_variance_axis(X=X_sparse, axis=axis)

    assert_array_almost_equal(means0, means_w0)
    assert_array_almost_equal(means0, means_simple)
    assert_array_almost_equal(vars0, vars_w0)
    assert_array_almost_equal(vars0, vars_simple)
    assert_array_almost_equal(n_incr0, n_incr_w0)

    # check second round for incremental
    means1, vars1, n_incr1 = incr_mean_variance_axis(
        X=X_sparse,
        axis=axis,
        last_mean=means0,
        last_var=vars0,
        last_n=n_incr0,
        weights=None,
    )

    means_w1, vars_w1, n_incr_w1 = incr_mean_variance_axis(
        X=Xw_sparse,
        axis=axis,
        last_mean=means_w0,
        last_var=vars_w0,
        last_n=n_incr_w0,
        weights=weights,
    )

    assert_array_almost_equal(means1, means_w1)
    assert_array_almost_equal(vars1, vars_w1)
    assert_array_almost_equal(n_incr1, n_incr_w1)

    assert means_w1.dtype == dtype
    assert vars_w1.dtype == dtype
    assert n_incr_w1.dtype == dtype


@pytest.mark.parametrize(
    ["Xw", "X", "weights"],
    [
        ([[0, 0, 1], [0, 2, 3]], [[0, 0, 1], [0, 2, 3]], [1, 1]),
        ([[0, 0, 1], [0, 1, 1]], [[0, 0, 1], [0, 1, 1], [0, 1, 1]], [1, 2]),
        ([[0, 0, 1], [0, 1, 1]], [[0, 0, 1], [0, 1, 1]], None),
        (
            [[0, np.nan, 2], [0, np.nan, np.nan]],
            [[0, np.nan, 2], [0, np.nan, np.nan]],
            [1.0, 1.0],
        ),
        (
            [[0, 0, 1, np.nan, 2, 0], [0, 3, np.nan, np.nan, np.nan, 2]],
            [
                [0, 0, 1, np.nan, 2, 0],
                [0, 0, 1, np.nan, 2, 0],
                [0, 3, np.nan, np.nan, np.nan, 2],
            ],
            [2.0, 1.0],
        ),
        (
            [[1, 0, 1], [0, 0, 1]],
            [[1, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]],
            np.array([1, 3]),
        ),
    ],
)
@pytest.mark.parametrize("sparse_constructor", [sp.csc_matrix, sp.csr_matrix])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_incr_mean_variance_axis_weighted_axis0(
    Xw, X, weights, sparse_constructor, dtype
):
    axis = 0
    Xw_sparse = sparse_constructor(Xw).astype(dtype)
    X_sparse = sparse_constructor(X).astype(dtype)

    last_mean = np.zeros(np.size(Xw, 1), dtype=dtype)
    last_var = np.zeros_like(last_mean)
    last_n = np.zeros_like(last_mean, dtype=np.int64)
    means0, vars0, n_incr0 = incr_mean_variance_axis(
        X=X_sparse,
        axis=axis,
        last_mean=last_mean,
        last_var=last_var,
        last_n=last_n,
        weights=None,
    )

    means_w0, vars_w0, n_incr_w0 = incr_mean_variance_axis(
        X=Xw_sparse,
        axis=axis,
        last_mean=last_mean,
        last_var=last_var,
        last_n=last_n,
        weights=weights,
    )

    assert means_w0.dtype == dtype
    assert vars_w0.dtype == dtype
    assert n_incr_w0.dtype == dtype

    means_simple, vars_simple = mean_variance_axis(X=X_sparse, axis=axis)

    assert_array_almost_equal(means0, means_w0)
    assert_array_almost_equal(means0, means_simple)
    assert_array_almost_equal(vars0, vars_w0)
    assert_array_almost_equal(vars0, vars_simple)
    assert_array_almost_equal(n_incr0, n_incr_w0)

    # check second round for incremental
    means1, vars1, n_incr1 = incr_mean_variance_axis(
        X=X_sparse,
        axis=axis,
        last_mean=means0,
        last_var=vars0,
        last_n=n_incr0,
        weights=None,
    )

    means_w1, vars_w1, n_incr_w1 = incr_mean_variance_axis(
        X=Xw_sparse,
        axis=axis,
        last_mean=means_w0,
        last_var=vars_w0,
        last_n=n_incr_w0,
        weights=weights,
    )

    assert_array_almost_equal(means1, means_w1)
    assert_array_almost_equal(vars1, vars_w1)
    assert_array_almost_equal(n_incr1, n_incr_w1)

    assert means_w1.dtype == dtype
    assert vars_w1.dtype == dtype
    assert n_incr_w1.dtype == dtype


def test_incr_mean_variance_axis():
    for axis in [0, 1]:
        rng = np.random.RandomState(0)
        n_features = 50
        n_samples = 10
        if axis == 0:
            data_chunks = [rng.randint(0, 2, size=n_features) for i in range(n_samples)]
        else:
            data_chunks = [rng.randint(0, 2, size=n_samples) for i in range(n_features)]

        # default params for incr_mean_variance
        last_mean = np.zeros(n_features) if axis == 0 else np.zeros(n_samples)
        last_var = np.zeros_like(last_mean)
        last_n = np.zeros_like(last_mean, dtype=np.int64)

        # Test errors
        X = np.array(data_chunks[0])
        X = np.atleast_2d(X)
        X = X.T if axis == 1 else X
        X_lil = sp.lil_matrix(X)
        X_csr = sp.csr_matrix(X_lil)

        with pytest.raises(TypeError):
            incr_mean_variance_axis(
                X=axis, axis=last_mean, last_mean=last_var, last_var=last_n
            )
        with pytest.raises(TypeError):
            incr_mean_variance_axis(
                X_lil, axis=axis, last_mean=last_mean, last_var=last_var, last_n=last_n
            )

        # Test _incr_mean_and_var with a 1 row input
        X_means, X_vars = mean_variance_axis(X_csr, axis)
        X_means_incr, X_vars_incr, n_incr = incr_mean_variance_axis(
            X_csr, axis=axis, last_mean=last_mean, last_var=last_var, last_n=last_n
        )
        assert_array_almost_equal(X_means, X_means_incr)
        assert_array_almost_equal(X_vars, X_vars_incr)
        # X.shape[axis] picks # samples
        assert_array_equal(X.shape[axis], n_incr)

        X_csc = sp.csc_matrix(X_lil)
        X_means, X_vars = mean_variance_axis(X_csc, axis)
        assert_array_almost_equal(X_means, X_means_incr)
        assert_array_almost_equal(X_vars, X_vars_incr)
        assert_array_equal(X.shape[axis], n_incr)

        # Test _incremental_mean_and_var with whole data
        X = np.vstack(data_chunks)
        X = X.T if axis == 1 else X
        X_lil = sp.lil_matrix(X)
        X_csr = sp.csr_matrix(X_lil)
        X_csc = sp.csc_matrix(X_lil)

        expected_dtypes = [
            (np.float32, np.float32),
            (np.float64, np.float64),
            (np.int32, np.float64),
            (np.int64, np.float64),
        ]

        for input_dtype, output_dtype in expected_dtypes:
            for X_sparse in (X_csr, X_csc):
                X_sparse = X_sparse.astype(input_dtype)
                last_mean = last_mean.astype(output_dtype)
                last_var = last_var.astype(output_dtype)
                X_means, X_vars = mean_variance_axis(X_sparse, axis)
                X_means_incr, X_vars_incr, n_incr = incr_mean_variance_axis(
                    X_sparse,
                    axis=axis,
                    last_mean=last_mean,
                    last_var=last_var,
                    last_n=last_n,
                )
                assert X_means_incr.dtype == output_dtype
                assert X_vars_incr.dtype == output_dtype
                assert_array_almost_equal(X_means, X_means_incr)
                assert_array_almost_equal(X_vars, X_vars_incr)
                assert_array_equal(X.shape[axis], n_incr)


@pytest.mark.parametrize("sparse_constructor", [sp.csc_matrix, sp.csr_matrix])
def test_incr_mean_variance_axis_dim_mismatch(sparse_constructor):
    """Check that we raise proper error when axis=1 and the dimension mismatch.
    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/pull/18655
    """
    n_samples, n_features = 60, 4
    rng = np.random.RandomState(42)
    X = sparse_constructor(rng.rand(n_samples, n_features))

    last_mean = np.zeros(n_features)
    last_var = np.zeros_like(last_mean)
    last_n = np.zeros(last_mean.shape, dtype=np.int64)

    kwargs = dict(last_mean=last_mean, last_var=last_var, last_n=last_n)
    mean0, var0, _ = incr_mean_variance_axis(X, axis=0, **kwargs)
    assert_allclose(np.mean(X.toarray(), axis=0), mean0)
    assert_allclose(np.var(X.toarray(), axis=0), var0)

    # test ValueError if axis=1 and last_mean.size == n_features
    with pytest.raises(ValueError):
        incr_mean_variance_axis(X, axis=1, **kwargs)

    # test inconsistent shapes of last_mean, last_var, last_n
    kwargs = dict(last_mean=last_mean[:-1], last_var=last_var, last_n=last_n)
    with pytest.raises(ValueError):
        incr_mean_variance_axis(X, axis=0, **kwargs)


@pytest.mark.parametrize(
    "X1, X2",
    [
        (
            sp.random(5, 2, density=0.8, format="csr", random_state=0),
            sp.random(13, 2, density=0.8, format="csr", random_state=0),
        ),
        (
            sp.random(5, 2, density=0.8, format="csr", random_state=0),
            sp.hstack(
                [
                    sp.csr_matrix(np.full((13, 1), fill_value=np.nan)),
                    sp.random(13, 1, density=0.8, random_state=42),
                ],
                format="csr",
            ),
        ),
    ],
)
def test_incr_mean_variance_axis_equivalence_mean_variance(X1, X2):
    # non-regression test for:
    # https://github.com/scikit-learn/scikit-learn/issues/16448
    # check that computing the incremental mean and variance is equivalent to
    # computing the mean and variance on the stacked dataset.
    axis = 0
    last_mean, last_var = np.zeros(X1.shape[1]), np.zeros(X1.shape[1])
    last_n = np.zeros(X1.shape[1], dtype=np.int64)
    updated_mean, updated_var, updated_n = incr_mean_variance_axis(
        X1, axis=axis, last_mean=last_mean, last_var=last_var, last_n=last_n
    )
    updated_mean, updated_var, updated_n = incr_mean_variance_axis(
        X2, axis=axis, last_mean=updated_mean, last_var=updated_var, last_n=updated_n
    )
    X = sp.vstack([X1, X2])
    assert_allclose(updated_mean, np.nanmean(X.A, axis=axis))
    assert_allclose(updated_var, np.nanvar(X.A, axis=axis))
    assert_allclose(updated_n, np.count_nonzero(~np.isnan(X.A), axis=0))


def test_incr_mean_variance_no_new_n():
    # check the behaviour when we update the variance with an empty matrix
    axis = 0
    X1 = sp.random(5, 1, density=0.8, random_state=0).tocsr()
    X2 = sp.random(0, 1, density=0.8, random_state=0).tocsr()
    last_mean, last_var = np.zeros(X1.shape[1]), np.zeros(X1.shape[1])
    last_n = np.zeros(X1.shape[1], dtype=np.int64)
    last_mean, last_var, last_n = incr_mean_variance_axis(
        X1, axis=axis, last_mean=last_mean, last_var=last_var, last_n=last_n
    )
    # update statistic with a column which should ignored
    updated_mean, updated_var, updated_n = incr_mean_variance_axis(
        X2, axis=axis, last_mean=last_mean, last_var=last_var, last_n=last_n
    )
    assert_allclose(updated_mean, last_mean)
    assert_allclose(updated_var, last_var)
    assert_allclose(updated_n, last_n)


def test_incr_mean_variance_n_float():
    # check the behaviour when last_n is just a number
    axis = 0
    X = sp.random(5, 2, density=0.8, random_state=0).tocsr()
    last_mean, last_var = np.zeros(X.shape[1]), np.zeros(X.shape[1])
    last_n = 0
    _, _, new_n = incr_mean_variance_axis(
        X, axis=axis, last_mean=last_mean, last_var=last_var, last_n=last_n
    )
    assert_allclose(new_n, np.full(X.shape[1], X.shape[0]))


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("sparse_constructor", [sp.csc_matrix, sp.csr_matrix])
def test_incr_mean_variance_axis_ignore_nan(axis, sparse_constructor):
    old_means = np.array([535.0, 535.0, 535.0, 535.0])
    old_variances = np.array([4225.0, 4225.0, 4225.0, 4225.0])
    old_sample_count = np.array([2, 2, 2, 2], dtype=np.int64)

    X = sparse_constructor(
        np.array([[170, 170, 170, 170], [430, 430, 430, 430], [300, 300, 300, 300]])
    )

    X_nan = sparse_constructor(
        np.array(
            [
                [170, np.nan, 170, 170],
                [np.nan, 170, 430, 430],
                [430, 430, np.nan, 300],
                [300, 300, 300, np.nan],
            ]
        )
    )

    # we avoid creating specific data for axis 0 and 1: translating the data is
    # enough.
    if axis:
        X = X.T
        X_nan = X_nan.T

    # take a copy of the old statistics since they are modified in place.
    X_means, X_vars, X_sample_count = incr_mean_variance_axis(
        X,
        axis=axis,
        last_mean=old_means.copy(),
        last_var=old_variances.copy(),
        last_n=old_sample_count.copy(),
    )
    X_nan_means, X_nan_vars, X_nan_sample_count = incr_mean_variance_axis(
        X_nan,
        axis=axis,
        last_mean=old_means.copy(),
        last_var=old_variances.copy(),
        last_n=old_sample_count.copy(),
    )

    assert_allclose(X_nan_means, X_means)
    assert_allclose(X_nan_vars, X_vars)
    assert_allclose(X_nan_sample_count, X_sample_count)


def test_mean_variance_illegal_axis():
    X, _ = make_classification(5, 4, random_state=0)
    # Sparsify the array a little bit
    X[0, 0] = 0
    X[2, 1] = 0
    X[4, 3] = 0
    X_csr = sp.csr_matrix(X)
    with pytest.raises(ValueError):
        mean_variance_axis(X_csr, axis=-3)
    with pytest.raises(ValueError):
        mean_variance_axis(X_csr, axis=2)
    with pytest.raises(ValueError):
        mean_variance_axis(X_csr, axis=-1)

    with pytest.raises(ValueError):
        incr_mean_variance_axis(
            X_csr, axis=-3, last_mean=None, last_var=None, last_n=None
        )

    with pytest.raises(ValueError):
        incr_mean_variance_axis(
            X_csr, axis=2, last_mean=None, last_var=None, last_n=None
        )

    with pytest.raises(ValueError):
        incr_mean_variance_axis(
            X_csr, axis=-1, last_mean=None, last_var=None, last_n=None
        )


def test_densify_rows():
    for dtype in (np.float32, np.float64):
        X = sp.csr_matrix(
            [[0, 3, 0], [2, 4, 0], [0, 0, 0], [9, 8, 7], [4, 0, 5]], dtype=dtype
        )
        X_rows = np.array([0, 2, 3], dtype=np.intp)
        out = np.ones((6, X.shape[1]), dtype=dtype)
        out_rows = np.array([1, 3, 4], dtype=np.intp)

        expect = np.ones_like(out)
        expect[out_rows] = X[X_rows, :].toarray()

        assign_rows_csr(X, X_rows, out_rows, out)
        assert_array_equal(out, expect)


def test_inplace_column_scale():
    rng = np.random.RandomState(0)
    X = sp.rand(100, 200, 0.05)
    Xr = X.tocsr()
    Xc = X.tocsc()
    XA = X.toarray()
    scale = rng.rand(200)
    XA *= scale

    inplace_column_scale(Xc, scale)
    inplace_column_scale(Xr, scale)
    assert_array_almost_equal(Xr.toarray(), Xc.toarray())
    assert_array_almost_equal(XA, Xc.toarray())
    assert_array_almost_equal(XA, Xr.toarray())
    with pytest.raises(TypeError):
        inplace_column_scale(X.tolil(), scale)

    X = X.astype(np.float32)
    scale = scale.astype(np.float32)
    Xr = X.tocsr()
    Xc = X.tocsc()
    XA = X.toarray()
    XA *= scale
    inplace_column_scale(Xc, scale)
    inplace_column_scale(Xr, scale)
    assert_array_almost_equal(Xr.toarray(), Xc.toarray())
    assert_array_almost_equal(XA, Xc.toarray())
    assert_array_almost_equal(XA, Xr.toarray())
    with pytest.raises(TypeError):
        inplace_column_scale(X.tolil(), scale)


def test_inplace_row_scale():
    rng = np.random.RandomState(0)
    X = sp.rand(100, 200, 0.05)
    Xr = X.tocsr()
    Xc = X.tocsc()
    XA = X.toarray()
    scale = rng.rand(100)
    XA *= scale.reshape(-1, 1)

    inplace_row_scale(Xc, scale)
    inplace_row_scale(Xr, scale)
    assert_array_almost_equal(Xr.toarray(), Xc.toarray())
    assert_array_almost_equal(XA, Xc.toarray())
    assert_array_almost_equal(XA, Xr.toarray())
    with pytest.raises(TypeError):
        inplace_column_scale(X.tolil(), scale)

    X = X.astype(np.float32)
    scale = scale.astype(np.float32)
    Xr = X.tocsr()
    Xc = X.tocsc()
    XA = X.toarray()
    XA *= scale.reshape(-1, 1)
    inplace_row_scale(Xc, scale)
    inplace_row_scale(Xr, scale)
    assert_array_almost_equal(Xr.toarray(), Xc.toarray())
    assert_array_almost_equal(XA, Xc.toarray())
    assert_array_almost_equal(XA, Xr.toarray())
    with pytest.raises(TypeError):
        inplace_column_scale(X.tolil(), scale)


def test_inplace_swap_row():
    X = np.array(
        [[0, 3, 0], [2, 4, 0], [0, 0, 0], [9, 8, 7], [4, 0, 5]], dtype=np.float64
    )
    X_csr = sp.csr_matrix(X)
    X_csc = sp.csc_matrix(X)

    swap = linalg.get_blas_funcs(("swap",), (X,))
    swap = swap[0]
    X[0], X[-1] = swap(X[0], X[-1])
    inplace_swap_row(X_csr, 0, -1)
    inplace_swap_row(X_csc, 0, -1)
    assert_array_equal(X_csr.toarray(), X_csc.toarray())
    assert_array_equal(X, X_csc.toarray())
    assert_array_equal(X, X_csr.toarray())

    X[2], X[3] = swap(X[2], X[3])
    inplace_swap_row(X_csr, 2, 3)
    inplace_swap_row(X_csc, 2, 3)
    assert_array_equal(X_csr.toarray(), X_csc.toarray())
    assert_array_equal(X, X_csc.toarray())
    assert_array_equal(X, X_csr.toarray())
    with pytest.raises(TypeError):
        inplace_swap_row(X_csr.tolil())

    X = np.array(
        [[0, 3, 0], [2, 4, 0], [0, 0, 0], [9, 8, 7], [4, 0, 5]], dtype=np.float32
    )
    X_csr = sp.csr_matrix(X)
    X_csc = sp.csc_matrix(X)
    swap = linalg.get_blas_funcs(("swap",), (X,))
    swap = swap[0]
    X[0], X[-1] = swap(X[0], X[-1])
    inplace_swap_row(X_csr, 0, -1)
    inplace_swap_row(X_csc, 0, -1)
    assert_array_equal(X_csr.toarray(), X_csc.toarray())
    assert_array_equal(X, X_csc.toarray())
    assert_array_equal(X, X_csr.toarray())
    X[2], X[3] = swap(X[2], X[3])
    inplace_swap_row(X_csr, 2, 3)
    inplace_swap_row(X_csc, 2, 3)
    assert_array_equal(X_csr.toarray(), X_csc.toarray())
    assert_array_equal(X, X_csc.toarray())
    assert_array_equal(X, X_csr.toarray())
    with pytest.raises(TypeError):
        inplace_swap_row(X_csr.tolil())


def test_inplace_swap_column():
    X = np.array(
        [[0, 3, 0], [2, 4, 0], [0, 0, 0], [9, 8, 7], [4, 0, 5]], dtype=np.float64
    )
    X_csr = sp.csr_matrix(X)
    X_csc = sp.csc_matrix(X)

    swap = linalg.get_blas_funcs(("swap",), (X,))
    swap = swap[0]
    X[:, 0], X[:, -1] = swap(X[:, 0], X[:, -1])
    inplace_swap_column(X_csr, 0, -1)
    inplace_swap_column(X_csc, 0, -1)
    assert_array_equal(X_csr.toarray(), X_csc.toarray())
    assert_array_equal(X, X_csc.toarray())
    assert_array_equal(X, X_csr.toarray())

    X[:, 0], X[:, 1] = swap(X[:, 0], X[:, 1])
    inplace_swap_column(X_csr, 0, 1)
    inplace_swap_column(X_csc, 0, 1)
    assert_array_equal(X_csr.toarray(), X_csc.toarray())
    assert_array_equal(X, X_csc.toarray())
    assert_array_equal(X, X_csr.toarray())
    with pytest.raises(TypeError):
        inplace_swap_column(X_csr.tolil())

    X = np.array(
        [[0, 3, 0], [2, 4, 0], [0, 0, 0], [9, 8, 7], [4, 0, 5]], dtype=np.float32
    )
    X_csr = sp.csr_matrix(X)
    X_csc = sp.csc_matrix(X)
    swap = linalg.get_blas_funcs(("swap",), (X,))
    swap = swap[0]
    X[:, 0], X[:, -1] = swap(X[:, 0], X[:, -1])
    inplace_swap_column(X_csr, 0, -1)
    inplace_swap_column(X_csc, 0, -1)
    assert_array_equal(X_csr.toarray(), X_csc.toarray())
    assert_array_equal(X, X_csc.toarray())
    assert_array_equal(X, X_csr.toarray())
    X[:, 0], X[:, 1] = swap(X[:, 0], X[:, 1])
    inplace_swap_column(X_csr, 0, 1)
    inplace_swap_column(X_csc, 0, 1)
    assert_array_equal(X_csr.toarray(), X_csc.toarray())
    assert_array_equal(X, X_csc.toarray())
    assert_array_equal(X, X_csr.toarray())
    with pytest.raises(TypeError):
        inplace_swap_column(X_csr.tolil())


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("axis", [0, 1, None])
@pytest.mark.parametrize("sparse_format", [sp.csr_matrix, sp.csc_matrix])
@pytest.mark.parametrize(
    "missing_values, min_func, max_func, ignore_nan",
    [(0, np.min, np.max, False), (np.nan, np.nanmin, np.nanmax, True)],
)
@pytest.mark.parametrize("large_indices", [True, False])
def test_min_max(
    dtype,
    axis,
    sparse_format,
    missing_values,
    min_func,
    max_func,
    ignore_nan,
    large_indices,
):
    X = np.array(
        [
            [0, 3, 0],
            [2, -1, missing_values],
            [0, 0, 0],
            [9, missing_values, 7],
            [4, 0, 5],
        ],
        dtype=dtype,
    )
    X_sparse = sparse_format(X)
    if large_indices:
        X_sparse.indices = X_sparse.indices.astype("int64")
        X_sparse.indptr = X_sparse.indptr.astype("int64")

    mins_sparse, maxs_sparse = min_max_axis(X_sparse, axis=axis, ignore_nan=ignore_nan)
    assert_array_equal(mins_sparse, min_func(X, axis=axis))
    assert_array_equal(maxs_sparse, max_func(X, axis=axis))


def test_min_max_axis_errors():
    X = np.array(
        [[0, 3, 0], [2, -1, 0], [0, 0, 0], [9, 8, 7], [4, 0, 5]], dtype=np.float64
    )
    X_csr = sp.csr_matrix(X)
    X_csc = sp.csc_matrix(X)
    with pytest.raises(TypeError):
        min_max_axis(X_csr.tolil(), axis=0)
    with pytest.raises(ValueError):
        min_max_axis(X_csr, axis=2)
    with pytest.raises(ValueError):
        min_max_axis(X_csc, axis=-3)


def test_count_nonzero():
    X = np.array(
        [[0, 3, 0], [2, -1, 0], [0, 0, 0], [9, 8, 7], [4, 0, 5]], dtype=np.float64
    )
    X_csr = sp.csr_matrix(X)
    X_csc = sp.csc_matrix(X)
    X_nonzero = X != 0
    sample_weight = [0.5, 0.2, 0.3, 0.1, 0.1]
    X_nonzero_weighted = X_nonzero * np.array(sample_weight)[:, None]

    for axis in [0, 1, -1, -2, None]:
        assert_array_almost_equal(
            count_nonzero(X_csr, axis=axis), X_nonzero.sum(axis=axis)
        )
        assert_array_almost_equal(
            count_nonzero(X_csr, axis=axis, sample_weight=sample_weight),
            X_nonzero_weighted.sum(axis=axis),
        )

    with pytest.raises(TypeError):
        count_nonzero(X_csc)
    with pytest.raises(ValueError):
        count_nonzero(X_csr, axis=2)

    assert count_nonzero(X_csr, axis=0).dtype == count_nonzero(X_csr, axis=1).dtype
    assert (
        count_nonzero(X_csr, axis=0, sample_weight=sample_weight).dtype
        == count_nonzero(X_csr, axis=1, sample_weight=sample_weight).dtype
    )

    # Check dtypes with large sparse matrices too
    # XXX: test fails on 32bit (Windows/Linux)
    try:
        X_csr.indices = X_csr.indices.astype(np.int64)
        X_csr.indptr = X_csr.indptr.astype(np.int64)
        assert count_nonzero(X_csr, axis=0).dtype == count_nonzero(X_csr, axis=1).dtype
        assert (
            count_nonzero(X_csr, axis=0, sample_weight=sample_weight).dtype
            == count_nonzero(X_csr, axis=1, sample_weight=sample_weight).dtype
        )
    except TypeError as e:
        assert "according to the rule 'safe'" in e.args[0] and np.intp().nbytes < 8, e


def test_csc_row_median():
    # Test csc_row_median actually calculates the median.

    # Test that it gives the same output when X is dense.
    rng = np.random.RandomState(0)
    X = rng.rand(100, 50)
    dense_median = np.median(X, axis=0)
    csc = sp.csc_matrix(X)
    sparse_median = csc_median_axis_0(csc)
    assert_array_equal(sparse_median, dense_median)

    # Test that it gives the same output when X is sparse
    X = rng.rand(51, 100)
    X[X < 0.7] = 0.0
    ind = rng.randint(0, 50, 10)
    X[ind] = -X[ind]
    csc = sp.csc_matrix(X)
    dense_median = np.median(X, axis=0)
    sparse_median = csc_median_axis_0(csc)
    assert_array_equal(sparse_median, dense_median)

    # Test for toy data.
    X = [[0, -2], [-1, -1], [1, 0], [2, 1]]
    csc = sp.csc_matrix(X)
    assert_array_equal(csc_median_axis_0(csc), np.array([0.5, -0.5]))
    X = [[0, -2], [-1, -5], [1, -3]]
    csc = sp.csc_matrix(X)
    assert_array_equal(csc_median_axis_0(csc), np.array([0.0, -3]))

    # Test that it raises an Error for non-csc matrices.
    with pytest.raises(TypeError):
        csc_median_axis_0(sp.csr_matrix(X))


def test_inplace_normalize():
    ones = np.ones((10, 1))
    rs = RandomState(10)

    for inplace_csr_row_normalize in (
        inplace_csr_row_normalize_l1,
        inplace_csr_row_normalize_l2,
    ):
        for dtype in (np.float64, np.float32):
            X = rs.randn(10, 5).astype(dtype)
            X_csr = sp.csr_matrix(X)
            for index_dtype in [np.int32, np.int64]:
                # csr_matrix will use int32 indices by default,
                # up-casting those to int64 when necessary
                if index_dtype is np.int64:
                    X_csr.indptr = X_csr.indptr.astype(index_dtype)
                    X_csr.indices = X_csr.indices.astype(index_dtype)
                assert X_csr.indices.dtype == index_dtype
                assert X_csr.indptr.dtype == index_dtype
                inplace_csr_row_normalize(X_csr)
                assert X_csr.dtype == dtype
                if inplace_csr_row_normalize is inplace_csr_row_normalize_l2:
                    X_csr.data **= 2
                assert_array_almost_equal(np.abs(X_csr).sum(axis=1), ones)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_csr_row_norms(dtype):
    # checks that csr_row_norms returns the same output as
    # scipy.sparse.linalg.norm, and that the dype is the same as X.dtype.
    X = sp.random(100, 10, format="csr", dtype=dtype, random_state=42)

    scipy_norms = sp.linalg.norm(X, axis=1) ** 2
    norms = csr_row_norms(X)

    assert norms.dtype == dtype
    rtol = 1e-6 if dtype == np.float32 else 1e-7
    assert_allclose(norms, scipy_norms, rtol=rtol)
