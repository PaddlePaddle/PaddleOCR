# Authors:
#
#          Giorgio Patrini
#
# License: BSD 3 clause

import warnings
import itertools

import re
import numpy as np
import numpy.linalg as la
from scipy import sparse, stats

import pytest

from sklearn.utils import gen_batches

from sklearn.utils._testing import assert_almost_equal
from sklearn.utils._testing import assert_array_almost_equal
from sklearn.utils._testing import assert_array_equal
from sklearn.utils._testing import assert_array_less
from sklearn.utils._testing import assert_allclose
from sklearn.utils._testing import assert_allclose_dense_sparse
from sklearn.utils._testing import skip_if_32bit
from sklearn.utils._testing import _convert_container

from sklearn.utils.sparsefuncs import mean_variance_axis
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import KernelCenterer
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import quantile_transform
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import maxabs_scale
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import robust_scale
from sklearn.preprocessing import add_dummy_feature
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import power_transform
from sklearn.preprocessing._data import _handle_zeros_in_scale
from sklearn.preprocessing._data import BOUNDS_THRESHOLD

from sklearn.exceptions import NotFittedError

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict
from sklearn.svm import SVR
from sklearn.utils import shuffle

from sklearn import datasets


iris = datasets.load_iris()

# Make some data to be used many times
rng = np.random.RandomState(0)
n_features = 30
n_samples = 1000
offsets = rng.uniform(-1, 1, size=n_features)
scales = rng.uniform(1, 10, size=n_features)
X_2d = rng.randn(n_samples, n_features) * scales + offsets
X_1row = X_2d[0, :].reshape(1, n_features)
X_1col = X_2d[:, 0].reshape(n_samples, 1)
X_list_1row = X_1row.tolist()
X_list_1col = X_1col.tolist()


def toarray(a):
    if hasattr(a, "toarray"):
        a = a.toarray()
    return a


def _check_dim_1axis(a):
    return np.asarray(a).shape[0]


def assert_correct_incr(i, batch_start, batch_stop, n, chunk_size, n_samples_seen):
    if batch_stop != n:
        assert (i + 1) * chunk_size == n_samples_seen
    else:
        assert i * chunk_size + (batch_stop - batch_start) == n_samples_seen


def test_raises_value_error_if_sample_weights_greater_than_1d():
    # Sample weights must be either scalar or 1D

    n_sampless = [2, 3]
    n_featuress = [3, 2]

    for n_samples, n_features in zip(n_sampless, n_featuress):

        X = rng.randn(n_samples, n_features)
        y = rng.randn(n_samples)

        scaler = StandardScaler()

        # make sure Error is raised the sample weights greater than 1d
        sample_weight_notOK = rng.randn(n_samples, 1) ** 2
        with pytest.raises(ValueError):
            scaler.fit(X, y, sample_weight=sample_weight_notOK)


@pytest.mark.parametrize(
    ["Xw", "X", "sample_weight"],
    [
        ([[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [1, 2, 3], [4, 5, 6]], [2.0, 1.0]),
        (
            [[1, 0, 1], [0, 0, 1]],
            [[1, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]],
            np.array([1, 3]),
        ),
        (
            [[1, np.nan, 1], [np.nan, np.nan, 1]],
            [
                [1, np.nan, 1],
                [np.nan, np.nan, 1],
                [np.nan, np.nan, 1],
                [np.nan, np.nan, 1],
            ],
            np.array([1, 3]),
        ),
    ],
)
@pytest.mark.parametrize("array_constructor", ["array", "sparse_csr", "sparse_csc"])
def test_standard_scaler_sample_weight(Xw, X, sample_weight, array_constructor):
    with_mean = not array_constructor.startswith("sparse")
    X = _convert_container(X, array_constructor)
    Xw = _convert_container(Xw, array_constructor)

    # weighted StandardScaler
    yw = np.ones(Xw.shape[0])
    scaler_w = StandardScaler(with_mean=with_mean)
    scaler_w.fit(Xw, yw, sample_weight=sample_weight)

    # unweighted, but with repeated samples
    y = np.ones(X.shape[0])
    scaler = StandardScaler(with_mean=with_mean)
    scaler.fit(X, y)

    X_test = [[1.5, 2.5, 3.5], [3.5, 4.5, 5.5]]

    assert_almost_equal(scaler.mean_, scaler_w.mean_)
    assert_almost_equal(scaler.var_, scaler_w.var_)
    assert_almost_equal(scaler.transform(X_test), scaler_w.transform(X_test))


def test_standard_scaler_1d():
    # Test scaling of dataset along single axis
    for X in [X_1row, X_1col, X_list_1row, X_list_1row]:
        scaler = StandardScaler()
        X_scaled = scaler.fit(X).transform(X, copy=True)

        if isinstance(X, list):
            X = np.array(X)  # cast only after scaling done

        if _check_dim_1axis(X) == 1:
            assert_almost_equal(scaler.mean_, X.ravel())
            assert_almost_equal(scaler.scale_, np.ones(n_features))
            assert_array_almost_equal(X_scaled.mean(axis=0), np.zeros_like(n_features))
            assert_array_almost_equal(X_scaled.std(axis=0), np.zeros_like(n_features))
        else:
            assert_almost_equal(scaler.mean_, X.mean())
            assert_almost_equal(scaler.scale_, X.std())
            assert_array_almost_equal(X_scaled.mean(axis=0), np.zeros_like(n_features))
            assert_array_almost_equal(X_scaled.mean(axis=0), 0.0)
            assert_array_almost_equal(X_scaled.std(axis=0), 1.0)
        assert scaler.n_samples_seen_ == X.shape[0]

        # check inverse transform
        X_scaled_back = scaler.inverse_transform(X_scaled)
        assert_array_almost_equal(X_scaled_back, X)

    # Constant feature
    X = np.ones((5, 1))
    scaler = StandardScaler()
    X_scaled = scaler.fit(X).transform(X, copy=True)
    assert_almost_equal(scaler.mean_, 1.0)
    assert_almost_equal(scaler.scale_, 1.0)
    assert_array_almost_equal(X_scaled.mean(axis=0), 0.0)
    assert_array_almost_equal(X_scaled.std(axis=0), 0.0)
    assert scaler.n_samples_seen_ == X.shape[0]


@pytest.mark.parametrize(
    "sparse_constructor", [None, sparse.csc_matrix, sparse.csr_matrix]
)
@pytest.mark.parametrize("add_sample_weight", [False, True])
def test_standard_scaler_dtype(add_sample_weight, sparse_constructor):
    # Ensure scaling does not affect dtype
    rng = np.random.RandomState(0)
    n_samples = 10
    n_features = 3
    if add_sample_weight:
        sample_weight = np.ones(n_samples)
    else:
        sample_weight = None
    with_mean = True
    for dtype in [np.float16, np.float32, np.float64]:
        X = rng.randn(n_samples, n_features).astype(dtype)
        if sparse_constructor is not None:
            X = sparse_constructor(X)
            with_mean = False

        scaler = StandardScaler(with_mean=with_mean)
        X_scaled = scaler.fit(X, sample_weight=sample_weight).transform(X)
        assert X.dtype == X_scaled.dtype
        assert scaler.mean_.dtype == np.float64
        assert scaler.scale_.dtype == np.float64


@pytest.mark.parametrize(
    "scaler",
    [
        StandardScaler(with_mean=False),
        RobustScaler(with_centering=False),
    ],
)
@pytest.mark.parametrize(
    "sparse_constructor", [np.asarray, sparse.csc_matrix, sparse.csr_matrix]
)
@pytest.mark.parametrize("add_sample_weight", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("constant", [0, 1.0, 100.0])
def test_standard_scaler_constant_features(
    scaler, add_sample_weight, sparse_constructor, dtype, constant
):

    if isinstance(scaler, RobustScaler) and add_sample_weight:
        pytest.skip(f"{scaler.__class__.__name__} does not yet support sample_weight")

    rng = np.random.RandomState(0)
    n_samples = 100
    n_features = 1
    if add_sample_weight:
        fit_params = dict(sample_weight=rng.uniform(size=n_samples) * 2)
    else:
        fit_params = {}
    X_array = np.full(shape=(n_samples, n_features), fill_value=constant, dtype=dtype)
    X = sparse_constructor(X_array)
    X_scaled = scaler.fit(X, **fit_params).transform(X)

    if isinstance(scaler, StandardScaler):
        # The variance info should be close to zero for constant features.
        assert_allclose(scaler.var_, np.zeros(X.shape[1]), atol=1e-7)

    # Constant features should not be scaled (scale of 1.):
    assert_allclose(scaler.scale_, np.ones(X.shape[1]))

    if hasattr(X_scaled, "toarray"):
        assert_allclose(X_scaled.toarray(), X_array)
    else:
        assert_allclose(X_scaled, X)

    if isinstance(scaler, StandardScaler) and not add_sample_weight:
        # Also check consistency with the standard scale function.
        X_scaled_2 = scale(X, with_mean=scaler.with_mean)
        if hasattr(X_scaled_2, "toarray"):
            assert_allclose(X_scaled_2.toarray(), X_scaled_2.toarray())
        else:
            assert_allclose(X_scaled_2, X_scaled_2)


@pytest.mark.parametrize("n_samples", [10, 100, 10_000])
@pytest.mark.parametrize("average", [1e-10, 1, 1e10])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "array_constructor", [np.asarray, sparse.csc_matrix, sparse.csr_matrix]
)
def test_standard_scaler_near_constant_features(
    n_samples, array_constructor, average, dtype
):
    # Check that when the variance is too small (var << mean**2) the feature
    # is considered constant and not scaled.

    scale_min, scale_max = -30, 19
    scales = np.array([10 ** i for i in range(scale_min, scale_max + 1)], dtype=dtype)

    n_features = scales.shape[0]
    X = np.empty((n_samples, n_features), dtype=dtype)
    # Make a dataset of known var = scales**2 and mean = average
    X[: n_samples // 2, :] = average + scales
    X[n_samples // 2 :, :] = average - scales
    X_array = array_constructor(X)

    scaler = StandardScaler(with_mean=False).fit(X_array)

    # StandardScaler uses float64 accumulators even if the data has a float32
    # dtype.
    eps = np.finfo(np.float64).eps

    # if var < bound = N.eps.var + N².eps².mean², the feature is considered
    # constant and the scale_ attribute is set to 1.
    bounds = n_samples * eps * scales ** 2 + n_samples ** 2 * eps ** 2 * average ** 2
    within_bounds = scales ** 2 <= bounds

    # Check that scale_min is small enough to have some scales below the
    # bound and therefore detected as constant:
    assert np.any(within_bounds)

    # Check that such features are actually treated as constant by the scaler:
    assert all(scaler.var_[within_bounds] <= bounds[within_bounds])
    assert_allclose(scaler.scale_[within_bounds], 1.0)

    # Depending the on the dtype of X, some features might not actually be
    # representable as non constant for small scales (even if above the
    # precision bound of the float64 variance estimate). Such feature should
    # be correctly detected as constants with 0 variance by StandardScaler.
    representable_diff = X[0, :] - X[-1, :] != 0
    assert_allclose(scaler.var_[np.logical_not(representable_diff)], 0)
    assert_allclose(scaler.scale_[np.logical_not(representable_diff)], 1)

    # The other features are scaled and scale_ is equal to sqrt(var_) assuming
    # that scales are large enough for average + scale and average - scale to
    # be distinct in X (depending on X's dtype).
    common_mask = np.logical_and(scales ** 2 > bounds, representable_diff)
    assert_allclose(scaler.scale_[common_mask], np.sqrt(scaler.var_)[common_mask])


def test_scale_1d():
    # 1-d inputs
    X_list = [1.0, 3.0, 5.0, 0.0]
    X_arr = np.array(X_list)

    for X in [X_list, X_arr]:
        X_scaled = scale(X)
        assert_array_almost_equal(X_scaled.mean(), 0.0)
        assert_array_almost_equal(X_scaled.std(), 1.0)
        assert_array_equal(scale(X, with_mean=False, with_std=False), X)


@skip_if_32bit
def test_standard_scaler_numerical_stability():
    # Test numerical stability of scaling
    # np.log(1e-5) is taken because of its floating point representation
    # was empirically found to cause numerical problems with np.mean & np.std.
    x = np.full(8, np.log(1e-5), dtype=np.float64)
    # This does not raise a warning as the number of samples is too low
    # to trigger the problem in recent numpy
    with pytest.warns(None) as record:
        scale(x)
    assert len(record) == 0
    assert_array_almost_equal(scale(x), np.zeros(8))

    # with 2 more samples, the std computation run into numerical issues:
    x = np.full(10, np.log(1e-5), dtype=np.float64)
    warning_message = "standard deviation of the data is probably very close to 0"
    with pytest.warns(UserWarning, match=warning_message):
        x_scaled = scale(x)
    assert_array_almost_equal(x_scaled, np.zeros(10))

    x = np.full(10, 1e-100, dtype=np.float64)
    with pytest.warns(None) as record:
        x_small_scaled = scale(x)
    assert len(record) == 0
    assert_array_almost_equal(x_small_scaled, np.zeros(10))

    # Large values can cause (often recoverable) numerical stability issues:
    x_big = np.full(10, 1e100, dtype=np.float64)
    warning_message = "Dataset may contain too large values"
    with pytest.warns(UserWarning, match=warning_message):
        x_big_scaled = scale(x_big)
    assert_array_almost_equal(x_big_scaled, np.zeros(10))
    assert_array_almost_equal(x_big_scaled, x_small_scaled)
    with pytest.warns(UserWarning, match=warning_message):
        x_big_centered = scale(x_big, with_std=False)
    assert_array_almost_equal(x_big_centered, np.zeros(10))
    assert_array_almost_equal(x_big_centered, x_small_scaled)


def test_scaler_2d_arrays():
    # Test scaling of 2d array along first axis
    rng = np.random.RandomState(0)
    n_features = 5
    n_samples = 4
    X = rng.randn(n_samples, n_features)
    X[:, 0] = 0.0  # first feature is always of zero

    scaler = StandardScaler()
    X_scaled = scaler.fit(X).transform(X, copy=True)
    assert not np.any(np.isnan(X_scaled))
    assert scaler.n_samples_seen_ == n_samples

    assert_array_almost_equal(X_scaled.mean(axis=0), n_features * [0.0])
    assert_array_almost_equal(X_scaled.std(axis=0), [0.0, 1.0, 1.0, 1.0, 1.0])
    # Check that X has been copied
    assert X_scaled is not X

    # check inverse transform
    X_scaled_back = scaler.inverse_transform(X_scaled)
    assert X_scaled_back is not X
    assert X_scaled_back is not X_scaled
    assert_array_almost_equal(X_scaled_back, X)

    X_scaled = scale(X, axis=1, with_std=False)
    assert not np.any(np.isnan(X_scaled))
    assert_array_almost_equal(X_scaled.mean(axis=1), n_samples * [0.0])
    X_scaled = scale(X, axis=1, with_std=True)
    assert not np.any(np.isnan(X_scaled))
    assert_array_almost_equal(X_scaled.mean(axis=1), n_samples * [0.0])
    assert_array_almost_equal(X_scaled.std(axis=1), n_samples * [1.0])
    # Check that the data hasn't been modified
    assert X_scaled is not X

    X_scaled = scaler.fit(X).transform(X, copy=False)
    assert not np.any(np.isnan(X_scaled))
    assert_array_almost_equal(X_scaled.mean(axis=0), n_features * [0.0])
    assert_array_almost_equal(X_scaled.std(axis=0), [0.0, 1.0, 1.0, 1.0, 1.0])
    # Check that X has not been copied
    assert X_scaled is X

    X = rng.randn(4, 5)
    X[:, 0] = 1.0  # first feature is a constant, non zero feature
    scaler = StandardScaler()
    X_scaled = scaler.fit(X).transform(X, copy=True)
    assert not np.any(np.isnan(X_scaled))
    assert_array_almost_equal(X_scaled.mean(axis=0), n_features * [0.0])
    assert_array_almost_equal(X_scaled.std(axis=0), [0.0, 1.0, 1.0, 1.0, 1.0])
    # Check that X has not been copied
    assert X_scaled is not X


def test_scaler_float16_overflow():
    # Test if the scaler will not overflow on float16 numpy arrays
    rng = np.random.RandomState(0)
    # float16 has a maximum of 65500.0. On the worst case 5 * 200000 is 100000
    # which is enough to overflow the data type
    X = rng.uniform(5, 10, [200000, 1]).astype(np.float16)

    with np.errstate(over="raise"):
        scaler = StandardScaler().fit(X)
        X_scaled = scaler.transform(X)

    # Calculate the float64 equivalent to verify result
    X_scaled_f64 = StandardScaler().fit_transform(X.astype(np.float64))

    # Overflow calculations may cause -inf, inf, or nan. Since there is no nan
    # input, all of the outputs should be finite. This may be redundant since a
    # FloatingPointError exception will be thrown on overflow above.
    assert np.all(np.isfinite(X_scaled))

    # The normal distribution is very unlikely to go above 4. At 4.0-8.0 the
    # float16 precision is 2^-8 which is around 0.004. Thus only 2 decimals are
    # checked to account for precision differences.
    assert_array_almost_equal(X_scaled, X_scaled_f64, decimal=2)


def test_handle_zeros_in_scale():
    s1 = np.array([0, 1e-16, 1, 2, 3])
    s2 = _handle_zeros_in_scale(s1, copy=True)

    assert_allclose(s1, np.array([0, 1e-16, 1, 2, 3]))
    assert_allclose(s2, np.array([1, 1, 1, 2, 3]))


def test_minmax_scaler_partial_fit():
    # Test if partial_fit run over many batches of size 1 and 50
    # gives the same results as fit
    X = X_2d
    n = X.shape[0]

    for chunk_size in [1, 2, 50, n, n + 42]:
        # Test mean at the end of the process
        scaler_batch = MinMaxScaler().fit(X)

        scaler_incr = MinMaxScaler()
        for batch in gen_batches(n_samples, chunk_size):
            scaler_incr = scaler_incr.partial_fit(X[batch])

        assert_array_almost_equal(scaler_batch.data_min_, scaler_incr.data_min_)
        assert_array_almost_equal(scaler_batch.data_max_, scaler_incr.data_max_)
        assert scaler_batch.n_samples_seen_ == scaler_incr.n_samples_seen_
        assert_array_almost_equal(scaler_batch.data_range_, scaler_incr.data_range_)
        assert_array_almost_equal(scaler_batch.scale_, scaler_incr.scale_)
        assert_array_almost_equal(scaler_batch.min_, scaler_incr.min_)

        # Test std after 1 step
        batch0 = slice(0, chunk_size)
        scaler_batch = MinMaxScaler().fit(X[batch0])
        scaler_incr = MinMaxScaler().partial_fit(X[batch0])

        assert_array_almost_equal(scaler_batch.data_min_, scaler_incr.data_min_)
        assert_array_almost_equal(scaler_batch.data_max_, scaler_incr.data_max_)
        assert scaler_batch.n_samples_seen_ == scaler_incr.n_samples_seen_
        assert_array_almost_equal(scaler_batch.data_range_, scaler_incr.data_range_)
        assert_array_almost_equal(scaler_batch.scale_, scaler_incr.scale_)
        assert_array_almost_equal(scaler_batch.min_, scaler_incr.min_)

        # Test std until the end of partial fits, and
        scaler_batch = MinMaxScaler().fit(X)
        scaler_incr = MinMaxScaler()  # Clean estimator
        for i, batch in enumerate(gen_batches(n_samples, chunk_size)):
            scaler_incr = scaler_incr.partial_fit(X[batch])
            assert_correct_incr(
                i,
                batch_start=batch.start,
                batch_stop=batch.stop,
                n=n,
                chunk_size=chunk_size,
                n_samples_seen=scaler_incr.n_samples_seen_,
            )


def test_standard_scaler_partial_fit():
    # Test if partial_fit run over many batches of size 1 and 50
    # gives the same results as fit
    X = X_2d
    n = X.shape[0]

    for chunk_size in [1, 2, 50, n, n + 42]:
        # Test mean at the end of the process
        scaler_batch = StandardScaler(with_std=False).fit(X)

        scaler_incr = StandardScaler(with_std=False)
        for batch in gen_batches(n_samples, chunk_size):
            scaler_incr = scaler_incr.partial_fit(X[batch])
        assert_array_almost_equal(scaler_batch.mean_, scaler_incr.mean_)
        assert scaler_batch.var_ == scaler_incr.var_  # Nones
        assert scaler_batch.n_samples_seen_ == scaler_incr.n_samples_seen_

        # Test std after 1 step
        batch0 = slice(0, chunk_size)
        scaler_incr = StandardScaler().partial_fit(X[batch0])
        if chunk_size == 1:
            assert_array_almost_equal(
                np.zeros(n_features, dtype=np.float64), scaler_incr.var_
            )
            assert_array_almost_equal(
                np.ones(n_features, dtype=np.float64), scaler_incr.scale_
            )
        else:
            assert_array_almost_equal(np.var(X[batch0], axis=0), scaler_incr.var_)
            assert_array_almost_equal(
                np.std(X[batch0], axis=0), scaler_incr.scale_
            )  # no constants

        # Test std until the end of partial fits, and
        scaler_batch = StandardScaler().fit(X)
        scaler_incr = StandardScaler()  # Clean estimator
        for i, batch in enumerate(gen_batches(n_samples, chunk_size)):
            scaler_incr = scaler_incr.partial_fit(X[batch])
            assert_correct_incr(
                i,
                batch_start=batch.start,
                batch_stop=batch.stop,
                n=n,
                chunk_size=chunk_size,
                n_samples_seen=scaler_incr.n_samples_seen_,
            )

        assert_array_almost_equal(scaler_batch.var_, scaler_incr.var_)
        assert scaler_batch.n_samples_seen_ == scaler_incr.n_samples_seen_


def test_standard_scaler_partial_fit_numerical_stability():
    # Test if the incremental computation introduces significative errors
    # for large datasets with values of large magniture
    rng = np.random.RandomState(0)
    n_features = 2
    n_samples = 100
    offsets = rng.uniform(-1e15, 1e15, size=n_features)
    scales = rng.uniform(1e3, 1e6, size=n_features)
    X = rng.randn(n_samples, n_features) * scales + offsets

    scaler_batch = StandardScaler().fit(X)
    scaler_incr = StandardScaler()
    for chunk in X:
        scaler_incr = scaler_incr.partial_fit(chunk.reshape(1, n_features))

    # Regardless of abs values, they must not be more diff 6 significant digits
    tol = 10 ** (-6)
    assert_allclose(scaler_incr.mean_, scaler_batch.mean_, rtol=tol)
    assert_allclose(scaler_incr.var_, scaler_batch.var_, rtol=tol)
    assert_allclose(scaler_incr.scale_, scaler_batch.scale_, rtol=tol)
    # NOTE Be aware that for much larger offsets std is very unstable (last
    # assert) while mean is OK.

    # Sparse input
    size = (100, 3)
    scale = 1e20
    X = rng.randint(0, 2, size).astype(np.float64) * scale
    X_csr = sparse.csr_matrix(X)
    X_csc = sparse.csc_matrix(X)

    for X in [X_csr, X_csc]:
        # with_mean=False is required with sparse input
        scaler = StandardScaler(with_mean=False).fit(X)
        scaler_incr = StandardScaler(with_mean=False)

        for chunk in X:
            # chunk = sparse.csr_matrix(data_chunks)
            scaler_incr = scaler_incr.partial_fit(chunk)

        # Regardless of magnitude, they must not differ more than of 6 digits
        tol = 10 ** (-6)
        assert scaler.mean_ is not None
        assert_allclose(scaler_incr.var_, scaler.var_, rtol=tol)
        assert_allclose(scaler_incr.scale_, scaler.scale_, rtol=tol)


@pytest.mark.parametrize("sample_weight", [True, None])
def test_partial_fit_sparse_input(sample_weight):
    # Check that sparsity is not destroyed
    X = np.array([[1.0], [0.0], [0.0], [5.0]])
    X_csr = sparse.csr_matrix(X)
    X_csc = sparse.csc_matrix(X)

    if sample_weight:
        sample_weight = rng.rand(X_csc.shape[0])

    null_transform = StandardScaler(with_mean=False, with_std=False, copy=True)
    for X in [X_csr, X_csc]:

        X_null = null_transform.partial_fit(X, sample_weight=sample_weight).transform(X)
        assert_array_equal(X_null.toarray(), X.toarray())
        X_orig = null_transform.inverse_transform(X_null)
        assert_array_equal(X_orig.toarray(), X_null.toarray())
        assert_array_equal(X_orig.toarray(), X.toarray())


@pytest.mark.parametrize("sample_weight", [True, None])
def test_standard_scaler_trasform_with_partial_fit(sample_weight):
    # Check some postconditions after applying partial_fit and transform
    X = X_2d[:100, :]

    if sample_weight:
        sample_weight = rng.rand(X.shape[0])

    scaler_incr = StandardScaler()
    for i, batch in enumerate(gen_batches(X.shape[0], 1)):

        X_sofar = X[: (i + 1), :]
        chunks_copy = X_sofar.copy()
        if sample_weight is None:
            scaled_batch = StandardScaler().fit_transform(X_sofar)
            scaler_incr = scaler_incr.partial_fit(X[batch])
        else:
            scaled_batch = StandardScaler().fit_transform(
                X_sofar, sample_weight=sample_weight[: i + 1]
            )
            scaler_incr = scaler_incr.partial_fit(
                X[batch], sample_weight=sample_weight[batch]
            )
        scaled_incr = scaler_incr.transform(X_sofar)

        assert_array_almost_equal(scaled_batch, scaled_incr)
        assert_array_almost_equal(X_sofar, chunks_copy)  # No change
        right_input = scaler_incr.inverse_transform(scaled_incr)
        assert_array_almost_equal(X_sofar, right_input)

        zero = np.zeros(X.shape[1])
        epsilon = np.finfo(float).eps
        assert_array_less(zero, scaler_incr.var_ + epsilon)  # as less or equal
        assert_array_less(zero, scaler_incr.scale_ + epsilon)
        if sample_weight is None:
            # (i+1) because the Scaler has been already fitted
            assert (i + 1) == scaler_incr.n_samples_seen_
        else:
            assert np.sum(sample_weight[: i + 1]) == pytest.approx(
                scaler_incr.n_samples_seen_
            )


def test_standard_check_array_of_inverse_transform():
    # Check if StandardScaler inverse_transform is
    # converting the integer array to float
    x = np.array(
        [
            [1, 1, 1, 0, 1, 0],
            [1, 1, 1, 0, 1, 0],
            [0, 8, 0, 1, 0, 0],
            [1, 4, 1, 1, 0, 0],
            [0, 1, 0, 0, 1, 0],
            [0, 4, 0, 1, 0, 1],
        ],
        dtype=np.int32,
    )

    scaler = StandardScaler()
    scaler.fit(x)

    # The of inverse_transform should be converted
    # to a float array.
    # If not X *= self.scale_ will fail.
    scaler.inverse_transform(x)


def test_min_max_scaler_iris():
    X = iris.data
    scaler = MinMaxScaler()
    # default params
    X_trans = scaler.fit_transform(X)
    assert_array_almost_equal(X_trans.min(axis=0), 0)
    assert_array_almost_equal(X_trans.max(axis=0), 1)
    X_trans_inv = scaler.inverse_transform(X_trans)
    assert_array_almost_equal(X, X_trans_inv)

    # not default params: min=1, max=2
    scaler = MinMaxScaler(feature_range=(1, 2))
    X_trans = scaler.fit_transform(X)
    assert_array_almost_equal(X_trans.min(axis=0), 1)
    assert_array_almost_equal(X_trans.max(axis=0), 2)
    X_trans_inv = scaler.inverse_transform(X_trans)
    assert_array_almost_equal(X, X_trans_inv)

    # min=-.5, max=.6
    scaler = MinMaxScaler(feature_range=(-0.5, 0.6))
    X_trans = scaler.fit_transform(X)
    assert_array_almost_equal(X_trans.min(axis=0), -0.5)
    assert_array_almost_equal(X_trans.max(axis=0), 0.6)
    X_trans_inv = scaler.inverse_transform(X_trans)
    assert_array_almost_equal(X, X_trans_inv)

    # raises on invalid range
    scaler = MinMaxScaler(feature_range=(2, 1))
    with pytest.raises(ValueError):
        scaler.fit(X)


def test_min_max_scaler_zero_variance_features():
    # Check min max scaler on toy data with zero variance features
    X = [[0.0, 1.0, +0.5], [0.0, 1.0, -0.1], [0.0, 1.0, +1.1]]

    X_new = [[+0.0, 2.0, 0.5], [-1.0, 1.0, 0.0], [+0.0, 1.0, 1.5]]

    # default params
    scaler = MinMaxScaler()
    X_trans = scaler.fit_transform(X)
    X_expected_0_1 = [[0.0, 0.0, 0.5], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    assert_array_almost_equal(X_trans, X_expected_0_1)
    X_trans_inv = scaler.inverse_transform(X_trans)
    assert_array_almost_equal(X, X_trans_inv)

    X_trans_new = scaler.transform(X_new)
    X_expected_0_1_new = [[+0.0, 1.0, 0.500], [-1.0, 0.0, 0.083], [+0.0, 0.0, 1.333]]
    assert_array_almost_equal(X_trans_new, X_expected_0_1_new, decimal=2)

    # not default params
    scaler = MinMaxScaler(feature_range=(1, 2))
    X_trans = scaler.fit_transform(X)
    X_expected_1_2 = [[1.0, 1.0, 1.5], [1.0, 1.0, 1.0], [1.0, 1.0, 2.0]]
    assert_array_almost_equal(X_trans, X_expected_1_2)

    # function interface
    X_trans = minmax_scale(X)
    assert_array_almost_equal(X_trans, X_expected_0_1)
    X_trans = minmax_scale(X, feature_range=(1, 2))
    assert_array_almost_equal(X_trans, X_expected_1_2)


def test_minmax_scale_axis1():
    X = iris.data
    X_trans = minmax_scale(X, axis=1)
    assert_array_almost_equal(np.min(X_trans, axis=1), 0)
    assert_array_almost_equal(np.max(X_trans, axis=1), 1)


def test_min_max_scaler_1d():
    # Test scaling of dataset along single axis
    for X in [X_1row, X_1col, X_list_1row, X_list_1row]:

        scaler = MinMaxScaler(copy=True)
        X_scaled = scaler.fit(X).transform(X)

        if isinstance(X, list):
            X = np.array(X)  # cast only after scaling done

        if _check_dim_1axis(X) == 1:
            assert_array_almost_equal(X_scaled.min(axis=0), np.zeros(n_features))
            assert_array_almost_equal(X_scaled.max(axis=0), np.zeros(n_features))
        else:
            assert_array_almost_equal(X_scaled.min(axis=0), 0.0)
            assert_array_almost_equal(X_scaled.max(axis=0), 1.0)
        assert scaler.n_samples_seen_ == X.shape[0]

        # check inverse transform
        X_scaled_back = scaler.inverse_transform(X_scaled)
        assert_array_almost_equal(X_scaled_back, X)

    # Constant feature
    X = np.ones((5, 1))
    scaler = MinMaxScaler()
    X_scaled = scaler.fit(X).transform(X)
    assert X_scaled.min() >= 0.0
    assert X_scaled.max() <= 1.0
    assert scaler.n_samples_seen_ == X.shape[0]

    # Function interface
    X_1d = X_1row.ravel()
    min_ = X_1d.min()
    max_ = X_1d.max()
    assert_array_almost_equal(
        (X_1d - min_) / (max_ - min_), minmax_scale(X_1d, copy=True)
    )


@pytest.mark.parametrize("sample_weight", [True, None])
def test_scaler_without_centering(sample_weight):
    rng = np.random.RandomState(42)
    X = rng.randn(4, 5)
    X[:, 0] = 0.0  # first feature is always of zero
    X_csr = sparse.csr_matrix(X)
    X_csc = sparse.csc_matrix(X)

    if sample_weight:
        sample_weight = rng.rand(X.shape[0])

    with pytest.raises(ValueError):
        StandardScaler().fit(X_csr)
    with pytest.raises(ValueError):
        StandardScaler().fit(X_csc)

    null_transform = StandardScaler(with_mean=False, with_std=False, copy=True)
    X_null = null_transform.fit_transform(X_csr)
    assert_array_equal(X_null.data, X_csr.data)
    X_orig = null_transform.inverse_transform(X_null)
    assert_array_equal(X_orig.data, X_csr.data)

    scaler = StandardScaler(with_mean=False).fit(X, sample_weight=sample_weight)
    X_scaled = scaler.transform(X, copy=True)
    assert not np.any(np.isnan(X_scaled))

    scaler_csr = StandardScaler(with_mean=False).fit(X_csr, sample_weight=sample_weight)
    X_csr_scaled = scaler_csr.transform(X_csr, copy=True)
    assert not np.any(np.isnan(X_csr_scaled.data))

    scaler_csc = StandardScaler(with_mean=False).fit(X_csc, sample_weight=sample_weight)
    X_csc_scaled = scaler_csc.transform(X_csc, copy=True)
    assert not np.any(np.isnan(X_csc_scaled.data))

    assert_array_almost_equal(scaler.mean_, scaler_csr.mean_)
    assert_array_almost_equal(scaler.var_, scaler_csr.var_)
    assert_array_almost_equal(scaler.scale_, scaler_csr.scale_)
    assert_array_almost_equal(scaler.n_samples_seen_, scaler_csr.n_samples_seen_)

    assert_array_almost_equal(scaler.mean_, scaler_csc.mean_)
    assert_array_almost_equal(scaler.var_, scaler_csc.var_)
    assert_array_almost_equal(scaler.scale_, scaler_csc.scale_)
    assert_array_almost_equal(scaler.n_samples_seen_, scaler_csc.n_samples_seen_)

    if sample_weight is None:
        assert_array_almost_equal(
            X_scaled.mean(axis=0), [0.0, -0.01, 2.24, -0.35, -0.78], 2
        )
        assert_array_almost_equal(X_scaled.std(axis=0), [0.0, 1.0, 1.0, 1.0, 1.0])

    X_csr_scaled_mean, X_csr_scaled_var = mean_variance_axis(X_csr_scaled, 0)
    assert_array_almost_equal(X_csr_scaled_mean, X_scaled.mean(axis=0))
    assert_array_almost_equal(X_csr_scaled_var, X_scaled.var(axis=0))

    # Check that X has not been modified (copy)
    assert X_scaled is not X
    assert X_csr_scaled is not X_csr

    X_scaled_back = scaler.inverse_transform(X_scaled)
    assert X_scaled_back is not X
    assert X_scaled_back is not X_scaled
    assert_array_almost_equal(X_scaled_back, X)

    X_csr_scaled_back = scaler_csr.inverse_transform(X_csr_scaled)
    assert X_csr_scaled_back is not X_csr
    assert X_csr_scaled_back is not X_csr_scaled
    assert_array_almost_equal(X_csr_scaled_back.toarray(), X)

    X_csc_scaled_back = scaler_csr.inverse_transform(X_csc_scaled.tocsc())
    assert X_csc_scaled_back is not X_csc
    assert X_csc_scaled_back is not X_csc_scaled
    assert_array_almost_equal(X_csc_scaled_back.toarray(), X)


@pytest.mark.parametrize("with_mean", [True, False])
@pytest.mark.parametrize("with_std", [True, False])
@pytest.mark.parametrize(
    "array_constructor", [np.asarray, sparse.csc_matrix, sparse.csr_matrix]
)
def test_scaler_n_samples_seen_with_nan(with_mean, with_std, array_constructor):
    X = np.array(
        [[0, 1, 3], [np.nan, 6, 10], [5, 4, np.nan], [8, 0, np.nan]], dtype=np.float64
    )
    X = array_constructor(X)

    if sparse.issparse(X) and with_mean:
        pytest.skip("'with_mean=True' cannot be used with sparse matrix.")

    transformer = StandardScaler(with_mean=with_mean, with_std=with_std)
    transformer.fit(X)

    assert_array_equal(transformer.n_samples_seen_, np.array([3, 4, 2]))


def _check_identity_scalers_attributes(scaler_1, scaler_2):
    assert scaler_1.mean_ is scaler_2.mean_ is None
    assert scaler_1.var_ is scaler_2.var_ is None
    assert scaler_1.scale_ is scaler_2.scale_ is None
    assert scaler_1.n_samples_seen_ == scaler_2.n_samples_seen_


def test_scaler_return_identity():
    # test that the scaler return identity when with_mean and with_std are
    # False
    X_dense = np.array([[0, 1, 3], [5, 6, 0], [8, 0, 10]], dtype=np.float64)
    X_csr = sparse.csr_matrix(X_dense)
    X_csc = X_csr.tocsc()

    transformer_dense = StandardScaler(with_mean=False, with_std=False)
    X_trans_dense = transformer_dense.fit_transform(X_dense)

    transformer_csr = clone(transformer_dense)
    X_trans_csr = transformer_csr.fit_transform(X_csr)

    transformer_csc = clone(transformer_dense)
    X_trans_csc = transformer_csc.fit_transform(X_csc)

    assert_allclose_dense_sparse(X_trans_csr, X_csr)
    assert_allclose_dense_sparse(X_trans_csc, X_csc)
    assert_allclose(X_trans_dense, X_dense)

    for trans_1, trans_2 in itertools.combinations(
        [transformer_dense, transformer_csr, transformer_csc], 2
    ):
        _check_identity_scalers_attributes(trans_1, trans_2)

    transformer_dense.partial_fit(X_dense)
    transformer_csr.partial_fit(X_csr)
    transformer_csc.partial_fit(X_csc)

    for trans_1, trans_2 in itertools.combinations(
        [transformer_dense, transformer_csr, transformer_csc], 2
    ):
        _check_identity_scalers_attributes(trans_1, trans_2)

    transformer_dense.fit(X_dense)
    transformer_csr.fit(X_csr)
    transformer_csc.fit(X_csc)

    for trans_1, trans_2 in itertools.combinations(
        [transformer_dense, transformer_csr, transformer_csc], 2
    ):
        _check_identity_scalers_attributes(trans_1, trans_2)


def test_scaler_int():
    # test that scaler converts integer input to floating
    # for both sparse and dense matrices
    rng = np.random.RandomState(42)
    X = rng.randint(20, size=(4, 5))
    X[:, 0] = 0  # first feature is always of zero
    X_csr = sparse.csr_matrix(X)
    X_csc = sparse.csc_matrix(X)

    null_transform = StandardScaler(with_mean=False, with_std=False, copy=True)
    with warnings.catch_warnings(record=True):
        X_null = null_transform.fit_transform(X_csr)
    assert_array_equal(X_null.data, X_csr.data)
    X_orig = null_transform.inverse_transform(X_null)
    assert_array_equal(X_orig.data, X_csr.data)

    with warnings.catch_warnings(record=True):
        scaler = StandardScaler(with_mean=False).fit(X)
        X_scaled = scaler.transform(X, copy=True)
    assert not np.any(np.isnan(X_scaled))

    with warnings.catch_warnings(record=True):
        scaler_csr = StandardScaler(with_mean=False).fit(X_csr)
        X_csr_scaled = scaler_csr.transform(X_csr, copy=True)
    assert not np.any(np.isnan(X_csr_scaled.data))

    with warnings.catch_warnings(record=True):
        scaler_csc = StandardScaler(with_mean=False).fit(X_csc)
        X_csc_scaled = scaler_csc.transform(X_csc, copy=True)
    assert not np.any(np.isnan(X_csc_scaled.data))

    assert_array_almost_equal(scaler.mean_, scaler_csr.mean_)
    assert_array_almost_equal(scaler.var_, scaler_csr.var_)
    assert_array_almost_equal(scaler.scale_, scaler_csr.scale_)

    assert_array_almost_equal(scaler.mean_, scaler_csc.mean_)
    assert_array_almost_equal(scaler.var_, scaler_csc.var_)
    assert_array_almost_equal(scaler.scale_, scaler_csc.scale_)

    assert_array_almost_equal(
        X_scaled.mean(axis=0), [0.0, 1.109, 1.856, 21.0, 1.559], 2
    )
    assert_array_almost_equal(X_scaled.std(axis=0), [0.0, 1.0, 1.0, 1.0, 1.0])

    X_csr_scaled_mean, X_csr_scaled_std = mean_variance_axis(
        X_csr_scaled.astype(float), 0
    )
    assert_array_almost_equal(X_csr_scaled_mean, X_scaled.mean(axis=0))
    assert_array_almost_equal(X_csr_scaled_std, X_scaled.std(axis=0))

    # Check that X has not been modified (copy)
    assert X_scaled is not X
    assert X_csr_scaled is not X_csr

    X_scaled_back = scaler.inverse_transform(X_scaled)
    assert X_scaled_back is not X
    assert X_scaled_back is not X_scaled
    assert_array_almost_equal(X_scaled_back, X)

    X_csr_scaled_back = scaler_csr.inverse_transform(X_csr_scaled)
    assert X_csr_scaled_back is not X_csr
    assert X_csr_scaled_back is not X_csr_scaled
    assert_array_almost_equal(X_csr_scaled_back.toarray(), X)

    X_csc_scaled_back = scaler_csr.inverse_transform(X_csc_scaled.tocsc())
    assert X_csc_scaled_back is not X_csc
    assert X_csc_scaled_back is not X_csc_scaled
    assert_array_almost_equal(X_csc_scaled_back.toarray(), X)


def test_scaler_without_copy():
    # Check that StandardScaler.fit does not change input
    rng = np.random.RandomState(42)
    X = rng.randn(4, 5)
    X[:, 0] = 0.0  # first feature is always of zero
    X_csr = sparse.csr_matrix(X)
    X_csc = sparse.csc_matrix(X)

    X_copy = X.copy()
    StandardScaler(copy=False).fit(X)
    assert_array_equal(X, X_copy)

    X_csr_copy = X_csr.copy()
    StandardScaler(with_mean=False, copy=False).fit(X_csr)
    assert_array_equal(X_csr.toarray(), X_csr_copy.toarray())

    X_csc_copy = X_csc.copy()
    StandardScaler(with_mean=False, copy=False).fit(X_csc)
    assert_array_equal(X_csc.toarray(), X_csc_copy.toarray())


def test_scale_sparse_with_mean_raise_exception():
    rng = np.random.RandomState(42)
    X = rng.randn(4, 5)
    X_csr = sparse.csr_matrix(X)
    X_csc = sparse.csc_matrix(X)

    # check scaling and fit with direct calls on sparse data
    with pytest.raises(ValueError):
        scale(X_csr, with_mean=True)
    with pytest.raises(ValueError):
        StandardScaler(with_mean=True).fit(X_csr)

    with pytest.raises(ValueError):
        scale(X_csc, with_mean=True)
    with pytest.raises(ValueError):
        StandardScaler(with_mean=True).fit(X_csc)

    # check transform and inverse_transform after a fit on a dense array
    scaler = StandardScaler(with_mean=True).fit(X)
    with pytest.raises(ValueError):
        scaler.transform(X_csr)
    with pytest.raises(ValueError):
        scaler.transform(X_csc)

    X_transformed_csr = sparse.csr_matrix(scaler.transform(X))
    with pytest.raises(ValueError):
        scaler.inverse_transform(X_transformed_csr)

    X_transformed_csc = sparse.csc_matrix(scaler.transform(X))
    with pytest.raises(ValueError):
        scaler.inverse_transform(X_transformed_csc)


def test_scale_input_finiteness_validation():
    # Check if non finite inputs raise ValueError
    X = [[np.inf, 5, 6, 7, 8]]
    with pytest.raises(
        ValueError, match="Input contains infinity or a value too large"
    ):
        scale(X)


def test_robust_scaler_error_sparse():
    X_sparse = sparse.rand(1000, 10)
    scaler = RobustScaler(with_centering=True)
    err_msg = "Cannot center sparse matrices"
    with pytest.raises(ValueError, match=err_msg):
        scaler.fit(X_sparse)


@pytest.mark.parametrize("with_centering", [True, False])
@pytest.mark.parametrize("with_scaling", [True, False])
@pytest.mark.parametrize("X", [np.random.randn(10, 3), sparse.rand(10, 3, density=0.5)])
def test_robust_scaler_attributes(X, with_centering, with_scaling):
    # check consistent type of attributes
    if with_centering and sparse.issparse(X):
        pytest.skip("RobustScaler cannot center sparse matrix")

    scaler = RobustScaler(with_centering=with_centering, with_scaling=with_scaling)
    scaler.fit(X)

    if with_centering:
        assert isinstance(scaler.center_, np.ndarray)
    else:
        assert scaler.center_ is None
    if with_scaling:
        assert isinstance(scaler.scale_, np.ndarray)
    else:
        assert scaler.scale_ is None


def test_robust_scaler_col_zero_sparse():
    # check that the scaler is working when there is not data materialized in a
    # column of a sparse matrix
    X = np.random.randn(10, 5)
    X[:, 0] = 0
    X = sparse.csr_matrix(X)

    scaler = RobustScaler(with_centering=False)
    scaler.fit(X)
    assert scaler.scale_[0] == pytest.approx(1)

    X_trans = scaler.transform(X)
    assert_allclose(X[:, 0].toarray(), X_trans[:, 0].toarray())


def test_robust_scaler_2d_arrays():
    # Test robust scaling of 2d array along first axis
    rng = np.random.RandomState(0)
    X = rng.randn(4, 5)
    X[:, 0] = 0.0  # first feature is always of zero

    scaler = RobustScaler()
    X_scaled = scaler.fit(X).transform(X)

    assert_array_almost_equal(np.median(X_scaled, axis=0), 5 * [0.0])
    assert_array_almost_equal(X_scaled.std(axis=0)[0], 0)


@pytest.mark.parametrize("density", [0, 0.05, 0.1, 0.5, 1])
@pytest.mark.parametrize("strictly_signed", ["positive", "negative", "zeros", None])
def test_robust_scaler_equivalence_dense_sparse(density, strictly_signed):
    # Check the equivalence of the fitting with dense and sparse matrices
    X_sparse = sparse.rand(1000, 5, density=density).tocsc()
    if strictly_signed == "positive":
        X_sparse.data = np.abs(X_sparse.data)
    elif strictly_signed == "negative":
        X_sparse.data = -np.abs(X_sparse.data)
    elif strictly_signed == "zeros":
        X_sparse.data = np.zeros(X_sparse.data.shape, dtype=np.float64)
    X_dense = X_sparse.toarray()

    scaler_sparse = RobustScaler(with_centering=False)
    scaler_dense = RobustScaler(with_centering=False)

    scaler_sparse.fit(X_sparse)
    scaler_dense.fit(X_dense)

    assert_allclose(scaler_sparse.scale_, scaler_dense.scale_)


def test_robust_scaler_transform_one_row_csr():
    # Check RobustScaler on transforming csr matrix with one row
    rng = np.random.RandomState(0)
    X = rng.randn(4, 5)
    single_row = np.array([[0.1, 1.0, 2.0, 0.0, -1.0]])
    scaler = RobustScaler(with_centering=False)
    scaler = scaler.fit(X)
    row_trans = scaler.transform(sparse.csr_matrix(single_row))
    row_expected = single_row / scaler.scale_
    assert_array_almost_equal(row_trans.toarray(), row_expected)
    row_scaled_back = scaler.inverse_transform(row_trans)
    assert_array_almost_equal(single_row, row_scaled_back.toarray())


def test_robust_scaler_iris():
    X = iris.data
    scaler = RobustScaler()
    X_trans = scaler.fit_transform(X)
    assert_array_almost_equal(np.median(X_trans, axis=0), 0)
    X_trans_inv = scaler.inverse_transform(X_trans)
    assert_array_almost_equal(X, X_trans_inv)
    q = np.percentile(X_trans, q=(25, 75), axis=0)
    iqr = q[1] - q[0]
    assert_array_almost_equal(iqr, 1)


def test_robust_scaler_iris_quantiles():
    X = iris.data
    scaler = RobustScaler(quantile_range=(10, 90))
    X_trans = scaler.fit_transform(X)
    assert_array_almost_equal(np.median(X_trans, axis=0), 0)
    X_trans_inv = scaler.inverse_transform(X_trans)
    assert_array_almost_equal(X, X_trans_inv)
    q = np.percentile(X_trans, q=(10, 90), axis=0)
    q_range = q[1] - q[0]
    assert_array_almost_equal(q_range, 1)


def test_quantile_transform_iris():
    X = iris.data
    # uniform output distribution
    transformer = QuantileTransformer(n_quantiles=30)
    X_trans = transformer.fit_transform(X)
    X_trans_inv = transformer.inverse_transform(X_trans)
    assert_array_almost_equal(X, X_trans_inv)
    # normal output distribution
    transformer = QuantileTransformer(n_quantiles=30, output_distribution="normal")
    X_trans = transformer.fit_transform(X)
    X_trans_inv = transformer.inverse_transform(X_trans)
    assert_array_almost_equal(X, X_trans_inv)
    # make sure it is possible to take the inverse of a sparse matrix
    # which contain negative value; this is the case in the iris dataset
    X_sparse = sparse.csc_matrix(X)
    X_sparse_tran = transformer.fit_transform(X_sparse)
    X_sparse_tran_inv = transformer.inverse_transform(X_sparse_tran)
    assert_array_almost_equal(X_sparse.A, X_sparse_tran_inv.A)


def test_quantile_transform_check_error():
    X = np.transpose(
        [
            [0, 25, 50, 0, 0, 0, 75, 0, 0, 100],
            [2, 4, 0, 0, 6, 8, 0, 10, 0, 0],
            [0, 0, 2.6, 4.1, 0, 0, 2.3, 0, 9.5, 0.1],
        ]
    )
    X = sparse.csc_matrix(X)
    X_neg = np.transpose(
        [
            [0, 25, 50, 0, 0, 0, 75, 0, 0, 100],
            [-2, 4, 0, 0, 6, 8, 0, 10, 0, 0],
            [0, 0, 2.6, 4.1, 0, 0, 2.3, 0, 9.5, 0.1],
        ]
    )
    X_neg = sparse.csc_matrix(X_neg)

    err_msg = "Invalid value for 'n_quantiles': 0."
    with pytest.raises(ValueError, match=err_msg):
        QuantileTransformer(n_quantiles=0).fit(X)
    err_msg = "Invalid value for 'subsample': 0."
    with pytest.raises(ValueError, match=err_msg):
        QuantileTransformer(subsample=0).fit(X)
    err_msg = (
        "The number of quantiles cannot be greater than "
        "the number of samples used. Got 1000 quantiles "
        "and 10 samples."
    )
    with pytest.raises(ValueError, match=err_msg):
        QuantileTransformer(subsample=10).fit(X)

    transformer = QuantileTransformer(n_quantiles=10)
    err_msg = "QuantileTransformer only accepts non-negative sparse matrices."
    with pytest.raises(ValueError, match=err_msg):
        transformer.fit(X_neg)
    transformer.fit(X)
    err_msg = "QuantileTransformer only accepts non-negative sparse matrices."
    with pytest.raises(ValueError, match=err_msg):
        transformer.transform(X_neg)

    X_bad_feat = np.transpose(
        [[0, 25, 50, 0, 0, 0, 75, 0, 0, 100], [0, 0, 2.6, 4.1, 0, 0, 2.3, 0, 9.5, 0.1]]
    )
    err_msg = (
        "X has 2 features, but QuantileTransformer is expecting 3 features as input."
    )
    with pytest.raises(ValueError, match=err_msg):
        transformer.inverse_transform(X_bad_feat)

    transformer = QuantileTransformer(n_quantiles=10, output_distribution="rnd")
    # check that an error is raised at fit time
    err_msg = (
        "'output_distribution' has to be either 'normal' or "
        "'uniform'. Got 'rnd' instead."
    )
    with pytest.raises(ValueError, match=err_msg):
        transformer.fit(X)
    # check that an error is raised at transform time
    transformer.output_distribution = "uniform"
    transformer.fit(X)
    X_tran = transformer.transform(X)
    transformer.output_distribution = "rnd"
    err_msg = (
        "'output_distribution' has to be either 'normal' or 'uniform'."
        " Got 'rnd' instead."
    )
    with pytest.raises(ValueError, match=err_msg):
        transformer.transform(X)
    # check that an error is raised at inverse_transform time
    err_msg = (
        "'output_distribution' has to be either 'normal' or 'uniform'."
        " Got 'rnd' instead."
    )
    with pytest.raises(ValueError, match=err_msg):
        transformer.inverse_transform(X_tran)
    # check that an error is raised if input is scalar
    with pytest.raises(ValueError, match="Expected 2D array, got scalar array instead"):
        transformer.transform(10)
    # check that a warning is raised is n_quantiles > n_samples
    transformer = QuantileTransformer(n_quantiles=100)
    warn_msg = "n_quantiles is set to n_samples"
    with pytest.warns(UserWarning, match=warn_msg) as record:
        transformer.fit(X)
    assert len(record) == 1
    assert transformer.n_quantiles_ == X.shape[0]


def test_quantile_transform_sparse_ignore_zeros():
    X = np.array([[0, 1], [0, 0], [0, 2], [0, 2], [0, 1]])
    X_sparse = sparse.csc_matrix(X)
    transformer = QuantileTransformer(ignore_implicit_zeros=True, n_quantiles=5)

    # dense case -> warning raise
    warning_message = (
        "'ignore_implicit_zeros' takes effect"
        " only with sparse matrix. This parameter has no"
        " effect."
    )
    with pytest.warns(UserWarning, match=warning_message):
        transformer.fit(X)

    X_expected = np.array([[0, 0], [0, 0], [0, 1], [0, 1], [0, 0]])
    X_trans = transformer.fit_transform(X_sparse)
    assert_almost_equal(X_expected, X_trans.A)

    # consider the case where sparse entries are missing values and user-given
    # zeros are to be considered
    X_data = np.array([0, 0, 1, 0, 2, 2, 1, 0, 1, 2, 0])
    X_col = np.array([0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    X_row = np.array([0, 4, 0, 1, 2, 3, 4, 5, 6, 7, 8])
    X_sparse = sparse.csc_matrix((X_data, (X_row, X_col)))
    X_trans = transformer.fit_transform(X_sparse)
    X_expected = np.array(
        [
            [0.0, 0.5],
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 0.5],
            [0.0, 0.0],
            [0.0, 0.5],
            [0.0, 1.0],
            [0.0, 0.0],
        ]
    )
    assert_almost_equal(X_expected, X_trans.A)

    transformer = QuantileTransformer(ignore_implicit_zeros=True, n_quantiles=5)
    X_data = np.array([-1, -1, 1, 0, 0, 0, 1, -1, 1])
    X_col = np.array([0, 0, 1, 1, 1, 1, 1, 1, 1])
    X_row = np.array([0, 4, 0, 1, 2, 3, 4, 5, 6])
    X_sparse = sparse.csc_matrix((X_data, (X_row, X_col)))
    X_trans = transformer.fit_transform(X_sparse)
    X_expected = np.array(
        [[0, 1], [0, 0.375], [0, 0.375], [0, 0.375], [0, 1], [0, 0], [0, 1]]
    )
    assert_almost_equal(X_expected, X_trans.A)
    assert_almost_equal(X_sparse.A, transformer.inverse_transform(X_trans).A)

    # check in conjunction with subsampling
    transformer = QuantileTransformer(
        ignore_implicit_zeros=True, n_quantiles=5, subsample=8, random_state=0
    )
    X_trans = transformer.fit_transform(X_sparse)
    assert_almost_equal(X_expected, X_trans.A)
    assert_almost_equal(X_sparse.A, transformer.inverse_transform(X_trans).A)


def test_quantile_transform_dense_toy():
    X = np.array(
        [[0, 2, 2.6], [25, 4, 4.1], [50, 6, 2.3], [75, 8, 9.5], [100, 10, 0.1]]
    )

    transformer = QuantileTransformer(n_quantiles=5)
    transformer.fit(X)

    # using the a uniform output, each entry of X should be map between 0 and 1
    # and equally spaced
    X_trans = transformer.fit_transform(X)
    X_expected = np.tile(np.linspace(0, 1, num=5), (3, 1)).T
    assert_almost_equal(np.sort(X_trans, axis=0), X_expected)

    X_test = np.array(
        [
            [-1, 1, 0],
            [101, 11, 10],
        ]
    )
    X_expected = np.array(
        [
            [0, 0, 0],
            [1, 1, 1],
        ]
    )
    assert_array_almost_equal(transformer.transform(X_test), X_expected)

    X_trans_inv = transformer.inverse_transform(X_trans)
    assert_array_almost_equal(X, X_trans_inv)


def test_quantile_transform_subsampling():
    # Test that subsampling the input yield to a consistent results We check
    # that the computed quantiles are almost mapped to a [0, 1] vector where
    # values are equally spaced. The infinite norm is checked to be smaller
    # than a given threshold. This is repeated 5 times.

    # dense support
    n_samples = 1000000
    n_quantiles = 1000
    X = np.sort(np.random.sample((n_samples, 1)), axis=0)
    ROUND = 5
    inf_norm_arr = []
    for random_state in range(ROUND):
        transformer = QuantileTransformer(
            random_state=random_state,
            n_quantiles=n_quantiles,
            subsample=n_samples // 10,
        )
        transformer.fit(X)
        diff = np.linspace(0, 1, n_quantiles) - np.ravel(transformer.quantiles_)
        inf_norm = np.max(np.abs(diff))
        assert inf_norm < 1e-2
        inf_norm_arr.append(inf_norm)
    # each random subsampling yield a unique approximation to the expected
    # linspace CDF
    assert len(np.unique(inf_norm_arr)) == len(inf_norm_arr)

    # sparse support

    X = sparse.rand(n_samples, 1, density=0.99, format="csc", random_state=0)
    inf_norm_arr = []
    for random_state in range(ROUND):
        transformer = QuantileTransformer(
            random_state=random_state,
            n_quantiles=n_quantiles,
            subsample=n_samples // 10,
        )
        transformer.fit(X)
        diff = np.linspace(0, 1, n_quantiles) - np.ravel(transformer.quantiles_)
        inf_norm = np.max(np.abs(diff))
        assert inf_norm < 1e-1
        inf_norm_arr.append(inf_norm)
    # each random subsampling yield a unique approximation to the expected
    # linspace CDF
    assert len(np.unique(inf_norm_arr)) == len(inf_norm_arr)


def test_quantile_transform_sparse_toy():
    X = np.array(
        [
            [0.0, 2.0, 0.0],
            [25.0, 4.0, 0.0],
            [50.0, 0.0, 2.6],
            [0.0, 0.0, 4.1],
            [0.0, 6.0, 0.0],
            [0.0, 8.0, 0.0],
            [75.0, 0.0, 2.3],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 9.5],
            [100.0, 0.0, 0.1],
        ]
    )

    X = sparse.csc_matrix(X)

    transformer = QuantileTransformer(n_quantiles=10)
    transformer.fit(X)

    X_trans = transformer.fit_transform(X)
    assert_array_almost_equal(np.min(X_trans.toarray(), axis=0), 0.0)
    assert_array_almost_equal(np.max(X_trans.toarray(), axis=0), 1.0)

    X_trans_inv = transformer.inverse_transform(X_trans)
    assert_array_almost_equal(X.toarray(), X_trans_inv.toarray())

    transformer_dense = QuantileTransformer(n_quantiles=10).fit(X.toarray())

    X_trans = transformer_dense.transform(X)
    assert_array_almost_equal(np.min(X_trans.toarray(), axis=0), 0.0)
    assert_array_almost_equal(np.max(X_trans.toarray(), axis=0), 1.0)

    X_trans_inv = transformer_dense.inverse_transform(X_trans)
    assert_array_almost_equal(X.toarray(), X_trans_inv.toarray())


def test_quantile_transform_axis1():
    X = np.array([[0, 25, 50, 75, 100], [2, 4, 6, 8, 10], [2.6, 4.1, 2.3, 9.5, 0.1]])

    X_trans_a0 = quantile_transform(X.T, axis=0, n_quantiles=5)
    X_trans_a1 = quantile_transform(X, axis=1, n_quantiles=5)
    assert_array_almost_equal(X_trans_a0, X_trans_a1.T)


def test_quantile_transform_bounds():
    # Lower and upper bounds are manually mapped. We checked that in the case
    # of a constant feature and binary feature, the bounds are properly mapped.
    X_dense = np.array([[0, 0], [0, 0], [1, 0]])
    X_sparse = sparse.csc_matrix(X_dense)

    # check sparse and dense are consistent
    X_trans = QuantileTransformer(n_quantiles=3, random_state=0).fit_transform(X_dense)
    assert_array_almost_equal(X_trans, X_dense)
    X_trans_sp = QuantileTransformer(n_quantiles=3, random_state=0).fit_transform(
        X_sparse
    )
    assert_array_almost_equal(X_trans_sp.A, X_dense)
    assert_array_almost_equal(X_trans, X_trans_sp.A)

    # check the consistency of the bounds by learning on 1 matrix
    # and transforming another
    X = np.array([[0, 1], [0, 0.5], [1, 0]])
    X1 = np.array([[0, 0.1], [0, 0.5], [1, 0.1]])
    transformer = QuantileTransformer(n_quantiles=3).fit(X)
    X_trans = transformer.transform(X1)
    assert_array_almost_equal(X_trans, X1)

    # check that values outside of the range learned will be mapped properly.
    X = np.random.random((1000, 1))
    transformer = QuantileTransformer()
    transformer.fit(X)
    assert transformer.transform([[-10]]) == transformer.transform([[np.min(X)]])
    assert transformer.transform([[10]]) == transformer.transform([[np.max(X)]])
    assert transformer.inverse_transform([[-10]]) == transformer.inverse_transform(
        [[np.min(transformer.references_)]]
    )
    assert transformer.inverse_transform([[10]]) == transformer.inverse_transform(
        [[np.max(transformer.references_)]]
    )


def test_quantile_transform_and_inverse():
    X_1 = iris.data
    X_2 = np.array([[0.0], [BOUNDS_THRESHOLD / 10], [1.5], [2], [3], [3], [4]])
    for X in [X_1, X_2]:
        transformer = QuantileTransformer(n_quantiles=1000, random_state=0)
        X_trans = transformer.fit_transform(X)
        X_trans_inv = transformer.inverse_transform(X_trans)
        assert_array_almost_equal(X, X_trans_inv, decimal=9)


def test_quantile_transform_nan():
    X = np.array([[np.nan, 0, 0, 1], [np.nan, np.nan, 0, 0.5], [np.nan, 1, 1, 0]])

    transformer = QuantileTransformer(n_quantiles=10, random_state=42)
    transformer.fit_transform(X)

    # check that the quantile of the first column is all NaN
    assert np.isnan(transformer.quantiles_[:, 0]).all()
    # all other column should not contain NaN
    assert not np.isnan(transformer.quantiles_[:, 1:]).any()


@pytest.mark.parametrize("array_type", ["array", "sparse"])
def test_quantile_transformer_sorted_quantiles(array_type):
    # Non-regression test for:
    # https://github.com/scikit-learn/scikit-learn/issues/15733
    # Taken from upstream bug report:
    # https://github.com/numpy/numpy/issues/14685
    X = np.array([0, 1, 1, 2, 2, 3, 3, 4, 5, 5, 1, 1, 9, 9, 9, 8, 8, 7] * 10)
    X = 0.1 * X.reshape(-1, 1)
    X = _convert_container(X, array_type)

    n_quantiles = 100
    qt = QuantileTransformer(n_quantiles=n_quantiles).fit(X)

    # Check that the estimated quantile thresholds are monotically
    # increasing:
    quantiles = qt.quantiles_[:, 0]
    assert len(quantiles) == 100
    assert all(np.diff(quantiles) >= 0)


def test_robust_scaler_invalid_range():
    for range_ in [
        (-1, 90),
        (-2, -3),
        (10, 101),
        (100.5, 101),
        (90, 50),
    ]:
        scaler = RobustScaler(quantile_range=range_)

        with pytest.raises(ValueError, match=r"Invalid quantile range: \("):
            scaler.fit(iris.data)


def test_scale_function_without_centering():
    rng = np.random.RandomState(42)
    X = rng.randn(4, 5)
    X[:, 0] = 0.0  # first feature is always of zero
    X_csr = sparse.csr_matrix(X)

    X_scaled = scale(X, with_mean=False)
    assert not np.any(np.isnan(X_scaled))

    X_csr_scaled = scale(X_csr, with_mean=False)
    assert not np.any(np.isnan(X_csr_scaled.data))

    # test csc has same outcome
    X_csc_scaled = scale(X_csr.tocsc(), with_mean=False)
    assert_array_almost_equal(X_scaled, X_csc_scaled.toarray())

    # raises value error on axis != 0
    with pytest.raises(ValueError):
        scale(X_csr, with_mean=False, axis=1)

    assert_array_almost_equal(
        X_scaled.mean(axis=0), [0.0, -0.01, 2.24, -0.35, -0.78], 2
    )
    assert_array_almost_equal(X_scaled.std(axis=0), [0.0, 1.0, 1.0, 1.0, 1.0])
    # Check that X has not been copied
    assert X_scaled is not X

    X_csr_scaled_mean, X_csr_scaled_std = mean_variance_axis(X_csr_scaled, 0)
    assert_array_almost_equal(X_csr_scaled_mean, X_scaled.mean(axis=0))
    assert_array_almost_equal(X_csr_scaled_std, X_scaled.std(axis=0))

    # null scale
    X_csr_scaled = scale(X_csr, with_mean=False, with_std=False, copy=True)
    assert_array_almost_equal(X_csr.toarray(), X_csr_scaled.toarray())


def test_robust_scale_axis1():
    X = iris.data
    X_trans = robust_scale(X, axis=1)
    assert_array_almost_equal(np.median(X_trans, axis=1), 0)
    q = np.percentile(X_trans, q=(25, 75), axis=1)
    iqr = q[1] - q[0]
    assert_array_almost_equal(iqr, 1)


def test_robust_scale_1d_array():
    X = iris.data[:, 1]
    X_trans = robust_scale(X)
    assert_array_almost_equal(np.median(X_trans), 0)
    q = np.percentile(X_trans, q=(25, 75))
    iqr = q[1] - q[0]
    assert_array_almost_equal(iqr, 1)


def test_robust_scaler_zero_variance_features():
    # Check RobustScaler on toy data with zero variance features
    X = [[0.0, 1.0, +0.5], [0.0, 1.0, -0.1], [0.0, 1.0, +1.1]]

    scaler = RobustScaler()
    X_trans = scaler.fit_transform(X)

    # NOTE: for such a small sample size, what we expect in the third column
    # depends HEAVILY on the method used to calculate quantiles. The values
    # here were calculated to fit the quantiles produces by np.percentile
    # using numpy 1.9 Calculating quantiles with
    # scipy.stats.mstats.scoreatquantile or scipy.stats.mstats.mquantiles
    # would yield very different results!
    X_expected = [[0.0, 0.0, +0.0], [0.0, 0.0, -1.0], [0.0, 0.0, +1.0]]
    assert_array_almost_equal(X_trans, X_expected)
    X_trans_inv = scaler.inverse_transform(X_trans)
    assert_array_almost_equal(X, X_trans_inv)

    # make sure new data gets transformed correctly
    X_new = [[+0.0, 2.0, 0.5], [-1.0, 1.0, 0.0], [+0.0, 1.0, 1.5]]
    X_trans_new = scaler.transform(X_new)
    X_expected_new = [[+0.0, 1.0, +0.0], [-1.0, 0.0, -0.83333], [+0.0, 0.0, +1.66667]]
    assert_array_almost_equal(X_trans_new, X_expected_new, decimal=3)


def test_robust_scaler_unit_variance():
    # Check RobustScaler with unit_variance=True on standard normal data with
    # outliers
    rng = np.random.RandomState(42)
    X = rng.randn(1000000, 1)
    X_with_outliers = np.vstack([X, np.ones((100, 1)) * 100, np.ones((100, 1)) * -100])

    quantile_range = (1, 99)
    robust_scaler = RobustScaler(quantile_range=quantile_range, unit_variance=True).fit(
        X_with_outliers
    )
    X_trans = robust_scaler.transform(X)

    assert robust_scaler.center_ == pytest.approx(0, abs=1e-3)
    assert robust_scaler.scale_ == pytest.approx(1, abs=1e-2)
    assert X_trans.std() == pytest.approx(1, abs=1e-2)


def test_maxabs_scaler_zero_variance_features():
    # Check MaxAbsScaler on toy data with zero variance features
    X = [[0.0, 1.0, +0.5], [0.0, 1.0, -0.3], [0.0, 1.0, +1.5], [0.0, 0.0, +0.0]]

    scaler = MaxAbsScaler()
    X_trans = scaler.fit_transform(X)
    X_expected = [
        [0.0, 1.0, 1.0 / 3.0],
        [0.0, 1.0, -0.2],
        [0.0, 1.0, 1.0],
        [0.0, 0.0, 0.0],
    ]
    assert_array_almost_equal(X_trans, X_expected)
    X_trans_inv = scaler.inverse_transform(X_trans)
    assert_array_almost_equal(X, X_trans_inv)

    # make sure new data gets transformed correctly
    X_new = [[+0.0, 2.0, 0.5], [-1.0, 1.0, 0.0], [+0.0, 1.0, 1.5]]
    X_trans_new = scaler.transform(X_new)
    X_expected_new = [[+0.0, 2.0, 1.0 / 3.0], [-1.0, 1.0, 0.0], [+0.0, 1.0, 1.0]]

    assert_array_almost_equal(X_trans_new, X_expected_new, decimal=2)

    # function interface
    X_trans = maxabs_scale(X)
    assert_array_almost_equal(X_trans, X_expected)

    # sparse data
    X_csr = sparse.csr_matrix(X)
    X_csc = sparse.csc_matrix(X)
    X_trans_csr = scaler.fit_transform(X_csr)
    X_trans_csc = scaler.fit_transform(X_csc)
    X_expected = [
        [0.0, 1.0, 1.0 / 3.0],
        [0.0, 1.0, -0.2],
        [0.0, 1.0, 1.0],
        [0.0, 0.0, 0.0],
    ]
    assert_array_almost_equal(X_trans_csr.A, X_expected)
    assert_array_almost_equal(X_trans_csc.A, X_expected)
    X_trans_csr_inv = scaler.inverse_transform(X_trans_csr)
    X_trans_csc_inv = scaler.inverse_transform(X_trans_csc)
    assert_array_almost_equal(X, X_trans_csr_inv.A)
    assert_array_almost_equal(X, X_trans_csc_inv.A)


def test_maxabs_scaler_large_negative_value():
    # Check MaxAbsScaler on toy data with a large negative value
    X = [
        [0.0, 1.0, +0.5, -1.0],
        [0.0, 1.0, -0.3, -0.5],
        [0.0, 1.0, -100.0, 0.0],
        [0.0, 0.0, +0.0, -2.0],
    ]

    scaler = MaxAbsScaler()
    X_trans = scaler.fit_transform(X)
    X_expected = [
        [0.0, 1.0, 0.005, -0.5],
        [0.0, 1.0, -0.003, -0.25],
        [0.0, 1.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, -1.0],
    ]
    assert_array_almost_equal(X_trans, X_expected)


def test_maxabs_scaler_transform_one_row_csr():
    # Check MaxAbsScaler on transforming csr matrix with one row
    X = sparse.csr_matrix([[0.5, 1.0, 1.0]])
    scaler = MaxAbsScaler()
    scaler = scaler.fit(X)
    X_trans = scaler.transform(X)
    X_expected = sparse.csr_matrix([[1.0, 1.0, 1.0]])
    assert_array_almost_equal(X_trans.toarray(), X_expected.toarray())
    X_scaled_back = scaler.inverse_transform(X_trans)
    assert_array_almost_equal(X.toarray(), X_scaled_back.toarray())


def test_maxabs_scaler_1d():
    # Test scaling of dataset along single axis
    for X in [X_1row, X_1col, X_list_1row, X_list_1row]:

        scaler = MaxAbsScaler(copy=True)
        X_scaled = scaler.fit(X).transform(X)

        if isinstance(X, list):
            X = np.array(X)  # cast only after scaling done

        if _check_dim_1axis(X) == 1:
            assert_array_almost_equal(np.abs(X_scaled.max(axis=0)), np.ones(n_features))
        else:
            assert_array_almost_equal(np.abs(X_scaled.max(axis=0)), 1.0)
        assert scaler.n_samples_seen_ == X.shape[0]

        # check inverse transform
        X_scaled_back = scaler.inverse_transform(X_scaled)
        assert_array_almost_equal(X_scaled_back, X)

    # Constant feature
    X = np.ones((5, 1))
    scaler = MaxAbsScaler()
    X_scaled = scaler.fit(X).transform(X)
    assert_array_almost_equal(np.abs(X_scaled.max(axis=0)), 1.0)
    assert scaler.n_samples_seen_ == X.shape[0]

    # function interface
    X_1d = X_1row.ravel()
    max_abs = np.abs(X_1d).max()
    assert_array_almost_equal(X_1d / max_abs, maxabs_scale(X_1d, copy=True))


def test_maxabs_scaler_partial_fit():
    # Test if partial_fit run over many batches of size 1 and 50
    # gives the same results as fit
    X = X_2d[:100, :]
    n = X.shape[0]

    for chunk_size in [1, 2, 50, n, n + 42]:
        # Test mean at the end of the process
        scaler_batch = MaxAbsScaler().fit(X)

        scaler_incr = MaxAbsScaler()
        scaler_incr_csr = MaxAbsScaler()
        scaler_incr_csc = MaxAbsScaler()
        for batch in gen_batches(n, chunk_size):
            scaler_incr = scaler_incr.partial_fit(X[batch])
            X_csr = sparse.csr_matrix(X[batch])
            scaler_incr_csr = scaler_incr_csr.partial_fit(X_csr)
            X_csc = sparse.csc_matrix(X[batch])
            scaler_incr_csc = scaler_incr_csc.partial_fit(X_csc)

        assert_array_almost_equal(scaler_batch.max_abs_, scaler_incr.max_abs_)
        assert_array_almost_equal(scaler_batch.max_abs_, scaler_incr_csr.max_abs_)
        assert_array_almost_equal(scaler_batch.max_abs_, scaler_incr_csc.max_abs_)
        assert scaler_batch.n_samples_seen_ == scaler_incr.n_samples_seen_
        assert scaler_batch.n_samples_seen_ == scaler_incr_csr.n_samples_seen_
        assert scaler_batch.n_samples_seen_ == scaler_incr_csc.n_samples_seen_
        assert_array_almost_equal(scaler_batch.scale_, scaler_incr.scale_)
        assert_array_almost_equal(scaler_batch.scale_, scaler_incr_csr.scale_)
        assert_array_almost_equal(scaler_batch.scale_, scaler_incr_csc.scale_)
        assert_array_almost_equal(scaler_batch.transform(X), scaler_incr.transform(X))

        # Test std after 1 step
        batch0 = slice(0, chunk_size)
        scaler_batch = MaxAbsScaler().fit(X[batch0])
        scaler_incr = MaxAbsScaler().partial_fit(X[batch0])

        assert_array_almost_equal(scaler_batch.max_abs_, scaler_incr.max_abs_)
        assert scaler_batch.n_samples_seen_ == scaler_incr.n_samples_seen_
        assert_array_almost_equal(scaler_batch.scale_, scaler_incr.scale_)
        assert_array_almost_equal(scaler_batch.transform(X), scaler_incr.transform(X))

        # Test std until the end of partial fits, and
        scaler_batch = MaxAbsScaler().fit(X)
        scaler_incr = MaxAbsScaler()  # Clean estimator
        for i, batch in enumerate(gen_batches(n, chunk_size)):
            scaler_incr = scaler_incr.partial_fit(X[batch])
            assert_correct_incr(
                i,
                batch_start=batch.start,
                batch_stop=batch.stop,
                n=n,
                chunk_size=chunk_size,
                n_samples_seen=scaler_incr.n_samples_seen_,
            )


def test_normalizer_l1():
    rng = np.random.RandomState(0)
    X_dense = rng.randn(4, 5)
    X_sparse_unpruned = sparse.csr_matrix(X_dense)

    # set the row number 3 to zero
    X_dense[3, :] = 0.0

    # set the row number 3 to zero without pruning (can happen in real life)
    indptr_3 = X_sparse_unpruned.indptr[3]
    indptr_4 = X_sparse_unpruned.indptr[4]
    X_sparse_unpruned.data[indptr_3:indptr_4] = 0.0

    # build the pruned variant using the regular constructor
    X_sparse_pruned = sparse.csr_matrix(X_dense)

    # check inputs that support the no-copy optim
    for X in (X_dense, X_sparse_pruned, X_sparse_unpruned):

        normalizer = Normalizer(norm="l1", copy=True)
        X_norm = normalizer.transform(X)
        assert X_norm is not X
        X_norm1 = toarray(X_norm)

        normalizer = Normalizer(norm="l1", copy=False)
        X_norm = normalizer.transform(X)
        assert X_norm is X
        X_norm2 = toarray(X_norm)

        for X_norm in (X_norm1, X_norm2):
            row_sums = np.abs(X_norm).sum(axis=1)
            for i in range(3):
                assert_almost_equal(row_sums[i], 1.0)
            assert_almost_equal(row_sums[3], 0.0)

    # check input for which copy=False won't prevent a copy
    for init in (sparse.coo_matrix, sparse.csc_matrix, sparse.lil_matrix):
        X = init(X_dense)
        X_norm = normalizer = Normalizer(norm="l2", copy=False).transform(X)

        assert X_norm is not X
        assert isinstance(X_norm, sparse.csr_matrix)

        X_norm = toarray(X_norm)
        for i in range(3):
            assert_almost_equal(row_sums[i], 1.0)
        assert_almost_equal(la.norm(X_norm[3]), 0.0)


def test_normalizer_l2():
    rng = np.random.RandomState(0)
    X_dense = rng.randn(4, 5)
    X_sparse_unpruned = sparse.csr_matrix(X_dense)

    # set the row number 3 to zero
    X_dense[3, :] = 0.0

    # set the row number 3 to zero without pruning (can happen in real life)
    indptr_3 = X_sparse_unpruned.indptr[3]
    indptr_4 = X_sparse_unpruned.indptr[4]
    X_sparse_unpruned.data[indptr_3:indptr_4] = 0.0

    # build the pruned variant using the regular constructor
    X_sparse_pruned = sparse.csr_matrix(X_dense)

    # check inputs that support the no-copy optim
    for X in (X_dense, X_sparse_pruned, X_sparse_unpruned):

        normalizer = Normalizer(norm="l2", copy=True)
        X_norm1 = normalizer.transform(X)
        assert X_norm1 is not X
        X_norm1 = toarray(X_norm1)

        normalizer = Normalizer(norm="l2", copy=False)
        X_norm2 = normalizer.transform(X)
        assert X_norm2 is X
        X_norm2 = toarray(X_norm2)

        for X_norm in (X_norm1, X_norm2):
            for i in range(3):
                assert_almost_equal(la.norm(X_norm[i]), 1.0)
            assert_almost_equal(la.norm(X_norm[3]), 0.0)

    # check input for which copy=False won't prevent a copy
    for init in (sparse.coo_matrix, sparse.csc_matrix, sparse.lil_matrix):
        X = init(X_dense)
        X_norm = normalizer = Normalizer(norm="l2", copy=False).transform(X)

        assert X_norm is not X
        assert isinstance(X_norm, sparse.csr_matrix)

        X_norm = toarray(X_norm)
        for i in range(3):
            assert_almost_equal(la.norm(X_norm[i]), 1.0)
        assert_almost_equal(la.norm(X_norm[3]), 0.0)


def test_normalizer_max():
    rng = np.random.RandomState(0)
    X_dense = rng.randn(4, 5)
    X_sparse_unpruned = sparse.csr_matrix(X_dense)

    # set the row number 3 to zero
    X_dense[3, :] = 0.0

    # set the row number 3 to zero without pruning (can happen in real life)
    indptr_3 = X_sparse_unpruned.indptr[3]
    indptr_4 = X_sparse_unpruned.indptr[4]
    X_sparse_unpruned.data[indptr_3:indptr_4] = 0.0

    # build the pruned variant using the regular constructor
    X_sparse_pruned = sparse.csr_matrix(X_dense)

    # check inputs that support the no-copy optim
    for X in (X_dense, X_sparse_pruned, X_sparse_unpruned):

        normalizer = Normalizer(norm="max", copy=True)
        X_norm1 = normalizer.transform(X)
        assert X_norm1 is not X
        X_norm1 = toarray(X_norm1)

        normalizer = Normalizer(norm="max", copy=False)
        X_norm2 = normalizer.transform(X)
        assert X_norm2 is X
        X_norm2 = toarray(X_norm2)

        for X_norm in (X_norm1, X_norm2):
            row_maxs = abs(X_norm).max(axis=1)
            for i in range(3):
                assert_almost_equal(row_maxs[i], 1.0)
            assert_almost_equal(row_maxs[3], 0.0)

    # check input for which copy=False won't prevent a copy
    for init in (sparse.coo_matrix, sparse.csc_matrix, sparse.lil_matrix):
        X = init(X_dense)
        X_norm = normalizer = Normalizer(norm="l2", copy=False).transform(X)

        assert X_norm is not X
        assert isinstance(X_norm, sparse.csr_matrix)

        X_norm = toarray(X_norm)
        for i in range(3):
            assert_almost_equal(row_maxs[i], 1.0)
        assert_almost_equal(la.norm(X_norm[3]), 0.0)


def test_normalizer_max_sign():
    # check that we normalize by a positive number even for negative data
    rng = np.random.RandomState(0)
    X_dense = rng.randn(4, 5)
    # set the row number 3 to zero
    X_dense[3, :] = 0.0
    # check for mixed data where the value with
    # largest magnitude is negative
    X_dense[2, abs(X_dense[2, :]).argmax()] *= -1
    X_all_neg = -np.abs(X_dense)
    X_all_neg_sparse = sparse.csr_matrix(X_all_neg)

    for X in (X_dense, X_all_neg, X_all_neg_sparse):
        normalizer = Normalizer(norm="max")
        X_norm = normalizer.transform(X)
        assert X_norm is not X
        X_norm = toarray(X_norm)
        assert_array_equal(np.sign(X_norm), np.sign(toarray(X)))


def test_normalize():
    # Test normalize function
    # Only tests functionality not used by the tests for Normalizer.
    X = np.random.RandomState(37).randn(3, 2)
    assert_array_equal(normalize(X, copy=False), normalize(X.T, axis=0, copy=False).T)
    with pytest.raises(ValueError):
        normalize([[0]], axis=2)
    with pytest.raises(ValueError):
        normalize([[0]], norm="l3")

    rs = np.random.RandomState(0)
    X_dense = rs.randn(10, 5)
    X_sparse = sparse.csr_matrix(X_dense)
    ones = np.ones((10))
    for X in (X_dense, X_sparse):
        for dtype in (np.float32, np.float64):
            for norm in ("l1", "l2"):
                X = X.astype(dtype)
                X_norm = normalize(X, norm=norm)
                assert X_norm.dtype == dtype

                X_norm = toarray(X_norm)
                if norm == "l1":
                    row_sums = np.abs(X_norm).sum(axis=1)
                else:
                    X_norm_squared = X_norm ** 2
                    row_sums = X_norm_squared.sum(axis=1)

                assert_array_almost_equal(row_sums, ones)

    # Test return_norm
    X_dense = np.array([[3.0, 0, 4.0], [1.0, 0.0, 0.0], [2.0, 3.0, 0.0]])
    for norm in ("l1", "l2", "max"):
        _, norms = normalize(X_dense, norm=norm, return_norm=True)
        if norm == "l1":
            assert_array_almost_equal(norms, np.array([7.0, 1.0, 5.0]))
        elif norm == "l2":
            assert_array_almost_equal(norms, np.array([5.0, 1.0, 3.60555127]))
        else:
            assert_array_almost_equal(norms, np.array([4.0, 1.0, 3.0]))

    X_sparse = sparse.csr_matrix(X_dense)
    for norm in ("l1", "l2"):
        with pytest.raises(NotImplementedError):
            normalize(X_sparse, norm=norm, return_norm=True)
    _, norms = normalize(X_sparse, norm="max", return_norm=True)
    assert_array_almost_equal(norms, np.array([4.0, 1.0, 3.0]))


def test_binarizer():
    X_ = np.array([[1, 0, 5], [2, 3, -1]])

    for init in (np.array, list, sparse.csr_matrix, sparse.csc_matrix):

        X = init(X_.copy())

        binarizer = Binarizer(threshold=2.0, copy=True)
        X_bin = toarray(binarizer.transform(X))
        assert np.sum(X_bin == 0) == 4
        assert np.sum(X_bin == 1) == 2
        X_bin = binarizer.transform(X)
        assert sparse.issparse(X) == sparse.issparse(X_bin)

        binarizer = Binarizer(copy=True).fit(X)
        X_bin = toarray(binarizer.transform(X))
        assert X_bin is not X
        assert np.sum(X_bin == 0) == 2
        assert np.sum(X_bin == 1) == 4

        binarizer = Binarizer(copy=True)
        X_bin = binarizer.transform(X)
        assert X_bin is not X
        X_bin = toarray(X_bin)
        assert np.sum(X_bin == 0) == 2
        assert np.sum(X_bin == 1) == 4

        binarizer = Binarizer(copy=False)
        X_bin = binarizer.transform(X)
        if init is not list:
            assert X_bin is X

        binarizer = Binarizer(copy=False)
        X_float = np.array([[1, 0, 5], [2, 3, -1]], dtype=np.float64)
        X_bin = binarizer.transform(X_float)
        if init is not list:
            assert X_bin is X_float

        X_bin = toarray(X_bin)
        assert np.sum(X_bin == 0) == 2
        assert np.sum(X_bin == 1) == 4

    binarizer = Binarizer(threshold=-0.5, copy=True)
    for init in (np.array, list):
        X = init(X_.copy())

        X_bin = toarray(binarizer.transform(X))
        assert np.sum(X_bin == 0) == 1
        assert np.sum(X_bin == 1) == 5
        X_bin = binarizer.transform(X)

    # Cannot use threshold < 0 for sparse
    with pytest.raises(ValueError):
        binarizer.transform(sparse.csc_matrix(X))


def test_center_kernel():
    # Test that KernelCenterer is equivalent to StandardScaler
    # in feature space
    rng = np.random.RandomState(0)
    X_fit = rng.random_sample((5, 4))
    scaler = StandardScaler(with_std=False)
    scaler.fit(X_fit)
    X_fit_centered = scaler.transform(X_fit)
    K_fit = np.dot(X_fit, X_fit.T)

    # center fit time matrix
    centerer = KernelCenterer()
    K_fit_centered = np.dot(X_fit_centered, X_fit_centered.T)
    K_fit_centered2 = centerer.fit_transform(K_fit)
    assert_array_almost_equal(K_fit_centered, K_fit_centered2)

    # center predict time matrix
    X_pred = rng.random_sample((2, 4))
    K_pred = np.dot(X_pred, X_fit.T)
    X_pred_centered = scaler.transform(X_pred)
    K_pred_centered = np.dot(X_pred_centered, X_fit_centered.T)
    K_pred_centered2 = centerer.transform(K_pred)
    assert_array_almost_equal(K_pred_centered, K_pred_centered2)

    # check the results coherence with the method proposed in:
    # B. Schölkopf, A. Smola, and K.R. Müller,
    # "Nonlinear component analysis as a kernel eigenvalue problem"
    # equation (B.3)

    # K_centered3 = (I - 1_M) K (I - 1_M)
    #             =  K - 1_M K - K 1_M + 1_M K 1_M
    ones_M = np.ones_like(K_fit) / K_fit.shape[0]
    K_fit_centered3 = K_fit - ones_M @ K_fit - K_fit @ ones_M + ones_M @ K_fit @ ones_M
    assert_allclose(K_fit_centered, K_fit_centered3)

    # K_test_centered3 = (K_test - 1'_M K)(I - 1_M)
    #                  = K_test - 1'_M K - K_test 1_M + 1'_M K 1_M
    ones_prime_M = np.ones_like(K_pred) / K_fit.shape[0]
    K_pred_centered3 = (
        K_pred - ones_prime_M @ K_fit - K_pred @ ones_M + ones_prime_M @ K_fit @ ones_M
    )
    assert_allclose(K_pred_centered, K_pred_centered3)


def test_kernelcenterer_non_linear_kernel():
    """Check kernel centering for non-linear kernel."""
    rng = np.random.RandomState(0)
    X, X_test = rng.randn(100, 50), rng.randn(20, 50)

    def phi(X):
        """Our mapping function phi."""
        return np.vstack(
            [
                np.clip(X, a_min=0, a_max=None),
                -np.clip(X, a_min=None, a_max=0),
            ]
        )

    phi_X = phi(X)
    phi_X_test = phi(X_test)

    # centered the projection
    scaler = StandardScaler(with_std=False)
    phi_X_center = scaler.fit_transform(phi_X)
    phi_X_test_center = scaler.transform(phi_X_test)

    # create the different kernel
    K = phi_X @ phi_X.T
    K_test = phi_X_test @ phi_X.T
    K_center = phi_X_center @ phi_X_center.T
    K_test_center = phi_X_test_center @ phi_X_center.T

    kernel_centerer = KernelCenterer()
    kernel_centerer.fit(K)

    assert_allclose(kernel_centerer.transform(K), K_center)
    assert_allclose(kernel_centerer.transform(K_test), K_test_center)

    # check the results coherence with the method proposed in:
    # B. Schölkopf, A. Smola, and K.R. Müller,
    # "Nonlinear component analysis as a kernel eigenvalue problem"
    # equation (B.3)

    # K_centered = (I - 1_M) K (I - 1_M)
    #            =  K - 1_M K - K 1_M + 1_M K 1_M
    ones_M = np.ones_like(K) / K.shape[0]
    K_centered = K - ones_M @ K - K @ ones_M + ones_M @ K @ ones_M
    assert_allclose(kernel_centerer.transform(K), K_centered)

    # K_test_centered = (K_test - 1'_M K)(I - 1_M)
    #                 = K_test - 1'_M K - K_test 1_M + 1'_M K 1_M
    ones_prime_M = np.ones_like(K_test) / K.shape[0]
    K_test_centered = (
        K_test - ones_prime_M @ K - K_test @ ones_M + ones_prime_M @ K @ ones_M
    )
    assert_allclose(kernel_centerer.transform(K_test), K_test_centered)


def test_cv_pipeline_precomputed():
    # Cross-validate a regression on four coplanar points with the same
    # value. Use precomputed kernel to ensure Pipeline with KernelCenterer
    # is treated as a pairwise operation.
    X = np.array([[3, 0, 0], [0, 3, 0], [0, 0, 3], [1, 1, 1]])
    y_true = np.ones((4,))
    K = X.dot(X.T)
    kcent = KernelCenterer()
    pipeline = Pipeline([("kernel_centerer", kcent), ("svr", SVR())])

    # did the pipeline set the pairwise attribute?
    assert pipeline._get_tags()["pairwise"]

    # TODO: Remove in 1.1
    msg = r"Attribute `_pairwise` was deprecated in version 0\.24"
    with pytest.warns(FutureWarning, match=msg):
        assert pipeline._pairwise

    # test cross-validation, score should be almost perfect
    # NB: this test is pretty vacuous -- it's mainly to test integration
    #     of Pipeline and KernelCenterer
    y_pred = cross_val_predict(pipeline, K, y_true, cv=2)
    assert_array_almost_equal(y_true, y_pred)


# TODO: Remove in 1.1
def test_pairwise_deprecated():
    kcent = KernelCenterer()
    msg = r"Attribute `_pairwise` was deprecated in version 0\.24"
    with pytest.warns(FutureWarning, match=msg):
        kcent._pairwise


def test_fit_transform():
    rng = np.random.RandomState(0)
    X = rng.random_sample((5, 4))
    for obj in (StandardScaler(), Normalizer(), Binarizer()):
        X_transformed = obj.fit(X).transform(X)
        X_transformed2 = obj.fit_transform(X)
        assert_array_equal(X_transformed, X_transformed2)


def test_add_dummy_feature():
    X = [[1, 0], [0, 1], [0, 1]]
    X = add_dummy_feature(X)
    assert_array_equal(X, [[1, 1, 0], [1, 0, 1], [1, 0, 1]])


def test_add_dummy_feature_coo():
    X = sparse.coo_matrix([[1, 0], [0, 1], [0, 1]])
    X = add_dummy_feature(X)
    assert sparse.isspmatrix_coo(X), X
    assert_array_equal(X.toarray(), [[1, 1, 0], [1, 0, 1], [1, 0, 1]])


def test_add_dummy_feature_csc():
    X = sparse.csc_matrix([[1, 0], [0, 1], [0, 1]])
    X = add_dummy_feature(X)
    assert sparse.isspmatrix_csc(X), X
    assert_array_equal(X.toarray(), [[1, 1, 0], [1, 0, 1], [1, 0, 1]])


def test_add_dummy_feature_csr():
    X = sparse.csr_matrix([[1, 0], [0, 1], [0, 1]])
    X = add_dummy_feature(X)
    assert sparse.isspmatrix_csr(X), X
    assert_array_equal(X.toarray(), [[1, 1, 0], [1, 0, 1], [1, 0, 1]])


def test_fit_cold_start():
    X = iris.data
    X_2d = X[:, :2]

    # Scalers that have a partial_fit method
    scalers = [
        StandardScaler(with_mean=False, with_std=False),
        MinMaxScaler(),
        MaxAbsScaler(),
    ]

    for scaler in scalers:
        scaler.fit_transform(X)
        # with a different shape, this may break the scaler unless the internal
        # state is reset
        scaler.fit_transform(X_2d)


def test_quantile_transform_valid_axis():
    X = np.array([[0, 25, 50, 75, 100], [2, 4, 6, 8, 10], [2.6, 4.1, 2.3, 9.5, 0.1]])

    with pytest.raises(
        ValueError, match="axis should be either equal to 0 or 1. Got axis=2"
    ):
        quantile_transform(X.T, axis=2)


@pytest.mark.parametrize("method", ["box-cox", "yeo-johnson"])
def test_power_transformer_notfitted(method):
    pt = PowerTransformer(method=method)
    X = np.abs(X_1col)
    with pytest.raises(NotFittedError):
        pt.transform(X)
    with pytest.raises(NotFittedError):
        pt.inverse_transform(X)


@pytest.mark.parametrize("method", ["box-cox", "yeo-johnson"])
@pytest.mark.parametrize("standardize", [True, False])
@pytest.mark.parametrize("X", [X_1col, X_2d])
def test_power_transformer_inverse(method, standardize, X):
    # Make sure we get the original input when applying transform and then
    # inverse transform
    X = np.abs(X) if method == "box-cox" else X
    pt = PowerTransformer(method=method, standardize=standardize)
    X_trans = pt.fit_transform(X)
    assert_almost_equal(X, pt.inverse_transform(X_trans))


def test_power_transformer_1d():
    X = np.abs(X_1col)

    for standardize in [True, False]:
        pt = PowerTransformer(method="box-cox", standardize=standardize)

        X_trans = pt.fit_transform(X)
        X_trans_func = power_transform(X, method="box-cox", standardize=standardize)

        X_expected, lambda_expected = stats.boxcox(X.flatten())

        if standardize:
            X_expected = scale(X_expected)

        assert_almost_equal(X_expected.reshape(-1, 1), X_trans)
        assert_almost_equal(X_expected.reshape(-1, 1), X_trans_func)

        assert_almost_equal(X, pt.inverse_transform(X_trans))
        assert_almost_equal(lambda_expected, pt.lambdas_[0])

        assert len(pt.lambdas_) == X.shape[1]
        assert isinstance(pt.lambdas_, np.ndarray)


def test_power_transformer_2d():
    X = np.abs(X_2d)

    for standardize in [True, False]:
        pt = PowerTransformer(method="box-cox", standardize=standardize)

        X_trans_class = pt.fit_transform(X)
        X_trans_func = power_transform(X, method="box-cox", standardize=standardize)

        for X_trans in [X_trans_class, X_trans_func]:
            for j in range(X_trans.shape[1]):
                X_expected, lmbda = stats.boxcox(X[:, j].flatten())

                if standardize:
                    X_expected = scale(X_expected)

                assert_almost_equal(X_trans[:, j], X_expected)
                assert_almost_equal(lmbda, pt.lambdas_[j])

            # Test inverse transformation
            X_inv = pt.inverse_transform(X_trans)
            assert_array_almost_equal(X_inv, X)

        assert len(pt.lambdas_) == X.shape[1]
        assert isinstance(pt.lambdas_, np.ndarray)


def test_power_transformer_boxcox_strictly_positive_exception():
    # Exceptions should be raised for negative arrays and zero arrays when
    # method is boxcox

    pt = PowerTransformer(method="box-cox")
    pt.fit(np.abs(X_2d))
    X_with_negatives = X_2d
    not_positive_message = "strictly positive"

    with pytest.raises(ValueError, match=not_positive_message):
        pt.transform(X_with_negatives)

    with pytest.raises(ValueError, match=not_positive_message):
        pt.fit(X_with_negatives)

    with pytest.raises(ValueError, match=not_positive_message):
        power_transform(X_with_negatives, method="box-cox")

    with pytest.raises(ValueError, match=not_positive_message):
        pt.transform(np.zeros(X_2d.shape))

    with pytest.raises(ValueError, match=not_positive_message):
        pt.fit(np.zeros(X_2d.shape))

    with pytest.raises(ValueError, match=not_positive_message):
        power_transform(np.zeros(X_2d.shape), method="box-cox")


@pytest.mark.parametrize("X", [X_2d, np.abs(X_2d), -np.abs(X_2d), np.zeros(X_2d.shape)])
def test_power_transformer_yeojohnson_any_input(X):
    # Yeo-Johnson method should support any kind of input
    power_transform(X, method="yeo-johnson")


@pytest.mark.parametrize("method", ["box-cox", "yeo-johnson"])
def test_power_transformer_shape_exception(method):
    pt = PowerTransformer(method=method)
    X = np.abs(X_2d)
    pt.fit(X)

    # Exceptions should be raised for arrays with different num_columns
    # than during fitting
    wrong_shape_message = (
        r"X has \d+ features, but PowerTransformer is " r"expecting \d+ features"
    )

    with pytest.raises(ValueError, match=wrong_shape_message):
        pt.transform(X[:, 0:1])

    with pytest.raises(ValueError, match=wrong_shape_message):
        pt.inverse_transform(X[:, 0:1])


def test_power_transformer_method_exception():
    pt = PowerTransformer(method="monty-python")
    X = np.abs(X_2d)

    # An exception should be raised if PowerTransformer.method isn't valid
    bad_method_message = "'method' must be one of"
    with pytest.raises(ValueError, match=bad_method_message):
        pt.fit(X)


def test_power_transformer_lambda_zero():
    pt = PowerTransformer(method="box-cox", standardize=False)
    X = np.abs(X_2d)[:, 0:1]

    # Test the lambda = 0 case
    pt.lambdas_ = np.array([0])
    X_trans = pt.transform(X)
    assert_array_almost_equal(pt.inverse_transform(X_trans), X)


def test_power_transformer_lambda_one():
    # Make sure lambda = 1 corresponds to the identity for yeo-johnson
    pt = PowerTransformer(method="yeo-johnson", standardize=False)
    X = np.abs(X_2d)[:, 0:1]

    pt.lambdas_ = np.array([1])
    X_trans = pt.transform(X)
    assert_array_almost_equal(X_trans, X)


@pytest.mark.parametrize(
    "method, lmbda",
    [
        ("box-cox", 0.1),
        ("box-cox", 0.5),
        ("yeo-johnson", 0.1),
        ("yeo-johnson", 0.5),
        ("yeo-johnson", 1.0),
    ],
)
def test_optimization_power_transformer(method, lmbda):
    # Test the optimization procedure:
    # - set a predefined value for lambda
    # - apply inverse_transform to a normal dist (we get X_inv)
    # - apply fit_transform to X_inv (we get X_inv_trans)
    # - check that X_inv_trans is roughly equal to X

    rng = np.random.RandomState(0)
    n_samples = 20000
    X = rng.normal(loc=0, scale=1, size=(n_samples, 1))

    pt = PowerTransformer(method=method, standardize=False)
    pt.lambdas_ = [lmbda]
    X_inv = pt.inverse_transform(X)

    pt = PowerTransformer(method=method, standardize=False)
    X_inv_trans = pt.fit_transform(X_inv)

    assert_almost_equal(0, np.linalg.norm(X - X_inv_trans) / n_samples, decimal=2)
    assert_almost_equal(0, X_inv_trans.mean(), decimal=1)
    assert_almost_equal(1, X_inv_trans.std(), decimal=1)


def test_yeo_johnson_darwin_example():
    # test from original paper "A new family of power transformations to
    # improve normality or symmetry" by Yeo and Johnson.
    X = [6.1, -8.4, 1.0, 2.0, 0.7, 2.9, 3.5, 5.1, 1.8, 3.6, 7.0, 3.0, 9.3, 7.5, -6.0]
    X = np.array(X).reshape(-1, 1)
    lmbda = PowerTransformer(method="yeo-johnson").fit(X).lambdas_
    assert np.allclose(lmbda, 1.305, atol=1e-3)


@pytest.mark.parametrize("method", ["box-cox", "yeo-johnson"])
def test_power_transformer_nans(method):
    # Make sure lambda estimation is not influenced by NaN values
    # and that transform() supports NaN silently

    X = np.abs(X_1col)
    pt = PowerTransformer(method=method)
    pt.fit(X)
    lmbda_no_nans = pt.lambdas_[0]

    # concat nans at the end and check lambda stays the same
    X = np.concatenate([X, np.full_like(X, np.nan)])
    X = shuffle(X, random_state=0)

    pt.fit(X)
    lmbda_nans = pt.lambdas_[0]

    assert_almost_equal(lmbda_no_nans, lmbda_nans, decimal=5)

    X_trans = pt.transform(X)
    assert_array_equal(np.isnan(X_trans), np.isnan(X))


@pytest.mark.parametrize("method", ["box-cox", "yeo-johnson"])
@pytest.mark.parametrize("standardize", [True, False])
def test_power_transformer_fit_transform(method, standardize):
    # check that fit_transform() and fit().transform() return the same values
    X = X_1col
    if method == "box-cox":
        X = np.abs(X)

    pt = PowerTransformer(method, standardize=standardize)
    assert_array_almost_equal(pt.fit(X).transform(X), pt.fit_transform(X))


@pytest.mark.parametrize("method", ["box-cox", "yeo-johnson"])
@pytest.mark.parametrize("standardize", [True, False])
def test_power_transformer_copy_True(method, standardize):
    # Check that neither fit, transform, fit_transform nor inverse_transform
    # modify X inplace when copy=True
    X = X_1col
    if method == "box-cox":
        X = np.abs(X)

    X_original = X.copy()
    assert X is not X_original  # sanity checks
    assert_array_almost_equal(X, X_original)

    pt = PowerTransformer(method, standardize=standardize, copy=True)

    pt.fit(X)
    assert_array_almost_equal(X, X_original)
    X_trans = pt.transform(X)
    assert X_trans is not X

    X_trans = pt.fit_transform(X)
    assert_array_almost_equal(X, X_original)
    assert X_trans is not X

    X_inv_trans = pt.inverse_transform(X_trans)
    assert X_trans is not X_inv_trans


@pytest.mark.parametrize("method", ["box-cox", "yeo-johnson"])
@pytest.mark.parametrize("standardize", [True, False])
def test_power_transformer_copy_False(method, standardize):
    # check that when copy=False fit doesn't change X inplace but transform,
    # fit_transform and inverse_transform do.
    X = X_1col
    if method == "box-cox":
        X = np.abs(X)

    X_original = X.copy()
    assert X is not X_original  # sanity checks
    assert_array_almost_equal(X, X_original)

    pt = PowerTransformer(method, standardize=standardize, copy=False)

    pt.fit(X)
    assert_array_almost_equal(X, X_original)  # fit didn't change X

    X_trans = pt.transform(X)
    assert X_trans is X

    if method == "box-cox":
        X = np.abs(X)
    X_trans = pt.fit_transform(X)
    assert X_trans is X

    X_inv_trans = pt.inverse_transform(X_trans)
    assert X_trans is X_inv_trans


@pytest.mark.parametrize(
    "X_2",
    [
        sparse.random(10, 1, density=0.8, random_state=0),
        sparse.csr_matrix(np.full((10, 1), fill_value=np.nan)),
    ],
)
def test_standard_scaler_sparse_partial_fit_finite_variance(X_2):
    # non-regression test for:
    # https://github.com/scikit-learn/scikit-learn/issues/16448
    X_1 = sparse.random(5, 1, density=0.8)
    scaler = StandardScaler(with_mean=False)
    scaler.fit(X_1).partial_fit(X_2)
    assert np.isfinite(scaler.var_[0])


@pytest.mark.parametrize("feature_range", [(0, 1), (-10, 10)])
def test_minmax_scaler_clip(feature_range):
    # test behaviour of the parameter 'clip' in MinMaxScaler
    X = iris.data
    scaler = MinMaxScaler(feature_range=feature_range, clip=True).fit(X)
    X_min, X_max = np.min(X, axis=0), np.max(X, axis=0)
    X_test = [np.r_[X_min[:2] - 10, X_max[2:] + 10]]
    X_transformed = scaler.transform(X_test)
    assert_allclose(
        X_transformed,
        [[feature_range[0], feature_range[0], feature_range[1], feature_range[1]]],
    )


def test_standard_scaler_raise_error_for_1d_input():
    """Check that `inverse_transform` from `StandardScaler` raises an error
    with 1D array.
    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/19518
    """
    scaler = StandardScaler().fit(X_2d)
    err_msg = "Expected 2D array, got 1D array instead"
    with pytest.raises(ValueError, match=err_msg):
        scaler.inverse_transform(X_2d[:, 0])


@pytest.mark.parametrize(
    "Transformer",
    [
        MinMaxScaler,
        MaxAbsScaler,
        RobustScaler,
        StandardScaler,
        QuantileTransformer,
        PowerTransformer,
    ],
)
def test_one_to_one_features(Transformer):
    """Check one-to-one transformers give correct feature names."""
    tr = Transformer().fit(iris.data)
    names_out = tr.get_feature_names_out(iris.feature_names)
    assert_array_equal(names_out, iris.feature_names)


@pytest.mark.parametrize(
    "Transformer",
    [
        MinMaxScaler,
        MaxAbsScaler,
        RobustScaler,
        StandardScaler,
        QuantileTransformer,
        PowerTransformer,
    ],
)
def test_one_to_one_features_pandas(Transformer):
    """Check one-to-one transformers give correct feature names."""
    pd = pytest.importorskip("pandas")

    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    tr = Transformer().fit(df)

    names_out_df_default = tr.get_feature_names_out()
    assert_array_equal(names_out_df_default, iris.feature_names)

    names_out_df_valid_in = tr.get_feature_names_out(iris.feature_names)
    assert_array_equal(names_out_df_valid_in, iris.feature_names)

    msg = re.escape("input_features is not equal to feature_names_in_")
    with pytest.raises(ValueError, match=msg):
        invalid_names = list("abcd")
        tr.get_feature_names_out(invalid_names)
