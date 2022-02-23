"""Testing for Gaussian process regression """

# Author: Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>
# Modified by: Pete Green <p.l.green@liverpool.ac.uk>
# License: BSD 3 clause

import sys
import re
import numpy as np
import warnings

from scipy.optimize import approx_fprime

import pytest

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.gaussian_process.kernels import DotProduct, ExpSineSquared
from sklearn.gaussian_process.tests._mini_sequence_kernel import MiniSeqKernel
from sklearn.exceptions import ConvergenceWarning

from sklearn.utils._testing import (
    assert_array_less,
    assert_almost_equal,
    assert_array_almost_equal,
    assert_allclose,
)


def f(x):
    return x * np.sin(x)


X = np.atleast_2d([1.0, 3.0, 5.0, 6.0, 7.0, 8.0]).T
X2 = np.atleast_2d([2.0, 4.0, 5.5, 6.5, 7.5]).T
y = f(X).ravel()

fixed_kernel = RBF(length_scale=1.0, length_scale_bounds="fixed")
kernels = [
    RBF(length_scale=1.0),
    fixed_kernel,
    RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3)),
    C(1.0, (1e-2, 1e2)) * RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3)),
    C(1.0, (1e-2, 1e2)) * RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3))
    + C(1e-5, (1e-5, 1e2)),
    C(0.1, (1e-2, 1e2)) * RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3))
    + C(1e-5, (1e-5, 1e2)),
]
non_fixed_kernels = [kernel for kernel in kernels if kernel != fixed_kernel]


@pytest.mark.parametrize("kernel", kernels)
def test_gpr_interpolation(kernel):
    if sys.maxsize <= 2 ** 32:
        pytest.xfail("This test may fail on 32 bit Python")

    # Test the interpolating property for different kernels.
    gpr = GaussianProcessRegressor(kernel=kernel).fit(X, y)
    y_pred, y_cov = gpr.predict(X, return_cov=True)

    assert_almost_equal(y_pred, y)
    assert_almost_equal(np.diag(y_cov), 0.0)


def test_gpr_interpolation_structured():
    # Test the interpolating property for different kernels.
    kernel = MiniSeqKernel(baseline_similarity_bounds="fixed")
    X = ["A", "B", "C"]
    y = np.array([1, 2, 3])
    gpr = GaussianProcessRegressor(kernel=kernel).fit(X, y)
    y_pred, y_cov = gpr.predict(X, return_cov=True)

    assert_almost_equal(
        kernel(X, eval_gradient=True)[1].ravel(), (1 - np.eye(len(X))).ravel()
    )
    assert_almost_equal(y_pred, y)
    assert_almost_equal(np.diag(y_cov), 0.0)


@pytest.mark.parametrize("kernel", non_fixed_kernels)
def test_lml_improving(kernel):
    if sys.maxsize <= 2 ** 32:
        pytest.xfail("This test may fail on 32 bit Python")

    # Test that hyperparameter-tuning improves log-marginal likelihood.
    gpr = GaussianProcessRegressor(kernel=kernel).fit(X, y)
    assert gpr.log_marginal_likelihood(gpr.kernel_.theta) > gpr.log_marginal_likelihood(
        kernel.theta
    )


@pytest.mark.parametrize("kernel", kernels)
def test_lml_precomputed(kernel):
    # Test that lml of optimized kernel is stored correctly.
    gpr = GaussianProcessRegressor(kernel=kernel).fit(X, y)
    assert gpr.log_marginal_likelihood(gpr.kernel_.theta) == pytest.approx(
        gpr.log_marginal_likelihood()
    )


@pytest.mark.parametrize("kernel", kernels)
def test_lml_without_cloning_kernel(kernel):
    # Test that lml of optimized kernel is stored correctly.
    gpr = GaussianProcessRegressor(kernel=kernel).fit(X, y)
    input_theta = np.ones(gpr.kernel_.theta.shape, dtype=np.float64)

    gpr.log_marginal_likelihood(input_theta, clone_kernel=False)
    assert_almost_equal(gpr.kernel_.theta, input_theta, 7)


@pytest.mark.parametrize("kernel", non_fixed_kernels)
def test_converged_to_local_maximum(kernel):
    # Test that we are in local maximum after hyperparameter-optimization.
    gpr = GaussianProcessRegressor(kernel=kernel).fit(X, y)

    lml, lml_gradient = gpr.log_marginal_likelihood(gpr.kernel_.theta, True)

    assert np.all(
        (np.abs(lml_gradient) < 1e-4)
        | (gpr.kernel_.theta == gpr.kernel_.bounds[:, 0])
        | (gpr.kernel_.theta == gpr.kernel_.bounds[:, 1])
    )


@pytest.mark.parametrize("kernel", non_fixed_kernels)
def test_solution_inside_bounds(kernel):
    # Test that hyperparameter-optimization remains in bounds#
    gpr = GaussianProcessRegressor(kernel=kernel).fit(X, y)

    bounds = gpr.kernel_.bounds
    max_ = np.finfo(gpr.kernel_.theta.dtype).max
    tiny = 1e-10
    bounds[~np.isfinite(bounds[:, 1]), 1] = max_

    assert_array_less(bounds[:, 0], gpr.kernel_.theta + tiny)
    assert_array_less(gpr.kernel_.theta, bounds[:, 1] + tiny)


@pytest.mark.parametrize("kernel", kernels)
def test_lml_gradient(kernel):
    # Compare analytic and numeric gradient of log marginal likelihood.
    gpr = GaussianProcessRegressor(kernel=kernel).fit(X, y)

    lml, lml_gradient = gpr.log_marginal_likelihood(kernel.theta, True)
    lml_gradient_approx = approx_fprime(
        kernel.theta, lambda theta: gpr.log_marginal_likelihood(theta, False), 1e-10
    )

    assert_almost_equal(lml_gradient, lml_gradient_approx, 3)


@pytest.mark.parametrize("kernel", kernels)
def test_prior(kernel):
    # Test that GP prior has mean 0 and identical variances.
    gpr = GaussianProcessRegressor(kernel=kernel)

    y_mean, y_cov = gpr.predict(X, return_cov=True)

    assert_almost_equal(y_mean, 0, 5)
    if len(gpr.kernel.theta) > 1:
        # XXX: quite hacky, works only for current kernels
        assert_almost_equal(np.diag(y_cov), np.exp(kernel.theta[0]), 5)
    else:
        assert_almost_equal(np.diag(y_cov), 1, 5)


@pytest.mark.parametrize("kernel", kernels)
def test_sample_statistics(kernel):
    # Test that statistics of samples drawn from GP are correct.
    gpr = GaussianProcessRegressor(kernel=kernel).fit(X, y)

    y_mean, y_cov = gpr.predict(X2, return_cov=True)

    samples = gpr.sample_y(X2, 300000)

    # More digits accuracy would require many more samples
    assert_almost_equal(y_mean, np.mean(samples, 1), 1)
    assert_almost_equal(
        np.diag(y_cov) / np.diag(y_cov).max(),
        np.var(samples, 1) / np.diag(y_cov).max(),
        1,
    )


def test_no_optimizer():
    # Test that kernel parameters are unmodified when optimizer is None.
    kernel = RBF(1.0)
    gpr = GaussianProcessRegressor(kernel=kernel, optimizer=None).fit(X, y)
    assert np.exp(gpr.kernel_.theta) == 1.0


@pytest.mark.parametrize("kernel", kernels)
@pytest.mark.parametrize("target", [y, np.ones(X.shape[0], dtype=np.float64)])
def test_predict_cov_vs_std(kernel, target):
    if sys.maxsize <= 2 ** 32:
        pytest.xfail("This test may fail on 32 bit Python")

    # Test that predicted std.-dev. is consistent with cov's diagonal.
    gpr = GaussianProcessRegressor(kernel=kernel).fit(X, y)
    y_mean, y_cov = gpr.predict(X2, return_cov=True)
    y_mean, y_std = gpr.predict(X2, return_std=True)
    assert_almost_equal(np.sqrt(np.diag(y_cov)), y_std)


def test_anisotropic_kernel():
    # Test that GPR can identify meaningful anisotropic length-scales.
    # We learn a function which varies in one dimension ten-times slower
    # than in the other. The corresponding length-scales should differ by at
    # least a factor 5
    rng = np.random.RandomState(0)
    X = rng.uniform(-1, 1, (50, 2))
    y = X[:, 0] + 0.1 * X[:, 1]

    kernel = RBF([1.0, 1.0])
    gpr = GaussianProcessRegressor(kernel=kernel).fit(X, y)
    assert np.exp(gpr.kernel_.theta[1]) > np.exp(gpr.kernel_.theta[0]) * 5


def test_random_starts():
    # Test that an increasing number of random-starts of GP fitting only
    # increases the log marginal likelihood of the chosen theta.
    n_samples, n_features = 25, 2
    rng = np.random.RandomState(0)
    X = rng.randn(n_samples, n_features) * 2 - 1
    y = (
        np.sin(X).sum(axis=1)
        + np.sin(3 * X).sum(axis=1)
        + rng.normal(scale=0.1, size=n_samples)
    )

    kernel = C(1.0, (1e-2, 1e2)) * RBF(
        length_scale=[1.0] * n_features, length_scale_bounds=[(1e-4, 1e2)] * n_features
    ) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-5, 1e1))
    last_lml = -np.inf
    for n_restarts_optimizer in range(5):
        gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=n_restarts_optimizer,
            random_state=0,
        ).fit(X, y)
        lml = gp.log_marginal_likelihood(gp.kernel_.theta)
        assert lml > last_lml - np.finfo(np.float32).eps
        last_lml = lml


@pytest.mark.parametrize("kernel", kernels)
def test_y_normalization(kernel):
    """
    Test normalization of the target values in GP

    Fitting non-normalizing GP on normalized y and fitting normalizing GP
    on unnormalized y should yield identical results. Note that, here,
    'normalized y' refers to y that has been made zero mean and unit
    variance.

    """

    y_mean = np.mean(y)
    y_std = np.std(y)
    y_norm = (y - y_mean) / y_std

    # Fit non-normalizing GP on normalized y
    gpr = GaussianProcessRegressor(kernel=kernel)
    gpr.fit(X, y_norm)

    # Fit normalizing GP on unnormalized y
    gpr_norm = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
    gpr_norm.fit(X, y)

    # Compare predicted mean, std-devs and covariances
    y_pred, y_pred_std = gpr.predict(X2, return_std=True)
    y_pred = y_pred * y_std + y_mean
    y_pred_std = y_pred_std * y_std
    y_pred_norm, y_pred_std_norm = gpr_norm.predict(X2, return_std=True)

    assert_almost_equal(y_pred, y_pred_norm)
    assert_almost_equal(y_pred_std, y_pred_std_norm)

    _, y_cov = gpr.predict(X2, return_cov=True)
    y_cov = y_cov * y_std ** 2
    _, y_cov_norm = gpr_norm.predict(X2, return_cov=True)

    assert_almost_equal(y_cov, y_cov_norm)


def test_large_variance_y():
    """
    Here we test that, when noramlize_y=True, our GP can produce a
    sensible fit to training data whose variance is significantly
    larger than unity. This test was made in response to issue #15612.

    GP predictions are verified against predictions that were made
    using GPy which, here, is treated as the 'gold standard'. Note that we
    only investigate the RBF kernel here, as that is what was used in the
    GPy implementation.

    The following code can be used to recreate the GPy data:

    --------------------------------------------------------------------------
    import GPy

    kernel_gpy = GPy.kern.RBF(input_dim=1, lengthscale=1.)
    gpy = GPy.models.GPRegression(X, np.vstack(y_large), kernel_gpy)
    gpy.optimize()
    y_pred_gpy, y_var_gpy = gpy.predict(X2)
    y_pred_std_gpy = np.sqrt(y_var_gpy)
    --------------------------------------------------------------------------
    """

    # Here we utilise a larger variance version of the training data
    y_large = 10 * y

    # Standard GP with normalize_y=True
    RBF_params = {"length_scale": 1.0}
    kernel = RBF(**RBF_params)
    gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
    gpr.fit(X, y_large)
    y_pred, y_pred_std = gpr.predict(X2, return_std=True)

    # 'Gold standard' mean predictions from GPy
    y_pred_gpy = np.array(
        [15.16918303, -27.98707845, -39.31636019, 14.52605515, 69.18503589]
    )

    # 'Gold standard' std predictions from GPy
    y_pred_std_gpy = np.array(
        [7.78860962, 3.83179178, 0.63149951, 0.52745188, 0.86170042]
    )

    # Based on numerical experiments, it's reasonable to expect our
    # GP's mean predictions to get within 7% of predictions of those
    # made by GPy.
    assert_allclose(y_pred, y_pred_gpy, rtol=0.07, atol=0)

    # Based on numerical experiments, it's reasonable to expect our
    # GP's std predictions to get within 15% of predictions of those
    # made by GPy.
    assert_allclose(y_pred_std, y_pred_std_gpy, rtol=0.15, atol=0)


def test_y_multioutput():
    # Test that GPR can deal with multi-dimensional target values
    y_2d = np.vstack((y, y * 2)).T

    # Test for fixed kernel that first dimension of 2d GP equals the output
    # of 1d GP and that second dimension is twice as large
    kernel = RBF(length_scale=1.0)

    gpr = GaussianProcessRegressor(kernel=kernel, optimizer=None, normalize_y=False)
    gpr.fit(X, y)

    gpr_2d = GaussianProcessRegressor(kernel=kernel, optimizer=None, normalize_y=False)
    gpr_2d.fit(X, y_2d)

    y_pred_1d, y_std_1d = gpr.predict(X2, return_std=True)
    y_pred_2d, y_std_2d = gpr_2d.predict(X2, return_std=True)
    _, y_cov_1d = gpr.predict(X2, return_cov=True)
    _, y_cov_2d = gpr_2d.predict(X2, return_cov=True)

    assert_almost_equal(y_pred_1d, y_pred_2d[:, 0])
    assert_almost_equal(y_pred_1d, y_pred_2d[:, 1] / 2)

    # Standard deviation and covariance do not depend on output
    assert_almost_equal(y_std_1d, y_std_2d)
    assert_almost_equal(y_cov_1d, y_cov_2d)

    y_sample_1d = gpr.sample_y(X2, n_samples=10)
    y_sample_2d = gpr_2d.sample_y(X2, n_samples=10)
    assert_almost_equal(y_sample_1d, y_sample_2d[:, 0])

    # Test hyperparameter optimization
    for kernel in kernels:
        gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
        gpr.fit(X, y)

        gpr_2d = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
        gpr_2d.fit(X, np.vstack((y, y)).T)

        assert_almost_equal(gpr.kernel_.theta, gpr_2d.kernel_.theta, 4)


@pytest.mark.parametrize("kernel", non_fixed_kernels)
def test_custom_optimizer(kernel):
    # Test that GPR can use externally defined optimizers.
    # Define a dummy optimizer that simply tests 50 random hyperparameters
    def optimizer(obj_func, initial_theta, bounds):
        rng = np.random.RandomState(0)
        theta_opt, func_min = initial_theta, obj_func(
            initial_theta, eval_gradient=False
        )
        for _ in range(50):
            theta = np.atleast_1d(
                rng.uniform(np.maximum(-2, bounds[:, 0]), np.minimum(1, bounds[:, 1]))
            )
            f = obj_func(theta, eval_gradient=False)
            if f < func_min:
                theta_opt, func_min = theta, f
        return theta_opt, func_min

    gpr = GaussianProcessRegressor(kernel=kernel, optimizer=optimizer)
    gpr.fit(X, y)
    # Checks that optimizer improved marginal likelihood
    assert gpr.log_marginal_likelihood(gpr.kernel_.theta) > gpr.log_marginal_likelihood(
        gpr.kernel.theta
    )


def test_gpr_correct_error_message():
    X = np.arange(12).reshape(6, -1)
    y = np.ones(6)
    kernel = DotProduct()
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=0.0)
    message = (
        "The kernel, %s, is not returning a "
        "positive definite matrix. Try gradually increasing "
        "the 'alpha' parameter of your "
        "GaussianProcessRegressor estimator." % kernel
    )
    with pytest.raises(np.linalg.LinAlgError, match=re.escape(message)):
        gpr.fit(X, y)


@pytest.mark.parametrize("kernel", kernels)
def test_duplicate_input(kernel):
    # Test GPR can handle two different output-values for the same input.
    gpr_equal_inputs = GaussianProcessRegressor(kernel=kernel, alpha=1e-2)
    gpr_similar_inputs = GaussianProcessRegressor(kernel=kernel, alpha=1e-2)

    X_ = np.vstack((X, X[0]))
    y_ = np.hstack((y, y[0] + 1))
    gpr_equal_inputs.fit(X_, y_)

    X_ = np.vstack((X, X[0] + 1e-15))
    y_ = np.hstack((y, y[0] + 1))
    gpr_similar_inputs.fit(X_, y_)

    X_test = np.linspace(0, 10, 100)[:, None]
    y_pred_equal, y_std_equal = gpr_equal_inputs.predict(X_test, return_std=True)
    y_pred_similar, y_std_similar = gpr_similar_inputs.predict(X_test, return_std=True)

    assert_almost_equal(y_pred_equal, y_pred_similar)
    assert_almost_equal(y_std_equal, y_std_similar)


def test_no_fit_default_predict():
    # Test that GPR predictions without fit does not break by default.
    default_kernel = C(1.0, constant_value_bounds="fixed") * RBF(
        1.0, length_scale_bounds="fixed"
    )
    gpr1 = GaussianProcessRegressor()
    _, y_std1 = gpr1.predict(X, return_std=True)
    _, y_cov1 = gpr1.predict(X, return_cov=True)

    gpr2 = GaussianProcessRegressor(kernel=default_kernel)
    _, y_std2 = gpr2.predict(X, return_std=True)
    _, y_cov2 = gpr2.predict(X, return_cov=True)

    assert_array_almost_equal(y_std1, y_std2)
    assert_array_almost_equal(y_cov1, y_cov2)


def test_warning_bounds():
    kernel = RBF(length_scale_bounds=[1e-5, 1e-3])
    gpr = GaussianProcessRegressor(kernel=kernel)
    warning_message = (
        "The optimal value found for dimension 0 of parameter "
        "length_scale is close to the specified upper bound "
        "0.001. Increasing the bound and calling fit again may "
        "find a better value."
    )
    with pytest.warns(ConvergenceWarning, match=warning_message):
        gpr.fit(X, y)

    kernel_sum = WhiteKernel(noise_level_bounds=[1e-5, 1e-3]) + RBF(
        length_scale_bounds=[1e3, 1e5]
    )
    gpr_sum = GaussianProcessRegressor(kernel=kernel_sum)
    with pytest.warns(None) as record:
        with warnings.catch_warnings():
            # scipy 1.3.0 uses tostring which is deprecated in numpy
            warnings.filterwarnings("ignore", "tostring", DeprecationWarning)
            gpr_sum.fit(X, y)

    assert len(record) == 2
    assert (
        record[0].message.args[0]
        == "The optimal value found for "
        "dimension 0 of parameter "
        "k1__noise_level is close to the "
        "specified upper bound 0.001. "
        "Increasing the bound and calling "
        "fit again may find a better value."
    )

    assert (
        record[1].message.args[0]
        == "The optimal value found for "
        "dimension 0 of parameter "
        "k2__length_scale is close to the "
        "specified lower bound 1000.0. "
        "Decreasing the bound and calling "
        "fit again may find a better value."
    )

    X_tile = np.tile(X, 2)
    kernel_dims = RBF(length_scale=[1.0, 2.0], length_scale_bounds=[1e1, 1e2])
    gpr_dims = GaussianProcessRegressor(kernel=kernel_dims)

    with pytest.warns(None) as record:
        with warnings.catch_warnings():
            # scipy 1.3.0 uses tostring which is deprecated in numpy
            warnings.filterwarnings("ignore", "tostring", DeprecationWarning)
            gpr_dims.fit(X_tile, y)

    assert len(record) == 2
    assert (
        record[0].message.args[0]
        == "The optimal value found for "
        "dimension 0 of parameter "
        "length_scale is close to the "
        "specified lower bound 10.0. "
        "Decreasing the bound and calling "
        "fit again may find a better value."
    )

    assert (
        record[1].message.args[0]
        == "The optimal value found for "
        "dimension 1 of parameter "
        "length_scale is close to the "
        "specified lower bound 10.0. "
        "Decreasing the bound and calling "
        "fit again may find a better value."
    )


def test_bound_check_fixed_hyperparameter():
    # Regression test for issue #17943
    # Check that having a hyperparameter with fixed bounds doesn't cause an
    # error
    k1 = 50.0 ** 2 * RBF(length_scale=50.0)  # long term smooth rising trend
    k2 = ExpSineSquared(
        length_scale=1.0, periodicity=1.0, periodicity_bounds="fixed"
    )  # seasonal component
    kernel = k1 + k2
    GaussianProcessRegressor(kernel=kernel).fit(X, y)


# FIXME: we should test for multitargets as well. However, GPR is broken:
# see: https://github.com/scikit-learn/scikit-learn/pull/19706
@pytest.mark.parametrize("kernel", kernels)
def test_constant_target(kernel):
    """Check that the std. dev. is affected to 1 when normalizing a constant
    feature.
    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/18318
    NaN where affected to the target when scaling due to null std. dev. with
    constant target.
    """
    y_constant = np.ones(X.shape[0], dtype=np.float64)

    gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
    gpr.fit(X, y_constant)
    assert gpr._y_train_std == pytest.approx(1.0)

    y_pred, y_cov = gpr.predict(X, return_cov=True)
    assert_allclose(y_pred, y_constant)
    # set atol because we compare to zero
    assert_allclose(np.diag(y_cov), 0.0, atol=1e-9)


def test_gpr_consistency_std_cov_non_invertible_kernel():
    """Check the consistency between the returned std. dev. and the covariance.
    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/19936
    Inconsistencies were observed when the kernel cannot be inverted (or
    numerically stable).
    """
    kernel = C(8.98576054e05, (1e-12, 1e12)) * RBF(
        [5.91326520e02, 1.32584051e03], (1e-12, 1e12)
    ) + WhiteKernel(noise_level=1e-5)
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=0, optimizer=None)
    X_train = np.array(
        [
            [0.0, 0.0],
            [1.54919334, -0.77459667],
            [-1.54919334, 0.0],
            [0.0, -1.54919334],
            [0.77459667, 0.77459667],
            [-0.77459667, 1.54919334],
        ]
    )
    y_train = np.array(
        [
            [-2.14882017e-10],
            [-4.66975823e00],
            [4.01823986e00],
            [-1.30303674e00],
            [-1.35760156e00],
            [3.31215668e00],
        ]
    )
    gpr.fit(X_train, y_train)
    X_test = np.array(
        [
            [-1.93649167, -1.93649167],
            [1.93649167, -1.93649167],
            [-1.93649167, 1.93649167],
            [1.93649167, 1.93649167],
        ]
    )
    pred1, std = gpr.predict(X_test, return_std=True)
    pred2, cov = gpr.predict(X_test, return_cov=True)
    assert_allclose(std, np.sqrt(np.diagonal(cov)), rtol=1e-5)


@pytest.mark.parametrize(
    "params, TypeError, err_msg",
    [
        ({"kernel": RBF(), "optimizer": "unknown"}, ValueError, "Unknown optimizer"),
        ({"alpha": np.zeros(100)}, ValueError, "alpha must be a scalar or an array"),
        (
            {
                "kernel": WhiteKernel(noise_level_bounds=(-np.inf, np.inf)),
                "n_restarts_optimizer": 2,
            },
            ValueError,
            "requires that all bounds are finite",
        ),
    ],
)
def test_gpr_fit_error(params, TypeError, err_msg):
    """Check that expected error are raised during fit."""
    gpr = GaussianProcessRegressor(**params)
    with pytest.raises(TypeError, match=err_msg):
        gpr.fit(X, y)


def test_gpr_lml_error():
    """Check that we raise the proper error in the LML method."""
    gpr = GaussianProcessRegressor(kernel=RBF()).fit(X, y)

    err_msg = "Gradient can only be evaluated for theta!=None"
    with pytest.raises(ValueError, match=err_msg):
        gpr.log_marginal_likelihood(eval_gradient=True)


def test_gpr_predict_error():
    """Check that we raise the proper error during predict."""
    gpr = GaussianProcessRegressor(kernel=RBF()).fit(X, y)

    err_msg = "At most one of return_std or return_cov can be requested."
    with pytest.raises(RuntimeError, match=err_msg):
        gpr.predict(X, return_cov=True, return_std=True)


def test_y_std_with_multitarget_normalized():
    """Check the proper normalization of `y_std` and `y_cov` in multi-target scene.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/17394
    https://github.com/scikit-learn/scikit-learn/issues/18065
    """
    rng = np.random.RandomState(1234)

    n_samples, n_features, n_targets = 12, 10, 6

    X_train = rng.randn(n_samples, n_features)
    y_train = rng.randn(n_samples, n_targets)
    X_test = rng.randn(n_samples, n_features)

    # Generic kernel
    kernel = WhiteKernel(1.0, (1e-1, 1e3)) * C(10.0, (1e-3, 1e3))

    model = GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=10, alpha=0.1, normalize_y=True
    )
    model.fit(X_train, y_train)
    y_pred, y_std = model.predict(X_test, return_std=True)
    _, y_cov = model.predict(X_test, return_cov=True)

    assert y_pred.shape == (n_samples, n_targets)
    assert y_std.shape == (n_samples, n_targets)
    assert y_cov.shape == (n_samples, n_samples, n_targets)
