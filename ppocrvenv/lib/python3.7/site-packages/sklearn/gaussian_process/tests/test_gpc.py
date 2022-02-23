"""Testing for Gaussian process classification """

# Author: Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>
# License: BSD 3 clause

import warnings
import numpy as np

from scipy.optimize import approx_fprime

import pytest

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.gaussian_process.tests._mini_sequence_kernel import MiniSeqKernel
from sklearn.exceptions import ConvergenceWarning

from sklearn.utils._testing import assert_almost_equal, assert_array_equal


def f(x):
    return np.sin(x)


X = np.atleast_2d(np.linspace(0, 10, 30)).T
X2 = np.atleast_2d([2.0, 4.0, 5.5, 6.5, 7.5]).T
y = np.array(f(X).ravel() > 0, dtype=int)
fX = f(X).ravel()
y_mc = np.empty(y.shape, dtype=int)  # multi-class
y_mc[fX < -0.35] = 0
y_mc[(fX >= -0.35) & (fX < 0.35)] = 1
y_mc[fX > 0.35] = 2


fixed_kernel = RBF(length_scale=1.0, length_scale_bounds="fixed")
kernels = [
    RBF(length_scale=0.1),
    fixed_kernel,
    RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3)),
    C(1.0, (1e-2, 1e2)) * RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3)),
]
non_fixed_kernels = [kernel for kernel in kernels if kernel != fixed_kernel]


@pytest.mark.parametrize("kernel", kernels)
def test_predict_consistent(kernel):
    # Check binary predict decision has also predicted probability above 0.5.
    gpc = GaussianProcessClassifier(kernel=kernel).fit(X, y)
    assert_array_equal(gpc.predict(X), gpc.predict_proba(X)[:, 1] >= 0.5)


def test_predict_consistent_structured():
    # Check binary predict decision has also predicted probability above 0.5.
    X = ["A", "AB", "B"]
    y = np.array([True, False, True])
    kernel = MiniSeqKernel(baseline_similarity_bounds="fixed")
    gpc = GaussianProcessClassifier(kernel=kernel).fit(X, y)
    assert_array_equal(gpc.predict(X), gpc.predict_proba(X)[:, 1] >= 0.5)


@pytest.mark.parametrize("kernel", non_fixed_kernels)
def test_lml_improving(kernel):
    # Test that hyperparameter-tuning improves log-marginal likelihood.
    gpc = GaussianProcessClassifier(kernel=kernel).fit(X, y)
    assert gpc.log_marginal_likelihood(gpc.kernel_.theta) > gpc.log_marginal_likelihood(
        kernel.theta
    )


@pytest.mark.parametrize("kernel", kernels)
def test_lml_precomputed(kernel):
    # Test that lml of optimized kernel is stored correctly.
    gpc = GaussianProcessClassifier(kernel=kernel).fit(X, y)
    assert_almost_equal(
        gpc.log_marginal_likelihood(gpc.kernel_.theta), gpc.log_marginal_likelihood(), 7
    )


@pytest.mark.parametrize("kernel", kernels)
def test_lml_without_cloning_kernel(kernel):
    # Test that clone_kernel=False has side-effects of kernel.theta.
    gpc = GaussianProcessClassifier(kernel=kernel).fit(X, y)
    input_theta = np.ones(gpc.kernel_.theta.shape, dtype=np.float64)

    gpc.log_marginal_likelihood(input_theta, clone_kernel=False)
    assert_almost_equal(gpc.kernel_.theta, input_theta, 7)


@pytest.mark.parametrize("kernel", non_fixed_kernels)
def test_converged_to_local_maximum(kernel):
    # Test that we are in local maximum after hyperparameter-optimization.
    gpc = GaussianProcessClassifier(kernel=kernel).fit(X, y)

    lml, lml_gradient = gpc.log_marginal_likelihood(gpc.kernel_.theta, True)

    assert np.all(
        (np.abs(lml_gradient) < 1e-4)
        | (gpc.kernel_.theta == gpc.kernel_.bounds[:, 0])
        | (gpc.kernel_.theta == gpc.kernel_.bounds[:, 1])
    )


@pytest.mark.parametrize("kernel", kernels)
def test_lml_gradient(kernel):
    # Compare analytic and numeric gradient of log marginal likelihood.
    gpc = GaussianProcessClassifier(kernel=kernel).fit(X, y)

    lml, lml_gradient = gpc.log_marginal_likelihood(kernel.theta, True)
    lml_gradient_approx = approx_fprime(
        kernel.theta, lambda theta: gpc.log_marginal_likelihood(theta, False), 1e-10
    )

    assert_almost_equal(lml_gradient, lml_gradient_approx, 3)


def test_random_starts():
    # Test that an increasing number of random-starts of GP fitting only
    # increases the log marginal likelihood of the chosen theta.
    n_samples, n_features = 25, 2
    rng = np.random.RandomState(0)
    X = rng.randn(n_samples, n_features) * 2 - 1
    y = (np.sin(X).sum(axis=1) + np.sin(3 * X).sum(axis=1)) > 0

    kernel = C(1.0, (1e-2, 1e2)) * RBF(
        length_scale=[1e-3] * n_features, length_scale_bounds=[(1e-4, 1e2)] * n_features
    )
    last_lml = -np.inf
    for n_restarts_optimizer in range(5):
        gp = GaussianProcessClassifier(
            kernel=kernel, n_restarts_optimizer=n_restarts_optimizer, random_state=0
        ).fit(X, y)
        lml = gp.log_marginal_likelihood(gp.kernel_.theta)
        assert lml > last_lml - np.finfo(np.float32).eps
        last_lml = lml


@pytest.mark.parametrize("kernel", non_fixed_kernels)
def test_custom_optimizer(kernel):
    # Test that GPC can use externally defined optimizers.
    # Define a dummy optimizer that simply tests 10 random hyperparameters
    def optimizer(obj_func, initial_theta, bounds):
        rng = np.random.RandomState(0)
        theta_opt, func_min = initial_theta, obj_func(
            initial_theta, eval_gradient=False
        )
        for _ in range(10):
            theta = np.atleast_1d(
                rng.uniform(np.maximum(-2, bounds[:, 0]), np.minimum(1, bounds[:, 1]))
            )
            f = obj_func(theta, eval_gradient=False)
            if f < func_min:
                theta_opt, func_min = theta, f
        return theta_opt, func_min

    gpc = GaussianProcessClassifier(kernel=kernel, optimizer=optimizer)
    gpc.fit(X, y_mc)
    # Checks that optimizer improved marginal likelihood
    assert gpc.log_marginal_likelihood(gpc.kernel_.theta) > gpc.log_marginal_likelihood(
        kernel.theta
    )


@pytest.mark.parametrize("kernel", kernels)
def test_multi_class(kernel):
    # Test GPC for multi-class classification problems.
    gpc = GaussianProcessClassifier(kernel=kernel)
    gpc.fit(X, y_mc)

    y_prob = gpc.predict_proba(X2)
    assert_almost_equal(y_prob.sum(1), 1)

    y_pred = gpc.predict(X2)
    assert_array_equal(np.argmax(y_prob, 1), y_pred)


@pytest.mark.parametrize("kernel", kernels)
def test_multi_class_n_jobs(kernel):
    # Test that multi-class GPC produces identical results with n_jobs>1.
    gpc = GaussianProcessClassifier(kernel=kernel)
    gpc.fit(X, y_mc)

    gpc_2 = GaussianProcessClassifier(kernel=kernel, n_jobs=2)
    gpc_2.fit(X, y_mc)

    y_prob = gpc.predict_proba(X2)
    y_prob_2 = gpc_2.predict_proba(X2)
    assert_almost_equal(y_prob, y_prob_2)


def test_warning_bounds():
    kernel = RBF(length_scale_bounds=[1e-5, 1e-3])
    gpc = GaussianProcessClassifier(kernel=kernel)
    warning_message = (
        "The optimal value found for dimension 0 of parameter "
        "length_scale is close to the specified upper bound "
        "0.001. Increasing the bound and calling fit again may "
        "find a better value."
    )
    with pytest.warns(ConvergenceWarning, match=warning_message):
        gpc.fit(X, y)

    kernel_sum = WhiteKernel(noise_level_bounds=[1e-5, 1e-3]) + RBF(
        length_scale_bounds=[1e3, 1e5]
    )
    gpc_sum = GaussianProcessClassifier(kernel=kernel_sum)
    with pytest.warns(None) as record:
        with warnings.catch_warnings():
            # scipy 1.3.0 uses tostring which is deprecated in numpy
            warnings.filterwarnings("ignore", "tostring", DeprecationWarning)
            gpc_sum.fit(X, y)

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
    gpc_dims = GaussianProcessClassifier(kernel=kernel_dims)

    with pytest.warns(None) as record:
        with warnings.catch_warnings():
            # scipy 1.3.0 uses tostring which is deprecated in numpy
            warnings.filterwarnings("ignore", "tostring", DeprecationWarning)
            gpc_dims.fit(X_tile, y)

    assert len(record) == 2
    assert (
        record[0].message.args[0]
        == "The optimal value found for "
        "dimension 0 of parameter "
        "length_scale is close to the "
        "specified upper bound 100.0. "
        "Increasing the bound and calling "
        "fit again may find a better value."
    )

    assert (
        record[1].message.args[0]
        == "The optimal value found for "
        "dimension 1 of parameter "
        "length_scale is close to the "
        "specified upper bound 100.0. "
        "Increasing the bound and calling "
        "fit again may find a better value."
    )
