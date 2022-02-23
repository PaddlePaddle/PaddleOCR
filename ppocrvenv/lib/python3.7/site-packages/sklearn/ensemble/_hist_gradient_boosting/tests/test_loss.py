import numpy as np
from numpy.testing import assert_almost_equal
from numpy.testing import assert_allclose
from scipy.optimize import newton
from scipy.special import logit
from sklearn.utils import assert_all_finite
from sklearn.utils.fixes import sp_version, parse_version
import pytest

from sklearn.ensemble._hist_gradient_boosting.loss import _LOSSES
from sklearn.ensemble._hist_gradient_boosting.common import Y_DTYPE
from sklearn.ensemble._hist_gradient_boosting.common import G_H_DTYPE
from sklearn.utils._testing import skip_if_32bit
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads

n_threads = _openmp_effective_n_threads()


def get_derivatives_helper(loss):
    """Return get_gradients() and get_hessians() functions for a given loss."""

    def get_gradients(y_true, raw_predictions):
        # create gradients and hessians array, update inplace, and return
        gradients = np.empty_like(raw_predictions, dtype=G_H_DTYPE)
        hessians = np.empty_like(raw_predictions, dtype=G_H_DTYPE)
        loss.update_gradients_and_hessians(
            gradients, hessians, y_true, raw_predictions, None
        )
        return gradients

    def get_hessians(y_true, raw_predictions):
        # create gradients and hessians array, update inplace, and return
        gradients = np.empty_like(raw_predictions, dtype=G_H_DTYPE)
        hessians = np.empty_like(raw_predictions, dtype=G_H_DTYPE)
        loss.update_gradients_and_hessians(
            gradients, hessians, y_true, raw_predictions, None
        )

        if loss.__class__.__name__ == "LeastSquares":
            # hessians aren't updated because they're constant:
            # the value is 1 (and not 2) because the loss is actually an half
            # least squares loss.
            hessians = np.full_like(raw_predictions, fill_value=1)
        elif loss.__class__.__name__ == "LeastAbsoluteDeviation":
            # hessians aren't updated because they're constant
            hessians = np.full_like(raw_predictions, fill_value=0)

        return hessians

    return get_gradients, get_hessians


@pytest.mark.parametrize(
    "loss, x0, y_true",
    [
        ("squared_error", -2.0, 42),
        ("squared_error", 117.0, 1.05),
        ("squared_error", 0.0, 0.0),
        # The argmin of binary_crossentropy for y_true=0 and y_true=1 is resp. -inf
        # and +inf due to logit, cf. "complete separation". Therefore, we use
        # 0 < y_true < 1.
        ("binary_crossentropy", 0.3, 0.1),
        ("binary_crossentropy", -12, 0.2),
        ("binary_crossentropy", 30, 0.9),
        ("poisson", 12.0, 1.0),
        ("poisson", 0.0, 2.0),
        ("poisson", -22.0, 10.0),
    ],
)
@pytest.mark.skipif(
    sp_version == parse_version("1.2.0"),
    reason="bug in scipy 1.2.0, see scipy issue #9608",
)
@skip_if_32bit
def test_derivatives(loss, x0, y_true):
    # Check that gradients are zero when the loss is minimized on a single
    # value/sample using Halley's method with the first and second order
    # derivatives computed by the Loss instance.
    # Note that methods of Loss instances operate on arrays while the newton
    # root finder expects a scalar or a one-element array for this purpose.

    loss = _LOSSES[loss](sample_weight=None)
    y_true = np.array([y_true], dtype=Y_DTYPE)
    x0 = np.array([x0], dtype=Y_DTYPE).reshape(1, 1)
    get_gradients, get_hessians = get_derivatives_helper(loss)

    def func(x: np.ndarray) -> np.ndarray:
        if isinstance(loss, _LOSSES["binary_crossentropy"]):
            # Subtract a constant term such that the binary cross entropy
            # has its minimum at zero, which is needed for the newton method.
            actual_min = loss.pointwise_loss(y_true, logit(y_true))
            return loss.pointwise_loss(y_true, x) - actual_min
        else:
            return loss.pointwise_loss(y_true, x)

    def fprime(x: np.ndarray) -> np.ndarray:
        return get_gradients(y_true, x)

    def fprime2(x: np.ndarray) -> np.ndarray:
        return get_hessians(y_true, x)

    optimum = newton(func, x0=x0, fprime=fprime, fprime2=fprime2, maxiter=70, tol=2e-8)

    # Need to ravel arrays because assert_allclose requires matching dimensions
    y_true = y_true.ravel()
    optimum = optimum.ravel()
    assert_allclose(loss.inverse_link_function(optimum), y_true)
    assert_allclose(func(optimum), 0, atol=1e-14)
    assert_allclose(get_gradients(y_true, optimum), 0, atol=1e-6)


@pytest.mark.parametrize(
    "loss, n_classes, prediction_dim",
    [
        ("squared_error", 0, 1),
        ("absolute_error", 0, 1),
        ("binary_crossentropy", 2, 1),
        ("categorical_crossentropy", 3, 3),
        ("poisson", 0, 1),
    ],
)
@pytest.mark.skipif(
    Y_DTYPE != np.float64, reason="Need 64 bits float precision for numerical checks"
)
def test_numerical_gradients(loss, n_classes, prediction_dim, seed=0):
    # Make sure gradients and hessians computed in the loss are correct, by
    # comparing with their approximations computed with finite central
    # differences.
    # See https://en.wikipedia.org/wiki/Finite_difference.

    rng = np.random.RandomState(seed)
    n_samples = 100
    if loss in ("squared_error", "absolute_error"):
        y_true = rng.normal(size=n_samples).astype(Y_DTYPE)
    elif loss in ("poisson"):
        y_true = rng.poisson(size=n_samples).astype(Y_DTYPE)
    else:
        y_true = rng.randint(0, n_classes, size=n_samples).astype(Y_DTYPE)
    raw_predictions = rng.normal(size=(prediction_dim, n_samples)).astype(Y_DTYPE)
    loss = _LOSSES[loss](sample_weight=None, n_threads=n_threads)
    get_gradients, get_hessians = get_derivatives_helper(loss)

    # only take gradients and hessians of first tree / class.
    gradients = get_gradients(y_true, raw_predictions)[0, :].ravel()
    hessians = get_hessians(y_true, raw_predictions)[0, :].ravel()

    # Approximate gradients
    # For multiclass loss, we should only change the predictions of one tree
    # (here the first), hence the use of offset[0, :] += eps
    # As a softmax is computed, offsetting the whole array by a constant would
    # have no effect on the probabilities, and thus on the loss
    eps = 1e-9
    offset = np.zeros_like(raw_predictions)
    offset[0, :] = eps
    f_plus_eps = loss.pointwise_loss(y_true, raw_predictions + offset / 2)
    f_minus_eps = loss.pointwise_loss(y_true, raw_predictions - offset / 2)
    numerical_gradients = (f_plus_eps - f_minus_eps) / eps

    # Approximate hessians
    eps = 1e-4  # need big enough eps as we divide by its square
    offset[0, :] = eps
    f_plus_eps = loss.pointwise_loss(y_true, raw_predictions + offset)
    f_minus_eps = loss.pointwise_loss(y_true, raw_predictions - offset)
    f = loss.pointwise_loss(y_true, raw_predictions)
    numerical_hessians = (f_plus_eps + f_minus_eps - 2 * f) / eps ** 2

    assert_allclose(numerical_gradients, gradients, rtol=1e-4, atol=1e-7)
    assert_allclose(numerical_hessians, hessians, rtol=1e-4, atol=1e-7)


def test_baseline_least_squares():
    rng = np.random.RandomState(0)

    loss = _LOSSES["squared_error"](sample_weight=None)
    y_train = rng.normal(size=100)
    baseline_prediction = loss.get_baseline_prediction(y_train, None, 1)
    assert baseline_prediction.shape == tuple()  # scalar
    assert baseline_prediction.dtype == y_train.dtype
    # Make sure baseline prediction is the mean of all targets
    assert_almost_equal(baseline_prediction, y_train.mean())
    assert np.allclose(
        loss.inverse_link_function(baseline_prediction), baseline_prediction
    )


def test_baseline_absolute_error():
    rng = np.random.RandomState(0)

    loss = _LOSSES["absolute_error"](sample_weight=None)
    y_train = rng.normal(size=100)
    baseline_prediction = loss.get_baseline_prediction(y_train, None, 1)
    assert baseline_prediction.shape == tuple()  # scalar
    assert baseline_prediction.dtype == y_train.dtype
    # Make sure baseline prediction is the median of all targets
    assert np.allclose(
        loss.inverse_link_function(baseline_prediction), baseline_prediction
    )
    assert baseline_prediction == pytest.approx(np.median(y_train))


def test_baseline_poisson():
    rng = np.random.RandomState(0)

    loss = _LOSSES["poisson"](sample_weight=None)
    y_train = rng.poisson(size=100).astype(np.float64)
    # Sanity check, make sure at least one sample is non-zero so we don't take
    # log(0)
    assert y_train.sum() > 0
    baseline_prediction = loss.get_baseline_prediction(y_train, None, 1)
    assert np.isscalar(baseline_prediction)
    assert baseline_prediction.dtype == y_train.dtype
    assert_all_finite(baseline_prediction)
    # Make sure baseline prediction produces the log of the mean of all targets
    assert_almost_equal(np.log(y_train.mean()), baseline_prediction)

    # Test baseline for y_true = 0
    y_train.fill(0.0)
    baseline_prediction = loss.get_baseline_prediction(y_train, None, 1)
    assert_all_finite(baseline_prediction)


def test_baseline_binary_crossentropy():
    rng = np.random.RandomState(0)

    loss = _LOSSES["binary_crossentropy"](sample_weight=None)
    for y_train in (np.zeros(shape=100), np.ones(shape=100)):
        y_train = y_train.astype(np.float64)
        baseline_prediction = loss.get_baseline_prediction(y_train, None, 1)
        assert_all_finite(baseline_prediction)
        assert np.allclose(loss.inverse_link_function(baseline_prediction), y_train[0])

    # Make sure baseline prediction is equal to link_function(p), where p
    # is the proba of the positive class. We want predict_proba() to return p,
    # and by definition
    # p = inverse_link_function(raw_prediction) = sigmoid(raw_prediction)
    # So we want raw_prediction = link_function(p) = log(p / (1 - p))
    y_train = rng.randint(0, 2, size=100).astype(np.float64)
    baseline_prediction = loss.get_baseline_prediction(y_train, None, 1)
    assert baseline_prediction.shape == tuple()  # scalar
    assert baseline_prediction.dtype == y_train.dtype
    p = y_train.mean()
    assert np.allclose(baseline_prediction, np.log(p / (1 - p)))


def test_baseline_categorical_crossentropy():
    rng = np.random.RandomState(0)

    prediction_dim = 4
    loss = _LOSSES["categorical_crossentropy"](sample_weight=None)
    for y_train in (np.zeros(shape=100), np.ones(shape=100)):
        y_train = y_train.astype(np.float64)
        baseline_prediction = loss.get_baseline_prediction(
            y_train, None, prediction_dim
        )
        assert baseline_prediction.dtype == y_train.dtype
        assert_all_finite(baseline_prediction)

    # Same logic as for above test. Here inverse_link_function = softmax and
    # link_function = log
    y_train = rng.randint(0, prediction_dim + 1, size=100).astype(np.float32)
    baseline_prediction = loss.get_baseline_prediction(y_train, None, prediction_dim)
    assert baseline_prediction.shape == (prediction_dim, 1)
    for k in range(prediction_dim):
        p = (y_train == k).mean()
        assert np.allclose(baseline_prediction[k, :], np.log(p))


@pytest.mark.parametrize(
    "loss, problem",
    [
        ("squared_error", "regression"),
        ("absolute_error", "regression"),
        ("binary_crossentropy", "classification"),
        ("categorical_crossentropy", "classification"),
        ("poisson", "poisson_regression"),
    ],
)
@pytest.mark.parametrize("sample_weight", ["ones", "random"])
def test_sample_weight_multiplies_gradients(loss, problem, sample_weight):
    # Make sure that passing sample weights to the gradient and hessians
    # computation methods is equivalent to multiplying by the weights.

    rng = np.random.RandomState(42)
    n_samples = 1000

    if loss == "categorical_crossentropy":
        n_classes = prediction_dim = 3
    else:
        n_classes = prediction_dim = 1

    if problem == "regression":
        y_true = rng.normal(size=n_samples).astype(Y_DTYPE)
    elif problem == "poisson_regression":
        y_true = rng.poisson(size=n_samples).astype(Y_DTYPE)
    else:
        y_true = rng.randint(0, n_classes, size=n_samples).astype(Y_DTYPE)

    if sample_weight == "ones":
        sample_weight = np.ones(shape=n_samples, dtype=Y_DTYPE)
    else:
        sample_weight = rng.normal(size=n_samples).astype(Y_DTYPE)

    loss_ = _LOSSES[loss](sample_weight=sample_weight, n_threads=n_threads)

    baseline_prediction = loss_.get_baseline_prediction(y_true, None, prediction_dim)
    raw_predictions = np.zeros(
        shape=(prediction_dim, n_samples), dtype=baseline_prediction.dtype
    )
    raw_predictions += baseline_prediction

    gradients = np.empty(shape=(prediction_dim, n_samples), dtype=G_H_DTYPE)
    hessians = np.ones(shape=(prediction_dim, n_samples), dtype=G_H_DTYPE)
    loss_.update_gradients_and_hessians(
        gradients, hessians, y_true, raw_predictions, None
    )

    gradients_sw = np.empty(shape=(prediction_dim, n_samples), dtype=G_H_DTYPE)
    hessians_sw = np.ones(shape=(prediction_dim, n_samples), dtype=G_H_DTYPE)
    loss_.update_gradients_and_hessians(
        gradients_sw, hessians_sw, y_true, raw_predictions, sample_weight
    )

    assert np.allclose(gradients * sample_weight, gradients_sw)
    assert np.allclose(hessians * sample_weight, hessians_sw)


def test_init_gradient_and_hessians_sample_weight():
    # Make sure that passing sample_weight to a loss correctly influences the
    # hessians_are_constant attribute, and consequently the shape of the
    # hessians array.

    prediction_dim = 2
    n_samples = 5
    sample_weight = None
    loss = _LOSSES["squared_error"](sample_weight=sample_weight)
    _, hessians = loss.init_gradients_and_hessians(
        n_samples=n_samples, prediction_dim=prediction_dim, sample_weight=None
    )
    assert loss.hessians_are_constant
    assert hessians.shape == (1, 1)

    sample_weight = np.ones(n_samples)
    loss = _LOSSES["squared_error"](sample_weight=sample_weight)
    _, hessians = loss.init_gradients_and_hessians(
        n_samples=n_samples, prediction_dim=prediction_dim, sample_weight=sample_weight
    )
    assert not loss.hessians_are_constant
    assert hessians.shape == (prediction_dim, n_samples)
