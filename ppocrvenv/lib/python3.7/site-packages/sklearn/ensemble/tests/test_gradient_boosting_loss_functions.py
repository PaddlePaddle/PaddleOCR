"""
Testing for the gradient boosting loss functions and initial estimators.
"""
from itertools import product
import numpy as np
from numpy.testing import assert_allclose
import pytest
from pytest import approx

from sklearn.utils import check_random_state
from sklearn.metrics import mean_pinball_loss
from sklearn.ensemble._gb_losses import RegressionLossFunction
from sklearn.ensemble._gb_losses import LeastSquaresError
from sklearn.ensemble._gb_losses import LeastAbsoluteError
from sklearn.ensemble._gb_losses import HuberLossFunction
from sklearn.ensemble._gb_losses import QuantileLossFunction
from sklearn.ensemble._gb_losses import BinomialDeviance
from sklearn.ensemble._gb_losses import MultinomialDeviance
from sklearn.ensemble._gb_losses import ExponentialLoss
from sklearn.ensemble._gb_losses import LOSS_FUNCTIONS


def test_binomial_deviance():
    # Check binomial deviance loss.
    # Check against alternative definitions in ESLII.
    bd = BinomialDeviance(2)

    # pred has the same BD for y in {0, 1}
    assert bd(np.array([0.0]), np.array([0.0])) == bd(np.array([1.0]), np.array([0.0]))

    assert bd(np.array([1.0, 1, 1]), np.array([100.0, 100, 100])) == approx(0)
    assert bd(np.array([1.0, 0, 0]), np.array([100.0, -100, -100])) == approx(0)

    # check if same results as alternative definition of deviance, from ESLII
    # Eq. (10.18): -loglike = log(1 + exp(-2*z*f))
    # Note:
    # - We use y = {0, 1}, ESL (10.18) uses z in {-1, 1}, hence y=2*y-1
    # - ESL 2*f = pred_raw, hence the factor 2 of ESL disappears.
    # - Deviance = -2*loglike + .., hence a factor of 2 in front.
    def alt_dev(y, raw_pred):
        z = 2 * y - 1
        return 2 * np.mean(np.log(1 + np.exp(-z * raw_pred)))

    test_data = product(
        (np.array([0.0, 0, 0]), np.array([1.0, 1, 1])),
        (np.array([-5.0, -5, -5]), np.array([3.0, 3, 3])),
    )

    for datum in test_data:
        assert bd(*datum) == approx(alt_dev(*datum))

    # check the negative gradient against alternative formula from ESLII
    # Note: negative_gradient is half the negative gradient.
    def alt_ng(y, raw_pred):
        z = 2 * y - 1
        return z / (1 + np.exp(z * raw_pred))

    for datum in test_data:
        assert bd.negative_gradient(*datum) == approx(alt_ng(*datum))


def test_sample_weight_smoke():
    rng = check_random_state(13)
    y = rng.rand(100)
    pred = rng.rand(100)

    # least squares
    loss = LeastSquaresError()
    loss_wo_sw = loss(y, pred)
    loss_w_sw = loss(y, pred, np.ones(pred.shape[0], dtype=np.float32))
    assert loss_wo_sw == approx(loss_w_sw)


def test_sample_weight_init_estimators():
    # Smoke test for init estimators with sample weights.
    rng = check_random_state(13)
    X = rng.rand(100, 2)
    sample_weight = np.ones(100)
    reg_y = rng.rand(100)

    clf_y = rng.randint(0, 2, size=100)

    for Loss in LOSS_FUNCTIONS.values():
        if Loss is None:
            continue
        if issubclass(Loss, RegressionLossFunction):
            y = reg_y
            loss = Loss()
        else:
            k = 2
            y = clf_y
            if Loss.is_multi_class:
                # skip multiclass
                continue
            loss = Loss(k)

        init_est = loss.init_estimator()
        init_est.fit(X, y)
        out = loss.get_init_raw_predictions(X, init_est)
        assert out.shape == (y.shape[0], 1)

        sw_init_est = loss.init_estimator()
        sw_init_est.fit(X, y, sample_weight=sample_weight)
        sw_out = loss.get_init_raw_predictions(X, sw_init_est)
        assert sw_out.shape == (y.shape[0], 1)

        # check if predictions match
        assert_allclose(out, sw_out, rtol=1e-2)


def test_quantile_loss_function():
    # Non regression test for the QuantileLossFunction object
    # There was a sign problem when evaluating the function
    # for negative values of 'ytrue - ypred'
    x = np.asarray([-1.0, 0.0, 1.0])
    y_found = QuantileLossFunction(0.9)(x, np.zeros_like(x))
    y_expected = np.asarray([0.1, 0.0, 0.9]).mean()
    np.testing.assert_allclose(y_found, y_expected)
    y_found_p = mean_pinball_loss(x, np.zeros_like(x), alpha=0.9)
    np.testing.assert_allclose(y_found, y_found_p)


def test_sample_weight_deviance():
    # Test if deviance supports sample weights.
    rng = check_random_state(13)
    sample_weight = np.ones(100)
    reg_y = rng.rand(100)
    clf_y = rng.randint(0, 2, size=100)
    mclf_y = rng.randint(0, 3, size=100)

    for Loss in LOSS_FUNCTIONS.values():
        if Loss is None:
            continue
        if issubclass(Loss, RegressionLossFunction):
            y = reg_y
            p = reg_y
            loss = Loss()
        else:
            k = 2
            y = clf_y
            p = clf_y
            if Loss.is_multi_class:
                k = 3
                y = mclf_y
                # one-hot encoding
                p = np.zeros((y.shape[0], k), dtype=np.float64)
                for i in range(k):
                    p[:, i] = y == i
            loss = Loss(k)

        deviance_w_w = loss(y, p, sample_weight)
        deviance_wo_w = loss(y, p)
        assert deviance_wo_w == deviance_w_w


@pytest.mark.parametrize("n_classes, n_samples", [(3, 100), (5, 57), (7, 13)])
def test_multinomial_deviance(n_classes, n_samples):
    # Check multinomial deviance with and without sample weights.
    rng = np.random.RandomState(13)
    sample_weight = np.ones(n_samples)
    y_true = rng.randint(0, n_classes, size=n_samples)
    y_pred = np.zeros((n_samples, n_classes), dtype=np.float64)
    for klass in range(y_pred.shape[1]):
        y_pred[:, klass] = y_true == klass

    loss = MultinomialDeviance(n_classes)
    loss_wo_sw = loss(y_true, y_pred)
    assert loss_wo_sw > 0
    loss_w_sw = loss(y_true, y_pred, sample_weight=sample_weight)
    assert loss_wo_sw == approx(loss_w_sw)

    # Multinomial deviance uses weighted average loss rather than
    # weighted sum loss, so we make sure that the value remains the same
    # when we device the weight by 2.
    loss_w_sw = loss(y_true, y_pred, sample_weight=0.5 * sample_weight)
    assert loss_wo_sw == approx(loss_w_sw)


def test_mdl_computation_weighted():
    raw_predictions = np.array([[1.0, -1.0, -0.1], [-2.0, 1.0, 2.0]])
    y_true = np.array([0, 1])
    weights = np.array([1, 3])
    expected_loss = 1.0909323
    # MultinomialDeviance loss computation with weights.
    loss = MultinomialDeviance(3)
    assert loss(y_true, raw_predictions, weights) == approx(expected_loss)


@pytest.mark.parametrize("n", [0, 1, 2])
def test_mdl_exception(n):
    # Check that MultinomialDeviance throws an exception when n_classes <= 2
    err_msg = "MultinomialDeviance requires more than 2 classes."
    with pytest.raises(ValueError, match=err_msg):
        MultinomialDeviance(n)


def test_init_raw_predictions_shapes():
    # Make sure get_init_raw_predictions returns float64 arrays with shape
    # (n_samples, K) where K is 1 for binary classification and regression, and
    # K = n_classes for multiclass classification
    rng = np.random.RandomState(0)

    n_samples = 100
    X = rng.normal(size=(n_samples, 5))
    y = rng.normal(size=n_samples)
    for loss in (
        LeastSquaresError(),
        LeastAbsoluteError(),
        QuantileLossFunction(),
        HuberLossFunction(),
    ):
        init_estimator = loss.init_estimator().fit(X, y)
        raw_predictions = loss.get_init_raw_predictions(y, init_estimator)
        assert raw_predictions.shape == (n_samples, 1)
        assert raw_predictions.dtype == np.float64

    y = rng.randint(0, 2, size=n_samples)
    for loss in (BinomialDeviance(n_classes=2), ExponentialLoss(n_classes=2)):
        init_estimator = loss.init_estimator().fit(X, y)
        raw_predictions = loss.get_init_raw_predictions(y, init_estimator)
        assert raw_predictions.shape == (n_samples, 1)
        assert raw_predictions.dtype == np.float64

    for n_classes in range(3, 5):
        y = rng.randint(0, n_classes, size=n_samples)
        loss = MultinomialDeviance(n_classes=n_classes)
        init_estimator = loss.init_estimator().fit(X, y)
        raw_predictions = loss.get_init_raw_predictions(y, init_estimator)
        assert raw_predictions.shape == (n_samples, n_classes)
        assert raw_predictions.dtype == np.float64


def test_init_raw_predictions_values():
    # Make sure the get_init_raw_predictions() returns the expected values for
    # each loss.
    rng = np.random.RandomState(0)

    n_samples = 100
    X = rng.normal(size=(n_samples, 5))
    y = rng.normal(size=n_samples)

    # Least squares loss
    loss = LeastSquaresError()
    init_estimator = loss.init_estimator().fit(X, y)
    raw_predictions = loss.get_init_raw_predictions(y, init_estimator)
    # Make sure baseline prediction is the mean of all targets
    assert_allclose(raw_predictions, y.mean())

    # Least absolute and huber loss
    for Loss in (LeastAbsoluteError, HuberLossFunction):
        loss = Loss()
        init_estimator = loss.init_estimator().fit(X, y)
        raw_predictions = loss.get_init_raw_predictions(y, init_estimator)
        # Make sure baseline prediction is the median of all targets
        assert_allclose(raw_predictions, np.median(y))

    # Quantile loss
    for alpha in (0.1, 0.5, 0.9):
        loss = QuantileLossFunction(alpha=alpha)
        init_estimator = loss.init_estimator().fit(X, y)
        raw_predictions = loss.get_init_raw_predictions(y, init_estimator)
        # Make sure baseline prediction is the alpha-quantile of all targets
        assert_allclose(raw_predictions, np.percentile(y, alpha * 100))

    y = rng.randint(0, 2, size=n_samples)

    # Binomial deviance
    loss = BinomialDeviance(n_classes=2)
    init_estimator = loss.init_estimator().fit(X, y)
    # Make sure baseline prediction is equal to link_function(p), where p
    # is the proba of the positive class. We want predict_proba() to return p,
    # and by definition
    # p = inverse_link_function(raw_prediction) = sigmoid(raw_prediction)
    # So we want raw_prediction = link_function(p) = log(p / (1 - p))
    raw_predictions = loss.get_init_raw_predictions(y, init_estimator)
    p = y.mean()
    assert_allclose(raw_predictions, np.log(p / (1 - p)))

    # Exponential loss
    loss = ExponentialLoss(n_classes=2)
    init_estimator = loss.init_estimator().fit(X, y)
    raw_predictions = loss.get_init_raw_predictions(y, init_estimator)
    p = y.mean()
    assert_allclose(raw_predictions, 0.5 * np.log(p / (1 - p)))

    # Multinomial deviance loss
    for n_classes in range(3, 5):
        y = rng.randint(0, n_classes, size=n_samples)
        loss = MultinomialDeviance(n_classes=n_classes)
        init_estimator = loss.init_estimator().fit(X, y)
        raw_predictions = loss.get_init_raw_predictions(y, init_estimator)
        for k in range(n_classes):
            p = (y == k).mean()
            assert_allclose(raw_predictions[:, k], np.log(p))


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("alpha", [0.4, 0.5, 0.6])
def test_lad_equals_quantiles(seed, alpha):
    # Make sure quantile loss with alpha = .5 is equivalent to LAD
    lad = LeastAbsoluteError()
    ql = QuantileLossFunction(alpha=alpha)

    n_samples = 50
    rng = np.random.RandomState(seed)
    raw_predictions = rng.normal(size=(n_samples))
    y_true = rng.normal(size=(n_samples))

    lad_loss = lad(y_true, raw_predictions)
    ql_loss = ql(y_true, raw_predictions)
    if alpha == 0.5:
        assert lad_loss == approx(2 * ql_loss)

    weights = np.linspace(0, 1, n_samples) ** 2
    lad_weighted_loss = lad(y_true, raw_predictions, sample_weight=weights)
    ql_weighted_loss = ql(y_true, raw_predictions, sample_weight=weights)
    if alpha == 0.5:
        assert lad_weighted_loss == approx(2 * ql_weighted_loss)
    pbl_weighted_loss = mean_pinball_loss(
        y_true, raw_predictions, sample_weight=weights, alpha=alpha
    )
    assert pbl_weighted_loss == approx(ql_weighted_loss)


def test_exponential_loss():
    """Check that we compute the negative gradient of the exponential loss.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/9666
    """
    loss = ExponentialLoss(n_classes=2)
    y_true = np.array([0])
    y_pred = np.array([0])
    # we expect to have loss = exp(0) = 1
    assert loss(y_true, y_pred) == pytest.approx(1)
    # we expect to have negative gradient = -1 * (1 * exp(0)) = -1
    assert_allclose(loss.negative_gradient(y_true, y_pred), -1)
