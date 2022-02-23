"""
Testing for Multi-layer Perceptron module (sklearn.neural_network)
"""

# Author: Issam H. Laradji
# License: BSD 3 clause

import pytest
import sys
import warnings
import re

import numpy as np
import joblib

from numpy.testing import (
    assert_almost_equal,
    assert_array_equal,
    assert_allclose,
)

from sklearn.datasets import load_digits, load_iris
from sklearn.datasets import make_regression, make_multilabel_classification
from sklearn.exceptions import ConvergenceWarning
from io import StringIO
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler, scale
from scipy.sparse import csr_matrix
from sklearn.utils._testing import ignore_warnings


ACTIVATION_TYPES = ["identity", "logistic", "tanh", "relu"]

X_digits, y_digits = load_digits(n_class=3, return_X_y=True)

X_digits_multi = MinMaxScaler().fit_transform(X_digits[:200])
y_digits_multi = y_digits[:200]

X_digits, y_digits = load_digits(n_class=2, return_X_y=True)

X_digits_binary = MinMaxScaler().fit_transform(X_digits[:200])
y_digits_binary = y_digits[:200]

classification_datasets = [
    (X_digits_multi, y_digits_multi),
    (X_digits_binary, y_digits_binary),
]

X_reg, y_reg = make_regression(
    n_samples=200, n_features=10, bias=20.0, noise=100.0, random_state=7
)
y_reg = scale(y_reg)
regression_datasets = [(X_reg, y_reg)]

iris = load_iris()

X_iris = iris.data
y_iris = iris.target


def test_alpha():
    # Test that larger alpha yields weights closer to zero
    X = X_digits_binary[:100]
    y = y_digits_binary[:100]

    alpha_vectors = []
    alpha_values = np.arange(2)
    absolute_sum = lambda x: np.sum(np.abs(x))

    for alpha in alpha_values:
        mlp = MLPClassifier(hidden_layer_sizes=10, alpha=alpha, random_state=1)
        with ignore_warnings(category=ConvergenceWarning):
            mlp.fit(X, y)
        alpha_vectors.append(
            np.array([absolute_sum(mlp.coefs_[0]), absolute_sum(mlp.coefs_[1])])
        )

    for i in range(len(alpha_values) - 1):
        assert (alpha_vectors[i] > alpha_vectors[i + 1]).all()


def test_fit():
    # Test that the algorithm solution is equal to a worked out example.
    X = np.array([[0.6, 0.8, 0.7]])
    y = np.array([0])
    mlp = MLPClassifier(
        solver="sgd",
        learning_rate_init=0.1,
        alpha=0.1,
        activation="logistic",
        random_state=1,
        max_iter=1,
        hidden_layer_sizes=2,
        momentum=0,
    )
    # set weights
    mlp.coefs_ = [0] * 2
    mlp.intercepts_ = [0] * 2
    mlp.n_outputs_ = 1
    mlp.coefs_[0] = np.array([[0.1, 0.2], [0.3, 0.1], [0.5, 0]])
    mlp.coefs_[1] = np.array([[0.1], [0.2]])
    mlp.intercepts_[0] = np.array([0.1, 0.1])
    mlp.intercepts_[1] = np.array([1.0])
    mlp._coef_grads = [] * 2
    mlp._intercept_grads = [] * 2
    mlp.n_features_in_ = 3

    # Initialize parameters
    mlp.n_iter_ = 0
    mlp.learning_rate_ = 0.1

    # Compute the number of layers
    mlp.n_layers_ = 3

    # Pre-allocate gradient matrices
    mlp._coef_grads = [0] * (mlp.n_layers_ - 1)
    mlp._intercept_grads = [0] * (mlp.n_layers_ - 1)

    mlp.out_activation_ = "logistic"
    mlp.t_ = 0
    mlp.best_loss_ = np.inf
    mlp.loss_curve_ = []
    mlp._no_improvement_count = 0
    mlp._intercept_velocity = [
        np.zeros_like(intercepts) for intercepts in mlp.intercepts_
    ]
    mlp._coef_velocity = [np.zeros_like(coefs) for coefs in mlp.coefs_]

    mlp.partial_fit(X, y, classes=[0, 1])
    # Manually worked out example
    # h1 = g(X1 * W_i1 + b11) = g(0.6 * 0.1 + 0.8 * 0.3 + 0.7 * 0.5 + 0.1)
    #       =  0.679178699175393
    # h2 = g(X2 * W_i2 + b12) = g(0.6 * 0.2 + 0.8 * 0.1 + 0.7 * 0 + 0.1)
    #         = 0.574442516811659
    # o1 = g(h * W2 + b21) = g(0.679 * 0.1 + 0.574 * 0.2 + 1)
    #       = 0.7654329236196236
    # d21 = -(0 - 0.765) = 0.765
    # d11 = (1 - 0.679) * 0.679 * 0.765 * 0.1 = 0.01667
    # d12 = (1 - 0.574) * 0.574 * 0.765 * 0.2 = 0.0374
    # W1grad11 = X1 * d11 + alpha * W11 = 0.6 * 0.01667 + 0.1 * 0.1 = 0.0200
    # W1grad11 = X1 * d12 + alpha * W12 = 0.6 * 0.0374 + 0.1 * 0.2 = 0.04244
    # W1grad21 = X2 * d11 + alpha * W13 = 0.8 * 0.01667 + 0.1 * 0.3 = 0.043336
    # W1grad22 = X2 * d12 + alpha * W14 = 0.8 * 0.0374 + 0.1 * 0.1 = 0.03992
    # W1grad31 = X3 * d11 + alpha * W15 = 0.6 * 0.01667 + 0.1 * 0.5 = 0.060002
    # W1grad32 = X3 * d12 + alpha * W16 = 0.6 * 0.0374 + 0.1 * 0 = 0.02244
    # W2grad1 = h1 * d21 + alpha * W21 = 0.679 * 0.765 + 0.1 * 0.1 = 0.5294
    # W2grad2 = h2 * d21 + alpha * W22 = 0.574 * 0.765 + 0.1 * 0.2 = 0.45911
    # b1grad1 = d11 = 0.01667
    # b1grad2 = d12 = 0.0374
    # b2grad = d21 = 0.765
    # W1 = W1 - eta * [W1grad11, .., W1grad32] = [[0.1, 0.2], [0.3, 0.1],
    #          [0.5, 0]] - 0.1 * [[0.0200, 0.04244], [0.043336, 0.03992],
    #          [0.060002, 0.02244]] = [[0.098, 0.195756], [0.2956664,
    #          0.096008], [0.4939998, -0.002244]]
    # W2 = W2 - eta * [W2grad1, W2grad2] = [[0.1], [0.2]] - 0.1 *
    #        [[0.5294], [0.45911]] = [[0.04706], [0.154089]]
    # b1 = b1 - eta * [b1grad1, b1grad2] = 0.1 - 0.1 * [0.01667, 0.0374]
    #         = [0.098333, 0.09626]
    # b2 = b2 - eta * b2grad = 1.0 - 0.1 * 0.765 = 0.9235
    assert_almost_equal(
        mlp.coefs_[0],
        np.array([[0.098, 0.195756], [0.2956664, 0.096008], [0.4939998, -0.002244]]),
        decimal=3,
    )
    assert_almost_equal(mlp.coefs_[1], np.array([[0.04706], [0.154089]]), decimal=3)
    assert_almost_equal(mlp.intercepts_[0], np.array([0.098333, 0.09626]), decimal=3)
    assert_almost_equal(mlp.intercepts_[1], np.array(0.9235), decimal=3)
    # Testing output
    #  h1 = g(X1 * W_i1 + b11) = g(0.6 * 0.098 + 0.8 * 0.2956664 +
    #               0.7 * 0.4939998 + 0.098333) = 0.677
    #  h2 = g(X2 * W_i2 + b12) = g(0.6 * 0.195756 + 0.8 * 0.096008 +
    #            0.7 * -0.002244 + 0.09626) = 0.572
    #  o1 = h * W2 + b21 = 0.677 * 0.04706 +
    #             0.572 * 0.154089 + 0.9235 = 1.043
    #  prob = sigmoid(o1) = 0.739
    assert_almost_equal(mlp.predict_proba(X)[0, 1], 0.739, decimal=3)


def test_gradient():
    # Test gradient.

    # This makes sure that the activation functions and their derivatives
    # are correct. The numerical and analytical computation of the gradient
    # should be close.
    for n_labels in [2, 3]:
        n_samples = 5
        n_features = 10
        random_state = np.random.RandomState(seed=42)
        X = random_state.rand(n_samples, n_features)
        y = 1 + np.mod(np.arange(n_samples) + 1, n_labels)
        Y = LabelBinarizer().fit_transform(y)

        for activation in ACTIVATION_TYPES:
            mlp = MLPClassifier(
                activation=activation,
                hidden_layer_sizes=10,
                solver="lbfgs",
                alpha=1e-5,
                learning_rate_init=0.2,
                max_iter=1,
                random_state=1,
            )
            mlp.fit(X, y)

            theta = np.hstack([l.ravel() for l in mlp.coefs_ + mlp.intercepts_])

            layer_units = [X.shape[1]] + [mlp.hidden_layer_sizes] + [mlp.n_outputs_]

            activations = []
            deltas = []
            coef_grads = []
            intercept_grads = []

            activations.append(X)
            for i in range(mlp.n_layers_ - 1):
                activations.append(np.empty((X.shape[0], layer_units[i + 1])))
                deltas.append(np.empty((X.shape[0], layer_units[i + 1])))

                fan_in = layer_units[i]
                fan_out = layer_units[i + 1]
                coef_grads.append(np.empty((fan_in, fan_out)))
                intercept_grads.append(np.empty(fan_out))

            # analytically compute the gradients
            def loss_grad_fun(t):
                return mlp._loss_grad_lbfgs(
                    t, X, Y, activations, deltas, coef_grads, intercept_grads
                )

            [value, grad] = loss_grad_fun(theta)
            numgrad = np.zeros(np.size(theta))
            n = np.size(theta, 0)
            E = np.eye(n)
            epsilon = 1e-5
            # numerically compute the gradients
            for i in range(n):
                dtheta = E[:, i] * epsilon
                numgrad[i] = (
                    loss_grad_fun(theta + dtheta)[0] - loss_grad_fun(theta - dtheta)[0]
                ) / (epsilon * 2.0)
            assert_almost_equal(numgrad, grad)


@pytest.mark.parametrize("X,y", classification_datasets)
def test_lbfgs_classification(X, y):
    # Test lbfgs on classification.
    # It should achieve a score higher than 0.95 for the binary and multi-class
    # versions of the digits dataset.
    X_train = X[:150]
    y_train = y[:150]
    X_test = X[150:]
    expected_shape_dtype = (X_test.shape[0], y_train.dtype.kind)

    for activation in ACTIVATION_TYPES:
        mlp = MLPClassifier(
            solver="lbfgs",
            hidden_layer_sizes=50,
            max_iter=150,
            shuffle=True,
            random_state=1,
            activation=activation,
        )
        mlp.fit(X_train, y_train)
        y_predict = mlp.predict(X_test)
        assert mlp.score(X_train, y_train) > 0.95
        assert (y_predict.shape[0], y_predict.dtype.kind) == expected_shape_dtype


@pytest.mark.parametrize("X,y", regression_datasets)
def test_lbfgs_regression(X, y):
    # Test lbfgs on the regression dataset.
    for activation in ACTIVATION_TYPES:
        mlp = MLPRegressor(
            solver="lbfgs",
            hidden_layer_sizes=50,
            max_iter=150,
            shuffle=True,
            random_state=1,
            activation=activation,
        )
        mlp.fit(X, y)
        if activation == "identity":
            assert mlp.score(X, y) > 0.80
        else:
            # Non linear models perform much better than linear bottleneck:
            assert mlp.score(X, y) > 0.98


@pytest.mark.parametrize("X,y", classification_datasets)
def test_lbfgs_classification_maxfun(X, y):
    # Test lbfgs parameter max_fun.
    # It should independently limit the number of iterations for lbfgs.
    max_fun = 10
    # classification tests
    for activation in ACTIVATION_TYPES:
        mlp = MLPClassifier(
            solver="lbfgs",
            hidden_layer_sizes=50,
            max_iter=150,
            max_fun=max_fun,
            shuffle=True,
            random_state=1,
            activation=activation,
        )
        with pytest.warns(ConvergenceWarning):
            mlp.fit(X, y)
            assert max_fun >= mlp.n_iter_


@pytest.mark.parametrize("X,y", regression_datasets)
def test_lbfgs_regression_maxfun(X, y):
    # Test lbfgs parameter max_fun.
    # It should independently limit the number of iterations for lbfgs.
    max_fun = 10
    # regression tests
    for activation in ACTIVATION_TYPES:
        mlp = MLPRegressor(
            solver="lbfgs",
            hidden_layer_sizes=50,
            tol=0.0,
            max_iter=150,
            max_fun=max_fun,
            shuffle=True,
            random_state=1,
            activation=activation,
        )
        with pytest.warns(ConvergenceWarning):
            mlp.fit(X, y)
            assert max_fun >= mlp.n_iter_

    mlp.max_fun = -1
    with pytest.raises(ValueError):
        mlp.fit(X, y)


def test_learning_rate_warmstart():
    # Tests that warm_start reuse past solutions.
    X = [[3, 2], [1, 6], [5, 6], [-2, -4]]
    y = [1, 1, 1, 0]
    for learning_rate in ["invscaling", "constant"]:
        mlp = MLPClassifier(
            solver="sgd",
            hidden_layer_sizes=4,
            learning_rate=learning_rate,
            max_iter=1,
            power_t=0.25,
            warm_start=True,
        )
        with ignore_warnings(category=ConvergenceWarning):
            mlp.fit(X, y)
            prev_eta = mlp._optimizer.learning_rate
            mlp.fit(X, y)
            post_eta = mlp._optimizer.learning_rate

        if learning_rate == "constant":
            assert prev_eta == post_eta
        elif learning_rate == "invscaling":
            assert mlp.learning_rate_init / pow(8 + 1, mlp.power_t) == post_eta


def test_multilabel_classification():
    # Test that multi-label classification works as expected.
    # test fit method
    X, y = make_multilabel_classification(
        n_samples=50, random_state=0, return_indicator=True
    )
    mlp = MLPClassifier(
        solver="lbfgs",
        hidden_layer_sizes=50,
        alpha=1e-5,
        max_iter=150,
        random_state=0,
        activation="logistic",
        learning_rate_init=0.2,
    )
    mlp.fit(X, y)
    assert mlp.score(X, y) > 0.97

    # test partial fit method
    mlp = MLPClassifier(
        solver="sgd",
        hidden_layer_sizes=50,
        max_iter=150,
        random_state=0,
        activation="logistic",
        alpha=1e-5,
        learning_rate_init=0.2,
    )
    for i in range(100):
        mlp.partial_fit(X, y, classes=[0, 1, 2, 3, 4])
    assert mlp.score(X, y) > 0.9

    # Make sure early stopping still work now that splitting is stratified by
    # default (it is disabled for multilabel classification)
    mlp = MLPClassifier(early_stopping=True)
    mlp.fit(X, y).predict(X)


def test_multioutput_regression():
    # Test that multi-output regression works as expected
    X, y = make_regression(n_samples=200, n_targets=5)
    mlp = MLPRegressor(
        solver="lbfgs", hidden_layer_sizes=50, max_iter=200, random_state=1
    )
    mlp.fit(X, y)
    assert mlp.score(X, y) > 0.9


def test_partial_fit_classes_error():
    # Tests that passing different classes to partial_fit raises an error
    X = [[3, 2]]
    y = [0]
    clf = MLPClassifier(solver="sgd")
    clf.partial_fit(X, y, classes=[0, 1])
    with pytest.raises(ValueError):
        clf.partial_fit(X, y, classes=[1, 2])


def test_partial_fit_classification():
    # Test partial_fit on classification.
    # `partial_fit` should yield the same results as 'fit' for binary and
    # multi-class classification.
    for X, y in classification_datasets:
        mlp = MLPClassifier(
            solver="sgd",
            max_iter=100,
            random_state=1,
            tol=0,
            alpha=1e-5,
            learning_rate_init=0.2,
        )

        with ignore_warnings(category=ConvergenceWarning):
            mlp.fit(X, y)
        pred1 = mlp.predict(X)
        mlp = MLPClassifier(
            solver="sgd", random_state=1, alpha=1e-5, learning_rate_init=0.2
        )
        for i in range(100):
            mlp.partial_fit(X, y, classes=np.unique(y))
        pred2 = mlp.predict(X)
        assert_array_equal(pred1, pred2)
        assert mlp.score(X, y) > 0.95


def test_partial_fit_unseen_classes():
    # Non regression test for bug 6994
    # Tests for labeling errors in partial fit

    clf = MLPClassifier(random_state=0)
    clf.partial_fit([[1], [2], [3]], ["a", "b", "c"], classes=["a", "b", "c", "d"])
    clf.partial_fit([[4]], ["d"])
    assert clf.score([[1], [2], [3], [4]], ["a", "b", "c", "d"]) > 0


def test_partial_fit_regression():
    # Test partial_fit on regression.
    # `partial_fit` should yield the same results as 'fit' for regression.
    X = X_reg
    y = y_reg

    for momentum in [0, 0.9]:
        mlp = MLPRegressor(
            solver="sgd",
            max_iter=100,
            activation="relu",
            random_state=1,
            learning_rate_init=0.01,
            batch_size=X.shape[0],
            momentum=momentum,
        )
        with warnings.catch_warnings(record=True):
            # catch convergence warning
            mlp.fit(X, y)
        pred1 = mlp.predict(X)
        mlp = MLPRegressor(
            solver="sgd",
            activation="relu",
            learning_rate_init=0.01,
            random_state=1,
            batch_size=X.shape[0],
            momentum=momentum,
        )
        for i in range(100):
            mlp.partial_fit(X, y)

        pred2 = mlp.predict(X)
        assert_allclose(pred1, pred2)
        score = mlp.score(X, y)
        assert score > 0.65


def test_partial_fit_errors():
    # Test partial_fit error handling.
    X = [[3, 2], [1, 6]]
    y = [1, 0]

    # no classes passed
    with pytest.raises(ValueError):
        MLPClassifier(solver="sgd").partial_fit(X, y, classes=[2])

    # lbfgs doesn't support partial_fit
    assert not hasattr(MLPClassifier(solver="lbfgs"), "partial_fit")


@pytest.mark.parametrize(
    "args",
    [
        {"hidden_layer_sizes": -1},
        {"max_iter": -1},
        {"shuffle": "true"},
        {"alpha": -1},
        {"learning_rate_init": -1},
        {"momentum": 2},
        {"momentum": -0.5},
        {"nesterovs_momentum": "invalid"},
        {"early_stopping": "invalid"},
        {"validation_fraction": 1},
        {"validation_fraction": -0.5},
        {"beta_1": 1},
        {"beta_1": -0.5},
        {"beta_2": 1},
        {"beta_2": -0.5},
        {"epsilon": -0.5},
        {"n_iter_no_change": -1},
        {"solver": "hadoken"},
        {"learning_rate": "converge"},
        {"activation": "cloak"},
    ],
)
def test_params_errors(args):
    # Test that invalid parameters raise value error
    X = [[3, 2], [1, 6]]
    y = [1, 0]
    clf = MLPClassifier

    with pytest.raises(ValueError):
        clf(**args).fit(X, y)


def test_predict_proba_binary():
    # Test that predict_proba works as expected for binary class.
    X = X_digits_binary[:50]
    y = y_digits_binary[:50]

    clf = MLPClassifier(hidden_layer_sizes=5, activation="logistic", random_state=1)
    with ignore_warnings(category=ConvergenceWarning):
        clf.fit(X, y)
    y_proba = clf.predict_proba(X)
    y_log_proba = clf.predict_log_proba(X)

    (n_samples, n_classes) = y.shape[0], 2

    proba_max = y_proba.argmax(axis=1)
    proba_log_max = y_log_proba.argmax(axis=1)

    assert y_proba.shape == (n_samples, n_classes)
    assert_array_equal(proba_max, proba_log_max)
    assert_allclose(y_log_proba, np.log(y_proba))

    assert roc_auc_score(y, y_proba[:, 1]) == 1.0


def test_predict_proba_multiclass():
    # Test that predict_proba works as expected for multi class.
    X = X_digits_multi[:10]
    y = y_digits_multi[:10]

    clf = MLPClassifier(hidden_layer_sizes=5)
    with ignore_warnings(category=ConvergenceWarning):
        clf.fit(X, y)
    y_proba = clf.predict_proba(X)
    y_log_proba = clf.predict_log_proba(X)

    (n_samples, n_classes) = y.shape[0], np.unique(y).size

    proba_max = y_proba.argmax(axis=1)
    proba_log_max = y_log_proba.argmax(axis=1)

    assert y_proba.shape == (n_samples, n_classes)
    assert_array_equal(proba_max, proba_log_max)
    assert_allclose(y_log_proba, np.log(y_proba))


def test_predict_proba_multilabel():
    # Test that predict_proba works as expected for multilabel.
    # Multilabel should not use softmax which makes probabilities sum to 1
    X, Y = make_multilabel_classification(
        n_samples=50, random_state=0, return_indicator=True
    )
    n_samples, n_classes = Y.shape

    clf = MLPClassifier(solver="lbfgs", hidden_layer_sizes=30, random_state=0)
    clf.fit(X, Y)
    y_proba = clf.predict_proba(X)

    assert y_proba.shape == (n_samples, n_classes)
    assert_array_equal(y_proba > 0.5, Y)

    y_log_proba = clf.predict_log_proba(X)
    proba_max = y_proba.argmax(axis=1)
    proba_log_max = y_log_proba.argmax(axis=1)

    assert (y_proba.sum(1) - 1).dot(y_proba.sum(1) - 1) > 1e-10
    assert_array_equal(proba_max, proba_log_max)
    assert_allclose(y_log_proba, np.log(y_proba))


def test_shuffle():
    # Test that the shuffle parameter affects the training process (it should)
    X, y = make_regression(n_samples=50, n_features=5, n_targets=1, random_state=0)

    # The coefficients will be identical if both do or do not shuffle
    for shuffle in [True, False]:
        mlp1 = MLPRegressor(
            hidden_layer_sizes=1,
            max_iter=1,
            batch_size=1,
            random_state=0,
            shuffle=shuffle,
        )
        mlp2 = MLPRegressor(
            hidden_layer_sizes=1,
            max_iter=1,
            batch_size=1,
            random_state=0,
            shuffle=shuffle,
        )
        mlp1.fit(X, y)
        mlp2.fit(X, y)

        assert np.array_equal(mlp1.coefs_[0], mlp2.coefs_[0])

    # The coefficients will be slightly different if shuffle=True
    mlp1 = MLPRegressor(
        hidden_layer_sizes=1, max_iter=1, batch_size=1, random_state=0, shuffle=True
    )
    mlp2 = MLPRegressor(
        hidden_layer_sizes=1, max_iter=1, batch_size=1, random_state=0, shuffle=False
    )
    mlp1.fit(X, y)
    mlp2.fit(X, y)

    assert not np.array_equal(mlp1.coefs_[0], mlp2.coefs_[0])


def test_sparse_matrices():
    # Test that sparse and dense input matrices output the same results.
    X = X_digits_binary[:50]
    y = y_digits_binary[:50]
    X_sparse = csr_matrix(X)
    mlp = MLPClassifier(solver="lbfgs", hidden_layer_sizes=15, random_state=1)
    mlp.fit(X, y)
    pred1 = mlp.predict(X)
    mlp.fit(X_sparse, y)
    pred2 = mlp.predict(X_sparse)
    assert_almost_equal(pred1, pred2)
    pred1 = mlp.predict(X)
    pred2 = mlp.predict(X_sparse)
    assert_array_equal(pred1, pred2)


def test_tolerance():
    # Test tolerance.
    # It should force the solver to exit the loop when it converges.
    X = [[3, 2], [1, 6]]
    y = [1, 0]
    clf = MLPClassifier(tol=0.5, max_iter=3000, solver="sgd")
    clf.fit(X, y)
    assert clf.max_iter > clf.n_iter_


def test_verbose_sgd():
    # Test verbose.
    X = [[3, 2], [1, 6]]
    y = [1, 0]
    clf = MLPClassifier(solver="sgd", max_iter=2, verbose=10, hidden_layer_sizes=2)
    old_stdout = sys.stdout
    sys.stdout = output = StringIO()

    with ignore_warnings(category=ConvergenceWarning):
        clf.fit(X, y)
    clf.partial_fit(X, y)

    sys.stdout = old_stdout
    assert "Iteration" in output.getvalue()


def test_early_stopping():
    X = X_digits_binary[:100]
    y = y_digits_binary[:100]
    tol = 0.2
    clf = MLPClassifier(tol=tol, max_iter=3000, solver="sgd", early_stopping=True)
    clf.fit(X, y)
    assert clf.max_iter > clf.n_iter_

    valid_scores = clf.validation_scores_
    best_valid_score = clf.best_validation_score_
    assert max(valid_scores) == best_valid_score
    assert best_valid_score + tol > valid_scores[-2]
    assert best_valid_score + tol > valid_scores[-1]


def test_adaptive_learning_rate():
    X = [[3, 2], [1, 6]]
    y = [1, 0]
    clf = MLPClassifier(tol=0.5, max_iter=3000, solver="sgd", learning_rate="adaptive")
    clf.fit(X, y)
    assert clf.max_iter > clf.n_iter_
    assert 1e-6 > clf._optimizer.learning_rate


@ignore_warnings(category=RuntimeWarning)
def test_warm_start():
    X = X_iris
    y = y_iris

    y_2classes = np.array([0] * 75 + [1] * 75)
    y_3classes = np.array([0] * 40 + [1] * 40 + [2] * 70)
    y_3classes_alt = np.array([0] * 50 + [1] * 50 + [3] * 50)
    y_4classes = np.array([0] * 37 + [1] * 37 + [2] * 38 + [3] * 38)
    y_5classes = np.array([0] * 30 + [1] * 30 + [2] * 30 + [3] * 30 + [4] * 30)

    # No error raised
    clf = MLPClassifier(hidden_layer_sizes=2, solver="lbfgs", warm_start=True).fit(X, y)
    clf.fit(X, y)
    clf.fit(X, y_3classes)

    for y_i in (y_2classes, y_3classes_alt, y_4classes, y_5classes):
        clf = MLPClassifier(hidden_layer_sizes=2, solver="lbfgs", warm_start=True).fit(
            X, y
        )
        message = (
            "warm_start can only be used where `y` has the same "
            "classes as in the previous call to fit."
            " Previously got [0 1 2], `y` has %s"
            % np.unique(y_i)
        )
        with pytest.raises(ValueError, match=re.escape(message)):
            clf.fit(X, y_i)


@pytest.mark.parametrize("MLPEstimator", [MLPClassifier, MLPRegressor])
def test_warm_start_full_iteration(MLPEstimator):
    # Non-regression test for:
    # https://github.com/scikit-learn/scikit-learn/issues/16812
    # Check that the MLP estimator accomplish `max_iter` with a
    # warm started estimator.
    X, y = X_iris, y_iris
    max_iter = 3
    clf = MLPEstimator(
        hidden_layer_sizes=2, solver="sgd", warm_start=True, max_iter=max_iter
    )
    clf.fit(X, y)
    assert max_iter == clf.n_iter_
    clf.fit(X, y)
    assert 2 * max_iter == clf.n_iter_


def test_n_iter_no_change():
    # test n_iter_no_change using binary data set
    # the classifying fitting process is not prone to loss curve fluctuations
    X = X_digits_binary[:100]
    y = y_digits_binary[:100]
    tol = 0.01
    max_iter = 3000

    # test multiple n_iter_no_change
    for n_iter_no_change in [2, 5, 10, 50, 100]:
        clf = MLPClassifier(
            tol=tol, max_iter=max_iter, solver="sgd", n_iter_no_change=n_iter_no_change
        )
        clf.fit(X, y)

        # validate n_iter_no_change
        assert clf._no_improvement_count == n_iter_no_change + 1
        assert max_iter > clf.n_iter_


@ignore_warnings(category=ConvergenceWarning)
def test_n_iter_no_change_inf():
    # test n_iter_no_change using binary data set
    # the fitting process should go to max_iter iterations
    X = X_digits_binary[:100]
    y = y_digits_binary[:100]

    # set a ridiculous tolerance
    # this should always trigger _update_no_improvement_count()
    tol = 1e9

    # fit
    n_iter_no_change = np.inf
    max_iter = 3000
    clf = MLPClassifier(
        tol=tol, max_iter=max_iter, solver="sgd", n_iter_no_change=n_iter_no_change
    )
    clf.fit(X, y)

    # validate n_iter_no_change doesn't cause early stopping
    assert clf.n_iter_ == max_iter

    # validate _update_no_improvement_count() was always triggered
    assert clf._no_improvement_count == clf.n_iter_ - 1


def test_early_stopping_stratified():
    # Make sure data splitting for early stopping is stratified
    X = [[1, 2], [2, 3], [3, 4], [4, 5]]
    y = [0, 0, 0, 1]

    mlp = MLPClassifier(early_stopping=True)
    with pytest.raises(
        ValueError, match="The least populated class in y has only 1 member"
    ):
        mlp.fit(X, y)


def test_mlp_classifier_dtypes_casting():
    # Compare predictions for different dtypes
    mlp_64 = MLPClassifier(
        alpha=1e-5, hidden_layer_sizes=(5, 3), random_state=1, max_iter=50
    )
    mlp_64.fit(X_digits[:300], y_digits[:300])
    pred_64 = mlp_64.predict(X_digits[300:])
    proba_64 = mlp_64.predict_proba(X_digits[300:])

    mlp_32 = MLPClassifier(
        alpha=1e-5, hidden_layer_sizes=(5, 3), random_state=1, max_iter=50
    )
    mlp_32.fit(X_digits[:300].astype(np.float32), y_digits[:300])
    pred_32 = mlp_32.predict(X_digits[300:].astype(np.float32))
    proba_32 = mlp_32.predict_proba(X_digits[300:].astype(np.float32))

    assert_array_equal(pred_64, pred_32)
    assert_allclose(proba_64, proba_32, rtol=1e-02)


def test_mlp_regressor_dtypes_casting():
    mlp_64 = MLPRegressor(
        alpha=1e-5, hidden_layer_sizes=(5, 3), random_state=1, max_iter=50
    )
    mlp_64.fit(X_digits[:300], y_digits[:300])
    pred_64 = mlp_64.predict(X_digits[300:])

    mlp_32 = MLPRegressor(
        alpha=1e-5, hidden_layer_sizes=(5, 3), random_state=1, max_iter=50
    )
    mlp_32.fit(X_digits[:300].astype(np.float32), y_digits[:300])
    pred_32 = mlp_32.predict(X_digits[300:].astype(np.float32))

    assert_allclose(pred_64, pred_32, rtol=1e-04)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("Estimator", [MLPClassifier, MLPRegressor])
def test_mlp_param_dtypes(dtype, Estimator):
    # Checks if input dtype is used for network parameters
    # and predictions
    X, y = X_digits.astype(dtype), y_digits
    mlp = Estimator(alpha=1e-5, hidden_layer_sizes=(5, 3), random_state=1, max_iter=50)
    mlp.fit(X[:300], y[:300])
    pred = mlp.predict(X[300:])

    assert all([intercept.dtype == dtype for intercept in mlp.intercepts_])

    assert all([coef.dtype == dtype for coef in mlp.coefs_])

    if Estimator == MLPRegressor:
        assert pred.dtype == dtype


def test_mlp_loading_from_joblib_partial_fit(tmp_path):
    """Loading from MLP and partial fitting updates weights. Non-regression
    test for #19626."""
    pre_trained_estimator = MLPRegressor(
        hidden_layer_sizes=(42,), random_state=42, learning_rate_init=0.01, max_iter=200
    )
    features, target = [[2]], [4]

    # Fit on x=2, y=4
    pre_trained_estimator.fit(features, target)

    # dump and load model
    pickled_file = tmp_path / "mlp.pkl"
    joblib.dump(pre_trained_estimator, pickled_file)
    load_estimator = joblib.load(pickled_file)

    # Train for a more epochs on point x=2, y=1
    fine_tune_features, fine_tune_target = [[2]], [1]

    for _ in range(200):
        load_estimator.partial_fit(fine_tune_features, fine_tune_target)

    # finetuned model learned the new target
    predicted_value = load_estimator.predict(fine_tune_features)
    assert_allclose(predicted_value, fine_tune_target, rtol=1e-4)
