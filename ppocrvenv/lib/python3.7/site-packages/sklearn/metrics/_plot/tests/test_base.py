import numpy as np
import pytest

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from sklearn.metrics._plot.base import _get_response


@pytest.mark.parametrize(
    "estimator, err_msg, params",
    [
        (
            DecisionTreeRegressor(),
            "Expected 'estimator' to be a binary classifier",
            {"response_method": "auto"},
        ),
        (
            DecisionTreeClassifier(),
            "The class provided by 'pos_label' is unknown.",
            {"response_method": "auto", "pos_label": "unknown"},
        ),
        (
            DecisionTreeClassifier(),
            "fit on multiclass",
            {"response_method": "predict_proba"},
        ),
    ],
)
def test_get_response_error(estimator, err_msg, params):
    """Check that we raise the proper error messages in `_get_response`."""
    X, y = load_iris(return_X_y=True)

    estimator.fit(X, y)
    with pytest.raises(ValueError, match=err_msg):
        _get_response(X, estimator, **params)


def test_get_response_predict_proba():
    """Check the behaviour of `_get_response` using `predict_proba`."""
    X, y = load_iris(return_X_y=True)
    X_binary, y_binary = X[:100], y[:100]

    classifier = DecisionTreeClassifier().fit(X_binary, y_binary)
    y_proba, pos_label = _get_response(
        X_binary, classifier, response_method="predict_proba"
    )
    np.testing.assert_allclose(y_proba, classifier.predict_proba(X_binary)[:, 1])
    assert pos_label == 1

    y_proba, pos_label = _get_response(
        X_binary, classifier, response_method="predict_proba", pos_label=0
    )
    np.testing.assert_allclose(y_proba, classifier.predict_proba(X_binary)[:, 0])
    assert pos_label == 0


def test_get_response_decision_function():
    """Check the behaviour of `get_response` using `decision_function`."""
    X, y = load_iris(return_X_y=True)
    X_binary, y_binary = X[:100], y[:100]

    classifier = LogisticRegression().fit(X_binary, y_binary)
    y_score, pos_label = _get_response(
        X_binary, classifier, response_method="decision_function"
    )
    np.testing.assert_allclose(y_score, classifier.decision_function(X_binary))
    assert pos_label == 1

    y_score, pos_label = _get_response(
        X_binary, classifier, response_method="decision_function", pos_label=0
    )
    np.testing.assert_allclose(y_score, classifier.decision_function(X_binary) * -1)
    assert pos_label == 0
