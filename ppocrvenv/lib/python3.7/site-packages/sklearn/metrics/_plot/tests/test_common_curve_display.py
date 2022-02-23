import pytest

from sklearn.base import ClassifierMixin, clone
from sklearn.compose import make_column_transformer
from sklearn.datasets import load_iris
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    DetCurveDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)


@pytest.fixture(scope="module")
def data():
    return load_iris(return_X_y=True)


@pytest.fixture(scope="module")
def data_binary(data):
    X, y = data
    return X[y < 2], y[y < 2]


@pytest.mark.parametrize(
    "Display", [DetCurveDisplay, PrecisionRecallDisplay, RocCurveDisplay]
)
def test_display_curve_error_non_binary(pyplot, data, Display):
    """Check that a proper error is raised when only binary classification is
    supported."""
    X, y = data
    clf = DecisionTreeClassifier().fit(X, y)

    msg = (
        "Expected 'estimator' to be a binary classifier, but got DecisionTreeClassifier"
    )
    with pytest.raises(ValueError, match=msg):
        Display.from_estimator(clf, X, y)


@pytest.mark.parametrize(
    "response_method, msg",
    [
        (
            "predict_proba",
            "response method predict_proba is not defined in MyClassifier",
        ),
        (
            "decision_function",
            "response method decision_function is not defined in MyClassifier",
        ),
        (
            "auto",
            "response method decision_function or predict_proba is not "
            "defined in MyClassifier",
        ),
        (
            "bad_method",
            "response_method must be 'predict_proba', 'decision_function' or 'auto'",
        ),
    ],
)
@pytest.mark.parametrize(
    "Display", [DetCurveDisplay, PrecisionRecallDisplay, RocCurveDisplay]
)
def test_display_curve_error_no_response(
    pyplot,
    data_binary,
    response_method,
    msg,
    Display,
):
    """Check that a proper error is raised when the response method requested
    is not defined for the given trained classifier."""
    X, y = data_binary

    class MyClassifier(ClassifierMixin):
        def fit(self, X, y):
            self.classes_ = [0, 1]
            return self

    clf = MyClassifier().fit(X, y)

    with pytest.raises(ValueError, match=msg):
        Display.from_estimator(clf, X, y, response_method=response_method)


@pytest.mark.parametrize(
    "Display", [DetCurveDisplay, PrecisionRecallDisplay, RocCurveDisplay]
)
@pytest.mark.parametrize("constructor_name", ["from_estimator", "from_predictions"])
def test_display_curve_estimator_name_multiple_calls(
    pyplot,
    data_binary,
    Display,
    constructor_name,
):
    """Check that passing `name` when calling `plot` will overwrite the original name
    in the legend."""
    X, y = data_binary
    clf_name = "my hand-crafted name"
    clf = LogisticRegression().fit(X, y)
    y_pred = clf.predict_proba(X)[:, 1]

    # safe guard for the binary if/else construction
    assert constructor_name in ("from_estimator", "from_predictions")

    if constructor_name == "from_estimator":
        disp = Display.from_estimator(clf, X, y, name=clf_name)
    else:
        disp = Display.from_predictions(y, y_pred, name=clf_name)
    assert disp.estimator_name == clf_name
    pyplot.close("all")
    disp.plot()
    assert clf_name in disp.line_.get_label()
    pyplot.close("all")
    clf_name = "another_name"
    disp.plot(name=clf_name)
    assert clf_name in disp.line_.get_label()


@pytest.mark.parametrize(
    "clf",
    [
        LogisticRegression(),
        make_pipeline(StandardScaler(), LogisticRegression()),
        make_pipeline(
            make_column_transformer((StandardScaler(), [0, 1])), LogisticRegression()
        ),
    ],
)
@pytest.mark.parametrize(
    "Display", [DetCurveDisplay, PrecisionRecallDisplay, RocCurveDisplay]
)
def test_display_curve_not_fitted_errors(pyplot, data_binary, clf, Display):
    """Check that a proper error is raised when the classifier is not
    fitted."""
    X, y = data_binary
    # clone since we parametrize the test and the classifier will be fitted
    # when testing the second and subsequent plotting function
    model = clone(clf)
    with pytest.raises(NotFittedError):
        Display.from_estimator(model, X, y)
    model.fit(X, y)
    disp = Display.from_estimator(model, X, y)
    assert model.__class__.__name__ in disp.line_.get_label()
    assert disp.estimator_name == model.__class__.__name__
