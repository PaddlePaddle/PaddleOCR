import pytest

from sklearn.base import ClassifierMixin
from sklearn.base import clone
from sklearn.compose import make_column_transformer
from sklearn.datasets import load_iris
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import plot_det_curve
from sklearn.metrics import plot_roc_curve

pytestmark = pytest.mark.filterwarnings(
    "ignore:Function plot_roc_curve is deprecated",
)


@pytest.fixture(scope="module")
def data():
    return load_iris(return_X_y=True)


@pytest.fixture(scope="module")
def data_binary(data):
    X, y = data
    return X[y < 2], y[y < 2]


@pytest.mark.filterwarnings("ignore: Function plot_det_curve is deprecated")
@pytest.mark.parametrize("plot_func", [plot_det_curve, plot_roc_curve])
def test_plot_curve_error_non_binary(pyplot, data, plot_func):
    X, y = data
    clf = DecisionTreeClassifier()
    clf.fit(X, y)

    msg = (
        "Expected 'estimator' to be a binary classifier, but got DecisionTreeClassifier"
    )
    with pytest.raises(ValueError, match=msg):
        plot_func(clf, X, y)


@pytest.mark.filterwarnings("ignore: Function plot_det_curve is deprecated")
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
@pytest.mark.parametrize("plot_func", [plot_det_curve, plot_roc_curve])
def test_plot_curve_error_no_response(
    pyplot,
    data_binary,
    response_method,
    msg,
    plot_func,
):
    X, y = data_binary

    class MyClassifier(ClassifierMixin):
        def fit(self, X, y):
            self.classes_ = [0, 1]
            return self

    clf = MyClassifier().fit(X, y)

    with pytest.raises(ValueError, match=msg):
        plot_func(clf, X, y, response_method=response_method)


@pytest.mark.filterwarnings("ignore: Function plot_det_curve is deprecated")
@pytest.mark.parametrize("plot_func", [plot_det_curve, plot_roc_curve])
def test_plot_curve_estimator_name_multiple_calls(pyplot, data_binary, plot_func):
    # non-regression test checking that the `name` used when calling
    # `plot_func` is used as well when calling `disp.plot()`
    X, y = data_binary
    clf_name = "my hand-crafted name"
    clf = LogisticRegression().fit(X, y)
    disp = plot_func(clf, X, y, name=clf_name)
    assert disp.estimator_name == clf_name
    pyplot.close("all")
    disp.plot()
    assert clf_name in disp.line_.get_label()
    pyplot.close("all")
    clf_name = "another_name"
    disp.plot(name=clf_name)
    assert clf_name in disp.line_.get_label()


@pytest.mark.filterwarnings("ignore: Function plot_det_curve is deprecated")
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
@pytest.mark.parametrize("plot_func", [plot_det_curve, plot_roc_curve])
def test_plot_det_curve_not_fitted_errors(pyplot, data_binary, clf, plot_func):
    X, y = data_binary
    # clone since we parametrize the test and the classifier will be fitted
    # when testing the second and subsequent plotting function
    model = clone(clf)
    with pytest.raises(NotFittedError):
        plot_func(model, X, y)
    model.fit(X, y)
    disp = plot_func(model, X, y)
    assert model.__class__.__name__ in disp.line_.get_label()
    assert disp.estimator_name == model.__class__.__name__
