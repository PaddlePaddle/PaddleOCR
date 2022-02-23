import pytest
import numpy as np
from numpy.testing import assert_allclose

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.datasets import make_classification
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.compose import make_column_transformer

pytestmark = pytest.mark.filterwarnings(
    # TODO: Remove when https://github.com/numpy/numpy/issues/14397 is resolved
    "ignore:In future, it will be an error for 'np.bool_':DeprecationWarning:"
    "matplotlib.*",
    # TODO: Remove in 1.2 (as well as all the tests below)
    "ignore:Function plot_precision_recall_curve is deprecated",
)


def test_errors(pyplot):
    X, y_multiclass = make_classification(
        n_classes=3, n_samples=50, n_informative=3, random_state=0
    )
    y_binary = y_multiclass == 0

    # Unfitted classifier
    binary_clf = DecisionTreeClassifier()
    with pytest.raises(NotFittedError):
        plot_precision_recall_curve(binary_clf, X, y_binary)
    binary_clf.fit(X, y_binary)

    multi_clf = DecisionTreeClassifier().fit(X, y_multiclass)

    # Fitted multiclass classifier with binary data
    msg = (
        "Expected 'estimator' to be a binary classifier, but got DecisionTreeClassifier"
    )
    with pytest.raises(ValueError, match=msg):
        plot_precision_recall_curve(multi_clf, X, y_binary)

    reg = DecisionTreeRegressor().fit(X, y_multiclass)
    msg = (
        "Expected 'estimator' to be a binary classifier, but got DecisionTreeRegressor"
    )
    with pytest.raises(ValueError, match=msg):
        plot_precision_recall_curve(reg, X, y_binary)


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
def test_error_bad_response(pyplot, response_method, msg):
    X, y = make_classification(n_classes=2, n_samples=50, random_state=0)

    class MyClassifier(ClassifierMixin, BaseEstimator):
        def fit(self, X, y):
            self.fitted_ = True
            self.classes_ = [0, 1]
            return self

    clf = MyClassifier().fit(X, y)

    with pytest.raises(ValueError, match=msg):
        plot_precision_recall_curve(clf, X, y, response_method=response_method)


@pytest.mark.parametrize("response_method", ["predict_proba", "decision_function"])
@pytest.mark.parametrize("with_sample_weight", [True, False])
def test_plot_precision_recall(pyplot, response_method, with_sample_weight):
    X, y = make_classification(n_classes=2, n_samples=50, random_state=0)

    lr = LogisticRegression().fit(X, y)

    if with_sample_weight:
        rng = np.random.RandomState(42)
        sample_weight = rng.randint(0, 4, size=X.shape[0])
    else:
        sample_weight = None

    disp = plot_precision_recall_curve(
        lr,
        X,
        y,
        alpha=0.8,
        response_method=response_method,
        sample_weight=sample_weight,
    )

    y_score = getattr(lr, response_method)(X)
    if response_method == "predict_proba":
        y_score = y_score[:, 1]

    prec, recall, _ = precision_recall_curve(y, y_score, sample_weight=sample_weight)
    avg_prec = average_precision_score(y, y_score, sample_weight=sample_weight)

    assert_allclose(disp.precision, prec)
    assert_allclose(disp.recall, recall)
    assert disp.average_precision == pytest.approx(avg_prec)

    assert disp.estimator_name == "LogisticRegression"

    # cannot fail thanks to pyplot fixture
    import matplotlib as mpl  # noqa

    assert isinstance(disp.line_, mpl.lines.Line2D)
    assert disp.line_.get_alpha() == 0.8
    assert isinstance(disp.ax_, mpl.axes.Axes)
    assert isinstance(disp.figure_, mpl.figure.Figure)

    expected_label = "LogisticRegression (AP = {:0.2f})".format(avg_prec)
    assert disp.line_.get_label() == expected_label
    assert disp.ax_.get_xlabel() == "Recall (Positive label: 1)"
    assert disp.ax_.get_ylabel() == "Precision (Positive label: 1)"

    # draw again with another label
    disp.plot(name="MySpecialEstimator")
    expected_label = "MySpecialEstimator (AP = {:0.2f})".format(avg_prec)
    assert disp.line_.get_label() == expected_label


@pytest.mark.parametrize(
    "clf",
    [
        make_pipeline(StandardScaler(), LogisticRegression()),
        make_pipeline(
            make_column_transformer((StandardScaler(), [0, 1])), LogisticRegression()
        ),
    ],
)
def test_precision_recall_curve_pipeline(pyplot, clf):
    X, y = make_classification(n_classes=2, n_samples=50, random_state=0)
    with pytest.raises(NotFittedError):
        plot_precision_recall_curve(clf, X, y)
    clf.fit(X, y)
    disp = plot_precision_recall_curve(clf, X, y)
    assert disp.estimator_name == clf.__class__.__name__


def test_precision_recall_curve_string_labels(pyplot):
    # regression test #15738
    cancer = load_breast_cancer()
    X = cancer.data
    y = cancer.target_names[cancer.target]

    lr = make_pipeline(StandardScaler(), LogisticRegression())
    lr.fit(X, y)
    for klass in cancer.target_names:
        assert klass in lr.classes_
    disp = plot_precision_recall_curve(lr, X, y)

    y_pred = lr.predict_proba(X)[:, 1]
    avg_prec = average_precision_score(y, y_pred, pos_label=lr.classes_[1])

    assert disp.average_precision == pytest.approx(avg_prec)
    assert disp.estimator_name == lr.__class__.__name__


def test_plot_precision_recall_curve_estimator_name_multiple_calls(pyplot):
    # non-regression test checking that the `name` used when calling
    # `plot_precision_recall_curve` is used as well when calling `disp.plot()`
    X, y = make_classification(n_classes=2, n_samples=50, random_state=0)
    clf_name = "my hand-crafted name"
    clf = LogisticRegression().fit(X, y)
    disp = plot_precision_recall_curve(clf, X, y, name=clf_name)
    assert disp.estimator_name == clf_name
    pyplot.close("all")
    disp.plot()
    assert clf_name in disp.line_.get_label()
    pyplot.close("all")
    clf_name = "another_name"
    disp.plot(name=clf_name)
    assert clf_name in disp.line_.get_label()


@pytest.mark.parametrize("response_method", ["predict_proba", "decision_function"])
def test_plot_precision_recall_pos_label(pyplot, response_method):
    # check that we can provide the positive label and display the proper
    # statistics
    X, y = load_breast_cancer(return_X_y=True)
    # create an highly imbalanced version of the breast cancer dataset
    idx_positive = np.flatnonzero(y == 1)
    idx_negative = np.flatnonzero(y == 0)
    idx_selected = np.hstack([idx_negative, idx_positive[:25]])
    X, y = X[idx_selected], y[idx_selected]
    X, y = shuffle(X, y, random_state=42)
    # only use 2 features to make the problem even harder
    X = X[:, :2]
    y = np.array(["cancer" if c == 1 else "not cancer" for c in y], dtype=object)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        stratify=y,
        random_state=0,
    )

    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)

    # sanity check to be sure the positive class is classes_[0] and that we
    # are betrayed by the class imbalance
    assert classifier.classes_.tolist() == ["cancer", "not cancer"]

    disp = plot_precision_recall_curve(
        classifier, X_test, y_test, pos_label="cancer", response_method=response_method
    )
    # we should obtain the statistics of the "cancer" class
    avg_prec_limit = 0.65
    assert disp.average_precision < avg_prec_limit
    assert -np.trapz(disp.precision, disp.recall) < avg_prec_limit

    # otherwise we should obtain the statistics of the "not cancer" class
    disp = plot_precision_recall_curve(
        classifier,
        X_test,
        y_test,
        response_method=response_method,
    )
    avg_prec_limit = 0.95
    assert disp.average_precision > avg_prec_limit
    assert -np.trapz(disp.precision, disp.recall) > avg_prec_limit
