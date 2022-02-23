import numpy as np
import pytest

from sklearn.compose import make_column_transformer
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.utils import shuffle

from sklearn.metrics import PrecisionRecallDisplay, plot_precision_recall_curve

# TODO: Remove when https://github.com/numpy/numpy/issues/14397 is resolved
pytestmark = pytest.mark.filterwarnings(
    "ignore:In future, it will be an error for 'np.bool_':DeprecationWarning:"
    "matplotlib.*"
)


def test_precision_recall_display_validation(pyplot):
    """Check that we raise the proper error when validating parameters."""
    X, y = make_classification(
        n_samples=100, n_informative=5, n_classes=5, random_state=0
    )

    with pytest.raises(NotFittedError):
        PrecisionRecallDisplay.from_estimator(SVC(), X, y)

    regressor = SVR().fit(X, y)
    y_pred_regressor = regressor.predict(X)
    classifier = SVC(probability=True).fit(X, y)
    y_pred_classifier = classifier.predict_proba(X)[:, -1]

    err_msg = "PrecisionRecallDisplay.from_estimator only supports classifiers"
    with pytest.raises(ValueError, match=err_msg):
        PrecisionRecallDisplay.from_estimator(regressor, X, y)

    err_msg = "Expected 'estimator' to be a binary classifier, but got SVC"
    with pytest.raises(ValueError, match=err_msg):
        PrecisionRecallDisplay.from_estimator(classifier, X, y)

    err_msg = "{} format is not supported"
    with pytest.raises(ValueError, match=err_msg.format("continuous")):
        # Force `y_true` to be seen as a regression problem
        PrecisionRecallDisplay.from_predictions(y + 0.5, y_pred_classifier, pos_label=1)
    with pytest.raises(ValueError, match=err_msg.format("multiclass")):
        PrecisionRecallDisplay.from_predictions(y, y_pred_regressor, pos_label=1)

    err_msg = "Found input variables with inconsistent numbers of samples"
    with pytest.raises(ValueError, match=err_msg):
        PrecisionRecallDisplay.from_predictions(y, y_pred_classifier[::2])

    X, y = make_classification(n_classes=2, n_samples=50, random_state=0)
    y += 10
    classifier.fit(X, y)
    y_pred_classifier = classifier.predict_proba(X)[:, -1]
    err_msg = r"y_true takes value in {10, 11} and pos_label is not specified"
    with pytest.raises(ValueError, match=err_msg):
        PrecisionRecallDisplay.from_predictions(y, y_pred_classifier)


# FIXME: Remove in 1.2
def test_plot_precision_recall_curve_deprecation(pyplot):
    """Check that we raise a FutureWarning when calling
    `plot_precision_recall_curve`."""

    X, y = make_classification(random_state=0)
    clf = LogisticRegression().fit(X, y)
    deprecation_warning = "Function plot_precision_recall_curve is deprecated"
    with pytest.warns(FutureWarning, match=deprecation_warning):
        plot_precision_recall_curve(clf, X, y)


@pytest.mark.parametrize("constructor_name", ["from_estimator", "from_predictions"])
@pytest.mark.parametrize("response_method", ["predict_proba", "decision_function"])
def test_precision_recall_display_plotting(pyplot, constructor_name, response_method):
    """Check the overall plotting rendering."""
    X, y = make_classification(n_classes=2, n_samples=50, random_state=0)
    pos_label = 1

    classifier = LogisticRegression().fit(X, y)
    classifier.fit(X, y)

    y_pred = getattr(classifier, response_method)(X)
    y_pred = y_pred if y_pred.ndim == 1 else y_pred[:, pos_label]

    # safe guard for the binary if/else construction
    assert constructor_name in ("from_estimator", "from_predictions")

    if constructor_name == "from_estimator":
        display = PrecisionRecallDisplay.from_estimator(
            classifier, X, y, response_method=response_method
        )
    else:
        display = PrecisionRecallDisplay.from_predictions(
            y, y_pred, pos_label=pos_label
        )

    precision, recall, _ = precision_recall_curve(y, y_pred, pos_label=pos_label)
    average_precision = average_precision_score(y, y_pred, pos_label=pos_label)

    np.testing.assert_allclose(display.precision, precision)
    np.testing.assert_allclose(display.recall, recall)
    assert display.average_precision == pytest.approx(average_precision)

    import matplotlib as mpl

    assert isinstance(display.line_, mpl.lines.Line2D)
    assert isinstance(display.ax_, mpl.axes.Axes)
    assert isinstance(display.figure_, mpl.figure.Figure)

    assert display.ax_.get_xlabel() == "Recall (Positive label: 1)"
    assert display.ax_.get_ylabel() == "Precision (Positive label: 1)"

    # plotting passing some new parameters
    display.plot(alpha=0.8, name="MySpecialEstimator")
    expected_label = f"MySpecialEstimator (AP = {average_precision:0.2f})"
    assert display.line_.get_label() == expected_label
    assert display.line_.get_alpha() == pytest.approx(0.8)


@pytest.mark.parametrize(
    "constructor_name, default_label",
    [
        ("from_estimator", "LogisticRegression (AP = {:.2f})"),
        ("from_predictions", "Classifier (AP = {:.2f})"),
    ],
)
def test_precision_recall_display_name(pyplot, constructor_name, default_label):
    """Check the behaviour of the name parameters"""
    X, y = make_classification(n_classes=2, n_samples=100, random_state=0)
    pos_label = 1

    classifier = LogisticRegression().fit(X, y)
    classifier.fit(X, y)

    y_pred = classifier.predict_proba(X)[:, pos_label]

    # safe guard for the binary if/else construction
    assert constructor_name in ("from_estimator", "from_predictions")

    if constructor_name == "from_estimator":
        display = PrecisionRecallDisplay.from_estimator(classifier, X, y)
    else:
        display = PrecisionRecallDisplay.from_predictions(
            y, y_pred, pos_label=pos_label
        )

    average_precision = average_precision_score(y, y_pred, pos_label=pos_label)

    # check that the default name is used
    assert display.line_.get_label() == default_label.format(average_precision)

    # check that the name can be set
    display.plot(name="MySpecialEstimator")
    assert (
        display.line_.get_label()
        == f"MySpecialEstimator (AP = {average_precision:.2f})"
    )


@pytest.mark.parametrize(
    "clf",
    [
        make_pipeline(StandardScaler(), LogisticRegression()),
        make_pipeline(
            make_column_transformer((StandardScaler(), [0, 1])), LogisticRegression()
        ),
    ],
)
def test_precision_recall_display_pipeline(pyplot, clf):
    X, y = make_classification(n_classes=2, n_samples=50, random_state=0)
    with pytest.raises(NotFittedError):
        PrecisionRecallDisplay.from_estimator(clf, X, y)
    clf.fit(X, y)
    display = PrecisionRecallDisplay.from_estimator(clf, X, y)
    assert display.estimator_name == clf.__class__.__name__


def test_precision_recall_display_string_labels(pyplot):
    # regression test #15738
    cancer = load_breast_cancer()
    X, y = cancer.data, cancer.target_names[cancer.target]

    lr = make_pipeline(StandardScaler(), LogisticRegression())
    lr.fit(X, y)
    for klass in cancer.target_names:
        assert klass in lr.classes_
    display = PrecisionRecallDisplay.from_estimator(lr, X, y)

    y_pred = lr.predict_proba(X)[:, 1]
    avg_prec = average_precision_score(y, y_pred, pos_label=lr.classes_[1])

    assert display.average_precision == pytest.approx(avg_prec)
    assert display.estimator_name == lr.__class__.__name__

    err_msg = r"y_true takes value in {'benign', 'malignant'}"
    with pytest.raises(ValueError, match=err_msg):
        PrecisionRecallDisplay.from_predictions(y, y_pred)

    display = PrecisionRecallDisplay.from_predictions(
        y, y_pred, pos_label=lr.classes_[1]
    )
    assert display.average_precision == pytest.approx(avg_prec)


@pytest.mark.parametrize(
    "average_precision, estimator_name, expected_label",
    [
        (0.9, None, "AP = 0.90"),
        (None, "my_est", "my_est"),
        (0.8, "my_est2", "my_est2 (AP = 0.80)"),
    ],
)
def test_default_labels(pyplot, average_precision, estimator_name, expected_label):
    """Check the default labels used in the display."""
    precision = np.array([1, 0.5, 0])
    recall = np.array([0, 0.5, 1])
    display = PrecisionRecallDisplay(
        precision,
        recall,
        average_precision=average_precision,
        estimator_name=estimator_name,
    )
    display.plot()
    assert display.line_.get_label() == expected_label


@pytest.mark.parametrize("constructor_name", ["from_estimator", "from_predictions"])
@pytest.mark.parametrize("response_method", ["predict_proba", "decision_function"])
def test_plot_precision_recall_pos_label(pyplot, constructor_name, response_method):
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

    y_pred = getattr(classifier, response_method)(X_test)
    # we select the correcponding probability columns or reverse the decision
    #  function otherwise
    y_pred_cancer = -1 * y_pred if y_pred.ndim == 1 else y_pred[:, 0]
    y_pred_not_cancer = y_pred if y_pred.ndim == 1 else y_pred[:, 1]

    if constructor_name == "from_estimator":
        display = PrecisionRecallDisplay.from_estimator(
            classifier,
            X_test,
            y_test,
            pos_label="cancer",
            response_method=response_method,
        )
    else:
        display = PrecisionRecallDisplay.from_predictions(
            y_test,
            y_pred_cancer,
            pos_label="cancer",
        )
    # we should obtain the statistics of the "cancer" class
    avg_prec_limit = 0.65
    assert display.average_precision < avg_prec_limit
    assert -np.trapz(display.precision, display.recall) < avg_prec_limit

    # otherwise we should obtain the statistics of the "not cancer" class
    if constructor_name == "from_estimator":
        display = PrecisionRecallDisplay.from_estimator(
            classifier,
            X_test,
            y_test,
            response_method=response_method,
            pos_label="not cancer",
        )
    else:
        display = PrecisionRecallDisplay.from_predictions(
            y_test,
            y_pred_not_cancer,
            pos_label="not cancer",
        )
    avg_prec_limit = 0.95
    assert display.average_precision > avg_prec_limit
    assert -np.trapz(display.precision, display.recall) > avg_prec_limit
