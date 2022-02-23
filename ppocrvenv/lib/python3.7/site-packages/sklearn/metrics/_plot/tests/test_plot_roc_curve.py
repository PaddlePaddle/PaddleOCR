import pytest
import numpy as np
from numpy.testing import assert_allclose

from sklearn.metrics import plot_roc_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.compose import make_column_transformer

# TODO: Remove when https://github.com/numpy/numpy/issues/14397 is resolved
pytestmark = pytest.mark.filterwarnings(
    "ignore:In future, it will be an error for 'np.bool_':DeprecationWarning:"
    "matplotlib.*",
    "ignore:Function plot_roc_curve is deprecated",
)


@pytest.fixture(scope="module")
def data():
    return load_iris(return_X_y=True)


@pytest.fixture(scope="module")
def data_binary(data):
    X, y = data
    return X[y < 2], y[y < 2]


@pytest.mark.parametrize("response_method", ["predict_proba", "decision_function"])
@pytest.mark.parametrize("with_sample_weight", [True, False])
@pytest.mark.parametrize("drop_intermediate", [True, False])
@pytest.mark.parametrize("with_strings", [True, False])
def test_plot_roc_curve(
    pyplot,
    response_method,
    data_binary,
    with_sample_weight,
    drop_intermediate,
    with_strings,
):
    X, y = data_binary

    pos_label = None
    if with_strings:
        y = np.array(["c", "b"])[y]
        pos_label = "c"

    if with_sample_weight:
        rng = np.random.RandomState(42)
        sample_weight = rng.randint(1, 4, size=(X.shape[0]))
    else:
        sample_weight = None

    lr = LogisticRegression()
    lr.fit(X, y)

    viz = plot_roc_curve(
        lr,
        X,
        y,
        alpha=0.8,
        sample_weight=sample_weight,
        drop_intermediate=drop_intermediate,
    )

    y_pred = getattr(lr, response_method)(X)
    if y_pred.ndim == 2:
        y_pred = y_pred[:, 1]

    fpr, tpr, _ = roc_curve(
        y,
        y_pred,
        sample_weight=sample_weight,
        drop_intermediate=drop_intermediate,
        pos_label=pos_label,
    )

    assert_allclose(viz.roc_auc, auc(fpr, tpr))
    assert_allclose(viz.fpr, fpr)
    assert_allclose(viz.tpr, tpr)

    assert viz.estimator_name == "LogisticRegression"

    # cannot fail thanks to pyplot fixture
    import matplotlib as mpl  # noqal

    assert isinstance(viz.line_, mpl.lines.Line2D)
    assert viz.line_.get_alpha() == 0.8
    assert isinstance(viz.ax_, mpl.axes.Axes)
    assert isinstance(viz.figure_, mpl.figure.Figure)

    expected_label = "LogisticRegression (AUC = {:0.2f})".format(viz.roc_auc)
    assert viz.line_.get_label() == expected_label

    expected_pos_label = 1 if pos_label is None else pos_label
    expected_ylabel = f"True Positive Rate (Positive label: {expected_pos_label})"
    expected_xlabel = f"False Positive Rate (Positive label: {expected_pos_label})"

    assert viz.ax_.get_ylabel() == expected_ylabel
    assert viz.ax_.get_xlabel() == expected_xlabel


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
def test_roc_curve_not_fitted_errors(pyplot, data_binary, clf):
    X, y = data_binary
    with pytest.raises(NotFittedError):
        plot_roc_curve(clf, X, y)
    clf.fit(X, y)
    disp = plot_roc_curve(clf, X, y)
    assert clf.__class__.__name__ in disp.line_.get_label()
    assert disp.estimator_name == clf.__class__.__name__


@pytest.mark.parametrize("response_method", ["predict_proba", "decision_function"])
def test_plot_roc_curve_pos_label(pyplot, response_method):
    # check that we can provide the positive label and display the proper
    # statistics
    X, y = load_breast_cancer(return_X_y=True)
    # create an highly imbalanced
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

    disp = plot_roc_curve(
        classifier, X_test, y_test, pos_label="cancer", response_method=response_method
    )

    roc_auc_limit = 0.95679

    assert disp.roc_auc == pytest.approx(roc_auc_limit)
    assert np.trapz(disp.tpr, disp.fpr) == pytest.approx(roc_auc_limit)

    disp = plot_roc_curve(
        classifier,
        X_test,
        y_test,
        response_method=response_method,
    )

    assert disp.roc_auc == pytest.approx(roc_auc_limit)
    assert np.trapz(disp.tpr, disp.fpr) == pytest.approx(roc_auc_limit)
