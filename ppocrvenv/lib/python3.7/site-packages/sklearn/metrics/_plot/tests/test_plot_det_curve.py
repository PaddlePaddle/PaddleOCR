# TODO: remove this file when plot_det_curve will be deprecated in 1.2
import pytest
import numpy as np
from numpy.testing import assert_allclose

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import det_curve
from sklearn.metrics import plot_det_curve


@pytest.fixture(scope="module")
def data():
    return load_iris(return_X_y=True)


@pytest.fixture(scope="module")
def data_binary(data):
    X, y = data
    return X[y < 2], y[y < 2]


@pytest.mark.filterwarnings("ignore: Function plot_det_curve is deprecated")
@pytest.mark.parametrize("response_method", ["predict_proba", "decision_function"])
@pytest.mark.parametrize("with_sample_weight", [True, False])
@pytest.mark.parametrize("with_strings", [True, False])
def test_plot_det_curve(
    pyplot, response_method, data_binary, with_sample_weight, with_strings
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

    viz = plot_det_curve(
        lr,
        X,
        y,
        alpha=0.8,
        sample_weight=sample_weight,
    )

    y_pred = getattr(lr, response_method)(X)
    if y_pred.ndim == 2:
        y_pred = y_pred[:, 1]

    fpr, fnr, _ = det_curve(
        y,
        y_pred,
        sample_weight=sample_weight,
        pos_label=pos_label,
    )

    assert_allclose(viz.fpr, fpr)
    assert_allclose(viz.fnr, fnr)

    assert viz.estimator_name == "LogisticRegression"

    # cannot fail thanks to pyplot fixture
    import matplotlib as mpl  # noqal

    assert isinstance(viz.line_, mpl.lines.Line2D)
    assert viz.line_.get_alpha() == 0.8
    assert isinstance(viz.ax_, mpl.axes.Axes)
    assert isinstance(viz.figure_, mpl.figure.Figure)
    assert viz.line_.get_label() == "LogisticRegression"

    expected_pos_label = 1 if pos_label is None else pos_label
    expected_ylabel = f"False Negative Rate (Positive label: {expected_pos_label})"
    expected_xlabel = f"False Positive Rate (Positive label: {expected_pos_label})"
    assert viz.ax_.get_ylabel() == expected_ylabel
    assert viz.ax_.get_xlabel() == expected_xlabel
