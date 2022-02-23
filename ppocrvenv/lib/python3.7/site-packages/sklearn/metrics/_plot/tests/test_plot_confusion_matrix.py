# TODO: remove this file when plot_confusion_matrix will be deprecated in 1.2
import pytest
import numpy as np
from numpy.testing import assert_allclose
from numpy.testing import assert_array_equal

from sklearn.compose import make_column_transformer
from sklearn.datasets import make_classification
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


# TODO: Remove when https://github.com/numpy/numpy/issues/14397 is resolved
pytestmark = pytest.mark.filterwarnings(
    "ignore:In future, it will be an error for 'np.bool_':DeprecationWarning:"
    "matplotlib.*"
)


@pytest.fixture(scope="module")
def n_classes():
    return 5


@pytest.fixture(scope="module")
def data(n_classes):
    X, y = make_classification(
        n_samples=100, n_informative=5, n_classes=n_classes, random_state=0
    )
    return X, y


@pytest.fixture(scope="module")
def fitted_clf(data):
    return SVC(kernel="linear", C=0.01).fit(*data)


@pytest.fixture(scope="module")
def y_pred(data, fitted_clf):
    X, _ = data
    return fitted_clf.predict(X)


@pytest.mark.filterwarnings("ignore: Function plot_confusion_matrix is deprecated")
def test_error_on_regressor(pyplot, data):
    X, y = data
    est = SVR().fit(X, y)

    msg = "plot_confusion_matrix only supports classifiers"
    with pytest.raises(ValueError, match=msg):
        plot_confusion_matrix(est, X, y)


@pytest.mark.filterwarnings("ignore: Function plot_confusion_matrix is deprecated")
def test_error_on_invalid_option(pyplot, fitted_clf, data):
    X, y = data
    msg = r"normalize must be one of \{'true', 'pred', 'all', " r"None\}"

    with pytest.raises(ValueError, match=msg):
        plot_confusion_matrix(fitted_clf, X, y, normalize="invalid")


@pytest.mark.filterwarnings("ignore: Function plot_confusion_matrix is deprecated")
@pytest.mark.parametrize("with_labels", [True, False])
@pytest.mark.parametrize("with_display_labels", [True, False])
def test_plot_confusion_matrix_custom_labels(
    pyplot, data, y_pred, fitted_clf, n_classes, with_labels, with_display_labels
):
    X, y = data
    ax = pyplot.gca()
    labels = [2, 1, 0, 3, 4] if with_labels else None
    display_labels = ["b", "d", "a", "e", "f"] if with_display_labels else None

    cm = confusion_matrix(y, y_pred, labels=labels)
    disp = plot_confusion_matrix(
        fitted_clf, X, y, ax=ax, display_labels=display_labels, labels=labels
    )

    assert_allclose(disp.confusion_matrix, cm)

    if with_display_labels:
        expected_display_labels = display_labels
    elif with_labels:
        expected_display_labels = labels
    else:
        expected_display_labels = list(range(n_classes))

    expected_display_labels_str = [str(name) for name in expected_display_labels]

    x_ticks = [tick.get_text() for tick in disp.ax_.get_xticklabels()]
    y_ticks = [tick.get_text() for tick in disp.ax_.get_yticklabels()]

    assert_array_equal(disp.display_labels, expected_display_labels)
    assert_array_equal(x_ticks, expected_display_labels_str)
    assert_array_equal(y_ticks, expected_display_labels_str)


@pytest.mark.filterwarnings("ignore: Function plot_confusion_matrix is deprecated")
@pytest.mark.parametrize("normalize", ["true", "pred", "all", None])
@pytest.mark.parametrize("include_values", [True, False])
def test_plot_confusion_matrix(
    pyplot, data, y_pred, n_classes, fitted_clf, normalize, include_values
):
    X, y = data
    ax = pyplot.gca()
    cmap = "plasma"
    cm = confusion_matrix(y, y_pred)
    disp = plot_confusion_matrix(
        fitted_clf,
        X,
        y,
        normalize=normalize,
        cmap=cmap,
        ax=ax,
        include_values=include_values,
    )

    assert disp.ax_ == ax

    if normalize == "true":
        cm = cm / cm.sum(axis=1, keepdims=True)
    elif normalize == "pred":
        cm = cm / cm.sum(axis=0, keepdims=True)
    elif normalize == "all":
        cm = cm / cm.sum()

    assert_allclose(disp.confusion_matrix, cm)
    import matplotlib as mpl

    assert isinstance(disp.im_, mpl.image.AxesImage)
    assert disp.im_.get_cmap().name == cmap
    assert isinstance(disp.ax_, pyplot.Axes)
    assert isinstance(disp.figure_, pyplot.Figure)

    assert disp.ax_.get_ylabel() == "True label"
    assert disp.ax_.get_xlabel() == "Predicted label"

    x_ticks = [tick.get_text() for tick in disp.ax_.get_xticklabels()]
    y_ticks = [tick.get_text() for tick in disp.ax_.get_yticklabels()]

    expected_display_labels = list(range(n_classes))

    expected_display_labels_str = [str(name) for name in expected_display_labels]

    assert_array_equal(disp.display_labels, expected_display_labels)
    assert_array_equal(x_ticks, expected_display_labels_str)
    assert_array_equal(y_ticks, expected_display_labels_str)

    image_data = disp.im_.get_array().data
    assert_allclose(image_data, cm)

    if include_values:
        assert disp.text_.shape == (n_classes, n_classes)
        fmt = ".2g"
        expected_text = np.array([format(v, fmt) for v in cm.ravel(order="C")])
        text_text = np.array([t.get_text() for t in disp.text_.ravel(order="C")])
        assert_array_equal(expected_text, text_text)
    else:
        assert disp.text_ is None


@pytest.mark.filterwarnings("ignore: Function plot_confusion_matrix is deprecated")
def test_confusion_matrix_display(pyplot, data, fitted_clf, y_pred, n_classes):
    X, y = data

    cm = confusion_matrix(y, y_pred)
    disp = plot_confusion_matrix(
        fitted_clf,
        X,
        y,
        normalize=None,
        include_values=True,
        cmap="viridis",
        xticks_rotation=45.0,
    )

    assert_allclose(disp.confusion_matrix, cm)
    assert disp.text_.shape == (n_classes, n_classes)

    rotations = [tick.get_rotation() for tick in disp.ax_.get_xticklabels()]
    assert_allclose(rotations, 45.0)

    image_data = disp.im_.get_array().data
    assert_allclose(image_data, cm)

    disp.plot(cmap="plasma")
    assert disp.im_.get_cmap().name == "plasma"

    disp.plot(include_values=False)
    assert disp.text_ is None

    disp.plot(xticks_rotation=90.0)
    rotations = [tick.get_rotation() for tick in disp.ax_.get_xticklabels()]
    assert_allclose(rotations, 90.0)

    disp.plot(values_format="e")
    expected_text = np.array([format(v, "e") for v in cm.ravel(order="C")])
    text_text = np.array([t.get_text() for t in disp.text_.ravel(order="C")])
    assert_array_equal(expected_text, text_text)


def test_confusion_matrix_contrast(pyplot):
    # make sure text color is appropriate depending on background

    cm = np.eye(2) / 2
    disp = ConfusionMatrixDisplay(cm, display_labels=[0, 1])

    disp.plot(cmap=pyplot.cm.gray)
    # diagonal text is black
    assert_allclose(disp.text_[0, 0].get_color(), [0.0, 0.0, 0.0, 1.0])
    assert_allclose(disp.text_[1, 1].get_color(), [0.0, 0.0, 0.0, 1.0])

    # off-diagonal text is white
    assert_allclose(disp.text_[0, 1].get_color(), [1.0, 1.0, 1.0, 1.0])
    assert_allclose(disp.text_[1, 0].get_color(), [1.0, 1.0, 1.0, 1.0])

    disp.plot(cmap=pyplot.cm.gray_r)
    # diagonal text is white
    assert_allclose(disp.text_[0, 1].get_color(), [0.0, 0.0, 0.0, 1.0])
    assert_allclose(disp.text_[1, 0].get_color(), [0.0, 0.0, 0.0, 1.0])

    # off-diagonal text is black
    assert_allclose(disp.text_[0, 0].get_color(), [1.0, 1.0, 1.0, 1.0])
    assert_allclose(disp.text_[1, 1].get_color(), [1.0, 1.0, 1.0, 1.0])

    # Regression test for #15920
    cm = np.array([[19, 34], [32, 58]])
    disp = ConfusionMatrixDisplay(cm, display_labels=[0, 1])

    disp.plot(cmap=pyplot.cm.Blues)
    min_color = pyplot.cm.Blues(0)
    max_color = pyplot.cm.Blues(255)
    assert_allclose(disp.text_[0, 0].get_color(), max_color)
    assert_allclose(disp.text_[0, 1].get_color(), max_color)
    assert_allclose(disp.text_[1, 0].get_color(), max_color)
    assert_allclose(disp.text_[1, 1].get_color(), min_color)


@pytest.mark.filterwarnings("ignore: Function plot_confusion_matrix is deprecated")
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
def test_confusion_matrix_pipeline(pyplot, clf, data, n_classes):
    X, y = data
    with pytest.raises(NotFittedError):
        plot_confusion_matrix(clf, X, y)
    clf.fit(X, y)
    y_pred = clf.predict(X)

    disp = plot_confusion_matrix(clf, X, y)
    cm = confusion_matrix(y, y_pred)

    assert_allclose(disp.confusion_matrix, cm)
    assert disp.text_.shape == (n_classes, n_classes)


@pytest.mark.filterwarnings("ignore: Function plot_confusion_matrix is deprecated")
@pytest.mark.parametrize("colorbar", [True, False])
def test_plot_confusion_matrix_colorbar(pyplot, data, fitted_clf, colorbar):
    X, y = data

    def _check_colorbar(disp, has_colorbar):
        if has_colorbar:
            assert disp.im_.colorbar is not None
            assert disp.im_.colorbar.__class__.__name__ == "Colorbar"
        else:
            assert disp.im_.colorbar is None

    disp = plot_confusion_matrix(fitted_clf, X, y, colorbar=colorbar)
    _check_colorbar(disp, colorbar)
    # attempt a plot with the opposite effect of colorbar
    disp.plot(colorbar=not colorbar)
    _check_colorbar(disp, not colorbar)


@pytest.mark.filterwarnings("ignore: Function plot_confusion_matrix is deprecated")
@pytest.mark.parametrize("values_format", ["e", "n"])
def test_confusion_matrix_text_format(
    pyplot, data, y_pred, n_classes, fitted_clf, values_format
):
    # Make sure plot text is formatted with 'values_format'.
    X, y = data
    cm = confusion_matrix(y, y_pred)
    disp = plot_confusion_matrix(
        fitted_clf, X, y, include_values=True, values_format=values_format
    )

    assert disp.text_.shape == (n_classes, n_classes)

    expected_text = np.array([format(v, values_format) for v in cm.ravel()])
    text_text = np.array([t.get_text() for t in disp.text_.ravel()])
    assert_array_equal(expected_text, text_text)


def test_confusion_matrix_standard_format(pyplot):
    cm = np.array([[10000000, 0], [123456, 12345678]])
    plotted_text = ConfusionMatrixDisplay(cm, display_labels=[False, True]).plot().text_
    # Values should be shown as whole numbers 'd',
    # except the first number which should be shown as 1e+07 (longer length)
    # and the last number will be shown as 1.2e+07 (longer length)
    test = [t.get_text() for t in plotted_text.ravel()]
    assert test == ["1e+07", "0", "123456", "1.2e+07"]

    cm = np.array([[0.1, 10], [100, 0.525]])
    plotted_text = ConfusionMatrixDisplay(cm, display_labels=[False, True]).plot().text_
    # Values should now formatted as '.2g', since there's a float in
    # Values are have two dec places max, (e.g 100 becomes 1e+02)
    test = [t.get_text() for t in plotted_text.ravel()]
    assert test == ["0.1", "10", "1e+02", "0.53"]


@pytest.mark.parametrize(
    "display_labels, expected_labels",
    [
        (None, ["0", "1"]),
        (["cat", "dog"], ["cat", "dog"]),
    ],
)
def test_default_labels(pyplot, display_labels, expected_labels):
    cm = np.array([[10, 0], [12, 120]])
    disp = ConfusionMatrixDisplay(cm, display_labels=display_labels).plot()

    x_ticks = [tick.get_text() for tick in disp.ax_.get_xticklabels()]
    y_ticks = [tick.get_text() for tick in disp.ax_.get_yticklabels()]

    assert_array_equal(x_ticks, expected_labels)
    assert_array_equal(y_ticks, expected_labels)


@pytest.mark.filterwarnings("ignore: Function plot_confusion_matrix is deprecated")
def test_error_on_a_dataset_with_unseen_labels(pyplot, fitted_clf, data, n_classes):
    """Check that when labels=None, the unique values in `y_pred` and `y_true`
    will be used.
    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/pull/18405
    """
    X, y = data

    # create unseen labels in `y_true` not seen during fitting and not present
    # in 'fitted_clf.classes_'
    y = y + 1
    disp = plot_confusion_matrix(fitted_clf, X, y)

    display_labels = [tick.get_text() for tick in disp.ax_.get_xticklabels()]
    expected_labels = [str(i) for i in range(n_classes + 1)]
    assert_array_equal(expected_labels, display_labels)


def test_plot_confusion_matrix_deprecation_warning(pyplot, fitted_clf, data):
    with pytest.warns(FutureWarning):
        plot_confusion_matrix(fitted_clf, *data)
