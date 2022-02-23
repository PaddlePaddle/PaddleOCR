from numpy.testing import (
    assert_allclose,
    assert_array_equal,
)
import numpy as np
import pytest

from sklearn.datasets import make_classification
from sklearn.compose import make_column_transformer
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix


# TODO: Remove when https://github.com/numpy/numpy/issues/14397 is resolved
pytestmark = pytest.mark.filterwarnings(
    "ignore:In future, it will be an error for 'np.bool_':DeprecationWarning:"
    "matplotlib.*"
)


def test_confusion_matrix_display_validation(pyplot):
    """Check that we raise the proper error when validating parameters."""
    X, y = make_classification(
        n_samples=100, n_informative=5, n_classes=5, random_state=0
    )

    with pytest.raises(NotFittedError):
        ConfusionMatrixDisplay.from_estimator(SVC(), X, y)

    regressor = SVR().fit(X, y)
    y_pred_regressor = regressor.predict(X)
    y_pred_classifier = SVC().fit(X, y).predict(X)

    err_msg = "ConfusionMatrixDisplay.from_estimator only supports classifiers"
    with pytest.raises(ValueError, match=err_msg):
        ConfusionMatrixDisplay.from_estimator(regressor, X, y)

    err_msg = "Mix type of y not allowed, got types"
    with pytest.raises(ValueError, match=err_msg):
        # Force `y_true` to be seen as a regression problem
        ConfusionMatrixDisplay.from_predictions(y + 0.5, y_pred_classifier)
    with pytest.raises(ValueError, match=err_msg):
        ConfusionMatrixDisplay.from_predictions(y, y_pred_regressor)

    err_msg = "Found input variables with inconsistent numbers of samples"
    with pytest.raises(ValueError, match=err_msg):
        ConfusionMatrixDisplay.from_predictions(y, y_pred_classifier[::2])


@pytest.mark.parametrize("constructor_name", ["from_estimator", "from_predictions"])
def test_confusion_matrix_display_invalid_option(pyplot, constructor_name):
    """Check the error raise if an invalid parameter value is passed."""
    X, y = make_classification(
        n_samples=100, n_informative=5, n_classes=5, random_state=0
    )
    classifier = SVC().fit(X, y)
    y_pred = classifier.predict(X)

    # safe guard for the binary if/else construction
    assert constructor_name in ("from_estimator", "from_predictions")
    extra_params = {"normalize": "invalid"}

    err_msg = r"normalize must be one of \{'true', 'pred', 'all', None\}"
    with pytest.raises(ValueError, match=err_msg):
        if constructor_name == "from_estimator":
            ConfusionMatrixDisplay.from_estimator(classifier, X, y, **extra_params)
        else:
            ConfusionMatrixDisplay.from_predictions(y, y_pred, **extra_params)


@pytest.mark.parametrize("constructor_name", ["from_estimator", "from_predictions"])
@pytest.mark.parametrize("with_labels", [True, False])
@pytest.mark.parametrize("with_display_labels", [True, False])
def test_confusion_matrix_display_custom_labels(
    pyplot, constructor_name, with_labels, with_display_labels
):
    """Check the resulting plot when labels are given."""
    n_classes = 5
    X, y = make_classification(
        n_samples=100, n_informative=5, n_classes=n_classes, random_state=0
    )
    classifier = SVC().fit(X, y)
    y_pred = classifier.predict(X)

    # safe guard for the binary if/else construction
    assert constructor_name in ("from_estimator", "from_predictions")

    ax = pyplot.gca()
    labels = [2, 1, 0, 3, 4] if with_labels else None
    display_labels = ["b", "d", "a", "e", "f"] if with_display_labels else None

    cm = confusion_matrix(y, y_pred, labels=labels)
    common_kwargs = {
        "ax": ax,
        "display_labels": display_labels,
        "labels": labels,
    }
    if constructor_name == "from_estimator":
        disp = ConfusionMatrixDisplay.from_estimator(classifier, X, y, **common_kwargs)
    else:
        disp = ConfusionMatrixDisplay.from_predictions(y, y_pred, **common_kwargs)
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


@pytest.mark.parametrize("constructor_name", ["from_estimator", "from_predictions"])
@pytest.mark.parametrize("normalize", ["true", "pred", "all", None])
@pytest.mark.parametrize("include_values", [True, False])
def test_confusion_matrix_display_plotting(
    pyplot,
    constructor_name,
    normalize,
    include_values,
):
    """Check the overall plotting rendering."""
    n_classes = 5
    X, y = make_classification(
        n_samples=100, n_informative=5, n_classes=n_classes, random_state=0
    )
    classifier = SVC().fit(X, y)
    y_pred = classifier.predict(X)

    # safe guard for the binary if/else construction
    assert constructor_name in ("from_estimator", "from_predictions")

    ax = pyplot.gca()
    cmap = "plasma"

    cm = confusion_matrix(y, y_pred)
    common_kwargs = {
        "normalize": normalize,
        "cmap": cmap,
        "ax": ax,
        "include_values": include_values,
    }
    if constructor_name == "from_estimator":
        disp = ConfusionMatrixDisplay.from_estimator(classifier, X, y, **common_kwargs)
    else:
        disp = ConfusionMatrixDisplay.from_predictions(y, y_pred, **common_kwargs)

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


@pytest.mark.parametrize("constructor_name", ["from_estimator", "from_predictions"])
def test_confusion_matrix_display(pyplot, constructor_name):
    """Check the behaviour of the default constructor without using the class
    methods."""
    n_classes = 5
    X, y = make_classification(
        n_samples=100, n_informative=5, n_classes=n_classes, random_state=0
    )
    classifier = SVC().fit(X, y)
    y_pred = classifier.predict(X)

    # safe guard for the binary if/else construction
    assert constructor_name in ("from_estimator", "from_predictions")

    cm = confusion_matrix(y, y_pred)
    common_kwargs = {
        "normalize": None,
        "include_values": True,
        "cmap": "viridis",
        "xticks_rotation": 45.0,
    }
    if constructor_name == "from_estimator":
        disp = ConfusionMatrixDisplay.from_estimator(classifier, X, y, **common_kwargs)
    else:
        disp = ConfusionMatrixDisplay.from_predictions(y, y_pred, **common_kwargs)

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
    """Check that the text color is appropriate depending on background."""

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


@pytest.mark.parametrize(
    "clf",
    [
        LogisticRegression(),
        make_pipeline(StandardScaler(), LogisticRegression()),
        make_pipeline(
            make_column_transformer((StandardScaler(), [0, 1])),
            LogisticRegression(),
        ),
    ],
    ids=["clf", "pipeline-clf", "pipeline-column_transformer-clf"],
)
def test_confusion_matrix_pipeline(pyplot, clf):
    """Check the behaviour of the plotting with more complex pipeline."""
    n_classes = 5
    X, y = make_classification(
        n_samples=100, n_informative=5, n_classes=n_classes, random_state=0
    )
    with pytest.raises(NotFittedError):
        ConfusionMatrixDisplay.from_estimator(clf, X, y)
    clf.fit(X, y)
    y_pred = clf.predict(X)

    disp = ConfusionMatrixDisplay.from_estimator(clf, X, y)
    cm = confusion_matrix(y, y_pred)

    assert_allclose(disp.confusion_matrix, cm)
    assert disp.text_.shape == (n_classes, n_classes)


@pytest.mark.parametrize("constructor_name", ["from_estimator", "from_predictions"])
def test_confusion_matrix_with_unknown_labels(pyplot, constructor_name):
    """Check that when labels=None, the unique values in `y_pred` and `y_true`
    will be used.
    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/pull/18405
    """
    n_classes = 5
    X, y = make_classification(
        n_samples=100, n_informative=5, n_classes=n_classes, random_state=0
    )
    classifier = SVC().fit(X, y)
    y_pred = classifier.predict(X)
    # create unseen labels in `y_true` not seen during fitting and not present
    # in 'classifier.classes_'
    y = y + 1

    # safe guard for the binary if/else construction
    assert constructor_name in ("from_estimator", "from_predictions")

    common_kwargs = {"labels": None}
    if constructor_name == "from_estimator":
        disp = ConfusionMatrixDisplay.from_estimator(classifier, X, y, **common_kwargs)
    else:
        disp = ConfusionMatrixDisplay.from_predictions(y, y_pred, **common_kwargs)

    display_labels = [tick.get_text() for tick in disp.ax_.get_xticklabels()]
    expected_labels = [str(i) for i in range(n_classes + 1)]
    assert_array_equal(expected_labels, display_labels)


def test_colormap_max(pyplot):
    """Check that the max color is used for the color of the text."""

    from matplotlib import cm

    gray = cm.get_cmap("gray", 1024)
    confusion_matrix = np.array([[1.0, 0.0], [0.0, 1.0]])

    disp = ConfusionMatrixDisplay(confusion_matrix)
    disp.plot(cmap=gray)

    color = disp.text_[1, 0].get_color()
    assert_allclose(color, [1.0, 1.0, 1.0, 1.0])
