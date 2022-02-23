import scipy as sp

from .base import _get_response

from .. import det_curve
from .._base import _check_pos_label_consistency

from ...utils import check_matplotlib_support
from ...utils import deprecated


class DetCurveDisplay:
    """DET curve visualization.

    It is recommend to use :func:`~sklearn.metrics.DetCurveDisplay.from_estimator`
    or :func:`~sklearn.metrics.DetCurveDisplay.from_predictions` to create a
    visualizer. All parameters are stored as attributes.

    Read more in the :ref:`User Guide <visualizations>`.

    .. versionadded:: 0.24

    Parameters
    ----------
    fpr : ndarray
        False positive rate.

    fnr : ndarray
        False negative rate.

    estimator_name : str, default=None
        Name of estimator. If None, the estimator name is not shown.

    pos_label : str or int, default=None
        The label of the positive class.

    Attributes
    ----------
    line_ : matplotlib Artist
        DET Curve.

    ax_ : matplotlib Axes
        Axes with DET Curve.

    figure_ : matplotlib Figure
        Figure containing the curve.

    See Also
    --------
    det_curve : Compute error rates for different probability thresholds.
    DetCurveDisplay.from_estimator : Plot DET curve given an estimator and
        some data.
    DetCurveDisplay.from_predictions : Plot DET curve given the true and
        predicted labels.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.metrics import det_curve, DetCurveDisplay
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.svm import SVC
    >>> X, y = make_classification(n_samples=1000, random_state=0)
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X, y, test_size=0.4, random_state=0)
    >>> clf = SVC(random_state=0).fit(X_train, y_train)
    >>> y_pred = clf.decision_function(X_test)
    >>> fpr, fnr, _ = det_curve(y_test, y_pred)
    >>> display = DetCurveDisplay(
    ...     fpr=fpr, fnr=fnr, estimator_name="SVC"
    ... )
    >>> display.plot()
    <...>
    >>> plt.show()
    """

    def __init__(self, *, fpr, fnr, estimator_name=None, pos_label=None):
        self.fpr = fpr
        self.fnr = fnr
        self.estimator_name = estimator_name
        self.pos_label = pos_label

    @classmethod
    def from_estimator(
        cls,
        estimator,
        X,
        y,
        *,
        sample_weight=None,
        response_method="auto",
        pos_label=None,
        name=None,
        ax=None,
        **kwargs,
    ):
        """Plot DET curve given an estimator and data.

        Read more in the :ref:`User Guide <visualizations>`.

        .. versionadded:: 1.0

        Parameters
        ----------
        estimator : estimator instance
            Fitted classifier or a fitted :class:`~sklearn.pipeline.Pipeline`
            in which the last estimator is a classifier.

        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Input values.

        y : array-like of shape (n_samples,)
            Target values.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        response_method : {'predict_proba', 'decision_function', 'auto'} \
                default='auto'
            Specifies whether to use :term:`predict_proba` or
            :term:`decision_function` as the predicted target response. If set
            to 'auto', :term:`predict_proba` is tried first and if it does not
            exist :term:`decision_function` is tried next.

        pos_label : str or int, default=None
            The label of the positive class. When `pos_label=None`, if `y_true`
            is in {-1, 1} or {0, 1}, `pos_label` is set to 1, otherwise an
            error will be raised.

        name : str, default=None
            Name of DET curve for labeling. If `None`, use the name of the
            estimator.

        ax : matplotlib axes, default=None
            Axes object to plot on. If `None`, a new figure and axes is
            created.

        **kwargs : dict
            Additional keywords arguments passed to matplotlib `plot` function.

        Returns
        -------
        display : :class:`~sklearn.metrics.DetCurveDisplay`
            Object that stores computed values.

        See Also
        --------
        det_curve : Compute error rates for different probability thresholds.
        DetCurveDisplay.from_predictions : Plot DET curve given the true and
            predicted labels.
        plot_roc_curve : Plot Receiver operating characteristic (ROC) curve.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> from sklearn.datasets import make_classification
        >>> from sklearn.metrics import DetCurveDisplay
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn.svm import SVC
        >>> X, y = make_classification(n_samples=1000, random_state=0)
        >>> X_train, X_test, y_train, y_test = train_test_split(
        ...     X, y, test_size=0.4, random_state=0)
        >>> clf = SVC(random_state=0).fit(X_train, y_train)
        >>> DetCurveDisplay.from_estimator(
        ...    clf, X_test, y_test)
        <...>
        >>> plt.show()
        """
        check_matplotlib_support(f"{cls.__name__}.from_estimator")

        name = estimator.__class__.__name__ if name is None else name

        y_pred, pos_label = _get_response(
            X,
            estimator,
            response_method,
            pos_label=pos_label,
        )

        return cls.from_predictions(
            y_true=y,
            y_pred=y_pred,
            sample_weight=sample_weight,
            name=name,
            ax=ax,
            pos_label=pos_label,
            **kwargs,
        )

    @classmethod
    def from_predictions(
        cls,
        y_true,
        y_pred,
        *,
        sample_weight=None,
        pos_label=None,
        name=None,
        ax=None,
        **kwargs,
    ):
        """Plot DET curve given the true and
        predicted labels.

        Read more in the :ref:`User Guide <visualizations>`.

        .. versionadded:: 1.0

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            True labels.

        y_pred : array-like of shape (n_samples,)
            Target scores, can either be probability estimates of the positive
            class, confidence values, or non-thresholded measure of decisions
            (as returned by `decision_function` on some classifiers).

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        pos_label : str or int, default=None
            The label of the positive class. When `pos_label=None`, if `y_true`
            is in {-1, 1} or {0, 1}, `pos_label` is set to 1, otherwise an
            error will be raised.

        name : str, default=None
            Name of DET curve for labeling. If `None`, name will be set to
            `"Classifier"`.

        ax : matplotlib axes, default=None
            Axes object to plot on. If `None`, a new figure and axes is
            created.

        **kwargs : dict
            Additional keywords arguments passed to matplotlib `plot` function.

        Returns
        -------
        display : :class:`~sklearn.metrics.DetCurveDisplay`
            Object that stores computed values.

        See Also
        --------
        det_curve : Compute error rates for different probability thresholds.
        DetCurveDisplay.from_estimator : Plot DET curve given an estimator and
            some data.
        plot_roc_curve : Plot Receiver operating characteristic (ROC) curve.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> from sklearn.datasets import make_classification
        >>> from sklearn.metrics import DetCurveDisplay
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn.svm import SVC
        >>> X, y = make_classification(n_samples=1000, random_state=0)
        >>> X_train, X_test, y_train, y_test = train_test_split(
        ...     X, y, test_size=0.4, random_state=0)
        >>> clf = SVC(random_state=0).fit(X_train, y_train)
        >>> y_pred = clf.decision_function(X_test)
        >>> DetCurveDisplay.from_predictions(
        ...    y_test, y_pred)
        <...>
        >>> plt.show()
        """
        check_matplotlib_support(f"{cls.__name__}.from_predictions")
        fpr, fnr, _ = det_curve(
            y_true,
            y_pred,
            pos_label=pos_label,
            sample_weight=sample_weight,
        )

        pos_label = _check_pos_label_consistency(pos_label, y_true)
        name = "Classifier" if name is None else name

        viz = DetCurveDisplay(
            fpr=fpr,
            fnr=fnr,
            estimator_name=name,
            pos_label=pos_label,
        )

        return viz.plot(ax=ax, name=name, **kwargs)

    def plot(self, ax=None, *, name=None, **kwargs):
        """Plot visualization.

        Parameters
        ----------
        ax : matplotlib axes, default=None
            Axes object to plot on. If `None`, a new figure and axes is
            created.

        name : str, default=None
            Name of DET curve for labeling. If `None`, use `estimator_name` if
            it is not `None`, otherwise no labeling is shown.

        **kwargs : dict
            Additional keywords arguments passed to matplotlib `plot` function.

        Returns
        -------
        display : :class:`~sklearn.metrics.plot.DetCurveDisplay`
            Object that stores computed values.
        """
        check_matplotlib_support("DetCurveDisplay.plot")

        name = self.estimator_name if name is None else name
        line_kwargs = {} if name is None else {"label": name}
        line_kwargs.update(**kwargs)

        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()

        (self.line_,) = ax.plot(
            sp.stats.norm.ppf(self.fpr),
            sp.stats.norm.ppf(self.fnr),
            **line_kwargs,
        )
        info_pos_label = (
            f" (Positive label: {self.pos_label})" if self.pos_label is not None else ""
        )

        xlabel = "False Positive Rate" + info_pos_label
        ylabel = "False Negative Rate" + info_pos_label
        ax.set(xlabel=xlabel, ylabel=ylabel)

        if "label" in line_kwargs:
            ax.legend(loc="lower right")

        ticks = [0.001, 0.01, 0.05, 0.20, 0.5, 0.80, 0.95, 0.99, 0.999]
        tick_locations = sp.stats.norm.ppf(ticks)
        tick_labels = [
            "{:.0%}".format(s) if (100 * s).is_integer() else "{:.1%}".format(s)
            for s in ticks
        ]
        ax.set_xticks(tick_locations)
        ax.set_xticklabels(tick_labels)
        ax.set_xlim(-3, 3)
        ax.set_yticks(tick_locations)
        ax.set_yticklabels(tick_labels)
        ax.set_ylim(-3, 3)

        self.ax_ = ax
        self.figure_ = ax.figure
        return self


@deprecated(
    "Function plot_det_curve is deprecated in 1.0 and will be "
    "removed in 1.2. Use one of the class methods: "
    "DetCurveDisplay.from_predictions or "
    "DetCurveDisplay.from_estimator."
)
def plot_det_curve(
    estimator,
    X,
    y,
    *,
    sample_weight=None,
    response_method="auto",
    name=None,
    ax=None,
    pos_label=None,
    **kwargs,
):
    """Plot detection error tradeoff (DET) curve.

    Extra keyword arguments will be passed to matplotlib's `plot`.

    Read more in the :ref:`User Guide <visualizations>`.

    .. versionadded:: 0.24

    .. deprecated:: 1.0
       `plot_det_curve` is deprecated in 1.0 and will be removed in
       1.2. Use one of the following class methods:
       :func:`~sklearn.metrics.DetCurveDisplay.from_predictions` or
       :func:`~sklearn.metrics.DetCurveDisplay.from_estimator`.

    Parameters
    ----------
    estimator : estimator instance
        Fitted classifier or a fitted :class:`~sklearn.pipeline.Pipeline`
        in which the last estimator is a classifier.

    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Input values.

    y : array-like of shape (n_samples,)
        Target values.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    response_method : {'predict_proba', 'decision_function', 'auto'} \
            default='auto'
        Specifies whether to use :term:`predict_proba` or
        :term:`decision_function` as the predicted target response. If set to
        'auto', :term:`predict_proba` is tried first and if it does not exist
        :term:`decision_function` is tried next.

    name : str, default=None
        Name of DET curve for labeling. If `None`, use the name of the
        estimator.

    ax : matplotlib axes, default=None
        Axes object to plot on. If `None`, a new figure and axes is created.

    pos_label : str or int, default=None
        The label of the positive class.
        When `pos_label=None`, if `y_true` is in {-1, 1} or {0, 1},
        `pos_label` is set to 1, otherwise an error will be raised.

    **kwargs : dict
            Additional keywords arguments passed to matplotlib `plot` function.

    Returns
    -------
    display : :class:`~sklearn.metrics.DetCurveDisplay`
        Object that stores computed values.

    See Also
    --------
    det_curve : Compute error rates for different probability thresholds.
    DetCurveDisplay : DET curve visualization.
    DetCurveDisplay.from_estimator : Plot DET curve given an estimator and
        some data.
    DetCurveDisplay.from_predictions : Plot DET curve given the true and
        predicted labels.
    RocCurveDisplay.from_estimator : Plot Receiver Operating Characteristic
        (ROC) curve given an estimator and some data.
    RocCurveDisplay.from_predictions : Plot Receiver Operating Characteristic
        (ROC) curve given the true and predicted values.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.metrics import plot_det_curve
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.svm import SVC
    >>> X, y = make_classification(n_samples=1000, random_state=0)
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X, y, test_size=0.4, random_state=0)
    >>> clf = SVC(random_state=0).fit(X_train, y_train)
    >>> plot_det_curve(clf, X_test, y_test)  # doctest: +SKIP
    <...>
    >>> plt.show()
    """
    check_matplotlib_support("plot_det_curve")

    y_pred, pos_label = _get_response(
        X, estimator, response_method, pos_label=pos_label
    )

    fpr, fnr, _ = det_curve(
        y,
        y_pred,
        pos_label=pos_label,
        sample_weight=sample_weight,
    )

    name = estimator.__class__.__name__ if name is None else name

    viz = DetCurveDisplay(fpr=fpr, fnr=fnr, estimator_name=name, pos_label=pos_label)

    return viz.plot(ax=ax, name=name, **kwargs)
