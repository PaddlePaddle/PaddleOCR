import numbers
from itertools import chain
from math import ceil

import numpy as np
from scipy import sparse
from scipy.stats.mstats import mquantiles
from joblib import Parallel

from .. import partial_dependence
from ...base import is_regressor
from ...utils import check_array
from ...utils import deprecated
from ...utils import check_matplotlib_support  # noqa
from ...utils import check_random_state
from ...utils import _safe_indexing
from ...utils.validation import _deprecate_positional_args
from ...utils.fixes import delayed


@deprecated(
    "Function `plot_partial_dependence` is deprecated in 1.0 and will be "
    "removed in 1.2. Use PartialDependenceDisplay.from_estimator instead"
)
def plot_partial_dependence(
    estimator,
    X,
    features,
    *,
    feature_names=None,
    target=None,
    response_method="auto",
    n_cols=3,
    grid_resolution=100,
    percentiles=(0.05, 0.95),
    method="auto",
    n_jobs=None,
    verbose=0,
    line_kw=None,
    ice_lines_kw=None,
    pd_line_kw=None,
    contour_kw=None,
    ax=None,
    kind="average",
    subsample=1000,
    random_state=None,
):
    """Partial dependence (PD) and individual conditional expectation (ICE)
    plots.

    Partial dependence plots, individual conditional expectation plots or an
    overlay of both of them can be plotted by setting the ``kind``
    parameter.
    The ``len(features)`` plots are arranged in a grid with ``n_cols``
    columns. Two-way partial dependence plots are plotted as contour plots. The
    deciles of the feature values will be shown with tick marks on the x-axes
    for one-way plots, and on both axes for two-way plots.

    Read more in the :ref:`User Guide <partial_dependence>`.

    .. note::

        :func:`plot_partial_dependence` does not support using the same axes
        with multiple calls. To plot the the partial dependence for multiple
        estimators, please pass the axes created by the first call to the
        second call::

          >>> from sklearn.inspection import plot_partial_dependence
          >>> from sklearn.datasets import make_friedman1
          >>> from sklearn.linear_model import LinearRegression
          >>> from sklearn.ensemble import RandomForestRegressor
          >>> X, y = make_friedman1()
          >>> est1 = LinearRegression().fit(X, y)
          >>> est2 = RandomForestRegressor().fit(X, y)
          >>> disp1 = plot_partial_dependence(est1, X,
          ...                                 [1, 2])  # doctest: +SKIP
          >>> disp2 = plot_partial_dependence(est2, X, [1, 2],
          ...                                 ax=disp1.axes_)  # doctest: +SKIP

    .. warning::

        For :class:`~sklearn.ensemble.GradientBoostingClassifier` and
        :class:`~sklearn.ensemble.GradientBoostingRegressor`, the
        `'recursion'` method (used by default) will not account for the `init`
        predictor of the boosting process. In practice, this will produce
        the same values as `'brute'` up to a constant offset in the target
        response, provided that `init` is a constant estimator (which is the
        default). However, if `init` is not a constant estimator, the
        partial dependence values are incorrect for `'recursion'` because the
        offset will be sample-dependent. It is preferable to use the `'brute'`
        method. Note that this only applies to
        :class:`~sklearn.ensemble.GradientBoostingClassifier` and
        :class:`~sklearn.ensemble.GradientBoostingRegressor`, not to
        :class:`~sklearn.ensemble.HistGradientBoostingClassifier` and
        :class:`~sklearn.ensemble.HistGradientBoostingRegressor`.

    .. deprecated:: 1.0
       `plot_partial_dependence` is deprecated in 1.0 and will be removed in
       1.2. Please use the class method:
       :func:`~sklearn.metrics.PartialDependenceDisplay.from_estimator`.

    Parameters
    ----------
    estimator : BaseEstimator
        A fitted estimator object implementing :term:`predict`,
        :term:`predict_proba`, or :term:`decision_function`.
        Multioutput-multiclass classifiers are not supported.

    X : {array-like, dataframe} of shape (n_samples, n_features)
        ``X`` is used to generate a grid of values for the target
        ``features`` (where the partial dependence will be evaluated), and
        also to generate values for the complement features when the
        `method` is `'brute'`.

    features : list of {int, str, pair of int, pair of str}
        The target features for which to create the PDPs.
        If `features[i]` is an integer or a string, a one-way PDP is created;
        if `features[i]` is a tuple, a two-way PDP is created (only supported
        with `kind='average'`). Each tuple must be of size 2.
        if any entry is a string, then it must be in ``feature_names``.

    feature_names : array-like of shape (n_features,), dtype=str, default=None
        Name of each feature; `feature_names[i]` holds the name of the feature
        with index `i`.
        By default, the name of the feature corresponds to their numerical
        index for NumPy array and their column name for pandas dataframe.

    target : int, default=None
        - In a multiclass setting, specifies the class for which the PDPs
          should be computed. Note that for binary classification, the
          positive class (index 1) is always used.
        - In a multioutput setting, specifies the task for which the PDPs
          should be computed.

        Ignored in binary classification or classical regression settings.

    response_method : {'auto', 'predict_proba', 'decision_function'}, \
            default='auto'
        Specifies whether to use :term:`predict_proba` or
        :term:`decision_function` as the target response. For regressors
        this parameter is ignored and the response is always the output of
        :term:`predict`. By default, :term:`predict_proba` is tried first
        and we revert to :term:`decision_function` if it doesn't exist. If
        ``method`` is `'recursion'`, the response is always the output of
        :term:`decision_function`.

    n_cols : int, default=3
        The maximum number of columns in the grid plot. Only active when `ax`
        is a single axis or `None`.

    grid_resolution : int, default=100
        The number of equally spaced points on the axes of the plots, for each
        target feature.

    percentiles : tuple of float, default=(0.05, 0.95)
        The lower and upper percentile used to create the extreme values
        for the PDP axes. Must be in [0, 1].

    method : str, default='auto'
        The method used to calculate the averaged predictions:

        - `'recursion'` is only supported for some tree-based estimators
          (namely
          :class:`~sklearn.ensemble.GradientBoostingClassifier`,
          :class:`~sklearn.ensemble.GradientBoostingRegressor`,
          :class:`~sklearn.ensemble.HistGradientBoostingClassifier`,
          :class:`~sklearn.ensemble.HistGradientBoostingRegressor`,
          :class:`~sklearn.tree.DecisionTreeRegressor`,
          :class:`~sklearn.ensemble.RandomForestRegressor`
          but is more efficient in terms of speed.
          With this method, the target response of a
          classifier is always the decision function, not the predicted
          probabilities. Since the `'recursion'` method implicitly computes
          the average of the ICEs by design, it is not compatible with ICE and
          thus `kind` must be `'average'`.

        - `'brute'` is supported for any estimator, but is more
          computationally intensive.

        - `'auto'`: the `'recursion'` is used for estimators that support it,
          and `'brute'` is used otherwise.

        Please see :ref:`this note <pdp_method_differences>` for
        differences between the `'brute'` and `'recursion'` method.

    n_jobs : int, default=None
        The number of CPUs to use to compute the partial dependences.
        Computation is parallelized over features specified by the `features`
        parameter.

        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    verbose : int, default=0
        Verbose output during PD computations.

    line_kw : dict, default=None
        Dict with keywords passed to the ``matplotlib.pyplot.plot`` call.
        For one-way partial dependence plots. It can be used to define common
        properties for both `ice_lines_kw` and `pdp_line_kw`.

    ice_lines_kw : dict, default=None
        Dictionary with keywords passed to the `matplotlib.pyplot.plot` call.
        For ICE lines in the one-way partial dependence plots.
        The key value pairs defined in `ice_lines_kw` takes priority over
        `line_kw`.

        .. versionadded:: 1.0

    pd_line_kw : dict, default=None
        Dictionary with keywords passed to the `matplotlib.pyplot.plot` call.
        For partial dependence in one-way partial dependence plots.
        The key value pairs defined in `pd_line_kw` takes priority over
        `line_kw`.

        .. versionadded:: 1.0

    contour_kw : dict, default=None
        Dict with keywords passed to the ``matplotlib.pyplot.contourf`` call.
        For two-way partial dependence plots.

    ax : Matplotlib axes or array-like of Matplotlib axes, default=None
        - If a single axis is passed in, it is treated as a bounding axes
          and a grid of partial dependence plots will be drawn within
          these bounds. The `n_cols` parameter controls the number of
          columns in the grid.
        - If an array-like of axes are passed in, the partial dependence
          plots will be drawn directly into these axes.
        - If `None`, a figure and a bounding axes is created and treated
          as the single axes case.

        .. versionadded:: 0.22

    kind : {'average', 'individual', 'both'}, default='average'
        Whether to plot the partial dependence averaged across all the samples
        in the dataset or one line per sample or both.

        - ``kind='average'`` results in the traditional PD plot;
        - ``kind='individual'`` results in the ICE plot.

       Note that the fast ``method='recursion'`` option is only available for
       ``kind='average'``. Plotting individual dependencies requires using the
       slower ``method='brute'`` option.

        .. versionadded:: 0.24

    subsample : float, int or None, default=1000
        Sampling for ICE curves when `kind` is 'individual' or 'both'.
        If `float`, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to be used to plot ICE curves. If `int`, represents the
        absolute number samples to use.

        Note that the full dataset is still used to calculate averaged partial
        dependence when `kind='both'`.

        .. versionadded:: 0.24

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the selected samples when subsamples is not
        `None` and `kind` is either `'both'` or `'individual'`.
        See :term:`Glossary <random_state>` for details.

        .. versionadded:: 0.24

    Returns
    -------
    display : :class:`~sklearn.inspection.PartialDependenceDisplay`

    See Also
    --------
    partial_dependence : Compute Partial Dependence values.
    PartialDependenceDisplay : Partial Dependence visualization.
    PartialDependenceDisplay.from_estimator : Plot Partial Dependence.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from sklearn.datasets import make_friedman1
    >>> from sklearn.ensemble import GradientBoostingRegressor
    >>> from sklearn.inspection import plot_partial_dependence
    >>> X, y = make_friedman1()
    >>> clf = GradientBoostingRegressor(n_estimators=10).fit(X, y)
    >>> plot_partial_dependence(clf, X, [0, (0, 1)])  # doctest: +SKIP
    <...>
    >>> plt.show()  # doctest: +SKIP
    """
    check_matplotlib_support("plot_partial_dependence")  # noqa
    return _plot_partial_dependence(
        estimator,
        X,
        features,
        feature_names=feature_names,
        target=target,
        response_method=response_method,
        n_cols=n_cols,
        grid_resolution=grid_resolution,
        percentiles=percentiles,
        method=method,
        n_jobs=n_jobs,
        verbose=verbose,
        line_kw=line_kw,
        ice_lines_kw=ice_lines_kw,
        pd_line_kw=pd_line_kw,
        contour_kw=contour_kw,
        ax=ax,
        kind=kind,
        subsample=subsample,
        random_state=random_state,
    )


# TODO: Move into PartialDependenceDisplay.from_estimator in 1.2
def _plot_partial_dependence(
    estimator,
    X,
    features,
    *,
    feature_names=None,
    target=None,
    response_method="auto",
    n_cols=3,
    grid_resolution=100,
    percentiles=(0.05, 0.95),
    method="auto",
    n_jobs=None,
    verbose=0,
    line_kw=None,
    ice_lines_kw=None,
    pd_line_kw=None,
    contour_kw=None,
    ax=None,
    kind="average",
    subsample=1000,
    random_state=None,
):
    """See PartialDependenceDisplay.from_estimator for details"""
    import matplotlib.pyplot as plt  # noqa

    # set target_idx for multi-class estimators
    if hasattr(estimator, "classes_") and np.size(estimator.classes_) > 2:
        if target is None:
            raise ValueError("target must be specified for multi-class")
        target_idx = np.searchsorted(estimator.classes_, target)
        if (
            not (0 <= target_idx < len(estimator.classes_))
            or estimator.classes_[target_idx] != target
        ):
            raise ValueError("target not in est.classes_, got {}".format(target))
    else:
        # regression and binary classification
        target_idx = 0

    # Use check_array only on lists and other non-array-likes / sparse. Do not
    # convert DataFrame into a NumPy array.
    if not (hasattr(X, "__array__") or sparse.issparse(X)):
        X = check_array(X, force_all_finite="allow-nan", dtype=object)
    n_features = X.shape[1]

    # convert feature_names to list
    if feature_names is None:
        if hasattr(X, "loc"):
            # get the column names for a pandas dataframe
            feature_names = X.columns.tolist()
        else:
            # define a list of numbered indices for a numpy array
            feature_names = [str(i) for i in range(n_features)]
    elif hasattr(feature_names, "tolist"):
        # convert numpy array or pandas index to a list
        feature_names = feature_names.tolist()
    if len(set(feature_names)) != len(feature_names):
        raise ValueError("feature_names should not contain duplicates.")

    def convert_feature(fx):
        if isinstance(fx, str):
            try:
                fx = feature_names.index(fx)
            except ValueError as e:
                raise ValueError("Feature %s not in feature_names" % fx) from e
        return int(fx)

    # convert features into a seq of int tuples
    tmp_features = []
    for fxs in features:
        if isinstance(fxs, (numbers.Integral, str)):
            fxs = (fxs,)
        try:
            fxs = tuple(convert_feature(fx) for fx in fxs)
        except TypeError as e:
            raise ValueError(
                "Each entry in features must be either an int, "
                "a string, or an iterable of size at most 2."
            ) from e
        if not 1 <= np.size(fxs) <= 2:
            raise ValueError(
                "Each entry in features must be either an int, "
                "a string, or an iterable of size at most 2."
            )
        if kind != "average" and np.size(fxs) > 1:
            raise ValueError(
                "It is not possible to display individual effects for more "
                f"than one feature at a time. Got: features={features}."
            )
        tmp_features.append(fxs)

    features = tmp_features

    # Early exit if the axes does not have the correct number of axes
    if ax is not None and not isinstance(ax, plt.Axes):
        axes = np.asarray(ax, dtype=object)
        if axes.size != len(features):
            raise ValueError(
                "Expected ax to have {} axes, got {}".format(len(features), axes.size)
            )

    for i in chain.from_iterable(features):
        if i >= len(feature_names):
            raise ValueError(
                "All entries of features must be less than "
                "len(feature_names) = {0}, got {1}.".format(len(feature_names), i)
            )

    if isinstance(subsample, numbers.Integral):
        if subsample <= 0:
            raise ValueError(
                f"When an integer, subsample={subsample} should be positive."
            )
    elif isinstance(subsample, numbers.Real):
        if subsample <= 0 or subsample >= 1:
            raise ValueError(
                f"When a floating-point, subsample={subsample} should be in "
                "the (0, 1) range."
            )

    # compute predictions and/or averaged predictions
    pd_results = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(partial_dependence)(
            estimator,
            X,
            fxs,
            response_method=response_method,
            method=method,
            grid_resolution=grid_resolution,
            percentiles=percentiles,
            kind=kind,
        )
        for fxs in features
    )

    # For multioutput regression, we can only check the validity of target
    # now that we have the predictions.
    # Also note: as multiclass-multioutput classifiers are not supported,
    # multiclass and multioutput scenario are mutually exclusive. So there is
    # no risk of overwriting target_idx here.
    pd_result = pd_results[0]  # checking the first result is enough
    n_tasks = (
        pd_result.average.shape[0]
        if kind == "average"
        else pd_result.individual.shape[0]
    )
    if is_regressor(estimator) and n_tasks > 1:
        if target is None:
            raise ValueError("target must be specified for multi-output regressors")
        if not 0 <= target <= n_tasks:
            raise ValueError("target must be in [0, n_tasks], got {}.".format(target))
        target_idx = target

    # get global min and max average predictions of PD grouped by plot type
    pdp_lim = {}
    for pdp in pd_results:
        values = pdp["values"]
        preds = pdp.average if kind == "average" else pdp.individual
        min_pd = preds[target_idx].min()
        max_pd = preds[target_idx].max()
        n_fx = len(values)
        old_min_pd, old_max_pd = pdp_lim.get(n_fx, (min_pd, max_pd))
        min_pd = min(min_pd, old_min_pd)
        max_pd = max(max_pd, old_max_pd)
        pdp_lim[n_fx] = (min_pd, max_pd)

    deciles = {}
    for fx in chain.from_iterable(features):
        if fx not in deciles:
            X_col = _safe_indexing(X, fx, axis=1)
            deciles[fx] = mquantiles(X_col, prob=np.arange(0.1, 1.0, 0.1))

    display = PartialDependenceDisplay(
        pd_results=pd_results,
        features=features,
        feature_names=feature_names,
        target_idx=target_idx,
        pdp_lim=pdp_lim,
        deciles=deciles,
        kind=kind,
        subsample=subsample,
        random_state=random_state,
    )
    return display.plot(
        ax=ax,
        n_cols=n_cols,
        line_kw=line_kw,
        ice_lines_kw=ice_lines_kw,
        pd_line_kw=pd_line_kw,
        contour_kw=contour_kw,
    )


class PartialDependenceDisplay:
    """Partial Dependence Plot (PDP).

    This can also display individual partial dependencies which are often
    referred to as: Individual Condition Expectation (ICE).

    It is recommended to use
    :func:`~sklearn.inspection.PartialDependenceDisplay.from_estimator` to create a
    :class:`~sklearn.inspection.PartialDependenceDisplay`. All parameters are
    stored as attributes.

    Read more in
    :ref:`sphx_glr_auto_examples_miscellaneous_plot_partial_dependence_visualization_api.py`
    and the :ref:`User Guide <visualizations>`.

        .. versionadded:: 0.22

    Parameters
    ----------
    pd_results : list of Bunch
        Results of :func:`~sklearn.inspection.partial_dependence` for
        ``features``.

    features : list of (int,) or list of (int, int)
        Indices of features for a given plot. A tuple of one integer will plot
        a partial dependence curve of one feature. A tuple of two integers will
        plot a two-way partial dependence curve as a contour plot.

    feature_names : list of str
        Feature names corresponding to the indices in ``features``.

    target_idx : int

        - In a multiclass setting, specifies the class for which the PDPs
          should be computed. Note that for binary classification, the
          positive class (index 1) is always used.
        - In a multioutput setting, specifies the task for which the PDPs
          should be computed.

        Ignored in binary classification or classical regression settings.

    pdp_lim : dict
        Global min and max average predictions, such that all plots will have
        the same scale and y limits. `pdp_lim[1]` is the global min and max for
        single partial dependence curves. `pdp_lim[2]` is the global min and
        max for two-way partial dependence curves.

    deciles : dict
        Deciles for feature indices in ``features``.

    kind : {'average', 'individual', 'both'}, default='average'
        Whether to plot the partial dependence averaged across all the samples
        in the dataset or one line per sample or both.

        - ``kind='average'`` results in the traditional PD plot;
        - ``kind='individual'`` results in the ICE plot.

       Note that the fast ``method='recursion'`` option is only available for
       ``kind='average'``. Plotting individual dependencies requires using the
       slower ``method='brute'`` option.

        .. versionadded:: 0.24

    subsample : float, int or None, default=1000
        Sampling for ICE curves when `kind` is 'individual' or 'both'.
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to be used to plot ICE curves. If int, represents the
        maximum absolute number of samples to use.

        Note that the full dataset is still used to calculate partial
        dependence when `kind='both'`.

        .. versionadded:: 0.24

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the selected samples when subsamples is not
        `None`. See :term:`Glossary <random_state>` for details.

        .. versionadded:: 0.24

    Attributes
    ----------
    bounding_ax_ : matplotlib Axes or None
        If `ax` is an axes or None, the `bounding_ax_` is the axes where the
        grid of partial dependence plots are drawn. If `ax` is a list of axes
        or a numpy array of axes, `bounding_ax_` is None.

    axes_ : ndarray of matplotlib Axes
        If `ax` is an axes or None, `axes_[i, j]` is the axes on the i-th row
        and j-th column. If `ax` is a list of axes, `axes_[i]` is the i-th item
        in `ax`. Elements that are None correspond to a nonexisting axes in
        that position.

    lines_ : ndarray of matplotlib Artists
        If `ax` is an axes or None, `lines_[i, j]` is the partial dependence
        curve on the i-th row and j-th column. If `ax` is a list of axes,
        `lines_[i]` is the partial dependence curve corresponding to the i-th
        item in `ax`. Elements that are None correspond to a nonexisting axes
        or an axes that does not include a line plot.

    deciles_vlines_ : ndarray of matplotlib LineCollection
        If `ax` is an axes or None, `vlines_[i, j]` is the line collection
        representing the x axis deciles of the i-th row and j-th column. If
        `ax` is a list of axes, `vlines_[i]` corresponds to the i-th item in
        `ax`. Elements that are None correspond to a nonexisting axes or an
        axes that does not include a PDP plot.

        .. versionadded:: 0.23

    deciles_hlines_ : ndarray of matplotlib LineCollection
        If `ax` is an axes or None, `vlines_[i, j]` is the line collection
        representing the y axis deciles of the i-th row and j-th column. If
        `ax` is a list of axes, `vlines_[i]` corresponds to the i-th item in
        `ax`. Elements that are None correspond to a nonexisting axes or an
        axes that does not include a 2-way plot.

        .. versionadded:: 0.23

    contours_ : ndarray of matplotlib Artists
        If `ax` is an axes or None, `contours_[i, j]` is the partial dependence
        plot on the i-th row and j-th column. If `ax` is a list of axes,
        `contours_[i]` is the partial dependence plot corresponding to the i-th
        item in `ax`. Elements that are None correspond to a nonexisting axes
        or an axes that does not include a contour plot.

    figure_ : matplotlib Figure
        Figure containing partial dependence plots.

    See Also
    --------
    partial_dependence : Compute Partial Dependence values.
    PartialDependenceDisplay.from_estimator : Plot Partial Dependence.
    """

    def __init__(
        self,
        pd_results,
        *,
        features,
        feature_names,
        target_idx,
        pdp_lim,
        deciles,
        kind="average",
        subsample=1000,
        random_state=None,
    ):
        self.pd_results = pd_results
        self.features = features
        self.feature_names = feature_names
        self.target_idx = target_idx
        self.pdp_lim = pdp_lim
        self.deciles = deciles
        self.kind = kind
        self.subsample = subsample
        self.random_state = random_state

    @classmethod
    def from_estimator(
        cls,
        estimator,
        X,
        features,
        *,
        feature_names=None,
        target=None,
        response_method="auto",
        n_cols=3,
        grid_resolution=100,
        percentiles=(0.05, 0.95),
        method="auto",
        n_jobs=None,
        verbose=0,
        line_kw=None,
        ice_lines_kw=None,
        pd_line_kw=None,
        contour_kw=None,
        ax=None,
        kind="average",
        subsample=1000,
        random_state=None,
    ):
        """Partial dependence (PD) and individual conditional expectation (ICE) plots.

        Partial dependence plots, individual conditional expectation plots or an
        overlay of both of them can be plotted by setting the ``kind``
        parameter. The ``len(features)`` plots are arranged in a grid with
        ``n_cols`` columns. Two-way partial dependence plots are plotted as
        contour plots. The deciles of the feature values will be shown with tick
        marks on the x-axes for one-way plots, and on both axes for two-way
        plots.

        Read more in the :ref:`User Guide <partial_dependence>`.

        .. note::

            :func:`PartialDependenceDisplay.from_estimator` does not support using the
            same axes with multiple calls. To plot the the partial dependence for
            multiple estimators, please pass the axes created by the first call to the
            second call::

               >>> from sklearn.inspection import PartialDependenceDisplay
               >>> from sklearn.datasets import make_friedman1
               >>> from sklearn.linear_model import LinearRegression
               >>> from sklearn.ensemble import RandomForestRegressor
               >>> X, y = make_friedman1()
               >>> est1 = LinearRegression().fit(X, y)
               >>> est2 = RandomForestRegressor().fit(X, y)
               >>> disp1 = PartialDependenceDisplay.from_estimator(est1, X,
               ...                                                 [1, 2])
               >>> disp2 = PartialDependenceDisplay.from_estimator(est2, X, [1, 2],
               ...                                                 ax=disp1.axes_)

        .. warning::

            For :class:`~sklearn.ensemble.GradientBoostingClassifier` and
            :class:`~sklearn.ensemble.GradientBoostingRegressor`, the
            `'recursion'` method (used by default) will not account for the `init`
            predictor of the boosting process. In practice, this will produce
            the same values as `'brute'` up to a constant offset in the target
            response, provided that `init` is a constant estimator (which is the
            default). However, if `init` is not a constant estimator, the
            partial dependence values are incorrect for `'recursion'` because the
            offset will be sample-dependent. It is preferable to use the `'brute'`
            method. Note that this only applies to
            :class:`~sklearn.ensemble.GradientBoostingClassifier` and
            :class:`~sklearn.ensemble.GradientBoostingRegressor`, not to
            :class:`~sklearn.ensemble.HistGradientBoostingClassifier` and
            :class:`~sklearn.ensemble.HistGradientBoostingRegressor`.

        .. versionadded:: 1.0

        Parameters
        ----------
        estimator : BaseEstimator
            A fitted estimator object implementing :term:`predict`,
            :term:`predict_proba`, or :term:`decision_function`.
            Multioutput-multiclass classifiers are not supported.

        X : {array-like, dataframe} of shape (n_samples, n_features)
            ``X`` is used to generate a grid of values for the target
            ``features`` (where the partial dependence will be evaluated), and
            also to generate values for the complement features when the
            `method` is `'brute'`.

        features : list of {int, str, pair of int, pair of str}
            The target features for which to create the PDPs.
            If `features[i]` is an integer or a string, a one-way PDP is created;
            if `features[i]` is a tuple, a two-way PDP is created (only supported
            with `kind='average'`). Each tuple must be of size 2.
            if any entry is a string, then it must be in ``feature_names``.

        feature_names : array-like of shape (n_features,), dtype=str, default=None
            Name of each feature; `feature_names[i]` holds the name of the feature
            with index `i`.
            By default, the name of the feature corresponds to their numerical
            index for NumPy array and their column name for pandas dataframe.

        target : int, default=None
            - In a multiclass setting, specifies the class for which the PDPs
              should be computed. Note that for binary classification, the
              positive class (index 1) is always used.
            - In a multioutput setting, specifies the task for which the PDPs
              should be computed.

            Ignored in binary classification or classical regression settings.

        response_method : {'auto', 'predict_proba', 'decision_function'}, \
                default='auto'
            Specifies whether to use :term:`predict_proba` or
            :term:`decision_function` as the target response. For regressors
            this parameter is ignored and the response is always the output of
            :term:`predict`. By default, :term:`predict_proba` is tried first
            and we revert to :term:`decision_function` if it doesn't exist. If
            ``method`` is `'recursion'`, the response is always the output of
            :term:`decision_function`.

        n_cols : int, default=3
            The maximum number of columns in the grid plot. Only active when `ax`
            is a single axis or `None`.

        grid_resolution : int, default=100
            The number of equally spaced points on the axes of the plots, for each
            target feature.

        percentiles : tuple of float, default=(0.05, 0.95)
            The lower and upper percentile used to create the extreme values
            for the PDP axes. Must be in [0, 1].

        method : str, default='auto'
            The method used to calculate the averaged predictions:

            - `'recursion'` is only supported for some tree-based estimators
              (namely
              :class:`~sklearn.ensemble.GradientBoostingClassifier`,
              :class:`~sklearn.ensemble.GradientBoostingRegressor`,
              :class:`~sklearn.ensemble.HistGradientBoostingClassifier`,
              :class:`~sklearn.ensemble.HistGradientBoostingRegressor`,
              :class:`~sklearn.tree.DecisionTreeRegressor`,
              :class:`~sklearn.ensemble.RandomForestRegressor`
              but is more efficient in terms of speed.
              With this method, the target response of a
              classifier is always the decision function, not the predicted
              probabilities. Since the `'recursion'` method implicitly computes
              the average of the ICEs by design, it is not compatible with ICE and
              thus `kind` must be `'average'`.

            - `'brute'` is supported for any estimator, but is more
              computationally intensive.

            - `'auto'`: the `'recursion'` is used for estimators that support it,
              and `'brute'` is used otherwise.

            Please see :ref:`this note <pdp_method_differences>` for
            differences between the `'brute'` and `'recursion'` method.

        n_jobs : int, default=None
            The number of CPUs to use to compute the partial dependences.
            Computation is parallelized over features specified by the `features`
            parameter.

            ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
            ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
            for more details.

        verbose : int, default=0
            Verbose output during PD computations.

        line_kw : dict, default=None
            Dict with keywords passed to the ``matplotlib.pyplot.plot`` call.
            For one-way partial dependence plots. It can be used to define common
            properties for both `ice_lines_kw` and `pdp_line_kw`.

        ice_lines_kw : dict, default=None
            Dictionary with keywords passed to the `matplotlib.pyplot.plot` call.
            For ICE lines in the one-way partial dependence plots.
            The key value pairs defined in `ice_lines_kw` takes priority over
            `line_kw`.

        pd_line_kw : dict, default=None
            Dictionary with keywords passed to the `matplotlib.pyplot.plot` call.
            For partial dependence in one-way partial dependence plots.
            The key value pairs defined in `pd_line_kw` takes priority over
            `line_kw`.

        contour_kw : dict, default=None
            Dict with keywords passed to the ``matplotlib.pyplot.contourf`` call.
            For two-way partial dependence plots.

        ax : Matplotlib axes or array-like of Matplotlib axes, default=None
            - If a single axis is passed in, it is treated as a bounding axes
              and a grid of partial dependence plots will be drawn within
              these bounds. The `n_cols` parameter controls the number of
              columns in the grid.
            - If an array-like of axes are passed in, the partial dependence
              plots will be drawn directly into these axes.
            - If `None`, a figure and a bounding axes is created and treated
              as the single axes case.

        kind : {'average', 'individual', 'both'}, default='average'
            Whether to plot the partial dependence averaged across all the samples
            in the dataset or one line per sample or both.

            - ``kind='average'`` results in the traditional PD plot;
            - ``kind='individual'`` results in the ICE plot.

           Note that the fast ``method='recursion'`` option is only available for
           ``kind='average'``. Plotting individual dependencies requires using the
           slower ``method='brute'`` option.

        subsample : float, int or None, default=1000
            Sampling for ICE curves when `kind` is 'individual' or 'both'.
            If `float`, should be between 0.0 and 1.0 and represent the proportion
            of the dataset to be used to plot ICE curves. If `int`, represents the
            absolute number samples to use.

            Note that the full dataset is still used to calculate averaged partial
            dependence when `kind='both'`.

        random_state : int, RandomState instance or None, default=None
            Controls the randomness of the selected samples when subsamples is not
            `None` and `kind` is either `'both'` or `'individual'`.
            See :term:`Glossary <random_state>` for details.

        Returns
        -------
        display : :class:`~sklearn.inspection.PartialDependenceDisplay`

        See Also
        --------
        partial_dependence : Compute Partial Dependence values.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> from sklearn.datasets import make_friedman1
        >>> from sklearn.ensemble import GradientBoostingRegressor
        >>> from sklearn.inspection import PartialDependenceDisplay
        >>> X, y = make_friedman1()
        >>> clf = GradientBoostingRegressor(n_estimators=10).fit(X, y)
        >>> PartialDependenceDisplay.from_estimator(clf, X, [0, (0, 1)])
        <...>
        >>> plt.show()
        """
        check_matplotlib_support(f"{cls.__name__}.from_estimator")  # noqa
        return _plot_partial_dependence(
            estimator,
            X,
            features,
            feature_names=feature_names,
            target=target,
            response_method=response_method,
            n_cols=n_cols,
            grid_resolution=grid_resolution,
            percentiles=percentiles,
            method=method,
            n_jobs=n_jobs,
            verbose=verbose,
            line_kw=line_kw,
            ice_lines_kw=ice_lines_kw,
            pd_line_kw=pd_line_kw,
            contour_kw=contour_kw,
            ax=ax,
            kind=kind,
            subsample=subsample,
            random_state=random_state,
        )

    def _get_sample_count(self, n_samples):
        """Compute the number of samples as an integer."""
        if isinstance(self.subsample, numbers.Integral):
            if self.subsample < n_samples:
                return self.subsample
            return n_samples
        elif isinstance(self.subsample, numbers.Real):
            return ceil(n_samples * self.subsample)
        return n_samples

    def _plot_ice_lines(
        self,
        preds,
        feature_values,
        n_ice_to_plot,
        ax,
        pd_plot_idx,
        n_total_lines_by_plot,
        individual_line_kw,
    ):
        """Plot the ICE lines.

        Parameters
        ----------
        preds : ndarray of shape \
                (n_instances, n_grid_points)
            The predictions computed for all points of `feature_values` for a
            given feature for all samples in `X`.
        feature_values : ndarray of shape (n_grid_points,)
            The feature values for which the predictions have been computed.
        n_ice_to_plot : int
            The number of ICE lines to plot.
        ax : Matplotlib axes
            The axis on which to plot the ICE lines.
        pd_plot_idx : int
            The sequential index of the plot. It will be unraveled to find the
            matching 2D position in the grid layout.
        n_total_lines_by_plot : int
            The total number of lines expected to be plot on the axis.
        individual_line_kw : dict
            Dict with keywords passed when plotting the ICE lines.
        """
        rng = check_random_state(self.random_state)
        # subsample ice
        ice_lines_idx = rng.choice(
            preds.shape[0],
            n_ice_to_plot,
            replace=False,
        )
        ice_lines_subsampled = preds[ice_lines_idx, :]
        # plot the subsampled ice
        for ice_idx, ice in enumerate(ice_lines_subsampled):
            line_idx = np.unravel_index(
                pd_plot_idx * n_total_lines_by_plot + ice_idx, self.lines_.shape
            )
            self.lines_[line_idx] = ax.plot(
                feature_values, ice.ravel(), **individual_line_kw
            )[0]

    def _plot_average_dependence(
        self,
        avg_preds,
        feature_values,
        ax,
        pd_line_idx,
        line_kw,
    ):
        """Plot the average partial dependence.

        Parameters
        ----------
        avg_preds : ndarray of shape (n_grid_points,)
            The average predictions for all points of `feature_values` for a
            given feature for all samples in `X`.
        feature_values : ndarray of shape (n_grid_points,)
            The feature values for which the predictions have been computed.
        ax : Matplotlib axes
            The axis on which to plot the ICE lines.
        pd_line_idx : int
            The sequential index of the plot. It will be unraveled to find the
            matching 2D position in the grid layout.
        line_kw : dict
            Dict with keywords passed when plotting the PD plot.
        """
        line_idx = np.unravel_index(pd_line_idx, self.lines_.shape)
        self.lines_[line_idx] = ax.plot(
            feature_values,
            avg_preds,
            **line_kw,
        )[0]

    def _plot_one_way_partial_dependence(
        self,
        preds,
        avg_preds,
        feature_values,
        feature_idx,
        n_ice_lines,
        ax,
        n_cols,
        pd_plot_idx,
        n_lines,
        ice_lines_kw,
        pd_line_kw,
    ):
        """Plot 1-way partial dependence: ICE and PDP.

        Parameters
        ----------
        preds : ndarray of shape \
                (n_instances, n_grid_points) or None
            The predictions computed for all points of `feature_values` for a
            given feature for all samples in `X`.
        avg_preds : ndarray of shape (n_grid_points,)
            The average predictions for all points of `feature_values` for a
            given feature for all samples in `X`.
        feature_values : ndarray of shape (n_grid_points,)
            The feature values for which the predictions have been computed.
        feature_idx : int
            The index corresponding to the target feature.
        n_ice_lines : int
            The number of ICE lines to plot.
        ax : Matplotlib axes
            The axis on which to plot the ICE and PDP lines.
        n_cols : int or None
            The number of column in the axis.
        pd_plot_idx : int
            The sequential index of the plot. It will be unraveled to find the
            matching 2D position in the grid layout.
        n_lines : int
            The total number of lines expected to be plot on the axis.
        ice_lines_kw : dict
            Dict with keywords passed when plotting the ICE lines.
        pd_line_kw : dict
            Dict with keywords passed when plotting the PD plot.
        """
        from matplotlib import transforms  # noqa

        if self.kind in ("individual", "both"):
            self._plot_ice_lines(
                preds[self.target_idx],
                feature_values,
                n_ice_lines,
                ax,
                pd_plot_idx,
                n_lines,
                ice_lines_kw,
            )

        if self.kind in ("average", "both"):
            # the average is stored as the last line
            if self.kind == "average":
                pd_line_idx = pd_plot_idx
            else:
                pd_line_idx = pd_plot_idx * n_lines + n_ice_lines
            self._plot_average_dependence(
                avg_preds[self.target_idx].ravel(),
                feature_values,
                ax,
                pd_line_idx,
                pd_line_kw,
            )

        trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
        # create the decile line for the vertical axis
        vlines_idx = np.unravel_index(pd_plot_idx, self.deciles_vlines_.shape)
        self.deciles_vlines_[vlines_idx] = ax.vlines(
            self.deciles[feature_idx[0]],
            0,
            0.05,
            transform=trans,
            color="k",
        )
        # reset ylim which was overwritten by vlines
        ax.set_ylim(self.pdp_lim[1])

        # Set xlabel if it is not already set
        if not ax.get_xlabel():
            ax.set_xlabel(self.feature_names[feature_idx[0]])

        if n_cols is None or pd_plot_idx % n_cols == 0:
            if not ax.get_ylabel():
                ax.set_ylabel("Partial dependence")
        else:
            ax.set_yticklabels([])

        if pd_line_kw.get("label", None) and self.kind != "individual":
            ax.legend()

    def _plot_two_way_partial_dependence(
        self,
        avg_preds,
        feature_values,
        feature_idx,
        ax,
        pd_plot_idx,
        Z_level,
        contour_kw,
    ):
        """Plot 2-way partial dependence.

        Parameters
        ----------
        avg_preds : ndarray of shape \
                (n_instances, n_grid_points, n_grid_points)
            The average predictions for all points of `feature_values[0]` and
            `feature_values[1]` for some given features for all samples in `X`.
        feature_values : seq of 1d array
            A sequence of array of the feature values for which the predictions
            have been computed.
        feature_idx : tuple of int
            The indices of the target features
        ax : Matplotlib axes
            The axis on which to plot the ICE and PDP lines.
        pd_plot_idx : int
            The sequential index of the plot. It will be unraveled to find the
            matching 2D position in the grid layout.
        Z_level : ndarray of shape (8, 8)
            The Z-level used to encode the average predictions.
        contour_kw : dict
            Dict with keywords passed when plotting the contours.
        """
        from matplotlib import transforms  # noqa

        XX, YY = np.meshgrid(feature_values[0], feature_values[1])
        Z = avg_preds[self.target_idx].T
        CS = ax.contour(XX, YY, Z, levels=Z_level, linewidths=0.5, colors="k")
        contour_idx = np.unravel_index(pd_plot_idx, self.contours_.shape)
        self.contours_[contour_idx] = ax.contourf(
            XX,
            YY,
            Z,
            levels=Z_level,
            vmax=Z_level[-1],
            vmin=Z_level[0],
            **contour_kw,
        )
        ax.clabel(CS, fmt="%2.2f", colors="k", fontsize=10, inline=True)

        trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
        # create the decile line for the vertical axis
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        vlines_idx = np.unravel_index(pd_plot_idx, self.deciles_vlines_.shape)
        self.deciles_vlines_[vlines_idx] = ax.vlines(
            self.deciles[feature_idx[0]],
            0,
            0.05,
            transform=trans,
            color="k",
        )
        # create the decile line for the horizontal axis
        hlines_idx = np.unravel_index(pd_plot_idx, self.deciles_hlines_.shape)
        self.deciles_hlines_[hlines_idx] = ax.hlines(
            self.deciles[feature_idx[1]],
            0,
            0.05,
            transform=trans,
            color="k",
        )
        # reset xlim and ylim since they are overwritten by hlines and vlines
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        # set xlabel if it is not already set
        if not ax.get_xlabel():
            ax.set_xlabel(self.feature_names[feature_idx[0]])
        ax.set_ylabel(self.feature_names[feature_idx[1]])

    @_deprecate_positional_args(version="1.1")
    def plot(
        self,
        *,
        ax=None,
        n_cols=3,
        line_kw=None,
        ice_lines_kw=None,
        pd_line_kw=None,
        contour_kw=None,
    ):
        """Plot partial dependence plots.

        Parameters
        ----------
        ax : Matplotlib axes or array-like of Matplotlib axes, default=None
            - If a single axis is passed in, it is treated as a bounding axes
                and a grid of partial dependence plots will be drawn within
                these bounds. The `n_cols` parameter controls the number of
                columns in the grid.
            - If an array-like of axes are passed in, the partial dependence
                plots will be drawn directly into these axes.
            - If `None`, a figure and a bounding axes is created and treated
                as the single axes case.

        n_cols : int, default=3
            The maximum number of columns in the grid plot. Only active when
            `ax` is a single axes or `None`.

        line_kw : dict, default=None
            Dict with keywords passed to the `matplotlib.pyplot.plot` call.
            For one-way partial dependence plots.

        ice_lines_kw : dict, default=None
            Dictionary with keywords passed to the `matplotlib.pyplot.plot` call.
            For ICE lines in the one-way partial dependence plots.
            The key value pairs defined in `ice_lines_kw` takes priority over
            `line_kw`.

            .. versionadded:: 1.0

        pd_line_kw : dict, default=None
            Dictionary with keywords passed to the `matplotlib.pyplot.plot` call.
            For partial dependence in one-way partial dependence plots.
            The key value pairs defined in `pd_line_kw` takes priority over
            `line_kw`.

            .. versionadded:: 1.0

        contour_kw : dict, default=None
            Dict with keywords passed to the `matplotlib.pyplot.contourf`
            call for two-way partial dependence plots.

        Returns
        -------
        display : :class:`~sklearn.inspection.PartialDependenceDisplay`
        """

        check_matplotlib_support("plot_partial_dependence")
        import matplotlib.pyplot as plt  # noqa
        from matplotlib.gridspec import GridSpecFromSubplotSpec  # noqa

        if line_kw is None:
            line_kw = {}
        if ice_lines_kw is None:
            ice_lines_kw = {}
        if pd_line_kw is None:
            pd_line_kw = {}
        if contour_kw is None:
            contour_kw = {}

        if ax is None:
            _, ax = plt.subplots()

        default_contour_kws = {"alpha": 0.75}
        contour_kw = {**default_contour_kws, **contour_kw}

        default_line_kws = {
            "color": "C0",
            "label": "average" if self.kind == "both" else None,
        }
        if self.kind in ("individual", "both"):
            default_ice_lines_kws = {"alpha": 0.3, "linewidth": 0.5}
        else:
            default_ice_lines_kws = {}

        ice_lines_kw = {
            **default_line_kws,
            **line_kw,
            **default_ice_lines_kws,
            **ice_lines_kw,
        }
        del ice_lines_kw["label"]

        pd_line_kw = {**default_line_kws, **line_kw, **pd_line_kw}

        n_features = len(self.features)
        if self.kind in ("individual", "both"):
            n_ice_lines = self._get_sample_count(len(self.pd_results[0].individual[0]))
            if self.kind == "individual":
                n_lines = n_ice_lines
            else:
                n_lines = n_ice_lines + 1
        else:
            n_ice_lines = 0
            n_lines = 1

        if isinstance(ax, plt.Axes):
            # If ax was set off, it has most likely been set to off
            # by a previous call to plot.
            if not ax.axison:
                raise ValueError(
                    "The ax was already used in another plot "
                    "function, please set ax=display.axes_ "
                    "instead"
                )

            ax.set_axis_off()
            self.bounding_ax_ = ax
            self.figure_ = ax.figure

            n_cols = min(n_cols, n_features)
            n_rows = int(np.ceil(n_features / float(n_cols)))

            self.axes_ = np.empty((n_rows, n_cols), dtype=object)
            if self.kind == "average":
                self.lines_ = np.empty((n_rows, n_cols), dtype=object)
            else:
                self.lines_ = np.empty((n_rows, n_cols, n_lines), dtype=object)
            self.contours_ = np.empty((n_rows, n_cols), dtype=object)

            axes_ravel = self.axes_.ravel()

            gs = GridSpecFromSubplotSpec(
                n_rows, n_cols, subplot_spec=ax.get_subplotspec()
            )
            for i, spec in zip(range(n_features), gs):
                axes_ravel[i] = self.figure_.add_subplot(spec)

        else:  # array-like
            ax = np.asarray(ax, dtype=object)
            if ax.size != n_features:
                raise ValueError(
                    "Expected ax to have {} axes, got {}".format(n_features, ax.size)
                )

            if ax.ndim == 2:
                n_cols = ax.shape[1]
            else:
                n_cols = None

            self.bounding_ax_ = None
            self.figure_ = ax.ravel()[0].figure
            self.axes_ = ax
            if self.kind == "average":
                self.lines_ = np.empty_like(ax, dtype=object)
            else:
                self.lines_ = np.empty(ax.shape + (n_lines,), dtype=object)
            self.contours_ = np.empty_like(ax, dtype=object)

        # create contour levels for two-way plots
        if 2 in self.pdp_lim:
            Z_level = np.linspace(*self.pdp_lim[2], num=8)

        self.deciles_vlines_ = np.empty_like(self.axes_, dtype=object)
        self.deciles_hlines_ = np.empty_like(self.axes_, dtype=object)

        for pd_plot_idx, (axi, feature_idx, pd_result) in enumerate(
            zip(self.axes_.ravel(), self.features, self.pd_results)
        ):
            avg_preds = None
            preds = None
            feature_values = pd_result["values"]
            if self.kind == "individual":
                preds = pd_result.individual
            elif self.kind == "average":
                avg_preds = pd_result.average
            else:  # kind='both'
                avg_preds = pd_result.average
                preds = pd_result.individual

            if len(feature_values) == 1:
                self._plot_one_way_partial_dependence(
                    preds,
                    avg_preds,
                    feature_values[0],
                    feature_idx,
                    n_ice_lines,
                    axi,
                    n_cols,
                    pd_plot_idx,
                    n_lines,
                    ice_lines_kw,
                    pd_line_kw,
                )
            else:
                self._plot_two_way_partial_dependence(
                    avg_preds,
                    feature_values,
                    feature_idx,
                    axi,
                    pd_plot_idx,
                    Z_level,
                    contour_kw,
                )

        return self
