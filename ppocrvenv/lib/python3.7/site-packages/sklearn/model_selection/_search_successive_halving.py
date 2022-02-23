from copy import deepcopy
from math import ceil, floor, log
from abc import abstractmethod
from numbers import Integral

import numpy as np
from ._search import _check_param_grid
from ._search import BaseSearchCV
from . import ParameterGrid, ParameterSampler
from ..base import is_classifier
from ._split import check_cv, _yields_constant_splits
from ..utils import resample
from ..utils.multiclass import check_classification_targets
from ..utils.validation import _num_samples


__all__ = ["HalvingGridSearchCV", "HalvingRandomSearchCV"]


class _SubsampleMetaSplitter:
    """Splitter that subsamples a given fraction of the dataset"""

    def __init__(self, *, base_cv, fraction, subsample_test, random_state):
        self.base_cv = base_cv
        self.fraction = fraction
        self.subsample_test = subsample_test
        self.random_state = random_state

    def split(self, X, y, groups=None):
        for train_idx, test_idx in self.base_cv.split(X, y, groups):
            train_idx = resample(
                train_idx,
                replace=False,
                random_state=self.random_state,
                n_samples=int(self.fraction * train_idx.shape[0]),
            )
            if self.subsample_test:
                test_idx = resample(
                    test_idx,
                    replace=False,
                    random_state=self.random_state,
                    n_samples=int(self.fraction * test_idx.shape[0]),
                )
            yield train_idx, test_idx


def _top_k(results, k, itr):
    # Return the best candidates of a given iteration
    iteration, mean_test_score, params = (
        np.asarray(a)
        for a in (results["iter"], results["mean_test_score"], results["params"])
    )
    iter_indices = np.flatnonzero(iteration == itr)
    sorted_indices = np.argsort(mean_test_score[iter_indices])
    return np.array(params[iter_indices][sorted_indices[-k:]])


class BaseSuccessiveHalving(BaseSearchCV):
    """Implements successive halving.

    Ref:
    Almost optimal exploration in multi-armed bandits, ICML 13
    Zohar Karnin, Tomer Koren, Oren Somekh
    """

    def __init__(
        self,
        estimator,
        *,
        scoring=None,
        n_jobs=None,
        refit=True,
        cv=5,
        verbose=0,
        random_state=None,
        error_score=np.nan,
        return_train_score=True,
        max_resources="auto",
        min_resources="exhaust",
        resource="n_samples",
        factor=3,
        aggressive_elimination=False,
    ):
        super().__init__(
            estimator,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=refit,
            cv=cv,
            verbose=verbose,
            error_score=error_score,
            return_train_score=return_train_score,
        )

        self.random_state = random_state
        self.max_resources = max_resources
        self.resource = resource
        self.factor = factor
        self.min_resources = min_resources
        self.aggressive_elimination = aggressive_elimination

    def _check_input_parameters(self, X, y, groups):

        if self.scoring is not None and not (
            isinstance(self.scoring, str) or callable(self.scoring)
        ):
            raise ValueError(
                "scoring parameter must be a string, "
                "a callable or None. Multimetric scoring is not "
                "supported."
            )

        # We need to enforce that successive calls to cv.split() yield the same
        # splits: see https://github.com/scikit-learn/scikit-learn/issues/15149
        if not _yields_constant_splits(self._checked_cv_orig):
            raise ValueError(
                "The cv parameter must yield consistent folds across "
                "calls to split(). Set its random_state to an int, or set "
                "shuffle=False."
            )

        if (
            self.resource != "n_samples"
            and self.resource not in self.estimator.get_params()
        ):
            raise ValueError(
                f"Cannot use resource={self.resource} which is not supported "
                f"by estimator {self.estimator.__class__.__name__}"
            )

        if isinstance(self.max_resources, str) and self.max_resources != "auto":
            raise ValueError(
                "max_resources must be either 'auto' or a positive integer"
            )
        if self.max_resources != "auto" and (
            not isinstance(self.max_resources, Integral) or self.max_resources <= 0
        ):
            raise ValueError(
                "max_resources must be either 'auto' or a positive integer"
            )

        if self.min_resources not in ("smallest", "exhaust") and (
            not isinstance(self.min_resources, Integral) or self.min_resources <= 0
        ):
            raise ValueError(
                "min_resources must be either 'smallest', 'exhaust', "
                "or a positive integer "
                "no greater than max_resources."
            )

        if isinstance(self, HalvingRandomSearchCV):
            if self.min_resources == self.n_candidates == "exhaust":
                # for n_candidates=exhaust to work, we need to know what
                # min_resources is. Similarly min_resources=exhaust needs to
                # know the actual number of candidates.
                raise ValueError(
                    "n_candidates and min_resources cannot be both set to 'exhaust'."
                )
            if self.n_candidates != "exhaust" and (
                not isinstance(self.n_candidates, Integral) or self.n_candidates <= 0
            ):
                raise ValueError(
                    "n_candidates must be either 'exhaust' or a positive integer"
                )

        self.min_resources_ = self.min_resources
        if self.min_resources_ in ("smallest", "exhaust"):
            if self.resource == "n_samples":
                n_splits = self._checked_cv_orig.get_n_splits(X, y, groups)
                # please see https://gph.is/1KjihQe for a justification
                magic_factor = 2
                self.min_resources_ = n_splits * magic_factor
                if is_classifier(self.estimator):
                    y = self._validate_data(X="no_validation", y=y)
                    check_classification_targets(y)
                    n_classes = np.unique(y).shape[0]
                    self.min_resources_ *= n_classes
            else:
                self.min_resources_ = 1
            # if 'exhaust', min_resources_ might be set to a higher value later
            # in _run_search

        self.max_resources_ = self.max_resources
        if self.max_resources_ == "auto":
            if not self.resource == "n_samples":
                raise ValueError(
                    "max_resources can only be 'auto' if resource='n_samples'"
                )
            self.max_resources_ = _num_samples(X)

        if self.min_resources_ > self.max_resources_:
            raise ValueError(
                f"min_resources_={self.min_resources_} is greater "
                f"than max_resources_={self.max_resources_}."
            )

        if self.min_resources_ == 0:
            raise ValueError(
                f"min_resources_={self.min_resources_}: you might have passed "
                "an empty dataset X."
            )

        if not isinstance(self.refit, bool):
            raise ValueError(
                f"refit is expected to be a boolean. Got {type(self.refit)} instead."
            )

    @staticmethod
    def _select_best_index(refit, refit_metric, results):
        """Custom refit callable to return the index of the best candidate.

        We want the best candidate out of the last iteration. By default
        BaseSearchCV would return the best candidate out of all iterations.

        Currently, we only support for a single metric thus `refit` and
        `refit_metric` are not required.
        """
        last_iter = np.max(results["iter"])
        last_iter_indices = np.flatnonzero(results["iter"] == last_iter)
        best_idx = np.argmax(results["mean_test_score"][last_iter_indices])
        return last_iter_indices[best_idx]

    def fit(self, X, y=None, groups=None, **fit_params):
        """Run fit with all sets of parameters.

        Parameters
        ----------

        X : array-like, shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like, shape (n_samples,) or (n_samples, n_output), optional
            Target relative to X for classification or regression;
            None for unsupervised learning.

        groups : array-like of shape (n_samples,), default=None
            Group labels for the samples used while splitting the dataset into
            train/test set. Only used in conjunction with a "Group" :term:`cv`
            instance (e.g., :class:`~sklearn.model_selection.GroupKFold`).

        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of the estimator.

        Returns
        -------
        self : object
            Instance of fitted estimator.
        """
        self._checked_cv_orig = check_cv(
            self.cv, y, classifier=is_classifier(self.estimator)
        )

        self._check_input_parameters(
            X=X,
            y=y,
            groups=groups,
        )

        self._n_samples_orig = _num_samples(X)

        super().fit(X, y=y, groups=groups, **fit_params)

        # Set best_score_: BaseSearchCV does not set it, as refit is a callable
        self.best_score_ = self.cv_results_["mean_test_score"][self.best_index_]

        return self

    def _run_search(self, evaluate_candidates):
        candidate_params = self._generate_candidate_params()

        if self.resource != "n_samples" and any(
            self.resource in candidate for candidate in candidate_params
        ):
            # Can only check this now since we need the candidates list
            raise ValueError(
                f"Cannot use parameter {self.resource} as the resource since "
                "it is part of the searched parameters."
            )

        # n_required_iterations is the number of iterations needed so that the
        # last iterations evaluates less than `factor` candidates.
        n_required_iterations = 1 + floor(log(len(candidate_params), self.factor))

        if self.min_resources == "exhaust":
            # To exhaust the resources, we want to start with the biggest
            # min_resources possible so that the last (required) iteration
            # uses as many resources as possible
            last_iteration = n_required_iterations - 1
            self.min_resources_ = max(
                self.min_resources_,
                self.max_resources_ // self.factor ** last_iteration,
            )

        # n_possible_iterations is the number of iterations that we can
        # actually do starting from min_resources and without exceeding
        # max_resources. Depending on max_resources and the number of
        # candidates, this may be higher or smaller than
        # n_required_iterations.
        n_possible_iterations = 1 + floor(
            log(self.max_resources_ // self.min_resources_, self.factor)
        )

        if self.aggressive_elimination:
            n_iterations = n_required_iterations
        else:
            n_iterations = min(n_possible_iterations, n_required_iterations)

        if self.verbose:
            print(f"n_iterations: {n_iterations}")
            print(f"n_required_iterations: {n_required_iterations}")
            print(f"n_possible_iterations: {n_possible_iterations}")
            print(f"min_resources_: {self.min_resources_}")
            print(f"max_resources_: {self.max_resources_}")
            print(f"aggressive_elimination: {self.aggressive_elimination}")
            print(f"factor: {self.factor}")

        self.n_resources_ = []
        self.n_candidates_ = []

        for itr in range(n_iterations):

            power = itr  # default
            if self.aggressive_elimination:
                # this will set n_resources to the initial value (i.e. the
                # value of n_resources at the first iteration) for as many
                # iterations as needed (while candidates are being
                # eliminated), and then go on as usual.
                power = max(0, itr - n_required_iterations + n_possible_iterations)

            n_resources = int(self.factor ** power * self.min_resources_)
            # guard, probably not needed
            n_resources = min(n_resources, self.max_resources_)
            self.n_resources_.append(n_resources)

            n_candidates = len(candidate_params)
            self.n_candidates_.append(n_candidates)

            if self.verbose:
                print("-" * 10)
                print(f"iter: {itr}")
                print(f"n_candidates: {n_candidates}")
                print(f"n_resources: {n_resources}")

            if self.resource == "n_samples":
                # subsampling will be done in cv.split()
                cv = _SubsampleMetaSplitter(
                    base_cv=self._checked_cv_orig,
                    fraction=n_resources / self._n_samples_orig,
                    subsample_test=True,
                    random_state=self.random_state,
                )

            else:
                # Need copy so that the n_resources of next iteration does
                # not overwrite
                candidate_params = [c.copy() for c in candidate_params]
                for candidate in candidate_params:
                    candidate[self.resource] = n_resources
                cv = self._checked_cv_orig

            more_results = {
                "iter": [itr] * n_candidates,
                "n_resources": [n_resources] * n_candidates,
            }

            results = evaluate_candidates(
                candidate_params, cv, more_results=more_results
            )

            n_candidates_to_keep = ceil(n_candidates / self.factor)
            candidate_params = _top_k(results, n_candidates_to_keep, itr)

        self.n_remaining_candidates_ = len(candidate_params)
        self.n_required_iterations_ = n_required_iterations
        self.n_possible_iterations_ = n_possible_iterations
        self.n_iterations_ = n_iterations

    @abstractmethod
    def _generate_candidate_params(self):
        pass

    def _more_tags(self):
        tags = deepcopy(super()._more_tags())
        tags["_xfail_checks"].update(
            {
                "check_fit2d_1sample": (
                    "Fail during parameter check since min/max resources requires"
                    " more samples"
                ),
            }
        )
        return tags


class HalvingGridSearchCV(BaseSuccessiveHalving):
    """Search over specified parameter values with successive halving.

    The search strategy starts evaluating all the candidates with a small
    amount of resources and iteratively selects the best candidates, using
    more and more resources.

    Read more in the :ref:`User guide <successive_halving_user_guide>`.

    .. note::

      This estimator is still **experimental** for now: the predictions
      and the API might change without any deprecation cycle. To use it,
      you need to explicitly import ``enable_halving_search_cv``::

        >>> # explicitly require this experimental feature
        >>> from sklearn.experimental import enable_halving_search_cv # noqa
        >>> # now you can import normally from model_selection
        >>> from sklearn.model_selection import HalvingGridSearchCV

    Parameters
    ----------
    estimator : estimator object
        This is assumed to implement the scikit-learn estimator interface.
        Either estimator needs to provide a ``score`` function,
        or ``scoring`` must be passed.

    param_grid : dict or list of dictionaries
        Dictionary with parameters names (string) as keys and lists of
        parameter settings to try as values, or a list of such
        dictionaries, in which case the grids spanned by each dictionary
        in the list are explored. This enables searching over any sequence
        of parameter settings.

    factor : int or float, default=3
        The 'halving' parameter, which determines the proportion of candidates
        that are selected for each subsequent iteration. For example,
        ``factor=3`` means that only one third of the candidates are selected.

    resource : ``'n_samples'`` or str, default='n_samples'
        Defines the resource that increases with each iteration. By default,
        the resource is the number of samples. It can also be set to any
        parameter of the base estimator that accepts positive integer
        values, e.g. 'n_iterations' or 'n_estimators' for a gradient
        boosting estimator. In this case ``max_resources`` cannot be 'auto'
        and must be set explicitly.

    max_resources : int, default='auto'
        The maximum amount of resource that any candidate is allowed to use
        for a given iteration. By default, this is set to ``n_samples`` when
        ``resource='n_samples'`` (default), else an error is raised.

    min_resources : {'exhaust', 'smallest'} or int, default='exhaust'
        The minimum amount of resource that any candidate is allowed to use
        for a given iteration. Equivalently, this defines the amount of
        resources `r0` that are allocated for each candidate at the first
        iteration.

        - 'smallest' is a heuristic that sets `r0` to a small value:

            - ``n_splits * 2`` when ``resource='n_samples'`` for a regression
              problem
            - ``n_classes * n_splits * 2`` when ``resource='n_samples'`` for a
              classification problem
            - ``1`` when ``resource != 'n_samples'``

        - 'exhaust' will set `r0` such that the **last** iteration uses as
          much resources as possible. Namely, the last iteration will use the
          highest value smaller than ``max_resources`` that is a multiple of
          both ``min_resources`` and ``factor``. In general, using 'exhaust'
          leads to a more accurate estimator, but is slightly more time
          consuming.

        Note that the amount of resources used at each iteration is always a
        multiple of ``min_resources``.

    aggressive_elimination : bool, default=False
        This is only relevant in cases where there isn't enough resources to
        reduce the remaining candidates to at most `factor` after the last
        iteration. If ``True``, then the search process will 'replay' the
        first iteration for as long as needed until the number of candidates
        is small enough. This is ``False`` by default, which means that the
        last iteration may evaluate more than ``factor`` candidates. See
        :ref:`aggressive_elimination` for more details.

    cv : int, cross-validation generator or iterable, default=5
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - integer, to specify the number of folds in a `(Stratified)KFold`,
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used. These splitters are instantiated
        with `shuffle=False` so the splits will be the same across calls.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

        .. note::
            Due to implementation details, the folds produced by `cv` must be
            the same across multiple calls to `cv.split()`. For
            built-in `scikit-learn` iterators, this can be achieved by
            deactivating shuffling (`shuffle=False`), or by setting the
            `cv`'s `random_state` parameter to an integer.

    scoring : str, callable, or None, default=None
        A single string (see :ref:`scoring_parameter`) or a callable
        (see :ref:`scoring`) to evaluate the predictions on the test set.
        If None, the estimator's score method is used.

    refit : bool, default=True
        If True, refit an estimator using the best found parameters on the
        whole dataset.

        The refitted estimator is made available at the ``best_estimator_``
        attribute and permits using ``predict`` directly on this
        ``HalvingGridSearchCV`` instance.

    error_score : 'raise' or numeric
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised. If a numeric value is given,
        FitFailedWarning is raised. This parameter does not affect the refit
        step, which will always raise the error. Default is ``np.nan``.

    return_train_score : bool, default=False
        If ``False``, the ``cv_results_`` attribute will not include training
        scores.
        Computing training scores is used to get insights on how different
        parameter settings impact the overfitting/underfitting trade-off.
        However computing the scores on the training set can be computationally
        expensive and is not strictly required to select the parameters that
        yield the best generalization performance.

    random_state : int, RandomState instance or None, default=None
        Pseudo random number generator state used for subsampling the dataset
        when `resources != 'n_samples'`. Ignored otherwise.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    n_jobs : int or None, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    verbose : int
        Controls the verbosity: the higher, the more messages.

    Attributes
    ----------
    n_resources_ : list of int
        The amount of resources used at each iteration.

    n_candidates_ : list of int
        The number of candidate parameters that were evaluated at each
        iteration.

    n_remaining_candidates_ : int
        The number of candidate parameters that are left after the last
        iteration. It corresponds to `ceil(n_candidates[-1] / factor)`

    max_resources_ : int
        The maximum number of resources that any candidate is allowed to use
        for a given iteration. Note that since the number of resources used
        at each iteration must be a multiple of ``min_resources_``, the
        actual number of resources used at the last iteration may be smaller
        than ``max_resources_``.

    min_resources_ : int
        The amount of resources that are allocated for each candidate at the
        first iteration.

    n_iterations_ : int
        The actual number of iterations that were run. This is equal to
        ``n_required_iterations_`` if ``aggressive_elimination`` is ``True``.
        Else, this is equal to ``min(n_possible_iterations_,
        n_required_iterations_)``.

    n_possible_iterations_ : int
        The number of iterations that are possible starting with
        ``min_resources_`` resources and without exceeding
        ``max_resources_``.

    n_required_iterations_ : int
        The number of iterations that are required to end up with less than
        ``factor`` candidates at the last iteration, starting with
        ``min_resources_`` resources. This will be smaller than
        ``n_possible_iterations_`` when there isn't enough resources.

    cv_results_ : dict of numpy (masked) ndarrays
        A dict with keys as column headers and values as columns, that can be
        imported into a pandas ``DataFrame``. It contains lots of information
        for analysing the results of a search.
        Please refer to the :ref:`User guide<successive_halving_cv_results>`
        for details.

    best_estimator_ : estimator or dict
        Estimator that was chosen by the search, i.e. estimator
        which gave highest score (or smallest loss if specified)
        on the left out data. Not available if ``refit=False``.

    best_score_ : float
        Mean cross-validated score of the best_estimator.

    best_params_ : dict
        Parameter setting that gave the best results on the hold out data.

    best_index_ : int
        The index (of the ``cv_results_`` arrays) which corresponds to the best
        candidate parameter setting.

        The dict at ``search.cv_results_['params'][search.best_index_]`` gives
        the parameter setting for the best model, that gives the highest
        mean score (``search.best_score_``).

    scorer_ : function or a dict
        Scorer function used on the held out data to choose the best
        parameters for the model.

    n_splits_ : int
        The number of cross-validation splits (folds/iterations).

    refit_time_ : float
        Seconds used for refitting the best model on the whole dataset.

        This is present only if ``refit`` is not False.

    multimetric_ : bool
        Whether or not the scorers compute several metrics.

    classes_ : ndarray of shape (n_classes,)
        The classes labels. This is present only if ``refit`` is specified and
        the underlying estimator is a classifier.

    n_features_in_ : int
        Number of features seen during :term:`fit`. Only defined if
        `best_estimator_` is defined (see the documentation for the `refit`
        parameter for more details) and that `best_estimator_` exposes
        `n_features_in_` when fit.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Only defined if
        `best_estimator_` is defined (see the documentation for the `refit`
        parameter for more details) and that `best_estimator_` exposes
        `feature_names_in_` when fit.

        .. versionadded:: 1.0

    See Also
    --------
    :class:`HalvingRandomSearchCV`:
        Random search over a set of parameters using successive halving.

    Notes
    -----
    The parameters selected are those that maximize the score of the held-out
    data, according to the scoring parameter.

    Examples
    --------

    >>> from sklearn.datasets import load_iris
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.experimental import enable_halving_search_cv  # noqa
    >>> from sklearn.model_selection import HalvingGridSearchCV
    ...
    >>> X, y = load_iris(return_X_y=True)
    >>> clf = RandomForestClassifier(random_state=0)
    ...
    >>> param_grid = {"max_depth": [3, None],
    ...               "min_samples_split": [5, 10]}
    >>> search = HalvingGridSearchCV(clf, param_grid, resource='n_estimators',
    ...                              max_resources=10,
    ...                              random_state=0).fit(X, y)
    >>> search.best_params_  # doctest: +SKIP
    {'max_depth': None, 'min_samples_split': 10, 'n_estimators': 9}
    """

    _required_parameters = ["estimator", "param_grid"]

    def __init__(
        self,
        estimator,
        param_grid,
        *,
        factor=3,
        resource="n_samples",
        max_resources="auto",
        min_resources="exhaust",
        aggressive_elimination=False,
        cv=5,
        scoring=None,
        refit=True,
        error_score=np.nan,
        return_train_score=True,
        random_state=None,
        n_jobs=None,
        verbose=0,
    ):
        super().__init__(
            estimator,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=refit,
            verbose=verbose,
            cv=cv,
            random_state=random_state,
            error_score=error_score,
            return_train_score=return_train_score,
            max_resources=max_resources,
            resource=resource,
            factor=factor,
            min_resources=min_resources,
            aggressive_elimination=aggressive_elimination,
        )
        self.param_grid = param_grid
        _check_param_grid(self.param_grid)

    def _generate_candidate_params(self):
        return ParameterGrid(self.param_grid)


class HalvingRandomSearchCV(BaseSuccessiveHalving):
    """Randomized search on hyper parameters.

    The search strategy starts evaluating all the candidates with a small
    amount of resources and iteratively selects the best candidates, using more
    and more resources.

    The candidates are sampled at random from the parameter space and the
    number of sampled candidates is determined by ``n_candidates``.

    Read more in the :ref:`User guide<successive_halving_user_guide>`.

    .. note::

      This estimator is still **experimental** for now: the predictions
      and the API might change without any deprecation cycle. To use it,
      you need to explicitly import ``enable_halving_search_cv``::

        >>> # explicitly require this experimental feature
        >>> from sklearn.experimental import enable_halving_search_cv # noqa
        >>> # now you can import normally from model_selection
        >>> from sklearn.model_selection import HalvingRandomSearchCV

    Parameters
    ----------
    estimator : estimator object
        This is assumed to implement the scikit-learn estimator interface.
        Either estimator needs to provide a ``score`` function,
        or ``scoring`` must be passed.

    param_distributions : dict
        Dictionary with parameters names (string) as keys and distributions
        or lists of parameters to try. Distributions must provide a ``rvs``
        method for sampling (such as those from scipy.stats.distributions).
        If a list is given, it is sampled uniformly.

    n_candidates : int, default='exhaust'
        The number of candidate parameters to sample, at the first
        iteration. Using 'exhaust' will sample enough candidates so that the
        last iteration uses as many resources as possible, based on
        `min_resources`, `max_resources` and `factor`. In this case,
        `min_resources` cannot be 'exhaust'.

    factor : int or float, default=3
        The 'halving' parameter, which determines the proportion of candidates
        that are selected for each subsequent iteration. For example,
        ``factor=3`` means that only one third of the candidates are selected.

    resource : ``'n_samples'`` or str, default='n_samples'
        Defines the resource that increases with each iteration. By default,
        the resource is the number of samples. It can also be set to any
        parameter of the base estimator that accepts positive integer
        values, e.g. 'n_iterations' or 'n_estimators' for a gradient
        boosting estimator. In this case ``max_resources`` cannot be 'auto'
        and must be set explicitly.

    max_resources : int, default='auto'
        The maximum number of resources that any candidate is allowed to use
        for a given iteration. By default, this is set ``n_samples`` when
        ``resource='n_samples'`` (default), else an error is raised.

    min_resources : {'exhaust', 'smallest'} or int, default='smallest'
        The minimum amount of resource that any candidate is allowed to use
        for a given iteration. Equivalently, this defines the amount of
        resources `r0` that are allocated for each candidate at the first
        iteration.

        - 'smallest' is a heuristic that sets `r0` to a small value:

            - ``n_splits * 2`` when ``resource='n_samples'`` for a regression
              problem
            - ``n_classes * n_splits * 2`` when ``resource='n_samples'`` for a
              classification problem
            - ``1`` when ``resource != 'n_samples'``

        - 'exhaust' will set `r0` such that the **last** iteration uses as
          much resources as possible. Namely, the last iteration will use the
          highest value smaller than ``max_resources`` that is a multiple of
          both ``min_resources`` and ``factor``. In general, using 'exhaust'
          leads to a more accurate estimator, but is slightly more time
          consuming. 'exhaust' isn't available when `n_candidates='exhaust'`.

        Note that the amount of resources used at each iteration is always a
        multiple of ``min_resources``.

    aggressive_elimination : bool, default=False
        This is only relevant in cases where there isn't enough resources to
        reduce the remaining candidates to at most `factor` after the last
        iteration. If ``True``, then the search process will 'replay' the
        first iteration for as long as needed until the number of candidates
        is small enough. This is ``False`` by default, which means that the
        last iteration may evaluate more than ``factor`` candidates. See
        :ref:`aggressive_elimination` for more details.

    cv : int, cross-validation generator or an iterable, default=5
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - integer, to specify the number of folds in a `(Stratified)KFold`,
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used. These splitters are instantiated
        with `shuffle=False` so the splits will be the same across calls.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

        .. note::
            Due to implementation details, the folds produced by `cv` must be
            the same across multiple calls to `cv.split()`. For
            built-in `scikit-learn` iterators, this can be achieved by
            deactivating shuffling (`shuffle=False`), or by setting the
            `cv`'s `random_state` parameter to an integer.

    scoring : str, callable, or None, default=None
        A single string (see :ref:`scoring_parameter`) or a callable
        (see :ref:`scoring`) to evaluate the predictions on the test set.
        If None, the estimator's score method is used.

    refit : bool, default=True
        If True, refit an estimator using the best found parameters on the
        whole dataset.

        The refitted estimator is made available at the ``best_estimator_``
        attribute and permits using ``predict`` directly on this
        ``HalvingRandomSearchCV`` instance.

    error_score : 'raise' or numeric
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised. If a numeric value is given,
        FitFailedWarning is raised. This parameter does not affect the refit
        step, which will always raise the error. Default is ``np.nan``.

    return_train_score : bool, default=False
        If ``False``, the ``cv_results_`` attribute will not include training
        scores.
        Computing training scores is used to get insights on how different
        parameter settings impact the overfitting/underfitting trade-off.
        However computing the scores on the training set can be computationally
        expensive and is not strictly required to select the parameters that
        yield the best generalization performance.

    random_state : int, RandomState instance or None, default=None
        Pseudo random number generator state used for subsampling the dataset
        when `resources != 'n_samples'`. Also used for random uniform
        sampling from lists of possible values instead of scipy.stats
        distributions.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    n_jobs : int or None, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    verbose : int
        Controls the verbosity: the higher, the more messages.

    Attributes
    ----------
    n_resources_ : list of int
        The amount of resources used at each iteration.

    n_candidates_ : list of int
        The number of candidate parameters that were evaluated at each
        iteration.

    n_remaining_candidates_ : int
        The number of candidate parameters that are left after the last
        iteration. It corresponds to `ceil(n_candidates[-1] / factor)`

    max_resources_ : int
        The maximum number of resources that any candidate is allowed to use
        for a given iteration. Note that since the number of resources used at
        each iteration must be a multiple of ``min_resources_``, the actual
        number of resources used at the last iteration may be smaller than
        ``max_resources_``.

    min_resources_ : int
        The amount of resources that are allocated for each candidate at the
        first iteration.

    n_iterations_ : int
        The actual number of iterations that were run. This is equal to
        ``n_required_iterations_`` if ``aggressive_elimination`` is ``True``.
        Else, this is equal to ``min(n_possible_iterations_,
        n_required_iterations_)``.

    n_possible_iterations_ : int
        The number of iterations that are possible starting with
        ``min_resources_`` resources and without exceeding
        ``max_resources_``.

    n_required_iterations_ : int
        The number of iterations that are required to end up with less than
        ``factor`` candidates at the last iteration, starting with
        ``min_resources_`` resources. This will be smaller than
        ``n_possible_iterations_`` when there isn't enough resources.

    cv_results_ : dict of numpy (masked) ndarrays
        A dict with keys as column headers and values as columns, that can be
        imported into a pandas ``DataFrame``. It contains lots of information
        for analysing the results of a search.
        Please refer to the :ref:`User guide<successive_halving_cv_results>`
        for details.

    best_estimator_ : estimator or dict
        Estimator that was chosen by the search, i.e. estimator
        which gave highest score (or smallest loss if specified)
        on the left out data. Not available if ``refit=False``.

    best_score_ : float
        Mean cross-validated score of the best_estimator.

    best_params_ : dict
        Parameter setting that gave the best results on the hold out data.

    best_index_ : int
        The index (of the ``cv_results_`` arrays) which corresponds to the best
        candidate parameter setting.

        The dict at ``search.cv_results_['params'][search.best_index_]`` gives
        the parameter setting for the best model, that gives the highest
        mean score (``search.best_score_``).

    scorer_ : function or a dict
        Scorer function used on the held out data to choose the best
        parameters for the model.

    n_splits_ : int
        The number of cross-validation splits (folds/iterations).

    refit_time_ : float
        Seconds used for refitting the best model on the whole dataset.

        This is present only if ``refit`` is not False.

    multimetric_ : bool
        Whether or not the scorers compute several metrics.

    classes_ : ndarray of shape (n_classes,)
        The classes labels. This is present only if ``refit`` is specified and
        the underlying estimator is a classifier.

    n_features_in_ : int
        Number of features seen during :term:`fit`. Only defined if
        `best_estimator_` is defined (see the documentation for the `refit`
        parameter for more details) and that `best_estimator_` exposes
        `n_features_in_` when fit.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Only defined if
        `best_estimator_` is defined (see the documentation for the `refit`
        parameter for more details) and that `best_estimator_` exposes
        `feature_names_in_` when fit.

        .. versionadded:: 1.0

    See Also
    --------
    :class:`HalvingGridSearchCV`:
        Search over a grid of parameters using successive halving.

    Notes
    -----
    The parameters selected are those that maximize the score of the held-out
    data, according to the scoring parameter.

    Examples
    --------

    >>> from sklearn.datasets import load_iris
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.experimental import enable_halving_search_cv  # noqa
    >>> from sklearn.model_selection import HalvingRandomSearchCV
    >>> from scipy.stats import randint
    >>> import numpy as np
    ...
    >>> X, y = load_iris(return_X_y=True)
    >>> clf = RandomForestClassifier(random_state=0)
    >>> np.random.seed(0)
    ...
    >>> param_distributions = {"max_depth": [3, None],
    ...                        "min_samples_split": randint(2, 11)}
    >>> search = HalvingRandomSearchCV(clf, param_distributions,
    ...                                resource='n_estimators',
    ...                                max_resources=10,
    ...                                random_state=0).fit(X, y)
    >>> search.best_params_  # doctest: +SKIP
    {'max_depth': None, 'min_samples_split': 10, 'n_estimators': 9}
    """

    _required_parameters = ["estimator", "param_distributions"]

    def __init__(
        self,
        estimator,
        param_distributions,
        *,
        n_candidates="exhaust",
        factor=3,
        resource="n_samples",
        max_resources="auto",
        min_resources="smallest",
        aggressive_elimination=False,
        cv=5,
        scoring=None,
        refit=True,
        error_score=np.nan,
        return_train_score=True,
        random_state=None,
        n_jobs=None,
        verbose=0,
    ):
        super().__init__(
            estimator,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=refit,
            verbose=verbose,
            cv=cv,
            random_state=random_state,
            error_score=error_score,
            return_train_score=return_train_score,
            max_resources=max_resources,
            resource=resource,
            factor=factor,
            min_resources=min_resources,
            aggressive_elimination=aggressive_elimination,
        )
        self.param_distributions = param_distributions
        self.n_candidates = n_candidates

    def _generate_candidate_params(self):
        n_candidates_first_iter = self.n_candidates
        if n_candidates_first_iter == "exhaust":
            # This will generate enough candidate so that the last iteration
            # uses as much resources as possible
            n_candidates_first_iter = self.max_resources_ // self.min_resources_
        return ParameterSampler(
            self.param_distributions,
            n_candidates_first_iter,
            random_state=self.random_state,
        )
