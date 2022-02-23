"""Permutation importance for estimators."""
import numbers
import numpy as np
from joblib import Parallel

from ..ensemble._bagging import _generate_indices
from ..metrics import check_scoring
from ..metrics._scorer import _check_multimetric_scoring, _MultimetricScorer
from ..model_selection._validation import _aggregate_score_dicts
from ..utils import Bunch, _safe_indexing
from ..utils import check_random_state
from ..utils import check_array
from ..utils.fixes import delayed


def _weights_scorer(scorer, estimator, X, y, sample_weight):
    if sample_weight is not None:
        return scorer(estimator, X, y, sample_weight)
    return scorer(estimator, X, y)


def _calculate_permutation_scores(
    estimator,
    X,
    y,
    sample_weight,
    col_idx,
    random_state,
    n_repeats,
    scorer,
    max_samples,
):
    """Calculate score when `col_idx` is permuted."""
    random_state = check_random_state(random_state)

    # Work on a copy of X to to ensure thread-safety in case of threading based
    # parallelism. Furthermore, making a copy is also useful when the joblib
    # backend is 'loky' (default) or the old 'multiprocessing': in those cases,
    # if X is large it will be automatically be backed by a readonly memory map
    # (memmap). X.copy() on the other hand is always guaranteed to return a
    # writable data-structure whose columns can be shuffled inplace.
    if max_samples < X.shape[0]:
        row_indices = _generate_indices(
            random_state=random_state,
            bootstrap=False,
            n_population=X.shape[0],
            n_samples=max_samples,
        )
        X_permuted = _safe_indexing(X, row_indices, axis=0)
        y = _safe_indexing(y, row_indices, axis=0)
    else:
        X_permuted = X.copy()

    scores = []
    shuffling_idx = np.arange(X_permuted.shape[0])
    for _ in range(n_repeats):
        random_state.shuffle(shuffling_idx)
        if hasattr(X_permuted, "iloc"):
            col = X_permuted.iloc[shuffling_idx, col_idx]
            col.index = X_permuted.index
            X_permuted.iloc[:, col_idx] = col
        else:
            X_permuted[:, col_idx] = X_permuted[shuffling_idx, col_idx]
        scores.append(_weights_scorer(scorer, estimator, X_permuted, y, sample_weight))

    if isinstance(scores[0], dict):
        scores = _aggregate_score_dicts(scores)
    else:
        scores = np.array(scores)

    return scores


def _create_importances_bunch(baseline_score, permuted_score):
    """Compute the importances as the decrease in score.

    Parameters
    ----------
    baseline_score : ndarray of shape (n_features,)
        The baseline score without permutation.
    permuted_score : ndarray of shape (n_features, n_repeats)
        The permuted scores for the `n` repetitions.

    Returns
    -------
    importances : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.
        importances_mean : ndarray, shape (n_features, )
            Mean of feature importance over `n_repeats`.
        importances_std : ndarray, shape (n_features, )
            Standard deviation over `n_repeats`.
        importances : ndarray, shape (n_features, n_repeats)
            Raw permutation importance scores.
    """
    importances = baseline_score - permuted_score
    return Bunch(
        importances_mean=np.mean(importances, axis=1),
        importances_std=np.std(importances, axis=1),
        importances=importances,
    )


def permutation_importance(
    estimator,
    X,
    y,
    *,
    scoring=None,
    n_repeats=5,
    n_jobs=None,
    random_state=None,
    sample_weight=None,
    max_samples=1.0,
):
    """Permutation importance for feature evaluation [BRE]_.

    The :term:`estimator` is required to be a fitted estimator. `X` can be the
    data set used to train the estimator or a hold-out set. The permutation
    importance of a feature is calculated as follows. First, a baseline metric,
    defined by :term:`scoring`, is evaluated on a (potentially different)
    dataset defined by the `X`. Next, a feature column from the validation set
    is permuted and the metric is evaluated again. The permutation importance
    is defined to be the difference between the baseline metric and metric from
    permutating the feature column.

    Read more in the :ref:`User Guide <permutation_importance>`.

    Parameters
    ----------
    estimator : object
        An estimator that has already been :term:`fitted` and is compatible
        with :term:`scorer`.

    X : ndarray or DataFrame, shape (n_samples, n_features)
        Data on which permutation importance will be computed.

    y : array-like or None, shape (n_samples, ) or (n_samples, n_classes)
        Targets for supervised or `None` for unsupervised.

    scoring : str, callable, list, tuple, or dict, default=None
        Scorer to use.
        If `scoring` represents a single score, one can use:

        - a single string (see :ref:`scoring_parameter`);
        - a callable (see :ref:`scoring`) that returns a single value.

        If `scoring` represents multiple scores, one can use:

        - a list or tuple of unique strings;
        - a callable returning a dictionary where the keys are the metric
          names and the values are the metric scores;
        - a dictionary with metric names as keys and callables a values.

        Passing multiple scores to `scoring` is more efficient than calling
        `permutation_importance` for each of the scores as it reuses
        predictions to avoid redundant computation.

        If None, the estimator's default scorer is used.

    n_repeats : int, default=5
        Number of times to permute a feature.

    n_jobs : int or None, default=None
        Number of jobs to run in parallel. The computation is done by computing
        permutation score for each columns and parallelized over the columns.
        `None` means 1 unless in a :obj:`joblib.parallel_backend` context.
        `-1` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    random_state : int, RandomState instance, default=None
        Pseudo-random number generator to control the permutations of each
        feature.
        Pass an int to get reproducible results across function calls.
        See :term:`Glossary <random_state>`.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights used in scoring.

        .. versionadded:: 0.24

    max_samples : int or float, default=1.0
        The number of samples to draw from X to compute feature importance
        in each repeat (without replacement).

        - If int, then draw `max_samples` samples.
        - If float, then draw `max_samples * X.shape[0]` samples.
        - If `max_samples` is equal to `1.0` or `X.shape[0]`, all samples
          will be used.

        While using this option may provide less accurate importance estimates,
        it keeps the method tractable when evaluating feature importance on
        large datasets. In combination with `n_repeats`, this allows to control
        the computational speed vs statistical accuracy trade-off of this method.

        .. versionadded:: 1.0

    Returns
    -------
    result : :class:`~sklearn.utils.Bunch` or dict of such instances
        Dictionary-like object, with the following attributes.

        importances_mean : ndarray of shape (n_features, )
            Mean of feature importance over `n_repeats`.
        importances_std : ndarray of shape (n_features, )
            Standard deviation over `n_repeats`.
        importances : ndarray of shape (n_features, n_repeats)
            Raw permutation importance scores.

        If there are multiple scoring metrics in the scoring parameter
        `result` is a dict with scorer names as keys (e.g. 'roc_auc') and
        `Bunch` objects like above as values.

    References
    ----------
    .. [BRE] L. Breiman, "Random Forests", Machine Learning, 45(1), 5-32,
             2001. https://doi.org/10.1023/A:1010933404324

    Examples
    --------
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.inspection import permutation_importance
    >>> X = [[1, 9, 9],[1, 9, 9],[1, 9, 9],
    ...      [0, 9, 9],[0, 9, 9],[0, 9, 9]]
    >>> y = [1, 1, 1, 0, 0, 0]
    >>> clf = LogisticRegression().fit(X, y)
    >>> result = permutation_importance(clf, X, y, n_repeats=10,
    ...                                 random_state=0)
    >>> result.importances_mean
    array([0.4666..., 0.       , 0.       ])
    >>> result.importances_std
    array([0.2211..., 0.       , 0.       ])
    """
    if not hasattr(X, "iloc"):
        X = check_array(X, force_all_finite="allow-nan", dtype=None)

    # Precompute random seed from the random state to be used
    # to get a fresh independent RandomState instance for each
    # parallel call to _calculate_permutation_scores, irrespective of
    # the fact that variables are shared or not depending on the active
    # joblib backend (sequential, thread-based or process-based).
    random_state = check_random_state(random_state)
    random_seed = random_state.randint(np.iinfo(np.int32).max + 1)

    if not isinstance(max_samples, numbers.Integral):
        max_samples = int(max_samples * X.shape[0])
    elif not (0 < max_samples <= X.shape[0]):
        raise ValueError("max_samples must be in (0, n_samples]")

    if callable(scoring):
        scorer = scoring
    elif scoring is None or isinstance(scoring, str):
        scorer = check_scoring(estimator, scoring=scoring)
    else:
        scorers_dict = _check_multimetric_scoring(estimator, scoring)
        scorer = _MultimetricScorer(**scorers_dict)

    baseline_score = _weights_scorer(scorer, estimator, X, y, sample_weight)

    scores = Parallel(n_jobs=n_jobs)(
        delayed(_calculate_permutation_scores)(
            estimator,
            X,
            y,
            sample_weight,
            col_idx,
            random_seed,
            n_repeats,
            scorer,
            max_samples,
        )
        for col_idx in range(X.shape[1])
    )

    if isinstance(baseline_score, dict):
        return {
            name: _create_importances_bunch(
                baseline_score[name],
                # unpack the permuted scores
                np.array([scores[col_idx][name] for col_idx in range(X.shape[1])]),
            )
            for name in baseline_score
        }
    else:
        return _create_importances_bunch(baseline_score, np.array(scores))
