from math import ceil

import pytest
from scipy.stats import norm, randint
import numpy as np

from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import LeavePGroupsOut
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.svm import LinearSVC
from sklearn.model_selection._search_successive_halving import (
    _SubsampleMetaSplitter,
    _top_k,
)


class FastClassifier(DummyClassifier):
    """Dummy classifier that accepts parameters a, b, ... z.

    These parameter don't affect the predictions and are useful for fast
    grid searching."""

    def __init__(
        self, strategy="stratified", random_state=None, constant=None, **kwargs
    ):
        super().__init__(
            strategy=strategy, random_state=random_state, constant=constant
        )

    def get_params(self, deep=False):
        params = super().get_params(deep=deep)
        for char in range(ord("a"), ord("z") + 1):
            params[chr(char)] = "whatever"
        return params


@pytest.mark.parametrize("Est", (HalvingGridSearchCV, HalvingRandomSearchCV))
@pytest.mark.parametrize(
    "aggressive_elimination,"
    "max_resources,"
    "expected_n_iterations,"
    "expected_n_required_iterations,"
    "expected_n_possible_iterations,"
    "expected_n_remaining_candidates,"
    "expected_n_candidates,"
    "expected_n_resources,",
    [
        # notice how it loops at the beginning
        # also, the number of candidates evaluated at the last iteration is
        # <= factor
        (True, "limited", 4, 4, 3, 1, [60, 20, 7, 3], [20, 20, 60, 180]),
        # no aggressive elimination: we end up with less iterations, and
        # the number of candidates at the last iter is > factor, which isn't
        # ideal
        (False, "limited", 3, 4, 3, 3, [60, 20, 7], [20, 60, 180]),
        #  # When the amount of resource isn't limited, aggressive_elimination
        #  # has no effect. Here the default min_resources='exhaust' will take
        #  # over.
        (True, "unlimited", 4, 4, 4, 1, [60, 20, 7, 3], [37, 111, 333, 999]),
        (False, "unlimited", 4, 4, 4, 1, [60, 20, 7, 3], [37, 111, 333, 999]),
    ],
)
def test_aggressive_elimination(
    Est,
    aggressive_elimination,
    max_resources,
    expected_n_iterations,
    expected_n_required_iterations,
    expected_n_possible_iterations,
    expected_n_remaining_candidates,
    expected_n_candidates,
    expected_n_resources,
):
    # Test the aggressive_elimination parameter.

    n_samples = 1000
    X, y = make_classification(n_samples=n_samples, random_state=0)
    param_grid = {"a": ("l1", "l2"), "b": list(range(30))}
    base_estimator = FastClassifier()

    if max_resources == "limited":
        max_resources = 180
    else:
        max_resources = n_samples

    sh = Est(
        base_estimator,
        param_grid,
        aggressive_elimination=aggressive_elimination,
        max_resources=max_resources,
        factor=3,
    )
    sh.set_params(verbose=True)  # just for test coverage

    if Est is HalvingRandomSearchCV:
        # same number of candidates as with the grid
        sh.set_params(n_candidates=2 * 30, min_resources="exhaust")

    sh.fit(X, y)

    assert sh.n_iterations_ == expected_n_iterations
    assert sh.n_required_iterations_ == expected_n_required_iterations
    assert sh.n_possible_iterations_ == expected_n_possible_iterations
    assert sh.n_resources_ == expected_n_resources
    assert sh.n_candidates_ == expected_n_candidates
    assert sh.n_remaining_candidates_ == expected_n_remaining_candidates
    assert ceil(sh.n_candidates_[-1] / sh.factor) == sh.n_remaining_candidates_


@pytest.mark.parametrize("Est", (HalvingGridSearchCV, HalvingRandomSearchCV))
@pytest.mark.parametrize(
    "min_resources,"
    "max_resources,"
    "expected_n_iterations,"
    "expected_n_possible_iterations,"
    "expected_n_resources,",
    [
        # with enough resources
        ("smallest", "auto", 2, 4, [20, 60]),
        # with enough resources but min_resources set manually
        (50, "auto", 2, 3, [50, 150]),
        # without enough resources, only one iteration can be done
        ("smallest", 30, 1, 1, [20]),
        # with exhaust: use as much resources as possible at the last iter
        ("exhaust", "auto", 2, 2, [333, 999]),
        ("exhaust", 1000, 2, 2, [333, 999]),
        ("exhaust", 999, 2, 2, [333, 999]),
        ("exhaust", 600, 2, 2, [200, 600]),
        ("exhaust", 599, 2, 2, [199, 597]),
        ("exhaust", 300, 2, 2, [100, 300]),
        ("exhaust", 60, 2, 2, [20, 60]),
        ("exhaust", 50, 1, 1, [20]),
        ("exhaust", 20, 1, 1, [20]),
    ],
)
def test_min_max_resources(
    Est,
    min_resources,
    max_resources,
    expected_n_iterations,
    expected_n_possible_iterations,
    expected_n_resources,
):
    # Test the min_resources and max_resources parameters, and how they affect
    # the number of resources used at each iteration
    n_samples = 1000
    X, y = make_classification(n_samples=n_samples, random_state=0)
    param_grid = {"a": [1, 2], "b": [1, 2, 3]}
    base_estimator = FastClassifier()

    sh = Est(
        base_estimator,
        param_grid,
        factor=3,
        min_resources=min_resources,
        max_resources=max_resources,
    )
    if Est is HalvingRandomSearchCV:
        sh.set_params(n_candidates=6)  # same number as with the grid

    sh.fit(X, y)

    expected_n_required_iterations = 2  # given 6 combinations and factor = 3
    assert sh.n_iterations_ == expected_n_iterations
    assert sh.n_required_iterations_ == expected_n_required_iterations
    assert sh.n_possible_iterations_ == expected_n_possible_iterations
    assert sh.n_resources_ == expected_n_resources
    if min_resources == "exhaust":
        assert sh.n_possible_iterations_ == sh.n_iterations_ == len(sh.n_resources_)


@pytest.mark.parametrize("Est", (HalvingRandomSearchCV, HalvingGridSearchCV))
@pytest.mark.parametrize(
    "max_resources, n_iterations, n_possible_iterations",
    [
        ("auto", 5, 9),  # all resources are used
        (1024, 5, 9),
        (700, 5, 8),
        (512, 5, 8),
        (511, 5, 7),
        (32, 4, 4),
        (31, 3, 3),
        (16, 3, 3),
        (4, 1, 1),  # max_resources == min_resources, only one iteration is
        # possible
    ],
)
def test_n_iterations(Est, max_resources, n_iterations, n_possible_iterations):
    # test the number of actual iterations that were run depending on
    # max_resources

    n_samples = 1024
    X, y = make_classification(n_samples=n_samples, random_state=1)
    param_grid = {"a": [1, 2], "b": list(range(10))}
    base_estimator = FastClassifier()
    factor = 2

    sh = Est(
        base_estimator,
        param_grid,
        cv=2,
        factor=factor,
        max_resources=max_resources,
        min_resources=4,
    )
    if Est is HalvingRandomSearchCV:
        sh.set_params(n_candidates=20)  # same as for HalvingGridSearchCV
    sh.fit(X, y)
    assert sh.n_required_iterations_ == 5
    assert sh.n_iterations_ == n_iterations
    assert sh.n_possible_iterations_ == n_possible_iterations


@pytest.mark.parametrize("Est", (HalvingRandomSearchCV, HalvingGridSearchCV))
def test_resource_parameter(Est):
    # Test the resource parameter

    n_samples = 1000
    X, y = make_classification(n_samples=n_samples, random_state=0)
    param_grid = {"a": [1, 2], "b": list(range(10))}
    base_estimator = FastClassifier()
    sh = Est(base_estimator, param_grid, cv=2, resource="c", max_resources=10, factor=3)
    sh.fit(X, y)
    assert set(sh.n_resources_) == set([1, 3, 9])
    for r_i, params, param_c in zip(
        sh.cv_results_["n_resources"],
        sh.cv_results_["params"],
        sh.cv_results_["param_c"],
    ):
        assert r_i == params["c"] == param_c

    with pytest.raises(
        ValueError, match="Cannot use resource=1234 which is not supported "
    ):
        sh = HalvingGridSearchCV(
            base_estimator, param_grid, cv=2, resource="1234", max_resources=10
        )
        sh.fit(X, y)

    with pytest.raises(
        ValueError,
        match=(
            "Cannot use parameter c as the resource since it is part "
            "of the searched parameters."
        ),
    ):
        param_grid = {"a": [1, 2], "b": [1, 2], "c": [1, 3]}
        sh = HalvingGridSearchCV(
            base_estimator, param_grid, cv=2, resource="c", max_resources=10
        )
        sh.fit(X, y)


@pytest.mark.parametrize(
    "max_resources, n_candidates, expected_n_candidates",
    [
        (512, "exhaust", 128),  # generate exactly as much as needed
        (32, "exhaust", 8),
        (32, 8, 8),
        (32, 7, 7),  # ask for less than what we could
        (32, 9, 9),  # ask for more than 'reasonable'
    ],
)
def test_random_search(max_resources, n_candidates, expected_n_candidates):
    # Test random search and make sure the number of generated candidates is
    # as expected

    n_samples = 1024
    X, y = make_classification(n_samples=n_samples, random_state=0)
    param_grid = {"a": norm, "b": norm}
    base_estimator = FastClassifier()
    sh = HalvingRandomSearchCV(
        base_estimator,
        param_grid,
        n_candidates=n_candidates,
        cv=2,
        max_resources=max_resources,
        factor=2,
        min_resources=4,
    )
    sh.fit(X, y)
    assert sh.n_candidates_[0] == expected_n_candidates
    if n_candidates == "exhaust":
        # Make sure 'exhaust' makes the last iteration use as much resources as
        # we can
        assert sh.n_resources_[-1] == max_resources


@pytest.mark.parametrize(
    "param_distributions, expected_n_candidates",
    [
        ({"a": [1, 2]}, 2),  # all lists, sample less than n_candidates
        ({"a": randint(1, 3)}, 10),  # not all list, respect n_candidates
    ],
)
def test_random_search_discrete_distributions(
    param_distributions, expected_n_candidates
):
    # Make sure random search samples the appropriate number of candidates when
    # we ask for more than what's possible. How many parameters are sampled
    # depends whether the distributions are 'all lists' or not (see
    # ParameterSampler for details). This is somewhat redundant with the checks
    # in ParameterSampler but interaction bugs were discovered during
    # development of SH

    n_samples = 1024
    X, y = make_classification(n_samples=n_samples, random_state=0)
    base_estimator = FastClassifier()
    sh = HalvingRandomSearchCV(base_estimator, param_distributions, n_candidates=10)
    sh.fit(X, y)
    assert sh.n_candidates_[0] == expected_n_candidates


@pytest.mark.parametrize("Est", (HalvingGridSearchCV, HalvingRandomSearchCV))
@pytest.mark.parametrize(
    "params, expected_error_message",
    [
        ({"scoring": {"accuracy", "accuracy"}}, "Multimetric scoring is not supported"),
        (
            {"resource": "not_a_parameter"},
            "Cannot use resource=not_a_parameter which is not supported",
        ),
        (
            {"resource": "a", "max_resources": 100},
            "Cannot use parameter a as the resource since it is part of",
        ),
        ({"max_resources": "not_auto"}, "max_resources must be either"),
        ({"max_resources": 100.5}, "max_resources must be either"),
        ({"max_resources": -10}, "max_resources must be either"),
        ({"min_resources": "bad str"}, "min_resources must be either"),
        ({"min_resources": 0.5}, "min_resources must be either"),
        ({"min_resources": -10}, "min_resources must be either"),
        (
            {"max_resources": "auto", "resource": "b"},
            "max_resources can only be 'auto' if resource='n_samples'",
        ),
        (
            {"min_resources": 15, "max_resources": 14},
            "min_resources_=15 is greater than max_resources_=14",
        ),
        ({"cv": KFold(shuffle=True)}, "must yield consistent folds"),
        ({"cv": ShuffleSplit()}, "must yield consistent folds"),
        ({"refit": "whatever"}, "refit is expected to be a boolean"),
    ],
)
def test_input_errors(Est, params, expected_error_message):
    base_estimator = FastClassifier()
    param_grid = {"a": [1]}
    X, y = make_classification(100)

    sh = Est(base_estimator, param_grid, **params)

    with pytest.raises(ValueError, match=expected_error_message):
        sh.fit(X, y)


@pytest.mark.parametrize(
    "params, expected_error_message",
    [
        (
            {"n_candidates": "exhaust", "min_resources": "exhaust"},
            "cannot be both set to 'exhaust'",
        ),
        ({"n_candidates": "bad"}, "either 'exhaust' or a positive integer"),
        ({"n_candidates": 0}, "either 'exhaust' or a positive integer"),
    ],
)
def test_input_errors_randomized(params, expected_error_message):
    # tests specific to HalvingRandomSearchCV

    base_estimator = FastClassifier()
    param_grid = {"a": [1]}
    X, y = make_classification(100)

    sh = HalvingRandomSearchCV(base_estimator, param_grid, **params)

    with pytest.raises(ValueError, match=expected_error_message):
        sh.fit(X, y)


@pytest.mark.parametrize(
    "fraction, subsample_test, expected_train_size, expected_test_size",
    [
        (0.5, True, 40, 10),
        (0.5, False, 40, 20),
        (0.2, True, 16, 4),
        (0.2, False, 16, 20),
    ],
)
def test_subsample_splitter_shapes(
    fraction, subsample_test, expected_train_size, expected_test_size
):
    # Make sure splits returned by SubsampleMetaSplitter are of appropriate
    # size

    n_samples = 100
    X, y = make_classification(n_samples)
    cv = _SubsampleMetaSplitter(
        base_cv=KFold(5),
        fraction=fraction,
        subsample_test=subsample_test,
        random_state=None,
    )

    for train, test in cv.split(X, y):
        assert train.shape[0] == expected_train_size
        assert test.shape[0] == expected_test_size
        if subsample_test:
            assert train.shape[0] + test.shape[0] == int(n_samples * fraction)
        else:
            assert test.shape[0] == n_samples // cv.base_cv.get_n_splits()


@pytest.mark.parametrize("subsample_test", (True, False))
def test_subsample_splitter_determinism(subsample_test):
    # Make sure _SubsampleMetaSplitter is consistent across calls to split():
    # - we're OK having training sets differ (they're always sampled with a
    #   different fraction anyway)
    # - when we don't subsample the test set, we want it to be always the same.
    #   This check is the most important. This is ensured by the determinism
    #   of the base_cv.

    # Note: we could force both train and test splits to be always the same if
    # we drew an int seed in _SubsampleMetaSplitter.__init__

    n_samples = 100
    X, y = make_classification(n_samples)
    cv = _SubsampleMetaSplitter(
        base_cv=KFold(5), fraction=0.5, subsample_test=subsample_test, random_state=None
    )

    folds_a = list(cv.split(X, y, groups=None))
    folds_b = list(cv.split(X, y, groups=None))

    for (train_a, test_a), (train_b, test_b) in zip(folds_a, folds_b):
        assert not np.all(train_a == train_b)

        if subsample_test:
            assert not np.all(test_a == test_b)
        else:
            assert np.all(test_a == test_b)
            assert np.all(X[test_a] == X[test_b])


@pytest.mark.parametrize(
    "k, itr, expected",
    [
        (1, 0, ["c"]),
        (2, 0, ["a", "c"]),
        (4, 0, ["d", "b", "a", "c"]),
        (10, 0, ["d", "b", "a", "c"]),
        (1, 1, ["e"]),
        (2, 1, ["f", "e"]),
        (10, 1, ["f", "e"]),
        (1, 2, ["i"]),
        (10, 2, ["g", "h", "i"]),
    ],
)
def test_top_k(k, itr, expected):

    results = {  # this isn't a 'real world' result dict
        "iter": [0, 0, 0, 0, 1, 1, 2, 2, 2],
        "mean_test_score": [4, 3, 5, 1, 11, 10, 5, 6, 9],
        "params": ["a", "b", "c", "d", "e", "f", "g", "h", "i"],
    }
    got = _top_k(results, k=k, itr=itr)
    assert np.all(got == expected)


@pytest.mark.parametrize("Est", (HalvingRandomSearchCV, HalvingGridSearchCV))
def test_cv_results(Est):
    # test that the cv_results_ matches correctly the logic of the
    # tournament: in particular that the candidates continued in each
    # successive iteration are those that were best in the previous iteration
    pd = pytest.importorskip("pandas")

    rng = np.random.RandomState(0)

    n_samples = 1000
    X, y = make_classification(n_samples=n_samples, random_state=0)
    param_grid = {"a": ("l1", "l2"), "b": list(range(30))}
    base_estimator = FastClassifier()

    # generate random scores: we want to avoid ties, which would otherwise
    # mess with the ordering and make testing harder
    def scorer(est, X, y):
        return rng.rand()

    sh = Est(base_estimator, param_grid, factor=2, scoring=scorer)
    if Est is HalvingRandomSearchCV:
        # same number of candidates as with the grid
        sh.set_params(n_candidates=2 * 30, min_resources="exhaust")

    sh.fit(X, y)

    # non-regression check for
    # https://github.com/scikit-learn/scikit-learn/issues/19203
    assert isinstance(sh.cv_results_["iter"], np.ndarray)
    assert isinstance(sh.cv_results_["n_resources"], np.ndarray)

    cv_results_df = pd.DataFrame(sh.cv_results_)

    # just make sure we don't have ties
    assert len(cv_results_df["mean_test_score"].unique()) == len(cv_results_df)

    cv_results_df["params_str"] = cv_results_df["params"].apply(str)
    table = cv_results_df.pivot(
        index="params_str", columns="iter", values="mean_test_score"
    )

    # table looks like something like this:
    # iter                    0      1       2        3   4   5
    # params_str
    # {'a': 'l2', 'b': 23} 0.75    NaN     NaN      NaN NaN NaN
    # {'a': 'l1', 'b': 30} 0.90  0.875     NaN      NaN NaN NaN
    # {'a': 'l1', 'b': 0}  0.75    NaN     NaN      NaN NaN NaN
    # {'a': 'l2', 'b': 3}  0.85  0.925  0.9125  0.90625 NaN NaN
    # {'a': 'l1', 'b': 5}  0.80    NaN     NaN      NaN NaN NaN
    # ...

    # where a NaN indicates that the candidate wasn't evaluated at a given
    # iteration, because it wasn't part of the top-K at some previous
    # iteration. We here make sure that candidates that aren't in the top-k at
    # any given iteration are indeed not evaluated at the subsequent
    # iterations.
    nan_mask = pd.isna(table)
    n_iter = sh.n_iterations_
    for it in range(n_iter - 1):
        already_discarded_mask = nan_mask[it]

        # make sure that if a candidate is already discarded, we don't evaluate
        # it later
        assert (
            already_discarded_mask & nan_mask[it + 1] == already_discarded_mask
        ).all()

        # make sure that the number of discarded candidate is correct
        discarded_now_mask = ~already_discarded_mask & nan_mask[it + 1]
        kept_mask = ~already_discarded_mask & ~discarded_now_mask
        assert kept_mask.sum() == sh.n_candidates_[it + 1]

        # make sure that all discarded candidates have a lower score than the
        # kept candidates
        discarded_max_score = table[it].where(discarded_now_mask).max()
        kept_min_score = table[it].where(kept_mask).min()
        assert discarded_max_score < kept_min_score

    # We now make sure that the best candidate is chosen only from the last
    # iteration.
    # We also make sure this is true even if there were higher scores in
    # earlier rounds (this isn't generally the case, but worth ensuring it's
    # possible).

    last_iter = cv_results_df["iter"].max()
    idx_best_last_iter = cv_results_df[cv_results_df["iter"] == last_iter][
        "mean_test_score"
    ].idxmax()
    idx_best_all_iters = cv_results_df["mean_test_score"].idxmax()

    assert sh.best_params_ == cv_results_df.iloc[idx_best_last_iter]["params"]
    assert (
        cv_results_df.iloc[idx_best_last_iter]["mean_test_score"]
        < cv_results_df.iloc[idx_best_all_iters]["mean_test_score"]
    )
    assert (
        cv_results_df.iloc[idx_best_last_iter]["params"]
        != cv_results_df.iloc[idx_best_all_iters]["params"]
    )


@pytest.mark.parametrize("Est", (HalvingGridSearchCV, HalvingRandomSearchCV))
def test_base_estimator_inputs(Est):
    # make sure that the base estimators are passed the correct parameters and
    # number of samples at each iteration.
    pd = pytest.importorskip("pandas")

    passed_n_samples_fit = []
    passed_n_samples_predict = []
    passed_params = []

    class FastClassifierBookKeeping(FastClassifier):
        def fit(self, X, y):
            passed_n_samples_fit.append(X.shape[0])
            return super().fit(X, y)

        def predict(self, X):
            passed_n_samples_predict.append(X.shape[0])
            return super().predict(X)

        def set_params(self, **params):
            passed_params.append(params)
            return super().set_params(**params)

    n_samples = 1024
    n_splits = 2
    X, y = make_classification(n_samples=n_samples, random_state=0)
    param_grid = {"a": ("l1", "l2"), "b": list(range(30))}
    base_estimator = FastClassifierBookKeeping()

    sh = Est(
        base_estimator,
        param_grid,
        factor=2,
        cv=n_splits,
        return_train_score=False,
        refit=False,
    )
    if Est is HalvingRandomSearchCV:
        # same number of candidates as with the grid
        sh.set_params(n_candidates=2 * 30, min_resources="exhaust")

    sh.fit(X, y)

    assert len(passed_n_samples_fit) == len(passed_n_samples_predict)
    passed_n_samples = [
        x + y for (x, y) in zip(passed_n_samples_fit, passed_n_samples_predict)
    ]

    # Lists are of length n_splits * n_iter * n_candidates_at_i.
    # Each chunk of size n_splits corresponds to the n_splits folds for the
    # same candidate at the same iteration, so they contain equal values. We
    # subsample such that the lists are of length n_iter * n_candidates_at_it
    passed_n_samples = passed_n_samples[::n_splits]
    passed_params = passed_params[::n_splits]

    cv_results_df = pd.DataFrame(sh.cv_results_)

    assert len(passed_params) == len(passed_n_samples) == len(cv_results_df)

    uniques, counts = np.unique(passed_n_samples, return_counts=True)
    assert (sh.n_resources_ == uniques).all()
    assert (sh.n_candidates_ == counts).all()

    assert (cv_results_df["params"] == passed_params).all()
    assert (cv_results_df["n_resources"] == passed_n_samples).all()


@pytest.mark.parametrize("Est", (HalvingGridSearchCV, HalvingRandomSearchCV))
def test_groups_support(Est):
    # Check if ValueError (when groups is None) propagates to
    # HalvingGridSearchCV and HalvingRandomSearchCV
    # And also check if groups is correctly passed to the cv object
    rng = np.random.RandomState(0)

    X, y = make_classification(n_samples=50, n_classes=2, random_state=0)
    groups = rng.randint(0, 3, 50)

    clf = LinearSVC(random_state=0)
    grid = {"C": [1]}

    group_cvs = [
        LeaveOneGroupOut(),
        LeavePGroupsOut(2),
        GroupKFold(n_splits=3),
        GroupShuffleSplit(random_state=0),
    ]
    error_msg = "The 'groups' parameter should not be None."
    for cv in group_cvs:
        gs = Est(clf, grid, cv=cv, random_state=0)
        with pytest.raises(ValueError, match=error_msg):
            gs.fit(X, y)
        gs.fit(X, y, groups=groups)

    non_group_cvs = [StratifiedKFold(), StratifiedShuffleSplit(random_state=0)]
    for cv in non_group_cvs:
        gs = Est(clf, grid, cv=cv)
        # Should not raise an error
        gs.fit(X, y)


@pytest.mark.parametrize("SearchCV", [HalvingRandomSearchCV, HalvingGridSearchCV])
def test_min_resources_null(SearchCV):
    """Check that we raise an error if the minimum resources is set to 0."""
    base_estimator = FastClassifier()
    param_grid = {"a": [1]}
    X = np.empty(0).reshape(0, 3)

    search = SearchCV(base_estimator, param_grid, min_resources="smallest")

    err_msg = "min_resources_=0: you might have passed an empty dataset X."
    with pytest.raises(ValueError, match=err_msg):
        search.fit(X, [])


@pytest.mark.parametrize("SearchCV", [HalvingGridSearchCV, HalvingRandomSearchCV])
def test_select_best_index(SearchCV):
    """Check the selection strategy of the halving search."""
    results = {  # this isn't a 'real world' result dict
        "iter": np.array([0, 0, 0, 0, 1, 1, 2, 2, 2]),
        "mean_test_score": np.array([4, 3, 5, 1, 11, 10, 5, 6, 9]),
        "params": np.array(["a", "b", "c", "d", "e", "f", "g", "h", "i"]),
    }

    # we expect the index of 'i'
    best_index = SearchCV._select_best_index(None, None, results)
    assert best_index == 8
