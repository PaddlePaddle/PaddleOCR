import numpy as np
from numpy.testing import assert_array_equal
from numpy.testing import assert_allclose

import pytest

from sklearn.base import clone
from sklearn.datasets import make_classification, make_regression

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import check_scoring


X_classification, y_classification = make_classification(random_state=0)
X_regression, y_regression = make_regression(random_state=0)


def _assert_predictor_equal(gb_1, gb_2, X):
    """Assert that two HistGBM instances are identical."""
    # Check identical nodes for each tree
    for (pred_ith_1, pred_ith_2) in zip(gb_1._predictors, gb_2._predictors):
        for (predictor_1, predictor_2) in zip(pred_ith_1, pred_ith_2):
            assert_array_equal(predictor_1.nodes, predictor_2.nodes)

    # Check identical predictions
    assert_allclose(gb_1.predict(X), gb_2.predict(X))


@pytest.mark.parametrize(
    "GradientBoosting, X, y",
    [
        (HistGradientBoostingClassifier, X_classification, y_classification),
        (HistGradientBoostingRegressor, X_regression, y_regression),
    ],
)
def test_max_iter_with_warm_start_validation(GradientBoosting, X, y):
    # Check that a ValueError is raised when the maximum number of iterations
    # is smaller than the number of iterations from the previous fit when warm
    # start is True.

    estimator = GradientBoosting(max_iter=10, early_stopping=False, warm_start=True)
    estimator.fit(X, y)
    estimator.set_params(max_iter=5)
    err_msg = (
        "max_iter=5 must be larger than or equal to n_iter_=10 when warm_start==True"
    )
    with pytest.raises(ValueError, match=err_msg):
        estimator.fit(X, y)


@pytest.mark.parametrize(
    "GradientBoosting, X, y",
    [
        (HistGradientBoostingClassifier, X_classification, y_classification),
        (HistGradientBoostingRegressor, X_regression, y_regression),
    ],
)
def test_warm_start_yields_identical_results(GradientBoosting, X, y):
    # Make sure that fitting 50 iterations and then 25 with warm start is
    # equivalent to fitting 75 iterations.

    rng = 42
    gb_warm_start = GradientBoosting(
        n_iter_no_change=100, max_iter=50, random_state=rng, warm_start=True
    )
    gb_warm_start.fit(X, y).set_params(max_iter=75).fit(X, y)

    gb_no_warm_start = GradientBoosting(
        n_iter_no_change=100, max_iter=75, random_state=rng, warm_start=False
    )
    gb_no_warm_start.fit(X, y)

    # Check that both predictors are equal
    _assert_predictor_equal(gb_warm_start, gb_no_warm_start, X)


@pytest.mark.parametrize(
    "GradientBoosting, X, y",
    [
        (HistGradientBoostingClassifier, X_classification, y_classification),
        (HistGradientBoostingRegressor, X_regression, y_regression),
    ],
)
def test_warm_start_max_depth(GradientBoosting, X, y):
    # Test if possible to fit trees of different depth in ensemble.
    gb = GradientBoosting(
        max_iter=20,
        min_samples_leaf=1,
        warm_start=True,
        max_depth=2,
        early_stopping=False,
    )
    gb.fit(X, y)
    gb.set_params(max_iter=30, max_depth=3, n_iter_no_change=110)
    gb.fit(X, y)

    # First 20 trees have max_depth == 2
    for i in range(20):
        assert gb._predictors[i][0].get_max_depth() == 2
    # Last 10 trees have max_depth == 3
    for i in range(1, 11):
        assert gb._predictors[-i][0].get_max_depth() == 3


@pytest.mark.parametrize(
    "GradientBoosting, X, y",
    [
        (HistGradientBoostingClassifier, X_classification, y_classification),
        (HistGradientBoostingRegressor, X_regression, y_regression),
    ],
)
@pytest.mark.parametrize("scoring", (None, "loss"))
def test_warm_start_early_stopping(GradientBoosting, X, y, scoring):
    # Make sure that early stopping occurs after a small number of iterations
    # when fitting a second time with warm starting.

    n_iter_no_change = 5
    gb = GradientBoosting(
        n_iter_no_change=n_iter_no_change,
        max_iter=10000,
        early_stopping=True,
        random_state=42,
        warm_start=True,
        tol=1e-3,
        scoring=scoring,
    )
    gb.fit(X, y)
    n_iter_first_fit = gb.n_iter_
    gb.fit(X, y)
    n_iter_second_fit = gb.n_iter_
    assert 0 < n_iter_second_fit - n_iter_first_fit < n_iter_no_change


@pytest.mark.parametrize(
    "GradientBoosting, X, y",
    [
        (HistGradientBoostingClassifier, X_classification, y_classification),
        (HistGradientBoostingRegressor, X_regression, y_regression),
    ],
)
def test_warm_start_equal_n_estimators(GradientBoosting, X, y):
    # Test if warm start with equal n_estimators does nothing
    gb_1 = GradientBoosting(max_depth=2, early_stopping=False)
    gb_1.fit(X, y)

    gb_2 = clone(gb_1)
    gb_2.set_params(max_iter=gb_1.max_iter, warm_start=True, n_iter_no_change=5)
    gb_2.fit(X, y)

    # Check that both predictors are equal
    _assert_predictor_equal(gb_1, gb_2, X)


@pytest.mark.parametrize(
    "GradientBoosting, X, y",
    [
        (HistGradientBoostingClassifier, X_classification, y_classification),
        (HistGradientBoostingRegressor, X_regression, y_regression),
    ],
)
def test_warm_start_clear(GradientBoosting, X, y):
    # Test if fit clears state.
    gb_1 = GradientBoosting(n_iter_no_change=5, random_state=42)
    gb_1.fit(X, y)

    gb_2 = GradientBoosting(n_iter_no_change=5, random_state=42, warm_start=True)
    gb_2.fit(X, y)  # inits state
    gb_2.set_params(warm_start=False)
    gb_2.fit(X, y)  # clears old state and equals est

    # Check that both predictors have the same train_score_ and
    # validation_score_ attributes
    assert_allclose(gb_1.train_score_, gb_2.train_score_)
    assert_allclose(gb_1.validation_score_, gb_2.validation_score_)

    # Check that both predictors are equal
    _assert_predictor_equal(gb_1, gb_2, X)


@pytest.mark.parametrize(
    "GradientBoosting, X, y",
    [
        (HistGradientBoostingClassifier, X_classification, y_classification),
        (HistGradientBoostingRegressor, X_regression, y_regression),
    ],
)
@pytest.mark.parametrize("rng_type", ("none", "int", "instance"))
def test_random_seeds_warm_start(GradientBoosting, X, y, rng_type):
    # Make sure the seeds for train/val split and small trainset subsampling
    # are correctly set in a warm start context.
    def _get_rng(rng_type):
        # Helper to avoid consuming rngs
        if rng_type == "none":
            return None
        elif rng_type == "int":
            return 42
        else:
            return np.random.RandomState(0)

    random_state = _get_rng(rng_type)
    gb_1 = GradientBoosting(early_stopping=True, max_iter=2, random_state=random_state)
    gb_1.set_params(scoring=check_scoring(gb_1))
    gb_1.fit(X, y)
    random_seed_1_1 = gb_1._random_seed

    gb_1.fit(X, y)
    random_seed_1_2 = gb_1._random_seed  # clear the old state, different seed

    random_state = _get_rng(rng_type)
    gb_2 = GradientBoosting(
        early_stopping=True, max_iter=2, random_state=random_state, warm_start=True
    )
    gb_2.set_params(scoring=check_scoring(gb_2))
    gb_2.fit(X, y)  # inits state
    random_seed_2_1 = gb_2._random_seed
    gb_2.fit(X, y)  # clears old state and equals est
    random_seed_2_2 = gb_2._random_seed

    # Without warm starting, the seeds should be
    # * all different if random state is None
    # * all equal if random state is an integer
    # * different when refitting and equal with a new estimator (because
    #   the random state is mutated)
    if rng_type == "none":
        assert random_seed_1_1 != random_seed_1_2 != random_seed_2_1
    elif rng_type == "int":
        assert random_seed_1_1 == random_seed_1_2 == random_seed_2_1
    else:
        assert random_seed_1_1 == random_seed_2_1 != random_seed_1_2

    # With warm starting, the seeds must be equal
    assert random_seed_2_1 == random_seed_2_2
