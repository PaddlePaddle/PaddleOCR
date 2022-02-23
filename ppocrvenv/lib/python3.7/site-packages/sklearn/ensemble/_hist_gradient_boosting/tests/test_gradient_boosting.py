import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.datasets import make_classification, make_regression
from sklearn.datasets import make_low_rank_matrix
from sklearn.preprocessing import KBinsDiscretizer, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.base import clone, BaseEstimator, TransformerMixin
from sklearn.base import is_regressor
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_poisson_deviance
from sklearn.dummy import DummyRegressor
from sklearn.exceptions import NotFittedError
from sklearn.compose import make_column_transformer

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble._hist_gradient_boosting.loss import _LOSSES
from sklearn.ensemble._hist_gradient_boosting.loss import LeastSquares
from sklearn.ensemble._hist_gradient_boosting.loss import BinaryCrossEntropy
from sklearn.ensemble._hist_gradient_boosting.grower import TreeGrower
from sklearn.ensemble._hist_gradient_boosting.binning import _BinMapper
from sklearn.utils import shuffle
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads

n_threads = _openmp_effective_n_threads()


X_classification, y_classification = make_classification(random_state=0)
X_regression, y_regression = make_regression(random_state=0)
X_multi_classification, y_multi_classification = make_classification(
    n_classes=3, n_informative=3, random_state=0
)


def _make_dumb_dataset(n_samples):
    """Make a dumb dataset to test early stopping."""
    rng = np.random.RandomState(42)
    X_dumb = rng.randn(n_samples, 1)
    y_dumb = (X_dumb[:, 0] > 0).astype("int64")
    return X_dumb, y_dumb


@pytest.mark.parametrize(
    "GradientBoosting, X, y",
    [
        (HistGradientBoostingClassifier, X_classification, y_classification),
        (HistGradientBoostingRegressor, X_regression, y_regression),
    ],
)
@pytest.mark.parametrize(
    "params, err_msg",
    [
        ({"loss": "blah"}, "Loss blah is not supported for"),
        ({"learning_rate": 0}, "learning_rate=0 must be strictly positive"),
        ({"learning_rate": -1}, "learning_rate=-1 must be strictly positive"),
        ({"max_iter": 0}, "max_iter=0 must not be smaller than 1"),
        ({"max_leaf_nodes": 0}, "max_leaf_nodes=0 should not be smaller than 2"),
        ({"max_leaf_nodes": 1}, "max_leaf_nodes=1 should not be smaller than 2"),
        ({"max_depth": 0}, "max_depth=0 should not be smaller than 1"),
        ({"min_samples_leaf": 0}, "min_samples_leaf=0 should not be smaller"),
        ({"l2_regularization": -1}, "l2_regularization=-1 must be positive"),
        ({"max_bins": 1}, "max_bins=1 should be no smaller than 2 and no larger"),
        ({"max_bins": 256}, "max_bins=256 should be no smaller than 2 and no"),
        ({"n_iter_no_change": -1}, "n_iter_no_change=-1 must be positive"),
        ({"validation_fraction": -1}, "validation_fraction=-1 must be strictly"),
        ({"validation_fraction": 0}, "validation_fraction=0 must be strictly"),
        ({"tol": -1}, "tol=-1 must not be smaller than 0"),
    ],
)
def test_init_parameters_validation(GradientBoosting, X, y, params, err_msg):

    with pytest.raises(ValueError, match=err_msg):
        GradientBoosting(**params).fit(X, y)


def test_invalid_classification_loss():
    binary_clf = HistGradientBoostingClassifier(loss="binary_crossentropy")
    err_msg = (
        "loss='binary_crossentropy' is not defined for multiclass "
        "classification with n_classes=3, use "
        "loss='categorical_crossentropy' instead"
    )
    with pytest.raises(ValueError, match=err_msg):
        binary_clf.fit(np.zeros(shape=(3, 2)), np.arange(3))


@pytest.mark.parametrize(
    "scoring, validation_fraction, early_stopping, n_iter_no_change, tol",
    [
        ("neg_mean_squared_error", 0.1, True, 5, 1e-7),  # use scorer
        ("neg_mean_squared_error", None, True, 5, 1e-1),  # use scorer on train
        (None, 0.1, True, 5, 1e-7),  # same with default scorer
        (None, None, True, 5, 1e-1),
        ("loss", 0.1, True, 5, 1e-7),  # use loss
        ("loss", None, True, 5, 1e-1),  # use loss on training data
        (None, None, False, 5, 0.0),  # no early stopping
    ],
)
def test_early_stopping_regression(
    scoring, validation_fraction, early_stopping, n_iter_no_change, tol
):

    max_iter = 200

    X, y = make_regression(n_samples=50, random_state=0)

    gb = HistGradientBoostingRegressor(
        verbose=1,  # just for coverage
        min_samples_leaf=5,  # easier to overfit fast
        scoring=scoring,
        tol=tol,
        early_stopping=early_stopping,
        validation_fraction=validation_fraction,
        max_iter=max_iter,
        n_iter_no_change=n_iter_no_change,
        random_state=0,
    )
    gb.fit(X, y)

    if early_stopping:
        assert n_iter_no_change <= gb.n_iter_ < max_iter
    else:
        assert gb.n_iter_ == max_iter


@pytest.mark.parametrize(
    "data",
    (
        make_classification(n_samples=30, random_state=0),
        make_classification(
            n_samples=30, n_classes=3, n_clusters_per_class=1, random_state=0
        ),
    ),
)
@pytest.mark.parametrize(
    "scoring, validation_fraction, early_stopping, n_iter_no_change, tol",
    [
        ("accuracy", 0.1, True, 5, 1e-7),  # use scorer
        ("accuracy", None, True, 5, 1e-1),  # use scorer on training data
        (None, 0.1, True, 5, 1e-7),  # same with default scorer
        (None, None, True, 5, 1e-1),
        ("loss", 0.1, True, 5, 1e-7),  # use loss
        ("loss", None, True, 5, 1e-1),  # use loss on training data
        (None, None, False, 5, 0.0),  # no early stopping
    ],
)
def test_early_stopping_classification(
    data, scoring, validation_fraction, early_stopping, n_iter_no_change, tol
):

    max_iter = 50

    X, y = data

    gb = HistGradientBoostingClassifier(
        verbose=1,  # just for coverage
        min_samples_leaf=5,  # easier to overfit fast
        scoring=scoring,
        tol=tol,
        early_stopping=early_stopping,
        validation_fraction=validation_fraction,
        max_iter=max_iter,
        n_iter_no_change=n_iter_no_change,
        random_state=0,
    )
    gb.fit(X, y)

    if early_stopping is True:
        assert n_iter_no_change <= gb.n_iter_ < max_iter
    else:
        assert gb.n_iter_ == max_iter


@pytest.mark.parametrize(
    "GradientBoosting, X, y",
    [
        (HistGradientBoostingClassifier, *_make_dumb_dataset(10000)),
        (HistGradientBoostingClassifier, *_make_dumb_dataset(10001)),
        (HistGradientBoostingRegressor, *_make_dumb_dataset(10000)),
        (HistGradientBoostingRegressor, *_make_dumb_dataset(10001)),
    ],
)
def test_early_stopping_default(GradientBoosting, X, y):
    # Test that early stopping is enabled by default if and only if there
    # are more than 10000 samples
    gb = GradientBoosting(max_iter=10, n_iter_no_change=2, tol=1e-1)
    gb.fit(X, y)
    if X.shape[0] > 10000:
        assert gb.n_iter_ < gb.max_iter
    else:
        assert gb.n_iter_ == gb.max_iter


@pytest.mark.parametrize(
    "scores, n_iter_no_change, tol, stopping",
    [
        ([], 1, 0.001, False),  # not enough iterations
        ([1, 1, 1], 5, 0.001, False),  # not enough iterations
        ([1, 1, 1, 1, 1], 5, 0.001, False),  # not enough iterations
        ([1, 2, 3, 4, 5, 6], 5, 0.001, False),  # significant improvement
        ([1, 2, 3, 4, 5, 6], 5, 0.0, False),  # significant improvement
        ([1, 2, 3, 4, 5, 6], 5, 0.999, False),  # significant improvement
        ([1, 2, 3, 4, 5, 6], 5, 5 - 1e-5, False),  # significant improvement
        ([1] * 6, 5, 0.0, True),  # no significant improvement
        ([1] * 6, 5, 0.001, True),  # no significant improvement
        ([1] * 6, 5, 5, True),  # no significant improvement
    ],
)
def test_should_stop(scores, n_iter_no_change, tol, stopping):

    gbdt = HistGradientBoostingClassifier(n_iter_no_change=n_iter_no_change, tol=tol)
    assert gbdt._should_stop(scores) == stopping


def test_absolute_error():
    # For coverage only.
    X, y = make_regression(n_samples=500, random_state=0)
    gbdt = HistGradientBoostingRegressor(loss="absolute_error", random_state=0)
    gbdt.fit(X, y)
    assert gbdt.score(X, y) > 0.9


def test_absolute_error_sample_weight():
    # non regression test for issue #19400
    # make sure no error is thrown during fit of
    # HistGradientBoostingRegressor with absolute_error loss function
    # and passing sample_weight
    rng = np.random.RandomState(0)
    n_samples = 100
    X = rng.uniform(-1, 1, size=(n_samples, 2))
    y = rng.uniform(-1, 1, size=n_samples)
    sample_weight = rng.uniform(0, 1, size=n_samples)
    gbdt = HistGradientBoostingRegressor(loss="absolute_error")
    gbdt.fit(X, y, sample_weight=sample_weight)


@pytest.mark.parametrize("y", [([1.0, -2.0, 0.0]), ([0.0, 0.0, 0.0])])
def test_poisson_y_positive(y):
    # Test that ValueError is raised if either one y_i < 0 or sum(y_i) <= 0.
    err_msg = r"loss='poisson' requires non-negative y and sum\(y\) > 0."
    gbdt = HistGradientBoostingRegressor(loss="poisson", random_state=0)
    with pytest.raises(ValueError, match=err_msg):
        gbdt.fit(np.zeros(shape=(len(y), 1)), y)


def test_poisson():
    # For Poisson distributed target, Poisson loss should give better results
    # than least squares measured in Poisson deviance as metric.
    rng = np.random.RandomState(42)
    n_train, n_test, n_features = 500, 100, 100
    X = make_low_rank_matrix(
        n_samples=n_train + n_test, n_features=n_features, random_state=rng
    )
    # We create a log-linear Poisson model and downscale coef as it will get
    # exponentiated.
    coef = rng.uniform(low=-2, high=2, size=n_features) / np.max(X, axis=0)
    y = rng.poisson(lam=np.exp(X @ coef))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=n_test, random_state=rng
    )
    gbdt_pois = HistGradientBoostingRegressor(loss="poisson", random_state=rng)
    gbdt_ls = HistGradientBoostingRegressor(loss="squared_error", random_state=rng)
    gbdt_pois.fit(X_train, y_train)
    gbdt_ls.fit(X_train, y_train)
    dummy = DummyRegressor(strategy="mean").fit(X_train, y_train)

    for X, y in [(X_train, y_train), (X_test, y_test)]:
        metric_pois = mean_poisson_deviance(y, gbdt_pois.predict(X))
        # squared_error might produce non-positive predictions => clip
        metric_ls = mean_poisson_deviance(y, np.clip(gbdt_ls.predict(X), 1e-15, None))
        metric_dummy = mean_poisson_deviance(y, dummy.predict(X))
        assert metric_pois < metric_ls
        assert metric_pois < metric_dummy


def test_binning_train_validation_are_separated():
    # Make sure training and validation data are binned separately.
    # See issue 13926

    rng = np.random.RandomState(0)
    validation_fraction = 0.2
    gb = HistGradientBoostingClassifier(
        early_stopping=True, validation_fraction=validation_fraction, random_state=rng
    )
    gb.fit(X_classification, y_classification)
    mapper_training_data = gb._bin_mapper

    # Note that since the data is small there is no subsampling and the
    # random_state doesn't matter
    mapper_whole_data = _BinMapper(random_state=0)
    mapper_whole_data.fit(X_classification)

    n_samples = X_classification.shape[0]
    assert np.all(
        mapper_training_data.n_bins_non_missing_
        == int((1 - validation_fraction) * n_samples)
    )
    assert np.all(
        mapper_training_data.n_bins_non_missing_
        != mapper_whole_data.n_bins_non_missing_
    )


def test_missing_values_trivial():
    # sanity check for missing values support. With only one feature and
    # y == isnan(X), the gbdt is supposed to reach perfect accuracy on the
    # training set.

    n_samples = 100
    n_features = 1
    rng = np.random.RandomState(0)

    X = rng.normal(size=(n_samples, n_features))
    mask = rng.binomial(1, 0.5, size=X.shape).astype(bool)
    X[mask] = np.nan
    y = mask.ravel()
    gb = HistGradientBoostingClassifier()
    gb.fit(X, y)

    assert gb.score(X, y) == pytest.approx(1)


@pytest.mark.parametrize("problem", ("classification", "regression"))
@pytest.mark.parametrize(
    "missing_proportion, expected_min_score_classification, "
    "expected_min_score_regression",
    [(0.1, 0.97, 0.89), (0.2, 0.93, 0.81), (0.5, 0.79, 0.52)],
)
def test_missing_values_resilience(
    problem,
    missing_proportion,
    expected_min_score_classification,
    expected_min_score_regression,
):
    # Make sure the estimators can deal with missing values and still yield
    # decent predictions

    rng = np.random.RandomState(0)
    n_samples = 1000
    n_features = 2
    if problem == "regression":
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_features,
            random_state=rng,
        )
        gb = HistGradientBoostingRegressor()
        expected_min_score = expected_min_score_regression
    else:
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_features,
            n_redundant=0,
            n_repeated=0,
            random_state=rng,
        )
        gb = HistGradientBoostingClassifier()
        expected_min_score = expected_min_score_classification

    mask = rng.binomial(1, missing_proportion, size=X.shape).astype(bool)
    X[mask] = np.nan

    gb.fit(X, y)

    assert gb.score(X, y) > expected_min_score


@pytest.mark.parametrize(
    "data",
    [
        make_classification(random_state=0, n_classes=2),
        make_classification(random_state=0, n_classes=3, n_informative=3),
    ],
    ids=["binary_crossentropy", "categorical_crossentropy"],
)
def test_zero_division_hessians(data):
    # non regression test for issue #14018
    # make sure we avoid zero division errors when computing the leaves values.

    # If the learning rate is too high, the raw predictions are bad and will
    # saturate the softmax (or sigmoid in binary classif). This leads to
    # probabilities being exactly 0 or 1, gradients being constant, and
    # hessians being zero.
    X, y = data
    gb = HistGradientBoostingClassifier(learning_rate=100, max_iter=10)
    gb.fit(X, y)


def test_small_trainset():
    # Make sure that the small trainset is stratified and has the expected
    # length (10k samples)
    n_samples = 20000
    original_distrib = {0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4}
    rng = np.random.RandomState(42)
    X = rng.randn(n_samples).reshape(n_samples, 1)
    y = [
        [class_] * int(prop * n_samples) for (class_, prop) in original_distrib.items()
    ]
    y = shuffle(np.concatenate(y))
    gb = HistGradientBoostingClassifier()

    # Compute the small training set
    X_small, y_small, _ = gb._get_small_trainset(
        X, y, seed=42, sample_weight_train=None
    )

    # Compute the class distribution in the small training set
    unique, counts = np.unique(y_small, return_counts=True)
    small_distrib = {class_: count / 10000 for (class_, count) in zip(unique, counts)}

    # Test that the small training set has the expected length
    assert X_small.shape[0] == 10000
    assert y_small.shape[0] == 10000

    # Test that the class distributions in the whole dataset and in the small
    # training set are identical
    assert small_distrib == pytest.approx(original_distrib)


def test_missing_values_minmax_imputation():
    # Compare the buit-in missing value handling of Histogram GBC with an
    # a-priori missing value imputation strategy that should yield the same
    # results in terms of decision function.
    #
    # Each feature (containing NaNs) is replaced by 2 features:
    # - one where the nans are replaced by min(feature) - 1
    # - one where the nans are replaced by max(feature) + 1
    # A split where nans go to the left has an equivalent split in the
    # first (min) feature, and a split where nans go to the right has an
    # equivalent split in the second (max) feature.
    #
    # Assuming the data is such that there is never a tie to select the best
    # feature to split on during training, the learned decision trees should be
    # strictly equivalent (learn a sequence of splits that encode the same
    # decision function).
    #
    # The MinMaxImputer transformer is meant to be a toy implementation of the
    # "Missing In Attributes" (MIA) missing value handling for decision trees
    # https://www.sciencedirect.com/science/article/abs/pii/S0167865508000305
    # The implementation of MIA as an imputation transformer was suggested by
    # "Remark 3" in https://arxiv.org/abs/1902.06931

    class MinMaxImputer(TransformerMixin, BaseEstimator):
        def fit(self, X, y=None):
            mm = MinMaxScaler().fit(X)
            self.data_min_ = mm.data_min_
            self.data_max_ = mm.data_max_
            return self

        def transform(self, X):
            X_min, X_max = X.copy(), X.copy()

            for feature_idx in range(X.shape[1]):
                nan_mask = np.isnan(X[:, feature_idx])
                X_min[nan_mask, feature_idx] = self.data_min_[feature_idx] - 1
                X_max[nan_mask, feature_idx] = self.data_max_[feature_idx] + 1

            return np.concatenate([X_min, X_max], axis=1)

    def make_missing_value_data(n_samples=int(1e4), seed=0):
        rng = np.random.RandomState(seed)
        X, y = make_regression(n_samples=n_samples, n_features=4, random_state=rng)

        # Pre-bin the data to ensure a deterministic handling by the 2
        # strategies and also make it easier to insert np.nan in a structured
        # way:
        X = KBinsDiscretizer(n_bins=42, encode="ordinal").fit_transform(X)

        # First feature has missing values completely at random:
        rnd_mask = rng.rand(X.shape[0]) > 0.9
        X[rnd_mask, 0] = np.nan

        # Second and third features have missing values for extreme values
        # (censoring missingness):
        low_mask = X[:, 1] == 0
        X[low_mask, 1] = np.nan

        high_mask = X[:, 2] == X[:, 2].max()
        X[high_mask, 2] = np.nan

        # Make the last feature nan pattern very informative:
        y_max = np.percentile(y, 70)
        y_max_mask = y >= y_max
        y[y_max_mask] = y_max
        X[y_max_mask, 3] = np.nan

        # Check that there is at least one missing value in each feature:
        for feature_idx in range(X.shape[1]):
            assert any(np.isnan(X[:, feature_idx]))

        # Let's use a test set to check that the learned decision function is
        # the same as evaluated on unseen data. Otherwise it could just be the
        # case that we find two independent ways to overfit the training set.
        return train_test_split(X, y, random_state=rng)

    # n_samples need to be large enough to minimize the likelihood of having
    # several candidate splits with the same gain value in a given tree.
    X_train, X_test, y_train, y_test = make_missing_value_data(
        n_samples=int(1e4), seed=0
    )

    # Use a small number of leaf nodes and iterations so as to keep
    # under-fitting models to minimize the likelihood of ties when training the
    # model.
    gbm1 = HistGradientBoostingRegressor(max_iter=100, max_leaf_nodes=5, random_state=0)
    gbm1.fit(X_train, y_train)

    gbm2 = make_pipeline(MinMaxImputer(), clone(gbm1))
    gbm2.fit(X_train, y_train)

    # Check that the model reach the same score:
    assert gbm1.score(X_train, y_train) == pytest.approx(gbm2.score(X_train, y_train))

    assert gbm1.score(X_test, y_test) == pytest.approx(gbm2.score(X_test, y_test))

    # Check the individual prediction match as a finer grained
    # decision function check.
    assert_allclose(gbm1.predict(X_train), gbm2.predict(X_train))
    assert_allclose(gbm1.predict(X_test), gbm2.predict(X_test))


def test_infinite_values():
    # Basic test for infinite values

    X = np.array([-np.inf, 0, 1, np.inf]).reshape(-1, 1)
    y = np.array([0, 0, 1, 1])

    gbdt = HistGradientBoostingRegressor(min_samples_leaf=1)
    gbdt.fit(X, y)
    np.testing.assert_allclose(gbdt.predict(X), y, atol=1e-4)


def test_consistent_lengths():
    X = np.array([-np.inf, 0, 1, np.inf]).reshape(-1, 1)
    y = np.array([0, 0, 1, 1])
    sample_weight = np.array([0.1, 0.3, 0.1])
    gbdt = HistGradientBoostingRegressor()
    with pytest.raises(ValueError, match=r"sample_weight.shape == \(3,\), expected"):
        gbdt.fit(X, y, sample_weight)

    with pytest.raises(
        ValueError, match="Found input variables with inconsistent number"
    ):
        gbdt.fit(X, y[1:])


def test_infinite_values_missing_values():
    # High level test making sure that inf and nan values are properly handled
    # when both are present. This is similar to
    # test_split_on_nan_with_infinite_values() in test_grower.py, though we
    # cannot check the predictions for binned values here.

    X = np.asarray([-np.inf, 0, 1, np.inf, np.nan]).reshape(-1, 1)
    y_isnan = np.isnan(X.ravel())
    y_isinf = X.ravel() == np.inf

    stump_clf = HistGradientBoostingClassifier(
        min_samples_leaf=1, max_iter=1, learning_rate=1, max_depth=2
    )

    assert stump_clf.fit(X, y_isinf).score(X, y_isinf) == 1
    assert stump_clf.fit(X, y_isnan).score(X, y_isnan) == 1


def test_crossentropy_binary_problem():
    # categorical_crossentropy should only be used if there are more than two
    # classes present. PR #14869
    X = [[1], [0]]
    y = [0, 1]
    gbrt = HistGradientBoostingClassifier(loss="categorical_crossentropy")
    with pytest.raises(
        ValueError, match="'categorical_crossentropy' is not suitable for"
    ):
        gbrt.fit(X, y)


@pytest.mark.parametrize("scoring", [None, "loss"])
def test_string_target_early_stopping(scoring):
    # Regression tests for #14709 where the targets need to be encoded before
    # to compute the score
    rng = np.random.RandomState(42)
    X = rng.randn(100, 10)
    y = np.array(["x"] * 50 + ["y"] * 50, dtype=object)
    gbrt = HistGradientBoostingClassifier(n_iter_no_change=10, scoring=scoring)
    gbrt.fit(X, y)


def test_zero_sample_weights_regression():
    # Make sure setting a SW to zero amounts to ignoring the corresponding
    # sample

    X = [[1, 0], [1, 0], [1, 0], [0, 1]]
    y = [0, 0, 1, 0]
    # ignore the first 2 training samples by setting their weight to 0
    sample_weight = [0, 0, 1, 1]
    gb = HistGradientBoostingRegressor(min_samples_leaf=1)
    gb.fit(X, y, sample_weight=sample_weight)
    assert gb.predict([[1, 0]])[0] > 0.5


def test_zero_sample_weights_classification():
    # Make sure setting a SW to zero amounts to ignoring the corresponding
    # sample

    X = [[1, 0], [1, 0], [1, 0], [0, 1]]
    y = [0, 0, 1, 0]
    # ignore the first 2 training samples by setting their weight to 0
    sample_weight = [0, 0, 1, 1]
    gb = HistGradientBoostingClassifier(loss="binary_crossentropy", min_samples_leaf=1)
    gb.fit(X, y, sample_weight=sample_weight)
    assert_array_equal(gb.predict([[1, 0]]), [1])

    X = [[1, 0], [1, 0], [1, 0], [0, 1], [1, 1]]
    y = [0, 0, 1, 0, 2]
    # ignore the first 2 training samples by setting their weight to 0
    sample_weight = [0, 0, 1, 1, 1]
    gb = HistGradientBoostingClassifier(
        loss="categorical_crossentropy", min_samples_leaf=1
    )
    gb.fit(X, y, sample_weight=sample_weight)
    assert_array_equal(gb.predict([[1, 0]]), [1])


@pytest.mark.parametrize(
    "problem", ("regression", "binary_classification", "multiclass_classification")
)
@pytest.mark.parametrize("duplication", ("half", "all"))
def test_sample_weight_effect(problem, duplication):
    # High level test to make sure that duplicating a sample is equivalent to
    # giving it weight of 2.

    # fails for n_samples > 255 because binning does not take sample weights
    # into account. Keeping n_samples <= 255 makes
    # sure only unique values are used so SW have no effect on binning.
    n_samples = 255
    n_features = 2
    if problem == "regression":
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_features,
            random_state=0,
        )
        Klass = HistGradientBoostingRegressor
    else:
        n_classes = 2 if problem == "binary_classification" else 3
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_features,
            n_redundant=0,
            n_clusters_per_class=1,
            n_classes=n_classes,
            random_state=0,
        )
        Klass = HistGradientBoostingClassifier

    # This test can't pass if min_samples_leaf > 1 because that would force 2
    # samples to be in the same node in est_sw, while these samples would be
    # free to be separate in est_dup: est_dup would just group together the
    # duplicated samples.
    est = Klass(min_samples_leaf=1)

    # Create dataset with duplicate and corresponding sample weights
    if duplication == "half":
        lim = n_samples // 2
    else:
        lim = n_samples
    X_dup = np.r_[X, X[:lim]]
    y_dup = np.r_[y, y[:lim]]
    sample_weight = np.ones(shape=(n_samples))
    sample_weight[:lim] = 2

    est_sw = clone(est).fit(X, y, sample_weight=sample_weight)
    est_dup = clone(est).fit(X_dup, y_dup)

    # checking raw_predict is stricter than just predict for classification
    assert np.allclose(est_sw._raw_predict(X_dup), est_dup._raw_predict(X_dup))


@pytest.mark.parametrize("loss_name", ("squared_error", "absolute_error"))
def test_sum_hessians_are_sample_weight(loss_name):
    # For losses with constant hessians, the sum_hessians field of the
    # histograms must be equal to the sum of the sample weight of samples at
    # the corresponding bin.

    rng = np.random.RandomState(0)
    n_samples = 1000
    n_features = 2
    X, y = make_regression(n_samples=n_samples, n_features=n_features, random_state=rng)
    bin_mapper = _BinMapper()
    X_binned = bin_mapper.fit_transform(X)

    sample_weight = rng.normal(size=n_samples)

    loss = _LOSSES[loss_name](sample_weight=sample_weight, n_threads=n_threads)
    gradients, hessians = loss.init_gradients_and_hessians(
        n_samples=n_samples, prediction_dim=1, sample_weight=sample_weight
    )
    raw_predictions = rng.normal(size=(1, n_samples))
    loss.update_gradients_and_hessians(
        gradients, hessians, y, raw_predictions, sample_weight
    )

    # build sum_sample_weight which contains the sum of the sample weights at
    # each bin (for each feature). This must be equal to the sum_hessians
    # field of the corresponding histogram
    sum_sw = np.zeros(shape=(n_features, bin_mapper.n_bins))
    for feature_idx in range(n_features):
        for sample_idx in range(n_samples):
            sum_sw[feature_idx, X_binned[sample_idx, feature_idx]] += sample_weight[
                sample_idx
            ]

    # Build histogram
    grower = TreeGrower(X_binned, gradients[0], hessians[0], n_bins=bin_mapper.n_bins)
    histograms = grower.histogram_builder.compute_histograms_brute(
        grower.root.sample_indices
    )

    for feature_idx in range(n_features):
        for bin_idx in range(bin_mapper.n_bins):
            assert histograms[feature_idx, bin_idx]["sum_hessians"] == (
                pytest.approx(sum_sw[feature_idx, bin_idx], rel=1e-5)
            )


def test_max_depth_max_leaf_nodes():
    # Non regression test for
    # https://github.com/scikit-learn/scikit-learn/issues/16179
    # there was a bug when the max_depth and the max_leaf_nodes criteria were
    # met at the same time, which would lead to max_leaf_nodes not being
    # respected.
    X, y = make_classification(random_state=0)
    est = HistGradientBoostingClassifier(max_depth=2, max_leaf_nodes=3, max_iter=1).fit(
        X, y
    )
    tree = est._predictors[0][0]
    assert tree.get_max_depth() == 2
    assert tree.get_n_leaf_nodes() == 3  # would be 4 prior to bug fix


def test_early_stopping_on_test_set_with_warm_start():
    # Non regression test for #16661 where second fit fails with
    # warm_start=True, early_stopping is on, and no validation set
    X, y = make_classification(random_state=0)
    gb = HistGradientBoostingClassifier(
        max_iter=1,
        scoring="loss",
        warm_start=True,
        early_stopping=True,
        n_iter_no_change=1,
        validation_fraction=None,
    )

    gb.fit(X, y)
    # does not raise on second call
    gb.set_params(max_iter=2)
    gb.fit(X, y)


@pytest.mark.parametrize(
    "Est", (HistGradientBoostingClassifier, HistGradientBoostingRegressor)
)
def test_single_node_trees(Est):
    # Make sure it's still possible to build single-node trees. In that case
    # the value of the root is set to 0. That's a correct value: if the tree is
    # single-node that's because min_gain_to_split is not respected right from
    # the root, so we don't want the tree to have any impact on the
    # predictions.

    X, y = make_classification(random_state=0)
    y[:] = 1  # constant target will lead to a single root node

    est = Est(max_iter=20)
    est.fit(X, y)

    assert all(len(predictor[0].nodes) == 1 for predictor in est._predictors)
    assert all(predictor[0].nodes[0]["value"] == 0 for predictor in est._predictors)
    # Still gives correct predictions thanks to the baseline prediction
    assert_allclose(est.predict(X), y)


@pytest.mark.parametrize(
    "Est, loss, X, y",
    [
        (
            HistGradientBoostingClassifier,
            BinaryCrossEntropy(sample_weight=None),
            X_classification,
            y_classification,
        ),
        (
            HistGradientBoostingRegressor,
            LeastSquares(sample_weight=None),
            X_regression,
            y_regression,
        ),
    ],
)
def test_custom_loss(Est, loss, X, y):
    est = Est(loss=loss, max_iter=20)
    est.fit(X, y)


@pytest.mark.parametrize(
    "HistGradientBoosting, X, y",
    [
        (HistGradientBoostingClassifier, X_classification, y_classification),
        (HistGradientBoostingRegressor, X_regression, y_regression),
        (
            HistGradientBoostingClassifier,
            X_multi_classification,
            y_multi_classification,
        ),
    ],
)
def test_staged_predict(HistGradientBoosting, X, y):

    # Test whether staged predictor eventually gives
    # the same prediction.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=0
    )
    gb = HistGradientBoosting(max_iter=10)

    # test raise NotFittedError if not fitted
    with pytest.raises(NotFittedError):
        next(gb.staged_predict(X_test))

    gb.fit(X_train, y_train)

    # test if the staged predictions of each iteration
    # are equal to the corresponding predictions of the same estimator
    # trained from scratch.
    # this also test limit case when max_iter = 1
    method_names = (
        ["predict"]
        if is_regressor(gb)
        else ["predict", "predict_proba", "decision_function"]
    )
    for method_name in method_names:

        staged_method = getattr(gb, "staged_" + method_name)
        staged_predictions = list(staged_method(X_test))
        assert len(staged_predictions) == gb.n_iter_
        for n_iter, staged_predictions in enumerate(staged_method(X_test), 1):
            aux = HistGradientBoosting(max_iter=n_iter)
            aux.fit(X_train, y_train)
            pred_aux = getattr(aux, method_name)(X_test)

            assert_allclose(staged_predictions, pred_aux)
            assert staged_predictions.shape == pred_aux.shape


@pytest.mark.parametrize("insert_missing", [False, True])
@pytest.mark.parametrize(
    "Est", (HistGradientBoostingRegressor, HistGradientBoostingClassifier)
)
@pytest.mark.parametrize("bool_categorical_parameter", [True, False])
def test_unknown_categories_nan(insert_missing, Est, bool_categorical_parameter):
    # Make sure no error is raised at predict if a category wasn't seen during
    # fit. We also make sure they're treated as nans.

    rng = np.random.RandomState(0)
    n_samples = 1000
    f1 = rng.rand(n_samples)
    f2 = rng.randint(4, size=n_samples)
    X = np.c_[f1, f2]
    y = np.zeros(shape=n_samples)
    y[X[:, 1] % 2 == 0] = 1

    if bool_categorical_parameter:
        categorical_features = [False, True]
    else:
        categorical_features = [1]

    if insert_missing:
        mask = rng.binomial(1, 0.01, size=X.shape).astype(bool)
        assert mask.sum() > 0
        X[mask] = np.nan

    est = Est(max_iter=20, categorical_features=categorical_features).fit(X, y)
    assert_array_equal(est.is_categorical_, [False, True])

    # Make sure no error is raised on unknown categories and nans
    # unknown categories will be treated as nans
    X_test = np.zeros((10, X.shape[1]), dtype=float)
    X_test[:5, 1] = 30
    X_test[5:, 1] = np.nan
    assert len(np.unique(est.predict(X_test))) == 1


def test_categorical_encoding_strategies():
    # Check native categorical handling vs different encoding strategies. We
    # make sure that native encoding needs only 1 split to achieve a perfect
    # prediction on a simple dataset. In contrast, OneHotEncoded data needs
    # more depth / splits, and treating categories as ordered (just using
    # OrdinalEncoder) requires even more depth.

    # dataset with one random continuous feature, and one categorical feature
    # with values in [0, 5], e.g. from an OrdinalEncoder.
    # class == 1 iff categorical value in {0, 2, 4}
    rng = np.random.RandomState(0)
    n_samples = 10_000
    f1 = rng.rand(n_samples)
    f2 = rng.randint(6, size=n_samples)
    X = np.c_[f1, f2]
    y = np.zeros(shape=n_samples)
    y[X[:, 1] % 2 == 0] = 1

    # make sure dataset is balanced so that the baseline_prediction doesn't
    # influence predictions too much with max_iter = 1
    assert 0.49 < y.mean() < 0.51

    clf_cat = HistGradientBoostingClassifier(
        max_iter=1, max_depth=1, categorical_features=[False, True]
    )

    # Using native categorical encoding, we get perfect predictions with just
    # one split
    assert cross_val_score(clf_cat, X, y).mean() == 1

    # quick sanity check for the bitset: 0, 2, 4 = 2**0 + 2**2 + 2**4 = 21
    expected_left_bitset = [21, 0, 0, 0, 0, 0, 0, 0]
    left_bitset = clf_cat.fit(X, y)._predictors[0][0].raw_left_cat_bitsets[0]
    assert_array_equal(left_bitset, expected_left_bitset)

    # Treating categories as ordered, we need more depth / more splits to get
    # the same predictions
    clf_no_cat = HistGradientBoostingClassifier(
        max_iter=1, max_depth=4, categorical_features=None
    )
    assert cross_val_score(clf_no_cat, X, y).mean() < 0.9

    clf_no_cat.set_params(max_depth=5)
    assert cross_val_score(clf_no_cat, X, y).mean() == 1

    # Using OHEd data, we need less splits than with pure OEd data, but we
    # still need more splits than with the native categorical splits
    ct = make_column_transformer(
        (OneHotEncoder(sparse=False), [1]), remainder="passthrough"
    )
    X_ohe = ct.fit_transform(X)
    clf_no_cat.set_params(max_depth=2)
    assert cross_val_score(clf_no_cat, X_ohe, y).mean() < 0.9

    clf_no_cat.set_params(max_depth=3)
    assert cross_val_score(clf_no_cat, X_ohe, y).mean() == 1


@pytest.mark.parametrize(
    "Est", (HistGradientBoostingClassifier, HistGradientBoostingRegressor)
)
@pytest.mark.parametrize(
    "categorical_features, monotonic_cst, expected_msg",
    [
        (
            ["hello", "world"],
            None,
            "categorical_features must be an array-like of bools or array-like of "
            "ints.",
        ),
        (
            [0, -1],
            None,
            (
                r"categorical_features set as integer indices must be in "
                r"\[0, n_features - 1\]"
            ),
        ),
        (
            [True, True, False, False, True],
            None,
            r"categorical_features set as a boolean mask must have shape "
            r"\(n_features,\)",
        ),
        (
            [True, True, False, False],
            [0, -1, 0, 1],
            "Categorical features cannot have monotonic constraints",
        ),
    ],
)
def test_categorical_spec_errors(
    Est, categorical_features, monotonic_cst, expected_msg
):
    # Test errors when categories are specified incorrectly
    n_samples = 100
    X, y = make_classification(random_state=0, n_features=4, n_samples=n_samples)
    rng = np.random.RandomState(0)
    X[:, 0] = rng.randint(0, 10, size=n_samples)
    X[:, 1] = rng.randint(0, 10, size=n_samples)
    est = Est(categorical_features=categorical_features, monotonic_cst=monotonic_cst)

    with pytest.raises(ValueError, match=expected_msg):
        est.fit(X, y)


@pytest.mark.parametrize(
    "Est", (HistGradientBoostingClassifier, HistGradientBoostingRegressor)
)
@pytest.mark.parametrize("categorical_features", ([False, False], []))
@pytest.mark.parametrize("as_array", (True, False))
def test_categorical_spec_no_categories(Est, categorical_features, as_array):
    # Make sure we can properly detect that no categorical features are present
    # even if the categorical_features parameter is not None
    X = np.arange(10).reshape(5, 2)
    y = np.arange(5)
    if as_array:
        categorical_features = np.asarray(categorical_features)
    est = Est(categorical_features=categorical_features).fit(X, y)
    assert est.is_categorical_ is None


@pytest.mark.parametrize(
    "Est", (HistGradientBoostingClassifier, HistGradientBoostingRegressor)
)
def test_categorical_bad_encoding_errors(Est):
    # Test errors when categories are encoded incorrectly

    gb = Est(categorical_features=[True], max_bins=2)

    X = np.array([[0, 1, 2]]).T
    y = np.arange(3)
    msg = "Categorical feature at index 0 is expected to have a cardinality <= 2"
    with pytest.raises(ValueError, match=msg):
        gb.fit(X, y)

    X = np.array([[0, 2]]).T
    y = np.arange(2)
    msg = "Categorical feature at index 0 is expected to be encoded with values < 2"
    with pytest.raises(ValueError, match=msg):
        gb.fit(X, y)

    # nans are ignored in the counts
    X = np.array([[0, 1, np.nan]]).T
    y = np.arange(3)
    gb.fit(X, y)


@pytest.mark.parametrize(
    "Est", (HistGradientBoostingClassifier, HistGradientBoostingRegressor)
)
def test_uint8_predict(Est):
    # Non regression test for
    # https://github.com/scikit-learn/scikit-learn/issues/18408
    # Make sure X can be of dtype uint8 (i.e. X_BINNED_DTYPE) in predict. It
    # will be converted to X_DTYPE.

    rng = np.random.RandomState(0)

    X = rng.randint(0, 100, size=(10, 2)).astype(np.uint8)
    y = rng.randint(0, 2, size=10).astype(np.uint8)
    est = Est()
    est.fit(X, y)
    est.predict(X)


# TODO: Remove in v1.2
@pytest.mark.parametrize(
    "old_loss, new_loss",
    [
        ("least_squares", "squared_error"),
        ("least_absolute_deviation", "absolute_error"),
    ],
)
def test_loss_deprecated(old_loss, new_loss):
    X, y = make_regression(n_samples=50, random_state=0)
    est1 = HistGradientBoostingRegressor(loss=old_loss, random_state=0)

    with pytest.warns(FutureWarning, match=f"The loss '{old_loss}' was deprecated"):
        est1.fit(X, y)

    est2 = HistGradientBoostingRegressor(loss=new_loss, random_state=0)
    est2.fit(X, y)
    assert_allclose(est1.predict(X), est2.predict(X))
