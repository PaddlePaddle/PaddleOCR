import pytest
import numpy as np

from numpy.testing import assert_allclose

from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_diabetes
from sklearn.datasets import load_iris
from sklearn.datasets import make_classification
from sklearn.datasets import make_regression
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    get_scorer,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale
from sklearn.utils import parallel_backend
from sklearn.utils._testing import _convert_container


@pytest.mark.parametrize("n_jobs", [1, 2])
@pytest.mark.parametrize("max_samples", [0.5, 1.0])
def test_permutation_importance_correlated_feature_regression(n_jobs, max_samples):
    # Make sure that feature highly correlated to the target have a higher
    # importance
    rng = np.random.RandomState(42)
    n_repeats = 5

    X, y = load_diabetes(return_X_y=True)
    y_with_little_noise = (y + rng.normal(scale=0.001, size=y.shape[0])).reshape(-1, 1)

    X = np.hstack([X, y_with_little_noise])

    clf = RandomForestRegressor(n_estimators=10, random_state=42)
    clf.fit(X, y)

    result = permutation_importance(
        clf,
        X,
        y,
        n_repeats=n_repeats,
        random_state=rng,
        n_jobs=n_jobs,
        max_samples=max_samples,
    )

    assert result.importances.shape == (X.shape[1], n_repeats)

    # the correlated feature with y was added as the last column and should
    # have the highest importance
    assert np.all(result.importances_mean[-1] > result.importances_mean[:-1])


@pytest.mark.parametrize("n_jobs", [1, 2])
@pytest.mark.parametrize("max_samples", [0.5, 1.0])
def test_permutation_importance_correlated_feature_regression_pandas(
    n_jobs, max_samples
):
    pd = pytest.importorskip("pandas")

    # Make sure that feature highly correlated to the target have a higher
    # importance
    rng = np.random.RandomState(42)
    n_repeats = 5

    dataset = load_iris()
    X, y = dataset.data, dataset.target
    y_with_little_noise = (y + rng.normal(scale=0.001, size=y.shape[0])).reshape(-1, 1)

    # Adds feature correlated with y as the last column
    X = pd.DataFrame(X, columns=dataset.feature_names)
    X["correlated_feature"] = y_with_little_noise

    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y)

    result = permutation_importance(
        clf,
        X,
        y,
        n_repeats=n_repeats,
        random_state=rng,
        n_jobs=n_jobs,
        max_samples=max_samples,
    )

    assert result.importances.shape == (X.shape[1], n_repeats)

    # the correlated feature with y was added as the last column and should
    # have the highest importance
    assert np.all(result.importances_mean[-1] > result.importances_mean[:-1])


@pytest.mark.parametrize("n_jobs", [1, 2])
@pytest.mark.parametrize("max_samples", [0.5, 1.0])
def test_robustness_to_high_cardinality_noisy_feature(n_jobs, max_samples, seed=42):
    # Permutation variable importance should not be affected by the high
    # cardinality bias of traditional feature importances, especially when
    # computed on a held-out test set:
    rng = np.random.RandomState(seed)
    n_repeats = 5
    n_samples = 1000
    n_classes = 5
    n_informative_features = 2
    n_noise_features = 1
    n_features = n_informative_features + n_noise_features

    # Generate a multiclass classification dataset and a set of informative
    # binary features that can be used to predict some classes of y exactly
    # while leaving some classes unexplained to make the problem harder.
    classes = np.arange(n_classes)
    y = rng.choice(classes, size=n_samples)
    X = np.hstack([(y == c).reshape(-1, 1) for c in classes[:n_informative_features]])
    X = X.astype(np.float32)

    # Not all target classes are explained by the binary class indicator
    # features:
    assert n_informative_features < n_classes

    # Add 10 other noisy features with high cardinality (numerical) values
    # that can be used to overfit the training data.
    X = np.concatenate([X, rng.randn(n_samples, n_noise_features)], axis=1)
    assert X.shape == (n_samples, n_features)

    # Split the dataset to be able to evaluate on a held-out test set. The
    # Test size should be large enough for importance measurements to be
    # stable:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=rng
    )
    clf = RandomForestClassifier(n_estimators=5, random_state=rng)
    clf.fit(X_train, y_train)

    # Variable importances computed by impurity decrease on the tree node
    # splits often use the noisy features in splits. This can give misleading
    # impression that high cardinality noisy variables are the most important:
    tree_importances = clf.feature_importances_
    informative_tree_importances = tree_importances[:n_informative_features]
    noisy_tree_importances = tree_importances[n_informative_features:]
    assert informative_tree_importances.max() < noisy_tree_importances.min()

    # Let's check that permutation-based feature importances do not have this
    # problem.
    r = permutation_importance(
        clf,
        X_test,
        y_test,
        n_repeats=n_repeats,
        random_state=rng,
        n_jobs=n_jobs,
        max_samples=max_samples,
    )

    assert r.importances.shape == (X.shape[1], n_repeats)

    # Split the importances between informative and noisy features
    informative_importances = r.importances_mean[:n_informative_features]
    noisy_importances = r.importances_mean[n_informative_features:]

    # Because we do not have a binary variable explaining each target classes,
    # the RF model will have to use the random variable to make some
    # (overfitting) splits (as max_depth is not set). Therefore the noisy
    # variables will be non-zero but with small values oscillating around
    # zero:
    assert max(np.abs(noisy_importances)) > 1e-7
    assert noisy_importances.max() < 0.05

    # The binary features correlated with y should have a higher importance
    # than the high cardinality noisy features.
    # The maximum test accuracy is 2 / 5 == 0.4, each informative feature
    # contributing approximately a bit more than 0.2 of accuracy.
    assert informative_importances.min() > 0.15


def test_permutation_importance_mixed_types():
    rng = np.random.RandomState(42)
    n_repeats = 4

    # Last column is correlated with y
    X = np.array([[1.0, 2.0, 3.0, np.nan], [2, 1, 2, 1]]).T
    y = np.array([0, 1, 0, 1])

    clf = make_pipeline(SimpleImputer(), LogisticRegression(solver="lbfgs"))
    clf.fit(X, y)
    result = permutation_importance(clf, X, y, n_repeats=n_repeats, random_state=rng)

    assert result.importances.shape == (X.shape[1], n_repeats)

    # the correlated feature with y is the last column and should
    # have the highest importance
    assert np.all(result.importances_mean[-1] > result.importances_mean[:-1])

    # use another random state
    rng = np.random.RandomState(0)
    result2 = permutation_importance(clf, X, y, n_repeats=n_repeats, random_state=rng)
    assert result2.importances.shape == (X.shape[1], n_repeats)

    assert not np.allclose(result.importances, result2.importances)

    # the correlated feature with y is the last column and should
    # have the highest importance
    assert np.all(result2.importances_mean[-1] > result2.importances_mean[:-1])


def test_permutation_importance_mixed_types_pandas():
    pd = pytest.importorskip("pandas")
    rng = np.random.RandomState(42)
    n_repeats = 5

    # Last column is correlated with y
    X = pd.DataFrame({"col1": [1.0, 2.0, 3.0, np.nan], "col2": ["a", "b", "a", "b"]})
    y = np.array([0, 1, 0, 1])

    num_preprocess = make_pipeline(SimpleImputer(), StandardScaler())
    preprocess = ColumnTransformer(
        [("num", num_preprocess, ["col1"]), ("cat", OneHotEncoder(), ["col2"])]
    )
    clf = make_pipeline(preprocess, LogisticRegression(solver="lbfgs"))
    clf.fit(X, y)

    result = permutation_importance(clf, X, y, n_repeats=n_repeats, random_state=rng)

    assert result.importances.shape == (X.shape[1], n_repeats)
    # the correlated feature with y is the last column and should
    # have the highest importance
    assert np.all(result.importances_mean[-1] > result.importances_mean[:-1])


def test_permutation_importance_linear_regresssion():
    X, y = make_regression(n_samples=500, n_features=10, random_state=0)

    X = scale(X)
    y = scale(y)

    lr = LinearRegression().fit(X, y)

    # this relationship can be computed in closed form
    expected_importances = 2 * lr.coef_ ** 2
    results = permutation_importance(
        lr, X, y, n_repeats=50, scoring="neg_mean_squared_error"
    )
    assert_allclose(
        expected_importances, results.importances_mean, rtol=1e-1, atol=1e-6
    )


@pytest.mark.parametrize("max_samples", [500, 1.0])
def test_permutation_importance_equivalence_sequential_parallel(max_samples):
    # regression test to make sure that sequential and parallel calls will
    # output the same results.
    # Also tests that max_samples equal to number of samples is equivalent to 1.0
    X, y = make_regression(n_samples=500, n_features=10, random_state=0)
    lr = LinearRegression().fit(X, y)

    importance_sequential = permutation_importance(
        lr, X, y, n_repeats=5, random_state=0, n_jobs=1, max_samples=max_samples
    )

    # First check that the problem is structured enough and that the model is
    # complex enough to not yield trivial, constant importances:
    imp_min = importance_sequential["importances"].min()
    imp_max = importance_sequential["importances"].max()
    assert imp_max - imp_min > 0.3

    # The actually check that parallelism does not impact the results
    # either with shared memory (threading) or without isolated memory
    # via process-based parallelism using the default backend
    # ('loky' or 'multiprocessing') depending on the joblib version:

    # process-based parallelism (by default):
    importance_processes = permutation_importance(
        lr, X, y, n_repeats=5, random_state=0, n_jobs=2
    )
    assert_allclose(
        importance_processes["importances"], importance_sequential["importances"]
    )

    # thread-based parallelism:
    with parallel_backend("threading"):
        importance_threading = permutation_importance(
            lr, X, y, n_repeats=5, random_state=0, n_jobs=2
        )
    assert_allclose(
        importance_threading["importances"], importance_sequential["importances"]
    )


@pytest.mark.parametrize("n_jobs", [None, 1, 2])
@pytest.mark.parametrize("max_samples", [0.5, 1.0])
def test_permutation_importance_equivalence_array_dataframe(n_jobs, max_samples):
    # This test checks that the column shuffling logic has the same behavior
    # both a dataframe and a simple numpy array.
    pd = pytest.importorskip("pandas")

    # regression test to make sure that sequential and parallel calls will
    # output the same results.
    X, y = make_regression(n_samples=100, n_features=5, random_state=0)
    X_df = pd.DataFrame(X)

    # Add a categorical feature that is statistically linked to y:
    binner = KBinsDiscretizer(n_bins=3, encode="ordinal")
    cat_column = binner.fit_transform(y.reshape(-1, 1))

    # Concatenate the extra column to the numpy array: integers will be
    # cast to float values
    X = np.hstack([X, cat_column])
    assert X.dtype.kind == "f"

    # Insert extra column as a non-numpy-native dtype (while keeping backward
    # compat for old pandas versions):
    if hasattr(pd, "Categorical"):
        cat_column = pd.Categorical(cat_column.ravel())
    else:
        cat_column = cat_column.ravel()
    new_col_idx = len(X_df.columns)
    X_df[new_col_idx] = cat_column
    assert X_df[new_col_idx].dtype == cat_column.dtype

    # Stich an arbitrary index to the dataframe:
    X_df.index = np.arange(len(X_df)).astype(str)

    rf = RandomForestRegressor(n_estimators=5, max_depth=3, random_state=0)
    rf.fit(X, y)

    n_repeats = 3
    importance_array = permutation_importance(
        rf,
        X,
        y,
        n_repeats=n_repeats,
        random_state=0,
        n_jobs=n_jobs,
        max_samples=max_samples,
    )

    # First check that the problem is structured enough and that the model is
    # complex enough to not yield trivial, constant importances:
    imp_min = importance_array["importances"].min()
    imp_max = importance_array["importances"].max()
    assert imp_max - imp_min > 0.3

    # Now check that importances computed on dataframe matche the values
    # of those computed on the array with the same data.
    importance_dataframe = permutation_importance(
        rf,
        X_df,
        y,
        n_repeats=n_repeats,
        random_state=0,
        n_jobs=n_jobs,
        max_samples=max_samples,
    )
    assert_allclose(
        importance_array["importances"], importance_dataframe["importances"]
    )


@pytest.mark.parametrize("input_type", ["array", "dataframe"])
def test_permutation_importance_large_memmaped_data(input_type):
    # Smoke, non-regression test for:
    # https://github.com/scikit-learn/scikit-learn/issues/15810
    n_samples, n_features = int(5e4), 4
    X, y = make_classification(
        n_samples=n_samples, n_features=n_features, random_state=0
    )
    assert X.nbytes > 1e6  # trigger joblib memmaping

    X = _convert_container(X, input_type)
    clf = DummyClassifier(strategy="prior").fit(X, y)

    # Actual smoke test: should not raise any error:
    n_repeats = 5
    r = permutation_importance(clf, X, y, n_repeats=n_repeats, n_jobs=2)

    # Auxiliary check: DummyClassifier is feature independent:
    # permutating feature should not change the predictions
    expected_importances = np.zeros((n_features, n_repeats))
    assert_allclose(expected_importances, r.importances)


def test_permutation_importance_sample_weight():
    # Creating data with 2 features and 1000 samples, where the target
    # variable is a linear combination of the two features, such that
    # in half of the samples the impact of feature 1 is twice the impact of
    # feature 2, and vice versa on the other half of the samples.
    rng = np.random.RandomState(1)
    n_samples = 1000
    n_features = 2
    n_half_samples = n_samples // 2
    x = rng.normal(0.0, 0.001, (n_samples, n_features))
    y = np.zeros(n_samples)
    y[:n_half_samples] = 2 * x[:n_half_samples, 0] + x[:n_half_samples, 1]
    y[n_half_samples:] = x[n_half_samples:, 0] + 2 * x[n_half_samples:, 1]

    # Fitting linear regression with perfect prediction
    lr = LinearRegression(fit_intercept=False)
    lr.fit(x, y)

    # When all samples are weighted with the same weights, the ratio of
    # the two features importance should equal to 1 on expectation (when using
    # mean absolutes error as the loss function).
    pi = permutation_importance(
        lr, x, y, random_state=1, scoring="neg_mean_absolute_error", n_repeats=200
    )
    x1_x2_imp_ratio_w_none = pi.importances_mean[0] / pi.importances_mean[1]
    assert x1_x2_imp_ratio_w_none == pytest.approx(1, 0.01)

    # When passing a vector of ones as the sample_weight, results should be
    # the same as in the case that sample_weight=None.
    w = np.ones(n_samples)
    pi = permutation_importance(
        lr,
        x,
        y,
        random_state=1,
        scoring="neg_mean_absolute_error",
        n_repeats=200,
        sample_weight=w,
    )
    x1_x2_imp_ratio_w_ones = pi.importances_mean[0] / pi.importances_mean[1]
    assert x1_x2_imp_ratio_w_ones == pytest.approx(x1_x2_imp_ratio_w_none, 0.01)

    # When the ratio between the weights of the first half of the samples and
    # the second half of the samples approaches to infinity, the ratio of
    # the two features importance should equal to 2 on expectation (when using
    # mean absolutes error as the loss function).
    w = np.hstack(
        [np.repeat(10.0 ** 10, n_half_samples), np.repeat(1.0, n_half_samples)]
    )
    lr.fit(x, y, w)
    pi = permutation_importance(
        lr,
        x,
        y,
        random_state=1,
        scoring="neg_mean_absolute_error",
        n_repeats=200,
        sample_weight=w,
    )
    x1_x2_imp_ratio_w = pi.importances_mean[0] / pi.importances_mean[1]
    assert x1_x2_imp_ratio_w / x1_x2_imp_ratio_w_none == pytest.approx(2, 0.01)


def test_permutation_importance_no_weights_scoring_function():
    # Creating a scorer function that does not takes sample_weight
    def my_scorer(estimator, X, y):
        return 1

    # Creating some data and estimator for the permutation test
    x = np.array([[1, 2], [3, 4]])
    y = np.array([1, 2])
    w = np.array([1, 1])
    lr = LinearRegression()
    lr.fit(x, y)

    # test that permutation_importance does not return error when
    # sample_weight is None
    try:
        permutation_importance(lr, x, y, random_state=1, scoring=my_scorer, n_repeats=1)
    except TypeError:
        pytest.fail(
            "permutation_test raised an error when using a scorer "
            "function that does not accept sample_weight even though "
            "sample_weight was None"
        )

    # test that permutation_importance raise exception when sample_weight is
    # not None
    with pytest.raises(TypeError):
        permutation_importance(
            lr, x, y, random_state=1, scoring=my_scorer, n_repeats=1, sample_weight=w
        )


@pytest.mark.parametrize(
    "list_single_scorer, multi_scorer",
    [
        (["r2", "neg_mean_squared_error"], ["r2", "neg_mean_squared_error"]),
        (
            ["r2", "neg_mean_squared_error"],
            {
                "r2": get_scorer("r2"),
                "neg_mean_squared_error": get_scorer("neg_mean_squared_error"),
            },
        ),
        (
            ["r2", "neg_mean_squared_error"],
            lambda estimator, X, y: {
                "r2": r2_score(y, estimator.predict(X)),
                "neg_mean_squared_error": -mean_squared_error(y, estimator.predict(X)),
            },
        ),
    ],
)
def test_permutation_importance_multi_metric(list_single_scorer, multi_scorer):
    # Test permutation importance when scoring contains multiple scorers

    # Creating some data and estimator for the permutation test
    x, y = make_regression(n_samples=500, n_features=10, random_state=0)
    lr = LinearRegression().fit(x, y)

    multi_importance = permutation_importance(
        lr, x, y, random_state=1, scoring=multi_scorer, n_repeats=2
    )
    assert set(multi_importance.keys()) == set(list_single_scorer)

    for scorer in list_single_scorer:
        multi_result = multi_importance[scorer]
        single_result = permutation_importance(
            lr, x, y, random_state=1, scoring=scorer, n_repeats=2
        )

        assert_allclose(multi_result.importances, single_result.importances)


@pytest.mark.parametrize("max_samples", [-1, 5])
def test_permutation_importance_max_samples_error(max_samples):
    """Check that a proper error message is raised when `max_samples` is not
    set to a valid input value.
    """
    X = np.array([(1.0, 2.0, 3.0, 4.0)]).T
    y = np.array([0, 1, 0, 1])

    clf = LogisticRegression()
    clf.fit(X, y)

    err_msg = r"max_samples must be in \(0, n_samples\]"

    with pytest.raises(ValueError, match=err_msg):
        permutation_importance(clf, X, y, max_samples=max_samples)
