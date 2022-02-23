import pytest
import numpy as np

from sklearn.utils._testing import assert_array_almost_equal
from sklearn.utils._testing import assert_array_equal
from sklearn.utils._testing import assert_allclose
from sklearn.utils._testing import skip_if_32bit
from sklearn.utils._testing import MinimalClassifier

from sklearn import datasets
from sklearn.linear_model import LogisticRegression, SGDClassifier, Lasso
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.base import BaseEstimator
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA


class NaNTag(BaseEstimator):
    def _more_tags(self):
        return {"allow_nan": True}


class NoNaNTag(BaseEstimator):
    def _more_tags(self):
        return {"allow_nan": False}


class NaNTagRandomForest(RandomForestClassifier):
    def _more_tags(self):
        return {"allow_nan": True}


iris = datasets.load_iris()
data, y = iris.data, iris.target
rng = np.random.RandomState(0)


def test_invalid_input():
    clf = SGDClassifier(
        alpha=0.1, max_iter=10, shuffle=True, random_state=None, tol=None
    )
    for threshold in ["gobbledigook", ".5 * gobbledigook"]:
        model = SelectFromModel(clf, threshold=threshold)
        model.fit(data, y)
        with pytest.raises(ValueError):
            model.transform(data)


def test_input_estimator_unchanged():
    # Test that SelectFromModel fits on a clone of the estimator.
    est = RandomForestClassifier()
    transformer = SelectFromModel(estimator=est)
    transformer.fit(data, y)
    assert transformer.estimator is est


@pytest.mark.parametrize(
    "max_features, err_type, err_msg",
    [
        (-1, ValueError, "'max_features' should be 0 and"),
        (data.shape[1] + 1, ValueError, "'max_features' should be 0 and"),
        ("gobbledigook", TypeError, "should be an integer"),
        ("all", TypeError, "should be an integer"),
    ],
)
def test_max_features_error(max_features, err_type, err_msg):
    clf = RandomForestClassifier(n_estimators=50, random_state=0)

    transformer = SelectFromModel(
        estimator=clf, max_features=max_features, threshold=-np.inf
    )
    with pytest.raises(err_type, match=err_msg):
        transformer.fit(data, y)


@pytest.mark.parametrize("max_features", [0, 2, data.shape[1]])
def test_max_features_dim(max_features):
    clf = RandomForestClassifier(n_estimators=50, random_state=0)
    transformer = SelectFromModel(
        estimator=clf, max_features=max_features, threshold=-np.inf
    )
    X_trans = transformer.fit_transform(data, y)
    assert X_trans.shape[1] == max_features


class FixedImportanceEstimator(BaseEstimator):
    def __init__(self, importances):
        self.importances = importances

    def fit(self, X, y=None):
        self.feature_importances_ = np.array(self.importances)


def test_max_features():
    # Test max_features parameter using various values
    X, y = datasets.make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=3,
        n_redundant=0,
        n_repeated=0,
        shuffle=False,
        random_state=0,
    )
    max_features = X.shape[1]
    est = RandomForestClassifier(n_estimators=50, random_state=0)

    transformer1 = SelectFromModel(estimator=est, threshold=-np.inf)
    transformer2 = SelectFromModel(
        estimator=est, max_features=max_features, threshold=-np.inf
    )
    X_new1 = transformer1.fit_transform(X, y)
    X_new2 = transformer2.fit_transform(X, y)
    assert_allclose(X_new1, X_new2)

    # Test max_features against actual model.
    transformer1 = SelectFromModel(estimator=Lasso(alpha=0.025, random_state=42))
    X_new1 = transformer1.fit_transform(X, y)
    scores1 = np.abs(transformer1.estimator_.coef_)
    candidate_indices1 = np.argsort(-scores1, kind="mergesort")

    for n_features in range(1, X_new1.shape[1] + 1):
        transformer2 = SelectFromModel(
            estimator=Lasso(alpha=0.025, random_state=42),
            max_features=n_features,
            threshold=-np.inf,
        )
        X_new2 = transformer2.fit_transform(X, y)
        scores2 = np.abs(transformer2.estimator_.coef_)
        candidate_indices2 = np.argsort(-scores2, kind="mergesort")
        assert_allclose(
            X[:, candidate_indices1[:n_features]], X[:, candidate_indices2[:n_features]]
        )
    assert_allclose(transformer1.estimator_.coef_, transformer2.estimator_.coef_)


def test_max_features_tiebreak():
    # Test if max_features can break tie among feature importance
    X, y = datasets.make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=3,
        n_redundant=0,
        n_repeated=0,
        shuffle=False,
        random_state=0,
    )
    max_features = X.shape[1]

    feature_importances = np.array([4, 4, 4, 4, 3, 3, 3, 2, 2, 1])
    for n_features in range(1, max_features + 1):
        transformer = SelectFromModel(
            FixedImportanceEstimator(feature_importances),
            max_features=n_features,
            threshold=-np.inf,
        )
        X_new = transformer.fit_transform(X, y)
        selected_feature_indices = np.where(transformer._get_support_mask())[0]
        assert_array_equal(selected_feature_indices, np.arange(n_features))
        assert X_new.shape[1] == n_features


def test_threshold_and_max_features():
    X, y = datasets.make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=3,
        n_redundant=0,
        n_repeated=0,
        shuffle=False,
        random_state=0,
    )
    est = RandomForestClassifier(n_estimators=50, random_state=0)

    transformer1 = SelectFromModel(estimator=est, max_features=3, threshold=-np.inf)
    X_new1 = transformer1.fit_transform(X, y)

    transformer2 = SelectFromModel(estimator=est, threshold=0.04)
    X_new2 = transformer2.fit_transform(X, y)

    transformer3 = SelectFromModel(estimator=est, max_features=3, threshold=0.04)
    X_new3 = transformer3.fit_transform(X, y)
    assert X_new3.shape[1] == min(X_new1.shape[1], X_new2.shape[1])
    selected_indices = transformer3.transform(np.arange(X.shape[1])[np.newaxis, :])
    assert_allclose(X_new3, X[:, selected_indices[0]])


@skip_if_32bit
def test_feature_importances():
    X, y = datasets.make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=3,
        n_redundant=0,
        n_repeated=0,
        shuffle=False,
        random_state=0,
    )

    est = RandomForestClassifier(n_estimators=50, random_state=0)
    for threshold, func in zip(["mean", "median"], [np.mean, np.median]):
        transformer = SelectFromModel(estimator=est, threshold=threshold)
        transformer.fit(X, y)
        assert hasattr(transformer.estimator_, "feature_importances_")

        X_new = transformer.transform(X)
        assert X_new.shape[1] < X.shape[1]
        importances = transformer.estimator_.feature_importances_

        feature_mask = np.abs(importances) > func(importances)
        assert_array_almost_equal(X_new, X[:, feature_mask])


def test_sample_weight():
    # Ensure sample weights are passed to underlying estimator
    X, y = datasets.make_classification(
        n_samples=100,
        n_features=10,
        n_informative=3,
        n_redundant=0,
        n_repeated=0,
        shuffle=False,
        random_state=0,
    )

    # Check with sample weights
    sample_weight = np.ones(y.shape)
    sample_weight[y == 1] *= 100

    est = LogisticRegression(random_state=0, fit_intercept=False)
    transformer = SelectFromModel(estimator=est)
    transformer.fit(X, y, sample_weight=None)
    mask = transformer._get_support_mask()
    transformer.fit(X, y, sample_weight=sample_weight)
    weighted_mask = transformer._get_support_mask()
    assert not np.all(weighted_mask == mask)
    transformer.fit(X, y, sample_weight=3 * sample_weight)
    reweighted_mask = transformer._get_support_mask()
    assert np.all(weighted_mask == reweighted_mask)


def test_coef_default_threshold():
    X, y = datasets.make_classification(
        n_samples=100,
        n_features=10,
        n_informative=3,
        n_redundant=0,
        n_repeated=0,
        shuffle=False,
        random_state=0,
    )

    # For the Lasso and related models, the threshold defaults to 1e-5
    transformer = SelectFromModel(estimator=Lasso(alpha=0.1, random_state=42))
    transformer.fit(X, y)
    X_new = transformer.transform(X)
    mask = np.abs(transformer.estimator_.coef_) > 1e-5
    assert_array_almost_equal(X_new, X[:, mask])


@skip_if_32bit
def test_2d_coef():
    X, y = datasets.make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=3,
        n_redundant=0,
        n_repeated=0,
        shuffle=False,
        random_state=0,
        n_classes=4,
    )

    est = LogisticRegression()
    for threshold, func in zip(["mean", "median"], [np.mean, np.median]):
        for order in [1, 2, np.inf]:
            # Fit SelectFromModel a multi-class problem
            transformer = SelectFromModel(
                estimator=LogisticRegression(), threshold=threshold, norm_order=order
            )
            transformer.fit(X, y)
            assert hasattr(transformer.estimator_, "coef_")
            X_new = transformer.transform(X)
            assert X_new.shape[1] < X.shape[1]

            # Manually check that the norm is correctly performed
            est.fit(X, y)
            importances = np.linalg.norm(est.coef_, axis=0, ord=order)
            feature_mask = importances > func(importances)
            assert_array_almost_equal(X_new, X[:, feature_mask])


def test_partial_fit():
    est = PassiveAggressiveClassifier(
        random_state=0, shuffle=False, max_iter=5, tol=None
    )
    transformer = SelectFromModel(estimator=est)
    transformer.partial_fit(data, y, classes=np.unique(y))
    old_model = transformer.estimator_
    transformer.partial_fit(data, y, classes=np.unique(y))
    new_model = transformer.estimator_
    assert old_model is new_model

    X_transform = transformer.transform(data)
    transformer.fit(np.vstack((data, data)), np.concatenate((y, y)))
    assert_array_almost_equal(X_transform, transformer.transform(data))

    # check that if est doesn't have partial_fit, neither does SelectFromModel
    transformer = SelectFromModel(estimator=RandomForestClassifier())
    assert not hasattr(transformer, "partial_fit")


def test_calling_fit_reinitializes():
    est = LinearSVC(random_state=0)
    transformer = SelectFromModel(estimator=est)
    transformer.fit(data, y)
    transformer.set_params(estimator__C=100)
    transformer.fit(data, y)
    assert transformer.estimator_.C == 100


def test_prefit():
    # Test all possible combinations of the prefit parameter.

    # Passing a prefit parameter with the selected model
    # and fitting a unfit model with prefit=False should give same results.
    clf = SGDClassifier(alpha=0.1, max_iter=10, shuffle=True, random_state=0, tol=None)
    model = SelectFromModel(clf)
    model.fit(data, y)
    X_transform = model.transform(data)
    clf.fit(data, y)
    model = SelectFromModel(clf, prefit=True)
    assert_array_almost_equal(model.transform(data), X_transform)

    # Check that the model is rewritten if prefit=False and a fitted model is
    # passed
    model = SelectFromModel(clf, prefit=False)
    model.fit(data, y)
    assert_array_almost_equal(model.transform(data), X_transform)

    # Check that prefit=True and calling fit raises a ValueError
    model = SelectFromModel(clf, prefit=True)
    with pytest.raises(ValueError):
        model.fit(data, y)


def test_threshold_string():
    est = RandomForestClassifier(n_estimators=50, random_state=0)
    model = SelectFromModel(est, threshold="0.5*mean")
    model.fit(data, y)
    X_transform = model.transform(data)

    # Calculate the threshold from the estimator directly.
    est.fit(data, y)
    threshold = 0.5 * np.mean(est.feature_importances_)
    mask = est.feature_importances_ > threshold
    assert_array_almost_equal(X_transform, data[:, mask])


def test_threshold_without_refitting():
    # Test that the threshold can be set without refitting the model.
    clf = SGDClassifier(alpha=0.1, max_iter=10, shuffle=True, random_state=0, tol=None)
    model = SelectFromModel(clf, threshold="0.1 * mean")
    model.fit(data, y)
    X_transform = model.transform(data)

    # Set a higher threshold to filter out more features.
    model.threshold = "1.0 * mean"
    assert X_transform.shape[1] > model.transform(data).shape[1]


def test_fit_accepts_nan_inf():
    # Test that fit doesn't check for np.inf and np.nan values.
    clf = HistGradientBoostingClassifier(random_state=0)

    model = SelectFromModel(estimator=clf)

    nan_data = data.copy()
    nan_data[0] = np.NaN
    nan_data[1] = np.Inf

    model.fit(data, y)


def test_transform_accepts_nan_inf():
    # Test that transform doesn't check for np.inf and np.nan values.
    clf = NaNTagRandomForest(n_estimators=100, random_state=0)
    nan_data = data.copy()

    model = SelectFromModel(estimator=clf)
    model.fit(nan_data, y)

    nan_data[0] = np.NaN
    nan_data[1] = np.Inf

    model.transform(nan_data)


def test_allow_nan_tag_comes_from_estimator():
    allow_nan_est = NaNTag()
    model = SelectFromModel(estimator=allow_nan_est)
    assert model._get_tags()["allow_nan"] is True

    no_nan_est = NoNaNTag()
    model = SelectFromModel(estimator=no_nan_est)
    assert model._get_tags()["allow_nan"] is False


def _pca_importances(pca_estimator):
    return np.abs(pca_estimator.explained_variance_)


@pytest.mark.parametrize(
    "estimator, importance_getter",
    [
        (
            make_pipeline(PCA(random_state=0), LogisticRegression()),
            "named_steps.logisticregression.coef_",
        ),
        (PCA(random_state=0), _pca_importances),
    ],
)
def test_importance_getter(estimator, importance_getter):
    selector = SelectFromModel(
        estimator, threshold="mean", importance_getter=importance_getter
    )
    selector.fit(data, y)
    assert selector.transform(data).shape[1] == 1


def test_estimator_does_not_support_feature_names():
    """SelectFromModel works with estimators that do not support feature_names_in_.

    Non-regression test for #21949.
    """
    pytest.importorskip("pandas")
    X, y = datasets.load_iris(as_frame=True, return_X_y=True)
    all_feature_names = set(X.columns)

    def importance_getter(estimator):
        return np.arange(X.shape[1])

    selector = SelectFromModel(
        MinimalClassifier(), importance_getter=importance_getter
    ).fit(X, y)

    # selector learns the feature names itself
    assert_array_equal(selector.feature_names_in_, X.columns)

    feature_names_out = set(selector.get_feature_names_out())
    assert feature_names_out < all_feature_names

    with pytest.warns(None) as records:
        selector.transform(X.iloc[1:3])
    assert not [str(record.message) for record in records]
