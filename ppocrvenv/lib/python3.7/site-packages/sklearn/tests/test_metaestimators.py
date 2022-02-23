"""Common tests for metaestimators"""
import functools
from inspect import signature

import numpy as np
import pytest

from sklearn.base import BaseEstimator
from sklearn.base import is_regressor
from sklearn.datasets import make_classification
from sklearn.utils import all_estimators
from sklearn.utils.estimator_checks import _enforce_estimator_tags_x
from sklearn.utils.estimator_checks import _enforce_estimator_tags_y
from sklearn.utils.validation import check_is_fitted
from sklearn.utils._testing import set_random_state
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import RFE, RFECV
from sklearn.ensemble import BaggingClassifier
from sklearn.exceptions import NotFittedError
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler, MaxAbsScaler


class DelegatorData:
    def __init__(
        self, name, construct, skip_methods=(), fit_args=make_classification()
    ):
        self.name = name
        self.construct = construct
        self.fit_args = fit_args
        self.skip_methods = skip_methods


DELEGATING_METAESTIMATORS = [
    DelegatorData("Pipeline", lambda est: Pipeline([("est", est)])),
    DelegatorData(
        "GridSearchCV",
        lambda est: GridSearchCV(est, param_grid={"param": [5]}, cv=2),
        skip_methods=["score"],
    ),
    DelegatorData(
        "RandomizedSearchCV",
        lambda est: RandomizedSearchCV(
            est, param_distributions={"param": [5]}, cv=2, n_iter=1
        ),
        skip_methods=["score"],
    ),
    DelegatorData("RFE", RFE, skip_methods=["transform", "inverse_transform"]),
    DelegatorData("RFECV", RFECV, skip_methods=["transform", "inverse_transform"]),
    DelegatorData(
        "BaggingClassifier",
        BaggingClassifier,
        skip_methods=[
            "transform",
            "inverse_transform",
            "score",
            "predict_proba",
            "predict_log_proba",
            "predict",
        ],
    ),
    DelegatorData(
        "SelfTrainingClassifier",
        lambda est: SelfTrainingClassifier(est),
        skip_methods=["transform", "inverse_transform", "predict_proba"],
    ),
]


def test_metaestimator_delegation():
    # Ensures specified metaestimators have methods iff subestimator does
    def hides(method):
        @property
        def wrapper(obj):
            if obj.hidden_method == method.__name__:
                raise AttributeError("%r is hidden" % obj.hidden_method)
            return functools.partial(method, obj)

        return wrapper

    class SubEstimator(BaseEstimator):
        def __init__(self, param=1, hidden_method=None):
            self.param = param
            self.hidden_method = hidden_method

        def fit(self, X, y=None, *args, **kwargs):
            self.coef_ = np.arange(X.shape[1])
            self.classes_ = []
            return True

        def _check_fit(self):
            check_is_fitted(self)

        @hides
        def inverse_transform(self, X, *args, **kwargs):
            self._check_fit()
            return X

        @hides
        def transform(self, X, *args, **kwargs):
            self._check_fit()
            return X

        @hides
        def predict(self, X, *args, **kwargs):
            self._check_fit()
            return np.ones(X.shape[0])

        @hides
        def predict_proba(self, X, *args, **kwargs):
            self._check_fit()
            return np.ones(X.shape[0])

        @hides
        def predict_log_proba(self, X, *args, **kwargs):
            self._check_fit()
            return np.ones(X.shape[0])

        @hides
        def decision_function(self, X, *args, **kwargs):
            self._check_fit()
            return np.ones(X.shape[0])

        @hides
        def score(self, X, y, *args, **kwargs):
            self._check_fit()
            return 1.0

    methods = [
        k
        for k in SubEstimator.__dict__.keys()
        if not k.startswith("_") and not k.startswith("fit")
    ]
    methods.sort()

    for delegator_data in DELEGATING_METAESTIMATORS:
        delegate = SubEstimator()
        delegator = delegator_data.construct(delegate)
        for method in methods:
            if method in delegator_data.skip_methods:
                continue
            assert hasattr(delegate, method)
            assert hasattr(
                delegator, method
            ), "%s does not have method %r when its delegate does" % (
                delegator_data.name,
                method,
            )
            # delegation before fit raises a NotFittedError
            if method == "score":
                with pytest.raises(NotFittedError):
                    getattr(delegator, method)(
                        delegator_data.fit_args[0], delegator_data.fit_args[1]
                    )
            else:
                with pytest.raises(NotFittedError):
                    getattr(delegator, method)(delegator_data.fit_args[0])

        delegator.fit(*delegator_data.fit_args)
        for method in methods:
            if method in delegator_data.skip_methods:
                continue
            # smoke test delegation
            if method == "score":
                getattr(delegator, method)(
                    delegator_data.fit_args[0], delegator_data.fit_args[1]
                )
            else:
                getattr(delegator, method)(delegator_data.fit_args[0])

        for method in methods:
            if method in delegator_data.skip_methods:
                continue
            delegate = SubEstimator(hidden_method=method)
            delegator = delegator_data.construct(delegate)
            assert not hasattr(delegate, method)
            assert not hasattr(
                delegator, method
            ), "%s has method %r when its delegate does not" % (
                delegator_data.name,
                method,
            )


def _generate_meta_estimator_instances_with_pipeline():
    """Generate instances of meta-estimators fed with a pipeline

    Are considered meta-estimators all estimators accepting one of "estimator",
    "base_estimator" or "estimators".
    """
    for _, Estimator in sorted(all_estimators()):
        sig = set(signature(Estimator).parameters)

        if "estimator" in sig or "base_estimator" in sig or "regressor" in sig:
            if is_regressor(Estimator):
                estimator = make_pipeline(TfidfVectorizer(), Ridge())
                param_grid = {"ridge__alpha": [0.1, 1.0]}
            else:
                estimator = make_pipeline(TfidfVectorizer(), LogisticRegression())
                param_grid = {"logisticregression__C": [0.1, 1.0]}

            if "param_grid" in sig or "param_distributions" in sig:
                # SearchCV estimators
                extra_params = {"n_iter": 2} if "n_iter" in sig else {}
                yield Estimator(estimator, param_grid, **extra_params)
            else:
                yield Estimator(estimator)

        elif "transformer_list" in sig:
            # FeatureUnion
            transformer_list = [
                ("trans1", make_pipeline(TfidfVectorizer(), MaxAbsScaler())),
                (
                    "trans2",
                    make_pipeline(TfidfVectorizer(), StandardScaler(with_mean=False)),
                ),
            ]
            yield Estimator(transformer_list)

        elif "estimators" in sig:
            # stacking, voting
            if is_regressor(Estimator):
                estimator = [
                    ("est1", make_pipeline(TfidfVectorizer(), Ridge(alpha=0.1))),
                    ("est2", make_pipeline(TfidfVectorizer(), Ridge(alpha=1))),
                ]
            else:
                estimator = [
                    (
                        "est1",
                        make_pipeline(TfidfVectorizer(), LogisticRegression(C=0.1)),
                    ),
                    ("est2", make_pipeline(TfidfVectorizer(), LogisticRegression(C=1))),
                ]
            yield Estimator(estimator)

        else:
            continue


# TODO: remove data validation for the following estimators
# They should be able to work on any data and delegate data validation to
# their inner estimator(s).
DATA_VALIDATION_META_ESTIMATORS_TO_IGNORE = [
    "AdaBoostClassifier",
    "AdaBoostRegressor",
    "BaggingClassifier",
    "BaggingRegressor",
    "ClassifierChain",  # data validation is necessary
    "IterativeImputer",
    "OneVsOneClassifier",  # input validation can't be avoided
    "RANSACRegressor",
    "RFE",
    "RFECV",
    "RegressorChain",  # data validation is necessary
    "SelfTrainingClassifier",
    "SequentialFeatureSelector",  # not applicable (2D data mandatory)
]

DATA_VALIDATION_META_ESTIMATORS = [
    est
    for est in _generate_meta_estimator_instances_with_pipeline()
    if est.__class__.__name__ not in DATA_VALIDATION_META_ESTIMATORS_TO_IGNORE
]


def _get_meta_estimator_id(estimator):
    return estimator.__class__.__name__


@pytest.mark.parametrize(
    "estimator", DATA_VALIDATION_META_ESTIMATORS, ids=_get_meta_estimator_id
)
def test_meta_estimators_delegate_data_validation(estimator):
    # Check that meta-estimators delegate data validation to the inner
    # estimator(s).
    rng = np.random.RandomState(0)
    set_random_state(estimator)

    n_samples = 30
    X = rng.choice(np.array(["aa", "bb", "cc"], dtype=object), size=n_samples)

    if is_regressor(estimator):
        y = rng.normal(size=n_samples)
    else:
        y = rng.randint(3, size=n_samples)

    # We convert to lists to make sure it works on array-like
    X = _enforce_estimator_tags_x(estimator, X).tolist()
    y = _enforce_estimator_tags_y(estimator, y).tolist()

    # Calling fit should not raise any data validation exception since X is a
    # valid input datastructure for the first step of the pipeline passed as
    # base estimator to the meta estimator.
    estimator.fit(X, y)

    # n_features_in_ should not be defined since data is not tabular data.
    assert not hasattr(estimator, "n_features_in_")
