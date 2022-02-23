# Author: Maria Telenczuk <https://github.com/maikia>
#
# License: BSD 3 clause

import pytest

import sys
import numpy as np

from sklearn.base import is_classifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import ARDRegression

from sklearn.utils.fixes import np_version, parse_version
from sklearn.utils import check_random_state


@pytest.mark.parametrize(
    "normalize, n_warnings, warning_category",
    [(True, 1, FutureWarning), (False, 1, FutureWarning), ("deprecated", 0, None)],
)
@pytest.mark.parametrize(
    "estimator",
    [
        LinearRegression,
        Ridge,
        RidgeCV,
        RidgeClassifier,
        RidgeClassifierCV,
        BayesianRidge,
        ARDRegression,
    ],
)
# FIXME remove test in 1.2
@pytest.mark.xfail(
    sys.platform == "darwin" and np_version < parse_version("1.22"),
    reason="https://github.com/scikit-learn/scikit-learn/issues/21395",
)
def test_linear_model_normalize_deprecation_message(
    estimator, normalize, n_warnings, warning_category
):
    # check that we issue a FutureWarning when normalize was set in
    # linear model
    rng = check_random_state(0)
    n_samples = 200
    n_features = 2
    X = rng.randn(n_samples, n_features)
    X[X < 0.1] = 0.0
    y = rng.rand(n_samples)
    if is_classifier(estimator):
        y = np.sign(y)

    model = estimator(normalize=normalize)
    with pytest.warns(warning_category) as record:
        model.fit(X, y)
    # Filter record in case other unrelated warnings are raised
    unwanted = [r for r in record if r.category != warning_category]
    if len(unwanted):
        msg = "unexpected warnings:\n"
        for w in unwanted:
            msg += str(w)
            msg += "\n"
        raise AssertionError(msg)
    wanted = [r for r in record if r.category == warning_category]
    if warning_category is not None:
        assert "'normalize' was deprecated" in str(wanted[0].message)
    assert len(wanted) == n_warnings
