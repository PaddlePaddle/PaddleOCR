"""
The :mod:`sklearn.linear_model` module implements a variety of linear models.
"""

# See http://scikit-learn.sourceforge.net/modules/sgd.html and
# http://scikit-learn.sourceforge.net/modules/linear_model.html for
# complete documentation.

from ._base import LinearRegression
from ._bayes import BayesianRidge, ARDRegression
from ._least_angle import (
    Lars,
    LassoLars,
    lars_path,
    lars_path_gram,
    LarsCV,
    LassoLarsCV,
    LassoLarsIC,
)
from ._coordinate_descent import (
    Lasso,
    ElasticNet,
    LassoCV,
    ElasticNetCV,
    lasso_path,
    enet_path,
    MultiTaskLasso,
    MultiTaskElasticNet,
    MultiTaskElasticNetCV,
    MultiTaskLassoCV,
)
from ._glm import PoissonRegressor, GammaRegressor, TweedieRegressor
from ._huber import HuberRegressor
from ._sgd_fast import Hinge, Log, ModifiedHuber, SquaredLoss, Huber
from ._stochastic_gradient import SGDClassifier, SGDRegressor, SGDOneClassSVM
from ._ridge import Ridge, RidgeCV, RidgeClassifier, RidgeClassifierCV, ridge_regression
from ._logistic import LogisticRegression, LogisticRegressionCV
from ._omp import (
    orthogonal_mp,
    orthogonal_mp_gram,
    OrthogonalMatchingPursuit,
    OrthogonalMatchingPursuitCV,
)
from ._passive_aggressive import PassiveAggressiveClassifier
from ._passive_aggressive import PassiveAggressiveRegressor
from ._perceptron import Perceptron

from ._quantile import QuantileRegressor
from ._ransac import RANSACRegressor
from ._theil_sen import TheilSenRegressor

__all__ = [
    "ARDRegression",
    "BayesianRidge",
    "ElasticNet",
    "ElasticNetCV",
    "Hinge",
    "Huber",
    "HuberRegressor",
    "Lars",
    "LarsCV",
    "Lasso",
    "LassoCV",
    "LassoLars",
    "LassoLarsCV",
    "LassoLarsIC",
    "LinearRegression",
    "Log",
    "LogisticRegression",
    "LogisticRegressionCV",
    "ModifiedHuber",
    "MultiTaskElasticNet",
    "MultiTaskElasticNetCV",
    "MultiTaskLasso",
    "MultiTaskLassoCV",
    "OrthogonalMatchingPursuit",
    "OrthogonalMatchingPursuitCV",
    "PassiveAggressiveClassifier",
    "PassiveAggressiveRegressor",
    "Perceptron",
    "QuantileRegressor",
    "Ridge",
    "RidgeCV",
    "RidgeClassifier",
    "RidgeClassifierCV",
    "SGDClassifier",
    "SGDRegressor",
    "SGDOneClassSVM",
    "SquaredLoss",
    "TheilSenRegressor",
    "enet_path",
    "lars_path",
    "lars_path_gram",
    "lasso_path",
    "orthogonal_mp",
    "orthogonal_mp_gram",
    "ridge_regression",
    "RANSACRegressor",
    "PoissonRegressor",
    "GammaRegressor",
    "TweedieRegressor",
]
