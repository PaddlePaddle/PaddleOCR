"""
The :mod:`sklearn.neural_network` module includes models based on neural
networks.
"""

# License: BSD 3 clause

from ._rbm import BernoulliRBM

from ._multilayer_perceptron import MLPClassifier
from ._multilayer_perceptron import MLPRegressor

__all__ = ["BernoulliRBM", "MLPClassifier", "MLPRegressor"]
