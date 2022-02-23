import numpy as np
import scipy.sparse as sp
import pytest

from sklearn.utils._testing import assert_allclose
from sklearn.utils._testing import assert_array_almost_equal
from sklearn.utils import check_random_state
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron

iris = load_iris()
random_state = check_random_state(12)
indices = np.arange(iris.data.shape[0])
random_state.shuffle(indices)
X = iris.data[indices]
y = iris.target[indices]
X_csr = sp.csr_matrix(X)
X_csr.sort_indices()


class MyPerceptron:
    def __init__(self, n_iter=1):
        self.n_iter = n_iter

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features, dtype=np.float64)
        self.b = 0.0

        for t in range(self.n_iter):
            for i in range(n_samples):
                if self.predict(X[i])[0] != y[i]:
                    self.w += y[i] * X[i]
                    self.b += y[i]

    def project(self, X):
        return np.dot(X, self.w) + self.b

    def predict(self, X):
        X = np.atleast_2d(X)
        return np.sign(self.project(X))


def test_perceptron_accuracy():
    for data in (X, X_csr):
        clf = Perceptron(max_iter=100, tol=None, shuffle=False)
        clf.fit(data, y)
        score = clf.score(data, y)
        assert score > 0.7


def test_perceptron_correctness():
    y_bin = y.copy()
    y_bin[y != 1] = -1

    clf1 = MyPerceptron(n_iter=2)
    clf1.fit(X, y_bin)

    clf2 = Perceptron(max_iter=2, shuffle=False, tol=None)
    clf2.fit(X, y_bin)

    assert_array_almost_equal(clf1.w, clf2.coef_.ravel())


def test_undefined_methods():
    clf = Perceptron(max_iter=100)
    for meth in ("predict_proba", "predict_log_proba"):
        with pytest.raises(AttributeError):
            getattr(clf, meth)


def test_perceptron_l1_ratio():
    """Check that `l1_ratio` has an impact when `penalty='elasticnet'`"""
    clf1 = Perceptron(l1_ratio=0, penalty="elasticnet")
    clf1.fit(X, y)

    clf2 = Perceptron(l1_ratio=0.15, penalty="elasticnet")
    clf2.fit(X, y)

    assert clf1.score(X, y) != clf2.score(X, y)

    # check that the bounds of elastic net which should correspond to an l1 or
    # l2 penalty depending of `l1_ratio` value.
    clf_l1 = Perceptron(penalty="l1").fit(X, y)
    clf_elasticnet = Perceptron(l1_ratio=1, penalty="elasticnet").fit(X, y)
    assert_allclose(clf_l1.coef_, clf_elasticnet.coef_)

    clf_l2 = Perceptron(penalty="l2").fit(X, y)
    clf_elasticnet = Perceptron(l1_ratio=0, penalty="elasticnet").fit(X, y)
    assert_allclose(clf_l2.coef_, clf_elasticnet.coef_)
