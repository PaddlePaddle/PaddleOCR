"""
Testing for Elliptic Envelope algorithm (sklearn.covariance.elliptic_envelope).
"""

import numpy as np
import pytest

from sklearn.covariance import EllipticEnvelope
from sklearn.utils._testing import assert_almost_equal
from sklearn.utils._testing import assert_array_almost_equal
from sklearn.utils._testing import assert_array_equal
from sklearn.exceptions import NotFittedError


def test_elliptic_envelope():
    rnd = np.random.RandomState(0)
    X = rnd.randn(100, 10)
    clf = EllipticEnvelope(contamination=0.1)
    with pytest.raises(NotFittedError):
        clf.predict(X)
    with pytest.raises(NotFittedError):
        clf.decision_function(X)
    clf.fit(X)
    y_pred = clf.predict(X)
    scores = clf.score_samples(X)
    decisions = clf.decision_function(X)

    assert_array_almost_equal(scores, -clf.mahalanobis(X))
    assert_array_almost_equal(clf.mahalanobis(X), clf.dist_)
    assert_almost_equal(
        clf.score(X, np.ones(100)), (100 - y_pred[y_pred == -1].size) / 100.0
    )
    assert sum(y_pred == -1) == sum(decisions < 0)


def test_score_samples():
    X_train = [[1, 1], [1, 2], [2, 1]]
    clf1 = EllipticEnvelope(contamination=0.2).fit(X_train)
    clf2 = EllipticEnvelope().fit(X_train)
    assert_array_equal(
        clf1.score_samples([[2.0, 2.0]]),
        clf1.decision_function([[2.0, 2.0]]) + clf1.offset_,
    )
    assert_array_equal(
        clf2.score_samples([[2.0, 2.0]]),
        clf2.decision_function([[2.0, 2.0]]) + clf2.offset_,
    )
    assert_array_equal(
        clf1.score_samples([[2.0, 2.0]]), clf2.score_samples([[2.0, 2.0]])
    )
