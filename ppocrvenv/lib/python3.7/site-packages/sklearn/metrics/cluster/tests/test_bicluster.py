"""Testing for bicluster metrics module"""

import numpy as np

from sklearn.utils._testing import assert_almost_equal

from sklearn.metrics.cluster._bicluster import _jaccard
from sklearn.metrics import consensus_score


def test_jaccard():
    a1 = np.array([True, True, False, False])
    a2 = np.array([True, True, True, True])
    a3 = np.array([False, True, True, False])
    a4 = np.array([False, False, True, True])

    assert _jaccard(a1, a1, a1, a1) == 1
    assert _jaccard(a1, a1, a2, a2) == 0.25
    assert _jaccard(a1, a1, a3, a3) == 1.0 / 7
    assert _jaccard(a1, a1, a4, a4) == 0


def test_consensus_score():
    a = [[True, True, False, False], [False, False, True, True]]
    b = a[::-1]

    assert consensus_score((a, a), (a, a)) == 1
    assert consensus_score((a, a), (b, b)) == 1
    assert consensus_score((a, b), (a, b)) == 1
    assert consensus_score((a, b), (b, a)) == 1

    assert consensus_score((a, a), (b, a)) == 0
    assert consensus_score((a, a), (a, b)) == 0
    assert consensus_score((b, b), (a, b)) == 0
    assert consensus_score((b, b), (b, a)) == 0


def test_consensus_score_issue2445():
    """Different number of biclusters in A and B"""
    a_rows = np.array(
        [
            [True, True, False, False],
            [False, False, True, True],
            [False, False, False, True],
        ]
    )
    a_cols = np.array(
        [
            [True, True, False, False],
            [False, False, True, True],
            [False, False, False, True],
        ]
    )
    idx = [0, 2]
    s = consensus_score((a_rows, a_cols), (a_rows[idx], a_cols[idx]))
    # B contains 2 of the 3 biclusters in A, so score should be 2/3
    assert_almost_equal(s, 2.0 / 3.0)
