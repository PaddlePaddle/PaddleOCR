import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest

from sklearn.manifold import _mds as mds
from sklearn.utils._testing import ignore_warnings


def test_smacof():
    # test metric smacof using the data of "Modern Multidimensional Scaling",
    # Borg & Groenen, p 154
    sim = np.array([[0, 5, 3, 4], [5, 0, 2, 2], [3, 2, 0, 1], [4, 2, 1, 0]])
    Z = np.array([[-0.266, -0.539], [0.451, 0.252], [0.016, -0.238], [-0.200, 0.524]])
    X, _ = mds.smacof(sim, init=Z, n_components=2, max_iter=1, n_init=1)
    X_true = np.array(
        [[-1.415, -2.471], [1.633, 1.107], [0.249, -0.067], [-0.468, 1.431]]
    )
    assert_array_almost_equal(X, X_true, decimal=3)


def test_smacof_error():
    # Not symmetric similarity matrix:
    sim = np.array([[0, 5, 9, 4], [5, 0, 2, 2], [3, 2, 0, 1], [4, 2, 1, 0]])

    with pytest.raises(ValueError):
        mds.smacof(sim)

    # Not squared similarity matrix:
    sim = np.array([[0, 5, 9, 4], [5, 0, 2, 2], [4, 2, 1, 0]])

    with pytest.raises(ValueError):
        mds.smacof(sim)

    # init not None and not correct format:
    sim = np.array([[0, 5, 3, 4], [5, 0, 2, 2], [3, 2, 0, 1], [4, 2, 1, 0]])

    Z = np.array([[-0.266, -0.539], [0.016, -0.238], [-0.200, 0.524]])
    with pytest.raises(ValueError):
        mds.smacof(sim, init=Z, n_init=1)


def test_MDS():
    sim = np.array([[0, 5, 3, 4], [5, 0, 2, 2], [3, 2, 0, 1], [4, 2, 1, 0]])
    mds_clf = mds.MDS(metric=False, n_jobs=3, dissimilarity="precomputed")
    mds_clf.fit(sim)


# TODO: Remove in 1.1
def test_MDS_pairwise_deprecated():
    mds_clf = mds.MDS(metric="precomputed")
    msg = r"Attribute `_pairwise` was deprecated in version 0\.24"
    with pytest.warns(FutureWarning, match=msg):
        mds_clf._pairwise


# TODO: Remove in 1.1
@ignore_warnings(category=FutureWarning)
@pytest.mark.parametrize(
    "dissimilarity, expected_pairwise",
    [
        ("precomputed", True),
        ("euclidean", False),
    ],
)
def test_MDS_pairwise(dissimilarity, expected_pairwise):
    # _pairwise attribute is set correctly
    mds_clf = mds.MDS(dissimilarity=dissimilarity)
    assert mds_clf._pairwise == expected_pairwise
