import numpy as np
import pytest
from scipy import sparse as sp

from numpy.testing import assert_array_equal

from sklearn.base import BaseEstimator
from sklearn.feature_selection._base import SelectorMixin
from sklearn.utils import check_array


class StepSelector(SelectorMixin, BaseEstimator):
    """Retain every `step` features (beginning with 0)"""

    def __init__(self, step=2):
        self.step = step

    def fit(self, X, y=None):
        X = check_array(X, accept_sparse="csc")
        self.n_input_feats = X.shape[1]
        return self

    def _get_support_mask(self):
        mask = np.zeros(self.n_input_feats, dtype=bool)
        mask[:: self.step] = True
        return mask


support = [True, False] * 5
support_inds = [0, 2, 4, 6, 8]
X = np.arange(20).reshape(2, 10)
Xt = np.arange(0, 20, 2).reshape(2, 5)
Xinv = X.copy()
Xinv[:, 1::2] = 0
y = [0, 1]
feature_names = list("ABCDEFGHIJ")
feature_names_t = feature_names[::2]
feature_names_inv = np.array(feature_names)
feature_names_inv[1::2] = ""


def test_transform_dense():
    sel = StepSelector()
    Xt_actual = sel.fit(X, y).transform(X)
    Xt_actual2 = StepSelector().fit_transform(X, y)
    assert_array_equal(Xt, Xt_actual)
    assert_array_equal(Xt, Xt_actual2)

    # Check dtype matches
    assert np.int32 == sel.transform(X.astype(np.int32)).dtype
    assert np.float32 == sel.transform(X.astype(np.float32)).dtype

    # Check 1d list and other dtype:
    names_t_actual = sel.transform([feature_names])
    assert_array_equal(feature_names_t, names_t_actual.ravel())

    # Check wrong shape raises error
    with pytest.raises(ValueError):
        sel.transform(np.array([[1], [2]]))


def test_transform_sparse():
    sparse = sp.csc_matrix
    sel = StepSelector()
    Xt_actual = sel.fit(sparse(X)).transform(sparse(X))
    Xt_actual2 = sel.fit_transform(sparse(X))
    assert_array_equal(Xt, Xt_actual.toarray())
    assert_array_equal(Xt, Xt_actual2.toarray())

    # Check dtype matches
    assert np.int32 == sel.transform(sparse(X).astype(np.int32)).dtype
    assert np.float32 == sel.transform(sparse(X).astype(np.float32)).dtype

    # Check wrong shape raises error
    with pytest.raises(ValueError):
        sel.transform(np.array([[1], [2]]))


def test_inverse_transform_dense():
    sel = StepSelector()
    Xinv_actual = sel.fit(X, y).inverse_transform(Xt)
    assert_array_equal(Xinv, Xinv_actual)

    # Check dtype matches
    assert np.int32 == sel.inverse_transform(Xt.astype(np.int32)).dtype
    assert np.float32 == sel.inverse_transform(Xt.astype(np.float32)).dtype

    # Check 1d list and other dtype:
    names_inv_actual = sel.inverse_transform([feature_names_t])
    assert_array_equal(feature_names_inv, names_inv_actual.ravel())

    # Check wrong shape raises error
    with pytest.raises(ValueError):
        sel.inverse_transform(np.array([[1], [2]]))


def test_inverse_transform_sparse():
    sparse = sp.csc_matrix
    sel = StepSelector()
    Xinv_actual = sel.fit(sparse(X)).inverse_transform(sparse(Xt))
    assert_array_equal(Xinv, Xinv_actual.toarray())

    # Check dtype matches
    assert np.int32 == sel.inverse_transform(sparse(Xt).astype(np.int32)).dtype
    assert np.float32 == sel.inverse_transform(sparse(Xt).astype(np.float32)).dtype

    # Check wrong shape raises error
    with pytest.raises(ValueError):
        sel.inverse_transform(np.array([[1], [2]]))


def test_get_support():
    sel = StepSelector()
    sel.fit(X, y)
    assert_array_equal(support, sel.get_support())
    assert_array_equal(support_inds, sel.get_support(indices=True))
