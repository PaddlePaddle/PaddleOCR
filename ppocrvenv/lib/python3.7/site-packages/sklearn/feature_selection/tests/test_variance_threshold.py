import numpy as np
import pytest

from sklearn.utils._testing import assert_array_equal

from scipy.sparse import bsr_matrix, csc_matrix, csr_matrix

from sklearn.feature_selection import VarianceThreshold

data = [[0, 1, 2, 3, 4], [0, 2, 2, 3, 5], [1, 1, 2, 4, 0]]

data2 = [[-0.13725701]] * 10


def test_zero_variance():
    # Test VarianceThreshold with default setting, zero variance.

    for X in [data, csr_matrix(data), csc_matrix(data), bsr_matrix(data)]:
        sel = VarianceThreshold().fit(X)
        assert_array_equal([0, 1, 3, 4], sel.get_support(indices=True))

    with pytest.raises(ValueError):
        VarianceThreshold().fit([[0, 1, 2, 3]])
    with pytest.raises(ValueError):
        VarianceThreshold().fit([[0, 1], [0, 1]])


def test_variance_threshold():
    # Test VarianceThreshold with custom variance.
    for X in [data, csr_matrix(data)]:
        X = VarianceThreshold(threshold=0.4).fit_transform(X)
        assert (len(data), 1) == X.shape


@pytest.mark.parametrize("X", [data, csr_matrix(data)])
def test_variance_negative(X):
    """Test VarianceThreshold with negative variance."""
    var_threshold = VarianceThreshold(threshold=-1.0)
    msg = r"^Threshold must be non-negative. Got: -1.0$"
    with pytest.raises(ValueError, match=msg):
        var_threshold.fit(X)


@pytest.mark.skipif(
    np.var(data2) == 0,
    reason=(
        "This test is not valid for this platform, "
        "as it relies on numerical instabilities."
    ),
)
def test_zero_variance_floating_point_error():
    # Test that VarianceThreshold(0.0).fit eliminates features that have
    # the same value in every sample, even when floating point errors
    # cause np.var not to be 0 for the feature.
    # See #13691

    for X in [data2, csr_matrix(data2), csc_matrix(data2), bsr_matrix(data2)]:
        msg = "No feature in X meets the variance threshold 0.00000"
        with pytest.raises(ValueError, match=msg):
            VarianceThreshold().fit(X)


def test_variance_nan():
    arr = np.array(data, dtype=np.float64)
    # add single NaN and feature should still be included
    arr[0, 0] = np.NaN
    # make all values in feature NaN and feature should be rejected
    arr[:, 1] = np.NaN

    for X in [arr, csr_matrix(arr), csc_matrix(arr), bsr_matrix(arr)]:
        sel = VarianceThreshold().fit(X)
        assert_array_equal([0, 3, 4], sel.get_support(indices=True))
