import numpy as np
import pytest
from scipy.sparse import csr_matrix

from sklearn.utils import check_random_state
from sklearn.utils._testing import assert_array_equal, assert_almost_equal
from sklearn.feature_selection._mutual_info import _compute_mi
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif


def test_compute_mi_dd():
    # In discrete case computations are straightforward and can be done
    # by hand on given vectors.
    x = np.array([0, 1, 1, 0, 0])
    y = np.array([1, 0, 0, 0, 1])

    H_x = H_y = -(3 / 5) * np.log(3 / 5) - (2 / 5) * np.log(2 / 5)
    H_xy = -1 / 5 * np.log(1 / 5) - 2 / 5 * np.log(2 / 5) - 2 / 5 * np.log(2 / 5)
    I_xy = H_x + H_y - H_xy

    assert_almost_equal(_compute_mi(x, y, True, True), I_xy)


def test_compute_mi_cc():
    # For two continuous variables a good approach is to test on bivariate
    # normal distribution, where mutual information is known.

    # Mean of the distribution, irrelevant for mutual information.
    mean = np.zeros(2)

    # Setup covariance matrix with correlation coeff. equal 0.5.
    sigma_1 = 1
    sigma_2 = 10
    corr = 0.5
    cov = np.array(
        [
            [sigma_1 ** 2, corr * sigma_1 * sigma_2],
            [corr * sigma_1 * sigma_2, sigma_2 ** 2],
        ]
    )

    # True theoretical mutual information.
    I_theory = np.log(sigma_1) + np.log(sigma_2) - 0.5 * np.log(np.linalg.det(cov))

    rng = check_random_state(0)
    Z = rng.multivariate_normal(mean, cov, size=1000)

    x, y = Z[:, 0], Z[:, 1]

    # Theory and computed values won't be very close, assert that the
    # first figures after decimal point match.
    for n_neighbors in [3, 5, 7]:
        I_computed = _compute_mi(x, y, False, False, n_neighbors)
        assert_almost_equal(I_computed, I_theory, 1)


def test_compute_mi_cd():
    # To test define a joint distribution as follows:
    # p(x, y) = p(x) p(y | x)
    # X ~ Bernoulli(p)
    # (Y | x = 0) ~ Uniform(-1, 1)
    # (Y | x = 1) ~ Uniform(0, 2)

    # Use the following formula for mutual information:
    # I(X; Y) = H(Y) - H(Y | X)
    # Two entropies can be computed by hand:
    # H(Y) = -(1-p)/2 * ln((1-p)/2) - p/2*log(p/2) - 1/2*log(1/2)
    # H(Y | X) = ln(2)

    # Now we need to implement sampling from out distribution, which is
    # done easily using conditional distribution logic.

    n_samples = 1000
    rng = check_random_state(0)

    for p in [0.3, 0.5, 0.7]:
        x = rng.uniform(size=n_samples) > p

        y = np.empty(n_samples)
        mask = x == 0
        y[mask] = rng.uniform(-1, 1, size=np.sum(mask))
        y[~mask] = rng.uniform(0, 2, size=np.sum(~mask))

        I_theory = -0.5 * (
            (1 - p) * np.log(0.5 * (1 - p)) + p * np.log(0.5 * p) + np.log(0.5)
        ) - np.log(2)

        # Assert the same tolerance.
        for n_neighbors in [3, 5, 7]:
            I_computed = _compute_mi(x, y, True, False, n_neighbors)
            assert_almost_equal(I_computed, I_theory, 1)


def test_compute_mi_cd_unique_label():
    # Test that adding unique label doesn't change MI.
    n_samples = 100
    x = np.random.uniform(size=n_samples) > 0.5

    y = np.empty(n_samples)
    mask = x == 0
    y[mask] = np.random.uniform(-1, 1, size=np.sum(mask))
    y[~mask] = np.random.uniform(0, 2, size=np.sum(~mask))

    mi_1 = _compute_mi(x, y, True, False)

    x = np.hstack((x, 2))
    y = np.hstack((y, 10))
    mi_2 = _compute_mi(x, y, True, False)

    assert mi_1 == mi_2


# We are going test that feature ordering by MI matches our expectations.
def test_mutual_info_classif_discrete():
    X = np.array([[0, 0, 0], [1, 1, 0], [2, 0, 1], [2, 0, 1], [2, 0, 1]])
    y = np.array([0, 1, 2, 2, 1])

    # Here X[:, 0] is the most informative feature, and X[:, 1] is weakly
    # informative.
    mi = mutual_info_classif(X, y, discrete_features=True)
    assert_array_equal(np.argsort(-mi), np.array([0, 2, 1]))


def test_mutual_info_regression():
    # We generate sample from multivariate normal distribution, using
    # transformation from initially uncorrelated variables. The zero
    # variables after transformation is selected as the target vector,
    # it has the strongest correlation with the variable 2, and
    # the weakest correlation with the variable 1.
    T = np.array([[1, 0.5, 2, 1], [0, 1, 0.1, 0.0], [0, 0.1, 1, 0.1], [0, 0.1, 0.1, 1]])
    cov = T.dot(T.T)
    mean = np.zeros(4)

    rng = check_random_state(0)
    Z = rng.multivariate_normal(mean, cov, size=1000)
    X = Z[:, 1:]
    y = Z[:, 0]

    mi = mutual_info_regression(X, y, random_state=0)
    assert_array_equal(np.argsort(-mi), np.array([1, 2, 0]))


def test_mutual_info_classif_mixed():
    # Here the target is discrete and there are two continuous and one
    # discrete feature. The idea of this test is clear from the code.
    rng = check_random_state(0)
    X = rng.rand(1000, 3)
    X[:, 1] += X[:, 0]
    y = ((0.5 * X[:, 0] + X[:, 2]) > 0.5).astype(int)
    X[:, 2] = X[:, 2] > 0.5

    mi = mutual_info_classif(X, y, discrete_features=[2], n_neighbors=3, random_state=0)
    assert_array_equal(np.argsort(-mi), [2, 0, 1])
    for n_neighbors in [5, 7, 9]:
        mi_nn = mutual_info_classif(
            X, y, discrete_features=[2], n_neighbors=n_neighbors, random_state=0
        )
        # Check that the continuous values have an higher MI with greater
        # n_neighbors
        assert mi_nn[0] > mi[0]
        assert mi_nn[1] > mi[1]
        # The n_neighbors should not have any effect on the discrete value
        # The MI should be the same
        assert mi_nn[2] == mi[2]


def test_mutual_info_options():
    X = np.array([[0, 0, 0], [1, 1, 0], [2, 0, 1], [2, 0, 1], [2, 0, 1]], dtype=float)
    y = np.array([0, 1, 2, 2, 1], dtype=float)
    X_csr = csr_matrix(X)

    for mutual_info in (mutual_info_regression, mutual_info_classif):
        with pytest.raises(ValueError):
            mutual_info(X_csr, y, discrete_features=False)
        with pytest.raises(ValueError):
            mutual_info(X, y, discrete_features="manual")
        with pytest.raises(ValueError):
            mutual_info(X_csr, y, discrete_features=[True, False, True])
        with pytest.raises(IndexError):
            mutual_info(X, y, discrete_features=[True, False, True, False])
        with pytest.raises(IndexError):
            mutual_info(X, y, discrete_features=[1, 4])

        mi_1 = mutual_info(X, y, discrete_features="auto", random_state=0)
        mi_2 = mutual_info(X, y, discrete_features=False, random_state=0)
        mi_3 = mutual_info(X_csr, y, discrete_features="auto", random_state=0)
        mi_4 = mutual_info(X_csr, y, discrete_features=True, random_state=0)
        mi_5 = mutual_info(X, y, discrete_features=[True, False, True], random_state=0)
        mi_6 = mutual_info(X, y, discrete_features=[0, 2], random_state=0)

        assert_array_equal(mi_1, mi_2)
        assert_array_equal(mi_3, mi_4)
        assert_array_equal(mi_5, mi_6)

    assert not np.allclose(mi_1, mi_3)
