# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#         Gael Varoquaux <gael.varoquaux@normalesup.org>
#         Virgile Fritsch <virgile.fritsch@inria.fr>
#
# License: BSD 3 clause

import numpy as np
import pytest

from sklearn.utils._testing import assert_almost_equal
from sklearn.utils._testing import assert_array_almost_equal
from sklearn.utils._testing import assert_array_equal

from sklearn import datasets
from sklearn.covariance import (
    empirical_covariance,
    EmpiricalCovariance,
    ShrunkCovariance,
    shrunk_covariance,
    LedoitWolf,
    ledoit_wolf,
    ledoit_wolf_shrinkage,
    OAS,
    oas,
)

X, _ = datasets.load_diabetes(return_X_y=True)
X_1d = X[:, 0]
n_samples, n_features = X.shape


def test_covariance():
    # Tests Covariance module on a simple dataset.
    # test covariance fit from data
    cov = EmpiricalCovariance()
    cov.fit(X)
    emp_cov = empirical_covariance(X)
    assert_array_almost_equal(emp_cov, cov.covariance_, 4)
    assert_almost_equal(cov.error_norm(emp_cov), 0)
    assert_almost_equal(cov.error_norm(emp_cov, norm="spectral"), 0)
    assert_almost_equal(cov.error_norm(emp_cov, norm="frobenius"), 0)
    assert_almost_equal(cov.error_norm(emp_cov, scaling=False), 0)
    assert_almost_equal(cov.error_norm(emp_cov, squared=False), 0)
    with pytest.raises(NotImplementedError):
        cov.error_norm(emp_cov, norm="foo")
    # Mahalanobis distances computation test
    mahal_dist = cov.mahalanobis(X)
    assert np.amin(mahal_dist) > 0

    # test with n_features = 1
    X_1d = X[:, 0].reshape((-1, 1))
    cov = EmpiricalCovariance()
    cov.fit(X_1d)
    assert_array_almost_equal(empirical_covariance(X_1d), cov.covariance_, 4)
    assert_almost_equal(cov.error_norm(empirical_covariance(X_1d)), 0)
    assert_almost_equal(cov.error_norm(empirical_covariance(X_1d), norm="spectral"), 0)

    # test with one sample
    # Create X with 1 sample and 5 features
    X_1sample = np.arange(5).reshape(1, 5)
    cov = EmpiricalCovariance()
    warn_msg = "Only one sample available. You may want to reshape your data array"
    with pytest.warns(UserWarning, match=warn_msg):
        cov.fit(X_1sample)

    assert_array_almost_equal(cov.covariance_, np.zeros(shape=(5, 5), dtype=np.float64))

    # test integer type
    X_integer = np.asarray([[0, 1], [1, 0]])
    result = np.asarray([[0.25, -0.25], [-0.25, 0.25]])
    assert_array_almost_equal(empirical_covariance(X_integer), result)

    # test centered case
    cov = EmpiricalCovariance(assume_centered=True)
    cov.fit(X)
    assert_array_equal(cov.location_, np.zeros(X.shape[1]))


def test_shrunk_covariance():
    # Tests ShrunkCovariance module on a simple dataset.
    # compare shrunk covariance obtained from data and from MLE estimate
    cov = ShrunkCovariance(shrinkage=0.5)
    cov.fit(X)
    assert_array_almost_equal(
        shrunk_covariance(empirical_covariance(X), shrinkage=0.5), cov.covariance_, 4
    )

    # same test with shrinkage not provided
    cov = ShrunkCovariance()
    cov.fit(X)
    assert_array_almost_equal(
        shrunk_covariance(empirical_covariance(X)), cov.covariance_, 4
    )

    # same test with shrinkage = 0 (<==> empirical_covariance)
    cov = ShrunkCovariance(shrinkage=0.0)
    cov.fit(X)
    assert_array_almost_equal(empirical_covariance(X), cov.covariance_, 4)

    # test with n_features = 1
    X_1d = X[:, 0].reshape((-1, 1))
    cov = ShrunkCovariance(shrinkage=0.3)
    cov.fit(X_1d)
    assert_array_almost_equal(empirical_covariance(X_1d), cov.covariance_, 4)

    # test shrinkage coeff on a simple data set (without saving precision)
    cov = ShrunkCovariance(shrinkage=0.5, store_precision=False)
    cov.fit(X)
    assert cov.precision_ is None


def test_ledoit_wolf():
    # Tests LedoitWolf module on a simple dataset.
    # test shrinkage coeff on a simple data set
    X_centered = X - X.mean(axis=0)
    lw = LedoitWolf(assume_centered=True)
    lw.fit(X_centered)
    shrinkage_ = lw.shrinkage_

    score_ = lw.score(X_centered)
    assert_almost_equal(
        ledoit_wolf_shrinkage(X_centered, assume_centered=True), shrinkage_
    )
    assert_almost_equal(
        ledoit_wolf_shrinkage(X_centered, assume_centered=True, block_size=6),
        shrinkage_,
    )
    # compare shrunk covariance obtained from data and from MLE estimate
    lw_cov_from_mle, lw_shrinkage_from_mle = ledoit_wolf(
        X_centered, assume_centered=True
    )
    assert_array_almost_equal(lw_cov_from_mle, lw.covariance_, 4)
    assert_almost_equal(lw_shrinkage_from_mle, lw.shrinkage_)
    # compare estimates given by LW and ShrunkCovariance
    scov = ShrunkCovariance(shrinkage=lw.shrinkage_, assume_centered=True)
    scov.fit(X_centered)
    assert_array_almost_equal(scov.covariance_, lw.covariance_, 4)

    # test with n_features = 1
    X_1d = X[:, 0].reshape((-1, 1))
    lw = LedoitWolf(assume_centered=True)
    lw.fit(X_1d)
    lw_cov_from_mle, lw_shrinkage_from_mle = ledoit_wolf(X_1d, assume_centered=True)
    assert_array_almost_equal(lw_cov_from_mle, lw.covariance_, 4)
    assert_almost_equal(lw_shrinkage_from_mle, lw.shrinkage_)
    assert_array_almost_equal((X_1d ** 2).sum() / n_samples, lw.covariance_, 4)

    # test shrinkage coeff on a simple data set (without saving precision)
    lw = LedoitWolf(store_precision=False, assume_centered=True)
    lw.fit(X_centered)
    assert_almost_equal(lw.score(X_centered), score_, 4)
    assert lw.precision_ is None

    # Same tests without assuming centered data
    # test shrinkage coeff on a simple data set
    lw = LedoitWolf()
    lw.fit(X)
    assert_almost_equal(lw.shrinkage_, shrinkage_, 4)
    assert_almost_equal(lw.shrinkage_, ledoit_wolf_shrinkage(X))
    assert_almost_equal(lw.shrinkage_, ledoit_wolf(X)[1])
    assert_almost_equal(lw.score(X), score_, 4)
    # compare shrunk covariance obtained from data and from MLE estimate
    lw_cov_from_mle, lw_shrinkage_from_mle = ledoit_wolf(X)
    assert_array_almost_equal(lw_cov_from_mle, lw.covariance_, 4)
    assert_almost_equal(lw_shrinkage_from_mle, lw.shrinkage_)
    # compare estimates given by LW and ShrunkCovariance
    scov = ShrunkCovariance(shrinkage=lw.shrinkage_)
    scov.fit(X)
    assert_array_almost_equal(scov.covariance_, lw.covariance_, 4)

    # test with n_features = 1
    X_1d = X[:, 0].reshape((-1, 1))
    lw = LedoitWolf()
    lw.fit(X_1d)
    lw_cov_from_mle, lw_shrinkage_from_mle = ledoit_wolf(X_1d)
    assert_array_almost_equal(lw_cov_from_mle, lw.covariance_, 4)
    assert_almost_equal(lw_shrinkage_from_mle, lw.shrinkage_)
    assert_array_almost_equal(empirical_covariance(X_1d), lw.covariance_, 4)

    # test with one sample
    # warning should be raised when using only 1 sample
    X_1sample = np.arange(5).reshape(1, 5)
    lw = LedoitWolf()

    warn_msg = "Only one sample available. You may want to reshape your data array"
    with pytest.warns(UserWarning, match=warn_msg):
        lw.fit(X_1sample)

    assert_array_almost_equal(lw.covariance_, np.zeros(shape=(5, 5), dtype=np.float64))

    # test shrinkage coeff on a simple data set (without saving precision)
    lw = LedoitWolf(store_precision=False)
    lw.fit(X)
    assert_almost_equal(lw.score(X), score_, 4)
    assert lw.precision_ is None


def _naive_ledoit_wolf_shrinkage(X):
    # A simple implementation of the formulas from Ledoit & Wolf

    # The computation below achieves the following computations of the
    # "O. Ledoit and M. Wolf, A Well-Conditioned Estimator for
    # Large-Dimensional Covariance Matrices"
    # beta and delta are given in the beginning of section 3.2
    n_samples, n_features = X.shape
    emp_cov = empirical_covariance(X, assume_centered=False)
    mu = np.trace(emp_cov) / n_features
    delta_ = emp_cov.copy()
    delta_.flat[:: n_features + 1] -= mu
    delta = (delta_ ** 2).sum() / n_features
    X2 = X ** 2
    beta_ = (
        1.0
        / (n_features * n_samples)
        * np.sum(np.dot(X2.T, X2) / n_samples - emp_cov ** 2)
    )

    beta = min(beta_, delta)
    shrinkage = beta / delta
    return shrinkage


def test_ledoit_wolf_small():
    # Compare our blocked implementation to the naive implementation
    X_small = X[:, :4]
    lw = LedoitWolf()
    lw.fit(X_small)
    shrinkage_ = lw.shrinkage_

    assert_almost_equal(shrinkage_, _naive_ledoit_wolf_shrinkage(X_small))


def test_ledoit_wolf_large():
    # test that ledoit_wolf doesn't error on data that is wider than block_size
    rng = np.random.RandomState(0)
    # use a number of features that is larger than the block-size
    X = rng.normal(size=(10, 20))
    lw = LedoitWolf(block_size=10).fit(X)
    # check that covariance is about diagonal (random normal noise)
    assert_almost_equal(lw.covariance_, np.eye(20), 0)
    cov = lw.covariance_

    # check that the result is consistent with not splitting data into blocks.
    lw = LedoitWolf(block_size=25).fit(X)
    assert_almost_equal(lw.covariance_, cov)


@pytest.mark.parametrize(
    "ledoit_wolf_fitting_function", [LedoitWolf().fit, ledoit_wolf_shrinkage]
)
def test_ledoit_wolf_empty_array(ledoit_wolf_fitting_function):
    """Check that we validate X and raise proper error with 0-sample array."""
    X_empty = np.zeros((0, 2))
    with pytest.raises(ValueError, match="Found array with 0 sample"):
        ledoit_wolf_fitting_function(X_empty)


def test_oas():
    # Tests OAS module on a simple dataset.
    # test shrinkage coeff on a simple data set
    X_centered = X - X.mean(axis=0)
    oa = OAS(assume_centered=True)
    oa.fit(X_centered)
    shrinkage_ = oa.shrinkage_
    score_ = oa.score(X_centered)
    # compare shrunk covariance obtained from data and from MLE estimate
    oa_cov_from_mle, oa_shrinkage_from_mle = oas(X_centered, assume_centered=True)
    assert_array_almost_equal(oa_cov_from_mle, oa.covariance_, 4)
    assert_almost_equal(oa_shrinkage_from_mle, oa.shrinkage_)
    # compare estimates given by OAS and ShrunkCovariance
    scov = ShrunkCovariance(shrinkage=oa.shrinkage_, assume_centered=True)
    scov.fit(X_centered)
    assert_array_almost_equal(scov.covariance_, oa.covariance_, 4)

    # test with n_features = 1
    X_1d = X[:, 0:1]
    oa = OAS(assume_centered=True)
    oa.fit(X_1d)
    oa_cov_from_mle, oa_shrinkage_from_mle = oas(X_1d, assume_centered=True)
    assert_array_almost_equal(oa_cov_from_mle, oa.covariance_, 4)
    assert_almost_equal(oa_shrinkage_from_mle, oa.shrinkage_)
    assert_array_almost_equal((X_1d ** 2).sum() / n_samples, oa.covariance_, 4)

    # test shrinkage coeff on a simple data set (without saving precision)
    oa = OAS(store_precision=False, assume_centered=True)
    oa.fit(X_centered)
    assert_almost_equal(oa.score(X_centered), score_, 4)
    assert oa.precision_ is None

    # Same tests without assuming centered data--------------------------------
    # test shrinkage coeff on a simple data set
    oa = OAS()
    oa.fit(X)
    assert_almost_equal(oa.shrinkage_, shrinkage_, 4)
    assert_almost_equal(oa.score(X), score_, 4)
    # compare shrunk covariance obtained from data and from MLE estimate
    oa_cov_from_mle, oa_shrinkage_from_mle = oas(X)
    assert_array_almost_equal(oa_cov_from_mle, oa.covariance_, 4)
    assert_almost_equal(oa_shrinkage_from_mle, oa.shrinkage_)
    # compare estimates given by OAS and ShrunkCovariance
    scov = ShrunkCovariance(shrinkage=oa.shrinkage_)
    scov.fit(X)
    assert_array_almost_equal(scov.covariance_, oa.covariance_, 4)

    # test with n_features = 1
    X_1d = X[:, 0].reshape((-1, 1))
    oa = OAS()
    oa.fit(X_1d)
    oa_cov_from_mle, oa_shrinkage_from_mle = oas(X_1d)
    assert_array_almost_equal(oa_cov_from_mle, oa.covariance_, 4)
    assert_almost_equal(oa_shrinkage_from_mle, oa.shrinkage_)
    assert_array_almost_equal(empirical_covariance(X_1d), oa.covariance_, 4)

    # test with one sample
    # warning should be raised when using only 1 sample
    X_1sample = np.arange(5).reshape(1, 5)
    oa = OAS()
    warn_msg = "Only one sample available. You may want to reshape your data array"
    with pytest.warns(UserWarning, match=warn_msg):
        oa.fit(X_1sample)

    assert_array_almost_equal(oa.covariance_, np.zeros(shape=(5, 5), dtype=np.float64))

    # test shrinkage coeff on a simple data set (without saving precision)
    oa = OAS(store_precision=False)
    oa.fit(X)
    assert_almost_equal(oa.score(X), score_, 4)
    assert oa.precision_ is None


def test_EmpiricalCovariance_validates_mahalanobis():
    """Checks that EmpiricalCovariance validates data with mahalanobis."""
    cov = EmpiricalCovariance().fit(X)

    msg = f"X has 2 features, but \\w+ is expecting {X.shape[1]} features as input"
    with pytest.raises(ValueError, match=msg):
        cov.mahalanobis(X[:, :2])
