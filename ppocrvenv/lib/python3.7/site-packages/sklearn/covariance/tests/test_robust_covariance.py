# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#         Gael Varoquaux <gael.varoquaux@normalesup.org>
#         Virgile Fritsch <virgile.fritsch@inria.fr>
#
# License: BSD 3 clause

import itertools

import numpy as np
import pytest

from sklearn.utils._testing import assert_array_almost_equal

from sklearn import datasets
from sklearn.covariance import empirical_covariance, MinCovDet
from sklearn.covariance import fast_mcd

X = datasets.load_iris().data
X_1d = X[:, 0]
n_samples, n_features = X.shape


def test_mcd():
    # Tests the FastMCD algorithm implementation
    # Small data set
    # test without outliers (random independent normal data)
    launch_mcd_on_dataset(100, 5, 0, 0.01, 0.1, 80)
    # test with a contaminated data set (medium contamination)
    launch_mcd_on_dataset(100, 5, 20, 0.01, 0.01, 70)
    # test with a contaminated data set (strong contamination)
    launch_mcd_on_dataset(100, 5, 40, 0.1, 0.1, 50)

    # Medium data set
    launch_mcd_on_dataset(1000, 5, 450, 0.1, 0.1, 540)

    # Large data set
    launch_mcd_on_dataset(1700, 5, 800, 0.1, 0.1, 870)

    # 1D data set
    launch_mcd_on_dataset(500, 1, 100, 0.001, 0.001, 350)


def test_fast_mcd_on_invalid_input():
    X = np.arange(100)
    msg = "Expected 2D array, got 1D array instead"
    with pytest.raises(ValueError, match=msg):
        fast_mcd(X)


def test_mcd_class_on_invalid_input():
    X = np.arange(100)
    mcd = MinCovDet()
    msg = "Expected 2D array, got 1D array instead"
    with pytest.raises(ValueError, match=msg):
        mcd.fit(X)


def launch_mcd_on_dataset(
    n_samples, n_features, n_outliers, tol_loc, tol_cov, tol_support
):

    rand_gen = np.random.RandomState(0)
    data = rand_gen.randn(n_samples, n_features)
    # add some outliers
    outliers_index = rand_gen.permutation(n_samples)[:n_outliers]
    outliers_offset = 10.0 * (rand_gen.randint(2, size=(n_outliers, n_features)) - 0.5)
    data[outliers_index] += outliers_offset
    inliers_mask = np.ones(n_samples).astype(bool)
    inliers_mask[outliers_index] = False

    pure_data = data[inliers_mask]
    # compute MCD by fitting an object
    mcd_fit = MinCovDet(random_state=rand_gen).fit(data)
    T = mcd_fit.location_
    S = mcd_fit.covariance_
    H = mcd_fit.support_
    # compare with the estimates learnt from the inliers
    error_location = np.mean((pure_data.mean(0) - T) ** 2)
    assert error_location < tol_loc
    error_cov = np.mean((empirical_covariance(pure_data) - S) ** 2)
    assert error_cov < tol_cov
    assert np.sum(H) >= tol_support
    assert_array_almost_equal(mcd_fit.mahalanobis(data), mcd_fit.dist_)


def test_mcd_issue1127():
    # Check that the code does not break with X.shape = (3, 1)
    # (i.e. n_support = n_samples)
    rnd = np.random.RandomState(0)
    X = rnd.normal(size=(3, 1))
    mcd = MinCovDet()
    mcd.fit(X)


def test_mcd_issue3367():
    # Check that MCD completes when the covariance matrix is singular
    # i.e. one of the rows and columns are all zeros
    rand_gen = np.random.RandomState(0)

    # Think of these as the values for X and Y -> 10 values between -5 and 5
    data_values = np.linspace(-5, 5, 10).tolist()
    # Get the cartesian product of all possible coordinate pairs from above set
    data = np.array(list(itertools.product(data_values, data_values)))

    # Add a third column that's all zeros to make our data a set of point
    # within a plane, which means that the covariance matrix will be singular
    data = np.hstack((data, np.zeros((data.shape[0], 1))))

    # The below line of code should raise an exception if the covariance matrix
    # is singular. As a further test, since we have points in XYZ, the
    # principle components (Eigenvectors) of these directly relate to the
    # geometry of the points. Since it's a plane, we should be able to test
    # that the Eigenvector that corresponds to the smallest Eigenvalue is the
    # plane normal, specifically [0, 0, 1], since everything is in the XY plane
    # (as I've set it up above). To do this one would start by:
    #
    #     evals, evecs = np.linalg.eigh(mcd_fit.covariance_)
    #     normal = evecs[:, np.argmin(evals)]
    #
    # After which we need to assert that our `normal` is equal to [0, 0, 1].
    # Do note that there is floating point error associated with this, so it's
    # best to subtract the two and then compare some small tolerance (e.g.
    # 1e-12).
    MinCovDet(random_state=rand_gen).fit(data)


def test_mcd_support_covariance_is_zero():
    # Check that MCD returns a ValueError with informative message when the
    # covariance of the support data is equal to 0.
    X_1 = np.array([0.5, 0.1, 0.1, 0.1, 0.957, 0.1, 0.1, 0.1, 0.4285, 0.1])
    X_1 = X_1.reshape(-1, 1)
    X_2 = np.array([0.5, 0.3, 0.3, 0.3, 0.957, 0.3, 0.3, 0.3, 0.4285, 0.3])
    X_2 = X_2.reshape(-1, 1)
    msg = (
        "The covariance matrix of the support data is equal to 0, try to "
        "increase support_fraction"
    )
    for X in [X_1, X_2]:
        with pytest.raises(ValueError, match=msg):
            MinCovDet().fit(X)


def test_mcd_increasing_det_warning():
    # Check that a warning is raised if we observe increasing determinants
    # during the c_step. In theory the sequence of determinants should be
    # decreasing. Increasing determinants are likely due to ill-conditioned
    # covariance matrices that result in poor precision matrices.

    X = [
        [5.1, 3.5, 1.4, 0.2],
        [4.9, 3.0, 1.4, 0.2],
        [4.7, 3.2, 1.3, 0.2],
        [4.6, 3.1, 1.5, 0.2],
        [5.0, 3.6, 1.4, 0.2],
        [4.6, 3.4, 1.4, 0.3],
        [5.0, 3.4, 1.5, 0.2],
        [4.4, 2.9, 1.4, 0.2],
        [4.9, 3.1, 1.5, 0.1],
        [5.4, 3.7, 1.5, 0.2],
        [4.8, 3.4, 1.6, 0.2],
        [4.8, 3.0, 1.4, 0.1],
        [4.3, 3.0, 1.1, 0.1],
        [5.1, 3.5, 1.4, 0.3],
        [5.7, 3.8, 1.7, 0.3],
        [5.4, 3.4, 1.7, 0.2],
        [4.6, 3.6, 1.0, 0.2],
        [5.0, 3.0, 1.6, 0.2],
        [5.2, 3.5, 1.5, 0.2],
    ]

    mcd = MinCovDet(random_state=1)
    warn_msg = "Determinant has increased"
    with pytest.warns(RuntimeWarning, match=warn_msg):
        mcd.fit(X)
