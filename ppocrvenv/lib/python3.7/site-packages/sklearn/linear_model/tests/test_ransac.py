import numpy as np
import pytest
from scipy import sparse

from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_array_equal

from sklearn.utils import check_random_state
from sklearn.utils._testing import assert_allclose
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression, RANSACRegressor, Ridge
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model._ransac import _dynamic_max_trials
from sklearn.exceptions import ConvergenceWarning


# Generate coordinates of line
X = np.arange(-200, 200)
y = 0.2 * X + 20
data = np.column_stack([X, y])

# Add some faulty data
rng = np.random.RandomState(1000)
outliers = np.unique(rng.randint(len(X), size=200))
data[outliers, :] += 50 + rng.rand(len(outliers), 2) * 10

X = data[:, 0][:, np.newaxis]
y = data[:, 1]


def test_ransac_inliers_outliers():

    base_estimator = LinearRegression()
    ransac_estimator = RANSACRegressor(
        base_estimator, min_samples=2, residual_threshold=5, random_state=0
    )

    # Estimate parameters of corrupted data
    ransac_estimator.fit(X, y)

    # Ground truth / reference inlier mask
    ref_inlier_mask = np.ones_like(ransac_estimator.inlier_mask_).astype(np.bool_)
    ref_inlier_mask[outliers] = False

    assert_array_equal(ransac_estimator.inlier_mask_, ref_inlier_mask)


def test_ransac_is_data_valid():
    def is_data_valid(X, y):
        assert X.shape[0] == 2
        assert y.shape[0] == 2
        return False

    rng = np.random.RandomState(0)
    X = rng.rand(10, 2)
    y = rng.rand(10, 1)

    base_estimator = LinearRegression()
    ransac_estimator = RANSACRegressor(
        base_estimator,
        min_samples=2,
        residual_threshold=5,
        is_data_valid=is_data_valid,
        random_state=0,
    )
    with pytest.raises(ValueError):
        ransac_estimator.fit(X, y)


def test_ransac_is_model_valid():
    def is_model_valid(estimator, X, y):
        assert X.shape[0] == 2
        assert y.shape[0] == 2
        return False

    base_estimator = LinearRegression()
    ransac_estimator = RANSACRegressor(
        base_estimator,
        min_samples=2,
        residual_threshold=5,
        is_model_valid=is_model_valid,
        random_state=0,
    )
    with pytest.raises(ValueError):
        ransac_estimator.fit(X, y)


def test_ransac_max_trials():
    base_estimator = LinearRegression()

    ransac_estimator = RANSACRegressor(
        base_estimator,
        min_samples=2,
        residual_threshold=5,
        max_trials=0,
        random_state=0,
    )
    with pytest.raises(ValueError):
        ransac_estimator.fit(X, y)

    # there is a 1e-9 chance it will take these many trials. No good reason
    # 1e-2 isn't enough, can still happen
    # 2 is the what ransac defines  as min_samples = X.shape[1] + 1
    max_trials = _dynamic_max_trials(len(X) - len(outliers), X.shape[0], 2, 1 - 1e-9)
    ransac_estimator = RANSACRegressor(base_estimator, min_samples=2)
    for i in range(50):
        ransac_estimator.set_params(min_samples=2, random_state=i)
        ransac_estimator.fit(X, y)
        assert ransac_estimator.n_trials_ < max_trials + 1


def test_ransac_stop_n_inliers():
    base_estimator = LinearRegression()
    ransac_estimator = RANSACRegressor(
        base_estimator,
        min_samples=2,
        residual_threshold=5,
        stop_n_inliers=2,
        random_state=0,
    )
    ransac_estimator.fit(X, y)

    assert ransac_estimator.n_trials_ == 1


def test_ransac_stop_score():
    base_estimator = LinearRegression()
    ransac_estimator = RANSACRegressor(
        base_estimator,
        min_samples=2,
        residual_threshold=5,
        stop_score=0,
        random_state=0,
    )
    ransac_estimator.fit(X, y)

    assert ransac_estimator.n_trials_ == 1


def test_ransac_score():
    X = np.arange(100)[:, None]
    y = np.zeros((100,))
    y[0] = 1
    y[1] = 100

    base_estimator = LinearRegression()
    ransac_estimator = RANSACRegressor(
        base_estimator, min_samples=2, residual_threshold=0.5, random_state=0
    )
    ransac_estimator.fit(X, y)

    assert ransac_estimator.score(X[2:], y[2:]) == 1
    assert ransac_estimator.score(X[:2], y[:2]) < 1


def test_ransac_predict():
    X = np.arange(100)[:, None]
    y = np.zeros((100,))
    y[0] = 1
    y[1] = 100

    base_estimator = LinearRegression()
    ransac_estimator = RANSACRegressor(
        base_estimator, min_samples=2, residual_threshold=0.5, random_state=0
    )
    ransac_estimator.fit(X, y)

    assert_array_equal(ransac_estimator.predict(X), np.zeros(100))


def test_ransac_residuals_threshold_no_inliers():
    # When residual_threshold=nan there are no inliers and a
    # ValueError with a message should be raised
    base_estimator = LinearRegression()
    ransac_estimator = RANSACRegressor(
        base_estimator,
        min_samples=2,
        residual_threshold=float("nan"),
        random_state=0,
        max_trials=5,
    )

    msg = "RANSAC could not find a valid consensus set"
    with pytest.raises(ValueError, match=msg):
        ransac_estimator.fit(X, y)
    assert ransac_estimator.n_skips_no_inliers_ == 5
    assert ransac_estimator.n_skips_invalid_data_ == 0
    assert ransac_estimator.n_skips_invalid_model_ == 0


def test_ransac_no_valid_data():
    def is_data_valid(X, y):
        return False

    base_estimator = LinearRegression()
    ransac_estimator = RANSACRegressor(
        base_estimator, is_data_valid=is_data_valid, max_trials=5
    )

    msg = "RANSAC could not find a valid consensus set"
    with pytest.raises(ValueError, match=msg):
        ransac_estimator.fit(X, y)
    assert ransac_estimator.n_skips_no_inliers_ == 0
    assert ransac_estimator.n_skips_invalid_data_ == 5
    assert ransac_estimator.n_skips_invalid_model_ == 0


def test_ransac_no_valid_model():
    def is_model_valid(estimator, X, y):
        return False

    base_estimator = LinearRegression()
    ransac_estimator = RANSACRegressor(
        base_estimator, is_model_valid=is_model_valid, max_trials=5
    )

    msg = "RANSAC could not find a valid consensus set"
    with pytest.raises(ValueError, match=msg):
        ransac_estimator.fit(X, y)
    assert ransac_estimator.n_skips_no_inliers_ == 0
    assert ransac_estimator.n_skips_invalid_data_ == 0
    assert ransac_estimator.n_skips_invalid_model_ == 5


def test_ransac_exceed_max_skips():
    def is_data_valid(X, y):
        return False

    base_estimator = LinearRegression()
    ransac_estimator = RANSACRegressor(
        base_estimator, is_data_valid=is_data_valid, max_trials=5, max_skips=3
    )

    msg = "RANSAC skipped more iterations than `max_skips`"
    with pytest.raises(ValueError, match=msg):
        ransac_estimator.fit(X, y)
    assert ransac_estimator.n_skips_no_inliers_ == 0
    assert ransac_estimator.n_skips_invalid_data_ == 4
    assert ransac_estimator.n_skips_invalid_model_ == 0


def test_ransac_warn_exceed_max_skips():
    global cause_skip
    cause_skip = False

    def is_data_valid(X, y):
        global cause_skip
        if not cause_skip:
            cause_skip = True
            return True
        else:
            return False

    base_estimator = LinearRegression()
    ransac_estimator = RANSACRegressor(
        base_estimator, is_data_valid=is_data_valid, max_skips=3, max_trials=5
    )
    warning_message = (
        "RANSAC found a valid consensus set but exited "
        "early due to skipping more iterations than "
        "`max_skips`. See estimator attributes for "
        "diagnostics."
    )
    with pytest.warns(ConvergenceWarning, match=warning_message):
        ransac_estimator.fit(X, y)
    assert ransac_estimator.n_skips_no_inliers_ == 0
    assert ransac_estimator.n_skips_invalid_data_ == 4
    assert ransac_estimator.n_skips_invalid_model_ == 0


def test_ransac_sparse_coo():
    X_sparse = sparse.coo_matrix(X)

    base_estimator = LinearRegression()
    ransac_estimator = RANSACRegressor(
        base_estimator, min_samples=2, residual_threshold=5, random_state=0
    )
    ransac_estimator.fit(X_sparse, y)

    ref_inlier_mask = np.ones_like(ransac_estimator.inlier_mask_).astype(np.bool_)
    ref_inlier_mask[outliers] = False

    assert_array_equal(ransac_estimator.inlier_mask_, ref_inlier_mask)


def test_ransac_sparse_csr():
    X_sparse = sparse.csr_matrix(X)

    base_estimator = LinearRegression()
    ransac_estimator = RANSACRegressor(
        base_estimator, min_samples=2, residual_threshold=5, random_state=0
    )
    ransac_estimator.fit(X_sparse, y)

    ref_inlier_mask = np.ones_like(ransac_estimator.inlier_mask_).astype(np.bool_)
    ref_inlier_mask[outliers] = False

    assert_array_equal(ransac_estimator.inlier_mask_, ref_inlier_mask)


def test_ransac_sparse_csc():
    X_sparse = sparse.csc_matrix(X)

    base_estimator = LinearRegression()
    ransac_estimator = RANSACRegressor(
        base_estimator, min_samples=2, residual_threshold=5, random_state=0
    )
    ransac_estimator.fit(X_sparse, y)

    ref_inlier_mask = np.ones_like(ransac_estimator.inlier_mask_).astype(np.bool_)
    ref_inlier_mask[outliers] = False

    assert_array_equal(ransac_estimator.inlier_mask_, ref_inlier_mask)


def test_ransac_none_estimator():

    base_estimator = LinearRegression()

    ransac_estimator = RANSACRegressor(
        base_estimator, min_samples=2, residual_threshold=5, random_state=0
    )
    ransac_none_estimator = RANSACRegressor(
        None, min_samples=2, residual_threshold=5, random_state=0
    )

    ransac_estimator.fit(X, y)
    ransac_none_estimator.fit(X, y)

    assert_array_almost_equal(
        ransac_estimator.predict(X), ransac_none_estimator.predict(X)
    )


def test_ransac_min_n_samples():
    base_estimator = LinearRegression()
    ransac_estimator1 = RANSACRegressor(
        base_estimator, min_samples=2, residual_threshold=5, random_state=0
    )
    ransac_estimator2 = RANSACRegressor(
        base_estimator,
        min_samples=2.0 / X.shape[0],
        residual_threshold=5,
        random_state=0,
    )
    ransac_estimator3 = RANSACRegressor(
        base_estimator, min_samples=-1, residual_threshold=5, random_state=0
    )
    ransac_estimator4 = RANSACRegressor(
        base_estimator, min_samples=5.2, residual_threshold=5, random_state=0
    )
    ransac_estimator5 = RANSACRegressor(
        base_estimator, min_samples=2.0, residual_threshold=5, random_state=0
    )
    ransac_estimator6 = RANSACRegressor(
        base_estimator, residual_threshold=5, random_state=0
    )
    ransac_estimator7 = RANSACRegressor(
        base_estimator, min_samples=X.shape[0] + 1, residual_threshold=5, random_state=0
    )
    # GH #19390
    ransac_estimator8 = RANSACRegressor(
        Ridge(), min_samples=None, residual_threshold=5, random_state=0
    )

    ransac_estimator1.fit(X, y)
    ransac_estimator2.fit(X, y)
    ransac_estimator5.fit(X, y)
    ransac_estimator6.fit(X, y)

    assert_array_almost_equal(
        ransac_estimator1.predict(X), ransac_estimator2.predict(X)
    )
    assert_array_almost_equal(
        ransac_estimator1.predict(X), ransac_estimator5.predict(X)
    )
    assert_array_almost_equal(
        ransac_estimator1.predict(X), ransac_estimator6.predict(X)
    )

    with pytest.raises(ValueError):
        ransac_estimator3.fit(X, y)

    with pytest.raises(ValueError):
        ransac_estimator4.fit(X, y)

    with pytest.raises(ValueError):
        ransac_estimator7.fit(X, y)

    err_msg = "From version 1.2, `min_samples` needs to be explicitly set"
    with pytest.warns(FutureWarning, match=err_msg):
        ransac_estimator8.fit(X, y)


def test_ransac_multi_dimensional_targets():

    base_estimator = LinearRegression()
    ransac_estimator = RANSACRegressor(
        base_estimator, min_samples=2, residual_threshold=5, random_state=0
    )

    # 3-D target values
    yyy = np.column_stack([y, y, y])

    # Estimate parameters of corrupted data
    ransac_estimator.fit(X, yyy)

    # Ground truth / reference inlier mask
    ref_inlier_mask = np.ones_like(ransac_estimator.inlier_mask_).astype(np.bool_)
    ref_inlier_mask[outliers] = False

    assert_array_equal(ransac_estimator.inlier_mask_, ref_inlier_mask)


def test_ransac_residual_loss():
    def loss_multi1(y_true, y_pred):
        return np.sum(np.abs(y_true - y_pred), axis=1)

    def loss_multi2(y_true, y_pred):
        return np.sum((y_true - y_pred) ** 2, axis=1)

    def loss_mono(y_true, y_pred):
        return np.abs(y_true - y_pred)

    yyy = np.column_stack([y, y, y])

    base_estimator = LinearRegression()
    ransac_estimator0 = RANSACRegressor(
        base_estimator, min_samples=2, residual_threshold=5, random_state=0
    )
    ransac_estimator1 = RANSACRegressor(
        base_estimator,
        min_samples=2,
        residual_threshold=5,
        random_state=0,
        loss=loss_multi1,
    )
    ransac_estimator2 = RANSACRegressor(
        base_estimator,
        min_samples=2,
        residual_threshold=5,
        random_state=0,
        loss=loss_multi2,
    )

    # multi-dimensional
    ransac_estimator0.fit(X, yyy)
    ransac_estimator1.fit(X, yyy)
    ransac_estimator2.fit(X, yyy)
    assert_array_almost_equal(
        ransac_estimator0.predict(X), ransac_estimator1.predict(X)
    )
    assert_array_almost_equal(
        ransac_estimator0.predict(X), ransac_estimator2.predict(X)
    )

    # one-dimensional
    ransac_estimator0.fit(X, y)
    ransac_estimator2.loss = loss_mono
    ransac_estimator2.fit(X, y)
    assert_array_almost_equal(
        ransac_estimator0.predict(X), ransac_estimator2.predict(X)
    )
    ransac_estimator3 = RANSACRegressor(
        base_estimator,
        min_samples=2,
        residual_threshold=5,
        random_state=0,
        loss="squared_error",
    )
    ransac_estimator3.fit(X, y)
    assert_array_almost_equal(
        ransac_estimator0.predict(X), ransac_estimator2.predict(X)
    )


def test_ransac_default_residual_threshold():
    base_estimator = LinearRegression()
    ransac_estimator = RANSACRegressor(base_estimator, min_samples=2, random_state=0)

    # Estimate parameters of corrupted data
    ransac_estimator.fit(X, y)

    # Ground truth / reference inlier mask
    ref_inlier_mask = np.ones_like(ransac_estimator.inlier_mask_).astype(np.bool_)
    ref_inlier_mask[outliers] = False

    assert_array_equal(ransac_estimator.inlier_mask_, ref_inlier_mask)


def test_ransac_dynamic_max_trials():
    # Numbers hand-calculated and confirmed on page 119 (Table 4.3) in
    #   Hartley, R.~I. and Zisserman, A., 2004,
    #   Multiple View Geometry in Computer Vision, Second Edition,
    #   Cambridge University Press, ISBN: 0521540518

    # e = 0%, min_samples = X
    assert _dynamic_max_trials(100, 100, 2, 0.99) == 1

    # e = 5%, min_samples = 2
    assert _dynamic_max_trials(95, 100, 2, 0.99) == 2
    # e = 10%, min_samples = 2
    assert _dynamic_max_trials(90, 100, 2, 0.99) == 3
    # e = 30%, min_samples = 2
    assert _dynamic_max_trials(70, 100, 2, 0.99) == 7
    # e = 50%, min_samples = 2
    assert _dynamic_max_trials(50, 100, 2, 0.99) == 17

    # e = 5%, min_samples = 8
    assert _dynamic_max_trials(95, 100, 8, 0.99) == 5
    # e = 10%, min_samples = 8
    assert _dynamic_max_trials(90, 100, 8, 0.99) == 9
    # e = 30%, min_samples = 8
    assert _dynamic_max_trials(70, 100, 8, 0.99) == 78
    # e = 50%, min_samples = 8
    assert _dynamic_max_trials(50, 100, 8, 0.99) == 1177

    # e = 0%, min_samples = 10
    assert _dynamic_max_trials(1, 100, 10, 0) == 0
    assert _dynamic_max_trials(1, 100, 10, 1) == float("inf")

    base_estimator = LinearRegression()
    ransac_estimator = RANSACRegressor(
        base_estimator, min_samples=2, stop_probability=-0.1
    )

    with pytest.raises(ValueError):
        ransac_estimator.fit(X, y)

    ransac_estimator = RANSACRegressor(
        base_estimator, min_samples=2, stop_probability=1.1
    )
    with pytest.raises(ValueError):
        ransac_estimator.fit(X, y)


def test_ransac_fit_sample_weight():
    ransac_estimator = RANSACRegressor(random_state=0)
    n_samples = y.shape[0]
    weights = np.ones(n_samples)
    ransac_estimator.fit(X, y, weights)
    # sanity check
    assert ransac_estimator.inlier_mask_.shape[0] == n_samples

    ref_inlier_mask = np.ones_like(ransac_estimator.inlier_mask_).astype(np.bool_)
    ref_inlier_mask[outliers] = False
    # check that mask is correct
    assert_array_equal(ransac_estimator.inlier_mask_, ref_inlier_mask)

    # check that fit(X)  = fit([X1, X2, X3],sample_weight = [n1, n2, n3]) where
    #   X = X1 repeated n1 times, X2 repeated n2 times and so forth
    random_state = check_random_state(0)
    X_ = random_state.randint(0, 200, [10, 1])
    y_ = np.ndarray.flatten(0.2 * X_ + 2)
    sample_weight = random_state.randint(0, 10, 10)
    outlier_X = random_state.randint(0, 1000, [1, 1])
    outlier_weight = random_state.randint(0, 10, 1)
    outlier_y = random_state.randint(-1000, 0, 1)

    X_flat = np.append(
        np.repeat(X_, sample_weight, axis=0),
        np.repeat(outlier_X, outlier_weight, axis=0),
        axis=0,
    )
    y_flat = np.ndarray.flatten(
        np.append(
            np.repeat(y_, sample_weight, axis=0),
            np.repeat(outlier_y, outlier_weight, axis=0),
            axis=0,
        )
    )
    ransac_estimator.fit(X_flat, y_flat)
    ref_coef_ = ransac_estimator.estimator_.coef_

    sample_weight = np.append(sample_weight, outlier_weight)
    X_ = np.append(X_, outlier_X, axis=0)
    y_ = np.append(y_, outlier_y)
    ransac_estimator.fit(X_, y_, sample_weight)

    assert_allclose(ransac_estimator.estimator_.coef_, ref_coef_)

    # check that if base_estimator.fit doesn't support
    # sample_weight, raises error
    base_estimator = OrthogonalMatchingPursuit()
    ransac_estimator = RANSACRegressor(base_estimator, min_samples=10)

    err_msg = f"{base_estimator.__class__.__name__} does not support sample_weight."
    with pytest.raises(ValueError, match=err_msg):
        ransac_estimator.fit(X, y, weights)


def test_ransac_final_model_fit_sample_weight():
    X, y = make_regression(n_samples=1000, random_state=10)
    rng = check_random_state(42)
    sample_weight = rng.randint(1, 4, size=y.shape[0])
    sample_weight = sample_weight / sample_weight.sum()
    ransac = RANSACRegressor(base_estimator=LinearRegression(), random_state=0)
    ransac.fit(X, y, sample_weight=sample_weight)

    final_model = LinearRegression()
    mask_samples = ransac.inlier_mask_
    final_model.fit(
        X[mask_samples], y[mask_samples], sample_weight=sample_weight[mask_samples]
    )

    assert_allclose(ransac.estimator_.coef_, final_model.coef_, atol=1e-12)


def test_perfect_horizontal_line():
    """Check that we can fit a line where all samples are inliers.
    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/19497
    """
    X = np.arange(100)[:, None]
    y = np.zeros((100,))

    base_estimator = LinearRegression()
    ransac_estimator = RANSACRegressor(base_estimator, random_state=0)
    ransac_estimator.fit(X, y)

    assert_allclose(ransac_estimator.estimator_.coef_, 0.0)
    assert_allclose(ransac_estimator.estimator_.intercept_, 0.0)


# TODO: Remove in v1.2
@pytest.mark.parametrize(
    "old_loss, new_loss",
    [
        ("absolute_loss", "squared_error"),
        ("squared_loss", "absolute_error"),
    ],
)
def test_loss_deprecated(old_loss, new_loss):
    est1 = RANSACRegressor(loss=old_loss, random_state=0)

    with pytest.warns(FutureWarning, match=f"The loss '{old_loss}' was deprecated"):
        est1.fit(X, y)

    est2 = RANSACRegressor(loss=new_loss, random_state=0)
    est2.fit(X, y)
    assert_allclose(est1.predict(X), est2.predict(X))
