import numpy as np
import pytest

from sklearn import config_context
from sklearn.impute import KNNImputer
from sklearn.metrics.pairwise import nan_euclidean_distances
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import KNeighborsRegressor
from sklearn.utils._testing import assert_allclose


@pytest.mark.parametrize("weights", ["uniform", "distance"])
@pytest.mark.parametrize("n_neighbors", range(1, 6))
def test_knn_imputer_shape(weights, n_neighbors):
    # Verify the shapes of the imputed matrix for different weights and
    # number of neighbors.
    n_rows = 10
    n_cols = 2
    X = np.random.rand(n_rows, n_cols)
    X[0, 0] = np.nan

    imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights)
    X_imputed = imputer.fit_transform(X)
    assert X_imputed.shape == (n_rows, n_cols)


@pytest.mark.parametrize("na", [np.nan, -1])
def test_knn_imputer_default_with_invalid_input(na):
    # Test imputation with default values and invalid input

    # Test with inf present
    X = np.array(
        [
            [np.inf, 1, 1, 2, na],
            [2, 1, 2, 2, 3],
            [3, 2, 3, 3, 8],
            [na, 6, 0, 5, 13],
            [na, 7, 0, 7, 8],
            [6, 6, 2, 5, 7],
        ]
    )
    with pytest.raises(ValueError, match="Input contains (infinity|NaN)"):
        KNNImputer(missing_values=na).fit(X)

    # Test with inf present in matrix passed in transform()
    X = np.array(
        [
            [np.inf, 1, 1, 2, na],
            [2, 1, 2, 2, 3],
            [3, 2, 3, 3, 8],
            [na, 6, 0, 5, 13],
            [na, 7, 0, 7, 8],
            [6, 6, 2, 5, 7],
        ]
    )

    X_fit = np.array(
        [
            [0, 1, 1, 2, na],
            [2, 1, 2, 2, 3],
            [3, 2, 3, 3, 8],
            [na, 6, 0, 5, 13],
            [na, 7, 0, 7, 8],
            [6, 6, 2, 5, 7],
        ]
    )
    imputer = KNNImputer(missing_values=na).fit(X_fit)
    with pytest.raises(ValueError, match="Input contains (infinity|NaN)"):
        imputer.transform(X)

    # negative n_neighbors
    with pytest.raises(ValueError, match="Expected n_neighbors > 0"):
        KNNImputer(missing_values=na, n_neighbors=0).fit(X_fit)

    # Test with missing_values=0 when NaN present
    imputer = KNNImputer(missing_values=0, n_neighbors=2, weights="uniform")
    X = np.array(
        [
            [np.nan, 0, 0, 0, 5],
            [np.nan, 1, 0, np.nan, 3],
            [np.nan, 2, 0, 0, 0],
            [np.nan, 6, 0, 5, 13],
        ]
    )
    msg = (
        r"Input contains NaN, infinity or a value too large for " r"dtype\('float64'\)"
    )
    with pytest.raises(ValueError, match=msg):
        imputer.fit(X)

    X = np.array(
        [
            [0, 0],
            [np.nan, 2],
        ]
    )

    # Test with a metric type without NaN support
    imputer = KNNImputer(metric="euclidean")
    bad_metric_msg = "The selected metric does not support NaN values"
    with pytest.raises(ValueError, match=bad_metric_msg):
        imputer.fit(X)


@pytest.mark.parametrize("na", [np.nan, -1])
def test_knn_imputer_removes_all_na_features(na):
    X = np.array(
        [
            [1, 1, na, 1, 1, 1.0],
            [2, 3, na, 2, 2, 2],
            [3, 4, na, 3, 3, na],
            [6, 4, na, na, 6, 6],
        ]
    )
    knn = KNNImputer(missing_values=na, n_neighbors=2).fit(X)

    X_transform = knn.transform(X)
    assert not np.isnan(X_transform).any()
    assert X_transform.shape == (4, 5)

    X_test = np.arange(0, 12).reshape(2, 6)
    X_transform = knn.transform(X_test)
    assert_allclose(X_test[:, [0, 1, 3, 4, 5]], X_transform)


@pytest.mark.parametrize("na", [np.nan, -1])
def test_knn_imputer_zero_nan_imputes_the_same(na):
    # Test with an imputable matrix and compare with different missing_values
    X_zero = np.array(
        [
            [1, 0, 1, 1, 1.0],
            [2, 2, 2, 2, 2],
            [3, 3, 3, 3, 0],
            [6, 6, 0, 6, 6],
        ]
    )

    X_nan = np.array(
        [
            [1, na, 1, 1, 1.0],
            [2, 2, 2, 2, 2],
            [3, 3, 3, 3, na],
            [6, 6, na, 6, 6],
        ]
    )

    X_imputed = np.array(
        [
            [1, 2.5, 1, 1, 1.0],
            [2, 2, 2, 2, 2],
            [3, 3, 3, 3, 1.5],
            [6, 6, 2.5, 6, 6],
        ]
    )

    imputer_zero = KNNImputer(missing_values=0, n_neighbors=2, weights="uniform")

    imputer_nan = KNNImputer(missing_values=na, n_neighbors=2, weights="uniform")

    assert_allclose(imputer_zero.fit_transform(X_zero), X_imputed)
    assert_allclose(
        imputer_zero.fit_transform(X_zero), imputer_nan.fit_transform(X_nan)
    )


@pytest.mark.parametrize("na", [np.nan, -1])
def test_knn_imputer_verify(na):
    # Test with an imputable matrix
    X = np.array(
        [
            [1, 0, 0, 1],
            [2, 1, 2, na],
            [3, 2, 3, na],
            [na, 4, 5, 5],
            [6, na, 6, 7],
            [8, 8, 8, 8],
            [16, 15, 18, 19],
        ]
    )

    X_imputed = np.array(
        [
            [1, 0, 0, 1],
            [2, 1, 2, 8],
            [3, 2, 3, 8],
            [4, 4, 5, 5],
            [6, 3, 6, 7],
            [8, 8, 8, 8],
            [16, 15, 18, 19],
        ]
    )

    imputer = KNNImputer(missing_values=na)
    assert_allclose(imputer.fit_transform(X), X_imputed)

    # Test when there is not enough neighbors
    X = np.array(
        [
            [1, 0, 0, na],
            [2, 1, 2, na],
            [3, 2, 3, na],
            [4, 4, 5, na],
            [6, 7, 6, na],
            [8, 8, 8, na],
            [20, 20, 20, 20],
            [22, 22, 22, 22],
        ]
    )

    # Not enough neighbors, use column mean from training
    X_impute_value = (20 + 22) / 2
    X_imputed = np.array(
        [
            [1, 0, 0, X_impute_value],
            [2, 1, 2, X_impute_value],
            [3, 2, 3, X_impute_value],
            [4, 4, 5, X_impute_value],
            [6, 7, 6, X_impute_value],
            [8, 8, 8, X_impute_value],
            [20, 20, 20, 20],
            [22, 22, 22, 22],
        ]
    )

    imputer = KNNImputer(missing_values=na)
    assert_allclose(imputer.fit_transform(X), X_imputed)

    # Test when data in fit() and transform() are different
    X = np.array([[0, 0], [na, 2], [4, 3], [5, 6], [7, 7], [9, 8], [11, 16]])

    X1 = np.array([[1, 0], [3, 2], [4, na]])

    X_2_1 = (0 + 3 + 6 + 7 + 8) / 5
    X1_imputed = np.array([[1, 0], [3, 2], [4, X_2_1]])

    imputer = KNNImputer(missing_values=na)
    assert_allclose(imputer.fit(X).transform(X1), X1_imputed)


@pytest.mark.parametrize("na", [np.nan, -1])
def test_knn_imputer_one_n_neighbors(na):

    X = np.array([[0, 0], [na, 2], [4, 3], [5, na], [7, 7], [na, 8], [14, 13]])

    X_imputed = np.array([[0, 0], [4, 2], [4, 3], [5, 3], [7, 7], [7, 8], [14, 13]])

    imputer = KNNImputer(n_neighbors=1, missing_values=na)

    assert_allclose(imputer.fit_transform(X), X_imputed)


@pytest.mark.parametrize("na", [np.nan, -1])
def test_knn_imputer_all_samples_are_neighbors(na):
    X = np.array([[0, 0], [na, 2], [4, 3], [5, na], [7, 7], [na, 8], [14, 13]])

    X_imputed = np.array([[0, 0], [6, 2], [4, 3], [5, 5.5], [7, 7], [6, 8], [14, 13]])

    n_neighbors = X.shape[0] - 1
    imputer = KNNImputer(n_neighbors=n_neighbors, missing_values=na)

    assert_allclose(imputer.fit_transform(X), X_imputed)

    n_neighbors = X.shape[0]
    imputer_plus1 = KNNImputer(n_neighbors=n_neighbors, missing_values=na)
    assert_allclose(imputer_plus1.fit_transform(X), X_imputed)


@pytest.mark.parametrize("na", [np.nan, -1])
def test_knn_imputer_weight_uniform(na):

    X = np.array([[0, 0], [na, 2], [4, 3], [5, 6], [7, 7], [9, 8], [11, 10]])

    # Test with "uniform" weight (or unweighted)
    X_imputed_uniform = np.array(
        [[0, 0], [5, 2], [4, 3], [5, 6], [7, 7], [9, 8], [11, 10]]
    )

    imputer = KNNImputer(weights="uniform", missing_values=na)
    assert_allclose(imputer.fit_transform(X), X_imputed_uniform)

    # Test with "callable" weight
    def no_weight(dist):
        return None

    imputer = KNNImputer(weights=no_weight, missing_values=na)
    assert_allclose(imputer.fit_transform(X), X_imputed_uniform)

    # Test with "callable" uniform weight
    def uniform_weight(dist):
        return np.ones_like(dist)

    imputer = KNNImputer(weights=uniform_weight, missing_values=na)
    assert_allclose(imputer.fit_transform(X), X_imputed_uniform)


@pytest.mark.parametrize("na", [np.nan, -1])
def test_knn_imputer_weight_distance(na):
    X = np.array([[0, 0], [na, 2], [4, 3], [5, 6], [7, 7], [9, 8], [11, 10]])

    # Test with "distance" weight
    nn = KNeighborsRegressor(metric="euclidean", weights="distance")
    X_rows_idx = [0, 2, 3, 4, 5, 6]
    nn.fit(X[X_rows_idx, 1:], X[X_rows_idx, 0])
    knn_imputed_value = nn.predict(X[1:2, 1:])[0]

    # Manual calculation
    X_neighbors_idx = [0, 2, 3, 4, 5]
    dist = nan_euclidean_distances(X[1:2, :], X, missing_values=na)
    weights = 1 / dist[:, X_neighbors_idx].ravel()
    manual_imputed_value = np.average(X[X_neighbors_idx, 0], weights=weights)

    X_imputed_distance1 = np.array(
        [[0, 0], [manual_imputed_value, 2], [4, 3], [5, 6], [7, 7], [9, 8], [11, 10]]
    )

    # NearestNeighbor calculation
    X_imputed_distance2 = np.array(
        [[0, 0], [knn_imputed_value, 2], [4, 3], [5, 6], [7, 7], [9, 8], [11, 10]]
    )

    imputer = KNNImputer(weights="distance", missing_values=na)
    assert_allclose(imputer.fit_transform(X), X_imputed_distance1)
    assert_allclose(imputer.fit_transform(X), X_imputed_distance2)

    # Test with weights = "distance" and n_neighbors=2
    X = np.array(
        [
            [na, 0, 0],
            [2, 1, 2],
            [3, 2, 3],
            [4, 5, 5],
        ]
    )

    # neighbors are rows 1, 2, the nan_euclidean_distances are:
    dist_0_1 = np.sqrt((3 / 2) * ((1 - 0) ** 2 + (2 - 0) ** 2))
    dist_0_2 = np.sqrt((3 / 2) * ((2 - 0) ** 2 + (3 - 0) ** 2))
    imputed_value = np.average([2, 3], weights=[1 / dist_0_1, 1 / dist_0_2])

    X_imputed = np.array(
        [
            [imputed_value, 0, 0],
            [2, 1, 2],
            [3, 2, 3],
            [4, 5, 5],
        ]
    )

    imputer = KNNImputer(n_neighbors=2, weights="distance", missing_values=na)
    assert_allclose(imputer.fit_transform(X), X_imputed)

    # Test with varying missingness patterns
    X = np.array(
        [
            [1, 0, 0, 1],
            [0, na, 1, na],
            [1, 1, 1, na],
            [0, 1, 0, 0],
            [0, 0, 0, 0],
            [1, 0, 1, 1],
            [10, 10, 10, 10],
        ]
    )

    # Get weights of donor neighbors
    dist = nan_euclidean_distances(X, missing_values=na)
    r1c1_nbor_dists = dist[1, [0, 2, 3, 4, 5]]
    r1c3_nbor_dists = dist[1, [0, 3, 4, 5, 6]]
    r1c1_nbor_wt = 1 / r1c1_nbor_dists
    r1c3_nbor_wt = 1 / r1c3_nbor_dists

    r2c3_nbor_dists = dist[2, [0, 3, 4, 5, 6]]
    r2c3_nbor_wt = 1 / r2c3_nbor_dists

    # Collect donor values
    col1_donor_values = np.ma.masked_invalid(X[[0, 2, 3, 4, 5], 1]).copy()
    col3_donor_values = np.ma.masked_invalid(X[[0, 3, 4, 5, 6], 3]).copy()

    # Final imputed values
    r1c1_imp = np.ma.average(col1_donor_values, weights=r1c1_nbor_wt)
    r1c3_imp = np.ma.average(col3_donor_values, weights=r1c3_nbor_wt)
    r2c3_imp = np.ma.average(col3_donor_values, weights=r2c3_nbor_wt)

    X_imputed = np.array(
        [
            [1, 0, 0, 1],
            [0, r1c1_imp, 1, r1c3_imp],
            [1, 1, 1, r2c3_imp],
            [0, 1, 0, 0],
            [0, 0, 0, 0],
            [1, 0, 1, 1],
            [10, 10, 10, 10],
        ]
    )

    imputer = KNNImputer(weights="distance", missing_values=na)
    assert_allclose(imputer.fit_transform(X), X_imputed)

    X = np.array(
        [
            [0, 0, 0, na],
            [1, 1, 1, na],
            [2, 2, na, 2],
            [3, 3, 3, 3],
            [4, 4, 4, 4],
            [5, 5, 5, 5],
            [6, 6, 6, 6],
            [na, 7, 7, 7],
        ]
    )

    dist = pairwise_distances(
        X, metric="nan_euclidean", squared=False, missing_values=na
    )

    # Calculate weights
    r0c3_w = 1.0 / dist[0, 2:-1]
    r1c3_w = 1.0 / dist[1, 2:-1]
    r2c2_w = 1.0 / dist[2, (0, 1, 3, 4, 5)]
    r7c0_w = 1.0 / dist[7, 2:7]

    # Calculate weighted averages
    r0c3 = np.average(X[2:-1, -1], weights=r0c3_w)
    r1c3 = np.average(X[2:-1, -1], weights=r1c3_w)
    r2c2 = np.average(X[(0, 1, 3, 4, 5), 2], weights=r2c2_w)
    r7c0 = np.average(X[2:7, 0], weights=r7c0_w)

    X_imputed = np.array(
        [
            [0, 0, 0, r0c3],
            [1, 1, 1, r1c3],
            [2, 2, r2c2, 2],
            [3, 3, 3, 3],
            [4, 4, 4, 4],
            [5, 5, 5, 5],
            [6, 6, 6, 6],
            [r7c0, 7, 7, 7],
        ]
    )

    imputer_comp_wt = KNNImputer(missing_values=na, weights="distance")
    assert_allclose(imputer_comp_wt.fit_transform(X), X_imputed)


def test_knn_imputer_callable_metric():

    # Define callable metric that returns the l1 norm:
    def custom_callable(x, y, missing_values=np.nan, squared=False):
        x = np.ma.array(x, mask=np.isnan(x))
        y = np.ma.array(y, mask=np.isnan(y))
        dist = np.nansum(np.abs(x - y))
        return dist

    X = np.array([[4, 3, 3, np.nan], [6, 9, 6, 9], [4, 8, 6, 9], [np.nan, 9, 11, 10.0]])

    X_0_3 = (9 + 9) / 2
    X_3_0 = (6 + 4) / 2
    X_imputed = np.array(
        [[4, 3, 3, X_0_3], [6, 9, 6, 9], [4, 8, 6, 9], [X_3_0, 9, 11, 10.0]]
    )

    imputer = KNNImputer(n_neighbors=2, metric=custom_callable)
    assert_allclose(imputer.fit_transform(X), X_imputed)


@pytest.mark.parametrize("working_memory", [None, 0])
@pytest.mark.parametrize("na", [-1, np.nan])
# Note that we use working_memory=0 to ensure that chunking is tested, even
# for a small dataset. However, it should raise a UserWarning that we ignore.
@pytest.mark.filterwarnings("ignore:adhere to working_memory")
def test_knn_imputer_with_simple_example(na, working_memory):

    X = np.array(
        [
            [0, na, 0, na],
            [1, 1, 1, na],
            [2, 2, na, 2],
            [3, 3, 3, 3],
            [4, 4, 4, 4],
            [5, 5, 5, 5],
            [6, 6, 6, 6],
            [na, 7, 7, 7],
        ]
    )

    r0c1 = np.mean(X[1:6, 1])
    r0c3 = np.mean(X[2:-1, -1])
    r1c3 = np.mean(X[2:-1, -1])
    r2c2 = np.mean(X[[0, 1, 3, 4, 5], 2])
    r7c0 = np.mean(X[2:-1, 0])

    X_imputed = np.array(
        [
            [0, r0c1, 0, r0c3],
            [1, 1, 1, r1c3],
            [2, 2, r2c2, 2],
            [3, 3, 3, 3],
            [4, 4, 4, 4],
            [5, 5, 5, 5],
            [6, 6, 6, 6],
            [r7c0, 7, 7, 7],
        ]
    )

    with config_context(working_memory=working_memory):
        imputer_comp = KNNImputer(missing_values=na)
        assert_allclose(imputer_comp.fit_transform(X), X_imputed)


@pytest.mark.parametrize("na", [-1, np.nan])
@pytest.mark.parametrize("weights", ["uniform", "distance"])
def test_knn_imputer_not_enough_valid_distances(na, weights):
    # Samples with needed feature has nan distance
    X1 = np.array([[na, 11], [na, 1], [3, na]])
    X1_imputed = np.array([[3, 11], [3, 1], [3, 6]])

    knn = KNNImputer(missing_values=na, n_neighbors=1, weights=weights)
    assert_allclose(knn.fit_transform(X1), X1_imputed)

    X2 = np.array([[4, na]])
    X2_imputed = np.array([[4, 6]])
    assert_allclose(knn.transform(X2), X2_imputed)


@pytest.mark.parametrize("na", [-1, np.nan])
def test_knn_imputer_drops_all_nan_features(na):
    X1 = np.array([[na, 1], [na, 2]])
    knn = KNNImputer(missing_values=na, n_neighbors=1)
    X1_expected = np.array([[1], [2]])
    assert_allclose(knn.fit_transform(X1), X1_expected)

    X2 = np.array([[1, 2], [3, na]])
    X2_expected = np.array([[2], [1.5]])
    assert_allclose(knn.transform(X2), X2_expected)


@pytest.mark.parametrize("working_memory", [None, 0])
@pytest.mark.parametrize("na", [-1, np.nan])
def test_knn_imputer_distance_weighted_not_enough_neighbors(na, working_memory):
    X = np.array([[3, na], [2, na], [na, 4], [5, 6], [6, 8], [na, 5]])

    dist = pairwise_distances(
        X, metric="nan_euclidean", squared=False, missing_values=na
    )

    X_01 = np.average(X[3:5, 1], weights=1 / dist[0, 3:5])
    X_11 = np.average(X[3:5, 1], weights=1 / dist[1, 3:5])
    X_20 = np.average(X[3:5, 0], weights=1 / dist[2, 3:5])
    X_50 = np.average(X[3:5, 0], weights=1 / dist[5, 3:5])

    X_expected = np.array([[3, X_01], [2, X_11], [X_20, 4], [5, 6], [6, 8], [X_50, 5]])

    with config_context(working_memory=working_memory):
        knn_3 = KNNImputer(missing_values=na, n_neighbors=3, weights="distance")
        assert_allclose(knn_3.fit_transform(X), X_expected)

        knn_4 = KNNImputer(missing_values=na, n_neighbors=4, weights="distance")
        assert_allclose(knn_4.fit_transform(X), X_expected)


@pytest.mark.parametrize("na, allow_nan", [(-1, False), (np.nan, True)])
def test_knn_tags(na, allow_nan):
    knn = KNNImputer(missing_values=na)
    assert knn._get_tags()["allow_nan"] == allow_nan
