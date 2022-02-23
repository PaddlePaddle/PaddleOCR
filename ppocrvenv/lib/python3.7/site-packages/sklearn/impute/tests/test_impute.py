import pytest

import numpy as np
from scipy import sparse
from scipy.stats import kstest

import io

from sklearn.utils._testing import assert_allclose
from sklearn.utils._testing import assert_allclose_dense_sparse
from sklearn.utils._testing import assert_array_equal
from sklearn.utils._testing import assert_array_almost_equal

# make IterativeImputer available
from sklearn.experimental import enable_iterative_imputer  # noqa

from sklearn.datasets import load_diabetes
from sklearn.impute import MissingIndicator
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import BayesianRidge, ARDRegression, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_union
from sklearn.model_selection import GridSearchCV
from sklearn import tree
from sklearn.random_projection import _sparse_random_matrix
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute._base import _most_frequent


def _check_statistics(X, X_true, strategy, statistics, missing_values):
    """Utility function for testing imputation for a given strategy.

    Test with dense and sparse arrays

    Check that:
        - the statistics (mean, median, mode) are correct
        - the missing values are imputed correctly"""

    err_msg = "Parameters: strategy = %s, missing_values = %s, sparse = {0}" % (
        strategy,
        missing_values,
    )

    assert_ae = assert_array_equal

    if X.dtype.kind == "f" or X_true.dtype.kind == "f":
        assert_ae = assert_array_almost_equal

    # Normal matrix
    imputer = SimpleImputer(missing_values=missing_values, strategy=strategy)
    X_trans = imputer.fit(X).transform(X.copy())
    assert_ae(imputer.statistics_, statistics, err_msg=err_msg.format(False))
    assert_ae(X_trans, X_true, err_msg=err_msg.format(False))

    # Sparse matrix
    imputer = SimpleImputer(missing_values=missing_values, strategy=strategy)
    imputer.fit(sparse.csc_matrix(X))
    X_trans = imputer.transform(sparse.csc_matrix(X.copy()))

    if sparse.issparse(X_trans):
        X_trans = X_trans.toarray()

    assert_ae(imputer.statistics_, statistics, err_msg=err_msg.format(True))
    assert_ae(X_trans, X_true, err_msg=err_msg.format(True))


@pytest.mark.parametrize("strategy", ["mean", "median", "most_frequent", "constant"])
def test_imputation_shape(strategy):
    # Verify the shapes of the imputed matrix for different strategies.
    X = np.random.randn(10, 2)
    X[::2] = np.nan

    imputer = SimpleImputer(strategy=strategy)
    X_imputed = imputer.fit_transform(sparse.csr_matrix(X))
    assert X_imputed.shape == (10, 2)
    X_imputed = imputer.fit_transform(X)
    assert X_imputed.shape == (10, 2)

    iterative_imputer = IterativeImputer(initial_strategy=strategy)
    X_imputed = iterative_imputer.fit_transform(X)
    assert X_imputed.shape == (10, 2)


@pytest.mark.parametrize("strategy", ["const", 101, None])
def test_imputation_error_invalid_strategy(strategy):
    X = np.ones((3, 5))
    X[0, 0] = np.nan

    with pytest.raises(ValueError, match=str(strategy)):
        imputer = SimpleImputer(strategy=strategy)
        imputer.fit_transform(X)


@pytest.mark.parametrize("strategy", ["mean", "median", "most_frequent"])
def test_imputation_deletion_warning(strategy):
    X = np.ones((3, 5))
    X[:, 0] = np.nan

    with pytest.warns(UserWarning, match="Deleting"):
        imputer = SimpleImputer(strategy=strategy, verbose=True)
        imputer.fit_transform(X)


@pytest.mark.parametrize("strategy", ["mean", "median", "most_frequent", "constant"])
def test_imputation_error_sparse_0(strategy):
    # check that error are raised when missing_values = 0 and input is sparse
    X = np.ones((3, 5))
    X[0] = 0
    X = sparse.csc_matrix(X)

    imputer = SimpleImputer(strategy=strategy, missing_values=0)
    with pytest.raises(ValueError, match="Provide a dense array"):
        imputer.fit(X)

    imputer.fit(X.toarray())
    with pytest.raises(ValueError, match="Provide a dense array"):
        imputer.transform(X)


def safe_median(arr, *args, **kwargs):
    # np.median([]) raises a TypeError for numpy >= 1.10.1
    length = arr.size if hasattr(arr, "size") else len(arr)
    return np.nan if length == 0 else np.median(arr, *args, **kwargs)


def safe_mean(arr, *args, **kwargs):
    # np.mean([]) raises a RuntimeWarning for numpy >= 1.10.1
    length = arr.size if hasattr(arr, "size") else len(arr)
    return np.nan if length == 0 else np.mean(arr, *args, **kwargs)


def test_imputation_mean_median():
    # Test imputation using the mean and median strategies, when
    # missing_values != 0.
    rng = np.random.RandomState(0)

    dim = 10
    dec = 10
    shape = (dim * dim, dim + dec)

    zeros = np.zeros(shape[0])
    values = np.arange(1, shape[0] + 1)
    values[4::2] = -values[4::2]

    tests = [
        ("mean", np.nan, lambda z, v, p: safe_mean(np.hstack((z, v)))),
        ("median", np.nan, lambda z, v, p: safe_median(np.hstack((z, v)))),
    ]

    for strategy, test_missing_values, true_value_fun in tests:
        X = np.empty(shape)
        X_true = np.empty(shape)
        true_statistics = np.empty(shape[1])

        # Create a matrix X with columns
        #    - with only zeros,
        #    - with only missing values
        #    - with zeros, missing values and values
        # And a matrix X_true containing all true values
        for j in range(shape[1]):
            nb_zeros = (j - dec + 1 > 0) * (j - dec + 1) * (j - dec + 1)
            nb_missing_values = max(shape[0] + dec * dec - (j + dec) * (j + dec), 0)
            nb_values = shape[0] - nb_zeros - nb_missing_values

            z = zeros[:nb_zeros]
            p = np.repeat(test_missing_values, nb_missing_values)
            v = values[rng.permutation(len(values))[:nb_values]]

            true_statistics[j] = true_value_fun(z, v, p)

            # Create the columns
            X[:, j] = np.hstack((v, z, p))

            if 0 == test_missing_values:
                # XXX unreached code as of v0.22
                X_true[:, j] = np.hstack(
                    (v, np.repeat(true_statistics[j], nb_missing_values + nb_zeros))
                )
            else:
                X_true[:, j] = np.hstack(
                    (v, z, np.repeat(true_statistics[j], nb_missing_values))
                )

            # Shuffle them the same way
            np.random.RandomState(j).shuffle(X[:, j])
            np.random.RandomState(j).shuffle(X_true[:, j])

        # Mean doesn't support columns containing NaNs, median does
        if strategy == "median":
            cols_to_keep = ~np.isnan(X_true).any(axis=0)
        else:
            cols_to_keep = ~np.isnan(X_true).all(axis=0)

        X_true = X_true[:, cols_to_keep]

        _check_statistics(X, X_true, strategy, true_statistics, test_missing_values)


def test_imputation_median_special_cases():
    # Test median imputation with sparse boundary cases
    X = np.array(
        [
            [0, np.nan, np.nan],  # odd: implicit zero
            [5, np.nan, np.nan],  # odd: explicit nonzero
            [0, 0, np.nan],  # even: average two zeros
            [-5, 0, np.nan],  # even: avg zero and neg
            [0, 5, np.nan],  # even: avg zero and pos
            [4, 5, np.nan],  # even: avg nonzeros
            [-4, -5, np.nan],  # even: avg negatives
            [-1, 2, np.nan],  # even: crossing neg and pos
        ]
    ).transpose()

    X_imputed_median = np.array(
        [
            [0, 0, 0],
            [5, 5, 5],
            [0, 0, 0],
            [-5, 0, -2.5],
            [0, 5, 2.5],
            [4, 5, 4.5],
            [-4, -5, -4.5],
            [-1, 2, 0.5],
        ]
    ).transpose()
    statistics_median = [0, 5, 0, -2.5, 2.5, 4.5, -4.5, 0.5]

    _check_statistics(X, X_imputed_median, "median", statistics_median, np.nan)


@pytest.mark.parametrize("strategy", ["mean", "median"])
@pytest.mark.parametrize("dtype", [None, object, str])
def test_imputation_mean_median_error_invalid_type(strategy, dtype):
    X = np.array([["a", "b", 3], [4, "e", 6], ["g", "h", 9]], dtype=dtype)
    msg = "non-numeric data:\ncould not convert string to float: '"
    with pytest.raises(ValueError, match=msg):
        imputer = SimpleImputer(strategy=strategy)
        imputer.fit_transform(X)


@pytest.mark.parametrize("strategy", ["mean", "median"])
@pytest.mark.parametrize("type", ["list", "dataframe"])
def test_imputation_mean_median_error_invalid_type_list_pandas(strategy, type):
    X = [["a", "b", 3], [4, "e", 6], ["g", "h", 9]]
    if type == "dataframe":
        pd = pytest.importorskip("pandas")
        X = pd.DataFrame(X)
    msg = "non-numeric data:\ncould not convert string to float: '"
    with pytest.raises(ValueError, match=msg):
        imputer = SimpleImputer(strategy=strategy)
        imputer.fit_transform(X)


@pytest.mark.parametrize("strategy", ["constant", "most_frequent"])
@pytest.mark.parametrize("dtype", [str, np.dtype("U"), np.dtype("S")])
def test_imputation_const_mostf_error_invalid_types(strategy, dtype):
    # Test imputation on non-numeric data using "most_frequent" and "constant"
    # strategy
    X = np.array(
        [
            [np.nan, np.nan, "a", "f"],
            [np.nan, "c", np.nan, "d"],
            [np.nan, "b", "d", np.nan],
            [np.nan, "c", "d", "h"],
        ],
        dtype=dtype,
    )

    err_msg = "SimpleImputer does not support data"
    with pytest.raises(ValueError, match=err_msg):
        imputer = SimpleImputer(strategy=strategy)
        imputer.fit(X).transform(X)


def test_imputation_most_frequent():
    # Test imputation using the most-frequent strategy.
    X = np.array(
        [
            [-1, -1, 0, 5],
            [-1, 2, -1, 3],
            [-1, 1, 3, -1],
            [-1, 2, 3, 7],
        ]
    )

    X_true = np.array(
        [
            [2, 0, 5],
            [2, 3, 3],
            [1, 3, 3],
            [2, 3, 7],
        ]
    )

    # scipy.stats.mode, used in SimpleImputer, doesn't return the first most
    # frequent as promised in the doc but the lowest most frequent. When this
    # test will fail after an update of scipy, SimpleImputer will need to be
    # updated to be consistent with the new (correct) behaviour
    _check_statistics(X, X_true, "most_frequent", [np.nan, 2, 3, 3], -1)


@pytest.mark.parametrize("marker", [None, np.nan, "NAN", "", 0])
def test_imputation_most_frequent_objects(marker):
    # Test imputation using the most-frequent strategy.
    X = np.array(
        [
            [marker, marker, "a", "f"],
            [marker, "c", marker, "d"],
            [marker, "b", "d", marker],
            [marker, "c", "d", "h"],
        ],
        dtype=object,
    )

    X_true = np.array(
        [
            ["c", "a", "f"],
            ["c", "d", "d"],
            ["b", "d", "d"],
            ["c", "d", "h"],
        ],
        dtype=object,
    )

    imputer = SimpleImputer(missing_values=marker, strategy="most_frequent")
    X_trans = imputer.fit(X).transform(X)

    assert_array_equal(X_trans, X_true)


@pytest.mark.parametrize("dtype", [object, "category"])
def test_imputation_most_frequent_pandas(dtype):
    # Test imputation using the most frequent strategy on pandas df
    pd = pytest.importorskip("pandas")

    f = io.StringIO("Cat1,Cat2,Cat3,Cat4\n,i,x,\na,,y,\na,j,,\nb,j,x,")

    df = pd.read_csv(f, dtype=dtype)

    X_true = np.array(
        [["a", "i", "x"], ["a", "j", "y"], ["a", "j", "x"], ["b", "j", "x"]],
        dtype=object,
    )

    imputer = SimpleImputer(strategy="most_frequent")
    X_trans = imputer.fit_transform(df)

    assert_array_equal(X_trans, X_true)


@pytest.mark.parametrize("X_data, missing_value", [(1, 0), (1.0, np.nan)])
def test_imputation_constant_error_invalid_type(X_data, missing_value):
    # Verify that exceptions are raised on invalid fill_value type
    X = np.full((3, 5), X_data, dtype=float)
    X[0, 0] = missing_value

    with pytest.raises(ValueError, match="imputing numerical"):
        imputer = SimpleImputer(
            missing_values=missing_value, strategy="constant", fill_value="x"
        )
        imputer.fit_transform(X)


def test_imputation_constant_integer():
    # Test imputation using the constant strategy on integers
    X = np.array([[-1, 2, 3, -1], [4, -1, 5, -1], [6, 7, -1, -1], [8, 9, 0, -1]])

    X_true = np.array([[0, 2, 3, 0], [4, 0, 5, 0], [6, 7, 0, 0], [8, 9, 0, 0]])

    imputer = SimpleImputer(missing_values=-1, strategy="constant", fill_value=0)
    X_trans = imputer.fit_transform(X)

    assert_array_equal(X_trans, X_true)


@pytest.mark.parametrize("array_constructor", [sparse.csr_matrix, np.asarray])
def test_imputation_constant_float(array_constructor):
    # Test imputation using the constant strategy on floats
    X = np.array(
        [
            [np.nan, 1.1, 0, np.nan],
            [1.2, np.nan, 1.3, np.nan],
            [0, 0, np.nan, np.nan],
            [1.4, 1.5, 0, np.nan],
        ]
    )

    X_true = np.array(
        [[-1, 1.1, 0, -1], [1.2, -1, 1.3, -1], [0, 0, -1, -1], [1.4, 1.5, 0, -1]]
    )

    X = array_constructor(X)

    X_true = array_constructor(X_true)

    imputer = SimpleImputer(strategy="constant", fill_value=-1)
    X_trans = imputer.fit_transform(X)

    assert_allclose_dense_sparse(X_trans, X_true)


@pytest.mark.parametrize("marker", [None, np.nan, "NAN", "", 0])
def test_imputation_constant_object(marker):
    # Test imputation using the constant strategy on objects
    X = np.array(
        [
            [marker, "a", "b", marker],
            ["c", marker, "d", marker],
            ["e", "f", marker, marker],
            ["g", "h", "i", marker],
        ],
        dtype=object,
    )

    X_true = np.array(
        [
            ["missing", "a", "b", "missing"],
            ["c", "missing", "d", "missing"],
            ["e", "f", "missing", "missing"],
            ["g", "h", "i", "missing"],
        ],
        dtype=object,
    )

    imputer = SimpleImputer(
        missing_values=marker, strategy="constant", fill_value="missing"
    )
    X_trans = imputer.fit_transform(X)

    assert_array_equal(X_trans, X_true)


@pytest.mark.parametrize("dtype", [object, "category"])
def test_imputation_constant_pandas(dtype):
    # Test imputation using the constant strategy on pandas df
    pd = pytest.importorskip("pandas")

    f = io.StringIO("Cat1,Cat2,Cat3,Cat4\n,i,x,\na,,y,\na,j,,\nb,j,x,")

    df = pd.read_csv(f, dtype=dtype)

    X_true = np.array(
        [
            ["missing_value", "i", "x", "missing_value"],
            ["a", "missing_value", "y", "missing_value"],
            ["a", "j", "missing_value", "missing_value"],
            ["b", "j", "x", "missing_value"],
        ],
        dtype=object,
    )

    imputer = SimpleImputer(strategy="constant")
    X_trans = imputer.fit_transform(df)

    assert_array_equal(X_trans, X_true)


@pytest.mark.parametrize("X", [[[1], [2]], [[1], [np.nan]]])
def test_iterative_imputer_one_feature(X):
    # check we exit early when there is a single feature
    imputer = IterativeImputer().fit(X)
    assert imputer.n_iter_ == 0
    imputer = IterativeImputer()
    imputer.fit([[1], [2]])
    assert imputer.n_iter_ == 0
    imputer.fit([[1], [np.nan]])
    assert imputer.n_iter_ == 0


def test_imputation_pipeline_grid_search():
    # Test imputation within a pipeline + gridsearch.
    X = _sparse_random_matrix(100, 100, density=0.10)
    missing_values = X.data[0]

    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(missing_values=missing_values)),
            ("tree", tree.DecisionTreeRegressor(random_state=0)),
        ]
    )

    parameters = {"imputer__strategy": ["mean", "median", "most_frequent"]}

    Y = _sparse_random_matrix(100, 1, density=0.10).toarray()
    gs = GridSearchCV(pipeline, parameters)
    gs.fit(X, Y)


def test_imputation_copy():
    # Test imputation with copy
    X_orig = _sparse_random_matrix(5, 5, density=0.75, random_state=0)

    # copy=True, dense => copy
    X = X_orig.copy().toarray()
    imputer = SimpleImputer(missing_values=0, strategy="mean", copy=True)
    Xt = imputer.fit(X).transform(X)
    Xt[0, 0] = -1
    assert not np.all(X == Xt)

    # copy=True, sparse csr => copy
    X = X_orig.copy()
    imputer = SimpleImputer(missing_values=X.data[0], strategy="mean", copy=True)
    Xt = imputer.fit(X).transform(X)
    Xt.data[0] = -1
    assert not np.all(X.data == Xt.data)

    # copy=False, dense => no copy
    X = X_orig.copy().toarray()
    imputer = SimpleImputer(missing_values=0, strategy="mean", copy=False)
    Xt = imputer.fit(X).transform(X)
    Xt[0, 0] = -1
    assert_array_almost_equal(X, Xt)

    # copy=False, sparse csc => no copy
    X = X_orig.copy().tocsc()
    imputer = SimpleImputer(missing_values=X.data[0], strategy="mean", copy=False)
    Xt = imputer.fit(X).transform(X)
    Xt.data[0] = -1
    assert_array_almost_equal(X.data, Xt.data)

    # copy=False, sparse csr => copy
    X = X_orig.copy()
    imputer = SimpleImputer(missing_values=X.data[0], strategy="mean", copy=False)
    Xt = imputer.fit(X).transform(X)
    Xt.data[0] = -1
    assert not np.all(X.data == Xt.data)

    # Note: If X is sparse and if missing_values=0, then a (dense) copy of X is
    # made, even if copy=False.


def test_iterative_imputer_zero_iters():
    rng = np.random.RandomState(0)

    n = 100
    d = 10
    X = _sparse_random_matrix(n, d, density=0.10, random_state=rng).toarray()
    missing_flag = X == 0
    X[missing_flag] = np.nan

    imputer = IterativeImputer(max_iter=0)
    X_imputed = imputer.fit_transform(X)
    # with max_iter=0, only initial imputation is performed
    assert_allclose(X_imputed, imputer.initial_imputer_.transform(X))

    # repeat but force n_iter_ to 0
    imputer = IterativeImputer(max_iter=5).fit(X)
    # transformed should not be equal to initial imputation
    assert not np.all(imputer.transform(X) == imputer.initial_imputer_.transform(X))

    imputer.n_iter_ = 0
    # now they should be equal as only initial imputation is done
    assert_allclose(imputer.transform(X), imputer.initial_imputer_.transform(X))


def test_iterative_imputer_verbose():
    rng = np.random.RandomState(0)

    n = 100
    d = 3
    X = _sparse_random_matrix(n, d, density=0.10, random_state=rng).toarray()
    imputer = IterativeImputer(missing_values=0, max_iter=1, verbose=1)
    imputer.fit(X)
    imputer.transform(X)
    imputer = IterativeImputer(missing_values=0, max_iter=1, verbose=2)
    imputer.fit(X)
    imputer.transform(X)


def test_iterative_imputer_all_missing():
    n = 100
    d = 3
    X = np.zeros((n, d))
    imputer = IterativeImputer(missing_values=0, max_iter=1)
    X_imputed = imputer.fit_transform(X)
    assert_allclose(X_imputed, imputer.initial_imputer_.transform(X))


@pytest.mark.parametrize(
    "imputation_order", ["random", "roman", "ascending", "descending", "arabic"]
)
def test_iterative_imputer_imputation_order(imputation_order):
    rng = np.random.RandomState(0)
    n = 100
    d = 10
    max_iter = 2
    X = _sparse_random_matrix(n, d, density=0.10, random_state=rng).toarray()
    X[:, 0] = 1  # this column should not be discarded by IterativeImputer

    imputer = IterativeImputer(
        missing_values=0,
        max_iter=max_iter,
        n_nearest_features=5,
        sample_posterior=False,
        skip_complete=True,
        min_value=0,
        max_value=1,
        verbose=1,
        imputation_order=imputation_order,
        random_state=rng,
    )
    imputer.fit_transform(X)
    ordered_idx = [i.feat_idx for i in imputer.imputation_sequence_]

    assert len(ordered_idx) // imputer.n_iter_ == imputer.n_features_with_missing_

    if imputation_order == "roman":
        assert np.all(ordered_idx[: d - 1] == np.arange(1, d))
    elif imputation_order == "arabic":
        assert np.all(ordered_idx[: d - 1] == np.arange(d - 1, 0, -1))
    elif imputation_order == "random":
        ordered_idx_round_1 = ordered_idx[: d - 1]
        ordered_idx_round_2 = ordered_idx[d - 1 :]
        assert ordered_idx_round_1 != ordered_idx_round_2
    elif "ending" in imputation_order:
        assert len(ordered_idx) == max_iter * (d - 1)


@pytest.mark.parametrize(
    "estimator", [None, DummyRegressor(), BayesianRidge(), ARDRegression(), RidgeCV()]
)
def test_iterative_imputer_estimators(estimator):
    rng = np.random.RandomState(0)

    n = 100
    d = 10
    X = _sparse_random_matrix(n, d, density=0.10, random_state=rng).toarray()

    imputer = IterativeImputer(
        missing_values=0, max_iter=1, estimator=estimator, random_state=rng
    )
    imputer.fit_transform(X)

    # check that types are correct for estimators
    hashes = []
    for triplet in imputer.imputation_sequence_:
        expected_type = (
            type(estimator) if estimator is not None else type(BayesianRidge())
        )
        assert isinstance(triplet.estimator, expected_type)
        hashes.append(id(triplet.estimator))

    # check that each estimator is unique
    assert len(set(hashes)) == len(hashes)


def test_iterative_imputer_clip():
    rng = np.random.RandomState(0)
    n = 100
    d = 10
    X = _sparse_random_matrix(n, d, density=0.10, random_state=rng).toarray()

    imputer = IterativeImputer(
        missing_values=0, max_iter=1, min_value=0.1, max_value=0.2, random_state=rng
    )

    Xt = imputer.fit_transform(X)
    assert_allclose(np.min(Xt[X == 0]), 0.1)
    assert_allclose(np.max(Xt[X == 0]), 0.2)
    assert_allclose(Xt[X != 0], X[X != 0])


def test_iterative_imputer_clip_truncnorm():
    rng = np.random.RandomState(0)
    n = 100
    d = 10
    X = _sparse_random_matrix(n, d, density=0.10, random_state=rng).toarray()
    X[:, 0] = 1

    imputer = IterativeImputer(
        missing_values=0,
        max_iter=2,
        n_nearest_features=5,
        sample_posterior=True,
        min_value=0.1,
        max_value=0.2,
        verbose=1,
        imputation_order="random",
        random_state=rng,
    )
    Xt = imputer.fit_transform(X)
    assert_allclose(np.min(Xt[X == 0]), 0.1)
    assert_allclose(np.max(Xt[X == 0]), 0.2)
    assert_allclose(Xt[X != 0], X[X != 0])


def test_iterative_imputer_truncated_normal_posterior():
    #  test that the values that are imputed using `sample_posterior=True`
    #  with boundaries (`min_value` and `max_value` are not None) are drawn
    #  from a distribution that looks gaussian via the Kolmogorov Smirnov test.
    #  note that starting from the wrong random seed will make this test fail
    #  because random sampling doesn't occur at all when the imputation
    #  is outside of the (min_value, max_value) range
    rng = np.random.RandomState(42)

    X = rng.normal(size=(5, 5))
    X[0][0] = np.nan

    imputer = IterativeImputer(
        min_value=0, max_value=0.5, sample_posterior=True, random_state=rng
    )

    imputer.fit_transform(X)
    # generate multiple imputations for the single missing value
    imputations = np.array([imputer.transform(X)[0][0] for _ in range(100)])

    assert all(imputations >= 0)
    assert all(imputations <= 0.5)

    mu, sigma = imputations.mean(), imputations.std()
    ks_statistic, p_value = kstest((imputations - mu) / sigma, "norm")
    if sigma == 0:
        sigma += 1e-12
    ks_statistic, p_value = kstest((imputations - mu) / sigma, "norm")
    # we want to fail to reject null hypothesis
    # null hypothesis: distributions are the same
    assert ks_statistic < 0.2 or p_value > 0.1, "The posterior does appear to be normal"


@pytest.mark.parametrize("strategy", ["mean", "median", "most_frequent"])
def test_iterative_imputer_missing_at_transform(strategy):
    rng = np.random.RandomState(0)
    n = 100
    d = 10
    X_train = rng.randint(low=0, high=3, size=(n, d))
    X_test = rng.randint(low=0, high=3, size=(n, d))

    X_train[:, 0] = 1  # definitely no missing values in 0th column
    X_test[0, 0] = 0  # definitely missing value in 0th column

    imputer = IterativeImputer(
        missing_values=0, max_iter=1, initial_strategy=strategy, random_state=rng
    ).fit(X_train)
    initial_imputer = SimpleImputer(missing_values=0, strategy=strategy).fit(X_train)

    # if there were no missing values at time of fit, then imputer will
    # only use the initial imputer for that feature at transform
    assert_allclose(
        imputer.transform(X_test)[:, 0], initial_imputer.transform(X_test)[:, 0]
    )


def test_iterative_imputer_transform_stochasticity():
    rng1 = np.random.RandomState(0)
    rng2 = np.random.RandomState(1)
    n = 100
    d = 10
    X = _sparse_random_matrix(n, d, density=0.10, random_state=rng1).toarray()

    # when sample_posterior=True, two transforms shouldn't be equal
    imputer = IterativeImputer(
        missing_values=0, max_iter=1, sample_posterior=True, random_state=rng1
    )
    imputer.fit(X)

    X_fitted_1 = imputer.transform(X)
    X_fitted_2 = imputer.transform(X)

    # sufficient to assert that the means are not the same
    assert np.mean(X_fitted_1) != pytest.approx(np.mean(X_fitted_2))

    # when sample_posterior=False, and n_nearest_features=None
    # and imputation_order is not random
    # the two transforms should be identical even if rng are different
    imputer1 = IterativeImputer(
        missing_values=0,
        max_iter=1,
        sample_posterior=False,
        n_nearest_features=None,
        imputation_order="ascending",
        random_state=rng1,
    )

    imputer2 = IterativeImputer(
        missing_values=0,
        max_iter=1,
        sample_posterior=False,
        n_nearest_features=None,
        imputation_order="ascending",
        random_state=rng2,
    )
    imputer1.fit(X)
    imputer2.fit(X)

    X_fitted_1a = imputer1.transform(X)
    X_fitted_1b = imputer1.transform(X)
    X_fitted_2 = imputer2.transform(X)

    assert_allclose(X_fitted_1a, X_fitted_1b)
    assert_allclose(X_fitted_1a, X_fitted_2)


def test_iterative_imputer_no_missing():
    rng = np.random.RandomState(0)
    X = rng.rand(100, 100)
    X[:, 0] = np.nan
    m1 = IterativeImputer(max_iter=10, random_state=rng)
    m2 = IterativeImputer(max_iter=10, random_state=rng)
    pred1 = m1.fit(X).transform(X)
    pred2 = m2.fit_transform(X)
    # should exclude the first column entirely
    assert_allclose(X[:, 1:], pred1)
    # fit and fit_transform should both be identical
    assert_allclose(pred1, pred2)


def test_iterative_imputer_rank_one():
    rng = np.random.RandomState(0)
    d = 50
    A = rng.rand(d, 1)
    B = rng.rand(1, d)
    X = np.dot(A, B)
    nan_mask = rng.rand(d, d) < 0.5
    X_missing = X.copy()
    X_missing[nan_mask] = np.nan

    imputer = IterativeImputer(max_iter=5, verbose=1, random_state=rng)
    X_filled = imputer.fit_transform(X_missing)
    assert_allclose(X_filled, X, atol=0.02)


@pytest.mark.parametrize("rank", [3, 5])
def test_iterative_imputer_transform_recovery(rank):
    rng = np.random.RandomState(0)
    n = 70
    d = 70
    A = rng.rand(n, rank)
    B = rng.rand(rank, d)
    X_filled = np.dot(A, B)
    nan_mask = rng.rand(n, d) < 0.5
    X_missing = X_filled.copy()
    X_missing[nan_mask] = np.nan

    # split up data in half
    n = n // 2
    X_train = X_missing[:n]
    X_test_filled = X_filled[n:]
    X_test = X_missing[n:]

    imputer = IterativeImputer(
        max_iter=5, imputation_order="descending", verbose=1, random_state=rng
    ).fit(X_train)
    X_test_est = imputer.transform(X_test)
    assert_allclose(X_test_filled, X_test_est, atol=0.1)


def test_iterative_imputer_additive_matrix():
    rng = np.random.RandomState(0)
    n = 100
    d = 10
    A = rng.randn(n, d)
    B = rng.randn(n, d)
    X_filled = np.zeros(A.shape)
    for i in range(d):
        for j in range(d):
            X_filled[:, (i + j) % d] += (A[:, i] + B[:, j]) / 2
    # a quarter is randomly missing
    nan_mask = rng.rand(n, d) < 0.25
    X_missing = X_filled.copy()
    X_missing[nan_mask] = np.nan

    # split up data
    n = n // 2
    X_train = X_missing[:n]
    X_test_filled = X_filled[n:]
    X_test = X_missing[n:]

    imputer = IterativeImputer(max_iter=10, verbose=1, random_state=rng).fit(X_train)
    X_test_est = imputer.transform(X_test)
    assert_allclose(X_test_filled, X_test_est, rtol=1e-3, atol=0.01)


@pytest.mark.parametrize(
    "max_iter, tol, error_type, warning",
    [
        (-1, 1e-3, ValueError, "should be a positive integer"),
        (1, -1e-3, ValueError, "should be a non-negative float"),
    ],
)
def test_iterative_imputer_error_param(max_iter, tol, error_type, warning):
    X = np.zeros((100, 2))
    imputer = IterativeImputer(max_iter=max_iter, tol=tol)
    with pytest.raises(error_type, match=warning):
        imputer.fit_transform(X)


def test_iterative_imputer_early_stopping():
    rng = np.random.RandomState(0)
    n = 50
    d = 5
    A = rng.rand(n, 1)
    B = rng.rand(1, d)
    X = np.dot(A, B)
    nan_mask = rng.rand(n, d) < 0.5
    X_missing = X.copy()
    X_missing[nan_mask] = np.nan

    imputer = IterativeImputer(
        max_iter=100, tol=1e-2, sample_posterior=False, verbose=1, random_state=rng
    )
    X_filled_100 = imputer.fit_transform(X_missing)
    assert len(imputer.imputation_sequence_) == d * imputer.n_iter_

    imputer = IterativeImputer(
        max_iter=imputer.n_iter_, sample_posterior=False, verbose=1, random_state=rng
    )
    X_filled_early = imputer.fit_transform(X_missing)
    assert_allclose(X_filled_100, X_filled_early, atol=1e-7)

    imputer = IterativeImputer(
        max_iter=100, tol=0, sample_posterior=False, verbose=1, random_state=rng
    )
    imputer.fit(X_missing)
    assert imputer.n_iter_ == imputer.max_iter


def test_iterative_imputer_catch_warning():
    # check that we catch a RuntimeWarning due to a division by zero when a
    # feature is constant in the dataset
    X, y = load_diabetes(return_X_y=True)
    n_samples, n_features = X.shape

    # simulate that a feature only contain one category during fit
    X[:, 3] = 1

    # add some missing values
    rng = np.random.RandomState(0)
    missing_rate = 0.15
    for feat in range(n_features):
        sample_idx = rng.choice(
            np.arange(n_samples), size=int(n_samples * missing_rate), replace=False
        )
        X[sample_idx, feat] = np.nan

    imputer = IterativeImputer(n_nearest_features=5, sample_posterior=True)
    with pytest.warns(None) as record:
        X_fill = imputer.fit_transform(X, y)
    assert not [w.message for w in record.list]
    assert not np.any(np.isnan(X_fill))


@pytest.mark.parametrize(
    "min_value, max_value, correct_output",
    [
        (0, 100, np.array([[0] * 3, [100] * 3])),
        (None, None, np.array([[-np.inf] * 3, [np.inf] * 3])),
        (-np.inf, np.inf, np.array([[-np.inf] * 3, [np.inf] * 3])),
        ([-5, 5, 10], [100, 200, 300], np.array([[-5, 5, 10], [100, 200, 300]])),
        (
            [-5, -np.inf, 10],
            [100, 200, np.inf],
            np.array([[-5, -np.inf, 10], [100, 200, np.inf]]),
        ),
    ],
    ids=["scalars", "None-default", "inf", "lists", "lists-with-inf"],
)
def test_iterative_imputer_min_max_array_like(min_value, max_value, correct_output):
    # check that passing scalar or array-like
    # for min_value and max_value in IterativeImputer works
    X = np.random.RandomState(0).randn(10, 3)
    imputer = IterativeImputer(min_value=min_value, max_value=max_value)
    imputer.fit(X)

    assert isinstance(imputer._min_value, np.ndarray) and isinstance(
        imputer._max_value, np.ndarray
    )
    assert (imputer._min_value.shape[0] == X.shape[1]) and (
        imputer._max_value.shape[0] == X.shape[1]
    )

    assert_allclose(correct_output[0, :], imputer._min_value)
    assert_allclose(correct_output[1, :], imputer._max_value)


@pytest.mark.parametrize(
    "min_value, max_value, err_msg",
    [
        (100, 0, "min_value >= max_value."),
        (np.inf, -np.inf, "min_value >= max_value."),
        ([-5, 5], [100, 200, 0], "_value' should be of shape"),
    ],
)
def test_iterative_imputer_catch_min_max_error(min_value, max_value, err_msg):
    # check that passing scalar or array-like
    # for min_value and max_value in IterativeImputer works
    X = np.random.random((10, 3))
    imputer = IterativeImputer(min_value=min_value, max_value=max_value)
    with pytest.raises(ValueError, match=err_msg):
        imputer.fit(X)


@pytest.mark.parametrize(
    "min_max_1, min_max_2",
    [([None, None], [-np.inf, np.inf]), ([-10, 10], [[-10] * 4, [10] * 4])],
    ids=["None-vs-inf", "Scalar-vs-vector"],
)
def test_iterative_imputer_min_max_array_like_imputation(min_max_1, min_max_2):
    # Test that None/inf and scalar/vector give the same imputation
    X_train = np.array(
        [
            [np.nan, 2, 2, 1],
            [10, np.nan, np.nan, 7],
            [3, 1, np.nan, 1],
            [np.nan, 4, 2, np.nan],
        ]
    )
    X_test = np.array(
        [[np.nan, 2, np.nan, 5], [2, 4, np.nan, np.nan], [np.nan, 1, 10, 1]]
    )
    imputer1 = IterativeImputer(
        min_value=min_max_1[0], max_value=min_max_1[1], random_state=0
    )
    imputer2 = IterativeImputer(
        min_value=min_max_2[0], max_value=min_max_2[1], random_state=0
    )
    X_test_imputed1 = imputer1.fit(X_train).transform(X_test)
    X_test_imputed2 = imputer2.fit(X_train).transform(X_test)
    assert_allclose(X_test_imputed1[:, 0], X_test_imputed2[:, 0])


@pytest.mark.parametrize("skip_complete", [True, False])
def test_iterative_imputer_skip_non_missing(skip_complete):
    # check the imputing strategy when missing data are present in the
    # testing set only.
    # taken from: https://github.com/scikit-learn/scikit-learn/issues/14383
    rng = np.random.RandomState(0)
    X_train = np.array([[5, 2, 2, 1], [10, 1, 2, 7], [3, 1, 1, 1], [8, 4, 2, 2]])
    X_test = np.array([[np.nan, 2, 4, 5], [np.nan, 4, 1, 2], [np.nan, 1, 10, 1]])
    imputer = IterativeImputer(
        initial_strategy="mean", skip_complete=skip_complete, random_state=rng
    )
    X_test_est = imputer.fit(X_train).transform(X_test)
    if skip_complete:
        # impute with the initial strategy: 'mean'
        assert_allclose(X_test_est[:, 0], np.mean(X_train[:, 0]))
    else:
        assert_allclose(X_test_est[:, 0], [11, 7, 12], rtol=1e-4)


@pytest.mark.parametrize("rs_imputer", [None, 1, np.random.RandomState(seed=1)])
@pytest.mark.parametrize("rs_estimator", [None, 1, np.random.RandomState(seed=1)])
def test_iterative_imputer_dont_set_random_state(rs_imputer, rs_estimator):
    class ZeroEstimator:
        def __init__(self, random_state):
            self.random_state = random_state

        def fit(self, *args, **kgards):
            return self

        def predict(self, X):
            return np.zeros(X.shape[0])

    estimator = ZeroEstimator(random_state=rs_estimator)
    imputer = IterativeImputer(random_state=rs_imputer)
    X_train = np.zeros((10, 3))
    imputer.fit(X_train)
    assert estimator.random_state == rs_estimator


@pytest.mark.parametrize(
    "X_fit, X_trans, params, msg_err",
    [
        (
            np.array([[-1, 1], [1, 2]]),
            np.array([[-1, 1], [1, -1]]),
            {"features": "missing-only", "sparse": "auto"},
            "have missing values in transform but have no missing values in fit",
        ),
        (
            np.array([[-1, 1], [1, 2]]),
            np.array([[-1, 1], [1, 2]]),
            {"features": "random", "sparse": "auto"},
            "'features' has to be either 'missing-only' or 'all'",
        ),
        (
            np.array([[-1, 1], [1, 2]]),
            np.array([[-1, 1], [1, 2]]),
            {"features": "all", "sparse": "random"},
            "'sparse' has to be a boolean or 'auto'",
        ),
        (
            np.array([["a", "b"], ["c", "a"]], dtype=str),
            np.array([["a", "b"], ["c", "a"]], dtype=str),
            {},
            "MissingIndicator does not support data with dtype",
        ),
    ],
)
def test_missing_indicator_error(X_fit, X_trans, params, msg_err):
    indicator = MissingIndicator(missing_values=-1)
    indicator.set_params(**params)
    with pytest.raises(ValueError, match=msg_err):
        indicator.fit(X_fit).transform(X_trans)


@pytest.mark.parametrize(
    "missing_values, dtype, arr_type",
    [
        (np.nan, np.float64, np.array),
        (0, np.int32, np.array),
        (-1, np.int32, np.array),
        (np.nan, np.float64, sparse.csc_matrix),
        (-1, np.int32, sparse.csc_matrix),
        (np.nan, np.float64, sparse.csr_matrix),
        (-1, np.int32, sparse.csr_matrix),
        (np.nan, np.float64, sparse.coo_matrix),
        (-1, np.int32, sparse.coo_matrix),
        (np.nan, np.float64, sparse.lil_matrix),
        (-1, np.int32, sparse.lil_matrix),
        (np.nan, np.float64, sparse.bsr_matrix),
        (-1, np.int32, sparse.bsr_matrix),
    ],
)
@pytest.mark.parametrize(
    "param_features, n_features, features_indices",
    [("missing-only", 3, np.array([0, 1, 2])), ("all", 3, np.array([0, 1, 2]))],
)
def test_missing_indicator_new(
    missing_values, arr_type, dtype, param_features, n_features, features_indices
):
    X_fit = np.array([[missing_values, missing_values, 1], [4, 2, missing_values]])
    X_trans = np.array([[missing_values, missing_values, 1], [4, 12, 10]])
    X_fit_expected = np.array([[1, 1, 0], [0, 0, 1]])
    X_trans_expected = np.array([[1, 1, 0], [0, 0, 0]])

    # convert the input to the right array format and right dtype
    X_fit = arr_type(X_fit).astype(dtype)
    X_trans = arr_type(X_trans).astype(dtype)
    X_fit_expected = X_fit_expected.astype(dtype)
    X_trans_expected = X_trans_expected.astype(dtype)

    indicator = MissingIndicator(
        missing_values=missing_values, features=param_features, sparse=False
    )
    X_fit_mask = indicator.fit_transform(X_fit)
    X_trans_mask = indicator.transform(X_trans)

    assert X_fit_mask.shape[1] == n_features
    assert X_trans_mask.shape[1] == n_features

    assert_array_equal(indicator.features_, features_indices)
    assert_allclose(X_fit_mask, X_fit_expected[:, features_indices])
    assert_allclose(X_trans_mask, X_trans_expected[:, features_indices])

    assert X_fit_mask.dtype == bool
    assert X_trans_mask.dtype == bool
    assert isinstance(X_fit_mask, np.ndarray)
    assert isinstance(X_trans_mask, np.ndarray)

    indicator.set_params(sparse=True)
    X_fit_mask_sparse = indicator.fit_transform(X_fit)
    X_trans_mask_sparse = indicator.transform(X_trans)

    assert X_fit_mask_sparse.dtype == bool
    assert X_trans_mask_sparse.dtype == bool
    assert X_fit_mask_sparse.format == "csc"
    assert X_trans_mask_sparse.format == "csc"
    assert_allclose(X_fit_mask_sparse.toarray(), X_fit_mask)
    assert_allclose(X_trans_mask_sparse.toarray(), X_trans_mask)


@pytest.mark.parametrize(
    "arr_type",
    [
        sparse.csc_matrix,
        sparse.csr_matrix,
        sparse.coo_matrix,
        sparse.lil_matrix,
        sparse.bsr_matrix,
    ],
)
def test_missing_indicator_raise_on_sparse_with_missing_0(arr_type):
    # test for sparse input and missing_value == 0

    missing_values = 0
    X_fit = np.array([[missing_values, missing_values, 1], [4, missing_values, 2]])
    X_trans = np.array([[missing_values, missing_values, 1], [4, 12, 10]])

    # convert the input to the right array format
    X_fit_sparse = arr_type(X_fit)
    X_trans_sparse = arr_type(X_trans)

    indicator = MissingIndicator(missing_values=missing_values)

    with pytest.raises(ValueError, match="Sparse input with missing_values=0"):
        indicator.fit_transform(X_fit_sparse)

    indicator.fit_transform(X_fit)
    with pytest.raises(ValueError, match="Sparse input with missing_values=0"):
        indicator.transform(X_trans_sparse)


@pytest.mark.parametrize("param_sparse", [True, False, "auto"])
@pytest.mark.parametrize(
    "missing_values, arr_type",
    [
        (np.nan, np.array),
        (0, np.array),
        (np.nan, sparse.csc_matrix),
        (np.nan, sparse.csr_matrix),
        (np.nan, sparse.coo_matrix),
        (np.nan, sparse.lil_matrix),
    ],
)
def test_missing_indicator_sparse_param(arr_type, missing_values, param_sparse):
    # check the format of the output with different sparse parameter
    X_fit = np.array([[missing_values, missing_values, 1], [4, missing_values, 2]])
    X_trans = np.array([[missing_values, missing_values, 1], [4, 12, 10]])
    X_fit = arr_type(X_fit).astype(np.float64)
    X_trans = arr_type(X_trans).astype(np.float64)

    indicator = MissingIndicator(missing_values=missing_values, sparse=param_sparse)
    X_fit_mask = indicator.fit_transform(X_fit)
    X_trans_mask = indicator.transform(X_trans)

    if param_sparse is True:
        assert X_fit_mask.format == "csc"
        assert X_trans_mask.format == "csc"
    elif param_sparse == "auto" and missing_values == 0:
        assert isinstance(X_fit_mask, np.ndarray)
        assert isinstance(X_trans_mask, np.ndarray)
    elif param_sparse is False:
        assert isinstance(X_fit_mask, np.ndarray)
        assert isinstance(X_trans_mask, np.ndarray)
    else:
        if sparse.issparse(X_fit):
            assert X_fit_mask.format == "csc"
            assert X_trans_mask.format == "csc"
        else:
            assert isinstance(X_fit_mask, np.ndarray)
            assert isinstance(X_trans_mask, np.ndarray)


def test_missing_indicator_string():
    X = np.array([["a", "b", "c"], ["b", "c", "a"]], dtype=object)
    indicator = MissingIndicator(missing_values="a", features="all")
    X_trans = indicator.fit_transform(X)
    assert_array_equal(X_trans, np.array([[True, False, False], [False, False, True]]))


@pytest.mark.parametrize(
    "X, missing_values, X_trans_exp",
    [
        (
            np.array([["a", "b"], ["b", "a"]], dtype=object),
            "a",
            np.array([["b", "b", True, False], ["b", "b", False, True]], dtype=object),
        ),
        (
            np.array([[np.nan, 1.0], [1.0, np.nan]]),
            np.nan,
            np.array([[1.0, 1.0, True, False], [1.0, 1.0, False, True]]),
        ),
        (
            np.array([[np.nan, "b"], ["b", np.nan]], dtype=object),
            np.nan,
            np.array([["b", "b", True, False], ["b", "b", False, True]], dtype=object),
        ),
        (
            np.array([[None, "b"], ["b", None]], dtype=object),
            None,
            np.array([["b", "b", True, False], ["b", "b", False, True]], dtype=object),
        ),
    ],
)
def test_missing_indicator_with_imputer(X, missing_values, X_trans_exp):
    trans = make_union(
        SimpleImputer(missing_values=missing_values, strategy="most_frequent"),
        MissingIndicator(missing_values=missing_values),
    )
    X_trans = trans.fit_transform(X)
    assert_array_equal(X_trans, X_trans_exp)


@pytest.mark.parametrize("imputer_constructor", [SimpleImputer, IterativeImputer])
@pytest.mark.parametrize(
    "imputer_missing_values, missing_value, err_msg",
    [
        ("NaN", np.nan, "Input contains NaN"),
        ("-1", -1, "types are expected to be both numerical."),
    ],
)
def test_inconsistent_dtype_X_missing_values(
    imputer_constructor, imputer_missing_values, missing_value, err_msg
):
    # regression test for issue #11390. Comparison between incoherent dtype
    # for X and missing_values was not raising a proper error.
    rng = np.random.RandomState(42)
    X = rng.randn(10, 10)
    X[0, 0] = missing_value

    imputer = imputer_constructor(missing_values=imputer_missing_values)

    with pytest.raises(ValueError, match=err_msg):
        imputer.fit_transform(X)


def test_missing_indicator_no_missing():
    # check that all features are dropped if there are no missing values when
    # features='missing-only' (#13491)
    X = np.array([[1, 1], [1, 1]])

    mi = MissingIndicator(features="missing-only", missing_values=-1)
    Xt = mi.fit_transform(X)

    assert Xt.shape[1] == 0


def test_missing_indicator_sparse_no_explicit_zeros():
    # Check that non missing values don't become explicit zeros in the mask
    # generated by missing indicator when X is sparse. (#13491)
    X = sparse.csr_matrix([[0, 1, 2], [1, 2, 0], [2, 0, 1]])

    mi = MissingIndicator(features="all", missing_values=1)
    Xt = mi.fit_transform(X)

    assert Xt.getnnz() == Xt.sum()


@pytest.mark.parametrize("imputer_constructor", [SimpleImputer, IterativeImputer])
def test_imputer_without_indicator(imputer_constructor):
    X = np.array([[1, 1], [1, 1]])
    imputer = imputer_constructor()
    imputer.fit(X)

    assert imputer.indicator_ is None


@pytest.mark.parametrize(
    "arr_type",
    [
        sparse.csc_matrix,
        sparse.csr_matrix,
        sparse.coo_matrix,
        sparse.lil_matrix,
        sparse.bsr_matrix,
    ],
)
def test_simple_imputation_add_indicator_sparse_matrix(arr_type):
    X_sparse = arr_type([[np.nan, 1, 5], [2, np.nan, 1], [6, 3, np.nan], [1, 2, 9]])
    X_true = np.array(
        [
            [3.0, 1.0, 5.0, 1.0, 0.0, 0.0],
            [2.0, 2.0, 1.0, 0.0, 1.0, 0.0],
            [6.0, 3.0, 5.0, 0.0, 0.0, 1.0],
            [1.0, 2.0, 9.0, 0.0, 0.0, 0.0],
        ]
    )

    imputer = SimpleImputer(missing_values=np.nan, add_indicator=True)
    X_trans = imputer.fit_transform(X_sparse)

    assert sparse.issparse(X_trans)
    assert X_trans.shape == X_true.shape
    assert_allclose(X_trans.toarray(), X_true)


@pytest.mark.parametrize(
    "strategy, expected", [("most_frequent", "b"), ("constant", "missing_value")]
)
def test_simple_imputation_string_list(strategy, expected):
    X = [["a", "b"], ["c", np.nan]]

    X_true = np.array([["a", "b"], ["c", expected]], dtype=object)

    imputer = SimpleImputer(strategy=strategy)
    X_trans = imputer.fit_transform(X)

    assert_array_equal(X_trans, X_true)


@pytest.mark.parametrize(
    "order, idx_order",
    [("ascending", [3, 4, 2, 0, 1]), ("descending", [1, 0, 2, 4, 3])],
)
def test_imputation_order(order, idx_order):
    # regression test for #15393
    rng = np.random.RandomState(42)
    X = rng.rand(100, 5)
    X[:50, 1] = np.nan
    X[:30, 0] = np.nan
    X[:20, 2] = np.nan
    X[:10, 4] = np.nan

    with pytest.warns(ConvergenceWarning):
        trs = IterativeImputer(max_iter=1, imputation_order=order, random_state=0).fit(
            X
        )
        idx = [x.feat_idx for x in trs.imputation_sequence_]
        assert idx == idx_order


@pytest.mark.parametrize("missing_value", [-1, np.nan])
def test_simple_imputation_inverse_transform(missing_value):
    # Test inverse_transform feature for np.nan
    X_1 = np.array(
        [
            [9, missing_value, 3, -1],
            [4, -1, 5, 4],
            [6, 7, missing_value, -1],
            [8, 9, 0, missing_value],
        ]
    )

    X_2 = np.array(
        [
            [5, 4, 2, 1],
            [2, 1, missing_value, 3],
            [9, missing_value, 7, 1],
            [6, 4, 2, missing_value],
        ]
    )

    X_3 = np.array(
        [
            [1, missing_value, 5, 9],
            [missing_value, 4, missing_value, missing_value],
            [2, missing_value, 7, missing_value],
            [missing_value, 3, missing_value, 8],
        ]
    )

    X_4 = np.array(
        [
            [1, 1, 1, 3],
            [missing_value, 2, missing_value, 1],
            [2, 3, 3, 4],
            [missing_value, 4, missing_value, 2],
        ]
    )

    imputer = SimpleImputer(
        missing_values=missing_value, strategy="mean", add_indicator=True
    )

    X_1_trans = imputer.fit_transform(X_1)
    X_1_inv_trans = imputer.inverse_transform(X_1_trans)

    X_2_trans = imputer.transform(X_2)  # test on new data
    X_2_inv_trans = imputer.inverse_transform(X_2_trans)

    assert_array_equal(X_1_inv_trans, X_1)
    assert_array_equal(X_2_inv_trans, X_2)

    for X in [X_3, X_4]:
        X_trans = imputer.fit_transform(X)
        X_inv_trans = imputer.inverse_transform(X_trans)
        assert_array_equal(X_inv_trans, X)


@pytest.mark.parametrize("missing_value", [-1, np.nan])
def test_simple_imputation_inverse_transform_exceptions(missing_value):
    X_1 = np.array(
        [
            [9, missing_value, 3, -1],
            [4, -1, 5, 4],
            [6, 7, missing_value, -1],
            [8, 9, 0, missing_value],
        ]
    )

    imputer = SimpleImputer(missing_values=missing_value, strategy="mean")
    X_1_trans = imputer.fit_transform(X_1)
    with pytest.raises(
        ValueError, match=f"Got 'add_indicator={imputer.add_indicator}'"
    ):
        imputer.inverse_transform(X_1_trans)


@pytest.mark.parametrize(
    "expected,array,dtype,extra_value,n_repeat",
    [
        # array of object dtype
        ("extra_value", ["a", "b", "c"], object, "extra_value", 2),
        (
            "most_frequent_value",
            ["most_frequent_value", "most_frequent_value", "value"],
            object,
            "extra_value",
            1,
        ),
        ("a", ["min_value", "min_valuevalue"], object, "a", 2),
        ("min_value", ["min_value", "min_value", "value"], object, "z", 2),
        # array of numeric dtype
        (10, [1, 2, 3], int, 10, 2),
        (1, [1, 1, 2], int, 10, 1),
        (10, [20, 20, 1], int, 10, 2),
        (1, [1, 1, 20], int, 10, 2),
    ],
)
def test_most_frequent(expected, array, dtype, extra_value, n_repeat):
    assert expected == _most_frequent(
        np.array(array, dtype=dtype), extra_value, n_repeat
    )
