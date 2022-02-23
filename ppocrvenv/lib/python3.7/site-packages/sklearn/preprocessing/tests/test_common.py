import warnings

import pytest
import numpy as np

from scipy import sparse

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from sklearn.base import clone

from sklearn.preprocessing import maxabs_scale
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import scale
from sklearn.preprocessing import power_transform
from sklearn.preprocessing import quantile_transform
from sklearn.preprocessing import robust_scale

from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import RobustScaler

from sklearn.utils._testing import assert_array_equal
from sklearn.utils._testing import assert_allclose

iris = load_iris()


def _get_valid_samples_by_column(X, col):
    """Get non NaN samples in column of X"""
    return X[:, [col]][~np.isnan(X[:, col])]


@pytest.mark.parametrize(
    "est, func, support_sparse, strictly_positive, omit_kwargs",
    [
        (MaxAbsScaler(), maxabs_scale, True, False, []),
        (MinMaxScaler(), minmax_scale, False, False, ["clip"]),
        (StandardScaler(), scale, False, False, []),
        (StandardScaler(with_mean=False), scale, True, False, []),
        (PowerTransformer("yeo-johnson"), power_transform, False, False, []),
        (PowerTransformer("box-cox"), power_transform, False, True, []),
        (QuantileTransformer(n_quantiles=10), quantile_transform, True, False, []),
        (RobustScaler(), robust_scale, False, False, []),
        (RobustScaler(with_centering=False), robust_scale, True, False, []),
    ],
)
def test_missing_value_handling(
    est, func, support_sparse, strictly_positive, omit_kwargs
):
    # check that the preprocessing method let pass nan
    rng = np.random.RandomState(42)
    X = iris.data.copy()
    n_missing = 50
    X[
        rng.randint(X.shape[0], size=n_missing), rng.randint(X.shape[1], size=n_missing)
    ] = np.nan
    if strictly_positive:
        X += np.nanmin(X) + 0.1
    X_train, X_test = train_test_split(X, random_state=1)
    # sanity check
    assert not np.all(np.isnan(X_train), axis=0).any()
    assert np.any(np.isnan(X_train), axis=0).all()
    assert np.any(np.isnan(X_test), axis=0).all()
    X_test[:, 0] = np.nan  # make sure this boundary case is tested

    with pytest.warns(None) as records:
        Xt = est.fit(X_train).transform(X_test)
    # ensure no warnings are raised
    assert len(records) == 0
    # missing values should still be missing, and only them
    assert_array_equal(np.isnan(Xt), np.isnan(X_test))

    # check that the function leads to the same results as the class
    with pytest.warns(None) as records:
        Xt_class = est.transform(X_train)
    assert len(records) == 0
    kwargs = est.get_params()
    # remove the parameters which should be omitted because they
    # are not defined in the sister function of the preprocessing class
    for kwarg in omit_kwargs:
        _ = kwargs.pop(kwarg)
    Xt_func = func(X_train, **kwargs)
    assert_array_equal(np.isnan(Xt_func), np.isnan(Xt_class))
    assert_allclose(Xt_func[~np.isnan(Xt_func)], Xt_class[~np.isnan(Xt_class)])

    # check that the inverse transform keep NaN
    Xt_inv = est.inverse_transform(Xt)
    assert_array_equal(np.isnan(Xt_inv), np.isnan(X_test))
    # FIXME: we can introduce equal_nan=True in recent version of numpy.
    # For the moment which just check that non-NaN values are almost equal.
    assert_allclose(Xt_inv[~np.isnan(Xt_inv)], X_test[~np.isnan(X_test)])

    for i in range(X.shape[1]):
        # train only on non-NaN
        est.fit(_get_valid_samples_by_column(X_train, i))
        # check transforming with NaN works even when training without NaN
        with pytest.warns(None) as records:
            Xt_col = est.transform(X_test[:, [i]])
        assert len(records) == 0
        assert_allclose(Xt_col, Xt[:, [i]])
        # check non-NaN is handled as before - the 1st column is all nan
        if not np.isnan(X_test[:, i]).all():
            Xt_col_nonan = est.transform(_get_valid_samples_by_column(X_test, i))
            assert_array_equal(Xt_col_nonan, Xt_col[~np.isnan(Xt_col.squeeze())])

    if support_sparse:
        est_dense = clone(est)
        est_sparse = clone(est)

        with pytest.warns(None) as records:
            Xt_dense = est_dense.fit(X_train).transform(X_test)
            Xt_inv_dense = est_dense.inverse_transform(Xt_dense)
        assert len(records) == 0
        for sparse_constructor in (
            sparse.csr_matrix,
            sparse.csc_matrix,
            sparse.bsr_matrix,
            sparse.coo_matrix,
            sparse.dia_matrix,
            sparse.dok_matrix,
            sparse.lil_matrix,
        ):
            # check that the dense and sparse inputs lead to the same results
            # precompute the matrix to avoid catching side warnings
            X_train_sp = sparse_constructor(X_train)
            X_test_sp = sparse_constructor(X_test)
            with pytest.warns(None) as records:
                warnings.simplefilter("ignore", PendingDeprecationWarning)
                Xt_sp = est_sparse.fit(X_train_sp).transform(X_test_sp)
            assert len(records) == 0
            assert_allclose(Xt_sp.A, Xt_dense)
            with pytest.warns(None) as records:
                warnings.simplefilter("ignore", PendingDeprecationWarning)
                Xt_inv_sp = est_sparse.inverse_transform(Xt_sp)
            assert len(records) == 0
            assert_allclose(Xt_inv_sp.A, Xt_inv_dense)


@pytest.mark.parametrize(
    "est, func",
    [
        (MaxAbsScaler(), maxabs_scale),
        (MinMaxScaler(), minmax_scale),
        (StandardScaler(), scale),
        (StandardScaler(with_mean=False), scale),
        (PowerTransformer("yeo-johnson"), power_transform),
        (
            PowerTransformer("box-cox"),
            power_transform,
        ),
        (QuantileTransformer(n_quantiles=3), quantile_transform),
        (RobustScaler(), robust_scale),
        (RobustScaler(with_centering=False), robust_scale),
    ],
)
def test_missing_value_pandas_na_support(est, func):
    # Test pandas IntegerArray with pd.NA
    pd = pytest.importorskip("pandas", minversion="1.0")

    X = np.array(
        [
            [1, 2, 3, np.nan, np.nan, 4, 5, 1],
            [np.nan, np.nan, 8, 4, 6, np.nan, np.nan, 8],
            [1, 2, 3, 4, 5, 6, 7, 8],
        ]
    ).T

    # Creates dataframe with IntegerArrays with pd.NA
    X_df = pd.DataFrame(X, dtype="Int16", columns=["a", "b", "c"])
    X_df["c"] = X_df["c"].astype("int")

    X_trans = est.fit_transform(X)
    X_df_trans = est.fit_transform(X_df)

    assert_allclose(X_trans, X_df_trans)
