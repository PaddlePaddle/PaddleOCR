import pytest
import numpy as np
from scipy import sparse

from sklearn.preprocessing import FunctionTransformer
from sklearn.utils._testing import assert_array_equal, assert_allclose_dense_sparse


def _make_func(args_store, kwargs_store, func=lambda X, *a, **k: X):
    def _func(X, *args, **kwargs):
        args_store.append(X)
        args_store.extend(args)
        kwargs_store.update(kwargs)
        return func(X)

    return _func


def test_delegate_to_func():
    # (args|kwargs)_store will hold the positional and keyword arguments
    # passed to the function inside the FunctionTransformer.
    args_store = []
    kwargs_store = {}
    X = np.arange(10).reshape((5, 2))
    assert_array_equal(
        FunctionTransformer(_make_func(args_store, kwargs_store)).transform(X),
        X,
        "transform should have returned X unchanged",
    )

    # The function should only have received X.
    assert args_store == [
        X
    ], "Incorrect positional arguments passed to func: {args}".format(args=args_store)

    assert (
        not kwargs_store
    ), "Unexpected keyword arguments passed to func: {args}".format(args=kwargs_store)

    # reset the argument stores.
    args_store[:] = []
    kwargs_store.clear()
    transformed = FunctionTransformer(
        _make_func(args_store, kwargs_store),
    ).transform(X)

    assert_array_equal(
        transformed, X, err_msg="transform should have returned X unchanged"
    )

    # The function should have received X
    assert args_store == [
        X
    ], "Incorrect positional arguments passed to func: {args}".format(args=args_store)

    assert (
        not kwargs_store
    ), "Unexpected keyword arguments passed to func: {args}".format(args=kwargs_store)


def test_np_log():
    X = np.arange(10).reshape((5, 2))

    # Test that the numpy.log example still works.
    assert_array_equal(
        FunctionTransformer(np.log1p).transform(X),
        np.log1p(X),
    )


def test_kw_arg():
    X = np.linspace(0, 1, num=10).reshape((5, 2))

    F = FunctionTransformer(np.around, kw_args=dict(decimals=3))

    # Test that rounding is correct
    assert_array_equal(F.transform(X), np.around(X, decimals=3))


def test_kw_arg_update():
    X = np.linspace(0, 1, num=10).reshape((5, 2))

    F = FunctionTransformer(np.around, kw_args=dict(decimals=3))

    F.kw_args["decimals"] = 1

    # Test that rounding is correct
    assert_array_equal(F.transform(X), np.around(X, decimals=1))


def test_kw_arg_reset():
    X = np.linspace(0, 1, num=10).reshape((5, 2))

    F = FunctionTransformer(np.around, kw_args=dict(decimals=3))

    F.kw_args = dict(decimals=1)

    # Test that rounding is correct
    assert_array_equal(F.transform(X), np.around(X, decimals=1))


def test_inverse_transform():
    X = np.array([1, 4, 9, 16]).reshape((2, 2))

    # Test that inverse_transform works correctly
    F = FunctionTransformer(
        func=np.sqrt,
        inverse_func=np.around,
        inv_kw_args=dict(decimals=3),
    )
    assert_array_equal(
        F.inverse_transform(F.transform(X)),
        np.around(np.sqrt(X), decimals=3),
    )


def test_check_inverse():
    X_dense = np.array([1, 4, 9, 16], dtype=np.float64).reshape((2, 2))

    X_list = [X_dense, sparse.csr_matrix(X_dense), sparse.csc_matrix(X_dense)]

    for X in X_list:
        if sparse.issparse(X):
            accept_sparse = True
        else:
            accept_sparse = False
        trans = FunctionTransformer(
            func=np.sqrt,
            inverse_func=np.around,
            accept_sparse=accept_sparse,
            check_inverse=True,
            validate=True,
        )
        warning_message = (
            "The provided functions are not strictly"
            " inverse of each other. If you are sure you"
            " want to proceed regardless, set"
            " 'check_inverse=False'."
        )
        with pytest.warns(UserWarning, match=warning_message):
            trans.fit(X)

        trans = FunctionTransformer(
            func=np.expm1,
            inverse_func=np.log1p,
            accept_sparse=accept_sparse,
            check_inverse=True,
            validate=True,
        )
        with pytest.warns(None) as record:
            Xt = trans.fit_transform(X)
        assert len(record) == 0
        assert_allclose_dense_sparse(X, trans.inverse_transform(Xt))

    # check that we don't check inverse when one of the func or inverse is not
    # provided.
    trans = FunctionTransformer(
        func=np.expm1, inverse_func=None, check_inverse=True, validate=True
    )
    with pytest.warns(None) as record:
        trans.fit(X_dense)
    assert len(record) == 0
    trans = FunctionTransformer(
        func=None, inverse_func=np.expm1, check_inverse=True, validate=True
    )
    with pytest.warns(None) as record:
        trans.fit(X_dense)
    assert len(record) == 0


def test_function_transformer_frame():
    pd = pytest.importorskip("pandas")
    X_df = pd.DataFrame(np.random.randn(100, 10))
    transformer = FunctionTransformer()
    X_df_trans = transformer.fit_transform(X_df)
    assert hasattr(X_df_trans, "loc")


def test_function_transformer_validate_inverse():
    """Test that function transformer does not reset estimator in
    `inverse_transform`."""

    def add_constant_feature(X):
        X_one = np.ones((X.shape[0], 1))
        return np.concatenate((X, X_one), axis=1)

    def inverse_add_constant(X):
        return X[:, :-1]

    X = np.array([[1, 2], [3, 4], [3, 4]])
    trans = FunctionTransformer(
        func=add_constant_feature,
        inverse_func=inverse_add_constant,
        validate=True,
    )
    X_trans = trans.fit_transform(X)
    assert trans.n_features_in_ == X.shape[1]

    trans.inverse_transform(X_trans)
    assert trans.n_features_in_ == X.shape[1]
