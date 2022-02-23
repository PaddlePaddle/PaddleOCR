import numpy as np
import pytest

from sklearn.base import clone
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

from sklearn.dummy import DummyRegressor

from sklearn.utils._testing import assert_allclose
from sklearn.utils._testing import assert_no_warnings

from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression, OrthogonalMatchingPursuit

from sklearn import datasets

from sklearn.compose import TransformedTargetRegressor

friedman = datasets.make_friedman1(random_state=0)


def test_transform_target_regressor_error():
    X, y = friedman
    # provide a transformer and functions at the same time
    regr = TransformedTargetRegressor(
        regressor=LinearRegression(),
        transformer=StandardScaler(),
        func=np.exp,
        inverse_func=np.log,
    )
    with pytest.raises(
        ValueError,
        match="'transformer' and functions 'func'/'inverse_func' cannot both be set.",
    ):
        regr.fit(X, y)
    # fit with sample_weight with a regressor which does not support it
    sample_weight = np.ones((y.shape[0],))
    regr = TransformedTargetRegressor(
        regressor=OrthogonalMatchingPursuit(), transformer=StandardScaler()
    )
    with pytest.raises(
        TypeError,
        match=r"fit\(\) got an unexpected " "keyword argument 'sample_weight'",
    ):
        regr.fit(X, y, sample_weight=sample_weight)
    # func is given but inverse_func is not
    regr = TransformedTargetRegressor(func=np.exp)
    with pytest.raises(
        ValueError,
        match="When 'func' is provided, 'inverse_func' must also be provided",
    ):
        regr.fit(X, y)


def test_transform_target_regressor_invertible():
    X, y = friedman
    regr = TransformedTargetRegressor(
        regressor=LinearRegression(),
        func=np.sqrt,
        inverse_func=np.log,
        check_inverse=True,
    )
    with pytest.warns(
        UserWarning,
        match=(
            "The provided functions or"
            " transformer are not strictly inverse of each other."
        ),
    ):
        regr.fit(X, y)
    regr = TransformedTargetRegressor(
        regressor=LinearRegression(), func=np.sqrt, inverse_func=np.log
    )
    regr.set_params(check_inverse=False)
    assert_no_warnings(regr.fit, X, y)


def _check_standard_scaled(y, y_pred):
    y_mean = np.mean(y, axis=0)
    y_std = np.std(y, axis=0)
    assert_allclose((y - y_mean) / y_std, y_pred)


def _check_shifted_by_one(y, y_pred):
    assert_allclose(y + 1, y_pred)


def test_transform_target_regressor_functions():
    X, y = friedman
    regr = TransformedTargetRegressor(
        regressor=LinearRegression(), func=np.log, inverse_func=np.exp
    )
    y_pred = regr.fit(X, y).predict(X)
    # check the transformer output
    y_tran = regr.transformer_.transform(y.reshape(-1, 1)).squeeze()
    assert_allclose(np.log(y), y_tran)
    assert_allclose(
        y, regr.transformer_.inverse_transform(y_tran.reshape(-1, 1)).squeeze()
    )
    assert y.shape == y_pred.shape
    assert_allclose(y_pred, regr.inverse_func(regr.regressor_.predict(X)))
    # check the regressor output
    lr = LinearRegression().fit(X, regr.func(y))
    assert_allclose(regr.regressor_.coef_.ravel(), lr.coef_.ravel())


def test_transform_target_regressor_functions_multioutput():
    X = friedman[0]
    y = np.vstack((friedman[1], friedman[1] ** 2 + 1)).T
    regr = TransformedTargetRegressor(
        regressor=LinearRegression(), func=np.log, inverse_func=np.exp
    )
    y_pred = regr.fit(X, y).predict(X)
    # check the transformer output
    y_tran = regr.transformer_.transform(y)
    assert_allclose(np.log(y), y_tran)
    assert_allclose(y, regr.transformer_.inverse_transform(y_tran))
    assert y.shape == y_pred.shape
    assert_allclose(y_pred, regr.inverse_func(regr.regressor_.predict(X)))
    # check the regressor output
    lr = LinearRegression().fit(X, regr.func(y))
    assert_allclose(regr.regressor_.coef_.ravel(), lr.coef_.ravel())


@pytest.mark.parametrize(
    "X,y", [friedman, (friedman[0], np.vstack((friedman[1], friedman[1] ** 2 + 1)).T)]
)
def test_transform_target_regressor_1d_transformer(X, y):
    # All transformer in scikit-learn expect 2D data. FunctionTransformer with
    # validate=False lift this constraint without checking that the input is a
    # 2D vector. We check the consistency of the data shape using a 1D and 2D y
    # array.
    transformer = FunctionTransformer(
        func=lambda x: x + 1, inverse_func=lambda x: x - 1
    )
    regr = TransformedTargetRegressor(
        regressor=LinearRegression(), transformer=transformer
    )
    y_pred = regr.fit(X, y).predict(X)
    assert y.shape == y_pred.shape
    # consistency forward transform
    y_tran = regr.transformer_.transform(y)
    _check_shifted_by_one(y, y_tran)
    assert y.shape == y_pred.shape
    # consistency inverse transform
    assert_allclose(y, regr.transformer_.inverse_transform(y_tran).squeeze())
    # consistency of the regressor
    lr = LinearRegression()
    transformer2 = clone(transformer)
    lr.fit(X, transformer2.fit_transform(y))
    y_lr_pred = lr.predict(X)
    assert_allclose(y_pred, transformer2.inverse_transform(y_lr_pred))
    assert_allclose(regr.regressor_.coef_, lr.coef_)


@pytest.mark.parametrize(
    "X,y", [friedman, (friedman[0], np.vstack((friedman[1], friedman[1] ** 2 + 1)).T)]
)
def test_transform_target_regressor_2d_transformer(X, y):
    # Check consistency with transformer accepting only 2D array and a 1D/2D y
    # array.
    transformer = StandardScaler()
    regr = TransformedTargetRegressor(
        regressor=LinearRegression(), transformer=transformer
    )
    y_pred = regr.fit(X, y).predict(X)
    assert y.shape == y_pred.shape
    # consistency forward transform
    if y.ndim == 1:  # create a 2D array and squeeze results
        y_tran = regr.transformer_.transform(y.reshape(-1, 1))
    else:
        y_tran = regr.transformer_.transform(y)
    _check_standard_scaled(y, y_tran.squeeze())
    assert y.shape == y_pred.shape
    # consistency inverse transform
    assert_allclose(y, regr.transformer_.inverse_transform(y_tran).squeeze())
    # consistency of the regressor
    lr = LinearRegression()
    transformer2 = clone(transformer)
    if y.ndim == 1:  # create a 2D array and squeeze results
        lr.fit(X, transformer2.fit_transform(y.reshape(-1, 1)).squeeze())
        y_lr_pred = lr.predict(X).reshape(-1, 1)
        y_pred2 = transformer2.inverse_transform(y_lr_pred).squeeze()
    else:
        lr.fit(X, transformer2.fit_transform(y))
        y_lr_pred = lr.predict(X)
        y_pred2 = transformer2.inverse_transform(y_lr_pred)

    assert_allclose(y_pred, y_pred2)
    assert_allclose(regr.regressor_.coef_, lr.coef_)


def test_transform_target_regressor_2d_transformer_multioutput():
    # Check consistency with transformer accepting only 2D array and a 2D y
    # array.
    X = friedman[0]
    y = np.vstack((friedman[1], friedman[1] ** 2 + 1)).T
    transformer = StandardScaler()
    regr = TransformedTargetRegressor(
        regressor=LinearRegression(), transformer=transformer
    )
    y_pred = regr.fit(X, y).predict(X)
    assert y.shape == y_pred.shape
    # consistency forward transform
    y_tran = regr.transformer_.transform(y)
    _check_standard_scaled(y, y_tran)
    assert y.shape == y_pred.shape
    # consistency inverse transform
    assert_allclose(y, regr.transformer_.inverse_transform(y_tran).squeeze())
    # consistency of the regressor
    lr = LinearRegression()
    transformer2 = clone(transformer)
    lr.fit(X, transformer2.fit_transform(y))
    y_lr_pred = lr.predict(X)
    assert_allclose(y_pred, transformer2.inverse_transform(y_lr_pred))
    assert_allclose(regr.regressor_.coef_, lr.coef_)


def test_transform_target_regressor_3d_target():
    # Non-regression test for:
    # https://github.com/scikit-learn/scikit-learn/issues/18866
    # Check with a 3D target with a transformer that reshapes the target
    X = friedman[0]
    y = np.tile(friedman[1].reshape(-1, 1, 1), [1, 3, 2])

    def flatten_data(data):
        return data.reshape(data.shape[0], -1)

    def unflatten_data(data):
        return data.reshape(data.shape[0], -1, 2)

    transformer = FunctionTransformer(func=flatten_data, inverse_func=unflatten_data)
    regr = TransformedTargetRegressor(
        regressor=LinearRegression(), transformer=transformer
    )
    y_pred = regr.fit(X, y).predict(X)
    assert y.shape == y_pred.shape


def test_transform_target_regressor_multi_to_single():
    X = friedman[0]
    y = np.transpose([friedman[1], (friedman[1] ** 2 + 1)])

    def func(y):
        out = np.sqrt(y[:, 0] ** 2 + y[:, 1] ** 2)
        return out[:, np.newaxis]

    def inverse_func(y):
        return y

    tt = TransformedTargetRegressor(
        func=func, inverse_func=inverse_func, check_inverse=False
    )
    tt.fit(X, y)
    y_pred_2d_func = tt.predict(X)
    assert y_pred_2d_func.shape == (100, 1)

    # force that the function only return a 1D array
    def func(y):
        return np.sqrt(y[:, 0] ** 2 + y[:, 1] ** 2)

    tt = TransformedTargetRegressor(
        func=func, inverse_func=inverse_func, check_inverse=False
    )
    tt.fit(X, y)
    y_pred_1d_func = tt.predict(X)
    assert y_pred_1d_func.shape == (100, 1)

    assert_allclose(y_pred_1d_func, y_pred_2d_func)


class DummyCheckerArrayTransformer(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        assert isinstance(X, np.ndarray)
        return self

    def transform(self, X):
        assert isinstance(X, np.ndarray)
        return X

    def inverse_transform(self, X):
        assert isinstance(X, np.ndarray)
        return X


class DummyCheckerListRegressor(DummyRegressor):
    def fit(self, X, y, sample_weight=None):
        assert isinstance(X, list)
        return super().fit(X, y, sample_weight)

    def predict(self, X):
        assert isinstance(X, list)
        return super().predict(X)


def test_transform_target_regressor_ensure_y_array():
    # check that the target ``y`` passed to the transformer will always be a
    # numpy array. Similarly, if ``X`` is passed as a list, we check that the
    # predictor receive as it is.
    X, y = friedman
    tt = TransformedTargetRegressor(
        transformer=DummyCheckerArrayTransformer(),
        regressor=DummyCheckerListRegressor(),
        check_inverse=False,
    )
    tt.fit(X.tolist(), y.tolist())
    tt.predict(X.tolist())
    with pytest.raises(AssertionError):
        tt.fit(X, y.tolist())
    with pytest.raises(AssertionError):
        tt.predict(X)


class DummyTransformer(TransformerMixin, BaseEstimator):
    """Dummy transformer which count how many time fit was called."""

    def __init__(self, fit_counter=0):
        self.fit_counter = fit_counter

    def fit(self, X, y=None):
        self.fit_counter += 1
        return self

    def transform(self, X):
        return X

    def inverse_transform(self, X):
        return X


@pytest.mark.parametrize("check_inverse", [False, True])
def test_transform_target_regressor_count_fit(check_inverse):
    # regression test for gh-issue #11618
    # check that we only call a single time fit for the transformer
    X, y = friedman
    ttr = TransformedTargetRegressor(
        transformer=DummyTransformer(), check_inverse=check_inverse
    )
    ttr.fit(X, y)
    assert ttr.transformer_.fit_counter == 1


class DummyRegressorWithExtraFitParams(DummyRegressor):
    def fit(self, X, y, sample_weight=None, check_input=True):
        # on the test below we force this to false, we make sure this is
        # actually passed to the regressor
        assert not check_input
        return super().fit(X, y, sample_weight)


def test_transform_target_regressor_pass_fit_parameters():
    X, y = friedman
    regr = TransformedTargetRegressor(
        regressor=DummyRegressorWithExtraFitParams(), transformer=DummyTransformer()
    )

    regr.fit(X, y, check_input=False)
    assert regr.transformer_.fit_counter == 1


def test_transform_target_regressor_route_pipeline():
    X, y = friedman

    regr = TransformedTargetRegressor(
        regressor=DummyRegressorWithExtraFitParams(), transformer=DummyTransformer()
    )
    estimators = [("normalize", StandardScaler()), ("est", regr)]

    pip = Pipeline(estimators)
    pip.fit(X, y, **{"est__check_input": False})

    assert regr.transformer_.fit_counter == 1


class DummyRegressorWithExtraPredictParams(DummyRegressor):
    def predict(self, X, check_input=True):
        # In the test below we make sure that the check input parameter is
        # passed as false
        self.predict_called = True
        assert not check_input
        return super().predict(X)


def test_transform_target_regressor_pass_extra_predict_parameters():
    # Checks that predict kwargs are passed to regressor.
    X, y = friedman
    regr = TransformedTargetRegressor(
        regressor=DummyRegressorWithExtraPredictParams(), transformer=DummyTransformer()
    )

    regr.fit(X, y)
    regr.predict(X, check_input=False)
    assert regr.regressor_.predict_called
