"""
Test the ColumnTransformer.
"""
import re
import pickle

import numpy as np
from scipy import sparse
import pytest

from numpy.testing import assert_allclose
from sklearn.utils._testing import assert_array_equal
from sklearn.utils._testing import assert_allclose_dense_sparse
from sklearn.utils._testing import assert_almost_equal

from sklearn.base import BaseEstimator
from sklearn.compose import (
    ColumnTransformer,
    make_column_transformer,
    make_column_selector,
)
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler, Normalizer, OneHotEncoder
from sklearn.feature_extraction import DictVectorizer


class Trans(BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # 1D Series -> 2D DataFrame
        if hasattr(X, "to_frame"):
            return X.to_frame()
        # 1D array -> 2D array
        if X.ndim == 1:
            return np.atleast_2d(X).T
        return X


class DoubleTrans(BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return 2 * X


class SparseMatrixTrans(BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        n_samples = len(X)
        return sparse.eye(n_samples, n_samples).tocsr()


class TransNo2D(BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X


class TransRaise(BaseEstimator):
    def fit(self, X, y=None):
        raise ValueError("specific message")

    def transform(self, X, y=None):
        raise ValueError("specific message")


def test_column_transformer():
    X_array = np.array([[0, 1, 2], [2, 4, 6]]).T

    X_res_first1D = np.array([0, 1, 2])
    X_res_second1D = np.array([2, 4, 6])
    X_res_first = X_res_first1D.reshape(-1, 1)
    X_res_both = X_array

    cases = [
        # single column 1D / 2D
        (0, X_res_first),
        ([0], X_res_first),
        # list-like
        ([0, 1], X_res_both),
        (np.array([0, 1]), X_res_both),
        # slice
        (slice(0, 1), X_res_first),
        (slice(0, 2), X_res_both),
        # boolean mask
        (np.array([True, False]), X_res_first),
        ([True, False], X_res_first),
        (np.array([True, True]), X_res_both),
        ([True, True], X_res_both),
    ]

    for selection, res in cases:
        ct = ColumnTransformer([("trans", Trans(), selection)], remainder="drop")
        assert_array_equal(ct.fit_transform(X_array), res)
        assert_array_equal(ct.fit(X_array).transform(X_array), res)

        # callable that returns any of the allowed specifiers
        ct = ColumnTransformer(
            [("trans", Trans(), lambda x: selection)], remainder="drop"
        )
        assert_array_equal(ct.fit_transform(X_array), res)
        assert_array_equal(ct.fit(X_array).transform(X_array), res)

    ct = ColumnTransformer([("trans1", Trans(), [0]), ("trans2", Trans(), [1])])
    assert_array_equal(ct.fit_transform(X_array), X_res_both)
    assert_array_equal(ct.fit(X_array).transform(X_array), X_res_both)
    assert len(ct.transformers_) == 2

    # test with transformer_weights
    transformer_weights = {"trans1": 0.1, "trans2": 10}
    both = ColumnTransformer(
        [("trans1", Trans(), [0]), ("trans2", Trans(), [1])],
        transformer_weights=transformer_weights,
    )
    res = np.vstack(
        [
            transformer_weights["trans1"] * X_res_first1D,
            transformer_weights["trans2"] * X_res_second1D,
        ]
    ).T
    assert_array_equal(both.fit_transform(X_array), res)
    assert_array_equal(both.fit(X_array).transform(X_array), res)
    assert len(both.transformers_) == 2

    both = ColumnTransformer(
        [("trans", Trans(), [0, 1])], transformer_weights={"trans": 0.1}
    )
    assert_array_equal(both.fit_transform(X_array), 0.1 * X_res_both)
    assert_array_equal(both.fit(X_array).transform(X_array), 0.1 * X_res_both)
    assert len(both.transformers_) == 1


def test_column_transformer_dataframe():
    pd = pytest.importorskip("pandas")

    X_array = np.array([[0, 1, 2], [2, 4, 6]]).T
    X_df = pd.DataFrame(X_array, columns=["first", "second"])

    X_res_first = np.array([0, 1, 2]).reshape(-1, 1)
    X_res_both = X_array

    cases = [
        # String keys: label based
        # scalar
        ("first", X_res_first),
        # list
        (["first"], X_res_first),
        (["first", "second"], X_res_both),
        # slice
        (slice("first", "second"), X_res_both),
        # int keys: positional
        # scalar
        (0, X_res_first),
        # list
        ([0], X_res_first),
        ([0, 1], X_res_both),
        (np.array([0, 1]), X_res_both),
        # slice
        (slice(0, 1), X_res_first),
        (slice(0, 2), X_res_both),
        # boolean mask
        (np.array([True, False]), X_res_first),
        (pd.Series([True, False], index=["first", "second"]), X_res_first),
        ([True, False], X_res_first),
    ]

    for selection, res in cases:
        ct = ColumnTransformer([("trans", Trans(), selection)], remainder="drop")
        assert_array_equal(ct.fit_transform(X_df), res)
        assert_array_equal(ct.fit(X_df).transform(X_df), res)

        # callable that returns any of the allowed specifiers
        ct = ColumnTransformer(
            [("trans", Trans(), lambda X: selection)], remainder="drop"
        )
        assert_array_equal(ct.fit_transform(X_df), res)
        assert_array_equal(ct.fit(X_df).transform(X_df), res)

    ct = ColumnTransformer(
        [("trans1", Trans(), ["first"]), ("trans2", Trans(), ["second"])]
    )
    assert_array_equal(ct.fit_transform(X_df), X_res_both)
    assert_array_equal(ct.fit(X_df).transform(X_df), X_res_both)
    assert len(ct.transformers_) == 2
    assert ct.transformers_[-1][0] != "remainder"

    ct = ColumnTransformer([("trans1", Trans(), [0]), ("trans2", Trans(), [1])])
    assert_array_equal(ct.fit_transform(X_df), X_res_both)
    assert_array_equal(ct.fit(X_df).transform(X_df), X_res_both)
    assert len(ct.transformers_) == 2
    assert ct.transformers_[-1][0] != "remainder"

    # test with transformer_weights
    transformer_weights = {"trans1": 0.1, "trans2": 10}
    both = ColumnTransformer(
        [("trans1", Trans(), ["first"]), ("trans2", Trans(), ["second"])],
        transformer_weights=transformer_weights,
    )
    res = np.vstack(
        [
            transformer_weights["trans1"] * X_df["first"],
            transformer_weights["trans2"] * X_df["second"],
        ]
    ).T
    assert_array_equal(both.fit_transform(X_df), res)
    assert_array_equal(both.fit(X_df).transform(X_df), res)
    assert len(both.transformers_) == 2
    assert both.transformers_[-1][0] != "remainder"

    # test multiple columns
    both = ColumnTransformer(
        [("trans", Trans(), ["first", "second"])], transformer_weights={"trans": 0.1}
    )
    assert_array_equal(both.fit_transform(X_df), 0.1 * X_res_both)
    assert_array_equal(both.fit(X_df).transform(X_df), 0.1 * X_res_both)
    assert len(both.transformers_) == 1
    assert both.transformers_[-1][0] != "remainder"

    both = ColumnTransformer(
        [("trans", Trans(), [0, 1])], transformer_weights={"trans": 0.1}
    )
    assert_array_equal(both.fit_transform(X_df), 0.1 * X_res_both)
    assert_array_equal(both.fit(X_df).transform(X_df), 0.1 * X_res_both)
    assert len(both.transformers_) == 1
    assert both.transformers_[-1][0] != "remainder"

    # ensure pandas object is passed through

    class TransAssert(BaseEstimator):
        def fit(self, X, y=None):
            return self

        def transform(self, X, y=None):
            assert isinstance(X, (pd.DataFrame, pd.Series))
            if isinstance(X, pd.Series):
                X = X.to_frame()
            return X

    ct = ColumnTransformer([("trans", TransAssert(), "first")], remainder="drop")
    ct.fit_transform(X_df)
    ct = ColumnTransformer([("trans", TransAssert(), ["first", "second"])])
    ct.fit_transform(X_df)

    # integer column spec + integer column names -> still use positional
    X_df2 = X_df.copy()
    X_df2.columns = [1, 0]
    ct = ColumnTransformer([("trans", Trans(), 0)], remainder="drop")
    assert_array_equal(ct.fit_transform(X_df2), X_res_first)
    assert_array_equal(ct.fit(X_df2).transform(X_df2), X_res_first)

    assert len(ct.transformers_) == 2
    assert ct.transformers_[-1][0] == "remainder"
    assert ct.transformers_[-1][1] == "drop"
    assert_array_equal(ct.transformers_[-1][2], [1])


@pytest.mark.parametrize("pandas", [True, False], ids=["pandas", "numpy"])
@pytest.mark.parametrize(
    "column_selection",
    [[], np.array([False, False]), [False, False]],
    ids=["list", "bool", "bool_int"],
)
@pytest.mark.parametrize("callable_column", [False, True])
def test_column_transformer_empty_columns(pandas, column_selection, callable_column):
    # test case that ensures that the column transformer does also work when
    # a given transformer doesn't have any columns to work on
    X_array = np.array([[0, 1, 2], [2, 4, 6]]).T
    X_res_both = X_array

    if pandas:
        pd = pytest.importorskip("pandas")
        X = pd.DataFrame(X_array, columns=["first", "second"])
    else:
        X = X_array

    if callable_column:
        column = lambda X: column_selection  # noqa
    else:
        column = column_selection

    ct = ColumnTransformer(
        [("trans1", Trans(), [0, 1]), ("trans2", TransRaise(), column)]
    )
    assert_array_equal(ct.fit_transform(X), X_res_both)
    assert_array_equal(ct.fit(X).transform(X), X_res_both)
    assert len(ct.transformers_) == 2
    assert isinstance(ct.transformers_[1][1], TransRaise)

    ct = ColumnTransformer(
        [("trans1", TransRaise(), column), ("trans2", Trans(), [0, 1])]
    )
    assert_array_equal(ct.fit_transform(X), X_res_both)
    assert_array_equal(ct.fit(X).transform(X), X_res_both)
    assert len(ct.transformers_) == 2
    assert isinstance(ct.transformers_[0][1], TransRaise)

    ct = ColumnTransformer([("trans", TransRaise(), column)], remainder="passthrough")
    assert_array_equal(ct.fit_transform(X), X_res_both)
    assert_array_equal(ct.fit(X).transform(X), X_res_both)
    assert len(ct.transformers_) == 2  # including remainder
    assert isinstance(ct.transformers_[0][1], TransRaise)

    fixture = np.array([[], [], []])
    ct = ColumnTransformer([("trans", TransRaise(), column)], remainder="drop")
    assert_array_equal(ct.fit_transform(X), fixture)
    assert_array_equal(ct.fit(X).transform(X), fixture)
    assert len(ct.transformers_) == 2  # including remainder
    assert isinstance(ct.transformers_[0][1], TransRaise)


def test_column_transformer_output_indices():
    # Checks for the output_indices_ attribute
    X_array = np.arange(6).reshape(3, 2)

    ct = ColumnTransformer([("trans1", Trans(), [0]), ("trans2", Trans(), [1])])
    X_trans = ct.fit_transform(X_array)
    assert ct.output_indices_ == {
        "trans1": slice(0, 1),
        "trans2": slice(1, 2),
        "remainder": slice(0, 0),
    }
    assert_array_equal(X_trans[:, [0]], X_trans[:, ct.output_indices_["trans1"]])
    assert_array_equal(X_trans[:, [1]], X_trans[:, ct.output_indices_["trans2"]])

    # test with transformer_weights and multiple columns
    ct = ColumnTransformer(
        [("trans", Trans(), [0, 1])], transformer_weights={"trans": 0.1}
    )
    X_trans = ct.fit_transform(X_array)
    assert ct.output_indices_ == {"trans": slice(0, 2), "remainder": slice(0, 0)}
    assert_array_equal(X_trans[:, [0, 1]], X_trans[:, ct.output_indices_["trans"]])
    assert_array_equal(X_trans[:, []], X_trans[:, ct.output_indices_["remainder"]])

    # test case that ensures that the attribute does also work when
    # a given transformer doesn't have any columns to work on
    ct = ColumnTransformer([("trans1", Trans(), [0, 1]), ("trans2", TransRaise(), [])])
    X_trans = ct.fit_transform(X_array)
    assert ct.output_indices_ == {
        "trans1": slice(0, 2),
        "trans2": slice(0, 0),
        "remainder": slice(0, 0),
    }
    assert_array_equal(X_trans[:, [0, 1]], X_trans[:, ct.output_indices_["trans1"]])
    assert_array_equal(X_trans[:, []], X_trans[:, ct.output_indices_["trans2"]])
    assert_array_equal(X_trans[:, []], X_trans[:, ct.output_indices_["remainder"]])

    ct = ColumnTransformer([("trans", TransRaise(), [])], remainder="passthrough")
    X_trans = ct.fit_transform(X_array)
    assert ct.output_indices_ == {"trans": slice(0, 0), "remainder": slice(0, 2)}
    assert_array_equal(X_trans[:, []], X_trans[:, ct.output_indices_["trans"]])
    assert_array_equal(X_trans[:, [0, 1]], X_trans[:, ct.output_indices_["remainder"]])


def test_column_transformer_output_indices_df():
    # Checks for the output_indices_ attribute with data frames
    pd = pytest.importorskip("pandas")

    X_df = pd.DataFrame(np.arange(6).reshape(3, 2), columns=["first", "second"])

    ct = ColumnTransformer(
        [("trans1", Trans(), ["first"]), ("trans2", Trans(), ["second"])]
    )
    X_trans = ct.fit_transform(X_df)
    assert ct.output_indices_ == {
        "trans1": slice(0, 1),
        "trans2": slice(1, 2),
        "remainder": slice(0, 0),
    }
    assert_array_equal(X_trans[:, [0]], X_trans[:, ct.output_indices_["trans1"]])
    assert_array_equal(X_trans[:, [1]], X_trans[:, ct.output_indices_["trans2"]])
    assert_array_equal(X_trans[:, []], X_trans[:, ct.output_indices_["remainder"]])

    ct = ColumnTransformer([("trans1", Trans(), [0]), ("trans2", Trans(), [1])])
    X_trans = ct.fit_transform(X_df)
    assert ct.output_indices_ == {
        "trans1": slice(0, 1),
        "trans2": slice(1, 2),
        "remainder": slice(0, 0),
    }
    assert_array_equal(X_trans[:, [0]], X_trans[:, ct.output_indices_["trans1"]])
    assert_array_equal(X_trans[:, [1]], X_trans[:, ct.output_indices_["trans2"]])
    assert_array_equal(X_trans[:, []], X_trans[:, ct.output_indices_["remainder"]])


def test_column_transformer_sparse_array():
    X_sparse = sparse.eye(3, 2).tocsr()

    # no distinction between 1D and 2D
    X_res_first = X_sparse[:, 0]
    X_res_both = X_sparse

    for col in [0, [0], slice(0, 1)]:
        for remainder, res in [("drop", X_res_first), ("passthrough", X_res_both)]:
            ct = ColumnTransformer(
                [("trans", Trans(), col)], remainder=remainder, sparse_threshold=0.8
            )
            assert sparse.issparse(ct.fit_transform(X_sparse))
            assert_allclose_dense_sparse(ct.fit_transform(X_sparse), res)
            assert_allclose_dense_sparse(ct.fit(X_sparse).transform(X_sparse), res)

    for col in [[0, 1], slice(0, 2)]:
        ct = ColumnTransformer([("trans", Trans(), col)], sparse_threshold=0.8)
        assert sparse.issparse(ct.fit_transform(X_sparse))
        assert_allclose_dense_sparse(ct.fit_transform(X_sparse), X_res_both)
        assert_allclose_dense_sparse(ct.fit(X_sparse).transform(X_sparse), X_res_both)


def test_column_transformer_list():
    X_list = [[1, float("nan"), "a"], [0, 0, "b"]]
    expected_result = np.array(
        [
            [1, float("nan"), 1, 0],
            [-1, 0, 0, 1],
        ]
    )

    ct = ColumnTransformer(
        [
            ("numerical", StandardScaler(), [0, 1]),
            ("categorical", OneHotEncoder(), [2]),
        ]
    )

    assert_array_equal(ct.fit_transform(X_list), expected_result)
    assert_array_equal(ct.fit(X_list).transform(X_list), expected_result)


def test_column_transformer_sparse_stacking():
    X_array = np.array([[0, 1, 2], [2, 4, 6]]).T
    col_trans = ColumnTransformer(
        [("trans1", Trans(), [0]), ("trans2", SparseMatrixTrans(), 1)],
        sparse_threshold=0.8,
    )
    col_trans.fit(X_array)
    X_trans = col_trans.transform(X_array)
    assert sparse.issparse(X_trans)
    assert X_trans.shape == (X_trans.shape[0], X_trans.shape[0] + 1)
    assert_array_equal(X_trans.toarray()[:, 1:], np.eye(X_trans.shape[0]))
    assert len(col_trans.transformers_) == 2
    assert col_trans.transformers_[-1][0] != "remainder"

    col_trans = ColumnTransformer(
        [("trans1", Trans(), [0]), ("trans2", SparseMatrixTrans(), 1)],
        sparse_threshold=0.1,
    )
    col_trans.fit(X_array)
    X_trans = col_trans.transform(X_array)
    assert not sparse.issparse(X_trans)
    assert X_trans.shape == (X_trans.shape[0], X_trans.shape[0] + 1)
    assert_array_equal(X_trans[:, 1:], np.eye(X_trans.shape[0]))


def test_column_transformer_mixed_cols_sparse():
    df = np.array([["a", 1, True], ["b", 2, False]], dtype="O")

    ct = make_column_transformer(
        (OneHotEncoder(), [0]), ("passthrough", [1, 2]), sparse_threshold=1.0
    )

    # this shouldn't fail, since boolean can be coerced into a numeric
    # See: https://github.com/scikit-learn/scikit-learn/issues/11912
    X_trans = ct.fit_transform(df)
    assert X_trans.getformat() == "csr"
    assert_array_equal(X_trans.toarray(), np.array([[1, 0, 1, 1], [0, 1, 2, 0]]))

    ct = make_column_transformer(
        (OneHotEncoder(), [0]), ("passthrough", [0]), sparse_threshold=1.0
    )
    with pytest.raises(ValueError, match="For a sparse output, all columns should"):
        # this fails since strings `a` and `b` cannot be
        # coerced into a numeric.
        ct.fit_transform(df)


def test_column_transformer_sparse_threshold():
    X_array = np.array([["a", "b"], ["A", "B"]], dtype=object).T
    # above data has sparsity of 4 / 8 = 0.5

    # apply threshold even if all sparse
    col_trans = ColumnTransformer(
        [("trans1", OneHotEncoder(), [0]), ("trans2", OneHotEncoder(), [1])],
        sparse_threshold=0.2,
    )
    res = col_trans.fit_transform(X_array)
    assert not sparse.issparse(res)
    assert not col_trans.sparse_output_

    # mixed -> sparsity of (4 + 2) / 8 = 0.75
    for thres in [0.75001, 1]:
        col_trans = ColumnTransformer(
            [
                ("trans1", OneHotEncoder(sparse=True), [0]),
                ("trans2", OneHotEncoder(sparse=False), [1]),
            ],
            sparse_threshold=thres,
        )
        res = col_trans.fit_transform(X_array)
        assert sparse.issparse(res)
        assert col_trans.sparse_output_

    for thres in [0.75, 0]:
        col_trans = ColumnTransformer(
            [
                ("trans1", OneHotEncoder(sparse=True), [0]),
                ("trans2", OneHotEncoder(sparse=False), [1]),
            ],
            sparse_threshold=thres,
        )
        res = col_trans.fit_transform(X_array)
        assert not sparse.issparse(res)
        assert not col_trans.sparse_output_

    # if nothing is sparse -> no sparse
    for thres in [0.33, 0, 1]:
        col_trans = ColumnTransformer(
            [
                ("trans1", OneHotEncoder(sparse=False), [0]),
                ("trans2", OneHotEncoder(sparse=False), [1]),
            ],
            sparse_threshold=thres,
        )
        res = col_trans.fit_transform(X_array)
        assert not sparse.issparse(res)
        assert not col_trans.sparse_output_


def test_column_transformer_error_msg_1D():
    X_array = np.array([[0.0, 1.0, 2.0], [2.0, 4.0, 6.0]]).T

    col_trans = ColumnTransformer([("trans", StandardScaler(), 0)])
    msg = "1D data passed to a transformer"
    with pytest.raises(ValueError, match=msg):
        col_trans.fit(X_array)

    with pytest.raises(ValueError, match=msg):
        col_trans.fit_transform(X_array)

    col_trans = ColumnTransformer([("trans", TransRaise(), 0)])
    for func in [col_trans.fit, col_trans.fit_transform]:
        with pytest.raises(ValueError, match="specific message"):
            func(X_array)


def test_2D_transformer_output():
    X_array = np.array([[0, 1, 2], [2, 4, 6]]).T

    # if one transformer is dropped, test that name is still correct
    ct = ColumnTransformer([("trans1", "drop", 0), ("trans2", TransNo2D(), 1)])

    msg = "the 'trans2' transformer should be 2D"
    with pytest.raises(ValueError, match=msg):
        ct.fit_transform(X_array)
    # because fit is also doing transform, this raises already on fit
    with pytest.raises(ValueError, match=msg):
        ct.fit(X_array)


def test_2D_transformer_output_pandas():
    pd = pytest.importorskip("pandas")

    X_array = np.array([[0, 1, 2], [2, 4, 6]]).T
    X_df = pd.DataFrame(X_array, columns=["col1", "col2"])

    # if one transformer is dropped, test that name is still correct
    ct = ColumnTransformer([("trans1", TransNo2D(), "col1")])
    msg = "the 'trans1' transformer should be 2D"
    with pytest.raises(ValueError, match=msg):
        ct.fit_transform(X_df)
    # because fit is also doing transform, this raises already on fit
    with pytest.raises(ValueError, match=msg):
        ct.fit(X_df)


@pytest.mark.parametrize("remainder", ["drop", "passthrough"])
def test_column_transformer_invalid_columns(remainder):
    X_array = np.array([[0, 1, 2], [2, 4, 6]]).T

    # general invalid
    for col in [1.5, ["string", 1], slice(1, "s"), np.array([1.0])]:
        ct = ColumnTransformer([("trans", Trans(), col)], remainder=remainder)
        with pytest.raises(ValueError, match="No valid specification"):
            ct.fit(X_array)

    # invalid for arrays
    for col in ["string", ["string", "other"], slice("a", "b")]:
        ct = ColumnTransformer([("trans", Trans(), col)], remainder=remainder)
        with pytest.raises(ValueError, match="Specifying the columns"):
            ct.fit(X_array)

    # transformed n_features does not match fitted n_features
    col = [0, 1]
    ct = ColumnTransformer([("trans", Trans(), col)], remainder=remainder)
    ct.fit(X_array)
    X_array_more = np.array([[0, 1, 2], [2, 4, 6], [3, 6, 9]]).T
    msg = "X has 3 features, but ColumnTransformer is expecting 2 features as input."
    with pytest.raises(ValueError, match=msg):
        ct.transform(X_array_more)
    X_array_fewer = np.array(
        [
            [0, 1, 2],
        ]
    ).T
    err_msg = (
        "X has 1 features, but ColumnTransformer is expecting 2 features as input."
    )
    with pytest.raises(ValueError, match=err_msg):
        ct.transform(X_array_fewer)


def test_column_transformer_invalid_transformer():
    class NoTrans(BaseEstimator):
        def fit(self, X, y=None):
            return self

        def predict(self, X):
            return X

    X_array = np.array([[0, 1, 2], [2, 4, 6]]).T
    ct = ColumnTransformer([("trans", NoTrans(), [0])])
    msg = "All estimators should implement fit and transform"
    with pytest.raises(TypeError, match=msg):
        ct.fit(X_array)


def test_make_column_transformer():
    scaler = StandardScaler()
    norm = Normalizer()
    ct = make_column_transformer((scaler, "first"), (norm, ["second"]))
    names, transformers, columns = zip(*ct.transformers)
    assert names == ("standardscaler", "normalizer")
    assert transformers == (scaler, norm)
    assert columns == ("first", ["second"])


def test_make_column_transformer_pandas():
    pd = pytest.importorskip("pandas")
    X_array = np.array([[0, 1, 2], [2, 4, 6]]).T
    X_df = pd.DataFrame(X_array, columns=["first", "second"])
    norm = Normalizer()
    ct1 = ColumnTransformer([("norm", Normalizer(), X_df.columns)])
    ct2 = make_column_transformer((norm, X_df.columns))
    assert_almost_equal(ct1.fit_transform(X_df), ct2.fit_transform(X_df))


def test_make_column_transformer_kwargs():
    scaler = StandardScaler()
    norm = Normalizer()
    ct = make_column_transformer(
        (scaler, "first"),
        (norm, ["second"]),
        n_jobs=3,
        remainder="drop",
        sparse_threshold=0.5,
    )
    assert (
        ct.transformers
        == make_column_transformer((scaler, "first"), (norm, ["second"])).transformers
    )
    assert ct.n_jobs == 3
    assert ct.remainder == "drop"
    assert ct.sparse_threshold == 0.5
    # invalid keyword parameters should raise an error message
    msg = re.escape(
        "make_column_transformer() got an unexpected "
        "keyword argument 'transformer_weights'"
    )
    with pytest.raises(TypeError, match=msg):
        make_column_transformer(
            (scaler, "first"),
            (norm, ["second"]),
            transformer_weights={"pca": 10, "Transf": 1},
        )


def test_make_column_transformer_remainder_transformer():
    scaler = StandardScaler()
    norm = Normalizer()
    remainder = StandardScaler()
    ct = make_column_transformer(
        (scaler, "first"), (norm, ["second"]), remainder=remainder
    )
    assert ct.remainder == remainder


def test_column_transformer_get_set_params():
    ct = ColumnTransformer(
        [("trans1", StandardScaler(), [0]), ("trans2", StandardScaler(), [1])]
    )

    exp = {
        "n_jobs": None,
        "remainder": "drop",
        "sparse_threshold": 0.3,
        "trans1": ct.transformers[0][1],
        "trans1__copy": True,
        "trans1__with_mean": True,
        "trans1__with_std": True,
        "trans2": ct.transformers[1][1],
        "trans2__copy": True,
        "trans2__with_mean": True,
        "trans2__with_std": True,
        "transformers": ct.transformers,
        "transformer_weights": None,
        "verbose_feature_names_out": True,
        "verbose": False,
    }

    assert ct.get_params() == exp

    ct.set_params(trans1__with_mean=False)
    assert not ct.get_params()["trans1__with_mean"]

    ct.set_params(trans1="passthrough")
    exp = {
        "n_jobs": None,
        "remainder": "drop",
        "sparse_threshold": 0.3,
        "trans1": "passthrough",
        "trans2": ct.transformers[1][1],
        "trans2__copy": True,
        "trans2__with_mean": True,
        "trans2__with_std": True,
        "transformers": ct.transformers,
        "transformer_weights": None,
        "verbose_feature_names_out": True,
        "verbose": False,
    }

    assert ct.get_params() == exp


def test_column_transformer_named_estimators():
    X_array = np.array([[0.0, 1.0, 2.0], [2.0, 4.0, 6.0]]).T
    ct = ColumnTransformer(
        [
            ("trans1", StandardScaler(), [0]),
            ("trans2", StandardScaler(with_std=False), [1]),
        ]
    )
    assert not hasattr(ct, "transformers_")
    ct.fit(X_array)
    assert hasattr(ct, "transformers_")
    assert isinstance(ct.named_transformers_["trans1"], StandardScaler)
    assert isinstance(ct.named_transformers_.trans1, StandardScaler)
    assert isinstance(ct.named_transformers_["trans2"], StandardScaler)
    assert isinstance(ct.named_transformers_.trans2, StandardScaler)
    assert not ct.named_transformers_.trans2.with_std
    # check it are fitted transformers
    assert ct.named_transformers_.trans1.mean_ == 1.0


def test_column_transformer_cloning():
    X_array = np.array([[0.0, 1.0, 2.0], [2.0, 4.0, 6.0]]).T

    ct = ColumnTransformer([("trans", StandardScaler(), [0])])
    ct.fit(X_array)
    assert not hasattr(ct.transformers[0][1], "mean_")
    assert hasattr(ct.transformers_[0][1], "mean_")

    ct = ColumnTransformer([("trans", StandardScaler(), [0])])
    ct.fit_transform(X_array)
    assert not hasattr(ct.transformers[0][1], "mean_")
    assert hasattr(ct.transformers_[0][1], "mean_")


# TODO: Remove in 1.2 when get_feature_names is removed.
@pytest.mark.filterwarnings("ignore::FutureWarning:sklearn")
@pytest.mark.parametrize("get_names", ["get_feature_names", "get_feature_names_out"])
def test_column_transformer_get_feature_names(get_names):
    X_array = np.array([[0.0, 1.0, 2.0], [2.0, 4.0, 6.0]]).T
    ct = ColumnTransformer([("trans", Trans(), [0, 1])])
    # raise correct error when not fitted
    with pytest.raises(NotFittedError):
        getattr(ct, get_names)()
    # raise correct error when no feature names are available
    ct.fit(X_array)
    msg = re.escape(f"Transformer trans (type Trans) does not provide {get_names}")
    with pytest.raises(AttributeError, match=msg):
        getattr(ct, get_names)()


@pytest.mark.parametrize(
    "X, keys",
    [
        (
            np.array(
                [[{"a": 1, "b": 2}, {"a": 3, "b": 4}], [{"c": 5}, {"c": 6}]],
                dtype=object,
            ).T,
            ("a", "b", "c"),
        ),
        (
            np.array([[{1: 1, 2: 2}, {1: 3, 2: 4}], [{3: 5}, {3: 6}]], dtype=object).T,
            ("1", "2", "3"),
        ),
    ],
)
# TODO: Remove in 1.2 when get_feature_names is removed.
@pytest.mark.filterwarnings("ignore::FutureWarning:sklearn")
def test_column_transformer_get_feature_names_pipeline(X, keys):
    ct = ColumnTransformer([("col" + str(i), DictVectorizer(), i) for i in range(2)])
    ct.fit(X)
    assert ct.get_feature_names() == [f"col0__{key}" for key in keys[:2]] + [
        f"col1__{keys[2]}"
    ]

    # drop transformer
    ct = ColumnTransformer([("col0", DictVectorizer(), 0), ("col1", "drop", 1)])
    ct.fit(X)
    assert ct.get_feature_names() == [f"col0__{key}" for key in keys[:2]]

    # passthrough transformer
    ct = ColumnTransformer([("trans", "passthrough", [0, 1])])
    ct.fit(X)
    assert ct.get_feature_names() == ["x0", "x1"]

    ct = ColumnTransformer([("trans", DictVectorizer(), 0)], remainder="passthrough")
    ct.fit(X)
    assert ct.get_feature_names() == [f"trans__{key}" for key in keys[:2]] + ["x1"]

    ct = ColumnTransformer([("trans", "passthrough", [1])], remainder="passthrough")
    ct.fit(X)
    assert ct.get_feature_names() == ["x1", "x0"]

    ct = ColumnTransformer(
        [("trans", "passthrough", lambda x: [1])], remainder="passthrough"
    )
    ct.fit(X)
    assert ct.get_feature_names() == ["x1", "x0"]

    ct = ColumnTransformer(
        [("trans", "passthrough", np.array([False, True]))], remainder="passthrough"
    )
    ct.fit(X)
    assert ct.get_feature_names() == ["x1", "x0"]

    ct = ColumnTransformer(
        [("trans", "passthrough", slice(1, 2))], remainder="passthrough"
    )
    ct.fit(X)
    assert ct.get_feature_names() == ["x1", "x0"]


# TODO: Remove in 1.2 when get_feature_names is removed.
@pytest.mark.filterwarnings("ignore::FutureWarning:sklearn")
def test_column_transformer_get_feature_names_dataframe():
    # passthough transformer with a dataframe
    pd = pytest.importorskip("pandas")
    X = np.array(
        [[{"a": 1, "b": 2}, {"a": 3, "b": 4}], [{"c": 5}, {"c": 6}]], dtype=object
    ).T
    X_df = pd.DataFrame(X, columns=["col0", "col1"])

    ct = ColumnTransformer([("trans", "passthrough", ["col0", "col1"])])
    ct.fit(X_df)
    assert ct.get_feature_names() == ["col0", "col1"]

    ct = ColumnTransformer([("trans", "passthrough", [0, 1])])
    ct.fit(X_df)
    assert ct.get_feature_names() == ["col0", "col1"]

    ct = ColumnTransformer([("col0", DictVectorizer(), 0)], remainder="passthrough")
    ct.fit(X_df)
    assert ct.get_feature_names() == ["col0__a", "col0__b", "col1"]

    ct = ColumnTransformer(
        [("trans", "passthrough", ["col1"])], remainder="passthrough"
    )
    ct.fit(X_df)
    assert ct.get_feature_names() == ["col1", "col0"]

    ct = ColumnTransformer(
        [("trans", "passthrough", lambda x: x[["col1"]].columns)],
        remainder="passthrough",
    )
    ct.fit(X_df)
    assert ct.get_feature_names() == ["col1", "col0"]

    ct = ColumnTransformer(
        [("trans", "passthrough", np.array([False, True]))], remainder="passthrough"
    )
    ct.fit(X_df)
    assert ct.get_feature_names() == ["col1", "col0"]

    ct = ColumnTransformer(
        [("trans", "passthrough", slice(1, 2))], remainder="passthrough"
    )
    ct.fit(X_df)
    assert ct.get_feature_names() == ["col1", "col0"]

    ct = ColumnTransformer([("trans", "passthrough", [1])], remainder="passthrough")
    ct.fit(X_df)
    assert ct.get_feature_names() == ["col1", "col0"]


def test_column_transformer_special_strings():

    # one 'drop' -> ignore
    X_array = np.array([[0.0, 1.0, 2.0], [2.0, 4.0, 6.0]]).T
    ct = ColumnTransformer([("trans1", Trans(), [0]), ("trans2", "drop", [1])])
    exp = np.array([[0.0], [1.0], [2.0]])
    assert_array_equal(ct.fit_transform(X_array), exp)
    assert_array_equal(ct.fit(X_array).transform(X_array), exp)
    assert len(ct.transformers_) == 2
    assert ct.transformers_[-1][0] != "remainder"

    # all 'drop' -> return shape 0 array
    ct = ColumnTransformer([("trans1", "drop", [0]), ("trans2", "drop", [1])])
    assert_array_equal(ct.fit(X_array).transform(X_array).shape, (3, 0))
    assert_array_equal(ct.fit_transform(X_array).shape, (3, 0))
    assert len(ct.transformers_) == 2
    assert ct.transformers_[-1][0] != "remainder"

    # 'passthrough'
    X_array = np.array([[0.0, 1.0, 2.0], [2.0, 4.0, 6.0]]).T
    ct = ColumnTransformer([("trans1", Trans(), [0]), ("trans2", "passthrough", [1])])
    exp = X_array
    assert_array_equal(ct.fit_transform(X_array), exp)
    assert_array_equal(ct.fit(X_array).transform(X_array), exp)
    assert len(ct.transformers_) == 2
    assert ct.transformers_[-1][0] != "remainder"

    # None itself / other string is not valid
    for val in [None, "other"]:
        ct = ColumnTransformer([("trans1", Trans(), [0]), ("trans2", None, [1])])
        msg = "All estimators should implement"
        with pytest.raises(TypeError, match=msg):
            ct.fit_transform(X_array)
        with pytest.raises(TypeError, match=msg):
            ct.fit(X_array)


def test_column_transformer_remainder():
    X_array = np.array([[0, 1, 2], [2, 4, 6]]).T

    X_res_first = np.array([0, 1, 2]).reshape(-1, 1)
    X_res_second = np.array([2, 4, 6]).reshape(-1, 1)
    X_res_both = X_array

    # default drop
    ct = ColumnTransformer([("trans1", Trans(), [0])])
    assert_array_equal(ct.fit_transform(X_array), X_res_first)
    assert_array_equal(ct.fit(X_array).transform(X_array), X_res_first)
    assert len(ct.transformers_) == 2
    assert ct.transformers_[-1][0] == "remainder"
    assert ct.transformers_[-1][1] == "drop"
    assert_array_equal(ct.transformers_[-1][2], [1])

    # specify passthrough
    ct = ColumnTransformer([("trans", Trans(), [0])], remainder="passthrough")
    assert_array_equal(ct.fit_transform(X_array), X_res_both)
    assert_array_equal(ct.fit(X_array).transform(X_array), X_res_both)
    assert len(ct.transformers_) == 2
    assert ct.transformers_[-1][0] == "remainder"
    assert ct.transformers_[-1][1] == "passthrough"
    assert_array_equal(ct.transformers_[-1][2], [1])

    # column order is not preserved (passed through added to end)
    ct = ColumnTransformer([("trans1", Trans(), [1])], remainder="passthrough")
    assert_array_equal(ct.fit_transform(X_array), X_res_both[:, ::-1])
    assert_array_equal(ct.fit(X_array).transform(X_array), X_res_both[:, ::-1])
    assert len(ct.transformers_) == 2
    assert ct.transformers_[-1][0] == "remainder"
    assert ct.transformers_[-1][1] == "passthrough"
    assert_array_equal(ct.transformers_[-1][2], [0])

    # passthrough when all actual transformers are skipped
    ct = ColumnTransformer([("trans1", "drop", [0])], remainder="passthrough")
    assert_array_equal(ct.fit_transform(X_array), X_res_second)
    assert_array_equal(ct.fit(X_array).transform(X_array), X_res_second)
    assert len(ct.transformers_) == 2
    assert ct.transformers_[-1][0] == "remainder"
    assert ct.transformers_[-1][1] == "passthrough"
    assert_array_equal(ct.transformers_[-1][2], [1])

    # error on invalid arg
    ct = ColumnTransformer([("trans1", Trans(), [0])], remainder=1)
    msg = "remainder keyword needs to be one of 'drop', 'passthrough', or estimator."
    with pytest.raises(ValueError, match=msg):
        ct.fit(X_array)

    with pytest.raises(ValueError, match=msg):
        ct.fit_transform(X_array)

    # check default for make_column_transformer
    ct = make_column_transformer((Trans(), [0]))
    assert ct.remainder == "drop"


@pytest.mark.parametrize(
    "key", [[0], np.array([0]), slice(0, 1), np.array([True, False])]
)
def test_column_transformer_remainder_numpy(key):
    # test different ways that columns are specified with passthrough
    X_array = np.array([[0, 1, 2], [2, 4, 6]]).T
    X_res_both = X_array

    ct = ColumnTransformer([("trans1", Trans(), key)], remainder="passthrough")
    assert_array_equal(ct.fit_transform(X_array), X_res_both)
    assert_array_equal(ct.fit(X_array).transform(X_array), X_res_both)
    assert len(ct.transformers_) == 2
    assert ct.transformers_[-1][0] == "remainder"
    assert ct.transformers_[-1][1] == "passthrough"
    assert_array_equal(ct.transformers_[-1][2], [1])


@pytest.mark.parametrize(
    "key",
    [
        [0],
        slice(0, 1),
        np.array([True, False]),
        ["first"],
        "pd-index",
        np.array(["first"]),
        np.array(["first"], dtype=object),
        slice(None, "first"),
        slice("first", "first"),
    ],
)
def test_column_transformer_remainder_pandas(key):
    # test different ways that columns are specified with passthrough
    pd = pytest.importorskip("pandas")
    if isinstance(key, str) and key == "pd-index":
        key = pd.Index(["first"])

    X_array = np.array([[0, 1, 2], [2, 4, 6]]).T
    X_df = pd.DataFrame(X_array, columns=["first", "second"])
    X_res_both = X_array

    ct = ColumnTransformer([("trans1", Trans(), key)], remainder="passthrough")
    assert_array_equal(ct.fit_transform(X_df), X_res_both)
    assert_array_equal(ct.fit(X_df).transform(X_df), X_res_both)
    assert len(ct.transformers_) == 2
    assert ct.transformers_[-1][0] == "remainder"
    assert ct.transformers_[-1][1] == "passthrough"
    assert_array_equal(ct.transformers_[-1][2], [1])


@pytest.mark.parametrize(
    "key", [[0], np.array([0]), slice(0, 1), np.array([True, False, False])]
)
def test_column_transformer_remainder_transformer(key):
    X_array = np.array([[0, 1, 2], [2, 4, 6], [8, 6, 4]]).T
    X_res_both = X_array.copy()

    # second and third columns are doubled when remainder = DoubleTrans
    X_res_both[:, 1:3] *= 2

    ct = ColumnTransformer([("trans1", Trans(), key)], remainder=DoubleTrans())

    assert_array_equal(ct.fit_transform(X_array), X_res_both)
    assert_array_equal(ct.fit(X_array).transform(X_array), X_res_both)
    assert len(ct.transformers_) == 2
    assert ct.transformers_[-1][0] == "remainder"
    assert isinstance(ct.transformers_[-1][1], DoubleTrans)
    assert_array_equal(ct.transformers_[-1][2], [1, 2])


def test_column_transformer_no_remaining_remainder_transformer():
    X_array = np.array([[0, 1, 2], [2, 4, 6], [8, 6, 4]]).T

    ct = ColumnTransformer([("trans1", Trans(), [0, 1, 2])], remainder=DoubleTrans())

    assert_array_equal(ct.fit_transform(X_array), X_array)
    assert_array_equal(ct.fit(X_array).transform(X_array), X_array)
    assert len(ct.transformers_) == 1
    assert ct.transformers_[-1][0] != "remainder"


def test_column_transformer_drops_all_remainder_transformer():
    X_array = np.array([[0, 1, 2], [2, 4, 6], [8, 6, 4]]).T

    # columns are doubled when remainder = DoubleTrans
    X_res_both = 2 * X_array.copy()[:, 1:3]

    ct = ColumnTransformer([("trans1", "drop", [0])], remainder=DoubleTrans())

    assert_array_equal(ct.fit_transform(X_array), X_res_both)
    assert_array_equal(ct.fit(X_array).transform(X_array), X_res_both)
    assert len(ct.transformers_) == 2
    assert ct.transformers_[-1][0] == "remainder"
    assert isinstance(ct.transformers_[-1][1], DoubleTrans)
    assert_array_equal(ct.transformers_[-1][2], [1, 2])


def test_column_transformer_sparse_remainder_transformer():
    X_array = np.array([[0, 1, 2], [2, 4, 6], [8, 6, 4]]).T

    ct = ColumnTransformer(
        [("trans1", Trans(), [0])], remainder=SparseMatrixTrans(), sparse_threshold=0.8
    )

    X_trans = ct.fit_transform(X_array)
    assert sparse.issparse(X_trans)
    # SparseMatrixTrans creates 3 features for each column. There is
    # one column in ``transformers``, thus:
    assert X_trans.shape == (3, 3 + 1)

    exp_array = np.hstack((X_array[:, 0].reshape(-1, 1), np.eye(3)))
    assert_array_equal(X_trans.toarray(), exp_array)
    assert len(ct.transformers_) == 2
    assert ct.transformers_[-1][0] == "remainder"
    assert isinstance(ct.transformers_[-1][1], SparseMatrixTrans)
    assert_array_equal(ct.transformers_[-1][2], [1, 2])


def test_column_transformer_drop_all_sparse_remainder_transformer():
    X_array = np.array([[0, 1, 2], [2, 4, 6], [8, 6, 4]]).T
    ct = ColumnTransformer(
        [("trans1", "drop", [0])], remainder=SparseMatrixTrans(), sparse_threshold=0.8
    )

    X_trans = ct.fit_transform(X_array)
    assert sparse.issparse(X_trans)

    #  SparseMatrixTrans creates 3 features for each column, thus:
    assert X_trans.shape == (3, 3)
    assert_array_equal(X_trans.toarray(), np.eye(3))
    assert len(ct.transformers_) == 2
    assert ct.transformers_[-1][0] == "remainder"
    assert isinstance(ct.transformers_[-1][1], SparseMatrixTrans)
    assert_array_equal(ct.transformers_[-1][2], [1, 2])


def test_column_transformer_get_set_params_with_remainder():
    ct = ColumnTransformer(
        [("trans1", StandardScaler(), [0])], remainder=StandardScaler()
    )

    exp = {
        "n_jobs": None,
        "remainder": ct.remainder,
        "remainder__copy": True,
        "remainder__with_mean": True,
        "remainder__with_std": True,
        "sparse_threshold": 0.3,
        "trans1": ct.transformers[0][1],
        "trans1__copy": True,
        "trans1__with_mean": True,
        "trans1__with_std": True,
        "transformers": ct.transformers,
        "transformer_weights": None,
        "verbose_feature_names_out": True,
        "verbose": False,
    }

    assert ct.get_params() == exp

    ct.set_params(remainder__with_std=False)
    assert not ct.get_params()["remainder__with_std"]

    ct.set_params(trans1="passthrough")
    exp = {
        "n_jobs": None,
        "remainder": ct.remainder,
        "remainder__copy": True,
        "remainder__with_mean": True,
        "remainder__with_std": False,
        "sparse_threshold": 0.3,
        "trans1": "passthrough",
        "transformers": ct.transformers,
        "transformer_weights": None,
        "verbose_feature_names_out": True,
        "verbose": False,
    }
    assert ct.get_params() == exp


def test_column_transformer_no_estimators():
    X_array = np.array([[0, 1, 2], [2, 4, 6], [8, 6, 4]]).astype("float").T
    ct = ColumnTransformer([], remainder=StandardScaler())

    params = ct.get_params()
    assert params["remainder__with_mean"]

    X_trans = ct.fit_transform(X_array)
    assert X_trans.shape == X_array.shape
    assert len(ct.transformers_) == 1
    assert ct.transformers_[-1][0] == "remainder"
    assert ct.transformers_[-1][2] == [0, 1, 2]


@pytest.mark.parametrize(
    ["est", "pattern"],
    [
        (
            ColumnTransformer(
                [("trans1", Trans(), [0]), ("trans2", Trans(), [1])],
                remainder=DoubleTrans(),
            ),
            (
                r"\[ColumnTransformer\].*\(1 of 3\) Processing trans1.* total=.*\n"
                r"\[ColumnTransformer\].*\(2 of 3\) Processing trans2.* total=.*\n"
                r"\[ColumnTransformer\].*\(3 of 3\) Processing remainder.* total=.*\n$"
            ),
        ),
        (
            ColumnTransformer(
                [("trans1", Trans(), [0]), ("trans2", Trans(), [1])],
                remainder="passthrough",
            ),
            (
                r"\[ColumnTransformer\].*\(1 of 3\) Processing trans1.* total=.*\n"
                r"\[ColumnTransformer\].*\(2 of 3\) Processing trans2.* total=.*\n"
                r"\[ColumnTransformer\].*\(3 of 3\) Processing remainder.* total=.*\n$"
            ),
        ),
        (
            ColumnTransformer(
                [("trans1", Trans(), [0]), ("trans2", "drop", [1])],
                remainder="passthrough",
            ),
            (
                r"\[ColumnTransformer\].*\(1 of 2\) Processing trans1.* total=.*\n"
                r"\[ColumnTransformer\].*\(2 of 2\) Processing remainder.* total=.*\n$"
            ),
        ),
        (
            ColumnTransformer(
                [("trans1", Trans(), [0]), ("trans2", "passthrough", [1])],
                remainder="passthrough",
            ),
            (
                r"\[ColumnTransformer\].*\(1 of 3\) Processing trans1.* total=.*\n"
                r"\[ColumnTransformer\].*\(2 of 3\) Processing trans2.* total=.*\n"
                r"\[ColumnTransformer\].*\(3 of 3\) Processing remainder.* total=.*\n$"
            ),
        ),
        (
            ColumnTransformer([("trans1", Trans(), [0])], remainder="passthrough"),
            (
                r"\[ColumnTransformer\].*\(1 of 2\) Processing trans1.* total=.*\n"
                r"\[ColumnTransformer\].*\(2 of 2\) Processing remainder.* total=.*\n$"
            ),
        ),
        (
            ColumnTransformer(
                [("trans1", Trans(), [0]), ("trans2", Trans(), [1])], remainder="drop"
            ),
            (
                r"\[ColumnTransformer\].*\(1 of 2\) Processing trans1.* total=.*\n"
                r"\[ColumnTransformer\].*\(2 of 2\) Processing trans2.* total=.*\n$"
            ),
        ),
        (
            ColumnTransformer([("trans1", Trans(), [0])], remainder="drop"),
            r"\[ColumnTransformer\].*\(1 of 1\) Processing trans1.* total=.*\n$",
        ),
    ],
)
@pytest.mark.parametrize("method", ["fit", "fit_transform"])
def test_column_transformer_verbose(est, pattern, method, capsys):
    X_array = np.array([[0, 1, 2], [2, 4, 6], [8, 6, 4]]).T

    func = getattr(est, method)
    est.set_params(verbose=False)
    func(X_array)
    assert not capsys.readouterr().out, "Got output for verbose=False"

    est.set_params(verbose=True)
    func(X_array)
    assert re.match(pattern, capsys.readouterr()[0])


def test_column_transformer_no_estimators_set_params():
    ct = ColumnTransformer([]).set_params(n_jobs=2)
    assert ct.n_jobs == 2


def test_column_transformer_callable_specifier():
    # assert that function gets the full array
    X_array = np.array([[0, 1, 2], [2, 4, 6]]).T
    X_res_first = np.array([[0, 1, 2]]).T

    def func(X):
        assert_array_equal(X, X_array)
        return [0]

    ct = ColumnTransformer([("trans", Trans(), func)], remainder="drop")
    assert_array_equal(ct.fit_transform(X_array), X_res_first)
    assert_array_equal(ct.fit(X_array).transform(X_array), X_res_first)
    assert callable(ct.transformers[0][2])
    assert ct.transformers_[0][2] == [0]


def test_column_transformer_callable_specifier_dataframe():
    # assert that function gets the full dataframe
    pd = pytest.importorskip("pandas")
    X_array = np.array([[0, 1, 2], [2, 4, 6]]).T
    X_res_first = np.array([[0, 1, 2]]).T

    X_df = pd.DataFrame(X_array, columns=["first", "second"])

    def func(X):
        assert_array_equal(X.columns, X_df.columns)
        assert_array_equal(X.values, X_df.values)
        return ["first"]

    ct = ColumnTransformer([("trans", Trans(), func)], remainder="drop")
    assert_array_equal(ct.fit_transform(X_df), X_res_first)
    assert_array_equal(ct.fit(X_df).transform(X_df), X_res_first)
    assert callable(ct.transformers[0][2])
    assert ct.transformers_[0][2] == ["first"]


def test_column_transformer_negative_column_indexes():
    X = np.random.randn(2, 2)
    X_categories = np.array([[1], [2]])
    X = np.concatenate([X, X_categories], axis=1)

    ohe = OneHotEncoder()

    tf_1 = ColumnTransformer([("ohe", ohe, [-1])], remainder="passthrough")
    tf_2 = ColumnTransformer([("ohe", ohe, [2])], remainder="passthrough")
    assert_array_equal(tf_1.fit_transform(X), tf_2.fit_transform(X))


@pytest.mark.parametrize("array_type", [np.asarray, sparse.csr_matrix])
def test_column_transformer_mask_indexing(array_type):
    # Regression test for #14510
    # Boolean array-like does not behave as boolean array with NumPy < 1.12
    # and sparse matrices as well
    X = np.transpose([[1, 2, 3], [4, 5, 6], [5, 6, 7], [8, 9, 10]])
    X = array_type(X)
    column_transformer = ColumnTransformer(
        [("identity", FunctionTransformer(), [False, True, False, True])]
    )
    X_trans = column_transformer.fit_transform(X)
    assert X_trans.shape == (3, 2)


def test_n_features_in():
    # make sure n_features_in is what is passed as input to the column
    # transformer.

    X = [[1, 2], [3, 4], [5, 6]]
    ct = ColumnTransformer([("a", DoubleTrans(), [0]), ("b", DoubleTrans(), [1])])
    assert not hasattr(ct, "n_features_in_")
    ct.fit(X)
    assert ct.n_features_in_ == 2


@pytest.mark.parametrize(
    "cols, pattern, include, exclude",
    [
        (["col_int", "col_float"], None, np.number, None),
        (["col_int", "col_float"], None, None, object),
        (["col_int", "col_float"], None, [int, float], None),
        (["col_str"], None, [object], None),
        (["col_str"], None, object, None),
        (["col_float"], None, float, None),
        (["col_float"], "at$", [np.number], None),
        (["col_int"], None, [int], None),
        (["col_int"], "^col_int", [np.number], None),
        (["col_float", "col_str"], "float|str", None, None),
        (["col_str"], "^col_s", None, [int]),
        ([], "str$", float, None),
        (["col_int", "col_float", "col_str"], None, [np.number, object], None),
    ],
)
def test_make_column_selector_with_select_dtypes(cols, pattern, include, exclude):
    pd = pytest.importorskip("pandas")

    X_df = pd.DataFrame(
        {
            "col_int": np.array([0, 1, 2], dtype=int),
            "col_float": np.array([0.0, 1.0, 2.0], dtype=float),
            "col_str": ["one", "two", "three"],
        },
        columns=["col_int", "col_float", "col_str"],
    )

    selector = make_column_selector(
        dtype_include=include, dtype_exclude=exclude, pattern=pattern
    )

    assert_array_equal(selector(X_df), cols)


def test_column_transformer_with_make_column_selector():
    # Functional test for column transformer + column selector
    pd = pytest.importorskip("pandas")
    X_df = pd.DataFrame(
        {
            "col_int": np.array([0, 1, 2], dtype=int),
            "col_float": np.array([0.0, 1.0, 2.0], dtype=float),
            "col_cat": ["one", "two", "one"],
            "col_str": ["low", "middle", "high"],
        },
        columns=["col_int", "col_float", "col_cat", "col_str"],
    )
    X_df["col_str"] = X_df["col_str"].astype("category")

    cat_selector = make_column_selector(dtype_include=["category", object])
    num_selector = make_column_selector(dtype_include=np.number)

    ohe = OneHotEncoder()
    scaler = StandardScaler()

    ct_selector = make_column_transformer((ohe, cat_selector), (scaler, num_selector))
    ct_direct = make_column_transformer(
        (ohe, ["col_cat", "col_str"]), (scaler, ["col_float", "col_int"])
    )

    X_selector = ct_selector.fit_transform(X_df)
    X_direct = ct_direct.fit_transform(X_df)

    assert_allclose(X_selector, X_direct)


def test_make_column_selector_error():
    selector = make_column_selector(dtype_include=np.number)
    X = np.array([[0.1, 0.2]])
    msg = "make_column_selector can only be applied to pandas dataframes"
    with pytest.raises(ValueError, match=msg):
        selector(X)


def test_make_column_selector_pickle():
    pd = pytest.importorskip("pandas")

    X_df = pd.DataFrame(
        {
            "col_int": np.array([0, 1, 2], dtype=int),
            "col_float": np.array([0.0, 1.0, 2.0], dtype=float),
            "col_str": ["one", "two", "three"],
        },
        columns=["col_int", "col_float", "col_str"],
    )

    selector = make_column_selector(dtype_include=[object])
    selector_picked = pickle.loads(pickle.dumps(selector))

    assert_array_equal(selector(X_df), selector_picked(X_df))


# TODO: Remove in 1.2 when get_feature_names is removed.
@pytest.mark.filterwarnings("ignore::FutureWarning:sklearn")
@pytest.mark.parametrize(
    "empty_col",
    [[], np.array([], dtype=int), lambda x: []],
    ids=["list", "array", "callable"],
)
@pytest.mark.parametrize(
    "get_names, expected_names",
    [
        ("get_feature_names", ["ohe__x0_a", "ohe__x0_b", "ohe__x1_z"]),
        ("get_feature_names_out", ["ohe__col1_a", "ohe__col1_b", "ohe__col2_z"]),
    ],
)
def test_feature_names_empty_columns(empty_col, get_names, expected_names):
    pd = pytest.importorskip("pandas")

    df = pd.DataFrame({"col1": ["a", "a", "b"], "col2": ["z", "z", "z"]})

    ct = ColumnTransformer(
        transformers=[
            ("ohe", OneHotEncoder(), ["col1", "col2"]),
            ("empty_features", OneHotEncoder(), empty_col),
        ],
    )

    ct.fit(df)
    assert_array_equal(getattr(ct, get_names)(), expected_names)


@pytest.mark.parametrize(
    "selector",
    [
        [1],
        lambda x: [1],
        ["col2"],
        lambda x: ["col2"],
        [False, True],
        lambda x: [False, True],
    ],
)
def test_feature_names_out_pandas(selector):
    """Checks name when selecting only the second column"""
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"col1": ["a", "a", "b"], "col2": ["z", "z", "z"]})
    ct = ColumnTransformer([("ohe", OneHotEncoder(), selector)])
    ct.fit(df)

    assert_array_equal(ct.get_feature_names_out(), ["ohe__col2_z"])


@pytest.mark.parametrize(
    "selector", [[1], lambda x: [1], [False, True], lambda x: [False, True]]
)
def test_feature_names_out_non_pandas(selector):
    """Checks name when selecting the second column with numpy array"""
    X = [["a", "z"], ["a", "z"], ["b", "z"]]
    ct = ColumnTransformer([("ohe", OneHotEncoder(), selector)])
    ct.fit(X)

    assert_array_equal(ct.get_feature_names_out(), ["ohe__x1_z"])


@pytest.mark.parametrize("remainder", ["passthrough", StandardScaler()])
def test_sk_visual_block_remainder(remainder):
    # remainder='passthrough' or an estimator will be shown in repr_html
    ohe = OneHotEncoder()
    ct = ColumnTransformer(
        transformers=[("ohe", ohe, ["col1", "col2"])], remainder=remainder
    )
    visual_block = ct._sk_visual_block_()
    assert visual_block.names == ("ohe", "remainder")
    assert visual_block.name_details == (["col1", "col2"], "")
    assert visual_block.estimators == (ohe, remainder)


def test_sk_visual_block_remainder_drop():
    # remainder='drop' is not shown in repr_html
    ohe = OneHotEncoder()
    ct = ColumnTransformer(transformers=[("ohe", ohe, ["col1", "col2"])])
    visual_block = ct._sk_visual_block_()
    assert visual_block.names == ("ohe",)
    assert visual_block.name_details == (["col1", "col2"],)
    assert visual_block.estimators == (ohe,)


@pytest.mark.parametrize("remainder", ["passthrough", StandardScaler()])
def test_sk_visual_block_remainder_fitted_pandas(remainder):
    # Remainder shows the columns after fitting
    pd = pytest.importorskip("pandas")
    ohe = OneHotEncoder()
    ct = ColumnTransformer(
        transformers=[("ohe", ohe, ["col1", "col2"])], remainder=remainder
    )
    df = pd.DataFrame(
        {
            "col1": ["a", "b", "c"],
            "col2": ["z", "z", "z"],
            "col3": [1, 2, 3],
            "col4": [3, 4, 5],
        }
    )
    ct.fit(df)
    visual_block = ct._sk_visual_block_()
    assert visual_block.names == ("ohe", "remainder")
    assert visual_block.name_details == (["col1", "col2"], ["col3", "col4"])
    assert visual_block.estimators == (ohe, remainder)


@pytest.mark.parametrize("remainder", ["passthrough", StandardScaler()])
def test_sk_visual_block_remainder_fitted_numpy(remainder):
    # Remainder shows the indices after fitting
    X = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
    scaler = StandardScaler()
    ct = ColumnTransformer(
        transformers=[("scale", scaler, [0, 2])], remainder=remainder
    )
    ct.fit(X)
    visual_block = ct._sk_visual_block_()
    assert visual_block.names == ("scale", "remainder")
    assert visual_block.name_details == ([0, 2], [1])
    assert visual_block.estimators == (scaler, remainder)


# TODO: Remove in 1.2 when get_feature_names is removed
def test_column_transformers_get_feature_names_deprecated():
    """Check that get_feature_names is deprecated"""
    X = np.array([[0, 1], [2, 4]])
    ct = ColumnTransformer([("trans", "passthrough", [0, 1])])
    ct.fit(X)

    msg = "get_feature_names is deprecated in 1.0"
    with pytest.warns(FutureWarning, match=msg):
        ct.get_feature_names()


@pytest.mark.parametrize("explicit_colname", ["first", "second", 0, 1])
@pytest.mark.parametrize("remainder", [Trans(), "passthrough", "drop"])
def test_column_transformer_reordered_column_names_remainder(
    explicit_colname, remainder
):
    """Test the interaction between remainder and column transformer"""
    pd = pytest.importorskip("pandas")

    X_fit_array = np.array([[0, 1, 2], [2, 4, 6]]).T
    X_fit_df = pd.DataFrame(X_fit_array, columns=["first", "second"])

    X_trans_array = np.array([[2, 4, 6], [0, 1, 2]]).T
    X_trans_df = pd.DataFrame(X_trans_array, columns=["second", "first"])

    tf = ColumnTransformer([("bycol", Trans(), explicit_colname)], remainder=remainder)

    tf.fit(X_fit_df)
    X_fit_trans = tf.transform(X_fit_df)

    # Changing the order still works
    X_trans = tf.transform(X_trans_df)
    assert_allclose(X_trans, X_fit_trans)

    # extra columns are ignored
    X_extended_df = X_fit_df.copy()
    X_extended_df["third"] = [3, 6, 9]
    X_trans = tf.transform(X_extended_df)
    assert_allclose(X_trans, X_fit_trans)

    if isinstance(explicit_colname, str):
        # Raise error if columns are specified by names but input only allows
        # to specify by position, e.g. numpy array instead of a pandas df.
        X_array = X_fit_array.copy()
        err_msg = "Specifying the columns"
        with pytest.raises(ValueError, match=err_msg):
            tf.transform(X_array)


def test_feature_name_validation_missing_columns_drop_passthough():
    """Test the interaction between {'drop', 'passthrough'} and
    missing column names."""
    pd = pytest.importorskip("pandas")

    X = np.ones(shape=(3, 4))
    df = pd.DataFrame(X, columns=["a", "b", "c", "d"])

    df_dropped = df.drop("c", axis=1)

    # with remainder='passthrough', all columns seen during `fit` must be
    # present
    tf = ColumnTransformer([("bycol", Trans(), [1])], remainder="passthrough")
    tf.fit(df)
    msg = r"columns are missing: {'c'}"
    with pytest.raises(ValueError, match=msg):
        tf.transform(df_dropped)

    # with remainder='drop', it is allowed to have column 'c' missing
    tf = ColumnTransformer([("bycol", Trans(), [1])], remainder="drop")
    tf.fit(df)

    df_dropped_trans = tf.transform(df_dropped)
    df_fit_trans = tf.transform(df)
    assert_allclose(df_dropped_trans, df_fit_trans)

    # bycol drops 'c', thus it is allowed for 'c' to be missing
    tf = ColumnTransformer([("bycol", "drop", ["c"])], remainder="passthrough")
    tf.fit(df)
    df_dropped_trans = tf.transform(df_dropped)
    df_fit_trans = tf.transform(df)
    assert_allclose(df_dropped_trans, df_fit_trans)


# TODO: Remove in 1.2 when get_feature_names is removed.
@pytest.mark.filterwarnings("ignore::FutureWarning:sklearn")
@pytest.mark.parametrize("selector", [[], [False, False]])
def test_get_feature_names_empty_selection(selector):
    """Test that get_feature_names is only called for transformers that
    were selected. Non-regression test for #19550.
    """
    ct = ColumnTransformer([("ohe", OneHotEncoder(drop="first"), selector)])
    ct.fit([[1, 2], [3, 4]])
    assert ct.get_feature_names() == []


def test_feature_names_in_():
    """Feature names are stored in column transformer.

    Column transformer deliberately does not check for column name consistency.
    It only checks that the non-dropped names seen in `fit` are seen
    in `transform`. This behavior is already tested in
    `test_feature_name_validation_missing_columns_drop_passthough`"""

    pd = pytest.importorskip("pandas")

    feature_names = ["a", "c", "d"]
    df = pd.DataFrame([[1, 2, 3]], columns=feature_names)
    ct = ColumnTransformer([("bycol", Trans(), ["a", "d"])], remainder="passthrough")

    ct.fit(df)
    assert_array_equal(ct.feature_names_in_, feature_names)
    assert isinstance(ct.feature_names_in_, np.ndarray)
    assert ct.feature_names_in_.dtype == object


class TransWithNames(Trans):
    def __init__(self, feature_names_out=None):
        self.feature_names_out = feature_names_out

    def get_feature_names_out(self, input_features=None):
        if self.feature_names_out is not None:
            return np.asarray(self.feature_names_out, dtype=object)
        return input_features


@pytest.mark.parametrize(
    "transformers, remainder, expected_names",
    [
        (
            [
                ("bycol1", TransWithNames(), ["d", "c"]),
                ("bycol2", "passthrough", ["d"]),
            ],
            "passthrough",
            ["bycol1__d", "bycol1__c", "bycol2__d", "remainder__a", "remainder__b"],
        ),
        (
            [
                ("bycol1", TransWithNames(), ["d", "c"]),
                ("bycol2", "passthrough", ["d"]),
            ],
            "drop",
            ["bycol1__d", "bycol1__c", "bycol2__d"],
        ),
        (
            [
                ("bycol1", TransWithNames(), ["b"]),
                ("bycol2", "drop", ["d"]),
            ],
            "passthrough",
            ["bycol1__b", "remainder__a", "remainder__c"],
        ),
        (
            [
                ("bycol1", TransWithNames(["pca1", "pca2"]), ["a", "b", "d"]),
            ],
            "passthrough",
            ["bycol1__pca1", "bycol1__pca2", "remainder__c"],
        ),
        (
            [
                ("bycol1", TransWithNames(["a", "b"]), ["d"]),
                ("bycol2", "passthrough", ["b"]),
            ],
            "drop",
            ["bycol1__a", "bycol1__b", "bycol2__b"],
        ),
        (
            [
                ("bycol1", TransWithNames([f"pca{i}" for i in range(2)]), ["b"]),
                ("bycol2", TransWithNames([f"pca{i}" for i in range(2)]), ["b"]),
            ],
            "passthrough",
            [
                "bycol1__pca0",
                "bycol1__pca1",
                "bycol2__pca0",
                "bycol2__pca1",
                "remainder__a",
                "remainder__c",
                "remainder__d",
            ],
        ),
        (
            [
                ("bycol1", "drop", ["d"]),
            ],
            "drop",
            [],
        ),
    ],
)
def test_verbose_feature_names_out_true(transformers, remainder, expected_names):
    """Check feature_names_out for verbose_feature_names_out=True (default)"""
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame([[1, 2, 3, 4]], columns=["a", "b", "c", "d"])
    ct = ColumnTransformer(
        transformers,
        remainder=remainder,
    )
    ct.fit(df)

    names = ct.get_feature_names_out()
    assert isinstance(names, np.ndarray)
    assert names.dtype == object
    assert_array_equal(names, expected_names)


@pytest.mark.parametrize(
    "transformers, remainder, expected_names",
    [
        (
            [
                ("bycol1", TransWithNames(), ["d", "c"]),
                ("bycol2", "passthrough", ["a"]),
            ],
            "passthrough",
            ["d", "c", "a", "b"],
        ),
        (
            [
                ("bycol1", TransWithNames(["a"]), ["d", "c"]),
                ("bycol2", "passthrough", ["d"]),
            ],
            "drop",
            ["a", "d"],
        ),
        (
            [
                ("bycol1", TransWithNames(), ["b"]),
                ("bycol2", "drop", ["d"]),
            ],
            "passthrough",
            ["b", "a", "c"],
        ),
        (
            [
                ("bycol1", TransWithNames(["pca1", "pca2"]), ["a", "b", "d"]),
            ],
            "passthrough",
            ["pca1", "pca2", "c"],
        ),
        (
            [
                ("bycol1", TransWithNames(["a", "c"]), ["d"]),
                ("bycol2", "passthrough", ["d"]),
            ],
            "drop",
            ["a", "c", "d"],
        ),
        (
            [
                ("bycol1", TransWithNames([f"pca{i}" for i in range(2)]), ["b"]),
                ("bycol2", TransWithNames([f"kpca{i}" for i in range(2)]), ["b"]),
            ],
            "passthrough",
            ["pca0", "pca1", "kpca0", "kpca1", "a", "c", "d"],
        ),
        (
            [
                ("bycol1", "drop", ["d"]),
            ],
            "drop",
            [],
        ),
    ],
)
def test_verbose_feature_names_out_false(transformers, remainder, expected_names):
    """Check feature_names_out for verbose_feature_names_out=False"""
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame([[1, 2, 3, 4]], columns=["a", "b", "c", "d"])
    ct = ColumnTransformer(
        transformers,
        remainder=remainder,
        verbose_feature_names_out=False,
    )
    ct.fit(df)

    names = ct.get_feature_names_out()
    assert isinstance(names, np.ndarray)
    assert names.dtype == object
    assert_array_equal(names, expected_names)


@pytest.mark.parametrize(
    "transformers, remainder, colliding_columns",
    [
        (
            [
                ("bycol1", TransWithNames(), ["b"]),
                ("bycol2", "passthrough", ["b"]),
            ],
            "drop",
            "['b']",
        ),
        (
            [
                ("bycol1", TransWithNames(["c", "d"]), ["c"]),
                ("bycol2", "passthrough", ["c"]),
            ],
            "drop",
            "['c']",
        ),
        (
            [
                ("bycol1", TransWithNames(["a"]), ["b"]),
                ("bycol2", "passthrough", ["b"]),
            ],
            "passthrough",
            "['a']",
        ),
        (
            [
                ("bycol1", TransWithNames(["a"]), ["b"]),
                ("bycol2", "drop", ["b"]),
            ],
            "passthrough",
            "['a']",
        ),
        (
            [
                ("bycol1", TransWithNames(["c", "b"]), ["b"]),
                ("bycol2", "passthrough", ["c", "b"]),
            ],
            "drop",
            "['b', 'c']",
        ),
        (
            [
                ("bycol1", TransWithNames(["a"]), ["b"]),
                ("bycol2", "passthrough", ["a"]),
                ("bycol3", TransWithNames(["a"]), ["b"]),
            ],
            "passthrough",
            "['a']",
        ),
        (
            [
                ("bycol1", TransWithNames(["a", "b"]), ["b"]),
                ("bycol2", "passthrough", ["a"]),
                ("bycol3", TransWithNames(["b"]), ["c"]),
            ],
            "passthrough",
            "['a', 'b']",
        ),
        (
            [
                ("bycol1", TransWithNames([f"pca{i}" for i in range(6)]), ["b"]),
                ("bycol2", TransWithNames([f"pca{i}" for i in range(6)]), ["b"]),
            ],
            "passthrough",
            "['pca0', 'pca1', 'pca2', 'pca3', 'pca4', ...]",
        ),
    ],
)
def test_verbose_feature_names_out_false_errors(
    transformers, remainder, colliding_columns
):
    """Check feature_names_out for verbose_feature_names_out=False"""

    pd = pytest.importorskip("pandas")
    df = pd.DataFrame([[1, 2, 3, 4]], columns=["a", "b", "c", "d"])
    ct = ColumnTransformer(
        transformers,
        remainder=remainder,
        verbose_feature_names_out=False,
    )
    ct.fit(df)

    msg = re.escape(
        f"Output feature names: {colliding_columns} are not unique. Please set "
        "verbose_feature_names_out=True to add prefixes to feature names"
    )
    with pytest.raises(ValueError, match=msg):
        ct.get_feature_names_out()
