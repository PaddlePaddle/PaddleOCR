"""Tests for input validation functions"""

import numbers
import warnings
import os
import re

from tempfile import NamedTemporaryFile
from itertools import product
from operator import itemgetter

import pytest
from pytest import importorskip
import numpy as np
import scipy.sparse as sp

from sklearn.utils._testing import assert_no_warnings
from sklearn.utils._testing import ignore_warnings
from sklearn.utils._testing import SkipTest
from sklearn.utils._testing import assert_array_equal
from sklearn.utils._testing import assert_allclose_dense_sparse
from sklearn.utils._testing import assert_allclose
from sklearn.utils._testing import _convert_container
from sklearn.utils import as_float_array, check_array, check_symmetric
from sklearn.utils import check_X_y
from sklearn.utils import deprecated
from sklearn.utils._mocking import MockDataFrame
from sklearn.utils.fixes import parse_version
from sklearn.utils.estimator_checks import _NotAnArray
from sklearn.random_projection import _sparse_random_matrix
from sklearn.linear_model import ARDRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.datasets import make_blobs
from sklearn.utils import _safe_indexing
from sklearn.utils.validation import (
    has_fit_parameter,
    check_is_fitted,
    check_consistent_length,
    assert_all_finite,
    check_memory,
    check_non_negative,
    _num_samples,
    check_scalar,
    _check_psd_eigenvalues,
    _check_y,
    _deprecate_positional_args,
    _check_sample_weight,
    _allclose_dense_sparse,
    _num_features,
    FLOAT_DTYPES,
    _get_feature_names,
    _check_feature_names_in,
)
from sklearn.utils.validation import _check_fit_params
from sklearn.base import BaseEstimator
import sklearn

from sklearn.exceptions import NotFittedError, PositiveSpectrumWarning

from sklearn.utils._testing import TempMemmap


# TODO: Remove np.matrix usage in 1.2
@pytest.mark.filterwarnings("ignore:np.matrix usage is deprecated in 1.0:FutureWarning")
@pytest.mark.filterwarnings("ignore:the matrix subclass:PendingDeprecationWarning")
def test_as_float_array():
    # Test function for as_float_array
    X = np.ones((3, 10), dtype=np.int32)
    X = X + np.arange(10, dtype=np.int32)
    X2 = as_float_array(X, copy=False)
    assert X2.dtype == np.float32
    # Another test
    X = X.astype(np.int64)
    X2 = as_float_array(X, copy=True)
    # Checking that the array wasn't overwritten
    assert as_float_array(X, copy=False) is not X
    assert X2.dtype == np.float64
    # Test int dtypes <= 32bit
    tested_dtypes = [bool, np.int8, np.int16, np.int32, np.uint8, np.uint16, np.uint32]
    for dtype in tested_dtypes:
        X = X.astype(dtype)
        X2 = as_float_array(X)
        assert X2.dtype == np.float32

    # Test object dtype
    X = X.astype(object)
    X2 = as_float_array(X, copy=True)
    assert X2.dtype == np.float64

    # Here, X is of the right type, it shouldn't be modified
    X = np.ones((3, 2), dtype=np.float32)
    assert as_float_array(X, copy=False) is X
    # Test that if X is fortran ordered it stays
    X = np.asfortranarray(X)
    assert np.isfortran(as_float_array(X, copy=True))

    # Test the copy parameter with some matrices
    matrices = [
        np.matrix(np.arange(5)),
        sp.csc_matrix(np.arange(5)).toarray(),
        _sparse_random_matrix(10, 10, density=0.10).toarray(),
    ]
    for M in matrices:
        N = as_float_array(M, copy=True)
        N[0, 0] = np.nan
        assert not np.isnan(M).any()


@pytest.mark.parametrize("X", [(np.random.random((10, 2))), (sp.rand(10, 2).tocsr())])
def test_as_float_array_nan(X):
    X[5, 0] = np.nan
    X[6, 1] = np.nan
    X_converted = as_float_array(X, force_all_finite="allow-nan")
    assert_allclose_dense_sparse(X_converted, X)


# TODO: Remove np.matrix usage in 1.2
@pytest.mark.filterwarnings("ignore:np.matrix usage is deprecated in 1.0:FutureWarning")
@pytest.mark.filterwarnings("ignore:the matrix subclass:PendingDeprecationWarning")
def test_np_matrix():
    # Confirm that input validation code does not return np.matrix
    X = np.arange(12).reshape(3, 4)

    assert not isinstance(as_float_array(X), np.matrix)
    assert not isinstance(as_float_array(np.matrix(X)), np.matrix)
    assert not isinstance(as_float_array(sp.csc_matrix(X)), np.matrix)


def test_memmap():
    # Confirm that input validation code doesn't copy memory mapped arrays

    asflt = lambda x: as_float_array(x, copy=False)

    with NamedTemporaryFile(prefix="sklearn-test") as tmp:
        M = np.memmap(tmp, shape=(10, 10), dtype=np.float32)
        M[:] = 0

        for f in (check_array, np.asarray, asflt):
            X = f(M)
            X[:] = 1
            assert_array_equal(X.ravel(), M.ravel())
            X[:] = 0


def test_ordering():
    # Check that ordering is enforced correctly by validation utilities.
    # We need to check each validation utility, because a 'copy' without
    # 'order=K' will kill the ordering.
    X = np.ones((10, 5))
    for A in X, X.T:
        for copy in (True, False):
            B = check_array(A, order="C", copy=copy)
            assert B.flags["C_CONTIGUOUS"]
            B = check_array(A, order="F", copy=copy)
            assert B.flags["F_CONTIGUOUS"]
            if copy:
                assert A is not B

    X = sp.csr_matrix(X)
    X.data = X.data[::-1]
    assert not X.data.flags["C_CONTIGUOUS"]


@pytest.mark.parametrize(
    "value, force_all_finite", [(np.inf, False), (np.nan, "allow-nan"), (np.nan, False)]
)
@pytest.mark.parametrize("retype", [np.asarray, sp.csr_matrix])
def test_check_array_force_all_finite_valid(value, force_all_finite, retype):
    X = retype(np.arange(4).reshape(2, 2).astype(float))
    X[0, 0] = value
    X_checked = check_array(X, force_all_finite=force_all_finite, accept_sparse=True)
    assert_allclose_dense_sparse(X, X_checked)


@pytest.mark.parametrize(
    "value, force_all_finite, match_msg",
    [
        (np.inf, True, "Input contains NaN, infinity"),
        (np.inf, "allow-nan", "Input contains infinity"),
        (np.nan, True, "Input contains NaN, infinity"),
        (np.nan, "allow-inf", 'force_all_finite should be a bool or "allow-nan"'),
        (np.nan, 1, "Input contains NaN, infinity"),
    ],
)
@pytest.mark.parametrize("retype", [np.asarray, sp.csr_matrix])
def test_check_array_force_all_finiteinvalid(
    value, force_all_finite, match_msg, retype
):
    X = retype(np.arange(4).reshape(2, 2).astype(float))
    X[0, 0] = value
    with pytest.raises(ValueError, match=match_msg):
        check_array(X, force_all_finite=force_all_finite, accept_sparse=True)


def test_check_array_force_all_finite_object():
    X = np.array([["a", "b", np.nan]], dtype=object).T

    X_checked = check_array(X, dtype=None, force_all_finite="allow-nan")
    assert X is X_checked

    X_checked = check_array(X, dtype=None, force_all_finite=False)
    assert X is X_checked

    with pytest.raises(ValueError, match="Input contains NaN"):
        check_array(X, dtype=None, force_all_finite=True)


@pytest.mark.parametrize(
    "X, err_msg",
    [
        (
            np.array([[1, np.nan]]),
            "Input contains NaN, infinity or a value too large for.*int",
        ),
        (
            np.array([[1, np.nan]]),
            "Input contains NaN, infinity or a value too large for.*int",
        ),
        (
            np.array([[1, np.inf]]),
            "Input contains NaN, infinity or a value too large for.*int",
        ),
        (np.array([[1, np.nan]], dtype=object), "cannot convert float NaN to integer"),
    ],
)
@pytest.mark.parametrize("force_all_finite", [True, False])
def test_check_array_force_all_finite_object_unsafe_casting(
    X, err_msg, force_all_finite
):
    # casting a float array containing NaN or inf to int dtype should
    # raise an error irrespective of the force_all_finite parameter.
    with pytest.raises(ValueError, match=err_msg):
        check_array(X, dtype=int, force_all_finite=force_all_finite)


@ignore_warnings
def test_check_array():
    # accept_sparse == False
    # raise error on sparse inputs
    X = [[1, 2], [3, 4]]
    X_csr = sp.csr_matrix(X)
    with pytest.raises(TypeError):
        check_array(X_csr)

    # ensure_2d=False
    X_array = check_array([0, 1, 2], ensure_2d=False)
    assert X_array.ndim == 1
    # ensure_2d=True with 1d array
    with pytest.raises(ValueError, match="Expected 2D array, got 1D array instead"):
        check_array([0, 1, 2], ensure_2d=True)

    # ensure_2d=True with scalar array
    with pytest.raises(ValueError, match="Expected 2D array, got scalar array instead"):
        check_array(10, ensure_2d=True)

    # don't allow ndim > 3
    X_ndim = np.arange(8).reshape(2, 2, 2)
    with pytest.raises(ValueError):
        check_array(X_ndim)
    check_array(X_ndim, allow_nd=True)  # doesn't raise

    # dtype and order enforcement.
    X_C = np.arange(4).reshape(2, 2).copy("C")
    X_F = X_C.copy("F")
    X_int = X_C.astype(int)
    X_float = X_C.astype(float)
    Xs = [X_C, X_F, X_int, X_float]
    dtypes = [np.int32, int, float, np.float32, None, bool, object]
    orders = ["C", "F", None]
    copys = [True, False]

    for X, dtype, order, copy in product(Xs, dtypes, orders, copys):
        X_checked = check_array(X, dtype=dtype, order=order, copy=copy)
        if dtype is not None:
            assert X_checked.dtype == dtype
        else:
            assert X_checked.dtype == X.dtype
        if order == "C":
            assert X_checked.flags["C_CONTIGUOUS"]
            assert not X_checked.flags["F_CONTIGUOUS"]
        elif order == "F":
            assert X_checked.flags["F_CONTIGUOUS"]
            assert not X_checked.flags["C_CONTIGUOUS"]
        if copy:
            assert X is not X_checked
        else:
            # doesn't copy if it was already good
            if (
                X.dtype == X_checked.dtype
                and X_checked.flags["C_CONTIGUOUS"] == X.flags["C_CONTIGUOUS"]
                and X_checked.flags["F_CONTIGUOUS"] == X.flags["F_CONTIGUOUS"]
            ):
                assert X is X_checked

    # allowed sparse != None
    X_csc = sp.csc_matrix(X_C)
    X_coo = X_csc.tocoo()
    X_dok = X_csc.todok()
    X_int = X_csc.astype(int)
    X_float = X_csc.astype(float)

    Xs = [X_csc, X_coo, X_dok, X_int, X_float]
    accept_sparses = [["csr", "coo"], ["coo", "dok"]]
    for X, dtype, accept_sparse, copy in product(Xs, dtypes, accept_sparses, copys):
        with warnings.catch_warnings(record=True) as w:
            X_checked = check_array(
                X, dtype=dtype, accept_sparse=accept_sparse, copy=copy
            )
        if (dtype is object or sp.isspmatrix_dok(X)) and len(w):
            # XXX unreached code as of v0.22
            message = str(w[0].message)
            messages = [
                "object dtype is not supported by sparse matrices",
                "Can't check dok sparse matrix for nan or inf.",
            ]
            assert message in messages
        else:
            assert len(w) == 0
        if dtype is not None:
            assert X_checked.dtype == dtype
        else:
            assert X_checked.dtype == X.dtype
        if X.format in accept_sparse:
            # no change if allowed
            assert X.format == X_checked.format
        else:
            # got converted
            assert X_checked.format == accept_sparse[0]
        if copy:
            assert X is not X_checked
        else:
            # doesn't copy if it was already good
            if X.dtype == X_checked.dtype and X.format == X_checked.format:
                assert X is X_checked

    # other input formats
    # convert lists to arrays
    X_dense = check_array([[1, 2], [3, 4]])
    assert isinstance(X_dense, np.ndarray)
    # raise on too deep lists
    with pytest.raises(ValueError):
        check_array(X_ndim.tolist())
    check_array(X_ndim.tolist(), allow_nd=True)  # doesn't raise

    # convert weird stuff to arrays
    X_no_array = _NotAnArray(X_dense)
    result = check_array(X_no_array)
    assert isinstance(result, np.ndarray)


# TODO: Check for error in 1.1 when implicit conversion is removed
@pytest.mark.parametrize(
    "X",
    [
        [["1", "2"], ["3", "4"]],
        np.array([["1", "2"], ["3", "4"]], dtype="U"),
        np.array([["1", "2"], ["3", "4"]], dtype="S"),
        [[b"1", b"2"], [b"3", b"4"]],
        np.array([[b"1", b"2"], [b"3", b"4"]], dtype="V1"),
    ],
)
def test_check_array_numeric_warns(X):
    """Test that check_array warns when it converts a bytes/string into a
    float."""
    expected_msg = (
        r"Arrays of bytes/strings is being converted to decimal .*"
        r"deprecated in 0.24 and will be removed in 1.1"
    )
    with pytest.warns(FutureWarning, match=expected_msg):
        check_array(X, dtype="numeric")


# TODO: remove in 1.1
@ignore_warnings(category=FutureWarning)
@pytest.mark.parametrize(
    "X",
    [
        [["11", "12"], ["13", "xx"]],
        np.array([["11", "12"], ["13", "xx"]], dtype="U"),
        np.array([["11", "12"], ["13", "xx"]], dtype="S"),
        [[b"a", b"b"], [b"c", b"d"]],
    ],
)
def test_check_array_dtype_numeric_errors(X):
    """Error when string-ike array can not be converted"""
    expected_warn_msg = "Unable to convert array of bytes/strings"
    with pytest.raises(ValueError, match=expected_warn_msg):
        check_array(X, dtype="numeric")


@pytest.mark.parametrize("pd_dtype", ["Int8", "Int16", "UInt8", "UInt16"])
@pytest.mark.parametrize(
    "dtype, expected_dtype",
    [
        ([np.float32, np.float64], np.float32),
        (np.float64, np.float64),
        ("numeric", np.float64),
    ],
)
def test_check_array_pandas_na_support(pd_dtype, dtype, expected_dtype):
    # Test pandas IntegerArray with pd.NA
    pd = pytest.importorskip("pandas", minversion="1.0")

    X_np = np.array(
        [[1, 2, 3, np.nan, np.nan], [np.nan, np.nan, 8, 4, 6], [1, 2, 3, 4, 5]]
    ).T

    # Creates dataframe with IntegerArrays with pd.NA
    X = pd.DataFrame(X_np, dtype=pd_dtype, columns=["a", "b", "c"])
    # column c has no nans
    X["c"] = X["c"].astype("float")
    X_checked = check_array(X, force_all_finite="allow-nan", dtype=dtype)
    assert_allclose(X_checked, X_np)
    assert X_checked.dtype == expected_dtype

    X_checked = check_array(X, force_all_finite=False, dtype=dtype)
    assert_allclose(X_checked, X_np)
    assert X_checked.dtype == expected_dtype

    msg = "Input contains NaN, infinity"
    with pytest.raises(ValueError, match=msg):
        check_array(X, force_all_finite=True)


# TODO: remove test in 1.1 once this behavior is deprecated
def test_check_array_pandas_dtype_object_conversion():
    # test that data-frame like objects with dtype object
    # get converted
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=object)
    X_df = MockDataFrame(X)
    with pytest.warns(FutureWarning):
        assert check_array(X_df).dtype.kind == "f"
    with pytest.warns(FutureWarning):
        assert check_array(X_df, ensure_2d=False).dtype.kind == "f"
    # smoke-test against dataframes with column named "dtype"
    X_df.dtype = "Hans"
    with pytest.warns(FutureWarning):
        assert check_array(X_df, ensure_2d=False).dtype.kind == "f"


def test_check_array_pandas_dtype_casting():
    # test that data-frames with homogeneous dtype are not upcast
    pd = pytest.importorskip("pandas")
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    X_df = pd.DataFrame(X)
    assert check_array(X_df).dtype == np.float32
    assert check_array(X_df, dtype=FLOAT_DTYPES).dtype == np.float32

    X_df.iloc[:, 0] = X_df.iloc[:, 0].astype(np.float16)
    assert_array_equal(X_df.dtypes, (np.float16, np.float32, np.float32))
    assert check_array(X_df).dtype == np.float32
    assert check_array(X_df, dtype=FLOAT_DTYPES).dtype == np.float32

    X_df.iloc[:, 1] = X_df.iloc[:, 1].astype(np.int16)
    # float16, int16, float32 casts to float32
    assert check_array(X_df).dtype == np.float32
    assert check_array(X_df, dtype=FLOAT_DTYPES).dtype == np.float32

    X_df.iloc[:, 2] = X_df.iloc[:, 2].astype(np.float16)
    # float16, int16, float16 casts to float32
    assert check_array(X_df).dtype == np.float32
    assert check_array(X_df, dtype=FLOAT_DTYPES).dtype == np.float32

    X_df = X_df.astype(np.int16)
    assert check_array(X_df).dtype == np.int16
    # we're not using upcasting rules for determining
    # the target type yet, so we cast to the default of float64
    assert check_array(X_df, dtype=FLOAT_DTYPES).dtype == np.float64

    # check that we handle pandas dtypes in a semi-reasonable way
    # this is actually tricky because we can't really know that this
    # should be integer ahead of converting it.
    cat_df = pd.DataFrame({"cat_col": pd.Categorical([1, 2, 3])})
    assert check_array(cat_df).dtype == np.int64
    assert check_array(cat_df, dtype=FLOAT_DTYPES).dtype == np.float64


def test_check_array_on_mock_dataframe():
    arr = np.array([[0.2, 0.7], [0.6, 0.5], [0.4, 0.1], [0.7, 0.2]])
    mock_df = MockDataFrame(arr)
    checked_arr = check_array(mock_df)
    assert checked_arr.dtype == arr.dtype
    checked_arr = check_array(mock_df, dtype=np.float32)
    assert checked_arr.dtype == np.dtype(np.float32)


def test_check_array_dtype_stability():
    # test that lists with ints don't get converted to floats
    X = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    assert check_array(X).dtype.kind == "i"
    assert check_array(X, ensure_2d=False).dtype.kind == "i"


def test_check_array_dtype_warning():
    X_int_list = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    X_float32 = np.asarray(X_int_list, dtype=np.float32)
    X_int64 = np.asarray(X_int_list, dtype=np.int64)
    X_csr_float32 = sp.csr_matrix(X_float32)
    X_csc_float32 = sp.csc_matrix(X_float32)
    X_csc_int32 = sp.csc_matrix(X_int64, dtype=np.int32)
    integer_data = [X_int64, X_csc_int32]
    float32_data = [X_float32, X_csr_float32, X_csc_float32]
    for X in integer_data:
        X_checked = assert_no_warnings(
            check_array, X, dtype=np.float64, accept_sparse=True
        )
        assert X_checked.dtype == np.float64

    for X in float32_data:
        X_checked = assert_no_warnings(
            check_array, X, dtype=[np.float64, np.float32], accept_sparse=True
        )
        assert X_checked.dtype == np.float32
        assert X_checked is X

        X_checked = assert_no_warnings(
            check_array,
            X,
            dtype=[np.float64, np.float32],
            accept_sparse=["csr", "dok"],
            copy=True,
        )
        assert X_checked.dtype == np.float32
        assert X_checked is not X

    X_checked = assert_no_warnings(
        check_array,
        X_csc_float32,
        dtype=[np.float64, np.float32],
        accept_sparse=["csr", "dok"],
        copy=False,
    )
    assert X_checked.dtype == np.float32
    assert X_checked is not X_csc_float32
    assert X_checked.format == "csr"


def test_check_array_accept_sparse_type_exception():
    X = [[1, 2], [3, 4]]
    X_csr = sp.csr_matrix(X)
    invalid_type = SVR()

    msg = (
        "A sparse matrix was passed, but dense data is required. "
        r"Use X.toarray\(\) to convert to a dense numpy array."
    )
    with pytest.raises(TypeError, match=msg):
        check_array(X_csr, accept_sparse=False)

    msg = (
        "Parameter 'accept_sparse' should be a string, "
        "boolean or list of strings. You provided 'accept_sparse=.*'."
    )
    with pytest.raises(ValueError, match=msg):
        check_array(X_csr, accept_sparse=invalid_type)

    msg = (
        "When providing 'accept_sparse' as a tuple or list, "
        "it must contain at least one string value."
    )
    with pytest.raises(ValueError, match=msg):
        check_array(X_csr, accept_sparse=[])
    with pytest.raises(ValueError, match=msg):
        check_array(X_csr, accept_sparse=())
    with pytest.raises(TypeError, match="SVR"):
        check_array(X_csr, accept_sparse=[invalid_type])


def test_check_array_accept_sparse_no_exception():
    X = [[1, 2], [3, 4]]
    X_csr = sp.csr_matrix(X)

    check_array(X_csr, accept_sparse=True)
    check_array(X_csr, accept_sparse="csr")
    check_array(X_csr, accept_sparse=["csr"])
    check_array(X_csr, accept_sparse=("csr",))


@pytest.fixture(params=["csr", "csc", "coo", "bsr"])
def X_64bit(request):
    X = sp.rand(20, 10, format=request.param)
    for attr in ["indices", "indptr", "row", "col"]:
        if hasattr(X, attr):
            setattr(X, attr, getattr(X, attr).astype("int64"))
    yield X


def test_check_array_accept_large_sparse_no_exception(X_64bit):
    # When large sparse are allowed
    check_array(X_64bit, accept_large_sparse=True, accept_sparse=True)


def test_check_array_accept_large_sparse_raise_exception(X_64bit):
    # When large sparse are not allowed
    msg = (
        "Only sparse matrices with 32-bit integer indices "
        "are accepted. Got int64 indices."
    )
    with pytest.raises(ValueError, match=msg):
        check_array(X_64bit, accept_sparse=True, accept_large_sparse=False)


def test_check_array_min_samples_and_features_messages():
    # empty list is considered 2D by default:
    msg = r"0 feature\(s\) \(shape=\(1, 0\)\) while a minimum of 1 is" " required."
    with pytest.raises(ValueError, match=msg):
        check_array([[]])

    # If considered a 1D collection when ensure_2d=False, then the minimum
    # number of samples will break:
    msg = r"0 sample\(s\) \(shape=\(0,\)\) while a minimum of 1 is required."
    with pytest.raises(ValueError, match=msg):
        check_array([], ensure_2d=False)

    # Invalid edge case when checking the default minimum sample of a scalar
    msg = r"Singleton array array\(42\) cannot be considered a valid" " collection."
    with pytest.raises(TypeError, match=msg):
        check_array(42, ensure_2d=False)

    # Simulate a model that would need at least 2 samples to be well defined
    X = np.ones((1, 10))
    y = np.ones(1)
    msg = r"1 sample\(s\) \(shape=\(1, 10\)\) while a minimum of 2 is" " required."
    with pytest.raises(ValueError, match=msg):
        check_X_y(X, y, ensure_min_samples=2)

    # The same message is raised if the data has 2 dimensions even if this is
    # not mandatory
    with pytest.raises(ValueError, match=msg):
        check_X_y(X, y, ensure_min_samples=2, ensure_2d=False)

    # Simulate a model that would require at least 3 features (e.g. SelectKBest
    # with k=3)
    X = np.ones((10, 2))
    y = np.ones(2)
    msg = r"2 feature\(s\) \(shape=\(10, 2\)\) while a minimum of 3 is" " required."
    with pytest.raises(ValueError, match=msg):
        check_X_y(X, y, ensure_min_features=3)

    # Only the feature check is enabled whenever the number of dimensions is 2
    # even if allow_nd is enabled:
    with pytest.raises(ValueError, match=msg):
        check_X_y(X, y, ensure_min_features=3, allow_nd=True)

    # Simulate a case where a pipeline stage as trimmed all the features of a
    # 2D dataset.
    X = np.empty(0).reshape(10, 0)
    y = np.ones(10)
    msg = r"0 feature\(s\) \(shape=\(10, 0\)\) while a minimum of 1 is" " required."
    with pytest.raises(ValueError, match=msg):
        check_X_y(X, y)

    # nd-data is not checked for any minimum number of features by default:
    X = np.ones((10, 0, 28, 28))
    y = np.ones(10)
    X_checked, y_checked = check_X_y(X, y, allow_nd=True)
    assert_array_equal(X, X_checked)
    assert_array_equal(y, y_checked)


def test_check_array_complex_data_error():
    X = np.array([[1 + 2j, 3 + 4j, 5 + 7j], [2 + 3j, 4 + 5j, 6 + 7j]])
    with pytest.raises(ValueError, match="Complex data not supported"):
        check_array(X)

    # list of lists
    X = [[1 + 2j, 3 + 4j, 5 + 7j], [2 + 3j, 4 + 5j, 6 + 7j]]
    with pytest.raises(ValueError, match="Complex data not supported"):
        check_array(X)

    # tuple of tuples
    X = ((1 + 2j, 3 + 4j, 5 + 7j), (2 + 3j, 4 + 5j, 6 + 7j))
    with pytest.raises(ValueError, match="Complex data not supported"):
        check_array(X)

    # list of np arrays
    X = [np.array([1 + 2j, 3 + 4j, 5 + 7j]), np.array([2 + 3j, 4 + 5j, 6 + 7j])]
    with pytest.raises(ValueError, match="Complex data not supported"):
        check_array(X)

    # tuple of np arrays
    X = (np.array([1 + 2j, 3 + 4j, 5 + 7j]), np.array([2 + 3j, 4 + 5j, 6 + 7j]))
    with pytest.raises(ValueError, match="Complex data not supported"):
        check_array(X)

    # dataframe
    X = MockDataFrame(np.array([[1 + 2j, 3 + 4j, 5 + 7j], [2 + 3j, 4 + 5j, 6 + 7j]]))
    with pytest.raises(ValueError, match="Complex data not supported"):
        check_array(X)

    # sparse matrix
    X = sp.coo_matrix([[0, 1 + 2j], [0, 0]])
    with pytest.raises(ValueError, match="Complex data not supported"):
        check_array(X)

    # target variable does not always go through check_array but should
    # never accept complex data either.
    y = np.array([1 + 2j, 3 + 4j, 5 + 7j, 2 + 3j, 4 + 5j, 6 + 7j])
    with pytest.raises(ValueError, match="Complex data not supported"):
        _check_y(y)


def test_has_fit_parameter():
    assert not has_fit_parameter(KNeighborsClassifier, "sample_weight")
    assert has_fit_parameter(RandomForestRegressor, "sample_weight")
    assert has_fit_parameter(SVR, "sample_weight")
    assert has_fit_parameter(SVR(), "sample_weight")

    class TestClassWithDeprecatedFitMethod:
        @deprecated("Deprecated for the purpose of testing has_fit_parameter")
        def fit(self, X, y, sample_weight=None):
            pass

    assert has_fit_parameter(
        TestClassWithDeprecatedFitMethod, "sample_weight"
    ), "has_fit_parameter fails for class with deprecated fit method."


def test_check_symmetric():
    arr_sym = np.array([[0, 1], [1, 2]])
    arr_bad = np.ones(2)
    arr_asym = np.array([[0, 2], [0, 2]])

    test_arrays = {
        "dense": arr_asym,
        "dok": sp.dok_matrix(arr_asym),
        "csr": sp.csr_matrix(arr_asym),
        "csc": sp.csc_matrix(arr_asym),
        "coo": sp.coo_matrix(arr_asym),
        "lil": sp.lil_matrix(arr_asym),
        "bsr": sp.bsr_matrix(arr_asym),
    }

    # check error for bad inputs
    with pytest.raises(ValueError):
        check_symmetric(arr_bad)

    # check that asymmetric arrays are properly symmetrized
    for arr_format, arr in test_arrays.items():
        # Check for warnings and errors
        with pytest.warns(UserWarning):
            check_symmetric(arr)
        with pytest.raises(ValueError):
            check_symmetric(arr, raise_exception=True)

        output = check_symmetric(arr, raise_warning=False)
        if sp.issparse(output):
            assert output.format == arr_format
            assert_array_equal(output.toarray(), arr_sym)
        else:
            assert_array_equal(output, arr_sym)


def test_check_is_fitted_with_is_fitted():
    class Estimator(BaseEstimator):
        def fit(self, **kwargs):
            self._is_fitted = True
            return self

        def __sklearn_is_fitted__(self):
            return hasattr(self, "_is_fitted") and self._is_fitted

    with pytest.raises(NotFittedError):
        check_is_fitted(Estimator())
    check_is_fitted(Estimator().fit())


def test_check_is_fitted():
    # Check is TypeError raised when non estimator instance passed
    with pytest.raises(TypeError):
        check_is_fitted(ARDRegression)
    with pytest.raises(TypeError):
        check_is_fitted("SVR")

    ard = ARDRegression()
    svr = SVR()

    try:
        with pytest.raises(NotFittedError):
            check_is_fitted(ard)
        with pytest.raises(NotFittedError):
            check_is_fitted(svr)
    except ValueError:
        assert False, "check_is_fitted failed with ValueError"

    # NotFittedError is a subclass of both ValueError and AttributeError
    try:
        check_is_fitted(ard, msg="Random message %(name)s, %(name)s")
    except ValueError as e:
        assert str(e) == "Random message ARDRegression, ARDRegression"

    try:
        check_is_fitted(svr, msg="Another message %(name)s, %(name)s")
    except AttributeError as e:
        assert str(e) == "Another message SVR, SVR"

    ard.fit(*make_blobs())
    svr.fit(*make_blobs())

    assert check_is_fitted(ard) is None
    assert check_is_fitted(svr) is None


def test_check_is_fitted_attributes():
    class MyEstimator:
        def fit(self, X, y):
            return self

    msg = "not fitted"
    est = MyEstimator()

    with pytest.raises(NotFittedError, match=msg):
        check_is_fitted(est, attributes=["a_", "b_"])
    with pytest.raises(NotFittedError, match=msg):
        check_is_fitted(est, attributes=["a_", "b_"], all_or_any=all)
    with pytest.raises(NotFittedError, match=msg):
        check_is_fitted(est, attributes=["a_", "b_"], all_or_any=any)

    est.a_ = "a"
    with pytest.raises(NotFittedError, match=msg):
        check_is_fitted(est, attributes=["a_", "b_"])
    with pytest.raises(NotFittedError, match=msg):
        check_is_fitted(est, attributes=["a_", "b_"], all_or_any=all)
    check_is_fitted(est, attributes=["a_", "b_"], all_or_any=any)

    est.b_ = "b"
    check_is_fitted(est, attributes=["a_", "b_"])
    check_is_fitted(est, attributes=["a_", "b_"], all_or_any=all)
    check_is_fitted(est, attributes=["a_", "b_"], all_or_any=any)


@pytest.mark.parametrize(
    "wrap", [itemgetter(0), list, tuple], ids=["single", "list", "tuple"]
)
def test_check_is_fitted_with_attributes(wrap):
    ard = ARDRegression()
    with pytest.raises(NotFittedError, match="is not fitted yet"):
        check_is_fitted(ard, wrap(["coef_"]))

    ard.fit(*make_blobs())

    # Does not raise
    check_is_fitted(ard, wrap(["coef_"]))

    # Raises when using attribute that is not defined
    with pytest.raises(NotFittedError, match="is not fitted yet"):
        check_is_fitted(ard, wrap(["coef_bad_"]))


def test_check_consistent_length():
    check_consistent_length([1], [2], [3], [4], [5])
    check_consistent_length([[1, 2], [[1, 2]]], [1, 2], ["a", "b"])
    check_consistent_length([1], (2,), np.array([3]), sp.csr_matrix((1, 2)))
    with pytest.raises(ValueError, match="inconsistent numbers of samples"):
        check_consistent_length([1, 2], [1])
    with pytest.raises(TypeError, match=r"got <\w+ 'int'>"):
        check_consistent_length([1, 2], 1)
    with pytest.raises(TypeError, match=r"got <\w+ 'object'>"):
        check_consistent_length([1, 2], object())

    with pytest.raises(TypeError):
        check_consistent_length([1, 2], np.array(1))

    # Despite ensembles having __len__ they must raise TypeError
    with pytest.raises(TypeError, match="Expected sequence or array-like"):
        check_consistent_length([1, 2], RandomForestRegressor())
    # XXX: We should have a test with a string, but what is correct behaviour?


def test_check_dataframe_fit_attribute():
    # check pandas dataframe with 'fit' column does not raise error
    # https://github.com/scikit-learn/scikit-learn/issues/8415
    try:
        import pandas as pd

        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        X_df = pd.DataFrame(X, columns=["a", "b", "fit"])
        check_consistent_length(X_df)
    except ImportError:
        raise SkipTest("Pandas not found")


def test_suppress_validation():
    X = np.array([0, np.inf])
    with pytest.raises(ValueError):
        assert_all_finite(X)
    sklearn.set_config(assume_finite=True)
    assert_all_finite(X)
    sklearn.set_config(assume_finite=False)
    with pytest.raises(ValueError):
        assert_all_finite(X)


def test_check_array_series():
    # regression test that check_array works on pandas Series
    pd = importorskip("pandas")
    res = check_array(pd.Series([1, 2, 3]), ensure_2d=False)
    assert_array_equal(res, np.array([1, 2, 3]))

    # with categorical dtype (not a numpy dtype) (GH12699)
    s = pd.Series(["a", "b", "c"]).astype("category")
    res = check_array(s, dtype=None, ensure_2d=False)
    assert_array_equal(res, np.array(["a", "b", "c"], dtype=object))


def test_check_dataframe_mixed_float_dtypes():
    # pandas dataframe will coerce a boolean into a object, this is a mismatch
    # with np.result_type which will return a float
    # check_array needs to explicitly check for bool dtype in a dataframe for
    # this situation
    # https://github.com/scikit-learn/scikit-learn/issues/15787

    pd = importorskip("pandas")
    df = pd.DataFrame(
        {"int": [1, 2, 3], "float": [0, 0.1, 2.1], "bool": [True, False, True]},
        columns=["int", "float", "bool"],
    )

    array = check_array(df, dtype=(np.float64, np.float32, np.float16))
    expected_array = np.array(
        [[1.0, 0.0, 1.0], [2.0, 0.1, 0.0], [3.0, 2.1, 1.0]], dtype=float
    )
    assert_allclose_dense_sparse(array, expected_array)


class DummyMemory:
    def cache(self, func):
        return func


class WrongDummyMemory:
    pass


@pytest.mark.filterwarnings("ignore:The 'cachedir' attribute")
def test_check_memory():
    memory = check_memory("cache_directory")
    assert memory.cachedir == os.path.join("cache_directory", "joblib")
    memory = check_memory(None)
    assert memory.cachedir is None
    dummy = DummyMemory()
    memory = check_memory(dummy)
    assert memory is dummy

    msg = (
        "'memory' should be None, a string or have the same interface as"
        " joblib.Memory. Got memory='1' instead."
    )
    with pytest.raises(ValueError, match=msg):
        check_memory(1)
    dummy = WrongDummyMemory()
    msg = (
        "'memory' should be None, a string or have the same interface as"
        " joblib.Memory. Got memory='{}' instead.".format(dummy)
    )
    with pytest.raises(ValueError, match=msg):
        check_memory(dummy)


@pytest.mark.parametrize("copy", [True, False])
def test_check_array_memmap(copy):
    X = np.ones((4, 4))
    with TempMemmap(X, mmap_mode="r") as X_memmap:
        X_checked = check_array(X_memmap, copy=copy)
        assert np.may_share_memory(X_memmap, X_checked) == (not copy)
        assert X_checked.flags["WRITEABLE"] == copy


@pytest.mark.parametrize(
    "retype",
    [
        np.asarray,
        sp.csr_matrix,
        sp.csc_matrix,
        sp.coo_matrix,
        sp.lil_matrix,
        sp.bsr_matrix,
        sp.dok_matrix,
        sp.dia_matrix,
    ],
)
def test_check_non_negative(retype):
    A = np.array([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    X = retype(A)
    check_non_negative(X, "")
    X = retype([[0, 0], [0, 0]])
    check_non_negative(X, "")

    A[0, 0] = -1
    X = retype(A)
    with pytest.raises(ValueError, match="Negative "):
        check_non_negative(X, "")


def test_check_X_y_informative_error():
    X = np.ones((2, 2))
    y = None
    with pytest.raises(ValueError, match="y cannot be None"):
        check_X_y(X, y)


def test_retrieve_samples_from_non_standard_shape():
    class TestNonNumericShape:
        def __init__(self):
            self.shape = ("not numeric",)

        def __len__(self):
            return len([1, 2, 3])

    X = TestNonNumericShape()
    assert _num_samples(X) == len(X)

    # check that it gives a good error if there's no __len__
    class TestNoLenWeirdShape:
        def __init__(self):
            self.shape = ("not numeric",)

    with pytest.raises(TypeError, match="Expected sequence or array-like"):
        _num_samples(TestNoLenWeirdShape())


@pytest.mark.parametrize("x", [2, 3, 2.5, 5])
def test_check_scalar_valid(x):
    """Test that check_scalar returns no error/warning if valid inputs are
    provided"""
    with pytest.warns(None) as record:
        scalar = check_scalar(
            x,
            "test_name",
            target_type=numbers.Real,
            min_val=2,
            max_val=5,
            include_boundaries="both",
        )
    assert len(record) == 0
    assert scalar == x


@pytest.mark.parametrize(
    "x, target_name, target_type, min_val, max_val, include_boundaries, err_msg",
    [
        (
            1,
            "test_name1",
            float,
            2,
            4,
            "neither",
            TypeError(
                "test_name1 must be an instance of <class 'float'>, not <class 'int'>."
            ),
        ),
        (
            1,
            "test_name2",
            int,
            2,
            4,
            "neither",
            ValueError("test_name2 == 1, must be > 2."),
        ),
        (
            5,
            "test_name3",
            int,
            2,
            4,
            "neither",
            ValueError("test_name3 == 5, must be < 4."),
        ),
        (
            2,
            "test_name4",
            int,
            2,
            4,
            "right",
            ValueError("test_name4 == 2, must be > 2."),
        ),
        (
            4,
            "test_name5",
            int,
            2,
            4,
            "left",
            ValueError("test_name5 == 4, must be < 4."),
        ),
        (
            4,
            "test_name6",
            int,
            2,
            4,
            "bad parameter value",
            ValueError(
                "Unknown value for `include_boundaries`: 'bad parameter value'. "
                "Possible values are: ('left', 'right', 'both', 'neither')."
            ),
        ),
    ],
)
def test_check_scalar_invalid(
    x, target_name, target_type, min_val, max_val, include_boundaries, err_msg
):
    """Test that check_scalar returns the right error if a wrong input is
    given"""
    with pytest.raises(Exception) as raised_error:
        check_scalar(
            x,
            target_name,
            target_type=target_type,
            min_val=min_val,
            max_val=max_val,
            include_boundaries=include_boundaries,
        )
    assert str(raised_error.value) == str(err_msg)
    assert type(raised_error.value) == type(err_msg)


_psd_cases_valid = {
    "nominal": ((1, 2), np.array([1, 2]), None, ""),
    "nominal_np_array": (np.array([1, 2]), np.array([1, 2]), None, ""),
    "insignificant_imag": (
        (5, 5e-5j),
        np.array([5, 0]),
        PositiveSpectrumWarning,
        "There are imaginary parts in eigenvalues \\(1e\\-05 of the maximum real part",
    ),
    "insignificant neg": ((5, -5e-5), np.array([5, 0]), PositiveSpectrumWarning, ""),
    "insignificant neg float32": (
        np.array([1, -1e-6], dtype=np.float32),
        np.array([1, 0], dtype=np.float32),
        PositiveSpectrumWarning,
        "There are negative eigenvalues \\(1e\\-06 of the maximum positive",
    ),
    "insignificant neg float64": (
        np.array([1, -1e-10], dtype=np.float64),
        np.array([1, 0], dtype=np.float64),
        PositiveSpectrumWarning,
        "There are negative eigenvalues \\(1e\\-10 of the maximum positive",
    ),
    "insignificant pos": (
        (5, 4e-12),
        np.array([5, 0]),
        PositiveSpectrumWarning,
        "the largest eigenvalue is more than 1e\\+12 times the smallest",
    ),
}


@pytest.mark.parametrize(
    "lambdas, expected_lambdas, w_type, w_msg",
    list(_psd_cases_valid.values()),
    ids=list(_psd_cases_valid.keys()),
)
@pytest.mark.parametrize("enable_warnings", [True, False])
def test_check_psd_eigenvalues_valid(
    lambdas, expected_lambdas, w_type, w_msg, enable_warnings
):
    # Test that ``_check_psd_eigenvalues`` returns the right output for valid
    # input, possibly raising the right warning

    if not enable_warnings:
        w_type = None
        w_msg = ""

    with pytest.warns(w_type, match=w_msg) as w:
        assert_array_equal(
            _check_psd_eigenvalues(lambdas, enable_warnings=enable_warnings),
            expected_lambdas,
        )
    if w_type is None:
        assert not w


_psd_cases_invalid = {
    "significant_imag": (
        (5, 5j),
        ValueError,
        "There are significant imaginary parts in eigenv",
    ),
    "all negative": (
        (-5, -1),
        ValueError,
        "All eigenvalues are negative \\(maximum is -1",
    ),
    "significant neg": (
        (5, -1),
        ValueError,
        "There are significant negative eigenvalues",
    ),
    "significant neg float32": (
        np.array([3e-4, -2e-6], dtype=np.float32),
        ValueError,
        "There are significant negative eigenvalues",
    ),
    "significant neg float64": (
        np.array([1e-5, -2e-10], dtype=np.float64),
        ValueError,
        "There are significant negative eigenvalues",
    ),
}


@pytest.mark.parametrize(
    "lambdas, err_type, err_msg",
    list(_psd_cases_invalid.values()),
    ids=list(_psd_cases_invalid.keys()),
)
def test_check_psd_eigenvalues_invalid(lambdas, err_type, err_msg):
    # Test that ``_check_psd_eigenvalues`` raises the right error for invalid
    # input

    with pytest.raises(err_type, match=err_msg):
        _check_psd_eigenvalues(lambdas)


def test_check_sample_weight():
    # check array order
    sample_weight = np.ones(10)[::2]
    assert not sample_weight.flags["C_CONTIGUOUS"]
    sample_weight = _check_sample_weight(sample_weight, X=np.ones((5, 1)))
    assert sample_weight.flags["C_CONTIGUOUS"]

    # check None input
    sample_weight = _check_sample_weight(None, X=np.ones((5, 2)))
    assert_allclose(sample_weight, np.ones(5))

    # check numbers input
    sample_weight = _check_sample_weight(2.0, X=np.ones((5, 2)))
    assert_allclose(sample_weight, 2 * np.ones(5))

    # check wrong number of dimensions
    with pytest.raises(ValueError, match="Sample weights must be 1D array or scalar"):
        _check_sample_weight(np.ones((2, 4)), X=np.ones((2, 2)))

    # check incorrect n_samples
    msg = r"sample_weight.shape == \(4,\), expected \(2,\)!"
    with pytest.raises(ValueError, match=msg):
        _check_sample_weight(np.ones(4), X=np.ones((2, 2)))

    # float32 dtype is preserved
    X = np.ones((5, 2))
    sample_weight = np.ones(5, dtype=np.float32)
    sample_weight = _check_sample_weight(sample_weight, X)
    assert sample_weight.dtype == np.float32

    # int dtype will be converted to float64 instead
    X = np.ones((5, 2), dtype=int)
    sample_weight = _check_sample_weight(None, X, dtype=X.dtype)
    assert sample_weight.dtype == np.float64


@pytest.mark.parametrize("toarray", [np.array, sp.csr_matrix, sp.csc_matrix])
def test_allclose_dense_sparse_equals(toarray):
    base = np.arange(9).reshape(3, 3)
    x, y = toarray(base), toarray(base)
    assert _allclose_dense_sparse(x, y)


@pytest.mark.parametrize("toarray", [np.array, sp.csr_matrix, sp.csc_matrix])
def test_allclose_dense_sparse_not_equals(toarray):
    base = np.arange(9).reshape(3, 3)
    x, y = toarray(base), toarray(base + 1)
    assert not _allclose_dense_sparse(x, y)


@pytest.mark.parametrize("toarray", [sp.csr_matrix, sp.csc_matrix])
def test_allclose_dense_sparse_raise(toarray):
    x = np.arange(9).reshape(3, 3)
    y = toarray(x + 1)

    msg = "Can only compare two sparse matrices, not a sparse matrix and an array"
    with pytest.raises(ValueError, match=msg):
        _allclose_dense_sparse(x, y)


def test_deprecate_positional_args_warns_for_function():
    @_deprecate_positional_args
    def f1(a, b, *, c=1, d=1):
        pass

    with pytest.warns(FutureWarning, match=r"Pass c=3 as keyword args"):
        f1(1, 2, 3)

    with pytest.warns(FutureWarning, match=r"Pass c=3, d=4 as keyword args"):
        f1(1, 2, 3, 4)

    @_deprecate_positional_args
    def f2(a=1, *, b=1, c=1, d=1):
        pass

    with pytest.warns(FutureWarning, match=r"Pass b=2 as keyword args"):
        f2(1, 2)

    # The * is place before a keyword only argument without a default value
    @_deprecate_positional_args
    def f3(a, *, b, c=1, d=1):
        pass

    with pytest.warns(FutureWarning, match=r"Pass b=2 as keyword args"):
        f3(1, 2)


def test_deprecate_positional_args_warns_for_function_version():
    @_deprecate_positional_args(version="1.1")
    def f1(a, *, b):
        pass

    with pytest.warns(
        FutureWarning, match=r"From version 1.1 passing these as positional"
    ):
        f1(1, 2)


def test_deprecate_positional_args_warns_for_class():
    class A1:
        @_deprecate_positional_args
        def __init__(self, a, b, *, c=1, d=1):
            pass

    with pytest.warns(FutureWarning, match=r"Pass c=3 as keyword args"):
        A1(1, 2, 3)

    with pytest.warns(FutureWarning, match=r"Pass c=3, d=4 as keyword args"):
        A1(1, 2, 3, 4)

    class A2:
        @_deprecate_positional_args
        def __init__(self, a=1, b=1, *, c=1, d=1):
            pass

    with pytest.warns(FutureWarning, match=r"Pass c=3 as keyword args"):
        A2(1, 2, 3)

    with pytest.warns(FutureWarning, match=r"Pass c=3, d=4 as keyword args"):
        A2(1, 2, 3, 4)


@pytest.mark.parametrize("indices", [None, [1, 3]])
def test_check_fit_params(indices):
    X = np.random.randn(4, 2)
    fit_params = {
        "list": [1, 2, 3, 4],
        "array": np.array([1, 2, 3, 4]),
        "sparse-col": sp.csc_matrix([1, 2, 3, 4]).T,
        "sparse-row": sp.csc_matrix([1, 2, 3, 4]),
        "scalar-int": 1,
        "scalar-str": "xxx",
        "None": None,
    }
    result = _check_fit_params(X, fit_params, indices)
    indices_ = indices if indices is not None else list(range(X.shape[0]))

    for key in ["sparse-row", "scalar-int", "scalar-str", "None"]:
        assert result[key] is fit_params[key]

    assert result["list"] == _safe_indexing(fit_params["list"], indices_)
    assert_array_equal(result["array"], _safe_indexing(fit_params["array"], indices_))
    assert_allclose_dense_sparse(
        result["sparse-col"], _safe_indexing(fit_params["sparse-col"], indices_)
    )


@pytest.mark.parametrize("sp_format", [True, "csr", "csc", "coo", "bsr"])
def test_check_sparse_pandas_sp_format(sp_format):
    # check_array converts pandas dataframe with only sparse arrays into
    # sparse matrix
    pd = pytest.importorskip("pandas", minversion="0.25.0")
    sp_mat = _sparse_random_matrix(10, 3)

    sdf = pd.DataFrame.sparse.from_spmatrix(sp_mat)
    result = check_array(sdf, accept_sparse=sp_format)

    if sp_format is True:
        # by default pandas converts to coo when accept_sparse is True
        sp_format = "coo"

    assert sp.issparse(result)
    assert result.format == sp_format
    assert_allclose_dense_sparse(sp_mat, result)


@pytest.mark.parametrize(
    "ntype1, ntype2",
    [
        ("longdouble", "float16"),
        ("float16", "float32"),
        ("float32", "double"),
        ("int16", "int32"),
        ("int32", "long"),
        ("byte", "uint16"),
        ("ushort", "uint32"),
        ("uint32", "uint64"),
        ("uint8", "int8"),
    ],
)
def test_check_pandas_sparse_invalid(ntype1, ntype2):
    """check that we raise an error with dataframe having
    sparse extension arrays with unsupported mixed dtype
    and pandas version below 1.1. pandas versions 1.1 and
    above fixed this issue so no error will be raised."""
    pd = pytest.importorskip("pandas", minversion="0.25.0")
    df = pd.DataFrame(
        {
            "col1": pd.arrays.SparseArray([0, 1, 0], dtype=ntype1, fill_value=0),
            "col2": pd.arrays.SparseArray([1, 0, 1], dtype=ntype2, fill_value=0),
        }
    )

    if parse_version(pd.__version__) < parse_version("1.1"):
        err_msg = "Pandas DataFrame with mixed sparse extension arrays"
        with pytest.raises(ValueError, match=err_msg):
            check_array(df, accept_sparse=["csr", "csc"])
    else:
        # pandas fixed this issue at 1.1 so from here on,
        # no error will be raised.
        check_array(df, accept_sparse=["csr", "csc"])


@pytest.mark.parametrize(
    "ntype1, ntype2, expected_subtype",
    [
        ("longfloat", "longdouble", np.floating),
        ("float16", "half", np.floating),
        ("single", "float32", np.floating),
        ("double", "float64", np.floating),
        ("int8", "byte", np.integer),
        ("short", "int16", np.integer),
        ("intc", "int32", np.integer),
        ("int0", "long", np.integer),
        ("int", "long", np.integer),
        ("int64", "longlong", np.integer),
        ("int_", "intp", np.integer),
        ("ubyte", "uint8", np.unsignedinteger),
        ("uint16", "ushort", np.unsignedinteger),
        ("uintc", "uint32", np.unsignedinteger),
        ("uint", "uint64", np.unsignedinteger),
        ("uintp", "ulonglong", np.unsignedinteger),
    ],
)
def test_check_pandas_sparse_valid(ntype1, ntype2, expected_subtype):
    # check that we support the conversion of sparse dataframe with mixed
    # type which can be converted safely.
    pd = pytest.importorskip("pandas", minversion="0.25.0")
    df = pd.DataFrame(
        {
            "col1": pd.arrays.SparseArray([0, 1, 0], dtype=ntype1, fill_value=0),
            "col2": pd.arrays.SparseArray([1, 0, 1], dtype=ntype2, fill_value=0),
        }
    )
    arr = check_array(df, accept_sparse=["csr", "csc"])
    assert np.issubdtype(arr.dtype, expected_subtype)


@pytest.mark.parametrize(
    "constructor_name",
    ["list", "tuple", "array", "dataframe", "sparse_csr", "sparse_csc"],
)
def test_num_features(constructor_name):
    """Check _num_features for array-likes."""
    X = [[1, 2, 3], [4, 5, 6]]
    X = _convert_container(X, constructor_name)
    assert _num_features(X) == 3


@pytest.mark.parametrize(
    "X",
    [
        [1, 2, 3],
        ["a", "b", "c"],
        [False, True, False],
        [1.0, 3.4, 4.0],
        [{"a": 1}, {"b": 2}, {"c": 3}],
    ],
    ids=["int", "str", "bool", "float", "dict"],
)
@pytest.mark.parametrize("constructor_name", ["list", "tuple", "array", "series"])
def test_num_features_errors_1d_containers(X, constructor_name):
    X = _convert_container(X, constructor_name)
    if constructor_name == "array":
        expected_type_name = "numpy.ndarray"
    elif constructor_name == "series":
        expected_type_name = "pandas.core.series.Series"
    else:
        expected_type_name = constructor_name
    message = (
        f"Unable to find the number of features from X of type {expected_type_name}"
    )
    if hasattr(X, "shape"):
        message += " with shape (3,)"
    elif isinstance(X[0], str):
        message += " where the samples are of type str"
    elif isinstance(X[0], dict):
        message += " where the samples are of type dict"
    with pytest.raises(TypeError, match=re.escape(message)):
        _num_features(X)


@pytest.mark.parametrize("X", [1, "b", False, 3.0], ids=["int", "str", "bool", "float"])
def test_num_features_errors_scalars(X):
    msg = f"Unable to find the number of features from X of type {type(X).__qualname__}"
    with pytest.raises(TypeError, match=msg):
        _num_features(X)


# TODO: Remove in 1.2
@pytest.mark.filterwarnings("ignore:the matrix subclass:PendingDeprecationWarning")
def test_check_array_deprecated_matrix():
    """Test that matrix support is deprecated in 1.0."""

    X = np.matrix(np.arange(5))
    msg = (
        "np.matrix usage is deprecated in 1.0 and will raise a TypeError "
        "in 1.2. Please convert to a numpy array with np.asarray."
    )
    with pytest.warns(FutureWarning, match=msg):
        check_array(X)


@pytest.mark.parametrize(
    "names",
    [list(range(2)), range(2), None],
    ids=["list-int", "range", "default"],
)
def test_get_feature_names_pandas_with_ints_no_warning(names):
    """Get feature names with pandas dataframes with ints without warning"""
    pd = pytest.importorskip("pandas")
    X = pd.DataFrame([[1, 2], [4, 5], [5, 6]], columns=names)

    with pytest.warns(None) as record:
        names = _get_feature_names(X)
    assert not record
    assert names is None


def test_get_feature_names_pandas():
    """Get feature names with pandas dataframes."""
    pd = pytest.importorskip("pandas")
    columns = [f"col_{i}" for i in range(3)]
    X = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=columns)
    feature_names = _get_feature_names(X)

    assert_array_equal(feature_names, columns)


def test_get_feature_names_numpy():
    """Get feature names return None for numpy arrays."""
    X = np.array([[1, 2, 3], [4, 5, 6]])
    names = _get_feature_names(X)
    assert names is None


# TODO: Convert to a error in 1.2
@pytest.mark.parametrize(
    "names, dtypes",
    [
        ([["a", "b"], ["c", "d"]], "['tuple']"),
        (["a", 1], "['int', 'str']"),
    ],
    ids=["multi-index", "mixed"],
)
def test_get_feature_names_invalid_dtypes_warns(names, dtypes):
    """Get feature names warns when the feature names have mixed dtypes"""
    pd = pytest.importorskip("pandas")
    X = pd.DataFrame([[1, 2], [4, 5], [5, 6]], columns=names)

    msg = re.escape(
        "Feature names only support names that are all strings. "
        f"Got feature names with dtypes: {dtypes}. An error will be raised"
    )
    with pytest.warns(FutureWarning, match=msg):
        names = _get_feature_names(X)
    assert names is None


class PassthroughTransformer(BaseEstimator):
    def fit(self, X, y=None):
        self._validate_data(X, reset=True)
        return self

    def transform(self, X):
        return X

    def get_feature_names_out(self, input_features=None):
        return _check_feature_names_in(self, input_features)


def test_check_feature_names_in():
    """Check behavior of check_feature_names_in for arrays."""
    X = np.array([[0.0, 1.0, 2.0]])
    est = PassthroughTransformer().fit(X)

    names = est.get_feature_names_out()
    assert_array_equal(names, ["x0", "x1", "x2"])

    incorrect_len_names = ["x10", "x1"]
    with pytest.raises(ValueError, match="input_features should have length equal to"):
        est.get_feature_names_out(incorrect_len_names)

    # remove n_feature_in_
    del est.n_features_in_
    with pytest.raises(ValueError, match="Unable to generate feature names"):
        est.get_feature_names_out()


def test_check_feature_names_in_pandas():
    """Check behavior of check_feature_names_in for pandas dataframes."""
    pd = pytest.importorskip("pandas")
    names = ["a", "b", "c"]
    df = pd.DataFrame([[0.0, 1.0, 2.0]], columns=names)
    est = PassthroughTransformer().fit(df)

    names = est.get_feature_names_out()
    assert_array_equal(names, ["a", "b", "c"])

    with pytest.raises(ValueError, match="input_features is not equal to"):
        est.get_feature_names_out(["x1", "x2", "x3"])
