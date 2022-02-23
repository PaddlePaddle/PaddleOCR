from copy import copy
from itertools import chain
import warnings
import string
import timeit

import pytest
import numpy as np
import scipy.sparse as sp

from sklearn.utils._testing import (
    assert_array_equal,
    assert_allclose_dense_sparse,
    assert_no_warnings,
    _convert_container,
)
from sklearn.utils import check_random_state
from sklearn.utils import _determine_key_type
from sklearn.utils import deprecated
from sklearn.utils import gen_batches
from sklearn.utils import _get_column_indices
from sklearn.utils import resample
from sklearn.utils import safe_mask
from sklearn.utils import column_or_1d
from sklearn.utils import _safe_indexing
from sklearn.utils import shuffle
from sklearn.utils import gen_even_slices
from sklearn.utils import _message_with_time, _print_elapsed_time
from sklearn.utils import get_chunk_n_rows
from sklearn.utils import is_scalar_nan
from sklearn.utils import _to_object_array
from sklearn.utils import _approximate_mode
from sklearn.utils.fixes import parse_version
from sklearn.utils._mocking import MockDataFrame
from sklearn.utils._testing import SkipTest
from sklearn import config_context

# toy array
X_toy = np.arange(9).reshape((3, 3))


def test_make_rng():
    # Check the check_random_state utility function behavior
    assert check_random_state(None) is np.random.mtrand._rand
    assert check_random_state(np.random) is np.random.mtrand._rand

    rng_42 = np.random.RandomState(42)
    assert check_random_state(42).randint(100) == rng_42.randint(100)

    rng_42 = np.random.RandomState(42)
    assert check_random_state(rng_42) is rng_42

    rng_42 = np.random.RandomState(42)
    assert check_random_state(43).randint(100) != rng_42.randint(100)

    with pytest.raises(ValueError):
        check_random_state("some invalid seed")


def test_gen_batches():
    # Make sure gen_batches errors on invalid batch_size

    assert_array_equal(list(gen_batches(4, 2)), [slice(0, 2, None), slice(2, 4, None)])
    msg_zero = "gen_batches got batch_size=0, must be positive"
    with pytest.raises(ValueError, match=msg_zero):
        next(gen_batches(4, 0))

    msg_float = "gen_batches got batch_size=0.5, must be an integer"
    with pytest.raises(TypeError, match=msg_float):
        next(gen_batches(4, 0.5))


def test_deprecated():
    # Test whether the deprecated decorator issues appropriate warnings
    # Copied almost verbatim from https://docs.python.org/library/warnings.html

    # First a function...
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        @deprecated()
        def ham():
            return "spam"

        spam = ham()

        assert spam == "spam"  # function must remain usable

        assert len(w) == 1
        assert issubclass(w[0].category, FutureWarning)
        assert "deprecated" in str(w[0].message).lower()

    # ... then a class.
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        @deprecated("don't use this")
        class Ham:
            SPAM = 1

        ham = Ham()

        assert hasattr(ham, "SPAM")

        assert len(w) == 1
        assert issubclass(w[0].category, FutureWarning)
        assert "deprecated" in str(w[0].message).lower()


def test_resample():
    # Border case not worth mentioning in doctests
    assert resample() is None

    # Check that invalid arguments yield ValueError
    with pytest.raises(ValueError):
        resample([0], [0, 1])
    with pytest.raises(ValueError):
        resample([0, 1], [0, 1], replace=False, n_samples=3)

    # Issue:6581, n_samples can be more when replace is True (default).
    assert len(resample([1, 2], n_samples=5)) == 5


def test_resample_stratified():
    # Make sure resample can stratify
    rng = np.random.RandomState(0)
    n_samples = 100
    p = 0.9
    X = rng.normal(size=(n_samples, 1))
    y = rng.binomial(1, p, size=n_samples)

    _, y_not_stratified = resample(X, y, n_samples=10, random_state=0, stratify=None)
    assert np.all(y_not_stratified == 1)

    _, y_stratified = resample(X, y, n_samples=10, random_state=0, stratify=y)
    assert not np.all(y_stratified == 1)
    assert np.sum(y_stratified) == 9  # all 1s, one 0


def test_resample_stratified_replace():
    # Make sure stratified resampling supports the replace parameter
    rng = np.random.RandomState(0)
    n_samples = 100
    X = rng.normal(size=(n_samples, 1))
    y = rng.randint(0, 2, size=n_samples)

    X_replace, _ = resample(
        X, y, replace=True, n_samples=50, random_state=rng, stratify=y
    )
    X_no_replace, _ = resample(
        X, y, replace=False, n_samples=50, random_state=rng, stratify=y
    )
    assert np.unique(X_replace).shape[0] < 50
    assert np.unique(X_no_replace).shape[0] == 50

    # make sure n_samples can be greater than X.shape[0] if we sample with
    # replacement
    X_replace, _ = resample(
        X, y, replace=True, n_samples=1000, random_state=rng, stratify=y
    )
    assert X_replace.shape[0] == 1000
    assert np.unique(X_replace).shape[0] == 100


def test_resample_stratify_2dy():
    # Make sure y can be 2d when stratifying
    rng = np.random.RandomState(0)
    n_samples = 100
    X = rng.normal(size=(n_samples, 1))
    y = rng.randint(0, 2, size=(n_samples, 2))
    X, y = resample(X, y, n_samples=50, random_state=rng, stratify=y)
    assert y.ndim == 2


def test_resample_stratify_sparse_error():
    # resample must be ndarray
    rng = np.random.RandomState(0)
    n_samples = 100
    X = rng.normal(size=(n_samples, 2))
    y = rng.randint(0, 2, size=n_samples)
    stratify = sp.csr_matrix(y)
    with pytest.raises(TypeError, match="A sparse matrix was passed"):
        X, y = resample(X, y, n_samples=50, random_state=rng, stratify=stratify)


def test_safe_mask():
    random_state = check_random_state(0)
    X = random_state.rand(5, 4)
    X_csr = sp.csr_matrix(X)
    mask = [False, False, True, True, True]

    mask = safe_mask(X, mask)
    assert X[mask].shape[0] == 3

    mask = safe_mask(X_csr, mask)
    assert X_csr[mask].shape[0] == 3


def test_column_or_1d():
    EXAMPLES = [
        ("binary", ["spam", "egg", "spam"]),
        ("binary", [0, 1, 0, 1]),
        ("continuous", np.arange(10) / 20.0),
        ("multiclass", [1, 2, 3]),
        ("multiclass", [0, 1, 2, 2, 0]),
        ("multiclass", [[1], [2], [3]]),
        ("multilabel-indicator", [[0, 1, 0], [0, 0, 1]]),
        ("multiclass-multioutput", [[1, 2, 3]]),
        ("multiclass-multioutput", [[1, 1], [2, 2], [3, 1]]),
        ("multiclass-multioutput", [[5, 1], [4, 2], [3, 1]]),
        ("multiclass-multioutput", [[1, 2, 3]]),
        ("continuous-multioutput", np.arange(30).reshape((-1, 3))),
    ]

    for y_type, y in EXAMPLES:
        if y_type in ["binary", "multiclass", "continuous"]:
            assert_array_equal(column_or_1d(y), np.ravel(y))
        else:
            with pytest.raises(ValueError):
                column_or_1d(y)


@pytest.mark.parametrize(
    "key, dtype",
    [
        (0, "int"),
        ("0", "str"),
        (True, "bool"),
        (np.bool_(True), "bool"),
        ([0, 1, 2], "int"),
        (["0", "1", "2"], "str"),
        ((0, 1, 2), "int"),
        (("0", "1", "2"), "str"),
        (slice(None, None), None),
        (slice(0, 2), "int"),
        (np.array([0, 1, 2], dtype=np.int32), "int"),
        (np.array([0, 1, 2], dtype=np.int64), "int"),
        (np.array([0, 1, 2], dtype=np.uint8), "int"),
        ([True, False], "bool"),
        ((True, False), "bool"),
        (np.array([True, False]), "bool"),
        ("col_0", "str"),
        (["col_0", "col_1", "col_2"], "str"),
        (("col_0", "col_1", "col_2"), "str"),
        (slice("begin", "end"), "str"),
        (np.array(["col_0", "col_1", "col_2"]), "str"),
        (np.array(["col_0", "col_1", "col_2"], dtype=object), "str"),
    ],
)
def test_determine_key_type(key, dtype):
    assert _determine_key_type(key) == dtype


def test_determine_key_type_error():
    with pytest.raises(ValueError, match="No valid specification of the"):
        _determine_key_type(1.0)


def test_determine_key_type_slice_error():
    with pytest.raises(TypeError, match="Only array-like or scalar are"):
        _determine_key_type(slice(0, 2, 1), accept_slice=False)


@pytest.mark.parametrize("array_type", ["list", "array", "sparse", "dataframe"])
@pytest.mark.parametrize("indices_type", ["list", "tuple", "array", "series", "slice"])
def test_safe_indexing_2d_container_axis_0(array_type, indices_type):
    indices = [1, 2]
    if indices_type == "slice" and isinstance(indices[1], int):
        indices[1] += 1
    array = _convert_container([[1, 2, 3], [4, 5, 6], [7, 8, 9]], array_type)
    indices = _convert_container(indices, indices_type)
    subset = _safe_indexing(array, indices, axis=0)
    assert_allclose_dense_sparse(
        subset, _convert_container([[4, 5, 6], [7, 8, 9]], array_type)
    )


@pytest.mark.parametrize("array_type", ["list", "array", "series"])
@pytest.mark.parametrize("indices_type", ["list", "tuple", "array", "series", "slice"])
def test_safe_indexing_1d_container(array_type, indices_type):
    indices = [1, 2]
    if indices_type == "slice" and isinstance(indices[1], int):
        indices[1] += 1
    array = _convert_container([1, 2, 3, 4, 5, 6, 7, 8, 9], array_type)
    indices = _convert_container(indices, indices_type)
    subset = _safe_indexing(array, indices, axis=0)
    assert_allclose_dense_sparse(subset, _convert_container([2, 3], array_type))


@pytest.mark.parametrize("array_type", ["array", "sparse", "dataframe"])
@pytest.mark.parametrize("indices_type", ["list", "tuple", "array", "series", "slice"])
@pytest.mark.parametrize("indices", [[1, 2], ["col_1", "col_2"]])
def test_safe_indexing_2d_container_axis_1(array_type, indices_type, indices):
    # validation of the indices
    # we make a copy because indices is mutable and shared between tests
    indices_converted = copy(indices)
    if indices_type == "slice" and isinstance(indices[1], int):
        indices_converted[1] += 1

    columns_name = ["col_0", "col_1", "col_2"]
    array = _convert_container(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]], array_type, columns_name
    )
    indices_converted = _convert_container(indices_converted, indices_type)

    if isinstance(indices[0], str) and array_type != "dataframe":
        err_msg = (
            "Specifying the columns using strings is only supported "
            "for pandas DataFrames"
        )
        with pytest.raises(ValueError, match=err_msg):
            _safe_indexing(array, indices_converted, axis=1)
    else:
        subset = _safe_indexing(array, indices_converted, axis=1)
        assert_allclose_dense_sparse(
            subset, _convert_container([[2, 3], [5, 6], [8, 9]], array_type)
        )


@pytest.mark.parametrize("array_read_only", [True, False])
@pytest.mark.parametrize("indices_read_only", [True, False])
@pytest.mark.parametrize("array_type", ["array", "sparse", "dataframe"])
@pytest.mark.parametrize("indices_type", ["array", "series"])
@pytest.mark.parametrize(
    "axis, expected_array", [(0, [[4, 5, 6], [7, 8, 9]]), (1, [[2, 3], [5, 6], [8, 9]])]
)
def test_safe_indexing_2d_read_only_axis_1(
    array_read_only, indices_read_only, array_type, indices_type, axis, expected_array
):
    array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    if array_read_only:
        array.setflags(write=False)
    array = _convert_container(array, array_type)
    indices = np.array([1, 2])
    if indices_read_only:
        indices.setflags(write=False)
    indices = _convert_container(indices, indices_type)
    subset = _safe_indexing(array, indices, axis=axis)
    assert_allclose_dense_sparse(subset, _convert_container(expected_array, array_type))


@pytest.mark.parametrize("array_type", ["list", "array", "series"])
@pytest.mark.parametrize("indices_type", ["list", "tuple", "array", "series"])
def test_safe_indexing_1d_container_mask(array_type, indices_type):
    indices = [False] + [True] * 2 + [False] * 6
    array = _convert_container([1, 2, 3, 4, 5, 6, 7, 8, 9], array_type)
    indices = _convert_container(indices, indices_type)
    subset = _safe_indexing(array, indices, axis=0)
    assert_allclose_dense_sparse(subset, _convert_container([2, 3], array_type))


@pytest.mark.parametrize("array_type", ["array", "sparse", "dataframe"])
@pytest.mark.parametrize("indices_type", ["list", "tuple", "array", "series"])
@pytest.mark.parametrize(
    "axis, expected_subset",
    [(0, [[4, 5, 6], [7, 8, 9]]), (1, [[2, 3], [5, 6], [8, 9]])],
)
def test_safe_indexing_2d_mask(array_type, indices_type, axis, expected_subset):
    columns_name = ["col_0", "col_1", "col_2"]
    array = _convert_container(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]], array_type, columns_name
    )
    indices = [False, True, True]
    indices = _convert_container(indices, indices_type)

    subset = _safe_indexing(array, indices, axis=axis)
    assert_allclose_dense_sparse(
        subset, _convert_container(expected_subset, array_type)
    )


@pytest.mark.parametrize(
    "array_type, expected_output_type",
    [
        ("list", "list"),
        ("array", "array"),
        ("sparse", "sparse"),
        ("dataframe", "series"),
    ],
)
def test_safe_indexing_2d_scalar_axis_0(array_type, expected_output_type):
    array = _convert_container([[1, 2, 3], [4, 5, 6], [7, 8, 9]], array_type)
    indices = 2
    subset = _safe_indexing(array, indices, axis=0)
    expected_array = _convert_container([7, 8, 9], expected_output_type)
    assert_allclose_dense_sparse(subset, expected_array)


@pytest.mark.parametrize("array_type", ["list", "array", "series"])
def test_safe_indexing_1d_scalar(array_type):
    array = _convert_container([1, 2, 3, 4, 5, 6, 7, 8, 9], array_type)
    indices = 2
    subset = _safe_indexing(array, indices, axis=0)
    assert subset == 3


@pytest.mark.parametrize(
    "array_type, expected_output_type",
    [("array", "array"), ("sparse", "sparse"), ("dataframe", "series")],
)
@pytest.mark.parametrize("indices", [2, "col_2"])
def test_safe_indexing_2d_scalar_axis_1(array_type, expected_output_type, indices):
    columns_name = ["col_0", "col_1", "col_2"]
    array = _convert_container(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]], array_type, columns_name
    )

    if isinstance(indices, str) and array_type != "dataframe":
        err_msg = (
            "Specifying the columns using strings is only supported "
            "for pandas DataFrames"
        )
        with pytest.raises(ValueError, match=err_msg):
            _safe_indexing(array, indices, axis=1)
    else:
        subset = _safe_indexing(array, indices, axis=1)
        expected_output = [3, 6, 9]
        if expected_output_type == "sparse":
            # sparse matrix are keeping the 2D shape
            expected_output = [[3], [6], [9]]
        expected_array = _convert_container(expected_output, expected_output_type)
        assert_allclose_dense_sparse(subset, expected_array)


@pytest.mark.parametrize("array_type", ["list", "array", "sparse"])
def test_safe_indexing_None_axis_0(array_type):
    X = _convert_container([[1, 2, 3], [4, 5, 6], [7, 8, 9]], array_type)
    X_subset = _safe_indexing(X, None, axis=0)
    assert_allclose_dense_sparse(X_subset, X)


def test_safe_indexing_pandas_no_matching_cols_error():
    pd = pytest.importorskip("pandas")
    err_msg = "No valid specification of the columns."
    X = pd.DataFrame(X_toy)
    with pytest.raises(ValueError, match=err_msg):
        _safe_indexing(X, [1.0], axis=1)


@pytest.mark.parametrize("axis", [None, 3])
def test_safe_indexing_error_axis(axis):
    with pytest.raises(ValueError, match="'axis' should be either 0"):
        _safe_indexing(X_toy, [0, 1], axis=axis)


@pytest.mark.parametrize("X_constructor", ["array", "series"])
def test_safe_indexing_1d_array_error(X_constructor):
    # check that we are raising an error if the array-like passed is 1D and
    # we try to index on the 2nd dimension
    X = list(range(5))
    if X_constructor == "array":
        X_constructor = np.asarray(X)
    elif X_constructor == "series":
        pd = pytest.importorskip("pandas")
        X_constructor = pd.Series(X)

    err_msg = "'X' should be a 2D NumPy array, 2D sparse matrix or pandas"
    with pytest.raises(ValueError, match=err_msg):
        _safe_indexing(X_constructor, [0, 1], axis=1)


def test_safe_indexing_container_axis_0_unsupported_type():
    indices = ["col_1", "col_2"]
    array = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    err_msg = "String indexing is not supported with 'axis=0'"
    with pytest.raises(ValueError, match=err_msg):
        _safe_indexing(array, indices, axis=0)


def test_safe_indexing_pandas_no_settingwithcopy_warning():
    # Using safe_indexing with an array-like indexer gives a copy of the
    # DataFrame -> ensure it doesn't raise a warning if modified
    pd = pytest.importorskip("pandas")
    if parse_version(pd.__version__) < parse_version("0.25.0"):
        raise SkipTest(
            "Older pandas version still raise a SettingWithCopyWarning warning"
        )
    X = pd.DataFrame({"a": [1, 2, 3], "b": [3, 4, 5]})
    subset = _safe_indexing(X, [0, 1], axis=0)
    with pytest.warns(None) as record:
        subset.iloc[0, 0] = 10
    assert len(record) == 0, f"{[str(rec.message) for rec in record]}"
    # The original dataframe is unaffected by the assignment on the subset:
    assert X.iloc[0, 0] == 1


@pytest.mark.parametrize(
    "key, err_msg",
    [
        (10, r"all features must be in \[0, 2\]"),
        ("whatever", "A given column is not a column of the dataframe"),
    ],
)
def test_get_column_indices_error(key, err_msg):
    pd = pytest.importorskip("pandas")
    X_df = pd.DataFrame(X_toy, columns=["col_0", "col_1", "col_2"])

    with pytest.raises(ValueError, match=err_msg):
        _get_column_indices(X_df, key)


@pytest.mark.parametrize(
    "key", [["col1"], ["col2"], ["col1", "col2"], ["col1", "col3"], ["col2", "col3"]]
)
def test_get_column_indices_pandas_nonunique_columns_error(key):
    pd = pytest.importorskip("pandas")
    toy = np.zeros((1, 5), dtype=int)
    columns = ["col1", "col1", "col2", "col3", "col2"]
    X = pd.DataFrame(toy, columns=columns)

    err_msg = "Selected columns, {}, are not unique in dataframe".format(key)
    with pytest.raises(ValueError) as exc_info:
        _get_column_indices(X, key)
    assert str(exc_info.value) == err_msg


def test_shuffle_on_ndim_equals_three():
    def to_tuple(A):  # to make the inner arrays hashable
        return tuple(tuple(tuple(C) for C in B) for B in A)

    A = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])  # A.shape = (2,2,2)
    S = set(to_tuple(A))
    shuffle(A)  # shouldn't raise a ValueError for dim = 3
    assert set(to_tuple(A)) == S


def test_shuffle_dont_convert_to_array():
    # Check that shuffle does not try to convert to numpy arrays with float
    # dtypes can let any indexable datastructure pass-through.
    a = ["a", "b", "c"]
    b = np.array(["a", "b", "c"], dtype=object)
    c = [1, 2, 3]
    d = MockDataFrame(np.array([["a", 0], ["b", 1], ["c", 2]], dtype=object))
    e = sp.csc_matrix(np.arange(6).reshape(3, 2))
    a_s, b_s, c_s, d_s, e_s = shuffle(a, b, c, d, e, random_state=0)

    assert a_s == ["c", "b", "a"]
    assert type(a_s) == list

    assert_array_equal(b_s, ["c", "b", "a"])
    assert b_s.dtype == object

    assert c_s == [3, 2, 1]
    assert type(c_s) == list

    assert_array_equal(d_s, np.array([["c", 2], ["b", 1], ["a", 0]], dtype=object))
    assert type(d_s) == MockDataFrame

    assert_array_equal(e_s.toarray(), np.array([[4, 5], [2, 3], [0, 1]]))


def test_gen_even_slices():
    # check that gen_even_slices contains all samples
    some_range = range(10)
    joined_range = list(chain(*[some_range[slice] for slice in gen_even_slices(10, 3)]))
    assert_array_equal(some_range, joined_range)

    # check that passing negative n_chunks raises an error
    slices = gen_even_slices(10, -1)
    with pytest.raises(ValueError, match="gen_even_slices got n_packs=-1, must be >=1"):
        next(slices)


@pytest.mark.parametrize(
    ("row_bytes", "max_n_rows", "working_memory", "expected", "warn_msg"),
    [
        (1024, None, 1, 1024, None),
        (1024, None, 0.99999999, 1023, None),
        (1023, None, 1, 1025, None),
        (1025, None, 1, 1023, None),
        (1024, None, 2, 2048, None),
        (1024, 7, 1, 7, None),
        (1024 * 1024, None, 1, 1, None),
        (
            1024 * 1024 + 1,
            None,
            1,
            1,
            "Could not adhere to working_memory config. Currently 1MiB, 2MiB required.",
        ),
    ],
)
def test_get_chunk_n_rows(row_bytes, max_n_rows, working_memory, expected, warn_msg):
    warning = None if warn_msg is None else UserWarning
    with pytest.warns(warning, match=warn_msg) as w:
        actual = get_chunk_n_rows(
            row_bytes=row_bytes,
            max_n_rows=max_n_rows,
            working_memory=working_memory,
        )

    assert actual == expected
    assert type(actual) is type(expected)
    if warn_msg is None:
        assert len(w) == 0
    with config_context(working_memory=working_memory):
        with pytest.warns(warning, match=warn_msg) as w:
            actual = get_chunk_n_rows(row_bytes=row_bytes, max_n_rows=max_n_rows)
        assert actual == expected
        assert type(actual) is type(expected)
        if warn_msg is None:
            assert len(w) == 0


@pytest.mark.parametrize(
    ["source", "message", "is_long"],
    [
        ("ABC", string.ascii_lowercase, False),
        ("ABCDEF", string.ascii_lowercase, False),
        ("ABC", string.ascii_lowercase * 3, True),
        ("ABC" * 10, string.ascii_lowercase, True),
        ("ABC", string.ascii_lowercase + "\u1048", False),
    ],
)
@pytest.mark.parametrize(
    ["time", "time_str"],
    [
        (0.2, "   0.2s"),
        (20, "  20.0s"),
        (2000, "33.3min"),
        (20000, "333.3min"),
    ],
)
def test_message_with_time(source, message, is_long, time, time_str):
    out = _message_with_time(source, message, time)
    if is_long:
        assert len(out) > 70
    else:
        assert len(out) == 70

    assert out.startswith("[" + source + "] ")
    out = out[len(source) + 3 :]

    assert out.endswith(time_str)
    out = out[: -len(time_str)]
    assert out.endswith(", total=")
    out = out[: -len(", total=")]
    assert out.endswith(message)
    out = out[: -len(message)]
    assert out.endswith(" ")
    out = out[:-1]

    if is_long:
        assert not out
    else:
        assert list(set(out)) == ["."]


@pytest.mark.parametrize(
    ["message", "expected"],
    [
        ("hello", _message_with_time("ABC", "hello", 0.1) + "\n"),
        ("", _message_with_time("ABC", "", 0.1) + "\n"),
        (None, ""),
    ],
)
def test_print_elapsed_time(message, expected, capsys, monkeypatch):
    monkeypatch.setattr(timeit, "default_timer", lambda: 0)
    with _print_elapsed_time("ABC", message):
        monkeypatch.setattr(timeit, "default_timer", lambda: 0.1)
    assert capsys.readouterr().out == expected


@pytest.mark.parametrize(
    "value, result",
    [
        (float("nan"), True),
        (np.nan, True),
        (float(np.nan), True),
        (np.float32(np.nan), True),
        (np.float64(np.nan), True),
        (0, False),
        (0.0, False),
        (None, False),
        ("", False),
        ("nan", False),
        ([np.nan], False),
        (9867966753463435747313673, False),  # Python int that overflows with C type
    ],
)
def test_is_scalar_nan(value, result):
    assert is_scalar_nan(value) is result
    # make sure that we are returning a Python bool
    assert isinstance(is_scalar_nan(value), bool)


def test_approximate_mode():
    """Make sure sklearn.utils._approximate_mode returns valid
    results for cases where "class_counts * n_draws" is enough
    to overflow 32-bit signed integer.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/20774
    """
    X = np.array([99000, 1000], dtype=np.int32)
    ret = _approximate_mode(class_counts=X, n_draws=25000, rng=0)

    # Draws 25% of the total population, so in this case a fair draw means:
    # 25% * 99.000 = 24.750
    # 25% *  1.000 =    250
    assert_array_equal(ret, [24750, 250])


def dummy_func():
    pass


def test_deprecation_joblib_api(tmpdir):

    # Only parallel_backend and register_parallel_backend are not deprecated in
    # sklearn.utils
    from sklearn.utils import parallel_backend, register_parallel_backend

    assert_no_warnings(parallel_backend, "loky", None)
    assert_no_warnings(register_parallel_backend, "failing", None)

    from sklearn.utils._joblib import joblib

    del joblib.parallel.BACKENDS["failing"]


@pytest.mark.parametrize("sequence", [[np.array(1), np.array(2)], [[1, 2], [3, 4]]])
def test_to_object_array(sequence):
    out = _to_object_array(sequence)
    assert isinstance(out, np.ndarray)
    assert out.dtype.kind == "O"
    assert out.ndim == 1
