import numpy as np
import pytest

from pandas import (
    DataFrame,
    Index,
    Series,
    Timestamp,
)
import pandas._testing as tm


def _assert_almost_equal_both(a, b, **kwargs):
    """
    Check that two objects are approximately equal.

    This check is performed commutatively.

    Parameters
    ----------
    a : object
        The first object to compare.
    b : object
        The second object to compare.
    **kwargs
        The arguments passed to `tm.assert_almost_equal`.
    """
    tm.assert_almost_equal(a, b, **kwargs)
    tm.assert_almost_equal(b, a, **kwargs)


def _assert_not_almost_equal(a, b, **kwargs):
    """
    Check that two objects are not approximately equal.

    Parameters
    ----------
    a : object
        The first object to compare.
    b : object
        The second object to compare.
    **kwargs
        The arguments passed to `tm.assert_almost_equal`.
    """
    try:
        tm.assert_almost_equal(a, b, **kwargs)
        msg = f"{a} and {b} were approximately equal when they shouldn't have been"
        pytest.fail(msg=msg)
    except AssertionError:
        pass


def _assert_not_almost_equal_both(a, b, **kwargs):
    """
    Check that two objects are not approximately equal.

    This check is performed commutatively.

    Parameters
    ----------
    a : object
        The first object to compare.
    b : object
        The second object to compare.
    **kwargs
        The arguments passed to `tm.assert_almost_equal`.
    """
    _assert_not_almost_equal(a, b, **kwargs)
    _assert_not_almost_equal(b, a, **kwargs)


@pytest.mark.parametrize(
    "a,b,check_less_precise",
    [(1.1, 1.1, False), (1.1, 1.100001, True), (1.1, 1.1001, 2)],
)
def test_assert_almost_equal_deprecated(a, b, check_less_precise):
    # GH#30562
    with tm.assert_produces_warning(FutureWarning):
        _assert_almost_equal_both(a, b, check_less_precise=check_less_precise)


@pytest.mark.parametrize(
    "a,b",
    [
        (1.1, 1.1),
        (1.1, 1.100001),
        (np.int16(1), 1.000001),
        (np.float64(1.1), 1.1),
        (np.uint32(5), 5),
    ],
)
def test_assert_almost_equal_numbers(a, b):
    _assert_almost_equal_both(a, b)


@pytest.mark.parametrize(
    "a,b",
    [
        (1.1, 1),
        (1.1, True),
        (1, 2),
        (1.0001, np.int16(1)),
        # The following two examples are not "almost equal" due to tol.
        (0.1, 0.1001),
        (0.0011, 0.0012),
    ],
)
def test_assert_not_almost_equal_numbers(a, b):
    _assert_not_almost_equal_both(a, b)


@pytest.mark.parametrize(
    "a,b",
    [
        (1.1, 1.1),
        (1.1, 1.100001),
        (1.1, 1.1001),
        (0.000001, 0.000005),
        (1000.0, 1000.0005),
        # Testing this example, as per #13357
        (0.000011, 0.000012),
    ],
)
def test_assert_almost_equal_numbers_atol(a, b):
    # Equivalent to the deprecated check_less_precise=True
    _assert_almost_equal_both(a, b, rtol=0.5e-3, atol=0.5e-3)


@pytest.mark.parametrize("a,b", [(1.1, 1.11), (0.1, 0.101), (0.000011, 0.001012)])
def test_assert_not_almost_equal_numbers_atol(a, b):
    _assert_not_almost_equal_both(a, b, atol=1e-3)


@pytest.mark.parametrize(
    "a,b",
    [
        (1.1, 1.1),
        (1.1, 1.100001),
        (1.1, 1.1001),
        (1000.0, 1000.0005),
        (1.1, 1.11),
        (0.1, 0.101),
    ],
)
def test_assert_almost_equal_numbers_rtol(a, b):
    _assert_almost_equal_both(a, b, rtol=0.05)


@pytest.mark.parametrize("a,b", [(0.000011, 0.000012), (0.000001, 0.000005)])
def test_assert_not_almost_equal_numbers_rtol(a, b):
    _assert_not_almost_equal_both(a, b, rtol=0.05)


@pytest.mark.parametrize(
    "a,b,rtol",
    [
        (1.00001, 1.00005, 0.001),
        (-0.908356 + 0.2j, -0.908358 + 0.2j, 1e-3),
        (0.1 + 1.009j, 0.1 + 1.006j, 0.1),
        (0.1001 + 2.0j, 0.1 + 2.001j, 0.01),
    ],
)
def test_assert_almost_equal_complex_numbers(a, b, rtol):
    _assert_almost_equal_both(a, b, rtol=rtol)
    _assert_almost_equal_both(np.complex64(a), np.complex64(b), rtol=rtol)
    _assert_almost_equal_both(np.complex128(a), np.complex128(b), rtol=rtol)


@pytest.mark.parametrize(
    "a,b,rtol",
    [
        (0.58310768, 0.58330768, 1e-7),
        (-0.908 + 0.2j, -0.978 + 0.2j, 0.001),
        (0.1 + 1j, 0.1 + 2j, 0.01),
        (-0.132 + 1.001j, -0.132 + 1.005j, 1e-5),
        (0.58310768j, 0.58330768j, 1e-9),
    ],
)
def test_assert_not_almost_equal_complex_numbers(a, b, rtol):
    _assert_not_almost_equal_both(a, b, rtol=rtol)
    _assert_not_almost_equal_both(np.complex64(a), np.complex64(b), rtol=rtol)
    _assert_not_almost_equal_both(np.complex128(a), np.complex128(b), rtol=rtol)


@pytest.mark.parametrize("a,b", [(0, 0), (0, 0.0), (0, np.float64(0)), (0.00000001, 0)])
def test_assert_almost_equal_numbers_with_zeros(a, b):
    _assert_almost_equal_both(a, b)


@pytest.mark.parametrize("a,b", [(0.001, 0), (1, 0)])
def test_assert_not_almost_equal_numbers_with_zeros(a, b):
    _assert_not_almost_equal_both(a, b)


@pytest.mark.parametrize("a,b", [(1, "abc"), (1, [1]), (1, object())])
def test_assert_not_almost_equal_numbers_with_mixed(a, b):
    _assert_not_almost_equal_both(a, b)


@pytest.mark.parametrize(
    "left_dtype", ["M8[ns]", "m8[ns]", "float64", "int64", "object"]
)
@pytest.mark.parametrize(
    "right_dtype", ["M8[ns]", "m8[ns]", "float64", "int64", "object"]
)
def test_assert_almost_equal_edge_case_ndarrays(left_dtype, right_dtype):
    # Empty compare.
    _assert_almost_equal_both(
        np.array([], dtype=left_dtype),
        np.array([], dtype=right_dtype),
        check_dtype=False,
    )


def test_assert_almost_equal_dicts():
    _assert_almost_equal_both({"a": 1, "b": 2}, {"a": 1, "b": 2})


@pytest.mark.parametrize(
    "a,b",
    [
        ({"a": 1, "b": 2}, {"a": 1, "b": 3}),
        ({"a": 1, "b": 2}, {"a": 1, "b": 2, "c": 3}),
        ({"a": 1}, 1),
        ({"a": 1}, "abc"),
        ({"a": 1}, [1]),
    ],
)
def test_assert_not_almost_equal_dicts(a, b):
    _assert_not_almost_equal_both(a, b)


@pytest.mark.parametrize("val", [1, 2])
def test_assert_almost_equal_dict_like_object(val):
    dict_val = 1
    real_dict = {"a": val}

    class DictLikeObj:
        def keys(self):
            return ("a",)

        def __getitem__(self, item):
            if item == "a":
                return dict_val

    func = (
        _assert_almost_equal_both if val == dict_val else _assert_not_almost_equal_both
    )
    func(real_dict, DictLikeObj(), check_dtype=False)


def test_assert_almost_equal_strings():
    _assert_almost_equal_both("abc", "abc")


@pytest.mark.parametrize(
    "a,b", [("abc", "abcd"), ("abc", "abd"), ("abc", 1), ("abc", [1])]
)
def test_assert_not_almost_equal_strings(a, b):
    _assert_not_almost_equal_both(a, b)


@pytest.mark.parametrize(
    "a,b", [([1, 2, 3], [1, 2, 3]), (np.array([1, 2, 3]), np.array([1, 2, 3]))]
)
def test_assert_almost_equal_iterables(a, b):
    _assert_almost_equal_both(a, b)


@pytest.mark.parametrize(
    "a,b",
    [
        # Class is different.
        (np.array([1, 2, 3]), [1, 2, 3]),
        # Dtype is different.
        (np.array([1, 2, 3]), np.array([1.0, 2.0, 3.0])),
        # Can't compare generators.
        (iter([1, 2, 3]), [1, 2, 3]),
        ([1, 2, 3], [1, 2, 4]),
        ([1, 2, 3], [1, 2, 3, 4]),
        ([1, 2, 3], 1),
    ],
)
def test_assert_not_almost_equal_iterables(a, b):
    _assert_not_almost_equal(a, b)


def test_assert_almost_equal_null():
    _assert_almost_equal_both(None, None)


@pytest.mark.parametrize("a,b", [(None, np.NaN), (None, 0), (np.NaN, 0)])
def test_assert_not_almost_equal_null(a, b):
    _assert_not_almost_equal(a, b)


@pytest.mark.parametrize(
    "a,b",
    [
        (np.inf, np.inf),
        (np.inf, float("inf")),
        (np.array([np.inf, np.nan, -np.inf]), np.array([np.inf, np.nan, -np.inf])),
        (
            np.array([np.inf, None, -np.inf], dtype=np.object_),
            np.array([np.inf, np.nan, -np.inf], dtype=np.object_),
        ),
    ],
)
def test_assert_almost_equal_inf(a, b):
    _assert_almost_equal_both(a, b)


def test_assert_not_almost_equal_inf():
    _assert_not_almost_equal_both(np.inf, 0)


@pytest.mark.parametrize(
    "a,b",
    [
        (Index([1.0, 1.1]), Index([1.0, 1.100001])),
        (Series([1.0, 1.1]), Series([1.0, 1.100001])),
        (np.array([1.1, 2.000001]), np.array([1.1, 2.0])),
        (DataFrame({"a": [1.0, 1.1]}), DataFrame({"a": [1.0, 1.100001]})),
    ],
)
def test_assert_almost_equal_pandas(a, b):
    _assert_almost_equal_both(a, b)


def test_assert_almost_equal_object():
    a = [Timestamp("2011-01-01"), Timestamp("2011-01-01")]
    b = [Timestamp("2011-01-01"), Timestamp("2011-01-01")]
    _assert_almost_equal_both(a, b)


def test_assert_almost_equal_value_mismatch():
    msg = "expected 2\\.00000 but got 1\\.00000, with rtol=1e-05, atol=1e-08"

    with pytest.raises(AssertionError, match=msg):
        tm.assert_almost_equal(1, 2)


@pytest.mark.parametrize(
    "a,b,klass1,klass2",
    [(np.array([1]), 1, "ndarray", "int"), (1, np.array([1]), "int", "ndarray")],
)
def test_assert_almost_equal_class_mismatch(a, b, klass1, klass2):

    msg = f"""numpy array are different

numpy array classes are different
\\[left\\]:  {klass1}
\\[right\\]: {klass2}"""

    with pytest.raises(AssertionError, match=msg):
        tm.assert_almost_equal(a, b)


def test_assert_almost_equal_value_mismatch1():
    msg = """numpy array are different

numpy array values are different \\(66\\.66667 %\\)
\\[left\\]:  \\[nan, 2\\.0, 3\\.0\\]
\\[right\\]: \\[1\\.0, nan, 3\\.0\\]"""

    with pytest.raises(AssertionError, match=msg):
        tm.assert_almost_equal(np.array([np.nan, 2, 3]), np.array([1, np.nan, 3]))


def test_assert_almost_equal_value_mismatch2():
    msg = """numpy array are different

numpy array values are different \\(50\\.0 %\\)
\\[left\\]:  \\[1, 2\\]
\\[right\\]: \\[1, 3\\]"""

    with pytest.raises(AssertionError, match=msg):
        tm.assert_almost_equal(np.array([1, 2]), np.array([1, 3]))


def test_assert_almost_equal_value_mismatch3():
    msg = """numpy array are different

numpy array values are different \\(16\\.66667 %\\)
\\[left\\]:  \\[\\[1, 2\\], \\[3, 4\\], \\[5, 6\\]\\]
\\[right\\]: \\[\\[1, 3\\], \\[3, 4\\], \\[5, 6\\]\\]"""

    with pytest.raises(AssertionError, match=msg):
        tm.assert_almost_equal(
            np.array([[1, 2], [3, 4], [5, 6]]), np.array([[1, 3], [3, 4], [5, 6]])
        )


def test_assert_almost_equal_value_mismatch4():
    msg = """numpy array are different

numpy array values are different \\(25\\.0 %\\)
\\[left\\]:  \\[\\[1, 2\\], \\[3, 4\\]\\]
\\[right\\]: \\[\\[1, 3\\], \\[3, 4\\]\\]"""

    with pytest.raises(AssertionError, match=msg):
        tm.assert_almost_equal(np.array([[1, 2], [3, 4]]), np.array([[1, 3], [3, 4]]))


def test_assert_almost_equal_shape_mismatch_override():
    msg = """Index are different

Index shapes are different
\\[left\\]:  \\(2L*,\\)
\\[right\\]: \\(3L*,\\)"""
    with pytest.raises(AssertionError, match=msg):
        tm.assert_almost_equal(np.array([1, 2]), np.array([3, 4, 5]), obj="Index")


def test_assert_almost_equal_unicode():
    # see gh-20503
    msg = """numpy array are different

numpy array values are different \\(33\\.33333 %\\)
\\[left\\]:  \\[á, à, ä\\]
\\[right\\]: \\[á, à, å\\]"""

    with pytest.raises(AssertionError, match=msg):
        tm.assert_almost_equal(np.array(["á", "à", "ä"]), np.array(["á", "à", "å"]))


def test_assert_almost_equal_timestamp():
    a = np.array([Timestamp("2011-01-01"), Timestamp("2011-01-01")])
    b = np.array([Timestamp("2011-01-01"), Timestamp("2011-01-02")])

    msg = """numpy array are different

numpy array values are different \\(50\\.0 %\\)
\\[left\\]:  \\[2011-01-01 00:00:00, 2011-01-01 00:00:00\\]
\\[right\\]: \\[2011-01-01 00:00:00, 2011-01-02 00:00:00\\]"""

    with pytest.raises(AssertionError, match=msg):
        tm.assert_almost_equal(a, b)


def test_assert_almost_equal_iterable_length_mismatch():
    msg = """Iterable are different

Iterable length are different
\\[left\\]:  2
\\[right\\]: 3"""

    with pytest.raises(AssertionError, match=msg):
        tm.assert_almost_equal([1, 2], [3, 4, 5])


def test_assert_almost_equal_iterable_values_mismatch():
    msg = """Iterable are different

Iterable values are different \\(50\\.0 %\\)
\\[left\\]:  \\[1, 2\\]
\\[right\\]: \\[1, 3\\]"""

    with pytest.raises(AssertionError, match=msg):
        tm.assert_almost_equal([1, 2], [1, 3])
