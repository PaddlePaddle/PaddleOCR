import pytest

from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    Series,
    _testing as tm,
    get_option,
)
from pandas.core import strings as strings


def test_api(any_string_dtype):

    # GH 6106, GH 9322
    assert Series.str is strings.StringMethods
    assert isinstance(Series([""], dtype=any_string_dtype).str, strings.StringMethods)


def test_api_mi_raises():
    # GH 23679
    mi = MultiIndex.from_arrays([["a", "b", "c"]])
    msg = "Can only use .str accessor with Index, not MultiIndex"
    with pytest.raises(AttributeError, match=msg):
        mi.str
    assert not hasattr(mi, "str")


@pytest.mark.parametrize("dtype", [object, "category"])
def test_api_per_dtype(index_or_series, dtype, any_skipna_inferred_dtype):
    # one instance of parametrized fixture
    box = index_or_series
    inferred_dtype, values = any_skipna_inferred_dtype

    t = box(values, dtype=dtype)  # explicit dtype to avoid casting

    types_passing_constructor = [
        "string",
        "unicode",
        "empty",
        "bytes",
        "mixed",
        "mixed-integer",
    ]
    if inferred_dtype in types_passing_constructor:
        # GH 6106
        assert isinstance(t.str, strings.StringMethods)
    else:
        # GH 9184, GH 23011, GH 23163
        msg = "Can only use .str accessor with string values.*"
        with pytest.raises(AttributeError, match=msg):
            t.str
        assert not hasattr(t, "str")


@pytest.mark.parametrize("dtype", [object, "category"])
def test_api_per_method(
    index_or_series,
    dtype,
    any_allowed_skipna_inferred_dtype,
    any_string_method,
    request,
):
    # this test does not check correctness of the different methods,
    # just that the methods work on the specified (inferred) dtypes,
    # and raise on all others
    box = index_or_series

    # one instance of each parametrized fixture
    inferred_dtype, values = any_allowed_skipna_inferred_dtype
    method_name, args, kwargs = any_string_method

    # TODO: get rid of these xfails
    reason = None
    if box is Index and values.size == 0:
        if method_name in ["partition", "rpartition"] and kwargs.get("expand", True):
            raises = TypeError
            reason = "Method cannot deal with empty Index"
        elif method_name == "split" and kwargs.get("expand", None):
            raises = TypeError
            reason = "Split fails on empty Series when expand=True"
        elif method_name == "get_dummies":
            raises = ValueError
            reason = "Need to fortify get_dummies corner cases"

    elif (
        box is Index
        and inferred_dtype == "empty"
        and dtype == object
        and method_name == "get_dummies"
    ):
        raises = ValueError
        reason = "Need to fortify get_dummies corner cases"

    if reason is not None:
        mark = pytest.mark.xfail(raises=raises, reason=reason)
        request.node.add_marker(mark)

    t = box(values, dtype=dtype)  # explicit dtype to avoid casting
    method = getattr(t.str, method_name)

    bytes_allowed = method_name in ["decode", "get", "len", "slice"]
    # as of v0.23.4, all methods except 'cat' are very lenient with the
    # allowed data types, just returning NaN for entries that error.
    # This could be changed with an 'errors'-kwarg to the `str`-accessor,
    # see discussion in GH 13877
    mixed_allowed = method_name not in ["cat"]

    allowed_types = (
        ["string", "unicode", "empty"]
        + ["bytes"] * bytes_allowed
        + ["mixed", "mixed-integer"] * mixed_allowed
    )

    if inferred_dtype in allowed_types:
        # xref GH 23555, GH 23556
        method(*args, **kwargs)  # works!
    else:
        # GH 23011, GH 23163
        msg = (
            f"Cannot use .str.{method_name} with values of "
            f"inferred dtype {repr(inferred_dtype)}."
        )
        with pytest.raises(TypeError, match=msg):
            method(*args, **kwargs)


def test_api_for_categorical(any_string_method, any_string_dtype, request):
    # https://github.com/pandas-dev/pandas/issues/10661

    if any_string_dtype == "string[pyarrow]" or (
        any_string_dtype == "string" and get_option("string_storage") == "pyarrow"
    ):
        # unsupported operand type(s) for +: 'ArrowStringArray' and 'str'
        mark = pytest.mark.xfail(raises=TypeError, reason="Not Implemented")
        request.node.add_marker(mark)

    s = Series(list("aabb"), dtype=any_string_dtype)
    s = s + " " + s
    c = s.astype("category")
    assert isinstance(c.str, strings.StringMethods)

    method_name, args, kwargs = any_string_method

    result = getattr(c.str, method_name)(*args, **kwargs)
    expected = getattr(s.astype("object").str, method_name)(*args, **kwargs)

    if isinstance(result, DataFrame):
        tm.assert_frame_equal(result, expected)
    elif isinstance(result, Series):
        tm.assert_series_equal(result, expected)
    else:
        # str.cat(others=None) returns string, for example
        assert result == expected
