import numpy as np
import pytest

from pandas import Series
from pandas.core import strings as strings

_any_string_method = [
    ("cat", (), {"sep": ","}),
    ("cat", (Series(list("zyx")),), {"sep": ",", "join": "left"}),
    ("center", (10,), {}),
    ("contains", ("a",), {}),
    ("count", ("a",), {}),
    ("decode", ("UTF-8",), {}),
    ("encode", ("UTF-8",), {}),
    ("endswith", ("a",), {}),
    ("endswith", ("a",), {"na": True}),
    ("endswith", ("a",), {"na": False}),
    ("extract", ("([a-z]*)",), {"expand": False}),
    ("extract", ("([a-z]*)",), {"expand": True}),
    ("extractall", ("([a-z]*)",), {}),
    ("find", ("a",), {}),
    ("findall", ("a",), {}),
    ("get", (0,), {}),
    # because "index" (and "rindex") fail intentionally
    # if the string is not found, search only for empty string
    ("index", ("",), {}),
    ("join", (",",), {}),
    ("ljust", (10,), {}),
    ("match", ("a",), {}),
    ("fullmatch", ("a",), {}),
    ("normalize", ("NFC",), {}),
    ("pad", (10,), {}),
    ("partition", (" ",), {"expand": False}),
    ("partition", (" ",), {"expand": True}),
    ("repeat", (3,), {}),
    ("replace", ("a", "z"), {}),
    ("rfind", ("a",), {}),
    ("rindex", ("",), {}),
    ("rjust", (10,), {}),
    ("rpartition", (" ",), {"expand": False}),
    ("rpartition", (" ",), {"expand": True}),
    ("slice", (0, 1), {}),
    ("slice_replace", (0, 1, "z"), {}),
    ("split", (" ",), {"expand": False}),
    ("split", (" ",), {"expand": True}),
    ("startswith", ("a",), {}),
    ("startswith", ("a",), {"na": True}),
    ("startswith", ("a",), {"na": False}),
    # translating unicode points of "a" to "d"
    ("translate", ({97: 100},), {}),
    ("wrap", (2,), {}),
    ("zfill", (10,), {}),
] + list(
    zip(
        [
            # methods without positional arguments: zip with empty tuple and empty dict
            "capitalize",
            "cat",
            "get_dummies",
            "isalnum",
            "isalpha",
            "isdecimal",
            "isdigit",
            "islower",
            "isnumeric",
            "isspace",
            "istitle",
            "isupper",
            "len",
            "lower",
            "lstrip",
            "partition",
            "rpartition",
            "rsplit",
            "rstrip",
            "slice",
            "slice_replace",
            "split",
            "strip",
            "swapcase",
            "title",
            "upper",
            "casefold",
        ],
        [()] * 100,
        [{}] * 100,
    )
)
ids, _, _ = zip(*_any_string_method)  # use method name as fixture-id
missing_methods = {
    f for f in dir(strings.StringMethods) if not f.startswith("_")
} - set(ids)

# test that the above list captures all methods of StringMethods
assert not missing_methods


@pytest.fixture(params=_any_string_method, ids=ids)
def any_string_method(request):
    """
    Fixture for all public methods of `StringMethods`

    This fixture returns a tuple of the method name and sample arguments
    necessary to call the method.

    Returns
    -------
    method_name : str
        The name of the method in `StringMethods`
    args : tuple
        Sample values for the positional arguments
    kwargs : dict
        Sample values for the keyword arguments

    Examples
    --------
    >>> def test_something(any_string_method):
    ...     s = Series(['a', 'b', np.nan, 'd'])
    ...
    ...     method_name, args, kwargs = any_string_method
    ...     method = getattr(s.str, method_name)
    ...     # will not raise
    ...     method(*args, **kwargs)
    """
    return request.param


# subset of the full set from pandas/conftest.py
_any_allowed_skipna_inferred_dtype = [
    ("string", ["a", np.nan, "c"]),
    ("bytes", [b"a", np.nan, b"c"]),
    ("empty", [np.nan, np.nan, np.nan]),
    ("empty", []),
    ("mixed-integer", ["a", np.nan, 2]),
]
ids, _ = zip(*_any_allowed_skipna_inferred_dtype)  # use inferred type as id


@pytest.fixture(params=_any_allowed_skipna_inferred_dtype, ids=ids)
def any_allowed_skipna_inferred_dtype(request):
    """
    Fixture for all (inferred) dtypes allowed in StringMethods.__init__

    The covered (inferred) types are:
    * 'string'
    * 'empty'
    * 'bytes'
    * 'mixed'
    * 'mixed-integer'

    Returns
    -------
    inferred_dtype : str
        The string for the inferred dtype from _libs.lib.infer_dtype
    values : np.ndarray
        An array of object dtype that will be inferred to have
        `inferred_dtype`

    Examples
    --------
    >>> import pandas._libs.lib as lib
    >>>
    >>> def test_something(any_allowed_skipna_inferred_dtype):
    ...     inferred_dtype, values = any_allowed_skipna_inferred_dtype
    ...     # will pass
    ...     assert lib.infer_dtype(values, skipna=True) == inferred_dtype
    ...
    ...     # constructor for .str-accessor will also pass
    ...     Series(values).str
    """
    inferred_dtype, values = request.param
    values = np.array(values, dtype=object)  # object dtype to avoid casting

    # correctness of inference tested in tests/dtypes/test_inference.py
    return inferred_dtype, values
