"""Testing utilities."""

# Copyright (c) 2011, 2012
# Authors: Pietro Berkes,
#          Andreas Muller
#          Mathieu Blondel
#          Olivier Grisel
#          Arnaud Joly
#          Denis Engemann
#          Giorgio Patrini
#          Thierry Guillemot
# License: BSD 3 clause
import os
import os.path as op
import inspect
import warnings
import sys
import functools
import tempfile
from subprocess import check_output, STDOUT, CalledProcessError
from subprocess import TimeoutExpired
import re
import contextlib
from collections.abc import Iterable

import scipy as sp
from functools import wraps
from inspect import signature

import shutil
import atexit
import unittest
from unittest import TestCase

# WindowsError only exist on Windows
try:
    WindowsError
except NameError:
    WindowsError = None

from numpy.testing import assert_allclose
from numpy.testing import assert_almost_equal
from numpy.testing import assert_approx_equal
from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_array_less
import numpy as np
import joblib

import sklearn
from sklearn.utils import IS_PYPY, _IS_32BIT, deprecated
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import (
    check_array,
    check_is_fitted,
    check_X_y,
)


__all__ = [
    "assert_raises",
    "assert_raises_regexp",
    "assert_array_equal",
    "assert_almost_equal",
    "assert_array_almost_equal",
    "assert_array_less",
    "assert_approx_equal",
    "assert_allclose",
    "assert_run_python_script",
    "SkipTest",
]

_dummy = TestCase("__init__")
assert_raises = _dummy.assertRaises
SkipTest = unittest.case.SkipTest
assert_dict_equal = _dummy.assertDictEqual

assert_raises_regex = _dummy.assertRaisesRegex
# assert_raises_regexp is deprecated in Python 3.4 in favor of
# assert_raises_regex but lets keep the backward compat in scikit-learn with
# the old name for now
assert_raises_regexp = assert_raises_regex


# TODO: Remove in 1.2
@deprecated(  # type: ignore
    "`assert_warns` is deprecated in 1.0 and will be removed in 1.2."
    "Use `pytest.warns` instead."
)
def assert_warns(warning_class, func, *args, **kw):
    """Test that a certain warning occurs.

    .. deprecated:: 1.0
        `assert_warns` is deprecated in 1.0 and will be removed in 1.2.
        Use `pytest.warns` instead.

    Parameters
    ----------
    warning_class : the warning class
        The class to test for, e.g. UserWarning.

    func : callable
        Callable object to trigger warnings.

    *args : the positional arguments to `func`.

    **kw : the keyword arguments to `func`

    Returns
    -------
    result : the return value of `func`

    """
    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        # Trigger a warning.
        result = func(*args, **kw)
        if hasattr(np, "FutureWarning"):
            # Filter out numpy-specific warnings in numpy >= 1.9
            w = [e for e in w if e.category is not np.VisibleDeprecationWarning]

        # Verify some things
        if not len(w) > 0:
            raise AssertionError("No warning raised when calling %s" % func.__name__)

        found = any(warning.category is warning_class for warning in w)
        if not found:
            raise AssertionError(
                "%s did not give warning: %s( is %s)"
                % (func.__name__, warning_class, w)
            )
    return result


# TODO: Remove in 1.2
@deprecated(  # type: ignore
    "`assert_warns_message` is deprecated in 1.0 and will be removed in 1.2."
    "Use `pytest.warns` instead."
)
def assert_warns_message(warning_class, message, func, *args, **kw):
    # very important to avoid uncontrolled state propagation
    """Test that a certain warning occurs and with a certain message.

    .. deprecated:: 1.0
        `assert_warns_message` is deprecated in 1.0 and will be removed in 1.2.
        Use `pytest.warns` instead.

    Parameters
    ----------
    warning_class : the warning class
        The class to test for, e.g. UserWarning.

    message : str or callable
        The message or a substring of the message to test for. If callable,
        it takes a string as the argument and will trigger an AssertionError
        if the callable returns `False`.

    func : callable
        Callable object to trigger warnings.

    *args : the positional arguments to `func`.

    **kw : the keyword arguments to `func`.

    Returns
    -------
    result : the return value of `func`

    """
    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        if hasattr(np, "FutureWarning"):
            # Let's not catch the numpy internal DeprecationWarnings
            warnings.simplefilter("ignore", np.VisibleDeprecationWarning)
        # Trigger a warning.
        result = func(*args, **kw)
        # Verify some things
        if not len(w) > 0:
            raise AssertionError("No warning raised when calling %s" % func.__name__)

        found = [issubclass(warning.category, warning_class) for warning in w]
        if not any(found):
            raise AssertionError(
                "No warning raised for %s with class %s"
                % (func.__name__, warning_class)
            )

        message_found = False
        # Checks the message of all warnings belong to warning_class
        for index in [i for i, x in enumerate(found) if x]:
            # substring will match, the entire message with typo won't
            msg = w[index].message  # For Python 3 compatibility
            msg = str(msg.args[0] if hasattr(msg, "args") else msg)
            if callable(message):  # add support for certain tests
                check_in_message = message
            else:

                def check_in_message(msg):
                    return message in msg

            if check_in_message(msg):
                message_found = True
                break

        if not message_found:
            raise AssertionError(
                "Did not receive the message you expected ('%s') for <%s>, got: '%s'"
                % (message, func.__name__, msg)
            )

    return result


# To remove when we support numpy 1.7
def assert_no_warnings(func, *args, **kw):
    """
    Parameters
    ----------
    func
    *args
    **kw
    """
    # very important to avoid uncontrolled state propagation
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        result = func(*args, **kw)
        if hasattr(np, "FutureWarning"):
            # Filter out numpy-specific warnings in numpy >= 1.9
            w = [e for e in w if e.category is not np.VisibleDeprecationWarning]

        if len(w) > 0:
            raise AssertionError(
                "Got warnings when calling %s: [%s]"
                % (func.__name__, ", ".join(str(warning) for warning in w))
            )
    return result


def ignore_warnings(obj=None, category=Warning):
    """Context manager and decorator to ignore warnings.

    Note: Using this (in both variants) will clear all warnings
    from all python modules loaded. In case you need to test
    cross-module-warning-logging, this is not your tool of choice.

    Parameters
    ----------
    obj : callable, default=None
        callable where you want to ignore the warnings.
    category : warning class, default=Warning
        The category to filter. If Warning, all categories will be muted.

    Examples
    --------
    >>> import warnings
    >>> from sklearn.utils._testing import ignore_warnings
    >>> with ignore_warnings():
    ...     warnings.warn('buhuhuhu')

    >>> def nasty_warn():
    ...     warnings.warn('buhuhuhu')
    ...     print(42)

    >>> ignore_warnings(nasty_warn)()
    42
    """
    if isinstance(obj, type) and issubclass(obj, Warning):
        # Avoid common pitfall of passing category as the first positional
        # argument which result in the test not being run
        warning_name = obj.__name__
        raise ValueError(
            "'obj' should be a callable where you want to ignore warnings. "
            "You passed a warning class instead: 'obj={warning_name}'. "
            "If you want to pass a warning class to ignore_warnings, "
            "you should use 'category={warning_name}'".format(warning_name=warning_name)
        )
    elif callable(obj):
        return _IgnoreWarnings(category=category)(obj)
    else:
        return _IgnoreWarnings(category=category)


class _IgnoreWarnings:
    """Improved and simplified Python warnings context manager and decorator.

    This class allows the user to ignore the warnings raised by a function.
    Copied from Python 2.7.5 and modified as required.

    Parameters
    ----------
    category : tuple of warning class, default=Warning
        The category to filter. By default, all the categories will be muted.

    """

    def __init__(self, category):
        self._record = True
        self._module = sys.modules["warnings"]
        self._entered = False
        self.log = []
        self.category = category

    def __call__(self, fn):
        """Decorator to catch and hide warnings without visual nesting."""

        @wraps(fn)
        def wrapper(*args, **kwargs):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", self.category)
                return fn(*args, **kwargs)

        return wrapper

    def __repr__(self):
        args = []
        if self._record:
            args.append("record=True")
        if self._module is not sys.modules["warnings"]:
            args.append("module=%r" % self._module)
        name = type(self).__name__
        return "%s(%s)" % (name, ", ".join(args))

    def __enter__(self):
        if self._entered:
            raise RuntimeError("Cannot enter %r twice" % self)
        self._entered = True
        self._filters = self._module.filters
        self._module.filters = self._filters[:]
        self._showwarning = self._module.showwarning
        warnings.simplefilter("ignore", self.category)

    def __exit__(self, *exc_info):
        if not self._entered:
            raise RuntimeError("Cannot exit %r without entering first" % self)
        self._module.filters = self._filters
        self._module.showwarning = self._showwarning
        self.log[:] = []


def assert_raise_message(exceptions, message, function, *args, **kwargs):
    """Helper function to test the message raised in an exception.

    Given an exception, a callable to raise the exception, and
    a message string, tests that the correct exception is raised and
    that the message is a substring of the error thrown. Used to test
    that the specific message thrown during an exception is correct.

    Parameters
    ----------
    exceptions : exception or tuple of exception
        An Exception object.

    message : str
        The error message or a substring of the error message.

    function : callable
        Callable object to raise error.

    *args : the positional arguments to `function`.

    **kwargs : the keyword arguments to `function`.
    """
    try:
        function(*args, **kwargs)
    except exceptions as e:
        error_message = str(e)
        if message not in error_message:
            raise AssertionError(
                "Error message does not include the expected"
                " string: %r. Observed error message: %r" % (message, error_message)
            )
    else:
        # concatenate exception names
        if isinstance(exceptions, tuple):
            names = " or ".join(e.__name__ for e in exceptions)
        else:
            names = exceptions.__name__

        raise AssertionError("%s not raised by %s" % (names, function.__name__))


def assert_allclose_dense_sparse(x, y, rtol=1e-07, atol=1e-9, err_msg=""):
    """Assert allclose for sparse and dense data.

    Both x and y need to be either sparse or dense, they
    can't be mixed.

    Parameters
    ----------
    x : {array-like, sparse matrix}
        First array to compare.

    y : {array-like, sparse matrix}
        Second array to compare.

    rtol : float, default=1e-07
        relative tolerance; see numpy.allclose.

    atol : float, default=1e-9
        absolute tolerance; see numpy.allclose. Note that the default here is
        more tolerant than the default for numpy.testing.assert_allclose, where
        atol=0.

    err_msg : str, default=''
        Error message to raise.
    """
    if sp.sparse.issparse(x) and sp.sparse.issparse(y):
        x = x.tocsr()
        y = y.tocsr()
        x.sum_duplicates()
        y.sum_duplicates()
        assert_array_equal(x.indices, y.indices, err_msg=err_msg)
        assert_array_equal(x.indptr, y.indptr, err_msg=err_msg)
        assert_allclose(x.data, y.data, rtol=rtol, atol=atol, err_msg=err_msg)
    elif not sp.sparse.issparse(x) and not sp.sparse.issparse(y):
        # both dense
        assert_allclose(x, y, rtol=rtol, atol=atol, err_msg=err_msg)
    else:
        raise ValueError(
            "Can only compare two sparse matrices, not a sparse matrix and an array."
        )


def set_random_state(estimator, random_state=0):
    """Set random state of an estimator if it has the `random_state` param.

    Parameters
    ----------
    estimator : object
        The estimator.
    random_state : int, RandomState instance or None, default=0
        Pseudo random number generator state.
        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.
    """
    if "random_state" in estimator.get_params():
        estimator.set_params(random_state=random_state)


try:
    import pytest

    skip_if_32bit = pytest.mark.skipif(_IS_32BIT, reason="skipped on 32bit platforms")
    skip_travis = pytest.mark.skipif(
        os.environ.get("TRAVIS") == "true", reason="skip on travis"
    )
    fails_if_pypy = pytest.mark.xfail(IS_PYPY, reason="not compatible with PyPy")
    skip_if_no_parallel = pytest.mark.skipif(
        not joblib.parallel.mp, reason="joblib is in serial mode"
    )

    #  Decorator for tests involving both BLAS calls and multiprocessing.
    #
    #  Under POSIX (e.g. Linux or OSX), using multiprocessing in conjunction
    #  with some implementation of BLAS (or other libraries that manage an
    #  internal posix thread pool) can cause a crash or a freeze of the Python
    #  process.
    #
    #  In practice all known packaged distributions (from Linux distros or
    #  Anaconda) of BLAS under Linux seems to be safe. So we this problem seems
    #  to only impact OSX users.
    #
    #  This wrapper makes it possible to skip tests that can possibly cause
    #  this crash under OS X with.
    #
    #  Under Python 3.4+ it is possible to use the `forkserver` start method
    #  for multiprocessing to avoid this issue. However it can cause pickling
    #  errors on interactively defined functions. It therefore not enabled by
    #  default.

    if_safe_multiprocessing_with_blas = pytest.mark.skipif(
        sys.platform == "darwin", reason="Possible multi-process bug with some BLAS"
    )
except ImportError:
    pass


def check_skip_network():
    if int(os.environ.get("SKLEARN_SKIP_NETWORK_TESTS", 0)):
        raise SkipTest("Text tutorial requires large dataset download")


def _delete_folder(folder_path, warn=False):
    """Utility function to cleanup a temporary folder if still existing.

    Copy from joblib.pool (for independence).
    """
    try:
        if os.path.exists(folder_path):
            # This can fail under windows,
            #  but will succeed when called by atexit
            shutil.rmtree(folder_path)
    except WindowsError:
        if warn:
            warnings.warn("Could not delete temporary folder %s" % folder_path)


class TempMemmap:
    """
    Parameters
    ----------
    data
    mmap_mode : str, default='r'
    """

    def __init__(self, data, mmap_mode="r"):
        self.mmap_mode = mmap_mode
        self.data = data

    def __enter__(self):
        data_read_only, self.temp_folder = create_memmap_backed_data(
            self.data, mmap_mode=self.mmap_mode, return_folder=True
        )
        return data_read_only

    def __exit__(self, exc_type, exc_val, exc_tb):
        _delete_folder(self.temp_folder)


def create_memmap_backed_data(data, mmap_mode="r", return_folder=False, aligned=False):
    """
    Parameters
    ----------
    data
    mmap_mode : str, default='r'
    return_folder :  bool, default=False
    aligned : bool, default=False
        If True, if input is a single numpy array and if the input array is aligned,
        the memory mapped array will also be aligned. This is a workaround for
        https://github.com/joblib/joblib/issues/563.
    """
    temp_folder = tempfile.mkdtemp(prefix="sklearn_testing_")
    atexit.register(functools.partial(_delete_folder, temp_folder, warn=True))
    if aligned:
        if isinstance(data, np.ndarray) and data.flags.aligned:
            # https://numpy.org/doc/stable/reference/generated/numpy.memmap.html
            filename = op.join(temp_folder, "data.dat")
            fp = np.memmap(filename, dtype=data.dtype, mode="w+", shape=data.shape)
            fp[:] = data[:]  # write data to memmap array
            fp.flush()
            memmap_backed_data = np.memmap(
                filename, dtype=data.dtype, mode=mmap_mode, shape=data.shape
            )
        else:
            raise ValueError("If aligned=True, input must be a single numpy array.")
    else:
        filename = op.join(temp_folder, "data.pkl")
        joblib.dump(data, filename)
        memmap_backed_data = joblib.load(filename, mmap_mode=mmap_mode)
    result = (
        memmap_backed_data if not return_folder else (memmap_backed_data, temp_folder)
    )
    return result


# Utils to test docstrings


def _get_args(function, varargs=False):
    """Helper to get function arguments."""

    try:
        params = signature(function).parameters
    except ValueError:
        # Error on builtin C function
        return []
    args = [
        key
        for key, param in params.items()
        if param.kind not in (param.VAR_POSITIONAL, param.VAR_KEYWORD)
    ]
    if varargs:
        varargs = [
            param.name
            for param in params.values()
            if param.kind == param.VAR_POSITIONAL
        ]
        if len(varargs) == 0:
            varargs = None
        return args, varargs
    else:
        return args


def _get_func_name(func):
    """Get function full name.

    Parameters
    ----------
    func : callable
        The function object.

    Returns
    -------
    name : str
        The function name.
    """
    parts = []
    module = inspect.getmodule(func)
    if module:
        parts.append(module.__name__)

    qualname = func.__qualname__
    if qualname != func.__name__:
        parts.append(qualname[: qualname.find(".")])

    parts.append(func.__name__)
    return ".".join(parts)


def check_docstring_parameters(func, doc=None, ignore=None):
    """Helper to check docstring.

    Parameters
    ----------
    func : callable
        The function object to test.
    doc : str, default=None
        Docstring if it is passed manually to the test.
    ignore : list, default=None
        Parameters to ignore.

    Returns
    -------
    incorrect : list
        A list of string describing the incorrect results.
    """
    from numpydoc import docscrape

    incorrect = []
    ignore = [] if ignore is None else ignore

    func_name = _get_func_name(func)
    if not func_name.startswith("sklearn.") or func_name.startswith(
        "sklearn.externals"
    ):
        return incorrect
    # Don't check docstring for property-functions
    if inspect.isdatadescriptor(func):
        return incorrect
    # Don't check docstring for setup / teardown pytest functions
    if func_name.split(".")[-1] in ("setup_module", "teardown_module"):
        return incorrect
    # Dont check estimator_checks module
    if func_name.split(".")[2] == "estimator_checks":
        return incorrect
    # Get the arguments from the function signature
    param_signature = list(filter(lambda x: x not in ignore, _get_args(func)))
    # drop self
    if len(param_signature) > 0 and param_signature[0] == "self":
        param_signature.remove("self")

    # Analyze function's docstring
    if doc is None:
        with warnings.catch_warnings(record=True) as w:
            try:
                doc = docscrape.FunctionDoc(func)
            except Exception as exp:
                incorrect += [func_name + " parsing error: " + str(exp)]
                return incorrect
        if len(w):
            raise RuntimeError("Error for %s:\n%s" % (func_name, w[0]))

    param_docs = []
    for name, type_definition, param_doc in doc["Parameters"]:
        # Type hints are empty only if parameter name ended with :
        if not type_definition.strip():
            if ":" in name and name[: name.index(":")][-1:].strip():
                incorrect += [
                    func_name
                    + " There was no space between the param name and colon (%r)" % name
                ]
            elif name.rstrip().endswith(":"):
                incorrect += [
                    func_name
                    + " Parameter %r has an empty type spec. Remove the colon"
                    % (name.lstrip())
                ]

        # Create a list of parameters to compare with the parameters gotten
        # from the func signature
        if "*" not in name:
            param_docs.append(name.split(":")[0].strip("` "))

    # If one of the docstring's parameters had an error then return that
    # incorrect message
    if len(incorrect) > 0:
        return incorrect

    # Remove the parameters that should be ignored from list
    param_docs = list(filter(lambda x: x not in ignore, param_docs))

    # The following is derived from pytest, Copyright (c) 2004-2017 Holger
    # Krekel and others, Licensed under MIT License. See
    # https://github.com/pytest-dev/pytest

    message = []
    for i in range(min(len(param_docs), len(param_signature))):
        if param_signature[i] != param_docs[i]:
            message += [
                "There's a parameter name mismatch in function"
                " docstring w.r.t. function signature, at index %s"
                " diff: %r != %r" % (i, param_signature[i], param_docs[i])
            ]
            break
    if len(param_signature) > len(param_docs):
        message += [
            "Parameters in function docstring have less items w.r.t."
            " function signature, first missing item: %s"
            % param_signature[len(param_docs)]
        ]

    elif len(param_signature) < len(param_docs):
        message += [
            "Parameters in function docstring have more items w.r.t."
            " function signature, first extra item: %s"
            % param_docs[len(param_signature)]
        ]

    # If there wasn't any difference in the parameters themselves between
    # docstring and signature including having the same length then return
    # empty list
    if len(message) == 0:
        return []

    import difflib
    import pprint

    param_docs_formatted = pprint.pformat(param_docs).splitlines()
    param_signature_formatted = pprint.pformat(param_signature).splitlines()

    message += ["Full diff:"]

    message.extend(
        line.strip()
        for line in difflib.ndiff(param_signature_formatted, param_docs_formatted)
    )

    incorrect.extend(message)

    # Prepend function name
    incorrect = ["In function: " + func_name] + incorrect

    return incorrect


def assert_run_python_script(source_code, timeout=60):
    """Utility to check assertions in an independent Python subprocess.

    The script provided in the source code should return 0 and not print
    anything on stderr or stdout.

    This is a port from cloudpickle https://github.com/cloudpipe/cloudpickle

    Parameters
    ----------
    source_code : str
        The Python source code to execute.
    timeout : int, default=60
        Time in seconds before timeout.
    """
    fd, source_file = tempfile.mkstemp(suffix="_src_test_sklearn.py")
    os.close(fd)
    try:
        with open(source_file, "wb") as f:
            f.write(source_code.encode("utf-8"))
        cmd = [sys.executable, source_file]
        cwd = op.normpath(op.join(op.dirname(sklearn.__file__), ".."))
        env = os.environ.copy()
        try:
            env["PYTHONPATH"] = os.pathsep.join([cwd, env["PYTHONPATH"]])
        except KeyError:
            env["PYTHONPATH"] = cwd
        kwargs = {"cwd": cwd, "stderr": STDOUT, "env": env}
        # If coverage is running, pass the config file to the subprocess
        coverage_rc = os.environ.get("COVERAGE_PROCESS_START")
        if coverage_rc:
            kwargs["env"]["COVERAGE_PROCESS_START"] = coverage_rc

        kwargs["timeout"] = timeout
        try:
            try:
                out = check_output(cmd, **kwargs)
            except CalledProcessError as e:
                raise RuntimeError(
                    "script errored with output:\n%s" % e.output.decode("utf-8")
                )
            if out != b"":
                raise AssertionError(out.decode("utf-8"))
        except TimeoutExpired as e:
            raise RuntimeError(
                "script timeout, output so far:\n%s" % e.output.decode("utf-8")
            )
    finally:
        os.unlink(source_file)


def _convert_container(container, constructor_name, columns_name=None, dtype=None):
    """Convert a given container to a specific array-like with a dtype.

    Parameters
    ----------
    container : array-like
        The container to convert.
    constructor_name : {"list", "tuple", "array", "sparse", "dataframe", \
            "series", "index", "slice", "sparse_csr", "sparse_csc"}
        The type of the returned container.
    columns_name : index or array-like, default=None
        For pandas container supporting `columns_names`, it will affect
        specific names.
    dtype : dtype, default=None
        Force the dtype of the container. Does not apply to `"slice"`
        container.

    Returns
    -------
    converted_container
    """
    if constructor_name == "list":
        if dtype is None:
            return list(container)
        else:
            return np.asarray(container, dtype=dtype).tolist()
    elif constructor_name == "tuple":
        if dtype is None:
            return tuple(container)
        else:
            return tuple(np.asarray(container, dtype=dtype).tolist())
    elif constructor_name == "array":
        return np.asarray(container, dtype=dtype)
    elif constructor_name == "sparse":
        return sp.sparse.csr_matrix(container, dtype=dtype)
    elif constructor_name == "dataframe":
        pd = pytest.importorskip("pandas")
        return pd.DataFrame(container, columns=columns_name, dtype=dtype)
    elif constructor_name == "series":
        pd = pytest.importorskip("pandas")
        return pd.Series(container, dtype=dtype)
    elif constructor_name == "index":
        pd = pytest.importorskip("pandas")
        return pd.Index(container, dtype=dtype)
    elif constructor_name == "slice":
        return slice(container[0], container[1])
    elif constructor_name == "sparse_csr":
        return sp.sparse.csr_matrix(container, dtype=dtype)
    elif constructor_name == "sparse_csc":
        return sp.sparse.csc_matrix(container, dtype=dtype)


def raises(expected_exc_type, match=None, may_pass=False, err_msg=None):
    """Context manager to ensure exceptions are raised within a code block.

    This is similar to and inspired from pytest.raises, but supports a few
    other cases.

    This is only intended to be used in estimator_checks.py where we don't
    want to use pytest. In the rest of the code base, just use pytest.raises
    instead.

    Parameters
    ----------
    excepted_exc_type : Exception or list of Exception
        The exception that should be raised by the block. If a list, the block
        should raise one of the exceptions.
    match : str or list of str, default=None
        A regex that the exception message should match. If a list, one of
        the entries must match. If None, match isn't enforced.
    may_pass : bool, default=False
        If True, the block is allowed to not raise an exception. Useful in
        cases where some estimators may support a feature but others must
        fail with an appropriate error message. By default, the context
        manager will raise an exception if the block does not raise an
        exception.
    err_msg : str, default=None
        If the context manager fails (e.g. the block fails to raise the
        proper exception, or fails to match), then an AssertionError is
        raised with this message. By default, an AssertionError is raised
        with a default error message (depends on the kind of failure). Use
        this to indicate how users should fix their estimators to pass the
        checks.

    Attributes
    ----------
    raised_and_matched : bool
        True if an exception was raised and a match was found, False otherwise.
    """
    return _Raises(expected_exc_type, match, may_pass, err_msg)


class _Raises(contextlib.AbstractContextManager):
    # see raises() for parameters
    def __init__(self, expected_exc_type, match, may_pass, err_msg):
        self.expected_exc_types = (
            expected_exc_type
            if isinstance(expected_exc_type, Iterable)
            else [expected_exc_type]
        )
        self.matches = [match] if isinstance(match, str) else match
        self.may_pass = may_pass
        self.err_msg = err_msg
        self.raised_and_matched = False

    def __exit__(self, exc_type, exc_value, _):
        # see
        # https://docs.python.org/2.5/whatsnew/pep-343.html#SECTION000910000000000000000

        if exc_type is None:  # No exception was raised in the block
            if self.may_pass:
                return True  # CM is happy
            else:
                err_msg = self.err_msg or f"Did not raise: {self.expected_exc_types}"
                raise AssertionError(err_msg)

        if not any(
            issubclass(exc_type, expected_type)
            for expected_type in self.expected_exc_types
        ):
            if self.err_msg is not None:
                raise AssertionError(self.err_msg) from exc_value
            else:
                return False  # will re-raise the original exception

        if self.matches is not None:
            err_msg = self.err_msg or (
                "The error message should contain one of the following "
                "patterns:\n{}\nGot {}".format("\n".join(self.matches), str(exc_value))
            )
            if not any(re.search(match, str(exc_value)) for match in self.matches):
                raise AssertionError(err_msg) from exc_value
            self.raised_and_matched = True

        return True


class MinimalClassifier:
    """Minimal classifier implementation with inheriting from BaseEstimator.

    This estimator should be tested with:

    * `check_estimator` in `test_estimator_checks.py`;
    * within a `Pipeline` in `test_pipeline.py`;
    * within a `SearchCV` in `test_search.py`.
    """

    _estimator_type = "classifier"

    def __init__(self, param=None):
        self.param = param

    def get_params(self, deep=True):
        return {"param": self.param}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_, counts = np.unique(y, return_counts=True)
        self._most_frequent_class_idx = counts.argmax()
        return self

    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)
        proba_shape = (X.shape[0], self.classes_.size)
        y_proba = np.zeros(shape=proba_shape, dtype=np.float64)
        y_proba[:, self._most_frequent_class_idx] = 1.0
        return y_proba

    def predict(self, X):
        y_proba = self.predict_proba(X)
        y_pred = y_proba.argmax(axis=1)
        return self.classes_[y_pred]

    def score(self, X, y):
        from sklearn.metrics import accuracy_score

        return accuracy_score(y, self.predict(X))


class MinimalRegressor:
    """Minimal regressor implementation with inheriting from BaseEstimator.

    This estimator should be tested with:

    * `check_estimator` in `test_estimator_checks.py`;
    * within a `Pipeline` in `test_pipeline.py`;
    * within a `SearchCV` in `test_search.py`.
    """

    _estimator_type = "regressor"

    def __init__(self, param=None):
        self.param = param

    def get_params(self, deep=True):
        return {"param": self.param}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.is_fitted_ = True
        self._mean = np.mean(y)
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        return np.ones(shape=(X.shape[0],)) * self._mean

    def score(self, X, y):
        from sklearn.metrics import r2_score

        return r2_score(y, self.predict(X))


class MinimalTransformer:
    """Minimal transformer implementation with inheriting from
    BaseEstimator.

    This estimator should be tested with:

    * `check_estimator` in `test_estimator_checks.py`;
    * within a `Pipeline` in `test_pipeline.py`;
    * within a `SearchCV` in `test_search.py`.
    """

    def __init__(self, param=None):
        self.param = param

    def get_params(self, deep=True):
        return {"param": self.param}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit(self, X, y=None):
        check_array(X)
        self.is_fitted_ = True
        return self

    def transform(self, X, y=None):
        check_is_fitted(self)
        X = check_array(X)
        return X

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X, y)
