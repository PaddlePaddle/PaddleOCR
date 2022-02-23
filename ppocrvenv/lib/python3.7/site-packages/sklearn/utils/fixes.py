"""Compatibility fixes for older version of python, numpy and scipy

If you add content to this file, please give the version of the package
at which the fix is no longer needed.
"""
# Authors: Emmanuelle Gouillart <emmanuelle.gouillart@normalesup.org>
#          Gael Varoquaux <gael.varoquaux@normalesup.org>
#          Fabian Pedregosa <fpedregosa@acm.org>
#          Lars Buitinck
#
# License: BSD 3 clause

from functools import update_wrapper
import functools

import sklearn
import numpy as np
import scipy.sparse as sp
import scipy
import scipy.stats
from scipy.sparse.linalg import lsqr as sparse_lsqr  # noqa
import threadpoolctl
from .._config import config_context, get_config
from ..externals._packaging.version import parse as parse_version


np_version = parse_version(np.__version__)
sp_version = parse_version(scipy.__version__)


if sp_version >= parse_version("1.4"):
    from scipy.sparse.linalg import lobpcg
else:
    # Backport of lobpcg functionality from scipy 1.4.0, can be removed
    # once support for sp_version < parse_version('1.4') is dropped
    # mypy error: Name 'lobpcg' already defined (possibly by an import)
    from ..externals._lobpcg import lobpcg  # type: ignore  # noqa

try:
    from scipy.optimize._linesearch import line_search_wolfe2, line_search_wolfe1
except ImportError:  # SciPy < 1.8
    from scipy.optimize.linesearch import line_search_wolfe2, line_search_wolfe1  # type: ignore  # noqa


def _object_dtype_isnan(X):
    return X != X


# TODO: replace by copy=False, when only scipy > 1.1 is supported.
def _astype_copy_false(X):
    """Returns the copy=False parameter for
    {ndarray, csr_matrix, csc_matrix}.astype when possible,
    otherwise don't specify
    """
    if sp_version >= parse_version("1.1") or not sp.issparse(X):
        return {"copy": False}
    else:
        return {}


def _joblib_parallel_args(**kwargs):
    """Set joblib.Parallel arguments in a compatible way for 0.11 and 0.12+

    For joblib 0.11 this maps both ``prefer`` and ``require`` parameters to
    a specific ``backend``.

    Parameters
    ----------

    prefer : str in {'processes', 'threads'} or None
        Soft hint to choose the default backend if no specific backend
        was selected with the parallel_backend context manager.

    require : 'sharedmem' or None
        Hard condstraint to select the backend. If set to 'sharedmem',
        the selected backend will be single-host and thread-based even
        if the user asked for a non-thread based backend with
        parallel_backend.

    See joblib.Parallel documentation for more details
    """
    import joblib

    if parse_version(joblib.__version__) >= parse_version("0.12"):
        return kwargs

    extra_args = set(kwargs.keys()).difference({"prefer", "require"})
    if extra_args:
        raise NotImplementedError(
            "unhandled arguments %s with joblib %s"
            % (list(extra_args), joblib.__version__)
        )
    args = {}
    if "prefer" in kwargs:
        prefer = kwargs["prefer"]
        if prefer not in ["threads", "processes", None]:
            raise ValueError("prefer=%s is not supported" % prefer)
        args["backend"] = {
            "threads": "threading",
            "processes": "multiprocessing",
            None: None,
        }[prefer]

    if "require" in kwargs:
        require = kwargs["require"]
        if require not in [None, "sharedmem"]:
            raise ValueError("require=%s is not supported" % require)
        if require == "sharedmem":
            args["backend"] = "threading"
    return args


class loguniform(scipy.stats.reciprocal):
    """A class supporting log-uniform random variables.

    Parameters
    ----------
    low : float
        The minimum value
    high : float
        The maximum value

    Methods
    -------
    rvs(self, size=None, random_state=None)
        Generate log-uniform random variables

    The most useful method for Scikit-learn usage is highlighted here.
    For a full list, see
    `scipy.stats.reciprocal
    <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.reciprocal.html>`_.
    This list includes all functions of ``scipy.stats`` continuous
    distributions such as ``pdf``.

    Notes
    -----
    This class generates values between ``low`` and ``high`` or

        low <= loguniform(low, high).rvs() <= high

    The logarithmic probability density function (PDF) is uniform. When
    ``x`` is a uniformly distributed random variable between 0 and 1, ``10**x``
    are random variables that are equally likely to be returned.

    This class is an alias to ``scipy.stats.reciprocal``, which uses the
    reciprocal distribution:
    https://en.wikipedia.org/wiki/Reciprocal_distribution

    Examples
    --------

    >>> from sklearn.utils.fixes import loguniform
    >>> rv = loguniform(1e-3, 1e1)
    >>> rvs = rv.rvs(random_state=42, size=1000)
    >>> rvs.min()  # doctest: +SKIP
    0.0010435856341129003
    >>> rvs.max()  # doctest: +SKIP
    9.97403052786026
    """


def _take_along_axis(arr, indices, axis):
    """Implements a simplified version of np.take_along_axis if numpy
    version < 1.15"""
    if np_version >= parse_version("1.15"):
        return np.take_along_axis(arr=arr, indices=indices, axis=axis)
    else:
        if axis is None:
            arr = arr.flatten()

        if not np.issubdtype(indices.dtype, np.intp):
            raise IndexError("`indices` must be an integer array")
        if arr.ndim != indices.ndim:
            raise ValueError(
                "`indices` and `arr` must have the same number of dimensions"
            )

        shape_ones = (1,) * indices.ndim
        dest_dims = list(range(axis)) + [None] + list(range(axis + 1, indices.ndim))

        # build a fancy index, consisting of orthogonal aranges, with the
        # requested index inserted at the right location
        fancy_index = []
        for dim, n in zip(dest_dims, arr.shape):
            if dim is None:
                fancy_index.append(indices)
            else:
                ind_shape = shape_ones[:dim] + (-1,) + shape_ones[dim + 1 :]
                fancy_index.append(np.arange(n).reshape(ind_shape))

        fancy_index = tuple(fancy_index)
        return arr[fancy_index]


# remove when https://github.com/joblib/joblib/issues/1071 is fixed
def delayed(function):
    """Decorator used to capture the arguments of a function."""

    @functools.wraps(function)
    def delayed_function(*args, **kwargs):
        return _FuncWrapper(function), args, kwargs

    return delayed_function


class _FuncWrapper:
    """ "Load the global configuration before calling the function."""

    def __init__(self, function):
        self.function = function
        self.config = get_config()
        update_wrapper(self, self.function)

    def __call__(self, *args, **kwargs):
        with config_context(**self.config):
            return self.function(*args, **kwargs)


def linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0):
    """Implements a simplified linspace function as of numpy version >= 1.16.

    As of numpy 1.16, the arguments start and stop can be array-like and
    there is an optional argument `axis`.
    For simplicity, we only allow 1d array-like to be passed to start and stop.
    See: https://github.com/numpy/numpy/pull/12388 and numpy 1.16 release
    notes about start and stop arrays for linspace logspace and geomspace.

    Returns
    -------
    out : ndarray of shape (num, n_start) or (num,)
        The output array with `n_start=start.shape[0]` columns.
    """
    if np_version < parse_version("1.16"):
        start = np.asanyarray(start) * 1.0
        stop = np.asanyarray(stop) * 1.0
        dt = np.result_type(start, stop, float(num))
        if dtype is None:
            dtype = dt

        if start.ndim == 0 == stop.ndim:
            return np.linspace(
                start=start,
                stop=stop,
                num=num,
                endpoint=endpoint,
                retstep=retstep,
                dtype=dtype,
            )

        if start.ndim != 1 or stop.ndim != 1 or start.shape != stop.shape:
            raise ValueError("start and stop must be 1d array-like of same shape.")
        n_start = start.shape[0]
        out = np.empty((num, n_start), dtype=dtype)
        step = np.empty(n_start, dtype=np.float)
        for i in range(n_start):
            out[:, i], step[i] = np.linspace(
                start=start[i],
                stop=stop[i],
                num=num,
                endpoint=endpoint,
                retstep=True,
                dtype=dtype,
            )
        if axis != 0:
            out = np.moveaxis(out, 0, axis)

        if retstep:
            return out, step
        else:
            return out
    else:
        return np.linspace(
            start=start,
            stop=stop,
            num=num,
            endpoint=endpoint,
            retstep=retstep,
            dtype=dtype,
            axis=axis,
        )


# Rename the `method` kwarg to `interpolation` for NumPy < 1.22, because
# `interpolation` kwarg was deprecated in favor of `method` in NumPy >= 1.22.
def _percentile(a, q, *, method="linear", **kwargs):
    return np.percentile(a, q, interpolation=method, **kwargs)


if np_version < parse_version("1.22"):
    percentile = _percentile
else:  # >= 1.22
    from numpy import percentile  # type: ignore  # noqa


# compatibility fix for threadpoolctl >= 3.0.0
# since version 3 it's possible to setup a global threadpool controller to avoid
# looping through all loaded shared libraries each time.
# the global controller is created during the first call to threadpoolctl.
def _get_threadpool_controller():
    if not hasattr(threadpoolctl, "ThreadpoolController"):
        return None

    if not hasattr(sklearn, "_sklearn_threadpool_controller"):
        sklearn._sklearn_threadpool_controller = threadpoolctl.ThreadpoolController()

    return sklearn._sklearn_threadpool_controller


def threadpool_limits(limits=None, user_api=None):
    controller = _get_threadpool_controller()
    if controller is not None:
        return controller.limit(limits=limits, user_api=user_api)
    else:
        return threadpoolctl.threadpool_limits(limits=limits, user_api=user_api)


threadpool_limits.__doc__ = threadpoolctl.threadpool_limits.__doc__


def threadpool_info():
    controller = _get_threadpool_controller()
    if controller is not None:
        return controller.info()
    else:
        return threadpoolctl.threadpool_info()


threadpool_info.__doc__ = threadpoolctl.threadpool_info.__doc__
