import os
import functools
import operator
from distutils.version import LooseVersion

import numpy as np
from numpy.testing import assert_
import pytest

import scipy.special as sc

__all__ = ['with_special_errors', 'assert_func_equal', 'FuncData']


#------------------------------------------------------------------------------
# Check if a module is present to be used in tests
#------------------------------------------------------------------------------

class MissingModule:
    def __init__(self, name):
        self.name = name


def check_version(module, min_ver):
    if type(module) == MissingModule:
        return pytest.mark.skip(reason="{} is not installed".format(module.name))
    return pytest.mark.skipif(LooseVersion(module.__version__) < LooseVersion(min_ver),
                              reason="{} version >= {} required".format(module.__name__, min_ver))


#------------------------------------------------------------------------------
# Enable convergence and loss of precision warnings -- turn off one by one
#------------------------------------------------------------------------------

def with_special_errors(func):
    """
    Enable special function errors (such as underflow, overflow,
    loss of precision, etc.)
    """
    @functools.wraps(func)
    def wrapper(*a, **kw):
        with sc.errstate(all='raise'):
            res = func(*a, **kw)
        return res
    return wrapper


#------------------------------------------------------------------------------
# Comparing function values at many data points at once, with helpful
# error reports
#------------------------------------------------------------------------------

def assert_func_equal(func, results, points, rtol=None, atol=None,
                      param_filter=None, knownfailure=None,
                      vectorized=True, dtype=None, nan_ok=False,
                      ignore_inf_sign=False, distinguish_nan_and_inf=True):
    if hasattr(points, 'next'):
        # it's a generator
        points = list(points)

    points = np.asarray(points)
    if points.ndim == 1:
        points = points[:,None]
    nparams = points.shape[1]

    if hasattr(results, '__name__'):
        # function
        data = points
        result_columns = None
        result_func = results
    else:
        # dataset
        data = np.c_[points, results]
        result_columns = list(range(nparams, data.shape[1]))
        result_func = None

    fdata = FuncData(func, data, list(range(nparams)),
                     result_columns=result_columns, result_func=result_func,
                     rtol=rtol, atol=atol, param_filter=param_filter,
                     knownfailure=knownfailure, nan_ok=nan_ok, vectorized=vectorized,
                     ignore_inf_sign=ignore_inf_sign,
                     distinguish_nan_and_inf=distinguish_nan_and_inf)
    fdata.check()


class FuncData:
    """
    Data set for checking a special function.

    Parameters
    ----------
    func : function
        Function to test
    data : numpy array
        columnar data to use for testing
    param_columns : int or tuple of ints
        Columns indices in which the parameters to `func` lie.
        Can be imaginary integers to indicate that the parameter
        should be cast to complex.
    result_columns : int or tuple of ints, optional
        Column indices for expected results from `func`.
    result_func : callable, optional
        Function to call to obtain results.
    rtol : float, optional
        Required relative tolerance. Default is 5*eps.
    atol : float, optional
        Required absolute tolerance. Default is 5*tiny.
    param_filter : function, or tuple of functions/Nones, optional
        Filter functions to exclude some parameter ranges.
        If omitted, no filtering is done.
    knownfailure : str, optional
        Known failure error message to raise when the test is run.
        If omitted, no exception is raised.
    nan_ok : bool, optional
        If nan is always an accepted result.
    vectorized : bool, optional
        Whether all functions passed in are vectorized.
    ignore_inf_sign : bool, optional
        Whether to ignore signs of infinities.
        (Doesn't matter for complex-valued functions.)
    distinguish_nan_and_inf : bool, optional
        If True, treat numbers which contain nans or infs as as
        equal. Sets ignore_inf_sign to be True.

    """

    def __init__(self, func, data, param_columns, result_columns=None,
                 result_func=None, rtol=None, atol=None, param_filter=None,
                 knownfailure=None, dataname=None, nan_ok=False, vectorized=True,
                 ignore_inf_sign=False, distinguish_nan_and_inf=True):
        self.func = func
        self.data = data
        self.dataname = dataname
        if not hasattr(param_columns, '__len__'):
            param_columns = (param_columns,)
        self.param_columns = tuple(param_columns)
        if result_columns is not None:
            if not hasattr(result_columns, '__len__'):
                result_columns = (result_columns,)
            self.result_columns = tuple(result_columns)
            if result_func is not None:
                raise ValueError("Only result_func or result_columns should be provided")
        elif result_func is not None:
            self.result_columns = None
        else:
            raise ValueError("Either result_func or result_columns should be provided")
        self.result_func = result_func
        self.rtol = rtol
        self.atol = atol
        if not hasattr(param_filter, '__len__'):
            param_filter = (param_filter,)
        self.param_filter = param_filter
        self.knownfailure = knownfailure
        self.nan_ok = nan_ok
        self.vectorized = vectorized
        self.ignore_inf_sign = ignore_inf_sign
        self.distinguish_nan_and_inf = distinguish_nan_and_inf
        if not self.distinguish_nan_and_inf:
            self.ignore_inf_sign = True

    def get_tolerances(self, dtype):
        if not np.issubdtype(dtype, np.inexact):
            dtype = np.dtype(float)
        info = np.finfo(dtype)
        rtol, atol = self.rtol, self.atol
        if rtol is None:
            rtol = 5*info.eps
        if atol is None:
            atol = 5*info.tiny
        return rtol, atol

    def check(self, data=None, dtype=None, dtypes=None):
        """Check the special function against the data."""
        __tracebackhide__ = operator.methodcaller(
            'errisinstance', AssertionError
        )

        if self.knownfailure:
            pytest.xfail(reason=self.knownfailure)

        if data is None:
            data = self.data

        if dtype is None:
            dtype = data.dtype
        else:
            data = data.astype(dtype)

        rtol, atol = self.get_tolerances(dtype)

        # Apply given filter functions
        if self.param_filter:
            param_mask = np.ones((data.shape[0],), np.bool_)
            for j, filter in zip(self.param_columns, self.param_filter):
                if filter:
                    param_mask &= list(filter(data[:,j]))
            data = data[param_mask]

        # Pick parameters from the correct columns
        params = []
        for idx, j in enumerate(self.param_columns):
            if np.iscomplexobj(j):
                j = int(j.imag)
                params.append(data[:,j].astype(complex))
            elif dtypes and idx < len(dtypes):
                params.append(data[:, j].astype(dtypes[idx]))
            else:
                params.append(data[:,j])

        # Helper for evaluating results
        def eval_func_at_params(func, skip_mask=None):
            if self.vectorized:
                got = func(*params)
            else:
                got = []
                for j in range(len(params[0])):
                    if skip_mask is not None and skip_mask[j]:
                        got.append(np.nan)
                        continue
                    got.append(func(*tuple([params[i][j] for i in range(len(params))])))
                got = np.asarray(got)
            if not isinstance(got, tuple):
                got = (got,)
            return got

        # Evaluate function to be tested
        got = eval_func_at_params(self.func)

        # Grab the correct results
        if self.result_columns is not None:
            # Correct results passed in with the data
            wanted = tuple([data[:,icol] for icol in self.result_columns])
        else:
            # Function producing correct results passed in
            skip_mask = None
            if self.nan_ok and len(got) == 1:
                # Don't spend time evaluating what doesn't need to be evaluated
                skip_mask = np.isnan(got[0])
            wanted = eval_func_at_params(self.result_func, skip_mask=skip_mask)

        # Check the validity of each output returned
        assert_(len(got) == len(wanted))

        for output_num, (x, y) in enumerate(zip(got, wanted)):
            if np.issubdtype(x.dtype, np.complexfloating) or self.ignore_inf_sign:
                pinf_x = np.isinf(x)
                pinf_y = np.isinf(y)
                minf_x = np.isinf(x)
                minf_y = np.isinf(y)
            else:
                pinf_x = np.isposinf(x)
                pinf_y = np.isposinf(y)
                minf_x = np.isneginf(x)
                minf_y = np.isneginf(y)
            nan_x = np.isnan(x)
            nan_y = np.isnan(y)

            with np.errstate(all='ignore'):
                abs_y = np.absolute(y)
                abs_y[~np.isfinite(abs_y)] = 0
                diff = np.absolute(x - y)
                diff[~np.isfinite(diff)] = 0

                rdiff = diff / np.absolute(y)
                rdiff[~np.isfinite(rdiff)] = 0

            tol_mask = (diff <= atol + rtol*abs_y)
            pinf_mask = (pinf_x == pinf_y)
            minf_mask = (minf_x == minf_y)

            nan_mask = (nan_x == nan_y)

            bad_j = ~(tol_mask & pinf_mask & minf_mask & nan_mask)

            point_count = bad_j.size
            if self.nan_ok:
                bad_j &= ~nan_x
                bad_j &= ~nan_y
                point_count -= (nan_x | nan_y).sum()

            if not self.distinguish_nan_and_inf and not self.nan_ok:
                # If nan's are okay we've already covered all these cases
                inf_x = np.isinf(x)
                inf_y = np.isinf(y)
                both_nonfinite = (inf_x & nan_y) | (nan_x & inf_y)
                bad_j &= ~both_nonfinite
                point_count -= both_nonfinite.sum()

            if np.any(bad_j):
                # Some bad results: inform what, where, and how bad
                msg = [""]
                msg.append("Max |adiff|: %g" % diff[bad_j].max())
                msg.append("Max |rdiff|: %g" % rdiff[bad_j].max())
                msg.append("Bad results (%d out of %d) for the following points (in output %d):"
                           % (np.sum(bad_j), point_count, output_num,))
                for j in np.nonzero(bad_j)[0]:
                    j = int(j)
                    fmt = lambda x: "%30s" % np.array2string(x[j], precision=18)
                    a = "  ".join(map(fmt, params))
                    b = "  ".join(map(fmt, got))
                    c = "  ".join(map(fmt, wanted))
                    d = fmt(rdiff)
                    msg.append("%s => %s != %s  (rdiff %s)" % (a, b, c, d))
                assert_(False, "\n".join(msg))

    def __repr__(self):
        """Pretty-printing, esp. for Nose output"""
        if np.any(list(map(np.iscomplexobj, self.param_columns))):
            is_complex = " (complex)"
        else:
            is_complex = ""
        if self.dataname:
            return "<Data for %s%s: %s>" % (self.func.__name__, is_complex,
                                            os.path.basename(self.dataname))
        else:
            return "<Data for %s%s>" % (self.func.__name__, is_complex)
