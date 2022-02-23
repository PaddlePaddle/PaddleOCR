from __future__ import annotations

import functools
from typing import (
    Any,
    Callable,
)

import numpy as np

from pandas._typing import Scalar
from pandas.compat._optional import import_optional_dependency

from pandas.core.util.numba_ import (
    NUMBA_FUNC_CACHE,
    get_jit_arguments,
    jit_user_function,
)


def generate_numba_apply_func(
    args: tuple,
    kwargs: dict[str, Any],
    func: Callable[..., Scalar],
    engine_kwargs: dict[str, bool] | None,
    name: str,
):
    """
    Generate a numba jitted apply function specified by values from engine_kwargs.

    1. jit the user's function
    2. Return a rolling apply function with the jitted function inline

    Configurations specified in engine_kwargs apply to both the user's
    function _AND_ the rolling apply function.

    Parameters
    ----------
    args : tuple
        *args to be passed into the function
    kwargs : dict
        **kwargs to be passed into the function
    func : function
        function to be applied to each window and will be JITed
    engine_kwargs : dict
        dictionary of arguments to be passed into numba.jit
    name: str
        name of the caller (Rolling/Expanding)

    Returns
    -------
    Numba function
    """
    nopython, nogil, parallel = get_jit_arguments(engine_kwargs, kwargs)

    cache_key = (func, f"{name}_apply_single")
    if cache_key in NUMBA_FUNC_CACHE:
        return NUMBA_FUNC_CACHE[cache_key]

    numba_func = jit_user_function(func, nopython, nogil, parallel)
    numba = import_optional_dependency("numba")

    @numba.jit(nopython=nopython, nogil=nogil, parallel=parallel)
    def roll_apply(
        values: np.ndarray, begin: np.ndarray, end: np.ndarray, minimum_periods: int
    ) -> np.ndarray:
        result = np.empty(len(begin))
        for i in numba.prange(len(result)):
            start = begin[i]
            stop = end[i]
            window = values[start:stop]
            count_nan = np.sum(np.isnan(window))
            if len(window) - count_nan >= minimum_periods:
                result[i] = numba_func(window, *args)
            else:
                result[i] = np.nan
        return result

    return roll_apply


def generate_numba_ewma_func(
    engine_kwargs: dict[str, bool] | None,
    com: float,
    adjust: bool,
    ignore_na: bool,
    deltas: np.ndarray,
):
    """
    Generate a numba jitted ewma function specified by values
    from engine_kwargs.

    Parameters
    ----------
    engine_kwargs : dict
        dictionary of arguments to be passed into numba.jit
    com : float
    adjust : bool
    ignore_na : bool
    deltas : numpy.ndarray

    Returns
    -------
    Numba function
    """
    nopython, nogil, parallel = get_jit_arguments(engine_kwargs)

    cache_key = (lambda x: x, "ewma")
    if cache_key in NUMBA_FUNC_CACHE:
        return NUMBA_FUNC_CACHE[cache_key]

    numba = import_optional_dependency("numba")

    @numba.jit(nopython=nopython, nogil=nogil, parallel=parallel)
    def ewma(
        values: np.ndarray,
        begin: np.ndarray,
        end: np.ndarray,
        minimum_periods: int,
    ) -> np.ndarray:
        result = np.empty(len(values))
        alpha = 1.0 / (1.0 + com)
        old_wt_factor = 1.0 - alpha
        new_wt = 1.0 if adjust else alpha

        for i in numba.prange(len(begin)):
            start = begin[i]
            stop = end[i]
            window = values[start:stop]
            sub_result = np.empty(len(window))

            weighted_avg = window[0]
            nobs = int(not np.isnan(weighted_avg))
            sub_result[0] = weighted_avg if nobs >= minimum_periods else np.nan
            old_wt = 1.0

            for j in range(1, len(window)):
                cur = window[j]
                is_observation = not np.isnan(cur)
                nobs += is_observation
                if not np.isnan(weighted_avg):

                    if is_observation or not ignore_na:

                        # note that len(deltas) = len(vals) - 1 and deltas[i] is to be
                        # used in conjunction with vals[i+1]
                        old_wt *= old_wt_factor ** deltas[start + j - 1]
                        if is_observation:

                            # avoid numerical errors on constant series
                            if weighted_avg != cur:
                                weighted_avg = (
                                    (old_wt * weighted_avg) + (new_wt * cur)
                                ) / (old_wt + new_wt)
                            if adjust:
                                old_wt += new_wt
                            else:
                                old_wt = 1.0
                elif is_observation:
                    weighted_avg = cur

                sub_result[j] = weighted_avg if nobs >= minimum_periods else np.nan

            result[start:stop] = sub_result

        return result

    return ewma


def generate_numba_table_func(
    args: tuple,
    kwargs: dict[str, Any],
    func: Callable[..., np.ndarray],
    engine_kwargs: dict[str, bool] | None,
    name: str,
):
    """
    Generate a numba jitted function to apply window calculations table-wise.

    Func will be passed a M window size x N number of columns array, and
    must return a 1 x N number of columns array. Func is intended to operate
    row-wise, but the result will be transposed for axis=1.

    1. jit the user's function
    2. Return a rolling apply function with the jitted function inline

    Parameters
    ----------
    args : tuple
        *args to be passed into the function
    kwargs : dict
        **kwargs to be passed into the function
    func : function
        function to be applied to each window and will be JITed
    engine_kwargs : dict
        dictionary of arguments to be passed into numba.jit
    name : str
        caller (Rolling/Expanding) and original method name for numba cache key

    Returns
    -------
    Numba function
    """
    nopython, nogil, parallel = get_jit_arguments(engine_kwargs, kwargs)

    cache_key = (func, f"{name}_table")
    if cache_key in NUMBA_FUNC_CACHE:
        return NUMBA_FUNC_CACHE[cache_key]

    numba_func = jit_user_function(func, nopython, nogil, parallel)
    numba = import_optional_dependency("numba")

    @numba.jit(nopython=nopython, nogil=nogil, parallel=parallel)
    def roll_table(
        values: np.ndarray, begin: np.ndarray, end: np.ndarray, minimum_periods: int
    ):
        result = np.empty(values.shape)
        min_periods_mask = np.empty(values.shape)
        for i in numba.prange(len(result)):
            start = begin[i]
            stop = end[i]
            window = values[start:stop]
            count_nan = np.sum(np.isnan(window), axis=0)
            sub_result = numba_func(window, *args)
            nan_mask = len(window) - count_nan >= minimum_periods
            min_periods_mask[i, :] = nan_mask
            result[i, :] = sub_result
        result = np.where(min_periods_mask, result, np.nan)
        return result

    return roll_table


# This function will no longer be needed once numba supports
# axis for all np.nan* agg functions
# https://github.com/numba/numba/issues/1269
@functools.lru_cache(maxsize=None)
def generate_manual_numpy_nan_agg_with_axis(nan_func):
    numba = import_optional_dependency("numba")

    @numba.jit(nopython=True, nogil=True, parallel=True)
    def nan_agg_with_axis(table):
        result = np.empty(table.shape[1])
        for i in numba.prange(table.shape[1]):
            partition = table[:, i]
            result[i] = nan_func(partition)
        return result

    return nan_agg_with_axis
