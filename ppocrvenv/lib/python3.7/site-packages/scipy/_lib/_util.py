from contextlib import contextmanager
import functools
import operator
import sys
import warnings
import numbers
from collections import namedtuple
import inspect
import math
from typing import (
    Optional,
    Union,
    TYPE_CHECKING,
    TypeVar,
)

import numpy as np

IntNumber = Union[int, np.integer]
DecimalNumber = Union[float, np.floating, np.integer]

# Since Generator was introduced in numpy 1.17, the following condition is needed for
# backward compatibility
if TYPE_CHECKING:
    SeedType = Optional[Union[IntNumber, np.random.Generator,
                              np.random.RandomState]]
    GeneratorType = TypeVar("GeneratorType", bound=Union[np.random.Generator,
                                                         np.random.RandomState])

try:
    from numpy.random import Generator as Generator
except ImportError:
    class Generator():  # type: ignore[no-redef]
        pass


def _lazywhere(cond, arrays, f, fillvalue=None, f2=None):
    """
    np.where(cond, x, fillvalue) always evaluates x even where cond is False.
    This one only evaluates f(arr1[cond], arr2[cond], ...).

    Examples
    --------
    >>> a, b = np.array([1, 2, 3, 4]), np.array([5, 6, 7, 8])
    >>> def f(a, b):
    ...     return a*b
    >>> _lazywhere(a > 2, (a, b), f, np.nan)
    array([ nan,  nan,  21.,  32.])

    Notice, it assumes that all `arrays` are of the same shape, or can be
    broadcasted together.

    """
    cond = np.asarray(cond)
    if fillvalue is None:
        if f2 is None:
            raise ValueError("One of (fillvalue, f2) must be given.")
        else:
            fillvalue = np.nan
    else:
        if f2 is not None:
            raise ValueError("Only one of (fillvalue, f2) can be given.")

    args = np.broadcast_arrays(cond, *arrays)
    cond,  arrays = args[0], args[1:]
    temp = tuple(np.extract(cond, arr) for arr in arrays)
    tcode = np.mintypecode([a.dtype.char for a in arrays])
    out = np.full(np.shape(arrays[0]), fill_value=fillvalue, dtype=tcode)
    np.place(out, cond, f(*temp))
    if f2 is not None:
        temp = tuple(np.extract(~cond, arr) for arr in arrays)
        np.place(out, ~cond, f2(*temp))

    return out


def _lazyselect(condlist, choicelist, arrays, default=0):
    """
    Mimic `np.select(condlist, choicelist)`.

    Notice, it assumes that all `arrays` are of the same shape or can be
    broadcasted together.

    All functions in `choicelist` must accept array arguments in the order
    given in `arrays` and must return an array of the same shape as broadcasted
    `arrays`.

    Examples
    --------
    >>> x = np.arange(6)
    >>> np.select([x <3, x > 3], [x**2, x**3], default=0)
    array([  0,   1,   4,   0,  64, 125])

    >>> _lazyselect([x < 3, x > 3], [lambda x: x**2, lambda x: x**3], (x,))
    array([   0.,    1.,    4.,   0.,   64.,  125.])

    >>> a = -np.ones_like(x)
    >>> _lazyselect([x < 3, x > 3],
    ...             [lambda x, a: x**2, lambda x, a: a * x**3],
    ...             (x, a), default=np.nan)
    array([   0.,    1.,    4.,   nan,  -64., -125.])

    """
    arrays = np.broadcast_arrays(*arrays)
    tcode = np.mintypecode([a.dtype.char for a in arrays])
    out = np.full(np.shape(arrays[0]), fill_value=default, dtype=tcode)
    for index in range(len(condlist)):
        func, cond = choicelist[index], condlist[index]
        if np.all(cond is False):
            continue
        cond, _ = np.broadcast_arrays(cond, arrays[0])
        temp = tuple(np.extract(cond, arr) for arr in arrays)
        np.place(out, cond, func(*temp))
    return out


def _aligned_zeros(shape, dtype=float, order="C", align=None):
    """Allocate a new ndarray with aligned memory.

    Primary use case for this currently is working around a f2py issue
    in NumPy 1.9.1, where dtype.alignment is such that np.zeros() does
    not necessarily create arrays aligned up to it.

    """
    dtype = np.dtype(dtype)
    if align is None:
        align = dtype.alignment
    if not hasattr(shape, '__len__'):
        shape = (shape,)
    size = functools.reduce(operator.mul, shape) * dtype.itemsize
    buf = np.empty(size + align + 1, np.uint8)
    offset = buf.__array_interface__['data'][0] % align
    if offset != 0:
        offset = align - offset
    # Note: slices producing 0-size arrays do not necessarily change
    # data pointer --- so we use and allocate size+1
    buf = buf[offset:offset+size+1][:-1]
    data = np.ndarray(shape, dtype, buf, order=order)
    data.fill(0)
    return data


def _prune_array(array):
    """Return an array equivalent to the input array. If the input
    array is a view of a much larger array, copy its contents to a
    newly allocated array. Otherwise, return the input unchanged.
    """
    if array.base is not None and array.size < array.base.size // 2:
        return array.copy()
    return array


def prod(iterable):
    """
    Product of a sequence of numbers.

    Faster than np.prod for short lists like array shapes, and does
    not overflow if using Python integers.
    """
    product = 1
    for x in iterable:
        product *= x
    return product


def float_factorial(n: int) -> float:
    """Compute the factorial and return as a float

    Returns infinity when result is too large for a double
    """
    return float(math.factorial(n)) if n < 171 else np.inf


class DeprecatedImport:
    """
    Deprecated import with redirection and warning.

    Examples
    --------
    Suppose you previously had in some module::

        from foo import spam

    If this has to be deprecated, do::

        spam = DeprecatedImport("foo.spam", "baz")

    to redirect users to use "baz" module instead.

    """

    def __init__(self, old_module_name, new_module_name):
        self._old_name = old_module_name
        self._new_name = new_module_name
        __import__(self._new_name)
        self._mod = sys.modules[self._new_name]

    def __dir__(self):
        return dir(self._mod)

    def __getattr__(self, name):
        warnings.warn("Module %s is deprecated, use %s instead"
                      % (self._old_name, self._new_name),
                      DeprecationWarning)
        return getattr(self._mod, name)


# copy-pasted from scikit-learn utils/validation.py
# change this to scipy.stats._qmc.check_random_state once numpy 1.16 is dropped
def check_random_state(seed):
    """Turn `seed` into a `np.random.RandomState` instance.

    Parameters
    ----------
    seed : {None, int, `numpy.random.Generator`,
            `numpy.random.RandomState`}, optional

        If `seed` is None (or `np.random`), the `numpy.random.RandomState`
        singleton is used.
        If `seed` is an int, a new ``RandomState`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` or ``RandomState`` instance then
        that instance is used.

    Returns
    -------
    seed : {`numpy.random.Generator`, `numpy.random.RandomState`}
        Random number generator.

    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    try:
        # Generator is only available in numpy >= 1.17
        if isinstance(seed, np.random.Generator):
            return seed
    except AttributeError:
        pass
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)


def _asarray_validated(a, check_finite=True,
                       sparse_ok=False, objects_ok=False, mask_ok=False,
                       as_inexact=False):
    """
    Helper function for SciPy argument validation.

    Many SciPy linear algebra functions do support arbitrary array-like
    input arguments. Examples of commonly unsupported inputs include
    matrices containing inf/nan, sparse matrix representations, and
    matrices with complicated elements.

    Parameters
    ----------
    a : array_like
        The array-like input.
    check_finite : bool, optional
        Whether to check that the input matrices contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.
        Default: True
    sparse_ok : bool, optional
        True if scipy sparse matrices are allowed.
    objects_ok : bool, optional
        True if arrays with dype('O') are allowed.
    mask_ok : bool, optional
        True if masked arrays are allowed.
    as_inexact : bool, optional
        True to convert the input array to a np.inexact dtype.

    Returns
    -------
    ret : ndarray
        The converted validated array.

    """
    if not sparse_ok:
        import scipy.sparse
        if scipy.sparse.issparse(a):
            msg = ('Sparse matrices are not supported by this function. '
                   'Perhaps one of the scipy.sparse.linalg functions '
                   'would work instead.')
            raise ValueError(msg)
    if not mask_ok:
        if np.ma.isMaskedArray(a):
            raise ValueError('masked arrays are not supported')
    toarray = np.asarray_chkfinite if check_finite else np.asarray
    a = toarray(a)
    if not objects_ok:
        if a.dtype is np.dtype('O'):
            raise ValueError('object arrays are not supported')
    if as_inexact:
        if not np.issubdtype(a.dtype, np.inexact):
            a = toarray(a, dtype=np.float_)
    return a


def _validate_int(k, name, minimum=None):
    """
    Validate a scalar integer.

    This functon can be used to validate an argument to a function
    that expects the value to be an integer.  It uses `operator.index`
    to validate the value (so, for example, k=2.0 results in a
    TypeError).

    Parameters
    ----------
    k : int
        The value to be validated.
    name : str
        The name of the parameter.
    minimum : int, optional
        An optional lower bound.
    """
    try:
        k = operator.index(k)
    except TypeError:
        raise TypeError(f'{name} must be an integer.') from None
    if minimum is not None and k < minimum:
        raise ValueError(f'{name} must be an integer not less '
                         f'than {minimum}') from None
    return k


# Add a replacement for inspect.getfullargspec()/
# The version below is borrowed from Django,
# https://github.com/django/django/pull/4846.

# Note an inconsistency between inspect.getfullargspec(func) and
# inspect.signature(func). If `func` is a bound method, the latter does *not*
# list `self` as a first argument, while the former *does*.
# Hence, cook up a common ground replacement: `getfullargspec_no_self` which
# mimics `inspect.getfullargspec` but does not list `self`.
#
# This way, the caller code does not need to know whether it uses a legacy
# .getfullargspec or a bright and shiny .signature.

FullArgSpec = namedtuple('FullArgSpec',
                         ['args', 'varargs', 'varkw', 'defaults',
                          'kwonlyargs', 'kwonlydefaults', 'annotations'])


def getfullargspec_no_self(func):
    """inspect.getfullargspec replacement using inspect.signature.

    If func is a bound method, do not list the 'self' parameter.

    Parameters
    ----------
    func : callable
        A callable to inspect

    Returns
    -------
    fullargspec : FullArgSpec(args, varargs, varkw, defaults, kwonlyargs,
                              kwonlydefaults, annotations)

        NOTE: if the first argument of `func` is self, it is *not*, I repeat
        *not*, included in fullargspec.args.
        This is done for consistency between inspect.getargspec() under
        Python 2.x, and inspect.signature() under Python 3.x.

    """
    sig = inspect.signature(func)
    args = [
        p.name for p in sig.parameters.values()
        if p.kind in [inspect.Parameter.POSITIONAL_OR_KEYWORD,
                      inspect.Parameter.POSITIONAL_ONLY]
    ]
    varargs = [
        p.name for p in sig.parameters.values()
        if p.kind == inspect.Parameter.VAR_POSITIONAL
    ]
    varargs = varargs[0] if varargs else None
    varkw = [
        p.name for p in sig.parameters.values()
        if p.kind == inspect.Parameter.VAR_KEYWORD
    ]
    varkw = varkw[0] if varkw else None
    defaults = tuple(
        p.default for p in sig.parameters.values()
        if (p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD and
            p.default is not p.empty)
    ) or None
    kwonlyargs = [
        p.name for p in sig.parameters.values()
        if p.kind == inspect.Parameter.KEYWORD_ONLY
    ]
    kwdefaults = {p.name: p.default for p in sig.parameters.values()
                  if p.kind == inspect.Parameter.KEYWORD_ONLY and
                  p.default is not p.empty}
    annotations = {p.name: p.annotation for p in sig.parameters.values()
                   if p.annotation is not p.empty}
    return FullArgSpec(args, varargs, varkw, defaults, kwonlyargs,
                       kwdefaults or None, annotations)


class MapWrapper:
    """
    Parallelisation wrapper for working with map-like callables, such as
    `multiprocessing.Pool.map`.

    Parameters
    ----------
    pool : int or map-like callable
        If `pool` is an integer, then it specifies the number of threads to
        use for parallelization. If ``int(pool) == 1``, then no parallel
        processing is used and the map builtin is used.
        If ``pool == -1``, then the pool will utilize all available CPUs.
        If `pool` is a map-like callable that follows the same
        calling sequence as the built-in map function, then this callable is
        used for parallelization.
    """
    def __init__(self, pool=1):
        self.pool = None
        self._mapfunc = map
        self._own_pool = False

        if callable(pool):
            self.pool = pool
            self._mapfunc = self.pool
        else:
            from multiprocessing import Pool
            # user supplies a number
            if int(pool) == -1:
                # use as many processors as possible
                self.pool = Pool()
                self._mapfunc = self.pool.map
                self._own_pool = True
            elif int(pool) == 1:
                pass
            elif int(pool) > 1:
                # use the number of processors requested
                self.pool = Pool(processes=int(pool))
                self._mapfunc = self.pool.map
                self._own_pool = True
            else:
                raise RuntimeError("Number of workers specified must be -1,"
                                   " an int >= 1, or an object with a 'map' "
                                   "method")

    def __enter__(self):
        return self

    def terminate(self):
        if self._own_pool:
            self.pool.terminate()

    def join(self):
        if self._own_pool:
            self.pool.join()

    def close(self):
        if self._own_pool:
            self.pool.close()

    def __exit__(self, exc_type, exc_value, traceback):
        if self._own_pool:
            self.pool.close()
            self.pool.terminate()

    def __call__(self, func, iterable):
        # only accept one iterable because that's all Pool.map accepts
        try:
            return self._mapfunc(func, iterable)
        except TypeError as e:
            # wrong number of arguments
            raise TypeError("The map-like callable must be of the"
                            " form f(func, iterable)") from e


def rng_integers(gen, low, high=None, size=None, dtype='int64',
                 endpoint=False):
    """
    Return random integers from low (inclusive) to high (exclusive), or if
    endpoint=True, low (inclusive) to high (inclusive). Replaces
    `RandomState.randint` (with endpoint=False) and
    `RandomState.random_integers` (with endpoint=True).

    Return random integers from the "discrete uniform" distribution of the
    specified dtype. If high is None (the default), then results are from
    0 to low.

    Parameters
    ----------
    gen : {None, np.random.RandomState, np.random.Generator}
        Random number generator. If None, then the np.random.RandomState
        singleton is used.
    low : int or array-like of ints
        Lowest (signed) integers to be drawn from the distribution (unless
        high=None, in which case this parameter is 0 and this value is used
        for high).
    high : int or array-like of ints
        If provided, one above the largest (signed) integer to be drawn from
        the distribution (see above for behavior if high=None). If array-like,
        must contain integer values.
    size : array-like of ints, optional
        Output shape. If the given shape is, e.g., (m, n, k), then m * n * k
        samples are drawn. Default is None, in which case a single value is
        returned.
    dtype : {str, dtype}, optional
        Desired dtype of the result. All dtypes are determined by their name,
        i.e., 'int64', 'int', etc, so byteorder is not available and a specific
        precision may have different C types depending on the platform.
        The default value is np.int_.
    endpoint : bool, optional
        If True, sample from the interval [low, high] instead of the default
        [low, high) Defaults to False.

    Returns
    -------
    out: int or ndarray of ints
        size-shaped array of random integers from the appropriate distribution,
        or a single such random int if size not provided.
    """
    if isinstance(gen, Generator):
        return gen.integers(low, high=high, size=size, dtype=dtype,
                            endpoint=endpoint)
    else:
        if gen is None:
            # default is RandomState singleton used by np.random.
            gen = np.random.mtrand._rand
        if endpoint:
            # inclusive of endpoint
            # remember that low and high can be arrays, so don't modify in
            # place
            if high is None:
                return gen.randint(low + 1, size=size, dtype=dtype)
            if high is not None:
                return gen.randint(low, high=high + 1, size=size, dtype=dtype)

        # exclusive
        return gen.randint(low, high=high, size=size, dtype=dtype)


@contextmanager
def _fixed_default_rng(seed=1638083107694713882823079058616272161):
    """Context with a fixed np.random.default_rng seed."""
    orig_fun = np.random.default_rng
    np.random.default_rng = lambda seed=seed: orig_fun(seed)
    try:
        yield
    finally:
        np.random.default_rng = orig_fun
