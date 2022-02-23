import sys
import copy
import heapq
import collections
import functools

import numpy as np

from scipy._lib._util import MapWrapper


class LRUDict(collections.OrderedDict):
    def __init__(self, max_size):
        self.__max_size = max_size

    def __setitem__(self, key, value):
        existing_key = (key in self)
        super().__setitem__(key, value)
        if existing_key:
            self.move_to_end(key)
        elif len(self) > self.__max_size:
            self.popitem(last=False)

    def update(self, other):
        # Not needed below
        raise NotImplementedError()


class SemiInfiniteFunc:
    """
    Argument transform from (start, +-oo) to (0, 1)
    """
    def __init__(self, func, start, infty):
        self._func = func
        self._start = start
        self._sgn = -1 if infty < 0 else 1

        # Overflow threshold for the 1/t**2 factor
        self._tmin = sys.float_info.min**0.5

    def get_t(self, x):
        z = self._sgn * (x - self._start) + 1
        if z == 0:
            # Can happen only if point not in range
            return np.inf
        return 1 / z

    def __call__(self, t):
        if t < self._tmin:
            return 0.0
        else:
            x = self._start + self._sgn * (1 - t) / t
            f = self._func(x)
            return self._sgn * (f / t) / t


class DoubleInfiniteFunc:
    """
    Argument transform from (-oo, oo) to (-1, 1)
    """
    def __init__(self, func):
        self._func = func

        # Overflow threshold for the 1/t**2 factor
        self._tmin = sys.float_info.min**0.5

    def get_t(self, x):
        s = -1 if x < 0 else 1
        return s / (abs(x) + 1)

    def __call__(self, t):
        if abs(t) < self._tmin:
            return 0.0
        else:
            x = (1 - abs(t)) / t
            f = self._func(x)
            return (f / t) / t


def _max_norm(x):
    return np.amax(abs(x))


def _get_sizeof(obj):
    try:
        return sys.getsizeof(obj)
    except TypeError:
        # occurs on pypy
        if hasattr(obj, '__sizeof__'):
            return int(obj.__sizeof__())
        return 64


class _Bunch:
    def __init__(self, **kwargs):
        self.__keys = kwargs.keys()
        self.__dict__.update(**kwargs)

    def __repr__(self):
        return "_Bunch({})".format(", ".join("{}={}".format(k, repr(self.__dict__[k]))
                                             for k in self.__keys))


def quad_vec(f, a, b, epsabs=1e-200, epsrel=1e-8, norm='2', cache_size=100e6, limit=10000,
             workers=1, points=None, quadrature=None, full_output=False):
    r"""Adaptive integration of a vector-valued function.

    Parameters
    ----------
    f : callable
        Vector-valued function f(x) to integrate.
    a : float
        Initial point.
    b : float
        Final point.
    epsabs : float, optional
        Absolute tolerance.
    epsrel : float, optional
        Relative tolerance.
    norm : {'max', '2'}, optional
        Vector norm to use for error estimation.
    cache_size : int, optional
        Number of bytes to use for memoization.
    workers : int or map-like callable, optional
        If `workers` is an integer, part of the computation is done in
        parallel subdivided to this many tasks (using
        :class:`python:multiprocessing.pool.Pool`).
        Supply `-1` to use all cores available to the Process.
        Alternatively, supply a map-like callable, such as
        :meth:`python:multiprocessing.pool.Pool.map` for evaluating the
        population in parallel.
        This evaluation is carried out as ``workers(func, iterable)``.
    points : list, optional
        List of additional breakpoints.
    quadrature : {'gk21', 'gk15', 'trapezoid'}, optional
        Quadrature rule to use on subintervals.
        Options: 'gk21' (Gauss-Kronrod 21-point rule),
        'gk15' (Gauss-Kronrod 15-point rule),
        'trapezoid' (composite trapezoid rule).
        Default: 'gk21' for finite intervals and 'gk15' for (semi-)infinite
    full_output : bool, optional
        Return an additional ``info`` dictionary.

    Returns
    -------
    res : {float, array-like}
        Estimate for the result
    err : float
        Error estimate for the result in the given norm
    info : dict
        Returned only when ``full_output=True``.
        Info dictionary. Is an object with the attributes:

            success : bool
                Whether integration reached target precision.
            status : int
                Indicator for convergence, success (0),
                failure (1), and failure due to rounding error (2).
            neval : int
                Number of function evaluations.
            intervals : ndarray, shape (num_intervals, 2)
                Start and end points of subdivision intervals.
            integrals : ndarray, shape (num_intervals, ...)
                Integral for each interval.
                Note that at most ``cache_size`` values are recorded,
                and the array may contains *nan* for missing items.
            errors : ndarray, shape (num_intervals,)
                Estimated integration error for each interval.

    Notes
    -----
    The algorithm mainly follows the implementation of QUADPACK's
    DQAG* algorithms, implementing global error control and adaptive
    subdivision.

    The algorithm here has some differences to the QUADPACK approach:

    Instead of subdividing one interval at a time, the algorithm
    subdivides N intervals with largest errors at once. This enables
    (partial) parallelization of the integration.

    The logic of subdividing "next largest" intervals first is then
    not implemented, and we rely on the above extension to avoid
    concentrating on "small" intervals only.

    The Wynn epsilon table extrapolation is not used (QUADPACK uses it
    for infinite intervals). This is because the algorithm here is
    supposed to work on vector-valued functions, in an user-specified
    norm, and the extension of the epsilon algorithm to this case does
    not appear to be widely agreed. For max-norm, using elementwise
    Wynn epsilon could be possible, but we do not do this here with
    the hope that the epsilon extrapolation is mainly useful in
    special cases.

    References
    ----------
    [1] R. Piessens, E. de Doncker, QUADPACK (1983).

    Examples
    --------
    We can compute integrations of a vector-valued function:

    >>> from scipy.integrate import quad_vec
    >>> import matplotlib.pyplot as plt
    >>> alpha = np.linspace(0.0, 2.0, num=30)
    >>> f = lambda x: x**alpha
    >>> x0, x1 = 0, 2
    >>> y, err = quad_vec(f, x0, x1)
    >>> plt.plot(alpha, y)
    >>> plt.xlabel(r"$\alpha$")
    >>> plt.ylabel(r"$\int_{0}^{2} x^\alpha dx$")
    >>> plt.show()

    """
    a = float(a)
    b = float(b)

    # Use simple transformations to deal with integrals over infinite
    # intervals.
    kwargs = dict(epsabs=epsabs,
                  epsrel=epsrel,
                  norm=norm,
                  cache_size=cache_size,
                  limit=limit,
                  workers=workers,
                  points=points,
                  quadrature='gk15' if quadrature is None else quadrature,
                  full_output=full_output)
    if np.isfinite(a) and np.isinf(b):
        f2 = SemiInfiniteFunc(f, start=a, infty=b)
        if points is not None:
            kwargs['points'] = tuple(f2.get_t(xp) for xp in points)
        return quad_vec(f2, 0, 1, **kwargs)
    elif np.isfinite(b) and np.isinf(a):
        f2 = SemiInfiniteFunc(f, start=b, infty=a)
        if points is not None:
            kwargs['points'] = tuple(f2.get_t(xp) for xp in points)
        res = quad_vec(f2, 0, 1, **kwargs)
        return (-res[0],) + res[1:]
    elif np.isinf(a) and np.isinf(b):
        sgn = -1 if b < a else 1

        # NB. explicitly split integral at t=0, which separates
        # the positive and negative sides
        f2 = DoubleInfiniteFunc(f)
        if points is not None:
            kwargs['points'] = (0,) + tuple(f2.get_t(xp) for xp in points)
        else:
            kwargs['points'] = (0,)

        if a != b:
            res = quad_vec(f2, -1, 1, **kwargs)
        else:
            res = quad_vec(f2, 1, 1, **kwargs)

        return (res[0]*sgn,) + res[1:]
    elif not (np.isfinite(a) and np.isfinite(b)):
        raise ValueError("invalid integration bounds a={}, b={}".format(a, b))

    norm_funcs = {
        None: _max_norm,
        'max': _max_norm,
        '2': np.linalg.norm
    }
    if callable(norm):
        norm_func = norm
    else:
        norm_func = norm_funcs[norm]


    parallel_count = 128
    min_intervals = 2

    try:
        _quadrature = {None: _quadrature_gk21,
                       'gk21': _quadrature_gk21,
                       'gk15': _quadrature_gk15,
                       'trapz': _quadrature_trapezoid,  # alias for backcompat
                       'trapezoid': _quadrature_trapezoid}[quadrature]
    except KeyError as e:
        raise ValueError("unknown quadrature {!r}".format(quadrature)) from e

    # Initial interval set
    if points is None:
        initial_intervals = [(a, b)]
    else:
        prev = a
        initial_intervals = []
        for p in sorted(points):
            p = float(p)
            if not (a < p < b) or p == prev:
                continue
            initial_intervals.append((prev, p))
            prev = p
        initial_intervals.append((prev, b))

    global_integral = None
    global_error = None
    rounding_error = None
    interval_cache = None
    intervals = []
    neval = 0

    for x1, x2 in initial_intervals:
        ig, err, rnd = _quadrature(x1, x2, f, norm_func)
        neval += _quadrature.num_eval

        if global_integral is None:
            if isinstance(ig, (float, complex)):
                # Specialize for scalars
                if norm_func in (_max_norm, np.linalg.norm):
                    norm_func = abs

            global_integral = ig
            global_error = float(err)
            rounding_error = float(rnd)

            cache_count = cache_size // _get_sizeof(ig)
            interval_cache = LRUDict(cache_count)
        else:
            global_integral += ig
            global_error += err
            rounding_error += rnd

        interval_cache[(x1, x2)] = copy.copy(ig)
        intervals.append((-err, x1, x2))

    heapq.heapify(intervals)

    CONVERGED = 0
    NOT_CONVERGED = 1
    ROUNDING_ERROR = 2
    NOT_A_NUMBER = 3

    status_msg = {
        CONVERGED: "Target precision reached.",
        NOT_CONVERGED: "Target precision not reached.",
        ROUNDING_ERROR: "Target precision could not be reached due to rounding error.",
        NOT_A_NUMBER: "Non-finite values encountered."
    }

    # Process intervals
    with MapWrapper(workers) as mapwrapper:
        ier = NOT_CONVERGED

        while intervals and len(intervals) < limit:
            # Select intervals with largest errors for subdivision
            tol = max(epsabs, epsrel*norm_func(global_integral))

            to_process = []
            err_sum = 0

            for j in range(parallel_count):
                if not intervals:
                    break

                if j > 0 and err_sum > global_error - tol/8:
                    # avoid unnecessary parallel splitting
                    break

                interval = heapq.heappop(intervals)

                neg_old_err, a, b = interval
                old_int = interval_cache.pop((a, b), None)
                to_process.append(((-neg_old_err, a, b, old_int), f, norm_func, _quadrature))
                err_sum += -neg_old_err

            # Subdivide intervals
            for dint, derr, dround_err, subint, dneval in mapwrapper(_subdivide_interval, to_process):
                neval += dneval
                global_integral += dint
                global_error += derr
                rounding_error += dround_err
                for x in subint:
                    x1, x2, ig, err = x
                    interval_cache[(x1, x2)] = ig
                    heapq.heappush(intervals, (-err, x1, x2))

            # Termination check
            if len(intervals) >= min_intervals:
                tol = max(epsabs, epsrel*norm_func(global_integral))
                if global_error < tol/8:
                    ier = CONVERGED
                    break
                if global_error < rounding_error:
                    ier = ROUNDING_ERROR
                    break

            if not (np.isfinite(global_error) and np.isfinite(rounding_error)):
                ier = NOT_A_NUMBER
                break

    res = global_integral
    err = global_error + rounding_error

    if full_output:
        res_arr = np.asarray(res)
        dummy = np.full(res_arr.shape, np.nan, dtype=res_arr.dtype)
        integrals = np.array([interval_cache.get((z[1], z[2]), dummy)
                                      for z in intervals], dtype=res_arr.dtype)
        errors = np.array([-z[0] for z in intervals])
        intervals = np.array([[z[1], z[2]] for z in intervals])

        info = _Bunch(neval=neval,
                      success=(ier == CONVERGED),
                      status=ier,
                      message=status_msg[ier],
                      intervals=intervals,
                      integrals=integrals,
                      errors=errors)
        return (res, err, info)
    else:
        return (res, err)


def _subdivide_interval(args):
    interval, f, norm_func, _quadrature = args
    old_err, a, b, old_int = interval

    c = 0.5 * (a + b)

    # Left-hand side
    if getattr(_quadrature, 'cache_size', 0) > 0:
        f = functools.lru_cache(_quadrature.cache_size)(f)

    s1, err1, round1 = _quadrature(a, c, f, norm_func)
    dneval = _quadrature.num_eval
    s2, err2, round2 = _quadrature(c, b, f, norm_func)
    dneval += _quadrature.num_eval
    if old_int is None:
        old_int, _, _ = _quadrature(a, b, f, norm_func)
        dneval += _quadrature.num_eval

    if getattr(_quadrature, 'cache_size', 0) > 0:
        dneval = f.cache_info().misses

    dint = s1 + s2 - old_int
    derr = err1 + err2 - old_err
    dround_err = round1 + round2

    subintervals = ((a, c, s1, err1), (c, b, s2, err2))
    return dint, derr, dround_err, subintervals, dneval


def _quadrature_trapezoid(x1, x2, f, norm_func):
    """
    Composite trapezoid quadrature
    """
    x3 = 0.5*(x1 + x2)
    f1 = f(x1)
    f2 = f(x2)
    f3 = f(x3)

    s2 = 0.25 * (x2 - x1) * (f1 + 2*f3 + f2)

    round_err = 0.25 * abs(x2 - x1) * (float(norm_func(f1))
                                       + 2*float(norm_func(f3))
                                       + float(norm_func(f2))) * 2e-16

    s1 = 0.5 * (x2 - x1) * (f1 + f2)
    err = 1/3 * float(norm_func(s1 - s2))
    return s2, err, round_err


_quadrature_trapezoid.cache_size = 3 * 3
_quadrature_trapezoid.num_eval = 3


def _quadrature_gk(a, b, f, norm_func, x, w, v):
    """
    Generic Gauss-Kronrod quadrature
    """

    fv = [0.0]*len(x)

    c = 0.5 * (a + b)
    h = 0.5 * (b - a)

    # Gauss-Kronrod
    s_k = 0.0
    s_k_abs = 0.0
    for i in range(len(x)):
        ff = f(c + h*x[i])
        fv[i] = ff

        vv = v[i]

        # \int f(x)
        s_k += vv * ff
        # \int |f(x)|
        s_k_abs += vv * abs(ff)

    # Gauss
    s_g = 0.0
    for i in range(len(w)):
        s_g += w[i] * fv[2*i + 1]

    # Quadrature of abs-deviation from average
    s_k_dabs = 0.0
    y0 = s_k / 2.0
    for i in range(len(x)):
        # \int |f(x) - y0|
        s_k_dabs += v[i] * abs(fv[i] - y0)

    # Use similar error estimation as quadpack
    err = float(norm_func((s_k - s_g) * h))
    dabs = float(norm_func(s_k_dabs * h))
    if dabs != 0 and err != 0:
        err = dabs * min(1.0, (200 * err / dabs)**1.5)

    eps = sys.float_info.epsilon
    round_err = float(norm_func(50 * eps * h * s_k_abs))

    if round_err > sys.float_info.min:
        err = max(err, round_err)

    return h * s_k, err, round_err


def _quadrature_gk21(a, b, f, norm_func):
    """
    Gauss-Kronrod 21 quadrature with error estimate
    """
    # Gauss-Kronrod points
    x = (0.995657163025808080735527280689003,
         0.973906528517171720077964012084452,
         0.930157491355708226001207180059508,
         0.865063366688984510732096688423493,
         0.780817726586416897063717578345042,
         0.679409568299024406234327365114874,
         0.562757134668604683339000099272694,
         0.433395394129247190799265943165784,
         0.294392862701460198131126603103866,
         0.148874338981631210884826001129720,
         0,
         -0.148874338981631210884826001129720,
         -0.294392862701460198131126603103866,
         -0.433395394129247190799265943165784,
         -0.562757134668604683339000099272694,
         -0.679409568299024406234327365114874,
         -0.780817726586416897063717578345042,
         -0.865063366688984510732096688423493,
         -0.930157491355708226001207180059508,
         -0.973906528517171720077964012084452,
         -0.995657163025808080735527280689003)

    # 10-point weights
    w = (0.066671344308688137593568809893332,
         0.149451349150580593145776339657697,
         0.219086362515982043995534934228163,
         0.269266719309996355091226921569469,
         0.295524224714752870173892994651338,
         0.295524224714752870173892994651338,
         0.269266719309996355091226921569469,
         0.219086362515982043995534934228163,
         0.149451349150580593145776339657697,
         0.066671344308688137593568809893332)

    # 21-point weights
    v = (0.011694638867371874278064396062192,
         0.032558162307964727478818972459390,
         0.054755896574351996031381300244580,
         0.075039674810919952767043140916190,
         0.093125454583697605535065465083366,
         0.109387158802297641899210590325805,
         0.123491976262065851077958109831074,
         0.134709217311473325928054001771707,
         0.142775938577060080797094273138717,
         0.147739104901338491374841515972068,
         0.149445554002916905664936468389821,
         0.147739104901338491374841515972068,
         0.142775938577060080797094273138717,
         0.134709217311473325928054001771707,
         0.123491976262065851077958109831074,
         0.109387158802297641899210590325805,
         0.093125454583697605535065465083366,
         0.075039674810919952767043140916190,
         0.054755896574351996031381300244580,
         0.032558162307964727478818972459390,
         0.011694638867371874278064396062192)

    return _quadrature_gk(a, b, f, norm_func, x, w, v)


_quadrature_gk21.num_eval = 21


def _quadrature_gk15(a, b, f, norm_func):
    """
    Gauss-Kronrod 15 quadrature with error estimate
    """
    # Gauss-Kronrod points
    x = (0.991455371120812639206854697526329,
         0.949107912342758524526189684047851,
         0.864864423359769072789712788640926,
         0.741531185599394439863864773280788,
         0.586087235467691130294144838258730,
         0.405845151377397166906606412076961,
         0.207784955007898467600689403773245,
         0.000000000000000000000000000000000,
         -0.207784955007898467600689403773245,
         -0.405845151377397166906606412076961,
         -0.586087235467691130294144838258730,
         -0.741531185599394439863864773280788,
         -0.864864423359769072789712788640926,
         -0.949107912342758524526189684047851,
         -0.991455371120812639206854697526329)

    # 7-point weights
    w = (0.129484966168869693270611432679082,
         0.279705391489276667901467771423780,
         0.381830050505118944950369775488975,
         0.417959183673469387755102040816327,
         0.381830050505118944950369775488975,
         0.279705391489276667901467771423780,
         0.129484966168869693270611432679082)

    # 15-point weights
    v = (0.022935322010529224963732008058970,
         0.063092092629978553290700663189204,
         0.104790010322250183839876322541518,
         0.140653259715525918745189590510238,
         0.169004726639267902826583426598550,
         0.190350578064785409913256402421014,
         0.204432940075298892414161999234649,
         0.209482141084727828012999174891714,
         0.204432940075298892414161999234649,
         0.190350578064785409913256402421014,
         0.169004726639267902826583426598550,
         0.140653259715525918745189590510238,
         0.104790010322250183839876322541518,
         0.063092092629978553290700663189204,
         0.022935322010529224963732008058970)

    return _quadrature_gk(a, b, f, norm_func, x, w, v)


_quadrature_gk15.num_eval = 15
