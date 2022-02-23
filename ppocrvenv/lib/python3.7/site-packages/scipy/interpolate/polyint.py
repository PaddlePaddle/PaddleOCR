import numpy as np
from scipy.special import factorial
from scipy._lib._util import _asarray_validated, float_factorial


__all__ = ["KroghInterpolator", "krogh_interpolate", "BarycentricInterpolator",
           "barycentric_interpolate", "approximate_taylor_polynomial"]


def _isscalar(x):
    """Check whether x is if a scalar type, or 0-dim"""
    return np.isscalar(x) or hasattr(x, 'shape') and x.shape == ()


class _Interpolator1D:
    """
    Common features in univariate interpolation

    Deal with input data type and interpolation axis rolling. The
    actual interpolator can assume the y-data is of shape (n, r) where
    `n` is the number of x-points, and `r` the number of variables,
    and use self.dtype as the y-data type.

    Attributes
    ----------
    _y_axis
        Axis along which the interpolation goes in the original array
    _y_extra_shape
        Additional trailing shape of the input arrays, excluding
        the interpolation axis.
    dtype
        Dtype of the y-data arrays. Can be set via _set_dtype, which
        forces it to be float or complex.

    Methods
    -------
    __call__
    _prepare_x
    _finish_y
    _reshape_yi
    _set_yi
    _set_dtype
    _evaluate

    """

    __slots__ = ('_y_axis', '_y_extra_shape', 'dtype')

    def __init__(self, xi=None, yi=None, axis=None):
        self._y_axis = axis
        self._y_extra_shape = None
        self.dtype = None
        if yi is not None:
            self._set_yi(yi, xi=xi, axis=axis)

    def __call__(self, x):
        """
        Evaluate the interpolant

        Parameters
        ----------
        x : array_like
            Points to evaluate the interpolant at.

        Returns
        -------
        y : array_like
            Interpolated values. Shape is determined by replacing
            the interpolation axis in the original array with the shape of x.

        Notes
        -----
        Input values `x` must be convertible to `float` values like `int` 
        or `float`.

        """
        x, x_shape = self._prepare_x(x)
        y = self._evaluate(x)
        return self._finish_y(y, x_shape)

    def _evaluate(self, x):
        """
        Actually evaluate the value of the interpolator.
        """
        raise NotImplementedError()

    def _prepare_x(self, x):
        """Reshape input x array to 1-D"""
        x = _asarray_validated(x, check_finite=False, as_inexact=True)
        x_shape = x.shape
        return x.ravel(), x_shape

    def _finish_y(self, y, x_shape):
        """Reshape interpolated y back to an N-D array similar to initial y"""
        y = y.reshape(x_shape + self._y_extra_shape)
        if self._y_axis != 0 and x_shape != ():
            nx = len(x_shape)
            ny = len(self._y_extra_shape)
            s = (list(range(nx, nx + self._y_axis))
                 + list(range(nx)) + list(range(nx+self._y_axis, nx+ny)))
            y = y.transpose(s)
        return y

    def _reshape_yi(self, yi, check=False):
        yi = np.rollaxis(np.asarray(yi), self._y_axis)
        if check and yi.shape[1:] != self._y_extra_shape:
            ok_shape = "%r + (N,) + %r" % (self._y_extra_shape[-self._y_axis:],
                                           self._y_extra_shape[:-self._y_axis])
            raise ValueError("Data must be of shape %s" % ok_shape)
        return yi.reshape((yi.shape[0], -1))

    def _set_yi(self, yi, xi=None, axis=None):
        if axis is None:
            axis = self._y_axis
        if axis is None:
            raise ValueError("no interpolation axis specified")

        yi = np.asarray(yi)

        shape = yi.shape
        if shape == ():
            shape = (1,)
        if xi is not None and shape[axis] != len(xi):
            raise ValueError("x and y arrays must be equal in length along "
                             "interpolation axis.")

        self._y_axis = (axis % yi.ndim)
        self._y_extra_shape = yi.shape[:self._y_axis]+yi.shape[self._y_axis+1:]
        self.dtype = None
        self._set_dtype(yi.dtype)

    def _set_dtype(self, dtype, union=False):
        if np.issubdtype(dtype, np.complexfloating) \
               or np.issubdtype(self.dtype, np.complexfloating):
            self.dtype = np.complex_
        else:
            if not union or self.dtype != np.complex_:
                self.dtype = np.float_


class _Interpolator1DWithDerivatives(_Interpolator1D):
    def derivatives(self, x, der=None):
        """
        Evaluate many derivatives of the polynomial at the point x

        Produce an array of all derivative values at the point x.

        Parameters
        ----------
        x : array_like
            Point or points at which to evaluate the derivatives
        der : int or None, optional
            How many derivatives to extract; None for all potentially
            nonzero derivatives (that is a number equal to the number
            of points). This number includes the function value as 0th
            derivative.

        Returns
        -------
        d : ndarray
            Array with derivatives; d[j] contains the jth derivative.
            Shape of d[j] is determined by replacing the interpolation
            axis in the original array with the shape of x.

        Examples
        --------
        >>> from scipy.interpolate import KroghInterpolator
        >>> KroghInterpolator([0,0,0],[1,2,3]).derivatives(0)
        array([1.0,2.0,3.0])
        >>> KroghInterpolator([0,0,0],[1,2,3]).derivatives([0,0])
        array([[1.0,1.0],
               [2.0,2.0],
               [3.0,3.0]])

        """
        x, x_shape = self._prepare_x(x)
        y = self._evaluate_derivatives(x, der)

        y = y.reshape((y.shape[0],) + x_shape + self._y_extra_shape)
        if self._y_axis != 0 and x_shape != ():
            nx = len(x_shape)
            ny = len(self._y_extra_shape)
            s = ([0] + list(range(nx+1, nx + self._y_axis+1))
                 + list(range(1, nx+1)) +
                 list(range(nx+1+self._y_axis, nx+ny+1)))
            y = y.transpose(s)
        return y

    def derivative(self, x, der=1):
        """
        Evaluate one derivative of the polynomial at the point x

        Parameters
        ----------
        x : array_like
            Point or points at which to evaluate the derivatives

        der : integer, optional
            Which derivative to extract. This number includes the
            function value as 0th derivative.

        Returns
        -------
        d : ndarray
            Derivative interpolated at the x-points. Shape of d is
            determined by replacing the interpolation axis in the
            original array with the shape of x.

        Notes
        -----
        This is computed by evaluating all derivatives up to the desired
        one (using self.derivatives()) and then discarding the rest.

        """
        x, x_shape = self._prepare_x(x)
        y = self._evaluate_derivatives(x, der+1)
        return self._finish_y(y[der], x_shape)


class KroghInterpolator(_Interpolator1DWithDerivatives):
    """
    Interpolating polynomial for a set of points.

    The polynomial passes through all the pairs (xi,yi). One may
    additionally specify a number of derivatives at each point xi;
    this is done by repeating the value xi and specifying the
    derivatives as successive yi values.

    Allows evaluation of the polynomial and all its derivatives.
    For reasons of numerical stability, this function does not compute
    the coefficients of the polynomial, although they can be obtained
    by evaluating all the derivatives.

    Parameters
    ----------
    xi : array_like, length N
        Known x-coordinates. Must be sorted in increasing order.
    yi : array_like
        Known y-coordinates. When an xi occurs two or more times in
        a row, the corresponding yi's represent derivative values.
    axis : int, optional
        Axis in the yi array corresponding to the x-coordinate values.

    Notes
    -----
    Be aware that the algorithms implemented here are not necessarily
    the most numerically stable known. Moreover, even in a world of
    exact computation, unless the x coordinates are chosen very
    carefully - Chebyshev zeros (e.g., cos(i*pi/n)) are a good choice -
    polynomial interpolation itself is a very ill-conditioned process
    due to the Runge phenomenon. In general, even with well-chosen
    x values, degrees higher than about thirty cause problems with
    numerical instability in this code.

    Based on [1]_.

    References
    ----------
    .. [1] Krogh, "Efficient Algorithms for Polynomial Interpolation
        and Numerical Differentiation", 1970.

    Examples
    --------
    To produce a polynomial that is zero at 0 and 1 and has
    derivative 2 at 0, call

    >>> from scipy.interpolate import KroghInterpolator
    >>> KroghInterpolator([0,0,1],[0,2,0])

    This constructs the quadratic 2*X**2-2*X. The derivative condition
    is indicated by the repeated zero in the xi array; the corresponding
    yi values are 0, the function value, and 2, the derivative value.

    For another example, given xi, yi, and a derivative ypi for each
    point, appropriate arrays can be constructed as:

    >>> rng = np.random.default_rng()
    >>> xi = np.linspace(0, 1, 5)
    >>> yi, ypi = rng.random((2, 5))
    >>> xi_k, yi_k = np.repeat(xi, 2), np.ravel(np.dstack((yi,ypi)))
    >>> KroghInterpolator(xi_k, yi_k)

    To produce a vector-valued polynomial, supply a higher-dimensional
    array for yi:

    >>> KroghInterpolator([0,1],[[2,3],[4,5]])

    This constructs a linear polynomial giving (2,3) at 0 and (4,5) at 1.

    """

    def __init__(self, xi, yi, axis=0):
        _Interpolator1DWithDerivatives.__init__(self, xi, yi, axis)

        self.xi = np.asarray(xi)
        self.yi = self._reshape_yi(yi)
        self.n, self.r = self.yi.shape

        c = np.zeros((self.n+1, self.r), dtype=self.dtype)
        c[0] = self.yi[0]
        Vk = np.zeros((self.n, self.r), dtype=self.dtype)
        for k in range(1, self.n):
            s = 0
            while s <= k and xi[k-s] == xi[k]:
                s += 1
            s -= 1
            Vk[0] = self.yi[k]/float_factorial(s)
            for i in range(k-s):
                if xi[i] == xi[k]:
                    raise ValueError("Elements if `xi` can't be equal.")
                if s == 0:
                    Vk[i+1] = (c[i]-Vk[i])/(xi[i]-xi[k])
                else:
                    Vk[i+1] = (Vk[i+1]-Vk[i])/(xi[i]-xi[k])
            c[k] = Vk[k-s]
        self.c = c

    def _evaluate(self, x):
        pi = 1
        p = np.zeros((len(x), self.r), dtype=self.dtype)
        p += self.c[0,np.newaxis,:]
        for k in range(1, self.n):
            w = x - self.xi[k-1]
            pi = w*pi
            p += pi[:,np.newaxis] * self.c[k]
        return p

    def _evaluate_derivatives(self, x, der=None):
        n = self.n
        r = self.r

        if der is None:
            der = self.n
        pi = np.zeros((n, len(x)))
        w = np.zeros((n, len(x)))
        pi[0] = 1
        p = np.zeros((len(x), self.r), dtype=self.dtype)
        p += self.c[0, np.newaxis, :]

        for k in range(1, n):
            w[k-1] = x - self.xi[k-1]
            pi[k] = w[k-1] * pi[k-1]
            p += pi[k, :, np.newaxis] * self.c[k]

        cn = np.zeros((max(der, n+1), len(x), r), dtype=self.dtype)
        cn[:n+1, :, :] += self.c[:n+1, np.newaxis, :]
        cn[0] = p
        for k in range(1, n):
            for i in range(1, n-k+1):
                pi[i] = w[k+i-1]*pi[i-1] + pi[i]
                cn[k] = cn[k] + pi[i, :, np.newaxis]*cn[k+i]
            cn[k] *= float_factorial(k)

        cn[n, :, :] = 0
        return cn[:der]


def krogh_interpolate(xi, yi, x, der=0, axis=0):
    """
    Convenience function for polynomial interpolation.

    See `KroghInterpolator` for more details.

    Parameters
    ----------
    xi : array_like
        Known x-coordinates.
    yi : array_like
        Known y-coordinates, of shape ``(xi.size, R)``. Interpreted as
        vectors of length R, or scalars if R=1.
    x : array_like
        Point or points at which to evaluate the derivatives.
    der : int or list, optional
        How many derivatives to extract; None for all potentially
        nonzero derivatives (that is a number equal to the number
        of points), or a list of derivatives to extract. This number
        includes the function value as 0th derivative.
    axis : int, optional
        Axis in the yi array corresponding to the x-coordinate values.

    Returns
    -------
    d : ndarray
        If the interpolator's values are R-D then the
        returned array will be the number of derivatives by N by R.
        If `x` is a scalar, the middle dimension will be dropped; if
        the `yi` are scalars then the last dimension will be dropped.

    See Also
    --------
    KroghInterpolator : Krogh interpolator

    Notes
    -----
    Construction of the interpolating polynomial is a relatively expensive
    process. If you want to evaluate it repeatedly consider using the class
    KroghInterpolator (which is what this function uses).

    Examples
    --------
    We can interpolate 2D observed data using krogh interpolation:

    >>> import matplotlib.pyplot as plt
    >>> from scipy.interpolate import krogh_interpolate
    >>> x_observed = np.linspace(0.0, 10.0, 11)
    >>> y_observed = np.sin(x_observed)
    >>> x = np.linspace(min(x_observed), max(x_observed), num=100)
    >>> y = krogh_interpolate(x_observed, y_observed, x)
    >>> plt.plot(x_observed, y_observed, "o", label="observation")
    >>> plt.plot(x, y, label="krogh interpolation")
    >>> plt.legend()
    >>> plt.show()

    """
    P = KroghInterpolator(xi, yi, axis=axis)
    if der == 0:
        return P(x)
    elif _isscalar(der):
        return P.derivative(x,der=der)
    else:
        return P.derivatives(x,der=np.amax(der)+1)[der]


def approximate_taylor_polynomial(f,x,degree,scale,order=None):
    """
    Estimate the Taylor polynomial of f at x by polynomial fitting.

    Parameters
    ----------
    f : callable
        The function whose Taylor polynomial is sought. Should accept
        a vector of `x` values.
    x : scalar
        The point at which the polynomial is to be evaluated.
    degree : int
        The degree of the Taylor polynomial
    scale : scalar
        The width of the interval to use to evaluate the Taylor polynomial.
        Function values spread over a range this wide are used to fit the
        polynomial. Must be chosen carefully.
    order : int or None, optional
        The order of the polynomial to be used in the fitting; `f` will be
        evaluated ``order+1`` times. If None, use `degree`.

    Returns
    -------
    p : poly1d instance
        The Taylor polynomial (translated to the origin, so that
        for example p(0)=f(x)).

    Notes
    -----
    The appropriate choice of "scale" is a trade-off; too large and the
    function differs from its Taylor polynomial too much to get a good
    answer, too small and round-off errors overwhelm the higher-order terms.
    The algorithm used becomes numerically unstable around order 30 even
    under ideal circumstances.

    Choosing order somewhat larger than degree may improve the higher-order
    terms.

    Examples
    --------
    We can calculate Taylor approximation polynomials of sin function with
    various degrees:

    >>> import matplotlib.pyplot as plt
    >>> from scipy.interpolate import approximate_taylor_polynomial
    >>> x = np.linspace(-10.0, 10.0, num=100)
    >>> plt.plot(x, np.sin(x), label="sin curve")
    >>> for degree in np.arange(1, 15, step=2):
    ...     sin_taylor = approximate_taylor_polynomial(np.sin, 0, degree, 1,
    ...                                                order=degree + 2)
    ...     plt.plot(x, sin_taylor(x), label=f"degree={degree}")
    >>> plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
    ...            borderaxespad=0.0, shadow=True)
    >>> plt.tight_layout()
    >>> plt.axis([-10, 10, -10, 10])
    >>> plt.show()

    """
    if order is None:
        order = degree

    n = order+1
    # Choose n points that cluster near the endpoints of the interval in
    # a way that avoids the Runge phenomenon. Ensure, by including the
    # endpoint or not as appropriate, that one point always falls at x
    # exactly.
    xs = scale*np.cos(np.linspace(0,np.pi,n,endpoint=n % 1)) + x

    P = KroghInterpolator(xs, f(xs))
    d = P.derivatives(x,der=degree+1)

    return np.poly1d((d/factorial(np.arange(degree+1)))[::-1])


class BarycentricInterpolator(_Interpolator1D):
    """The interpolating polynomial for a set of points

    Constructs a polynomial that passes through a given set of points.
    Allows evaluation of the polynomial, efficient changing of the y
    values to be interpolated, and updating by adding more x values.
    For reasons of numerical stability, this function does not compute
    the coefficients of the polynomial.

    The values yi need to be provided before the function is
    evaluated, but none of the preprocessing depends on them, so rapid
    updates are possible.

    Parameters
    ----------
    xi : array_like
        1-D array of x coordinates of the points the polynomial
        should pass through
    yi : array_like, optional
        The y coordinates of the points the polynomial should pass through.
        If None, the y values will be supplied later via the `set_y` method.
    axis : int, optional
        Axis in the yi array corresponding to the x-coordinate values.

    Notes
    -----
    This class uses a "barycentric interpolation" method that treats
    the problem as a special case of rational function interpolation.
    This algorithm is quite stable, numerically, but even in a world of
    exact computation, unless the x coordinates are chosen very
    carefully - Chebyshev zeros (e.g., cos(i*pi/n)) are a good choice -
    polynomial interpolation itself is a very ill-conditioned process
    due to the Runge phenomenon.

    Based on Berrut and Trefethen 2004, "Barycentric Lagrange Interpolation".

    """
    def __init__(self, xi, yi=None, axis=0):
        _Interpolator1D.__init__(self, xi, yi, axis)

        self.xi = np.asfarray(xi)
        self.set_yi(yi)
        self.n = len(self.xi)

        self.wi = np.zeros(self.n)
        self.wi[0] = 1
        for j in range(1, self.n):
            self.wi[:j] *= (self.xi[j]-self.xi[:j])
            self.wi[j] = np.multiply.reduce(self.xi[:j]-self.xi[j])
        self.wi **= -1

    def set_yi(self, yi, axis=None):
        """
        Update the y values to be interpolated

        The barycentric interpolation algorithm requires the calculation
        of weights, but these depend only on the xi. The yi can be changed
        at any time.

        Parameters
        ----------
        yi : array_like
            The y coordinates of the points the polynomial should pass through.
            If None, the y values will be supplied later.
        axis : int, optional
            Axis in the yi array corresponding to the x-coordinate values.

        """
        if yi is None:
            self.yi = None
            return
        self._set_yi(yi, xi=self.xi, axis=axis)
        self.yi = self._reshape_yi(yi)
        self.n, self.r = self.yi.shape

    def add_xi(self, xi, yi=None):
        """
        Add more x values to the set to be interpolated

        The barycentric interpolation algorithm allows easy updating by
        adding more points for the polynomial to pass through.

        Parameters
        ----------
        xi : array_like
            The x coordinates of the points that the polynomial should pass
            through.
        yi : array_like, optional
            The y coordinates of the points the polynomial should pass through.
            Should have shape ``(xi.size, R)``; if R > 1 then the polynomial is
            vector-valued.
            If `yi` is not given, the y values will be supplied later. `yi` should
            be given if and only if the interpolator has y values specified.

        """
        if yi is not None:
            if self.yi is None:
                raise ValueError("No previous yi value to update!")
            yi = self._reshape_yi(yi, check=True)
            self.yi = np.vstack((self.yi,yi))
        else:
            if self.yi is not None:
                raise ValueError("No update to yi provided!")
        old_n = self.n
        self.xi = np.concatenate((self.xi,xi))
        self.n = len(self.xi)
        self.wi **= -1
        old_wi = self.wi
        self.wi = np.zeros(self.n)
        self.wi[:old_n] = old_wi
        for j in range(old_n, self.n):
            self.wi[:j] *= (self.xi[j]-self.xi[:j])
            self.wi[j] = np.multiply.reduce(self.xi[:j]-self.xi[j])
        self.wi **= -1

    def __call__(self, x):
        """Evaluate the interpolating polynomial at the points x

        Parameters
        ----------
        x : array_like
            Points to evaluate the interpolant at.

        Returns
        -------
        y : array_like
            Interpolated values. Shape is determined by replacing
            the interpolation axis in the original array with the shape of x.

        Notes
        -----
        Currently the code computes an outer product between x and the
        weights, that is, it constructs an intermediate array of size
        N by len(x), where N is the degree of the polynomial.
        """
        return _Interpolator1D.__call__(self, x)

    def _evaluate(self, x):
        if x.size == 0:
            p = np.zeros((0, self.r), dtype=self.dtype)
        else:
            c = x[...,np.newaxis]-self.xi
            z = c == 0
            c[z] = 1
            c = self.wi/c
            p = np.dot(c,self.yi)/np.sum(c,axis=-1)[...,np.newaxis]
            # Now fix where x==some xi
            r = np.nonzero(z)
            if len(r) == 1:  # evaluation at a scalar
                if len(r[0]) > 0:  # equals one of the points
                    p = self.yi[r[0][0]]
            else:
                p[r[:-1]] = self.yi[r[-1]]
        return p


def barycentric_interpolate(xi, yi, x, axis=0):
    """
    Convenience function for polynomial interpolation.

    Constructs a polynomial that passes through a given set of points,
    then evaluates the polynomial. For reasons of numerical stability,
    this function does not compute the coefficients of the polynomial.

    This function uses a "barycentric interpolation" method that treats
    the problem as a special case of rational function interpolation.
    This algorithm is quite stable, numerically, but even in a world of
    exact computation, unless the `x` coordinates are chosen very
    carefully - Chebyshev zeros (e.g., cos(i*pi/n)) are a good choice -
    polynomial interpolation itself is a very ill-conditioned process
    due to the Runge phenomenon.

    Parameters
    ----------
    xi : array_like
        1-D array of x coordinates of the points the polynomial should
        pass through
    yi : array_like
        The y coordinates of the points the polynomial should pass through.
    x : scalar or array_like
        Points to evaluate the interpolator at.
    axis : int, optional
        Axis in the yi array corresponding to the x-coordinate values.

    Returns
    -------
    y : scalar or array_like
        Interpolated values. Shape is determined by replacing
        the interpolation axis in the original array with the shape of x.

    See Also
    --------
    BarycentricInterpolator : Bary centric interpolator

    Notes
    -----
    Construction of the interpolation weights is a relatively slow process.
    If you want to call this many times with the same xi (but possibly
    varying yi or x) you should use the class `BarycentricInterpolator`.
    This is what this function uses internally.

    Examples
    --------
    We can interpolate 2D observed data using barycentric interpolation:

    >>> import matplotlib.pyplot as plt
    >>> from scipy.interpolate import barycentric_interpolate
    >>> x_observed = np.linspace(0.0, 10.0, 11)
    >>> y_observed = np.sin(x_observed)
    >>> x = np.linspace(min(x_observed), max(x_observed), num=100)
    >>> y = barycentric_interpolate(x_observed, y_observed, x)
    >>> plt.plot(x_observed, y_observed, "o", label="observation")
    >>> plt.plot(x, y, label="barycentric interpolation")
    >>> plt.legend()
    >>> plt.show()

    """
    return BarycentricInterpolator(xi, yi, axis=axis)(x)
