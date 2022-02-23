"""rbf - Radial basis functions for interpolation/smoothing scattered N-D data.

Written by John Travers <jtravs@gmail.com>, February 2007
Based closely on Matlab code by Alex Chirokov
Additional, large, improvements by Robert Hetland
Some additional alterations by Travis Oliphant
Interpolation with multi-dimensional target domain by Josua Sassen

Permission to use, modify, and distribute this software is given under the
terms of the SciPy (BSD style) license. See LICENSE.txt that came with
this distribution for specifics.

NO WARRANTY IS EXPRESSED OR IMPLIED. USE AT YOUR OWN RISK.

Copyright (c) 2006-2007, Robert Hetland <hetland@tamu.edu>
Copyright (c) 2007, John Travers <jtravs@gmail.com>

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

    * Redistributions of source code must retain the above copyright
       notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above
       copyright notice, this list of conditions and the following
       disclaimer in the documentation and/or other materials provided
       with the distribution.

    * Neither the name of Robert Hetland nor the names of any
       contributors may be used to endorse or promote products derived
       from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
import numpy as np

from scipy import linalg
from scipy.special import xlogy
from scipy.spatial.distance import cdist, pdist, squareform

__all__ = ['Rbf']


class Rbf:
    """
    Rbf(*args)

    A class for radial basis function interpolation of functions from
    N-D scattered data to an M-D domain.

    .. note::
        `Rbf` is legacy code, for new usage please use `RBFInterpolator`
        instead.

    Parameters
    ----------
    *args : arrays
        x, y, z, ..., d, where x, y, z, ... are the coordinates of the nodes
        and d is the array of values at the nodes
    function : str or callable, optional
        The radial basis function, based on the radius, r, given by the norm
        (default is Euclidean distance); the default is 'multiquadric'::

            'multiquadric': sqrt((r/self.epsilon)**2 + 1)
            'inverse': 1.0/sqrt((r/self.epsilon)**2 + 1)
            'gaussian': exp(-(r/self.epsilon)**2)
            'linear': r
            'cubic': r**3
            'quintic': r**5
            'thin_plate': r**2 * log(r)

        If callable, then it must take 2 arguments (self, r). The epsilon
        parameter will be available as self.epsilon. Other keyword
        arguments passed in will be available as well.

    epsilon : float, optional
        Adjustable constant for gaussian or multiquadrics functions
        - defaults to approximate average distance between nodes (which is
        a good start).
    smooth : float, optional
        Values greater than zero increase the smoothness of the
        approximation. 0 is for interpolation (default), the function will
        always go through the nodal points in this case.
    norm : str, callable, optional
        A function that returns the 'distance' between two points, with
        inputs as arrays of positions (x, y, z, ...), and an output as an
        array of distance. E.g., the default: 'euclidean', such that the result
        is a matrix of the distances from each point in ``x1`` to each point in
        ``x2``. For more options, see documentation of
        `scipy.spatial.distances.cdist`.
    mode : str, optional
        Mode of the interpolation, can be '1-D' (default) or 'N-D'. When it is
        '1-D' the data `d` will be considered as 1-D and flattened
        internally. When it is 'N-D' the data `d` is assumed to be an array of
        shape (n_samples, m), where m is the dimension of the target domain.


    Attributes
    ----------
    N : int
        The number of data points (as determined by the input arrays).
    di : ndarray
        The 1-D array of data values at each of the data coordinates `xi`.
    xi : ndarray
        The 2-D array of data coordinates.
    function : str or callable
        The radial basis function. See description under Parameters.
    epsilon : float
        Parameter used by gaussian or multiquadrics functions. See Parameters.
    smooth : float
        Smoothing parameter. See description under Parameters.
    norm : str or callable
        The distance function. See description under Parameters.
    mode : str
        Mode of the interpolation. See description under Parameters.
    nodes : ndarray
        A 1-D array of node values for the interpolation.
    A : internal property, do not use

    See Also
    --------
    RBFInterpolator

    Examples
    --------
    >>> from scipy.interpolate import Rbf
    >>> rng = np.random.default_rng()
    >>> x, y, z, d = rng.random((4, 50))
    >>> rbfi = Rbf(x, y, z, d)  # radial basis function interpolator instance
    >>> xi = yi = zi = np.linspace(0, 1, 20)
    >>> di = rbfi(xi, yi, zi)   # interpolated values
    >>> di.shape
    (20,)

    """
    # Available radial basis functions that can be selected as strings;
    # they all start with _h_ (self._init_function relies on that)
    def _h_multiquadric(self, r):
        return np.sqrt((1.0/self.epsilon*r)**2 + 1)

    def _h_inverse_multiquadric(self, r):
        return 1.0/np.sqrt((1.0/self.epsilon*r)**2 + 1)

    def _h_gaussian(self, r):
        return np.exp(-(1.0/self.epsilon*r)**2)

    def _h_linear(self, r):
        return r

    def _h_cubic(self, r):
        return r**3

    def _h_quintic(self, r):
        return r**5

    def _h_thin_plate(self, r):
        return xlogy(r**2, r)

    # Setup self._function and do smoke test on initial r
    def _init_function(self, r):
        if isinstance(self.function, str):
            self.function = self.function.lower()
            _mapped = {'inverse': 'inverse_multiquadric',
                       'inverse multiquadric': 'inverse_multiquadric',
                       'thin-plate': 'thin_plate'}
            if self.function in _mapped:
                self.function = _mapped[self.function]

            func_name = "_h_" + self.function
            if hasattr(self, func_name):
                self._function = getattr(self, func_name)
            else:
                functionlist = [x[3:] for x in dir(self)
                                if x.startswith('_h_')]
                raise ValueError("function must be a callable or one of " +
                                 ", ".join(functionlist))
            self._function = getattr(self, "_h_"+self.function)
        elif callable(self.function):
            allow_one = False
            if hasattr(self.function, 'func_code') or \
               hasattr(self.function, '__code__'):
                val = self.function
                allow_one = True
            elif hasattr(self.function, "__call__"):
                val = self.function.__call__.__func__
            else:
                raise ValueError("Cannot determine number of arguments to "
                                 "function")

            argcount = val.__code__.co_argcount
            if allow_one and argcount == 1:
                self._function = self.function
            elif argcount == 2:
                self._function = self.function.__get__(self, Rbf)
            else:
                raise ValueError("Function argument must take 1 or 2 "
                                 "arguments.")

        a0 = self._function(r)
        if a0.shape != r.shape:
            raise ValueError("Callable must take array and return array of "
                             "the same shape")
        return a0

    def __init__(self, *args, **kwargs):
        # `args` can be a variable number of arrays; we flatten them and store
        # them as a single 2-D array `xi` of shape (n_args-1, array_size),
        # plus a 1-D array `di` for the values.
        # All arrays must have the same number of elements
        self.xi = np.asarray([np.asarray(a, dtype=np.float_).flatten()
                              for a in args[:-1]])
        self.N = self.xi.shape[-1]

        self.mode = kwargs.pop('mode', '1-D')

        if self.mode == '1-D':
            self.di = np.asarray(args[-1]).flatten()
            self._target_dim = 1
        elif self.mode == 'N-D':
            self.di = np.asarray(args[-1])
            self._target_dim = self.di.shape[-1]
        else:
            raise ValueError("Mode has to be 1-D or N-D.")

        if not all([x.size == self.di.shape[0] for x in self.xi]):
            raise ValueError("All arrays must be equal length.")

        self.norm = kwargs.pop('norm', 'euclidean')
        self.epsilon = kwargs.pop('epsilon', None)
        if self.epsilon is None:
            # default epsilon is the "the average distance between nodes" based
            # on a bounding hypercube
            ximax = np.amax(self.xi, axis=1)
            ximin = np.amin(self.xi, axis=1)
            edges = ximax - ximin
            edges = edges[np.nonzero(edges)]
            self.epsilon = np.power(np.prod(edges)/self.N, 1.0/edges.size)

        self.smooth = kwargs.pop('smooth', 0.0)
        self.function = kwargs.pop('function', 'multiquadric')

        # attach anything left in kwargs to self for use by any user-callable
        # function or to save on the object returned.
        for item, value in kwargs.items():
            setattr(self, item, value)

        # Compute weights
        if self._target_dim > 1:  # If we have more than one target dimension,
            # we first factorize the matrix
            self.nodes = np.zeros((self.N, self._target_dim), dtype=self.di.dtype)
            lu, piv = linalg.lu_factor(self.A)
            for i in range(self._target_dim):
                self.nodes[:, i] = linalg.lu_solve((lu, piv), self.di[:, i])
        else:
            self.nodes = linalg.solve(self.A, self.di)

    @property
    def A(self):
        # this only exists for backwards compatibility: self.A was available
        # and, at least technically, public.
        r = squareform(pdist(self.xi.T, self.norm))  # Pairwise norm
        return self._init_function(r) - np.eye(self.N)*self.smooth

    def _call_norm(self, x1, x2):
        return cdist(x1.T, x2.T, self.norm)

    def __call__(self, *args):
        args = [np.asarray(x) for x in args]
        if not all([x.shape == y.shape for x in args for y in args]):
            raise ValueError("Array lengths must be equal")
        if self._target_dim > 1:
            shp = args[0].shape + (self._target_dim,)
        else:
            shp = args[0].shape
        xa = np.asarray([a.flatten() for a in args], dtype=np.float_)
        r = self._call_norm(xa, self.xi)
        return np.dot(self._function(r), self.nodes).reshape(shp)
