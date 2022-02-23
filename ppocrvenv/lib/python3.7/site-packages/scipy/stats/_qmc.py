"""Quasi-Monte Carlo engines and helpers."""
from __future__ import annotations

import os
import copy
import numbers
from abc import ABC, abstractmethod
import math
from typing import (
    ClassVar,
    List,
    Optional,
    overload,
    TYPE_CHECKING,
)
import warnings

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt
    from typing_extensions import Literal
    from scipy._lib._util import (
        DecimalNumber, GeneratorType, IntNumber, SeedType
    )


import scipy.stats as stats
from scipy._lib._util import rng_integers
from scipy.stats._sobol import (
    initialize_v, _cscramble, _fill_p_cumulative, _draw, _fast_forward,
    _categorize, initialize_direction_numbers, _MAXDIM, _MAXBIT
)
from scipy.stats._qmc_cy import (
    _cy_wrapper_centered_discrepancy,
    _cy_wrapper_wrap_around_discrepancy,
    _cy_wrapper_mixture_discrepancy,
    _cy_wrapper_l2_star_discrepancy,
    _cy_wrapper_update_discrepancy
)


__all__ = ['scale', 'discrepancy', 'update_discrepancy',
           'QMCEngine', 'Sobol', 'Halton', 'LatinHypercube',
           'MultinomialQMC', 'MultivariateNormalQMC']


@overload
def check_random_state(seed: Optional[IntNumber] = ...) -> np.random.Generator:
    ...

@overload
def check_random_state(seed: GeneratorType) -> GeneratorType:
    ...


# Based on scipy._lib._util.check_random_state
def check_random_state(seed=None):
    """Turn `seed` into a `numpy.random.Generator` instance.

    Parameters
    ----------
    seed : {None, int, `numpy.random.Generator`,
            `numpy.random.RandomState`}, optional

        If `seed` is None the `numpy.random.Generator` singleton is used.
        If `seed` is an int, a new ``Generator`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` or ``RandomState`` instance then
        that instance is used.

    Returns
    -------
    seed : {`numpy.random.Generator`, `numpy.random.RandomState`}
        Random number generator.

    """
    if seed is None or isinstance(seed, (numbers.Integral, np.integer)):
        if not hasattr(np.random, 'Generator'):
            # This can be removed once numpy 1.16 is dropped
            msg = ("NumPy 1.16 doesn't have Generator, use either "
                   "NumPy >= 1.17 or `seed=np.random.RandomState(seed)`")
            raise ValueError(msg)
        return np.random.default_rng(seed)
    elif isinstance(seed, np.random.RandomState):
        return seed
    elif isinstance(seed, np.random.Generator):
        # The two checks can be merged once numpy 1.16 is dropped
        return seed
    else:
        raise ValueError('%r cannot be used to seed a numpy.random.Generator'
                         ' instance' % seed)


def scale(
    sample: npt.ArrayLike,
    l_bounds: npt.ArrayLike,
    u_bounds: npt.ArrayLike,
    *,
    reverse: bool = False
) -> np.ndarray:
    r"""Sample scaling from unit hypercube to different bounds.

    To convert a sample from :math:`[0, 1)` to :math:`[a, b), b>a`,
    with :math:`a` the lower bounds and :math:`b` the upper bounds.
    The following transformation is used:

    .. math::

        (b - a) \cdot \text{sample} + a

    Parameters
    ----------
    sample : array_like (n, d)
        Sample to scale.
    l_bounds, u_bounds : array_like (d,)
        Lower and upper bounds (resp. :math:`a`, :math:`b`) of transformed
        data. If `reverse` is True, range of the original data to transform
        to the unit hypercube.
    reverse : bool, optional
        Reverse the transformation from different bounds to the unit hypercube.
        Default is False.

    Returns
    -------
    sample : array_like (n, d)
        Scaled sample.

    Examples
    --------
    Transform 3 samples in the unit hypercube to bounds:

    >>> from scipy.stats import qmc
    >>> l_bounds = [-2, 0]
    >>> u_bounds = [6, 5]
    >>> sample = [[0.5 , 0.75],
    ...           [0.5 , 0.5],
    ...           [0.75, 0.25]]
    >>> sample_scaled = qmc.scale(sample, l_bounds, u_bounds)
    >>> sample_scaled
    array([[2.  , 3.75],
           [2.  , 2.5 ],
           [4.  , 1.25]])

    And convert back to the unit hypercube:

    >>> sample_ = qmc.scale(sample_scaled, l_bounds, u_bounds, reverse=True)
    >>> sample_
    array([[0.5 , 0.75],
           [0.5 , 0.5 ],
           [0.75, 0.25]])

    """
    sample = np.asarray(sample)
    lower = np.atleast_1d(l_bounds)
    upper = np.atleast_1d(u_bounds)

    # Checking bounds and sample
    if not sample.ndim == 2:
        raise ValueError('Sample is not a 2D array')

    lower, upper = np.broadcast_arrays(lower, upper)

    if not np.all(lower < upper):
        raise ValueError('Bounds are not consistent a < b')

    if len(lower) != sample.shape[1]:
        raise ValueError('Sample dimension is different than bounds dimension')

    if not reverse:
        # Checking that sample is within the hypercube
        if not (np.all(sample >= 0) and np.all(sample <= 1)):
            raise ValueError('Sample is not in unit hypercube')

        return sample * (upper - lower) + lower
    else:
        # Checking that sample is within the bounds
        if not (np.all(sample >= lower) and np.all(sample <= upper)):
            raise ValueError('Sample is out of bounds')

        return (sample - lower) / (upper - lower)


def discrepancy(
        sample: npt.ArrayLike,
        *,
        iterative: bool = False,
        method: Literal["CD", "WD", "MD", "L2-star"] = "CD",
        workers: IntNumber = 1) -> float:
    """Discrepancy of a given sample.

    Parameters
    ----------
    sample : array_like (n, d)
        The sample to compute the discrepancy from.
    iterative : bool, optional
        Must be False if not using it for updating the discrepancy.
        Default is False. Refer to the notes for more details.
    method : str, optional
        Type of discrepancy, can be ``CD``, ``WD``, ``MD`` or ``L2-star``.
        Refer to the notes for more details. Default is ``CD``.
    workers : int, optional
        Number of workers to use for parallel processing. If -1 is given all
        CPU threads are used. Default is 1.

    Returns
    -------
    discrepancy : float
        Discrepancy.

    Notes
    -----
    The discrepancy is a uniformity criterion used to assess the space filling
    of a number of samples in a hypercube. A discrepancy quantifies the
    distance between the continuous uniform distribution on a hypercube and the
    discrete uniform distribution on :math:`n` distinct sample points.

    The lower the value is, the better the coverage of the parameter space is.

    For a collection of subsets of the hypercube, the discrepancy is the
    difference between the fraction of sample points in one of those
    subsets and the volume of that subset. There are different definitions of
    discrepancy corresponding to different collections of subsets. Some
    versions take a root mean square difference over subsets instead of
    a maximum.

    A measure of uniformity is reasonable if it satisfies the following
    criteria [1]_:

    1. It is invariant under permuting factors and/or runs.
    2. It is invariant under rotation of the coordinates.
    3. It can measure not only uniformity of the sample over the hypercube,
       but also the projection uniformity of the sample over non-empty
       subset of lower dimension hypercubes.
    4. There is some reasonable geometric meaning.
    5. It is easy to compute.
    6. It satisfies the Koksma-Hlawka-like inequality.
    7. It is consistent with other criteria in experimental design.

    Four methods are available:

    * ``CD``: Centered Discrepancy - subspace involves a corner of the
      hypercube
    * ``WD``: Wrap-around Discrepancy - subspace can wrap around bounds
    * ``MD``: Mixture Discrepancy - mix between CD/WD covering more criteria
    * ``L2-star``: L2-star discrepancy - like CD BUT variant to rotation

    See [2]_ for precise definitions of each method.

    Lastly, using ``iterative=True``, it is possible to compute the
    discrepancy as if we had :math:`n+1` samples. This is useful if we want
    to add a point to a sampling and check the candidate which would give the
    lowest discrepancy. Then you could just update the discrepancy with
    each candidate using `update_discrepancy`. This method is faster than
    computing the discrepancy for a large number of candidates.

    References
    ----------
    .. [1] Fang et al. "Design and modeling for computer experiments".
       Computer Science and Data Analysis Series, 2006.
    .. [2] Zhou Y.-D. et al. Mixture discrepancy for quasi-random point sets.
       Journal of Complexity, 29 (3-4) , pp. 283-301, 2013.
    .. [3] T. T. Warnock. "Computational investigations of low discrepancy
       point sets". Applications of Number Theory to Numerical
       Analysis, Academic Press, pp. 319-343, 1972.

    Examples
    --------
    Calculate the quality of the sample using the discrepancy:

    >>> from scipy.stats import qmc
    >>> space = np.array([[1, 3], [2, 6], [3, 2], [4, 5], [5, 1], [6, 4]])
    >>> l_bounds = [0.5, 0.5]
    >>> u_bounds = [6.5, 6.5]
    >>> space = qmc.scale(space, l_bounds, u_bounds, reverse=True)
    >>> space
    array([[0.08333333, 0.41666667],
           [0.25      , 0.91666667],
           [0.41666667, 0.25      ],
           [0.58333333, 0.75      ],
           [0.75      , 0.08333333],
           [0.91666667, 0.58333333]])
    >>> qmc.discrepancy(space)
    0.008142039609053464

    We can also compute iteratively the ``CD`` discrepancy by using
    ``iterative=True``.

    >>> disc_init = qmc.discrepancy(space[:-1], iterative=True)
    >>> disc_init
    0.04769081147119336
    >>> qmc.update_discrepancy(space[-1], space[:-1], disc_init)
    0.008142039609053513

    """
    sample = np.asarray(sample, dtype=np.float64, order="C")

    # Checking that sample is within the hypercube and 2D
    if not sample.ndim == 2:
        raise ValueError("Sample is not a 2D array")

    if not (np.all(sample >= 0) and np.all(sample <= 1)):
        raise ValueError("Sample is not in unit hypercube")

    workers = int(workers)
    if workers == -1:
        workers = os.cpu_count()  # type: ignore[assignment]
        if workers is None:
            raise NotImplementedError(
                "Cannot determine the number of cpus using os.cpu_count(), "
                "cannot use -1 for the number of workers"
            )
    elif workers <= 0:
        raise ValueError(f"Invalid number of workers: {workers}, must be -1 "
                         "or > 0")

    methods = {
        "CD": _cy_wrapper_centered_discrepancy,
        "WD": _cy_wrapper_wrap_around_discrepancy,
        "MD": _cy_wrapper_mixture_discrepancy,
        "L2-star": _cy_wrapper_l2_star_discrepancy,
    }

    if method in methods:
        return methods[method](sample, iterative, workers=workers)
    else:
        raise ValueError(f"{method!r} is not a valid method. It must be one of"
                         f" {set(methods)!r}")


def update_discrepancy(
        x_new: npt.ArrayLike,
        sample: npt.ArrayLike,
        initial_disc: DecimalNumber) -> float:
    """Update the centered discrepancy with a new sample.

    Parameters
    ----------
    x_new : array_like (1, d)
        The new sample to add in `sample`.
    sample : array_like (n, d)
        The initial sample.
    initial_disc : float
        Centered discrepancy of the `sample`.

    Returns
    -------
    discrepancy : float
        Centered discrepancy of the sample composed of `x_new` and `sample`.

    Examples
    --------
    We can also compute iteratively the discrepancy by using
    ``iterative=True``.

    >>> from scipy.stats import qmc
    >>> space = np.array([[1, 3], [2, 6], [3, 2], [4, 5], [5, 1], [6, 4]])
    >>> l_bounds = [0.5, 0.5]
    >>> u_bounds = [6.5, 6.5]
    >>> space = qmc.scale(space, l_bounds, u_bounds, reverse=True)
    >>> disc_init = qmc.discrepancy(space[:-1], iterative=True)
    >>> disc_init
    0.04769081147119336
    >>> qmc.update_discrepancy(space[-1], space[:-1], disc_init)
    0.008142039609053513

    """
    sample = np.asarray(sample, dtype=np.float64, order="C")
    x_new = np.asarray(x_new, dtype=np.float64, order="C")

    # Checking that sample is within the hypercube and 2D
    if not sample.ndim == 2:
        raise ValueError('Sample is not a 2D array')

    if not (np.all(sample >= 0) and np.all(sample <= 1)):
        raise ValueError('Sample is not in unit hypercube')

    # Checking that x_new is within the hypercube and 1D
    if not x_new.ndim == 1:
        raise ValueError('x_new is not a 1D array')

    if not (np.all(x_new >= 0) and np.all(x_new <= 1)):
        raise ValueError('x_new is not in unit hypercube')

    if x_new.shape[0] != sample.shape[1]:
        raise ValueError("x_new and sample must be broadcastable")

    return _cy_wrapper_update_discrepancy(x_new, sample, initial_disc)


def primes_from_2_to(n: int) -> np.ndarray:
    """Prime numbers from 2 to *n*.

    Parameters
    ----------
    n : int
        Sup bound with ``n >= 6``.

    Returns
    -------
    primes : list(int)
        Primes in ``2 <= p < n``.

    Notes
    -----
    Taken from [1]_ by P.T. Roy, written consent given on 23.04.2021
    by the original author, Bruno Astrolino, for free use in SciPy under
    the 3-clause BSD.

    References
    ----------
    .. [1] `StackOverflow <https://stackoverflow.com/questions/2068372>`_.

    """
    sieve = np.ones(n // 3 + (n % 6 == 2), dtype=bool)
    for i in range(1, int(n ** 0.5) // 3 + 1):
        k = 3 * i + 1 | 1
        sieve[k * k // 3::2 * k] = False
        sieve[k * (k - 2 * (i & 1) + 4) // 3::2 * k] = False
    return np.r_[2, 3, ((3 * np.nonzero(sieve)[0][1:] + 1) | 1)]


def n_primes(n: IntNumber) -> List[int]:
    """List of the n-first prime numbers.

    Parameters
    ----------
    n : int
        Number of prime numbers wanted.

    Returns
    -------
    primes : list(int)
        List of primes.

    """
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59,
              61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127,
              131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193,
              197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269,
              271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349,
              353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431,
              433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503,
              509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599,
              601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673,
              677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761,
              769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857,
              859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947,
              953, 967, 971, 977, 983, 991, 997][:n]  # type: ignore[misc]

    if len(primes) < n:
        big_number = 2000
        while 'Not enough primes':
            primes = primes_from_2_to(big_number)[:n]  # type: ignore[misc]
            if len(primes) == n:
                break
            big_number += 1000

    return primes


def van_der_corput(
        n: IntNumber,
        base: IntNumber = 2,
        *,
        start_index: IntNumber = 0,
        scramble: bool = False,
        seed: SeedType = None) -> np.ndarray:
    """Van der Corput sequence.

    Pseudo-random number generator based on a b-adic expansion.

    Scrambling uses permutations of the remainders (see [1]_). Multiple
    permutations are applied to construct a point. The sequence of
    permutations has to be the same for all points of the sequence.

    Parameters
    ----------
    n : int
        Number of element of the sequence.
    base : int, optional
        Base of the sequence. Default is 2.
    start_index : int, optional
        Index to start the sequence from. Default is 0.
    scramble : bool, optional
        If True, use Owen scrambling. Otherwise no scrambling is done.
        Default is True.
    seed : {None, int, `numpy.random.Generator`}, optional
        If `seed` is None the `numpy.random.Generator` singleton is used.
        If `seed` is an int, a new ``Generator`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` instance then that instance is
        used.

    Returns
    -------
    sequence : list (n,)
        Sequence of Van der Corput.

    References
    ----------
    .. [1] A. B. Owen. "A randomized Halton algorithm in R",
       arXiv:1706.02808, 2017.

    """
    rng = check_random_state(seed)
    sequence = np.zeros(n)

    quotient = np.arange(start_index, start_index + n)
    b2r = 1 / base

    while (1 - b2r) < 1:
        remainder = quotient % base

        if scramble:
            # permutation must be the same for all points of the sequence
            perm = rng.permutation(base)
            remainder = perm[np.array(remainder).astype(int)]

        sequence += remainder * b2r
        b2r /= base
        quotient = (quotient - remainder) / base

    return sequence


class QMCEngine(ABC):
    """A generic Quasi-Monte Carlo sampler class meant for subclassing.

    QMCEngine is a base class to construct a specific Quasi-Monte Carlo
    sampler. It cannot be used directly as a sampler.

    Parameters
    ----------
    d : int
        Dimension of the parameter space.
    seed : {None, int, `numpy.random.Generator`}, optional
        If `seed` is None the `numpy.random.Generator` singleton is used.
        If `seed` is an int, a new ``Generator`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` instance then that instance is
        used.

    Notes
    -----
    By convention samples are distributed over the half-open interval
    ``[0, 1)``. Instances of the class can access the attributes: ``d`` for
    the dimension; and ``rng`` for the random number generator (used for the
    ``seed``).

    **Subclassing**

    When subclassing `QMCEngine` to create a new sampler,  ``__init__`` and
    ``random`` must be redefined.

    * ``__init__(d, seed=None)``: at least fix the dimension. If the sampler
      does not take advantage of a ``seed`` (deterministic methods like
      Halton), this parameter can be omitted.
    * ``random(n)``: draw ``n`` from the engine and increase the counter
      ``num_generated`` by ``n``.

    Optionally, two other methods can be overwritten by subclasses:

    * ``reset``: Reset the engine to it's original state.
    * ``fast_forward``: If the sequence is deterministic (like Halton
      sequence), then ``fast_forward(n)`` is skipping the ``n`` first draw.

    Examples
    --------
    To create a random sampler based on ``np.random.random``, we would do the
    following:

    >>> from scipy.stats import qmc
    >>> class RandomEngine(qmc.QMCEngine):
    ...     def __init__(self, d, seed=None):
    ...         super().__init__(d=d, seed=seed)
    ...
    ...
    ...     def random(self, n=1):
    ...         self.num_generated += n
    ...         return self.rng.random((n, self.d))
    ...
    ...
    ...     def reset(self):
    ...         super().__init__(d=self.d, seed=self.rng_seed)
    ...         return self
    ...
    ...
    ...     def fast_forward(self, n):
    ...         self.random(n)
    ...         return self

    After subclassing `QMCEngine` to define the sampling strategy we want to
    use, we can create an instance to sample from.

    >>> engine = RandomEngine(2)
    >>> engine.random(5)
    array([[0.22733602, 0.31675834],  # random
           [0.79736546, 0.67625467],
           [0.39110955, 0.33281393],
           [0.59830875, 0.18673419],
           [0.67275604, 0.94180287]])

    We can also reset the state of the generator and resample again.

    >>> _ = engine.reset()
    >>> engine.random(5)
    array([[0.22733602, 0.31675834],  # random
           [0.79736546, 0.67625467],
           [0.39110955, 0.33281393],
           [0.59830875, 0.18673419],
           [0.67275604, 0.94180287]])

    """

    @abstractmethod
    def __init__(
            self,
            d: IntNumber,
            *,
            seed: SeedType = None
    ) -> None:
        if not np.issubdtype(type(d), np.integer):
            raise ValueError('d must be an integer value')

        self.d = d
        self.rng = check_random_state(seed)
        self.rng_seed = copy.deepcopy(seed)
        self.num_generated = 0

    @abstractmethod
    def random(self, n: IntNumber = 1) -> np.ndarray:
        """Draw `n` in the half-open interval ``[0, 1)``.

        Parameters
        ----------
        n : int, optional
            Number of samples to generate in the parameter space.
            Default is 1.

        Returns
        -------
        sample : array_like (n, d)
            QMC sample.

        """
        # self.num_generated += n

    def reset(self) -> QMCEngine:
        """Reset the engine to base state.

        Returns
        -------
        engine : QMCEngine
            Engine reset to its base state.

        """
        seed = copy.deepcopy(self.rng_seed)
        self.rng = check_random_state(seed)
        self.num_generated = 0
        return self

    def fast_forward(self, n: IntNumber) -> QMCEngine:
        """Fast-forward the sequence by `n` positions.

        Parameters
        ----------
        n : int
            Number of points to skip in the sequence.

        Returns
        -------
        engine : QMCEngine
            Engine reset to its base state.

        """
        self.random(n=n)
        return self


class Halton(QMCEngine):
    """Halton sequence.

    Pseudo-random number generator that generalize the Van der Corput sequence
    for multiple dimensions. The Halton sequence uses the base-two Van der
    Corput sequence for the first dimension, base-three for its second and
    base-:math:`n` for its n-dimension.

    Parameters
    ----------
    d : int
        Dimension of the parameter space.
    scramble : bool, optional
        If True, use Owen scrambling. Otherwise no scrambling is done.
        Default is True.
    seed : {None, int, `numpy.random.Generator`}, optional
        If `seed` is None the `numpy.random.Generator` singleton is used.
        If `seed` is an int, a new ``Generator`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` instance then that instance is
        used.

    Notes
    -----
    The Halton sequence has severe striping artifacts for even modestly
    large dimensions. These can be ameliorated by scrambling. Scrambling
    also supports replication-based error estimates and extends
    applicabiltiy to unbounded integrands.

    References
    ----------
    .. [1] Halton, "On the efficiency of certain quasi-random sequences of
       points in evaluating multi-dimensional integrals", Numerische
       Mathematik, 1960.
    .. [2] A. B. Owen. "A randomized Halton algorithm in R",
       arXiv:1706.02808, 2017.

    Examples
    --------
    Generate samples from a low discrepancy sequence of Halton.

    >>> from scipy.stats import qmc
    >>> sampler = qmc.Halton(d=2, scramble=False)
    >>> sample = sampler.random(n=5)
    >>> sample
    array([[0.        , 0.        ],
           [0.5       , 0.33333333],
           [0.25      , 0.66666667],
           [0.75      , 0.11111111],
           [0.125     , 0.44444444]])

    Compute the quality of the sample using the discrepancy criterion.

    >>> qmc.discrepancy(sample)
    0.088893711419753

    If some wants to continue an existing design, extra points can be obtained
    by calling again `random`. Alternatively, you can skip some points like:

    >>> _ = sampler.fast_forward(5)
    >>> sample_continued = sampler.random(n=5)
    >>> sample_continued
    array([[0.3125    , 0.37037037],
           [0.8125    , 0.7037037 ],
           [0.1875    , 0.14814815],
           [0.6875    , 0.48148148],
           [0.4375    , 0.81481481]])

    Finally, samples can be scaled to bounds.

    >>> l_bounds = [0, 2]
    >>> u_bounds = [10, 5]
    >>> qmc.scale(sample_continued, l_bounds, u_bounds)
    array([[3.125     , 3.11111111],
           [8.125     , 4.11111111],
           [1.875     , 2.44444444],
           [6.875     , 3.44444444],
           [4.375     , 4.44444444]])

    """

    def __init__(
            self, d: IntNumber, *, scramble: bool = True,
            seed: SeedType = None
    ) -> None:
        super().__init__(d=d, seed=seed)
        self.seed = seed
        self.base = n_primes(d)
        self.scramble = scramble

    def random(self, n: IntNumber = 1) -> np.ndarray:
        """Draw `n` in the half-open interval ``[0, 1)``.

        Parameters
        ----------
        n : int, optional
            Number of samples to generate in the parameter space. Default is 1.

        Returns
        -------
        sample : array_like (n, d)
            QMC sample.

        """
        # Generate a sample using a Van der Corput sequence per dimension.
        # important to have ``type(bdim) == int`` for performance reason
        sample = [van_der_corput(n, int(bdim), start_index=self.num_generated,
                                 scramble=self.scramble,
                                 seed=copy.deepcopy(self.seed))
                  for bdim in self.base]

        self.num_generated += n
        return np.array(sample).T.reshape(n, self.d)


class LatinHypercube(QMCEngine):
    """Latin hypercube sampling (LHS).

    A Latin hypercube sample [1]_ generates :math:`n` points in
    :math:`[0,1)^{d}`. Each univariate marginal distribution is stratified,
    placing exactly one point in :math:`[j/n, (j+1)/n)` for
    :math:`j=0,1,...,n-1`. They are still applicable when :math:`n << d`.
    LHS is extremely effective on integrands that are nearly additive [2]_.
    LHS on :math:`n` points never has more variance than plain MC on
    :math:`n-1` points [3]_. There is a central limit theorem for LHS [4]_,
    but not necessarily for optimized LHS.

    Parameters
    ----------
    d : int
        Dimension of the parameter space.
    centered : bool, optional
        Center the point within the multi-dimensional grid. Default is False.
    seed : {None, int, `numpy.random.Generator`}, optional
        If `seed` is None the `numpy.random.Generator` singleton is used.
        If `seed` is an int, a new ``Generator`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` instance then that instance is
        used.

    References
    ----------
    .. [1] Mckay et al., "A Comparison of Three Methods for Selecting Values
       of Input Variables in the Analysis of Output from a Computer Code",
       Technometrics, 1979.
    .. [2] M. Stein, "Large sample properties of simulations using Latin
       hypercube sampling." Technometrics 29, no. 2: 143-151, 1987.
    .. [3] A. B. Owen, "Monte Carlo variance of scrambled net quadrature."
       SIAM Journal on Numerical Analysis 34, no. 5: 1884-1910, 1997
    .. [4]  Loh, W.-L. "On Latin hypercube sampling." The annals of statistics
       24, no. 5: 2058-2080, 1996.

    Examples
    --------
    Generate samples from a Latin hypercube generator.

    >>> from scipy.stats import qmc
    >>> sampler = qmc.LatinHypercube(d=2)
    >>> sample = sampler.random(n=5)
    >>> sample
    array([[0.1545328 , 0.53664833],  # random
           [0.84052691, 0.06474907],
           [0.52177809, 0.93343721],
           [0.68033825, 0.36265316],
           [0.26544879, 0.61163943]])

    Compute the quality of the sample using the discrepancy criterion.

    >>> qmc.discrepancy(sample)
    0.019558034794794565  # random

    Finally, samples can be scaled to bounds.

    >>> l_bounds = [0, 2]
    >>> u_bounds = [10, 5]
    >>> qmc.scale(sample, l_bounds, u_bounds)
    array([[1.54532796, 3.609945  ],  # random
           [8.40526909, 2.1942472 ],
           [5.2177809 , 4.80031164],
           [6.80338249, 3.08795949],
           [2.65448791, 3.83491828]])

    """

    def __init__(
        self, d: IntNumber, *, centered: bool = False,
        seed: SeedType = None
    ) -> None:
        super().__init__(d=d, seed=seed)
        self.centered = centered

    def random(self, n: IntNumber = 1) -> np.ndarray:
        """Draw `n` in the half-open interval ``[0, 1)``.

        Parameters
        ----------
        n : int, optional
            Number of samples to generate in the parameter space. Default is 1.

        Returns
        -------
        sample : array_like (n, d)
            LHS sample.

        """
        if self.centered:
            samples = 0.5
        else:
            samples = self.rng.uniform(size=(n, self.d))  # type: ignore[assignment]

        perms = np.tile(np.arange(1, n + 1), (self.d, 1))
        for i in range(self.d):  # type: ignore[arg-type]
            self.rng.shuffle(perms[i, :])
        perms = perms.T

        samples = (perms - samples) / n
        self.num_generated += n
        return samples  # type: ignore[return-value]


class Sobol(QMCEngine):
    """Engine for generating (scrambled) Sobol' sequences.

    Sobol' sequences are low-discrepancy, quasi-random numbers. Points
    can be drawn using two methods:

    * `random_base2`: safely draw :math:`n=2^m` points. This method
      guarantees the balance properties of the sequence.
    * `random`: draw an arbitrary number of points from the
      sequence. See warning below.

    Parameters
    ----------
    d : int
        Dimensionality of the sequence. Max dimensionality is 21201.
    scramble : bool, optional
        If True, use Owen scrambling. Otherwise no scrambling is done.
        Default is True.
    seed : {None, int, `numpy.random.Generator`}, optional
        If `seed` is None the `numpy.random.Generator` singleton is used.
        If `seed` is an int, a new ``Generator`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` instance then that instance is
        used.

    Notes
    -----
    Sobol' sequences [1]_ provide :math:`n=2^m` low discrepancy points in
    :math:`[0,1)^{d}`. Scrambling them [2]_ makes them suitable for singular
    integrands, provides a means of error estimation, and can improve their
    rate of convergence.

    There are many versions of Sobol' sequences depending on their
    'direction numbers'. This code uses direction numbers from [3]_. Hence,
    the maximum number of dimension is 21201. The direction numbers have been
    precomputed with search criterion 6 and can be retrieved at
    https://web.maths.unsw.edu.au/~fkuo/sobol/.

    .. warning::

       Sobol' sequences are a quadrature rule and they lose their balance
       properties if one uses a sample size that is not a power of 2, or skips
       the first point, or thins the sequence [4]_.

       If :math:`n=2^m` points are not enough then one should take :math:`2^M`
       points for :math:`M>m`. When scrambling, the number R of independent
       replicates does not have to be a power of 2.

       Sobol' sequences are generated to some number :math:`B` of bits.
       After :math:`2^B` points have been generated, the sequence will repeat.
       Currently :math:`B=30`.

    References
    ----------
    .. [1] I. M. Sobol. The distribution of points in a cube and the accurate
       evaluation of integrals. Zh. Vychisl. Mat. i Mat. Phys., 7:784-802,
       1967.

    .. [2] Art B. Owen. Scrambling Sobol and Niederreiter-Xing points.
       Journal of Complexity, 14(4):466-489, December 1998.

    .. [3] S. Joe and F. Y. Kuo. Constructing sobol sequences with better
       two-dimensional projections. SIAM Journal on Scientific Computing,
       30(5):2635-2654, 2008.

    .. [4] Art B. Owen. On dropping the first Sobol' point. arXiv 2008.08051,
       2020.

    Examples
    --------
    Generate samples from a low discrepancy sequence of Sobol'.

    >>> from scipy.stats import qmc
    >>> sampler = qmc.Sobol(d=2, scramble=False)
    >>> sample = sampler.random_base2(m=3)
    >>> sample
    array([[0.   , 0.   ],
           [0.5  , 0.5  ],
           [0.75 , 0.25 ],
           [0.25 , 0.75 ],
           [0.375, 0.375],
           [0.875, 0.875],
           [0.625, 0.125],
           [0.125, 0.625]])

    Compute the quality of the sample using the discrepancy criterion.

    >>> qmc.discrepancy(sample)
    0.013882107204860938

    To continue an existing design, extra points can be obtained
    by calling again `random_base2`. Alternatively, you can skip some
    points like:

    >>> _ = sampler.reset()
    >>> _ = sampler.fast_forward(4)
    >>> sample_continued = sampler.random_base2(m=2)
    >>> sample_continued
    array([[0.375, 0.375],
           [0.875, 0.875],
           [0.625, 0.125],
           [0.125, 0.625]])

    Finally, samples can be scaled to bounds.

    >>> l_bounds = [0, 2]
    >>> u_bounds = [10, 5]
    >>> qmc.scale(sample_continued, l_bounds, u_bounds)
    array([[3.75 , 3.125],
           [8.75 , 4.625],
           [6.25 , 2.375],
           [1.25 , 3.875]])

    """

    MAXDIM: ClassVar[int] = _MAXDIM
    MAXBIT: ClassVar[int] = _MAXBIT

    def __init__(
            self, d: IntNumber, *, scramble: bool = True,
            seed: SeedType = None
    ) -> None:
        super().__init__(d=d, seed=seed)
        if d > self.MAXDIM:
            raise ValueError(
                "Maximum supported dimensionality is {}.".format(self.MAXDIM)
            )

        # initialize direction numbers
        initialize_direction_numbers()

        # v is d x MAXBIT matrix
        self._sv = np.zeros((d, self.MAXBIT), dtype=int)
        initialize_v(self._sv, d)

        if not scramble:
            self._shift = np.zeros(d, dtype=int)
        else:
            self._scramble()

        self._quasi = self._shift.copy()
        self._first_point = (self._quasi / 2 ** self.MAXBIT).reshape(1, -1)

    def _scramble(self) -> None:
        """Scramble the sequence."""
        # Generate shift vector
        self._shift = np.dot(
            rng_integers(self.rng, 2, size=(self.d, self.MAXBIT), dtype=int),
            2 ** np.arange(self.MAXBIT, dtype=int),
        )
        self._quasi = self._shift.copy()
        # Generate lower triangular matrices (stacked across dimensions)
        ltm = np.tril(rng_integers(self.rng, 2,
                                   size=(self.d, self.MAXBIT, self.MAXBIT),
                                   dtype=int))
        _cscramble(self.d, ltm, self._sv)
        self.num_generated = 0

    def random(self, n: IntNumber = 1) -> np.ndarray:
        """Draw next point(s) in the Sobol' sequence.

        Parameters
        ----------
        n : int, optional
            Number of samples to generate in the parameter space. Default is 1.

        Returns
        -------
        sample : array_like (n, d)
            Sobol' sample.

        """
        sample = np.empty((n, self.d), dtype=float)

        if self.num_generated == 0:
            # verify n is 2**n
            if not (n & (n - 1) == 0):
                warnings.warn("The balance properties of Sobol' points require"
                              " n to be a power of 2.")

            if n == 1:
                sample = self._first_point
            else:
                _draw(n - 1, self.num_generated, self.d, self._sv,
                      self._quasi, sample)
                sample = np.concatenate([self._first_point, sample])[:n]  # type: ignore[misc]
        else:
            _draw(n, self.num_generated - 1, self.d, self._sv,
                  self._quasi, sample)

        self.num_generated += n
        return sample

    def random_base2(self, m: IntNumber) -> np.ndarray:
        """Draw point(s) from the Sobol' sequence.

        This function draws :math:`n=2^m` points in the parameter space
        ensuring the balance properties of the sequence.

        Parameters
        ----------
        m : int
            Logarithm in base 2 of the number of samples; i.e., n = 2^m.

        Returns
        -------
        sample : array_like (n, d)
            Sobol' sample.

        """
        n = 2 ** m

        total_n = self.num_generated + n
        if not (total_n & (total_n - 1) == 0):
            raise ValueError("The balance properties of Sobol' points require "
                             "n to be a power of 2. {0} points have been "
                             "previously generated, then: n={0}+2**{1}={2}. "
                             "If you still want to do this, the function "
                             "'Sobol.random()' can be used."
                             .format(self.num_generated, m, total_n))

        return self.random(n)

    def reset(self) -> Sobol:
        """Reset the engine to base state.

        Returns
        -------
        engine : Sobol
            Engine reset to its base state.

        """
        super().reset()
        self._quasi = self._shift.copy()
        return self

    def fast_forward(self, n: IntNumber) -> Sobol:
        """Fast-forward the sequence by `n` positions.

        Parameters
        ----------
        n : int
            Number of points to skip in the sequence.

        Returns
        -------
        engine: Sobol
            The fast-forwarded engine.

        """
        if self.num_generated == 0:
            _fast_forward(n - 1, self.num_generated, self.d,
                          self._sv, self._quasi)
        else:
            _fast_forward(n, self.num_generated - 1, self.d,
                          self._sv, self._quasi)
        self.num_generated += n
        return self


class MultivariateNormalQMC(QMCEngine):
    r"""QMC sampling from a multivariate Normal :math:`N(\mu, \Sigma)`.

    Parameters
    ----------
    mean : array_like (d,)
        The mean vector. Where ``d`` is the dimension.
    cov : array_like (d, d), optional
        The covariance matrix. If omitted, use `cov_root` instead.
        If both `cov` and `cov_root` are omitted, use the identity matrix.
    cov_root : array_like (d, d'), optional
        A root decomposition of the covariance matrix, where ``d'`` may be less
        than ``d`` if the covariance is not full rank. If omitted, use `cov`.
    inv_transform : bool, optional
        If True, use inverse transform instead of Box-Muller. Default is True.
    engine : QMCEngine, optional
        Quasi-Monte Carlo engine sampler. If None, `Sobol` is used.
    seed : {None, int, `numpy.random.Generator`}, optional
        If `seed` is None the `numpy.random.Generator` singleton is used.
        If `seed` is an int, a new ``Generator`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` instance then that instance is
        used.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from scipy.stats import qmc
    >>> engine = qmc.MultivariateNormalQMC(mean=[0, 5], cov=[[1, 0], [0, 1]])
    >>> sample = engine.random(512)
    >>> _ = plt.scatter(sample[:, 0], sample[:, 1])
    >>> plt.show()

    """

    def __init__(
            self, mean: npt.ArrayLike, cov: Optional[npt.ArrayLike] = None, *,
            cov_root: Optional[npt.ArrayLike] = None,
            inv_transform: bool = True,
            engine: Optional[QMCEngine] = None,
            seed: SeedType = None
    ) -> None:
        mean = np.array(mean, copy=False, ndmin=1)
        d = mean.shape[0]
        if cov is not None:
            # covariance matrix provided
            cov = np.array(cov, copy=False, ndmin=2)
            # check for square/symmetric cov matrix and mean vector has the
            # same d
            if not mean.shape[0] == cov.shape[0]:
                raise ValueError("Dimension mismatch between mean and "
                                 "covariance.")
            if not np.allclose(cov, cov.transpose()):
                raise ValueError("Covariance matrix is not symmetric.")
            # compute Cholesky decomp; if it fails, do the eigen decomposition
            try:
                cov_root = np.linalg.cholesky(cov).transpose()
            except np.linalg.LinAlgError:
                eigval, eigvec = np.linalg.eigh(cov)
                if not np.all(eigval >= -1.0e-8):
                    raise ValueError("Covariance matrix not PSD.")
                eigval = np.clip(eigval, 0.0, None)
                cov_root = (eigvec * np.sqrt(eigval)).transpose()
        elif cov_root is not None:
            # root decomposition provided
            cov_root = np.atleast_2d(cov_root)
            if not mean.shape[0] == cov_root.shape[0]:
                raise ValueError("Dimension mismatch between mean and "
                                 "covariance.")
        else:
            # corresponds to identity covariance matrix
            cov_root = None

        super().__init__(d=d, seed=seed)
        self._inv_transform = inv_transform

        if not inv_transform:
            # to apply Box-Muller, we need an even number of dimensions
            engine_dim = 2 * math.ceil(d / 2)
        else:
            engine_dim = d
        if engine is None:
            self.engine = Sobol(d=engine_dim, scramble=True, seed=seed)  # type: QMCEngine
        elif isinstance(engine, QMCEngine) and engine.d != 1:
            if engine.d != d:
                raise ValueError("Dimension of `engine` must be consistent"
                                 " with dimensions of mean and covariance.")
            self.engine = engine
        else:
            raise ValueError("`engine` must be an instance of "
                             "`scipy.stats.qmc.QMCEngine` or `None`.")

        self._mean = mean
        self._corr_matrix = cov_root

    def random(self, n: IntNumber = 1) -> np.ndarray:
        """Draw `n` QMC samples from the multivariate Normal.

        Parameters
        ----------
        n : int, optional
            Number of samples to generate in the parameter space. Default is 1.

        Returns
        -------
        sample : array_like (n, d)
            Sample.

        """
        base_samples = self._standard_normal_samples(n)
        self.num_generated += n
        return self._correlate(base_samples)

    def reset(self) -> MultivariateNormalQMC:
        """Reset the engine to base state.

        Returns
        -------
        engine : MultivariateNormalQMC
            Engine reset to its base state.

        """
        super().reset()
        self.engine.reset()
        return self

    def _correlate(self, base_samples: np.ndarray) -> np.ndarray:
        if self._corr_matrix is not None:
            return base_samples @ self._corr_matrix + self._mean
        else:
            # avoid multiplying with identity here
            return base_samples + self._mean

    def _standard_normal_samples(self, n: IntNumber = 1) -> np.ndarray:
        """Draw `n` QMC samples from the standard Normal :math:`N(0, I_d)`.

        Parameters
        ----------
        n : int, optional
            Number of samples to generate in the parameter space. Default is 1.

        Returns
        -------
        sample : array_like (n, d)
            Sample.

        """
        # get base samples
        samples = self.engine.random(n)
        if self._inv_transform:
            # apply inverse transform
            # (values to close to 0/1 result in inf values)
            return stats.norm.ppf(0.5 + (1 - 1e-10) * (samples - 0.5))  # type: ignore[attr-defined]
        else:
            # apply Box-Muller transform (note: indexes starting from 1)
            even = np.arange(0, samples.shape[-1], 2)
            Rs = np.sqrt(-2 * np.log(samples[:, even]))
            thetas = 2 * math.pi * samples[:, 1 + even]
            cos = np.cos(thetas)
            sin = np.sin(thetas)
            transf_samples = np.stack([Rs * cos, Rs * sin],
                                      -1).reshape(n, -1)
            # make sure we only return the number of dimension requested
            return transf_samples[:, : self.d]  # type: ignore[misc]


class MultinomialQMC(QMCEngine):
    r"""QMC sampling from a multinomial distribution.

    Parameters
    ----------
    pvals : array_like (k,)
        Vector of probabilities of size ``k``, where ``k`` is the number
        of categories. Elements must be non-negative and sum to 1.
    engine : QMCEngine, optional
        Quasi-Monte Carlo engine sampler. If None, `Sobol` is used.
    seed : {None, int, `numpy.random.Generator`}, optional
        If `seed` is None the `numpy.random.Generator` singleton is used.
        If `seed` is an int, a new ``Generator`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` instance then that instance is
        used.

    Examples
    --------
    >>> from scipy.stats import qmc
    >>> engine = qmc.MultinomialQMC(pvals=[0.2, 0.4, 0.4])
    >>> sample = engine.random(10)

    """

    def __init__(
            self, pvals: npt.ArrayLike, *, engine: Optional[QMCEngine] = None,
            seed: SeedType = None
    ) -> None:
        self.pvals = np.array(pvals, copy=False, ndmin=1)
        if np.min(pvals) < 0:
            raise ValueError('Elements of pvals must be non-negative.')
        if not np.isclose(np.sum(pvals), 1):
            raise ValueError('Elements of pvals must sum to 1.')
        if engine is None:
            self.engine = Sobol(d=1, scramble=True, seed=seed)  # type: QMCEngine
        elif isinstance(engine, QMCEngine):
            if engine.d != 1:
                raise ValueError("Dimension of `engine` must be 1.")
            self.engine = engine
        else:
            raise ValueError("`engine` must be an instance of "
                             "`scipy.stats.qmc.QMCEngine` or `None`.")

        super().__init__(d=1, seed=seed)

    def random(self, n: IntNumber = 1) -> np.ndarray:
        """Draw `n` QMC samples from the multinomial distribution.

        Parameters
        ----------
        n : int, optional
            Number of samples to generate in the parameter space. Default is 1.

        Returns
        -------
        samples : array_like (pvals,)
            Vector of size ``p`` summing to `n`.

        """
        base_draws = self.engine.random(n).ravel()
        p_cumulative = np.empty_like(self.pvals, dtype=float)
        _fill_p_cumulative(np.array(self.pvals, dtype=float), p_cumulative)
        sample = np.zeros_like(self.pvals, dtype=int)
        _categorize(base_draws, p_cumulative, sample)
        self.num_generated += n
        return sample

    def reset(self) -> MultinomialQMC:
        """Reset the engine to base state.

        Returns
        -------
        engine : MultinomialQMC
            Engine reset to its base state.

        """
        super().reset()
        self.engine.reset()
        return self
