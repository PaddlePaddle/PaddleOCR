# -*- coding: utf-8 -*-
r"""
====================================================
Quasi-Monte Carlo submodule (:mod:`scipy.stats.qmc`)
====================================================

.. currentmodule:: scipy.stats.qmc

This module provides Quasi-Monte Carlo generators and associated helper
functions.


Quasi-Monte Carlo
=================

Engines
-------

.. autosummary::
   :toctree: generated/

   QMCEngine
   Sobol
   Halton
   LatinHypercube
   MultinomialQMC
   MultivariateNormalQMC

Helpers
-------

.. autosummary::
   :toctree: generated/

   discrepancy
   update_discrepancy
   scale


Introduction to Quasi-Monte Carlo
=================================

Quasi-Monte Carlo (QMC) methods [1]_, [2]_, [3]_ provide an
:math:`n \times d` array of numbers in :math:`[0,1]`. They can be used in
place of :math:`n` points from the :math:`U[0,1]^{d}` distribution. Compared to
random points, QMC points are designed to have fewer gaps and clumps. This is
quantified by discrepancy measures [4]_. From the Koksma-Hlawka
inequality [5]_ we know that low discrepancy reduces a bound on
integration error. Averaging a function :math:`f` over :math:`n` QMC points
can achieve an integration error close to :math:`O(n^{-1})` for well
behaved functions [2]_.

Most QMC constructions are designed for special values of :math:`n`
such as powers of 2 or large primes. Changing the sample
size by even one can degrade their performance, even their
rate of convergence [6]_. For instance :math:`n=100` points may give less
accuracy than :math:`n=64` if the method was designed for :math:`n=2^m`.

Some QMC constructions are extensible in :math:`n`: we can find
another special sample size :math:`n' > n` and often an infinite
sequence of increasing special sample sizes. Some QMC
constructions are extensible in :math:`d`: we can increase the dimension,
possibly to some upper bound, and typically without requiring
special values of :math:`d`. Some QMC methods are extensible in
both :math:`n` and :math:`d`.

QMC points are deterministic. That makes it hard to estimate the accuracy of
integrals estimated by averages over QMC points. Randomized QMC (RQMC) [7]_
points are constructed so that each point is individually :math:`U[0,1]^{d}`
while collectively the :math:`n` points retain their low discrepancy.
One can make :math:`R` independent replications of RQMC points to
see how stable a computation is. From :math:`R` independent values,
a t-test (or bootstrap t-test [8]_) then gives approximate confidence
intervals on the mean value. Some RQMC methods produce a
root mean squared error that is actually :math:`o(1/n)` and smaller than
the rate seen in unrandomized QMC. An intuitive explanation is
that the error is a sum of many small ones and random errors
cancel in a way that deterministic ones do not. RQMC also
has advantages on integrands that are singular or, for other
reasons, fail to be Riemann integrable.

(R)QMC cannot beat Bahkvalov's curse of dimension (see [9]_). For
any random or deterministic method, there are worst case functions
that will give it poor performance in high dimensions. A worst
case function for QMC might be 0 at all n points but very
large elsewhere. Worst case analyses get very pessimistic
in high dimensions. (R)QMC can bring a great improvement over
MC when the functions on which it is used are not worst case.
For instance (R)QMC can be especially effective on integrands
that are well approximated by sums of functions of
some small number of their input variables at a time [10]_, [11]_.
That property is often a surprising finding about those functions.

Also, to see an improvement over IID MC, (R)QMC requires a bit of smoothness of
the integrand, roughly the mixed first order derivative in each direction,
:math:`\partial^d f/\partial x_1 \cdots \partial x_d`, must be integral.
For instance, a function that is 1 inside the hypersphere and 0 outside of it
has infinite variation in the sense of Hardy and Krause for any dimension
:math:`d = 2`.

Scrambled nets are a kind of RQMC that have some valuable robustness
properties [12]_. If the integrand is square integrable, they give variance
:math:`var_{SNET} = o(1/n)`. There is a finite upper bound on
:math:`var_{SNET} / var_{MC}` that holds simultaneously for every square
integrable integrand. Scrambled nets satisfy a strong law of large numbers
for :math:`f` in :math:`L^p` when :math:`p>1`. In some
special cases there is a central limit theorem [13]_. For smooth enough
integrands they can achieve RMSE nearly :math:`O(n^{-3})`. See [12]_
for references about these properties.

The main kinds of QMC methods are lattice rules [14]_ and digital
nets and sequences [2]_, [15]_. The theories meet up in polynomial
lattice rules [16]_ which can produce digital nets. Lattice rules
require some form of search for good constructions. For digital
nets there are widely used default constructions.

The most widely used QMC methods are Sobol' sequences [17]_.
These are digital nets. They are extensible in both :math:`n` and :math:`d`.
They can be scrambled. The special sample sizes are powers
of 2. Another popular method are Halton sequences [18]_.
The constructions resemble those of digital nets. The earlier
dimensions have much better equidistribution properties than
later ones. There are essentially no special sample sizes.
They are not thought to be as accurate as Sobol' sequences.
They can be scrambled. The nets of Faure [19]_ are also widely
used. All dimensions are equally good, but the special sample
sizes grow rapidly with dimension :math:`d`. They can be scrambled.
The nets of Niederreiter and Xing [20]_ have the best asymptotic
properties but have not shown good empirical performance [21]_.

Higher order digital nets are formed by a digit interleaving process
in the digits of the constructed points. They can achieve higher
levels of asymptotic accuracy given higher smoothness conditions on :math:`f`
and they can be scrambled [22]_. There is little or no empirical work
showing the improved rate to be attained.

Using QMC is like using the entire period of a small random
number generator. The constructions are similar and so
therefore are the computational costs [23]_.

(R)QMC is sometimes improved by passing the points through
a baker's transformation (tent function) prior to using them.
That function has the form :math:`1-2|x-1/2|`. As :math:`x` goes from 0 to
1, this function goes from 0 to 1 and then back. It is very
useful to produce a periodic function for lattice rules [14]_,
and sometimes it improves the convergence rate [24]_.

It is not straightforward to apply QMC methods to Markov
chain Monte Carlo (MCMC).  We can think of MCMC as using
:math:`n=1` point in :math:`[0,1]^{d}` for very large :math:`d`, with
ergodic results corresponding to :math:`d \to \infty`. One proposal is
in [25]_ and under strong conditions an improved rate of convergence
has been shown [26]_.

Returning to Sobol' points: there are many versions depending
on what are called direction numbers. Those are the result of
searches and are tabulated. A very widely used set of direction
numbers come from [27]_. It is extensible in dimension up to
:math:`d=21201`.

References
----------
.. [1] Owen, Art B. "Monte Carlo Book: the Quasi-Monte Carlo parts." (2019).
.. [2] Niederreiter, Harald. Random number generation and quasi-Monte Carlo
   methods. Society for Industrial and Applied Mathematics, 1992.
.. [3] Dick, Josef, Frances Y. Kuo, and Ian H. Sloan. "High-dimensional
   integration: the quasi-Monte Carlo way." Acta Numerica 22 (2013): 133.
.. [4] Aho, A. V., C. Aistleitner, T. Anderson, K. Appel, V. Arnol'd, N.
   Aronszajn, D. Asotsky et al. "W. Chen et al.(eds.), A Panorama of
   Discrepancy Theory (2014): 679. Sringer International Publishing,
   Switzerland.
.. [5] Hickernell, Fred J. "Koksma-Hlawka Inequality." Wiley StatsRef:
   Statistics Reference Online (2014).
.. [6] Owen, Art B. "On dropping the first Sobol' point." arXiv preprint
   arXiv:2008.08051 (2020).
.. [7] L'Ecuyer, Pierre, and Christiane Lemieux. "Recent advances in randomized
   quasi-Monte Carlo methods." In Modeling uncertainty, pp. 419-474. Springer,
   New York, NY, 2002.
.. [8] DiCiccio, Thomas J., and Bradley Efron. "Bootstrap confidence
   intervals." Statistical science (1996): 189-212.
.. [9] Dimov, Ivan T. Monte Carlo methods for applied scientists. World
   Scientific, 2008.
.. [10] Caflisch, Russel E., William J. Morokoff, and Art B. Owen. Valuation of
   mortgage backed securities using Brownian bridges to reduce effective
   dimension. Journal of Computational Finance, (1997): 1, no. 1 27-46.
.. [11] Sloan, Ian H., and Henryk Wozniakowski. "When are quasi-Monte Carlo
   algorithms efficient for high dimensional integrals?." Journal of Complexity
   14, no. 1 (1998): 1-33.
.. [12] Owen, Art B., and Daniel Rudolf "A strong law of large numbers for
   scrambled net integration." SIAM Review, to appear.
.. [13] Loh, Wei-Liem. "On the asymptotic distribution of scrambled net
   quadrature." The Annals of Statistics 31, no. 4 (2003): 1282-1324.
.. [14] Sloan, Ian H. and S. Joe. Lattice methods for multiple integration.
   Oxford University Press, 1994.
.. [15] Dick, Josef, and Friedrich Pillichshammer. Digital nets and sequences:
   discrepancy theory and quasi-Monte Carlo integration. Cambridge University
   Press, 2010.
.. [16] Dick, Josef, F. Kuo, Friedrich Pillichshammer, and I. Sloan.
   "Construction algorithms for polynomial lattice rules for multivariate
   integration." Mathematics of computation 74, no. 252 (2005): 1895-1921.
.. [17] Sobol', Il'ya Meerovich. "On the distribution of points in a cube and
   the approximate evaluation of integrals." Zhurnal Vychislitel'noi Matematiki
   i Matematicheskoi Fiziki 7, no. 4 (1967): 784-802.
.. [18] Halton, John H. "On the efficiency of certain quasi-random sequences of
   points in evaluating multi-dimensional integrals." Numerische Mathematik 2,
   no. 1 (1960): 84-90.
.. [19] Faure, Henri. "Discrepance de suites associees a un systeme de
   numeration (en dimension s)." Acta arithmetica 41, no. 4 (1982): 337-351.
.. [20] Niederreiter, Harold, and Chaoping Xing. "Low-discrepancy sequences and
   global function fields with many rational places." Finite Fields and their
   applications 2, no. 3 (1996): 241-273.
.. [21] Hong, Hee Sun, and Fred J. Hickernell. "Algorithm 823: Implementing
   scrambled digital sequences." ACM Transactions on Mathematical Software
   (TOMS) 29, no. 2 (2003): 95-109.
.. [22] Dick, Josef. "Higher order scrambled digital nets achieve the optimal
   rate of the root mean square error for smooth integrands." The Annals of
   Statistics 39, no. 3 (2011): 1372-1398.
.. [23] Niederreiter, Harald. "Multidimensional numerical integration using
   pseudorandom numbers." In Stochastic Programming 84 Part I, pp. 17-38.
   Springer, Berlin, Heidelberg, 1986.
.. [24] Hickernell, Fred J. "Obtaining O (N-2+e) Convergence for Lattice
   Quadrature Rules." In Monte Carlo and Quasi-Monte Carlo Methods 2000,
   pp. 274-289. Springer, Berlin, Heidelberg, 2002.
.. [25] Owen, Art B., and Seth D. Tribble. "A quasi-Monte Carlo Metropolis
   algorithm." Proceedings of the National Academy of Sciences 102, no. 25
   (2005): 8844-8849.
.. [26] Chen, Su. "Consistency and convergence rate of Markov chain quasi Monte
   Carlo with examples." PhD diss., Stanford University, 2011.
.. [27] Joe, Stephen, and Frances Y. Kuo. "Constructing Sobol sequences with
   better two-dimensional projections." SIAM Journal on Scientific Computing
   30, no. 5 (2008): 2635-2654.

"""
from ._qmc import *
