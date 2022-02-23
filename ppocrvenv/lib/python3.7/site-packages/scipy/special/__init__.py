"""
========================================
Special functions (:mod:`scipy.special`)
========================================

.. currentmodule:: scipy.special

Nearly all of the functions below are universal functions and follow
broadcasting and automatic array-looping rules.

.. seealso::

   `scipy.special.cython_special` -- Typed Cython versions of special functions


Error handling
==============

Errors are handled by returning NaNs or other appropriate values.
Some of the special function routines can emit warnings or raise
exceptions when an error occurs. By default this is disabled; to
query and control the current error handling state the following
functions are provided.

.. autosummary::
   :toctree: generated/

   geterr                 -- Get the current way of handling special-function errors.
   seterr                 -- Set how special-function errors are handled.
   errstate               -- Context manager for special-function error handling.
   SpecialFunctionWarning -- Warning that can be emitted by special functions.
   SpecialFunctionError   -- Exception that can be raised by special functions.

Available functions
===================

Airy functions
--------------

.. autosummary::
   :toctree: generated/

   airy     -- Airy functions and their derivatives.
   airye    -- Exponentially scaled Airy functions and their derivatives.
   ai_zeros -- Compute `nt` zeros and values of the Airy function Ai and its derivative.
   bi_zeros -- Compute `nt` zeros and values of the Airy function Bi and its derivative.
   itairy   -- Integrals of Airy functions


Elliptic functions and integrals
--------------------------------

.. autosummary::
   :toctree: generated/

   ellipj    -- Jacobian elliptic functions.
   ellipk    -- Complete elliptic integral of the first kind.
   ellipkm1  -- Complete elliptic integral of the first kind around `m` = 1.
   ellipkinc -- Incomplete elliptic integral of the first kind.
   ellipe    -- Complete elliptic integral of the second kind.
   ellipeinc -- Incomplete elliptic integral of the second kind.

Bessel functions
----------------

.. autosummary::
   :toctree: generated/

   jv            -- Bessel function of the first kind of real order and \
                    complex argument.
   jve           -- Exponentially scaled Bessel function of order `v`.
   yn            -- Bessel function of the second kind of integer order and \
                    real argument.
   yv            -- Bessel function of the second kind of real order and \
                    complex argument.
   yve           -- Exponentially scaled Bessel function of the second kind \
                    of real order.
   kn            -- Modified Bessel function of the second kind of integer \
                    order `n`
   kv            -- Modified Bessel function of the second kind of real order \
                    `v`
   kve           -- Exponentially scaled modified Bessel function of the \
                    second kind.
   iv            -- Modified Bessel function of the first kind of real order.
   ive           -- Exponentially scaled modified Bessel function of the \
                    first kind.
   hankel1       -- Hankel function of the first kind.
   hankel1e      -- Exponentially scaled Hankel function of the first kind.
   hankel2       -- Hankel function of the second kind.
   hankel2e      -- Exponentially scaled Hankel function of the second kind.
   wright_bessel -- Wright's generalized Bessel function.

The following is not a universal function:

.. autosummary::
   :toctree: generated/

   lmbda -- Jahnke-Emden Lambda function, Lambdav(x).

Zeros of Bessel functions
^^^^^^^^^^^^^^^^^^^^^^^^^

These are not universal functions:

.. autosummary::
   :toctree: generated/

   jnjnp_zeros -- Compute zeros of integer-order Bessel functions Jn and Jn'.
   jnyn_zeros  -- Compute nt zeros of Bessel functions Jn(x), Jn'(x), Yn(x), and Yn'(x).
   jn_zeros    -- Compute zeros of integer-order Bessel function Jn(x).
   jnp_zeros   -- Compute zeros of integer-order Bessel function derivative Jn'(x).
   yn_zeros    -- Compute zeros of integer-order Bessel function Yn(x).
   ynp_zeros   -- Compute zeros of integer-order Bessel function derivative Yn'(x).
   y0_zeros    -- Compute nt zeros of Bessel function Y0(z), and derivative at each zero.
   y1_zeros    -- Compute nt zeros of Bessel function Y1(z), and derivative at each zero.
   y1p_zeros   -- Compute nt zeros of Bessel derivative Y1'(z), and value at each zero.

Faster versions of common Bessel functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/

   j0  -- Bessel function of the first kind of order 0.
   j1  -- Bessel function of the first kind of order 1.
   y0  -- Bessel function of the second kind of order 0.
   y1  -- Bessel function of the second kind of order 1.
   i0  -- Modified Bessel function of order 0.
   i0e -- Exponentially scaled modified Bessel function of order 0.
   i1  -- Modified Bessel function of order 1.
   i1e -- Exponentially scaled modified Bessel function of order 1.
   k0  -- Modified Bessel function of the second kind of order 0, :math:`K_0`.
   k0e -- Exponentially scaled modified Bessel function K of order 0
   k1  -- Modified Bessel function of the second kind of order 1, :math:`K_1(x)`.
   k1e -- Exponentially scaled modified Bessel function K of order 1.

Integrals of Bessel functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/

   itj0y0     -- Integrals of Bessel functions of order 0.
   it2j0y0    -- Integrals related to Bessel functions of order 0.
   iti0k0     -- Integrals of modified Bessel functions of order 0.
   it2i0k0    -- Integrals related to modified Bessel functions of order 0.
   besselpoly -- Weighted integral of a Bessel function.

Derivatives of Bessel functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/

   jvp  -- Compute nth derivative of Bessel function Jv(z) with respect to `z`.
   yvp  -- Compute nth derivative of Bessel function Yv(z) with respect to `z`.
   kvp  -- Compute nth derivative of real-order modified Bessel function Kv(z)
   ivp  -- Compute nth derivative of modified Bessel function Iv(z) with respect to `z`.
   h1vp -- Compute nth derivative of Hankel function H1v(z) with respect to `z`.
   h2vp -- Compute nth derivative of Hankel function H2v(z) with respect to `z`.

Spherical Bessel functions
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/

   spherical_jn -- Spherical Bessel function of the first kind or its derivative.
   spherical_yn -- Spherical Bessel function of the second kind or its derivative.
   spherical_in -- Modified spherical Bessel function of the first kind or its derivative.
   spherical_kn -- Modified spherical Bessel function of the second kind or its derivative.

Riccati-Bessel functions
^^^^^^^^^^^^^^^^^^^^^^^^

These are not universal functions:

.. autosummary::
   :toctree: generated/

   riccati_jn -- Compute Ricatti-Bessel function of the first kind and its derivative.
   riccati_yn -- Compute Ricatti-Bessel function of the second kind and its derivative.

Struve functions
----------------

.. autosummary::
   :toctree: generated/

   struve       -- Struve function.
   modstruve    -- Modified Struve function.
   itstruve0    -- Integral of the Struve function of order 0.
   it2struve0   -- Integral related to the Struve function of order 0.
   itmodstruve0 -- Integral of the modified Struve function of order 0.


Raw statistical functions
-------------------------

.. seealso:: :mod:`scipy.stats`: Friendly versions of these functions.

.. autosummary::
   :toctree: generated/

   bdtr         -- Binomial distribution cumulative distribution function.
   bdtrc        -- Binomial distribution survival function.
   bdtri        -- Inverse function to `bdtr` with respect to `p`.
   bdtrik       -- Inverse function to `bdtr` with respect to `k`.
   bdtrin       -- Inverse function to `bdtr` with respect to `n`.
   btdtr        -- Cumulative distribution function of the beta distribution.
   btdtri       -- The `p`-th quantile of the beta distribution.
   btdtria      -- Inverse of `btdtr` with respect to `a`.
   btdtrib      -- btdtria(a, p, x).
   fdtr         -- F cumulative distribution function.
   fdtrc        -- F survival function.
   fdtri        -- The `p`-th quantile of the F-distribution.
   fdtridfd     -- Inverse to `fdtr` vs dfd.
   gdtr         -- Gamma distribution cumulative distribution function.
   gdtrc        -- Gamma distribution survival function.
   gdtria       -- Inverse of `gdtr` vs a.
   gdtrib       -- Inverse of `gdtr` vs b.
   gdtrix       -- Inverse of `gdtr` vs x.
   nbdtr        -- Negative binomial cumulative distribution function.
   nbdtrc       -- Negative binomial survival function.
   nbdtri       -- Inverse of `nbdtr` vs `p`.
   nbdtrik      -- Inverse of `nbdtr` vs `k`.
   nbdtrin      -- Inverse of `nbdtr` vs `n`.
   ncfdtr       -- Cumulative distribution function of the non-central F distribution.
   ncfdtridfd   -- Calculate degrees of freedom (denominator) for the noncentral F-distribution.
   ncfdtridfn   -- Calculate degrees of freedom (numerator) for the noncentral F-distribution.
   ncfdtri      -- Inverse cumulative distribution function of the non-central F distribution.
   ncfdtrinc    -- Calculate non-centrality parameter for non-central F distribution.
   nctdtr       -- Cumulative distribution function of the non-central `t` distribution.
   nctdtridf    -- Calculate degrees of freedom for non-central t distribution.
   nctdtrit     -- Inverse cumulative distribution function of the non-central t distribution.
   nctdtrinc    -- Calculate non-centrality parameter for non-central t distribution.
   nrdtrimn     -- Calculate mean of normal distribution given other params.
   nrdtrisd     -- Calculate standard deviation of normal distribution given other params.
   pdtr         -- Poisson cumulative distribution function.
   pdtrc        -- Poisson survival function.
   pdtri        -- Inverse to `pdtr` vs m.
   pdtrik       -- Inverse to `pdtr` vs k.
   stdtr        -- Student t distribution cumulative distribution function.
   stdtridf     -- Inverse of `stdtr` vs df.
   stdtrit      -- Inverse of `stdtr` vs `t`.
   chdtr        -- Chi square cumulative distribution function.
   chdtrc       -- Chi square survival function.
   chdtri       -- Inverse to `chdtrc`.
   chdtriv      -- Inverse to `chdtr` vs `v`.
   ndtr         -- Gaussian cumulative distribution function.
   log_ndtr     -- Logarithm of Gaussian cumulative distribution function.
   ndtri        -- Inverse of `ndtr` vs x.
   ndtri_exp    -- Inverse of `log_ndtr` vs x.
   chndtr       -- Non-central chi square cumulative distribution function.
   chndtridf    -- Inverse to `chndtr` vs `df`.
   chndtrinc    -- Inverse to `chndtr` vs `nc`.
   chndtrix     -- Inverse to `chndtr` vs `x`.
   smirnov      -- Kolmogorov-Smirnov complementary cumulative distribution function.
   smirnovi     -- Inverse to `smirnov`.
   kolmogorov   -- Complementary cumulative distribution function of Kolmogorov distribution.
   kolmogi      -- Inverse function to `kolmogorov`.
   tklmbda      -- Tukey-Lambda cumulative distribution function.
   logit        -- Logit ufunc for ndarrays.
   expit        -- Expit ufunc for ndarrays.
   boxcox       -- Compute the Box-Cox transformation.
   boxcox1p     -- Compute the Box-Cox transformation of 1 + `x`.
   inv_boxcox   -- Compute the inverse of the Box-Cox transformation.
   inv_boxcox1p -- Compute the inverse of the Box-Cox transformation.
   owens_t      -- Owen's T Function.


Information Theory functions
----------------------------

.. autosummary::
   :toctree: generated/

   entr         -- Elementwise function for computing entropy.
   rel_entr     -- Elementwise function for computing relative entropy.
   kl_div       -- Elementwise function for computing Kullback-Leibler divergence.
   huber        -- Huber loss function.
   pseudo_huber -- Pseudo-Huber loss function.


Gamma and related functions
---------------------------

.. autosummary::
   :toctree: generated/

   gamma        -- Gamma function.
   gammaln      -- Logarithm of the absolute value of the Gamma function for real inputs.
   loggamma     -- Principal branch of the logarithm of the Gamma function.
   gammasgn     -- Sign of the gamma function.
   gammainc     -- Regularized lower incomplete gamma function.
   gammaincinv  -- Inverse to `gammainc`.
   gammaincc    -- Regularized upper incomplete gamma function.
   gammainccinv -- Inverse to `gammaincc`.
   beta         -- Beta function.
   betaln       -- Natural logarithm of absolute value of beta function.
   betainc      -- Incomplete beta integral.
   betaincinv   -- Inverse function to beta integral.
   psi          -- The digamma function.
   rgamma       -- Gamma function inverted.
   polygamma    -- Polygamma function n.
   multigammaln -- Returns the log of multivariate gamma, also sometimes called the generalized gamma.
   digamma      -- psi(x[, out]).
   poch         -- Rising factorial (z)_m.


Error function and Fresnel integrals
------------------------------------

.. autosummary::
   :toctree: generated/

   erf           -- Returns the error function of complex argument.
   erfc          -- Complementary error function, ``1 - erf(x)``.
   erfcx         -- Scaled complementary error function, ``exp(x**2) * erfc(x)``.
   erfi          -- Imaginary error function, ``-i erf(i z)``.
   erfinv        -- Inverse function for erf.
   erfcinv       -- Inverse function for erfc.
   wofz          -- Faddeeva function.
   dawsn         -- Dawson's integral.
   fresnel       -- Fresnel sin and cos integrals.
   fresnel_zeros -- Compute nt complex zeros of sine and cosine Fresnel integrals S(z) and C(z).
   modfresnelp   -- Modified Fresnel positive integrals.
   modfresnelm   -- Modified Fresnel negative integrals.
   voigt_profile -- Voigt profile.

These are not universal functions:

.. autosummary::
   :toctree: generated/

   erf_zeros      -- Compute nt complex zeros of error function erf(z).
   fresnelc_zeros -- Compute nt complex zeros of cosine Fresnel integral C(z).
   fresnels_zeros -- Compute nt complex zeros of sine Fresnel integral S(z).

Legendre functions
------------------

.. autosummary::
   :toctree: generated/

   lpmv     -- Associated Legendre function of integer order and real degree.
   sph_harm -- Compute spherical harmonics.

These are not universal functions:

.. autosummary::
   :toctree: generated/

   clpmn -- Associated Legendre function of the first kind for complex arguments.
   lpn   -- Legendre function of the first kind.
   lqn   -- Legendre function of the second kind.
   lpmn  -- Sequence of associated Legendre functions of the first kind.
   lqmn  -- Sequence of associated Legendre functions of the second kind.

Ellipsoidal harmonics
---------------------

.. autosummary::
   :toctree: generated/

   ellip_harm   -- Ellipsoidal harmonic functions E^p_n(l).
   ellip_harm_2 -- Ellipsoidal harmonic functions F^p_n(l).
   ellip_normal -- Ellipsoidal harmonic normalization constants gamma^p_n.

Orthogonal polynomials
----------------------

The following functions evaluate values of orthogonal polynomials:

.. autosummary::
   :toctree: generated/

   assoc_laguerre   -- Compute the generalized (associated) Laguerre polynomial of degree n and order k.
   eval_legendre    -- Evaluate Legendre polynomial at a point.
   eval_chebyt      -- Evaluate Chebyshev polynomial of the first kind at a point.
   eval_chebyu      -- Evaluate Chebyshev polynomial of the second kind at a point.
   eval_chebyc      -- Evaluate Chebyshev polynomial of the first kind on [-2, 2] at a point.
   eval_chebys      -- Evaluate Chebyshev polynomial of the second kind on [-2, 2] at a point.
   eval_jacobi      -- Evaluate Jacobi polynomial at a point.
   eval_laguerre    -- Evaluate Laguerre polynomial at a point.
   eval_genlaguerre -- Evaluate generalized Laguerre polynomial at a point.
   eval_hermite     -- Evaluate physicist's Hermite polynomial at a point.
   eval_hermitenorm -- Evaluate probabilist's (normalized) Hermite polynomial at a point.
   eval_gegenbauer  -- Evaluate Gegenbauer polynomial at a point.
   eval_sh_legendre -- Evaluate shifted Legendre polynomial at a point.
   eval_sh_chebyt   -- Evaluate shifted Chebyshev polynomial of the first kind at a point.
   eval_sh_chebyu   -- Evaluate shifted Chebyshev polynomial of the second kind at a point.
   eval_sh_jacobi   -- Evaluate shifted Jacobi polynomial at a point.

The following functions compute roots and quadrature weights for
orthogonal polynomials:

.. autosummary::
   :toctree: generated/

   roots_legendre    -- Gauss-Legendre quadrature.
   roots_chebyt      -- Gauss-Chebyshev (first kind) quadrature.
   roots_chebyu      -- Gauss-Chebyshev (second kind) quadrature.
   roots_chebyc      -- Gauss-Chebyshev (first kind) quadrature.
   roots_chebys      -- Gauss-Chebyshev (second kind) quadrature.
   roots_jacobi      -- Gauss-Jacobi quadrature.
   roots_laguerre    -- Gauss-Laguerre quadrature.
   roots_genlaguerre -- Gauss-generalized Laguerre quadrature.
   roots_hermite     -- Gauss-Hermite (physicst's) quadrature.
   roots_hermitenorm -- Gauss-Hermite (statistician's) quadrature.
   roots_gegenbauer  -- Gauss-Gegenbauer quadrature.
   roots_sh_legendre -- Gauss-Legendre (shifted) quadrature.
   roots_sh_chebyt   -- Gauss-Chebyshev (first kind, shifted) quadrature.
   roots_sh_chebyu   -- Gauss-Chebyshev (second kind, shifted) quadrature.
   roots_sh_jacobi   -- Gauss-Jacobi (shifted) quadrature.

The functions below, in turn, return the polynomial coefficients in
``orthopoly1d`` objects, which function similarly as `numpy.poly1d`.
The ``orthopoly1d`` class also has an attribute ``weights``, which returns
the roots, weights, and total weights for the appropriate form of Gaussian
quadrature. These are returned in an ``n x 3`` array with roots in the first
column, weights in the second column, and total weights in the final column.
Note that ``orthopoly1d`` objects are converted to `~numpy.poly1d` when doing
arithmetic, and lose information of the original orthogonal polynomial.

.. autosummary::
   :toctree: generated/

   legendre    -- Legendre polynomial.
   chebyt      -- Chebyshev polynomial of the first kind.
   chebyu      -- Chebyshev polynomial of the second kind.
   chebyc      -- Chebyshev polynomial of the first kind on :math:`[-2, 2]`.
   chebys      -- Chebyshev polynomial of the second kind on :math:`[-2, 2]`.
   jacobi      -- Jacobi polynomial.
   laguerre    -- Laguerre polynomial.
   genlaguerre -- Generalized (associated) Laguerre polynomial.
   hermite     -- Physicist's Hermite polynomial.
   hermitenorm -- Normalized (probabilist's) Hermite polynomial.
   gegenbauer  -- Gegenbauer (ultraspherical) polynomial.
   sh_legendre -- Shifted Legendre polynomial.
   sh_chebyt   -- Shifted Chebyshev polynomial of the first kind.
   sh_chebyu   -- Shifted Chebyshev polynomial of the second kind.
   sh_jacobi   -- Shifted Jacobi polynomial.

.. warning::

   Computing values of high-order polynomials (around ``order > 20``) using
   polynomial coefficients is numerically unstable. To evaluate polynomial
   values, the ``eval_*`` functions should be used instead.


Hypergeometric functions
------------------------

.. autosummary::
   :toctree: generated/

   hyp2f1 -- Gauss hypergeometric function 2F1(a, b; c; z).
   hyp1f1 -- Confluent hypergeometric function 1F1(a, b; x).
   hyperu -- Confluent hypergeometric function U(a, b, x) of the second kind.
   hyp0f1 -- Confluent hypergeometric limit function 0F1.


Parabolic cylinder functions
----------------------------

.. autosummary::
   :toctree: generated/

   pbdv -- Parabolic cylinder function D.
   pbvv -- Parabolic cylinder function V.
   pbwa -- Parabolic cylinder function W.

These are not universal functions:

.. autosummary::
   :toctree: generated/

   pbdv_seq -- Parabolic cylinder functions Dv(x) and derivatives.
   pbvv_seq -- Parabolic cylinder functions Vv(x) and derivatives.
   pbdn_seq -- Parabolic cylinder functions Dn(z) and derivatives.

Mathieu and related functions
-----------------------------

.. autosummary::
   :toctree: generated/

   mathieu_a -- Characteristic value of even Mathieu functions.
   mathieu_b -- Characteristic value of odd Mathieu functions.

These are not universal functions:

.. autosummary::
   :toctree: generated/

   mathieu_even_coef -- Fourier coefficients for even Mathieu and modified Mathieu functions.
   mathieu_odd_coef  -- Fourier coefficients for even Mathieu and modified Mathieu functions.

The following return both function and first derivative:

.. autosummary::
   :toctree: generated/

   mathieu_cem     -- Even Mathieu function and its derivative.
   mathieu_sem     -- Odd Mathieu function and its derivative.
   mathieu_modcem1 -- Even modified Mathieu function of the first kind and its derivative.
   mathieu_modcem2 -- Even modified Mathieu function of the second kind and its derivative.
   mathieu_modsem1 -- Odd modified Mathieu function of the first kind and its derivative.
   mathieu_modsem2 -- Odd modified Mathieu function of the second kind and its derivative.

Spheroidal wave functions
-------------------------

.. autosummary::
   :toctree: generated/

   pro_ang1   -- Prolate spheroidal angular function of the first kind and its derivative.
   pro_rad1   -- Prolate spheroidal radial function of the first kind and its derivative.
   pro_rad2   -- Prolate spheroidal radial function of the secon kind and its derivative.
   obl_ang1   -- Oblate spheroidal angular function of the first kind and its derivative.
   obl_rad1   -- Oblate spheroidal radial function of the first kind and its derivative.
   obl_rad2   -- Oblate spheroidal radial function of the second kind and its derivative.
   pro_cv     -- Characteristic value of prolate spheroidal function.
   obl_cv     -- Characteristic value of oblate spheroidal function.
   pro_cv_seq -- Characteristic values for prolate spheroidal wave functions.
   obl_cv_seq -- Characteristic values for oblate spheroidal wave functions.

The following functions require pre-computed characteristic value:

.. autosummary::
   :toctree: generated/

   pro_ang1_cv -- Prolate spheroidal angular function pro_ang1 for precomputed characteristic value.
   pro_rad1_cv -- Prolate spheroidal radial function pro_rad1 for precomputed characteristic value.
   pro_rad2_cv -- Prolate spheroidal radial function pro_rad2 for precomputed characteristic value.
   obl_ang1_cv -- Oblate spheroidal angular function obl_ang1 for precomputed characteristic value.
   obl_rad1_cv -- Oblate spheroidal radial function obl_rad1 for precomputed characteristic value.
   obl_rad2_cv -- Oblate spheroidal radial function obl_rad2 for precomputed characteristic value.

Kelvin functions
----------------

.. autosummary::
   :toctree: generated/

   kelvin       -- Kelvin functions as complex numbers.
   kelvin_zeros -- Compute nt zeros of all Kelvin functions.
   ber          -- Kelvin function ber.
   bei          -- Kelvin function bei
   berp         -- Derivative of the Kelvin function `ber`.
   beip         -- Derivative of the Kelvin function `bei`.
   ker          -- Kelvin function ker.
   kei          -- Kelvin function ker.
   kerp         -- Derivative of the Kelvin function ker.
   keip         -- Derivative of the Kelvin function kei.

These are not universal functions:

.. autosummary::
   :toctree: generated/

   ber_zeros  -- Compute nt zeros of the Kelvin function ber(x).
   bei_zeros  -- Compute nt zeros of the Kelvin function bei(x).
   berp_zeros -- Compute nt zeros of the Kelvin function ber'(x).
   beip_zeros -- Compute nt zeros of the Kelvin function bei'(x).
   ker_zeros  -- Compute nt zeros of the Kelvin function ker(x).
   kei_zeros  -- Compute nt zeros of the Kelvin function kei(x).
   kerp_zeros -- Compute nt zeros of the Kelvin function ker'(x).
   keip_zeros -- Compute nt zeros of the Kelvin function kei'(x).

Combinatorics
-------------

.. autosummary::
   :toctree: generated/

   comb -- The number of combinations of N things taken k at a time.
   perm -- Permutations of N things taken k at a time, i.e., k-permutations of N.

Lambert W and related functions
-------------------------------

.. autosummary::
   :toctree: generated/

   lambertw    -- Lambert W function.
   wrightomega -- Wright Omega function.

Other special functions
-----------------------

.. autosummary::
   :toctree: generated/

   agm         -- Arithmetic, Geometric Mean.
   bernoulli   -- Bernoulli numbers B0..Bn (inclusive).
   binom       -- Binomial coefficient
   diric       -- Periodic sinc function, also called the Dirichlet function.
   euler       -- Euler numbers E0..En (inclusive).
   expn        -- Exponential integral E_n.
   exp1        -- Exponential integral E_1 of complex argument z.
   expi        -- Exponential integral Ei.
   factorial   -- The factorial of a number or array of numbers.
   factorial2  -- Double factorial.
   factorialk  -- Multifactorial of n of order k, n(!!...!).
   shichi      -- Hyperbolic sine and cosine integrals.
   sici        -- Sine and cosine integrals.
   softmax     -- Softmax function.
   log_softmax -- Logarithm of softmax function.
   spence      -- Spence's function, also known as the dilogarithm.
   zeta        -- Riemann zeta function.
   zetac       -- Riemann zeta function minus 1.

Convenience functions
---------------------

.. autosummary::
   :toctree: generated/

   cbrt      -- Cube root of `x`.
   exp10     -- 10**x.
   exp2      -- 2**x.
   radian    -- Convert from degrees to radians.
   cosdg     -- Cosine of the angle `x` given in degrees.
   sindg     -- Sine of angle given in degrees.
   tandg     -- Tangent of angle x given in degrees.
   cotdg     -- Cotangent of the angle `x` given in degrees.
   log1p     -- Calculates log(1+x) for use when `x` is near zero.
   expm1     -- exp(x) - 1 for use when `x` is near zero.
   cosm1     -- cos(x) - 1 for use when `x` is near zero.
   round     -- Round to nearest integer.
   xlogy     -- Compute ``x*log(y)`` so that the result is 0 if ``x = 0``.
   xlog1py   -- Compute ``x*log1p(y)`` so that the result is 0 if ``x = 0``.
   logsumexp -- Compute the log of the sum of exponentials of input elements.
   exprel    -- Relative error exponential, (exp(x)-1)/x, for use when `x` is near zero.
   sinc      -- Return the sinc function.

"""

from .sf_error import SpecialFunctionWarning, SpecialFunctionError

from . import _ufuncs
from ._ufuncs import *

from . import _basic
from ._basic import *

from ._logsumexp import logsumexp, softmax, log_softmax

from . import orthogonal
from .orthogonal import *

from .spfun_stats import multigammaln
from ._ellip_harm import (
    ellip_harm,
    ellip_harm_2,
    ellip_normal
)
from ._lambertw import lambertw
from ._spherical_bessel import (
    spherical_jn,
    spherical_yn,
    spherical_in,
    spherical_kn
)

__all__ = _ufuncs.__all__ + _basic.__all__ + orthogonal.__all__ + [
    'SpecialFunctionWarning',
    'SpecialFunctionError',
    'orthogonal',  # Not public, but kept in __all__ for back-compat
    'logsumexp',
    'softmax',
    'log_softmax',
    'multigammaln',
    'ellip_harm',
    'ellip_harm_2',
    'ellip_normal',
    'lambertw',
    'spherical_jn',
    'spherical_yn',
    'spherical_in',
    'spherical_kn',
]

from scipy._lib._testutils import PytestTester
test = PytestTester(__name__)
del PytestTester
