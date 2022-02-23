"""
.. _statsrefmanual:

==========================================
Statistical functions (:mod:`scipy.stats`)
==========================================

.. currentmodule:: scipy.stats

This module contains a large number of probability distributions,
summary and frequency statistics, correlation functions and statistical
tests, masked statistics, kernel density estimation, quasi-Monte Carlo
functionality, and more.

Statistics is a very large area, and there are topics that are out of scope
for SciPy and are covered by other packages. Some of the most important ones
are:

- `statsmodels <https://www.statsmodels.org/stable/index.html>`__:
  regression, linear models, time series analysis, extensions to topics
  also covered by ``scipy.stats``.
- `Pandas <https://pandas.pydata.org/>`__: tabular data, time series
  functionality, interfaces to other statistical languages.
- `PyMC3 <https://docs.pymc.io/>`__: Bayesian statistical
  modeling, probabilistic machine learning.
- `scikit-learn <https://scikit-learn.org/>`__: classification, regression,
  model selection.
- `Seaborn <https://seaborn.pydata.org/>`__: statistical data visualization.
- `rpy2 <https://rpy2.github.io/>`__: Python to R bridge.


Probability distributions
=========================

Each univariate distribution is an instance of a subclass of `rv_continuous`
(`rv_discrete` for discrete distributions):

.. autosummary::
   :toctree: generated/

   rv_continuous
   rv_discrete
   rv_histogram

Continuous distributions
------------------------

.. autosummary::
   :toctree: generated/

   alpha             -- Alpha
   anglit            -- Anglit
   arcsine           -- Arcsine
   argus             -- Argus
   beta              -- Beta
   betaprime         -- Beta Prime
   bradford          -- Bradford
   burr              -- Burr (Type III)
   burr12            -- Burr (Type XII)
   cauchy            -- Cauchy
   chi               -- Chi
   chi2              -- Chi-squared
   cosine            -- Cosine
   crystalball       -- Crystalball
   dgamma            -- Double Gamma
   dweibull          -- Double Weibull
   erlang            -- Erlang
   expon             -- Exponential
   exponnorm         -- Exponentially Modified Normal
   exponweib         -- Exponentiated Weibull
   exponpow          -- Exponential Power
   f                 -- F (Snecdor F)
   fatiguelife       -- Fatigue Life (Birnbaum-Saunders)
   fisk              -- Fisk
   foldcauchy        -- Folded Cauchy
   foldnorm          -- Folded Normal
   genlogistic       -- Generalized Logistic
   gennorm           -- Generalized normal
   genpareto         -- Generalized Pareto
   genexpon          -- Generalized Exponential
   genextreme        -- Generalized Extreme Value
   gausshyper        -- Gauss Hypergeometric
   gamma             -- Gamma
   gengamma          -- Generalized gamma
   genhalflogistic   -- Generalized Half Logistic
   genhyperbolic     -- Generalized Hyperbolic
   geninvgauss       -- Generalized Inverse Gaussian
   gilbrat           -- Gilbrat
   gompertz          -- Gompertz (Truncated Gumbel)
   gumbel_r          -- Right Sided Gumbel, Log-Weibull, Fisher-Tippett, Extreme Value Type I
   gumbel_l          -- Left Sided Gumbel, etc.
   halfcauchy        -- Half Cauchy
   halflogistic      -- Half Logistic
   halfnorm          -- Half Normal
   halfgennorm       -- Generalized Half Normal
   hypsecant         -- Hyperbolic Secant
   invgamma          -- Inverse Gamma
   invgauss          -- Inverse Gaussian
   invweibull        -- Inverse Weibull
   johnsonsb         -- Johnson SB
   johnsonsu         -- Johnson SU
   kappa4            -- Kappa 4 parameter
   kappa3            -- Kappa 3 parameter
   ksone             -- Distribution of Kolmogorov-Smirnov one-sided test statistic
   kstwo             -- Distribution of Kolmogorov-Smirnov two-sided test statistic
   kstwobign         -- Limiting Distribution of scaled Kolmogorov-Smirnov two-sided test statistic.
   laplace           -- Laplace
   laplace_asymmetric    -- Asymmetric Laplace
   levy              -- Levy
   levy_l
   levy_stable
   logistic          -- Logistic
   loggamma          -- Log-Gamma
   loglaplace        -- Log-Laplace (Log Double Exponential)
   lognorm           -- Log-Normal
   loguniform        -- Log-Uniform
   lomax             -- Lomax (Pareto of the second kind)
   maxwell           -- Maxwell
   mielke            -- Mielke's Beta-Kappa
   moyal             -- Moyal
   nakagami          -- Nakagami
   ncx2              -- Non-central chi-squared
   ncf               -- Non-central F
   nct               -- Non-central Student's T
   norm              -- Normal (Gaussian)
   norminvgauss      -- Normal Inverse Gaussian
   pareto            -- Pareto
   pearson3          -- Pearson type III
   powerlaw          -- Power-function
   powerlognorm      -- Power log normal
   powernorm         -- Power normal
   rdist             -- R-distribution
   rayleigh          -- Rayleigh
   rice              -- Rice
   recipinvgauss     -- Reciprocal Inverse Gaussian
   semicircular      -- Semicircular
   skewcauchy        -- Skew Cauchy
   skewnorm          -- Skew normal
   studentized_range    -- Studentized Range
   t                 -- Student's T
   trapezoid         -- Trapezoidal
   triang            -- Triangular
   truncexpon        -- Truncated Exponential
   truncnorm         -- Truncated Normal
   tukeylambda       -- Tukey-Lambda
   uniform           -- Uniform
   vonmises          -- Von-Mises (Circular)
   vonmises_line     -- Von-Mises (Line)
   wald              -- Wald
   weibull_min       -- Minimum Weibull (see Frechet)
   weibull_max       -- Maximum Weibull (see Frechet)
   wrapcauchy        -- Wrapped Cauchy

Multivariate distributions
--------------------------

.. autosummary::
   :toctree: generated/

   multivariate_normal    -- Multivariate normal distribution
   matrix_normal          -- Matrix normal distribution
   dirichlet              -- Dirichlet
   wishart                -- Wishart
   invwishart             -- Inverse Wishart
   multinomial            -- Multinomial distribution
   special_ortho_group    -- SO(N) group
   ortho_group            -- O(N) group
   unitary_group          -- U(N) group
   random_correlation     -- random correlation matrices
   multivariate_t         -- Multivariate t-distribution
   multivariate_hypergeom -- Multivariate hypergeometric distribution

Discrete distributions
----------------------

.. autosummary::
   :toctree: generated/

   bernoulli                -- Bernoulli
   betabinom                -- Beta-Binomial
   binom                    -- Binomial
   boltzmann                -- Boltzmann (Truncated Discrete Exponential)
   dlaplace                 -- Discrete Laplacian
   geom                     -- Geometric
   hypergeom                -- Hypergeometric
   logser                   -- Logarithmic (Log-Series, Series)
   nbinom                   -- Negative Binomial
   nchypergeom_fisher       -- Fisher's Noncentral Hypergeometric
   nchypergeom_wallenius    -- Wallenius's Noncentral Hypergeometric
   nhypergeom               -- Negative Hypergeometric
   planck                   -- Planck (Discrete Exponential)
   poisson                  -- Poisson
   randint                  -- Discrete Uniform
   skellam                  -- Skellam
   yulesimon                -- Yule-Simon
   zipf                     -- Zipf (Zeta)
   zipfian                  -- Zipfian

An overview of statistical functions is given below.  Many of these functions
have a similar version in `scipy.stats.mstats` which work for masked arrays.

Summary statistics
==================

.. autosummary::
   :toctree: generated/

   describe          -- Descriptive statistics
   gmean             -- Geometric mean
   hmean             -- Harmonic mean
   kurtosis          -- Fisher or Pearson kurtosis
   mode              -- Modal value
   moment            -- Central moment
   skew              -- Skewness
   kstat             --
   kstatvar          --
   tmean             -- Truncated arithmetic mean
   tvar              -- Truncated variance
   tmin              --
   tmax              --
   tstd              --
   tsem              --
   variation         -- Coefficient of variation
   find_repeats
   trim_mean
   gstd              -- Geometric Standard Deviation
   iqr
   sem
   bayes_mvs
   mvsdist
   entropy
   differential_entropy
   median_absolute_deviation
   median_abs_deviation
   bootstrap

Frequency statistics
====================

.. autosummary::
   :toctree: generated/

   cumfreq
   itemfreq
   percentileofscore
   scoreatpercentile
   relfreq

.. autosummary::
   :toctree: generated/

   binned_statistic     -- Compute a binned statistic for a set of data.
   binned_statistic_2d  -- Compute a 2-D binned statistic for a set of data.
   binned_statistic_dd  -- Compute a d-D binned statistic for a set of data.

Correlation functions
=====================

.. autosummary::
   :toctree: generated/

   f_oneway
   alexandergovern
   pearsonr
   spearmanr
   pointbiserialr
   kendalltau
   weightedtau
   somersd
   linregress
   siegelslopes
   theilslopes
   multiscale_graphcorr

Statistical tests
=================

.. autosummary::
   :toctree: generated/

   ttest_1samp
   ttest_ind
   ttest_ind_from_stats
   ttest_rel
   chisquare
   cramervonmises
   cramervonmises_2samp
   power_divergence
   kstest
   ks_1samp
   ks_2samp
   epps_singleton_2samp
   mannwhitneyu
   tiecorrect
   rankdata
   ranksums
   wilcoxon
   kruskal
   friedmanchisquare
   brunnermunzel
   combine_pvalues
   jarque_bera
   page_trend_test

.. autosummary::
   :toctree: generated/

   ansari
   bartlett
   levene
   shapiro
   anderson
   anderson_ksamp
   binom_test
   binomtest
   fligner
   median_test
   mood
   skewtest
   kurtosistest
   normaltest


Quasi-Monte Carlo
=================

.. toctree::
   :maxdepth: 4

   stats.qmc


Masked statistics functions
===========================

.. toctree::

   stats.mstats


Other statistical functionality
===============================

Transformations
---------------

.. autosummary::
   :toctree: generated/

   boxcox
   boxcox_normmax
   boxcox_llf
   yeojohnson
   yeojohnson_normmax
   yeojohnson_llf
   obrientransform
   sigmaclip
   trimboth
   trim1
   zmap
   zscore

Statistical distances
---------------------

.. autosummary::
   :toctree: generated/

   wasserstein_distance
   energy_distance

Random variate generation / CDF Inversion
=========================================

.. autosummary::
   :toctree: generated/

   rvs_ratio_uniforms
   NumericalInverseHermite

Circular statistical functions
------------------------------

.. autosummary::
   :toctree: generated/

   circmean
   circvar
   circstd

Contingency table functions
---------------------------

.. autosummary::
   :toctree: generated/

   chi2_contingency
   contingency.crosstab
   contingency.expected_freq
   contingency.margins
   contingency.relative_risk
   contingency.association
   fisher_exact
   barnard_exact
   boschloo_exact

Plot-tests
----------

.. autosummary::
   :toctree: generated/

   ppcc_max
   ppcc_plot
   probplot
   boxcox_normplot
   yeojohnson_normplot

Univariate and multivariate kernel density estimation
-----------------------------------------------------

.. autosummary::
   :toctree: generated/

   gaussian_kde

Warnings used in :mod:`scipy.stats`
-----------------------------------

.. autosummary::
   :toctree: generated/

   F_onewayConstantInputWarning
   F_onewayBadInputSizesWarning
   PearsonRConstantInputWarning
   PearsonRNearConstantInputWarning
   SpearmanRConstantInputWarning

"""

from .stats import *
from .distributions import *
from .morestats import *
from ._binomtest import binomtest
from ._binned_statistic import *
from .kde import gaussian_kde
from . import mstats
from . import qmc
from ._multivariate import *
from . import contingency
from .contingency import chi2_contingency
from ._bootstrap import bootstrap
from ._entropy import *
from ._hypotests import *
from ._rvs_sampling import rvs_ratio_uniforms, NumericalInverseHermite
from ._page_trend_test import page_trend_test
from ._mannwhitneyu import mannwhitneyu

__all__ = [s for s in dir() if not s.startswith("_")]  # Remove dunders.

from scipy._lib._testutils import PytestTester
test = PytestTester(__name__)
del PytestTester
