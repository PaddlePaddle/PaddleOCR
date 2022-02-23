"""
===================================================================
Statistical functions for masked arrays (:mod:`scipy.stats.mstats`)
===================================================================

.. currentmodule:: scipy.stats.mstats

This module contains a large number of statistical functions that can
be used with masked arrays.

Most of these functions are similar to those in `scipy.stats` but might
have small differences in the API or in the algorithm used. Since this
is a relatively new package, some API changes are still possible.

Summary statistics
==================

.. autosummary::
   :toctree: generated/

   describe
   gmean
   hmean
   kurtosis
   mode
   mquantiles
   hdmedian
   hdquantiles
   hdquantiles_sd
   idealfourths
   plotting_positions
   meppf
   moment
   skew
   tmean
   tvar
   tmin
   tmax
   tsem
   variation
   find_repeats
   sem
   trimmed_mean
   trimmed_mean_ci
   trimmed_std
   trimmed_var

Frequency statistics
====================

.. autosummary::
   :toctree: generated/

   scoreatpercentile

Correlation functions
=====================

.. autosummary::
   :toctree: generated/

   f_oneway
   pearsonr
   spearmanr
   pointbiserialr
   kendalltau
   kendalltau_seasonal
   linregress
   siegelslopes
   theilslopes
   sen_seasonal_slopes

Statistical tests
=================

.. autosummary::
   :toctree: generated/

   ttest_1samp
   ttest_onesamp
   ttest_ind
   ttest_rel
   chisquare
   kstest
   ks_2samp
   ks_1samp
   ks_twosamp
   mannwhitneyu
   rankdata
   kruskal
   kruskalwallis
   friedmanchisquare
   brunnermunzel
   skewtest
   kurtosistest
   normaltest

Transformations
===============

.. autosummary::
   :toctree: generated/

   obrientransform
   trim
   trima
   trimmed_stde
   trimr
   trimtail
   trimboth
   winsorize
   zmap
   zscore

Other
=====

.. autosummary::
   :toctree: generated/

   argstoarray
   count_tied_groups
   msign
   compare_medians_ms
   median_cihs
   mjci
   mquantiles_cimj
   rsh

"""
from .mstats_basic import *
from .mstats_extras import *
# Functions that support masked array input in stats but need to be kept in the
# mstats namespace for backwards compatibility:
from scipy.stats import gmean, hmean, zmap, zscore, chisquare
