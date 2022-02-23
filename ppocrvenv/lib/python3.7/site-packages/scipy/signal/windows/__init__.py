"""
Window functions (:mod:`scipy.signal.windows`)
==============================================

The suite of window functions for filtering and spectral estimation.

.. currentmodule:: scipy.signal.windows

.. autosummary::
   :toctree: generated/

   get_window        -- Return a window of a given length and type.

   barthann          -- Bartlett-Hann window
   bartlett          -- Bartlett window
   blackman          -- Blackman window
   blackmanharris    -- Minimum 4-term Blackman-Harris window
   bohman            -- Bohman window
   boxcar            -- Boxcar window
   chebwin           -- Dolph-Chebyshev window
   cosine            -- Cosine window
   dpss              -- Discrete prolate spheroidal sequences
   exponential       -- Exponential window
   flattop           -- Flat top window
   gaussian          -- Gaussian window
   general_cosine    -- Generalized Cosine window
   general_gaussian  -- Generalized Gaussian window
   general_hamming   -- Generalized Hamming window
   hamming           -- Hamming window
   hann              -- Hann window
   hanning           -- Hann window
   kaiser            -- Kaiser window
   nuttall           -- Nuttall's minimum 4-term Blackman-Harris window
   parzen            -- Parzen window
   taylor            -- Taylor window
   triang            -- Triangular window
   tukey             -- Tukey window

"""

from .windows import *

__all__ = ['boxcar', 'triang', 'parzen', 'bohman', 'blackman', 'nuttall',
           'blackmanharris', 'flattop', 'bartlett', 'hanning', 'barthann',
           'hamming', 'kaiser', 'gaussian', 'general_gaussian', 'general_cosine',
           'general_hamming', 'chebwin', 'cosine', 'hann',
           'exponential', 'tukey', 'taylor', 'get_window', 'dpss']
