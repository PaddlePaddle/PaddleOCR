"""
==============================================
Discrete Fourier transforms (:mod:`scipy.fft`)
==============================================

.. currentmodule:: scipy.fft

Fast Fourier Transforms (FFTs)
==============================

.. autosummary::
   :toctree: generated/

   fft - Fast (discrete) Fourier Transform (FFT)
   ifft - Inverse FFT
   fft2 - 2-D FFT
   ifft2 - 2-D inverse FFT
   fftn - N-D FFT
   ifftn - N-D inverse FFT
   rfft - FFT of strictly real-valued sequence
   irfft - Inverse of rfft
   rfft2 - 2-D FFT of real sequence
   irfft2 - Inverse of rfft2
   rfftn - N-D FFT of real sequence
   irfftn - Inverse of rfftn
   hfft - FFT of a Hermitian sequence (real spectrum)
   ihfft - Inverse of hfft
   hfft2 - 2-D FFT of a Hermitian sequence
   ihfft2 - Inverse of hfft2
   hfftn - N-D FFT of a Hermitian sequence
   ihfftn - Inverse of hfftn

Discrete Sin and Cosine Transforms (DST and DCT)
================================================
.. autosummary::
   :toctree: generated/

   dct - Discrete cosine transform
   idct - Inverse discrete cosine transform
   dctn - N-D Discrete cosine transform
   idctn - N-D Inverse discrete cosine transform
   dst - Discrete sine transform
   idst - Inverse discrete sine transform
   dstn - N-D Discrete sine transform
   idstn - N-D Inverse discrete sine transform

Fast Hankel Transforms
======================

.. autosummary::
   :toctree: generated/

   fht - Fast Hankel transform
   ifht - Inverse of fht

Helper functions
================

.. autosummary::
   :toctree: generated/

   fftshift - Shift the zero-frequency component to the center of the spectrum
   ifftshift - The inverse of `fftshift`
   fftfreq - Return the Discrete Fourier Transform sample frequencies
   rfftfreq - DFT sample frequencies (for usage with rfft, irfft)
   fhtoffset - Compute an optimal offset for the Fast Hankel Transform
   next_fast_len - Find the optimal length to zero-pad an FFT for speed
   set_workers - Context manager to set default number of workers
   get_workers - Get the current default number of workers

Backend control
===============

.. autosummary::
   :toctree: generated/

   set_backend - Context manager to set the backend within a fixed scope
   skip_backend - Context manager to skip a backend within a fixed scope
   set_global_backend - Sets the global fft backend
   register_backend - Register a backend for permanent use

"""

from ._basic import (
    fft, ifft, fft2, ifft2, fftn, ifftn,
    rfft, irfft, rfft2, irfft2, rfftn, irfftn,
    hfft, ihfft, hfft2, ihfft2, hfftn, ihfftn)
from ._realtransforms import dct, idct, dst, idst, dctn, idctn, dstn, idstn
from ._fftlog import fht, ifht, fhtoffset
from ._helper import next_fast_len
from ._backend import (set_backend, skip_backend, set_global_backend,
                       register_backend)
from numpy.fft import fftfreq, rfftfreq, fftshift, ifftshift
from ._pocketfft.helper import set_workers, get_workers

__all__ = [
    'fft', 'ifft', 'fft2','ifft2', 'fftn', 'ifftn',
    'rfft', 'irfft', 'rfft2', 'irfft2', 'rfftn', 'irfftn',
    'hfft', 'ihfft', 'hfft2', 'ihfft2', 'hfftn', 'ihfftn',
    'fftfreq', 'rfftfreq', 'fftshift', 'ifftshift',
    'next_fast_len',
    'dct', 'idct', 'dst', 'idst', 'dctn', 'idctn', 'dstn', 'idstn',
    'fht', 'ifht',
    'fhtoffset',
    'set_backend', 'skip_backend', 'set_global_backend', 'register_backend',
    'get_workers', 'set_workers']


from scipy._lib._testutils import PytestTester
test = PytestTester(__name__)
del PytestTester
