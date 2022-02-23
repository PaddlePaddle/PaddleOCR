# Copyright (c) 2017 The PyWavelets Developers
#                    <https://github.com/PyWavelets/pywt>
# See COPYING for license details.
import inspect
import numpy as np
import sys
from collections.abc import Iterable

from ._extensions._pywt import (Wavelet, ContinuousWavelet,
                                DiscreteContinuousWavelet, Modes)


# define string_types as in six for Python 2/3 compatibility
if sys.version_info[0] == 3:
    string_types = str,
else:
    string_types = basestring,


def _as_wavelet(wavelet):
    """Convert wavelet name to a Wavelet object."""
    if not isinstance(wavelet, (ContinuousWavelet, Wavelet)):
        wavelet = DiscreteContinuousWavelet(wavelet)
    if isinstance(wavelet, ContinuousWavelet):
        raise ValueError(
            "A ContinuousWavelet object was provided, but only discrete "
            "Wavelet objects are supported by this function.  A list of all "
            "supported discrete wavelets can be obtained by running:\n"
            "print(pywt.wavelist(kind='discrete'))")
    return wavelet


def _wavelets_per_axis(wavelet, axes):
    """Initialize Wavelets for each axis to be transformed.

    Parameters
    ----------
    wavelet : Wavelet or tuple of Wavelets
        If a single Wavelet is provided, it will used for all axes.  Otherwise
        one Wavelet per axis must be provided.
    axes : list
        The tuple of axes to be transformed.

    Returns
    -------
    wavelets : list of Wavelet objects
        A tuple of Wavelets equal in length to ``axes``.

    """
    axes = tuple(axes)
    if isinstance(wavelet, string_types + (Wavelet, )):
        # same wavelet on all axes
        wavelets = [_as_wavelet(wavelet), ] * len(axes)
    elif isinstance(wavelet, Iterable):
        # (potentially) unique wavelet per axis (e.g. for dual-tree DWT)
        if len(wavelet) == 1:
            wavelets = [_as_wavelet(wavelet[0]), ] * len(axes)
        else:
            if len(wavelet) != len(axes):
                raise ValueError((
                    "The number of wavelets must match the number of axes "
                    "to be transformed."))
            wavelets = [_as_wavelet(w) for w in wavelet]
    else:
        raise ValueError("wavelet must be a str, Wavelet or iterable")
    return wavelets


def _modes_per_axis(modes, axes):
    """Initialize mode for each axis to be transformed.

    Parameters
    ----------
    modes : str or tuple of strings
        If a single mode is provided, it will used for all axes.  Otherwise
        one mode per axis must be provided.
    axes : tuple
        The tuple of axes to be transformed.

    Returns
    -------
    modes : tuple of int
        A tuple of Modes equal in length to ``axes``.

    """
    axes = tuple(axes)
    if isinstance(modes, string_types + (int, )):
        # same wavelet on all axes
        modes = [Modes.from_object(modes), ] * len(axes)
    elif isinstance(modes, Iterable):
        if len(modes) == 1:
            modes = [Modes.from_object(modes[0]), ] * len(axes)
        else:
            # (potentially) unique wavelet per axis (e.g. for dual-tree DWT)
            if len(modes) != len(axes):
                raise ValueError(("The number of modes must match the number "
                                  "of axes to be transformed."))
        modes = [Modes.from_object(mode) for mode in modes]
    else:
        raise ValueError("modes must be a str, Mode enum or iterable")
    return modes
