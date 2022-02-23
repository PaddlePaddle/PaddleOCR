# Author: Eric Larson
# 2014

"""Tools for MLS generation"""

import numpy as np

from ._max_len_seq_inner import _max_len_seq_inner

__all__ = ['max_len_seq']


# These are definitions of linear shift register taps for use in max_len_seq()
_mls_taps = {2: [1], 3: [2], 4: [3], 5: [3], 6: [5], 7: [6], 8: [7, 6, 1],
             9: [5], 10: [7], 11: [9], 12: [11, 10, 4], 13: [12, 11, 8],
             14: [13, 12, 2], 15: [14], 16: [15, 13, 4], 17: [14],
             18: [11], 19: [18, 17, 14], 20: [17], 21: [19], 22: [21],
             23: [18], 24: [23, 22, 17], 25: [22], 26: [25, 24, 20],
             27: [26, 25, 22], 28: [25], 29: [27], 30: [29, 28, 7],
             31: [28], 32: [31, 30, 10]}

def max_len_seq(nbits, state=None, length=None, taps=None):
    """
    Maximum length sequence (MLS) generator.

    Parameters
    ----------
    nbits : int
        Number of bits to use. Length of the resulting sequence will
        be ``(2**nbits) - 1``. Note that generating long sequences
        (e.g., greater than ``nbits == 16``) can take a long time.
    state : array_like, optional
        If array, must be of length ``nbits``, and will be cast to binary
        (bool) representation. If None, a seed of ones will be used,
        producing a repeatable representation. If ``state`` is all
        zeros, an error is raised as this is invalid. Default: None.
    length : int, optional
        Number of samples to compute. If None, the entire length
        ``(2**nbits) - 1`` is computed.
    taps : array_like, optional
        Polynomial taps to use (e.g., ``[7, 6, 1]`` for an 8-bit sequence).
        If None, taps will be automatically selected (for up to
        ``nbits == 32``).

    Returns
    -------
    seq : array
        Resulting MLS sequence of 0's and 1's.
    state : array
        The final state of the shift register.

    Notes
    -----
    The algorithm for MLS generation is generically described in:

        https://en.wikipedia.org/wiki/Maximum_length_sequence

    The default values for taps are specifically taken from the first
    option listed for each value of ``nbits`` in:

        https://web.archive.org/web/20181001062252/http://www.newwaveinstruments.com/resources/articles/m_sequence_linear_feedback_shift_register_lfsr.htm

    .. versionadded:: 0.15.0

    Examples
    --------
    MLS uses binary convention:

    >>> from scipy.signal import max_len_seq
    >>> max_len_seq(4)[0]
    array([1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0], dtype=int8)

    MLS has a white spectrum (except for DC):

    >>> import matplotlib.pyplot as plt
    >>> from numpy.fft import fft, ifft, fftshift, fftfreq
    >>> seq = max_len_seq(6)[0]*2-1  # +1 and -1
    >>> spec = fft(seq)
    >>> N = len(seq)
    >>> plt.plot(fftshift(fftfreq(N)), fftshift(np.abs(spec)), '.-')
    >>> plt.margins(0.1, 0.1)
    >>> plt.grid(True)
    >>> plt.show()

    Circular autocorrelation of MLS is an impulse:

    >>> acorrcirc = ifft(spec * np.conj(spec)).real
    >>> plt.figure()
    >>> plt.plot(np.arange(-N/2+1, N/2+1), fftshift(acorrcirc), '.-')
    >>> plt.margins(0.1, 0.1)
    >>> plt.grid(True)
    >>> plt.show()

    Linear autocorrelation of MLS is approximately an impulse:

    >>> acorr = np.correlate(seq, seq, 'full')
    >>> plt.figure()
    >>> plt.plot(np.arange(-N+1, N), acorr, '.-')
    >>> plt.margins(0.1, 0.1)
    >>> plt.grid(True)
    >>> plt.show()

    """
    if taps is None:
        if nbits not in _mls_taps:
            known_taps = np.array(list(_mls_taps.keys()))
            raise ValueError('nbits must be between %s and %s if taps is None'
                             % (known_taps.min(), known_taps.max()))
        taps = np.array(_mls_taps[nbits], np.intp)
    else:
        taps = np.unique(np.array(taps, np.intp))[::-1]
        if np.any(taps < 0) or np.any(taps > nbits) or taps.size < 1:
            raise ValueError('taps must be non-empty with values between '
                             'zero and nbits (inclusive)')
        taps = np.array(taps)  # needed for Cython and Pythran
    n_max = (2**nbits) - 1
    if length is None:
        length = n_max
    else:
        length = int(length)
        if length < 0:
            raise ValueError('length must be greater than or equal to 0')
    # We use int8 instead of bool here because NumPy arrays of bools
    # don't seem to work nicely with Cython
    if state is None:
        state = np.ones(nbits, dtype=np.int8, order='c')
    else:
        # makes a copy if need be, ensuring it's 0's and 1's
        state = np.array(state, dtype=bool, order='c').astype(np.int8)
    if state.ndim != 1 or state.size != nbits:
        raise ValueError('state must be a 1-D array of size nbits')
    if np.all(state == 0):
        raise ValueError('state must not be all zeros')

    seq = np.empty(length, dtype=np.int8, order='c')
    state = _max_len_seq_inner(taps, state, nbits, length, seq)
    return seq, state
