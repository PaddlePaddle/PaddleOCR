# Author: Pim Schellart
# 2010 - 2011

"""Tools for spectral analysis of unequally sampled signals."""

import numpy as np

#pythran export _lombscargle(float64[], float64[], float64[])
def _lombscargle(x, y, freqs):
    """
    _lombscargle(x, y, freqs)

    Computes the Lomb-Scargle periodogram.

    Parameters
    ----------
    x : array_like
        Sample times.
    y : array_like
        Measurement values (must be registered so the mean is zero).
    freqs : array_like
        Angular frequencies for output periodogram.

    Returns
    -------
    pgram : array_like
        Lomb-Scargle periodogram.

    Raises
    ------
    ValueError
        If the input arrays `x` and `y` do not have the same shape.

    See also
    --------
    lombscargle

    """

    # Check input sizes
    if x.shape != y.shape:
        raise ValueError("Input arrays do not have the same size.")

    # Create empty array for output periodogram
    pgram = np.empty_like(freqs)

    c = np.empty_like(x)
    s = np.empty_like(x)

    for i in range(freqs.shape[0]):

        xc = 0.
        xs = 0.
        cc = 0.
        ss = 0.
        cs = 0.

        c[:] = np.cos(freqs[i] * x)
        s[:] = np.sin(freqs[i] * x)

        for j in range(x.shape[0]):
            xc += y[j] * c[j]
            xs += y[j] * s[j]
            cc += c[j] * c[j]
            ss += s[j] * s[j]
            cs += c[j] * s[j]

        if freqs[i] == 0:
            raise ZeroDivisionError()

        tau = np.arctan2(2 * cs, cc - ss) / (2 * freqs[i])
        c_tau = np.cos(freqs[i] * tau)
        s_tau = np.sin(freqs[i] * tau)
        c_tau2 = c_tau * c_tau
        s_tau2 = s_tau * s_tau
        cs_tau = 2 * c_tau * s_tau

        pgram[i] = 0.5 * (((c_tau * xc + s_tau * xs)**2 / \
            (c_tau2 * cc + cs_tau * cs + s_tau2 * ss)) + \
            ((c_tau * xs - s_tau * xc)**2 / \
            (c_tau2 * ss - cs_tau * cs + s_tau2 * cc)))

    return pgram
