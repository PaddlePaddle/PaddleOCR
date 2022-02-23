"""
Functions for calculating the "distance" between colors.

Implicit in these definitions of "distance" is the notion of "Just Noticeable
Distance" (JND).  This represents the distance between colors where a human can
perceive different colors.  Humans are more sensitive to certain colors than
others, which different deltaE metrics correct for with varying degrees of
sophistication.

The literature often mentions 1 as the minimum distance for visual
differentiation, but more recent studies (Mahy 1994) peg JND at 2.3

The delta-E notation comes from the German word for "Sensation" (Empfindung).

Reference
---------
https://en.wikipedia.org/wiki/Color_difference

"""

import numpy as np

from .._shared.utils import _supported_float_type
from .colorconv import lab2lch, _cart2polar_2pi


def _float_inputs(lab1, lab2, allow_float32=True):
    lab1 = np.asarray(lab1)
    lab2 = np.asarray(lab2)
    if allow_float32:
        float_dtype = _supported_float_type([lab1.dtype, lab2.dtype])
    else:
        float_dtype = np.float64
    lab1 = lab1.astype(float_dtype, copy=False)
    lab2 = lab2.astype(float_dtype, copy=False)
    return lab1, lab2


def deltaE_cie76(lab1, lab2, channel_axis=-1):
    """Euclidean distance between two points in Lab color space

    Parameters
    ----------
    lab1 : array_like
        reference color (Lab colorspace)
    lab2 : array_like
        comparison color (Lab colorspace)
    channel_axis : int, optional
        This parameter indicates which axis of the arrays corresponds to
        channels.

        .. versionadded:: 0.19
           ``channel_axis`` was added in 0.19.

    Returns
    -------
    dE : array_like
        distance between colors `lab1` and `lab2`

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Color_difference
    .. [2] A. R. Robertson, "The CIE 1976 color-difference formulae,"
           Color Res. Appl. 2, 7-11 (1977).
    """
    lab1, lab2 = _float_inputs(lab1, lab2, allow_float32=True)
    L1, a1, b1 = np.moveaxis(lab1, source=channel_axis, destination=0)[:3]
    L2, a2, b2 = np.moveaxis(lab2, source=channel_axis, destination=0)[:3]
    return np.sqrt((L2 - L1) ** 2 + (a2 - a1) ** 2 + (b2 - b1) ** 2)


def deltaE_ciede94(lab1, lab2, kH=1, kC=1, kL=1, k1=0.045, k2=0.015, *,
                   channel_axis=-1):
    """Color difference according to CIEDE 94 standard

    Accommodates perceptual non-uniformities through the use of application
    specific scale factors (`kH`, `kC`, `kL`, `k1`, and `k2`).

    Parameters
    ----------
    lab1 : array_like
        reference color (Lab colorspace)
    lab2 : array_like
        comparison color (Lab colorspace)
    kH : float, optional
        Hue scale
    kC : float, optional
        Chroma scale
    kL : float, optional
        Lightness scale
    k1 : float, optional
        first scale parameter
    k2 : float, optional
        second scale parameter
    channel_axis : int, optional
        This parameter indicates which axis of the arrays corresponds to
        channels.

        .. versionadded:: 0.19
           ``channel_axis`` was added in 0.19.

    Returns
    -------
    dE : array_like
        color difference between `lab1` and `lab2`

    Notes
    -----
    deltaE_ciede94 is not symmetric with respect to lab1 and lab2.  CIEDE94
    defines the scales for the lightness, hue, and chroma in terms of the first
    color.  Consequently, the first color should be regarded as the "reference"
    color.

    `kL`, `k1`, `k2` depend on the application and default to the values
    suggested for graphic arts

    ==========  ==============  ==========
    Parameter    Graphic Arts    Textiles
    ==========  ==============  ==========
    `kL`         1.000           2.000
    `k1`         0.045           0.048
    `k2`         0.015           0.014
    ==========  ==============  ==========

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Color_difference
    .. [2] http://www.brucelindbloom.com/index.html?Eqn_DeltaE_CIE94.html
    """
    lab1, lab2 = _float_inputs(lab1, lab2, allow_float32=True)
    lab1 = np.moveaxis(lab1, source=channel_axis, destination=0)
    lab2 = np.moveaxis(lab2, source=channel_axis, destination=0)

    L1, C1 = lab2lch(lab1, channel_axis=0)[:2]
    L2, C2 = lab2lch(lab2, channel_axis=0)[:2]

    dL = L1 - L2
    dC = C1 - C2
    dH2 = get_dH2(lab1, lab2, channel_axis=0)

    SL = 1
    SC = 1 + k1 * C1
    SH = 1 + k2 * C1

    dE2 = (dL / (kL * SL)) ** 2
    dE2 += (dC / (kC * SC)) ** 2
    dE2 += dH2 / (kH * SH) ** 2
    return np.sqrt(np.maximum(dE2, 0))


def deltaE_ciede2000(lab1, lab2, kL=1, kC=1, kH=1, *, channel_axis=-1):
    """Color difference as given by the CIEDE 2000 standard.

    CIEDE 2000 is a major revision of CIDE94.  The perceptual calibration is
    largely based on experience with automotive paint on smooth surfaces.

    Parameters
    ----------
    lab1 : array_like
        reference color (Lab colorspace)
    lab2 : array_like
        comparison color (Lab colorspace)
    kL : float (range), optional
        lightness scale factor, 1 for "acceptably close"; 2 for "imperceptible"
        see deltaE_cmc
    kC : float (range), optional
        chroma scale factor, usually 1
    kH : float (range), optional
        hue scale factor, usually 1
    channel_axis : int, optional
        This parameter indicates which axis of the arrays corresponds to
        channels.

        .. versionadded:: 0.19
           ``channel_axis`` was added in 0.19.

    Returns
    -------
    deltaE : array_like
        The distance between `lab1` and `lab2`

    Notes
    -----
    CIEDE 2000 assumes parametric weighting factors for the lightness, chroma,
    and hue (`kL`, `kC`, `kH` respectively).  These default to 1.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Color_difference
    .. [2] http://www.ece.rochester.edu/~gsharma/ciede2000/ciede2000noteCRNA.pdf
           :DOI:`10.1364/AO.33.008069`
    .. [3] M. Melgosa, J. Quesada, and E. Hita, "Uniformity of some recent
           color metrics tested with an accurate color-difference tolerance
           dataset," Appl. Opt. 33, 8069-8077 (1994).
    """
    lab1, lab2 = _float_inputs(lab1, lab2, allow_float32=True)

    channel_axis = channel_axis % lab1.ndim
    unroll = False
    if lab1.ndim == 1 and lab2.ndim == 1:
        unroll = True
        if lab1.ndim == 1:
            lab1 = lab1[None, :]
        if lab2.ndim == 1:
            lab2 = lab2[None, :]
        channel_axis += 1
    L1, a1, b1 = np.moveaxis(lab1, source=channel_axis, destination=0)[:3]
    L2, a2, b2 = np.moveaxis(lab2, source=channel_axis, destination=0)[:3]


    # distort `a` based on average chroma
    # then convert to lch coordines from distorted `a`
    # all subsequence calculations are in the new coordiantes
    # (often denoted "prime" in the literature)
    Cbar = 0.5 * (np.hypot(a1, b1) + np.hypot(a2, b2))
    c7 = Cbar ** 7
    G = 0.5 * (1 - np.sqrt(c7 / (c7 + 25 ** 7)))
    scale = 1 + G
    C1, h1 = _cart2polar_2pi(a1 * scale, b1)
    C2, h2 = _cart2polar_2pi(a2 * scale, b2)
    # recall that c, h are polar coordiantes.  c==r, h==theta

    # cide2000 has four terms to delta_e:
    # 1) Luminance term
    # 2) Hue term
    # 3) Chroma term
    # 4) hue Rotation term

    # lightness term
    Lbar = 0.5 * (L1 + L2)
    tmp = (Lbar - 50) ** 2
    SL = 1 + 0.015 * tmp / np.sqrt(20 + tmp)
    L_term = (L2 - L1) / (kL * SL)

    # chroma term
    Cbar = 0.5 * (C1 + C2)  # new coordiantes
    SC = 1 + 0.045 * Cbar
    C_term = (C2 - C1) / (kC * SC)

    # hue term
    h_diff = h2 - h1
    h_sum = h1 + h2
    CC = C1 * C2

    dH = h_diff.copy()
    dH[h_diff > np.pi] -= 2 * np.pi
    dH[h_diff < -np.pi] += 2 * np.pi
    dH[CC == 0.] = 0.  # if r == 0, dtheta == 0
    dH_term = 2 * np.sqrt(CC) * np.sin(dH / 2)

    Hbar = h_sum.copy()
    mask = np.logical_and(CC != 0., np.abs(h_diff) > np.pi)
    Hbar[mask * (h_sum < 2 * np.pi)] += 2 * np.pi
    Hbar[mask * (h_sum >= 2 * np.pi)] -= 2 * np.pi
    Hbar[CC == 0.] *= 2
    Hbar *= 0.5

    T = (1 -
         0.17 * np.cos(Hbar - np.deg2rad(30)) +
         0.24 * np.cos(2 * Hbar) +
         0.32 * np.cos(3 * Hbar + np.deg2rad(6)) -
         0.20 * np.cos(4 * Hbar - np.deg2rad(63))
         )
    SH = 1 + 0.015 * Cbar * T

    H_term = dH_term / (kH * SH)

    # hue rotation
    c7 = Cbar ** 7
    Rc = 2 * np.sqrt(c7 / (c7 + 25 ** 7))
    dtheta = np.deg2rad(30) * np.exp(-((np.rad2deg(Hbar) - 275) / 25) ** 2)
    R_term = -np.sin(2 * dtheta) * Rc * C_term * H_term

    # put it all together
    dE2 = L_term ** 2
    dE2 += C_term ** 2
    dE2 += H_term ** 2
    dE2 += R_term
    ans = np.sqrt(np.maximum(dE2, 0))
    if unroll:
        ans = ans[0]
    return ans


def deltaE_cmc(lab1, lab2, kL=1, kC=1, *, channel_axis=-1):
    """Color difference from the  CMC l:c standard.

    This color difference was developed by the Colour Measurement Committee
    (CMC) of the Society of Dyers and Colourists (United Kingdom). It is
    intended for use in the textile industry.

    The scale factors `kL`, `kC` set the weight given to differences in
    lightness and chroma relative to differences in hue.  The usual values are
    ``kL=2``, ``kC=1`` for "acceptability" and ``kL=1``, ``kC=1`` for
    "imperceptibility".  Colors with ``dE > 1`` are "different" for the given
    scale factors.

    Parameters
    ----------
    lab1 : array_like
        reference color (Lab colorspace)
    lab2 : array_like
        comparison color (Lab colorspace)
    channel_axis : int, optional
        This parameter indicates which axis of the arrays corresponds to
        channels.

        .. versionadded:: 0.19
           ``channel_axis`` was added in 0.19.

    Returns
    -------
    dE : array_like
        distance between colors `lab1` and `lab2`

    Notes
    -----
    deltaE_cmc the defines the scales for the lightness, hue, and chroma
    in terms of the first color.  Consequently
    ``deltaE_cmc(lab1, lab2) != deltaE_cmc(lab2, lab1)``

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Color_difference
    .. [2] http://www.brucelindbloom.com/index.html?Eqn_DeltaE_CIE94.html
    .. [3] F. J. J. Clarke, R. McDonald, and B. Rigg, "Modification to the
           JPC79 colour-difference formula," J. Soc. Dyers Colour. 100, 128-132
           (1984).
    """
    lab1, lab2 = _float_inputs(lab1, lab2, allow_float32=True)
    lab1 = np.moveaxis(lab1, source=channel_axis, destination=0)
    lab2 = np.moveaxis(lab2, source=channel_axis, destination=0)
    L1, C1, h1 = lab2lch(lab1, channel_axis=0)[:3]
    L2, C2, h2 = lab2lch(lab2, channel_axis=0)[:3]

    dC = C1 - C2
    dL = L1 - L2
    dH2 = get_dH2(lab1, lab2, channel_axis=0)

    T = np.where(np.logical_and(np.rad2deg(h1) >= 164, np.rad2deg(h1) <= 345),
                 0.56 + 0.2 * np.abs(np.cos(h1 + np.deg2rad(168))),
                 0.36 + 0.4 * np.abs(np.cos(h1 + np.deg2rad(35)))
                 )
    c1_4 = C1 ** 4
    F = np.sqrt(c1_4 / (c1_4 + 1900))

    SL = np.where(L1 < 16, 0.511, 0.040975 * L1 / (1. + 0.01765 * L1))
    SC = 0.638 + 0.0638 * C1 / (1. + 0.0131 * C1)
    SH = SC * (F * T + 1 - F)

    dE2 = (dL / (kL * SL)) ** 2
    dE2 += (dC / (kC * SC)) ** 2
    dE2 += dH2 / (SH ** 2)

    return np.sqrt(np.maximum(dE2, 0))


def get_dH2(lab1, lab2, *, channel_axis=-1):
    """squared hue difference term occurring in deltaE_cmc and deltaE_ciede94

    Despite its name, "dH" is not a simple difference of hue values.  We avoid
    working directly with the hue value, since differencing angles is
    troublesome.  The hue term is usually written as:
        c1 = sqrt(a1**2 + b1**2)
        c2 = sqrt(a2**2 + b2**2)
        term = (a1-a2)**2 + (b1-b2)**2 - (c1-c2)**2
        dH = sqrt(term)

    However, this has poor roundoff properties when a or b is dominant.
    Instead, ab is a vector with elements a and b.  The same dH term can be
    re-written as:
        |ab1-ab2|**2 - (|ab1| - |ab2|)**2
    and then simplified to:
        2*|ab1|*|ab2| - 2*dot(ab1, ab2)
    """
    # This function needs double precision internally for accuracy
    input_is_float_32 = _supported_float_type([lab1, lab2]) == np.float32
    lab1, lab2 = _float_inputs(lab1, lab2, allow_float32=False)

    a1, b1 = np.moveaxis(lab1, source=channel_axis, destination=0)[1:3]
    a2, b2 = np.moveaxis(lab2, source=channel_axis, destination=0)[1:3]

    # magnitude of (a, b) is the chroma
    C1 = np.hypot(a1, b1)
    C2 = np.hypot(a2, b2)

    term = (C1 * C2) - (a1 * a2 + b1 * b2)
    out = 2 * term
    if input_is_float_32:
        out = out.astype(np.float32)
    return out
