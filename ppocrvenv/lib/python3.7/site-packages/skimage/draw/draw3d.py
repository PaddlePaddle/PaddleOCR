import numpy as np
from scipy.special import (ellipkinc as ellip_F, ellipeinc as ellip_E)


def ellipsoid(a, b, c, spacing=(1., 1., 1.), levelset=False):
    """
    Generates ellipsoid with semimajor axes aligned with grid dimensions
    on grid with specified `spacing`.

    Parameters
    ----------
    a : float
        Length of semimajor axis aligned with x-axis.
    b : float
        Length of semimajor axis aligned with y-axis.
    c : float
        Length of semimajor axis aligned with z-axis.
    spacing : tuple of floats, length 3
        Spacing in (x, y, z) spatial dimensions.
    levelset : bool
        If True, returns the level set for this ellipsoid (signed level
        set about zero, with positive denoting interior) as np.float64.
        False returns a binarized version of said level set.

    Returns
    -------
    ellip : (N, M, P) array
        Ellipsoid centered in a correctly sized array for given `spacing`.
        Boolean dtype unless `levelset=True`, in which case a float array is
        returned with the level set above 0.0 representing the ellipsoid.

    """
    if (a <= 0) or (b <= 0) or (c <= 0):
        raise ValueError('Parameters a, b, and c must all be > 0')

    offset = np.r_[1, 1, 1] * np.r_[spacing]

    # Calculate limits, and ensure output volume is odd & symmetric
    low = np.ceil((- np.r_[a, b, c] - offset))
    high = np.floor((np.r_[a, b, c] + offset + 1))

    for dim in range(3):
        if (high[dim] - low[dim]) % 2 == 0:
            low[dim] -= 1
        num = np.arange(low[dim], high[dim], spacing[dim])
        if 0 not in num:
            low[dim] -= np.max(num[num < 0])

    # Generate (anisotropic) spatial grid
    x, y, z = np.mgrid[low[0]:high[0]:spacing[0],
                       low[1]:high[1]:spacing[1],
                       low[2]:high[2]:spacing[2]]

    if not levelset:
        arr = ((x / float(a)) ** 2 +
               (y / float(b)) ** 2 +
               (z / float(c)) ** 2) <= 1
    else:
        arr = ((x / float(a)) ** 2 +
               (y / float(b)) ** 2 +
               (z / float(c)) ** 2) - 1

    return arr


def ellipsoid_stats(a, b, c):
    """
    Calculates analytical surface area and volume for ellipsoid with
    semimajor axes aligned with grid dimensions of specified `spacing`.

    Parameters
    ----------
    a : float
        Length of semimajor axis aligned with x-axis.
    b : float
        Length of semimajor axis aligned with y-axis.
    c : float
        Length of semimajor axis aligned with z-axis.

    Returns
    -------
    vol : float
        Calculated volume of ellipsoid.
    surf : float
        Calculated surface area of ellipsoid.

    """
    if (a <= 0) or (b <= 0) or (c <= 0):
        raise ValueError('Parameters a, b, and c must all be > 0')

    # Calculate volume & surface area
    # Surface calculation requires a >= b >= c and a != c.
    abc = [a, b, c]
    abc.sort(reverse=True)
    a = abc[0]
    b = abc[1]
    c = abc[2]

    # Volume
    vol = 4 / 3. * np.pi * a * b * c

    # Analytical ellipsoid surface area
    phi = np.arcsin((1. - (c ** 2 / (a ** 2.))) ** 0.5)
    d = float((a ** 2 - c ** 2) ** 0.5)
    m = (a ** 2 * (b ** 2 - c ** 2) /
         float(b ** 2 * (a ** 2 - c ** 2)))
    F = ellip_F(phi, m)
    E = ellip_E(phi, m)

    surf = 2 * np.pi * (c ** 2 +
                        b * c ** 2 / d * F +
                        b * d * E)

    return vol, surf
