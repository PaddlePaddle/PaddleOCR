from ._ufuncs import (_spherical_jn, _spherical_yn, _spherical_in,
                      _spherical_kn, _spherical_jn_d, _spherical_yn_d,
                      _spherical_in_d, _spherical_kn_d)

def spherical_jn(n, z, derivative=False):
    r"""Spherical Bessel function of the first kind or its derivative.

    Defined as [1]_,

    .. math:: j_n(z) = \sqrt{\frac{\pi}{2z}} J_{n + 1/2}(z),

    where :math:`J_n` is the Bessel function of the first kind.

    Parameters
    ----------
    n : int, array_like
        Order of the Bessel function (n >= 0).
    z : complex or float, array_like
        Argument of the Bessel function.
    derivative : bool, optional
        If True, the value of the derivative (rather than the function
        itself) is returned.

    Returns
    -------
    jn : ndarray

    Notes
    -----
    For real arguments greater than the order, the function is computed
    using the ascending recurrence [2]_. For small real or complex
    arguments, the definitional relation to the cylindrical Bessel function
    of the first kind is used.

    The derivative is computed using the relations [3]_,

    .. math::
        j_n'(z) = j_{n-1}(z) - \frac{n + 1}{z} j_n(z).

        j_0'(z) = -j_1(z)


    .. versionadded:: 0.18.0

    References
    ----------
    .. [1] https://dlmf.nist.gov/10.47.E3
    .. [2] https://dlmf.nist.gov/10.51.E1
    .. [3] https://dlmf.nist.gov/10.51.E2
    """
    if derivative:
        return _spherical_jn_d(n, z)
    else:
        return _spherical_jn(n, z)


def spherical_yn(n, z, derivative=False):
    r"""Spherical Bessel function of the second kind or its derivative.

    Defined as [1]_,

    .. math:: y_n(z) = \sqrt{\frac{\pi}{2z}} Y_{n + 1/2}(z),

    where :math:`Y_n` is the Bessel function of the second kind.

    Parameters
    ----------
    n : int, array_like
        Order of the Bessel function (n >= 0).
    z : complex or float, array_like
        Argument of the Bessel function.
    derivative : bool, optional
        If True, the value of the derivative (rather than the function
        itself) is returned.

    Returns
    -------
    yn : ndarray

    Notes
    -----
    For real arguments, the function is computed using the ascending
    recurrence [2]_.  For complex arguments, the definitional relation to
    the cylindrical Bessel function of the second kind is used.

    The derivative is computed using the relations [3]_,

    .. math::
        y_n' = y_{n-1} - \frac{n + 1}{z} y_n.

        y_0' = -y_1


    .. versionadded:: 0.18.0

    References
    ----------
    .. [1] https://dlmf.nist.gov/10.47.E4
    .. [2] https://dlmf.nist.gov/10.51.E1
    .. [3] https://dlmf.nist.gov/10.51.E2
    """
    if derivative:
        return _spherical_yn_d(n, z)
    else:
        return _spherical_yn(n, z)


def spherical_in(n, z, derivative=False):
    r"""Modified spherical Bessel function of the first kind or its derivative.

    Defined as [1]_,

    .. math:: i_n(z) = \sqrt{\frac{\pi}{2z}} I_{n + 1/2}(z),

    where :math:`I_n` is the modified Bessel function of the first kind.

    Parameters
    ----------
    n : int, array_like
        Order of the Bessel function (n >= 0).
    z : complex or float, array_like
        Argument of the Bessel function.
    derivative : bool, optional
        If True, the value of the derivative (rather than the function
        itself) is returned.

    Returns
    -------
    in : ndarray

    Notes
    -----
    The function is computed using its definitional relation to the
    modified cylindrical Bessel function of the first kind.

    The derivative is computed using the relations [2]_,

    .. math::
        i_n' = i_{n-1} - \frac{n + 1}{z} i_n.

        i_1' = i_0


    .. versionadded:: 0.18.0

    References
    ----------
    .. [1] https://dlmf.nist.gov/10.47.E7
    .. [2] https://dlmf.nist.gov/10.51.E5
    """
    if derivative:
        return _spherical_in_d(n, z)
    else:
        return _spherical_in(n, z)


def spherical_kn(n, z, derivative=False):
    r"""Modified spherical Bessel function of the second kind or its derivative.

    Defined as [1]_,

    .. math:: k_n(z) = \sqrt{\frac{\pi}{2z}} K_{n + 1/2}(z),

    where :math:`K_n` is the modified Bessel function of the second kind.

    Parameters
    ----------
    n : int, array_like
        Order of the Bessel function (n >= 0).
    z : complex or float, array_like
        Argument of the Bessel function.
    derivative : bool, optional
        If True, the value of the derivative (rather than the function
        itself) is returned.

    Returns
    -------
    kn : ndarray

    Notes
    -----
    The function is computed using its definitional relation to the
    modified cylindrical Bessel function of the second kind.

    The derivative is computed using the relations [2]_,

    .. math::
        k_n' = -k_{n-1} - \frac{n + 1}{z} k_n.

        k_0' = -k_1


    .. versionadded:: 0.18.0

    References
    ----------
    .. [1] https://dlmf.nist.gov/10.47.E9
    .. [2] https://dlmf.nist.gov/10.51.E5
    """
    if derivative:
        return _spherical_kn_d(n, z)
    else:
        return _spherical_kn(n, z)
