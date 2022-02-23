try:
    import mpmath as mp
except ImportError:
    pass

try:
    from sympy.abc import x  # type: ignore[import]
except ImportError:
    pass


def lagrange_inversion(a):
    """Given a series

    f(x) = a[1]*x + a[2]*x**2 + ... + a[n-1]*x**(n - 1),

    use the Lagrange inversion formula to compute a series

    g(x) = b[1]*x + b[2]*x**2 + ... + b[n-1]*x**(n - 1)

    so that f(g(x)) = g(f(x)) = x mod x**n. We must have a[0] = 0, so
    necessarily b[0] = 0 too.

    The algorithm is naive and could be improved, but speed isn't an
    issue here and it's easy to read.

    """
    n = len(a)
    f = sum(a[i]*x**i for i in range(len(a)))
    h = (x/f).series(x, 0, n).removeO()
    hpower = [h**0]
    for k in range(n):
        hpower.append((hpower[-1]*h).expand())
    b = [mp.mpf(0)]
    for k in range(1, n):
        b.append(hpower[k].coeff(x, k - 1)/k)
    b = map(lambda x: mp.mpf(x), b)
    return b
