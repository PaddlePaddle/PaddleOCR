"""Compute the Taylor series for zeta(x) - 1 around x = 0."""
try:
    import mpmath
except ImportError:
    pass


def zetac_series(N):
    coeffs = []
    with mpmath.workdps(100):
        coeffs.append(-1.5)
        for n in range(1, N):
            coeff = mpmath.diff(mpmath.zeta, 0, n)/mpmath.factorial(n)
            coeffs.append(coeff)
    return coeffs


def main():
    print(__doc__)
    coeffs = zetac_series(10)
    coeffs = [mpmath.nstr(x, 20, min_fixed=0, max_fixed=0)
              for x in coeffs]
    print("\n".join(coeffs[::-1]))


if __name__ == '__main__':
    main()
