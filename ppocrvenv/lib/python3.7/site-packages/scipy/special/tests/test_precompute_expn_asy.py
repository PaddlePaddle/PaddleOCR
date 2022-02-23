from numpy.testing import assert_equal

from scipy.special._testutils import check_version, MissingModule
from scipy.special._precompute.expn_asy import generate_A

try:
    import sympy  # type: ignore[import]
    from sympy import Poly
except ImportError:
    sympy = MissingModule("sympy")


@check_version(sympy, "1.0")
def test_generate_A():
    # Data from DLMF 8.20.5
    x = sympy.symbols('x')
    Astd = [Poly(1, x),
            Poly(1, x),
            Poly(1 - 2*x),
            Poly(1 - 8*x + 6*x**2)]
    Ares = generate_A(len(Astd))

    for p, q in zip(Astd, Ares):
        assert_equal(p, q)
