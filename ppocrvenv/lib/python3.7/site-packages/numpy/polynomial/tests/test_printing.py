import pytest
from numpy.core import array, arange, printoptions
import numpy.polynomial as poly
from numpy.testing import assert_equal, assert_

# For testing polynomial printing with object arrays
from fractions import Fraction
from decimal import Decimal


class TestStrUnicodeSuperSubscripts:

    @pytest.fixture(scope='class', autouse=True)
    def use_unicode(self):
        poly.set_default_printstyle('unicode')

    @pytest.mark.parametrize(('inp', 'tgt'), (
        ([1, 2, 3], "1.0 + 2.0·x¹ + 3.0·x²"),
        ([-1, 0, 3, -1], "-1.0 + 0.0·x¹ + 3.0·x² - 1.0·x³"),
        (arange(12), ("0.0 + 1.0·x¹ + 2.0·x² + 3.0·x³ + 4.0·x⁴ + 5.0·x⁵ + "
                      "6.0·x⁶ + 7.0·x⁷ +\n8.0·x⁸ + 9.0·x⁹ + 10.0·x¹⁰ + "
                      "11.0·x¹¹")),
    ))
    def test_polynomial_str(self, inp, tgt):
        res = str(poly.Polynomial(inp))
        assert_equal(res, tgt)

    @pytest.mark.parametrize(('inp', 'tgt'), (
        ([1, 2, 3], "1.0 + 2.0·T₁(x) + 3.0·T₂(x)"),
        ([-1, 0, 3, -1], "-1.0 + 0.0·T₁(x) + 3.0·T₂(x) - 1.0·T₃(x)"),
        (arange(12), ("0.0 + 1.0·T₁(x) + 2.0·T₂(x) + 3.0·T₃(x) + 4.0·T₄(x) + "
                      "5.0·T₅(x) +\n6.0·T₆(x) + 7.0·T₇(x) + 8.0·T₈(x) + "
                      "9.0·T₉(x) + 10.0·T₁₀(x) + 11.0·T₁₁(x)")),
    ))
    def test_chebyshev_str(self, inp, tgt):
        res = str(poly.Chebyshev(inp))
        assert_equal(res, tgt)

    @pytest.mark.parametrize(('inp', 'tgt'), (
        ([1, 2, 3], "1.0 + 2.0·P₁(x) + 3.0·P₂(x)"),
        ([-1, 0, 3, -1], "-1.0 + 0.0·P₁(x) + 3.0·P₂(x) - 1.0·P₃(x)"),
        (arange(12), ("0.0 + 1.0·P₁(x) + 2.0·P₂(x) + 3.0·P₃(x) + 4.0·P₄(x) + "
                      "5.0·P₅(x) +\n6.0·P₆(x) + 7.0·P₇(x) + 8.0·P₈(x) + "
                      "9.0·P₉(x) + 10.0·P₁₀(x) + 11.0·P₁₁(x)")),
    ))
    def test_legendre_str(self, inp, tgt):
        res = str(poly.Legendre(inp))
        assert_equal(res, tgt)

    @pytest.mark.parametrize(('inp', 'tgt'), (
        ([1, 2, 3], "1.0 + 2.0·H₁(x) + 3.0·H₂(x)"),
        ([-1, 0, 3, -1], "-1.0 + 0.0·H₁(x) + 3.0·H₂(x) - 1.0·H₃(x)"),
        (arange(12), ("0.0 + 1.0·H₁(x) + 2.0·H₂(x) + 3.0·H₃(x) + 4.0·H₄(x) + "
                      "5.0·H₅(x) +\n6.0·H₆(x) + 7.0·H₇(x) + 8.0·H₈(x) + "
                      "9.0·H₉(x) + 10.0·H₁₀(x) + 11.0·H₁₁(x)")),
    ))
    def test_hermite_str(self, inp, tgt):
        res = str(poly.Hermite(inp))
        assert_equal(res, tgt)

    @pytest.mark.parametrize(('inp', 'tgt'), (
        ([1, 2, 3], "1.0 + 2.0·He₁(x) + 3.0·He₂(x)"),
        ([-1, 0, 3, -1], "-1.0 + 0.0·He₁(x) + 3.0·He₂(x) - 1.0·He₃(x)"),
        (arange(12), ("0.0 + 1.0·He₁(x) + 2.0·He₂(x) + 3.0·He₃(x) + "
                      "4.0·He₄(x) + 5.0·He₅(x) +\n6.0·He₆(x) + 7.0·He₇(x) + "
                      "8.0·He₈(x) + 9.0·He₉(x) + 10.0·He₁₀(x) +\n"
                      "11.0·He₁₁(x)")),
    ))
    def test_hermiteE_str(self, inp, tgt):
        res = str(poly.HermiteE(inp))
        assert_equal(res, tgt)

    @pytest.mark.parametrize(('inp', 'tgt'), (
        ([1, 2, 3], "1.0 + 2.0·L₁(x) + 3.0·L₂(x)"),
        ([-1, 0, 3, -1], "-1.0 + 0.0·L₁(x) + 3.0·L₂(x) - 1.0·L₃(x)"),
        (arange(12), ("0.0 + 1.0·L₁(x) + 2.0·L₂(x) + 3.0·L₃(x) + 4.0·L₄(x) + "
                      "5.0·L₅(x) +\n6.0·L₆(x) + 7.0·L₇(x) + 8.0·L₈(x) + "
                      "9.0·L₉(x) + 10.0·L₁₀(x) + 11.0·L₁₁(x)")),
    ))
    def test_laguerre_str(self, inp, tgt):
        res = str(poly.Laguerre(inp))
        assert_equal(res, tgt)


class TestStrAscii:

    @pytest.fixture(scope='class', autouse=True)
    def use_ascii(self):
        poly.set_default_printstyle('ascii')

    @pytest.mark.parametrize(('inp', 'tgt'), (
        ([1, 2, 3], "1.0 + 2.0 x**1 + 3.0 x**2"),
        ([-1, 0, 3, -1], "-1.0 + 0.0 x**1 + 3.0 x**2 - 1.0 x**3"),
        (arange(12), ("0.0 + 1.0 x**1 + 2.0 x**2 + 3.0 x**3 + 4.0 x**4 + "
                      "5.0 x**5 + 6.0 x**6 +\n7.0 x**7 + 8.0 x**8 + "
                      "9.0 x**9 + 10.0 x**10 + 11.0 x**11")),
    ))
    def test_polynomial_str(self, inp, tgt):
        res = str(poly.Polynomial(inp))
        assert_equal(res, tgt)

    @pytest.mark.parametrize(('inp', 'tgt'), (
        ([1, 2, 3], "1.0 + 2.0 T_1(x) + 3.0 T_2(x)"),
        ([-1, 0, 3, -1], "-1.0 + 0.0 T_1(x) + 3.0 T_2(x) - 1.0 T_3(x)"),
        (arange(12), ("0.0 + 1.0 T_1(x) + 2.0 T_2(x) + 3.0 T_3(x) + "
                      "4.0 T_4(x) + 5.0 T_5(x) +\n6.0 T_6(x) + 7.0 T_7(x) + "
                      "8.0 T_8(x) + 9.0 T_9(x) + 10.0 T_10(x) +\n"
                      "11.0 T_11(x)")),
    ))
    def test_chebyshev_str(self, inp, tgt):
        res = str(poly.Chebyshev(inp))
        assert_equal(res, tgt)

    @pytest.mark.parametrize(('inp', 'tgt'), (
        ([1, 2, 3], "1.0 + 2.0 P_1(x) + 3.0 P_2(x)"),
        ([-1, 0, 3, -1], "-1.0 + 0.0 P_1(x) + 3.0 P_2(x) - 1.0 P_3(x)"),
        (arange(12), ("0.0 + 1.0 P_1(x) + 2.0 P_2(x) + 3.0 P_3(x) + "
                      "4.0 P_4(x) + 5.0 P_5(x) +\n6.0 P_6(x) + 7.0 P_7(x) + "
                      "8.0 P_8(x) + 9.0 P_9(x) + 10.0 P_10(x) +\n"
                      "11.0 P_11(x)")),
    ))
    def test_legendre_str(self, inp, tgt):
        res = str(poly.Legendre(inp))
        assert_equal(res, tgt)

    @pytest.mark.parametrize(('inp', 'tgt'), (
        ([1, 2, 3], "1.0 + 2.0 H_1(x) + 3.0 H_2(x)"),
        ([-1, 0, 3, -1], "-1.0 + 0.0 H_1(x) + 3.0 H_2(x) - 1.0 H_3(x)"),
        (arange(12), ("0.0 + 1.0 H_1(x) + 2.0 H_2(x) + 3.0 H_3(x) + "
                      "4.0 H_4(x) + 5.0 H_5(x) +\n6.0 H_6(x) + 7.0 H_7(x) + "
                      "8.0 H_8(x) + 9.0 H_9(x) + 10.0 H_10(x) +\n"
                      "11.0 H_11(x)")),
    ))
    def test_hermite_str(self, inp, tgt):
        res = str(poly.Hermite(inp))
        assert_equal(res, tgt)

    @pytest.mark.parametrize(('inp', 'tgt'), (
        ([1, 2, 3], "1.0 + 2.0 He_1(x) + 3.0 He_2(x)"),
        ([-1, 0, 3, -1], "-1.0 + 0.0 He_1(x) + 3.0 He_2(x) - 1.0 He_3(x)"),
        (arange(12), ("0.0 + 1.0 He_1(x) + 2.0 He_2(x) + 3.0 He_3(x) + "
                      "4.0 He_4(x) +\n5.0 He_5(x) + 6.0 He_6(x) + "
                      "7.0 He_7(x) + 8.0 He_8(x) + 9.0 He_9(x) +\n"
                      "10.0 He_10(x) + 11.0 He_11(x)")),
    ))
    def test_hermiteE_str(self, inp, tgt):
        res = str(poly.HermiteE(inp))
        assert_equal(res, tgt)

    @pytest.mark.parametrize(('inp', 'tgt'), (
        ([1, 2, 3], "1.0 + 2.0 L_1(x) + 3.0 L_2(x)"),
        ([-1, 0, 3, -1], "-1.0 + 0.0 L_1(x) + 3.0 L_2(x) - 1.0 L_3(x)"),
        (arange(12), ("0.0 + 1.0 L_1(x) + 2.0 L_2(x) + 3.0 L_3(x) + "
                      "4.0 L_4(x) + 5.0 L_5(x) +\n6.0 L_6(x) + 7.0 L_7(x) + "
                      "8.0 L_8(x) + 9.0 L_9(x) + 10.0 L_10(x) +\n"
                      "11.0 L_11(x)")),
    ))
    def test_laguerre_str(self, inp, tgt):
        res = str(poly.Laguerre(inp))
        assert_equal(res, tgt)


class TestLinebreaking:

    @pytest.fixture(scope='class', autouse=True)
    def use_ascii(self):
        poly.set_default_printstyle('ascii')

    def test_single_line_one_less(self):
        # With 'ascii' style, len(str(p)) is default linewidth - 1 (i.e. 74)
        p = poly.Polynomial([123456789, 123456789, 123456789, 1234, 1])
        assert_equal(len(str(p)), 74)
        assert_equal(str(p), (
            '123456789.0 + 123456789.0 x**1 + 123456789.0 x**2 + '
            '1234.0 x**3 + 1.0 x**4'
        ))

    def test_num_chars_is_linewidth(self):
        # len(str(p)) == default linewidth == 75
        p = poly.Polynomial([123456789, 123456789, 123456789, 1234, 10])
        assert_equal(len(str(p)), 75)
        assert_equal(str(p), (
            '123456789.0 + 123456789.0 x**1 + 123456789.0 x**2 + '
            '1234.0 x**3 +\n10.0 x**4'
        ))

    def test_first_linebreak_multiline_one_less_than_linewidth(self):
        # Multiline str where len(first_line) + len(next_term) == lw - 1 == 74
        p = poly.Polynomial(
                [123456789, 123456789, 123456789, 12, 1, 123456789]
            )
        assert_equal(len(str(p).split('\n')[0]), 74)
        assert_equal(str(p), (
            '123456789.0 + 123456789.0 x**1 + 123456789.0 x**2 + '
            '12.0 x**3 + 1.0 x**4 +\n123456789.0 x**5'
        ))

    def test_first_linebreak_multiline_on_linewidth(self):
        # First line is one character longer than previous test
        p = poly.Polynomial(
                [123456789, 123456789, 123456789, 123, 1, 123456789]
            )
        assert_equal(str(p), (
            '123456789.0 + 123456789.0 x**1 + 123456789.0 x**2 + '
            '123.0 x**3 +\n1.0 x**4 + 123456789.0 x**5'
        ))

    @pytest.mark.parametrize(('lw', 'tgt'), (
        (75, ('0.0 + 10.0 x**1 + 200.0 x**2 + 3000.0 x**3 + 40000.0 x**4 +\n'
              '500000.0 x**5 + 600000.0 x**6 + 70000.0 x**7 + 8000.0 x**8 + '
              '900.0 x**9')),
        (45, ('0.0 + 10.0 x**1 + 200.0 x**2 + 3000.0 x**3 +\n40000.0 x**4 + '
              '500000.0 x**5 +\n600000.0 x**6 + 70000.0 x**7 + 8000.0 x**8 +\n'
              '900.0 x**9')),
        (132, ('0.0 + 10.0 x**1 + 200.0 x**2 + 3000.0 x**3 + 40000.0 x**4 + '
               '500000.0 x**5 + 600000.0 x**6 + 70000.0 x**7 + 8000.0 x**8 + '
               '900.0 x**9')),
    ))
    def test_linewidth_printoption(self, lw, tgt):
        p = poly.Polynomial(
            [0, 10, 200, 3000, 40000, 500000, 600000, 70000, 8000, 900]
        )
        with printoptions(linewidth=lw):
            assert_equal(str(p), tgt)
            for line in str(p).split('\n'):
                assert_(len(line) < lw)


def test_set_default_printoptions():
    p = poly.Polynomial([1, 2, 3])
    c = poly.Chebyshev([1, 2, 3])
    poly.set_default_printstyle('ascii')
    assert_equal(str(p), "1.0 + 2.0 x**1 + 3.0 x**2")
    assert_equal(str(c), "1.0 + 2.0 T_1(x) + 3.0 T_2(x)")
    poly.set_default_printstyle('unicode')
    assert_equal(str(p), "1.0 + 2.0·x¹ + 3.0·x²")
    assert_equal(str(c), "1.0 + 2.0·T₁(x) + 3.0·T₂(x)")
    with pytest.raises(ValueError):
        poly.set_default_printstyle('invalid_input')


def test_complex_coefficients():
    """Test both numpy and built-in complex."""
    coefs = [0+1j, 1+1j, -2+2j, 3+0j]
    # numpy complex
    p1 = poly.Polynomial(coefs)
    # Python complex
    p2 = poly.Polynomial(array(coefs, dtype=object))
    poly.set_default_printstyle('unicode')
    assert_equal(str(p1), "1j + (1+1j)·x¹ - (2-2j)·x² + (3+0j)·x³")
    assert_equal(str(p2), "1j + (1+1j)·x¹ + (-2+2j)·x² + (3+0j)·x³")
    poly.set_default_printstyle('ascii')
    assert_equal(str(p1), "1j + (1+1j) x**1 - (2-2j) x**2 + (3+0j) x**3")
    assert_equal(str(p2), "1j + (1+1j) x**1 + (-2+2j) x**2 + (3+0j) x**3")


@pytest.mark.parametrize(('coefs', 'tgt'), (
    (array([Fraction(1, 2), Fraction(3, 4)], dtype=object), (
        "1/2 + 3/4·x¹"
    )),
    (array([1, 2, Fraction(5, 7)], dtype=object), (
        "1 + 2·x¹ + 5/7·x²"
    )),
    (array([Decimal('1.00'), Decimal('2.2'), 3], dtype=object), (
        "1.00 + 2.2·x¹ + 3·x²"
    )),
))
def test_numeric_object_coefficients(coefs, tgt):
    p = poly.Polynomial(coefs)
    poly.set_default_printstyle('unicode')
    assert_equal(str(p), tgt)


@pytest.mark.parametrize(('coefs', 'tgt'), (
    (array([1, 2, 'f'], dtype=object), '1 + 2·x¹ + f·x²'),
    (array([1, 2, [3, 4]], dtype=object), '1 + 2·x¹ + [3, 4]·x²'),
))
def test_nonnumeric_object_coefficients(coefs, tgt):
    """
    Test coef fallback for object arrays of non-numeric coefficients.
    """
    p = poly.Polynomial(coefs)
    poly.set_default_printstyle('unicode')
    assert_equal(str(p), tgt)


class TestFormat:
    def test_format_unicode(self):
        poly.set_default_printstyle('ascii')
        p = poly.Polynomial([1, 2, 0, -1])
        assert_equal(format(p, 'unicode'), "1.0 + 2.0·x¹ + 0.0·x² - 1.0·x³")

    def test_format_ascii(self):
        poly.set_default_printstyle('unicode')
        p = poly.Polynomial([1, 2, 0, -1])
        assert_equal(
            format(p, 'ascii'), "1.0 + 2.0 x**1 + 0.0 x**2 - 1.0 x**3"
        )

    def test_empty_formatstr(self):
        poly.set_default_printstyle('ascii')
        p = poly.Polynomial([1, 2, 3])
        assert_equal(format(p), "1.0 + 2.0 x**1 + 3.0 x**2")
        assert_equal(f"{p}", "1.0 + 2.0 x**1 + 3.0 x**2")

    def test_bad_formatstr(self):
        p = poly.Polynomial([1, 2, 0, -1])
        with pytest.raises(ValueError):
            format(p, '.2f')


class TestRepr:
    def test_polynomial_str(self):
        res = repr(poly.Polynomial([0, 1]))
        tgt = 'Polynomial([0., 1.], domain=[-1,  1], window=[-1,  1])'
        assert_equal(res, tgt)

    def test_chebyshev_str(self):
        res = repr(poly.Chebyshev([0, 1]))
        tgt = 'Chebyshev([0., 1.], domain=[-1,  1], window=[-1,  1])'
        assert_equal(res, tgt)

    def test_legendre_repr(self):
        res = repr(poly.Legendre([0, 1]))
        tgt = 'Legendre([0., 1.], domain=[-1,  1], window=[-1,  1])'
        assert_equal(res, tgt)

    def test_hermite_repr(self):
        res = repr(poly.Hermite([0, 1]))
        tgt = 'Hermite([0., 1.], domain=[-1,  1], window=[-1,  1])'
        assert_equal(res, tgt)

    def test_hermiteE_repr(self):
        res = repr(poly.HermiteE([0, 1]))
        tgt = 'HermiteE([0., 1.], domain=[-1,  1], window=[-1,  1])'
        assert_equal(res, tgt)

    def test_laguerre_repr(self):
        res = repr(poly.Laguerre([0, 1]))
        tgt = 'Laguerre([0., 1.], domain=[0, 1], window=[0, 1])'
        assert_equal(res, tgt)


class TestLatexRepr:
    """Test the latex repr used by Jupyter"""

    def as_latex(self, obj):
        # right now we ignore the formatting of scalars in our tests, since
        # it makes them too verbose. Ideally, the formatting of scalars will
        # be fixed such that tests below continue to pass
        obj._repr_latex_scalar = lambda x: str(x)
        try:
            return obj._repr_latex_()
        finally:
            del obj._repr_latex_scalar

    def test_simple_polynomial(self):
        # default input
        p = poly.Polynomial([1, 2, 3])
        assert_equal(self.as_latex(p),
            r'$x \mapsto 1.0 + 2.0\,x + 3.0\,x^{2}$')

        # translated input
        p = poly.Polynomial([1, 2, 3], domain=[-2, 0])
        assert_equal(self.as_latex(p),
            r'$x \mapsto 1.0 + 2.0\,\left(1.0 + x\right) + 3.0\,\left(1.0 + x\right)^{2}$')

        # scaled input
        p = poly.Polynomial([1, 2, 3], domain=[-0.5, 0.5])
        assert_equal(self.as_latex(p),
            r'$x \mapsto 1.0 + 2.0\,\left(2.0x\right) + 3.0\,\left(2.0x\right)^{2}$')

        # affine input
        p = poly.Polynomial([1, 2, 3], domain=[-1, 0])
        assert_equal(self.as_latex(p),
            r'$x \mapsto 1.0 + 2.0\,\left(1.0 + 2.0x\right) + 3.0\,\left(1.0 + 2.0x\right)^{2}$')

    def test_basis_func(self):
        p = poly.Chebyshev([1, 2, 3])
        assert_equal(self.as_latex(p),
            r'$x \mapsto 1.0\,{T}_{0}(x) + 2.0\,{T}_{1}(x) + 3.0\,{T}_{2}(x)$')
        # affine input - check no surplus parens are added
        p = poly.Chebyshev([1, 2, 3], domain=[-1, 0])
        assert_equal(self.as_latex(p),
            r'$x \mapsto 1.0\,{T}_{0}(1.0 + 2.0x) + 2.0\,{T}_{1}(1.0 + 2.0x) + 3.0\,{T}_{2}(1.0 + 2.0x)$')

    def test_multichar_basis_func(self):
        p = poly.HermiteE([1, 2, 3])
        assert_equal(self.as_latex(p),
            r'$x \mapsto 1.0\,{He}_{0}(x) + 2.0\,{He}_{1}(x) + 3.0\,{He}_{2}(x)$')
