import numpy as np  # np is actually used, in the decorators below.
import pytest

from scipy.special._testutils import MissingModule, check_version
from scipy.special._mptestutils import (
    Arg, IntArg, mp_assert_allclose, assert_mpmath_equal)
from scipy.special._precompute.gammainc_asy import (
    compute_g, compute_alpha, compute_d)
from scipy.special._precompute.gammainc_data import gammainc, gammaincc

try:
    import sympy  # type: ignore[import]
except ImportError:
    sympy = MissingModule('sympy')

try:
    import mpmath as mp
except ImportError:
    mp = MissingModule('mpmath')


@check_version(mp, '0.19')
def test_g():
    # Test data for the g_k. See DLMF 5.11.4.
    with mp.workdps(30):
        g = [mp.mpf(1), mp.mpf(1)/12, mp.mpf(1)/288,
             -mp.mpf(139)/51840, -mp.mpf(571)/2488320,
             mp.mpf(163879)/209018880, mp.mpf(5246819)/75246796800]
        mp_assert_allclose(compute_g(7), g)


@pytest.mark.slow
@check_version(mp, '0.19')
@check_version(sympy, '0.7')
@pytest.mark.xfail_on_32bit("rtol only 2e-11, see gh-6938")
def test_alpha():
    # Test data for the alpha_k. See DLMF 8.12.14.
    with mp.workdps(30):
        alpha = [mp.mpf(0), mp.mpf(1), mp.mpf(1)/3, mp.mpf(1)/36,
                 -mp.mpf(1)/270, mp.mpf(1)/4320, mp.mpf(1)/17010,
                 -mp.mpf(139)/5443200, mp.mpf(1)/204120]
        mp_assert_allclose(compute_alpha(9), alpha)


@pytest.mark.xslow
@check_version(mp, '0.19')
@check_version(sympy, '0.7')
def test_d():
    # Compare the d_{k, n} to the results in appendix F of [1].
    #
    # Sources
    # -------
    # [1] DiDonato and Morris, Computation of the Incomplete Gamma
    #     Function Ratios and their Inverse, ACM Transactions on
    #     Mathematical Software, 1986.

    with mp.workdps(50):
        dataset = [(0, 0, -mp.mpf('0.333333333333333333333333333333')),
                   (0, 12, mp.mpf('0.102618097842403080425739573227e-7')),
                   (1, 0, -mp.mpf('0.185185185185185185185185185185e-2')),
                   (1, 12, mp.mpf('0.119516285997781473243076536700e-7')),
                   (2, 0, mp.mpf('0.413359788359788359788359788360e-2')),
                   (2, 12, -mp.mpf('0.140925299108675210532930244154e-7')),
                   (3, 0, mp.mpf('0.649434156378600823045267489712e-3')),
                   (3, 12, -mp.mpf('0.191111684859736540606728140873e-7')),
                   (4, 0, -mp.mpf('0.861888290916711698604702719929e-3')),
                   (4, 12, mp.mpf('0.288658297427087836297341274604e-7')),
                   (5, 0, -mp.mpf('0.336798553366358150308767592718e-3')),
                   (5, 12, mp.mpf('0.482409670378941807563762631739e-7')),
                   (6, 0, mp.mpf('0.531307936463992223165748542978e-3')),
                   (6, 12, -mp.mpf('0.882860074633048352505085243179e-7')),
                   (7, 0, mp.mpf('0.344367606892377671254279625109e-3')),
                   (7, 12, -mp.mpf('0.175629733590604619378669693914e-6')),
                   (8, 0, -mp.mpf('0.652623918595309418922034919727e-3')),
                   (8, 12, mp.mpf('0.377358774161109793380344937299e-6')),
                   (9, 0, -mp.mpf('0.596761290192746250124390067179e-3')),
                   (9, 12, mp.mpf('0.870823417786464116761231237189e-6'))]
        d = compute_d(10, 13)
        res = [d[k][n] for k, n, std in dataset]
        std = map(lambda x: x[2], dataset)
        mp_assert_allclose(res, std)


@check_version(mp, '0.19')
def test_gammainc():
    # Quick check that the gammainc in
    # special._precompute.gammainc_data agrees with mpmath's
    # gammainc.
    assert_mpmath_equal(gammainc,
                        lambda a, x: mp.gammainc(a, b=x, regularized=True),
                        [Arg(0, 100, inclusive_a=False), Arg(0, 100)],
                        nan_ok=False, rtol=1e-17, n=50, dps=50)


@pytest.mark.xslow
@check_version(mp, '0.19')
def test_gammaincc():
    # Check that the gammaincc in special._precompute.gammainc_data
    # agrees with mpmath's gammainc.
    assert_mpmath_equal(lambda a, x: gammaincc(a, x, dps=1000),
                        lambda a, x: mp.gammainc(a, a=x, regularized=True),
                        [Arg(20, 100), Arg(20, 100)],
                        nan_ok=False, rtol=1e-17, n=50, dps=1000)

    # Test the fast integer path
    assert_mpmath_equal(gammaincc,
                        lambda a, x: mp.gammainc(a, a=x, regularized=True),
                        [IntArg(1, 100), Arg(0, 100)],
                        nan_ok=False, rtol=1e-17, n=50, dps=50)
