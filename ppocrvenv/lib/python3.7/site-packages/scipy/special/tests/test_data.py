import os

import numpy as np
from numpy import arccosh, arcsinh, arctanh
from numpy.testing import suppress_warnings
import pytest

from scipy.special import (
    lpn, lpmn, lpmv, lqn, lqmn, sph_harm, eval_legendre, eval_hermite,
    eval_laguerre, eval_genlaguerre, binom, cbrt, expm1, log1p, zeta,
    jn, jv, jvp, yn, yv, yvp, iv, ivp, kn, kv, kvp,
    gamma, gammaln, gammainc, gammaincc, gammaincinv, gammainccinv, digamma,
    beta, betainc, betaincinv, poch,
    ellipe, ellipeinc, ellipk, ellipkm1, ellipkinc, ellipj,
    erf, erfc, erfinv, erfcinv, exp1, expi, expn,
    bdtrik, btdtr, btdtri, btdtria, btdtrib, chndtr, gdtr, gdtrc, gdtrix, gdtrib,
    nbdtrik, pdtrik, owens_t,
    mathieu_a, mathieu_b, mathieu_cem, mathieu_sem, mathieu_modcem1,
    mathieu_modsem1, mathieu_modcem2, mathieu_modsem2,
    ellip_harm, ellip_harm_2, spherical_jn, spherical_yn, wright_bessel
)
from scipy.integrate import IntegrationWarning

from scipy.special._testutils import FuncData

DATASETS_BOOST = np.load(os.path.join(os.path.dirname(__file__),
                                      "data", "boost.npz"))

DATASETS_GSL = np.load(os.path.join(os.path.dirname(__file__),
                                    "data", "gsl.npz"))

DATASETS_LOCAL = np.load(os.path.join(os.path.dirname(__file__),
                                    "data", "local.npz"))


def data(func, dataname, *a, **kw):
    kw.setdefault('dataname', dataname)
    return FuncData(func, DATASETS_BOOST[dataname], *a, **kw)


def data_gsl(func, dataname, *a, **kw):
    kw.setdefault('dataname', dataname)
    return FuncData(func, DATASETS_GSL[dataname], *a, **kw)


def data_local(func, dataname, *a, **kw):
    kw.setdefault('dataname', dataname)
    return FuncData(func, DATASETS_LOCAL[dataname], *a, **kw)


def ellipk_(k):
    return ellipk(k*k)


def ellipkinc_(f, k):
    return ellipkinc(f, k*k)


def ellipe_(k):
    return ellipe(k*k)


def ellipeinc_(f, k):
    return ellipeinc(f, k*k)


def ellipj_(k):
    return ellipj(k*k)


def zeta_(x):
    return zeta(x, 1.)


def assoc_legendre_p_boost_(nu, mu, x):
    # the boost test data is for integer orders only
    return lpmv(mu, nu.astype(int), x)

def legendre_p_via_assoc_(nu, x):
    return lpmv(0, nu, x)

def lpn_(n, x):
    return lpn(n.astype('l'), x)[0][-1]

def lqn_(n, x):
    return lqn(n.astype('l'), x)[0][-1]

def legendre_p_via_lpmn(n, x):
    return lpmn(0, n, x)[0][0,-1]

def legendre_q_via_lqmn(n, x):
    return lqmn(0, n, x)[0][0,-1]

def mathieu_ce_rad(m, q, x):
    return mathieu_cem(m, q, x*180/np.pi)[0]


def mathieu_se_rad(m, q, x):
    return mathieu_sem(m, q, x*180/np.pi)[0]


def mathieu_mc1_scaled(m, q, x):
    # GSL follows a different normalization.
    # We follow Abramowitz & Stegun, they apparently something else.
    return mathieu_modcem1(m, q, x)[0] * np.sqrt(np.pi/2)


def mathieu_ms1_scaled(m, q, x):
    return mathieu_modsem1(m, q, x)[0] * np.sqrt(np.pi/2)


def mathieu_mc2_scaled(m, q, x):
    return mathieu_modcem2(m, q, x)[0] * np.sqrt(np.pi/2)


def mathieu_ms2_scaled(m, q, x):
    return mathieu_modsem2(m, q, x)[0] * np.sqrt(np.pi/2)

def eval_legendre_ld(n, x):
    return eval_legendre(n.astype('l'), x)

def eval_legendre_dd(n, x):
    return eval_legendre(n.astype('d'), x)

def eval_hermite_ld(n, x):
    return eval_hermite(n.astype('l'), x)

def eval_laguerre_ld(n, x):
    return eval_laguerre(n.astype('l'), x)

def eval_laguerre_dd(n, x):
    return eval_laguerre(n.astype('d'), x)

def eval_genlaguerre_ldd(n, a, x):
    return eval_genlaguerre(n.astype('l'), a, x)

def eval_genlaguerre_ddd(n, a, x):
    return eval_genlaguerre(n.astype('d'), a, x)

def bdtrik_comp(y, n, p):
    return bdtrik(1-y, n, p)

def btdtri_comp(a, b, p):
    return btdtri(a, b, 1-p)

def btdtria_comp(p, b, x):
    return btdtria(1-p, b, x)

def btdtrib_comp(a, p, x):
    return btdtrib(a, 1-p, x)

def gdtr_(p, x):
    return gdtr(1.0, p, x)

def gdtrc_(p, x):
    return gdtrc(1.0, p, x)

def gdtrix_(b, p):
    return gdtrix(1.0, b, p)

def gdtrix_comp(b, p):
    return gdtrix(1.0, b, 1-p)

def gdtrib_(p, x):
    return gdtrib(1.0, p, x)

def gdtrib_comp(p, x):
    return gdtrib(1.0, 1-p, x)

def nbdtrik_comp(y, n, p):
    return nbdtrik(1-y, n, p)

def pdtrik_comp(p, m):
    return pdtrik(1-p, m)

def poch_(z, m):
    return 1.0 / poch(z, m)

def poch_minus(z, m):
    return 1.0 / poch(z, -m)

def spherical_jn_(n, x):
    return spherical_jn(n.astype('l'), x)

def spherical_yn_(n, x):
    return spherical_yn(n.astype('l'), x)

def sph_harm_(m, n, theta, phi):
    y = sph_harm(m, n, theta, phi)
    return (y.real, y.imag)

def cexpm1(x, y):
    z = expm1(x + 1j*y)
    return z.real, z.imag

def clog1p(x, y):
    z = log1p(x + 1j*y)
    return z.real, z.imag


BOOST_TESTS = [
        data(arccosh, 'acosh_data_ipp-acosh_data', 0, 1, rtol=5e-13),
        data(arccosh, 'acosh_data_ipp-acosh_data', 0j, 1, rtol=5e-13),

        data(arcsinh, 'asinh_data_ipp-asinh_data', 0, 1, rtol=1e-11),
        data(arcsinh, 'asinh_data_ipp-asinh_data', 0j, 1, rtol=1e-11),

        data(arctanh, 'atanh_data_ipp-atanh_data', 0, 1, rtol=1e-11),
        data(arctanh, 'atanh_data_ipp-atanh_data', 0j, 1, rtol=1e-11),

        data(assoc_legendre_p_boost_, 'assoc_legendre_p_ipp-assoc_legendre_p', (0,1,2), 3, rtol=1e-11),

        data(legendre_p_via_assoc_, 'legendre_p_ipp-legendre_p', (0,1), 2, rtol=1e-11),
        data(legendre_p_via_assoc_, 'legendre_p_large_ipp-legendre_p_large', (0,1), 2, rtol=9.6e-14),
        data(legendre_p_via_lpmn, 'legendre_p_ipp-legendre_p', (0,1), 2, rtol=5e-14, vectorized=False),
        data(legendre_p_via_lpmn, 'legendre_p_large_ipp-legendre_p_large', (0,1), 2, rtol=9.6e-14, vectorized=False),
        data(lpn_, 'legendre_p_ipp-legendre_p', (0,1), 2, rtol=5e-14, vectorized=False),
        data(lpn_, 'legendre_p_large_ipp-legendre_p_large', (0,1), 2, rtol=3e-13, vectorized=False),
        data(eval_legendre_ld, 'legendre_p_ipp-legendre_p', (0,1), 2, rtol=6e-14),
        data(eval_legendre_ld, 'legendre_p_large_ipp-legendre_p_large', (0,1), 2, rtol=2e-13),
        data(eval_legendre_dd, 'legendre_p_ipp-legendre_p', (0,1), 2, rtol=2e-14),
        data(eval_legendre_dd, 'legendre_p_large_ipp-legendre_p_large', (0,1), 2, rtol=2e-13),

        data(lqn_, 'legendre_p_ipp-legendre_p', (0,1), 3, rtol=2e-14, vectorized=False),
        data(lqn_, 'legendre_p_large_ipp-legendre_p_large', (0,1), 3, rtol=2e-12, vectorized=False),
        data(legendre_q_via_lqmn, 'legendre_p_ipp-legendre_p', (0,1), 3, rtol=2e-14, vectorized=False),
        data(legendre_q_via_lqmn, 'legendre_p_large_ipp-legendre_p_large', (0,1), 3, rtol=2e-12, vectorized=False),

        data(beta, 'beta_exp_data_ipp-beta_exp_data', (0,1), 2, rtol=1e-13),
        data(beta, 'beta_exp_data_ipp-beta_exp_data', (0,1), 2, rtol=1e-13),
        data(beta, 'beta_med_data_ipp-beta_med_data', (0,1), 2, rtol=5e-13),

        data(betainc, 'ibeta_small_data_ipp-ibeta_small_data', (0,1,2), 5, rtol=6e-15),
        data(betainc, 'ibeta_data_ipp-ibeta_data', (0,1,2), 5, rtol=5e-13),
        data(betainc, 'ibeta_int_data_ipp-ibeta_int_data', (0,1,2), 5, rtol=2e-14),
        data(betainc, 'ibeta_large_data_ipp-ibeta_large_data', (0,1,2), 5, rtol=4e-10),

        data(betaincinv, 'ibeta_inv_data_ipp-ibeta_inv_data', (0,1,2), 3, rtol=1e-5),

        data(btdtr, 'ibeta_small_data_ipp-ibeta_small_data', (0,1,2), 5, rtol=6e-15),
        data(btdtr, 'ibeta_data_ipp-ibeta_data', (0,1,2), 5, rtol=4e-13),
        data(btdtr, 'ibeta_int_data_ipp-ibeta_int_data', (0,1,2), 5, rtol=2e-14),
        data(btdtr, 'ibeta_large_data_ipp-ibeta_large_data', (0,1,2), 5, rtol=4e-10),

        data(btdtri, 'ibeta_inv_data_ipp-ibeta_inv_data', (0,1,2), 3, rtol=1e-5),
        data(btdtri_comp, 'ibeta_inv_data_ipp-ibeta_inv_data', (0,1,2), 4, rtol=8e-7),

        data(btdtria, 'ibeta_inva_data_ipp-ibeta_inva_data', (2,0,1), 3, rtol=5e-9),
        data(btdtria_comp, 'ibeta_inva_data_ipp-ibeta_inva_data', (2,0,1), 4, rtol=5e-9),

        data(btdtrib, 'ibeta_inva_data_ipp-ibeta_inva_data', (0,2,1), 5, rtol=5e-9),
        data(btdtrib_comp, 'ibeta_inva_data_ipp-ibeta_inva_data', (0,2,1), 6, rtol=5e-9),

        data(binom, 'binomial_data_ipp-binomial_data', (0,1), 2, rtol=1e-13),
        data(binom, 'binomial_large_data_ipp-binomial_large_data', (0,1), 2, rtol=5e-13),

        data(bdtrik, 'binomial_quantile_ipp-binomial_quantile_data', (2,0,1), 3, rtol=5e-9),
        data(bdtrik_comp, 'binomial_quantile_ipp-binomial_quantile_data', (2,0,1), 4, rtol=5e-9),

        data(nbdtrik, 'negative_binomial_quantile_ipp-negative_binomial_quantile_data', (2,0,1), 3, rtol=4e-9),
        data(nbdtrik_comp, 'negative_binomial_quantile_ipp-negative_binomial_quantile_data', (2,0,1), 4, rtol=4e-9),

        data(pdtrik, 'poisson_quantile_ipp-poisson_quantile_data', (1,0), 2, rtol=3e-9),
        data(pdtrik_comp, 'poisson_quantile_ipp-poisson_quantile_data', (1,0), 3, rtol=4e-9),

        data(cbrt, 'cbrt_data_ipp-cbrt_data', 1, 0),

        data(digamma, 'digamma_data_ipp-digamma_data', 0, 1),
        data(digamma, 'digamma_data_ipp-digamma_data', 0j, 1),
        data(digamma, 'digamma_neg_data_ipp-digamma_neg_data', 0, 1, rtol=2e-13),
        data(digamma, 'digamma_neg_data_ipp-digamma_neg_data', 0j, 1, rtol=1e-13),
        data(digamma, 'digamma_root_data_ipp-digamma_root_data', 0, 1, rtol=1e-15),
        data(digamma, 'digamma_root_data_ipp-digamma_root_data', 0j, 1, rtol=1e-15),
        data(digamma, 'digamma_small_data_ipp-digamma_small_data', 0, 1, rtol=1e-15),
        data(digamma, 'digamma_small_data_ipp-digamma_small_data', 0j, 1, rtol=1e-14),

        data(ellipk_, 'ellint_k_data_ipp-ellint_k_data', 0, 1),
        data(ellipkinc_, 'ellint_f_data_ipp-ellint_f_data', (0,1), 2, rtol=1e-14),
        data(ellipe_, 'ellint_e_data_ipp-ellint_e_data', 0, 1),
        data(ellipeinc_, 'ellint_e2_data_ipp-ellint_e2_data', (0,1), 2, rtol=1e-14),

        data(erf, 'erf_data_ipp-erf_data', 0, 1),
        data(erf, 'erf_data_ipp-erf_data', 0j, 1, rtol=1e-13),
        data(erfc, 'erf_data_ipp-erf_data', 0, 2, rtol=6e-15),
        data(erf, 'erf_large_data_ipp-erf_large_data', 0, 1),
        data(erf, 'erf_large_data_ipp-erf_large_data', 0j, 1),
        data(erfc, 'erf_large_data_ipp-erf_large_data', 0, 2, rtol=4e-14),
        data(erf, 'erf_small_data_ipp-erf_small_data', 0, 1),
        data(erf, 'erf_small_data_ipp-erf_small_data', 0j, 1, rtol=1e-13),
        data(erfc, 'erf_small_data_ipp-erf_small_data', 0, 2),

        data(erfinv, 'erf_inv_data_ipp-erf_inv_data', 0, 1),
        data(erfcinv, 'erfc_inv_data_ipp-erfc_inv_data', 0, 1),
        data(erfcinv, 'erfc_inv_big_data_ipp-erfc_inv_big_data', 0, 1, param_filter=(lambda s: s > 0)),

        data(exp1, 'expint_1_data_ipp-expint_1_data', 1, 2, rtol=1e-13),
        data(exp1, 'expint_1_data_ipp-expint_1_data', 1j, 2, rtol=5e-9),
        data(expi, 'expinti_data_ipp-expinti_data', 0, 1, rtol=1e-13),
        data(expi, 'expinti_data_double_ipp-expinti_data_double', 0, 1, rtol=1e-13),
        data(expi, 'expinti_data_long_ipp-expinti_data_long', 0, 1),

        data(expn, 'expint_small_data_ipp-expint_small_data', (0,1), 2),
        data(expn, 'expint_data_ipp-expint_data', (0,1), 2, rtol=1e-14),

        data(gamma, 'test_gamma_data_ipp-near_0', 0, 1),
        data(gamma, 'test_gamma_data_ipp-near_1', 0, 1),
        data(gamma, 'test_gamma_data_ipp-near_2', 0, 1),
        data(gamma, 'test_gamma_data_ipp-near_m10', 0, 1),
        data(gamma, 'test_gamma_data_ipp-near_m55', 0, 1, rtol=7e-12),
        data(gamma, 'test_gamma_data_ipp-factorials', 0, 1, rtol=4e-14),
        data(gamma, 'test_gamma_data_ipp-near_0', 0j, 1, rtol=2e-9),
        data(gamma, 'test_gamma_data_ipp-near_1', 0j, 1, rtol=2e-9),
        data(gamma, 'test_gamma_data_ipp-near_2', 0j, 1, rtol=2e-9),
        data(gamma, 'test_gamma_data_ipp-near_m10', 0j, 1, rtol=2e-9),
        data(gamma, 'test_gamma_data_ipp-near_m55', 0j, 1, rtol=2e-9),
        data(gamma, 'test_gamma_data_ipp-factorials', 0j, 1, rtol=2e-13),
        data(gammaln, 'test_gamma_data_ipp-near_0', 0, 2, rtol=5e-11),
        data(gammaln, 'test_gamma_data_ipp-near_1', 0, 2, rtol=5e-11),
        data(gammaln, 'test_gamma_data_ipp-near_2', 0, 2, rtol=2e-10),
        data(gammaln, 'test_gamma_data_ipp-near_m10', 0, 2, rtol=5e-11),
        data(gammaln, 'test_gamma_data_ipp-near_m55', 0, 2, rtol=5e-11),
        data(gammaln, 'test_gamma_data_ipp-factorials', 0, 2),

        data(gammainc, 'igamma_small_data_ipp-igamma_small_data', (0,1), 5, rtol=5e-15),
        data(gammainc, 'igamma_med_data_ipp-igamma_med_data', (0,1), 5, rtol=2e-13),
        data(gammainc, 'igamma_int_data_ipp-igamma_int_data', (0,1), 5, rtol=2e-13),
        data(gammainc, 'igamma_big_data_ipp-igamma_big_data', (0,1), 5, rtol=1e-12),

        data(gdtr_, 'igamma_small_data_ipp-igamma_small_data', (0,1), 5, rtol=1e-13),
        data(gdtr_, 'igamma_med_data_ipp-igamma_med_data', (0,1), 5, rtol=2e-13),
        data(gdtr_, 'igamma_int_data_ipp-igamma_int_data', (0,1), 5, rtol=2e-13),
        data(gdtr_, 'igamma_big_data_ipp-igamma_big_data', (0,1), 5, rtol=2e-9),

        data(gammaincc, 'igamma_small_data_ipp-igamma_small_data', (0,1), 3, rtol=1e-13),
        data(gammaincc, 'igamma_med_data_ipp-igamma_med_data', (0,1), 3, rtol=2e-13),
        data(gammaincc, 'igamma_int_data_ipp-igamma_int_data', (0,1), 3, rtol=4e-14),
        data(gammaincc, 'igamma_big_data_ipp-igamma_big_data', (0,1), 3, rtol=1e-11),

        data(gdtrc_, 'igamma_small_data_ipp-igamma_small_data', (0,1), 3, rtol=1e-13),
        data(gdtrc_, 'igamma_med_data_ipp-igamma_med_data', (0,1), 3, rtol=2e-13),
        data(gdtrc_, 'igamma_int_data_ipp-igamma_int_data', (0,1), 3, rtol=4e-14),
        data(gdtrc_, 'igamma_big_data_ipp-igamma_big_data', (0,1), 3, rtol=1e-11),

        data(gdtrib_, 'igamma_inva_data_ipp-igamma_inva_data', (1,0), 2, rtol=5e-9),
        data(gdtrib_comp, 'igamma_inva_data_ipp-igamma_inva_data', (1,0), 3, rtol=5e-9),

        data(poch_, 'tgamma_delta_ratio_data_ipp-tgamma_delta_ratio_data', (0,1), 2, rtol=2e-13),
        data(poch_, 'tgamma_delta_ratio_int_ipp-tgamma_delta_ratio_int', (0,1), 2,),
        data(poch_, 'tgamma_delta_ratio_int2_ipp-tgamma_delta_ratio_int2', (0,1), 2,),
        data(poch_minus, 'tgamma_delta_ratio_data_ipp-tgamma_delta_ratio_data', (0,1), 3, rtol=2e-13),
        data(poch_minus, 'tgamma_delta_ratio_int_ipp-tgamma_delta_ratio_int', (0,1), 3),
        data(poch_minus, 'tgamma_delta_ratio_int2_ipp-tgamma_delta_ratio_int2', (0,1), 3),

        data(eval_hermite_ld, 'hermite_ipp-hermite', (0,1), 2, rtol=2e-14),

        data(eval_laguerre_ld, 'laguerre2_ipp-laguerre2', (0,1), 2, rtol=7e-12),
        data(eval_laguerre_dd, 'laguerre2_ipp-laguerre2', (0,1), 2, knownfailure='hyp2f1 insufficiently accurate.'),
        data(eval_genlaguerre_ldd, 'laguerre3_ipp-laguerre3', (0,1,2), 3, rtol=2e-13),
        data(eval_genlaguerre_ddd, 'laguerre3_ipp-laguerre3', (0,1,2), 3, knownfailure='hyp2f1 insufficiently accurate.'),

        data(log1p, 'log1p_expm1_data_ipp-log1p_expm1_data', 0, 1),
        data(expm1, 'log1p_expm1_data_ipp-log1p_expm1_data', 0, 2),

        data(iv, 'bessel_i_data_ipp-bessel_i_data', (0,1), 2, rtol=1e-12),
        data(iv, 'bessel_i_data_ipp-bessel_i_data', (0,1j), 2, rtol=2e-10, atol=1e-306),
        data(iv, 'bessel_i_int_data_ipp-bessel_i_int_data', (0,1), 2, rtol=1e-9),
        data(iv, 'bessel_i_int_data_ipp-bessel_i_int_data', (0,1j), 2, rtol=2e-10),

        data(ivp, 'bessel_i_prime_int_data_ipp-bessel_i_prime_int_data', (0,1), 2, rtol=1.2e-13),
        data(ivp, 'bessel_i_prime_int_data_ipp-bessel_i_prime_int_data', (0,1j), 2, rtol=1.2e-13, atol=1e-300),

        data(jn, 'bessel_j_int_data_ipp-bessel_j_int_data', (0,1), 2, rtol=1e-12),
        data(jn, 'bessel_j_int_data_ipp-bessel_j_int_data', (0,1j), 2, rtol=1e-12),
        data(jn, 'bessel_j_large_data_ipp-bessel_j_large_data', (0,1), 2, rtol=6e-11),
        data(jn, 'bessel_j_large_data_ipp-bessel_j_large_data', (0,1j), 2, rtol=6e-11),

        data(jv, 'bessel_j_int_data_ipp-bessel_j_int_data', (0,1), 2, rtol=1e-12),
        data(jv, 'bessel_j_int_data_ipp-bessel_j_int_data', (0,1j), 2, rtol=1e-12),
        data(jv, 'bessel_j_data_ipp-bessel_j_data', (0,1), 2, rtol=1e-12),
        data(jv, 'bessel_j_data_ipp-bessel_j_data', (0,1j), 2, rtol=1e-12),

        data(jvp, 'bessel_j_prime_int_data_ipp-bessel_j_prime_int_data', (0,1), 2, rtol=1e-13),
        data(jvp, 'bessel_j_prime_int_data_ipp-bessel_j_prime_int_data', (0,1j), 2, rtol=1e-13),
        data(jvp, 'bessel_j_prime_large_data_ipp-bessel_j_prime_large_data', (0,1), 2, rtol=1e-11),
        data(jvp, 'bessel_j_prime_large_data_ipp-bessel_j_prime_large_data', (0,1j), 2, rtol=1e-11),

        data(kn, 'bessel_k_int_data_ipp-bessel_k_int_data', (0,1), 2, rtol=1e-12),

        data(kv, 'bessel_k_int_data_ipp-bessel_k_int_data', (0,1), 2, rtol=1e-12),
        data(kv, 'bessel_k_int_data_ipp-bessel_k_int_data', (0,1j), 2, rtol=1e-12),
        data(kv, 'bessel_k_data_ipp-bessel_k_data', (0,1), 2, rtol=1e-12),
        data(kv, 'bessel_k_data_ipp-bessel_k_data', (0,1j), 2, rtol=1e-12),

        data(kvp, 'bessel_k_prime_int_data_ipp-bessel_k_prime_int_data', (0,1), 2, rtol=3e-14),
        data(kvp, 'bessel_k_prime_int_data_ipp-bessel_k_prime_int_data', (0,1j), 2, rtol=3e-14),
        data(kvp, 'bessel_k_prime_data_ipp-bessel_k_prime_data', (0,1), 2, rtol=7e-14),
        data(kvp, 'bessel_k_prime_data_ipp-bessel_k_prime_data', (0,1j), 2, rtol=7e-14),

        data(yn, 'bessel_y01_data_ipp-bessel_y01_data', (0,1), 2, rtol=1e-12),
        data(yn, 'bessel_yn_data_ipp-bessel_yn_data', (0,1), 2, rtol=1e-12),

        data(yv, 'bessel_yn_data_ipp-bessel_yn_data', (0,1), 2, rtol=1e-12),
        data(yv, 'bessel_yn_data_ipp-bessel_yn_data', (0,1j), 2, rtol=1e-12),
        data(yv, 'bessel_yv_data_ipp-bessel_yv_data', (0,1), 2, rtol=1e-10),
        data(yv, 'bessel_yv_data_ipp-bessel_yv_data', (0,1j), 2, rtol=1e-10),

        data(yvp, 'bessel_yv_prime_data_ipp-bessel_yv_prime_data', (0, 1), 2, rtol=4e-9),
        data(yvp, 'bessel_yv_prime_data_ipp-bessel_yv_prime_data', (0, 1j), 2, rtol=4e-9),

        data(zeta_, 'zeta_data_ipp-zeta_data', 0, 1, param_filter=(lambda s: s > 1)),
        data(zeta_, 'zeta_neg_data_ipp-zeta_neg_data', 0, 1, param_filter=(lambda s: s > 1)),
        data(zeta_, 'zeta_1_up_data_ipp-zeta_1_up_data', 0, 1, param_filter=(lambda s: s > 1)),
        data(zeta_, 'zeta_1_below_data_ipp-zeta_1_below_data', 0, 1, param_filter=(lambda s: s > 1)),

        data(gammaincinv, 'gamma_inv_small_data_ipp-gamma_inv_small_data', (0,1), 2, rtol=1e-11),
        data(gammaincinv, 'gamma_inv_data_ipp-gamma_inv_data', (0,1), 2, rtol=1e-14),
        data(gammaincinv, 'gamma_inv_big_data_ipp-gamma_inv_big_data', (0,1), 2, rtol=1e-11),

        data(gammainccinv, 'gamma_inv_small_data_ipp-gamma_inv_small_data', (0,1), 3, rtol=1e-12),
        data(gammainccinv, 'gamma_inv_data_ipp-gamma_inv_data', (0,1), 3, rtol=1e-14),
        data(gammainccinv, 'gamma_inv_big_data_ipp-gamma_inv_big_data', (0,1), 3, rtol=1e-14),

        data(gdtrix_, 'gamma_inv_small_data_ipp-gamma_inv_small_data', (0,1), 2, rtol=3e-13, knownfailure='gdtrix unflow some points'),
        data(gdtrix_, 'gamma_inv_data_ipp-gamma_inv_data', (0,1), 2, rtol=3e-15),
        data(gdtrix_, 'gamma_inv_big_data_ipp-gamma_inv_big_data', (0,1), 2),
        data(gdtrix_comp, 'gamma_inv_small_data_ipp-gamma_inv_small_data', (0,1), 2, knownfailure='gdtrix bad some points'),
        data(gdtrix_comp, 'gamma_inv_data_ipp-gamma_inv_data', (0,1), 3, rtol=6e-15),
        data(gdtrix_comp, 'gamma_inv_big_data_ipp-gamma_inv_big_data', (0,1), 3),

        data(chndtr, 'nccs_ipp-nccs', (2,0,1), 3, rtol=3e-5),
        data(chndtr, 'nccs_big_ipp-nccs_big', (2,0,1), 3, rtol=5e-4, knownfailure='chndtr inaccurate some points'),

        data(sph_harm_, 'spherical_harmonic_ipp-spherical_harmonic', (1,0,3,2), (4,5), rtol=5e-11,
             param_filter=(lambda p: np.ones(p.shape, '?'),
                           lambda p: np.ones(p.shape, '?'),
                           lambda p: np.logical_and(p < 2*np.pi, p >= 0),
                           lambda p: np.logical_and(p < np.pi, p >= 0))),

        data(spherical_jn_, 'sph_bessel_data_ipp-sph_bessel_data', (0,1), 2, rtol=1e-13),
        data(spherical_yn_, 'sph_neumann_data_ipp-sph_neumann_data', (0,1), 2, rtol=8e-15),

        data(owens_t, 'owens_t_ipp-owens_t', (0, 1), 2, rtol=5e-14),
        data(owens_t, 'owens_t_large_data_ipp-owens_t_large_data', (0, 1), 2, rtol=8e-12),

        # -- test data exists in boost but is not used in scipy --

        # ibeta_derivative_data_ipp/ibeta_derivative_data.txt
        # ibeta_derivative_int_data_ipp/ibeta_derivative_int_data.txt
        # ibeta_derivative_large_data_ipp/ibeta_derivative_large_data.txt
        # ibeta_derivative_small_data_ipp/ibeta_derivative_small_data.txt

        # bessel_y01_prime_data_ipp/bessel_y01_prime_data.txt
        # bessel_yn_prime_data_ipp/bessel_yn_prime_data.txt
        # sph_bessel_prime_data_ipp/sph_bessel_prime_data.txt
        # sph_neumann_prime_data_ipp/sph_neumann_prime_data.txt

        # ellint_d2_data_ipp/ellint_d2_data.txt
        # ellint_d_data_ipp/ellint_d_data.txt
        # ellint_pi2_data_ipp/ellint_pi2_data.txt
        # ellint_pi3_data_ipp/ellint_pi3_data.txt
        # ellint_pi3_large_data_ipp/ellint_pi3_large_data.txt
        # ellint_rc_data_ipp/ellint_rc_data.txt
        # ellint_rd_0xy_ipp/ellint_rd_0xy.txt
        # ellint_rd_0yy_ipp/ellint_rd_0yy.txt
        # ellint_rd_data_ipp/ellint_rd_data.txt
        # ellint_rd_xxx_ipp/ellint_rd_xxx.txt
        # ellint_rd_xxz_ipp/ellint_rd_xxz.txt
        # ellint_rd_xyy_ipp/ellint_rd_xyy.txt
        # ellint_rf_0yy_ipp/ellint_rf_0yy.txt
        # ellint_rf_data_ipp/ellint_rf_data.txt
        # ellint_rf_xxx_ipp/ellint_rf_xxx.txt
        # ellint_rf_xy0_ipp/ellint_rf_xy0.txt
        # ellint_rf_xyy_ipp/ellint_rf_xyy.txt
        # ellint_rg_00x_ipp/ellint_rg_00x.txt
        # ellint_rg_ipp/ellint_rg.txt
        # ellint_rg_xxx_ipp/ellint_rg_xxx.txt
        # ellint_rg_xy0_ipp/ellint_rg_xy0.txt
        # ellint_rg_xyy_ipp/ellint_rg_xyy.txt
        # ellint_rj_data_ipp/ellint_rj_data.txt
        # ellint_rj_e2_ipp/ellint_rj_e2.txt
        # ellint_rj_e3_ipp/ellint_rj_e3.txt
        # ellint_rj_e4_ipp/ellint_rj_e4.txt
        # ellint_rj_zp_ipp/ellint_rj_zp.txt

        # jacobi_elliptic_ipp/jacobi_elliptic.txt
        # jacobi_elliptic_small_ipp/jacobi_elliptic_small.txt
        # jacobi_large_phi_ipp/jacobi_large_phi.txt
        # jacobi_near_1_ipp/jacobi_near_1.txt
        # jacobi_zeta_big_phi_ipp/jacobi_zeta_big_phi.txt
        # jacobi_zeta_data_ipp/jacobi_zeta_data.txt

        # heuman_lambda_data_ipp/heuman_lambda_data.txt

        # hypergeometric_0F2_ipp/hypergeometric_0F2.txt
        # hypergeometric_1F1_big_ipp/hypergeometric_1F1_big.txt
        # hypergeometric_1F1_ipp/hypergeometric_1F1.txt
        # hypergeometric_1F1_small_random_ipp/hypergeometric_1F1_small_random.txt
        # hypergeometric_1F2_ipp/hypergeometric_1F2.txt
        # hypergeometric_1f1_large_regularized_ipp/hypergeometric_1f1_large_regularized.txt
        # hypergeometric_1f1_log_large_unsolved_ipp/hypergeometric_1f1_log_large_unsolved.txt
        # hypergeometric_2F0_half_ipp/hypergeometric_2F0_half.txt
        # hypergeometric_2F0_integer_a2_ipp/hypergeometric_2F0_integer_a2.txt
        # hypergeometric_2F0_ipp/hypergeometric_2F0.txt
        # hypergeometric_2F0_large_z_ipp/hypergeometric_2F0_large_z.txt
        # hypergeometric_2F1_ipp/hypergeometric_2F1.txt
        # hypergeometric_2F2_ipp/hypergeometric_2F2.txt

        # ncbeta_big_ipp/ncbeta_big.txt
        # nct_small_delta_ipp/nct_small_delta.txt
        # nct_asym_ipp/nct_asym.txt
        # ncbeta_ipp/ncbeta.txt

        # powm1_data_ipp/powm1_big_data.txt
        # powm1_sqrtp1m1_test_hpp/sqrtp1m1_data.txt

        # sinc_data_ipp/sinc_data.txt

        # test_gamma_data_ipp/gammap1m1_data.txt
        # tgamma_ratio_data_ipp/tgamma_ratio_data.txt

        # trig_data_ipp/trig_data.txt
        # trig_data2_ipp/trig_data2.txt
]


@pytest.mark.parametrize('test', BOOST_TESTS, ids=repr)
def test_boost(test):
    _test_factory(test)


GSL_TESTS = [
        data_gsl(mathieu_a, 'mathieu_ab', (0, 1), 2, rtol=1e-13, atol=1e-13),
        data_gsl(mathieu_b, 'mathieu_ab', (0, 1), 3, rtol=1e-13, atol=1e-13),

        # Also the GSL output has limited accuracy...
        data_gsl(mathieu_ce_rad, 'mathieu_ce_se', (0, 1, 2), 3, rtol=1e-7, atol=1e-13),
        data_gsl(mathieu_se_rad, 'mathieu_ce_se', (0, 1, 2), 4, rtol=1e-7, atol=1e-13),

        data_gsl(mathieu_mc1_scaled, 'mathieu_mc_ms', (0, 1, 2), 3, rtol=1e-7, atol=1e-13),
        data_gsl(mathieu_ms1_scaled, 'mathieu_mc_ms', (0, 1, 2), 4, rtol=1e-7, atol=1e-13),

        data_gsl(mathieu_mc2_scaled, 'mathieu_mc_ms', (0, 1, 2), 5, rtol=1e-7, atol=1e-13),
        data_gsl(mathieu_ms2_scaled, 'mathieu_mc_ms', (0, 1, 2), 6, rtol=1e-7, atol=1e-13),
]


@pytest.mark.parametrize('test', GSL_TESTS, ids=repr)
def test_gsl(test):
    _test_factory(test)


LOCAL_TESTS = [
    data_local(ellipkinc, 'ellipkinc_neg_m', (0, 1), 2),
    data_local(ellipkm1, 'ellipkm1', 0, 1),
    data_local(ellipeinc, 'ellipeinc_neg_m', (0, 1), 2),
    data_local(clog1p, 'log1p_expm1_complex', (0,1), (2,3), rtol=1e-14),
    data_local(cexpm1, 'log1p_expm1_complex', (0,1), (4,5), rtol=1e-14),
    data_local(gammainc, 'gammainc', (0, 1), 2, rtol=1e-12),
    data_local(gammaincc, 'gammaincc', (0, 1), 2, rtol=1e-11),
    data_local(ellip_harm_2, 'ellip',(0, 1, 2, 3, 4), 6, rtol=1e-10, atol=1e-13),
    data_local(ellip_harm, 'ellip',(0, 1, 2, 3, 4), 5, rtol=1e-10, atol=1e-13),
    data_local(wright_bessel, 'wright_bessel', (0, 1, 2), 3, rtol=1e-11),
]


@pytest.mark.parametrize('test', LOCAL_TESTS, ids=repr)
def test_local(test):
    _test_factory(test)


def _test_factory(test, dtype=np.double):
    """Boost test"""
    with suppress_warnings() as sup:
        sup.filter(IntegrationWarning, "The occurrence of roundoff error is detected")
        with np.errstate(all='ignore'):
            test.check(dtype=dtype)
