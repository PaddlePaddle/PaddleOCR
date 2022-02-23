import numpy as np
from numpy.testing import assert_, assert_allclose
import pytest

import scipy.special.orthogonal as orth
from scipy.special._testutils import FuncData


def test_eval_chebyt():
    n = np.arange(0, 10000, 7)
    x = 2*np.random.rand() - 1
    v1 = np.cos(n*np.arccos(x))
    v2 = orth.eval_chebyt(n, x)
    assert_(np.allclose(v1, v2, rtol=1e-15))


def test_eval_genlaguerre_restriction():
    # check it returns nan for alpha <= -1
    assert_(np.isnan(orth.eval_genlaguerre(0, -1, 0)))
    assert_(np.isnan(orth.eval_genlaguerre(0.1, -1, 0)))


def test_warnings():
    # ticket 1334
    with np.errstate(all='raise'):
        # these should raise no fp warnings
        orth.eval_legendre(1, 0)
        orth.eval_laguerre(1, 1)
        orth.eval_gegenbauer(1, 1, 0)


class TestPolys:
    """
    Check that the eval_* functions agree with the constructed polynomials

    """

    def check_poly(self, func, cls, param_ranges=[], x_range=[], nn=10,
                   nparam=10, nx=10, rtol=1e-8):
        np.random.seed(1234)

        dataset = []
        for n in np.arange(nn):
            params = [a + (b-a)*np.random.rand(nparam) for a,b in param_ranges]
            params = np.asarray(params).T
            if not param_ranges:
                params = [0]
            for p in params:
                if param_ranges:
                    p = (n,) + tuple(p)
                else:
                    p = (n,)
                x = x_range[0] + (x_range[1] - x_range[0])*np.random.rand(nx)
                x[0] = x_range[0]  # always include domain start point
                x[1] = x_range[1]  # always include domain end point
                poly = np.poly1d(cls(*p).coef)
                z = np.c_[np.tile(p, (nx,1)), x, poly(x)]
                dataset.append(z)

        dataset = np.concatenate(dataset, axis=0)

        def polyfunc(*p):
            p = (p[0].astype(int),) + p[1:]
            return func(*p)

        with np.errstate(all='raise'):
            ds = FuncData(polyfunc, dataset, list(range(len(param_ranges)+2)), -1,
                          rtol=rtol)
            ds.check()

    def test_jacobi(self):
        self.check_poly(orth.eval_jacobi, orth.jacobi,
                   param_ranges=[(-0.99, 10), (-0.99, 10)], x_range=[-1, 1],
                   rtol=1e-5)

    def test_sh_jacobi(self):
        self.check_poly(orth.eval_sh_jacobi, orth.sh_jacobi,
                   param_ranges=[(1, 10), (0, 1)], x_range=[0, 1],
                   rtol=1e-5)

    def test_gegenbauer(self):
        self.check_poly(orth.eval_gegenbauer, orth.gegenbauer,
                   param_ranges=[(-0.499, 10)], x_range=[-1, 1],
                   rtol=1e-7)

    def test_chebyt(self):
        self.check_poly(orth.eval_chebyt, orth.chebyt,
                   param_ranges=[], x_range=[-1, 1])

    def test_chebyu(self):
        self.check_poly(orth.eval_chebyu, orth.chebyu,
                   param_ranges=[], x_range=[-1, 1])

    def test_chebys(self):
        self.check_poly(orth.eval_chebys, orth.chebys,
                   param_ranges=[], x_range=[-2, 2])

    def test_chebyc(self):
        self.check_poly(orth.eval_chebyc, orth.chebyc,
                   param_ranges=[], x_range=[-2, 2])

    def test_sh_chebyt(self):
        with np.errstate(all='ignore'):
            self.check_poly(orth.eval_sh_chebyt, orth.sh_chebyt,
                            param_ranges=[], x_range=[0, 1])

    def test_sh_chebyu(self):
        self.check_poly(orth.eval_sh_chebyu, orth.sh_chebyu,
                   param_ranges=[], x_range=[0, 1])

    def test_legendre(self):
        self.check_poly(orth.eval_legendre, orth.legendre,
                   param_ranges=[], x_range=[-1, 1])

    def test_sh_legendre(self):
        with np.errstate(all='ignore'):
            self.check_poly(orth.eval_sh_legendre, orth.sh_legendre,
                            param_ranges=[], x_range=[0, 1])

    def test_genlaguerre(self):
        self.check_poly(orth.eval_genlaguerre, orth.genlaguerre,
                   param_ranges=[(-0.99, 10)], x_range=[0, 100])

    def test_laguerre(self):
        self.check_poly(orth.eval_laguerre, orth.laguerre,
                   param_ranges=[], x_range=[0, 100])

    def test_hermite(self):
        self.check_poly(orth.eval_hermite, orth.hermite,
                   param_ranges=[], x_range=[-100, 100])

    def test_hermitenorm(self):
        self.check_poly(orth.eval_hermitenorm, orth.hermitenorm,
                        param_ranges=[], x_range=[-100, 100])


class TestRecurrence:
    """
    Check that the eval_* functions sig='ld->d' and 'dd->d' agree.

    """

    def check_poly(self, func, param_ranges=[], x_range=[], nn=10,
                   nparam=10, nx=10, rtol=1e-8):
        np.random.seed(1234)

        dataset = []
        for n in np.arange(nn):
            params = [a + (b-a)*np.random.rand(nparam) for a,b in param_ranges]
            params = np.asarray(params).T
            if not param_ranges:
                params = [0]
            for p in params:
                if param_ranges:
                    p = (n,) + tuple(p)
                else:
                    p = (n,)
                x = x_range[0] + (x_range[1] - x_range[0])*np.random.rand(nx)
                x[0] = x_range[0]  # always include domain start point
                x[1] = x_range[1]  # always include domain end point
                kw = dict(sig=(len(p)+1)*'d'+'->d')
                z = np.c_[np.tile(p, (nx,1)), x, func(*(p + (x,)), **kw)]
                dataset.append(z)

        dataset = np.concatenate(dataset, axis=0)

        def polyfunc(*p):
            p = (p[0].astype(int),) + p[1:]
            kw = dict(sig='l'+(len(p)-1)*'d'+'->d')
            return func(*p, **kw)

        with np.errstate(all='raise'):
            ds = FuncData(polyfunc, dataset, list(range(len(param_ranges)+2)), -1,
                          rtol=rtol)
            ds.check()

    def test_jacobi(self):
        self.check_poly(orth.eval_jacobi,
                   param_ranges=[(-0.99, 10), (-0.99, 10)], x_range=[-1, 1])

    def test_sh_jacobi(self):
        self.check_poly(orth.eval_sh_jacobi,
                   param_ranges=[(1, 10), (0, 1)], x_range=[0, 1])

    def test_gegenbauer(self):
        self.check_poly(orth.eval_gegenbauer,
                   param_ranges=[(-0.499, 10)], x_range=[-1, 1])

    def test_chebyt(self):
        self.check_poly(orth.eval_chebyt,
                   param_ranges=[], x_range=[-1, 1])

    def test_chebyu(self):
        self.check_poly(orth.eval_chebyu,
                   param_ranges=[], x_range=[-1, 1])

    def test_chebys(self):
        self.check_poly(orth.eval_chebys,
                   param_ranges=[], x_range=[-2, 2])

    def test_chebyc(self):
        self.check_poly(orth.eval_chebyc,
                   param_ranges=[], x_range=[-2, 2])

    def test_sh_chebyt(self):
        self.check_poly(orth.eval_sh_chebyt,
                   param_ranges=[], x_range=[0, 1])

    def test_sh_chebyu(self):
        self.check_poly(orth.eval_sh_chebyu,
                   param_ranges=[], x_range=[0, 1])

    def test_legendre(self):
        self.check_poly(orth.eval_legendre,
                   param_ranges=[], x_range=[-1, 1])

    def test_sh_legendre(self):
        self.check_poly(orth.eval_sh_legendre,
                   param_ranges=[], x_range=[0, 1])

    def test_genlaguerre(self):
        self.check_poly(orth.eval_genlaguerre,
                   param_ranges=[(-0.99, 10)], x_range=[0, 100])

    def test_laguerre(self):
        self.check_poly(orth.eval_laguerre,
                   param_ranges=[], x_range=[0, 100])

    def test_hermite(self):
        v = orth.eval_hermite(70, 1.0)
        a = -1.457076485701412e60
        assert_allclose(v,a)


def test_hermite_domain():
    # Regression test for gh-11091.
    assert np.isnan(orth.eval_hermite(-1, 1.0))
    assert np.isnan(orth.eval_hermitenorm(-1, 1.0))


@pytest.mark.parametrize("n", [0, 1, 2])
@pytest.mark.parametrize("x", [0, 1, np.nan])
def test_hermite_nan(n, x):
    # Regression test for gh-11369.
    assert np.isnan(orth.eval_hermite(n, x)) == np.any(np.isnan([n, x]))
    assert np.isnan(orth.eval_hermitenorm(n, x)) == np.any(np.isnan([n, x]))


@pytest.mark.parametrize('n', [0, 1, 2, 3.2])
@pytest.mark.parametrize('alpha', [1, np.nan])
@pytest.mark.parametrize('x', [2, np.nan])
def test_genlaguerre_nan(n, alpha, x):
    # Regression test for gh-11361.
    nan_laguerre = np.isnan(orth.eval_genlaguerre(n, alpha, x))
    nan_arg = np.any(np.isnan([n, alpha, x]))
    assert nan_laguerre == nan_arg


@pytest.mark.parametrize('n', [0, 1, 2, 3.2])
@pytest.mark.parametrize('alpha', [0.0, 1, np.nan])
@pytest.mark.parametrize('x', [1e-6, 2, np.nan])
def test_gegenbauer_nan(n, alpha, x):
    # Regression test for gh-11370.
    nan_gegenbauer = np.isnan(orth.eval_gegenbauer(n, alpha, x))
    nan_arg = np.any(np.isnan([n, alpha, x]))
    assert nan_gegenbauer == nan_arg
