import itertools
import os

import numpy as np
from numpy.testing import (assert_equal, assert_allclose, assert_,
                           assert_almost_equal, assert_array_almost_equal)
from pytest import raises as assert_raises
import pytest
from scipy._lib._testutils import check_free_memory

from numpy import array, asarray, pi, sin, cos, arange, dot, ravel, sqrt, round
from scipy import interpolate
from scipy.interpolate.fitpack import (splrep, splev, bisplrep, bisplev,
     sproot, splprep, splint, spalde, splder, splantider, insert, dblint)
from scipy.interpolate.dfitpack import regrid_smth
from scipy.interpolate.fitpack2 import dfitpack_int


def data_file(basename):
    return os.path.join(os.path.abspath(os.path.dirname(__file__)),
                        'data', basename)


def norm2(x):
    return sqrt(dot(x.T,x))


def f1(x,d=0):
    if d is None:
        return "sin"
    if x is None:
        return "sin(x)"
    if d % 4 == 0:
        return sin(x)
    if d % 4 == 1:
        return cos(x)
    if d % 4 == 2:
        return -sin(x)
    if d % 4 == 3:
        return -cos(x)


def f2(x,y=0,dx=0,dy=0):
    if x is None:
        return "sin(x+y)"
    d = dx+dy
    if d % 4 == 0:
        return sin(x+y)
    if d % 4 == 1:
        return cos(x+y)
    if d % 4 == 2:
        return -sin(x+y)
    if d % 4 == 3:
        return -cos(x+y)


def makepairs(x, y):
    """Helper function to create an array of pairs of x and y."""
    xy = array(list(itertools.product(asarray(x), asarray(y))))
    return xy.T


def put(*a):
    """Produce some output if file run directly"""
    import sys
    if hasattr(sys.modules['__main__'], '__put_prints'):
        sys.stderr.write("".join(map(str, a)) + "\n")


class TestSmokeTests:
    """
    Smoke tests (with a few asserts) for fitpack routines -- mostly
    check that they are runnable
    """

    def check_1(self,f=f1,per=0,s=0,a=0,b=2*pi,N=20,at=0,xb=None,xe=None):
        if xb is None:
            xb = a
        if xe is None:
            xe = b
        x = a+(b-a)*arange(N+1,dtype=float)/float(N)    # nodes
        x1 = a+(b-a)*arange(1,N,dtype=float)/float(N-1)  # middle points of the nodes
        v = f(x)
        nk = []

        def err_est(k, d):
            # Assume f has all derivatives < 1
            h = 1.0/float(N)
            tol = 5 * h**(.75*(k-d))
            if s > 0:
                tol += 1e5*s
            return tol

        for k in range(1,6):
            tck = splrep(x,v,s=s,per=per,k=k,xe=xe)
            if at:
                t = tck[0][k:-k]
            else:
                t = x1
            nd = []
            for d in range(k+1):
                tol = err_est(k, d)
                err = norm2(f(t,d)-splev(t,tck,d)) / norm2(f(t,d))
                assert_(err < tol, (k, d, err, tol))
                nd.append((err, tol))
            nk.append(nd)
        put("\nf = %s  s=S_k(x;t,c)  x in [%s, %s] > [%s, %s]" % (f(None),
                                                        repr(round(xb,3)),repr(round(xe,3)),
                                                          repr(round(a,3)),repr(round(b,3))))
        if at:
            str = "at knots"
        else:
            str = "at the middle of nodes"
        put(" per=%d s=%s Evaluation %s" % (per,repr(s),str))
        put(" k :  |f-s|^2  |f'-s'| |f''-.. |f'''-. |f''''- |f'''''")
        k = 1
        for l in nk:
            put(' %d : ' % k)
            for r in l:
                put(' %.1e  %.1e' % r)
            put('\n')
            k = k+1

    def check_2(self,f=f1,per=0,s=0,a=0,b=2*pi,N=20,xb=None,xe=None,
              ia=0,ib=2*pi,dx=0.2*pi):
        if xb is None:
            xb = a
        if xe is None:
            xe = b
        x = a+(b-a)*arange(N+1,dtype=float)/float(N)    # nodes
        v = f(x)

        def err_est(k, d):
            # Assume f has all derivatives < 1
            h = 1.0/float(N)
            tol = 5 * h**(.75*(k-d))
            if s > 0:
                tol += 1e5*s
            return tol

        nk = []
        for k in range(1,6):
            tck = splrep(x,v,s=s,per=per,k=k,xe=xe)
            nk.append([splint(ia,ib,tck),spalde(dx,tck)])
        put("\nf = %s  s=S_k(x;t,c)  x in [%s, %s] > [%s, %s]" % (f(None),
                                                   repr(round(xb,3)),repr(round(xe,3)),
                                                    repr(round(a,3)),repr(round(b,3))))
        put(" per=%d s=%s N=%d [a, b] = [%s, %s]  dx=%s" % (per,repr(s),N,repr(round(ia,3)),repr(round(ib,3)),repr(round(dx,3))))
        put(" k :  int(s,[a,b]) Int.Error   Rel. error of s^(d)(dx) d = 0, .., k")
        k = 1
        for r in nk:
            if r[0] < 0:
                sr = '-'
            else:
                sr = ' '
            put(" %d   %s%.8f   %.1e " % (k,sr,abs(r[0]),
                                         abs(r[0]-(f(ib,-1)-f(ia,-1)))))
            d = 0
            for dr in r[1]:
                err = abs(1-dr/f(dx,d))
                tol = err_est(k, d)
                assert_(err < tol, (k, d))
                put(" %.1e %.1e" % (err, tol))
                d = d+1
            put("\n")
            k = k+1

    def check_3(self,f=f1,per=0,s=0,a=0,b=2*pi,N=20,xb=None,xe=None,
              ia=0,ib=2*pi,dx=0.2*pi):
        if xb is None:
            xb = a
        if xe is None:
            xe = b
        x = a+(b-a)*arange(N+1,dtype=float)/float(N)    # nodes
        v = f(x)
        put("  k  :     Roots of s(x) approx %s  x in [%s,%s]:" %
              (f(None),repr(round(a,3)),repr(round(b,3))))
        for k in range(1,6):
            tck = splrep(x, v, s=s, per=per, k=k, xe=xe)
            if k == 3:
                roots = sproot(tck)
                assert_allclose(splev(roots, tck), 0, atol=1e-10, rtol=1e-10)
                assert_allclose(roots, pi*array([1, 2, 3, 4]), rtol=1e-3)
                put('  %d  : %s' % (k, repr(roots.tolist())))
            else:
                assert_raises(ValueError, sproot, tck)

    def check_4(self,f=f1,per=0,s=0,a=0,b=2*pi,N=20,xb=None,xe=None,
              ia=0,ib=2*pi,dx=0.2*pi):
        if xb is None:
            xb = a
        if xe is None:
            xe = b
        x = a+(b-a)*arange(N+1,dtype=float)/float(N)    # nodes
        x1 = a + (b-a)*arange(1,N,dtype=float)/float(N-1)  # middle points of the nodes
        v, _ = f(x),f(x1)
        put(" u = %s   N = %d" % (repr(round(dx,3)),N))
        put("  k  :  [x(u), %s(x(u))]  Error of splprep  Error of splrep " % (f(0,None)))
        for k in range(1,6):
            tckp,u = splprep([x,v],s=s,per=per,k=k,nest=-1)
            tck = splrep(x,v,s=s,per=per,k=k)
            uv = splev(dx,tckp)
            err1 = abs(uv[1]-f(uv[0]))
            err2 = abs(splev(uv[0],tck)-f(uv[0]))
            assert_(err1 < 1e-2)
            assert_(err2 < 1e-2)
            put("  %d  :  %s    %.1e           %.1e" %
                  (k,repr([round(z,3) for z in uv]),
                   err1,
                   err2))
        put("Derivatives of parametric cubic spline at u (first function):")
        k = 3
        tckp,u = splprep([x,v],s=s,per=per,k=k,nest=-1)
        for d in range(1,k+1):
            uv = splev(dx,tckp,d)
            put(" %s " % (repr(uv[0])))

    def check_5(self,f=f2,kx=3,ky=3,xb=0,xe=2*pi,yb=0,ye=2*pi,Nx=20,Ny=20,s=0):
        x = xb+(xe-xb)*arange(Nx+1,dtype=float)/float(Nx)
        y = yb+(ye-yb)*arange(Ny+1,dtype=float)/float(Ny)
        xy = makepairs(x,y)
        tck = bisplrep(xy[0],xy[1],f(xy[0],xy[1]),s=s,kx=kx,ky=ky)
        tt = [tck[0][kx:-kx],tck[1][ky:-ky]]
        t2 = makepairs(tt[0],tt[1])
        v1 = bisplev(tt[0],tt[1],tck)
        v2 = f2(t2[0],t2[1])
        v2.shape = len(tt[0]),len(tt[1])
        err = norm2(ravel(v1-v2))
        assert_(err < 1e-2, err)
        put(err)

    def test_smoke_splrep_splev(self):
        put("***************** splrep/splev")
        self.check_1(s=1e-6)
        self.check_1()
        self.check_1(at=1)
        self.check_1(per=1)
        self.check_1(per=1,at=1)
        self.check_1(b=1.5*pi)
        self.check_1(b=1.5*pi,xe=2*pi,per=1,s=1e-1)

    def test_smoke_splint_spalde(self):
        put("***************** splint/spalde")
        self.check_2()
        self.check_2(per=1)
        self.check_2(ia=0.2*pi,ib=pi)
        self.check_2(ia=0.2*pi,ib=pi,N=50)

    def test_smoke_sproot(self):
        put("***************** sproot")
        self.check_3(a=0.1,b=15)

    def test_smoke_splprep_splrep_splev(self):
        put("***************** splprep/splrep/splev")
        self.check_4()
        self.check_4(N=50)

    def test_smoke_bisplrep_bisplev(self):
        put("***************** bisplev")
        self.check_5()


class TestSplev:
    def test_1d_shape(self):
        x = [1,2,3,4,5]
        y = [4,5,6,7,8]
        tck = splrep(x, y)
        z = splev([1], tck)
        assert_equal(z.shape, (1,))
        z = splev(1, tck)
        assert_equal(z.shape, ())

    def test_2d_shape(self):
        x = [1, 2, 3, 4, 5]
        y = [4, 5, 6, 7, 8]
        tck = splrep(x, y)
        t = np.array([[1.0, 1.5, 2.0, 2.5],
                      [3.0, 3.5, 4.0, 4.5]])
        z = splev(t, tck)
        z0 = splev(t[0], tck)
        z1 = splev(t[1], tck)
        assert_equal(z, np.row_stack((z0, z1)))

    def test_extrapolation_modes(self):
        # test extrapolation modes
        #    * if ext=0, return the extrapolated value.
        #    * if ext=1, return 0
        #    * if ext=2, raise a ValueError
        #    * if ext=3, return the boundary value.
        x = [1,2,3]
        y = [0,2,4]
        tck = splrep(x, y, k=1)

        rstl = [[-2, 6], [0, 0], None, [0, 4]]
        for ext in (0, 1, 3):
            assert_array_almost_equal(splev([0, 4], tck, ext=ext), rstl[ext])

        assert_raises(ValueError, splev, [0, 4], tck, ext=2)


class TestSplder:
    def setup_method(self):
        # non-uniform grid, just to make it sure
        x = np.linspace(0, 1, 100)**3
        y = np.sin(20 * x)
        self.spl = splrep(x, y)

        # double check that knots are non-uniform
        assert_(np.diff(self.spl[0]).ptp() > 0)

    def test_inverse(self):
        # Check that antiderivative + derivative is identity.
        for n in range(5):
            spl2 = splantider(self.spl, n)
            spl3 = splder(spl2, n)
            assert_allclose(self.spl[0], spl3[0])
            assert_allclose(self.spl[1], spl3[1])
            assert_equal(self.spl[2], spl3[2])

    def test_splder_vs_splev(self):
        # Check derivative vs. FITPACK

        for n in range(3+1):
            # Also extrapolation!
            xx = np.linspace(-1, 2, 2000)
            if n == 3:
                # ... except that FITPACK extrapolates strangely for
                # order 0, so let's not check that.
                xx = xx[(xx >= 0) & (xx <= 1)]

            dy = splev(xx, self.spl, n)
            spl2 = splder(self.spl, n)
            dy2 = splev(xx, spl2)
            if n == 1:
                assert_allclose(dy, dy2, rtol=2e-6)
            else:
                assert_allclose(dy, dy2)

    def test_splantider_vs_splint(self):
        # Check antiderivative vs. FITPACK
        spl2 = splantider(self.spl)

        # no extrapolation, splint assumes function is zero outside
        # range
        xx = np.linspace(0, 1, 20)

        for x1 in xx:
            for x2 in xx:
                y1 = splint(x1, x2, self.spl)
                y2 = splev(x2, spl2) - splev(x1, spl2)
                assert_allclose(y1, y2)

    def test_order0_diff(self):
        assert_raises(ValueError, splder, self.spl, 4)

    def test_kink(self):
        # Should refuse to differentiate splines with kinks

        spl2 = insert(0.5, self.spl, m=2)
        splder(spl2, 2)  # Should work
        assert_raises(ValueError, splder, spl2, 3)

        spl2 = insert(0.5, self.spl, m=3)
        splder(spl2, 1)  # Should work
        assert_raises(ValueError, splder, spl2, 2)

        spl2 = insert(0.5, self.spl, m=4)
        assert_raises(ValueError, splder, spl2, 1)

    def test_multidim(self):
        # c can have trailing dims
        for n in range(3):
            t, c, k = self.spl
            c2 = np.c_[c, c, c]
            c2 = np.dstack((c2, c2))

            spl2 = splantider((t, c2, k), n)
            spl3 = splder(spl2, n)

            assert_allclose(t, spl3[0])
            assert_allclose(c2, spl3[1])
            assert_equal(k, spl3[2])


class TestBisplrep:
    def test_overflow(self):
        from numpy.lib.stride_tricks import as_strided
        if dfitpack_int.itemsize == 8:
            size = 1500000**2
        else:
            size = 400**2
        # Don't allocate a real array, as it's very big, but rely
        # on that it's not referenced
        x = as_strided(np.zeros(()), shape=(size,))
        assert_raises(OverflowError, bisplrep, x, x, x, w=x,
                      xb=0, xe=1, yb=0, ye=1, s=0)

    def test_regression_1310(self):
        # Regression test for gh-1310
        data = np.load(data_file('bug-1310.npz'))['data']

        # Shouldn't crash -- the input data triggers work array sizes
        # that caused previously some data to not be aligned on
        # sizeof(double) boundaries in memory, which made the Fortran
        # code to crash when compiled with -O3
        bisplrep(data[:,0], data[:,1], data[:,2], kx=3, ky=3, s=0,
                 full_output=True)

    @pytest.mark.skipif(dfitpack_int != np.int64, reason="needs ilp64 fitpack")
    def test_ilp64_bisplrep(self):
        check_free_memory(28000)  # VM size, doesn't actually use the pages
        x = np.linspace(0, 1, 400)
        y = np.linspace(0, 1, 400)
        x, y = np.meshgrid(x, y)
        z = np.zeros_like(x)
        tck = bisplrep(x, y, z, kx=3, ky=3, s=0)
        assert_allclose(bisplev(0.5, 0.5, tck), 0.0)


def test_dblint():
    # Basic test to see it runs and gives the correct result on a trivial
    # problem. Note that `dblint` is not exposed in the interpolate namespace.
    x = np.linspace(0, 1)
    y = np.linspace(0, 1)
    xx, yy = np.meshgrid(x, y)
    rect = interpolate.RectBivariateSpline(x, y, 4 * xx * yy)
    tck = list(rect.tck)
    tck.extend(rect.degrees)

    assert_almost_equal(dblint(0, 1, 0, 1, tck), 1)
    assert_almost_equal(dblint(0, 0.5, 0, 1, tck), 0.25)
    assert_almost_equal(dblint(0.5, 1, 0, 1, tck), 0.75)
    assert_almost_equal(dblint(-100, 100, -100, 100, tck), 1)


def test_splev_der_k():
    # regression test for gh-2188: splev(x, tck, der=k) gives garbage or crashes
    # for x outside of knot range

    # test case from gh-2188
    tck = (np.array([0., 0., 2.5, 2.5]),
           np.array([-1.56679978, 2.43995873, 0., 0.]),
           1)
    t, c, k = tck
    x = np.array([-3, 0, 2.5, 3])

    # an explicit form of the linear spline
    assert_allclose(splev(x, tck), c[0] + (c[1] - c[0]) * x/t[2])
    assert_allclose(splev(x, tck, 1), (c[1]-c[0]) / t[2])

    # now check a random spline vs splder
    np.random.seed(1234)
    x = np.sort(np.random.random(30))
    y = np.random.random(30)
    t, c, k = splrep(x, y)

    x = [t[0] - 1., t[-1] + 1.]
    tck2 = splder((t, c, k), k)
    assert_allclose(splev(x, (t, c, k), k), splev(x, tck2))


def test_splprep_segfault():
    # regression test for gh-3847: splprep segfaults if knots are specified
    # for task=-1
    t = np.arange(0, 1.1, 0.1)
    x = np.sin(2*np.pi*t)
    y = np.cos(2*np.pi*t)
    tck, u = interpolate.splprep([x, y], s=0)
    unew = np.arange(0, 1.01, 0.01)

    uknots = tck[0]  # using the knots from the previous fitting
    tck, u = interpolate.splprep([x, y], task=-1, t=uknots)  # here is the crash


def test_bisplev_integer_overflow():
    np.random.seed(1)

    x = np.linspace(0, 1, 11)
    y = x
    z = np.random.randn(11, 11).ravel()
    kx = 1
    ky = 1

    nx, tx, ny, ty, c, fp, ier = regrid_smth(
        x, y, z, None, None, None, None, kx=kx, ky=ky, s=0.0)
    tck = (tx[:nx], ty[:ny], c[:(nx - kx - 1) * (ny - ky - 1)], kx, ky)

    xp = np.zeros([2621440])
    yp = np.zeros([2621440])

    assert_raises((RuntimeError, MemoryError), bisplev, xp, yp, tck)
