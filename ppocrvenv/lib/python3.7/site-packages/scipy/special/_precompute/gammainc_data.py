"""Compute gammainc and gammaincc for large arguments and parameters
and save the values to data files for use in tests. We can't just
compare to mpmath's gammainc in test_mpmath.TestSystematic because it
would take too long.

Note that mpmath's gammainc is computed using hypercomb, but since it
doesn't allow the user to increase the maximum number of terms used in
the series it doesn't converge for many arguments. To get around this
we copy the mpmath implementation but use more terms.

This takes about 17 minutes to run on a 2.3 GHz Macbook Pro with 4GB
ram.

Sources:
[1] Fredrik Johansson and others. mpmath: a Python library for
    arbitrary-precision floating-point arithmetic (version 0.19),
    December 2013. http://mpmath.org/.

"""
import os
from time import time
import numpy as np
from numpy import pi

from scipy.special._mptestutils import mpf2float

try:
    import mpmath as mp
except ImportError:
    pass


def gammainc(a, x, dps=50, maxterms=10**8):
    """Compute gammainc exactly like mpmath does but allow for more
    summands in hypercomb. See

    mpmath/functions/expintegrals.py#L134

    in the mpmath github repository.

    """
    with mp.workdps(dps):
        z, a, b = mp.mpf(a), mp.mpf(x), mp.mpf(x)
        G = [z]
        negb = mp.fneg(b, exact=True)

        def h(z):
            T1 = [mp.exp(negb), b, z], [1, z, -1], [], G, [1], [1+z], b
            return (T1,)

        res = mp.hypercomb(h, [z], maxterms=maxterms)
        return mpf2float(res)


def gammaincc(a, x, dps=50, maxterms=10**8):
    """Compute gammaincc exactly like mpmath does but allow for more
    terms in hypercomb. See

    mpmath/functions/expintegrals.py#L187

    in the mpmath github repository.

    """
    with mp.workdps(dps):
        z, a = a, x

        if mp.isint(z):
            try:
                # mpmath has a fast integer path
                return mpf2float(mp.gammainc(z, a=a, regularized=True))
            except mp.libmp.NoConvergence:
                pass
        nega = mp.fneg(a, exact=True)
        G = [z]
        # Use 2F0 series when possible; fall back to lower gamma representation
        try:
            def h(z):
                r = z-1
                return [([mp.exp(nega), a], [1, r], [], G, [1, -r], [], 1/nega)]
            return mpf2float(mp.hypercomb(h, [z], force_series=True))
        except mp.libmp.NoConvergence:
            def h(z):
                T1 = [], [1, z-1], [z], G, [], [], 0
                T2 = [-mp.exp(nega), a, z], [1, z, -1], [], G, [1], [1+z], a
                return T1, T2
            return mpf2float(mp.hypercomb(h, [z], maxterms=maxterms))


def main():
    t0 = time()
    # It would be nice to have data for larger values, but either this
    # requires prohibitively large precision (dps > 800) or mpmath has
    # a bug. For example, gammainc(1e20, 1e20, dps=800) returns a
    # value around 0.03, while the true value should be close to 0.5
    # (DLMF 8.12.15).
    print(__doc__)
    pwd = os.path.dirname(__file__)
    r = np.logspace(4, 14, 30)
    ltheta = np.logspace(np.log10(pi/4), np.log10(np.arctan(0.6)), 30)
    utheta = np.logspace(np.log10(pi/4), np.log10(np.arctan(1.4)), 30)

    regimes = [(gammainc, ltheta), (gammaincc, utheta)]
    for func, theta in regimes:
        rg, thetag = np.meshgrid(r, theta)
        a, x = rg*np.cos(thetag), rg*np.sin(thetag)
        a, x = a.flatten(), x.flatten()
        dataset = []
        for i, (a0, x0) in enumerate(zip(a, x)):
            if func == gammaincc:
                # Exploit the fast integer path in gammaincc whenever
                # possible so that the computation doesn't take too
                # long
                a0, x0 = np.floor(a0), np.floor(x0)
            dataset.append((a0, x0, func(a0, x0)))
        dataset = np.array(dataset)
        filename = os.path.join(pwd, '..', 'tests', 'data', 'local',
                                '{}.txt'.format(func.__name__))
        np.savetxt(filename, dataset)

    print("{} minutes elapsed".format((time() - t0)/60))


if __name__ == "__main__":
    main()
