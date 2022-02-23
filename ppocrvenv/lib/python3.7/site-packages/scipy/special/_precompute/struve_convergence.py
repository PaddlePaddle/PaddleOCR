"""
Convergence regions of the expansions used in ``struve.c``

Note that for v >> z both functions tend rapidly to 0,
and for v << -z, they tend to infinity.

The floating-point functions over/underflow in the lower left and right
corners of the figure.


Figure legend
=============

Red region
    Power series is close (1e-12) to the mpmath result

Blue region
    Asymptotic series is close to the mpmath result

Green region
    Bessel series is close to the mpmath result

Dotted colored lines
    Boundaries of the regions

Solid colored lines
    Boundaries estimated by the routine itself. These will be used
    for determining which of the results to use.

Black dashed line
    The line z = 0.7*|v| + 12

"""
import numpy as np
import matplotlib.pyplot as plt  # type: ignore[import]

import mpmath


def err_metric(a, b, atol=1e-290):
    m = abs(a - b) / (atol + abs(b))
    m[np.isinf(b) & (a == b)] = 0
    return m


def do_plot(is_h=True):
    from scipy.special._ufuncs import (_struve_power_series,
                                       _struve_asymp_large_z,
                                       _struve_bessel_series)

    vs = np.linspace(-1000, 1000, 91)
    zs = np.sort(np.r_[1e-5, 1.0, np.linspace(0, 700, 91)[1:]])

    rp = _struve_power_series(vs[:,None], zs[None,:], is_h)
    ra = _struve_asymp_large_z(vs[:,None], zs[None,:], is_h)
    rb = _struve_bessel_series(vs[:,None], zs[None,:], is_h)

    mpmath.mp.dps = 50
    if is_h:
        sh = lambda v, z: float(mpmath.struveh(mpmath.mpf(v), mpmath.mpf(z)))
    else:
        sh = lambda v, z: float(mpmath.struvel(mpmath.mpf(v), mpmath.mpf(z)))
    ex = np.vectorize(sh, otypes='d')(vs[:,None], zs[None,:])

    err_a = err_metric(ra[0], ex) + 1e-300
    err_p = err_metric(rp[0], ex) + 1e-300
    err_b = err_metric(rb[0], ex) + 1e-300

    err_est_a = abs(ra[1]/ra[0])
    err_est_p = abs(rp[1]/rp[0])
    err_est_b = abs(rb[1]/rb[0])

    z_cutoff = 0.7*abs(vs) + 12

    levels = [-1000, -12]

    plt.cla()

    plt.hold(1)
    plt.contourf(vs, zs, np.log10(err_p).T, levels=levels, colors=['r', 'r'], alpha=0.1)
    plt.contourf(vs, zs, np.log10(err_a).T, levels=levels, colors=['b', 'b'], alpha=0.1)
    plt.contourf(vs, zs, np.log10(err_b).T, levels=levels, colors=['g', 'g'], alpha=0.1)

    plt.contour(vs, zs, np.log10(err_p).T, levels=levels, colors=['r', 'r'], linestyles=[':', ':'])
    plt.contour(vs, zs, np.log10(err_a).T, levels=levels, colors=['b', 'b'], linestyles=[':', ':'])
    plt.contour(vs, zs, np.log10(err_b).T, levels=levels, colors=['g', 'g'], linestyles=[':', ':'])

    lp = plt.contour(vs, zs, np.log10(err_est_p).T, levels=levels, colors=['r', 'r'], linestyles=['-', '-'])
    la = plt.contour(vs, zs, np.log10(err_est_a).T, levels=levels, colors=['b', 'b'], linestyles=['-', '-'])
    lb = plt.contour(vs, zs, np.log10(err_est_b).T, levels=levels, colors=['g', 'g'], linestyles=['-', '-'])

    plt.clabel(lp, fmt={-1000: 'P', -12: 'P'})
    plt.clabel(la, fmt={-1000: 'A', -12: 'A'})
    plt.clabel(lb, fmt={-1000: 'B', -12: 'B'})

    plt.plot(vs, z_cutoff, 'k--')

    plt.xlim(vs.min(), vs.max())
    plt.ylim(zs.min(), zs.max())

    plt.xlabel('v')
    plt.ylabel('z')


def main():
    plt.clf()
    plt.subplot(121)
    do_plot(True)
    plt.title('Struve H')

    plt.subplot(122)
    do_plot(False)
    plt.title('Struve L')

    plt.savefig('struve_convergence.png')
    plt.show()


if __name__ == "__main__":
    main()
