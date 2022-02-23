"""Compute a Pade approximation for the principle branch of the
Lambert W function around 0 and compare it to various other
approximations.

"""
import numpy as np

try:
    import mpmath
    import matplotlib.pyplot as plt  # type: ignore[import]
except ImportError:
    pass


def lambertw_pade():
    derivs = [mpmath.diff(mpmath.lambertw, 0, n=n) for n in range(6)]
    p, q = mpmath.pade(derivs, 3, 2)
    return p, q


def main():
    print(__doc__)
    with mpmath.workdps(50):
        p, q = lambertw_pade()
        p, q = p[::-1], q[::-1]
        print("p = {}".format(p))
        print("q = {}".format(q))

    x, y = np.linspace(-1.5, 1.5, 75), np.linspace(-1.5, 1.5, 75)
    x, y = np.meshgrid(x, y)
    z = x + 1j*y
    lambertw_std = []
    for z0 in z.flatten():
        lambertw_std.append(complex(mpmath.lambertw(z0)))
    lambertw_std = np.array(lambertw_std).reshape(x.shape)

    fig, axes = plt.subplots(nrows=3, ncols=1)
    # Compare Pade approximation to true result
    p = np.array([float(p0) for p0 in p])
    q = np.array([float(q0) for q0 in q])
    pade_approx = np.polyval(p, z)/np.polyval(q, z)
    pade_err = abs(pade_approx - lambertw_std)
    axes[0].pcolormesh(x, y, pade_err)
    # Compare two terms of asymptotic series to true result
    asy_approx = np.log(z) - np.log(np.log(z))
    asy_err = abs(asy_approx - lambertw_std)
    axes[1].pcolormesh(x, y, asy_err)
    # Compare two terms of the series around the branch point to the
    # true result
    p = np.sqrt(2*(np.exp(1)*z + 1))
    series_approx = -1 + p - p**2/3
    series_err = abs(series_approx - lambertw_std)
    im = axes[2].pcolormesh(x, y, series_err)

    fig.colorbar(im, ax=axes.ravel().tolist())
    plt.show()

    fig, ax = plt.subplots(nrows=1, ncols=1)
    pade_better = pade_err < asy_err
    im = ax.pcolormesh(x, y, pade_better)
    t = np.linspace(-0.3, 0.3)
    ax.plot(-2.5*abs(t) - 0.2, t, 'r')
    fig.colorbar(im, ax=ax)
    plt.show()


if __name__ == '__main__':
    main()
