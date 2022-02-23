import numpy as np
from numpy.testing import assert_allclose
import scipy.special as sc


def test_first_harmonics():
    # Test against explicit representations of the first four
    # spherical harmonics which use `theta` as the azimuthal angle,
    # `phi` as the polar angle, and include the Condon-Shortley
    # phase.

    # Notation is Ymn
    def Y00(theta, phi):
        return 0.5*np.sqrt(1/np.pi)

    def Yn11(theta, phi):
        return 0.5*np.sqrt(3/(2*np.pi))*np.exp(-1j*theta)*np.sin(phi)

    def Y01(theta, phi):
        return 0.5*np.sqrt(3/np.pi)*np.cos(phi)

    def Y11(theta, phi):
        return -0.5*np.sqrt(3/(2*np.pi))*np.exp(1j*theta)*np.sin(phi)

    harms = [Y00, Yn11, Y01, Y11]
    m = [0, -1, 0, 1]
    n = [0, 1, 1, 1]

    theta = np.linspace(0, 2*np.pi)
    phi = np.linspace(0, np.pi)
    theta, phi = np.meshgrid(theta, phi)

    for harm, m, n in zip(harms, m, n):
        assert_allclose(sc.sph_harm(m, n, theta, phi),
                        harm(theta, phi),
                        rtol=1e-15, atol=1e-15,
                        err_msg="Y^{}_{} incorrect".format(m, n))
