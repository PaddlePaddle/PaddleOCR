import numpy as np
from scipy.linalg import solve_banded
from .rotation import Rotation


def _create_skew_matrix(x):
    """Create skew-symmetric matrices corresponding to vectors.

    Parameters
    ----------
    x : ndarray, shape (n, 3)
        Set of vectors.

    Returns
    -------
    ndarray, shape (n, 3, 3)
    """
    result = np.zeros((len(x), 3, 3))
    result[:, 0, 1] = -x[:, 2]
    result[:, 0, 2] = x[:, 1]
    result[:, 1, 0] = x[:, 2]
    result[:, 1, 2] = -x[:, 0]
    result[:, 2, 0] = -x[:, 1]
    result[:, 2, 1] = x[:, 0]
    return result


def _matrix_vector_product_of_stacks(A, b):
    """Compute the product of stack of matrices and vectors."""
    return np.einsum("ijk,ik->ij", A, b)


def _angular_rate_to_rotvec_dot_matrix(rotvecs):
    """Compute matrices to transform angular rates to rot. vector derivatives.

    The matrices depend on the current attitude represented as a rotation
    vector.

    Parameters
    ----------
    rotvecs : ndarray, shape (n, 3)
        Set of rotation vectors.

    Returns
    -------
    ndarray, shape (n, 3, 3)
    """
    norm = np.linalg.norm(rotvecs, axis=1)
    k = np.empty_like(norm)

    mask = norm > 1e-4
    nm = norm[mask]
    k[mask] = (1 - 0.5 * nm / np.tan(0.5 * nm)) / nm**2
    mask = ~mask
    nm = norm[mask]
    k[mask] = 1/12 + 1/720 * nm**2

    skew = _create_skew_matrix(rotvecs)

    result = np.empty((len(rotvecs), 3, 3))
    result[:] = np.identity(3)
    result[:] += 0.5 * skew
    result[:] += k[:, None, None] * np.matmul(skew, skew)

    return result


def _rotvec_dot_to_angular_rate_matrix(rotvecs):
    """Compute matrices to transform rot. vector derivatives to angular rates.

    The matrices depend on the current attitude represented as a rotation
    vector.

    Parameters
    ----------
    rotvecs : ndarray, shape (n, 3)
        Set of rotation vectors.

    Returns
    -------
    ndarray, shape (n, 3, 3)
    """
    norm = np.linalg.norm(rotvecs, axis=1)
    k1 = np.empty_like(norm)
    k2 = np.empty_like(norm)

    mask = norm > 1e-4
    nm = norm[mask]
    k1[mask] = (1 - np.cos(nm)) / nm ** 2
    k2[mask] = (nm - np.sin(nm)) / nm ** 3

    mask = ~mask
    nm = norm[mask]
    k1[mask] = 0.5 - nm ** 2 / 24
    k2[mask] = 1 / 6 - nm ** 2 / 120

    skew = _create_skew_matrix(rotvecs)

    result = np.empty((len(rotvecs), 3, 3))
    result[:] = np.identity(3)
    result[:] -= k1[:, None, None] * skew
    result[:] += k2[:, None, None] * np.matmul(skew, skew)

    return result


def _angular_acceleration_nonlinear_term(rotvecs, rotvecs_dot):
    """Compute the non-linear term in angular acceleration.

    The angular acceleration contains a quadratic term with respect to
    the derivative of the rotation vector. This function computes that.

    Parameters
    ----------
    rotvecs : ndarray, shape (n, 3)
        Set of rotation vectors.
    rotvecs_dot: ndarray, shape (n, 3)
        Set of rotation vector derivatives.

    Returns
    -------
    ndarray, shape (n, 3)
    """
    norm = np.linalg.norm(rotvecs, axis=1)
    dp = np.sum(rotvecs * rotvecs_dot, axis=1)
    cp = np.cross(rotvecs, rotvecs_dot)
    ccp = np.cross(rotvecs, cp)
    dccp = np.cross(rotvecs_dot, cp)

    k1 = np.empty_like(norm)
    k2 = np.empty_like(norm)
    k3 = np.empty_like(norm)

    mask = norm > 1e-4
    nm = norm[mask]
    k1[mask] = (-nm * np.sin(nm) - 2 * (np.cos(nm) - 1)) / nm ** 4
    k2[mask] = (-2 * nm + 3 * np.sin(nm) - nm * np.cos(nm)) / nm ** 5
    k3[mask] = (nm - np.sin(nm)) / nm ** 3

    mask = ~mask
    nm = norm[mask]
    k1[mask] = 1/12 - nm ** 2 / 180
    k2[mask] = -1/60 + nm ** 2 / 12604
    k3[mask] = 1/6 - nm ** 2 / 120

    dp = dp[:, None]
    k1 = k1[:, None]
    k2 = k2[:, None]
    k3 = k3[:, None]

    return dp * (k1 * cp + k2 * ccp) + k3 * dccp


def _compute_angular_rate(rotvecs, rotvecs_dot):
    """Compute angular rates given rotation vectors and its derivatives.

    Parameters
    ----------
    rotvecs : ndarray, shape (n, 3)
        Set of rotation vectors.
    rotvecs_dot : ndarray, shape (n, 3)
        Set of rotation vector derivatives.

    Returns
    -------
    ndarray, shape (n, 3)
    """
    return _matrix_vector_product_of_stacks(
        _rotvec_dot_to_angular_rate_matrix(rotvecs), rotvecs_dot)


def _compute_angular_acceleration(rotvecs, rotvecs_dot, rotvecs_dot_dot):
    """Compute angular acceleration given rotation vector and its derivatives.

    Parameters
    ----------
    rotvecs : ndarray, shape (n, 3)
        Set of rotation vectors.
    rotvecs_dot : ndarray, shape (n, 3)
        Set of rotation vector derivatives.
    rotvecs_dot_dot : ndarray, shape (n, 3)
        Set of rotation vector second derivatives.

    Returns
    -------
    ndarray, shape (n, 3)
    """
    return (_compute_angular_rate(rotvecs, rotvecs_dot_dot) +
            _angular_acceleration_nonlinear_term(rotvecs, rotvecs_dot))


def _create_block_3_diagonal_matrix(A, B, d):
    """Create a 3-diagonal block matrix as banded.

    The matrix has the following structure:

        DB...
        ADB..
        .ADB.
        ..ADB
        ...AD

    The blocks A, B and D are 3-by-3 matrices. The D matrices has the form
    d * I.

    Parameters
    ----------
    A : ndarray, shape (n, 3, 3)
        Stack of A blocks.
    B : ndarray, shape (n, 3, 3)
        Stack of B blocks.
    d : ndarray, shape (n + 1,)
        Values for diagonal blocks.

    Returns
    -------
    ndarray, shape (11, 3 * (n + 1))
        Matrix in the banded form as used by `scipy.linalg.solve_banded`.
    """
    ind = np.arange(3)
    ind_blocks = np.arange(len(A))

    A_i = np.empty_like(A, dtype=int)
    A_i[:] = ind[:, None]
    A_i += 3 * (1 + ind_blocks[:, None, None])

    A_j = np.empty_like(A, dtype=int)
    A_j[:] = ind
    A_j += 3 * ind_blocks[:, None, None]

    B_i = np.empty_like(B, dtype=int)
    B_i[:] = ind[:, None]
    B_i += 3 * ind_blocks[:, None, None]

    B_j = np.empty_like(B, dtype=int)
    B_j[:] = ind
    B_j += 3 * (1 + ind_blocks[:, None, None])

    diag_i = diag_j = np.arange(3 * len(d))
    i = np.hstack((A_i.ravel(), B_i.ravel(), diag_i))
    j = np.hstack((A_j.ravel(), B_j.ravel(), diag_j))
    values = np.hstack((A.ravel(), B.ravel(), np.repeat(d, 3)))

    u = 5
    l = 5
    result = np.zeros((u + l + 1, 3 * len(d)))
    result[u + i - j, j] = values
    return result


class RotationSpline:
    """Interpolate rotations with continuous angular rate and acceleration.

    The rotation vectors between each consecutive orientation are cubic
    functions of time and it is guaranteed that angular rate and acceleration
    are continuous. Such interpolation are analogous to cubic spline
    interpolation.

    Refer to [1]_ for math and implementation details.

    Parameters
    ----------
    times : array_like, shape (N,)
        Times of the known rotations. At least 2 times must be specified.
    rotations : `Rotation` instance
        Rotations to perform the interpolation between. Must contain N
        rotations.

    Methods
    -------
    __call__

    References
    ----------
    .. [1] `Smooth Attitude Interpolation
            <https://github.com/scipy/scipy/files/2932755/attitude_interpolation.pdf>`_

    Examples
    --------
    >>> from scipy.spatial.transform import Rotation, RotationSpline

    Define the sequence of times and rotations from the Euler angles:

    >>> times = [0, 10, 20, 40]
    >>> angles = [[-10, 20, 30], [0, 15, 40], [-30, 45, 30], [20, 45, 90]]
    >>> rotations = Rotation.from_euler('XYZ', angles, degrees=True)

    Create the interpolator object:

    >>> spline = RotationSpline(times, rotations)

    Interpolate the Euler angles, angular rate and acceleration:

    >>> angular_rate = np.rad2deg(spline(times, 1))
    >>> angular_acceleration = np.rad2deg(spline(times, 2))
    >>> times_plot = np.linspace(times[0], times[-1], 100)
    >>> angles_plot = spline(times_plot).as_euler('XYZ', degrees=True)
    >>> angular_rate_plot = np.rad2deg(spline(times_plot, 1))
    >>> angular_acceleration_plot = np.rad2deg(spline(times_plot, 2))

    On this plot you see that Euler angles are continuous and smooth:

    >>> import matplotlib.pyplot as plt
    >>> plt.plot(times_plot, angles_plot)
    >>> plt.plot(times, angles, 'x')
    >>> plt.title("Euler angles")
    >>> plt.show()

    The angular rate is also smooth:

    >>> plt.plot(times_plot, angular_rate_plot)
    >>> plt.plot(times, angular_rate, 'x')
    >>> plt.title("Angular rate")
    >>> plt.show()

    The angular acceleration is continuous, but not smooth. Also note that
    the angular acceleration is not a piecewise-linear function, because
    it is different from the second derivative of the rotation vector (which
    is a piecewise-linear function as in the cubic spline).

    >>> plt.plot(times_plot, angular_acceleration_plot)
    >>> plt.plot(times, angular_acceleration, 'x')
    >>> plt.title("Angular acceleration")
    >>> plt.show()
    """
    # Parameters for the solver for angular rate.
    MAX_ITER = 10
    TOL = 1e-9

    def _solve_for_angular_rates(self, dt, angular_rates, rotvecs):
        angular_rate_first = angular_rates[0].copy()

        A = _angular_rate_to_rotvec_dot_matrix(rotvecs)
        A_inv = _rotvec_dot_to_angular_rate_matrix(rotvecs)
        M = _create_block_3_diagonal_matrix(
            2 * A_inv[1:-1] / dt[1:-1, None, None],
            2 * A[1:-1] / dt[1:-1, None, None],
            4 * (1 / dt[:-1] + 1 / dt[1:]))

        b0 = 6 * (rotvecs[:-1] * dt[:-1, None] ** -2 +
                  rotvecs[1:] * dt[1:, None] ** -2)
        b0[0] -= 2 / dt[0] * A_inv[0].dot(angular_rate_first)
        b0[-1] -= 2 / dt[-1] * A[-1].dot(angular_rates[-1])

        for iteration in range(self.MAX_ITER):
            rotvecs_dot = _matrix_vector_product_of_stacks(A, angular_rates)
            delta_beta = _angular_acceleration_nonlinear_term(
                rotvecs[:-1], rotvecs_dot[:-1])
            b = b0 - delta_beta
            angular_rates_new = solve_banded((5, 5), M, b.ravel())
            angular_rates_new = angular_rates_new.reshape((-1, 3))

            delta = np.abs(angular_rates_new - angular_rates[:-1])
            angular_rates[:-1] = angular_rates_new
            if np.all(delta < self.TOL * (1 + np.abs(angular_rates_new))):
                break

        rotvecs_dot = _matrix_vector_product_of_stacks(A, angular_rates)
        angular_rates = np.vstack((angular_rate_first, angular_rates[:-1]))

        return angular_rates, rotvecs_dot

    def __init__(self, times, rotations):
        from scipy.interpolate import PPoly

        if rotations.single:
            raise ValueError("`rotations` must be a sequence of rotations.")

        if len(rotations) == 1:
            raise ValueError("`rotations` must contain at least 2 rotations.")

        times = np.asarray(times, dtype=float)
        if times.ndim != 1:
            raise ValueError("`times` must be 1-dimensional.")

        if len(times) != len(rotations):
            raise ValueError("Expected number of rotations to be equal to "
                             "number of timestamps given, got {} rotations "
                             "and {} timestamps."
                             .format(len(rotations), len(times)))

        dt = np.diff(times)
        if np.any(dt <= 0):
            raise ValueError("Values in `times` must be in a strictly "
                             "increasing order.")

        rotvecs = (rotations[:-1].inv() * rotations[1:]).as_rotvec()
        angular_rates = rotvecs / dt[:, None]

        if len(rotations) == 2:
            rotvecs_dot = angular_rates
        else:
            angular_rates, rotvecs_dot = self._solve_for_angular_rates(
                dt, angular_rates, rotvecs)

        dt = dt[:, None]
        coeff = np.empty((4, len(times) - 1, 3))
        coeff[0] = (-2 * rotvecs + dt * angular_rates
                    + dt * rotvecs_dot) / dt ** 3
        coeff[1] = (3 * rotvecs - 2 * dt * angular_rates
                    - dt * rotvecs_dot) / dt ** 2
        coeff[2] = angular_rates
        coeff[3] = 0

        self.times = times
        self.rotations = rotations
        self.interpolator = PPoly(coeff, times)

    def __call__(self, times, order=0):
        """Compute interpolated values.

        Parameters
        ----------
        times : float or array_like
            Times of interest.
        order : {0, 1, 2}, optional
            Order of differentiation:

                * 0 (default) : return Rotation
                * 1 : return the angular rate in rad/sec
                * 2 : return the angular acceleration in rad/sec/sec

        Returns
        -------
        Interpolated Rotation, angular rate or acceleration.
        """
        if order not in [0, 1, 2]:
            raise ValueError("`order` must be 0, 1 or 2.")

        times = np.asarray(times, dtype=float)
        if times.ndim > 1:
            raise ValueError("`times` must be at most 1-dimensional.")

        singe_time = times.ndim == 0
        times = np.atleast_1d(times)

        rotvecs = self.interpolator(times)
        if order == 0:
            index = np.searchsorted(self.times, times, side='right')
            index -= 1
            index[index < 0] = 0
            n_segments = len(self.times) - 1
            index[index > n_segments - 1] = n_segments - 1
            result = self.rotations[index] * Rotation.from_rotvec(rotvecs)
        elif order == 1:
            rotvecs_dot = self.interpolator(times, 1)
            result = _compute_angular_rate(rotvecs, rotvecs_dot)
        elif order == 2:
            rotvecs_dot = self.interpolator(times, 1)
            rotvecs_dot_dot = self.interpolator(times, 2)
            result = _compute_angular_acceleration(rotvecs, rotvecs_dot,
                                                   rotvecs_dot_dot)
        else:
            assert False

        if singe_time:
            result = result[0]

        return result
