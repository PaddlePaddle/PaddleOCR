"""
Distribution functions used in GLM
"""

# Author: Christian Lorentzen <lorentzen.ch@googlemail.com>
# License: BSD 3 clause

from abc import ABCMeta, abstractmethod
from collections import namedtuple
import numbers

import numpy as np
from scipy.special import xlogy


DistributionBoundary = namedtuple("DistributionBoundary", ("value", "inclusive"))


class ExponentialDispersionModel(metaclass=ABCMeta):
    r"""Base class for reproductive Exponential Dispersion Models (EDM).

    The pdf of :math:`Y\sim \mathrm{EDM}(y_\textrm{pred}, \phi)` is given by

    .. math:: p(y| \theta, \phi) = c(y, \phi)
        \exp\left(\frac{\theta y-A(\theta)}{\phi}\right)
        = \tilde{c}(y, \phi)
            \exp\left(-\frac{d(y, y_\textrm{pred})}{2\phi}\right)

    with mean :math:`\mathrm{E}[Y] = A'(\theta) = y_\textrm{pred}`,
    variance :math:`\mathrm{Var}[Y] = \phi \cdot v(y_\textrm{pred})`,
    unit variance :math:`v(y_\textrm{pred})` and
    unit deviance :math:`d(y,y_\textrm{pred})`.

    Methods
    -------
    deviance
    deviance_derivative
    in_y_range
    unit_deviance
    unit_deviance_derivative
    unit_variance

    References
    ----------
    https://en.wikipedia.org/wiki/Exponential_dispersion_model.
    """

    def in_y_range(self, y):
        """Returns ``True`` if y is in the valid range of Y~EDM.

        Parameters
        ----------
        y : array of shape (n_samples,)
            Target values.
        """
        # Note that currently supported distributions have +inf upper bound

        if not isinstance(self._lower_bound, DistributionBoundary):
            raise TypeError(
                "_lower_bound attribute must be of type DistributionBoundary"
            )

        if self._lower_bound.inclusive:
            return np.greater_equal(y, self._lower_bound.value)
        else:
            return np.greater(y, self._lower_bound.value)

    @abstractmethod
    def unit_variance(self, y_pred):
        r"""Compute the unit variance function.

        The unit variance :math:`v(y_\textrm{pred})` determines the variance as
        a function of the mean :math:`y_\textrm{pred}` by
        :math:`\mathrm{Var}[Y_i] = \phi/s_i*v(y_\textrm{pred}_i)`.
        It can also be derived from the unit deviance
        :math:`d(y,y_\textrm{pred})` as

        .. math:: v(y_\textrm{pred}) = \frac{2}{
            \frac{\partial^2 d(y,y_\textrm{pred})}{
            \partialy_\textrm{pred}^2}}\big|_{y=y_\textrm{pred}}

        See also :func:`variance`.

        Parameters
        ----------
        y_pred : array of shape (n_samples,)
            Predicted mean.
        """

    @abstractmethod
    def unit_deviance(self, y, y_pred, check_input=False):
        r"""Compute the unit deviance.

        The unit_deviance :math:`d(y,y_\textrm{pred})` can be defined by the
        log-likelihood as
        :math:`d(y,y_\textrm{pred}) = -2\phi\cdot
        \left(loglike(y,y_\textrm{pred},\phi) - loglike(y,y,\phi)\right).`

        Parameters
        ----------
        y : array of shape (n_samples,)
            Target values.

        y_pred : array of shape (n_samples,)
            Predicted mean.

        check_input : bool, default=False
            If True raise an exception on invalid y or y_pred values, otherwise
            they will be propagated as NaN.
        Returns
        -------
        deviance: array of shape (n_samples,)
            Computed deviance
        """

    def unit_deviance_derivative(self, y, y_pred):
        r"""Compute the derivative of the unit deviance w.r.t. y_pred.

        The derivative of the unit deviance is given by
        :math:`\frac{\partial}{\partialy_\textrm{pred}}d(y,y_\textrm{pred})
             = -2\frac{y-y_\textrm{pred}}{v(y_\textrm{pred})}`
        with unit variance :math:`v(y_\textrm{pred})`.

        Parameters
        ----------
        y : array of shape (n_samples,)
            Target values.

        y_pred : array of shape (n_samples,)
            Predicted mean.
        """
        return -2 * (y - y_pred) / self.unit_variance(y_pred)

    def deviance(self, y, y_pred, weights=1):
        r"""Compute the deviance.

        The deviance is a weighted sum of the per sample unit deviances,
        :math:`D = \sum_i s_i \cdot d(y_i, y_\textrm{pred}_i)`
        with weights :math:`s_i` and unit deviance
        :math:`d(y,y_\textrm{pred})`.
        In terms of the log-likelihood it is :math:`D = -2\phi\cdot
        \left(loglike(y,y_\textrm{pred},\frac{phi}{s})
        - loglike(y,y,\frac{phi}{s})\right)`.

        Parameters
        ----------
        y : array of shape (n_samples,)
            Target values.

        y_pred : array of shape (n_samples,)
            Predicted mean.

        weights : {int, array of shape (n_samples,)}, default=1
            Weights or exposure to which variance is inverse proportional.
        """
        return np.sum(weights * self.unit_deviance(y, y_pred))

    def deviance_derivative(self, y, y_pred, weights=1):
        r"""Compute the derivative of the deviance w.r.t. y_pred.

        It gives :math:`\frac{\partial}{\partial y_\textrm{pred}}
        D(y, \y_\textrm{pred}; weights)`.

        Parameters
        ----------
        y : array, shape (n_samples,)
            Target values.

        y_pred : array, shape (n_samples,)
            Predicted mean.

        weights : {int, array of shape (n_samples,)}, default=1
            Weights or exposure to which variance is inverse proportional.
        """
        return weights * self.unit_deviance_derivative(y, y_pred)


class TweedieDistribution(ExponentialDispersionModel):
    r"""A class for the Tweedie distribution.

    A Tweedie distribution with mean :math:`y_\textrm{pred}=\mathrm{E}[Y]`
    is uniquely defined by it's mean-variance relationship
    :math:`\mathrm{Var}[Y] \propto y_\textrm{pred}^power`.

    Special cases are:

    ===== ================
    Power Distribution
    ===== ================
    0     Normal
    1     Poisson
    (1,2) Compound Poisson
    2     Gamma
    3     Inverse Gaussian

    Parameters
    ----------
    power : float, default=0
            The variance power of the `unit_variance`
            :math:`v(y_\textrm{pred}) = y_\textrm{pred}^{power}`.
            For ``0<power<1``, no distribution exists.
    """

    def __init__(self, power=0):
        self.power = power

    @property
    def power(self):
        return self._power

    @power.setter
    def power(self, power):
        # We use a property with a setter, to update lower and
        # upper bound when the power parameter is updated e.g. in grid
        # search.
        if not isinstance(power, numbers.Real):
            raise TypeError("power must be a real number, input was {0}".format(power))

        if power <= 0:
            # Extreme Stable or Normal distribution
            self._lower_bound = DistributionBoundary(-np.Inf, inclusive=False)
        elif 0 < power < 1:
            raise ValueError(
                "Tweedie distribution is only defined for power<=0 and power>=1."
            )
        elif 1 <= power < 2:
            # Poisson or Compound Poisson distribution
            self._lower_bound = DistributionBoundary(0, inclusive=True)
        elif power >= 2:
            # Gamma, Positive Stable, Inverse Gaussian distributions
            self._lower_bound = DistributionBoundary(0, inclusive=False)
        else:  # pragma: no cover
            # this branch should be unreachable.
            raise ValueError

        self._power = power

    def unit_variance(self, y_pred):
        """Compute the unit variance of a Tweedie distribution
        v(y_\textrm{pred})=y_\textrm{pred}**power.

        Parameters
        ----------
        y_pred : array of shape (n_samples,)
            Predicted mean.
        """
        return np.power(y_pred, self.power)

    def unit_deviance(self, y, y_pred, check_input=False):
        r"""Compute the unit deviance.

        The unit_deviance :math:`d(y,y_\textrm{pred})` can be defined by the
        log-likelihood as
        :math:`d(y,y_\textrm{pred}) = -2\phi\cdot
        \left(loglike(y,y_\textrm{pred},\phi) - loglike(y,y,\phi)\right).`

        Parameters
        ----------
        y : array of shape (n_samples,)
            Target values.

        y_pred : array of shape (n_samples,)
            Predicted mean.

        check_input : bool, default=False
            If True raise an exception on invalid y or y_pred values, otherwise
            they will be propagated as NaN.
        Returns
        -------
        deviance: array of shape (n_samples,)
            Computed deviance
        """
        p = self.power

        if check_input:
            message = (
                "Mean Tweedie deviance error with power={} can only be used on ".format(
                    p
                )
            )
            if p < 0:
                # 'Extreme stable', y any real number, y_pred > 0
                if (y_pred <= 0).any():
                    raise ValueError(message + "strictly positive y_pred.")
            elif p == 0:
                # Normal, y and y_pred can be any real number
                pass
            elif 0 < p < 1:
                raise ValueError(
                    "Tweedie deviance is only defined for power<=0 and power>=1."
                )
            elif 1 <= p < 2:
                # Poisson and compound Poisson distribution, y >= 0, y_pred > 0
                if (y < 0).any() or (y_pred <= 0).any():
                    raise ValueError(
                        message + "non-negative y and strictly positive y_pred."
                    )
            elif p >= 2:
                # Gamma and Extreme stable distribution, y and y_pred > 0
                if (y <= 0).any() or (y_pred <= 0).any():
                    raise ValueError(message + "strictly positive y and y_pred.")
            else:  # pragma: nocover
                # Unreachable statement
                raise ValueError

        if p < 0:
            # 'Extreme stable', y any real number, y_pred > 0
            dev = 2 * (
                np.power(np.maximum(y, 0), 2 - p) / ((1 - p) * (2 - p))
                - y * np.power(y_pred, 1 - p) / (1 - p)
                + np.power(y_pred, 2 - p) / (2 - p)
            )

        elif p == 0:
            # Normal distribution, y and y_pred any real number
            dev = (y - y_pred) ** 2
        elif p < 1:
            raise ValueError(
                "Tweedie deviance is only defined for power<=0 and power>=1."
            )
        elif p == 1:
            # Poisson distribution
            dev = 2 * (xlogy(y, y / y_pred) - y + y_pred)
        elif p == 2:
            # Gamma distribution
            dev = 2 * (np.log(y_pred / y) + y / y_pred - 1)
        else:
            dev = 2 * (
                np.power(y, 2 - p) / ((1 - p) * (2 - p))
                - y * np.power(y_pred, 1 - p) / (1 - p)
                + np.power(y_pred, 2 - p) / (2 - p)
            )
        return dev


class NormalDistribution(TweedieDistribution):
    """Class for the Normal (aka Gaussian) distribution."""

    def __init__(self):
        super().__init__(power=0)


class PoissonDistribution(TweedieDistribution):
    """Class for the scaled Poisson distribution."""

    def __init__(self):
        super().__init__(power=1)


class GammaDistribution(TweedieDistribution):
    """Class for the Gamma distribution."""

    def __init__(self):
        super().__init__(power=2)


class InverseGaussianDistribution(TweedieDistribution):
    """Class for the scaled InverseGaussianDistribution distribution."""

    def __init__(self):
        super().__init__(power=3)


EDM_DISTRIBUTIONS = {
    "normal": NormalDistribution,
    "poisson": PoissonDistribution,
    "gamma": GammaDistribution,
    "inverse-gaussian": InverseGaussianDistribution,
}
