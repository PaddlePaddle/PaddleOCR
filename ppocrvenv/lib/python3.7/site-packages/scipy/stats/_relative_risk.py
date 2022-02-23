
import operator
from dataclasses import dataclass
import numpy as np
from scipy.special import ndtri
from ._common import ConfidenceInterval


def _validate_int(n, bound, name):
    msg = f'{name} must be an integer not less than {bound}, but got {n!r}'
    try:
        n = operator.index(n)
    except TypeError:
        raise TypeError(msg) from None
    if n < bound:
        raise ValueError(msg)
    return n


@dataclass
class RelativeRiskResult:
    """
    Result of `scipy.stats.contingency.relative_risk`.

    Attributes
    ----------
    relative_risk : float
        This is::

            (exposed_cases/exposed_total) / (control_cases/control_total)

    exposed_cases : int
        The number of "cases" (i.e. occurrence of disease or other event
        of interest) among the sample of "exposed" individuals.
    exposed_total : int
        The total number of "exposed" individuals in the sample.
    control_cases : int
        The number of "cases" among the sample of "control" or non-exposed
        individuals.
    control_total : int
        The total number of "control" individuals in the sample.

    Methods
    -------
    confidence_interval :
        Compute the confidence interval for the relative risk estimate.
    """

    relative_risk: float
    exposed_cases: int
    exposed_total: int
    control_cases: int
    control_total: int

    def confidence_interval(self, confidence_level=0.95):
        """
        Compute the confidence interval for the relative risk.

        The confidence interval is computed using the Katz method
        (i.e. "Method C" of [1]_; see also [2]_, section 3.1.2).

        Parameters
        ----------
        confidence_level : float, optional
            The confidence level to use for the confidence interval.
            Default is 0.95.

        Returns
        -------
        ci : ConfidenceInterval instance
            The return value is an object with attributes ``low`` and
            ``high`` that hold the confidence interval.

        References
        ----------
        .. [1] D. Katz, J. Baptista, S. P. Azen and M. C. Pike, "Obtaining
               confidence intervals for the risk ratio in cohort studies",
               Biometrics, 34, 469-474 (1978).
        .. [2] Hardeo Sahai and Anwer Khurshid, Statistics in Epidemiology,
               CRC Press LLC, Boca Raton, FL, USA (1996).


        Examples
        --------
        >>> from scipy.stats.contingency import relative_risk
        >>> result = relative_risk(exposed_cases=10, exposed_total=75,
        ...                        control_cases=12, control_total=225)
        >>> result.relative_risk
        2.5
        >>> result.confidence_interval()
        ConfidenceInterval(low=1.1261564003469628, high=5.549850800541033)
        """
        if not 0 <= confidence_level <= 1:
            raise ValueError('confidence_level must be in the interval '
                             '[0, 1].')

        # Handle edge cases where either exposed_cases or control_cases
        # is zero.  We follow the convention of the R function riskratio
        # from the epitools library.
        if self.exposed_cases == 0 and self.control_cases == 0:
            # relative risk is nan.
            return ConfidenceInterval(low=np.nan, high=np.nan)
        elif self.exposed_cases == 0:
            # relative risk is 0.
            return ConfidenceInterval(low=0.0, high=np.nan)
        elif self.control_cases == 0:
            # relative risk is inf
            return ConfidenceInterval(low=np.nan, high=np.inf)

        alpha = 1 - confidence_level
        z = ndtri(1 - alpha/2)
        rr = self.relative_risk

        # Estimate of the variance of log(rr) is
        # var(log(rr)) = 1/exposed_cases - 1/exposed_total +
        #                1/control_cases - 1/control_total
        # and the standard error is the square root of that.
        se = np.sqrt(1/self.exposed_cases - 1/self.exposed_total +
                     1/self.control_cases - 1/self.control_total)
        delta = z*se
        katz_lo = rr*np.exp(-delta)
        katz_hi = rr*np.exp(delta)
        return ConfidenceInterval(low=katz_lo, high=katz_hi)


def relative_risk(exposed_cases, exposed_total, control_cases, control_total):
    """
    Compute the relative risk (also known as the risk ratio).

    This function computes the relative risk associated with a 2x2
    contingency table ([1]_, section 2.2.3; [2]_, section 3.1.2). Instead
    of accepting a table as an argument, the individual numbers that are
    used to compute the relative risk are given as separate parameters.
    This is to avoid the ambiguity of which row or column of the contingency
    table corresponds to the "exposed" cases and which corresponds to the
    "control" cases.  Unlike, say, the odds ratio, the relative risk is not
    invariant under an interchange of the rows or columns.

    Parameters
    ----------
    exposed_cases : nonnegative int
        The number of "cases" (i.e. occurrence of disease or other event
        of interest) among the sample of "exposed" individuals.
    exposed_total : positive int
        The total number of "exposed" individuals in the sample.
    control_cases : nonnegative int
        The number of "cases" among the sample of "control" or non-exposed
        individuals.
    control_total : positive int
        The total number of "control" individuals in the sample.

    Returns
    -------
    result : instance of `~scipy.stats._result_classes.RelativeRiskResult`
        The object has the float attribute ``relative_risk``, which is::

            rr = (exposed_cases/exposed_total) / (control_cases/control_total)

        The object also has the method ``confidence_interval`` to compute
        the confidence interval of the relative risk for a given confidence
        level.

    Notes
    -----
    The R package epitools has the function `riskratio`, which accepts
    a table with the following layout::

                        disease=0   disease=1
        exposed=0 (ref)    n00         n01
        exposed=1          n10         n11

    With a 2x2 table in the above format, the estimate of the CI is
    computed by `riskratio` when the argument method="wald" is given,
    or with the function `riskratio.wald`.

    For example, in a test of the incidence of lung cancer among a
    sample of smokers and nonsmokers, the "exposed" category would
    correspond to "is a smoker" and the "disease" category would
    correspond to "has or had lung cancer".

    To pass the same data to ``relative_risk``, use::

        relative_risk(n11, n10 + n11, n01, n00 + n01)

    .. versionadded:: 1.7.0

    References
    ----------
    .. [1] Alan Agresti, An Introduction to Categorical Data Analysis
           (second edition), Wiley, Hoboken, NJ, USA (2007).
    .. [2] Hardeo Sahai and Anwer Khurshid, Statistics in Epidemiology,
           CRC Press LLC, Boca Raton, FL, USA (1996).

    Examples
    --------
    >>> from scipy.stats.contingency import relative_risk

    This example is from Example 3.1 of [2]_.  The results of a heart
    disease study are summarized in the following table::

                 High CAT   Low CAT    Total
                 --------   -------    -----
        CHD         27         44        71
        No CHD      95        443       538

        Total      122        487       609

    CHD is coronary heart disease, and CAT refers to the level of
    circulating catecholamine.  CAT is the "exposure" variable, and
    high CAT is the "exposed" category. So the data from the table
    to be passed to ``relative_risk`` is::

        exposed_cases = 27
        exposed_total = 122
        control_cases = 44
        control_total = 487

    >>> result = relative_risk(27, 122, 44, 487)
    >>> result.relative_risk
    2.4495156482861398

    Find the confidence interval for the relative risk.

    >>> result.confidence_interval(confidence_level=0.95)
    ConfidenceInterval(low=1.5836990926700116, high=3.7886786315466354)

    The interval does not contain 1, so the data supports the statement
    that high CAT is associated with greater risk of CHD.
    """
    # Relative risk is a trivial calculation.  The nontrivial part is in the
    # `confidence_interval` method of the RelativeRiskResult class.

    exposed_cases = _validate_int(exposed_cases, 0, "exposed_cases")
    exposed_total = _validate_int(exposed_total, 1, "exposed_total")
    control_cases = _validate_int(control_cases, 0, "control_cases")
    control_total = _validate_int(control_total, 1, "control_total")

    if exposed_cases > exposed_total:
        raise ValueError('exposed_cases must not exceed exposed_total.')
    if control_cases > control_total:
        raise ValueError('control_cases must not exceed control_total.')

    if exposed_cases == 0 and control_cases == 0:
        # relative risk is 0/0.
        rr = np.nan
    elif exposed_cases == 0:
        # relative risk is 0/nonzero
        rr = 0.0
    elif control_cases == 0:
        # relative risk is nonzero/0.
        rr = np.inf
    else:
        p1 = exposed_cases / exposed_total
        p2 = control_cases / control_total
        rr = p1 / p2
    return RelativeRiskResult(relative_risk=rr,
                              exposed_cases=exposed_cases,
                              exposed_total=exposed_total,
                              control_cases=control_cases,
                              control_total=control_total)
