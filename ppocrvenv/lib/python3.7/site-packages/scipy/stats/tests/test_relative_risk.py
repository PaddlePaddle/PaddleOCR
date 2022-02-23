
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal
from scipy.stats.contingency import relative_risk


# Test just the calculation of the relative risk, including edge
# cases that result in a relative risk of 0, inf or nan.
@pytest.mark.parametrize(
    'exposed_cases, exposed_total, control_cases, control_total, expected_rr',
    [(1, 4, 3, 8, 0.25 / 0.375),
     (0, 10, 5, 20, 0),
     (0, 10, 0, 20, np.nan),
     (5, 15, 0, 20, np.inf)]
)
def test_relative_risk(exposed_cases, exposed_total,
                       control_cases, control_total, expected_rr):
    result = relative_risk(exposed_cases, exposed_total,
                           control_cases, control_total)
    assert_allclose(result.relative_risk, expected_rr, rtol=1e-13)


def test_relative_risk_confidence_interval():
    result = relative_risk(exposed_cases=16, exposed_total=128,
                           control_cases=24, control_total=256)
    rr = result.relative_risk
    ci = result.confidence_interval(confidence_level=0.95)
    # The corresponding calculation in R using the epitools package.
    #
    # > library(epitools)
    # > c <- matrix(c(232, 112, 24, 16), nrow=2)
    # > result <- riskratio(c)
    # > result$measure
    #               risk ratio with 95% C.I.
    # Predictor  estimate     lower    upper
    #   Exposed1 1.000000        NA       NA
    #   Exposed2 1.333333 0.7347317 2.419628
    #
    # The last line is the result that we want.
    assert_allclose(rr, 4/3)
    assert_allclose((ci.low, ci.high), (0.7347317, 2.419628), rtol=5e-7)


def test_relative_risk_ci_conflevel0():
    result = relative_risk(exposed_cases=4, exposed_total=12,
                           control_cases=5, control_total=30)
    rr = result.relative_risk
    assert_allclose(rr, 2.0, rtol=1e-14)
    ci = result.confidence_interval(0)
    assert_allclose((ci.low, ci.high), (2.0, 2.0), rtol=1e-12)


def test_relative_risk_ci_conflevel1():
    result = relative_risk(exposed_cases=4, exposed_total=12,
                           control_cases=5, control_total=30)
    ci = result.confidence_interval(1)
    assert_equal((ci.low, ci.high), (0, np.inf))


def test_relative_risk_ci_edge_cases_00():
    result = relative_risk(exposed_cases=0, exposed_total=12,
                           control_cases=0, control_total=30)
    assert_equal(result.relative_risk, np.nan)
    ci = result.confidence_interval()
    assert_equal((ci.low, ci.high), (np.nan, np.nan))


def test_relative_risk_ci_edge_cases_01():
    result = relative_risk(exposed_cases=0, exposed_total=12,
                           control_cases=1, control_total=30)
    assert_equal(result.relative_risk, 0)
    ci = result.confidence_interval()
    assert_equal((ci.low, ci.high), (0.0, np.nan))


def test_relative_risk_ci_edge_cases_10():
    result = relative_risk(exposed_cases=1, exposed_total=12,
                           control_cases=0, control_total=30)
    assert_equal(result.relative_risk, np.inf)
    ci = result.confidence_interval()
    assert_equal((ci.low, ci.high), (np.nan, np.inf))


@pytest.mark.parametrize('ec, et, cc, ct', [(0, 0, 10, 20),
                                            (-1, 10, 1, 5),
                                            (1, 10, 0, 0),
                                            (1, 10, -1, 4)])
def test_relative_risk_bad_value(ec, et, cc, ct):
    with pytest.raises(ValueError, match="must be an integer not less than"):
        relative_risk(ec, et, cc, ct)


def test_relative_risk_bad_type():
    with pytest.raises(TypeError, match="must be an integer"):
        relative_risk(1, 10, 2.0, 40)
