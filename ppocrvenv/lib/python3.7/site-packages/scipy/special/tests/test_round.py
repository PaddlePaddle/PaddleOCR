import numpy as np
import pytest

from scipy.special import _test_round


@pytest.mark.skipif(not _test_round.have_fenv(), reason="no fenv()")
def test_add_round_up():
    np.random.seed(1234)
    _test_round.test_add_round(10**5, 'up')


@pytest.mark.skipif(not _test_round.have_fenv(), reason="no fenv()")
def test_add_round_down():
    np.random.seed(1234)
    _test_round.test_add_round(10**5, 'down')
