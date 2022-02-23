import sys
import pytest
import numpy as np


@pytest.mark.skipif(sys.version_info[:2] < (3, 7),
                    reason="requires python 3.7 or higher")
def test_financial_expired():
    match = 'NEP 32'
    with pytest.warns(DeprecationWarning, match=match):
        func = np.fv
    with pytest.raises(RuntimeError, match=match):
        func(1, 2, 3)
