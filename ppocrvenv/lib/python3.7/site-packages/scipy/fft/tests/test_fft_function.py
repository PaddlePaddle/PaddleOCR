import numpy as np
import subprocess
import sys

TEST_BODY = r"""
import pytest
import numpy as np
from numpy.testing import assert_allclose
import scipy
import sys
import pytest

if hasattr(scipy, 'fft'):
    raise AssertionError("scipy.fft should require an explicit import")

np.random.seed(1234)
x = np.random.randn(10) + 1j * np.random.randn(10)
X = np.fft.fft(x)
# Callable before scipy.fft is imported
with pytest.deprecated_call(match=r'2\.0\.0'):
    y = scipy.ifft(X)
assert_allclose(y, x)

# Callable after scipy.fft is imported
import scipy.fft
with pytest.deprecated_call(match=r'2\.0\.0'):
    y = scipy.ifft(X)
assert_allclose(y, x)

"""

def test_fft_function():
    # Historically, scipy.fft was an alias for numpy.fft.fft
    # Ensure there are no conflicts with the FFT module (gh-10253)

    # Test must run in a subprocess so scipy.fft is not already imported
    subprocess.check_call([sys.executable, '-c', TEST_BODY])

    # scipy.fft is the correct module
    from scipy import fft
    assert not callable(fft)
    assert fft.__name__ == 'scipy.fft'

    from scipy import ifft
    assert ifft.__wrapped__ is np.fft.ifft

