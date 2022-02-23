import re

import scipy
from numpy.testing import assert_


def test_valid_scipy_version():
    # Verify that the SciPy version is a valid one (no .post suffix or other
    # nonsense). See NumPy issue gh-6431 for an issue caused by an invalid
    # version.
    version_pattern = r"^[0-9]+\.[0-9]+\.[0-9]+(|a[0-9]|b[0-9]|rc[0-9])"
    dev_suffix = r"(\.dev0\+.+([0-9a-f]{7}|Unknown))"
    if scipy.version.release:
        res = re.match(version_pattern, scipy.__version__)
    else:
        res = re.match(version_pattern + dev_suffix, scipy.__version__)

    assert_(res is not None, scipy.__version__)
