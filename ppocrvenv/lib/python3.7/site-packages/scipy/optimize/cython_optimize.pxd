# Public Cython API declarations
#
# See doc/source/dev/contributor/public_cython_api.rst for guidelines


# The following cimport statement provides legacy ABI
# support. Changing it causes an ABI forward-compatibility break
# (gh-11793), so we currently leave it as is (no further cimport
# statements should be used in this file).
from .cython_optimize._zeros cimport (
    brentq, brenth, ridder, bisect, zeros_full_output)
