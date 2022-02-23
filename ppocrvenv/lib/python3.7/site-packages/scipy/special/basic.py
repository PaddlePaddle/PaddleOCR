# Deprecated in SciPy 1.4
from ._basic import *
from warnings import warn as _warn

_warn(
    'scipy.special.basic is deprecated, '
    'import directly from scipy.special instead',
    category=DeprecationWarning,
    stacklevel=2
)
