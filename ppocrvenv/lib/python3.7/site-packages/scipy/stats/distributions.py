#
# Author:  Travis Oliphant  2002-2011 with contributions from
#          SciPy Developers 2004-2011
#
# NOTE: To look at history using `git blame`, use `git blame -M -C -C`
#       instead of `git blame -Lxxx,+x`.
#
from ._distn_infrastructure import (rv_discrete, rv_continuous, rv_frozen)

from . import _continuous_distns
from . import _discrete_distns

from ._continuous_distns import *
from ._discrete_distns import *
from ._entropy import entropy

# For backwards compatibility e.g. pymc expects distributions.__all__.
__all__ = ['rv_discrete', 'rv_continuous', 'rv_histogram', 'entropy']

# Add only the distribution names, not the *_gen names.
__all__ += _continuous_distns._distn_names
__all__ += _discrete_distns._distn_names
