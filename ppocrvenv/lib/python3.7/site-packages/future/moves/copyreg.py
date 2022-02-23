from __future__ import absolute_import
from future.utils import PY3

if PY3:
    import copyreg, sys
    # A "*" import uses Python 3's copyreg.__all__ which does not include
    # all public names in the API surface for copyreg, this avoids that
    # problem by just making our module _be_ a reference to the actual module.
    sys.modules['future.moves.copyreg'] = copyreg
else:
    __future_module__ = True
    from copy_reg import *
