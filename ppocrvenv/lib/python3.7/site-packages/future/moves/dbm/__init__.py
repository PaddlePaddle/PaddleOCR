from __future__ import absolute_import
from future.utils import PY3

if PY3:
    from dbm import *
else:
    __future_module__ = True
    from whichdb import *
    from anydbm import *

# Py3.3's dbm/__init__.py imports ndbm but doesn't expose it via __all__.
# In case some (badly written) code depends on dbm.ndbm after import dbm,
# we simulate this:
if PY3:
    from dbm import ndbm
else:
    try:
        from future.moves.dbm import ndbm
    except ImportError:
        ndbm = None
