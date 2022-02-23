from __future__ import absolute_import
from future.utils import PY3

if PY3:
    from _dummy_thread import *
else:
    __future_module__ = True
    from dummy_thread import *
