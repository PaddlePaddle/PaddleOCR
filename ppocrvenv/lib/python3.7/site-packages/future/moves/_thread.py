from __future__ import absolute_import
from future.utils import PY3

if PY3:
    from _thread import *
else:
    __future_module__ = True
    from thread import *
