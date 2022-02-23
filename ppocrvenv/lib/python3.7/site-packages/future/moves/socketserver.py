from __future__ import absolute_import
from future.utils import PY3

if PY3:
    from socketserver import *
else:
    __future_module__ = True
    from SocketServer import *
