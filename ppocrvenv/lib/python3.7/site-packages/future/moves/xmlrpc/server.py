from __future__ import absolute_import
from future.utils import PY3

if PY3:
    from xmlrpc.server import *
else:
    from xmlrpclib import *
