from __future__ import absolute_import
from future.utils import PY3
__future_module__ = True

if PY3:
    from html.parser import *
else:
    from HTMLParser import *
