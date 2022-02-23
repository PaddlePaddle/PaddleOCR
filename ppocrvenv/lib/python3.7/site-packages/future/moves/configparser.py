from __future__ import absolute_import

from future.utils import PY2

if PY2:
    from ConfigParser import *
else:
    from configparser import *
