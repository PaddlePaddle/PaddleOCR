from __future__ import absolute_import
import sys

from future.utils import PY2, PY26
__future_module__ = True

from collections import *

if PY2:
    from UserDict import UserDict
    from UserList import UserList
    from UserString import UserString

if PY26:
    from future.backports.misc import OrderedDict, Counter

if sys.version_info < (3, 3):
    from future.backports.misc import ChainMap, _count_elements
