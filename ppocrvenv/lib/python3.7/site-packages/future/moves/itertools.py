from __future__ import absolute_import

from itertools import *
try:
    zip_longest = izip_longest
    filterfalse = ifilterfalse
except NameError:
    pass
