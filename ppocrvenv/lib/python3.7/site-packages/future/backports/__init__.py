"""
future.backports package
"""

from __future__ import absolute_import

import sys

__future_module__ = True
from future.standard_library import import_top_level_modules


if sys.version_info[0] >= 3:
    import_top_level_modules()


from .misc import (ceil,
                   OrderedDict,
                   Counter,
                   ChainMap,
                   check_output,
                   count,
                   recursive_repr,
                   _count_elements,
                   cmp_to_key
                  )
