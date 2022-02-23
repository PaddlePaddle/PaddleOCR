"""
A module that brings in equivalents of the new and modified Python 3
builtins into Py2. Has no effect on Py3.

See the docs `here <http://python-future.org/what-else.html>`_
(``docs/what-else.rst``) for more information.

"""

from future.builtins.iterators import (filter, map, zip)
# The isinstance import is no longer needed. We provide it only for
# backward-compatibility with future v0.8.2. It will be removed in future v1.0.
from future.builtins.misc import (ascii, chr, hex, input, isinstance, next,
                                  oct, open, pow, round, super, max, min)
from future.utils import PY3

if PY3:
    import builtins
    bytes = builtins.bytes
    dict = builtins.dict
    int = builtins.int
    list = builtins.list
    object = builtins.object
    range = builtins.range
    str = builtins.str
    __all__ = []
else:
    from future.types import (newbytes as bytes,
                              newdict as dict,
                              newint as int,
                              newlist as list,
                              newobject as object,
                              newrange as range,
                              newstr as str)
from future import utils


if not utils.PY3:
    # We only import names that shadow the builtins on Py2. No other namespace
    # pollution on Py2.

    # Only shadow builtins on Py2; no new names
    __all__ = ['filter', 'map', 'zip',
               'ascii', 'chr', 'hex', 'input', 'next', 'oct', 'open', 'pow',
               'round', 'super',
               'bytes', 'dict', 'int', 'list', 'object', 'range', 'str', 'max', 'min'
              ]

else:
    # No namespace pollution on Py3
    __all__ = []
