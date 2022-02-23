"""
Forward-ports of types from Python 2 for use with Python 3:

- ``basestring``: equivalent to ``(str, bytes)`` in ``isinstance`` checks
- ``dict``: with list-producing .keys() etc. methods
- ``str``: bytes-like, but iterating over them doesn't product integers
- ``long``: alias of Py3 int with ``L`` suffix in the ``repr``
- ``unicode``: alias of Py3 str with ``u`` prefix in the ``repr``

"""

from past import utils

if utils.PY2:
    import __builtin__
    basestring = __builtin__.basestring
    dict = __builtin__.dict
    str = __builtin__.str
    long = __builtin__.long
    unicode = __builtin__.unicode
    __all__ = []
else:
    from .basestring import basestring
    from .olddict import olddict
    from .oldstr import oldstr
    long = int
    unicode = str
    # from .unicode import unicode
    __all__ = ['basestring', 'olddict', 'oldstr', 'long', 'unicode']
