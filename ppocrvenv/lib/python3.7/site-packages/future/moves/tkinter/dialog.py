from __future__ import absolute_import

from future.utils import PY3

if PY3:
    from tkinter.dialog import *
else:
    try:
        from Dialog import *
    except ImportError:
        raise ImportError('The Dialog module is missing. Does your Py2 '
                          'installation include tkinter?')
