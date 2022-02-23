from __future__ import absolute_import

from future.utils import PY3

if PY3:
    from tkinter.constants import *
else:
    try:
        from Tkconstants import *
    except ImportError:
        raise ImportError('The Tkconstants module is missing. Does your Py2 '
                          'installation include tkinter?')
