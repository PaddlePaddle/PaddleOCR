from __future__ import absolute_import

from future.utils import PY3

if PY3:
    from tkinter.messagebox import *
else:
    try:
        from tkMessageBox import *
    except ImportError:
        raise ImportError('The tkMessageBox module is missing. Does your Py2 '
                          'installation include tkinter?')
