from __future__ import absolute_import

from future.utils import PY3

if PY3:
    from tkinter.colorchooser import *
else:
    try:
        from tkColorChooser import *
    except ImportError:
        raise ImportError('The tkColorChooser module is missing. Does your Py2 '
                          'installation include tkinter?')
