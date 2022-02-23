from __future__ import absolute_import

from future.utils import PY3

if PY3:
    from tkinter.simpledialog import *
else:
    try:
        from SimpleDialog import *
    except ImportError:
        raise ImportError('The SimpleDialog module is missing. Does your Py2 '
                          'installation include tkinter?')
