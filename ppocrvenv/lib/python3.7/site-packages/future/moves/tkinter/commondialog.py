from __future__ import absolute_import

from future.utils import PY3

if PY3:
    from tkinter.commondialog import *
else:
    try:
        from tkCommonDialog import *
    except ImportError:
        raise ImportError('The tkCommonDialog module is missing. Does your Py2 '
                          'installation include tkinter?')
