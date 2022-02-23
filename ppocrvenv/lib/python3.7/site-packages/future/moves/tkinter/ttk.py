from __future__ import absolute_import

from future.utils import PY3

if PY3:
    from tkinter.ttk import *
else:
    try:
        from ttk import *
    except ImportError:
        raise ImportError('The ttk module is missing. Does your Py2 '
                          'installation include tkinter?')
