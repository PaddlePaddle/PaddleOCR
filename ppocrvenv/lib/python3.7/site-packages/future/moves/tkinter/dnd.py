from __future__ import absolute_import

from future.utils import PY3

if PY3:
    from tkinter.dnd import *
else:
    try:
        from Tkdnd import *
    except ImportError:
        raise ImportError('The Tkdnd module is missing. Does your Py2 '
                          'installation include tkinter?')
