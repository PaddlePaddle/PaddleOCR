from __future__ import absolute_import

from future.utils import PY3

if PY3:
    from tkinter.filedialog import *
else:
    try:
        from FileDialog import *
    except ImportError:
        raise ImportError('The FileDialog module is missing. Does your Py2 '
                          'installation include tkinter?')
