from __future__ import absolute_import

from future.utils import PY3

if PY3:
    from tkinter.scrolledtext import *
else:
    try:
        from ScrolledText import *
    except ImportError:
        raise ImportError('The ScrolledText module is missing. Does your Py2 '
                          'installation include tkinter?')
