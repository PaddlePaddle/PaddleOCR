from __future__ import absolute_import
from future.standard_library import suspend_hooks

from future.utils import PY3

if PY3:
    from urllib.error import *
else:
    __future_module__ = True

    # We use this method to get at the original Py2 urllib before any renaming magic
    # ContentTooShortError = sys.py2_modules['urllib'].ContentTooShortError

    with suspend_hooks():
        from urllib import ContentTooShortError
        from urllib2 import URLError, HTTPError
