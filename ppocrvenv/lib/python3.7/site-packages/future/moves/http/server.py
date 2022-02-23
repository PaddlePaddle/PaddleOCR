from __future__ import absolute_import
from future.utils import PY3

if PY3:
    from http.server import *
else:
    __future_module__ = True
    from BaseHTTPServer import *
    from CGIHTTPServer import *
    from SimpleHTTPServer import *
    try:
        from CGIHTTPServer import _url_collapse_path     # needed for a test
    except ImportError:
        try:
            # Python 2.7.0 to 2.7.3
            from CGIHTTPServer import (
                _url_collapse_path_split as _url_collapse_path)
        except ImportError:
            # Doesn't exist on Python 2.6.x. Ignore it.
            pass
