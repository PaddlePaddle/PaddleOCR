from __future__ import absolute_import
from future.utils import PY3
__future_module__ = True

if PY3:
    from html import *
else:
    # cgi.escape isn't good enough for the single Py3.3 html test to pass.
    # Define it inline here instead. From the Py3.4 stdlib. Note that the
    # html.escape() function from the Py3.3 stdlib is not suitable for use on
    # Py2.x.
    """
    General functions for HTML manipulation.
    """

    def escape(s, quote=True):
        """
        Replace special characters "&", "<" and ">" to HTML-safe sequences.
        If the optional flag quote is true (the default), the quotation mark
        characters, both double quote (") and single quote (') characters are also
        translated.
        """
        s = s.replace("&", "&amp;") # Must be done first!
        s = s.replace("<", "&lt;")
        s = s.replace(">", "&gt;")
        if quote:
            s = s.replace('"', "&quot;")
            s = s.replace('\'', "&#x27;")
        return s

    __all__ = ['escape']
