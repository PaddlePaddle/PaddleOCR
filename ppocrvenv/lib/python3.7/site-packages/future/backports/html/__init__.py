"""
General functions for HTML manipulation, backported from Py3.

Note that this uses Python 2.7 code with the corresponding Python 3
module names and locations.
"""

from __future__ import unicode_literals


_escape_map = {ord('&'): '&amp;', ord('<'): '&lt;', ord('>'): '&gt;'}
_escape_map_full = {ord('&'): '&amp;', ord('<'): '&lt;', ord('>'): '&gt;',
                    ord('"'): '&quot;', ord('\''): '&#x27;'}

# NB: this is a candidate for a bytes/string polymorphic interface

def escape(s, quote=True):
    """
    Replace special characters "&", "<" and ">" to HTML-safe sequences.
    If the optional flag quote is true (the default), the quotation mark
    characters, both double quote (") and single quote (') characters are also
    translated.
    """
    assert not isinstance(s, bytes), 'Pass a unicode string'
    if quote:
        return s.translate(_escape_map_full)
    return s.translate(_escape_map)
