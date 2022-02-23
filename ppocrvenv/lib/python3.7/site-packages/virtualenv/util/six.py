"""Backward compatibility layer with older version of six.

This is used to avoid virtualenv requiring a version of six newer than what
the system may have.
"""
from __future__ import absolute_import

from six import PY2, PY3, binary_type, text_type

try:
    from six import ensure_text
except ImportError:

    def ensure_text(s, encoding="utf-8", errors="strict"):
        """Coerce *s* to six.text_type.
        For Python 2:
        - `unicode` -> `unicode`
        - `str` -> `unicode`
        For Python 3:
        - `str` -> `str`
        - `bytes` -> decoded to `str`
        """
        if isinstance(s, binary_type):
            return s.decode(encoding, errors)
        elif isinstance(s, text_type):
            return s
        else:
            raise TypeError("not expecting type '%s'" % type(s))


try:
    from six import ensure_str
except ImportError:

    def ensure_str(s, encoding="utf-8", errors="strict"):
        """Coerce *s* to `str`.
        For Python 2:
        - `unicode` -> encoded to `str`
        - `str` -> `str`
        For Python 3:
        - `str` -> `str`
        - `bytes` -> decoded to `str`
        """
        if not isinstance(s, (text_type, binary_type)):
            raise TypeError("not expecting type '%s'" % type(s))
        if PY2 and isinstance(s, text_type):
            s = s.encode(encoding, errors)
        elif PY3 and isinstance(s, binary_type):
            s = s.decode(encoding, errors)
        return s
