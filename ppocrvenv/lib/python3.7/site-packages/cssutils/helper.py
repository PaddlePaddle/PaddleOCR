"""cssutils helper TEST
"""

import os
import re
import urllib.request
import urllib.error
import urllib.parse


class Deprecated(object):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used.

    It accepts a single paramter ``msg`` which is shown with the warning.
    It should contain information which function or method to use instead.
    """

    def __init__(self, msg):
        self.msg = msg

    def __call__(self, func):
        def newFunc(*args, **kwargs):
            import warnings

            warnings.warn(
                "Call to deprecated method %r. %s" % (func.__name__, self.msg),
                category=DeprecationWarning,
                stacklevel=2,
            )
            return func(*args, **kwargs)

        newFunc.__name__ = func.__name__
        newFunc.__doc__ = func.__doc__
        newFunc.__dict__.update(func.__dict__)
        return newFunc


# simple escapes, all non unicodes
_simpleescapes = re.compile(r'(\\[^0-9a-fA-F])').sub


def normalize(x):
    r"""
    normalizes x, namely:

    - remove any \ before non unicode sequences (0-9a-zA-Z) so for
      x==r"c\olor\" return "color" (unicode escape sequences should have
      been resolved by the tokenizer already)
    - lowercase
    """
    if x:

        def removeescape(matchobj):
            return matchobj.group(0)[1:]

        x = _simpleescapes(removeescape, x)
        return x.lower()
    else:
        return x


def path2url(path):
    """Return file URL of `path`"""
    return 'file:' + urllib.request.pathname2url(os.path.abspath(path))


def pushtoken(token, tokens):
    """Return new generator starting with token followed by all tokens in
    ``tokens``"""
    # TODO: may use itertools.chain?
    yield token
    for t in tokens:
        yield t


def string(value):
    """
    Serialize value with quotes e.g.::

        ``a \'string`` => ``'a \'string'``
    """
    # \n = 0xa, \r = 0xd, \f = 0xc
    value = (
        value.replace('\n', '\\a ')
        .replace('\r', '\\d ')
        .replace('\f', '\\c ')
        .replace('"', '\\"')
    )

    if value.endswith('\\'):
        value = value[:-1] + '\\\\'

    return '"%s"' % value


def stringvalue(string):
    """
    Retrieve actual value of string without quotes. Escaped
    quotes inside the value are resolved, e.g.::

        ``'a \'string'`` => ``a 'string``
    """
    return string.replace('\\' + string[0], string[0])[1:-1]


_match_forbidden_in_uri = re.compile(r'''.*?[\(\)\s\;,'"]''', re.U).match


def uri(value):
    """
    Serialize value by adding ``url()`` and with quotes if needed e.g.::

        ``"`` => ``url("\"")``
    """
    if _match_forbidden_in_uri(value):
        value = string(value)
    return 'url(%s)' % value


def urivalue(uri):
    """
    Return actual content without surrounding "url(" and ")"
    and removed surrounding quotes too including contained
    escapes of quotes, e.g.::

         ``url("\"")`` => ``"``
    """
    uri = uri[uri.find('(') + 1 : -1].strip()
    if uri and (uri[0] in '\'"') and (uri[0] == uri[-1]):
        return stringvalue(uri)
    else:
        return uri


# def normalnumber(num):
#    """
#    Return normalized number as string.
#    """
#    sign = ''
#    if num.startswith('-'):
#        sign = '-'
#        num = num[1:]
#    elif num.startswith('+'):
#        num = num[1:]
#
#    if float(num) == 0.0:
#        return '0'
#    else:
#        if num.find('.') == -1:
#            return sign + str(int(num))
#        else:
#            a, b = num.split('.')
#            if not a:
#                a = '0'
#            return '%s%s.%s' % (sign, int(a), b)
