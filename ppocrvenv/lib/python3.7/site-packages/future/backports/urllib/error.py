"""Exception classes raised by urllib.

The base exception class is URLError, which inherits from IOError.  It
doesn't define any behavior of its own, but is the base class for all
exceptions defined in this package.

HTTPError is an exception class that is also a valid HTTP response
instance.  It behaves this way because HTTP protocol errors are valid
responses, with a status code, headers, and a body.  In some contexts,
an application may want to handle an exception like a regular
response.
"""
from __future__ import absolute_import, division, unicode_literals
from future import standard_library

from future.backports.urllib import response as urllib_response


__all__ = ['URLError', 'HTTPError', 'ContentTooShortError']


# do these error classes make sense?
# make sure all of the IOError stuff is overridden.  we just want to be
# subtypes.

class URLError(IOError):
    # URLError is a sub-type of IOError, but it doesn't share any of
    # the implementation.  need to override __init__ and __str__.
    # It sets self.args for compatibility with other EnvironmentError
    # subclasses, but args doesn't have the typical format with errno in
    # slot 0 and strerror in slot 1.  This may be better than nothing.
    def __init__(self, reason, filename=None):
        self.args = reason,
        self.reason = reason
        if filename is not None:
            self.filename = filename

    def __str__(self):
        return '<urlopen error %s>' % self.reason

class HTTPError(URLError, urllib_response.addinfourl):
    """Raised when HTTP error occurs, but also acts like non-error return"""
    __super_init = urllib_response.addinfourl.__init__

    def __init__(self, url, code, msg, hdrs, fp):
        self.code = code
        self.msg = msg
        self.hdrs = hdrs
        self.fp = fp
        self.filename = url
        # The addinfourl classes depend on fp being a valid file
        # object.  In some cases, the HTTPError may not have a valid
        # file object.  If this happens, the simplest workaround is to
        # not initialize the base classes.
        if fp is not None:
            self.__super_init(fp, hdrs, url, code)

    def __str__(self):
        return 'HTTP Error %s: %s' % (self.code, self.msg)

    # since URLError specifies a .reason attribute, HTTPError should also
    #  provide this attribute. See issue13211 for discussion.
    @property
    def reason(self):
        return self.msg

    def info(self):
        return self.hdrs


# exception raised when downloaded size does not match content-length
class ContentTooShortError(URLError):
    def __init__(self, message, content):
        URLError.__init__(self, message)
        self.content = content
