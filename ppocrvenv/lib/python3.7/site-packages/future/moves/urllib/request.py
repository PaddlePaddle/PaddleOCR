from __future__ import absolute_import

from future.standard_library import suspend_hooks
from future.utils import PY3

if PY3:
    from urllib.request import *
    # This aren't in __all__:
    from urllib.request import (getproxies,
                                pathname2url,
                                proxy_bypass,
                                quote,
                                request_host,
                                thishost,
                                unquote,
                                url2pathname,
                                urlcleanup,
                                urljoin,
                                urlopen,
                                urlparse,
                                urlretrieve,
                                urlsplit,
                                urlunparse)

    from urllib.parse import (splitattr,
                              splithost,
                              splitpasswd,
                              splitport,
                              splitquery,
                              splittag,
                              splittype,
                              splituser,
                              splitvalue,
                              to_bytes,
                              unwrap)
else:
    __future_module__ = True
    with suspend_hooks():
        from urllib import *
        from urllib2 import *
        from urlparse import *

        # Rename:
        from urllib import toBytes    # missing from __all__ on Py2.6
        to_bytes = toBytes

        # from urllib import (pathname2url,
        #                     url2pathname,
        #                     getproxies,
        #                     urlretrieve,
        #                     urlcleanup,
        #                     URLopener,
        #                     FancyURLopener,
        #                     proxy_bypass)

        # from urllib2 import (
        #                  AbstractBasicAuthHandler,
        #                  AbstractDigestAuthHandler,
        #                  BaseHandler,
        #                  CacheFTPHandler,
        #                  FileHandler,
        #                  FTPHandler,
        #                  HTTPBasicAuthHandler,
        #                  HTTPCookieProcessor,
        #                  HTTPDefaultErrorHandler,
        #                  HTTPDigestAuthHandler,
        #                  HTTPErrorProcessor,
        #                  HTTPHandler,
        #                  HTTPPasswordMgr,
        #                  HTTPPasswordMgrWithDefaultRealm,
        #                  HTTPRedirectHandler,
        #                  HTTPSHandler,
        #                  URLError,
        #                  build_opener,
        #                  install_opener,
        #                  OpenerDirector,
        #                  ProxyBasicAuthHandler,
        #                  ProxyDigestAuthHandler,
        #                  ProxyHandler,
        #                  Request,
        #                  UnknownHandler,
        #                  urlopen,
        #                 )

        # from urlparse import (
        #                  urldefrag
        #                  urljoin,
        #                  urlparse,
        #                  urlunparse,
        #                  urlsplit,
        #                  urlunsplit,
        #                  parse_qs,
        #                  parse_q"
        #                 )
