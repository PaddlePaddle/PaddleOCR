"""xmlrpclib.Transport implementation
"""

# The following comment should be removed at some point in the future.
# mypy: disallow-untyped-defs=False

import logging

from pip._vendor import requests
# NOTE: XMLRPC Client is not annotated in typeshed as on 2017-07-17, which is
#       why we ignore the type on this import
from pip._vendor.six.moves import xmlrpc_client  # type: ignore
from pip._vendor.six.moves.urllib import parse as urllib_parse

logger = logging.getLogger(__name__)


class PipXmlrpcTransport(xmlrpc_client.Transport):
    """Provide a `xmlrpclib.Transport` implementation via a `PipSession`
    object.
    """

    def __init__(self, index_url, session, use_datetime=False):
        xmlrpc_client.Transport.__init__(self, use_datetime)
        index_parts = urllib_parse.urlparse(index_url)
        self._scheme = index_parts.scheme
        self._session = session

    def request(self, host, handler, request_body, verbose=False):
        parts = (self._scheme, host, handler, None, None, None)
        url = urllib_parse.urlunparse(parts)
        try:
            headers = {'Content-Type': 'text/xml'}
            response = self._session.post(url, data=request_body,
                                          headers=headers, stream=True)
            response.raise_for_status()
            self.verbose = verbose
            return self.parse_response(response.raw)
        except requests.HTTPError as exc:
            logger.critical(
                "HTTP error %s while getting %s",
                exc.response.status_code, url,
            )
            raise
