"""GAE specific URL reading functions"""
__all__ = ['_defaultFetcher']

# raises ImportError of not on GAE
from google.appengine.api import urlfetch
import cgi
from . import errorhandler

log = errorhandler.ErrorHandler()


def _defaultFetcher(url):
    """
    uses GoogleAppEngine (GAE)
        fetch(url, payload=None, method=GET, headers={}, allow_truncated=False)

    Response
        content
            The body content of the response.
        content_was_truncated
            True if the allow_truncated parameter to fetch() was True and
            the response exceeded the maximum response size. In this case,
            the content attribute contains the truncated response.
        status_code
            The HTTP status code.
        headers
            The HTTP response headers, as a mapping of names to values.

    Exceptions
        exception InvalidURLError()
            The URL of the request was not a valid URL, or it used an
            unsupported method. Only http and https URLs are supported.
        exception DownloadError()
            There was an error retrieving the data.

            This exception is not raised if the server returns an HTTP
            error code: In that case, the response data comes back intact,
            including the error code.

        exception ResponseTooLargeError()
            The response data exceeded the maximum allowed size, and the
            allow_truncated parameter passed to fetch() was False.
    """
    # from google.appengine.api import urlfetch
    try:
        r = urlfetch.fetch(url, method=urlfetch.GET)
    except urlfetch.Error as e:
        log.warn('Error opening url=%r: %s' % (url, e), error=IOError)
    else:
        if r.status_code == 200:
            # find mimetype and encoding
            mimetype = 'application/octet-stream'
            try:
                mimetype, params = cgi.parse_header(r.headers['content-type'])
                encoding = params['charset']
            except KeyError:
                encoding = None
            if mimetype != 'text/css':
                log.error(
                    'Expected "text/css" mime type for url %r but found: %r'
                    % (url, mimetype),
                    error=ValueError,
                )
            return encoding, r.content
        else:
            # TODO: 301 etc
            log.warn(
                'Error opening url=%r: HTTP status %s' % (url, r.status_code),
                error=IOError,
            )
