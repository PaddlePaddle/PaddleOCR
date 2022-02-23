"""cssutils ErrorHandler

ErrorHandler
    used as log with usual levels (debug, info, warn, error)

    if instanciated with ``raiseExceptions=True`` raises exeptions instead
    of logging

log
    defaults to instance of ErrorHandler for any kind of log message from
    lexerm, parser etc.

    - raiseExceptions = [False, True]
    - setloglevel(loglevel)
"""
__all__ = ['ErrorHandler']

import logging
import urllib.request
import urllib.error
import urllib.parse
import xml.dom


class _ErrorHandler(object):
    """
    handles all errors and log messages
    """

    def __init__(self, log, defaultloglevel=logging.INFO, raiseExceptions=True):
        """
        inits log if none given

        log
            for parse messages, default logs to sys.stderr
        defaultloglevel
            if none give this is logging.DEBUG
        raiseExceptions
            - True: Errors will be raised e.g. during building
            - False: Errors will be written to the log, this is the
              default behaviour when parsing
        """
        # may be disabled during setting of known valid items
        self.enabled = True

        if log:
            self._log = log
        else:
            import sys

            self._log = logging.getLogger('CSSUTILS')
            hdlr = logging.StreamHandler(sys.stderr)
            formatter = logging.Formatter('%(levelname)s\t%(message)s')
            hdlr.setFormatter(formatter)
            self._log.addHandler(hdlr)
            self._log.setLevel(defaultloglevel)

        self.raiseExceptions = raiseExceptions

    def __getattr__(self, name):
        "use self._log items"
        calls = ('debug', 'info', 'warn', 'error', 'critical', 'fatal')
        other = ('setLevel', 'getEffectiveLevel', 'addHandler', 'removeHandler')

        if name in calls:
            if name == 'warn':
                name = 'warning'
            self._logcall = getattr(self._log, name)
            return self.__handle
        elif name in other:
            return getattr(self._log, name)
        else:
            raise AttributeError('(errorhandler) No Attribute %r found' % name)

    def __handle(
        self, msg='', token=None, error=xml.dom.SyntaxErr, neverraise=False, args=None
    ):
        """
        handles all calls
        logs or raises exception
        """
        if self.enabled:
            if error is None:
                error = xml.dom.SyntaxErr

            line, col = None, None
            if token:
                if isinstance(token, tuple):
                    value, line, col = token[1], token[2], token[3]
                else:
                    value, line, col = token.value, token.line, token.col
                msg = '%s [%s:%s: %s]' % (msg, line, col, value)

            if error and self.raiseExceptions and not neverraise:
                if isinstance(error, urllib.error.HTTPError) or isinstance(
                    error, urllib.error.URLError
                ):
                    raise
                elif issubclass(error, xml.dom.DOMException):
                    error.line = line
                    error.col = col
                raise error(msg)
            else:
                self._logcall(msg)

    def setLog(self, log):
        """set log of errorhandler's log"""
        self._log = log


class ErrorHandler(_ErrorHandler):
    "Singleton, see _ErrorHandler"
    instance = None

    def __init__(self, log=None, defaultloglevel=logging.INFO, raiseExceptions=True):

        if ErrorHandler.instance is None:
            ErrorHandler.instance = _ErrorHandler(
                log=log,
                defaultloglevel=defaultloglevel,
                raiseExceptions=raiseExceptions,
            )
        self.__dict__ = ErrorHandler.instance.__dict__
