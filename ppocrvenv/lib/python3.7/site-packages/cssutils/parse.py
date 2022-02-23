"""A validating CSSParser"""
__all__ = ['CSSParser']

from .helper import path2url
import codecs
import cssutils
from . import tokenize2

from cssutils import css


class CSSParser(object):
    """Parse a CSS StyleSheet from URL, string or file and return a DOM Level 2
    CSS StyleSheet object.

    Usage::

        parser = CSSParser()
        # optionally
        parser.setFetcher(fetcher)
        sheet = parser.parseFile('test1.css', 'ascii')
        print sheet.cssText
    """

    def __init__(
        self,
        log=None,
        loglevel=None,
        raiseExceptions=None,
        fetcher=None,
        parseComments=True,
        validate=True,
    ):
        """
        :param log:
            logging object
        :param loglevel:
            logging loglevel
        :param raiseExceptions:
            if log should simply log (default) or raise errors during
            parsing. Later while working with the resulting sheets
            the setting used in cssutils.log.raiseExeptions is used
        :param fetcher:
            see ``setFetcher(fetcher)``
        :param parseComments:
            if comments should be added to CSS DOM or simply omitted
        :param validate:
            if parsing should validate, may be overwritten in parse methods
        """
        if log is not None:
            cssutils.log.setLog(log)
        if loglevel is not None:
            cssutils.log.setLevel(loglevel)

        # remember global setting
        self.__globalRaising = cssutils.log.raiseExceptions
        if raiseExceptions:
            self.__parseRaising = raiseExceptions
        else:
            # DEFAULT during parse
            self.__parseRaising = False

        self.__tokenizer = tokenize2.Tokenizer(doComments=parseComments)
        self.setFetcher(fetcher)

        self._validate = validate

    def __parseSetting(self, parse):
        """during parse exceptions may be handled differently depending on
        init parameter ``raiseExceptions``
        """
        if parse:
            cssutils.log.raiseExceptions = self.__parseRaising
        else:
            cssutils.log.raiseExceptions = self.__globalRaising

    def parseStyle(self, cssText, encoding='utf-8', validate=None):
        """Parse given `cssText` which is assumed to be the content of
        a HTML style attribute.

        :param cssText:
            CSS string to parse
        :param encoding:
            It will be used to decode `cssText` if given as a (byte)
            string.
        :param validate:
            If given defines if validation is used. Uses CSSParser settings as
            fallback
        :returns:
            :class:`~cssutils.css.CSSStyleDeclaration`
        """
        self.__parseSetting(True)
        if isinstance(cssText, bytes):
            # TODO: use codecs.getdecoder('css') here?
            cssText = cssText.decode(encoding)
        if validate is None:
            validate = self._validate
        style = css.CSSStyleDeclaration(cssText, validating=validate)
        self.__parseSetting(False)
        return style

    def parseString(
        self, cssText, encoding=None, href=None, media=None, title=None, validate=None
    ):
        """Parse `cssText` as :class:`~cssutils.css.CSSStyleSheet`.
        Errors may be raised (e.g. UnicodeDecodeError).

        :param cssText:
            CSS string to parse
        :param encoding:
            If ``None`` the encoding will be read from BOM or an @charset
            rule or defaults to UTF-8.
            If given overrides any found encoding including the ones for
            imported sheets.
            It also will be used to decode `cssText` if given as a (byte)
            string.
        :param href:
            The ``href`` attribute to assign to the parsed style sheet.
            Used to resolve other urls in the parsed sheet like @import hrefs.
        :param media:
            The ``media`` attribute to assign to the parsed style sheet
            (may be a MediaList, list or a string).
        :param title:
            The ``title`` attribute to assign to the parsed style sheet.
        :param validate:
            If given defines if validation is used. Uses CSSParser settings as
            fallback
        :returns:
            :class:`~cssutils.css.CSSStyleSheet`.
        """
        self.__parseSetting(True)
        # TODO: py3 needs bytes here!
        if isinstance(cssText, bytes):
            cssText = codecs.getdecoder('css')(cssText, encoding=encoding)[0]

        if validate is None:
            validate = self._validate

        sheet = cssutils.css.CSSStyleSheet(
            href=href,
            media=cssutils.stylesheets.MediaList(media),
            title=title,
            validating=validate,
        )
        sheet._setFetcher(self.__fetcher)
        # tokenizing this ways closes open constructs and adds EOF
        sheet._setCssTextWithEncodingOverride(
            self.__tokenizer.tokenize(cssText, fullsheet=True),
            encodingOverride=encoding,
        )
        self.__parseSetting(False)
        return sheet

    def parseFile(
        self, filename, encoding=None, href=None, media=None, title=None, validate=None
    ):
        """Retrieve content from `filename` and parse it. Errors may be raised
        (e.g. IOError).

        :param filename:
            of the CSS file to parse, if no `href` is given filename is
            converted to a (file:) URL and set as ``href`` of resulting
            stylesheet.
            If `href` is given it is set as ``sheet.href``. Either way
            ``sheet.href`` is used to resolve e.g. stylesheet imports via
            @import rules.
        :param encoding:
            Value ``None`` defaults to encoding detection via BOM or an
            @charset rule.
            Other values override detected encoding for the sheet at
            `filename` including any imported sheets.
        :returns:
            :class:`~cssutils.css.CSSStyleSheet`.
        """
        if not href:
            # prepend // for file URL, urllib does not do this?
            # href = u'file:' + urllib.pathname2url(os.path.abspath(filename))
            href = path2url(filename)

        f = open(filename, 'rb')
        css = f.read()
        f.close()

        return self.parseString(
            css,
            encoding=encoding,  # read returns a str
            href=href,
            media=media,
            title=title,
            validate=validate,
        )

    def parseUrl(self, href, encoding=None, media=None, title=None, validate=None):
        """Retrieve content from URL `href` and parse it. Errors may be raised
        (e.g. URLError).

        :param href:
            URL of the CSS file to parse, will also be set as ``href`` of
            resulting stylesheet
        :param encoding:
            Value ``None`` defaults to encoding detection via HTTP, BOM or an
            @charset rule.
            A value overrides detected encoding for the sheet at ``href``
            including any imported sheets.
        :returns:
            :class:`~cssutils.css.CSSStyleSheet`.
        """
        encoding, enctype, text = cssutils.util._readUrl(
            href, fetcher=self.__fetcher, overrideEncoding=encoding
        )
        if enctype == 5:
            # do not use if defaulting to UTF-8
            encoding = None

        if text is not None:
            return self.parseString(
                text,
                encoding=encoding,
                href=href,
                media=media,
                title=title,
                validate=validate,
            )

    def setFetcher(self, fetcher=None):
        """Replace the default URL fetch function with a custom one.

        :param fetcher:
            A function which gets a single parameter

            ``url``
                the URL to read

            and must return ``(encoding, content)`` where ``encoding`` is the
            HTTP charset normally given via the Content-Type header (which may
            simply omit the charset in which case ``encoding`` would be
            ``None``) and ``content`` being the string (or unicode) content.

            The Mimetype should be 'text/css' but this has to be checked by the
            fetcher itself (the default fetcher emits a warning if encountering
            a different mimetype).

            Calling ``setFetcher`` with ``fetcher=None`` resets cssutils
            to use its default function.
        """
        self.__fetcher = fetcher
