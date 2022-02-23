"""A validating CSSParser"""

import codecs
import sys

from . import helper
from . import errorhandler
from . import tokenize2


class ErrorHandler(object):
    """Basic class for CSS error handlers.

    This class class provides a default implementation ignoring warnings and
    recoverable errors and throwing a SAXParseException for fatal errors.

    If a CSS application needs to implement customized error handling, it must
    extend this class and then register an instance with the CSS parser
    using the parser's setErrorHandler method. The parser will then report all
    errors and warnings through this interface.

    The parser shall use this class instead of throwing an exception: it is
    up to the application whether to throw an exception for different types of
    errors and warnings. Note, however, that there is no requirement that the
    parser continue to provide useful information after a call to fatalError
    (in other words, a CSS driver class could catch an exception and report a
    fatalError).
    """

    def __init__(self):
        self._log = errorhandler.ErrorHandler()

    def error(self, exception, token=None):
        self._log.error(exception, token, neverraise=True)

    def fatal(self, exception, token=None):
        self._log.fatal(exception, token)

    def warn(self, exception, token=None):
        self._log.warn(exception, token, neverraise=True)


class DocumentHandler(object):
    """
    void     endFontFace()
             Receive notification of the end of a font face statement.
    void     endMedia(SACMediaList media)
             Receive notification of the end of a media statement.
    void     endPage(java.lang.String name, java.lang.String pseudo_page)
             Receive notification of the end of a media statement.
    void     importStyle(java.lang.String uri, SACMediaList media,
             java.lang.String defaultNamespaceURI)
             Receive notification of a import statement in the style sheet.
    void     startFontFace()
             Receive notification of the beginning of a font face statement.
    void     startMedia(SACMediaList media)
             Receive notification of the beginning of a media statement.
    void     startPage(java.lang.String name, java.lang.String pseudo_page)
             Receive notification of the beginning of a page statement.
    """

    def __init__(self):
        def log(msg):
            sys.stderr.write('INFO\t%s\n' % msg)

        self._log = log

    def comment(self, text, line=None, col=None):
        "Receive notification of a comment."
        self._log("comment %r at [%s, %s]" % (text, line, col))

    def startDocument(self, encoding):
        "Receive notification of the beginning of a style sheet."
        # source
        self._log("startDocument encoding=%s" % encoding)

    def endDocument(self, source=None, line=None, col=None):
        "Receive notification of the end of a document."
        self._log("endDocument EOF")

    def importStyle(self, uri, media, name, line=None, col=None):
        "Receive notification of a import statement in the style sheet."
        # defaultNamespaceURI???
        self._log("importStyle at [%s, %s]" % (line, col))

    def namespaceDeclaration(self, prefix, uri, line=None, col=None):
        "Receive notification of an unknown rule t-rule not supported by this parser."
        # prefix might be None!
        self._log("namespaceDeclaration at [%s, %s]" % (line, col))

    def startSelector(self, selectors=None, line=None, col=None):
        "Receive notification of the beginning of a rule statement."
        # TODO selectorList!
        self._log("startSelector at [%s, %s]" % (line, col))

    def endSelector(self, selectors=None, line=None, col=None):
        "Receive notification of the end of a rule statement."
        self._log("endSelector at [%s, %s]" % (line, col))

    def property(self, name, value='TODO', important=False, line=None, col=None):
        "Receive notification of a declaration."
        # TODO: value is LexicalValue?
        self._log("property %r at [%s, %s]" % (name, line, col))

    def ignorableAtRule(self, atRule, line=None, col=None):
        "Receive notification of an unknown rule t-rule not supported by this parser."
        self._log("ignorableAtRule %r at [%s, %s]" % (atRule, line, col))


class EchoHandler(DocumentHandler):
    "Echos all input to property `out`"

    def __init__(self):
        super(EchoHandler, self).__init__()
        self._out = []

    out = property(lambda self: ''.join(self._out))

    def startDocument(self, encoding):
        super(EchoHandler, self).startDocument(encoding)
        if 'utf-8' != encoding:
            self._out.append('@charset "%s";\n' % encoding)

    #    def comment(self, text, line=None, col=None):
    #        self._out.append(u'/*%s*/' % text)

    def importStyle(self, uri, media, name, line=None, col=None):
        "Receive notification of a import statement in the style sheet."
        # defaultNamespaceURI???
        super(EchoHandler, self).importStyle(uri, media, name, line, col)
        self._out.append(
            '@import %s%s%s;\n'
            % (
                helper.string(uri),
                '%s ' % media if media else '',
                '%s ' % name if name else '',
            )
        )

    def namespaceDeclaration(self, prefix, uri, line=None, col=None):
        super(EchoHandler, self).namespaceDeclaration(prefix, uri, line, col)
        self._out.append(
            '@namespace %s%s;\n'
            % ('%s ' % prefix if prefix else '', helper.string(uri))
        )

    def startSelector(self, selectors=None, line=None, col=None):
        super(EchoHandler, self).startSelector(selectors, line, col)
        if selectors:
            self._out.append(', '.join(selectors))
        self._out.append(' {\n')

    def endSelector(self, selectors=None, line=None, col=None):
        self._out.append('    }')

    def property(self, name, value, important=False, line=None, col=None):
        super(EchoHandler, self).property(name, value, line, col)
        self._out.append(
            '    %s: %s%s;\n' % (name, value, ' !important' if important else '')
        )


class Parser(object):
    """
    java.lang.String     getParserVersion()
        Returns a string about which CSS language is supported by this parser.
     boolean     parsePriority(InputSource source)
          Parse a CSS priority value (e.g.
     LexicalUnit     parsePropertyValue(InputSource source)
          Parse a CSS property value.
     void     parseRule(InputSource source)
          Parse a CSS rule.
     SelectorList     parseSelectors(InputSource source)
          Parse a comma separated list of selectors.
     void     parseStyleDeclaration(InputSource source)
          Parse a CSS style declaration (without '{' and '}').
     void     parseStyleSheet(InputSource source)
          Parse a CSS document.
     void     parseStyleSheet(java.lang.String uri)
          Parse a CSS document from a URI.
     void     setConditionFactory(ConditionFactory conditionFactory)

     void     setDocumentHandler(DocumentHandler handler)
          Allow an application to register a document event handler.
     void     setErrorHandler(ErrorHandler handler)
          Allow an application to register an error event handler.
     void     setLocale(java.util.Locale locale)
          Allow an application to request a locale for errors and warnings.
     void     setSelectorFactory(SelectorFactory selectorFactory)
    """

    def __init__(self, documentHandler=None, errorHandler=None):
        self._tokenizer = tokenize2.Tokenizer()
        if documentHandler:
            self.setDocumentHandler(documentHandler)
        else:
            self.setDocumentHandler(DocumentHandler())

        if errorHandler:
            self.setErrorHandler(errorHandler)
        else:
            self.setErrorHandler(ErrorHandler())

    def parseString(self, cssText, encoding=None):  # noqa: C901
        if isinstance(cssText, str):
            cssText = codecs.getdecoder('css')(cssText, encoding=encoding)[0]

        tokens = self._tokenizer.tokenize(cssText, fullsheet=True)

        def COMMENT(val, line, col):
            self._handler.comment(val[2:-2], line, col)

        def EOF(val, line, col):
            self._handler.endDocument(val, line, col)

        def simple(t):
            map = {'COMMENT': COMMENT, 'S': lambda val, line, col: None, 'EOF': EOF}
            type_, val, line, col = t
            if type_ in map:
                map[type_](val, line, col)
                return True
            else:
                return False

        # START PARSING
        t = next(tokens)
        type_, val, line, col = t

        encoding = 'utf-8'
        if 'CHARSET_SYM' == type_:
            # @charset "encoding";
            # S
            encodingtoken = next(tokens)
            semicolontoken = next(tokens)
            if 'STRING' == type_:
                encoding = helper.stringvalue(val)
            # ;
            if 'STRING' == encodingtoken[0] and semicolontoken:
                encoding = helper.stringvalue(encodingtoken[1])
            else:
                self._errorHandler.fatal('Invalid @charset')

            t = next(tokens)
            type_, val, line, col = t

        self._handler.startDocument(encoding)

        while True:
            start = (line, col)
            try:
                if simple(t):
                    pass

                elif 'ATKEYWORD' == type_ or type_ in (
                    'PAGE_SYM',
                    'MEDIA_SYM',
                    'FONT_FACE_SYM',
                ):
                    atRule = [val]
                    braces = 0
                    while True:
                        # read till end ;
                        # TODO: or {}
                        t = next(tokens)
                        type_, val, line, col = t
                        atRule.append(val)
                        if ';' == val and not braces:
                            break
                        elif '{' == val:
                            braces += 1
                        elif '}' == val:
                            braces -= 1
                            if braces == 0:
                                break

                    self._handler.ignorableAtRule(''.join(atRule), *start)

                elif 'IMPORT_SYM' == type_:
                    # import URI or STRING media? name?
                    uri, media, name = None, None, None
                    while True:
                        t = next(tokens)
                        type_, val, line, col = t
                        if 'STRING' == type_:
                            uri = helper.stringvalue(val)
                        elif 'URI' == type_:
                            uri = helper.urivalue(val)
                        elif ';' == val:
                            break

                    if uri:
                        self._handler.importStyle(uri, media, name)
                    else:
                        self._errorHandler.error(
                            'Invalid @import' ' declaration at %r' % (start,)
                        )

                elif 'NAMESPACE_SYM' == type_:
                    prefix, uri = None, None
                    while True:
                        t = next(tokens)
                        type_, val, line, col = t
                        if 'IDENT' == type_:
                            prefix = val
                        elif 'STRING' == type_:
                            uri = helper.stringvalue(val)
                        elif 'URI' == type_:
                            uri = helper.urivalue(val)
                        elif ';' == val:
                            break
                    if uri:
                        self._handler.namespaceDeclaration(prefix, uri, *start)
                    else:
                        self._errorHandler.error(
                            'Invalid @namespace' ' declaration at %r' % (start,)
                        )

                else:
                    # CSSSTYLERULE
                    selector = []
                    selectors = []
                    while True:
                        # selectors[, selector]* {
                        if 'S' == type_:
                            selector.append(' ')
                        elif simple(t):
                            pass
                        elif ',' == val:
                            selectors.append(''.join(selector).strip())
                            selector = []
                        elif '{' == val:
                            selectors.append(''.join(selector).strip())
                            self._handler.startSelector(selectors, *start)
                            break
                        else:
                            selector.append(val)

                        t = next(tokens)
                        type_, val, line, col = t

                    end = None
                    while True:
                        # name: value [!important][;name: value [!important]]*;?
                        name, value, important = None, [], False

                        while True:
                            # name:
                            t = next(tokens)
                            type_, val, line, col = t
                            if 'S' == type_:
                                pass
                            elif simple(t):
                                pass
                            elif 'IDENT' == type_:
                                if name:
                                    self._errorHandler.error(
                                        'more than one property name', t
                                    )
                                else:
                                    name = val
                            elif ':' == val:
                                if not name:
                                    self._errorHandler.error('no property name', t)
                                break
                            elif ';' == val:
                                self._errorHandler.error('premature end of property', t)
                                end = val
                                break
                            elif '}' == val:
                                if name:
                                    self._errorHandler.error(
                                        'premature end of property', t
                                    )
                                end = val
                                break
                            else:
                                self._errorHandler.error(
                                    'unexpected property name token %r' % val, t
                                )

                        while not ';' == end and not '}' == end:
                            # value !;}
                            t = next(tokens)
                            type_, val, line, col = t

                            if 'S' == type_:
                                value.append(' ')
                            elif simple(t):
                                pass
                            elif '!' == val or ';' == val or '}' == val:
                                value = ''.join(value).strip()
                                if not value:
                                    self._errorHandler.error(
                                        'premature end of property (no value)', t
                                    )
                                end = val
                                break
                            else:
                                value.append(val)

                        while '!' == end:
                            # !important
                            t = next(tokens)
                            type_, val, line, col = t

                            if simple(t):
                                pass
                            elif 'IDENT' == type_ and not important:
                                important = True
                            elif ';' == val or '}' == val:
                                end = val
                                break
                            else:
                                self._errorHandler.error(
                                    'unexpected priority token %r' % val
                                )

                        if name and value:
                            self._handler.property(name, value, important)

                        if '}' == end:
                            self._handler.endSelector(selectors, line=line, col=col)
                            break
                        else:
                            # reset
                            end = None

                    else:
                        self._handler.endSelector(selectors, line=line, col=col)

                t = next(tokens)
                type_, val, line, col = t

            except StopIteration:
                break

    def setDocumentHandler(self, handler):
        "Allow an application to register a document event `handler`."
        self._handler = handler

    def setErrorHandler(self, handler):
        "TODO"
        self._errorHandler = handler
