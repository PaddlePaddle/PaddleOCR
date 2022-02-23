"""cssutils serializer"""
__all__ = ['CSSSerializer', 'Preferences']

from cssutils.helper import normalize
import codecs
import cssutils
from . import helper


def _escapecss(e):
    r"""
    Escapes characters not allowed in the current encoding the CSS way
    with a backslash followed by a uppercase hex code point

    E.g. the german umlaut 'ä' is escaped as \E4
    """
    s = e.object[e.start : e.end]
    return (
        ''.join(
            [r'\%s ' % str(hex(ord(x)))[2:].upper() for x in s]  # remove 0x from hex
        ),
        e.end,
    )


codecs.register_error('escapecss', _escapecss)


class Preferences(object):
    r"""Control output of CSSSerializer.

    defaultAtKeyword = True
        Should the literal @keyword from src CSS be used or the default
        form, e.g. if ``True``: ``@import`` else: ``@i\mport``
    defaultPropertyName = True
        Should the normalized propertyname be used or the one given in
        the src file, e.g. if ``True``: ``color`` else: ``c\olor``

        Only used if ``keepAllProperties==False``.

    defaultPropertyPriority = True
        Should the normalized or literal priority be used, e.g. ``!important``
        or ``!Im\portant``

    importHrefFormat = None
        Uses hreftype if ``None`` or format ``"URI"`` if ``'string'`` or
        format ``url(URI)`` if ``'uri'``
    indent = 4 * ' '
        Indentation of e.g Properties inside a CSSStyleDeclaration
    indentClosingBrace = True
        Defines if closing brace of block is indented to match indentation
        of the block (default) oder match indentation of selector.
    indentSpecificities = False (**EXPERIMENTAL**)
        Indent rules with subset of Selectors and higher Specitivity

    keepAllProperties = True
        If ``True`` all properties set in the original CSSStylesheet
        are kept meaning even properties set twice with the exact same
        same name are kept!
    keepComments = True
        If ``False`` removes all CSSComments
    keepEmptyRules = False
        defines if empty rules like e.g. ``a {}`` are kept in the resulting
        serialized sheet
    keepUnknownAtRules = True
        defines if unknown @rules like e.g. ``@three-dee {}`` are kept in the
        serialized sheet
    keepUsedNamespaceRulesOnly = False
        if True only namespace rules which are actually used are kept

    lineNumbers = False
        Only used if a complete CSSStyleSheet is serialized.
    lineSeparator = u'\\n'
        How to end a line. This may be set to e.g. u'' for serializing of
        CSSStyleDeclarations usable in HTML style attribute.
    listItemSpacer = u' '
        string which is used in ``css.SelectorList``, ``css.CSSValue`` and
        ``stylesheets.MediaList`` after the comma
    minimizeColorHash = True
        defines if colorhash should be minimized from full size to shorthand
        e.g minimize #FFFFFF to #FFF
    normalizedVarNames = True
        defines if variable names should be serialized normalized (they are
        used as being normalized anyway)
    omitLastSemicolon = True
        If ``True`` omits ; after last property of CSSStyleDeclaration
    omitLeadingZero = False
        defines if values between -1 and 1 should omit the 0, like ``.5px``
    paranthesisSpacer = u' '
        string which is used before an opening paranthesis like in a
        ``css.CSSMediaRule`` or ``css.CSSStyleRule``
    propertyNameSpacer = u' '
        string which is used after a Property name colon
    resolveVariables = True
        if ``True`` all variable references are tried to resolved and
        all CSSVariablesRules are removed from the output.
        Any variable reference not resolvable is simply kept untouched.
    selectorCombinatorSpacer = u' '
        string which is used before and after a Selector combinator like +, > or ~.
        CSSOM defines a single space for this which is also the default in cssutils.
    spacer = u' '
        general spacer, used e.g. by CSSUnknownRule

    validOnly = False
        if True only valid (Properties) are output

        A Property is valid if it is a known Property with a valid value.
    """

    def __init__(self, **initials):
        """Always use named instead of positional parameters."""
        self.useDefaults()
        for key, value in list(initials.items()):
            if value:
                self.__setattr__(key, value)

    def __repr__(self):
        return "cssutils.css.%s(%s)" % (
            self.__class__.__name__,
            ', '.join(
                ['\n    %s=%r' % (p, self.__getattribute__(p)) for p in self.__dict__]
            ),
        )

    def __str__(self):
        return "<cssutils.css.%s object %s at 0x%x" % (
            self.__class__.__name__,
            ' '.join(['%s=%r' % (p, self.__getattribute__(p)) for p in self.__dict__]),
            id(self),
        )

    def useDefaults(self):
        "Reset all preference options to their default value."
        self.defaultAtKeyword = True
        self.defaultPropertyName = True
        self.defaultPropertyPriority = True
        self.importHrefFormat = None
        self.indent = 4 * ' '
        self.indentClosingBrace = True
        self.indentSpecificities = False
        self.keepAllProperties = True
        self.keepComments = True
        self.keepEmptyRules = False
        self.keepUnknownAtRules = True
        self.keepUsedNamespaceRulesOnly = False
        self.lineNumbers = False
        self.lineSeparator = '\n'
        self.listItemSpacer = ' '
        self.minimizeColorHash = True
        self.normalizedVarNames = True
        self.omitLastSemicolon = True
        self.omitLeadingZero = False
        self.paranthesisSpacer = ' '
        self.propertyNameSpacer = ' '
        self.resolveVariables = True
        self.selectorCombinatorSpacer = ' '
        self.spacer = ' '
        self.validOnly = False  # should not be changed currently!!!

    def useMinified(self):
        """Set options resulting in a minified stylesheet.

        You may want to set preferences with this convenience method
        and override specific settings you want adjusted afterwards.
        """
        self.importHrefFormat = 'string'
        self.indent = ''
        self.keepComments = False
        self.keepEmptyRules = False
        self.keepUnknownAtRules = False
        self.keepUsedNamespaceRulesOnly = True
        self.lineNumbers = False
        self.lineSeparator = ''
        self.listItemSpacer = ''
        self.minimizeColorHash = True
        self.omitLastSemicolon = True
        self.omitLeadingZero = True
        self.paranthesisSpacer = ''
        self.propertyNameSpacer = ''
        self.selectorCombinatorSpacer = ''
        self.spacer = ''
        self.validOnly = False


class Out(object):
    """A simple class which makes appended items available as a combined string"""

    def __init__(self, ser):
        self.ser = ser
        self.out = []

    def _remove_last_if_S(self):
        if self.out and not self.out[-1].strip():
            # remove trailing S
            del self.out[-1]

    def append(  # noqa: C901
        self, val, type_=None, space=True, keepS=False, indent=False, alwaysS=False
    ):
        """Appends val. Adds a single S after each token except as follows:

        - typ COMMENT
            uses cssText depending on self.ser.prefs.keepComments
        - typ "Property", cssutils.css.CSSRule.UNKNOWN_RULE
            uses cssText
        - typ STRING
            escapes helper.string
        - typ S
            ignored except ``keepS=True``
        - typ URI
            calls helper.uri
        - val ``{``
            adds LF after
        - val ``;``, typ 'styletext'
            removes S before and adds LF after
        - val ``, :``
            removes S before
        - val ``+ > ~``
            encloses in prefs.selectorCombinatorSpacer
        - some other vals
            add ``*spacer`` except ``space=False``
        """
        prefspace = self.ser.prefs.spacer
        if val or type_ in ('STRING', 'URI'):
            # PRE
            if 'COMMENT' == type_:
                if self.ser.prefs.keepComments:
                    val = val.cssText
                else:
                    return
            elif 'S' == type_ and not keepS:
                return
            elif 'S' == type_ and keepS:
                val = ' '
            elif 'STRING' == type_:
                # may be empty but MUST not be None
                if val is None:
                    return
                val = helper.string(val)
                if not prefspace:
                    self._remove_last_if_S()
            elif 'URI' == type_:
                val = helper.uri(val)
            elif 'HASH' == type_:
                val = self.ser._hash(val)
            elif hasattr(val, 'cssText'):
                val = val.cssText
            elif hasattr(val, 'mediaText'):
                val = val.mediaText
            elif val in '+>~,:{;)]/=}' and not alwaysS:
                self._remove_last_if_S()
            # elif type_ in ('Property', cssutils.css.CSSRule.UNKNOWN_RULE):
            #     val = val.cssText
            # elif type_ in ('NUMBER', 'DIMENSION', 'PERCENTAGE') and val == u'0':
            #     # remove sign + or - if value is zero
            #     # TODO: only for lenghts!
            #     if self.out and self.out[-1] in u'+-':
            #         del self.out[-1]

            # APPEND

            if indent or (val == '}' and self.ser.prefs.indentClosingBrace):
                self.out.append(self.ser._indentblock(val, self.ser._level + 1))
            else:
                if val.endswith(' '):
                    self._remove_last_if_S()
                self.out.append(val)

            # POST
            if alwaysS and val in '-+*/':  # calc, */ not really but do anyway
                self.out.append(' ')
            elif val in '+>~':  # enclose selector combinator
                self.out.insert(-1, self.ser.prefs.selectorCombinatorSpacer)
                self.out.append(self.ser.prefs.selectorCombinatorSpacer)
            elif ')' == val and not keepS:  # CHAR funcend
                # TODO: pref?
                self.out.append(' ')
            elif ',' == val:  # list
                self.out.append(self.ser.prefs.listItemSpacer)
            elif ':' == val:  # prop
                self.out.append(self.ser.prefs.propertyNameSpacer)
            elif '{' == val:  # block start
                self.out.insert(-1, self.ser.prefs.paranthesisSpacer)
                self.out.append(self.ser.prefs.lineSeparator)
            elif ';' == val or 'styletext' == type_:  # end or prop or block
                self.out.append(self.ser.prefs.lineSeparator)
            elif val not in '}[]()/=' and space and type_ != 'FUNCTION':
                self.out.append(self.ser.prefs.spacer)
                if (
                    type_ != 'STRING'
                    and not self.ser.prefs.spacer
                    and self.out
                    and not self.out[-1].endswith(' ')
                ):
                    self.out.append(' ')

    def value(self, delim='', end=None, keepS=False):
        "returns all items joined by delim"
        if not keepS:
            self._remove_last_if_S()
        if end:
            self.out.append(end)
        return delim.join(self.out)


class CSSSerializer(object):
    """Serialize a CSSStylesheet and its parts.

    To use your own serializing method the easiest is to subclass CSS
    Serializer and overwrite the methods you like to customize.
    """

    def __init__(self, prefs=None):
        """
        :param prefs:
            instance of Preferences
        """
        if not prefs:
            prefs = Preferences()
        self.prefs = prefs
        self._level = 0  # current nesting level

        # TODO:
        self._selectors = []  # holds SelectorList
        self._selectorlevel = 0  # current specificity nesting level

    def _atkeyword(self, rule):
        "returns default or source atkeyword depending on prefs"
        if self.prefs.defaultAtKeyword:
            return rule.atkeyword  # default
        else:
            return rule._keyword

    def _indentblock(self, text, level):
        """
        indent a block like a CSSStyleDeclaration to the given level
        which may be higher than self._level (e.g. for CSSStyleDeclaration)
        """
        if not self.prefs.lineSeparator:
            return text
        return self.prefs.lineSeparator.join(
            [
                '%s%s' % (level * self.prefs.indent, line)
                for line in text.split(self.prefs.lineSeparator)
            ]
        )

    def _propertyname(self, property, actual):
        """
        used by all styledeclarations to get the propertyname used
        dependent on prefs setting defaultPropertyName and
        keepAllProperties
        """
        if self.prefs.defaultPropertyName and not self.prefs.keepAllProperties:
            return property.name
        else:
            return actual

    def _linenumnbers(self, text):
        if self.prefs.lineNumbers:
            pad = len(str(text.count(self.prefs.lineSeparator) + 1))
            out = []
            for i, line in enumerate(text.split(self.prefs.lineSeparator)):
                out.append(('%*i: %s') % (pad, i + 1, line))
            text = self.prefs.lineSeparator.join(out)
        return text

    def _hash(self, val, type_=None):
        """
        Short form of hash, e.g. #123 instead of #112233
        """
        if (
            self.prefs.minimizeColorHash
            and len(val) == 7
            and val[1] == val[2]
            and val[3] == val[4]
            and val[5] == val[6]
        ):
            return '#%s%s%s' % (val[1], val[3], val[5])
        return val

    def _valid(self, x):
        "checks items valid property and prefs.validOnly"
        return not self.prefs.validOnly or (self.prefs.validOnly and x.valid)

    def do_CSSStyleSheet(self, stylesheet):
        """serializes a complete CSSStyleSheet"""
        useduris = stylesheet._getUsedURIs()
        out = []
        for rule in stylesheet.cssRules:
            if (
                self.prefs.keepUsedNamespaceRulesOnly
                and rule.NAMESPACE_RULE == rule.type
                and rule.namespaceURI not in useduris
                and (rule.prefix or None not in useduris)
            ):
                continue

            cssText = rule.cssText
            if cssText:
                out.append(cssText)
        text = self._linenumnbers(self.prefs.lineSeparator.join(out))

        # get encoding of sheet, defaults to UTF-8
        try:
            encoding = stylesheet.cssRules[0].encoding
        except (IndexError, AttributeError):
            encoding = 'UTF-8'

        # TODO: py3 return b str but tests use unicode?
        return text.encode(encoding, 'escapecss')

    def do_CSSComment(self, rule):
        """
        serializes CSSComment which consists only of commentText
        """
        if rule._cssText and self.prefs.keepComments:
            return rule._cssText
        else:
            return ''

    def do_CSSCharsetRule(self, rule):
        """
        serializes CSSCharsetRule
        encoding: string

        always @charset "encoding";
        no comments or other things allowed!
        """
        if rule.wellformed:
            return '@charset %s;' % helper.string(rule.encoding)
        else:
            return ''

    def do_CSSVariablesRule(self, rule):
        """
        serializes CSSVariablesRule

        media
            TODO
        variables
            CSSStyleDeclaration

        + CSSComments
        """
        variablesText = rule.variables.cssText

        if variablesText and rule.wellformed and not self.prefs.resolveVariables:
            out = Out(self)
            out.append(self._atkeyword(rule))
            for item in rule.seq:
                # assume comments {
                out.append(item.value, item.type)
            out.append('{')
            out.append('%s%s}' % (variablesText, self.prefs.lineSeparator), indent=1)
            return out.value()
        else:
            return ''

    def do_CSSFontFaceRule(self, rule):
        """
        serializes CSSFontFaceRule

        style
            CSSStyleDeclaration

        + CSSComments
        """
        styleText = self.do_css_CSSStyleDeclaration(rule.style)

        if styleText and rule.wellformed:
            out = Out(self)
            out.append(self._atkeyword(rule))
            for item in rule.seq:
                # assume comments {
                out.append(item.value, item.type)
            out.append('{')
            out.append('%s%s}' % (styleText, self.prefs.lineSeparator), indent=1)
            return out.value()
        else:
            return ''

    def do_CSSImportRule(self, rule):
        """
        serializes CSSImportRule

        href
            string
        media
            optional cssutils.stylesheets.medialist.MediaList
        name
            optional string

        + CSSComments
        """
        if rule.wellformed:
            out = Out(self)
            out.append(self._atkeyword(rule))

            for item in rule.seq:
                type_, val = item.type, item.value
                if 'href' == type_:
                    # "href" or url(href)
                    if self.prefs.importHrefFormat == 'string' or (
                        self.prefs.importHrefFormat != 'uri'
                        and rule.hreftype == 'string'
                    ):
                        out.append(val, 'STRING')
                    else:
                        out.append(val, 'URI')
                elif 'media' == type_:
                    # media
                    mediaText = self.do_stylesheets_medialist(val)
                    if mediaText and mediaText != 'all':
                        out.append(mediaText)
                elif 'name' == type_:
                    out.append(val, 'STRING')
                else:
                    out.append(val, type_)

            return out.value(end=';')
        else:
            return ''

    def do_CSSNamespaceRule(self, rule):
        """
        serializes CSSNamespaceRule

        uri
            string
        prefix
            string

        + CSSComments
        """
        if rule.wellformed:
            out = Out(self)
            out.append(self._atkeyword(rule))
            for item in rule.seq:
                type_, val = item.type, item.value
                if 'namespaceURI' == type_:
                    out.append(val, 'STRING')
                else:
                    out.append(val, type_)

            return out.value(end=';')
        else:
            return ''

    def do_CSSMediaRule(self, rule):
        """
        serializes CSSMediaRule

        + CSSComments
        """
        # TODO: use Out()?

        # mediaquery
        if not rule.media.wellformed:
            return ''

        # @media
        out = [self._atkeyword(rule)]
        if not len(self.prefs.spacer):
            # for now always with space as only webkit supports @mediaall?
            out.append(' ')
        else:
            out.append(self.prefs.spacer)  # might be empty

        out.append(self.do_stylesheets_medialist(rule.media))

        # name, seq contains content after name only (Comments)
        if rule.name:
            out.append(self.prefs.spacer)
            nameout = Out(self)
            nameout.append(helper.string(rule.name))
            for item in rule.seq:
                nameout.append(item.value, item.type)
            out.append(nameout.value())

        #  {
        out.append(self.prefs.paranthesisSpacer)
        out.append('{')
        out.append(self.prefs.lineSeparator)

        # rules
        rulesout = []
        for r in rule.cssRules:
            rtext = r.cssText
            if rtext:
                # indent each line of cssText
                rulesout.append(self._indentblock(rtext, self._level + 1))
                rulesout.append(self.prefs.lineSeparator)
        if not self.prefs.keepEmptyRules and not ''.join(rulesout).strip():
            return ''
        out.extend(rulesout)

        #     }
        out.append(
            '%s}'
            % ((self._level + int(self.prefs.indentClosingBrace)) * self.prefs.indent)
        )

        return ''.join(out)

    def do_CSSPageRule(self, rule):
        """
        serializes CSSPageRule

        selectorText
            string
        style
            CSSStyleDeclaration
        cssRules
            CSSRuleList of MarginRule objects

        + CSSComments
        """
        # rules
        rulesout = []
        for r in rule.cssRules:
            rtext = r.cssText
            if rtext:
                rulesout.append(rtext)
                rulesout.append(self.prefs.lineSeparator)

        rulesText = ''.join(rulesout)  # .strip()

        # omit semicolon only if no MarginRules
        styleText = self.do_css_CSSStyleDeclaration(rule.style, omit=not rulesText)

        if (styleText or rulesText) and rule.wellformed:
            out = Out(self)
            out.append(self._atkeyword(rule))
            out.append(rule.selectorText)
            out.append('{')

            if styleText:
                if not rulesText:
                    out.append('%s%s' % (styleText, self.prefs.lineSeparator), indent=1)
                else:
                    out.append(styleText, type_='styletext', indent=1, space=False)

            if rulesText:
                out.append(rulesText, indent=1)
            # ?
            self._level -= 1
            out.append('}')
            self._level += 1

            return out.value()
        else:
            return ''

    def do_CSSPageRuleSelector(self, seq):
        "Serialize selector of a CSSPageRule"
        out = Out(self)
        for item in seq:
            if item.type == 'IDENT':
                out.append(item.value, item.type, space=False)
            else:
                out.append(item.value, item.type)
        return out.value()

    def do_MarginRule(self, rule):
        """
        serializes MarginRule

        atkeyword
            string
        style
            CSSStyleDeclaration

        + CSSComments
        """
        # might not be set at all?!
        if rule.atkeyword:
            styleText = self.do_css_CSSStyleDeclaration(rule.style)

            if styleText and rule.wellformed:
                out = Out(self)

                # # use seq but styledecl missing
                # for item in rule.seq:
                #     if item.type == 'ATKEYWORD':
                #         # move logic to Out
                #         out.append(self._atkeyword(rule), type_=item.type)
                #     else:
                #         print type_, val
                #         out.append(item.value, item.type)
                # return out.value()

                # ok for now:
                out.append(self._atkeyword(rule), type_='ATKEYWORD')
                out.append('{')
                out.append(
                    '%s%s'
                    % (
                        self._indentblock(styleText, self._level + 1),
                        self.prefs.lineSeparator,
                    )
                )
                out.append('}')
                return out.value()

        return ''

    def do_CSSUnknownRule(self, rule):
        """
        serializes CSSUnknownRule
        anything until ";" or "{...}"
        + CSSComments
        """
        if rule.wellformed and self.prefs.keepUnknownAtRules:
            out = Out(self)
            out.append(rule.atkeyword)

            stacks = []
            for item in rule.seq:
                type_, val = item.type, item.value
                # PRE
                if '}' == val:
                    # close last open item on stack
                    stackblock = stacks.pop().value()
                    if stackblock:
                        val = self._indentblock(
                            stackblock + self.prefs.lineSeparator + val,
                            min(1, len(stacks) + 1),
                        )
                    else:
                        val = self._indentblock(val, min(1, len(stacks) + 1))
                # APPEND
                if stacks:
                    stacks[-1].append(val, type_)
                else:
                    out.append(val, type_)

                # POST
                if '{' == val:
                    # new stack level
                    stacks.append(Out(self))

            return out.value()
        else:
            return ''

    def do_CSSStyleRule(self, rule):
        """
        serializes CSSStyleRule

        selectorList
        style

        + CSSComments
        """
        # TODO: use Out()

        # prepare for element nested rules
        # TODO: sort selectors!
        if self.prefs.indentSpecificities:
            # subselectorlist?
            elements = set([s.element for s in rule.selectorList])
            specitivities = [s.specificity for s in rule.selectorList]
            for selector in self._selectors:
                lastelements = set([s.element for s in selector])
                if elements.issubset(lastelements):
                    # higher specificity?
                    lastspecitivities = [s.specificity for s in selector]
                    if specitivities > lastspecitivities:
                        self._selectorlevel += 1
                        break
                elif self._selectorlevel > 0:
                    self._selectorlevel -= 1
            else:
                # save new reference
                self._selectors.append(rule.selectorList)
                self._selectorlevel = 0

        # TODO ^ RESOLVE!!!!

        selectorText = self.do_css_SelectorList(rule.selectorList)
        if not selectorText or not rule.wellformed:
            return ''
        self._level += 1
        styleText = ''
        try:
            styleText = self.do_css_CSSStyleDeclaration(rule.style)
        finally:
            self._level -= 1
        if not styleText:
            if self.prefs.keepEmptyRules:
                return '%s%s{}' % (selectorText, self.prefs.paranthesisSpacer)
        else:
            return self._indentblock(
                '%s%s{%s%s%s%s}'
                % (
                    selectorText,
                    self.prefs.paranthesisSpacer,
                    self.prefs.lineSeparator,
                    self._indentblock(styleText, self._level + 1),
                    self.prefs.lineSeparator,
                    (self._level + int(self.prefs.indentClosingBrace))
                    * self.prefs.indent,
                ),
                self._selectorlevel,
            )

    def do_css_SelectorList(self, selectorlist):
        "comma-separated list of Selectors"
        # does not need Out() as it is too simple
        if selectorlist.wellformed:
            out = []
            for part in selectorlist.seq:
                if isinstance(part, cssutils.css.Selector):
                    out.append(part.selectorText)
                else:
                    out.append(part)  # should not happen
            sep = ',%s' % self.prefs.listItemSpacer
            return sep.join(out)
        else:
            return ''

    def do_css_Selector(self, selector):
        """
        a single Selector including comments

        an element has syntax (namespaceURI, name) where namespaceURI may be:

        - cssutils._ANYNS => ``*|name``
        - None => ``name``
        - u'' => ``|name``
        - any other value: => ``prefix|name``
        """
        if selector.wellformed:
            out = Out(self)

            DEFAULTURI = selector._namespaces.get('', None)
            for item in selector.seq:
                type_, val = item.type, item.value
                if isinstance(val, tuple):
                    # namespaceURI|name (element or attribute)
                    namespaceURI, name = val
                    if DEFAULTURI == namespaceURI or (
                        not DEFAULTURI and namespaceURI is None
                    ):
                        out.append(name, type_, space=False)
                    else:
                        if namespaceURI == cssutils._ANYNS:
                            prefix = '*'
                        else:
                            try:
                                prefix = selector._namespaces.prefixForNamespaceURI(
                                    namespaceURI
                                )
                            except IndexError:
                                prefix = ''

                        out.append('%s|%s' % (prefix, name), type_, space=False)
                else:
                    out.append(val, type_, space=False, keepS=True)

            return out.value()
        else:
            return ''

    def do_css_CSSVariablesDeclaration(self, variables):
        """Variables of CSSVariableRule."""
        if len(variables.seq) > 0:
            out = Out(self)

            lastitem = len(variables.seq) - 1
            for i, item in enumerate(variables.seq):
                type_, val = item.type, item.value
                if 'var' == type_:
                    name, cssvalue = val
                    if self.prefs.normalizedVarNames:
                        name = normalize(name)
                    out.append(name)
                    out.append(':')
                    out.append(cssvalue.cssText)
                    if i < lastitem or not self.prefs.omitLastSemicolon:
                        out.append(';')

                elif isinstance(val, cssutils.css.CSSComment):
                    # CSSComment
                    out.append(val, 'COMMENT')
                    out.append(self.prefs.lineSeparator)
                else:
                    out.append(val.cssText, type_)
                    out.append(self.prefs.lineSeparator)

            return out.value().strip()

        else:
            return ''

    def do_css_CSSStyleDeclaration(  # noqa: C901
        self, style, separator=None, omit=True
    ):
        """
        Style declaration of CSSStyleRule
        """
        # TODO: use Out()

        # may be comments only
        if len(style.seq) > 0:
            if separator is None:
                separator = self.prefs.lineSeparator

            if self.prefs.keepAllProperties:
                # all
                seq = style.seq
            else:
                # only effective ones
                _effective = style.getProperties()
                seq = [
                    item
                    for item in style.seq
                    if (
                        isinstance(item.value, cssutils.css.Property)
                        and item.value in _effective
                    )
                    or not isinstance(item.value, cssutils.css.Property)
                ]

            out = []
            omitLastSemicolon = omit and self.prefs.omitLastSemicolon

            for i, item in enumerate(seq):
                val = item.value
                if isinstance(val, cssutils.css.CSSComment):
                    # CSSComment
                    if self.prefs.keepComments:
                        out.append(val.cssText)
                        out.append(separator)
                elif isinstance(val, cssutils.css.Property):
                    # PropertySimilarNameList
                    if val.cssText:
                        out.append(val.cssText)
                        if not (omitLastSemicolon and i == len(seq) - 1):
                            out.append(';')
                        out.append(separator)
                elif isinstance(val, cssutils.css.CSSUnknownRule):
                    # @rule
                    out.append(val.cssText)
                    out.append(separator)
                else:
                    # ?
                    out.append(val)
                    out.append(separator)

            if out and out[-1] == separator:
                del out[-1]

            return ''.join(out)

        else:
            return ''

    def do_Property(self, property):
        """
        Style declaration of CSSStyleRule

        Property has a seqs attribute which contains seq lists for
        name, a CSSvalue and a seq list for priority
        """
        # TODO: use Out()

        out = []
        if property.seqs[0] and property.wellformed and self._valid(property):
            nameseq, value, priorityseq = property.seqs

            # name
            for part in nameseq:
                if hasattr(part, 'cssText'):
                    out.append(part.cssText)
                elif property.literalname == part:
                    out.append(self._propertyname(property, part))
                else:
                    out.append(part)

            if out and (
                not property._mediaQuery or property._mediaQuery and value.cssText
            ):
                # MediaQuery may consist of name only
                out.append(':')
                out.append(self.prefs.propertyNameSpacer)

            # value
            out.append(value.cssText)

            # priority
            if out and priorityseq:
                out.append(' ')
                for part in priorityseq:
                    if hasattr(part, 'cssText'):  # comments
                        out.append(part.cssText)
                    else:
                        if (
                            part == property.literalpriority
                            and self.prefs.defaultPropertyPriority
                        ):
                            out.append(property.priority)
                        else:
                            out.append(part)

        return ''.join(out)

    def do_Property_priority(self, priorityseq):
        """
        a Properties priority "!" S* "important"
        """
        # TODO: use Out()
        out = []
        for part in priorityseq:
            if hasattr(part, 'cssText'):  # comments
                out.append(' ')
                out.append(part.cssText)
                out.append(' ')
            else:
                out.append(part)
        return ''.join(out).strip()

    def do_css_PropertyValue(self, value, valuesOnly=False):
        """Serializes a PropertyValue"""
        if not value:
            return ''
        else:
            out = Out(self)
            for item in value.seq:
                type_, val = item.type, item.value
                if valuesOnly and type_ == cssutils.css.CSSComment:
                    continue
                elif hasattr(val, 'cssText'):
                    # RGBColor or CSSValue if a CSSValueList
                    out.append(val.cssText, type_)
                else:
                    if val and val[0] == val[-1] and val[0] in '\'"':
                        val = helper.string(val[1:-1])
                    # S must be kept! in between values but no extra space
                    out.append(val, type_)

            return out.value()

    def _strip_zeros(self, s):
        i = s.index('.') + 2
        a, b = s[0:i], s[i : len(s)]
        b = b.rstrip('0')
        return a + b

    def do_css_Value(self, value, valuesOnly=None):
        """Serializes a Value, valuesOnly is ignored"""
        if not value:
            return ''
        else:
            out = Out(self)
            if value.type in ('DIMENSION', 'NUMBER', 'PERCENTAGE'):
                dim = value.dimension or ''
                if value.value == 0:
                    val = '0'
                    if value.dimension in (
                        'cm',
                        'mm',
                        'in',
                        'px',
                        'pc',
                        'pt',
                        'em',
                        'ex',
                    ):
                        dim = ''
                elif value.value == int(value.value):
                    # cut off after . which is zero anyway
                    val = str(int(value.value))
                elif self.prefs.omitLeadingZero and -1 < value.value < 1:
                    v = self._strip_zeros('%f' % value.value)  # issue #27
                    val = v
                    if value._sign == '-':
                        val = v[0] + v[2:]
                    else:
                        val = v[1:]
                else:

                    val = self._strip_zeros('%f' % value.value)  # issue #27

                # keep '+' if given
                if value.value != 0 and value._sign == '+':
                    sign = '+'
                else:
                    sign = ''

                out.append(sign + val + dim, value.type)

            else:
                # e.g. URI
                out.append(value.value, value.type)

        return out.value()

    def do_css_ColorValue(self, value, valuesOnly=False):
        """Serialize a ColorValue, a HASH simple value or FUNCTION"""
        try:
            return {
                'FUNCTION': self.do_css_CSSFunction,
                'HASH': self.do_css_Value,
                'IDENT': self.do_css_Value,
            }[value.colorType](value, valuesOnly=valuesOnly)
        except KeyError:
            return ''

    def do_css_CSSFunction(self, cssvalue, valuesOnly=False):
        """Serialize a CSS function value"""
        if not cssvalue:
            return ''
        else:
            out = Out(self)
            for item in cssvalue.seq:
                type_, val = item.type, item.value
                if valuesOnly and type_ == cssutils.css.CSSComment:
                    continue
                out.append(val, type_)
            return out.value()

    def do_css_CSSCalc(self, cssvalue, valuesOnly=False):
        """Serialize a CSS calc value"""
        if not cssvalue:
            return ''
        else:
            out = Out(self)
            for item in cssvalue.seq:
                type_, val = item.type, item.value

                if valuesOnly and type_ == cssutils.css.CSSComment:
                    continue
                elif hasattr(val, 'cssText'):
                    # RGBColor or CSSValue if a CSSValueList
                    out.append(val.cssText, type_)
                elif type_ == 'CHAR' and val in '-+*/':
                    out.append(val, type_, alwaysS=True)
                else:
                    out.append(val, type_)

            return out.value()

    def do_css_MSValue(self, cssvalue, valuesOnly=False):
        """Serialize an ExpressionValue (IE only),
        should at least keep the original syntax"""
        if not cssvalue:
            return ''
        else:
            out = Out(self)
            for item in cssvalue.seq:
                val = item.value
                # type_ = item.type
                # val = self._possiblezero(cssvalue, type_, val)
                # do no send type_ so no special cases!
                out.append(val, None, space=False)

            return out.value()

    def do_css_CSSVariable(self, variable, IGNORED=False):
        """Serializes a CSSVariable"""
        if not variable or not variable.name:
            return ''
        else:
            out = Out(self)
            v = variable.value
            if self.prefs.resolveVariables and v:
                # resolve variable
                out.append(v)

            else:
                # keep var(NAME)
                out.append('var(', 'FUNCTION')
                out.append(variable.name, 'IDENT')
                if variable.fallback:
                    out.append(',', 'COMMA')
                    out.append(variable.fallback.cssText)
                out.append(')')

            return out.value()

    def do_stylesheets_medialist(self, medialist):
        """
        comma-separated list of media, default is 'all'

        If "all" is in the list, every other media *except* "handheld" will
        be stripped. This is because how Opera handles CSS for PDAs.
        """
        if len(medialist) == 0:
            return 'all'
        else:
            seq = medialist.seq
            out = Out(self)
            firstdone = False

            for item in seq:
                if item.type == 'MediaQuery':
                    if firstdone:
                        out.append(',', 'CHAR')
                    else:
                        firstdone = True

                out.append(item.value, item.type)

            return out.value()

    def do_stylesheets_mediaquery(self, mediaquery):
        """
        a single media used in medialist
        """
        if not mediaquery.wellformed:
            return ''
        else:
            withsemi = []
            nextmq = False
            for item in mediaquery.seq:
                type_, val = item.type, item.value

                if type_ == 'MediaQuery' and nextmq:
                    withsemi.append(('CHAR', ','))
                    nextmq = False
                else:
                    nextmq = True

                withsemi.append((type_, val))

            out = Out(self)
            for (
                t,
                v,
            ) in withsemi:
                out.append(v, t)
            # for item in mediaquery.seq:
            #    type_, val = item.type, item.value
            #    out.append(val, type_)#, space=False)

            return out.value()

        # if mediaquery.wellformed:
        #    out = []
        #    for part in mediaquery.seq:
        #        if isinstance(part, cssutils.css.Property): # Property
        #            out.append(u'(%s)' % part.cssText)
        #        elif hasattr(part, 'cssText'): # comments
        #            out.append(part.cssText)
        #        else:
        #            # TODO: media queries!
        #            out.append(part)
        #    return u' '.join(out)
        # else:
        #    return u''
