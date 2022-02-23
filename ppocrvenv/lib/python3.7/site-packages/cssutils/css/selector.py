"""Selector is a single Selector of a CSSStyleRule SelectorList.
Partly implements http://www.w3.org/TR/css3-selectors/.

TODO
    - .contains(selector)
    - .isSubselector(selector)
"""
__all__ = ['Selector']

from cssutils.helper import Deprecated
from cssutils.util import _SimpleNamespaces
import cssutils
import xml.dom


class Selector(cssutils.util.Base2):
    """
    (cssutils) a single selector in a :class:`~cssutils.css.SelectorList`
    of a :class:`~cssutils.css.CSSStyleRule`.

    Format::

        # implemented in SelectorList
        selectors_group
          : selector [ COMMA S* selector ]*
          ;

        selector
          : simple_selector_sequence [ combinator simple_selector_sequence ]*
          ;

        combinator
          /* combinators can be surrounded by white space */
          : PLUS S* | GREATER S* | TILDE S* | S+
          ;

        simple_selector_sequence
          : [ type_selector | universal ]
            [ HASH | class | attrib | pseudo | negation ]*
          | [ HASH | class | attrib | pseudo | negation ]+
          ;

        type_selector
          : [ namespace_prefix ]? element_name
          ;

        namespace_prefix
          : [ IDENT | '*' ]? '|'
          ;

        element_name
          : IDENT
          ;

        universal
          : [ namespace_prefix ]? '*'
          ;

        class
          : '.' IDENT
          ;

        attrib
          : '[' S* [ namespace_prefix ]? IDENT S*
                [ [ PREFIXMATCH |
                    SUFFIXMATCH |
                    SUBSTRINGMATCH |
                    '=' |
                    INCLUDES |
                    DASHMATCH ] S* [ IDENT | STRING ] S*
                ]? ']'
          ;

        pseudo
          /* '::' starts a pseudo-element, ':' a pseudo-class */
          /* Exceptions: :first-line, :first-letter, :before and :after. */
          /* Note that pseudo-elements are restricted to one per selector and */
          /* occur only in the last simple_selector_sequence. */
          : ':' ':'? [ IDENT | functional_pseudo ]
          ;

        functional_pseudo
          : FUNCTION S* expression ')'
          ;

        expression
          /* In CSS3, the expressions are identifiers, strings, */
          /* or of the form "an+b" */
          : [ [ PLUS | '-' | DIMENSION | NUMBER | STRING | IDENT ] S* ]+
          ;

        negation
          : NOT S* negation_arg S* ')'
          ;

        negation_arg
          : type_selector | universal | HASH | class | attrib | pseudo
          ;

    """

    def __init__(self, selectorText=None, parent=None, readonly=False):
        """
        :Parameters:
            selectorText
                initial value of this selector
            parent
                a SelectorList
            readonly
                default to False
        """
        super(Selector, self).__init__()

        self.__namespaces = _SimpleNamespaces(log=self._log)
        self._element = None
        self._parent = parent
        self._specificity = (0, 0, 0, 0)

        if selectorText:
            self.selectorText = selectorText

        self._readonly = readonly

    def __repr__(self):
        if self.__getNamespaces():
            st = (self.selectorText, self._getUsedNamespaces())
        else:
            st = self.selectorText
        return "cssutils.css.%s(selectorText=%r)" % (self.__class__.__name__, st)

    def __str__(self):
        return (
            "<cssutils.css.%s object selectorText=%r specificity=%r"
            " _namespaces=%r at 0x%x>"
            % (
                self.__class__.__name__,
                self.selectorText,
                self.specificity,
                self._getUsedNamespaces(),
                id(self),
            )
        )

    def _getUsedUris(self):
        "Return list of actually used URIs in this Selector."
        uris = set()
        for item in self.seq:
            type_, val = item.type, item.value
            if (
                type_.endswith('-selector')
                or type_ == 'universal'
                and isinstance(val, tuple)
                and val[0] not in (None, '*')
            ):
                uris.add(val[0])
        return uris

    def _getUsedNamespaces(self):
        "Return actually used namespaces only."
        useduris = self._getUsedUris()
        namespaces = _SimpleNamespaces(log=self._log)
        for p, uri in list(self._namespaces.items()):
            if uri in useduris:
                namespaces[p] = uri
        return namespaces

    def __getNamespaces(self):
        "Use own namespaces if not attached to a sheet, else the sheet's ones."
        try:
            return self._parent.parentRule.parentStyleSheet.namespaces
        except AttributeError:
            return self.__namespaces

    _namespaces = property(
        __getNamespaces,
        doc="If this Selector is attached to a "
        "CSSStyleSheet the namespaces of that sheet "
        "are mirrored here. While the Selector (or "
        "parent SelectorList or parentRule(s) of that "
        "are not attached a own dict of {prefix: "
        "namespaceURI} is used.",
    )

    element = property(
        lambda self: self._element, doc="Effective element target of this selector."
    )

    parent = property(
        lambda self: self._parent,
        doc="(DOM) The SelectorList that contains this Selector "
        "or None if this Selector is not attached to a "
        "SelectorList.",
    )

    def _getSelectorText(self):
        """Return serialized format."""
        return cssutils.ser.do_css_Selector(self)

    def _setSelectorText(self, selectorText):  # noqa: C901
        """
        :param selectorText:
            parsable string or a tuple of (selectorText, dict-of-namespaces).
            Given namespaces are ignored if this object is attached to a
            CSSStyleSheet!

        :exceptions:
            - :exc:`~xml.dom.NamespaceErr`:
              Raised if the specified selector uses an unknown namespace
              prefix.
            - :exc:`~xml.dom.SyntaxErr`:
              Raised if the specified CSS string value has a syntax error
              and is unparsable.
            - :exc:`~xml.dom.NoModificationAllowedErr`:
              Raised if this rule is readonly.
        """
        self._checkReadonly()

        # might be (selectorText, namespaces)
        selectorText, namespaces = self._splitNamespacesOff(selectorText)

        try:
            # uses parent stylesheets namespaces if available,
            # otherwise given ones
            namespaces = self.parent.parentRule.parentStyleSheet.namespaces
        except AttributeError:
            pass
        tokenizer = self._tokenize2(selectorText)
        if not tokenizer:
            self._log.error('Selector: No selectorText given.')
        else:
            # prepare tokenlist:
            #     "*" -> type "universal"
            #     "*"|IDENT + "|" -> combined to "namespace_prefix"
            #     "|" -> type "namespace_prefix"
            #     "." + IDENT -> combined to "class"
            #     ":" + IDENT, ":" + FUNCTION -> pseudo-class
            #     FUNCTION "not(" -> negation
            #     "::" + IDENT, "::" + FUNCTION -> pseudo-element
            tokens = []
            for t in tokenizer:
                typ, val, lin, col = t
                if val == ':' and tokens and self._tokenvalue(tokens[-1]) == ':':
                    # combine ":" and ":"
                    tokens[-1] = (typ, '::', lin, col)

                elif typ == 'IDENT' and tokens and self._tokenvalue(tokens[-1]) == '.':
                    # class: combine to .IDENT
                    tokens[-1] = ('class', '.' + val, lin, col)
                elif (
                    typ == 'IDENT'
                    and tokens
                    and self._tokenvalue(tokens[-1]).startswith(':')
                    and not self._tokenvalue(tokens[-1]).endswith('(')
                ):
                    # pseudo-X: combine to :IDENT or ::IDENT but not ":a(" + "b"
                    if self._tokenvalue(tokens[-1]).startswith('::'):
                        t = 'pseudo-element'
                    else:
                        t = 'pseudo-class'
                    tokens[-1] = (t, self._tokenvalue(tokens[-1]) + val, lin, col)

                elif (
                    typ == 'FUNCTION'
                    and val == 'not('
                    and tokens
                    and ':' == self._tokenvalue(tokens[-1])
                ):
                    tokens[-1] = ('negation', ':' + val, lin, tokens[-1][3])
                elif (
                    typ == 'FUNCTION'
                    and tokens
                    and self._tokenvalue(tokens[-1]).startswith(':')
                ):
                    # pseudo-X: combine to :FUNCTION( or ::FUNCTION(
                    if self._tokenvalue(tokens[-1]).startswith('::'):
                        t = 'pseudo-element'
                    else:
                        t = 'pseudo-class'
                    tokens[-1] = (t, self._tokenvalue(tokens[-1]) + val, lin, col)

                elif (
                    val == '*'
                    and tokens
                    and self._type(tokens[-1]) == 'namespace_prefix'
                    and self._tokenvalue(tokens[-1]).endswith('|')
                ):
                    # combine prefix|*
                    tokens[-1] = (
                        'universal',
                        self._tokenvalue(tokens[-1]) + val,
                        lin,
                        col,
                    )
                elif val == '*':
                    # universal: "*"
                    tokens.append(('universal', val, lin, col))

                elif (
                    val == '|'
                    and tokens
                    and self._type(tokens[-1]) in (self._prods.IDENT, 'universal')
                    and self._tokenvalue(tokens[-1]).find('|') == -1
                ):
                    # namespace_prefix: "IDENT|" or "*|"
                    tokens[-1] = (
                        'namespace_prefix',
                        self._tokenvalue(tokens[-1]) + '|',
                        lin,
                        col,
                    )
                elif val == '|':
                    # namespace_prefix: "|"
                    tokens.append(('namespace_prefix', val, lin, col))

                else:
                    tokens.append(t)

            tokenizer = iter(tokens)

            # for closures: must be a mutable
            new = {
                'context': [''],  # stack of: 'attrib', 'negation', 'pseudo'
                'element': None,
                '_PREFIX': None,
                'specificity': [0, 0, 0, 0],  # mutable, finally a tuple!
                'wellformed': True,
            }
            # used for equality checks and setting of a space combinator
            S = ' '

            def append(seq, val, typ=None, token=None):
                """
                appends to seq

                namespace_prefix, IDENT will be combined to a tuple
                (prefix, name) where prefix might be None, the empty string
                or a prefix.

                Saved are also:
                    - specificity definition: style, id, class/att, type
                    - element: the element this Selector is for
                """
                context = new['context'][-1]
                if token:
                    line, col = token[2], token[3]
                else:
                    line, col = None, None

                if typ == '_PREFIX':
                    # SPECIAL TYPE: save prefix for combination with next
                    new['_PREFIX'] = val[:-1]
                    # handle next time
                    return

                if new['_PREFIX'] is not None:
                    # as saved from before and reset to None
                    prefix, new['_PREFIX'] = new['_PREFIX'], None
                elif typ == 'universal' and '|' in val:
                    # val == *|* or prefix|*
                    prefix, val = val.split('|')
                else:
                    prefix = None

                # namespace
                if (typ.endswith('-selector') or typ == 'universal') and not (
                    'attribute-selector' == typ and not prefix
                ):
                    # att **IS NOT** in default ns
                    if prefix == '*':
                        # *|name: in ANY_NS
                        namespaceURI = cssutils._ANYNS
                    elif prefix is None:
                        # e or *: default namespace with prefix u''
                        # or local-name()
                        namespaceURI = namespaces.get('', None)
                    elif prefix == '':
                        # |name or |*: in no (or the empty) namespace
                        namespaceURI = ''
                    else:
                        # explicit namespace prefix
                        # does not raise KeyError, see _SimpleNamespaces
                        namespaceURI = namespaces[prefix]

                        if namespaceURI is None:
                            new['wellformed'] = False
                            self._log.error(
                                'Selector: No namespaceURI found '
                                'for prefix %r' % prefix,
                                token=token,
                                error=xml.dom.NamespaceErr,
                            )
                            return

                    # val is now (namespaceprefix, name) tuple
                    val = (namespaceURI, val)

                # specificity
                if not context or context == 'negation':
                    if 'id' == typ:
                        new['specificity'][1] += 1
                    elif 'class' == typ or '[' == val:
                        new['specificity'][2] += 1
                    elif typ in (
                        'type-selector',
                        'negation-type-selector',
                        'pseudo-element',
                    ):
                        new['specificity'][3] += 1
                if not context and typ in ('type-selector', 'universal'):
                    # define element
                    new['element'] = val

                seq.append(val, typ, line=line, col=col)

            # expected constants
            simple_selector_sequence = (
                'type_selector universal HASH class ' 'attrib pseudo negation '
            )
            simple_selector_sequence2 = 'HASH class attrib pseudo negation '

            element_name = 'element_name'

            negation_arg = 'type_selector universal HASH class attrib pseudo'
            negationend = ')'

            attname = 'prefix attribute'
            attname2 = 'attribute'
            attcombinator = 'combinator ]'  # optional
            attvalue = 'value'  # optional
            attend = ']'

            expressionstart = 'PLUS - DIMENSION NUMBER STRING IDENT'
            expression = expressionstart + ' )'

            combinator = ' combinator'

            def _COMMENT(expected, seq, token, tokenizer=None):
                "special implementation for comment token"
                append(seq, cssutils.css.CSSComment([token]), 'COMMENT', token=token)
                return expected

            def _S(expected, seq, token, tokenizer=None):
                # S
                context = new['context'][-1]
                if context.startswith('pseudo-'):
                    if seq and seq[-1].value not in '+-':
                        # e.g. x:func(a + b)
                        append(seq, S, 'S', token=token)
                    return expected

                elif context != 'attrib' and 'combinator' in expected:
                    append(seq, S, 'descendant', token=token)
                    return simple_selector_sequence + combinator

                else:
                    return expected

            def _universal(expected, seq, token, tokenizer=None):
                # *|* or prefix|*
                context = new['context'][-1]
                val = self._tokenvalue(token)
                if 'universal' in expected:
                    append(seq, val, 'universal', token=token)

                    if 'negation' == context:
                        return negationend
                    else:
                        return simple_selector_sequence2 + combinator

                else:
                    new['wellformed'] = False
                    self._log.error('Selector: Unexpected universal.', token=token)
                    return expected

            def _namespace_prefix(expected, seq, token, tokenizer=None):
                # prefix| => element_name
                # or prefix| => attribute_name if attrib
                context = new['context'][-1]
                val = self._tokenvalue(token)
                if 'attrib' == context and 'prefix' in expected:
                    # [PREFIX|att]
                    append(seq, val, '_PREFIX', token=token)
                    return attname2
                elif 'type_selector' in expected:
                    # PREFIX|*
                    append(seq, val, '_PREFIX', token=token)
                    return element_name
                else:
                    new['wellformed'] = False
                    self._log.error(
                        'Selector: Unexpected namespace prefix.', token=token
                    )
                    return expected

            def _pseudo(expected, seq, token, tokenizer=None):
                # pseudo-class or pseudo-element :a ::a :a( ::a(
                """
                /* '::' starts a pseudo-element, ':' a pseudo-class */
                /* Exceptions: :first-line, :first-letter, :before and
                :after. */
                /* Note that pseudo-elements are restricted to one per selector
                and */
                /* occur only in the last simple_selector_sequence. */
                """
                context = new['context'][-1]
                val, typ = self._tokenvalue(token, normalize=True), self._type(token)
                if 'pseudo' in expected:
                    if val in (':first-line', ':first-letter', ':before', ':after'):
                        # always pseudo-element ???
                        typ = 'pseudo-element'
                    append(seq, val, typ, token=token)

                    if val.endswith('('):
                        # function
                        # "pseudo-" "class" or "element"
                        new['context'].append(typ)
                        return expressionstart
                    elif 'negation' == context:
                        return negationend
                    elif 'pseudo-element' == typ:
                        # only one per element, check at ) also!
                        return combinator
                    else:
                        return simple_selector_sequence2 + combinator

                else:
                    new['wellformed'] = False
                    self._log.error(
                        'Selector: Unexpected start of pseudo.', token=token
                    )
                    return expected

            def _expression(expected, seq, token, tokenizer=None):
                # [ [ PLUS | '-' | DIMENSION | NUMBER | STRING | IDENT ] S* ]+
                context = new['context'][-1]
                val, typ = self._tokenvalue(token), self._type(token)
                if context.startswith('pseudo-'):
                    append(seq, val, typ, token=token)
                    return expression
                else:
                    new['wellformed'] = False
                    self._log.error('Selector: Unexpected %s.' % typ, token=token)
                    return expected

            def _attcombinator(expected, seq, token, tokenizer=None):
                # context: attrib
                # PREFIXMATCH | SUFFIXMATCH | SUBSTRINGMATCH | INCLUDES |
                # DASHMATCH
                context = new['context'][-1]
                val, typ = self._tokenvalue(token), self._type(token)
                if 'attrib' == context and 'combinator' in expected:
                    # combinator in attrib
                    append(seq, val, typ.lower(), token=token)
                    return attvalue
                else:
                    new['wellformed'] = False
                    self._log.error('Selector: Unexpected %s.' % typ, token=token)
                    return expected

            def _string(expected, seq, token, tokenizer=None):
                # identifier
                context = new['context'][-1]
                typ, val = self._type(token), self._stringtokenvalue(token)

                # context: attrib
                if 'attrib' == context and 'value' in expected:
                    # attrib: [...=VALUE]
                    append(seq, val, typ, token=token)
                    return attend

                # context: pseudo
                elif context.startswith('pseudo-'):
                    # :func(...)
                    append(seq, val, typ, token=token)
                    return expression

                else:
                    new['wellformed'] = False
                    self._log.error('Selector: Unexpected STRING.', token=token)
                    return expected

            def _ident(expected, seq, token, tokenizer=None):
                # identifier
                context = new['context'][-1]
                val, typ = self._tokenvalue(token), self._type(token)

                # context: attrib
                if 'attrib' == context and 'attribute' in expected:
                    # attrib: [...|ATT...]
                    append(seq, val, 'attribute-selector', token=token)
                    return attcombinator

                elif 'attrib' == context and 'value' in expected:
                    # attrib: [...=VALUE]
                    append(seq, val, 'attribute-value', token=token)
                    return attend

                # context: negation
                elif 'negation' == context:
                    # negation: (prefix|IDENT)
                    append(seq, val, 'negation-type-selector', token=token)
                    return negationend

                # context: pseudo
                elif context.startswith('pseudo-'):
                    # :func(...)
                    append(seq, val, typ, token=token)
                    return expression

                elif 'type_selector' in expected or element_name == expected:
                    # element name after ns or complete type_selector
                    append(seq, val, 'type-selector', token=token)
                    return simple_selector_sequence2 + combinator

                else:
                    new['wellformed'] = False
                    self._log.error('Selector: Unexpected IDENT.', token=token)
                    return expected

            def _class(expected, seq, token, tokenizer=None):
                # .IDENT
                context = new['context'][-1]
                val = self._tokenvalue(token)
                if 'class' in expected:
                    append(seq, val, 'class', token=token)

                    if 'negation' == context:
                        return negationend
                    else:
                        return simple_selector_sequence2 + combinator

                else:
                    new['wellformed'] = False
                    self._log.error('Selector: Unexpected class.', token=token)
                    return expected

            def _hash(expected, seq, token, tokenizer=None):
                # #IDENT
                context = new['context'][-1]
                val = self._tokenvalue(token)
                if 'HASH' in expected:
                    append(seq, val, 'id', token=token)

                    if 'negation' == context:
                        return negationend
                    else:
                        return simple_selector_sequence2 + combinator

                else:
                    new['wellformed'] = False
                    self._log.error('Selector: Unexpected HASH.', token=token)
                    return expected

            def _char(expected, seq, token, tokenizer=None):
                # + > ~ ) [ ] + -
                context = new['context'][-1]
                val = self._tokenvalue(token)

                # context: attrib
                if ']' == val and 'attrib' == context and ']' in expected:
                    # end of attrib
                    append(seq, val, 'attribute-end', token=token)
                    context = new['context'].pop()  # attrib is done
                    context = new['context'][-1]
                    if 'negation' == context:
                        return negationend
                    else:
                        return simple_selector_sequence2 + combinator

                elif '=' == val and 'attrib' == context and 'combinator' in expected:
                    # combinator in attrib
                    append(seq, val, 'equals', token=token)
                    return attvalue

                # context: negation
                elif ')' == val and 'negation' == context and ')' in expected:
                    # not(negation_arg)"
                    append(seq, val, 'negation-end', token=token)
                    new['context'].pop()  # negation is done
                    context = new['context'][-1]
                    return simple_selector_sequence + combinator

                # context: pseudo (at least one expression)
                elif val in '+-' and context.startswith('pseudo-'):
                    # :func(+ -)"
                    _names = {'+': 'plus', '-': 'minus'}
                    if val == '+' and seq and seq[-1].value == S:
                        seq.replace(-1, val, _names[val])
                    else:
                        append(seq, val, _names[val], token=token)
                    return expression

                elif (
                    ')' == val
                    and context.startswith('pseudo-')
                    and expression == expected
                ):
                    # :func(expression)"
                    append(seq, val, 'function-end', token=token)
                    new['context'].pop()  # pseudo is done
                    if 'pseudo-element' == context:
                        return combinator
                    else:
                        return simple_selector_sequence + combinator

                # context: ROOT
                elif '[' == val and 'attrib' in expected:
                    # start of [attrib]
                    append(seq, val, 'attribute-start', token=token)
                    new['context'].append('attrib')
                    return attname

                elif val in '+>~' and 'combinator' in expected:
                    # no other combinator except S may be following
                    _names = {
                        '>': 'child',
                        '+': 'adjacent-sibling',
                        '~': 'following-sibling',
                    }
                    if seq and seq[-1].value == S:
                        seq.replace(-1, val, _names[val])
                    else:
                        append(seq, val, _names[val], token=token)
                    return simple_selector_sequence

                elif ',' == val:
                    # not a selectorlist
                    new['wellformed'] = False
                    self._log.error(
                        'Selector: Single selector only.',
                        error=xml.dom.InvalidModificationErr,
                        token=token,
                    )
                    return expected

                else:
                    new['wellformed'] = False
                    self._log.error('Selector: Unexpected CHAR.', token=token)
                    return expected

            def _negation(expected, seq, token, tokenizer=None):
                # not(
                val = self._tokenvalue(token, normalize=True)
                if 'negation' in expected:
                    new['context'].append('negation')
                    append(seq, val, 'negation-start', token=token)
                    return negation_arg
                else:
                    new['wellformed'] = False
                    self._log.error('Selector: Unexpected negation.', token=token)
                    return expected

            def _atkeyword(expected, seq, token, tokenizer=None):
                "invalidates selector"
                new['wellformed'] = False
                self._log.error('Selector: Unexpected ATKEYWORD.', token=token)
                return expected

            # expected: only|not or mediatype, mediatype, feature, and
            newseq = self._tempSeq()

            wellformed, expected = self._parse(
                expected=simple_selector_sequence,
                seq=newseq,
                tokenizer=tokenizer,
                productions={
                    'CHAR': _char,
                    'class': _class,
                    'HASH': _hash,
                    'STRING': _string,
                    'IDENT': _ident,
                    'namespace_prefix': _namespace_prefix,
                    'negation': _negation,
                    'pseudo-class': _pseudo,
                    'pseudo-element': _pseudo,
                    'universal': _universal,
                    # pseudo
                    'NUMBER': _expression,
                    'DIMENSION': _expression,
                    # attribute
                    'PREFIXMATCH': _attcombinator,
                    'SUFFIXMATCH': _attcombinator,
                    'SUBSTRINGMATCH': _attcombinator,
                    'DASHMATCH': _attcombinator,
                    'INCLUDES': _attcombinator,
                    'S': _S,
                    'COMMENT': _COMMENT,
                    'ATKEYWORD': _atkeyword,
                },
            )
            wellformed = wellformed and new['wellformed']

            # post condition
            if len(new['context']) > 1 or not newseq:
                wellformed = False
                self._log.error(
                    'Selector: Invalid or incomplete selector: %s'
                    % self._valuestr(selectorText)
                )

            if expected == 'element_name':
                wellformed = False
                self._log.error(
                    'Selector: No element name found: %s' % self._valuestr(selectorText)
                )

            if expected == simple_selector_sequence and newseq:
                wellformed = False
                self._log.error(
                    'Selector: Cannot end with combinator: %s'
                    % self._valuestr(selectorText)
                )

            if (
                newseq
                and hasattr(newseq[-1].value, 'strip')
                and newseq[-1].value.strip() == ''
            ):
                del newseq[-1]

            # set
            if wellformed:
                self.__namespaces = namespaces
                self._element = new['element']
                self._specificity = tuple(new['specificity'])
                self._setSeq(newseq)
                # filter that only used ones are kept
                self.__namespaces = self._getUsedNamespaces()

    selectorText = property(
        _getSelectorText,
        _setSelectorText,
        doc="(DOM) The parsable textual representation of " "the selector.",
    )

    specificity = property(
        lambda self: self._specificity,
        doc="""Specificity of this selector (READONLY).
                Tuple of (a, b, c, d) where:

                a
                    presence of style in document, always 0 if not used on a
                    document
                b
                    number of ID selectors
                c
                    number of .class selectors
                d
                    number of Element (type) selectors""",
    )

    wellformed = property(lambda self: bool(len(self.seq)))

    @Deprecated('Use property parent instead')
    def _getParentList(self):
        return self.parent

    parentList = property(_getParentList, doc="DEPRECATED, see property parent instead")
