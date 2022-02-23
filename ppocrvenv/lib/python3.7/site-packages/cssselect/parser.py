# -*- coding: utf-8 -*-
"""
    cssselect.parser
    ================

    Tokenizer, parser and parsed objects for CSS selectors.


    :copyright: (c) 2007-2012 Ian Bicking and contributors.
                See AUTHORS for more details.
    :license: BSD, see LICENSE for more details.

"""

import sys
import re
import operator


if sys.version_info[0] < 3:
    _unicode = unicode
    _unichr = unichr
else:
    _unicode = str
    _unichr = chr


def ascii_lower(string):
    """Lower-case, but only in the ASCII range."""
    return string.encode('utf8').lower().decode('utf8')


class SelectorError(Exception):
    """Common parent for :class:`SelectorSyntaxError` and
    :class:`ExpressionError`.

    You can just use ``except SelectorError:`` when calling
    :meth:`~GenericTranslator.css_to_xpath` and handle both exceptions types.

    """

class SelectorSyntaxError(SelectorError, SyntaxError):
    """Parsing a selector that does not match the grammar."""


#### Parsed objects

class Selector(object):
    """
    Represents a parsed selector.

    :meth:`~GenericTranslator.selector_to_xpath` accepts this object,
    but ignores :attr:`pseudo_element`. It is the user’s responsibility
    to account for pseudo-elements and reject selectors with unknown
    or unsupported pseudo-elements.

    """
    def __init__(self, tree, pseudo_element=None):
        self.parsed_tree = tree
        if pseudo_element is not None and not isinstance(
                pseudo_element, FunctionalPseudoElement):
            pseudo_element = ascii_lower(pseudo_element)
        #: A :class:`FunctionalPseudoElement`,
        #: or the identifier for the pseudo-element as a string,
        #  or ``None``.
        #:
        #: +-------------------------+----------------+--------------------------------+
        #: |                         | Selector       | Pseudo-element                 |
        #: +=========================+================+================================+
        #: | CSS3 syntax             | ``a::before``  | ``'before'``                   |
        #: +-------------------------+----------------+--------------------------------+
        #: | Older syntax            | ``a:before``   | ``'before'``                   |
        #: +-------------------------+----------------+--------------------------------+
        #: | From the Lists3_ draft, | ``li::marker`` | ``'marker'``                   |
        #: | not in Selectors3       |                |                                |
        #: +-------------------------+----------------+--------------------------------+
        #: | Invalid pseudo-class    | ``li:marker``  | ``None``                       |
        #: +-------------------------+----------------+--------------------------------+
        #: | Functional              | ``a::foo(2)``  | ``FunctionalPseudoElement(…)`` |
        #: +-------------------------+----------------+--------------------------------+
        #:
        #: .. _Lists3: http://www.w3.org/TR/2011/WD-css3-lists-20110524/#marker-pseudoelement
        self.pseudo_element = pseudo_element

    def __repr__(self):
        if isinstance(self.pseudo_element, FunctionalPseudoElement):
            pseudo_element = repr(self.pseudo_element)
        elif self.pseudo_element:
            pseudo_element = '::%s' % self.pseudo_element
        else:
            pseudo_element = ''
        return '%s[%r%s]' % (
            self.__class__.__name__, self.parsed_tree, pseudo_element)

    def canonical(self):
        """Return a CSS representation for this selector (a string)
        """
        if isinstance(self.pseudo_element, FunctionalPseudoElement):
            pseudo_element = '::%s' % self.pseudo_element.canonical()
        elif self.pseudo_element:
            pseudo_element = '::%s' % self.pseudo_element
        else:
            pseudo_element = ''
        res = '%s%s' % (self.parsed_tree.canonical(), pseudo_element)
        if len(res) > 1:
            res = res.lstrip('*')
        return res

    def specificity(self):
        """Return the specificity_ of this selector as a tuple of 3 integers.

        .. _specificity: http://www.w3.org/TR/selectors/#specificity

        """
        a, b, c = self.parsed_tree.specificity()
        if self.pseudo_element:
            c += 1
        return a, b, c


class Class(object):
    """
    Represents selector.class_name
    """
    def __init__(self, selector, class_name):
        self.selector = selector
        self.class_name = class_name

    def __repr__(self):
        return '%s[%r.%s]' % (
            self.__class__.__name__, self.selector, self.class_name)

    def canonical(self):
        return '%s.%s' % (self.selector.canonical(), self.class_name)

    def specificity(self):
        a, b, c = self.selector.specificity()
        b += 1
        return a, b, c


class FunctionalPseudoElement(object):
    """
    Represents selector::name(arguments)

    .. attribute:: name

        The name (identifier) of the pseudo-element, as a string.

    .. attribute:: arguments

        The arguments of the pseudo-element, as a list of tokens.

        **Note:** tokens are not part of the public API,
        and may change between cssselect versions.
        Use at your own risks.

    """
    def __init__(self, name, arguments):
        self.name = ascii_lower(name)
        self.arguments = arguments

    def __repr__(self):
        return '%s[::%s(%r)]' % (
            self.__class__.__name__, self.name,
            [token.value for token in self.arguments])

    def argument_types(self):
        return [token.type for token in self.arguments]

    def canonical(self):
        args = ''.join(token.css() for token in self.arguments)
        return '%s(%s)' % (self.name, args)

    def specificity(self):
        a, b, c = self.selector.specificity()
        b += 1
        return a, b, c


class Function(object):
    """
    Represents selector:name(expr)
    """
    def __init__(self, selector, name, arguments):
        self.selector = selector
        self.name = ascii_lower(name)
        self.arguments = arguments

    def __repr__(self):
        return '%s[%r:%s(%r)]' % (
            self.__class__.__name__, self.selector, self.name,
            [token.value for token in self.arguments])

    def argument_types(self):
        return [token.type for token in self.arguments]

    def canonical(self):
        args = ''.join(token.css() for token in self.arguments)
        return '%s:%s(%s)' % (self.selector.canonical(), self.name, args)

    def specificity(self):
        a, b, c = self.selector.specificity()
        b += 1
        return a, b, c


class Pseudo(object):
    """
    Represents selector:ident
    """
    def __init__(self, selector, ident):
        self.selector = selector
        self.ident = ascii_lower(ident)

    def __repr__(self):
        return '%s[%r:%s]' % (
            self.__class__.__name__, self.selector, self.ident)

    def canonical(self):
        return '%s:%s' % (self.selector.canonical(), self.ident)

    def specificity(self):
        a, b, c = self.selector.specificity()
        b += 1
        return a, b, c


class Negation(object):
    """
    Represents selector:not(subselector)
    """
    def __init__(self, selector, subselector):
        self.selector = selector
        self.subselector = subselector

    def __repr__(self):
        return '%s[%r:not(%r)]' % (
            self.__class__.__name__, self.selector, self.subselector)

    def canonical(self):
        subsel = self.subselector.canonical()
        if len(subsel) > 1:
            subsel = subsel.lstrip('*')
        return '%s:not(%s)' % (self.selector.canonical(), subsel)

    def specificity(self):
        a1, b1, c1 = self.selector.specificity()
        a2, b2, c2 = self.subselector.specificity()
        return a1 + a2, b1 + b2, c1 + c2


class Attrib(object):
    """
    Represents selector[namespace|attrib operator value]
    """
    def __init__(self, selector, namespace, attrib, operator, value):
        self.selector = selector
        self.namespace = namespace
        self.attrib = attrib
        self.operator = operator
        self.value = value

    def __repr__(self):
        if self.namespace:
            attrib = '%s|%s' % (self.namespace, self.attrib)
        else:
            attrib = self.attrib
        if self.operator == 'exists':
            return '%s[%r[%s]]' % (
                self.__class__.__name__, self.selector, attrib)
        else:
            return '%s[%r[%s %s %r]]' % (
                self.__class__.__name__, self.selector, attrib,
                self.operator, self.value.value)

    def canonical(self):
        if self.namespace:
            attrib = '%s|%s' % (self.namespace, self.attrib)
        else:
            attrib = self.attrib

        if self.operator == 'exists':
            op = attrib
        else:
            op = '%s%s%s' % (attrib, self.operator, self.value.css())

        return '%s[%s]' % (self.selector.canonical(), op)

    def specificity(self):
        a, b, c = self.selector.specificity()
        b += 1
        return a, b, c


class Element(object):
    """
    Represents namespace|element

    `None` is for the universal selector '*'

    """
    def __init__(self, namespace=None, element=None):
        self.namespace = namespace
        self.element = element

    def __repr__(self):
        return '%s[%s]' % (self.__class__.__name__, self.canonical())

    def canonical(self):
        element = self.element or '*'
        if self.namespace:
            element = '%s|%s' % (self.namespace, element)
        return element

    def specificity(self):
        if self.element:
            return 0, 0, 1
        else:
            return 0, 0, 0


class Hash(object):
    """
    Represents selector#id
    """
    def __init__(self, selector, id):
        self.selector = selector
        self.id = id

    def __repr__(self):
        return '%s[%r#%s]' % (
            self.__class__.__name__, self.selector, self.id)

    def canonical(self):
        return '%s#%s' % (self.selector.canonical(), self.id)

    def specificity(self):
        a, b, c = self.selector.specificity()
        a += 1
        return a, b, c


class CombinedSelector(object):
    def __init__(self, selector, combinator, subselector):
        assert selector is not None
        self.selector = selector
        self.combinator = combinator
        self.subselector = subselector

    def __repr__(self):
        if self.combinator == ' ':
            comb = '<followed>'
        else:
            comb = self.combinator
        return '%s[%r %s %r]' % (
            self.__class__.__name__, self.selector, comb, self.subselector)

    def canonical(self):
        subsel = self.subselector.canonical()
        if len(subsel) > 1:
            subsel = subsel.lstrip('*')
        return '%s %s %s' % (
            self.selector.canonical(), self.combinator, subsel)

    def specificity(self):
        a1, b1, c1 = self.selector.specificity()
        a2, b2, c2 = self.subselector.specificity()
        return a1 + a2, b1 + b2, c1 + c2


#### Parser

# foo
_el_re = re.compile(r'^[ \t\r\n\f]*([a-zA-Z]+)[ \t\r\n\f]*$')

# foo#bar or #bar
_id_re = re.compile(r'^[ \t\r\n\f]*([a-zA-Z]*)#([a-zA-Z0-9_-]+)[ \t\r\n\f]*$')

# foo.bar or .bar
_class_re = re.compile(
    r'^[ \t\r\n\f]*([a-zA-Z]*)\.([a-zA-Z][a-zA-Z0-9_-]*)[ \t\r\n\f]*$')


def parse(css):
    """Parse a CSS *group of selectors*.

    If you don't care about pseudo-elements or selector specificity,
    you can skip this and use :meth:`~GenericTranslator.css_to_xpath`.

    :param css:
        A *group of selectors* as an Unicode string.
    :raises:
        :class:`SelectorSyntaxError` on invalid selectors.
    :returns:
        A list of parsed :class:`Selector` objects, one for each
        selector in the comma-separated group.

    """
    # Fast path for simple cases
    match = _el_re.match(css)
    if match:
        return [Selector(Element(element=match.group(1)))]
    match = _id_re.match(css)
    if match is not None:
        return [Selector(Hash(Element(element=match.group(1) or None),
                              match.group(2)))]
    match = _class_re.match(css)
    if match is not None:
        return [Selector(Class(Element(element=match.group(1) or None),
                               match.group(2)))]

    stream = TokenStream(tokenize(css))
    stream.source = css
    return list(parse_selector_group(stream))
#    except SelectorSyntaxError:
#        e = sys.exc_info()[1]
#        message = "%s at %s -> %r" % (
#            e, stream.used, stream.peek())
#        e.msg = message
#        e.args = tuple([message])
#        raise


def parse_selector_group(stream):
    stream.skip_whitespace()
    while 1:
        yield Selector(*parse_selector(stream))
        if stream.peek() == ('DELIM', ','):
            stream.next()
            stream.skip_whitespace()
        else:
            break

def parse_selector(stream):
    result, pseudo_element = parse_simple_selector(stream)
    while 1:
        stream.skip_whitespace()
        peek = stream.peek()
        if peek in (('EOF', None), ('DELIM', ',')):
            break
        if pseudo_element:
            raise SelectorSyntaxError(
                'Got pseudo-element ::%s not at the end of a selector'
                % pseudo_element)
        if peek.is_delim('+', '>', '~'):
            # A combinator
            combinator = stream.next().value
            stream.skip_whitespace()
        else:
            # By exclusion, the last parse_simple_selector() ended
            # at peek == ' '
            combinator = ' '
        next_selector, pseudo_element = parse_simple_selector(stream)
        result = CombinedSelector(result, combinator, next_selector)
    return result, pseudo_element


def parse_simple_selector(stream, inside_negation=False):
    stream.skip_whitespace()
    selector_start = len(stream.used)
    peek = stream.peek()
    if peek.type == 'IDENT' or peek == ('DELIM', '*'):
        if peek.type == 'IDENT':
            namespace = stream.next().value
        else:
            stream.next()
            namespace = None
        if stream.peek() == ('DELIM', '|'):
            stream.next()
            element = stream.next_ident_or_star()
        else:
            element = namespace
            namespace = None
    else:
        element = namespace = None
    result = Element(namespace, element)
    pseudo_element = None
    while 1:
        peek = stream.peek()
        if peek.type in ('S', 'EOF') or peek.is_delim(',', '+', '>', '~') or (
                inside_negation and peek == ('DELIM', ')')):
            break
        if pseudo_element:
            raise SelectorSyntaxError(
                'Got pseudo-element ::%s not at the end of a selector'
                % pseudo_element)
        if peek.type == 'HASH':
            result = Hash(result, stream.next().value)
        elif peek == ('DELIM', '.'):
            stream.next()
            result = Class(result, stream.next_ident())
        elif peek == ('DELIM', '|'):
            stream.next()
            result = Element(None, stream.next_ident())
        elif peek == ('DELIM', '['):
            stream.next()
            result = parse_attrib(result, stream)
        elif peek == ('DELIM', ':'):
            stream.next()
            if stream.peek() == ('DELIM', ':'):
                stream.next()
                pseudo_element = stream.next_ident()
                if stream.peek() == ('DELIM', '('):
                    stream.next()
                    pseudo_element = FunctionalPseudoElement(
                        pseudo_element, parse_arguments(stream))
                continue
            ident = stream.next_ident()
            if ident.lower() in ('first-line', 'first-letter',
                                 'before', 'after'):
                # Special case: CSS 2.1 pseudo-elements can have a single ':'
                # Any new pseudo-element must have two.
                pseudo_element = _unicode(ident)
                continue
            if stream.peek() != ('DELIM', '('):
                result = Pseudo(result, ident)
                if result.__repr__() == 'Pseudo[Element[*]:scope]':
                    if not (len(stream.used) == 2 or
                            (len(stream.used) == 3
                             and stream.used[0].type == 'S')):
                        raise SelectorSyntaxError(
                            'Got immediate child pseudo-element ":scope" '
                            'not at the start of a selector')
                continue
            stream.next()
            stream.skip_whitespace()
            if ident.lower() == 'not':
                if inside_negation:
                    raise SelectorSyntaxError('Got nested :not()')
                argument, argument_pseudo_element = parse_simple_selector(
                    stream, inside_negation=True)
                next = stream.next()
                if argument_pseudo_element:
                    raise SelectorSyntaxError(
                        'Got pseudo-element ::%s inside :not() at %s'
                        % (argument_pseudo_element, next.pos))
                if next != ('DELIM', ')'):
                    raise SelectorSyntaxError("Expected ')', got %s" % (next,))
                result = Negation(result, argument)
            else:
                result = Function(result, ident, parse_arguments(stream))
        else:
            raise SelectorSyntaxError(
                "Expected selector, got %s" % (peek,))
    if len(stream.used) == selector_start:
        raise SelectorSyntaxError(
            "Expected selector, got %s" % (stream.peek(),))
    return result, pseudo_element


def parse_arguments(stream):
    arguments = []
    while 1:
        stream.skip_whitespace()
        next = stream.next()
        if next.type in ('IDENT', 'STRING', 'NUMBER') or next in [
                ('DELIM', '+'), ('DELIM', '-')]:
            arguments.append(next)
        elif next == ('DELIM', ')'):
            return arguments
        else:
            raise SelectorSyntaxError(
                "Expected an argument, got %s" % (next,))


def parse_attrib(selector, stream):
    stream.skip_whitespace()
    attrib = stream.next_ident_or_star()
    if attrib is None and stream.peek() != ('DELIM', '|'):
        raise SelectorSyntaxError(
            "Expected '|', got %s" % (stream.peek(),))
    if stream.peek() == ('DELIM', '|'):
        stream.next()
        if stream.peek() == ('DELIM', '='):
            namespace = None
            stream.next()
            op = '|='
        else:
            namespace = attrib
            attrib = stream.next_ident()
            op = None
    else:
        namespace = op = None
    if op is None:
        stream.skip_whitespace()
        next = stream.next()
        if next == ('DELIM', ']'):
            return Attrib(selector, namespace, attrib, 'exists', None)
        elif next == ('DELIM', '='):
            op = '='
        elif next.is_delim('^', '$', '*', '~', '|', '!') and (
                stream.peek() == ('DELIM', '=')):
            op = next.value + '='
            stream.next()
        else:
            raise SelectorSyntaxError(
                "Operator expected, got %s" % (next,))
    stream.skip_whitespace()
    value = stream.next()
    if value.type not in ('IDENT', 'STRING'):
        raise SelectorSyntaxError(
            "Expected string or ident, got %s" % (value,))
    stream.skip_whitespace()
    next = stream.next()
    if next != ('DELIM', ']'):
        raise SelectorSyntaxError(
            "Expected ']', got %s" % (next,))
    return Attrib(selector, namespace, attrib, op, value)


def parse_series(tokens):
    """
    Parses the arguments for :nth-child() and friends.

    :raises: A list of tokens
    :returns: :``(a, b)``

    """
    for token in tokens:
        if token.type == 'STRING':
            raise ValueError('String tokens not allowed in series.')
    s = ''.join(token.value for token in tokens).strip()
    if s == 'odd':
        return 2, 1
    elif s == 'even':
        return 2, 0
    elif s == 'n':
        return 1, 0
    if 'n' not in s:
        # Just b
        return 0, int(s)
    a, b = s.split('n', 1)
    if not a:
        a = 1
    elif a == '-' or a == '+':
        a = int(a+'1')
    else:
        a = int(a)
    if not b:
        b = 0
    else:
        b = int(b)
    return a, b


#### Token objects

class Token(tuple):
    def __new__(cls, type_, value, pos):
        obj = tuple.__new__(cls, (type_, value))
        obj.pos = pos
        return obj

    def __repr__(self):
        return "<%s '%s' at %i>" % (self.type, self.value, self.pos)

    def is_delim(self, *values):
        return self.type == 'DELIM' and self.value in values

    type = property(operator.itemgetter(0))
    value = property(operator.itemgetter(1))

    def css(self):
        if self.type == 'STRING':
            return repr(self.value)
        else:
            return self.value


class EOFToken(Token):
    def __new__(cls, pos):
        return Token.__new__(cls, 'EOF', None, pos)

    def __repr__(self):
        return '<%s at %i>' % (self.type, self.pos)


#### Tokenizer


class TokenMacros:
    unicode_escape = r'\\([0-9a-f]{1,6})(?:\r\n|[ \n\r\t\f])?'
    escape = unicode_escape + r'|\\[^\n\r\f0-9a-f]'
    string_escape = r'\\(?:\n|\r\n|\r|\f)|' + escape
    nonascii = r'[^\0-\177]'
    nmchar = '[_a-z0-9-]|%s|%s' % (escape, nonascii)
    nmstart = '[_a-z]|%s|%s' % (escape, nonascii)

def _compile(pattern):
    return re.compile(pattern % vars(TokenMacros), re.IGNORECASE).match

_match_whitespace = _compile(r'[ \t\r\n\f]+')
_match_number = _compile(r'[+-]?(?:[0-9]*\.[0-9]+|[0-9]+)')
_match_hash = _compile('#(?:%(nmchar)s)+')
_match_ident = _compile('-?(?:%(nmstart)s)(?:%(nmchar)s)*')
_match_string_by_quote = {
    "'": _compile(r"([^\n\r\f\\']|%(string_escape)s)*"),
    '"': _compile(r'([^\n\r\f\\"]|%(string_escape)s)*'),
}

_sub_simple_escape = re.compile(r'\\(.)').sub
_sub_unicode_escape = re.compile(TokenMacros.unicode_escape, re.I).sub
_sub_newline_escape =re.compile(r'\\(?:\n|\r\n|\r|\f)').sub

# Same as r'\1', but faster on CPython
_replace_simple = operator.methodcaller('group', 1)

def _replace_unicode(match):
    codepoint = int(match.group(1), 16)
    if codepoint > sys.maxunicode:
        codepoint = 0xFFFD
    return _unichr(codepoint)


def unescape_ident(value):
    value = _sub_unicode_escape(_replace_unicode, value)
    value = _sub_simple_escape(_replace_simple, value)
    return value


def tokenize(s):
    pos = 0
    len_s = len(s)
    while pos < len_s:
        match = _match_whitespace(s, pos=pos)
        if match:
            yield Token('S', ' ', pos)
            pos = match.end()
            continue

        match = _match_ident(s, pos=pos)
        if match:
            value = _sub_simple_escape(_replace_simple,
                    _sub_unicode_escape(_replace_unicode, match.group()))
            yield Token('IDENT', value, pos)
            pos = match.end()
            continue

        match = _match_hash(s, pos=pos)
        if match:
            value = _sub_simple_escape(_replace_simple,
                    _sub_unicode_escape(_replace_unicode, match.group()[1:]))
            yield Token('HASH', value, pos)
            pos = match.end()
            continue

        quote = s[pos]
        if quote in _match_string_by_quote:
            match = _match_string_by_quote[quote](s, pos=pos + 1)
            assert match, 'Should have found at least an empty match'
            end_pos = match.end()
            if end_pos == len_s:
                raise SelectorSyntaxError('Unclosed string at %s' % pos)
            if s[end_pos] != quote:
                raise SelectorSyntaxError('Invalid string at %s' % pos)
            value = _sub_simple_escape(_replace_simple,
                    _sub_unicode_escape(_replace_unicode,
                    _sub_newline_escape('', match.group())))
            yield Token('STRING', value, pos)
            pos = end_pos + 1
            continue

        match = _match_number(s, pos=pos)
        if match:
            value = match.group()
            yield Token('NUMBER', value, pos)
            pos = match.end()
            continue

        pos2 = pos + 2
        if s[pos:pos2] == '/*':
            pos = s.find('*/', pos2)
            if pos == -1:
                pos = len_s
            else:
                pos += 2
            continue

        yield Token('DELIM', s[pos], pos)
        pos += 1

    assert pos == len_s
    yield EOFToken(pos)


class TokenStream(object):
    def __init__(self, tokens, source=None):
        self.used = []
        self.tokens = iter(tokens)
        self.source = source
        self.peeked = None
        self._peeking = False
        try:
            self.next_token = self.tokens.next
        except AttributeError:
            # Python 3
            self.next_token = self.tokens.__next__

    def next(self):
        if self._peeking:
            self._peeking = False
            self.used.append(self.peeked)
            return self.peeked
        else:
            next = self.next_token()
            self.used.append(next)
            return next

    def peek(self):
        if not self._peeking:
            self.peeked = self.next_token()
            self._peeking = True
        return self.peeked

    def next_ident(self):
        next = self.next()
        if next.type != 'IDENT':
            raise SelectorSyntaxError('Expected ident, got %s' % (next,))
        return next.value

    def next_ident_or_star(self):
        next = self.next()
        if next.type == 'IDENT':
            return next.value
        elif next == ('DELIM', '*'):
            return None
        else:
            raise SelectorSyntaxError(
                "Expected ident or '*', got %s" % (next,))

    def skip_whitespace(self):
        peek = self.peek()
        if peek.type == 'S':
            self.next()
