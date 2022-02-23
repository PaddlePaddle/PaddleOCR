"""base classes and helper functions for css and stylesheets packages
"""
__all__ = []

from .helper import normalize
from itertools import chain
import cssutils
from . import codec
import codecs
from . import errorhandler
from . import tokenize2
import xml.dom
import re

try:
    from ._fetchgae import _defaultFetcher
except ImportError:
    from ._fetch import _defaultFetcher

log = errorhandler.ErrorHandler()


class _BaseClass(object):
    """
    Base class for Base, Base2 and _NewBase.

    **Base and Base2 will be removed in the future!**
    """

    _log = errorhandler.ErrorHandler()
    _prods = tokenize2.CSSProductions

    def _checkReadonly(self):
        "Raise xml.dom.NoModificationAllowedErr if rule/... is readonly"
        if hasattr(self, '_readonly') and self._readonly:
            raise xml.dom.NoModificationAllowedErr('%s is readonly.' % self.__class__)
            return True
        return False

    def _valuestr(self, t):
        """
        Return string value of t (t may be a string, a list of token tuples
        or a single tuple in format (type, value, line, col).
        Mainly used to get a string value of t for error messages.
        """
        if not t:
            return ''
        elif isinstance(t, str):
            return t
        else:
            return ''.join([x[1] for x in t])


class _NewBase(_BaseClass):
    """
    New base class for classes using ProdParser.

    **Currently CSSValue and related ones only.**
    """

    def __init__(self):
        self._seq = Seq()

    def _setSeq(self, newseq):
        """Set value of ``seq`` which is readonly."""
        newseq._readonly = True
        self._seq = newseq

    def _clearSeq(self):
        self._seq.clear()

    def _tempSeq(self, readonly=False):
        "Get a writeable Seq() which is used to set ``seq`` later"
        return Seq(readonly=readonly)

    seq = property(
        lambda self: self._seq, doc="Internal readonly attribute, **DO NOT USE**!"
    )


class _NewListBase(_NewBase):
    """
    (EXPERIMENTAL)
    A base class used for list classes like stylesheets.MediaList

    adds list like behaviour running on inhering class' property ``seq``

    - item in x => bool
    - len(x) => integer
    - get, set and del x[i]
    - for item in x
    - append(item)

    some methods must be overwritten in inheriting class
    """

    def __init__(self):
        self._seq = Seq()

    def __contains__(self, item):
        for it in self._seq:
            if item == it.value:
                return True
        return False

    def __delitem__(self, index):
        del self._seq[index]

    def __getitem__(self, index):
        return self._seq[index].value

    def __iter__(self):
        def gen():
            for x in self._seq:
                yield x.value

        return gen()

    def __len__(self):
        return len(self._seq)

    def __setitem__(self, index, item):
        "must be overwritten"
        raise NotImplementedError

    def append(self, item):
        "must be overwritten"
        raise NotImplementedError


class Base(_BaseClass):
    """
    **Superceded by _NewBase**

    **Superceded by Base2 which is used for new seq handling class.**

    Base class for most CSS and StyleSheets classes

    Contains helper methods for inheriting classes helping parsing

    ``_normalize`` is static as used by Preferences.
    """

    __tokenizer2 = tokenize2.Tokenizer()

    # for more on shorthand properties see
    # http://www.dustindiaz.com/css-shorthand/
    # format: shorthand: [(propname, mandatorycheck?)*]
    _SHORTHANDPROPERTIES = {
        'background': [],
        # u'background-position': [], # list of 2 values!
        'border': [],
        'border-left': [],
        'border-right': [],
        'border-top': [],
        'border-bottom': [],
        # u'border-color': [], # list or single but same values
        # u'border-style': [], # list or single but same values
        # u'border-width': [], # list or single but same values
        'cue': [],
        'font': [],
        'list-style': [],
        # u'margin': [], # list or single but same values
        'outline': [],
        # u'padding': [], # list or single but same values
        'pause': [],
    }

    @staticmethod
    def _normalize(x):
        r"""
        normalizes x, namely:

        - remove any \ before non unicode sequences (0-9a-zA-Z) so for
          x=="c\olor\" return "color" (unicode escape sequences should have
          been resolved by the tokenizer already)
        - lowercase
        """
        return normalize(x)

    def _splitNamespacesOff(self, text_namespaces_tuple):
        """
        returns tuple (text, dict-of-namespaces) or if no namespaces are
        in cssText returns (cssText, {})

        used in Selector, SelectorList, CSSStyleRule, CSSMediaRule and
        CSSStyleSheet
        """
        if isinstance(text_namespaces_tuple, tuple):
            return text_namespaces_tuple[0], _SimpleNamespaces(
                self._log, text_namespaces_tuple[1]
            )
        else:
            return text_namespaces_tuple, _SimpleNamespaces(log=self._log)

    def _tokenize2(self, textortokens):
        """
        returns tokens of textortokens which may already be tokens in which
        case simply returns input
        """
        if not textortokens:
            return None
        elif isinstance(textortokens, str):
            # needs to be tokenized
            return self.__tokenizer2.tokenize(textortokens)
        elif isinstance(textortokens, tuple):
            # a single token (like a comment)
            return [textortokens]
        else:
            # already tokenized but return an iterator
            return iter(textortokens)

    def _nexttoken(self, tokenizer, default=None):
        "returns next token in generator tokenizer or the default value"
        try:
            return next(tokenizer)
        # TypeError for py3
        except (StopIteration, AttributeError, TypeError):
            return default

    def _type(self, token):
        "returns type of Tokenizer token"
        if token:
            return token[0]
        else:
            return None

    def _tokenvalue(self, token, normalize=False):
        "returns value of Tokenizer token"
        if token and normalize:
            return Base._normalize(token[1])
        elif token:
            return token[1]
        else:
            return None

    def _stringtokenvalue(self, token):
        """
        for STRING returns the actual content without surrounding "" or ''
        and without respective escapes, e.g.::

             "with \" char" => with " char
        """
        if token:
            value = token[1]
            return value.replace('\\' + value[0], value[0])[1:-1]
        else:
            return None

    def _uritokenvalue(self, token):
        """
        for URI returns the actual content without surrounding url()
        or url(""), url('') and without respective escapes, e.g.::

             url("\"") => "
        """
        if token:
            value = token[1][4:-1].strip()
            if value and (value[0] in '\'"') and (value[0] == value[-1]):
                # a string "..." or '...'
                value = value.replace('\\' + value[0], value[0])[1:-1]
            return value
        else:
            return None

    def _tokensupto2(  # noqa: C901
        self,
        tokenizer,
        starttoken=None,
        blockstartonly=False,  # {
        blockendonly=False,  # }
        mediaendonly=False,
        importmediaqueryendonly=False,  # ; or STRING
        mediaqueryendonly=False,  # { or STRING
        semicolon=False,  # ;
        propertynameendonly=False,  # :
        propertyvalueendonly=False,  # ! ; }
        propertypriorityendonly=False,  # ; }
        selectorattendonly=False,  # ]
        funcendonly=False,  # )
        listseponly=False,  # ,
        separateEnd=False,  # returns (resulttokens, endtoken)
    ):
        """
        returns tokens upto end of atrule and end index
        end is defined by parameters, might be ; } ) or other

        default looks for ending "}" and ";"
        """
        ends = ';}'
        endtypes = ()
        brace = bracket = parant = 0  # {}, [], ()

        if blockstartonly:  # {
            ends = '{'
            brace = -1  # set to 0 with first {
        elif blockendonly:  # }
            ends = '}'
            brace = 1
        elif mediaendonly:  # }
            ends = '}'
            brace = 1  # rules } and mediarules }
        elif importmediaqueryendonly:
            # end of mediaquery which may be ; or STRING
            ends = ';'
            endtypes = ('STRING',)
        elif mediaqueryendonly:
            # end of mediaquery which may be { or STRING
            # special case, see below
            ends = '{'
            brace = -1  # set to 0 with first {
            endtypes = ('STRING',)
        elif semicolon:
            ends = ';'
        elif propertynameendonly:  # : and ; in case of an error
            ends = ':;'
        elif propertyvalueendonly:  # ; or !important
            ends = ';!'
        elif propertypriorityendonly:  # ;
            ends = ';'
        elif selectorattendonly:  # ]
            ends = ']'
            if starttoken and self._tokenvalue(starttoken) == '[':
                bracket = 1
        elif funcendonly:  # )
            ends = ')'
            parant = 1
        elif listseponly:  # ,
            ends = ','

        resulttokens = []
        if starttoken:
            resulttokens.append(starttoken)
            val = starttoken[1]
            if '[' == val:
                bracket += 1
            elif '{' == val:
                brace += 1
            elif '(' == val:
                parant += 1

        if tokenizer:
            for token in tokenizer:
                typ, val, line, col = token
                if 'EOF' == typ:
                    resulttokens.append(token)
                    break

                if '{' == val:
                    brace += 1
                elif '}' == val:
                    brace -= 1
                elif '[' == val:
                    bracket += 1
                elif ']' == val:
                    bracket -= 1
                # function( or single (
                elif '(' == val or Base._prods.FUNCTION == typ:
                    parant += 1
                elif ')' == val:
                    parant -= 1

                resulttokens.append(token)

                if (brace == bracket == parant == 0) and (
                    val in ends or typ in endtypes
                ):
                    break
                elif (
                    mediaqueryendonly
                    and brace == -1
                    and (bracket == parant == 0)
                    and typ in endtypes
                ):
                    # mediaqueryendonly with STRING
                    break
        if separateEnd:
            # TODO: use this method as generator, then this makes sense
            if resulttokens:
                return resulttokens[:-1], resulttokens[-1]
            else:
                return resulttokens, None
        else:
            return resulttokens

    def _adddefaultproductions(self, productions, new=None):
        """
        adds default productions if not already present, used by
        _parse only

        each production should return the next expected token
        normaly a name like "uri" or "EOF"
        some have no expectation like S or COMMENT, so simply return
        the current value of self.__expected
        """

        def ATKEYWORD(expected, seq, token, tokenizer=None):
            "default impl for unexpected @rule"
            if expected != 'EOF':
                # TODO: parentStyleSheet=self
                rule = cssutils.css.CSSUnknownRule()
                rule.cssText = self._tokensupto2(tokenizer, token)
                if rule.wellformed:
                    seq.append(rule)
                return expected
            else:
                new['wellformed'] = False
                self._log.error('Expected EOF.', token=token)
                return expected

        def COMMENT(expected, seq, token, tokenizer=None):
            "default implementation for COMMENT token adds CSSCommentRule"
            seq.append(cssutils.css.CSSComment([token]))
            return expected

        def S(expected, seq, token, tokenizer=None):
            "default implementation for S token, does nothing"
            return expected

        def EOF(expected=None, seq=None, token=None, tokenizer=None):
            "default implementation for EOF token"
            return 'EOF'

        p = {
            'ATKEYWORD': ATKEYWORD,
            'COMMENT': COMMENT,
            'S': S,
            'EOF': EOF,  # only available if fullsheet
        }
        p.update(productions)
        return p

    def _parse(
        self,
        expected,
        seq,
        tokenizer,
        productions,
        default=None,
        new=None,
        initialtoken=None,
    ):
        """
        puts parsed tokens in seq by calling a production with
            (seq, tokenizer, token)

        expected
            a name what token or value is expected next, e.g. 'uri'
        seq
            to add rules etc to
        tokenizer
            call tokenizer.next() to get next token
        productions
            callbacks {tokentype: callback}
        default
            default callback if tokentype not in productions
        new
            used to init default productions
        initialtoken
            will be used together with tokenizer running 1st this token
            and then all tokens in tokenizer

        returns (wellformed, expected) which the last prod might have set
        """
        wellformed = True

        if initialtoken:
            # add initialtoken to tokenizer
            def tokens():
                "Build new tokenizer including initialtoken"
                yield initialtoken
                for item in tokenizer:
                    yield item

            fulltokenizer = chain([initialtoken], tokenizer)
        else:
            fulltokenizer = tokenizer

        if fulltokenizer:
            prods = self._adddefaultproductions(productions, new)
            for token in fulltokenizer:
                p = prods.get(token[0], default)
                if p:
                    expected = p(expected, seq, token, tokenizer)
                else:
                    wellformed = False
                    self._log.error('Unexpected token (%s, %s, %s, %s)' % token)
        return wellformed, expected


class Base2(Base, _NewBase):
    """
    **Superceded by _NewBase.**

    Base class for new seq handling.
    """

    def __init__(self):
        self._seq = Seq()

    def _adddefaultproductions(self, productions, new=None):
        """
        adds default productions if not already present, used by
        _parse only

        each production should return the next expected token
        normaly a name like "uri" or "EOF"
        some have no expectation like S or COMMENT, so simply return
        the current value of self.__expected
        """

        def ATKEYWORD(expected, seq, token, tokenizer=None):
            "default impl for unexpected @rule"
            if expected != 'EOF':
                # TODO: parentStyleSheet=self
                rule = cssutils.css.CSSUnknownRule()
                rule.cssText = self._tokensupto2(tokenizer, token)
                if rule.wellformed:
                    seq.append(
                        rule,
                        cssutils.css.CSSRule.UNKNOWN_RULE,
                        line=token[2],
                        col=token[3],
                    )
                return expected
            else:
                new['wellformed'] = False
                self._log.error('Expected EOF.', token=token)
                return expected

        def COMMENT(expected, seq, token, tokenizer=None):
            "default impl, adds CSSCommentRule if not token == EOF"
            if expected == 'EOF':
                new['wellformed'] = False
                self._log.error('Expected EOF but found comment.', token=token)
            seq.append(cssutils.css.CSSComment([token]), 'COMMENT')
            return expected

        def S(expected, seq, token, tokenizer=None):
            "default impl, does nothing if not token == EOF"
            if expected == 'EOF':
                new['wellformed'] = False
                self._log.error('Expected EOF but found whitespace.', token=token)
            return expected

        def EOF(expected=None, seq=None, token=None, tokenizer=None):
            "default implementation for EOF token"
            return 'EOF'

        defaultproductions = {
            'ATKEYWORD': ATKEYWORD,
            'COMMENT': COMMENT,
            'S': S,
            'EOF': EOF,  # only available if fullsheet
        }
        defaultproductions.update(productions)
        return defaultproductions


class Seq(object):
    """
    property seq of Base2 inheriting classes, holds a list of Item objects.

    used only by Selector for now

    is normally readonly, only writable during parsing
    """

    def __init__(self, readonly=True):
        """
        only way to write to a Seq is to initialize it with new items
        each itemtuple has (value, type, line) where line is optional
        """
        self._seq = []
        self._readonly = readonly

    def __repr__(self):
        "returns a repr same as a list of tuples of (value, type)"
        return 'cssutils.%s.%s([\n    %s], readonly=%r)' % (
            self.__module__,
            self.__class__.__name__,
            ',\n    '.join(['%r' % item for item in self._seq]),
            self._readonly,
        )

    def __str__(self):
        vals = []
        for v in self:
            if isinstance(v.value, str):
                vals.append(v.value)
            elif isinstance(v, tuple):
                vals.append(v.value[1])
            else:
                vals.append(repr(v))

        return "<cssutils.%s.%s object length=%r items=%r readonly=%r at 0x%x>" % (
            self.__module__,
            self.__class__.__name__,
            len(self),
            vals,
            self._readonly,
            id(self),
        )

    def __delitem__(self, i):
        del self._seq[i]

    def __getitem__(self, i):
        return self._seq[i]

    def __setitem__(self, i, xxx_todo_changeme):
        (val, typ, line, col) = xxx_todo_changeme
        self._seq[i] = Item(val, typ, line, col)

    def __iter__(self):
        return iter(self._seq)

    def __len__(self):
        return len(self._seq)

    def append(self, val, typ=None, line=None, col=None):
        "If not readonly add new Item()"
        if self._readonly:
            raise AttributeError('Seq is readonly.')
        else:
            if isinstance(val, Item):
                self._seq.append(val)
            else:
                self._seq.append(Item(val, typ, line, col))

    def appendItem(self, item):
        "if not readonly add item which must be an Item"
        if self._readonly:
            raise AttributeError('Seq is readonly.')
        else:
            self._seq.append(item)

    def clear(self):
        del self._seq[:]

    def insert(self, index, val, typ, line=None, col=None):
        "Insert new Item() at index # even if readony!? TODO!"
        self._seq.insert(index, Item(val, typ, line, col))

    def replace(self, index=-1, val=None, typ=None, line=None, col=None):
        """
        if not readonly replace Item at index with new Item or
        simply replace value or type
        """
        if self._readonly:
            raise AttributeError('Seq is readonly.')
        else:
            self._seq[index] = Item(val, typ, line, col)

    def rstrip(self):
        "trims S items from end of Seq"
        while self._seq and self._seq[-1].type == tokenize2.CSSProductions.S:
            # TODO: removed S before CSSComment /**/ /**/
            del self._seq[-1]

    def appendToVal(self, val=None, index=-1):
        """
        if not readonly append to Item's value at index
        """
        if self._readonly:
            raise AttributeError('Seq is readonly.')
        else:
            old = self._seq[index]
            self._seq[index] = Item(old.value + val, old.type, old.line, old.col)


class Item(object):
    """
    an item in the seq list of classes (successor to tuple items in old seq)

    each item has attributes:

    type
        a sematic type like "element", "attribute"
    value
        the actual value which may be a string, number etc or an instance
        of e.g. a CSSComment
    *line*
        **NOT IMPLEMENTED YET, may contain the line in the source later**
    """

    def __init__(self, value, type, line=None, col=None):
        self.__value = value
        self.__type = type
        self.__line = line
        self.__col = col

    type = property(lambda self: self.__type)
    value = property(lambda self: self.__value)
    line = property(lambda self: self.__line)
    col = property(lambda self: self.__col)

    def __repr__(self):
        return "%s.%s(value=%r, type=%r, line=%r, col=%r)" % (
            self.__module__,
            self.__class__.__name__,
            self.__value,
            self.__type,
            self.__line,
            self.__col,
        )


class ListSeq(object):
    """
    (EXPERIMENTAL)
    A base class used for list classes like cssutils.css.SelectorList or
    stylesheets.MediaList

    adds list like behaviour running on inhering class' property ``seq``

    - item in x => bool
    - len(x) => integer
    - get, set and del x[i]
    - for item in x
    - append(item)

    some methods must be overwritten in inheriting class
    """

    def __init__(self):
        self.seq = []  # does not need to use ``Seq`` as simple list only

    def __contains__(self, item):
        return item in self.seq

    def __delitem__(self, index):
        del self.seq[index]

    def __getitem__(self, index):
        return self.seq[index]

    def __iter__(self):
        def gen():
            for x in self.seq:
                yield x

        return gen()

    def __len__(self):
        return len(self.seq)

    def __setitem__(self, index, item):
        "must be overwritten"
        raise NotImplementedError

    def append(self, item):
        "must be overwritten"
        raise NotImplementedError


class _Namespaces(object):
    """
    A dictionary like wrapper for @namespace rules used in a CSSStyleSheet.
    Works on effective namespaces, so e.g. if::

        @namespace p1 "uri";
        @namespace p2 "uri";

    only the second rule is effective and kept.

    namespaces
        a dictionary {prefix: namespaceURI} containing the effective namespaces
        only. These are the latest set in the CSSStyleSheet.
    parentStyleSheet
        the parent CSSStyleSheet
    """

    def __init__(self, parentStyleSheet, log=None, *args):
        "no initial values are set, only the relevant sheet is"
        self.parentStyleSheet = parentStyleSheet
        self._log = log

    def __repr__(self):
        return "%r" % self.namespaces

    def __contains__(self, prefix):
        return prefix in self.namespaces

    def __delitem__(self, prefix):
        """deletes CSSNamespaceRule(s) with rule.prefix == prefix

        prefix '' and None are handled the same
        """
        if not prefix:
            prefix = ''
        delrule = self.__findrule(prefix)
        for i, rule in enumerate(
            filter(lambda r: r.type == r.NAMESPACE_RULE, self.parentStyleSheet.cssRules)
        ):
            if rule == delrule:
                self.parentStyleSheet.deleteRule(i)
                return

        self._log.error('Prefix %s not found.' % prefix, error=xml.dom.NamespaceErr)

    def __getitem__(self, prefix):
        try:
            return self.namespaces[prefix]
        except KeyError:
            self._log.error('Prefix %s not found.' % prefix, error=xml.dom.NamespaceErr)

    def __iter__(self):
        return self.namespaces.__iter__()

    def __len__(self):
        return len(self.namespaces)

    def __setitem__(self, prefix, namespaceURI):
        "replaces prefix or sets new rule, may raise NoModificationAllowedErr"
        if not prefix:
            prefix = ''  # None or ''
        rule = self.__findrule(prefix)
        if not rule:
            self.parentStyleSheet.insertRule(
                cssutils.css.CSSNamespaceRule(prefix=prefix, namespaceURI=namespaceURI),
                inOrder=True,
            )
        else:
            if prefix in self.namespaces:
                rule.namespaceURI = namespaceURI  # raises NoModificationAllowedErr
            if namespaceURI in list(self.namespaces.values()):
                rule.prefix = prefix

    def __findrule(self, prefix):
        # returns namespace rule where prefix == key
        for rule in filter(
            lambda r: r.type == r.NAMESPACE_RULE,
            reversed(self.parentStyleSheet.cssRules),
        ):
            if rule.prefix == prefix:
                return rule

    @property
    def namespaces(self):
        """
        A property holding only effective @namespace rules in
        self.parentStyleSheets.
        """
        namespaces = {}
        for rule in filter(
            lambda r: r.type == r.NAMESPACE_RULE,
            reversed(self.parentStyleSheet.cssRules),
        ):
            if rule.namespaceURI not in list(namespaces.values()):
                namespaces[rule.prefix] = rule.namespaceURI
        return namespaces

    def get(self, prefix, default):
        return self.namespaces.get(prefix, default)

    def items(self):
        return list(self.namespaces.items())

    def keys(self):
        return list(self.namespaces.keys())

    def values(self):
        return list(self.namespaces.values())

    def prefixForNamespaceURI(self, namespaceURI):
        """
        returns effective prefix for given namespaceURI or raises IndexError
        if this cannot be found"""
        for prefix, uri in list(self.namespaces.items()):
            if uri == namespaceURI:
                return prefix
        raise IndexError('NamespaceURI %s not found.' % namespaceURI)

    def __str__(self):
        return "<cssutils.util.%s object parentStyleSheet=%r at 0x%x>" % (
            self.__class__.__name__,
            str(self.parentStyleSheet),
            id(self),
        )


class _SimpleNamespaces(_Namespaces):
    """
    namespaces used in objects like Selector as long as they are not connected
    to a CSSStyleSheet
    """

    def __init__(self, log=None, *args):
        """init"""
        super(_SimpleNamespaces, self).__init__(parentStyleSheet=None, log=log)
        self.__namespaces = dict(*args)

    def __setitem__(self, prefix, namespaceURI):
        self.__namespaces[prefix] = namespaceURI

    namespaces = property(
        lambda self: self.__namespaces,
        doc='Dict Wrapper for self.sheets @namespace rules.',
    )

    def __str__(self):
        return "<cssutils.util.%s object namespaces=%r at 0x%x>" % (
            self.__class__.__name__,
            self.namespaces,
            id(self),
        )

    def __repr__(self):
        return "cssutils.util.%s(%r)" % (self.__class__.__name__, self.namespaces)


def _readUrl(  # noqa: C901
    url, fetcher=None, overrideEncoding=None, parentEncoding=None
):
    """
    Read cssText from url and decode it using all relevant methods (HTTP
    header, BOM, @charset). Returns

    - encoding used to decode text (which is needed to set encoding of
      stylesheet properly)
    - type of encoding (how it was retrieved, see list below)
    - decodedCssText

    ``fetcher``
        see cssutils.CSSParser.setFetcher for details
    ``overrideEncoding``
        If given this encoding is used and all other encoding information is
        ignored (HTTP, BOM etc)
    ``parentEncoding``
        Encoding of parent stylesheet (while e.g. reading @import references
        sheets) or document if available.

    Priority or encoding information
    --------------------------------
    **cssutils only**: 0. overrideEncoding

    1. An HTTP "charset" parameter in a "Content-Type" field (or similar
       parameters in other protocols)
    2. BOM and/or @charset (see below)
    3. <link charset=""> or other metadata from the linking mechanism (if any)
    4. charset of referring style sheet or document (if any)
    5. Assume UTF-8

    """
    enctype = None

    if not fetcher:
        fetcher = _defaultFetcher

    r = fetcher(url)
    if r and len(r) == 2 and r[1] is not None:
        httpEncoding, content = r

        if overrideEncoding:
            enctype = 0  # 0. override encoding
            encoding = overrideEncoding
        elif httpEncoding:
            enctype = 1  # 1. HTTP
            encoding = httpEncoding
        else:
            # BOM or @charset
            if isinstance(content, str):
                contentEncoding, explicit = codec.detectencoding_unicode(content)
            else:
                contentEncoding, explicit = codec.detectencoding_str(content)

            if explicit:
                enctype = 2  # 2. BOM/@charset: explicitly
                encoding = contentEncoding

            elif parentEncoding:
                enctype = 4  # 4. parent stylesheet or document
                # may also be None in which case 5. is used in next step anyway
                encoding = parentEncoding

            else:
                enctype = 5  # 5. assume UTF-8
                encoding = 'utf-8'

        if isinstance(content, str):
            decodedCssText = content
        else:
            try:
                # encoding may still be wrong if encoding *is lying*!
                try:
                    decodedCssText = codecs.lookup("css")[1](
                        content, encoding=encoding
                    )[0]
                except AttributeError:
                    # at least in GAE
                    decodedCssText = content.decode(encoding if encoding else 'utf-8')

            except UnicodeDecodeError as e:
                log.warn(e, neverraise=True)
                decodedCssText = None

        return encoding, enctype, decodedCssText
    else:
        return None, None, None


class LazyRegex(object):
    """A class to represent a lazily compiled regular expression.

    The interface is kept similar to a `re.compile`ed object from the standard
    regex library.

    :ivar pattern: The original regular expression.
    :ivar flags: Flags of the regular expression.
    :ivar matcher: The compiled regular expression, or None if it is not yet
                   compiled.
    :ivar groups: The number of capturing groups in the pattern, or None.
    :ivar groupindex: A dictionary mapping any symbolic group names defined by
                      `(?P<id>)` to group numbers. The dictionary is empty if
                      no symbolic groups were used in the pattern. None if the
                      pattern is not yet compiled.
    """

    __slots__ = ('pattern', 'flags', 'matcher', 'groups', 'groupindex')

    def __init__(self, pattern, flags=0):
        self.pattern = pattern
        self.matcher = None
        self.flags = flags
        self.groups = None
        self.groupindex = None

    def ensure(self):
        """Make sure that the expression is compiled.

        If self.matcher is already a compiled expression, do nothing.
        """
        if self.matcher is not None:
            return
        self.matcher = re.compile(self.pattern, self.flags)
        self.flags = self.matcher.flags
        self.groups = self.matcher.groups
        self.groupindex = self.matcher.groupindex

    def __call__(self, string, pos=None, endpos=None):
        """Shortcut for self.match(string)."""
        return self.match(string, pos, endpos)

    def match(self, string, pos=None, endpos=None):
        """Attempt to do a match of the regular expression.

        This is similar to the `.match` method of `re` objects.
        """
        self.ensure()
        if pos is None:
            pos = 0
        if endpos is None:
            endpos = len(string)
        return self.matcher.match(string, pos, endpos)

    def search(self, string, pos=None, endpos=None):
        """Search the string for the pattern.

        This is similar to the `.search` method of `re` objects.
        """
        self.ensure()
        if pos is None:
            pos = 0
        if endpos is None:
            endpos = len(string)
        return self.matcher.search(string, pos, endpos)

    def split(self, string, maxsplit=0):
        """Split the string at the pattern.

        This is similar to the `.split` method of `re` objects.
        """
        self.ensure()
        return self.matcher.split(string, maxsplit)

    def findall(self, string, pos=None, endpos=None):
        """Find all instances of the pattern in the given string.

        This is similar to the `.findall` method of `re` objects.
        """
        self.ensure()
        if pos is None:
            pos = 0
        if endpos is None:
            endpos = len(string)
        return self.matcher.findall(string, pos, endpos)

    def finditer(self, string, pos=None, endpos=None):
        """Find all instances of the pattern in the given string.

        Returns an iterator.

        This is similar to the `.finditer` method of `re` objects.
        """
        self.ensure()
        if pos is None:
            pos = 0
        if endpos is None:
            endpos = len(string)
        return self.matcher.finditer(string, pos, endpos)

    def sub(self, repl, string, count=0):
        """Replace all occurences of the pattern with the given replacement.

        This is similar to the `.sub` method of `re` objects.
        """
        self.ensure()
        return self.matcher.sub(repl, string, count)

    def subn(self, repl, string, count=0):
        """Replace all occurences of the pattern with the given replacement.

        This is similar to the `.subn` method of `re` objects.
        """
        self.ensure()
        return self.matcher.subn(repl, string, count)
