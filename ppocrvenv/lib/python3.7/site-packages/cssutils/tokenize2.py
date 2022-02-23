"""New CSS Tokenizer (a generator)
"""
__all__ = ['Tokenizer', 'CSSProductions']

import itertools
import re
import sys

from .cssproductions import CSSProductions, MACROS, PRODUCTIONS
from .helper import normalize

_TOKENIZER_CACHE = {}


class Tokenizer(object):
    """
    generates a list of Token tuples:
        (Tokenname, value, startline, startcolumn)
    """

    _atkeywords = {
        '@font-face': CSSProductions.FONT_FACE_SYM,
        '@import': CSSProductions.IMPORT_SYM,
        '@media': CSSProductions.MEDIA_SYM,
        '@namespace': CSSProductions.NAMESPACE_SYM,
        '@page': CSSProductions.PAGE_SYM,
        '@variables': CSSProductions.VARIABLES_SYM,
    }
    _linesep = '\n'
    unicodesub = re.compile(r'\\[0-9a-fA-F]{1,6}(?:\r\n|[\t\r\n\f\x20])?').sub
    cleanstring = re.compile(r'\\((\r\n)|[\n\r\f])').sub

    def __init__(self, macros=None, productions=None, doComments=True):
        """
        inits tokenizer with given macros and productions which default to
        cssutils own macros and productions
        """
        if isinstance(macros, dict):
            macros_hash_key = sorted(macros.items())
        else:
            macros_hash_key = macros
        hash_key = str((macros_hash_key, productions))
        if hash_key in _TOKENIZER_CACHE:
            (tokenmatches, commentmatcher, urimatcher) = _TOKENIZER_CACHE[hash_key]
        else:
            if not macros:
                macros = MACROS
            if not productions:
                productions = PRODUCTIONS
            tokenmatches = self._compile_productions(
                self._expand_macros(macros, productions)
            )
            commentmatcher = [x[1] for x in tokenmatches if x[0] == 'COMMENT'][0]
            urimatcher = [x[1] for x in tokenmatches if x[0] == 'URI'][0]
            _TOKENIZER_CACHE[hash_key] = (tokenmatches, commentmatcher, urimatcher)

        self.tokenmatches = tokenmatches
        self.commentmatcher = commentmatcher
        self.urimatcher = urimatcher

        self._doComments = doComments
        self._pushed = []

    def _expand_macros(self, macros, productions):
        """returns macro expanded productions, order of productions is kept"""

        def macro_value(m):
            return '(?:%s)' % macros[m.groupdict()['macro']]

        expanded = []
        for key, value in productions:
            while re.search(r'{[a-zA-Z][a-zA-Z0-9-]*}', value):
                value = re.sub(
                    r'{(?P<macro>[a-zA-Z][a-zA-Z0-9-]*)}', macro_value, value
                )
            expanded.append((key, value))
        return expanded

    def _compile_productions(self, expanded_productions):
        """compile productions into callable match objects, order is kept"""
        compiled = []
        for key, value in expanded_productions:
            compiled.append((key, re.compile('(?:%s)' % value, re.U).match))
        return compiled

    def push(self, *tokens):
        """Push back tokens which have been pulled but not processed."""
        self._pushed = itertools.chain(tokens, self._pushed)

    def clear(self):
        self._pushed = []

    def tokenize(self, text, fullsheet=False):  # noqa: C901
        """Generator: Tokenize text and yield tokens, each token is a tuple
        of::

            (name, value, line, col)

        The token value will contain a normal string, meaning CSS unicode
        escapes have been resolved to normal characters. The serializer
        escapes needed characters back to unicode escapes depending on
        the stylesheet target encoding.

        text
            to be tokenized
        fullsheet
            if ``True`` appends EOF token as last one and completes incomplete
            COMMENT or INVALID (to STRING) tokens
        """

        def _repl(m):
            "used by unicodesub"
            num = int(m.group(0)[1:], 16)
            if num <= sys.maxunicode:
                return chr(num)
            else:
                return m.group(0)

        def _normalize(value):
            "normalize and do unicodesub"
            return normalize(self.unicodesub(_repl, value))

        line = col = 1
        # The current starting character. We just increase this instead of
        # splitting off the beginning of text to increase performance.
        pos = 0

        # check for BOM first as it should only be max one at the start
        (BOM, matcher), productions = self.tokenmatches[0], self.tokenmatches[1:]
        match = matcher(text, pos)
        if match:
            found = match.group(0)
            yield (BOM, found, line, col)
            pos += len(found)

        # check for @charset which is valid only at start of CSS
        if has_at(text, pos, '@charset '):
            found = '@charset '  # production has trailing S!
            yield (CSSProductions.CHARSET_SYM, found, line, col)
            pos += len(found)
            col += len(found)

        # Avoid repeated function call
        _len_text = len(text)
        _orig_text = text
        while pos < _len_text:
            # do pushed tokens before new ones
            for pushed in self._pushed:
                yield pushed

            # speed test for most used CHARs, sadly . not possible :(
            c = text[pos]
            if c in ',:;{}>[]':  # + but in num!
                yield ('CHAR', c, line, col)
                col += 1
                pos += 1

            else:
                # check all other productions, at least CHAR must match
                for name, matcher in productions:

                    # TODO: USE bad comment?
                    if fullsheet and name == 'CHAR' and has_at(text, pos, '/*'):
                        # before CHAR production test for incomplete comment
                        possiblecomment = '%s*/' % text[pos:]
                        match = self.commentmatcher(possiblecomment)
                        if match and self._doComments:
                            yield ('COMMENT', possiblecomment, line, col)
                            pos = _len_text  # ate all remaining text
                            break

                    match = matcher(text, pos)  # if no match try next production
                    if match:
                        found = match.group(0)  # needed later for line/col
                        # The ident regex also matches the beginning of
                        # functions, but we can't put the function regex before
                        # the ident regex, as otherwise 'and(' is recognized as
                        # function (even though it is valid in media queries).
                        # So we're doing this: if we find an ident, but the next
                        # character is a open parenthesis, we instead skip and
                        # let the FUNCTION production take over - except if the
                        # ident is "and"
                        if (
                            name == 'IDENT'
                            and found.lower() != "and"
                            and match.end(0) < len(text)
                            and text[match.end(0)] == '('
                        ):
                            continue
                        if fullsheet:
                            # check if found may be completed into a full token
                            if 'INVALID' == name and suffix_eq(text, pos, found):
                                # complete INVALID to STRING with start char " or '
                                name, found = 'STRING', '%s%s' % (found, found[0])

                            elif 'FUNCTION' == name and 'url(' == _normalize(found):
                                # url( is a FUNCTION if incomplete sheet
                                # FUNCTION production MUST BE after URI production
                                for end in ("')", '")', ')'):
                                    possibleuri = '%s%s' % (text[pos:], end)
                                    match = self.urimatcher(possibleuri)
                                    if match:
                                        name, found = 'URI', match.group(0)
                                        break

                        if name in (
                            'DIMENSION',
                            'IDENT',
                            'STRING',
                            'URI',
                            'HASH',
                            'COMMENT',
                            'FUNCTION',
                            'INVALID',
                            'UNICODE-RANGE',
                        ):
                            # may contain unicode escape, replace with normal
                            # char but do not _normalize (?)
                            value = self.unicodesub(_repl, found)
                            if name in ('STRING', 'INVALID'):  # 'URI'?
                                # remove \ followed by nl (so escaped) from string
                                value = self.cleanstring('', value)

                        else:
                            if 'ATKEYWORD' == name:
                                try:
                                    # get actual ATKEYWORD SYM
                                    name = self._atkeywords[_normalize(found)]
                                except KeyError:
                                    # might also be misplace @charset...
                                    if '@charset' == found and has_at(
                                        text, pos + len(found), ' '
                                    ):
                                        # @charset needs tailing S!
                                        name = CSSProductions.CHARSET_SYM
                                        found += ' '
                                    else:
                                        name = 'ATKEYWORD'

                            value = found  # should not contain unicode escape (?)

                        if self._doComments or (
                            not self._doComments and name != 'COMMENT'
                        ):
                            yield (name, value, line, col)

                        pos += len(found)
                        nls = found.count(self._linesep)
                        line += nls
                        if nls:
                            col = len(found[found.rfind(self._linesep) :])
                        else:
                            col += len(found)

                        break
            # Make sure we didn't accidentally modify text in the process
            assert text is _orig_text

        if fullsheet:
            yield ('EOF', '', line, col)


def has_at(text, pos, string):
    """Check if text has substring string as position pos.

    This is effectively equal to ``text[pos:pos+len(string)] == string``.

    If ``pos`` is 0, this is equal to ``text.startswith(string)``.

    :param str text: The text to check against.
    :param int pos: The starting position where the substring should be
                    searched.
    :param str string: The string to search.
    """
    return text[pos : pos + len(string)] == string


def suffix_eq(text, pos, expected):
    """Check if the suffix of text starting at pos equals expected.

    This is effectively equivalent to ``text[pos:] == expected``, but avoids
    allocating a huge substring.

    If ``pos`` is 0, this is equal to ``text == expected``.

    :param str text: Text to compare.
    :param int pos: Starting position for the suffix.
    :param str expected: String to compare to.
    """
    if not pos:
        return text == expected
    return len(text) - pos == len(expected) and all(
        text[i + pos] == expected[i] for i in range(len(expected))
    )
