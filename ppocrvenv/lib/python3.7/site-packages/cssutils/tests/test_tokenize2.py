"""Testcases for new cssutils.tokenize.Tokenizer

TODO: old tests as new ones are **not complete**!
"""

import sys
from . import basetest
import cssutils.tokenize2 as tokenize2
from cssutils.tokenize2 import Tokenizer


class TokenizerTestCase(basetest.BaseTestCase):

    testsall = {
        # IDENT
        'äöüß€': [('IDENT', 'äöüß€', 1, 1)],
        ' a ': [('S', ' ', 1, 1), ('IDENT', 'a', 1, 2), ('S', ' ', 1, 3)],
        '_a': [('IDENT', '_a', 1, 1)],
        '-a': [('IDENT', '-a', 1, 1)],
        'aA-_\200\377': [('IDENT', 'aA-_\200\377', 1, 1)],
        'a1': [('IDENT', 'a1', 1, 1)],
        # escapes must end with S or max 6 digits:
        '\\44 b': [('IDENT', 'Db', 1, 1)],
        '\\44  b': [('IDENT', 'D', 1, 1), ('S', ' ', 1, 5), ('IDENT', 'b', 1, 6)],
        '\\44\nb': [('IDENT', 'Db', 1, 1)],
        '\\44\rb': [('IDENT', 'Db', 1, 1)],
        '\\44\fb': [('IDENT', 'Db', 1, 1)],
        '\\44\n*': [('IDENT', 'D', 1, 1), ('CHAR', '*', 2, 1)],
        '\\44  a': [('IDENT', 'D', 1, 1), ('S', ' ', 1, 5), ('IDENT', 'a', 1, 6)],
        # TODO:
        # Note that this means that a "real" space after the escape sequence
        # must itself either be escaped or doubled:
        '\\44\\ x': [('IDENT', 'D\\ x', 1, 1)],
        '\\44  ': [('IDENT', 'D', 1, 1), ('S', ' ', 1, 5)],
        r'\44': [('IDENT', 'D', 1, 1)],
        r'\\': [('IDENT', r'\\', 1, 1)],
        r'\{': [('IDENT', r'\{', 1, 1)],
        r'\"': [('IDENT', r'\"', 1, 1)],
        r'\(': [('IDENT', r'\(', 1, 1)],
        r'\1 \22 \333 \4444 \55555 \666666 \777777 7 \7777777': [
            (
                ('IDENT', '\x01"\u0333\u4444\U00055555\\666666 \\777777 7', 1, 1)
                if sys.maxunicode > 0x10000
                else ('IDENT', '\x01"\u0333\u4444\\55555 \\666666 \\777777 7', 1, 1)
            ),
            ('S', ' ', 1, 43),
            ('IDENT', '\\7777777', 1, 44),
        ],
        # Not a function, important for media queries
        'and(': [('IDENT', 'and', 1, 1), ('CHAR', '(', 1, 4)],
        '\\1 b': [('IDENT', '\x01b', 1, 1)],
        # repeat: '\\44 b': [('IDENT', 'Db', 1, 1)],
        '\\123 b': [('IDENT', '\u0123b', 1, 1)],
        '\\1234 b': [('IDENT', '\u1234b', 1, 1)],
        '\\12345 b': [
            (
                ('IDENT', '\U00012345b', 1, 1)
                if sys.maxunicode > 0x10000
                else ('IDENT', '\\12345 b', 1, 1)
            )
        ],
        '\\123456 b': [('IDENT', '\\123456 b', 1, 1)],
        '\\1234567 b': [
            ('IDENT', '\\1234567', 1, 1),
            ('S', ' ', 1, 9),
            ('IDENT', 'b', 1, 10),
        ],
        '\\{\\}\\(\\)\\[\\]\\#\\@\\.\\,': [
            ('IDENT', '\\{\\}\\(\\)\\[\\]\\#\\@\\.\\,', 1, 1)
        ],
        # STRING
        ' "" ': [('S', ' ', 1, 1), ('STRING', '""', 1, 2), ('S', ' ', 1, 4)],
        ' "\'" ': [('S', ' ', 1, 1), ('STRING', '"\'"', 1, 2), ('S', ' ', 1, 5)],
        " '' ": [('S', ' ', 1, 1), ('STRING', "''", 1, 2), ('S', ' ', 1, 4)],
        # until 0.9.5.x
        # u"'\\\n'": [('STRING', u"'\\\n'", 1, 1)],
        # u"'\\\n\\\n\\\n'": [('STRING', u"'\\\n\\\n\\\n'", 1, 1)],
        # u"'\\\f'": [('STRING', u"'\\\f'", 1, 1)],
        # u"'\\\r'": [('STRING', u"'\\\r'", 1, 1)],
        # u"'\\\r\n'": [('STRING', u"'\\\r\n'", 1, 1)],
        # u"'1\\\n2'": [('STRING', u"'1\\\n2'", 1, 1)],
        # from 0.9.6a0 escaped nl is removed from string
        "'\\\n'": [('STRING', "''", 1, 1)],
        "'\\\n\\\n\\\n'": [('STRING', "''", 1, 1)],
        "'\\\f'": [('STRING', "''", 1, 1)],
        "'\\\r'": [('STRING', "''", 1, 1)],
        "'1\\\n2'": [('STRING', "'12'", 1, 1)],
        "'1\\\r\n2'": [('STRING', "'12'", 1, 1)],
        # ur'"\0020|\0020"': [('STRING', u'"\\0020|\\0020"', 1, 1)],
        r'"\61|\0061"': [('STRING', '"a|a"', 1, 1)],
        # HASH
        ' #a ': [('S', ' ', 1, 1), ('HASH', '#a', 1, 2), ('S', ' ', 1, 4)],
        '#ccc': [('HASH', '#ccc', 1, 1)],
        '#111': [('HASH', '#111', 1, 1)],
        '#a1a1a1': [('HASH', '#a1a1a1', 1, 1)],
        '#1a1a1a': [('HASH', '#1a1a1a', 1, 1)],
        # NUMBER, for plus see CSS3
        ' 0 ': [('S', ' ', 1, 1), ('NUMBER', '0', 1, 2), ('S', ' ', 1, 3)],
        ' 0.1 ': [('S', ' ', 1, 1), ('NUMBER', '0.1', 1, 2), ('S', ' ', 1, 5)],
        ' .0 ': [('S', ' ', 1, 1), ('NUMBER', '.0', 1, 2), ('S', ' ', 1, 4)],
        ' -0 ': [
            ('S', ' ', 1, 1),
            # ('CHAR', u'-', 1, 2),
            # ('NUMBER', u'0', 1, 3),
            ('NUMBER', '-0', 1, 2),
            ('S', ' ', 1, 4),
        ],
        # PERCENTAGE
        ' 0% ': [('S', ' ', 1, 1), ('PERCENTAGE', '0%', 1, 2), ('S', ' ', 1, 4)],
        ' .5% ': [('S', ' ', 1, 1), ('PERCENTAGE', '.5%', 1, 2), ('S', ' ', 1, 5)],
        # URI
        ' url() ': [('S', ' ', 1, 1), ('URI', 'url()', 1, 2), ('S', ' ', 1, 7)],
        ' url(a) ': [('S', ' ', 1, 1), ('URI', 'url(a)', 1, 2), ('S', ' ', 1, 8)],
        ' url("a") ': [('S', ' ', 1, 1), ('URI', 'url("a")', 1, 2), ('S', ' ', 1, 10)],
        ' url( a ) ': [('S', ' ', 1, 1), ('URI', 'url( a )', 1, 2), ('S', ' ', 1, 10)],
        # UNICODE-RANGE
        # CDO
        ' <!-- ': [('S', ' ', 1, 1), ('CDO', '<!--', 1, 2), ('S', ' ', 1, 6)],
        '"<!--""-->"': [('STRING', '"<!--"', 1, 1), ('STRING', '"-->"', 1, 7)],
        # CDC
        ' --> ': [('S', ' ', 1, 1), ('CDC', '-->', 1, 2), ('S', ' ', 1, 5)],
        # S
        ' ': [('S', ' ', 1, 1)],
        '  ': [('S', '  ', 1, 1)],
        '\r': [('S', '\r', 1, 1)],
        '\n': [('S', '\n', 1, 1)],
        '\r\n': [('S', '\r\n', 1, 1)],
        '\f': [('S', '\f', 1, 1)],
        '\t': [('S', '\t', 1, 1)],
        '\r\n\r\n\f\t ': [('S', '\r\n\r\n\f\t ', 1, 1)],
        # COMMENT, for incomplete see later
        '/*x*/ ': [('COMMENT', '/*x*/', 1, 1), ('S', ' ', 1, 6)],
        # FUNCTION
        ' x( ': [('S', ' ', 1, 1), ('FUNCTION', 'x(', 1, 2), ('S', ' ', 1, 4)],
        # INCLUDES
        ' ~= ': [('S', ' ', 1, 1), ('INCLUDES', '~=', 1, 2), ('S', ' ', 1, 4)],
        '~==': [('INCLUDES', '~=', 1, 1), ('CHAR', '=', 1, 3)],
        # DASHMATCH
        ' |= ': [('S', ' ', 1, 1), ('DASHMATCH', '|=', 1, 2), ('S', ' ', 1, 4)],
        '|==': [('DASHMATCH', '|=', 1, 1), ('CHAR', '=', 1, 3)],
        # CHAR
        ' @ ': [('S', ' ', 1, 1), ('CHAR', '@', 1, 2), ('S', ' ', 1, 3)],
        # --- overwritten for CSS 2.1 ---
        # LBRACE
        ' { ': [('S', ' ', 1, 1), ('CHAR', '{', 1, 2), ('S', ' ', 1, 3)],
        # PLUS
        ' + ': [('S', ' ', 1, 1), ('CHAR', '+', 1, 2), ('S', ' ', 1, 3)],
        # GREATER
        ' > ': [('S', ' ', 1, 1), ('CHAR', '>', 1, 2), ('S', ' ', 1, 3)],
        # COMMA
        ' , ': [('S', ' ', 1, 1), ('CHAR', ',', 1, 2), ('S', ' ', 1, 3)],
        # class
        ' . ': [('S', ' ', 1, 1), ('CHAR', '.', 1, 2), ('S', ' ', 1, 3)],
    }

    tests3 = {
        # UNICODE-RANGE
        ' u+0 ': [('S', ' ', 1, 1), ('UNICODE-RANGE', 'u+0', 1, 2), ('S', ' ', 1, 5)],
        ' u+01 ': [('S', ' ', 1, 1), ('UNICODE-RANGE', 'u+01', 1, 2), ('S', ' ', 1, 6)],
        ' u+012 ': [
            ('S', ' ', 1, 1),
            ('UNICODE-RANGE', 'u+012', 1, 2),
            ('S', ' ', 1, 7),
        ],
        ' u+0123 ': [
            ('S', ' ', 1, 1),
            ('UNICODE-RANGE', 'u+0123', 1, 2),
            ('S', ' ', 1, 8),
        ],
        ' u+01234 ': [
            ('S', ' ', 1, 1),
            ('UNICODE-RANGE', 'u+01234', 1, 2),
            ('S', ' ', 1, 9),
        ],
        ' u+012345 ': [
            ('S', ' ', 1, 1),
            ('UNICODE-RANGE', 'u+012345', 1, 2),
            ('S', ' ', 1, 10),
        ],
        ' u+0123456 ': [
            ('S', ' ', 1, 1),
            ('UNICODE-RANGE', 'u+012345', 1, 2),
            ('NUMBER', '6', 1, 10),
            ('S', ' ', 1, 11),
        ],
        ' U+123456 ': [
            ('S', ' ', 1, 1),
            ('UNICODE-RANGE', 'U+123456', 1, 2),
            ('S', ' ', 1, 10),
        ],
        ' \\55+abcdef ': [
            ('S', ' ', 1, 1),
            ('UNICODE-RANGE', 'U+abcdef', 1, 2),
            ('S', ' ', 1, 12),
        ],
        ' \\75+abcdef ': [
            ('S', ' ', 1, 1),
            ('UNICODE-RANGE', 'u+abcdef', 1, 2),
            ('S', ' ', 1, 12),
        ],
        ' u+0-1 ': [
            ('S', ' ', 1, 1),
            ('UNICODE-RANGE', 'u+0-1', 1, 2),
            ('S', ' ', 1, 7),
        ],
        ' u+0-1, u+123456-abcdef ': [
            ('S', ' ', 1, 1),
            ('UNICODE-RANGE', 'u+0-1', 1, 2),
            ('CHAR', ',', 1, 7),
            ('S', ' ', 1, 8),
            ('UNICODE-RANGE', 'u+123456-abcdef', 1, 9),
            ('S', ' ', 1, 24),
        ],
        # specials
        'c\\olor': [('IDENT', 'c\\olor', 1, 1)],
        # u'-1': [('CHAR', u'-', 1, 1), ('NUMBER', u'1', 1, 2)],
        # u'-1px': [('CHAR', u'-', 1, 1), ('DIMENSION', u'1px', 1, 2)],
        '-1': [('NUMBER', '-1', 1, 1)],
        '-1px': [('DIMENSION', '-1px', 1, 1)],
        # ATKEYWORD
        ' @x ': [('S', ' ', 1, 1), ('ATKEYWORD', '@x', 1, 2), ('S', ' ', 1, 4)],
        '@X': [('ATKEYWORD', '@X', 1, 1)],
        '@\\x': [('ATKEYWORD', '@\\x', 1, 1)],
        # -
        '@1x': [('CHAR', '@', 1, 1), ('DIMENSION', '1x', 1, 2)],
        # DIMENSION
        ' 0px ': [('S', ' ', 1, 1), ('DIMENSION', '0px', 1, 2), ('S', ' ', 1, 5)],
        ' 1s ': [('S', ' ', 1, 1), ('DIMENSION', '1s', 1, 2), ('S', ' ', 1, 4)],
        '0.2EM': [('DIMENSION', '0.2EM', 1, 1)],
        '1p\\x': [('DIMENSION', '1p\\x', 1, 1)],
        '1PX': [('DIMENSION', '1PX', 1, 1)],
        # NUMBER
        ' - 0 ': [
            ('S', ' ', 1, 1),
            ('CHAR', '-', 1, 2),
            ('S', ' ', 1, 3),
            ('NUMBER', '0', 1, 4),
            ('S', ' ', 1, 5),
        ],
        ' + 0 ': [
            ('S', ' ', 1, 1),
            ('CHAR', '+', 1, 2),
            ('S', ' ', 1, 3),
            ('NUMBER', '0', 1, 4),
            ('S', ' ', 1, 5),
        ],
        # PREFIXMATCH
        ' ^= ': [('S', ' ', 1, 1), ('PREFIXMATCH', '^=', 1, 2), ('S', ' ', 1, 4)],
        '^==': [('PREFIXMATCH', '^=', 1, 1), ('CHAR', '=', 1, 3)],
        # SUFFIXMATCH
        ' $= ': [('S', ' ', 1, 1), ('SUFFIXMATCH', '$=', 1, 2), ('S', ' ', 1, 4)],
        '$==': [('SUFFIXMATCH', '$=', 1, 1), ('CHAR', '=', 1, 3)],
        # SUBSTRINGMATCH
        ' *= ': [('S', ' ', 1, 1), ('SUBSTRINGMATCH', '*=', 1, 2), ('S', ' ', 1, 4)],
        '*==': [('SUBSTRINGMATCH', '*=', 1, 1), ('CHAR', '=', 1, 3)],
        # BOM only at start
        #        u'\xFEFF ': [('BOM', u'\xfeFF', 1, 1),
        #                  ('S', u' ', 1, 1)],
        #        u' \xFEFF ': [('S', u' ', 1, 1),
        #                  ('IDENT', u'\xfeFF', 1, 2),
        #                  ('S', u' ', 1, 5)],
        '\xfe\xff ': [('BOM', '\xfe\xff', 1, 1), ('S', ' ', 1, 1)],
        ' \xfe\xff ': [('S', ' ', 1, 1), ('IDENT', '\xfe\xff', 1, 2), ('S', ' ', 1, 4)],
        '\xef\xbb\xbf ': [('BOM', '\xef\xbb\xbf', 1, 1), ('S', ' ', 1, 1)],
        ' \xef\xbb\xbf ': [
            ('S', ' ', 1, 1),
            ('IDENT', '\xef\xbb\xbf', 1, 2),
            ('S', ' ', 1, 5),
        ],
    }

    tests2 = {
        # escapes work not for a-f!
        # IMPORT_SYM
        ' @import ': [
            ('S', ' ', 1, 1),
            ('IMPORT_SYM', '@import', 1, 2),
            ('S', ' ', 1, 9),
        ],
        '@IMPORT': [('IMPORT_SYM', '@IMPORT', 1, 1)],
        '@\\49\r\nMPORT': [('IMPORT_SYM', '@\\49\r\nMPORT', 1, 1)],
        r'@\i\m\p\o\r\t': [('IMPORT_SYM', r'@\i\m\p\o\r\t', 1, 1)],
        r'@\I\M\P\O\R\T': [('IMPORT_SYM', r'@\I\M\P\O\R\T', 1, 1)],
        r'@\49 \04d\0050\0004f\000052\54': [
            ('IMPORT_SYM', r'@\49 \04d\0050\0004f\000052\54', 1, 1)
        ],
        r'@\69 \06d\0070\0006f\000072\74': [
            ('IMPORT_SYM', r'@\69 \06d\0070\0006f\000072\74', 1, 1)
        ],
        # PAGE_SYM
        ' @page ': [('S', ' ', 1, 1), ('PAGE_SYM', '@page', 1, 2), ('S', ' ', 1, 7)],
        '@PAGE': [('PAGE_SYM', '@PAGE', 1, 1)],
        r'@\pa\ge': [('PAGE_SYM', r'@\pa\ge', 1, 1)],
        r'@\PA\GE': [('PAGE_SYM', r'@\PA\GE', 1, 1)],
        r'@\50\41\47\45': [('PAGE_SYM', r'@\50\41\47\45', 1, 1)],
        r'@\70\61\67\65': [('PAGE_SYM', r'@\70\61\67\65', 1, 1)],
        # MEDIA_SYM
        ' @media ': [('S', ' ', 1, 1), ('MEDIA_SYM', '@media', 1, 2), ('S', ' ', 1, 8)],
        '@MEDIA': [('MEDIA_SYM', '@MEDIA', 1, 1)],
        r'@\med\ia': [('MEDIA_SYM', r'@\med\ia', 1, 1)],
        r'@\MED\IA': [('MEDIA_SYM', r'@\MED\IA', 1, 1)],
        '@\\4d\n\\45\r\\44\t\\49\r\nA': [
            ('MEDIA_SYM', '@\\4d\n\\45\r\\44\t\\49\r\nA', 1, 1)
        ],
        '@\\4d\n\\45\r\\44\t\\49\r\\41\f': [
            ('MEDIA_SYM', '@\\4d\n\\45\r\\44\t\\49\r\\41\f', 1, 1)
        ],
        '@\\6d\n\\65\r\\64\t\\69\r\\61\f': [
            ('MEDIA_SYM', '@\\6d\n\\65\r\\64\t\\69\r\\61\f', 1, 1)
        ],
        # FONT_FACE_SYM
        ' @font-face ': [
            ('S', ' ', 1, 1),
            ('FONT_FACE_SYM', '@font-face', 1, 2),
            ('S', ' ', 1, 12),
        ],
        '@FONT-FACE': [('FONT_FACE_SYM', '@FONT-FACE', 1, 1)],
        r'@f\o\n\t\-face': [('FONT_FACE_SYM', r'@f\o\n\t\-face', 1, 1)],
        r'@F\O\N\T\-FACE': [('FONT_FACE_SYM', r'@F\O\N\T\-FACE', 1, 1)],
        # TODO: "-" as hex!
        r'@\46\4f\4e\54\-\46\41\43\45': [
            ('FONT_FACE_SYM', r'@\46\4f\4e\54\-\46\41\43\45', 1, 1)
        ],
        r'@\66\6f\6e\74\-\66\61\63\65': [
            ('FONT_FACE_SYM', r'@\66\6f\6e\74\-\66\61\63\65', 1, 1)
        ],
        # CHARSET_SYM only if "@charset "!
        '@charset  ': [('CHARSET_SYM', '@charset ', 1, 1), ('S', ' ', 1, 10)],
        ' @charset  ': [
            ('S', ' ', 1, 1),
            ('CHARSET_SYM', '@charset ', 1, 2),  # not at start
            ('S', ' ', 1, 11),
        ],
        '@charset': [('ATKEYWORD', '@charset', 1, 1)],  # no ending S
        '@CHARSET ': [('ATKEYWORD', '@CHARSET', 1, 1), ('S', ' ', 1, 9)],  # uppercase
        '@cha\\rset ': [
            ('ATKEYWORD', '@cha\\rset', 1, 1),  # not literal
            ('S', ' ', 1, 10),
        ],
        # NAMESPACE_SYM
        ' @namespace ': [
            ('S', ' ', 1, 1),
            ('NAMESPACE_SYM', '@namespace', 1, 2),
            ('S', ' ', 1, 12),
        ],
        r'@NAMESPACE': [('NAMESPACE_SYM', r'@NAMESPACE', 1, 1)],
        r'@\na\me\s\pace': [('NAMESPACE_SYM', r'@\na\me\s\pace', 1, 1)],
        r'@\NA\ME\S\PACE': [('NAMESPACE_SYM', r'@\NA\ME\S\PACE', 1, 1)],
        r'@\4e\41\4d\45\53\50\41\43\45': [
            ('NAMESPACE_SYM', r'@\4e\41\4d\45\53\50\41\43\45', 1, 1)
        ],
        r'@\6e\61\6d\65\73\70\61\63\65': [
            ('NAMESPACE_SYM', r'@\6e\61\6d\65\73\70\61\63\65', 1, 1)
        ],
        # ATKEYWORD
        ' @unknown ': [
            ('S', ' ', 1, 1),
            ('ATKEYWORD', '@unknown', 1, 2),
            ('S', ' ', 1, 10),
        ],
        # STRING
        # strings with linebreak in it
        ' "\\na"\na': [
            ('S', ' ', 1, 1),
            ('STRING', '"\\na"', 1, 2),
            ('S', '\n', 1, 7),
            ('IDENT', 'a', 2, 1),
        ],
        " '\\na'\na": [
            ('S', ' ', 1, 1),
            ('STRING', "'\\na'", 1, 2),
            ('S', '\n', 1, 7),
            ('IDENT', 'a', 2, 1),
        ],
        ' "\\r\\n\\t\\n\\ra"a': [
            ('S', ' ', 1, 1),
            ('STRING', '"\\r\\n\\t\\n\\ra"', 1, 2),
            ('IDENT', 'a', 1, 15),
        ],
        # IMPORTANT_SYM is not IDENT!!!
        ' !important ': [
            ('S', ' ', 1, 1),
            ('CHAR', '!', 1, 2),
            ('IDENT', 'important', 1, 3),
            ('S', ' ', 1, 12),
        ],
        '! /*1*/ important ': [
            ('CHAR', '!', 1, 1),
            ('S', ' ', 1, 2),
            ('COMMENT', '/*1*/', 1, 3),
            ('S', ' ', 1, 8),
            ('IDENT', 'important', 1, 9),
            ('S', ' ', 1, 18),
        ],
        '! important': [
            ('CHAR', '!', 1, 1),
            ('S', ' ', 1, 2),
            ('IDENT', 'important', 1, 3),
        ],
        '!\n\timportant': [
            ('CHAR', '!', 1, 1),
            ('S', '\n\t', 1, 2),
            ('IDENT', 'important', 2, 2),
        ],
        '!IMPORTANT': [('CHAR', '!', 1, 1), ('IDENT', 'IMPORTANT', 1, 2)],
        r'!\i\m\p\o\r\ta\n\t': [
            ('CHAR', '!', 1, 1),
            ('IDENT', r'\i\m\p\o\r\ta\n\t', 1, 2),
        ],
        r'!\I\M\P\O\R\Ta\N\T': [
            ('CHAR', '!', 1, 1),
            ('IDENT', r'\I\M\P\O\R\Ta\N\T', 1, 2),
        ],
        r'!\49\4d\50\4f\52\54\41\4e\54': [
            ('CHAR', '!', 1, 1),
            ('IDENT', r'IMPORTANT', 1, 2),
        ],
        r'!\69\6d\70\6f\72\74\61\6e\74': [
            ('CHAR', '!', 1, 1),
            ('IDENT', r'important', 1, 2),
        ],
    }

    # overwriting tests in testsall
    tests2only = {
        # LBRACE
        ' { ': [('S', ' ', 1, 1), ('LBRACE', '{', 1, 2), ('S', ' ', 1, 3)],
        # PLUS
        ' + ': [('S', ' ', 1, 1), ('PLUS', '+', 1, 2), ('S', ' ', 1, 3)],
        # GREATER
        ' > ': [('S', ' ', 1, 1), ('GREATER', '>', 1, 2), ('S', ' ', 1, 3)],
        # COMMA
        ' , ': [('S', ' ', 1, 1), ('COMMA', ',', 1, 2), ('S', ' ', 1, 3)],
        # class
        ' . ': [('S', ' ', 1, 1), ('CLASS', '.', 1, 2), ('S', ' ', 1, 3)],
    }

    testsfullsheet = {
        # escape ends with explicit space but \r\n as single space
        '\\65\r\nb': [('IDENT', 'eb', 1, 1)],
        # STRING
        r'"\""': [('STRING', r'"\""', 1, 1)],
        r'"\" "': [('STRING', r'"\" "', 1, 1)],
        """'\\''""": [('STRING', """'\\''""", 1, 1)],
        ' "\na': [
            ('S', ' ', 1, 1),
            ('INVALID', '"', 1, 2),
            ('S', '\n', 1, 3),
            ('IDENT', 'a', 2, 1),
        ],
        # strings with linebreak in it
        ' "\\na\na': [
            ('S', ' ', 1, 1),
            ('INVALID', '"\\na', 1, 2),
            ('S', '\n', 1, 6),
            ('IDENT', 'a', 2, 1),
        ],
        ' "\\r\\n\\t\\n\\ra\na': [
            ('S', ' ', 1, 1),
            ('INVALID', '"\\r\\n\\t\\n\\ra', 1, 2),
            ('S', '\n', 1, 14),
            ('IDENT', 'a', 2, 1),
        ],
        # URI
        'ur\\l(a)': [('URI', 'ur\\l(a)', 1, 1)],
        'url(a)': [('URI', 'url(a)', 1, 1)],
        '\\55r\\4c(a)': [('URI', 'UrL(a)', 1, 1)],
        '\\75r\\6c(a)': [('URI', 'url(a)', 1, 1)],
        ' url())': [('S', ' ', 1, 1), ('URI', 'url()', 1, 2), ('CHAR', ')', 1, 7)],
        'url("x"))': [('URI', 'url("x")', 1, 1), ('CHAR', ')', 1, 9)],
        "url('x'))": [('URI', "url('x')", 1, 1), ('CHAR', ')', 1, 9)],
    }

    # tests if fullsheet=False is set on tokenizer
    testsfullsheetfalse = {
        # COMMENT incomplete
        '/*': [('CHAR', '/', 1, 1), ('CHAR', '*', 1, 2)],
        # INVALID incomplete
        ' " ': [('S', ' ', 1, 1), ('INVALID', '" ', 1, 2)],
        " 'abc\"with quote\" in it": [
            ('S', ' ', 1, 1),
            ('INVALID', "'abc\"with quote\" in it", 1, 2),
        ],
        # URI incomplete
        'url(a': [('FUNCTION', 'url(', 1, 1), ('IDENT', 'a', 1, 5)],
        'url("a': [('FUNCTION', 'url(', 1, 1), ('INVALID', '"a', 1, 5)],
        "url('a": [('FUNCTION', 'url(', 1, 1), ('INVALID', "'a", 1, 5)],
        "UR\\l('a": [('FUNCTION', 'UR\\l(', 1, 1), ('INVALID', "'a", 1, 6)],
    }

    # tests if fullsheet=True is set on tokenizer
    testsfullsheettrue = {
        # COMMENT incomplete
        '/*': [('COMMENT', '/**/', 1, 1)],
        #        # INVALID incomplete => STRING
        ' " ': [('S', ' ', 1, 1), ('STRING', '" "', 1, 2)],
        " 'abc\"with quote\" in it": [
            ('S', ' ', 1, 1),
            ('STRING', "'abc\"with quote\" in it'", 1, 2),
        ],
        # URI incomplete FUNC => URI
        'url(a': [('URI', 'url(a)', 1, 1)],
        'url( a': [('URI', 'url( a)', 1, 1)],
        'url("a': [('URI', 'url("a")', 1, 1)],
        'url( "a ': [('URI', 'url( "a ")', 1, 1)],
        "url('a": [('URI', "url('a')", 1, 1)],
        'url("a"': [('URI', 'url("a")', 1, 1)],
        "url('a'": [('URI', "url('a')", 1, 1)],
    }

    def setUp(self):
        # log = cssutils.errorhandler.ErrorHandler()
        self.tokenizer = Tokenizer()

    #    NOT USED
    #    def test_push(self):
    #        "Tokenizer.push()"
    #        r = []
    #        def do():
    #            T = Tokenizer()
    #            x = False
    #            for t in T.tokenize('1 x 2 3'):
    #                if not x and t[1] == 'x':
    #                    T.push(t)
    #                    x = True
    #                r.append(t[1])
    #            return ''.join(r)
    #
    #        # push reinserts token into token stream, so x is doubled
    #        self.assertEqual('1 xx 2 3', do())

    #    def test_linenumbers(self):
    #        "Tokenizer line + col"
    #        pass

    def test_tokenize(self):
        "cssutils Tokenizer().tokenize()"
        import cssutils.cssproductions

        tokenizer = Tokenizer(
            cssutils.cssproductions.MACROS, cssutils.cssproductions.PRODUCTIONS
        )
        tests = {}
        tests.update(self.testsall)
        tests.update(self.tests2)
        tests.update(self.tests3)
        tests.update(self.testsfullsheet)
        tests.update(self.testsfullsheetfalse)
        for css in tests:
            # check token format
            tokens = tokenizer.tokenize(css)
            for i, actual in enumerate(tokens):
                expected = tests[css][i]
                self.assertEqual(expected, actual)

            # check if all same number of tokens
            tokens = list(tokenizer.tokenize(css))
            self.assertEqual(len(tokens), len(tests[css]))

    def test_tokenizefullsheet(self):
        "cssutils Tokenizer().tokenize(fullsheet=True)"
        import cssutils.cssproductions

        tokenizer = Tokenizer(
            cssutils.cssproductions.MACROS, cssutils.cssproductions.PRODUCTIONS
        )
        tests = {}
        tests.update(self.testsall)
        tests.update(self.tests2)
        tests.update(self.tests3)
        tests.update(self.testsfullsheet)
        tests.update(self.testsfullsheettrue)
        for css in tests:
            # check token format
            tokens = tokenizer.tokenize(css, fullsheet=True)
            for i, actual in enumerate(tokens):
                try:
                    expected = tests[css][i]
                except IndexError:
                    # EOF is added
                    self.assertEqual(actual[0], 'EOF')
                else:
                    self.assertEqual(expected, actual)

            # check if all same number of tokens
            tokens = list(tokenizer.tokenize(css, fullsheet=True))
            # EOF is added so -1
            self.assertEqual(len(tokens) - 1, len(tests[css]))


class TokenizerUtilsTestCase(basetest.BaseTestCase, metaclass=basetest.GenerateTests):
    """Tests for the util functions of tokenize"""

    def gen_test_has_at(self, string, pos, text, expected):
        self.assertEqual(tokenize2.has_at(string, pos, text), expected)

    gen_test_has_at.cases = [
        ('foo', 0, 'foo', True),
        ('foo', 0, 'f', True),
        ('foo', 1, 'o', True),
        ('foo', 1, 'oo', True),
        ('foo', 4, 'foo', False),
        ('foo', 0, 'bar', False),
        ('foo', 0, 'foobar', False),
    ]

    def gen_test_suffix_eq(self, string, pos, suffix, expected):
        self.assertEqual(tokenize2.suffix_eq(string, pos, suffix), expected)

    gen_test_suffix_eq.cases = [
        ('foobar', 0, 'foobar', True),
        ('foobar', 3, 'bar', True),
        ('foobar', 3, 'foo', False),
        ('foobar', 10, 'bar', False),
    ]


if __name__ == '__main__':
    import unittest

    unittest.main()
