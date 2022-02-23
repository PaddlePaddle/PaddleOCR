# cython: language_level=3, py2_import=True
#
#   Cython Scanner - Lexical Definitions
#

from __future__ import absolute_import, unicode_literals

raw_prefixes = "rR"
bytes_prefixes = "bB"
string_prefixes = "fFuU" + bytes_prefixes
char_prefixes = "cC"
any_string_prefix = raw_prefixes + string_prefixes + char_prefixes
IDENT = 'IDENT'


def make_lexicon():
    from ..Plex import \
        Str, Any, AnyBut, AnyChar, Rep, Rep1, Opt, Bol, Eol, Eof, \
        TEXT, IGNORE, State, Lexicon
    from .Scanning import Method

    letter = Any("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_")
    digit = Any("0123456789")
    bindigit = Any("01")
    octdigit = Any("01234567")
    hexdigit = Any("0123456789ABCDEFabcdef")
    indentation = Bol + Rep(Any(" \t"))

    def underscore_digits(d):
        return Rep1(d) + Rep(Str("_") + Rep1(d))

    decimal = underscore_digits(digit)
    dot = Str(".")
    exponent = Any("Ee") + Opt(Any("+-")) + decimal
    decimal_fract = (decimal + dot + Opt(decimal)) | (dot + decimal)

    name = letter + Rep(letter | digit)
    intconst = decimal | (Str("0") + ((Any("Xx") + underscore_digits(hexdigit)) |
                                      (Any("Oo") + underscore_digits(octdigit)) |
                                      (Any("Bb") + underscore_digits(bindigit)) ))
    intsuffix = (Opt(Any("Uu")) + Opt(Any("Ll")) + Opt(Any("Ll"))) | (Opt(Any("Ll")) + Opt(Any("Ll")) + Opt(Any("Uu")))
    intliteral = intconst + intsuffix
    fltconst = (decimal_fract + Opt(exponent)) | (decimal + exponent)
    imagconst = (intconst | fltconst) + Any("jJ")

    # invalid combinations of prefixes are caught in p_string_literal
    beginstring = Opt(Rep(Any(string_prefixes + raw_prefixes)) |
                      Any(char_prefixes)
                      ) + (Str("'") | Str('"') | Str("'''") | Str('"""'))
    two_oct = octdigit + octdigit
    three_oct = octdigit + octdigit + octdigit
    two_hex = hexdigit + hexdigit
    four_hex = two_hex + two_hex
    escapeseq = Str("\\") + (two_oct | three_oct |
                             Str('N{') + Rep(AnyBut('}')) + Str('}') |
                             Str('u') + four_hex | Str('x') + two_hex |
                             Str('U') + four_hex + four_hex | AnyChar)

    bra = Any("([{")
    ket = Any(")]}")
    punct = Any(":,;+-*/|&<>=.%`~^?!@")
    diphthong = Str("==", "<>", "!=", "<=", ">=", "<<", ">>", "**", "//",
                    "+=", "-=", "*=", "/=", "%=", "|=", "^=", "&=",
                    "<<=", ">>=", "**=", "//=", "->", "@=")
    spaces = Rep1(Any(" \t\f"))
    escaped_newline = Str("\\\n")
    lineterm = Eol + Opt(Str("\n"))

    comment = Str("#") + Rep(AnyBut("\n"))

    return Lexicon([
        (name, IDENT),
        (intliteral, Method('strip_underscores', symbol='INT')),
        (fltconst, Method('strip_underscores', symbol='FLOAT')),
        (imagconst, Method('strip_underscores', symbol='IMAG')),
        (punct | diphthong, TEXT),

        (bra, Method('open_bracket_action')),
        (ket, Method('close_bracket_action')),
        (lineterm, Method('newline_action')),

        (beginstring, Method('begin_string_action')),

        (comment, IGNORE),
        (spaces, IGNORE),
        (escaped_newline, IGNORE),

        State('INDENT', [
            (comment + lineterm, Method('commentline')),
            (Opt(spaces) + Opt(comment) + lineterm, IGNORE),
            (indentation, Method('indentation_action')),
            (Eof, Method('eof_action'))
        ]),

        State('SQ_STRING', [
            (escapeseq, 'ESCAPE'),
            (Rep1(AnyBut("'\"\n\\")), 'CHARS'),
            (Str('"'), 'CHARS'),
            (Str("\n"), Method('unclosed_string_action')),
            (Str("'"), Method('end_string_action')),
            (Eof, 'EOF')
        ]),

        State('DQ_STRING', [
            (escapeseq, 'ESCAPE'),
            (Rep1(AnyBut('"\n\\')), 'CHARS'),
            (Str("'"), 'CHARS'),
            (Str("\n"), Method('unclosed_string_action')),
            (Str('"'), Method('end_string_action')),
            (Eof, 'EOF')
        ]),

        State('TSQ_STRING', [
            (escapeseq, 'ESCAPE'),
            (Rep1(AnyBut("'\"\n\\")), 'CHARS'),
            (Any("'\""), 'CHARS'),
            (Str("\n"), 'NEWLINE'),
            (Str("'''"), Method('end_string_action')),
            (Eof, 'EOF')
        ]),

        State('TDQ_STRING', [
            (escapeseq, 'ESCAPE'),
            (Rep1(AnyBut('"\'\n\\')), 'CHARS'),
            (Any("'\""), 'CHARS'),
            (Str("\n"), 'NEWLINE'),
            (Str('"""'), Method('end_string_action')),
            (Eof, 'EOF')
        ]),

        (Eof, Method('eof_action'))
        ],

        # FIXME: Plex 1.9 needs different args here from Plex 1.1.4
        #debug_flags = scanner_debug_flags,
        #debug_file = scanner_dump_file
        )

