"""productions for CSS 2.1

CSS2_1_MACROS and CSS2_1_PRODUCTIONS are from both
http://www.w3.org/TR/CSS21/grammar.html and
http://www.w3.org/TR/css3-syntax/#grammar0


"""
__all__ = ['CSSProductions', 'MACROS', 'PRODUCTIONS']

# option case-insensitive
MACROS = {
    'h': r'[0-9a-f]',
    # 'nonascii': r'[\200-\377]',
    'nonascii': r'[^\0-\177]',  # CSS3
    'unicode': r'\\{h}{1,6}(\r\n|[ \t\r\n\f])?',
    'escape': r'{unicode}|\\[^\r\n\f0-9a-f]',
    'nmstart': r'[_a-zA-Z]|{nonascii}|{escape}',
    'nmchar': r'[_a-zA-Z0-9-]|{nonascii}|{escape}',
    'string1': r'\"([^\n\r\f\\"]|\\{nl}|{escape})*\"',
    'string2': r"\'([^\n\r\f\\']|\\{nl}|{escape})*\'",
    'invalid1': r'\"([^\n\r\f\\"]|\\{nl}|{escape})*',
    'invalid2': r"\'([^\n\r\f\\']|\\{nl}|{escape})*",
    'comment': r'\/\*[^*]*\*+([^/*][^*]*\*+)*\/',
    # CSS list 080725 19:43
    # \/\*([^*\\]|{escape})*\*+(([^/*\\]|{escape})[^*]*\*+)*\/
    'ident': r'[-]?{nmstart}{nmchar}*',
    'name': r'{nmchar}+',
    # CHANGED TO SPEC: added "-?"
    'num': r'-?[0-9]*\.[0-9]+|[0-9]+',
    'string': r'{string1}|{string2}',
    'invalid': r'{invalid1}|{invalid2}',
    'url': r'([!#$%&*-~]|{nonascii}|{escape})*',
    's': r'[ \t\r\n\f]+',
    'w': r'{s}?',
    'nl': r'\n|\r\n|\r|\f',
    'range': r'\?{1,6}|{h}(\?{0,5}|{h}(\?{0,4}|{h}(\?{0,3}|{h}(\?{0,2}|{h}(\??|{h})))))',
    'A': r'a|\\0{0,4}(41|61)(\r\n|[ \t\r\n\f])?',
    'C': r'c|\\0{0,4}(43|63)(\r\n|[ \t\r\n\f])?',
    'D': r'd|\\0{0,4}(44|64)(\r\n|[ \t\r\n\f])?',
    'E': r'e|\\0{0,4}(45|65)(\r\n|[ \t\r\n\f])?',
    'F': r'f|\\0{0,4}(46|66)(\r\n|[ \t\r\n\f])?',
    'G': r'g|\\0{0,4}(47|67)(\r\n|[ \t\r\n\f])?|\\g',
    'H': r'h|\\0{0,4}(48|68)(\r\n|[ \t\r\n\f])?|\\h',
    'I': r'i|\\0{0,4}(49|69)(\r\n|[ \t\r\n\f])?|\\i',
    'K': r'k|\\0{0,4}(4b|6b)(\r\n|[ \t\r\n\f])?|\\k',
    'M': r'm|\\0{0,4}(4d|6d)(\r\n|[ \t\r\n\f])?|\\m',
    'N': r'n|\\0{0,4}(4e|6e)(\r\n|[ \t\r\n\f])?|\\n',
    'O': r'o|\\0{0,4}(51|71)(\r\n|[ \t\r\n\f])?|\\o',
    'P': r'p|\\0{0,4}(50|70)(\r\n|[ \t\r\n\f])?|\\p',
    'R': r'r|\\0{0,4}(52|72)(\r\n|[ \t\r\n\f])?|\\r',
    'S': r's|\\0{0,4}(53|73)(\r\n|[ \t\r\n\f])?|\\s',
    'T': r't|\\0{0,4}(54|74)(\r\n|[ \t\r\n\f])?|\\t',
    'X': r'x|\\0{0,4}(58|78)(\r\n|[ \t\r\n\f])?|\\x',
    'Z': r'z|\\0{0,4}(5a|7a)(\r\n|[ \t\r\n\f])?|\\z',
}

PRODUCTIONS = [
    ('URI', r'url\({w}{string}{w}\)'),  # "url("{w}{string}{w}")"    {return URI;}
    ('URI', r'url\({w}{url}{w}\)'),  # "url("{w}{url}{w}")"    {return URI;}
    ('FUNCTION', r'{ident}\('),  # {ident}"("        {return FUNCTION;}
    ('IMPORT_SYM', r'@{I}{M}{P}{O}{R}{T}'),  # "@import"        {return IMPORT_SYM;}
    ('PAGE_SYM', r'@{P}{A}{G}{E}'),  # "@page"            {return PAGE_SYM;}
    ('MEDIA_SYM', r'@{M}{E}{D}{I}{A}'),  # "@media"        {return MEDIA_SYM;}
    (
        'FONT_FACE_SYM',
        r'@{F}{O}{N}{T}\-{F}{A}{C}{E}',
    ),  # "@font-face"        {return FONT_FACE_SYM;}
    # CHANGED TO SPEC: only @charset
    ('CHARSET_SYM', r'@charset '),  # "@charset "        {return CHARSET_SYM;}
    (
        'NAMESPACE_SYM',
        r'@{N}{A}{M}{E}{S}{P}{A}{C}{E}',
    ),  # "@namespace"        {return NAMESPACE_SYM;}
    # CHANGED TO SPEC: ATKEYWORD
    ('ATKEYWORD', r'\@{ident}'),
    ('IDENT', r'{ident}'),  # {ident}            {return IDENT;}
    ('STRING', r'{string}'),  # {string}        {return STRING;}
    ('INVALID', r'{invalid}'),  # {return INVALID; /* unclosed string */}
    ('HASH', r'\#{name}'),  # "#"{name}        {return HASH;}
    ('PERCENTAGE', r'{num}%'),  # {num}%            {return PERCENTAGE;}
    ('LENGTH', r'{num}{E}{M}'),  # {num}em            {return EMS;}
    ('LENGTH', r'{num}{E}{X}'),  # {num}ex            {return EXS;}
    ('LENGTH', r'{num}{P}{X}'),  # {num}px            {return LENGTH;}
    ('LENGTH', r'{num}{C}{M}'),  # {num}cm            {return LENGTH;}
    ('LENGTH', r'{num}{M}{M}'),  # {num}mm            {return LENGTH;}
    ('LENGTH', r'{num}{I}{N}'),  # {num}in            {return LENGTH;}
    ('LENGTH', r'{num}{P}{T}'),  # {num}pt            {return LENGTH;}
    ('LENGTH', r'{num}{P}{C}'),  # {num}pc            {return LENGTH;}
    ('ANGLE', r'{num}{D}{E}{G}'),  # {num}deg        {return ANGLE;}
    ('ANGLE', r'{num}{R}{A}{D}'),  # {num}rad        {return ANGLE;}
    ('ANGLE', r'{num}{G}{R}{A}{D}'),  # {num}grad        {return ANGLE;}
    ('TIME', r'{num}{M}{S}'),  # {num}ms            {return TIME;}
    ('TIME', r'{num}{S}'),  # {num}s            {return TIME;}
    ('FREQ', r'{num}{H}{Z}'),  # {num}Hz            {return FREQ;}
    ('FREQ', r'{num}{K}{H}{Z}'),  # {num}kHz        {return FREQ;}
    ('DIMEN', r'{num}{ident}'),  # {num}{ident}        {return DIMEN;}
    ('NUMBER', r'{num}'),  # {num}            {return NUMBER;}
    # ('UNICODERANGE', r'U\+{range}'), #U\+{range}        {return UNICODERANGE;}
    # ('UNICODERANGE', r'U\+{h}{1,6}-{h}{1,6}'), #U\+{h}{1,6}-{h}{1,6}    {return UNICODERANGE;}  # noqa
    # --- CSS3 ---
    ('UNICODE-RANGE', r'[0-9A-F?]{1,6}(\-[0-9A-F]{1,6})?'),
    ('CDO', r'\<\!\-\-'),  # "<!--"            {return CDO;}
    ('CDC', r'\-\-\>'),  # "-->"            {return CDC;}
    ('S', r'{s}'),  # {return S;}
    # \/\*[^*]*\*+([^/*][^*]*\*+)*\/		/* ignore comments */
    # {s}+\/\*[^*]*\*+([^/*][^*]*\*+)*\/	{unput(' '); /*replace by space*/}
    ('INCLUDES', r'\~\='),  # "~="			{return INCLUDES;}
    ('DASHMATCH', r'\|\='),  # "|="			{return DASHMATCH;}
    ('LBRACE', r'\{'),  # {w}"{"			{return LBRACE;}
    ('PLUS', r'\+'),  # {w}"+"			{return PLUS;}
    ('GREATER', r'\>'),  # {w}">"			{return GREATER;}
    ('COMMA', r'\,'),  # {w}","			{return COMMA;}
    (
        'IMPORTANT_SYM',
        r'\!({w}|{comment})*{I}{M}{P}{O}{R}{T}{A}{N}{T}',
    ),  # "!{w}important"        {return IMPORTANT_SYM;}
    ('COMMENT', r'\/\*[^*]*\*+([^/][^*]*\*+)*\/'),  # /* ignore comments */
    ('CLASS', r'\.'),  # .			{return *yytext;}
    # --- CSS3! ---
    ('CHAR', r'[^"\']'),
]


class CSSProductions(object):
    pass


for i, t in enumerate(PRODUCTIONS):
    setattr(CSSProductions, t[0].replace('-', '_'), t[0])
