import logging
import re
from typing import Dict, Iterable, Optional, cast

from .glyphlist import glyphname2unicode
from .latin_enc import ENCODING
from .psparser import PSLiteral

HEXADECIMAL = re.compile(r'[0-9a-fA-F]+')

log = logging.getLogger(__name__)


def name2unicode(name: str) -> str:
    """Converts Adobe glyph names to Unicode numbers.

    In contrast to the specification, this raises a KeyError instead of return
    an empty string when the key is unknown.
    This way the caller must explicitly define what to do
    when there is not a match.

    Reference:
    https://github.com/adobe-type-tools/agl-specification#2-the-mapping

    :returns unicode character if name resembles something,
    otherwise a KeyError
    """
    name = name.split('.')[0]
    components = name.split('_')

    if len(components) > 1:
        return ''.join(map(name2unicode, components))

    else:
        if name in glyphname2unicode:
            return glyphname2unicode[name]

        elif name.startswith('uni'):
            name_without_uni = name.strip('uni')

            if HEXADECIMAL.match(name_without_uni) and \
                    len(name_without_uni) % 4 == 0:
                unicode_digits = [int(name_without_uni[i:i + 4], base=16)
                                  for i in range(0, len(name_without_uni), 4)]
                for digit in unicode_digits:
                    raise_key_error_for_invalid_unicode(digit)
                characters = map(chr, unicode_digits)
                return ''.join(characters)

        elif name.startswith('u'):
            name_without_u = name.strip('u')

            if HEXADECIMAL.match(name_without_u) and \
                    4 <= len(name_without_u) <= 6:
                unicode_digit = int(name_without_u, base=16)
                raise_key_error_for_invalid_unicode(unicode_digit)
                return chr(unicode_digit)

    raise KeyError('Could not convert unicode name "%s" to character because '
                   'it does not match specification' % name)


def raise_key_error_for_invalid_unicode(unicode_digit: int) -> None:
    """Unicode values should not be in the range D800 through DFFF because
    that is used for surrogate pairs in UTF-16

    :raises KeyError if unicode digit is invalid
    """
    if 55295 < unicode_digit < 57344:
        raise KeyError('Unicode digit %d is invalid because '
                       'it is in the range D800 through DFFF' % unicode_digit)


class EncodingDB:

    std2unicode: Dict[int, str] = {}
    mac2unicode: Dict[int, str] = {}
    win2unicode: Dict[int, str] = {}
    pdf2unicode: Dict[int, str] = {}
    for (name, std, mac, win, pdf) in ENCODING:
        c = name2unicode(name)
        if std:
            std2unicode[std] = c
        if mac:
            mac2unicode[mac] = c
        if win:
            win2unicode[win] = c
        if pdf:
            pdf2unicode[pdf] = c

    encodings = {
        'StandardEncoding': std2unicode,
        'MacRomanEncoding': mac2unicode,
        'WinAnsiEncoding': win2unicode,
        'PDFDocEncoding': pdf2unicode,
    }

    @classmethod
    def get_encoding(
        cls,
        name: str,
        diff: Optional[Iterable[object]] = None
    ) -> Dict[int, str]:
        cid2unicode = cls.encodings.get(name, cls.std2unicode)
        if diff:
            cid2unicode = cid2unicode.copy()
            cid = 0
            for x in diff:
                if isinstance(x, int):
                    cid = x
                elif isinstance(x, PSLiteral):
                    try:
                        cid2unicode[cid] = name2unicode(cast(str, x.name))
                    except (KeyError, ValueError) as e:
                        log.debug(str(e))
                    cid += 1
        return cid2unicode
