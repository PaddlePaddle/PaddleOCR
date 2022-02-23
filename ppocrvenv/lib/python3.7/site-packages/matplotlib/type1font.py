"""
A class representing a Type 1 font.

This version reads pfa and pfb files and splits them for embedding in
pdf files. It also supports SlantFont and ExtendFont transformations,
similarly to pdfTeX and friends. There is no support yet for subsetting.

Usage::

   >>> font = Type1Font(filename)
   >>> clear_part, encrypted_part, finale = font.parts
   >>> slanted_font = font.transform({'slant': 0.167})
   >>> extended_font = font.transform({'extend': 1.2})

Sources:

* Adobe Technical Note #5040, Supporting Downloadable PostScript
  Language Fonts.

* Adobe Type 1 Font Format, Adobe Systems Incorporated, third printing,
  v1.1, 1993. ISBN 0-201-57044-0.
"""

import binascii
import enum
import itertools
import logging
import re
import struct

import numpy as np

from matplotlib.cbook import _format_approx
from . import _api

_log = logging.getLogger(__name__)

# token types
_TokenType = enum.Enum('_TokenType',
                       'whitespace name string delimiter number')


class Type1Font:
    """
    A class representing a Type-1 font, for use by backends.

    Attributes
    ----------
    parts : tuple
        A 3-tuple of the cleartext part, the encrypted part, and the finale of
        zeros.
    decrypted : bytes
        The decrypted form of parts[1].
    prop : dict[str, Any]
        A dictionary of font properties.
    """
    __slots__ = ('parts', 'decrypted', 'prop')

    def __init__(self, input):
        """
        Initialize a Type-1 font.

        Parameters
        ----------
        input : str or 3-tuple
            Either a pfb file name, or a 3-tuple of already-decoded Type-1
            font `~.Type1Font.parts`.
        """
        if isinstance(input, tuple) and len(input) == 3:
            self.parts = input
        else:
            with open(input, 'rb') as file:
                data = self._read(file)
            self.parts = self._split(data)

        self.decrypted = self._decrypt(self.parts[1], 'eexec')
        self._parse()

    def _read(self, file):
        """Read the font from a file, decoding into usable parts."""
        rawdata = file.read()
        if not rawdata.startswith(b'\x80'):
            return rawdata

        data = b''
        while rawdata:
            if not rawdata.startswith(b'\x80'):
                raise RuntimeError('Broken pfb file (expected byte 128, '
                                   'got %d)' % rawdata[0])
            type = rawdata[1]
            if type in (1, 2):
                length, = struct.unpack('<i', rawdata[2:6])
                segment = rawdata[6:6 + length]
                rawdata = rawdata[6 + length:]

            if type == 1:       # ASCII text: include verbatim
                data += segment
            elif type == 2:     # binary data: encode in hexadecimal
                data += binascii.hexlify(segment)
            elif type == 3:     # end of file
                break
            else:
                raise RuntimeError('Unknown segment type %d in pfb file' %
                                   type)

        return data

    def _split(self, data):
        """
        Split the Type 1 font into its three main parts.

        The three parts are: (1) the cleartext part, which ends in a
        eexec operator; (2) the encrypted part; (3) the fixed part,
        which contains 512 ASCII zeros possibly divided on various
        lines, a cleartomark operator, and possibly something else.
        """

        # Cleartext part: just find the eexec and skip whitespace
        idx = data.index(b'eexec')
        idx += len(b'eexec')
        while data[idx] in b' \t\r\n':
            idx += 1
        len1 = idx

        # Encrypted part: find the cleartomark operator and count
        # zeros backward
        idx = data.rindex(b'cleartomark') - 1
        zeros = 512
        while zeros and data[idx] in b'0' or data[idx] in b'\r\n':
            if data[idx] in b'0':
                zeros -= 1
            idx -= 1
        if zeros:
            # this may have been a problem on old implementations that
            # used the zeros as necessary padding
            _log.info('Insufficiently many zeros in Type 1 font')

        # Convert encrypted part to binary (if we read a pfb file, we may end
        # up converting binary to hexadecimal to binary again; but if we read
        # a pfa file, this part is already in hex, and I am not quite sure if
        # even the pfb format guarantees that it will be in binary).
        idx1 = len1 + ((idx - len1 + 2) & ~1)  # ensure an even number of bytes
        binary = binascii.unhexlify(data[len1:idx1])

        return data[:len1], binary, data[idx+1:]

    _whitespace_or_comment_re = re.compile(br'[\0\t\r\014\n ]+|%[^\r\n\v]*')
    _token_re = re.compile(br'/{0,2}[^]\0\t\r\v\n ()<>{}/%[]+')
    _instring_re = re.compile(br'[()\\]')

    @staticmethod
    def _decrypt(ciphertext, key, ndiscard=4):
        """
        Decrypt ciphertext using the Type-1 font algorithm

        The algorithm is described in Adobe's "Adobe Type 1 Font Format".
        The key argument can be an integer, or one of the strings
        'eexec' and 'charstring', which map to the key specified for the
        corresponding part of Type-1 fonts.

        The ndiscard argument should be an integer, usually 4.
        That number of bytes is discarded from the beginning of plaintext.
        """

        key = _api.check_getitem({'eexec': 55665, 'charstring': 4330}, key=key)
        plaintext = []
        for byte in ciphertext:
            plaintext.append(byte ^ (key >> 8))
            key = ((key+byte) * 52845 + 22719) & 0xffff

        return bytes(plaintext[ndiscard:])

    @staticmethod
    def _encrypt(plaintext, key, ndiscard=4):
        """
        Encrypt plaintext using the Type-1 font algorithm

        The algorithm is described in Adobe's "Adobe Type 1 Font Format".
        The key argument can be an integer, or one of the strings
        'eexec' and 'charstring', which map to the key specified for the
        corresponding part of Type-1 fonts.

        The ndiscard argument should be an integer, usually 4. That
        number of bytes is prepended to the plaintext before encryption.
        This function prepends NUL bytes for reproducibility, even though
        the original algorithm uses random bytes, presumably to avoid
        cryptanalysis.
        """

        key = _api.check_getitem({'eexec': 55665, 'charstring': 4330}, key=key)
        ciphertext = []
        for byte in b'\0' * ndiscard + plaintext:
            c = byte ^ (key >> 8)
            ciphertext.append(c)
            key = ((key + c) * 52845 + 22719) & 0xffff

        return bytes(ciphertext)

    @classmethod
    def _tokens(cls, text):
        """
        A PostScript tokenizer. Yield (token, value) pairs such as
        (_TokenType.whitespace, '   ') or (_TokenType.name, '/Foobar').
        """
        # Preload enum members for speed.
        tok_whitespace = _TokenType.whitespace
        tok_name = _TokenType.name
        tok_string = _TokenType.string
        tok_delimiter = _TokenType.delimiter
        tok_number = _TokenType.number
        pos = 0
        while pos < len(text):
            match = cls._whitespace_or_comment_re.match(text, pos)
            if match:
                yield (tok_whitespace, match.group())
                pos = match.end()
            elif text[pos:pos+1] == b'(':
                start = pos
                pos += 1
                depth = 1
                while depth:
                    match = cls._instring_re.search(text, pos)
                    if match is None:
                        return
                    pos = match.end()
                    if match.group() == b'(':
                        depth += 1
                    elif match.group() == b')':
                        depth -= 1
                    else:  # a backslash - skip the next character
                        pos += 1
                yield (tok_string, text[start:pos])
            elif text[pos:pos + 2] in (b'<<', b'>>'):
                yield (tok_delimiter, text[pos:pos + 2])
                pos += 2
            elif text[pos:pos+1] == b'<':
                start = pos
                pos = text.index(b'>', pos)
                yield (tok_string, text[start:pos])
            else:
                match = cls._token_re.match(text, pos)
                if match:
                    try:
                        float(match.group())
                        yield (tok_number, match.group())
                    except ValueError:
                        yield (tok_name, match.group())
                    pos = match.end()
                else:
                    yield (tok_delimiter, text[pos:pos + 1])
                    pos += 1

    def _parse(self):
        """
        Find the values of various font properties. This limited kind
        of parsing is described in Chapter 10 "Adobe Type Manager
        Compatibility" of the Type-1 spec.
        """
        # Preload enum members for speed.
        tok_whitespace = _TokenType.whitespace
        tok_name = _TokenType.name
        tok_string = _TokenType.string
        tok_number = _TokenType.number
        # Start with reasonable defaults
        prop = {'weight': 'Regular', 'ItalicAngle': 0.0, 'isFixedPitch': False,
                'UnderlinePosition': -100, 'UnderlineThickness': 50}
        filtered = ((token, value)
                    for token, value in self._tokens(self.parts[0])
                    if token is not tok_whitespace)
        # The spec calls this an ASCII format; in Python 2.x we could
        # just treat the strings and names as opaque bytes but let's
        # turn them into proper Unicode, and be lenient in case of high bytes.
        def convert(x): return x.decode('ascii', 'replace')
        for token, value in filtered:
            if token is tok_name and value.startswith(b'/'):
                key = convert(value[1:])
                token, value = next(filtered)
                if token is tok_name:
                    if value in (b'true', b'false'):
                        value = value == b'true'
                    else:
                        value = convert(value.lstrip(b'/'))
                elif token is tok_string:
                    value = convert(value.lstrip(b'(').rstrip(b')'))
                elif token is tok_number:
                    if b'.' in value:
                        value = float(value)
                    else:
                        value = int(value)
                else:  # more complicated value such as an array
                    value = None
                if key != 'FontInfo' and value is not None:
                    prop[key] = value

        # Fill in the various *Name properties
        if 'FontName' not in prop:
            prop['FontName'] = (prop.get('FullName') or
                                prop.get('FamilyName') or
                                'Unknown')
        if 'FullName' not in prop:
            prop['FullName'] = prop['FontName']
        if 'FamilyName' not in prop:
            extras = ('(?i)([ -](regular|plain|italic|oblique|(semi)?bold|'
                      '(ultra)?light|extra|condensed))+$')
            prop['FamilyName'] = re.sub(extras, '', prop['FullName'])

        self.prop = prop

    @classmethod
    def _transformer(cls, tokens, slant, extend):
        tok_whitespace = _TokenType.whitespace
        tok_name = _TokenType.name

        def fontname(name):
            result = name
            if slant:
                result += b'_Slant_%d' % int(1000 * slant)
            if extend != 1.0:
                result += b'_Extend_%d' % int(1000 * extend)
            return result

        def italicangle(angle):
            return b'%a' % round(
                float(angle) - np.arctan(slant) / np.pi * 180,
                5
            )

        def fontmatrix(array):
            array = array.lstrip(b'[').rstrip(b']').split()
            array = [float(x) for x in array]
            oldmatrix = np.eye(3, 3)
            oldmatrix[0:3, 0] = array[::2]
            oldmatrix[0:3, 1] = array[1::2]
            modifier = np.array([[extend, 0, 0],
                                 [slant, 1, 0],
                                 [0, 0, 1]])
            newmatrix = np.dot(modifier, oldmatrix)
            array[::2] = newmatrix[0:3, 0]
            array[1::2] = newmatrix[0:3, 1]
            return (
                '[%s]' % ' '.join(_format_approx(x, 6) for x in array)
            ).encode('ascii')

        def replace(fun):
            def replacer(tokens):
                token, value = next(tokens)      # name, e.g., /FontMatrix
                yield value
                token, value = next(tokens)      # possible whitespace
                while token is tok_whitespace:
                    yield value
                    token, value = next(tokens)
                if value != b'[':                # name/number/etc.
                    yield fun(value)
                else:                            # array, e.g., [1 2 3]
                    result = b''
                    while value != b']':
                        result += value
                        token, value = next(tokens)
                    result += value
                    yield fun(result)
            return replacer

        def suppress(tokens):
            for _ in itertools.takewhile(lambda x: x[1] != b'def', tokens):
                pass
            yield b''

        table = {b'/FontName': replace(fontname),
                 b'/ItalicAngle': replace(italicangle),
                 b'/FontMatrix': replace(fontmatrix),
                 b'/UniqueID': suppress}

        for token, value in tokens:
            if token is tok_name and value in table:
                yield from table[value](
                    itertools.chain([(token, value)], tokens))
            else:
                yield value

    def transform(self, effects):
        """
        Return a new font that is slanted and/or extended.

        Parameters
        ----------
        effects : dict
            A dict with optional entries:

            - 'slant' : float, default: 0
                Tangent of the angle that the font is to be slanted to the
                right. Negative values slant to the left.
            - 'extend' : float, default: 1
                Scaling factor for the font width. Values less than 1 condense
                the glyphs.

        Returns
        -------
        `Type1Font`
        """
        tokenizer = self._tokens(self.parts[0])
        transformed = self._transformer(tokenizer,
                                        slant=effects.get('slant', 0.0),
                                        extend=effects.get('extend', 1.0))
        return Type1Font((b"".join(transformed), self.parts[1], self.parts[2]))
