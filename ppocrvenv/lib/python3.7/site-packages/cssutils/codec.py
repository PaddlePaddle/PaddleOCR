"""Python codec for CSS."""

import codecs
import marshal
import functools

# We're using bits to store all possible candidate encodings (or variants, i.e.
# we have two bits for the variants of UTF-16 and two for the
# variants of UTF-32).
#
# Prefixes for various CSS encodings
# UTF-8-SIG   xEF  xBB  xBF
# UTF-16 (LE) xFF  xFE ~x00|~x00
# UTF-16 (BE) xFE  xFF
# UTF-16-LE    @   x00   @   x00
# UTF-16-BE   x00   @
# UTF-32 (LE) xFF  xFE  x00  x00
# UTF-32 (BE) x00  x00  xFE  xFF
# UTF-32-LE    @   x00  x00  x00
# UTF-32-BE   x00  x00  x00   @
# CHARSET      @    c    h    a  ...


def chars(bytestring):
    return "".join(chr(byte) for byte in bytestring)


def detectencoding_str(input, final=False):  # noqa: C901
    """
    Detect the encoding of the byte string ``input``, which contains the
    beginning of a CSS file. This function returns the detected encoding (or
    ``None`` if it hasn't got enough data), and a flag that indicates whether
    that encoding has been detected explicitely or implicitely. To detect the
    encoding the first few bytes are used (or if ``input`` is ASCII compatible
    and starts with a charset rule the encoding name from the rule). "Explicit"
    detection means that the bytes start with a BOM or a charset rule.

    If the encoding can't be detected yet, ``None`` is returned as the encoding.
    ``final`` specifies whether more data will be available in later calls or
    not. If ``final`` is true, ``detectencoding_str()`` will never return
    ``None`` as the encoding.
    """

    # A bit for every candidate
    CANDIDATE_UTF_8_SIG = 1
    CANDIDATE_UTF_16_AS_LE = 2
    CANDIDATE_UTF_16_AS_BE = 4
    CANDIDATE_UTF_16_LE = 8
    CANDIDATE_UTF_16_BE = 16
    CANDIDATE_UTF_32_AS_LE = 32
    CANDIDATE_UTF_32_AS_BE = 64
    CANDIDATE_UTF_32_LE = 128
    CANDIDATE_UTF_32_BE = 256
    CANDIDATE_CHARSET = 512

    candidates = 1023  # all candidates

    # input = chars(input)
    li = len(input)
    if li >= 1:
        # Check first byte
        c = input[0]
        if c != b"\xef"[0]:
            candidates &= ~CANDIDATE_UTF_8_SIG
        if c != b"\xff"[0]:
            candidates &= ~(CANDIDATE_UTF_32_AS_LE | CANDIDATE_UTF_16_AS_LE)
        if c != b"\xfe"[0]:
            candidates &= ~CANDIDATE_UTF_16_AS_BE
        if c != b"@"[0]:
            candidates &= ~(
                CANDIDATE_UTF_32_LE | CANDIDATE_UTF_16_LE | CANDIDATE_CHARSET
            )
        if c != b"\x00"[0]:
            candidates &= ~(
                CANDIDATE_UTF_32_AS_BE | CANDIDATE_UTF_32_BE | CANDIDATE_UTF_16_BE
            )
        if li >= 2:
            # Check second byte
            c = input[1]
            if c != b"\xbb"[0]:
                candidates &= ~CANDIDATE_UTF_8_SIG
            if c != b"\xfe"[0]:
                candidates &= ~(CANDIDATE_UTF_16_AS_LE | CANDIDATE_UTF_32_AS_LE)
            if c != b"\xff"[0]:
                candidates &= ~CANDIDATE_UTF_16_AS_BE
            if c != b"\x00"[0]:
                candidates &= ~(
                    CANDIDATE_UTF_16_LE
                    | CANDIDATE_UTF_32_AS_BE
                    | CANDIDATE_UTF_32_LE
                    | CANDIDATE_UTF_32_BE
                )
            if c != b"@"[0]:
                candidates &= ~CANDIDATE_UTF_16_BE
            if c != b"c"[0]:
                candidates &= ~CANDIDATE_CHARSET
            if li >= 3:
                # Check third byte
                c = input[2]
                if c != b"\xbf"[0]:
                    candidates &= ~CANDIDATE_UTF_8_SIG
                if c != b"c"[0]:
                    candidates &= ~CANDIDATE_UTF_16_LE
                if c != b"\x00"[0]:
                    candidates &= ~(
                        CANDIDATE_UTF_32_AS_LE
                        | CANDIDATE_UTF_32_LE
                        | CANDIDATE_UTF_32_BE
                    )
                if c != b"\xfe"[0]:
                    candidates &= ~CANDIDATE_UTF_32_AS_BE
                if c != b"h"[0]:
                    candidates &= ~CANDIDATE_CHARSET
                if li >= 4:
                    # Check fourth byte
                    c = input[3]
                    if input[2:4] == b"\x00\x00"[0:2]:
                        candidates &= ~CANDIDATE_UTF_16_AS_LE
                    if c != b"\x00"[0]:
                        candidates &= ~(
                            CANDIDATE_UTF_16_LE
                            | CANDIDATE_UTF_32_AS_LE
                            | CANDIDATE_UTF_32_LE
                        )
                    if c != b"\xff"[0]:
                        candidates &= ~CANDIDATE_UTF_32_AS_BE
                    if c != b"@"[0]:
                        candidates &= ~CANDIDATE_UTF_32_BE
                    if c != b"a"[0]:
                        candidates &= ~CANDIDATE_CHARSET
    if candidates == 0:
        return ("utf-8", False)
    if not (candidates & (candidates - 1)):  # only one candidate remaining
        if candidates == CANDIDATE_UTF_8_SIG and li >= 3:
            return ("utf-8-sig", True)
        elif candidates == CANDIDATE_UTF_16_AS_LE and li >= 2:
            return ("utf-16", True)
        elif candidates == CANDIDATE_UTF_16_AS_BE and li >= 2:
            return ("utf-16", True)
        elif candidates == CANDIDATE_UTF_16_LE and li >= 4:
            return ("utf-16-le", False)
        elif candidates == CANDIDATE_UTF_16_BE and li >= 2:
            return ("utf-16-be", False)
        elif candidates == CANDIDATE_UTF_32_AS_LE and li >= 4:
            return ("utf-32", True)
        elif candidates == CANDIDATE_UTF_32_AS_BE and li >= 4:
            return ("utf-32", True)
        elif candidates == CANDIDATE_UTF_32_LE and li >= 4:
            return ("utf-32-le", False)
        elif candidates == CANDIDATE_UTF_32_BE and li >= 4:
            return ("utf-32-be", False)
        elif candidates == CANDIDATE_CHARSET and li >= 4:
            prefix = '@charset "'
            charsinput = chars(input)
            if charsinput[: len(prefix)] == prefix:
                pos = charsinput.find('"', len(prefix))
                if pos >= 0:
                    # TODO: return str and not bytes!
                    return (charsinput[len(prefix) : pos], True)
    # if this is the last call, and we haven't determined an encoding yet,
    # we default to UTF-8
    if final:
        return ("utf-8", False)
    return (None, False)  # dont' know yet


def detectencoding_unicode(input, final=False):
    """
    Detect the encoding of the unicode string ``input``, which contains the
    beginning of a CSS file. The encoding is detected from the charset rule
    at the beginning of ``input``. If there is no charset rule, ``"utf-8"``
    will be returned.

    If the encoding can't be detected yet, ``None`` is returned. ``final``
    specifies whether more data will be available in later calls or not. If
    ``final`` is true, ``detectencoding_unicode()`` will never return ``None``.
    """
    prefix = '@charset "'
    if input.startswith(prefix):
        pos = input.find('"', len(prefix))
        if pos >= 0:
            return (input[len(prefix) : pos], True)
    elif final or not prefix.startswith(input):
        # if this is the last call, and we haven't determined an encoding yet,
        # (or the string definitely doesn't start with prefix) we default to UTF-8
        return ("utf-8", False)
    return (None, False)  # don't know yet


def _fixencoding(input, encoding, final=False):
    """
    Replace the name of the encoding in the charset rule at the beginning of
    ``input`` with ``encoding``. If ``input`` doesn't starts with a charset
    rule, ``input`` will be returned unmodified.

    If the encoding can't be found yet, ``None`` is returned. ``final``
    specifies whether more data will be available in later calls or not.
    If ``final`` is true, ``_fixencoding()`` will never return ``None``.
    """
    prefix = '@charset "'
    if len(input) > len(prefix):
        if input.startswith(prefix):
            pos = input.find('"', len(prefix))
            if pos >= 0:
                if encoding.replace("_", "-").lower() == "utf-8-sig":
                    encoding = "utf-8"
                return prefix + encoding + input[pos:]
            # we haven't seen the end of the encoding name yet => fall through
        else:
            return input  # doesn't start with prefix, so nothing to fix
    elif not prefix.startswith(input) or final:
        # can't turn out to be a @charset rule later (or there is no "later")
        return input
    if final:
        return input
    return None  # don't know yet


def decode(input, errors="strict", encoding=None, force=True):
    try:
        # py 3 only, memory?! object to bytes
        input = input.tobytes()
    except AttributeError:
        pass

    if encoding is None or not force:
        (_encoding, explicit) = detectencoding_str(input, True)
        if _encoding == "css":
            raise ValueError("css not allowed as encoding name")
        if (
            explicit and not force
        ) or encoding is None:  # Take the encoding from the input
            encoding = _encoding

    # NEEDS: change in parse.py (str to bytes!)
    (input, consumed) = codecs.getdecoder(encoding)(input, errors)
    return (_fixencoding(input, str(encoding), True), consumed)


def encode(input, errors="strict", encoding=None):
    consumed = len(input)
    if encoding is None:
        encoding = detectencoding_unicode(input, True)[0]
        if encoding.replace("_", "-").lower() == "utf-8-sig":
            input = _fixencoding(input, "utf-8", True)
    else:
        input = _fixencoding(input, str(encoding), True)
    if encoding == "css":
        raise ValueError("css not allowed as encoding name")
    encoder = codecs.getencoder(encoding)
    return (encoder(input, errors)[0], consumed)


def _bytes2int(bytes):
    # Helper: convert an 8 bit string into an ``int``.
    i = 0
    for byte in bytes:
        i = (i << 8) + ord(byte)
    return i


def _int2bytes(i):
    # Helper: convert an ``int`` into an 8-bit string.
    v = []
    while i:
        v.insert(0, chr(i & 0xFF))
        i >>= 8
    return "".join(v)


class IncrementalDecoder(codecs.IncrementalDecoder):
    def __init__(self, errors="strict", encoding=None, force=True):
        self.decoder = None
        self.encoding = encoding
        self.force = force
        codecs.IncrementalDecoder.__init__(self, errors)
        # Store ``errors`` somewhere else,
        # because we have to hide it in a property
        self._errors = errors
        self.buffer = b""
        self.headerfixed = False

    def iterdecode(self, input):
        for part in input:
            result = self.decode(part, False)
            if result:
                yield result
        result = self.decode("", True)
        if result:
            yield result

    def decode(self, input, final=False):
        # We're doing basically the same as a ``BufferedIncrementalDecoder``,
        # but since the buffer is only relevant until the encoding has been
        # detected (in which case the buffer of the underlying codec might
        # kick in), we're implementing buffering ourselves to avoid some
        # overhead.
        if self.decoder is None:
            input = self.buffer + input
            # Do we have to detect the encoding from the input?
            if self.encoding is None or not self.force:
                (encoding, explicit) = detectencoding_str(input, final)
                if encoding is None:  # no encoding determined yet
                    self.buffer = input  # retry the complete input on the next call
                    return ""  # no encoding determined yet, so no output
                elif encoding == "css":
                    raise ValueError("css not allowed as encoding name")
                if (
                    explicit and not self.force
                ) or self.encoding is None:  # Take the encoding from the input
                    self.encoding = encoding
            self.buffer = ""  # drop buffer, as the decoder might keep its own
            decoder = codecs.getincrementaldecoder(self.encoding)
            self.decoder = decoder(self._errors)
        if self.headerfixed:
            return self.decoder.decode(input, final)
        # If we haven't fixed the header yet,
        # the content of ``self.buffer`` is a ``unicode`` object
        output = self.buffer + self.decoder.decode(input, final)
        encoding = self.encoding
        if encoding.replace("_", "-").lower() == "utf-8-sig":
            encoding = "utf-8"
        newoutput = _fixencoding(output, str(encoding), final)
        if newoutput is None:
            # retry fixing the @charset rule (but keep the decoded stuff)
            self.buffer = output
            return ""
        self.headerfixed = True
        return newoutput

    def reset(self):
        codecs.IncrementalDecoder.reset(self)
        self.decoder = None
        self.buffer = b""
        self.headerfixed = False

    def _geterrors(self):
        return self._errors

    def _seterrors(self, errors):
        # Setting ``errors`` must be done on the real decoder too
        if self.decoder is not None:
            self.decoder.errors = errors
        self._errors = errors

    errors = property(_geterrors, _seterrors)

    def getstate(self):
        if self.decoder is not None:
            state = (
                self.encoding,
                self.buffer,
                self.headerfixed,
                True,
                self.decoder.getstate(),
            )
        else:
            state = (self.encoding, self.buffer, self.headerfixed, False, None)
        return ("", _bytes2int(marshal.dumps(state)))

    def setstate(self, state):
        state = _int2bytes(marshal.loads(state[1]))  # ignore buffered input
        self.encoding = state[0]
        self.buffer = state[1]
        self.headerfixed = state[2]
        if state[3] is not None:
            self.decoder = codecs.getincrementaldecoder(self.encoding)(self._errors)
            self.decoder.setstate(state[4])
        else:
            self.decoder = None


class IncrementalEncoder(codecs.IncrementalEncoder):
    def __init__(self, errors="strict", encoding=None):
        self.encoder = None
        self.encoding = encoding
        codecs.IncrementalEncoder.__init__(self, errors)
        # Store ``errors`` somewhere else,
        # because we have to hide it in a property
        self._errors = errors
        self.buffer = ""

    def iterencode(self, input):
        for part in input:
            result = self.encode(part, False)
            if result:
                yield result
        result = self.encode("", True)
        if result:
            yield result

    def encode(self, input, final=False):
        if self.encoder is None:
            input = self.buffer + input
            if self.encoding is not None:
                # Replace encoding in the @charset rule with the specified one
                encoding = self.encoding
                if encoding.replace("_", "-").lower() == "utf-8-sig":
                    encoding = "utf-8"
                newinput = _fixencoding(input, str(encoding), final)
                if newinput is None:  # @charset rule incomplete => Retry next time
                    self.buffer = input
                    return ""
                input = newinput
            else:
                # Use encoding from the @charset declaration
                self.encoding = detectencoding_unicode(input, final)[0]
            if self.encoding is not None:
                if self.encoding == "css":
                    raise ValueError("css not allowed as encoding name")
                info = codecs.lookup(self.encoding)
                encoding = self.encoding
                if self.encoding.replace("_", "-").lower() == "utf-8-sig":
                    input = _fixencoding(input, "utf-8", True)
                self.encoder = info.incrementalencoder(self._errors)
                self.buffer = ""
            else:
                self.buffer = input
                return ""
        return self.encoder.encode(input, final)

    def reset(self):
        codecs.IncrementalEncoder.reset(self)
        self.encoder = None
        self.buffer = ""

    def _geterrors(self):
        return self._errors

    def _seterrors(self, errors):
        # Setting ``errors ``must be done on the real encoder too
        if self.encoder is not None:
            self.encoder.errors = errors
        self._errors = errors

    errors = property(_geterrors, _seterrors)

    def getstate(self):
        if self.encoder is not None:
            state = (self.encoding, self.buffer, True, self.encoder.getstate())
        else:
            state = (self.encoding, self.buffer, False, None)
        return _bytes2int(marshal.dumps(state))

    def setstate(self, state):
        state = _int2bytes(marshal.loads(state))
        self.encoding = state[0]
        self.buffer = state[1]
        if state[2] is not None:
            self.encoder = codecs.getincrementalencoder(self.encoding)(self._errors)
            self.encoder.setstate(state[4])
        else:
            self.encoder = None


class StreamWriter(codecs.StreamWriter):
    def __init__(self, stream, errors="strict", encoding=None, header=False):
        codecs.StreamWriter.__init__(self, stream, errors)
        self.streamwriter = None
        self.encoding = encoding
        self._errors = errors
        self.buffer = ""

    def encode(self, input, errors="strict"):
        li = len(input)
        if self.streamwriter is None:
            input = self.buffer + input
            li = len(input)
            if self.encoding is not None:
                # Replace encoding in the @charset rule with the specified one
                encoding = self.encoding
                if encoding.replace("_", "-").lower() == "utf-8-sig":
                    encoding = "utf-8"
                newinput = _fixencoding(input, str(encoding), False)
                if newinput is None:  # @charset rule incomplete => Retry next time
                    self.buffer = input
                    return ("", 0)
                input = newinput
            else:
                # Use encoding from the @charset declaration
                self.encoding = detectencoding_unicode(input, False)[0]
            if self.encoding is not None:
                if self.encoding == "css":
                    raise ValueError("css not allowed as encoding name")
                self.streamwriter = codecs.getwriter(self.encoding)(
                    self.stream, self._errors
                )
                encoding = self.encoding
                if self.encoding.replace("_", "-").lower() == "utf-8-sig":
                    input = _fixencoding(input, "utf-8", True)
                self.buffer = ""
            else:
                self.buffer = input
                return ("", 0)
        return (self.streamwriter.encode(input, errors)[0], li)

    def _geterrors(self):
        return self._errors

    def _seterrors(self, errors):
        # Setting ``errors`` must be done on the streamwriter too
        try:
            if self.streamwriter is not None:
                self.streamwriter.errors = errors
        except AttributeError:
            # TODO: py3 only exception?
            pass

        self._errors = errors

    errors = property(_geterrors, _seterrors)


class StreamReader(codecs.StreamReader):
    def __init__(self, stream, errors="strict", encoding=None, force=True):
        codecs.StreamReader.__init__(self, stream, errors)
        self.streamreader = None
        self.encoding = encoding
        self.force = force
        self._errors = errors

    def decode(self, input, errors="strict"):
        if self.streamreader is None:
            if self.encoding is None or not self.force:
                (encoding, explicit) = detectencoding_str(input, False)
                if encoding is None:  # no encoding determined yet
                    return ("", 0)  # no encoding determined yet, so no output
                elif encoding == "css":
                    raise ValueError("css not allowed as encoding name")
                if (
                    explicit and not self.force
                ) or self.encoding is None:  # Take the encoding from the input
                    self.encoding = encoding
            streamreader = codecs.getreader(self.encoding)
            streamreader = streamreader(self.stream, self._errors)
            (output, consumed) = streamreader.decode(input, errors)
            encoding = self.encoding
            if encoding.replace("_", "-").lower() == "utf-8-sig":
                encoding = "utf-8"
            newoutput = _fixencoding(output, str(encoding), False)
            if newoutput is not None:
                self.streamreader = streamreader
                return (newoutput, consumed)
            return ("", 0)  # we will create a new streamreader on the next call
        return self.streamreader.decode(input, errors)

    def _geterrors(self):
        return self._errors

    def _seterrors(self, errors):
        # Setting ``errors`` must be done on the streamreader too
        try:
            if self.streamreader is not None:
                self.streamreader.errors = errors
        except AttributeError:
            # TODO: py3 only exception?
            pass

        self._errors = errors

    errors = property(_geterrors, _seterrors)


@codecs.register
def search_function(name):
    if name != "css":
        return
    return codecs.CodecInfo(
        name="css",
        encode=encode,
        decode=decode,
        incrementalencoder=IncrementalEncoder,
        incrementaldecoder=IncrementalDecoder,
        streamwriter=StreamWriter,
        streamreader=StreamReader,
    )


@functools.partial(codecs.register_error, "cssescape")
def cssescape(exc):
    """
    Error handler for CSS escaping.
    """
    if not isinstance(exc, UnicodeEncodeError):
        raise TypeError("don't know how to handle %r" % exc)
    return (
        "".join("\\%06x" % ord(c) for c in exc.object[exc.start : exc.end]),
        exc.end,
    )
