"""Testcases for cssutils.codec"""

import codecs
import unittest
import io

from cssutils import codec

try:
    codecs.lookup("utf-32")
except LookupError:
    haveutf32 = False
else:
    haveutf32 = True

iostream = io.BytesIO


class Queue(object):
    """
    queue: write bytes at one end, read bytes from the other end
    """

    def __init__(self):
        self._buffer = "".encode()

    def write(self, chars):
        # TODO ???
        if isinstance(chars, str):
            chars = chars.encode()
        elif isinstance(chars, int):
            chars = bytes([chars])

        self._buffer += chars

    def read(self, size=-1):
        if size < 0:
            s = self._buffer
            self._buffer = "".encode()
            return s
        else:
            s = self._buffer[:size]
            self._buffer = self._buffer[size:]
            return s


class CodecTestCase(unittest.TestCase):
    def test_detectencoding_str(self):
        "codec.detectencoding_str()"
        self.assertEqual(codec.detectencoding_str(''.encode()), (None, False))
        self.assertEqual(
            codec.detectencoding_str('\xef'.encode('latin1')), (None, False)
        )
        self.assertEqual(
            codec.detectencoding_str('\xef\x33'.encode("utf-8")), ("utf-8", False)
        )
        self.assertEqual(
            codec.detectencoding_str('\xc3\xaf3'.encode("utf-8")), ("utf-8", False)
        )
        self.assertEqual(
            codec.detectencoding_str('\xef\xbb'.encode("latin1")), (None, False)
        )
        self.assertEqual(
            codec.detectencoding_str('\xef\xbb\x33'.encode("utf-8")), ("utf-8", False)
        )
        self.assertEqual(
            codec.detectencoding_str('\xef\xbb\xbf'.encode("utf-8-sig")),
            ("utf-8-sig", True),
        )
        self.assertEqual(
            codec.detectencoding_str('\xff'.encode("latin1")), (None, False)
        )
        self.assertEqual(
            codec.detectencoding_str('\xff\x33'.encode("utf-8")), ("utf-8", False)
        )
        self.assertEqual(
            codec.detectencoding_str('\xff\xfe'.encode("latin1")), (None, False)
        )
        self.assertEqual(
            codec.detectencoding_str('\xff\xfe\x33'.encode("utf-16")), ("utf-16", True)
        )
        self.assertEqual(
            codec.detectencoding_str('\xff\xfe\x00'.encode("latin1")), (None, False)
        )
        self.assertEqual(
            codec.detectencoding_str('\xff\xfe\x00\x33'.encode("utf-16")),
            ("utf-16", True),
        )
        if haveutf32:
            self.assertEqual(
                codec.detectencoding_str('\xff\xfe\x00\x00'.encode("utf-32")),
                ("utf-32", True),
            )
        self.assertEqual(codec.detectencoding_str('\x00'.encode()), (None, False))
        self.assertEqual(
            codec.detectencoding_str('\x00\x33'.encode()), ("utf-8", False)
        )
        self.assertEqual(codec.detectencoding_str('\x00\x00'.encode()), (None, False))
        self.assertEqual(
            codec.detectencoding_str('\x00\x00\x33'.encode()), ("utf-8", False)
        )
        self.assertEqual(
            codec.detectencoding_str('\x00\x00\xfe'.encode('latin1')), (None, False)
        )
        self.assertEqual(
            codec.detectencoding_str('\x00\x00\x00\x33'.encode()), ("utf-8", False)
        )
        if haveutf32:
            self.assertEqual(
                codec.detectencoding_str('\x00\x00\x00@'.encode()), ("utf-32-be", False)
            )
            self.assertEqual(
                codec.detectencoding_str('\x00\x00\xfe\xff'.encode('utf-32')),
                ("utf-32", True),
            )
        self.assertEqual(codec.detectencoding_str('@'.encode()), (None, False))
        self.assertEqual(codec.detectencoding_str('@\x33'.encode()), ("utf-8", False))
        self.assertEqual(codec.detectencoding_str('@\x00'.encode()), (None, False))
        self.assertEqual(
            codec.detectencoding_str('@\x00\x33'.encode()), ("utf-8", False)
        )
        self.assertEqual(codec.detectencoding_str('@\x00\x00'.encode()), (None, False))
        self.assertEqual(
            codec.detectencoding_str('@\x00\x00\x33'.encode()), ("utf-8", False)
        )
        if haveutf32:
            self.assertEqual(
                codec.detectencoding_str('@\x00\x00\x00'.encode()), ("utf-32-le", False)
            )
        self.assertEqual(codec.detectencoding_str('@c'.encode()), (None, False))
        self.assertEqual(codec.detectencoding_str('@ch'.encode()), (None, False))
        self.assertEqual(codec.detectencoding_str('@cha'.encode()), (None, False))
        self.assertEqual(codec.detectencoding_str('@char'.encode()), (None, False))
        self.assertEqual(codec.detectencoding_str('@chars'.encode()), (None, False))
        self.assertEqual(codec.detectencoding_str('@charse'.encode()), (None, False))
        self.assertEqual(codec.detectencoding_str('@charset'.encode()), (None, False))
        self.assertEqual(codec.detectencoding_str('@charset '.encode()), (None, False))
        self.assertEqual(codec.detectencoding_str('@charset "'.encode()), (None, False))
        self.assertEqual(
            codec.detectencoding_str('@charset "x'.encode()), (None, False)
        )
        self.assertEqual(codec.detectencoding_str('@charset ""'.encode()), ("", True))
        self.assertEqual(codec.detectencoding_str('@charset "x"'.encode()), ("x", True))
        self.assertEqual(codec.detectencoding_str("@".encode(), False), (None, False))
        self.assertEqual(codec.detectencoding_str("@".encode(), True), ("utf-8", False))
        self.assertEqual(codec.detectencoding_str("@c".encode(), False), (None, False))
        self.assertEqual(
            codec.detectencoding_str("@c".encode(), True), ("utf-8", False)
        )

    def test_detectencoding_unicode(self):
        "codec.detectencoding_unicode()"
        # Unicode version (only parses the header)
        self.assertEqual(codec.detectencoding_unicode('@charset "x'), (None, False))
        self.assertEqual(codec.detectencoding_unicode('a {}'), ("utf-8", False))
        self.assertEqual(
            codec.detectencoding_unicode('@charset "x', True), (None, False)
        )
        self.assertEqual(codec.detectencoding_unicode('@charset "x"'), ("x", True))

    def test_fixencoding(self):
        "codec._fixencoding()"
        s = '@charset "'
        self.assertTrue(codec._fixencoding(s, "utf-8") is None)

        s = '@charset "x'
        self.assertTrue(codec._fixencoding(s, "utf-8") is None)

        s = '@charset "x'
        self.assertEqual(codec._fixencoding(s, "utf-8", True), s)

        s = '@charset x'
        self.assertEqual(codec._fixencoding(s, "utf-8"), s)

        s = '@charset "x"'
        self.assertEqual(codec._fixencoding(s, "utf-8"), s.replace('"x"', '"utf-8"'))

    def test_decoder(self):  # noqa: C901
        "codecs.decoder"

        def checkauto(encoding, input='@charset "x";g\xfcrk\u20ac{}'):
            outputencoding = encoding
            if outputencoding == "utf-8-sig":
                outputencoding = "utf-8"
            # Check stateless decoder with encoding autodetection
            d = codecs.getdecoder("css")
            self.assertEqual(
                d(input.encode(encoding))[0],
                input.replace('"x"', '"%s"' % outputencoding),
            )

            # Check stateless decoder with specified encoding
            self.assertEqual(
                d(input.encode(encoding), encoding=encoding)[0],
                input.replace('"x"', '"%s"' % outputencoding),
            )

            if hasattr(codec, "getincrementaldecoder"):
                # Check incremental decoder with encoding autodetection
                id = codecs.getincrementaldecoder("css")()
                self.assertEqual(
                    "".join(id.iterdecode(input.encode(encoding))),
                    input.replace('"x"', '"%s"' % outputencoding),
                )

                # Check incremental decoder with specified encoding
                id = codecs.getincrementaldecoder("css")(encoding=encoding)
                self.assertEqual(
                    "".join(id.iterdecode(input.encode(encoding))),
                    input.replace('"x"', '"%s"' % outputencoding),
                )

            # Check stream reader with encoding autodetection
            q = Queue()
            sr = codecs.getreader("css")(q)
            result = []
            # TODO: py3 only???
            for c in input.encode(encoding):
                q.write(c)
                result.append(sr.read())
            self.assertEqual(
                "".join(result), input.replace('"x"', '"%s"' % outputencoding)
            )

            # Check stream reader with specified encoding
            q = Queue()
            sr = codecs.getreader("css")(q, encoding=encoding)
            result = []
            for c in input.encode(encoding):
                q.write(c)
                result.append(sr.read())
            self.assertEqual(
                "".join(result), input.replace('"x"', '"%s"' % outputencoding)
            )

        # Autodetectable encodings
        checkauto("utf-8-sig")
        checkauto("utf-16")
        checkauto("utf-16-le")
        checkauto("utf-16-be")
        if haveutf32:
            checkauto("utf-32")
            checkauto("utf-32-le")
            checkauto("utf-32-be")

        def checkdecl(encoding, input='@charset "%s";g\xfcrk{}'):
            # Check stateless decoder with encoding autodetection
            d = codecs.getdecoder("css")
            input = input % encoding
            outputencoding = encoding
            if outputencoding == "utf-8-sig":
                outputencoding = "utf-8"
            self.assertEqual(d(input.encode(encoding))[0], input)

            # Check stateless decoder with specified encoding
            self.assertEqual(d(input.encode(encoding), encoding=encoding)[0], input)

            if hasattr(codec, "getincrementaldecoder"):
                # Check incremental decoder with encoding autodetection
                id = codecs.getincrementaldecoder("css")()
                self.assertEqual("".join(id.iterdecode(input.encode(encoding))), input)

                # Check incremental decoder with specified encoding
                id = codecs.getincrementaldecoder("css")(encoding)
                self.assertEqual("".join(id.iterdecode(input.encode(encoding))), input)

            # Check stream reader with encoding autodetection
            q = Queue()
            sr = codecs.getreader("css")(q)
            result = []
            for c in input.encode(encoding):
                q.write(c)
                result.append(sr.read())
            self.assertEqual("".join(result), input)

            # Check stream reader with specified encoding
            q = Queue()
            sr = codecs.getreader("css")(q, encoding=encoding)
            result = []
            for c in input.encode(encoding):
                q.write(c)
                result.append(sr.read())
            self.assertEqual("".join(result), input)

        # Use correct declaration
        checkdecl("utf-8")
        checkdecl("iso-8859-1", '@charset "%s";g\xfcrk')
        checkdecl("iso-8859-15")
        checkdecl("cp1252")

        # No recursion
        self.assertRaises(ValueError, '@charset "css";div{}'.encode().decode, "css")

    def test_encoder(self):
        "codec.encoder"

        def check(encoding, input='@charset "x";g\xfcrk\u20ac{}'):
            outputencoding = encoding
            if outputencoding == "utf-8-sig":
                outputencoding = "utf-8"

            # Check stateless encoder with encoding autodetection
            e = codecs.getencoder("css")
            inputdecl = input.replace('"x"', '"%s"' % encoding)
            outputdecl = input.replace('"x"', '"%s"' % outputencoding)
            self.assertEqual(e(inputdecl)[0].decode(encoding), outputdecl)

            # Check stateless encoder with specified encoding
            self.assertEqual(
                e(input, encoding=encoding)[0].decode(encoding), outputdecl
            )

            if hasattr(codec, "getincrementalencoder"):
                # Check incremental encoder with encoding autodetection
                ie = codecs.getincrementalencoder("css")()
                self.assertEqual(
                    "".join(ie.iterencode(inputdecl)).decode(encoding), outputdecl
                )

                # Check incremental encoder with specified encoding
                ie = codecs.getincrementalencoder("css")(encoding=encoding)
                self.assertEqual(
                    "".join(ie.iterencode(input)).decode(encoding), outputdecl
                )

            # Check stream writer with encoding autodetection
            q = Queue()
            sw = codecs.getwriter("css")(q)
            for c in inputdecl:  # .encode(outputencoding): # TODO: .encode()???
                sw.write(c)
            self.assertEqual(
                q.read().decode(encoding), input.replace('"x"', '"%s"' % outputencoding)
            )

            # Check stream writer with specified encoding
            q = Queue()
            sw = codecs.getwriter("css")(q, encoding=encoding)
            for c in input:
                sw.write(c)
            self.assertEqual(
                q.read().decode(encoding), input.replace('"x"', '"%s"' % outputencoding)
            )

        # Autodetectable encodings
        check("utf-8-sig")
        check("utf-16")
        check("utf-16-le")
        check("utf-16-be")
        if haveutf32:
            check("utf-32")
            check("utf-32-le")
            check("utf-32-be")
        check("utf-8")
        check("iso-8859-1", '@charset "x";g\xfcrk{}')
        check("iso-8859-15")
        check("cp1252")

        # No recursion
        self.assertRaises(ValueError, '@charset "css";div{}'.encode, "css")

    def test_decode_force(self):
        "codec.decode (force)"
        info = codecs.lookup("css")

        def decodeall(input, **kwargs):
            # Py 2.5: info.decode('@charset "utf-8"; x')
            return info[1](input, **kwargs)[0]

        def incdecode(input, **kwargs):
            decoder = info.incrementaldecoder(**kwargs)
            return decoder.decode(input)

        def streamdecode(input, **kwargs):
            stream = iostream(input)  # py3 .decode('utf-8') but still error?!
            reader = info.streamreader(stream, **kwargs)
            return reader.read()

        for d in (decodeall, incdecode, streamdecode):
            # input = '@charset "utf-8"; \xc3\xbf'
            # output = u'@charset "utf-8"; \xff'
            # self.assertEqual(d(input), output)
            #
            # input = '@charset "utf-8"; \xc3\xbf'
            # output = u'@charset "iso-8859-1"; \xc3\xbf'
            # self.assertEqual(d(input, encoding="iso-8859-1", force=True), output)
            #
            # input = '\xc3\xbf'
            # output = u'\xc3\xbf'
            # self.assertEqual(d(input, encoding="iso-8859-1", force=True), output)
            #
            # input = '@charset "utf-8"; \xc3\xbf'
            # output = u'@charset "utf-8"; \xff'
            # self.assertEqual(d(input, encoding="iso-8859-1", force=False), output)

            input = '@charset "utf-8"; \xff'.encode('utf-8')
            output = '@charset "utf-8"; \xff'
            self.assertEqual(d(input), output)

            # input = b'@charset "utf-8"; \xc3\xbf'
            input = '@charset "utf-8"; \xff'.encode('utf-8')
            output = '@charset "iso-8859-1"; \xc3\xbf'
            self.assertEqual(d(input, encoding="iso-8859-1", force=True), output)

            # input = b'\xc3\xbf'
            input = '\xff'.encode('utf-8')
            output = '\xc3\xbf'
            self.assertEqual(d(input, encoding="iso-8859-1", force=True), output)

            # input = b'@charset "utf-8"; \xc3\xbf'
            input = '@charset "utf-8"; \xff'.encode('utf-8')
            output = '@charset "utf-8"; \xff'
            self.assertEqual(d(input, encoding="iso-8859-1", force=False), output)


if __name__ == '__main__':
    import unittest

    unittest.main()
