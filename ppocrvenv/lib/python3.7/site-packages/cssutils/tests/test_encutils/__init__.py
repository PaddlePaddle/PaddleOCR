"""
tests for encutils.py
"""
import http.client
from io import StringIO
import unittest

try:
    import cssutils.encutils as encutils
except ImportError:
    import encutils

# helper log
log = encutils.buildlog(stream=StringIO())


class AutoEncodingTestCase(unittest.TestCase):
    def _fakeRes(self, content):
        "build a fake HTTP response"

        class FakeRes:
            def __init__(self, content):
                self._info = http.client.HTTPMessage()
                # Adjust to testdata.
                items = content.split(':')
                if len(items) > 1:
                    # Get the type by just
                    # using the data at the end.
                    t = items[-1].strip()
                    self._info.set_type(t)

            def info(self):
                return self._info

            def read(self):
                return content

        return FakeRes(content)

    def test_getTextTypeByMediaType(self):
        "encutils._getTextTypeByMediaType"
        tests = {
            'application/xml': encutils._XML_APPLICATION_TYPE,
            'application/xml-dtd': encutils._XML_APPLICATION_TYPE,
            'application/xml-external-parsed-entity': encutils._XML_APPLICATION_TYPE,
            'application/xhtml+xml': encutils._XML_APPLICATION_TYPE,
            'text/xml': encutils._XML_TEXT_TYPE,
            'text/xml-external-parsed-entity': encutils._XML_TEXT_TYPE,
            'text/xhtml+xml': encutils._XML_TEXT_TYPE,
            'text/html': encutils._HTML_TEXT_TYPE,
            'text/css': encutils._TEXT_UTF8,
            'text/plain': encutils._TEXT_TYPE,
            'x/x': encutils._OTHER_TYPE,
            'ANYTHING': encutils._OTHER_TYPE,
        }
        for test, exp in list(tests.items()):
            self.assertEqual(exp, encutils._getTextTypeByMediaType(test, log=log))

    def test_getTextType(self):
        "encutils._getTextType"
        tests = {
            '\x00\x00\xFE\xFF<?xml version="1.0"': encutils._XML_APPLICATION_TYPE,
            '\xFF\xFE\x00\x00<?xml version="1.0"': encutils._XML_APPLICATION_TYPE,
            '\xFE\xFF<?xml version="1.0"': encutils._XML_APPLICATION_TYPE,
            '\xFF\xFE<?xml version="1.0"': encutils._XML_APPLICATION_TYPE,
            '\xef\xbb\xbf<?xml version="1.0"': encutils._XML_APPLICATION_TYPE,
            '<?xml version="1.0"': encutils._XML_APPLICATION_TYPE,
            '\x00\x00\xFE\xFFanything': encutils._OTHER_TYPE,
            '\xFF\xFE\x00\x00anything': encutils._OTHER_TYPE,
            '\xFE\xFFanything': encutils._OTHER_TYPE,
            '\xFF\xFEanything': encutils._OTHER_TYPE,
            '\xef\xbb\xbfanything': encutils._OTHER_TYPE,
            'x/x': encutils._OTHER_TYPE,
            'ANYTHING': encutils._OTHER_TYPE,
        }
        for test, exp in list(tests.items()):
            self.assertEqual(exp, encutils._getTextType(test, log=log))

    def test_encodingByMediaType(self):
        "encutils.encodingByMediaType"
        tests = {
            'application/xml': 'utf-8',
            'application/xml-dtd': 'utf-8',
            'application/xml-external-parsed-entity': 'utf-8',
            'application/ANYTHING+xml': 'utf-8',
            '  application/xml  ': 'utf-8',
            'text/xml': 'ascii',
            'text/xml-external-parsed-entity': 'ascii',
            'text/ANYTHING+xml': 'ascii',
            'text/html': 'iso-8859-1',
            'text/css': 'utf-8',
            'text/plain': 'iso-8859-1',
            'ANYTHING': None,
        }
        for test, exp in list(tests.items()):
            self.assertEqual(exp, encutils.encodingByMediaType(test, log=log))

    def test_getMetaInfo(self):
        "encutils.getMetaInfo"
        tests = {
            """<meta tp-equiv='Content-Type' content='text/html; charset=ascii'>""": (
                None,
                None,
            ),
            """<meta http-equiv='ontent-Type' content='text/html; charset=ascii'>""": (
                None,
                None,
            ),
            """<meta http-equiv='Content-Type' content='text/html'>""": (
                'text/html',
                None,
            ),
            """<meta content='text/html' http-equiv='Content-Type'>""": (
                'text/html',
                None,
            ),
            """<meta content='text/html;charset=ascii' http-equiv='Content-Type'>""": (
                'text/html',
                'ascii',
            ),
            """<meta http-equiv='Content-Type' content='text/html ;charset=ascii'>""": (
                'text/html',
                'ascii',
            ),
            """<meta content='text/html;charset=iso-8859-1' http-equiv='Content-Type'>""": (
                'text/html',
                'iso-8859-1',
            ),
            """<meta http-equiv="Content-Type" content="text/html;charset = ascii">""": (
                'text/html',
                'ascii',
            ),
            """<meta http-equiv="Content-Type" content="text/html;charset=ascii;x=2">""": (
                'text/html',
                'ascii',
            ),
            """<meta http-equiv="Content-Type" content="text/html;x=2;charset=ascii">""": (
                'text/html',
                'ascii',
            ),
            """<meta http-equiv="Content-Type" content="text/html;x=2;charset=ascii;y=2">""": (
                'text/html',
                'ascii',
            ),
            """<meta http-equiv='Content-Type' content="text/html;charset=ascii">""": (
                'text/html',
                'ascii',
            ),
            """<meta http-equiv='Content-Type' content='text/html;charset=ascii'  />""": (
                'text/html',
                'ascii',
            ),
            """<meta http-equiv = " Content-Type" content = " text/html;charset=ascii " >""": (
                'text/html',
                'ascii',
            ),
            """<meta http-equiv = " \n Content-Type " content = "  \t text/html   ;  charset=ascii " >""": (
                'text/html',
                'ascii',
            ),
            """<meta content="text/html;charset=ascii" http-equiv="Content-Type">""": (
                'text/html',
                'ascii',
            ),
            """<meta content="text/html;charset=ascii" http-equiv="cONTENT-type">""": (
                'text/html',
                'ascii',
            ),
            """raises exception: </ >""": (None, None),
            """<meta content="text/html;charset=ascii" http-equiv="cONTENT-type">
                </ >""": (
                'text/html',
                'ascii',
            ),
            """</ >
                <meta content="text/html;charset=ascii" http-equiv="cONTENT-type">""": (
                'text/html',
                'ascii',
            ),
            # py 2.7.3 fixed HTMLParser so:  (None, None)
            """<meta content="text/html" http-equiv="cONTENT-type">
                </ >
                <meta content="text/html;charset=ascii" http-equiv="cONTENT-type">""": (
                'text/html',
                None,
            ),
        }
        for test, exp in list(tests.items()):
            self.assertEqual(exp, encutils.getMetaInfo(test, log=log))

    def test_detectXMLEncoding(self):
        "encutils.detectXMLEncoding"
        tests = (
            # BOM
            (('utf_32_be'), '\x00\x00\xFE\xFFanything'),
            (('utf_32_le'), '\xFF\xFE\x00\x00anything'),
            (('utf_16_be'), '\xFE\xFFanything'),
            (('utf_16_le'), '\xFF\xFEanything'),
            (('utf-8'), '\xef\xbb\xbfanything'),
            # encoding=
            (('ascii'), '<?xml version="1.0" encoding="ascii" ?>'),
            (('ascii'), "<?xml version='1.0' encoding='ascii' ?>"),
            (('iso-8859-1'), "<?xml version='1.0' encoding='iso-8859-1' ?>"),
            # default
            (('utf-8'), '<?xml version="1.0" ?>'),
            (('utf-8'), '<?xml version="1.0"?><x encoding="ascii"/>'),
        )
        for exp, test in tests:
            self.assertEqual(exp, encutils.detectXMLEncoding(test, log=log))

    def test_tryEncodings(self):
        "encutils.tryEncodings"
        try:
            tests = [
                ('ascii', 'abc'.encode('ascii')),
                ('windows-1252', '€'.encode('windows-1252')),
                ('ascii', '1'.encode('utf-8')),
            ]
        except ImportError:
            tests = [
                ('ascii', 'abc'.encode('ascii')),
                ('windows-1252', '€'.encode('windows-1252')),
                ('iso-8859-1', 'äöüß'.encode('iso-8859-1')),
                ('iso-8859-1', 'äöüß'.encode('windows-1252')),
                # ('utf-8', u'\u1111'.encode('utf-8'))
            ]
        for exp, test in tests:
            self.assertEqual(exp, encutils.tryEncodings(test))

    def test_getEncodingInfo(self):
        "encutils.getEncodingInfo"
        # (expectedencoding, expectedmismatch): (httpheader, filecontent)
        tests = [
            # --- application/xhtml+xml ---
            # header default and XML default
            (
                ('utf-8', False),
                (
                    '''Content-Type: application/xhtml+xml''',
                    '''<?xml version="1.0" ?>
                    <example>
                        <meta http-equiv="Content-Type"
                            content="application/xhtml+xml"/>
                    </example>''',
                ),
            ),
            # XML default
            (
                ('utf-8', False),
                (
                    None,
                    '''<?xml version="1.0" ?>
                    <example>
                        <meta http-equiv="Content-Type"
                            content="application/xhtml+xml"/>
                    </example>''',
                ),
            ),
            # meta is ignored!
            (
                ('utf-8', False),
                (
                    '''Content-Type: application/xhtml+xml''',
                    '''<?xml version="1.0" ?>
                    <example>
                        <meta http-equiv="Content-Type"
                            content="application/xhtml+xml;charset=iso_M"/>
                    </example>''',
                ),
            ),
            # header enc and XML default
            (
                ('iso-h', True),
                (
                    '''Content-Type: application/xhtml+xml;charset=iso-H''',
                    '''<?xml version="1.0" ?>
                    <example>
                        <meta http-equiv="Content-Type"
                            content="application/xhtml+xml"/>
                    </example>''',
                ),
            ),
            # mismatch header and XML explicit, header wins
            (
                ('iso-h', True),
                (
                    '''Content-Type: application/xhtml+xml;charset=iso-H''',
                    '''<?xml version="1.0" encoding="iso-X" ?>
                    <example/>''',
                ),
            ),
            # header == XML, meta ignored!
            (
                ('iso-h', False),
                (
                    '''Content-Type: application/xhtml+xml;charset=iso-H''',
                    '''<?xml version="1.0" encoding="iso-h" ?>
                    <example>
                        <meta http-equiv="Content-Type"
                            content="application/xhtml+xml;charset=iso_M"/>
                    </example>''',
                ),
            ),
            # XML only, meta ignored!
            (
                ('iso-x', False),
                (
                    '''Content-Type: application/xhtml+xml''',
                    '''<?xml version="1.0" encoding="iso-X" ?>
                    <example>
                        <meta http-equiv="Content-Type"
                            content="application/xhtml+xml;charset=iso_M"/>
                    </example>''',
                ),
            ),
            # no text or not enough text:
            (('iso-h', False), ('Content-Type: application/xml;charset=iso-h', '1')),
            (('utf-8', False), ('Content-Type: application/xml', None)),
            ((None, False), ('Content-Type: application/xml', '1')),
            # --- text/xml ---
            # default enc
            (
                ('ascii', False),
                (
                    '''Content-Type: text/xml''',
                    '''<?xml version="1.0" ?>
                    <example>
                        <meta http-equiv="Content-Type"
                            content="text/xml"/>
                    </example>''',
                ),
            ),
            # default as XML ignored and meta completely ignored
            (
                ('ascii', False),
                (
                    '''Content-Type: text/xml''',
                    '''<?xml version="1.0" encoding="iso-X" ?>
                    <example>
                        <meta http-equiv="Content-Type"
                            content="text/xml;charset=iso_M"/>
                    </example>''',
                ),
            ),
            (('ascii', False), ('Content-Type: text/xml', '1')),
            (('ascii', False), ('Content-Type: text/xml', None)),
            # header enc
            (
                ('iso-h', False),
                (
                    '''Content-Type: text/xml;charset=iso-H''',
                    '''<?xml version="1.0" ?>
                    <example>
                        <meta http-equiv="Content-Type"
                            content="text/xml"/>
                    </example>''',
                ),
            ),
            # header only, XML and meta ignored!
            (
                ('iso-h', False),
                (
                    '''Content-Type: text/xml;charset=iso-H''',
                    '''<?xml version="1.0" encoding="iso-X" ?>
                    <example/>''',
                ),
            ),
            (
                ('iso-h', False),
                (
                    '''Content-Type: text/xml;charset=iso-H''',
                    '''<?xml version="1.0"  encoding="iso-h" ?>
                    <example>
                        <meta http-equiv="Content-Type"
                            content="text/xml;charset=iso_M"/>
                    </example>''',
                ),
            ),
            # --- text/html ---
            # default enc
            (
                ('iso-8859-1', False),
                (
                    'Content-Type: text/html;',
                    '''<meta http-equiv="Content-Type"
                                        content="text/html">''',
                ),
            ),
            (('iso-8859-1', False), ('Content-Type: text/html;', None)),
            # header enc
            (
                ('iso-h', False),
                (
                    'Content-Type: text/html;charset=iso-H',
                    '''<meta http-equiv="Content-Type"
                                    content="text/html">''',
                ),
            ),
            # meta enc
            (
                ('iso-m', False),
                (
                    'Content-Type: text/html',
                    '''<meta http-equiv="Content-Type"
                                    content="text/html;charset=iso-m">''',
                ),
            ),
            # mismatch header and meta, header wins
            (
                ('iso-h', True),
                (
                    'Content-Type: text/html;charset=iso-H',
                    '''<meta http-equiv="Content-Type"
                                    content="text/html;charset=iso-m">''',
                ),
            ),
            # no header:
            (
                (None, False),
                (
                    None,
                    '''<meta http-equiv="Content-Type"
                                content="text/html;charset=iso-m">''',
                ),
            ),
            # no encoding at all
            (
                (None, False),
                (
                    None,
                    '''<meta http-equiv="Content-Type"
                                content="text/html">''',
                ),
            ),
            ((None, False), (None, '''text''')),
            # --- no header ---
            ((None, False), (None, '')),
            (('iso-8859-1', False), ('''NoContentType''', '''OnlyText''')),
            (('iso-8859-1', False), ('Content-Type: text/html;', None)),
            (('iso-8859-1', False), ('Content-Type: text/html;', '1')),
            # XML
            (('utf-8', False), (None, '''<?xml version=''')),
            (('iso-x', False), (None, '''<?xml version="1.0" encoding="iso-X"?>''')),
            # meta ignored
            (
                ('utf-8', False),
                (
                    None,
                    '''<?xml version="1.0" ?>
                                    <html><meta http-equiv="Content-Type"
                                    content="text/html;charset=iso-m"></html>''',
                ),
            ),
            (('utf-8', False), ('Content-Type: text/css;', '1')),
            (('iso-h', False), ('Content-Type: text/css;charset=iso-h', '1')),
            # only header is used by encutils
            (('utf-8', False), ('Content-Type: text/css', '@charset "ascii";')),
        ]
        for exp, test in tests:
            header, text = test
            if header:
                res = encutils.getEncodingInfo(self._fakeRes(header), text)
            else:
                res = encutils.getEncodingInfo(text=text)

            res = (res.encoding, res.mismatch)
            self.assertEqual(exp, res)


if __name__ == '__main__':
    unittest.main()
