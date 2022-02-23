"""Testcases for cssutils.helper"""

from . import basetest
from cssutils.helper import normalize, string, stringvalue, uri, urivalue


class HelperTestCase(basetest.BaseTestCase):
    def test_normalize(self):
        "helper._normalize()"
        tests = {
            'abcdefg ABCDEFG äöüß€ AÖÜ': r'abcdefg abcdefg äöüß€ aöü',
            r'\ga\Ga\\\ ': r'gaga\ ',
            r'0123456789': r'0123456789',
            r'"\x"': r'"x"',
            # unicode escape seqs should have been done by
            # the tokenizer...
        }
        for test, exp in list(tests.items()):
            self.assertEqual(normalize(test), exp)
            # static too
            self.assertEqual(normalize(test), exp)

    #    def test_normalnumber(self):
    #        "helper.normalnumber()"
    #        tests = {
    #                 '0': '0',
    #                 '00': '0',
    #                 '0.0': '0',
    #                 '00.0': '0',
    #                 '1': '1',
    #                 '01': '1',
    #                 '00.1': '0.1',
    #                 '0.00001': '0.00001',
    #                 '-0': '0',
    #                 '-00': '0',
    #                 '-0.0': '0',
    #                 '-00.0': '0',
    #                 '-1': '-1',
    #                 '-01': '-1',
    #                 '-00.1': '-0.1',
    #                 '-0.00001': '-0.00001',
    #                 }
    #        for test, exp in tests.items():
    #            self.assertEqual(exp, normalnumber(test))

    def test_string(self):
        "helper.string()"
        self.assertEqual('"x"', string('x'))
        self.assertEqual('"1 2ä€"', string('1 2ä€'))
        self.assertEqual(r'''"'"''', string("'"))
        self.assertEqual(r'"\""', string('"'))
        # \n = 0xa, \r = 0xd, \f = 0xc
        self.assertEqual(
            r'"\a "',
            string(
                '''
'''
            ),
        )
        self.assertEqual(r'"\c "', string('\f'))
        self.assertEqual(r'"\d "', string('\r'))
        self.assertEqual(r'"\d \a "', string('\r\n'))

    def test_stringvalue(self):
        "helper.stringvalue()"
        self.assertEqual('x', stringvalue('"x"'))
        self.assertEqual('"', stringvalue('"\\""'))
        self.assertEqual(r'x', stringvalue(r"\x "))

        # escapes should have been done by tokenizer
        # so this shoule not happen at all:
        self.assertEqual(r'a', stringvalue(r"\a "))

    def test_uri(self):
        "helper.uri()"
        self.assertEqual('url(x)', uri('x'))
        self.assertEqual('url("(")', uri('('))
        self.assertEqual('url(")")', uri(')'))
        self.assertEqual('url(" ")', uri(' '))
        self.assertEqual('url(";")', uri(';'))
        self.assertEqual('url(",")', uri(','))
        self.assertEqual('url("x)x")', uri('x)x'))

    def test_urivalue(self):
        "helper.urivalue()"
        self.assertEqual('x', urivalue('url(x)'))
        self.assertEqual('x', urivalue('url("x")'))
        self.assertEqual(')', urivalue('url(")")'))


if __name__ == '__main__':
    import unittest

    unittest.main()
