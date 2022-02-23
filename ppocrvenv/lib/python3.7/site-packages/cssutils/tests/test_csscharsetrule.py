"""Testcases for cssutils.css.CSSCharsetRule"""

import re
import xml.dom
from . import test_cssrule
import cssutils.css


class CSSCharsetRuleTestCase(test_cssrule.CSSRuleTestCase):
    def setUp(self):
        super(CSSCharsetRuleTestCase, self).setUp()
        self.r = cssutils.css.CSSCharsetRule()
        self.rRO = cssutils.css.CSSCharsetRule(readonly=True)
        self.r_type = cssutils.css.CSSCharsetRule.CHARSET_RULE
        self.r_typeString = 'CHARSET_RULE'

    def test_init(self):
        "CSSCharsetRule.__init__()"
        super(CSSCharsetRuleTestCase, self).test_init()
        self.assertEqual(None, self.r.encoding)
        self.assertEqual('', self.r.cssText)

        self.assertRaises(xml.dom.InvalidModificationErr, self.r._setCssText, 'xxx')

    def test_InvalidModificationErr(self):
        "CSSCharsetRule InvalidModificationErr"
        self._test_InvalidModificationErr('@charset')

    def test_init_encoding(self):
        "CSSCharsetRule.__init__(encoding)"
        for enc in (None, 'UTF-8', 'utf-8', 'iso-8859-1', 'ascii'):
            r = cssutils.css.CSSCharsetRule(enc)
            if enc is None:
                self.assertEqual(None, r.encoding)
                self.assertEqual('', r.cssText)
            else:
                self.assertEqual(enc.lower(), r.encoding)
                self.assertEqual('@charset "%s";' % enc.lower(), r.cssText)

        for enc in (' ascii ', ' ascii', 'ascii '):
            self.assertRaisesEx(
                xml.dom.SyntaxErr,
                cssutils.css.CSSCharsetRule,
                enc,
                exc_pattern=re.compile("Syntax Error"),
            )

        for enc in ('unknown',):
            self.assertRaisesEx(
                xml.dom.SyntaxErr,
                cssutils.css.CSSCharsetRule,
                enc,
                exc_pattern=re.compile(r"Unknown \(Python\) encoding"),
            )

    def test_encoding(self):
        "CSSCharsetRule.encoding"
        for enc in ('UTF-8', 'utf-8', 'iso-8859-1', 'ascii'):
            self.r.encoding = enc
            self.assertEqual(enc.lower(), self.r.encoding)
            self.assertEqual('@charset "%s";' % enc.lower(), self.r.cssText)

        for enc in (None, ' ascii ', ' ascii', 'ascii '):
            self.assertRaisesEx(
                xml.dom.SyntaxErr,
                self.r.__setattr__,
                'encoding',
                enc,
                exc_pattern=re.compile("Syntax Error"),
            )

        for enc in ('unknown',):
            self.assertRaisesEx(
                xml.dom.SyntaxErr,
                self.r.__setattr__,
                'encoding',
                enc,
                exc_pattern=re.compile(r"Unknown \(Python\) encoding"),
            )

    def test_cssText(self):
        """CSSCharsetRule.cssText

        setting cssText is ok to use @CHARSET or other but a file
        using parse MUST use ``@charset "ENCODING";``
        """
        tests = {
            '@charset "utf-8";': None,
            "@charset 'utf-8';": '@charset "utf-8";',
        }
        self.do_equal_r(tests)
        self.do_equal_p(tests)  # also parse

        tests = {
            # token is "@charset " with space!
            '@charset;"': xml.dom.InvalidModificationErr,
            '@CHARSET "UTF-8";': xml.dom.InvalidModificationErr,
            '@charset "";': xml.dom.SyntaxErr,
            '''@charset /*1*/"utf-8"/*2*/;''': xml.dom.SyntaxErr,
            '''@charset /*1*/"utf-8";''': xml.dom.SyntaxErr,
            '''@charset "utf-8"/*2*/;''': xml.dom.SyntaxErr,
            '@charset { utf-8 }': xml.dom.SyntaxErr,
            '@charset "utf-8"': xml.dom.SyntaxErr,
            '@charset a;': xml.dom.SyntaxErr,
            '@charset /**/;': xml.dom.SyntaxErr,
            # trailing content
            '@charset "utf-8";s': xml.dom.SyntaxErr,
            '@charset "utf-8";/**/': xml.dom.SyntaxErr,
            '@charset "utf-8"; ': xml.dom.SyntaxErr,
            # comments do not work in this rule!
            '@charset "utf-8"/*1*//*2*/;': xml.dom.SyntaxErr,
        }
        self.do_raise_r(tests)

    def test_repr(self):
        "CSSCharsetRule.__repr__()"
        self.r.encoding = 'utf-8'
        self.assertTrue('utf-8' in repr(self.r))

    def test_reprANDstr(self):
        "CSSCharsetRule.__repr__(), .__str__()"
        encoding = 'utf-8'

        s = cssutils.css.CSSCharsetRule(encoding=encoding)

        self.assertTrue(encoding in str(s))

        s2 = eval(repr(s))
        self.assertTrue(isinstance(s2, s.__class__))
        self.assertTrue(encoding == s2.encoding)


if __name__ == '__main__':
    import unittest

    unittest.main()
