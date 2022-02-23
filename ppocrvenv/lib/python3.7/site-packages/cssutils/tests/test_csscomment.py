"""Testcases for cssutils.css.CSSComment"""

import xml
from . import test_cssrule
import cssutils.css


class CSSCommentTestCase(test_cssrule.CSSRuleTestCase):
    def setUp(self):
        super(CSSCommentTestCase, self).setUp()
        self.r = cssutils.css.CSSComment()
        self.rRO = cssutils.css.CSSComment(readonly=True)
        self.r_type = cssutils.css.CSSComment.COMMENT
        self.r_typeString = 'COMMENT'

    def test_init(self):
        "CSSComment.type and init"
        super(CSSCommentTestCase, self).test_init()

    def test_csstext(self):
        "CSSComment.cssText"
        tests = {
            '/*öäüß€ÖÄÜ*/': '/*\xf6\xe4\xfc\xdf\u20ac\xd6\xc4\xdc*/',
            '/*x*/': None,
            '/* x */': None,
            '/*\t12\n*/': None,
            '/* /* */': None,
            '/* \\*/': None,
            '/*"*/': None,
            '''/*"
            */''': None,
            '/** / ** //*/': None,
        }
        self.do_equal_r(tests)  # set cssText
        tests.update(
            {
                '/*x': '/*x*/',
                '\n /*': '/**/',
            }
        )
        self.do_equal_p(tests)  # parse

        tests = {
            '/* */ ': xml.dom.InvalidModificationErr,
            '/* *//**/': xml.dom.InvalidModificationErr,
            '/* */1': xml.dom.InvalidModificationErr,
            '/* */ */': xml.dom.InvalidModificationErr,
            '  */ /* ': xml.dom.InvalidModificationErr,
            '*/': xml.dom.InvalidModificationErr,
            '@x /* x */': xml.dom.InvalidModificationErr,
        }
        self.do_raise_r(tests)  # set cssText
        # no raising of error possible?
        # self.do_raise_p(tests) # parse

    def test_InvalidModificationErr(self):
        "CSSComment.cssText InvalidModificationErr"
        self._test_InvalidModificationErr('/* comment */')

    def test_reprANDstr(self):
        "CSSComment.__repr__(), .__str__()"
        text = '/* test */'

        s = cssutils.css.CSSComment(cssText=text)

        s2 = eval(repr(s))
        self.assertTrue(isinstance(s2, s.__class__))
        self.assertTrue(text == s2.cssText)


if __name__ == '__main__':
    import unittest

    unittest.main()
