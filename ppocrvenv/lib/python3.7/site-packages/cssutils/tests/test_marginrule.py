"""Testcases for cssutils.css.CSSPageRule"""

import xml.dom
from . import test_cssrule
import cssutils


class MarginRuleTestCase(test_cssrule.CSSRuleTestCase):
    def setUp(self):
        super(MarginRuleTestCase, self).setUp()

        cssutils.ser.prefs.useDefaults()
        self.r = cssutils.css.MarginRule()
        self.rRO = cssutils.css.MarginRule(readonly=True)
        self.r_type = cssutils.css.MarginRule.MARGIN_RULE
        self.r_typeString = 'MARGIN_RULE'

    def tearDown(self):
        cssutils.ser.prefs.useDefaults()

    def test_init(self):
        "MarginRule.__init__()"

        r = cssutils.css.MarginRule()
        self.assertEqual(r.margin, None)
        self.assertEqual(r.atkeyword, None)
        self.assertEqual(r._keyword, None)
        self.assertEqual(r.style.cssText, '')
        self.assertEqual(r.cssText, '')

        r = cssutils.css.MarginRule(margin='@TOP-left')
        self.assertEqual(r.margin, '@top-left')
        self.assertEqual(r.atkeyword, '@top-left')
        self.assertEqual(r._keyword, '@TOP-left')
        self.assertEqual(r.style.cssText, '')
        self.assertEqual(r.cssText, '')

        self.assertRaises(xml.dom.InvalidModificationErr, cssutils.css.MarginRule, '@x')

    def test_InvalidModificationErr(self):
        "MarginRule.cssText InvalidModificationErr"
        # TODO: !!!

    #        self._test_InvalidModificationErr(u'@top-left')
    #        tests = {
    #            u'@x {}': xml.dom.InvalidModificationErr,
    #            }
    #        self.do_raise_r(tests)

    def test_incomplete(self):
        "MarginRule (incomplete)"
        # must be inside @page as not valid outside
        tests = {
            '@page { @top-left { ': '',  # no } and no content
            '@page { @top-left { /*1*/ ': '',  # no } and no content
            '@page { @top-left { color: red': '@page {\n    @top-left {\n        color: red\n        }\n    }',
        }
        self.do_equal_p(tests)  # parse

    def test_cssText(self):
        tests = {
            '@top-left {}': '',
            '@top-left { /**/ }': '',
            '@top-left { color: red }': '@top-left {\n    color: red\n    }',
            '@top-left{color:red;}': '@top-left {\n    color: red\n    }',
            '@top-left{color:red}': '@top-left {\n    color: red\n    }',
            '@top-left { color: red; left: 0 }': '@top-left {\n    color: red;\n    left: 0\n    }',
        }
        self.do_equal_r(tests)

        # TODO
        tests.update(
            {
                # false selector
                #            u'@top-left { color:': xml.dom.SyntaxErr, # no }
                #            u'@top-left { color': xml.dom.SyntaxErr, # no }
                #            u'@top-left {': xml.dom.SyntaxErr, # no }
                #            u'@top-left': xml.dom.SyntaxErr, # no }
                #            u'@top-left;': xml.dom.SyntaxErr, # no }
            }
        )

    #        self.do_raise_r(tests) # set cssText

    def test_reprANDstr(self):
        "MarginRule.__repr__(), .__str__()"
        margin = '@top-left'

        s = cssutils.css.MarginRule(margin=margin, style='left: 0')

        self.assertTrue(margin in str(s))

        s2 = eval(repr(s))
        self.assertTrue(isinstance(s2, s.__class__))
        self.assertTrue(margin == s2.margin)


if __name__ == '__main__':
    import unittest

    unittest.main()
