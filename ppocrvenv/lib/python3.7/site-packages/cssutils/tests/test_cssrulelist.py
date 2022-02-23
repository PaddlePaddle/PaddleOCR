"""Testcases for cssutils.css.CSSRuleList"""

from . import basetest
import cssutils


class CSSRuleListTestCase(basetest.BaseTestCase):
    def test_init(self):
        "CSSRuleList.__init__()"
        r = cssutils.css.CSSRuleList()
        self.assertEqual(0, r.length)
        self.assertEqual(None, r.item(2))

        # subclasses list but all setting options like append, extend etc
        # need to be added to an instance of this class by a using class!
        self.assertRaises(NotImplementedError, r.append, 1)

    def test_rulesOfType(self):
        "CSSRuleList.rulesOfType()"
        s = cssutils.parseString(
            '''
        /*c*/
        @namespace "a";
        a { color: red}
        b { left: 0 }'''
        )

        c = list(s.cssRules.rulesOfType(cssutils.css.CSSRule.COMMENT))
        self.assertEqual(1, len(c))
        self.assertEqual('/*c*/', c[0].cssText)

        r = list(s.cssRules.rulesOfType(cssutils.css.CSSRule.STYLE_RULE))
        self.assertEqual(2, len(r))
        self.assertEqual('b {\n    left: 0\n    }', r[1].cssText)


if __name__ == '__main__':
    import unittest

    unittest.main()
