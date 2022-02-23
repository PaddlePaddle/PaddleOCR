"""Testcases for cssutils.css.CSSPageRule"""

import xml.dom
from . import test_cssrule
import cssutils


class CSSVariablesRuleTestCase(test_cssrule.CSSRuleTestCase):
    def setUp(self):
        super(CSSVariablesRuleTestCase, self).setUp()
        self.r = cssutils.css.CSSVariablesRule()
        self.rRO = cssutils.css.CSSVariablesRule(readonly=True)
        self.r_type = cssutils.css.CSSPageRule.VARIABLES_RULE
        self.r_typeString = 'VARIABLES_RULE'

        cssutils.ser.prefs.resolveVariables = False

    def test_init(self):
        "CSSVariablesRule.__init__()"
        super(CSSVariablesRuleTestCase, self).test_init()

        r = cssutils.css.CSSVariablesRule()
        self.assertEqual(cssutils.css.CSSVariablesDeclaration, type(r.variables))
        self.assertEqual(r, r.variables.parentRule)

        # until any variables
        self.assertEqual('', r.cssText)

        # only possible to set @... similar name
        self.assertRaises(xml.dom.InvalidModificationErr, self.r._setAtkeyword, 'x')

    def test_InvalidModificationErr(self):
        "CSSVariablesRule.cssText InvalidModificationErr"
        self._test_InvalidModificationErr('@variables')
        tests = {
            '@var {}': xml.dom.InvalidModificationErr,
        }
        self.do_raise_r(tests)

    def test_incomplete(self):
        "CSSVariablesRule (incomplete)"
        tests = {
            '@variables { ': '',  # no } and no content
            '@variables { x: red': '@variables {\n    x: red\n    }',  # no }
        }
        self.do_equal_p(tests)  # parse

    def test_cssText(self):
        "CSSVariablesRule"
        EXP = '@variables {\n    margin: 0\n    }'
        tests = {
            '@variables {}': '',
            '@variables     {margin:0;}': EXP,
            '@variables     {margin:0}': EXP,
            '@VaRIables {   margin    :   0   ;    }': EXP,
            '@\\VaRIables {    margin : 0    }': EXP,
            '@variables {a:1;b:2}': '@variables {\n    a: 1;\n    b: 2\n    }',
            # comments
            '@variables   /*1*/   {margin:0;}': '@variables /*1*/ {\n    margin: 0\n    }',
            '@variables/*1*/{margin:0;}': '@variables /*1*/ {\n    margin: 0\n    }',
        }
        self.do_equal_r(tests)
        self.do_equal_p(tests)

    def test_media(self):
        "CSSVariablesRule.media"
        r = cssutils.css.CSSVariablesRule()
        self.assertRaises(AttributeError, r.__getattribute__, 'media')
        self.assertRaises(AttributeError, r.__setattr__, 'media', '?')

    def test_variables(self):
        "CSSVariablesRule.variables"
        r = cssutils.css.CSSVariablesRule(
            variables=cssutils.css.CSSVariablesDeclaration('x: 1')
        )
        self.assertEqual(r, r.variables.parentRule)

        # cssText
        r = cssutils.css.CSSVariablesRule()
        r.cssText = '@variables { x: 1 }'
        vars1 = r.variables
        self.assertEqual(r, r.variables.parentRule)
        self.assertEqual(vars1, r.variables)
        self.assertEqual(r.variables.cssText, 'x: 1')
        self.assertEqual(r.cssText, '@variables {\n    x: 1\n    }')

        r.cssText = '@variables {y:2}'
        self.assertEqual(r, r.variables.parentRule)
        self.assertNotEqual(vars1, r.variables)
        self.assertEqual(r.variables.cssText, 'y: 2')
        self.assertEqual(r.cssText, '@variables {\n    y: 2\n    }')

        vars2 = r.variables

        # fail
        try:
            r.cssText = '@variables {$:1}'
        except xml.dom.DOMException:
            pass

        self.assertEqual(vars2, r.variables)
        self.assertEqual(r.variables.cssText, 'y: 2')
        self.assertEqual(r.cssText, '@variables {\n    y: 2\n    }')

        # var decl
        vars3 = cssutils.css.CSSVariablesDeclaration('z: 3')
        r.variables = vars3

        self.assertEqual(r, r.variables.parentRule)
        self.assertEqual(vars3, r.variables)
        self.assertEqual(r.variables.cssText, 'z: 3')
        self.assertEqual(r.cssText, '@variables {\n    z: 3\n    }')

        # string
        r.variables = 'a: x'
        self.assertNotEqual(vars3, r.variables)
        self.assertEqual(r, r.variables.parentRule)
        self.assertEqual(r.variables.cssText, 'a: x')
        self.assertEqual(r.cssText, '@variables {\n    a: x\n    }')
        vars4 = r.variables

        # string fail
        try:
            r.variables = '$: x'
        except xml.dom.DOMException:
            pass
        self.assertEqual(vars4, r.variables)
        self.assertEqual(r, r.variables.parentRule)
        self.assertEqual(r.variables.cssText, 'a: x')
        self.assertEqual(r.cssText, '@variables {\n    a: x\n    }')

    def test_reprANDstr(self):
        "CSSVariablesRule.__repr__(), .__str__()"
        r = cssutils.css.CSSVariablesRule()
        r.cssText = '@variables { xxx: 1 }'
        self.assertTrue('xxx' in str(r))

        r2 = eval(repr(r))
        self.assertTrue(isinstance(r2, r.__class__))
        self.assertTrue(r.cssText == r2.cssText)


if __name__ == '__main__':
    import unittest

    unittest.main()
