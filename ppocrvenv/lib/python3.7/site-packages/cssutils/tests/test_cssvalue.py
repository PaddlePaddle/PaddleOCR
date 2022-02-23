"""Testcases for cssutils.css.CSSValue and CSSPrimitiveValue."""

#
# from decimal import Decimal # maybe for later tests?
# import xml.dom
# import basetest
# import cssutils
# import types
#
# class CSSValueTestCase(basetest.BaseTestCase):
#
#    def setUp(self):
#        self.r = cssutils.css.CSSValue() # needed for tests
#
#    def test_init(self):
#        "CSSValue.__init__()"
#        v = cssutils.css.CSSValue()
#        self.assertTrue(u'' == v.cssText)
#        self.assertTrue(None is  v.cssValueType)
#        self.assertTrue(None == v.cssValueTypeString)
#
#    def test_escapes(self):
#        "CSSValue Escapes"
#        v = cssutils.css.CSSValue()
#        v.cssText = u'1px'
#        self.assertTrue(v.CSS_PRIMITIVE_VALUE == v.cssValueType)
#        self.assertTrue(v.CSS_PX == v.primitiveType)
#        self.assertTrue(u'1px' == v.cssText)
#
#        v.cssText = u'1PX'
#        self.assertTrue(v.CSS_PRIMITIVE_VALUE == v.cssValueType)
#        self.assertTrue(v.CSS_PX == v.primitiveType)
#        self.assertTrue(u'1px' == v.cssText)
#
#        v.cssText = u'1p\\x'
#        self.assertTrue(v.CSS_PRIMITIVE_VALUE == v.cssValueType)
#        self.assertTrue(v.CSS_PX == v.primitiveType)
#        self.assertTrue(u'1px' == v.cssText)
#
#    def test_cssText(self):
#        "CSSValue.cssText"
#        v = cssutils.css.CSSValue()
#        v.cssText = u'1px'
#        self.assertTrue(v.CSS_PRIMITIVE_VALUE == v.cssValueType)
#        self.assertTrue(v.CSS_PX == v.primitiveType)
#        self.assertTrue(u'1px' == v.cssText)
#
#        v = cssutils.css.CSSValue()
#        v.cssText = u'1px'
#        self.assertTrue(v.CSS_PRIMITIVE_VALUE == v.cssValueType)
#        self.assertTrue(v.CSS_PX == v.primitiveType)
#        self.assertTrue(u'1px' == v.cssText)
#
#        v = cssutils.css.CSSValue()
#        v.cssText = u'a  ,b,  c  ,"d or d", "e, " '
#        self.assertEqual(v.CSS_PRIMITIVE_VALUE, v.cssValueType)
#        self.assertEqual(v.CSS_STRING, v.primitiveType)
#        self.assertEqual(u'a, b, c, "d or d", "e, "', v.cssText)
#
#        v.cssText = u'  1   px    '
#        self.assertTrue(v.CSS_VALUE_LIST == v.cssValueType)
#        self.assertTrue('1 px' == v.cssText)
#
#        v.cssText = u'  normal 1px a, b, "c" end   '
#        self.assertTrue(v.CSS_VALUE_LIST == v.cssValueType)
#        self.assertEqual('normal 1px a, b, "c" end', v.cssText)
#
#        for i, x in enumerate(v):
#            self.assertEqual(x.CSS_PRIMITIVE_VALUE, x.cssValueType)
#            if x == 0:
#                self.assertEqual(x.CSS_IDENT, x.primitiveType)
#                self.assertEqual(u'normal', x.cssText)
#            elif x == 1:
#                self.assertEqual(x.CSS_PX, x.primitiveType)
#                self.assertEqual(u'1px', x.cssText)
#            if x == 2:
#                self.assertEqual(x.CSS_STRING, x.primitiveType)
#                self.assertEqual(u'a, b, "c"', x.cssText)
#            if x == 3:
#                self.assertEqual(x.CSS_IDENT, x.primitiveType)
#                self.assertEqual(u'end', x.cssText)
#
#
#        v = cssutils.css.CSSValue()
#        v.cssText = u'  1   px    '
#        self.assertTrue(v.CSS_VALUE_LIST == v.cssValueType)
#        self.assertTrue(u'1 px' == v.cssText)
#
#        v.cssText = u'expression(document.body.clientWidth > 972 ? "1014px": "100%" )'
#        self.assertTrue(v.CSS_PRIMITIVE_VALUE == v.cssValueType)
#        self.assertTrue(v.CSS_UNKNOWN == v.primitiveType)
#        self.assertEqual(
#            u'expression(document.body.clientWidth > 972?"1014px": "100%")',
#            v.cssText)
#
#    def test_cssText2(self):
#        "CSSValue.cssText 2"
#        tests = {
#            # mix
#            u'a()1,-1,+1,1%,-1%,1px,-1px,"a",a,url(a),#aabb44':
#                u'a() 1, -1, 1, 1%, -1%, 1px, -1px, "a", a, url(a), #ab4',
#
#            # S or COMMENT
#            u'red': u'red',
#            u'red ': u'red',
#            u' red ': u'red',
#            u'/**/red': u'/**/ red',
#            u'red/**/': u'red /**/',
#            u'/**/red/**/': u'/**/ red /**/',
#            u'/**/ red': u'/**/ red',
#            u'red /**/': u'red /**/',
#            u'/**/ red /**/': u'/**/ red /**/',
#            u'red-': u'red-',
#
#            # num / dimension
#            u'.0': u'0',
#            u'0': u'0',
#            u'0.0': u'0',
#            u'00': u'0',
#            u'0%': u'0%',
#            u'0px': u'0',
#            u'-.0': u'0',
#            u'-0': u'0',
#            u'-0.0': u'0',
#            u'-00': u'0',
#            u'-0%': u'0%',
#            u'-0px': u'0',
#            u'+.0': u'0',
#            u'+0': u'0',
#            u'+0.0': u'0',
#            u'+00': u'0',
#            u'+0%': u'0%',
#            u'+0px': u'0',
#            u'1': u'1',
#            u'1.0': u'1',
#            u'1px': u'1px',
#            u'1%': u'1%',
#            u'1px1': u'1px1',
#            u'+1': u'1',
#            u'-1': u'-1',
#            u'+1.0': u'1',
#            u'-1.0': u'-1',
#
#            # string, escaped nl is removed during tokenizing
#            u'"x"': u'"x"',
#            u"'x'": u'"x"',
#            #ur''' "1\'2" ''': u'''"1'2"''', #???
#            #ur"'x\"'": ur'"x\""', #???
#            ur'''"x\
# y"''': u'''"xy"''',
#
#            # hash and rgb/a
#            u'#112234': u'#112234',
#            u'#112233': u'#123',
#            u'rgb(1,2,3)': u'rgb(1, 2, 3)',
#            u'rgb(  1  ,  2  ,  3  )': u'rgb(1, 2, 3)',
#            u'rgb(-1,+2,0)': u'rgb(-1, 2, 0)',
#            u'rgba(1,2,3,4)': u'rgba(1, 2, 3, 4)',
#            u'rgba(  1  ,  2  ,  3  ,  4 )': u'rgba(1, 2, 3, 4)',
#            u'rgba(-1,+2,0, 0)': u'rgba(-1, 2, 0, 0)',
#
#            # FUNCTION
#            u'f(1,2)': u'f(1, 2)',
#            u'f(  1  ,  2  )': u'f(1, 2)',
#            u'f(-1,+2)': u'f(-1, 2)',
#            u'f(  -1  ,  +2  )': u'f(-1, 2)',
#            u'fun(  -1  ,  +2  )': u'fun(-1, 2)',
#            u'local( x )': u'local(x)',
#            u'test(1px, #111, y, 1, 1%, "1", y(), var(x))':
#                u'test(1px, #111, y, 1, 1%, "1", y(), var(x))',
#            u'test(-1px, #111, y, -1, -1%, "1", -y())':
#                u'test(-1px, #111, y, -1, -1%, "1", -y())',
#            u'url(y)  format( "x" ,  "y" )': u'url(y) format("x", "y")',
#            u'f(1 2,3 4)': u'f(1 2, 3 4)',
#
#            # IE expression
#            ur'Expression()': u'Expression()',
#            ur'expression(-1 < +2)': u'expression(-1< + 2)',
#            ur'expression(document.width == "1")': u'expression(document.width=="1")',
#            u'alpha(opacity=80)': u'alpha(opacity=80)',
#            u'alpha( opacity = 80 , x=2  )': u'alpha(opacity=80, x=2)',
#
#            # unicode-range
#            'u+f': 'u+f',
#            'U+ABCdef': 'u+abcdef',
#
#            # url
#            'url(a)': 'url(a)',
#            'uRl(a)': 'url(a)',
#            'u\\rl(a)': 'url(a)',
#            'url("a")': 'url(a)',
#            'url(  "a"  )': 'url(a)',
#            'url(a)': 'url(a)',
#            'url(";")': 'url(";")',
#            'url(",")': 'url(",")',
#            'url(")")': 'url(")")',
#            '''url("'")''': '''url("'")''',
#            '''url('"')''': '''url("\\"")''',
#            '''url("'")''': '''url("'")''',
#
#            # operator
#            '1': '1',
#            '1 2': '1 2',
#            '1   2': '1 2',
#            '1,2': '1, 2',
#            '1,  2': '1, 2',
#            '1  ,2': '1, 2',
#            '1  ,  2': '1, 2',
#            '1/2': '1/2',
#            '1/  2': '1/2',
#            '1  /2': '1/2',
#            '1  /  2': '1/2',
#             # comment
#            '1/**/2': '1 /**/ 2',
#            '1 /**/2': '1 /**/ 2',
#            '1/**/ 2': '1 /**/ 2',
#            '1 /**/ 2': '1 /**/ 2',
#            '1  /*a*/  /*b*/  2': '1 /*a*/ /*b*/ 2',
#            # , before
#            '1,/**/2': '1, /**/ 2',
#            '1 ,/**/2': '1, /**/ 2',
#            '1, /**/2': '1, /**/ 2',
#            '1 , /**/2': '1, /**/ 2',
#            # , after
#            '1/**/,2': '1 /**/, 2',
#            '1/**/ ,2': '1 /**/, 2',
#            '1/**/, 2': '1 /**/, 2',
#            '1/**/ , 2': '1 /**/, 2',
#            # all
#            '1/*a*/  ,/*b*/  2': '1 /*a*/, /*b*/ 2',
#            '1  /*a*/,  /*b*/2': '1 /*a*/, /*b*/ 2',
#            '1  /*a*/  ,  /*b*/  2': '1 /*a*/, /*b*/ 2',
#
#            # list
#            'a b1,b2 b2,b3,b4': 'a b1, b2 b2, b3, b4',
#            'a b1  ,   b2   b2  ,  b3  ,   b4': 'a b1, b2 b2, b3, b4',
#            'u+1  ,   u+2-5': 'u+1, u+2-5',
#            u'local( x ),  url(y)  format( "x" ,  "y" )':
#                u'local(x), url(y) format("x", "y")',
#            # FUNCTION
#            u'attr( href )': u'attr(href)',
#            # PrinceXML extende FUNC syntax with nested FUNC
#            u'target-counter(attr(href),page)': u'target-counter(attr(href), page)'
#            }
#
#        self.do_equal_r(tests)
#
#        tests = {
#            u'a+': xml.dom.SyntaxErr,
#            u'-': xml.dom.SyntaxErr,
#            u'+': xml.dom.SyntaxErr,
#            u'-%': xml.dom.SyntaxErr,
#            u'+a': xml.dom.SyntaxErr,
#            u'--1px': xml.dom.SyntaxErr,
#            u'++1px': xml.dom.SyntaxErr,
#            u'#': xml.dom.SyntaxErr,
#            u'#00': xml.dom.SyntaxErr,
#            u'#0000': xml.dom.SyntaxErr,
#            u'#00000': xml.dom.SyntaxErr,
#            u'#0000000': xml.dom.SyntaxErr,
#            u'-#0': xml.dom.SyntaxErr,
#            # operator
#            u',': xml.dom.SyntaxErr,
#            u'1,,2': xml.dom.SyntaxErr,
#            u'1,/**/,2': xml.dom.SyntaxErr,
#            u'1  ,  /**/  ,  2': xml.dom.SyntaxErr,
#            u'1,': xml.dom.SyntaxErr,
#            u'1, ': xml.dom.SyntaxErr,
#            u'1 ,': xml.dom.SyntaxErr,
#            u'1 , ': xml.dom.SyntaxErr,
#            u'1  ,  ': xml.dom.SyntaxErr,
#            u'1//2': xml.dom.SyntaxErr,
#            # URL
#            u'url(x))': xml.dom.SyntaxErr,
#            # string
#            u'"': xml.dom.SyntaxErr,
#            u"'": xml.dom.SyntaxErr,
#            # function
#            u'f(-)': xml.dom.SyntaxErr,
#            u'f(x))': xml.dom.SyntaxErr
#            }
#        self.do_raise_r(tests)
#
#    def test_incomplete(self):
#        "CSSValue (incomplete)"
#        tests = {
#            u'url("a': u'url(a)',
#            u'url(a': u'url(a)'
#        }
#        for v, exp in tests.items():
#            s = cssutils.parseString('a { background: %s' % v)
#            v = s.cssRules[0].style.background
#            self.assertEqual(v, exp)
#
#    def test_cssValueType(self):
#        "CSSValue.cssValueType .cssValueTypeString"
#        tests = [
#            ([u'inherit', u'INhe\\rit'], 'CSS_INHERIT', cssutils.css.CSSValue),
#            (['1', '1%', '1em', '1ex', '1px', '1cm', '1mm', '1in', '1pt', '1pc',
#              '1deg', '1rad', '1grad', '1ms', '1s', '1hz', '1khz', '1other',
#               '"string"', "'string'", 'url(x)', 'red',
#               'attr(a)', 'counter(x)', 'rect(1px, 2px, 3px, 4px)',
#               'rgb(0, 0, 0)', '#000', '#123456', 'rgba(0, 0, 0, 0)',
#               'hsl(0, 0, 0)', 'hsla(0, 0, 0, 0)',
#               ],
#             'CSS_PRIMITIVE_VALUE', cssutils.css.CSSPrimitiveValue),
#            (
#                [u'1px 1px', 'red blue green x'],
#                'CSS_VALUE_LIST', cssutils.css.CSSValueList),
#            # what is a custom value?
#            #([], 'CSS_CUSTOM', cssutils.css.CSSValue)
#            ]
#        for values, name, cls in tests:
#            for value in values:
#                v = cssutils.css.CSSValue(cssText=value)
#                if value == "'string'":
#                    # will be changed to " always
#                    value = '"string"'
#                self.assertEqual(value, v.cssText)
#                self.assertEqual(name, v.cssValueTypeString)
#                self.assertEqual(getattr(v, name), v.cssValueType)
#                self.assertEqual(cls, type(v))
#
#    def test_readonly(self):
#        "(CSSValue._readonly)"
#        v = cssutils.css.CSSValue(cssText='inherit')
#        self.assertTrue(False is v._readonly)
#
#        v = cssutils.css.CSSValue(cssText='inherit', readonly=True)
#        self.assertTrue(True is v._readonly)
#        self.assertTrue(u'inherit', v.cssText)
#        self.assertRaises(xml.dom.NoModificationAllowedErr, v._setCssText, u'x')
#        self.assertTrue(u'inherit', v.cssText)
#
#    def test_reprANDstr(self):
#        "CSSValue.__repr__(), .__str__()"
#        cssText='inherit'
#
#        s = cssutils.css.CSSValue(cssText=cssText)
#
#        self.assertTrue(cssText in str(s))
#
#        s2 = eval(repr(s))
#        self.assertTrue(isinstance(s2, s.__class__))
#        self.assertTrue(cssText == s2.cssText)
#
#
# class CSSPrimitiveValueTestCase(basetest.BaseTestCase):
#
#    def test_init(self):
#        "CSSPrimitiveValue.__init__()"
#        v = cssutils.css.CSSPrimitiveValue(u'1')
#        self.assertTrue(u'1' == v.cssText)
#
#        self.assertTrue(v.CSS_PRIMITIVE_VALUE == v.cssValueType)
#        self.assertTrue("CSS_PRIMITIVE_VALUE" == v.cssValueTypeString)
#
#        self.assertTrue(v.CSS_NUMBER == v.primitiveType)
#        self.assertTrue("CSS_NUMBER" == v.primitiveTypeString)
#
#        # DUMMY to be able to test empty constructor call
#        #self.assertRaises(xml.dom.SyntaxErr, v.__init__, None)
#
#        self.assertRaises(xml.dom.InvalidAccessErr, v.getCounterValue)
#        self.assertRaises(xml.dom.InvalidAccessErr, v.getRGBColorValue)
#        self.assertRaises(xml.dom.InvalidAccessErr, v.getRectValue)
#        self.assertRaises(xml.dom.InvalidAccessErr, v.getStringValue)
#
#    def test_CSS_UNKNOWN(self):
#        "CSSPrimitiveValue.CSS_UNKNOWN"
#        v = cssutils.css.CSSPrimitiveValue(u'expression(false)')
#        self.assertTrue(v.CSS_UNKNOWN == v.primitiveType)
#        self.assertTrue('CSS_UNKNOWN' == v.primitiveTypeString)
#
#    def test_CSS_NUMBER_AND_OTHER_DIMENSIONS(self):
#        "CSSPrimitiveValue.CSS_NUMBER .. CSS_DIMENSION"
#        defs = [
#            ('', 'CSS_NUMBER'),
#            ('%', 'CSS_PERCENTAGE'),
#            ('em', 'CSS_EMS'),
#            ('ex', 'CSS_EXS'),
#            ('px', 'CSS_PX'),
#            ('cm', 'CSS_CM'),
#            ('mm', 'CSS_MM'),
#            ('in', 'CSS_IN'),
#            ('pt', 'CSS_PT'),
#            ('pc', 'CSS_PC'),
#            ('deg', 'CSS_DEG'),
#            ('rad', 'CSS_RAD'),
#            ('grad', 'CSS_GRAD'),
#            ('ms', 'CSS_MS'),
#            ('s', 'CSS_S'),
#            ('hz', 'CSS_HZ'),
#            ('khz', 'CSS_KHZ'),
#            ('other_dimension', 'CSS_DIMENSION')
#            ]
#        for dim, name in defs:
#            for n in (0, 1, 1.1, -1, -1.1, -0):
#                v = cssutils.css.CSSPrimitiveValue('%i%s' % (n, dim))
#                self.assertEqual(name, v.primitiveTypeString)
#                self.assertEqual(getattr(v, name), v.primitiveType)
#
#    def test_CSS_STRING_AND_OTHER(self):
#        "CSSPrimitiveValue.CSS_STRING .. CSS_RGBCOLOR"
#        defs = [
#                (('""', "''", '"some thing"', "' A\\ND '",
#                  # comma separated lists are STRINGS FOR NOW!
#                  'a, b',
#                  '"a", "b"',
#                  ), 'CSS_STRING'),
#                (('url(a)', 'url("a b")', "url(' ')"), 'CSS_URI'),
#                (('some', 'or_anth-er'), 'CSS_IDENT'),
#                (('attr(a)', 'attr(b)'), 'CSS_ATTR'),
#                (('counter(1)', 'counter(2)'), 'CSS_COUNTER'),
#                (('rect(1,2,3,4)',), 'CSS_RECT'),
#                (('rgb(1,2,3)', 'rgb(10%, 20%, 30%)', '#123', '#123456'),
#                 'CSS_RGBCOLOR'),
#                (('rgba(1,2,3,4)','rgba(10%, 20%, 30%, 40%)', ),
#                 'CSS_RGBACOLOR'),
#                (('U+0', 'u+ffffff', 'u+000000-f',
#                  'u+0-f, U+ee-ff'), 'CSS_UNICODE_RANGE')
#                ]
#
#        for examples, name in defs:
#            for x in examples:
#                v = cssutils.css.CSSPrimitiveValue(x)
#                self.assertEqual(getattr(v, name), v.primitiveType)
#                self.assertEqual(name, v.primitiveTypeString)
#
#    def test_getFloat(self):
#        "CSSPrimitiveValue.getFloatValue()"
#        # NOT TESTED are float values as it seems difficult to
#        # compare these. Maybe use decimal.Decimal?
#
#        v = cssutils.css.CSSPrimitiveValue(u'1px')
#        tests = {
#            '0': (v.CSS_NUMBER, 0),
#            '-1.1': (v.CSS_NUMBER, -1.1),
#            '1%': (v.CSS_PERCENTAGE, 1),
#            '-1%': (v.CSS_PERCENTAGE, -1),
#            '1em': (v.CSS_EMS, 1),
#            '-1.1em': (v.CSS_EMS, -1.1),
#            '1ex': (v.CSS_EXS, 1),
#            '1px': (v.CSS_PX, 1),
#
#            '1cm': (v.CSS_CM, 1),
#            '1cm': (v.CSS_MM, 10),
#            '254cm': (v.CSS_IN, 100),
#            '1mm': (v.CSS_MM, 1),
#            '10mm': (v.CSS_CM, 1),
#            '254mm': (v.CSS_IN, 10),
#            '1in': (v.CSS_IN, 1),
#            '100in': (v.CSS_CM, 254), # ROUNDED!!!
#            '10in': (v.CSS_MM, 254), # ROUNDED!!!
#
#            '1pt': (v.CSS_PT, 1),
#            '1pc': (v.CSS_PC, 1),
#
#            '1deg': (v.CSS_DEG, 1),
#            '1rad': (v.CSS_RAD, 1),
#            '1grad': (v.CSS_GRAD, 1),
#
#            '1ms': (v.CSS_MS, 1),
#            '1000ms': (v.CSS_S, 1),
#            '1s': (v.CSS_S, 1),
#            '1s': (v.CSS_MS, 1000),
#
#            '1hz': (v.CSS_HZ, 1),
#            '1000hz': (v.CSS_KHZ, 1),
#            '1khz': (v.CSS_KHZ, 1),
#            '1khz': (v.CSS_HZ, 1000),
#
#            '1DIMENSION': (v.CSS_DIMENSION, 1),
#            }
#        for cssText in tests:
#            v.cssText = cssText
#            unitType, exp = tests[cssText]
#            val = v.getFloatValue(unitType)
#            if unitType in (v.CSS_IN, v.CSS_CM):
#                val = round(val)
#            self.assertEqual(val , exp)
#
#    def test_setFloat(self):
#        "CSSPrimitiveValue.setFloatValue()"
#        V = cssutils.css.CSSPrimitiveValue
#
#        tests = {
#            # unitType, value
#            (V.CSS_NUMBER, 1): [
#                # unitType, setvalue,
#                #    getvalue or expected exception, msg or cssText
#                (V.CSS_NUMBER, 0, 0, '0'),
#                (V.CSS_NUMBER, 0.1, 0.1, '0.1'),
#                (V.CSS_NUMBER, -0, 0, '0'),
#                (V.CSS_NUMBER, 2, 2, '2'),
#                (V.CSS_NUMBER, 2.0, 2, '2'),
#                (V.CSS_NUMBER, 2.1, 2.1, '2.1'),
#                (V.CSS_NUMBER, -2.1, -2.1, '-2.1'),
#                # setting with string does work
#                (V.CSS_NUMBER, '1', 1, '1'),
#                (V.CSS_NUMBER, '1.1', 1.1, '1.1'),
#                (V.CSS_PX, 1, xml.dom.InvalidAccessErr, None),
#                (V.CSS_DEG, 1, xml.dom.InvalidAccessErr, None),
#                (V.CSS_RAD, 1, xml.dom.InvalidAccessErr, None),
#                (V.CSS_GRAD, 1, xml.dom.InvalidAccessErr, None),
#                (V.CSS_S, 1, xml.dom.InvalidAccessErr, None),
#                (V.CSS_MS, 1, xml.dom.InvalidAccessErr, None),
#                (V.CSS_KHZ, 1, xml.dom.InvalidAccessErr, None),
#                (V.CSS_HZ, 1, xml.dom.InvalidAccessErr, None),
#                (V.CSS_DIMENSION, 1, xml.dom.InvalidAccessErr, None),
#                (V.CSS_MM, 2, xml.dom.InvalidAccessErr, None),
#
#                (V.CSS_NUMBER, 'x', xml.dom.InvalidAccessErr,
#                 "CSSPrimitiveValue: floatValue 'x' is not a float"),
#                (V.CSS_NUMBER, '1x', xml.dom.InvalidAccessErr,
#                 "CSSPrimitiveValue: floatValue '1x' is not a float"),
#
#                (V.CSS_STRING, 'x', xml.dom.InvalidAccessErr,
#                 "CSSPrimitiveValue: unitType 'CSS_STRING' is not a float type"),
#                (V.CSS_URI, 'x', xml.dom.InvalidAccessErr,
#                 "CSSPrimitiveValue: unitType 'CSS_URI' is not a float type"),
#                (V.CSS_ATTR, 'x', xml.dom.InvalidAccessErr,
#                 "CSSPrimitiveValue: unitType 'CSS_ATTR' is not a float type"),
#                (V.CSS_IDENT, 'x', xml.dom.InvalidAccessErr,
#                 "CSSPrimitiveValue: unitType 'CSS_IDENT' is not a float type"),
#                (V.CSS_RGBCOLOR, 'x', xml.dom.InvalidAccessErr,
#                 "CSSPrimitiveValue: unitType 'CSS_RGBCOLOR' is not a float type"),
#                (V.CSS_RGBACOLOR, 'x', xml.dom.InvalidAccessErr,
#                 "CSSPrimitiveValue: unitType 'CSS_RGBACOLOR' is not a float type"),
#                (V.CSS_RECT, 'x', xml.dom.InvalidAccessErr,
#                 "CSSPrimitiveValue: unitType 'CSS_RECT' is not a float type"),
#                (V.CSS_COUNTER, 'x', xml.dom.InvalidAccessErr,
#                 "CSSPrimitiveValue: unitType 'CSS_COUNTER' is not a float type"),
#                (V.CSS_EMS, 1, xml.dom.InvalidAccessErr,
#                 "CSSPrimitiveValue: Cannot coerce "
#                 "primitiveType 'CSS_NUMBER' to 'CSS_EMS'"),
#                (V.CSS_EXS, 1, xml.dom.InvalidAccessErr,
#                 "CSSPrimitiveValue: Cannot coerce primitiveType "
#                 "'CSS_NUMBER' to 'CSS_EXS'")
#            ],
#            (V.CSS_MM, '1mm'): [
#                (V.CSS_MM, 2, 2, '2mm'),
#                (V.CSS_MM, 0, 0, '0mm'),
#                (V.CSS_MM, 0.1, 0.1, '0.1mm'),
#                (V.CSS_MM, -0, -0, '0mm'),
#                (V.CSS_MM, 3.0, 3, '3mm'),
#                (V.CSS_MM, 3.1, 3.1, '3.1mm'),
#                (V.CSS_MM, -3.1, -3.1, '-3.1mm'),
#                (V.CSS_CM, 1, 10, '10mm'),
#                (V.CSS_IN, 10, 254, '254mm'),
#                (V.CSS_PT, 1, 1828.8, '1828.8mm'),
#                (V.CSS_PX, 1, xml.dom.InvalidAccessErr, None),
#                (V.CSS_NUMBER, 2, xml.dom.InvalidAccessErr, None)
#            ],
#            (V.CSS_PT, '1pt'): [
#                (V.CSS_PT, 2, 2, '2pt'),
#                (V.CSS_PC, 12, 1, '1pt'),
#                (V.CSS_NUMBER, 1, xml.dom.InvalidAccessErr, None),
#                (V.CSS_DEG, 1, xml.dom.InvalidAccessErr, None),
#                (V.CSS_PX, 1, xml.dom.InvalidAccessErr, None)
#            ],
#            (V.CSS_KHZ, '1khz'): [
#                (V.CSS_HZ, 2000, 2, '2khz'),
#                (V.CSS_NUMBER, 1, xml.dom.InvalidAccessErr, None),
#                (V.CSS_DEG, 1, xml.dom.InvalidAccessErr, None),
#                (V.CSS_PX, 1, xml.dom.InvalidAccessErr, None)
#            ]
#        }
#        for test in tests:
#            initialType, initialValue = test
#            pv = cssutils.css.CSSPrimitiveValue(initialValue)
#            for setType, setValue, exp, cssText in tests[test]:
#                if type(exp) == types.TypeType or\
#                   type(exp) == types.ClassType: # 2.4 compatibility
#                    if cssText:
#                        self.assertRaisesMsg(
#                            exp, cssText, pv.setFloatValue, setType, setValue)
#                    else:
#                        self.assertRaises(
#                            exp, pv.setFloatValue, setType, setValue)
#                else:
#                    pv.setFloatValue(setType, setValue)
#                    self.assertEqual(pv._value[0], cssText)
#                    if cssText == '0mm':
#                        cssText = '0'
#                    self.assertEqual(pv.cssText, cssText)
#                    self.assertEqual(pv.getFloatValue(initialType), exp)
#
#    def test_getString(self):
#        "CSSPrimitiveValue.getStringValue()"
#        v = cssutils.css.CSSPrimitiveValue(u'1px')
#        self.assertTrue(v.primitiveType == v.CSS_PX)
#        self.assertRaises(xml.dom.InvalidAccessErr,
#                          v.getStringValue)
#
#        pv = cssutils.css.CSSPrimitiveValue
#        tests = {
#            pv.CSS_STRING: ("'red'", 'red'),
#            pv.CSS_STRING: ('"red"', 'red'),
#            pv.CSS_URI: ('url(http://example.com)', None),
#            pv.CSS_URI: ("url('http://example.com')",
#                         u"http://example.com"),
#            pv.CSS_URI: ('url("http://example.com")',
#                         u'http://example.com'),
#            pv.CSS_URI: ('url("http://example.com?)")',
#                         u'http://example.com?)'),
#            pv.CSS_IDENT: ('red', None),
#            pv.CSS_ATTR: ('attr(att-name)',
#                         u'att-name'), # the name of the attrr
#            }
#        for t in tests:
#            val, exp = tests[t]
#            if not exp:
#                exp = val
#
#            v = cssutils.css.CSSPrimitiveValue(val)
#            self.assertEqual(v.primitiveType, t)
#            self.assertEqual(v.getStringValue(), exp)
#
#    def test_setString(self):
#        "CSSPrimitiveValue.setStringValue()"
#        # CSS_STRING
#        v = cssutils.css.CSSPrimitiveValue(u'"a"')
#        self.assertTrue(v.CSS_STRING == v.primitiveType)
#        v.setStringValue(v.CSS_STRING, 'b')
#        self.assertTrue(('b', 'STRING') == v._value)
#        self.assertEqual('b', v.getStringValue())
#        self.assertRaisesMsg(xml.dom.InvalidAccessErr,
#            u"CSSPrimitiveValue: Cannot coerce primitiveType "
#            "'CSS_STRING' to 'CSS_URI'",
#            v.setStringValue, *(v.CSS_URI, 'x'))
#        self.assertRaisesMsg(xml.dom.InvalidAccessErr,
#            u"CSSPrimitiveValue: Cannot coerce primitiveType "
#            "'CSS_STRING' to 'CSS_IDENT'",
#            v.setStringValue, *(v.CSS_IDENT, 'x'))
#        self.assertRaisesMsg(xml.dom.InvalidAccessErr,
#            u"CSSPrimitiveValue: Cannot coerce primitiveType "
#            "'CSS_STRING' to 'CSS_ATTR'",
#            v.setStringValue, *(v.CSS_ATTR, 'x'))
#
#        # CSS_IDENT
#        v = cssutils.css.CSSPrimitiveValue('new')
#        v.setStringValue(v.CSS_IDENT, 'ident')
#        self.assertTrue(v.CSS_IDENT == v.primitiveType)
#        self.assertTrue(('ident', 'IDENT') == v._value)
#        self.assertTrue('ident' == v.getStringValue())
#        self.assertRaisesMsg(xml.dom.InvalidAccessErr,
#            u"CSSPrimitiveValue: Cannot coerce primitiveType "
#            "'CSS_IDENT' to 'CSS_URI'",
#            v.setStringValue, *(v.CSS_URI, 'x'))
#        self.assertRaisesMsg(xml.dom.InvalidAccessErr,
#            u"CSSPrimitiveValue: Cannot coerce primitiveType "
#            "'CSS_IDENT' to 'CSS_STRING'",
#            v.setStringValue, *(v.CSS_STRING, '"x"'))
#        self.assertRaisesMsg(xml.dom.InvalidAccessErr,
#            u"CSSPrimitiveValue: Cannot coerce primitiveType "
#            "'CSS_IDENT' to 'CSS_ATTR'",
#            v.setStringValue, *(v.CSS_ATTR, 'x'))
#
#        # CSS_URI
#        v = cssutils.css.CSSPrimitiveValue('url(old)')
#        v.setStringValue(v.CSS_URI, '(')
#        self.assertEqual((u'(', 'URI'), v._value)
#        self.assertEqual(u'(', v.getStringValue())
#
#        v.setStringValue(v.CSS_URI, ')')
#        self.assertEqual((u')', 'URI'), v._value)
#        self.assertEqual(u')', v.getStringValue())
#
#        v.setStringValue(v.CSS_URI, '"')
#        self.assertEqual(ur'"', v.getStringValue())
#        self.assertEqual((ur'"', 'URI'), v._value)
#
#        v.setStringValue(v.CSS_URI, "''")
#        self.assertEqual(ur"''", v.getStringValue())
#        self.assertEqual((ur"''", 'URI'), v._value)
#
#        v.setStringValue(v.CSS_URI, ',')
#        self.assertEqual(ur',', v.getStringValue())
#        self.assertEqual((ur',', 'URI'), v._value)
#
#        v.setStringValue(v.CSS_URI, ' ')
#        self.assertEqual((u' ', 'URI'), v._value)
#        self.assertEqual(u' ', v.getStringValue())
#
#        v.setStringValue(v.CSS_URI, 'a)')
#        self.assertEqual((u'a)', 'URI'), v._value)
#        self.assertEqual(u'a)', v.getStringValue())
#
#        v.setStringValue(v.CSS_URI, 'a')
#        self.assertTrue(v.CSS_URI == v.primitiveType)
#        self.assertEqual((u'a', 'URI'), v._value)
#        self.assertEqual(u'a', v.getStringValue())
#
#        self.assertRaisesMsg(xml.dom.InvalidAccessErr,
#            u"CSSPrimitiveValue: Cannot coerce primitiveType "
#            "'CSS_URI' to 'CSS_IDENT'",
#            v.setStringValue, *(v.CSS_IDENT, 'x'))
#        self.assertRaisesMsg(xml.dom.InvalidAccessErr,
#            u"CSSPrimitiveValue: Cannot coerce primitiveType "
#            "'CSS_URI' to 'CSS_STRING'",
#            v.setStringValue, *(v.CSS_STRING, '"x"'))
#        self.assertRaisesMsg(xml.dom.InvalidAccessErr,
#            u"CSSPrimitiveValue: Cannot coerce primitiveType "
#            "'CSS_URI' to 'CSS_ATTR'",
#            v.setStringValue, *(v.CSS_ATTR, 'x'))
#
#        # CSS_ATTR
#        v = cssutils.css.CSSPrimitiveValue('attr(old)')
#        v.setStringValue(v.CSS_ATTR, 'a')
#        self.assertTrue(v.CSS_ATTR == v.primitiveType)
#        self.assertTrue('a' == v.getStringValue())
#        self.assertRaisesMsg(xml.dom.InvalidAccessErr,
#            u"CSSPrimitiveValue: Cannot coerce primitiveType "
#            "'CSS_ATTR' to 'CSS_IDENT'",
#            v.setStringValue, *(v.CSS_IDENT, 'x'))
#        self.assertRaisesMsg(xml.dom.InvalidAccessErr,
#            u"CSSPrimitiveValue: Cannot coerce primitiveType "
#            "'CSS_ATTR' to 'CSS_STRING'",
#            v.setStringValue, *(v.CSS_STRING, '"x"'))
#        self.assertRaisesMsg(xml.dom.InvalidAccessErr,
#            u"CSSPrimitiveValue: Cannot coerce primitiveType "
#            "'CSS_ATTR' to 'CSS_URI'",
#            v.setStringValue, *(v.CSS_URI, 'x'))
#
#        # TypeError as 'x' is no valid type
#        self.assertRaisesMsg(xml.dom.InvalidAccessErr,
#            u"CSSPrimitiveValue: stringType 'x' (UNKNOWN TYPE) is not a string type",
#            v.setStringValue, *('x', 'brown'))
#        # IndexError as 111 is no valid type
#        self.assertRaisesMsg(xml.dom.InvalidAccessErr,
#            u"CSSPrimitiveValue: stringType 111 (UNKNOWN TYPE) is not a string type",
#            v.setStringValue, *(111, 'brown'))
#        # CSS_PX is no string type
#        self.assertRaisesMsg(xml.dom.InvalidAccessErr,
#            u"CSSPrimitiveValue: stringType CSS_PX is not a string type",
#            v.setStringValue, *(v.CSS_PX, 'brown'))
#
#    def test_typeRGBColor(self):
#        "RGBColor"
#        v = cssutils.css.CSSPrimitiveValue('RGB(1, 5, 10)')
#        self.assertEqual(v.CSS_RGBCOLOR, v.primitiveType)
#        self.assertEqual(u'rgb(1, 5, 10)', v.cssText)
#
#        v = cssutils.css.CSSPrimitiveValue('rgb(1, 5, 10)')
#        self.assertEqual(v.CSS_RGBCOLOR, v.primitiveType)
#        self.assertEqual(u'rgb(1, 5, 10)', v.cssText)
#
#        v = cssutils.css.CSSPrimitiveValue('rgb(1%, 5%, 10%)')
#        self.assertEqual(v.CSS_RGBCOLOR, v.primitiveType)
#        self.assertEqual(u'rgb(1%, 5%, 10%)', v.cssText)
#
#        v = cssutils.css.CSSPrimitiveValue('  rgb(  1 ,5,  10  )')
#        self.assertEqual(v.CSS_RGBCOLOR, v.primitiveType)
#        v = cssutils.css.CSSPrimitiveValue('rgb(1,5,10)')
#        self.assertEqual(v.CSS_RGBCOLOR, v.primitiveType)
#        v = cssutils.css.CSSPrimitiveValue('rgb(1%, .5%, 10.1%)')
#        self.assertEqual(v.CSS_RGBCOLOR, v.primitiveType)
#
#    def test_reprANDstr(self):
#        "CSSPrimitiveValue.__repr__(), .__str__()"
#        v='111'
#
#        s = cssutils.css.CSSPrimitiveValue(v)
#
#        self.assertTrue(v in str(s))
#        self.assertTrue('CSS_NUMBER' in str(s))
#
#        s2 = eval(repr(s))
#        self.assertTrue(isinstance(s2, s.__class__))
#        self.assertTrue(v == s2.cssText)
#
#
# class CSSValueListTestCase(basetest.BaseTestCase):
#
#    def test_init(self):
#        "CSSValueList.__init__()"
#        v = cssutils.css.CSSValue(cssText=u'red blue')
#        self.assertTrue(v.CSS_VALUE_LIST == v.cssValueType)
#        self.assertEqual('red blue', v.cssText)
#
#        self.assertTrue(2 == v.length)
#
#        item = v.item(0)
#        item.setStringValue(item.CSS_IDENT, 'green')
#        self.assertEqual('green blue', v.cssText)
#
#    def test_numbers(self):
#        "CSSValueList.cssText"
#        tests = {
#            u'0 0px -0px +0px': (u'0 0 0 0', 4),
#            u'1 2 3 4': (None, 4),
#            u'-1 -2 -3 -4': (None, 4),
#            u'-1 2': (None, 2),
#            u'-1px red "x"': (None, 3),
#            u'a, b c': (None, 2),
#            u'1px1 2% 3': (u'1px1 2% 3', 3),
#            u'f(+1pX, -2, 5%) 1': (u'f(1px, -2, 5%) 1', 2),
#            u'0 f()0': (u'0 f() 0', 3),
#            u'f()0': (u'f() 0', 2),
#            u'f()1%': (u'f() 1%', 2),
#            u'f()1px': (u'f() 1px', 2),
#            u'f()"str"': (u'f() "str"', 2),
#            u'f()ident': (u'f() ident', 2),
#            u'f()#123': (u'f() #123', 2),
#            u'f()url()': (u'f() url()', 2),
#            u'f()f()': (u'f() f()', 2),
#            u'url(x.gif)0 0': (u'url(x.gif) 0 0', 3),
#            u'url(x.gif)no-repeat': (u'url(x.gif) no-repeat', 2)
#            }
#        for test in tests:
#            exp, num = tests[test]
#            if not exp:
#                exp = test
#            v = cssutils.css.CSSValue(cssText=test)
#            self.assertTrue(v.CSS_VALUE_LIST == v.cssValueType)
#            self.assertEqual(num, v.length)
#            self.assertEqual(exp, v.cssText)
#
#    def test_reprANDstr(self):
#        "CSSValueList.__repr__(), .__str__()"
#        v='1px 2px'
#
#        s = cssutils.css.CSSValue(v)
#        self.assertTrue(isinstance(s, cssutils.css.CSSValueList))
#
#        self.assertTrue('length=2' in str(s))
#        self.assertTrue(v in str(s))
#
#        # not "eval()"able!
#        #s2 = eval(repr(s))
#
#
# if __name__ == '__main__':
#    import unittest
#    unittest.main()
