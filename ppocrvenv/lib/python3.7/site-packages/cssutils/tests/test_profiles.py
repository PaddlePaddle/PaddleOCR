"""Testcases for cssutils.css.CSSValue and CSSPrimitiveValue."""

import re

import pytest

import cssutils
from . import basetest

CSS2 = (cssutils.profile.CSS_LEVEL_2,)
C3BUI = (cssutils.profile.CSS3_BASIC_USER_INTERFACE,)
C3BB = (cssutils.profile.CSS3_BACKGROUNDS_AND_BORDERS,)
CM3 = (cssutils.profile.CSS3_COLOR,)
FM3 = (cssutils.profile.CSS3_FONTS,)
C3PM = (cssutils.profile.CSS3_PAGED_MEDIA,)
C3T = (cssutils.profile.CSS3_TEXT,)
FM3FF = (cssutils.profile.CSS3_FONT_FACE,)
CSS2_CM3 = (CM3[0], CSS2[0])
CSS2_FM3 = (FM3[0], CSS2[0])


class ProfilesTestCase(basetest.BaseTestCase):
    M1 = {'testvalue': 'x'}
    P1 = {
        '-test-tokenmacro': '({num}{w}){1,2}',
        '-test-macro': '{ident}|{percentage}',
        '-test-custommacro': '{testvalue}',
        # custom validation function
        '-test-funcval': lambda v: int(v) > 0,
    }

    def test_knownNames(self):
        "Profiles.knownNames"
        p = cssutils.profiles.Profiles()
        p.removeProfile(all=True)
        p.addProfile('test', self.P1, self.M1)
        self.assertEqual(p.knownNames, list(self.P1.keys()))
        p.removeProfile(all=True)
        self.assertEqual(p.knownNames, [])

    def test_profiles(self):
        "Profiles.profiles"
        p = cssutils.profiles.Profiles()
        p.removeProfile(all=True)
        p.addProfile('test', self.P1, self.M1)
        self.assertEqual(p.profiles, ['test'])
        p.removeProfile(all=True)
        self.assertEqual(p.profiles, [])

    def test_validate2(self):
        "Profiles.validate()"
        # save
        saved = cssutils.profile

        # test
        p = cssutils.profiles.Profiles()
        cssutils.profile = p

        pvs = [('color', 'red'), ('color', 'rgba(0,0,0,0)'), ('color', 'XXX')]

        def check(*results):
            for i, pv in enumerate(pvs):
                self.assertEqual(p.validate(*pv), results[i])

        check(True, True, False)

        p.removeProfile(p.CSS3_COLOR)
        check(True, False, False)

        cssutils.profile.addProfile('test', {}, {'color': 'XXX'})
        check(False, False, True)

        p.removeProfile(all=True)
        check(False, False, False)

        # TODO: validateWithProfile

        # restore
        cssutils.profile = saved

    @pytest.mark.usefixtures('saved_profiles')
    def test_addProfile(self):
        "Profiles.addProfile with custom validation function"
        # unknown profile
        self.assertRaises(
            cssutils.profiles.NoSuchProfileException,
            lambda: list(cssutils.profile.propertiesByProfile('NOTSET')),
        )

        # new profile
        cssutils.profile.addProfile('test', self.P1, self.M1)

        props = list(self.P1.keys())
        props.sort()
        self.assertEqual(props, list(cssutils.profile.propertiesByProfile('test')))

        cssutils.log.raiseExceptions = False
        tests = {
            ('-test-tokenmacro', '1'): True,
            ('-test-tokenmacro', '1 -2'): True,
            ('-test-tokenmacro', '1 2 3'): False,
            ('-test-tokenmacro', 'a'): False,
            ('-test-macro', 'a'): True,
            ('-test-macro', '0.1%'): True,
            ('-test-custommacro', 'x'): True,
            ('-test-custommacro', '1'): False,
            ('-test-custommacro', 'y'): False,
            ('-test-funcval', '1'): True,
            ('-test-funcval', '-1'): False,
            ('-test-funcval', 'x'): False,
        }
        for test, v in list(tests.items()):
            self.assertEqual(v, cssutils.profile.validate(*test))

            self.assertEqual(
                (v, v, ['test']), cssutils.profile.validateWithProfile(*test)
            )

        cssutils.log.raiseExceptions = True

        expmsg = re.escape("invalid literal for int() with base 10: 'x'")
        with pytest.raises(Exception, match=expmsg):
            cssutils.profile.validate('-test-funcval', 'x')

    def test_removeProfile(self):
        "Profiles.removeProfile()"
        p = cssutils.profiles.Profiles()
        self.assertEqual(9, len(p.profiles))
        p.removeProfile(p.CSS_LEVEL_2)
        self.assertEqual(8, len(p.profiles))
        p.removeProfile(all=True)
        self.assertEqual(0, len(p.profiles))

    # TODO: FIX
    # def test_validateWithProfile(self):
    #     "Profiles.validate(), Profiles.validateWithProfile()"
    #     p = cssutils.profiles.Profiles()
    #     tests = {
    #         ('color', 'red', None): (True, True, [p.CSS_LEVEL_2]),
    #         ('color', 'red', p.CSS_LEVEL_2): (True, True,[p.CSS_LEVEL_2]),
    #         ('color', 'red', p.CSS3_COLOR): (True, False, [p.CSS_LEVEL_2]),
    #         ('color', 'rgba(0,0,0,0)', None): (True, True, [p.CSS3_COLOR]),
    #         ('color', 'rgba(0,0,0,0)', p.CSS_LEVEL_2): (True, False, [p.CSS3_COLOR]),
    #         ('color', 'rgba(0,0,0,0)', p.CSS3_COLOR): (True, True, [p.CSS3_COLOR]),
    #         ('color', '1px', None): (False, False, [p.CSS3_COLOR, p.CSS_LEVEL_2]),
    #         ('color', '1px', p.CSS_LEVEL_2):
    #         (False, False, [p.CSS3_COLOR, p.CSS_LEVEL_2]),
    #         ('color', '1px', p.CSS3_COLOR):
    #         (False, False, [p.CSS3_COLOR, p.CSS_LEVEL_2]),
    #         ('color', 'aliceblue', None): (True, True, [p.CSS_LEVEL_2]),
    #
    #         ('opacity', '1', None): (True, True, [p.CSS3_COLOR]),
    #         ('opacity', '1', p.CSS_LEVEL_2): (True, False, [p.CSS3_COLOR]),
    #         ('opacity', '1', p.CSS3_COLOR): (True, True, [p.CSS3_COLOR]),
    #         ('opacity', '1px', None): (False, False, [p.CSS3_COLOR]),
    #         ('opacity', '1px', p.CSS_LEVEL_2): (False, False, [p.CSS3_COLOR]),
    #         ('opacity', '1px', p.CSS3_COLOR): (False, False, [p.CSS3_COLOR]),
    #
    #         ('-x', '1', None): (False, False, []),
    #         ('-x', '1', p.CSS_LEVEL_2): (False, False, []),
    #         ('-x', '1', p.CSS3_COLOR): (False, False, []),
    #     }
    #     for test, r in tests.items():
    #         self.assertEqual(p.validate(test[0], test[1]), r[0])
    #         self.assertEqual(p.validateWithProfile(*test), r)

    def test_propertiesByProfile(self):
        "Profiles.propertiesByProfile"
        self.assertEqual(
            ['opacity'],  # 'color',
            list(cssutils.profile.propertiesByProfile(cssutils.profile.CSS3_COLOR)),
        )

    def test_csscolorlevel3(self):
        "CSS Color Module Level 3"
        # (propname, propvalue): (valid, validprofile)
        namedcolors = '''transparent, orange,
                         aqua, black, blue, fuchsia, gray, green, lime, maroon,
                         navy, olive, purple, red, silver, teal, white, yellow'''
        for color in namedcolors.split(','):
            color = color.strip()
            self.assertEqual(True, cssutils.profile.validate('color', color))

            self.assertEqual(
                (True, True, list(CSS2)),
                cssutils.profile.validateWithProfile('color', color),
            )

        # CSS2 only:
        uicolor = (
            'ActiveBorder|ActiveCaption|AppWorkspace|Background|ButtonFace|'
            'ButtonHighlight|ButtonShadow|ButtonText|CaptionText|GrayText|Highlight|'
            'HighlightText|InactiveBorder|InactiveCaption|InactiveCaptionText|'
            'InfoBackground|InfoText|Menu|MenuText|Scrollbar|ThreeDDarkShadow|'
            'ThreeDFace|ThreeDHighlight|ThreeDLightShadow|ThreeDShadow|Window|'
            'WindowFrame|WindowText'
        )
        for color in uicolor.split('|'):
            self.assertEqual(False, cssutils.profile.validate('color', color))

            # TODO: Fix
            # self.assertEqual((True, True, list(CSS2)),
            #                 cssutils.profile.validateWithProfile('color', color))

    def test_validate(self):
        "Profiles.validate()"
        tests = {
            # name, values: valid, matching, profile
            # background-position
            (
                'background-position',
                (
                    'inherit',
                    '0',
                    '1%',
                    '1px',
                    '0 0',
                    '1% 1%',
                    '1px 1px',
                    '1px 1%',
                    'top',
                    'bottom',
                    'left',
                    'right',
                    'center center',
                    'center',
                    'top left',
                    'top center',
                    'top right',
                    'bottom left',
                    'bottom center',
                    'bottom right',
                    'center left',
                    'center center',
                    'center right',
                    '0 center',
                    'center 0',
                    '0 top',
                    '10% bottom',
                    'left 0',
                    'right 10%',
                    '1% center',
                    'center 1%',
                ),
            ): (True, True, CSS2),
            ('background-position', ('0 left', 'top 0')): (False, False, CSS2),
            (
                'border-top-right-radius',
                (
                    '1px',
                    '1%',
                    '1% -1px',
                    '1% 0',
                ),
            ): (True, True, C3BB),
            ('border-top-right-radius', ('1px 2px 2px', '/ 1px', 'black')): (
                False,
                False,
                C3BB,
            ),
            (
                'border-radius',
                (
                    '1px',
                    '1%',
                    '0',
                    '1px 1px',
                    '1px/ 1px',
                    '1px /1px',
                    '1px  /  1px',
                    '1px 1px 1px 1px',
                    '1px 1px 1px 1px / 1px 1px 1px 1px',
                ),
            ): (True, True, C3BB),
            (
                'border-radius',
                (
                    '1px /',
                    '/ 1px',
                    '1px / 1px / 1px',
                    '1px 1px 1px 1px 1px',
                    '1px / 1px 1px 1px 1px 1px',
                    'black',
                ),
            ): (False, False, C3BB),
            (
                'border',
                (
                    '1px',
                    'solid',
                    'red',
                    '1px solid red',
                    '1px red solid',
                    'red 1px solid',
                    'red solid 1px',
                    'solid 1px red',
                    'solid red 1px',
                ),
            ): (True, True, C3BB),
            (
                'border',
                (
                    '1px 1px',
                    'red red 1px',
                ),
            ): (False, False, C3BB),
            (
                'box-shadow',
                (
                    'none',
                    '1px 1px',
                    '1px 1px 1px',
                    '1px 1px 1px 1px',
                    '1px 1px 1px 1px red',
                    'inset 1px 1px',
                    'inset 1px 1px 1px 1px black',
                ),
            ): (True, True, C3BB),
            (
                'box-shadow',
                (
                    '1px',
                    '1px 1px 1px 1px 1px',
                    'x 1px 1px',
                    'inset',
                    '1px black',
                    'black',
                ),
            ): (False, False, C3BB),
            # color
            (
                'color',
                (
                    'x',
                    '#',
                    '#0',
                    '#00',
                    '#0000',
                    '#00000',
                    '#0000000',
                    '#00j',
                    '#j00000',
                    'rgb(0.0,1,1)',
                    'rgb(0)',
                    'rgb(0, 1)',
                    'rgb(0, 1, 1, 1)',
                    'rgb(0, 1, 0%)',
                    'rgba(0)',
                    'rgba(0, 1, 1.0, 1)',
                    'rgba(0, 1)',
                    'rgba(0, 1, 1, 1, 1)',
                    'rgba(100%, 0%, 0%, 1%)',
                    'rgba(100%, 0%, 0, 1)',
                    'hsl(1.5,1%,1%)',
                    'hsl(1,1,1%)',
                    'hsl(1,1%,1)',
                    'hsla(1.5,1%,1%, 1)',
                    'hsla(1,1,1%, 1)',
                    'hsla(1,1%,1, 1)',
                    'hsla(1,1%,1%, 1%)',
                ),
            ): (False, False, CSS2_CM3),
            (
                'color',
                (
                    'inherit',
                    'black',
                    '#000',
                    '#000000',
                    'rgb(0,1,1)',
                    'rgb( 0 , 1 , 1 )',
                    'rgb(-10,555,1)',
                    'rgb(100%, 1.5%, 0%)',
                    'rgb(150%, -20%, 0%)',
                ),
            ): (True, True, CSS2),
            (
                'color',
                (
                    'currentcolor',
                    'aliceblue',
                    'rgba(1,1,1,1)',
                    'rgba( 1 , 1 , 1 , 1 )',
                    'rgba(100%, 0%, 0%, 1)',
                    'hsl(1,1%,1%)',
                    'hsl( 1 , 1% , 1% )',
                    'hsl(-1000,555.5%,-61.5%)',
                    'hsla(1,1%,1%,1)',
                    'hsla( 1, 1% , 1% , 1 )',
                    'hsla(-1000,555.5%,-61.5%, 0.5)',
                ),
            ): (True, True, CM3),
            # TODO?:
            # ('color', 'rgb(/**/ 0 /**/ , /**/ 1 /**/ , /**/ 1 /**/ )'):
            # (True, True, CSS2),
            # content
            (
                'content',
                (
                    'none',
                    'normal',
                    '""',
                    "'x'",
                ),
            ): (True, True, CSS2),
            (
                'cursor',
                ('url(1), auto', 'url(1) 2 3, help', 'wait', 'inherit', 'none'),
            ): (True, True, C3BUI),
            ('cursor', ('url(1), auto, wait', 'url(1) 2, help', '1')): (
                False,
                False,
                C3BUI,
            ),
            # FONTS
            ('font-family', ('serif, x',)): (True, True, CSS2),  # CSS2_FM3),
            (
                'font-family',
                (
                    'inherit',
                    'a, b',
                    'a,b,c',
                    'a, "b", c',
                    '"a", b, "c"',
                    '"a", "b", "c"',
                    '"x y"',
                    'serif',
                    # valid but CSS2: font with name serif, CSS3: same as `serif`
                    '"serif"',
                    'a  b',  # should use quotes but valid
                    'a, b   b, d',
                ),
            ): (True, True, CSS2),
            (
                'font-weight',
                (
                    'normal',
                    'bold',
                    'bolder',
                    'lighter',
                    'inherit',
                    '100',
                    '200',
                    '300',
                    '400',
                    '500',
                    '600',
                    '700',
                    '800',
                    '900',
                ),
            ): (True, True, CSS2),
            (
                'font-stretch',
                (
                    'normal',
                    'wider',
                    'narrower',
                    'ultra-condensed',
                    'extra-condensed',
                    'condensed',
                    'semi-condensed',
                    'semi-expanded',
                    'expanded',
                    'extra-expanded',
                    'ultra-expanded',
                    'inherit',
                ),
            ): (True, True, FM3),
            ('font-style', ('normal', 'italic', 'oblique', 'inherit')): (
                True,
                True,
                CSS2,
            ),
            ('font-variant', ('normal', 'small-caps', 'inherit')): (True, True, CSS2),
            ('font-size', ('-1em',)): (False, False, CSS2),
            (
                'font-size',
                (
                    'xx-small',
                    'x-small',
                    'small',
                    'medium',
                    'large',
                    'x-large',
                    'xx-large',
                    'larger',
                    'smaller',
                    '1em',
                    '1%',
                    'inherit',
                ),
            ): (True, True, CSS2),
            ('font-size-adjust', ('1.0', 'none', 'inherit')): (True, True, FM3),
            (
                'font',
                (
                    'italic small-caps bold 1px/3 a, "b", serif',
                    '12pt/14pt sans-serif',
                    '80% sans-serif',
                    'x-large/110% "new century schoolbook", serif',
                    'bold italic large Palatino, serif',
                    'normal small-caps 120%/120% fantasy',
                    'oblique 12pt "Helvetica Nue", serif',
                    'caption',
                    'icon',
                    'menu',
                    'message-box',
                    'small-caption',
                    'status-bar',
                    'inherit',
                ),
            ): (True, True, CSS2),
            ('nav-index', ('1', 'auto', 'inherit')): (True, True, C3BUI),
            ('nav-index', ('x', '1 2', '1px')): (False, False, C3BUI),
            (
                'opacity',
                (
                    'inherit',
                    '0',
                    '0.0',
                    '0.42342',
                    '1',
                    '1.0',
                    # should be clipped but valid
                    '-0',
                    '-0.1',
                    '-10',
                    '2',
                ),
            ): (True, True, CM3),
            ('opacity', ('a', '#000', '+1')): (False, False, CM3),
            (
                'outline',
                (
                    'red dotted 1px',
                    'dotted 1px red',
                    '1px red dotted',
                    'red',
                    '1px',
                    'dotted',
                    'red 1px',
                    '1px dotted',
                    'red dotted',
                    'inherit',
                ),
            ): (True, True, C3BUI),
            ('outline', ('red #fff', 'solid dotted', 'Url(x)', '1px 1px')): (
                False,
                False,
                C3BUI,
            ),
            ('outline-color', ('red', '#fff', 'inherit')): (True, True, C3BUI),
            ('outline-color', ('0', '1em')): (False, False, C3BUI),
            ('outline-offset', ('0', '1em', 'inherit')): (True, True, C3BUI),
            ('outline-offset', ('1%', 'red')): (False, False, C3BUI),
            ('outline-style', ('auto', 'dotted', 'inherit')): (True, True, C3BUI),
            ('outline-style', ('0', '1em', 'red')): (False, False, C3BUI),
            ('outline-width', ('0', '1em', 'inherit')): (True, True, C3BUI),
            ('outline-width', ('auto', 'red', 'dotted')): (False, False, C3BUI),
            ('resize', ('none', 'both', 'horizontal', 'vertical', 'inherit')): (
                True,
                True,
                C3BUI,
            ),
            ('resize', ('1', 'auto', '1px', '2%')): (False, False, C3BUI),
            (
                'size',
                (
                    '1cm',
                    '1mm 20cm',
                    'auto',
                    'landscape letter',
                    'a4 portrait',
                    'landscape',
                    'a5',
                    # 'inherit'
                ),
            ): (True, True, C3PM),
            ('size', ('portrait landscape', 'a5 letter', '2%')): (False, False, C3PM),
            (
                'src',
                (
                    'url(  a  )',
                    'local(  x  )',
                    'local("x")',
                    'local(  "x"  )',
                    'url(../fonts/LateefRegAAT.ttf) format(  "truetype-aat"  )',
                    'url(a) format(  "123x"  , "a"   )',
                    'url(a) format( "123x"  , "a"   ), '
                    'url(a) format( "123x"  , "a"   )',
                    'local(HiraKakuPro-W3), local(Meiryo), local(IPAPGothic)',
                    'local(Gentium), url(/fonts/Gentium.ttf)',
                    'local("Gentium"), url("/fonts/Gentium.ttf")',
                    'local(Futura-Medium), url(fonts.svg#MyGeometricModern) '
                    'format("svg")',
                ),
            ): (True, True, FM3FF),
            (
                'text-shadow',
                (
                    'none',
                    '1px 1px',
                    '1px 1px 1px',
                    '1px 1px 1px 1px',
                    '1px 1px 1px 1px red',
                    'inset 1px 1px',
                    'inset 1px 1px 1px 1px black',
                ),
            ): (True, True, C3T),
            (
                'text-shadow',
                (
                    '1px',
                    '1px 1px 1px 1px 1px',
                    'x 1px 1px',
                    'inset',
                    '1px black',
                    'black',
                ),
            ): (False, False, C3T),
            ('unicode-range', ('u+1', 'U+111111-ffffff', 'u+123456  ,  U+1-f')): (
                True,
                True,
                FM3FF,
            ),
        }
        # TODO!!!
        for (name, values), (valid, matching, profile) in list(tests.items()):
            for value in values:
                self.assertEqual(valid, cssutils.profile.validate(name, value))


# if (valid, matching, list(profile)) !=
# cssutils.profile.validateWithProfile(name, value):
#     print
#     print '###############', name, value
#     print (valid, matching, list(profile)),
# cssutils.profile.validateWithProfile(name, value)

# TODO: fix
#                self.assertEqual((valid, matching, list(profile)),
#                                 cssutils.profile.validateWithProfile(name, value))

# TODO: fix
#    def test_validateByProfile(self):
#        "Profiles.validateByProfile()"
#        # testing for valid values overwritten in a profile
#        tests = {
#            (FM3FF, 'font-family', ('y', '"y"' # => name should be "y"!!!
#                                     )): (True, True, FM3FF),
#            (FM3FF, 'font-family', ('"y", "a"', 'a, b', 'a a'
#                                     )): (True, False, CSS2),
#            (FM3FF, 'font-stretch', ('normal', 'wider', 'narrower', 'inherit'
#                                     )): (True, False, FM3),
#            (FM3FF, 'font-style', ('inherit',
#                                     )): (True, False, CSS2),
#            (FM3FF, 'font-weight', ('bolder', 'lighter', 'inherit',
#                                     )): (True, False, CSS2),
#            }
#        for (profiles, name, values), (v, m, p) in tests.items():
#            for value in values:
#                self.assertEqual((v, m, list(p)),
#                                 cssutils.profile.validateWithProfile(name,
#                                                                      value,
#                                                                      profiles))


if __name__ == '__main__':
    import unittest

    unittest.main()
