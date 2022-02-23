"""Testcases for cssutils.css.property._Property."""

import copy
from . import basetest
import cssutils
from cssutils.css.property import Property

debug = False


class PropertiesTestCase(basetest.BaseTestCase):
    def setUp(self):
        "init test values"
        V = {
            '0': ('0', '-0'),  # , '+0'),
            'NUMBER': ('0', '-0', '100.1', '-100.1'),  # , '+0', '+100.1'),
            'PERCENTAGE': ('0%', '-0%', '100.1%', '-100.1%'),  # , '+0%', '+100.1%'),
            'EM': '1.2em',
            'EX': '1.2ex',
            'PX': '1.2px',
            'CM': '1.2cm',
            'MM': '1.2mm',
            'IN': '1.2in',
            'PT': '1.2pt',
            'PC': '1.2pc',
            'ANGLES': ('1deg', '1rad', '1grad'),
            'TIMES': ('1s', '1ms'),
            'FREQUENCIES': ('1hz', '1khz'),
            'DIMENSION': ('1dimension', '1_dimension', '1dimension2'),
            'STRING': ('"string"', "'STRING'"),
            'URI': ('url(x)', 'URL("x")', "url(')')"),
            'IDENT': ('ident', 'IDENT', '_IDENT', '_2', 'i-2'),
            # 'AUTO': 'auto', # explicit in list as an IDENT too
            # 'INHERIT': 'inherit', # explicit in list as an IDENT too
            'ATTR': ('attr(x)'),
            'RECT': ('rect(1,2,3,4)'),
            # ?
            'CLIP': ('rect(1,2,3,4)'),
            'FUNCTION': (),
            'HEX3': '#123',
            'HEX6': '#123abc',
            'RGB': 'rgb(1,2,3)',
            'RGB100': 'rgb(1%,2%,100%)',
            'RGBA': 'rgba(1,2,3, 1)',
            'RGBA100': 'rgba(1%,2%,100%, 0)',
            'HSL': 'hsl(1,2%,3%)',
            'HSLA': 'hsla(1,2%,3%, 1.0)',
        }

        def expanded(*keys):
            r = []
            for k in keys:
                if isinstance(V[k], str):
                    r.append(V[k])
                else:
                    r.extend(list(V[k]))
            return r

        # before adding combined
        self.V = V
        self.ALL = list(self._valuesofkeys(list(V.keys())))

        # combined values, only keys of V may be used!
        self.V['LENGTHS'] = expanded(
            '0', 'EM', 'EX', 'PX', 'CM', 'MM', 'IN', 'PT', 'PC'
        )
        self.V['COLORS'] = expanded('HEX3', 'HEX6', 'RGB', 'RGB100')
        self.V['COLORS3'] = expanded('RGBA', 'RGBA100', 'HSL', 'HSLA')

    def _allvalues(self):
        "Return list of **all** possible values as simple list"
        return copy.copy(self.ALL)

    def _valuesofkeys(self, keys):
        "Generate all distinct values in given keys of self.V"
        done = []
        for key in keys:
            if isinstance(key, list):
                # not a key but a list of values, return directly
                for v in key:
                    yield v
            else:
                v = self.V[key]
                if isinstance(v, str):
                    # single value
                    if v not in done:
                        done.append(v)
                        yield v
                else:
                    # a list of values
                    for value in v:
                        if value not in done:
                            done.append(value)
                            yield value

    def _check(self, name, keys):
        """
        Check each value in values if for property name p.name==exp.
        """
        notvalid = self._allvalues()

        for value in self._valuesofkeys(keys):
            if name == debug:
                print('+True?', Property(name, value).valid, value)
            self.assertEqual(True, Property(name, value).valid)
            if value in notvalid:
                notvalid.remove(value)
        for value in notvalid:
            if name == debug:
                print('-False?', Property(name, value).valid, value)
            self.assertEqual(False, Property(name, value).valid)

    def test_properties(self):
        "properties"
        tests = {
            # propname: key or [list of values]
            'color': ('COLORS', 'COLORS3', ['inherit', 'red']),
            'fit': (['fill', 'hidden', 'meet', 'slice'],),
            'fit-position': (
                'LENGTHS',
                'PERCENTAGE',
                ['auto', 'top left', '0% 50%', '1cm 5em', 'bottom'],
            ),
            'font-family': (
                'STRING',
                'IDENT',
                [
                    'a, b',
                    '"a", "b"',
                    'a, "b"',
                    '"a", b',
                    r'a\{b',
                    r'a\ b',
                    'a b' 'a b, c  d  , e',
                ],
            ),
            # 'src': ('STRING',),
            'font-weight': (
                [
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
                ],
            ),
            'font-stretch': (
                [
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
                ],
            ),
            'font-style': (['normal', 'italic', 'oblique', 'inherit'],),
            'font-variant': (['normal', 'small-caps', 'inherit'],),
            'font-size': (
                'LENGTHS',
                'PERCENTAGE',
                [
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
                ],
            ),
            'font-size-adjust': ('NUMBER', ['none', 'inherit']),
            # 'font': (['italic small-caps bold 1px/3 a, "b", serif',
            #           'caption', 'icon', 'menu', 'message-box', 'small-caption',
            #           'status-bar', 'inherit'],),
            'image-orientation': ('0', 'ANGLES', ['auto']),
            'left': ('LENGTHS', 'PERCENTAGE', ['inherit', 'auto']),
            'opacity': ('NUMBER', ['inherit']),
            'orphans': ('0', ['1', '99999', 'inherit']),
            'page': ('IDENT',),
            'page-break-inside': (['auto', 'inherit', 'avoid'],),
            'size': (
                'LENGTHS',
                ['auto', '1em 1em', 'a4 portrait', 'b4 landscape', 'A5 PORTRAIT'],
            ),
            'widows': ('0', ['1', '99999', 'inherit']),
        }
        for name, keys in list(tests.items()):
            # keep track of valid keys
            self._check(name, keys)

    def test_validate(self):
        "Property.validate() and Property.valid"
        tests = {
            # (default L2, no default, no profile, L2, Color L3)
            'red': (True, True, True, True, True),
            'rgba(1,2,3,1)': (False, True, True, False, True),
            '1': (False, False, False, False, False),
        }
        for v, rs in list(tests.items()):
            p = Property('color', v)

            # TODO: Fix
            #            cssutils.profile.defaultProfiles = \
            #                cssutils.profile.CSS_LEVEL_2
            #            self.assertEqual(rs[0], p.valid)

            cssutils.profile.defaultProfiles = None
            self.assertEqual(rs[1], p.valid)

            self.assertEqual(rs[2], p.validate())


#            self.assertEqual(rs[3], p.validate(
#                profiles=cssutils.profile.CSS_LEVEL_2))
#            self.assertEqual(rs[4], p.validate(
#                cssutils.profile.CSS3_COLOR))


if __name__ == '__main__':
    debug = 'font-family'
    import logging
    import unittest

    cssutils.log.setLevel(logging.FATAL)
    # debug = True
    unittest.main()
