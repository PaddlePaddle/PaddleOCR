"""
A module for parsing and generating `fontconfig patterns`_.

.. _fontconfig patterns:
   https://www.freedesktop.org/software/fontconfig/fontconfig-user.html
"""

# This class logically belongs in `matplotlib.font_manager`, but placing it
# there would have created cyclical dependency problems, because it also needs
# to be available from `matplotlib.rcsetup` (for parsing matplotlibrc files).

from functools import lru_cache
import re
import numpy as np
from pyparsing import (Literal, ZeroOrMore, Optional, Regex, StringEnd,
                       ParseException, Suppress)

family_punc = r'\\\-:,'
family_unescape = re.compile(r'\\([%s])' % family_punc).sub
family_escape = re.compile(r'([%s])' % family_punc).sub

value_punc = r'\\=_:,'
value_unescape = re.compile(r'\\([%s])' % value_punc).sub
value_escape = re.compile(r'([%s])' % value_punc).sub


class FontconfigPatternParser:
    """
    A simple pyparsing-based parser for `fontconfig patterns`_.

    .. _fontconfig patterns:
       https://www.freedesktop.org/software/fontconfig/fontconfig-user.html
    """

    _constants = {
        'thin':           ('weight', 'light'),
        'extralight':     ('weight', 'light'),
        'ultralight':     ('weight', 'light'),
        'light':          ('weight', 'light'),
        'book':           ('weight', 'book'),
        'regular':        ('weight', 'regular'),
        'normal':         ('weight', 'normal'),
        'medium':         ('weight', 'medium'),
        'demibold':       ('weight', 'demibold'),
        'semibold':       ('weight', 'semibold'),
        'bold':           ('weight', 'bold'),
        'extrabold':      ('weight', 'extra bold'),
        'black':          ('weight', 'black'),
        'heavy':          ('weight', 'heavy'),
        'roman':          ('slant', 'normal'),
        'italic':         ('slant', 'italic'),
        'oblique':        ('slant', 'oblique'),
        'ultracondensed': ('width', 'ultra-condensed'),
        'extracondensed': ('width', 'extra-condensed'),
        'condensed':      ('width', 'condensed'),
        'semicondensed':  ('width', 'semi-condensed'),
        'expanded':       ('width', 'expanded'),
        'extraexpanded':  ('width', 'extra-expanded'),
        'ultraexpanded':  ('width', 'ultra-expanded')
        }

    def __init__(self):

        family = Regex(
            r'([^%s]|(\\[%s]))*' % (family_punc, family_punc)
        ).setParseAction(self._family)

        size = Regex(
            r"([0-9]+\.?[0-9]*|\.[0-9]+)"
        ).setParseAction(self._size)

        name = Regex(
            r'[a-z]+'
        ).setParseAction(self._name)

        value = Regex(
            r'([^%s]|(\\[%s]))*' % (value_punc, value_punc)
        ).setParseAction(self._value)

        families = (
            family
            + ZeroOrMore(
                Literal(',')
                + family)
        ).setParseAction(self._families)

        point_sizes = (
            size
            + ZeroOrMore(
                Literal(',')
                + size)
        ).setParseAction(self._point_sizes)

        property = (
            (name
             + Suppress(Literal('='))
             + value
             + ZeroOrMore(
                 Suppress(Literal(','))
                 + value))
            | name
        ).setParseAction(self._property)

        pattern = (
            Optional(
                families)
            + Optional(
                Literal('-')
                + point_sizes)
            + ZeroOrMore(
                Literal(':')
                + property)
            + StringEnd()
        )

        self._parser = pattern
        self.ParseException = ParseException

    def parse(self, pattern):
        """
        Parse the given fontconfig *pattern* and return a dictionary
        of key/value pairs useful for initializing a
        `.font_manager.FontProperties` object.
        """
        props = self._properties = {}
        try:
            self._parser.parseString(pattern)
        except self.ParseException as e:
            raise ValueError(
                "Could not parse font string: '%s'\n%s" % (pattern, e)) from e

        self._properties = None

        self._parser.resetCache()

        return props

    def _family(self, s, loc, tokens):
        return [family_unescape(r'\1', str(tokens[0]))]

    def _size(self, s, loc, tokens):
        return [float(tokens[0])]

    def _name(self, s, loc, tokens):
        return [str(tokens[0])]

    def _value(self, s, loc, tokens):
        return [value_unescape(r'\1', str(tokens[0]))]

    def _families(self, s, loc, tokens):
        self._properties['family'] = [str(x) for x in tokens]
        return []

    def _point_sizes(self, s, loc, tokens):
        self._properties['size'] = [str(x) for x in tokens]
        return []

    def _property(self, s, loc, tokens):
        if len(tokens) == 1:
            if tokens[0] in self._constants:
                key, val = self._constants[tokens[0]]
                self._properties.setdefault(key, []).append(val)
        else:
            key = tokens[0]
            val = tokens[1:]
            self._properties.setdefault(key, []).extend(val)
        return []


# `parse_fontconfig_pattern` is a bottleneck during the tests because it is
# repeatedly called when the rcParams are reset (to validate the default
# fonts).  In practice, the cache size doesn't grow beyond a few dozen entries
# during the test suite.
parse_fontconfig_pattern = lru_cache()(FontconfigPatternParser().parse)


def _escape_val(val, escape_func):
    """
    Given a string value or a list of string values, run each value through
    the input escape function to make the values into legal font config
    strings.  The result is returned as a string.
    """
    if not np.iterable(val) or isinstance(val, str):
        val = [val]

    return ','.join(escape_func(r'\\\1', str(x)) for x in val
                    if x is not None)


def generate_fontconfig_pattern(d):
    """
    Given a dictionary of key/value pairs, generates a fontconfig
    pattern string.
    """
    props = []

    # Family is added first w/o a keyword
    family = d.get_family()
    if family is not None and family != []:
        props.append(_escape_val(family, family_escape))

    # The other keys are added as key=value
    for key in ['style', 'variant', 'weight', 'stretch', 'file', 'size']:
        val = getattr(d, 'get_' + key)()
        # Don't use 'if not val' because 0 is a valid input.
        if val is not None and val != []:
            props.append(":%s=%s" % (key, _escape_val(val, value_escape)))

    return ''.join(props)
