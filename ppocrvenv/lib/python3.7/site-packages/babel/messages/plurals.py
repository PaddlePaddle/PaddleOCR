# -*- coding: utf-8 -*-
"""
    babel.messages.plurals
    ~~~~~~~~~~~~~~~~~~~~~~

    Plural form definitions.

    :copyright: (c) 2013-2021 by the Babel Team.
    :license: BSD, see LICENSE for more details.
"""

from babel.core import default_locale, Locale
from operator import itemgetter


# XXX: remove this file, duplication with babel.plural


LC_CTYPE = default_locale('LC_CTYPE')


PLURALS = {
    # Afar
    # 'aa': (),
    # Abkhazian
    # 'ab': (),
    # Avestan
    # 'ae': (),
    # Afrikaans - From Pootle's PO's
    'af': (2, '(n != 1)'),
    # Akan
    # 'ak': (),
    # Amharic
    # 'am': (),
    # Aragonese
    # 'an': (),
    # Arabic - From Pootle's PO's
    'ar': (6, '(n==0 ? 0 : n==1 ? 1 : n==2 ? 2 : n%100>=3 && n%100<=10 ? 3 : n%100>=0 && n%100<=2 ? 4 : 5)'),
    # Assamese
    # 'as': (),
    # Avaric
    # 'av': (),
    # Aymara
    # 'ay': (),
    # Azerbaijani
    # 'az': (),
    # Bashkir
    # 'ba': (),
    # Belarusian
    'be': (3, '(n%10==1 && n%100!=11 ? 0 : n%10>=2 && n%10<=4 && (n%100<10 || n%100>=20) ? 1 : 2)'),
    # Bulgarian - From Pootle's PO's
    'bg': (2, '(n != 1)'),
    # Bihari
    # 'bh': (),
    # Bislama
    # 'bi': (),
    # Bambara
    # 'bm': (),
    # Bengali - From Pootle's PO's
    'bn': (2, '(n != 1)'),
    # Tibetan - as discussed in private with Andrew West
    'bo': (1, '0'),
    # Breton
    'br': (
        6,
        '(n==1 ? 0 : n%10==1 && n%100!=11 && n%100!=71 && n%100!=91 ? 1 : n%10==2 && n%100!=12 && n%100!=72 && '
        'n%100!=92 ? 2 : (n%10==3 || n%10==4 || n%10==9) && n%100!=13 && n%100!=14 && n%100!=19 && n%100!=73 && '
        'n%100!=74 && n%100!=79 && n%100!=93 && n%100!=94 && n%100!=99 ? 3 : n%1000000==0 ? 4 : 5)'
    ),
    # Bosnian
    'bs': (3, '(n%10==1 && n%100!=11 ? 0 : n%10>=2 && n%10<=4 && (n%100<10 || n%100>=20) ? 1 : 2)'),
    # Catalan - From Pootle's PO's
    'ca': (2, '(n != 1)'),
    # Chechen
    # 'ce': (),
    # Chamorro
    # 'ch': (),
    # Corsican
    # 'co': (),
    # Cree
    # 'cr': (),
    # Czech
    'cs': (3, '((n==1) ? 0 : (n>=2 && n<=4) ? 1 : 2)'),
    # Church Slavic
    # 'cu': (),
    # Chuvash
    'cv': (1, '0'),
    # Welsh
    'cy': (5, '(n==1 ? 1 : n==2 ? 2 : n==3 ? 3 : n==6 ? 4 : 0)'),
    # Danish
    'da': (2, '(n != 1)'),
    # German
    'de': (2, '(n != 1)'),
    # Divehi
    # 'dv': (),
    # Dzongkha
    'dz': (1, '0'),
    # Greek
    'el': (2, '(n != 1)'),
    # English
    'en': (2, '(n != 1)'),
    # Esperanto
    'eo': (2, '(n != 1)'),
    # Spanish
    'es': (2, '(n != 1)'),
    # Estonian
    'et': (2, '(n != 1)'),
    # Basque - From Pootle's PO's
    'eu': (2, '(n != 1)'),
    # Persian - From Pootle's PO's
    'fa': (1, '0'),
    # Finnish
    'fi': (2, '(n != 1)'),
    # French
    'fr': (2, '(n > 1)'),
    # Friulian - From Pootle's PO's
    'fur': (2, '(n > 1)'),
    # Irish
    'ga': (5, '(n==1 ? 0 : n==2 ? 1 : n>=3 && n<=6 ? 2 : n>=7 && n<=10 ? 3 : 4)'),
    # Galician - From Pootle's PO's
    'gl': (2, '(n != 1)'),
    # Hausa - From Pootle's PO's
    'ha': (2, '(n != 1)'),
    # Hebrew
    'he': (2, '(n != 1)'),
    # Hindi - From Pootle's PO's
    'hi': (2, '(n != 1)'),
    # Croatian
    'hr': (3, '(n%10==1 && n%100!=11 ? 0 : n%10>=2 && n%10<=4 && (n%100<10 || n%100>=20) ? 1 : 2)'),
    # Hungarian
    'hu': (1, '0'),
    # Armenian - From Pootle's PO's
    'hy': (1, '0'),
    # Icelandic - From Pootle's PO's
    'is': (2, '(n%10==1 && n%100!=11 ? 0 : 1)'),
    # Italian
    'it': (2, '(n != 1)'),
    # Japanese
    'ja': (1, '0'),
    # Georgian - From Pootle's PO's
    'ka': (1, '0'),
    # Kongo - From Pootle's PO's
    'kg': (2, '(n != 1)'),
    # Khmer - From Pootle's PO's
    'km': (1, '0'),
    # Korean
    'ko': (1, '0'),
    # Kurdish - From Pootle's PO's
    'ku': (2, '(n != 1)'),
    # Lao - Another member of the Tai language family, like Thai.
    'lo': (1, '0'),
    # Lithuanian
    'lt': (3, '(n%10==1 && n%100!=11 ? 0 : n%10>=2 && (n%100<10 || n%100>=20) ? 1 : 2)'),
    # Latvian
    'lv': (3, '(n%10==1 && n%100!=11 ? 0 : n != 0 ? 1 : 2)'),
    # Maltese - From Pootle's PO's
    'mt': (4, '(n==1 ? 0 : n==0 || ( n%100=>1 && n%100<=10) ? 1 : (n%100>10 && n%100<20 ) ? 2 : 3)'),
    # Norwegian Bokmål
    'nb': (2, '(n != 1)'),
    # Dutch
    'nl': (2, '(n != 1)'),
    # Norwegian Nynorsk
    'nn': (2, '(n != 1)'),
    # Norwegian
    'no': (2, '(n != 1)'),
    # Punjabi - From Pootle's PO's
    'pa': (2, '(n != 1)'),
    # Polish
    'pl': (3, '(n==1 ? 0 : n%10>=2 && n%10<=4 && (n%100<10 || n%100>=20) ? 1 : 2)'),
    # Portuguese
    'pt': (2, '(n != 1)'),
    # Brazilian
    'pt_BR': (2, '(n > 1)'),
    # Romanian - From Pootle's PO's
    'ro': (3, '(n==1 ? 0 : (n==0 || (n%100 > 0 && n%100 < 20)) ? 1 : 2)'),
    # Russian
    'ru': (3, '(n%10==1 && n%100!=11 ? 0 : n%10>=2 && n%10<=4 && (n%100<10 || n%100>=20) ? 1 : 2)'),
    # Slovak
    'sk': (3, '((n==1) ? 0 : (n>=2 && n<=4) ? 1 : 2)'),
    # Slovenian
    'sl': (4, '(n%100==1 ? 0 : n%100==2 ? 1 : n%100==3 || n%100==4 ? 2 : 3)'),
    # Serbian - From Pootle's PO's
    'sr': (3, '(n%10==1 && n%100!=11 ? 0 : n%10>=2 && n%10<=4 && (n%100<10 || n%100>=20) ? 1 : 2)'),
    # Southern Sotho - From Pootle's PO's
    'st': (2, '(n != 1)'),
    # Swedish
    'sv': (2, '(n != 1)'),
    # Thai
    'th': (1, '0'),
    # Turkish
    'tr': (1, '0'),
    # Ukrainian
    'uk': (3, '(n%10==1 && n%100!=11 ? 0 : n%10>=2 && n%10<=4 && (n%100<10 || n%100>=20) ? 1 : 2)'),
    # Venda - From Pootle's PO's
    've': (2, '(n != 1)'),
    # Vietnamese - From Pootle's PO's
    'vi': (1, '0'),
    # Xhosa - From Pootle's PO's
    'xh': (2, '(n != 1)'),
    # Chinese - From Pootle's PO's (modified)
    'zh': (1, '0'),
}


DEFAULT_PLURAL = (2, '(n != 1)')


class _PluralTuple(tuple):
    """A tuple with plural information."""

    __slots__ = ()
    num_plurals = property(itemgetter(0), doc="""
    The number of plurals used by the locale.""")
    plural_expr = property(itemgetter(1), doc="""
    The plural expression used by the locale.""")
    plural_forms = property(lambda x: 'nplurals=%s; plural=%s;' % x, doc="""
    The plural expression used by the catalog or locale.""")

    def __str__(self):
        return self.plural_forms


def get_plural(locale=LC_CTYPE):
    """A tuple with the information catalogs need to perform proper
    pluralization.  The first item of the tuple is the number of plural
    forms, the second the plural expression.

    >>> get_plural(locale='en')
    (2, '(n != 1)')
    >>> get_plural(locale='ga')
    (5, '(n==1 ? 0 : n==2 ? 1 : n>=3 && n<=6 ? 2 : n>=7 && n<=10 ? 3 : 4)')

    The object returned is a special tuple with additional members:

    >>> tup = get_plural("ja")
    >>> tup.num_plurals
    1
    >>> tup.plural_expr
    '0'
    >>> tup.plural_forms
    'nplurals=1; plural=0;'

    Converting the tuple into a string prints the plural forms for a
    gettext catalog:

    >>> str(tup)
    'nplurals=1; plural=0;'
    """
    locale = Locale.parse(locale)
    try:
        tup = PLURALS[str(locale)]
    except KeyError:
        try:
            tup = PLURALS[locale.language]
        except KeyError:
            tup = DEFAULT_PLURAL
    return _PluralTuple(tup)
