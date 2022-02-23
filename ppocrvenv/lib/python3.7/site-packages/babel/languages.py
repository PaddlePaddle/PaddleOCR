# -- encoding: UTF-8 --
from babel.core import get_global


def get_official_languages(territory, regional=False, de_facto=False):
    """
    Get the official language(s) for the given territory.

    The language codes, if any are known, are returned in order of descending popularity.

    If the `regional` flag is set, then languages which are regionally official are also returned.

    If the `de_facto` flag is set, then languages which are "de facto" official are also returned.

    .. warning:: Note that the data is as up to date as the current version of the CLDR used
                 by Babel.  If you need scientifically accurate information, use another source!

    :param territory: Territory code
    :type territory: str
    :param regional: Whether to return regionally official languages too
    :type regional: bool
    :param de_facto: Whether to return de-facto official languages too
    :type de_facto: bool
    :return: Tuple of language codes
    :rtype: tuple[str]
    """

    territory = str(territory).upper()
    allowed_stati = {"official"}
    if regional:
        allowed_stati.add("official_regional")
    if de_facto:
        allowed_stati.add("de_facto_official")

    languages = get_global("territory_languages").get(territory, {})
    pairs = [
        (info['population_percent'], language)
        for language, info in languages.items()
        if info.get('official_status') in allowed_stati
    ]
    pairs.sort(reverse=True)
    return tuple(lang for _, lang in pairs)


def get_territory_language_info(territory):
    """
    Get a dictionary of language information for a territory.

    The dictionary is keyed by language code; the values are dicts with more information.

    The following keys are currently known for the values:

    * `population_percent`: The percentage of the territory's population speaking the
                            language.
    * `official_status`: An optional string describing the officiality status of the language.
                         Known values are "official", "official_regional" and "de_facto_official".

    .. warning:: Note that the data is as up to date as the current version of the CLDR used
                 by Babel.  If you need scientifically accurate information, use another source!

    .. note:: Note that the format of the dict returned may change between Babel versions.

    See https://www.unicode.org/cldr/charts/latest/supplemental/territory_language_information.html

    :param territory: Territory code
    :type territory: str
    :return: Language information dictionary
    :rtype: dict[str, dict]
    """
    territory = str(territory).upper()
    return get_global("territory_languages").get(territory, {}).copy()
