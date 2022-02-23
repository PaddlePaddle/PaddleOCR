"""Module to build FeatureVariation tables:
https://docs.microsoft.com/en-us/typography/opentype/spec/chapter2#featurevariations-table

NOTE: The API is experimental and subject to change.
"""
from fontTools.misc.dictTools import hashdict
from fontTools.misc.intTools import popCount
from fontTools.ttLib import newTable
from fontTools.ttLib.tables import otTables as ot
from fontTools.otlLib.builder import buildLookup, buildSingleSubstSubtable
from collections import OrderedDict

from .errors import VarLibError, VarLibValidationError


def addFeatureVariations(font, conditionalSubstitutions, featureTag='rvrn'):
    """Add conditional substitutions to a Variable Font.

    The `conditionalSubstitutions` argument is a list of (Region, Substitutions)
    tuples.

    A Region is a list of Boxes. A Box is a dict mapping axisTags to
    (minValue, maxValue) tuples. Irrelevant axes may be omitted and they are
    interpretted as extending to end of axis in each direction.  A Box represents
    an orthogonal 'rectangular' subset of an N-dimensional design space.
    A Region represents a more complex subset of an N-dimensional design space,
    ie. the union of all the Boxes in the Region.
    For efficiency, Boxes within a Region should ideally not overlap, but
    functionality is not compromised if they do.

    The minimum and maximum values are expressed in normalized coordinates.

    A Substitution is a dict mapping source glyph names to substitute glyph names.

    Example:

    # >>> f = TTFont(srcPath)
    # >>> condSubst = [
    # ...     # A list of (Region, Substitution) tuples.
    # ...     ([{"wdth": (0.5, 1.0)}], {"cent": "cent.rvrn"}),
    # ...     ([{"wght": (0.5, 1.0)}], {"dollar": "dollar.rvrn"}),
    # ... ]
    # >>> addFeatureVariations(f, condSubst)
    # >>> f.save(dstPath)
    """


    substitutions = overlayFeatureVariations(conditionalSubstitutions)

    # turn substitution dicts into tuples of tuples, so they are hashable
    conditionalSubstitutions, allSubstitutions = makeSubstitutionsHashable(substitutions)
    if "GSUB" not in font:
        font["GSUB"] = buildGSUB()

    # setup lookups
    lookupMap = buildSubstitutionLookups(font["GSUB"].table, allSubstitutions)

    # addFeatureVariationsRaw takes a list of
    #  ( {condition}, [ lookup indices ] )
    # so rearrange our lookups to match
    conditionsAndLookups = []
    for conditionSet, substitutions in conditionalSubstitutions:
        conditionsAndLookups.append((conditionSet, [lookupMap[s] for s in substitutions]))

    addFeatureVariationsRaw(font, font["GSUB"].table,
                            conditionsAndLookups,
                            featureTag)

def overlayFeatureVariations(conditionalSubstitutions):
    """Compute overlaps between all conditional substitutions.

    The `conditionalSubstitutions` argument is a list of (Region, Substitutions)
    tuples.

    A Region is a list of Boxes. A Box is a dict mapping axisTags to
    (minValue, maxValue) tuples. Irrelevant axes may be omitted and they are
    interpretted as extending to end of axis in each direction.  A Box represents
    an orthogonal 'rectangular' subset of an N-dimensional design space.
    A Region represents a more complex subset of an N-dimensional design space,
    ie. the union of all the Boxes in the Region.
    For efficiency, Boxes within a Region should ideally not overlap, but
    functionality is not compromised if they do.

    The minimum and maximum values are expressed in normalized coordinates.

    A Substitution is a dict mapping source glyph names to substitute glyph names.

    Returns data is in similar but different format.  Overlaps of distinct
    substitution Boxes (*not* Regions) are explicitly listed as distinct rules,
    and rules with the same Box merged.  The more specific rules appear earlier
    in the resulting list.  Moreover, instead of just a dictionary of substitutions,
    a list of dictionaries is returned for substitutions corresponding to each
    unique space, with each dictionary being identical to one of the input
    substitution dictionaries.  These dictionaries are not merged to allow data
    sharing when they are converted into font tables.

    Example::

        >>> condSubst = [
        ...     # A list of (Region, Substitution) tuples.
        ...     ([{"wght": (0.5, 1.0)}], {"dollar": "dollar.rvrn"}),
        ...     ([{"wght": (0.5, 1.0)}], {"dollar": "dollar.rvrn"}),
        ...     ([{"wdth": (0.5, 1.0)}], {"cent": "cent.rvrn"}),
        ...     ([{"wght": (0.5, 1.0), "wdth": (-1, 1.0)}], {"dollar": "dollar.rvrn"}),
        ... ]
        >>> from pprint import pprint
        >>> pprint(overlayFeatureVariations(condSubst))
        [({'wdth': (0.5, 1.0), 'wght': (0.5, 1.0)},
          [{'dollar': 'dollar.rvrn'}, {'cent': 'cent.rvrn'}]),
         ({'wdth': (0.5, 1.0)}, [{'cent': 'cent.rvrn'}]),
         ({'wght': (0.5, 1.0)}, [{'dollar': 'dollar.rvrn'}])]

    """

    # Merge same-substitutions rules, as this creates fewer number oflookups.
    merged = OrderedDict()
    for value,key in conditionalSubstitutions:
        key = hashdict(key)
        if key in merged:
            merged[key].extend(value)
        else:
            merged[key] = value
    conditionalSubstitutions = [(v,dict(k)) for k,v in merged.items()]
    del merged

    # Merge same-region rules, as this is cheaper.
    # Also convert boxes to hashdict()
    #
    # Reversing is such that earlier entries win in case of conflicting substitution
    # rules for the same region.
    merged = OrderedDict()
    for key,value in reversed(conditionalSubstitutions):
        key = tuple(sorted((hashdict(cleanupBox(k)) for k in key),
                           key=lambda d: tuple(sorted(d.items()))))
        if key in merged:
            merged[key].update(value)
        else:
            merged[key] = dict(value)
    conditionalSubstitutions = list(reversed(merged.items()))
    del merged

    # Overlay
    #
    # Rank is the bit-set of the index of all contributing layers.
    initMapInit = ((hashdict(),0),) # Initializer representing the entire space
    boxMap = OrderedDict(initMapInit) # Map from Box to Rank
    for i,(currRegion,_) in enumerate(conditionalSubstitutions):
        newMap = OrderedDict(initMapInit)
        currRank = 1<<i
        for box,rank in boxMap.items():
            for currBox in currRegion:
                intersection, remainder = overlayBox(currBox, box)
                if intersection is not None:
                    intersection = hashdict(intersection)
                    newMap[intersection] = newMap.get(intersection, 0) | rank|currRank
                if remainder is not None:
                    remainder = hashdict(remainder)
                    newMap[remainder] = newMap.get(remainder, 0) | rank
        boxMap = newMap

    # Generate output
    items = []
    for box,rank in sorted(boxMap.items(),
                           key=(lambda BoxAndRank: -popCount(BoxAndRank[1]))):
        # Skip any box that doesn't have any substitution.
        if rank == 0:
            continue
        substsList = []
        i = 0
        while rank:
          if rank & 1:
              substsList.append(conditionalSubstitutions[i][1])
          rank >>= 1
          i += 1
        items.append((dict(box),substsList))
    return items


#
# Terminology:
#
# A 'Box' is a dict representing an orthogonal "rectangular" bit of N-dimensional space.
# The keys in the dict are axis tags, the values are (minValue, maxValue) tuples.
# Missing dimensions (keys) are substituted by the default min and max values
# from the corresponding axes.
#

def overlayBox(top, bot):
    """Overlays ``top`` box on top of ``bot`` box.

    Returns two items:

    * Box for intersection of ``top`` and ``bot``, or None if they don't intersect.
    * Box for remainder of ``bot``.  Remainder box might not be exact (since the
      remainder might not be a simple box), but is inclusive of the exact
      remainder.
    """

    # Intersection
    intersection = {}
    intersection.update(top)
    intersection.update(bot)
    for axisTag in set(top) & set(bot):
        min1, max1 = top[axisTag]
        min2, max2 = bot[axisTag]
        minimum = max(min1, min2)
        maximum = min(max1, max2)
        if not minimum < maximum:
            return None, bot # Do not intersect
        intersection[axisTag] = minimum,maximum

    # Remainder
    #
    # Remainder is empty if bot's each axis range lies within that of intersection.
    #
    # Remainder is shrank if bot's each, except for exactly one, axis range lies
    # within that of intersection, and that one axis, it spills out of the
    # intersection only on one side.
    #
    # Bot is returned in full as remainder otherwise, as true remainder is not
    # representable as a single box.

    remainder = dict(bot)
    exactlyOne = False
    fullyInside = False
    for axisTag in bot:
        if axisTag not in intersection:
            fullyInside = False
            continue # Axis range lies fully within
        min1, max1 = intersection[axisTag]
        min2, max2 = bot[axisTag]
        if min1 <= min2 and max2 <= max1:
            continue # Axis range lies fully within

        # Bot's range doesn't fully lie within that of top's for this axis.
        # We know they intersect, so it cannot lie fully without either; so they
        # overlap.

        # If we have had an overlapping axis before, remainder is not
        # representable as a box, so return full bottom and go home.
        if exactlyOne:
            return intersection, bot
        exactlyOne = True
        fullyInside = False

        # Otherwise, cut remainder on this axis and continue.
        if min1 <= min2:
            # Right side survives.
            minimum = max(max1, min2)
            maximum = max2
        elif max2 <= max1:
            # Left side survives.
            minimum = min2
            maximum = min(min1, max2)
        else:
            # Remainder leaks out from both sides.  Can't cut either.
            return intersection, bot

        remainder[axisTag] = minimum,maximum

    if fullyInside:
        # bot is fully within intersection.  Remainder is empty.
        return intersection, None

    return intersection, remainder

def cleanupBox(box):
    """Return a sparse copy of `box`, without redundant (default) values.

        >>> cleanupBox({})
        {}
        >>> cleanupBox({'wdth': (0.0, 1.0)})
        {'wdth': (0.0, 1.0)}
        >>> cleanupBox({'wdth': (-1.0, 1.0)})
        {}

    """
    return {tag: limit for tag, limit in box.items() if limit != (-1.0, 1.0)}


#
# Low level implementation
#

def addFeatureVariationsRaw(font, table, conditionalSubstitutions, featureTag='rvrn'):
    """Low level implementation of addFeatureVariations that directly
    models the possibilities of the FeatureVariations table."""

    #
    # if there is no <featureTag> feature:
    #     make empty <featureTag> feature
    #     sort features, get <featureTag> feature index
    #     add <featureTag> feature to all scripts
    # make lookups
    # add feature variations
    #
    if table.Version < 0x00010001:
        table.Version = 0x00010001  # allow table.FeatureVariations

    table.FeatureVariations = None  # delete any existing FeatureVariations

    varFeatureIndices = []
    for index, feature in enumerate(table.FeatureList.FeatureRecord):
        if feature.FeatureTag == featureTag:
            varFeatureIndices.append(index)

    if not varFeatureIndices:
        varFeature = buildFeatureRecord(featureTag, [])
        table.FeatureList.FeatureRecord.append(varFeature)
        table.FeatureList.FeatureCount = len(table.FeatureList.FeatureRecord)

        sortFeatureList(table)
        varFeatureIndex = table.FeatureList.FeatureRecord.index(varFeature)

        for scriptRecord in table.ScriptList.ScriptRecord:
            if scriptRecord.Script.DefaultLangSys is None:
                raise VarLibError(
                    "Feature variations require that the script "
                    f"'{scriptRecord.ScriptTag}' defines a default language system."
                )
            langSystems = [lsr.LangSys for lsr in scriptRecord.Script.LangSysRecord]
            for langSys in [scriptRecord.Script.DefaultLangSys] + langSystems:
                langSys.FeatureIndex.append(varFeatureIndex)

        varFeatureIndices = [varFeatureIndex]

    axisIndices = {axis.axisTag: axisIndex for axisIndex, axis in enumerate(font["fvar"].axes)}

    featureVariationRecords = []
    for conditionSet, lookupIndices in conditionalSubstitutions:
        conditionTable = []
        for axisTag, (minValue, maxValue) in sorted(conditionSet.items()):
            if minValue > maxValue:
                raise VarLibValidationError(
                    "A condition set has a minimum value above the maximum value."
                )
            ct = buildConditionTable(axisIndices[axisTag], minValue, maxValue)
            conditionTable.append(ct)
        records = []
        for varFeatureIndex in varFeatureIndices:
            existingLookupIndices = table.FeatureList.FeatureRecord[varFeatureIndex].Feature.LookupListIndex
            records.append(buildFeatureTableSubstitutionRecord(varFeatureIndex, existingLookupIndices + lookupIndices))
        featureVariationRecords.append(buildFeatureVariationRecord(conditionTable, records))

    table.FeatureVariations = buildFeatureVariations(featureVariationRecords)


#
# Building GSUB/FeatureVariations internals
#

def buildGSUB():
    """Build a GSUB table from scratch."""
    fontTable = newTable("GSUB")
    gsub = fontTable.table = ot.GSUB()
    gsub.Version = 0x00010001  # allow gsub.FeatureVariations

    gsub.ScriptList = ot.ScriptList()
    gsub.ScriptList.ScriptRecord = []
    gsub.FeatureList = ot.FeatureList()
    gsub.FeatureList.FeatureRecord = []
    gsub.LookupList = ot.LookupList()
    gsub.LookupList.Lookup = []

    srec = ot.ScriptRecord()
    srec.ScriptTag = 'DFLT'
    srec.Script = ot.Script()
    srec.Script.DefaultLangSys = None
    srec.Script.LangSysRecord = []

    langrec = ot.LangSysRecord()
    langrec.LangSys = ot.LangSys()
    langrec.LangSys.ReqFeatureIndex = 0xFFFF
    langrec.LangSys.FeatureIndex = []
    srec.Script.DefaultLangSys = langrec.LangSys

    gsub.ScriptList.ScriptRecord.append(srec)
    gsub.ScriptList.ScriptCount = 1
    gsub.FeatureVariations = None

    return fontTable


def makeSubstitutionsHashable(conditionalSubstitutions):
    """Turn all the substitution dictionaries in sorted tuples of tuples so
    they are hashable, to detect duplicates so we don't write out redundant
    data."""
    allSubstitutions = set()
    condSubst = []
    for conditionSet, substitutionMaps in conditionalSubstitutions:
        substitutions = []
        for substitutionMap in substitutionMaps:
            subst = tuple(sorted(substitutionMap.items()))
            substitutions.append(subst)
            allSubstitutions.add(subst)
        condSubst.append((conditionSet, substitutions))
    return condSubst, sorted(allSubstitutions)


def buildSubstitutionLookups(gsub, allSubstitutions):
    """Build the lookups for the glyph substitutions, return a dict mapping
    the substitution to lookup indices."""
    firstIndex = len(gsub.LookupList.Lookup)
    lookupMap = {}
    for i, substitutionMap in enumerate(allSubstitutions):
        lookupMap[substitutionMap] = i + firstIndex

    for subst in allSubstitutions:
        substMap = dict(subst)
        lookup = buildLookup([buildSingleSubstSubtable(substMap)])
        gsub.LookupList.Lookup.append(lookup)
        assert gsub.LookupList.Lookup[lookupMap[subst]] is lookup
    gsub.LookupList.LookupCount = len(gsub.LookupList.Lookup)
    return lookupMap


def buildFeatureVariations(featureVariationRecords):
    """Build the FeatureVariations subtable."""
    fv = ot.FeatureVariations()
    fv.Version = 0x00010000
    fv.FeatureVariationRecord = featureVariationRecords
    fv.FeatureVariationCount = len(featureVariationRecords)
    return fv


def buildFeatureRecord(featureTag, lookupListIndices):
    """Build a FeatureRecord."""
    fr = ot.FeatureRecord()
    fr.FeatureTag = featureTag
    fr.Feature = ot.Feature()
    fr.Feature.LookupListIndex = lookupListIndices
    fr.Feature.populateDefaults()
    return fr


def buildFeatureVariationRecord(conditionTable, substitutionRecords):
    """Build a FeatureVariationRecord."""
    fvr = ot.FeatureVariationRecord()
    fvr.ConditionSet = ot.ConditionSet()
    fvr.ConditionSet.ConditionTable = conditionTable
    fvr.ConditionSet.ConditionCount = len(conditionTable)
    fvr.FeatureTableSubstitution = ot.FeatureTableSubstitution()
    fvr.FeatureTableSubstitution.Version = 0x00010000
    fvr.FeatureTableSubstitution.SubstitutionRecord = substitutionRecords
    fvr.FeatureTableSubstitution.SubstitutionCount = len(substitutionRecords)
    return fvr


def buildFeatureTableSubstitutionRecord(featureIndex, lookupListIndices):
    """Build a FeatureTableSubstitutionRecord."""
    ftsr = ot.FeatureTableSubstitutionRecord()
    ftsr.FeatureIndex = featureIndex
    ftsr.Feature = ot.Feature()
    ftsr.Feature.LookupListIndex = lookupListIndices
    ftsr.Feature.LookupCount = len(lookupListIndices)
    return ftsr


def buildConditionTable(axisIndex, filterRangeMinValue, filterRangeMaxValue):
    """Build a ConditionTable."""
    ct = ot.ConditionTable()
    ct.Format = 1
    ct.AxisIndex = axisIndex
    ct.FilterRangeMinValue = filterRangeMinValue
    ct.FilterRangeMaxValue = filterRangeMaxValue
    return ct


def sortFeatureList(table):
    """Sort the feature list by feature tag, and remap the feature indices
    elsewhere. This is needed after the feature list has been modified.
    """
    # decorate, sort, undecorate, because we need to make an index remapping table
    tagIndexFea = [(fea.FeatureTag, index, fea) for index, fea in enumerate(table.FeatureList.FeatureRecord)]
    tagIndexFea.sort()
    table.FeatureList.FeatureRecord = [fea for tag, index, fea in tagIndexFea]
    featureRemap = dict(zip([index for tag, index, fea in tagIndexFea], range(len(tagIndexFea))))

    # Remap the feature indices
    remapFeatures(table, featureRemap)


def remapFeatures(table, featureRemap):
    """Go through the scripts list, and remap feature indices."""
    for scriptIndex, script in enumerate(table.ScriptList.ScriptRecord):
        defaultLangSys = script.Script.DefaultLangSys
        if defaultLangSys is not None:
            _remapLangSys(defaultLangSys, featureRemap)
        for langSysRecordIndex, langSysRec in enumerate(script.Script.LangSysRecord):
            langSys = langSysRec.LangSys
            _remapLangSys(langSys, featureRemap)

    if hasattr(table, "FeatureVariations") and table.FeatureVariations is not None:
        for fvr in table.FeatureVariations.FeatureVariationRecord:
            for ftsr in fvr.FeatureTableSubstitution.SubstitutionRecord:
                ftsr.FeatureIndex = featureRemap[ftsr.FeatureIndex]


def _remapLangSys(langSys, featureRemap):
    if langSys.ReqFeatureIndex != 0xffff:
        langSys.ReqFeatureIndex = featureRemap[langSys.ReqFeatureIndex]
    langSys.FeatureIndex = [featureRemap[index] for index in langSys.FeatureIndex]


if __name__ == "__main__":
    import doctest, sys
    sys.exit(doctest.testmod().failed)
