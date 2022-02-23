from collections import namedtuple, OrderedDict
import os
from fontTools.misc.fixedTools import fixedToFloat
from fontTools import ttLib
from fontTools.ttLib.tables import otTables as ot
from fontTools.ttLib.tables.otBase import (
    ValueRecord,
    valueRecordFormatDict,
    OTTableWriter,
    CountReference,
)
from fontTools.ttLib.tables import otBase
from fontTools.feaLib.ast import STATNameStatement
from fontTools.otlLib.optimize.gpos import GPOS_COMPACT_MODE_DEFAULT, GPOS_COMPACT_MODE_ENV_KEY, compact_lookup
from fontTools.otlLib.error import OpenTypeLibError
from functools import reduce
import logging
import copy


log = logging.getLogger(__name__)


def buildCoverage(glyphs, glyphMap):
    """Builds a coverage table.

    Coverage tables (as defined in the `OpenType spec <https://docs.microsoft.com/en-gb/typography/opentype/spec/chapter2#coverage-table>`__)
    are used in all OpenType Layout lookups apart from the Extension type, and
    define the glyphs involved in a layout subtable. This allows shaping engines
    to compare the glyph stream with the coverage table and quickly determine
    whether a subtable should be involved in a shaping operation.

    This function takes a list of glyphs and a glyphname-to-ID map, and
    returns a ``Coverage`` object representing the coverage table.

    Example::

        glyphMap = font.getReverseGlyphMap()
        glyphs = [ "A", "B", "C" ]
        coverage = buildCoverage(glyphs, glyphMap)

    Args:
        glyphs: a sequence of glyph names.
        glyphMap: a glyph name to ID map, typically returned from
            ``font.getReverseGlyphMap()``.

    Returns:
        An ``otTables.Coverage`` object or ``None`` if there are no glyphs
        supplied.
    """

    if not glyphs:
        return None
    self = ot.Coverage()
    self.glyphs = sorted(set(glyphs), key=glyphMap.__getitem__)
    return self


LOOKUP_FLAG_RIGHT_TO_LEFT = 0x0001
LOOKUP_FLAG_IGNORE_BASE_GLYPHS = 0x0002
LOOKUP_FLAG_IGNORE_LIGATURES = 0x0004
LOOKUP_FLAG_IGNORE_MARKS = 0x0008
LOOKUP_FLAG_USE_MARK_FILTERING_SET = 0x0010


def buildLookup(subtables, flags=0, markFilterSet=None):
    """Turns a collection of rules into a lookup.

    A Lookup (as defined in the `OpenType Spec <https://docs.microsoft.com/en-gb/typography/opentype/spec/chapter2#lookupTbl>`__)
    wraps the individual rules in a layout operation (substitution or
    positioning) in a data structure expressing their overall lookup type -
    for example, single substitution, mark-to-base attachment, and so on -
    as well as the lookup flags and any mark filtering sets. You may import
    the following constants to express lookup flags:

    - ``LOOKUP_FLAG_RIGHT_TO_LEFT``
    - ``LOOKUP_FLAG_IGNORE_BASE_GLYPHS``
    - ``LOOKUP_FLAG_IGNORE_LIGATURES``
    - ``LOOKUP_FLAG_IGNORE_MARKS``
    - ``LOOKUP_FLAG_USE_MARK_FILTERING_SET``

    Args:
        subtables: A list of layout subtable objects (e.g.
            ``MultipleSubst``, ``PairPos``, etc.) or ``None``.
        flags (int): This lookup's flags.
        markFilterSet: Either ``None`` if no mark filtering set is used, or
            an integer representing the filtering set to be used for this
            lookup. If a mark filtering set is provided,
            `LOOKUP_FLAG_USE_MARK_FILTERING_SET` will be set on the lookup's
            flags.

    Returns:
        An ``otTables.Lookup`` object or ``None`` if there are no subtables
        supplied.
    """
    if subtables is None:
        return None
    subtables = [st for st in subtables if st is not None]
    if not subtables:
        return None
    assert all(
        t.LookupType == subtables[0].LookupType for t in subtables
    ), "all subtables must have the same LookupType; got %s" % repr(
        [t.LookupType for t in subtables]
    )
    self = ot.Lookup()
    self.LookupType = subtables[0].LookupType
    self.LookupFlag = flags
    self.SubTable = subtables
    self.SubTableCount = len(self.SubTable)
    if markFilterSet is not None:
        self.LookupFlag |= LOOKUP_FLAG_USE_MARK_FILTERING_SET
        assert isinstance(markFilterSet, int), markFilterSet
        self.MarkFilteringSet = markFilterSet
    else:
        assert (self.LookupFlag & LOOKUP_FLAG_USE_MARK_FILTERING_SET) == 0, (
            "if markFilterSet is None, flags must not set "
            "LOOKUP_FLAG_USE_MARK_FILTERING_SET; flags=0x%04x" % flags
        )
    return self


class LookupBuilder(object):
    SUBTABLE_BREAK_ = "SUBTABLE_BREAK"

    def __init__(self, font, location, table, lookup_type):
        self.font = font
        self.glyphMap = font.getReverseGlyphMap()
        self.location = location
        self.table, self.lookup_type = table, lookup_type
        self.lookupflag = 0
        self.markFilterSet = None
        self.lookup_index = None  # assigned when making final tables
        assert table in ("GPOS", "GSUB")

    def equals(self, other):
        return (
            isinstance(other, self.__class__)
            and self.table == other.table
            and self.lookupflag == other.lookupflag
            and self.markFilterSet == other.markFilterSet
        )

    def inferGlyphClasses(self):
        """Infers glyph glasses for the GDEF table, such as {"cedilla":3}."""
        return {}

    def getAlternateGlyphs(self):
        """Helper for building 'aalt' features."""
        return {}

    def buildLookup_(self, subtables):
        return buildLookup(subtables, self.lookupflag, self.markFilterSet)

    def buildMarkClasses_(self, marks):
        """{"cedilla": ("BOTTOM", ast.Anchor), ...} --> {"BOTTOM":0, "TOP":1}

        Helper for MarkBasePostBuilder, MarkLigPosBuilder, and
        MarkMarkPosBuilder. Seems to return the same numeric IDs
        for mark classes as the AFDKO makeotf tool.
        """
        ids = {}
        for mark in sorted(marks.keys(), key=self.font.getGlyphID):
            markClassName, _markAnchor = marks[mark]
            if markClassName not in ids:
                ids[markClassName] = len(ids)
        return ids

    def setBacktrackCoverage_(self, prefix, subtable):
        subtable.BacktrackGlyphCount = len(prefix)
        subtable.BacktrackCoverage = []
        for p in reversed(prefix):
            coverage = buildCoverage(p, self.glyphMap)
            subtable.BacktrackCoverage.append(coverage)

    def setLookAheadCoverage_(self, suffix, subtable):
        subtable.LookAheadGlyphCount = len(suffix)
        subtable.LookAheadCoverage = []
        for s in suffix:
            coverage = buildCoverage(s, self.glyphMap)
            subtable.LookAheadCoverage.append(coverage)

    def setInputCoverage_(self, glyphs, subtable):
        subtable.InputGlyphCount = len(glyphs)
        subtable.InputCoverage = []
        for g in glyphs:
            coverage = buildCoverage(g, self.glyphMap)
            subtable.InputCoverage.append(coverage)

    def setCoverage_(self, glyphs, subtable):
        subtable.GlyphCount = len(glyphs)
        subtable.Coverage = []
        for g in glyphs:
            coverage = buildCoverage(g, self.glyphMap)
            subtable.Coverage.append(coverage)

    def build_subst_subtables(self, mapping, klass):
        substitutions = [{}]
        for key in mapping:
            if key[0] == self.SUBTABLE_BREAK_:
                substitutions.append({})
            else:
                substitutions[-1][key] = mapping[key]
        subtables = [klass(s) for s in substitutions]
        return subtables

    def add_subtable_break(self, location):
        """Add an explicit subtable break.

        Args:
            location: A string or tuple representing the location in the
                original source which produced this break, or ``None`` if
                no location is provided.
        """
        log.warning(
            OpenTypeLibError(
                'unsupported "subtable" statement for lookup type', location
            )
        )


class AlternateSubstBuilder(LookupBuilder):
    """Builds an Alternate Substitution (GSUB3) lookup.

    Users are expected to manually add alternate glyph substitutions to
    the ``alternates`` attribute after the object has been initialized,
    e.g.::

        builder.alternates["A"] = ["A.alt1", "A.alt2"]

    Attributes:
        font (``fontTools.TTLib.TTFont``): A font object.
        location: A string or tuple representing the location in the original
            source which produced this lookup.
        alternates: An ordered dictionary of alternates, mapping glyph names
            to a list of names of alternates.
        lookupflag (int): The lookup's flag
        markFilterSet: Either ``None`` if no mark filtering set is used, or
            an integer representing the filtering set to be used for this
            lookup. If a mark filtering set is provided,
            `LOOKUP_FLAG_USE_MARK_FILTERING_SET` will be set on the lookup's
            flags.
    """

    def __init__(self, font, location):
        LookupBuilder.__init__(self, font, location, "GSUB", 3)
        self.alternates = OrderedDict()

    def equals(self, other):
        return LookupBuilder.equals(self, other) and self.alternates == other.alternates

    def build(self):
        """Build the lookup.

        Returns:
            An ``otTables.Lookup`` object representing the alternate
            substitution lookup.
        """
        subtables = self.build_subst_subtables(
            self.alternates, buildAlternateSubstSubtable
        )
        return self.buildLookup_(subtables)

    def getAlternateGlyphs(self):
        return self.alternates

    def add_subtable_break(self, location):
        self.alternates[(self.SUBTABLE_BREAK_, location)] = self.SUBTABLE_BREAK_


class ChainContextualRule(
    namedtuple("ChainContextualRule", ["prefix", "glyphs", "suffix", "lookups"])
):
    @property
    def is_subtable_break(self):
        return self.prefix == LookupBuilder.SUBTABLE_BREAK_


class ChainContextualRuleset:
    def __init__(self):
        self.rules = []

    def addRule(self, rule):
        self.rules.append(rule)

    @property
    def hasPrefixOrSuffix(self):
        # Do we have any prefixes/suffixes? If this is False for all
        # rulesets, we can express the whole lookup as GPOS5/GSUB7.
        for rule in self.rules:
            if len(rule.prefix) > 0 or len(rule.suffix) > 0:
                return True
        return False

    @property
    def hasAnyGlyphClasses(self):
        # Do we use glyph classes anywhere in the rules? If this is False
        # we can express this subtable as a Format 1.
        for rule in self.rules:
            for coverage in (rule.prefix, rule.glyphs, rule.suffix):
                if any(len(x) > 1 for x in coverage):
                    return True
        return False

    def format2ClassDefs(self):
        PREFIX, GLYPHS, SUFFIX = 0, 1, 2
        classDefBuilders = []
        for ix in [PREFIX, GLYPHS, SUFFIX]:
            context = []
            for r in self.rules:
                context.append(r[ix])
            classes = self._classBuilderForContext(context)
            if not classes:
                return None
            classDefBuilders.append(classes)
        return classDefBuilders

    def _classBuilderForContext(self, context):
        classdefbuilder = ClassDefBuilder(useClass0=False)
        for position in context:
            for glyphset in position:
                glyphs = set(glyphset)
                if not classdefbuilder.canAdd(glyphs):
                    return None
                classdefbuilder.add(glyphs)
        return classdefbuilder


class ChainContextualBuilder(LookupBuilder):
    def equals(self, other):
        return LookupBuilder.equals(self, other) and self.rules == other.rules

    def rulesets(self):
        # Return a list of ChainContextRuleset objects, taking explicit
        # subtable breaks into account
        ruleset = [ChainContextualRuleset()]
        for rule in self.rules:
            if rule.is_subtable_break:
                ruleset.append(ChainContextualRuleset())
                continue
            ruleset[-1].addRule(rule)
        # Squish any empty subtables
        return [x for x in ruleset if len(x.rules) > 0]

    def getCompiledSize_(self, subtables):
        size = 0
        for st in subtables:
            w = OTTableWriter()
            w["LookupType"] = CountReference(
                {"LookupType": st.LookupType}, "LookupType"
            )
            # We need to make a copy here because compiling
            # modifies the subtable (finalizing formats etc.)
            copy.deepcopy(st).compile(w, self.font)
            size += len(w.getAllData())
        return size

    def build(self):
        """Build the lookup.

        Returns:
            An ``otTables.Lookup`` object representing the chained
            contextual positioning lookup.
        """
        subtables = []
        chaining = False
        rulesets = self.rulesets()
        chaining = any(ruleset.hasPrefixOrSuffix for ruleset in rulesets)
        for ruleset in rulesets:
            # Determine format strategy. We try to build formats 1, 2 and 3
            # subtables and then work out which is best. candidates list holds
            # the subtables in each format for this ruleset (including a dummy
            # "format 0" to make the addressing match the format numbers).

            # We can always build a format 3 lookup by accumulating each of
            # the rules into a list, so start with that.
            candidates = [None, None, None, []]
            for rule in ruleset.rules:
                candidates[3].append(self.buildFormat3Subtable(rule, chaining))

            # Can we express the whole ruleset as a format 2 subtable?
            classdefs = ruleset.format2ClassDefs()
            if classdefs:
                candidates[2] = [
                    self.buildFormat2Subtable(ruleset, classdefs, chaining)
                ]

            if not ruleset.hasAnyGlyphClasses:
                candidates[1] = [self.buildFormat1Subtable(ruleset, chaining)]

            for i in [1, 2, 3]:
                if candidates[i]:
                    try:
                        self.getCompiledSize_(candidates[i])
                    except Exception as e:
                        log.warning(
                            "Contextual format %i at %s overflowed (%s)"
                            % (i, str(self.location), e)
                        )
                        candidates[i] = None

            candidates = [x for x in candidates if x is not None]
            if not candidates:
                raise OpenTypeLibError("All candidates overflowed", self.location)

            winner = min(candidates, key=self.getCompiledSize_)
            subtables.extend(winner)

        # If we are not chaining, lookup type will be automatically fixed by
        # buildLookup_
        return self.buildLookup_(subtables)

    def buildFormat1Subtable(self, ruleset, chaining=True):
        st = self.newSubtable_(chaining=chaining)
        st.Format = 1
        st.populateDefaults()
        coverage = set()
        rulesetsByFirstGlyph = {}
        ruleAttr = self.ruleAttr_(format=1, chaining=chaining)

        for rule in ruleset.rules:
            ruleAsSubtable = self.newRule_(format=1, chaining=chaining)

            if chaining:
                ruleAsSubtable.BacktrackGlyphCount = len(rule.prefix)
                ruleAsSubtable.LookAheadGlyphCount = len(rule.suffix)
                ruleAsSubtable.Backtrack = [list(x)[0] for x in reversed(rule.prefix)]
                ruleAsSubtable.LookAhead = [list(x)[0] for x in rule.suffix]

                ruleAsSubtable.InputGlyphCount = len(rule.glyphs)
            else:
                ruleAsSubtable.GlyphCount = len(rule.glyphs)

            ruleAsSubtable.Input = [list(x)[0] for x in rule.glyphs[1:]]

            self.buildLookupList(rule, ruleAsSubtable)

            firstGlyph = list(rule.glyphs[0])[0]
            if firstGlyph not in rulesetsByFirstGlyph:
                coverage.add(firstGlyph)
                rulesetsByFirstGlyph[firstGlyph] = []
            rulesetsByFirstGlyph[firstGlyph].append(ruleAsSubtable)

        st.Coverage = buildCoverage(coverage, self.glyphMap)
        ruleSets = []
        for g in st.Coverage.glyphs:
            ruleSet = self.newRuleSet_(format=1, chaining=chaining)
            setattr(ruleSet, ruleAttr, rulesetsByFirstGlyph[g])
            setattr(ruleSet, f"{ruleAttr}Count", len(rulesetsByFirstGlyph[g]))
            ruleSets.append(ruleSet)

        setattr(st, self.ruleSetAttr_(format=1, chaining=chaining), ruleSets)
        setattr(
            st, self.ruleSetAttr_(format=1, chaining=chaining) + "Count", len(ruleSets)
        )

        return st

    def buildFormat2Subtable(self, ruleset, classdefs, chaining=True):
        st = self.newSubtable_(chaining=chaining)
        st.Format = 2
        st.populateDefaults()

        if chaining:
            (
                st.BacktrackClassDef,
                st.InputClassDef,
                st.LookAheadClassDef,
            ) = [c.build() for c in classdefs]
        else:
            st.ClassDef = classdefs[1].build()

        inClasses = classdefs[1].classes()

        classSets = []
        for _ in inClasses:
            classSet = self.newRuleSet_(format=2, chaining=chaining)
            classSets.append(classSet)

        coverage = set()
        classRuleAttr = self.ruleAttr_(format=2, chaining=chaining)

        for rule in ruleset.rules:
            ruleAsSubtable = self.newRule_(format=2, chaining=chaining)
            if chaining:
                ruleAsSubtable.BacktrackGlyphCount = len(rule.prefix)
                ruleAsSubtable.LookAheadGlyphCount = len(rule.suffix)
                # The glyphs in the rule may be list, tuple, odict_keys...
                # Order is not important anyway because they are guaranteed
                # to be members of the same class.
                ruleAsSubtable.Backtrack = [
                    st.BacktrackClassDef.classDefs[list(x)[0]]
                    for x in reversed(rule.prefix)
                ]
                ruleAsSubtable.LookAhead = [
                    st.LookAheadClassDef.classDefs[list(x)[0]] for x in rule.suffix
                ]

                ruleAsSubtable.InputGlyphCount = len(rule.glyphs)
                ruleAsSubtable.Input = [
                    st.InputClassDef.classDefs[list(x)[0]] for x in rule.glyphs[1:]
                ]
                setForThisRule = classSets[
                    st.InputClassDef.classDefs[list(rule.glyphs[0])[0]]
                ]
            else:
                ruleAsSubtable.GlyphCount = len(rule.glyphs)
                ruleAsSubtable.Class = [  # The spec calls this InputSequence
                    st.ClassDef.classDefs[list(x)[0]] for x in rule.glyphs[1:]
                ]
                setForThisRule = classSets[
                    st.ClassDef.classDefs[list(rule.glyphs[0])[0]]
                ]

            self.buildLookupList(rule, ruleAsSubtable)
            coverage |= set(rule.glyphs[0])

            getattr(setForThisRule, classRuleAttr).append(ruleAsSubtable)
            setattr(
                setForThisRule,
                f"{classRuleAttr}Count",
                getattr(setForThisRule, f"{classRuleAttr}Count") + 1,
            )
        setattr(st, self.ruleSetAttr_(format=2, chaining=chaining), classSets)
        setattr(
            st, self.ruleSetAttr_(format=2, chaining=chaining) + "Count", len(classSets)
        )
        st.Coverage = buildCoverage(coverage, self.glyphMap)
        return st

    def buildFormat3Subtable(self, rule, chaining=True):
        st = self.newSubtable_(chaining=chaining)
        st.Format = 3
        if chaining:
            self.setBacktrackCoverage_(rule.prefix, st)
            self.setLookAheadCoverage_(rule.suffix, st)
            self.setInputCoverage_(rule.glyphs, st)
        else:
            self.setCoverage_(rule.glyphs, st)
        self.buildLookupList(rule, st)
        return st

    def buildLookupList(self, rule, st):
        for sequenceIndex, lookupList in enumerate(rule.lookups):
            if lookupList is not None:
                if not isinstance(lookupList, list):
                    # Can happen with synthesised lookups
                    lookupList = [lookupList]
                for l in lookupList:
                    if l.lookup_index is None:
                        if isinstance(self, ChainContextPosBuilder):
                            other = "substitution"
                        else:
                            other = "positioning"
                        raise OpenTypeLibError(
                            "Missing index of the specified "
                            f"lookup, might be a {other} lookup",
                            self.location,
                        )
                    rec = self.newLookupRecord_(st)
                    rec.SequenceIndex = sequenceIndex
                    rec.LookupListIndex = l.lookup_index

    def add_subtable_break(self, location):
        self.rules.append(
            ChainContextualRule(
                self.SUBTABLE_BREAK_,
                self.SUBTABLE_BREAK_,
                self.SUBTABLE_BREAK_,
                [self.SUBTABLE_BREAK_],
            )
        )

    def newSubtable_(self, chaining=True):
        subtablename = f"Context{self.subtable_type}"
        if chaining:
            subtablename = "Chain" + subtablename
        st = getattr(ot, subtablename)()  # ot.ChainContextPos()/ot.ChainSubst()/etc.
        setattr(st, f"{self.subtable_type}Count", 0)
        setattr(st, f"{self.subtable_type}LookupRecord", [])
        return st

    # Format 1 and format 2 GSUB5/GSUB6/GPOS7/GPOS8 rulesets and rules form a family:
    #
    #       format 1 ruleset      format 1 rule      format 2 ruleset      format 2 rule
    # GSUB5 SubRuleSet            SubRule            SubClassSet           SubClassRule
    # GSUB6 ChainSubRuleSet       ChainSubRule       ChainSubClassSet      ChainSubClassRule
    # GPOS7 PosRuleSet            PosRule            PosClassSet           PosClassRule
    # GPOS8 ChainPosRuleSet       ChainPosRule       ChainPosClassSet      ChainPosClassRule
    #
    # The following functions generate the attribute names and subtables according
    # to this naming convention.
    def ruleSetAttr_(self, format=1, chaining=True):
        if format == 1:
            formatType = "Rule"
        elif format == 2:
            formatType = "Class"
        else:
            raise AssertionError(formatType)
        subtablename = f"{self.subtable_type[0:3]}{formatType}Set"  # Sub, not Subst.
        if chaining:
            subtablename = "Chain" + subtablename
        return subtablename

    def ruleAttr_(self, format=1, chaining=True):
        if format == 1:
            formatType = ""
        elif format == 2:
            formatType = "Class"
        else:
            raise AssertionError(formatType)
        subtablename = f"{self.subtable_type[0:3]}{formatType}Rule"  # Sub, not Subst.
        if chaining:
            subtablename = "Chain" + subtablename
        return subtablename

    def newRuleSet_(self, format=1, chaining=True):
        st = getattr(
            ot, self.ruleSetAttr_(format, chaining)
        )()  # ot.ChainPosRuleSet()/ot.SubRuleSet()/etc.
        st.populateDefaults()
        return st

    def newRule_(self, format=1, chaining=True):
        st = getattr(
            ot, self.ruleAttr_(format, chaining)
        )()  # ot.ChainPosClassRule()/ot.SubClassRule()/etc.
        st.populateDefaults()
        return st

    def attachSubtableWithCount_(
        self, st, subtable_name, count_name, existing=None, index=None, chaining=False
    ):
        if chaining:
            subtable_name = "Chain" + subtable_name
            count_name = "Chain" + count_name

        if not hasattr(st, count_name):
            setattr(st, count_name, 0)
            setattr(st, subtable_name, [])

        if existing:
            new_subtable = existing
        else:
            # Create a new, empty subtable from otTables
            new_subtable = getattr(ot, subtable_name)()

        setattr(st, count_name, getattr(st, count_name) + 1)

        if index:
            getattr(st, subtable_name).insert(index, new_subtable)
        else:
            getattr(st, subtable_name).append(new_subtable)

        return new_subtable

    def newLookupRecord_(self, st):
        return self.attachSubtableWithCount_(
            st,
            f"{self.subtable_type}LookupRecord",
            f"{self.subtable_type}Count",
            chaining=False,
        )  # Oddly, it isn't ChainSubstLookupRecord


class ChainContextPosBuilder(ChainContextualBuilder):
    """Builds a Chained Contextual Positioning (GPOS8) lookup.

    Users are expected to manually add rules to the ``rules`` attribute after
    the object has been initialized, e.g.::

        # pos [A B] [C D] x' lookup lu1 y' z' lookup lu2 E;

        prefix  = [ ["A", "B"], ["C", "D"] ]
        suffix  = [ ["E"] ]
        glyphs  = [ ["x"], ["y"], ["z"] ]
        lookups = [ [lu1], None,  [lu2] ]
        builder.rules.append( (prefix, glyphs, suffix, lookups) )

    Attributes:
        font (``fontTools.TTLib.TTFont``): A font object.
        location: A string or tuple representing the location in the original
            source which produced this lookup.
        rules: A list of tuples representing the rules in this lookup.
        lookupflag (int): The lookup's flag
        markFilterSet: Either ``None`` if no mark filtering set is used, or
            an integer representing the filtering set to be used for this
            lookup. If a mark filtering set is provided,
            `LOOKUP_FLAG_USE_MARK_FILTERING_SET` will be set on the lookup's
            flags.
    """

    def __init__(self, font, location):
        LookupBuilder.__init__(self, font, location, "GPOS", 8)
        self.rules = []
        self.subtable_type = "Pos"

    def find_chainable_single_pos(self, lookups, glyphs, value):
        """Helper for add_single_pos_chained_()"""
        res = None
        for lookup in lookups[::-1]:
            if lookup == self.SUBTABLE_BREAK_:
                return res
            if isinstance(lookup, SinglePosBuilder) and all(
                lookup.can_add(glyph, value) for glyph in glyphs
            ):
                res = lookup
        return res


class ChainContextSubstBuilder(ChainContextualBuilder):
    """Builds a Chained Contextual Substitution (GSUB6) lookup.

    Users are expected to manually add rules to the ``rules`` attribute after
    the object has been initialized, e.g.::

        # sub [A B] [C D] x' lookup lu1 y' z' lookup lu2 E;

        prefix  = [ ["A", "B"], ["C", "D"] ]
        suffix  = [ ["E"] ]
        glyphs  = [ ["x"], ["y"], ["z"] ]
        lookups = [ [lu1], None,  [lu2] ]
        builder.rules.append( (prefix, glyphs, suffix, lookups) )

    Attributes:
        font (``fontTools.TTLib.TTFont``): A font object.
        location: A string or tuple representing the location in the original
            source which produced this lookup.
        rules: A list of tuples representing the rules in this lookup.
        lookupflag (int): The lookup's flag
        markFilterSet: Either ``None`` if no mark filtering set is used, or
            an integer representing the filtering set to be used for this
            lookup. If a mark filtering set is provided,
            `LOOKUP_FLAG_USE_MARK_FILTERING_SET` will be set on the lookup's
            flags.
    """

    def __init__(self, font, location):
        LookupBuilder.__init__(self, font, location, "GSUB", 6)
        self.rules = []  # (prefix, input, suffix, lookups)
        self.subtable_type = "Subst"

    def getAlternateGlyphs(self):
        result = {}
        for rule in self.rules:
            if rule.is_subtable_break:
                continue
            for lookups in rule.lookups:
                if not isinstance(lookups, list):
                    lookups = [lookups]
                for lookup in lookups:
                    if lookup is not None:
                        alts = lookup.getAlternateGlyphs()
                        for glyph, replacements in alts.items():
                            result.setdefault(glyph, set()).update(replacements)
        return result

    def find_chainable_single_subst(self, glyphs):
        """Helper for add_single_subst_chained_()"""
        res = None
        for rule in self.rules[::-1]:
            if rule.is_subtable_break:
                return res
            for sub in rule.lookups:
                if isinstance(sub, SingleSubstBuilder) and not any(
                    g in glyphs for g in sub.mapping.keys()
                ):
                    res = sub
        return res


class LigatureSubstBuilder(LookupBuilder):
    """Builds a Ligature Substitution (GSUB4) lookup.

    Users are expected to manually add ligatures to the ``ligatures``
    attribute after the object has been initialized, e.g.::

        # sub f i by f_i;
        builder.ligatures[("f","f","i")] = "f_f_i"

    Attributes:
        font (``fontTools.TTLib.TTFont``): A font object.
        location: A string or tuple representing the location in the original
            source which produced this lookup.
        ligatures: An ordered dictionary mapping a tuple of glyph names to the
            ligature glyphname.
        lookupflag (int): The lookup's flag
        markFilterSet: Either ``None`` if no mark filtering set is used, or
            an integer representing the filtering set to be used for this
            lookup. If a mark filtering set is provided,
            `LOOKUP_FLAG_USE_MARK_FILTERING_SET` will be set on the lookup's
            flags.
    """

    def __init__(self, font, location):
        LookupBuilder.__init__(self, font, location, "GSUB", 4)
        self.ligatures = OrderedDict()  # {('f','f','i'): 'f_f_i'}

    def equals(self, other):
        return LookupBuilder.equals(self, other) and self.ligatures == other.ligatures

    def build(self):
        """Build the lookup.

        Returns:
            An ``otTables.Lookup`` object representing the ligature
            substitution lookup.
        """
        subtables = self.build_subst_subtables(
            self.ligatures, buildLigatureSubstSubtable
        )
        return self.buildLookup_(subtables)

    def add_subtable_break(self, location):
        self.ligatures[(self.SUBTABLE_BREAK_, location)] = self.SUBTABLE_BREAK_


class MultipleSubstBuilder(LookupBuilder):
    """Builds a Multiple Substitution (GSUB2) lookup.

    Users are expected to manually add substitutions to the ``mapping``
    attribute after the object has been initialized, e.g.::

        # sub uni06C0 by uni06D5.fina hamza.above;
        builder.mapping["uni06C0"] = [ "uni06D5.fina", "hamza.above"]

    Attributes:
        font (``fontTools.TTLib.TTFont``): A font object.
        location: A string or tuple representing the location in the original
            source which produced this lookup.
        mapping: An ordered dictionary mapping a glyph name to a list of
            substituted glyph names.
        lookupflag (int): The lookup's flag
        markFilterSet: Either ``None`` if no mark filtering set is used, or
            an integer representing the filtering set to be used for this
            lookup. If a mark filtering set is provided,
            `LOOKUP_FLAG_USE_MARK_FILTERING_SET` will be set on the lookup's
            flags.
    """

    def __init__(self, font, location):
        LookupBuilder.__init__(self, font, location, "GSUB", 2)
        self.mapping = OrderedDict()

    def equals(self, other):
        return LookupBuilder.equals(self, other) and self.mapping == other.mapping

    def build(self):
        subtables = self.build_subst_subtables(self.mapping, buildMultipleSubstSubtable)
        return self.buildLookup_(subtables)

    def add_subtable_break(self, location):
        self.mapping[(self.SUBTABLE_BREAK_, location)] = self.SUBTABLE_BREAK_


class CursivePosBuilder(LookupBuilder):
    """Builds a Cursive Positioning (GPOS3) lookup.

    Attributes:
        font (``fontTools.TTLib.TTFont``): A font object.
        location: A string or tuple representing the location in the original
            source which produced this lookup.
        attachments: An ordered dictionary mapping a glyph name to a two-element
            tuple of ``otTables.Anchor`` objects.
        lookupflag (int): The lookup's flag
        markFilterSet: Either ``None`` if no mark filtering set is used, or
            an integer representing the filtering set to be used for this
            lookup. If a mark filtering set is provided,
            `LOOKUP_FLAG_USE_MARK_FILTERING_SET` will be set on the lookup's
            flags.
    """

    def __init__(self, font, location):
        LookupBuilder.__init__(self, font, location, "GPOS", 3)
        self.attachments = {}

    def equals(self, other):
        return (
            LookupBuilder.equals(self, other) and self.attachments == other.attachments
        )

    def add_attachment(self, location, glyphs, entryAnchor, exitAnchor):
        """Adds attachment information to the cursive positioning lookup.

        Args:
            location: A string or tuple representing the location in the
                original source which produced this lookup. (Unused.)
            glyphs: A list of glyph names sharing these entry and exit
                anchor locations.
            entryAnchor: A ``otTables.Anchor`` object representing the
                entry anchor, or ``None`` if no entry anchor is present.
            exitAnchor: A ``otTables.Anchor`` object representing the
                exit anchor, or ``None`` if no exit anchor is present.
        """
        for glyph in glyphs:
            self.attachments[glyph] = (entryAnchor, exitAnchor)

    def build(self):
        """Build the lookup.

        Returns:
            An ``otTables.Lookup`` object representing the cursive
            positioning lookup.
        """
        st = buildCursivePosSubtable(self.attachments, self.glyphMap)
        return self.buildLookup_([st])


class MarkBasePosBuilder(LookupBuilder):
    """Builds a Mark-To-Base Positioning (GPOS4) lookup.

    Users are expected to manually add marks and bases to the ``marks``
    and ``bases`` attributes after the object has been initialized, e.g.::

        builder.marks["acute"]   = (0, a1)
        builder.marks["grave"]   = (0, a1)
        builder.marks["cedilla"] = (1, a2)
        builder.bases["a"] = {0: a3, 1: a5}
        builder.bases["b"] = {0: a4, 1: a5}

    Attributes:
        font (``fontTools.TTLib.TTFont``): A font object.
        location: A string or tuple representing the location in the original
            source which produced this lookup.
        marks: An dictionary mapping a glyph name to a two-element
            tuple containing a mark class ID and ``otTables.Anchor`` object.
        bases: An dictionary mapping a glyph name to a dictionary of
            mark class IDs and ``otTables.Anchor`` object.
        lookupflag (int): The lookup's flag
        markFilterSet: Either ``None`` if no mark filtering set is used, or
            an integer representing the filtering set to be used for this
            lookup. If a mark filtering set is provided,
            `LOOKUP_FLAG_USE_MARK_FILTERING_SET` will be set on the lookup's
            flags.
    """

    def __init__(self, font, location):
        LookupBuilder.__init__(self, font, location, "GPOS", 4)
        self.marks = {}  # glyphName -> (markClassName, anchor)
        self.bases = {}  # glyphName -> {markClassName: anchor}

    def equals(self, other):
        return (
            LookupBuilder.equals(self, other)
            and self.marks == other.marks
            and self.bases == other.bases
        )

    def inferGlyphClasses(self):
        result = {glyph: 1 for glyph in self.bases}
        result.update({glyph: 3 for glyph in self.marks})
        return result

    def build(self):
        """Build the lookup.

        Returns:
            An ``otTables.Lookup`` object representing the mark-to-base
            positioning lookup.
        """
        markClasses = self.buildMarkClasses_(self.marks)
        marks = {}
        for mark, (mc, anchor) in self.marks.items():
            if mc not in markClasses:
                raise ValueError("Mark class %s not found for mark glyph %s" % (mc, mark))
            marks[mark] = (markClasses[mc], anchor)
        bases = {}
        for glyph, anchors in self.bases.items():
            bases[glyph] = {}
            for mc, anchor in anchors.items():
                if mc not in markClasses:
                    raise ValueError("Mark class %s not found for base glyph %s" % (mc, mark))
                bases[glyph][markClasses[mc]] = anchor
        subtables = buildMarkBasePos(marks, bases, self.glyphMap)
        return self.buildLookup_(subtables)


class MarkLigPosBuilder(LookupBuilder):
    """Builds a Mark-To-Ligature Positioning (GPOS5) lookup.

    Users are expected to manually add marks and bases to the ``marks``
    and ``ligatures`` attributes after the object has been initialized, e.g.::

        builder.marks["acute"]   = (0, a1)
        builder.marks["grave"]   = (0, a1)
        builder.marks["cedilla"] = (1, a2)
        builder.ligatures["f_i"] = [
            { 0: a3, 1: a5 }, # f
            { 0: a4, 1: a5 }  # i
        ]

    Attributes:
        font (``fontTools.TTLib.TTFont``): A font object.
        location: A string or tuple representing the location in the original
            source which produced this lookup.
        marks: An dictionary mapping a glyph name to a two-element
            tuple containing a mark class ID and ``otTables.Anchor`` object.
        ligatures: An dictionary mapping a glyph name to an array with one
            element for each ligature component. Each array element should be
            a dictionary mapping mark class IDs to ``otTables.Anchor`` objects.
        lookupflag (int): The lookup's flag
        markFilterSet: Either ``None`` if no mark filtering set is used, or
            an integer representing the filtering set to be used for this
            lookup. If a mark filtering set is provided,
            `LOOKUP_FLAG_USE_MARK_FILTERING_SET` will be set on the lookup's
            flags.
    """

    def __init__(self, font, location):
        LookupBuilder.__init__(self, font, location, "GPOS", 5)
        self.marks = {}  # glyphName -> (markClassName, anchor)
        self.ligatures = {}  # glyphName -> [{markClassName: anchor}, ...]

    def equals(self, other):
        return (
            LookupBuilder.equals(self, other)
            and self.marks == other.marks
            and self.ligatures == other.ligatures
        )

    def inferGlyphClasses(self):
        result = {glyph: 2 for glyph in self.ligatures}
        result.update({glyph: 3 for glyph in self.marks})
        return result

    def build(self):
        """Build the lookup.

        Returns:
            An ``otTables.Lookup`` object representing the mark-to-ligature
            positioning lookup.
        """
        markClasses = self.buildMarkClasses_(self.marks)
        marks = {
            mark: (markClasses[mc], anchor) for mark, (mc, anchor) in self.marks.items()
        }
        ligs = {}
        for lig, components in self.ligatures.items():
            ligs[lig] = []
            for c in components:
                ligs[lig].append({markClasses[mc]: a for mc, a in c.items()})
        subtables = buildMarkLigPos(marks, ligs, self.glyphMap)
        return self.buildLookup_(subtables)


class MarkMarkPosBuilder(LookupBuilder):
    """Builds a Mark-To-Mark Positioning (GPOS6) lookup.

    Users are expected to manually add marks and bases to the ``marks``
    and ``baseMarks`` attributes after the object has been initialized, e.g.::

        builder.marks["acute"]     = (0, a1)
        builder.marks["grave"]     = (0, a1)
        builder.marks["cedilla"]   = (1, a2)
        builder.baseMarks["acute"] = {0: a3}

    Attributes:
        font (``fontTools.TTLib.TTFont``): A font object.
        location: A string or tuple representing the location in the original
            source which produced this lookup.
        marks: An dictionary mapping a glyph name to a two-element
            tuple containing a mark class ID and ``otTables.Anchor`` object.
        baseMarks: An dictionary mapping a glyph name to a dictionary
            containing one item: a mark class ID and a ``otTables.Anchor`` object.
        lookupflag (int): The lookup's flag
        markFilterSet: Either ``None`` if no mark filtering set is used, or
            an integer representing the filtering set to be used for this
            lookup. If a mark filtering set is provided,
            `LOOKUP_FLAG_USE_MARK_FILTERING_SET` will be set on the lookup's
            flags.
    """

    def __init__(self, font, location):
        LookupBuilder.__init__(self, font, location, "GPOS", 6)
        self.marks = {}  # glyphName -> (markClassName, anchor)
        self.baseMarks = {}  # glyphName -> {markClassName: anchor}

    def equals(self, other):
        return (
            LookupBuilder.equals(self, other)
            and self.marks == other.marks
            and self.baseMarks == other.baseMarks
        )

    def inferGlyphClasses(self):
        result = {glyph: 3 for glyph in self.baseMarks}
        result.update({glyph: 3 for glyph in self.marks})
        return result

    def build(self):
        """Build the lookup.

        Returns:
            An ``otTables.Lookup`` object representing the mark-to-mark
            positioning lookup.
        """
        markClasses = self.buildMarkClasses_(self.marks)
        markClassList = sorted(markClasses.keys(), key=markClasses.get)
        marks = {
            mark: (markClasses[mc], anchor) for mark, (mc, anchor) in self.marks.items()
        }

        st = ot.MarkMarkPos()
        st.Format = 1
        st.ClassCount = len(markClasses)
        st.Mark1Coverage = buildCoverage(marks, self.glyphMap)
        st.Mark2Coverage = buildCoverage(self.baseMarks, self.glyphMap)
        st.Mark1Array = buildMarkArray(marks, self.glyphMap)
        st.Mark2Array = ot.Mark2Array()
        st.Mark2Array.Mark2Count = len(st.Mark2Coverage.glyphs)
        st.Mark2Array.Mark2Record = []
        for base in st.Mark2Coverage.glyphs:
            anchors = [self.baseMarks[base].get(mc) for mc in markClassList]
            st.Mark2Array.Mark2Record.append(buildMark2Record(anchors))
        return self.buildLookup_([st])


class ReverseChainSingleSubstBuilder(LookupBuilder):
    """Builds a Reverse Chaining Contextual Single Substitution (GSUB8) lookup.

    Users are expected to manually add substitutions to the ``substitutions``
    attribute after the object has been initialized, e.g.::

        # reversesub [a e n] d' by d.alt;
        prefix = [ ["a", "e", "n"] ]
        suffix = []
        mapping = { "d": "d.alt" }
        builder.substitutions.append( (prefix, suffix, mapping) )

    Attributes:
        font (``fontTools.TTLib.TTFont``): A font object.
        location: A string or tuple representing the location in the original
            source which produced this lookup.
        substitutions: A three-element tuple consisting of a prefix sequence,
            a suffix sequence, and a dictionary of single substitutions.
        lookupflag (int): The lookup's flag
        markFilterSet: Either ``None`` if no mark filtering set is used, or
            an integer representing the filtering set to be used for this
            lookup. If a mark filtering set is provided,
            `LOOKUP_FLAG_USE_MARK_FILTERING_SET` will be set on the lookup's
            flags.
    """

    def __init__(self, font, location):
        LookupBuilder.__init__(self, font, location, "GSUB", 8)
        self.rules = []  # (prefix, suffix, mapping)

    def equals(self, other):
        return LookupBuilder.equals(self, other) and self.rules == other.rules

    def build(self):
        """Build the lookup.

        Returns:
            An ``otTables.Lookup`` object representing the chained
            contextual substitution lookup.
        """
        subtables = []
        for prefix, suffix, mapping in self.rules:
            st = ot.ReverseChainSingleSubst()
            st.Format = 1
            self.setBacktrackCoverage_(prefix, st)
            self.setLookAheadCoverage_(suffix, st)
            st.Coverage = buildCoverage(mapping.keys(), self.glyphMap)
            st.GlyphCount = len(mapping)
            st.Substitute = [mapping[g] for g in st.Coverage.glyphs]
            subtables.append(st)
        return self.buildLookup_(subtables)

    def add_subtable_break(self, location):
        # Nothing to do here, each substitution is in its own subtable.
        pass


class SingleSubstBuilder(LookupBuilder):
    """Builds a Single Substitution (GSUB1) lookup.

    Users are expected to manually add substitutions to the ``mapping``
    attribute after the object has been initialized, e.g.::

        # sub x by y;
        builder.mapping["x"] = "y"

    Attributes:
        font (``fontTools.TTLib.TTFont``): A font object.
        location: A string or tuple representing the location in the original
            source which produced this lookup.
        mapping: A dictionary mapping a single glyph name to another glyph name.
        lookupflag (int): The lookup's flag
        markFilterSet: Either ``None`` if no mark filtering set is used, or
            an integer representing the filtering set to be used for this
            lookup. If a mark filtering set is provided,
            `LOOKUP_FLAG_USE_MARK_FILTERING_SET` will be set on the lookup's
            flags.
    """

    def __init__(self, font, location):
        LookupBuilder.__init__(self, font, location, "GSUB", 1)
        self.mapping = OrderedDict()

    def equals(self, other):
        return LookupBuilder.equals(self, other) and self.mapping == other.mapping

    def build(self):
        """Build the lookup.

        Returns:
            An ``otTables.Lookup`` object representing the multiple
            substitution lookup.
        """
        subtables = self.build_subst_subtables(self.mapping, buildSingleSubstSubtable)
        return self.buildLookup_(subtables)

    def getAlternateGlyphs(self):
        return {glyph: set([repl]) for glyph, repl in self.mapping.items()}

    def add_subtable_break(self, location):
        self.mapping[(self.SUBTABLE_BREAK_, location)] = self.SUBTABLE_BREAK_


class ClassPairPosSubtableBuilder(object):
    """Builds class-based Pair Positioning (GPOS2 format 2) subtables.

    Note that this does *not* build a GPOS2 ``otTables.Lookup`` directly,
    but builds a list of ``otTables.PairPos`` subtables. It is used by the
    :class:`PairPosBuilder` below.

    Attributes:
        builder (PairPosBuilder): A pair positioning lookup builder.
    """

    def __init__(self, builder):
        self.builder_ = builder
        self.classDef1_, self.classDef2_ = None, None
        self.values_ = {}  # (glyphclass1, glyphclass2) --> (value1, value2)
        self.forceSubtableBreak_ = False
        self.subtables_ = []

    def addPair(self, gc1, value1, gc2, value2):
        """Add a pair positioning rule.

        Args:
            gc1: A set of glyph names for the "left" glyph
            value1: An ``otTables.ValueRecord`` object for the left glyph's
                positioning.
            gc2: A set of glyph names for the "right" glyph
            value2: An ``otTables.ValueRecord`` object for the right glyph's
                positioning.
        """
        mergeable = (
            not self.forceSubtableBreak_
            and self.classDef1_ is not None
            and self.classDef1_.canAdd(gc1)
            and self.classDef2_ is not None
            and self.classDef2_.canAdd(gc2)
        )
        if not mergeable:
            self.flush_()
            self.classDef1_ = ClassDefBuilder(useClass0=True)
            self.classDef2_ = ClassDefBuilder(useClass0=False)
            self.values_ = {}
        self.classDef1_.add(gc1)
        self.classDef2_.add(gc2)
        self.values_[(gc1, gc2)] = (value1, value2)

    def addSubtableBreak(self):
        """Add an explicit subtable break at this point."""
        self.forceSubtableBreak_ = True

    def subtables(self):
        """Return the list of ``otTables.PairPos`` subtables constructed."""
        self.flush_()
        return self.subtables_

    def flush_(self):
        if self.classDef1_ is None or self.classDef2_ is None:
            return
        st = buildPairPosClassesSubtable(self.values_, self.builder_.glyphMap)
        if st.Coverage is None:
            return
        self.subtables_.append(st)
        self.forceSubtableBreak_ = False


class PairPosBuilder(LookupBuilder):
    """Builds a Pair Positioning (GPOS2) lookup.

    Attributes:
        font (``fontTools.TTLib.TTFont``): A font object.
        location: A string or tuple representing the location in the original
            source which produced this lookup.
        pairs: An array of class-based pair positioning tuples. Usually
            manipulated with the :meth:`addClassPair` method below.
        glyphPairs: A dictionary mapping a tuple of glyph names to a tuple
            of ``otTables.ValueRecord`` objects. Usually manipulated with the
            :meth:`addGlyphPair` method below.
        lookupflag (int): The lookup's flag
        markFilterSet: Either ``None`` if no mark filtering set is used, or
            an integer representing the filtering set to be used for this
            lookup. If a mark filtering set is provided,
            `LOOKUP_FLAG_USE_MARK_FILTERING_SET` will be set on the lookup's
            flags.
    """

    def __init__(self, font, location):
        LookupBuilder.__init__(self, font, location, "GPOS", 2)
        self.pairs = []  # [(gc1, value1, gc2, value2)*]
        self.glyphPairs = {}  # (glyph1, glyph2) --> (value1, value2)
        self.locations = {}  # (gc1, gc2) --> (filepath, line, column)

    def addClassPair(self, location, glyphclass1, value1, glyphclass2, value2):
        """Add a class pair positioning rule to the current lookup.

        Args:
            location: A string or tuple representing the location in the
                original source which produced this rule. Unused.
            glyphclass1: A set of glyph names for the "left" glyph in the pair.
            value1: A ``otTables.ValueRecord`` for positioning the left glyph.
            glyphclass2: A set of glyph names for the "right" glyph in the pair.
            value2: A ``otTables.ValueRecord`` for positioning the right glyph.
        """
        self.pairs.append((glyphclass1, value1, glyphclass2, value2))

    def addGlyphPair(self, location, glyph1, value1, glyph2, value2):
        """Add a glyph pair positioning rule to the current lookup.

        Args:
            location: A string or tuple representing the location in the
                original source which produced this rule.
            glyph1: A glyph name for the "left" glyph in the pair.
            value1: A ``otTables.ValueRecord`` for positioning the left glyph.
            glyph2: A glyph name for the "right" glyph in the pair.
            value2: A ``otTables.ValueRecord`` for positioning the right glyph.
        """
        key = (glyph1, glyph2)
        oldValue = self.glyphPairs.get(key, None)
        if oldValue is not None:
            # the Feature File spec explicitly allows specific pairs generated
            # by an 'enum' rule to be overridden by preceding single pairs
            otherLoc = self.locations[key]
            log.debug(
                "Already defined position for pair %s %s at %s; "
                "choosing the first value",
                glyph1,
                glyph2,
                otherLoc,
            )
        else:
            self.glyphPairs[key] = (value1, value2)
            self.locations[key] = location

    def add_subtable_break(self, location):
        self.pairs.append(
            (
                self.SUBTABLE_BREAK_,
                self.SUBTABLE_BREAK_,
                self.SUBTABLE_BREAK_,
                self.SUBTABLE_BREAK_,
            )
        )

    def equals(self, other):
        return (
            LookupBuilder.equals(self, other)
            and self.glyphPairs == other.glyphPairs
            and self.pairs == other.pairs
        )

    def build(self):
        """Build the lookup.

        Returns:
            An ``otTables.Lookup`` object representing the pair positioning
            lookup.
        """
        builders = {}
        builder = None
        for glyphclass1, value1, glyphclass2, value2 in self.pairs:
            if glyphclass1 is self.SUBTABLE_BREAK_:
                if builder is not None:
                    builder.addSubtableBreak()
                continue
            valFormat1, valFormat2 = 0, 0
            if value1:
                valFormat1 = value1.getFormat()
            if value2:
                valFormat2 = value2.getFormat()
            builder = builders.get((valFormat1, valFormat2))
            if builder is None:
                builder = ClassPairPosSubtableBuilder(self)
                builders[(valFormat1, valFormat2)] = builder
            builder.addPair(glyphclass1, value1, glyphclass2, value2)
        subtables = []
        if self.glyphPairs:
            subtables.extend(buildPairPosGlyphs(self.glyphPairs, self.glyphMap))
        for key in sorted(builders.keys()):
            subtables.extend(builders[key].subtables())
        lookup = self.buildLookup_(subtables)

        # Compact the lookup
        # This is a good moment to do it because the compaction should create
        # smaller subtables, which may prevent overflows from happening.
        mode = os.environ.get(GPOS_COMPACT_MODE_ENV_KEY, GPOS_COMPACT_MODE_DEFAULT)
        if mode and mode != "0":
            log.info("Compacting GPOS...")
            compact_lookup(self.font, mode, lookup)

        return lookup


class SinglePosBuilder(LookupBuilder):
    """Builds a Single Positioning (GPOS1) lookup.

    Attributes:
        font (``fontTools.TTLib.TTFont``): A font object.
        location: A string or tuple representing the location in the original
            source which produced this lookup.
        mapping: A dictionary mapping a glyph name to a ``otTables.ValueRecord``
            objects. Usually manipulated with the :meth:`add_pos` method below.
        lookupflag (int): The lookup's flag
        markFilterSet: Either ``None`` if no mark filtering set is used, or
            an integer representing the filtering set to be used for this
            lookup. If a mark filtering set is provided,
            `LOOKUP_FLAG_USE_MARK_FILTERING_SET` will be set on the lookup's
            flags.
    """

    def __init__(self, font, location):
        LookupBuilder.__init__(self, font, location, "GPOS", 1)
        self.locations = {}  # glyph -> (filename, line, column)
        self.mapping = {}  # glyph -> ot.ValueRecord

    def add_pos(self, location, glyph, otValueRecord):
        """Add a single positioning rule.

        Args:
            location: A string or tuple representing the location in the
                original source which produced this lookup.
            glyph: A glyph name.
            otValueRection: A ``otTables.ValueRecord`` used to position the
                glyph.
        """
        if not self.can_add(glyph, otValueRecord):
            otherLoc = self.locations[glyph]
            raise OpenTypeLibError(
                'Already defined different position for glyph "%s" at %s'
                % (glyph, otherLoc),
                location,
            )
        if otValueRecord:
            self.mapping[glyph] = otValueRecord
        self.locations[glyph] = location

    def can_add(self, glyph, value):
        assert isinstance(value, ValueRecord)
        curValue = self.mapping.get(glyph)
        return curValue is None or curValue == value

    def equals(self, other):
        return LookupBuilder.equals(self, other) and self.mapping == other.mapping

    def build(self):
        """Build the lookup.

        Returns:
            An ``otTables.Lookup`` object representing the single positioning
            lookup.
        """
        subtables = buildSinglePos(self.mapping, self.glyphMap)
        return self.buildLookup_(subtables)


# GSUB


def buildSingleSubstSubtable(mapping):
    """Builds a single substitution (GSUB1) subtable.

    Note that if you are implementing a layout compiler, you may find it more
    flexible to use
    :py:class:`fontTools.otlLib.lookupBuilders.SingleSubstBuilder` instead.

    Args:
        mapping: A dictionary mapping input glyph names to output glyph names.

    Returns:
        An ``otTables.SingleSubst`` object, or ``None`` if the mapping dictionary
        is empty.
    """
    if not mapping:
        return None
    self = ot.SingleSubst()
    self.mapping = dict(mapping)
    return self


def buildMultipleSubstSubtable(mapping):
    """Builds a multiple substitution (GSUB2) subtable.

    Note that if you are implementing a layout compiler, you may find it more
    flexible to use
    :py:class:`fontTools.otlLib.lookupBuilders.MultipleSubstBuilder` instead.

    Example::

        # sub uni06C0 by uni06D5.fina hamza.above
        # sub uni06C2 by uni06C1.fina hamza.above;

        subtable = buildMultipleSubstSubtable({
            "uni06C0": [ "uni06D5.fina", "hamza.above"],
            "uni06C2": [ "uni06D1.fina", "hamza.above"]
        })

    Args:
        mapping: A dictionary mapping input glyph names to a list of output
            glyph names.

    Returns:
        An ``otTables.MultipleSubst`` object or ``None`` if the mapping dictionary
        is empty.
    """
    if not mapping:
        return None
    self = ot.MultipleSubst()
    self.mapping = dict(mapping)
    return self


def buildAlternateSubstSubtable(mapping):
    """Builds an alternate substitution (GSUB3) subtable.

    Note that if you are implementing a layout compiler, you may find it more
    flexible to use
    :py:class:`fontTools.otlLib.lookupBuilders.AlternateSubstBuilder` instead.

    Args:
        mapping: A dictionary mapping input glyph names to a list of output
            glyph names.

    Returns:
        An ``otTables.AlternateSubst`` object or ``None`` if the mapping dictionary
        is empty.
    """
    if not mapping:
        return None
    self = ot.AlternateSubst()
    self.alternates = dict(mapping)
    return self


def _getLigatureKey(components):
    # Computes a key for ordering ligatures in a GSUB Type-4 lookup.

    # When building the OpenType lookup, we need to make sure that
    # the longest sequence of components is listed first, so we
    # use the negative length as the primary key for sorting.
    # To make buildLigatureSubstSubtable() deterministic, we use the
    # component sequence as the secondary key.

    # For example, this will sort (f,f,f) < (f,f,i) < (f,f) < (f,i) < (f,l).
    return (-len(components), components)


def buildLigatureSubstSubtable(mapping):
    """Builds a ligature substitution (GSUB4) subtable.

    Note that if you are implementing a layout compiler, you may find it more
    flexible to use
    :py:class:`fontTools.otlLib.lookupBuilders.LigatureSubstBuilder` instead.

    Example::

        # sub f f i by f_f_i;
        # sub f i by f_i;

        subtable = buildLigatureSubstSubtable({
            ("f", "f", "i"): "f_f_i",
            ("f", "i"): "f_i",
        })

    Args:
        mapping: A dictionary mapping tuples of glyph names to output
            glyph names.

    Returns:
        An ``otTables.LigatureSubst`` object or ``None`` if the mapping dictionary
        is empty.
    """

    if not mapping:
        return None
    self = ot.LigatureSubst()
    # The following single line can replace the rest of this function
    # with fontTools >= 3.1:
    # self.ligatures = dict(mapping)
    self.ligatures = {}
    for components in sorted(mapping.keys(), key=_getLigatureKey):
        ligature = ot.Ligature()
        ligature.Component = components[1:]
        ligature.CompCount = len(ligature.Component) + 1
        ligature.LigGlyph = mapping[components]
        firstGlyph = components[0]
        self.ligatures.setdefault(firstGlyph, []).append(ligature)
    return self


# GPOS


def buildAnchor(x, y, point=None, deviceX=None, deviceY=None):
    """Builds an Anchor table.

    This determines the appropriate anchor format based on the passed parameters.

    Args:
        x (int): X coordinate.
        y (int): Y coordinate.
        point (int): Index of glyph contour point, if provided.
        deviceX (``otTables.Device``): X coordinate device table, if provided.
        deviceY (``otTables.Device``): Y coordinate device table, if provided.

    Returns:
        An ``otTables.Anchor`` object.
    """
    self = ot.Anchor()
    self.XCoordinate, self.YCoordinate = x, y
    self.Format = 1
    if point is not None:
        self.AnchorPoint = point
        self.Format = 2
    if deviceX is not None or deviceY is not None:
        assert (
            self.Format == 1
        ), "Either point, or both of deviceX/deviceY, must be None."
        self.XDeviceTable = deviceX
        self.YDeviceTable = deviceY
        self.Format = 3
    return self


def buildBaseArray(bases, numMarkClasses, glyphMap):
    """Builds a base array record.

    As part of building mark-to-base positioning rules, you will need to define
    a ``BaseArray`` record, which "defines for each base glyph an array of
    anchors, one for each mark class." This function builds the base array
    subtable.

    Example::

        bases = {"a": {0: a3, 1: a5}, "b": {0: a4, 1: a5}}
        basearray = buildBaseArray(bases, 2, font.getReverseGlyphMap())

    Args:
        bases (dict): A dictionary mapping anchors to glyphs; the keys being
            glyph names, and the values being dictionaries mapping mark class ID
            to the appropriate ``otTables.Anchor`` object used for attaching marks
            of that class.
        numMarkClasses (int): The total number of mark classes for which anchors
            are defined.
        glyphMap: a glyph name to ID map, typically returned from
            ``font.getReverseGlyphMap()``.

    Returns:
        An ``otTables.BaseArray`` object.
    """
    self = ot.BaseArray()
    self.BaseRecord = []
    for base in sorted(bases, key=glyphMap.__getitem__):
        b = bases[base]
        anchors = [b.get(markClass) for markClass in range(numMarkClasses)]
        self.BaseRecord.append(buildBaseRecord(anchors))
    self.BaseCount = len(self.BaseRecord)
    return self


def buildBaseRecord(anchors):
    # [otTables.Anchor, otTables.Anchor, ...] --> otTables.BaseRecord
    self = ot.BaseRecord()
    self.BaseAnchor = anchors
    return self


def buildComponentRecord(anchors):
    """Builds a component record.

    As part of building mark-to-ligature positioning rules, you will need to
    define ``ComponentRecord`` objects, which contain "an array of offsets...
    to the Anchor tables that define all the attachment points used to attach
    marks to the component." This function builds the component record.

    Args:
        anchors: A list of ``otTables.Anchor`` objects or ``None``.

    Returns:
        A ``otTables.ComponentRecord`` object or ``None`` if no anchors are
        supplied.
    """
    if not anchors:
        return None
    self = ot.ComponentRecord()
    self.LigatureAnchor = anchors
    return self


def buildCursivePosSubtable(attach, glyphMap):
    """Builds a cursive positioning (GPOS3) subtable.

    Cursive positioning lookups are made up of a coverage table of glyphs,
    and a set of ``EntryExitRecord`` records containing the anchors for
    each glyph. This function builds the cursive positioning subtable.

    Example::

        subtable = buildCursivePosSubtable({
            "AlifIni": (None, buildAnchor(0, 50)),
            "BehMed": (buildAnchor(500,250), buildAnchor(0,50)),
            # ...
        }, font.getReverseGlyphMap())

    Args:
        attach (dict): A mapping between glyph names and a tuple of two
            ``otTables.Anchor`` objects representing entry and exit anchors.
        glyphMap: a glyph name to ID map, typically returned from
            ``font.getReverseGlyphMap()``.

    Returns:
        An ``otTables.CursivePos`` object, or ``None`` if the attachment
        dictionary was empty.
    """
    if not attach:
        return None
    self = ot.CursivePos()
    self.Format = 1
    self.Coverage = buildCoverage(attach.keys(), glyphMap)
    self.EntryExitRecord = []
    for glyph in self.Coverage.glyphs:
        entryAnchor, exitAnchor = attach[glyph]
        rec = ot.EntryExitRecord()
        rec.EntryAnchor = entryAnchor
        rec.ExitAnchor = exitAnchor
        self.EntryExitRecord.append(rec)
    self.EntryExitCount = len(self.EntryExitRecord)
    return self


def buildDevice(deltas):
    """Builds a Device record as part of a ValueRecord or Anchor.

    Device tables specify size-specific adjustments to value records
    and anchors to reflect changes based on the resolution of the output.
    For example, one could specify that an anchor's Y position should be
    increased by 1 pixel when displayed at 8 pixels per em. This routine
    builds device records.

    Args:
        deltas: A dictionary mapping pixels-per-em sizes to the delta
            adjustment in pixels when the font is displayed at that size.

    Returns:
        An ``otTables.Device`` object if any deltas were supplied, or
        ``None`` otherwise.
    """
    if not deltas:
        return None
    self = ot.Device()
    keys = deltas.keys()
    self.StartSize = startSize = min(keys)
    self.EndSize = endSize = max(keys)
    assert 0 <= startSize <= endSize
    self.DeltaValue = deltaValues = [
        deltas.get(size, 0) for size in range(startSize, endSize + 1)
    ]
    maxDelta = max(deltaValues)
    minDelta = min(deltaValues)
    assert minDelta > -129 and maxDelta < 128
    if minDelta > -3 and maxDelta < 2:
        self.DeltaFormat = 1
    elif minDelta > -9 and maxDelta < 8:
        self.DeltaFormat = 2
    else:
        self.DeltaFormat = 3
    return self


def buildLigatureArray(ligs, numMarkClasses, glyphMap):
    """Builds a LigatureArray subtable.

    As part of building a mark-to-ligature lookup, you will need to define
    the set of anchors (for each mark class) on each component of the ligature
    where marks can be attached. For example, for an Arabic divine name ligature
    (lam lam heh), you may want to specify mark attachment positioning for
    superior marks (fatha, etc.) and inferior marks (kasra, etc.) on each glyph
    of the ligature. This routine builds the ligature array record.

    Example::

        buildLigatureArray({
            "lam-lam-heh": [
                { 0: superiorAnchor1, 1: inferiorAnchor1 }, # attach points for lam1
                { 0: superiorAnchor2, 1: inferiorAnchor2 }, # attach points for lam2
                { 0: superiorAnchor3, 1: inferiorAnchor3 }, # attach points for heh
            ]
        }, 2, font.getReverseGlyphMap())

    Args:
        ligs (dict): A mapping of ligature names to an array of dictionaries:
            for each component glyph in the ligature, an dictionary mapping
            mark class IDs to anchors.
        numMarkClasses (int): The number of mark classes.
        glyphMap: a glyph name to ID map, typically returned from
            ``font.getReverseGlyphMap()``.

    Returns:
        An ``otTables.LigatureArray`` object if deltas were supplied.
    """
    self = ot.LigatureArray()
    self.LigatureAttach = []
    for lig in sorted(ligs, key=glyphMap.__getitem__):
        anchors = []
        for component in ligs[lig]:
            anchors.append([component.get(mc) for mc in range(numMarkClasses)])
        self.LigatureAttach.append(buildLigatureAttach(anchors))
    self.LigatureCount = len(self.LigatureAttach)
    return self


def buildLigatureAttach(components):
    # [[Anchor, Anchor], [Anchor, Anchor, Anchor]] --> LigatureAttach
    self = ot.LigatureAttach()
    self.ComponentRecord = [buildComponentRecord(c) for c in components]
    self.ComponentCount = len(self.ComponentRecord)
    return self


def buildMarkArray(marks, glyphMap):
    """Builds a mark array subtable.

    As part of building mark-to-* positioning rules, you will need to define
    a MarkArray subtable, which "defines the class and the anchor point
    for a mark glyph." This function builds the mark array subtable.

    Example::

        mark = {
            "acute": (0, buildAnchor(300,712)),
            # ...
        }
        markarray = buildMarkArray(marks, font.getReverseGlyphMap())

    Args:
        marks (dict): A dictionary mapping anchors to glyphs; the keys being
            glyph names, and the values being a tuple of mark class number and
            an ``otTables.Anchor`` object representing the mark's attachment
            point.
        glyphMap: a glyph name to ID map, typically returned from
            ``font.getReverseGlyphMap()``.

    Returns:
        An ``otTables.MarkArray`` object.
    """
    self = ot.MarkArray()
    self.MarkRecord = []
    for mark in sorted(marks.keys(), key=glyphMap.__getitem__):
        markClass, anchor = marks[mark]
        markrec = buildMarkRecord(markClass, anchor)
        self.MarkRecord.append(markrec)
    self.MarkCount = len(self.MarkRecord)
    return self


def buildMarkBasePos(marks, bases, glyphMap):
    """Build a list of MarkBasePos (GPOS4) subtables.

    This routine turns a set of marks and bases into a list of mark-to-base
    positioning subtables. Currently the list will contain a single subtable
    containing all marks and bases, although at a later date it may return the
    optimal list of subtables subsetting the marks and bases into groups which
    save space. See :func:`buildMarkBasePosSubtable` below.

    Note that if you are implementing a layout compiler, you may find it more
    flexible to use
    :py:class:`fontTools.otlLib.lookupBuilders.MarkBasePosBuilder` instead.

    Example::

        # a1, a2, a3, a4, a5 = buildAnchor(500, 100), ...

        marks = {"acute": (0, a1), "grave": (0, a1), "cedilla": (1, a2)}
        bases = {"a": {0: a3, 1: a5}, "b": {0: a4, 1: a5}}
        markbaseposes = buildMarkBasePos(marks, bases, font.getReverseGlyphMap())

    Args:
        marks (dict): A dictionary mapping anchors to glyphs; the keys being
            glyph names, and the values being a tuple of mark class number and
            an ``otTables.Anchor`` object representing the mark's attachment
            point. (See :func:`buildMarkArray`.)
        bases (dict): A dictionary mapping anchors to glyphs; the keys being
            glyph names, and the values being dictionaries mapping mark class ID
            to the appropriate ``otTables.Anchor`` object used for attaching marks
            of that class. (See :func:`buildBaseArray`.)
        glyphMap: a glyph name to ID map, typically returned from
            ``font.getReverseGlyphMap()``.

    Returns:
        A list of ``otTables.MarkBasePos`` objects.
    """
    # TODO: Consider emitting multiple subtables to save space.
    # Partition the marks and bases into disjoint subsets, so that
    # MarkBasePos rules would only access glyphs from a single
    # subset. This would likely lead to smaller mark/base
    # matrices, so we might be able to omit many of the empty
    # anchor tables that we currently produce. Of course, this
    # would only work if the MarkBasePos rules of real-world fonts
    # allow partitioning into multiple subsets. We should find out
    # whether this is the case; if so, implement the optimization.
    # On the other hand, a very large number of subtables could
    # slow down layout engines; so this would need profiling.
    return [buildMarkBasePosSubtable(marks, bases, glyphMap)]


def buildMarkBasePosSubtable(marks, bases, glyphMap):
    """Build a single MarkBasePos (GPOS4) subtable.

    This builds a mark-to-base lookup subtable containing all of the referenced
    marks and bases. See :func:`buildMarkBasePos`.

    Args:
        marks (dict): A dictionary mapping anchors to glyphs; the keys being
            glyph names, and the values being a tuple of mark class number and
            an ``otTables.Anchor`` object representing the mark's attachment
            point. (See :func:`buildMarkArray`.)
        bases (dict): A dictionary mapping anchors to glyphs; the keys being
            glyph names, and the values being dictionaries mapping mark class ID
            to the appropriate ``otTables.Anchor`` object used for attaching marks
            of that class. (See :func:`buildBaseArray`.)
        glyphMap: a glyph name to ID map, typically returned from
            ``font.getReverseGlyphMap()``.

    Returns:
        A ``otTables.MarkBasePos`` object.
    """
    self = ot.MarkBasePos()
    self.Format = 1
    self.MarkCoverage = buildCoverage(marks, glyphMap)
    self.MarkArray = buildMarkArray(marks, glyphMap)
    self.ClassCount = max([mc for mc, _ in marks.values()]) + 1
    self.BaseCoverage = buildCoverage(bases, glyphMap)
    self.BaseArray = buildBaseArray(bases, self.ClassCount, glyphMap)
    return self


def buildMarkLigPos(marks, ligs, glyphMap):
    """Build a list of MarkLigPos (GPOS5) subtables.

    This routine turns a set of marks and ligatures into a list of mark-to-ligature
    positioning subtables. Currently the list will contain a single subtable
    containing all marks and ligatures, although at a later date it may return
    the optimal list of subtables subsetting the marks and ligatures into groups
    which save space. See :func:`buildMarkLigPosSubtable` below.

    Note that if you are implementing a layout compiler, you may find it more
    flexible to use
    :py:class:`fontTools.otlLib.lookupBuilders.MarkLigPosBuilder` instead.

    Example::

        # a1, a2, a3, a4, a5 = buildAnchor(500, 100), ...
        marks = {
            "acute": (0, a1),
            "grave": (0, a1),
            "cedilla": (1, a2)
        }
        ligs = {
            "f_i": [
                { 0: a3, 1: a5 }, # f
                { 0: a4, 1: a5 }  # i
                ],
        #   "c_t": [{...}, {...}]
        }
        markligposes = buildMarkLigPos(marks, ligs,
            font.getReverseGlyphMap())

    Args:
        marks (dict): A dictionary mapping anchors to glyphs; the keys being
            glyph names, and the values being a tuple of mark class number and
            an ``otTables.Anchor`` object representing the mark's attachment
            point. (See :func:`buildMarkArray`.)
        ligs (dict): A mapping of ligature names to an array of dictionaries:
            for each component glyph in the ligature, an dictionary mapping
            mark class IDs to anchors. (See :func:`buildLigatureArray`.)
        glyphMap: a glyph name to ID map, typically returned from
            ``font.getReverseGlyphMap()``.

    Returns:
        A list of ``otTables.MarkLigPos`` objects.

    """
    # TODO: Consider splitting into multiple subtables to save space,
    # as with MarkBasePos, this would be a trade-off that would need
    # profiling. And, depending on how typical fonts are structured,
    # it might not be worth doing at all.
    return [buildMarkLigPosSubtable(marks, ligs, glyphMap)]


def buildMarkLigPosSubtable(marks, ligs, glyphMap):
    """Build a single MarkLigPos (GPOS5) subtable.

    This builds a mark-to-base lookup subtable containing all of the referenced
    marks and bases. See :func:`buildMarkLigPos`.

    Args:
        marks (dict): A dictionary mapping anchors to glyphs; the keys being
            glyph names, and the values being a tuple of mark class number and
            an ``otTables.Anchor`` object representing the mark's attachment
            point. (See :func:`buildMarkArray`.)
        ligs (dict): A mapping of ligature names to an array of dictionaries:
            for each component glyph in the ligature, an dictionary mapping
            mark class IDs to anchors. (See :func:`buildLigatureArray`.)
        glyphMap: a glyph name to ID map, typically returned from
            ``font.getReverseGlyphMap()``.

    Returns:
        A ``otTables.MarkLigPos`` object.
    """
    self = ot.MarkLigPos()
    self.Format = 1
    self.MarkCoverage = buildCoverage(marks, glyphMap)
    self.MarkArray = buildMarkArray(marks, glyphMap)
    self.ClassCount = max([mc for mc, _ in marks.values()]) + 1
    self.LigatureCoverage = buildCoverage(ligs, glyphMap)
    self.LigatureArray = buildLigatureArray(ligs, self.ClassCount, glyphMap)
    return self


def buildMarkRecord(classID, anchor):
    assert isinstance(classID, int)
    assert isinstance(anchor, ot.Anchor)
    self = ot.MarkRecord()
    self.Class = classID
    self.MarkAnchor = anchor
    return self


def buildMark2Record(anchors):
    # [otTables.Anchor, otTables.Anchor, ...] --> otTables.Mark2Record
    self = ot.Mark2Record()
    self.Mark2Anchor = anchors
    return self


def _getValueFormat(f, values, i):
    # Helper for buildPairPos{Glyphs|Classes}Subtable.
    if f is not None:
        return f
    mask = 0
    for value in values:
        if value is not None and value[i] is not None:
            mask |= value[i].getFormat()
    return mask


def buildPairPosClassesSubtable(pairs, glyphMap, valueFormat1=None, valueFormat2=None):
    """Builds a class pair adjustment (GPOS2 format 2) subtable.

    Kerning tables are generally expressed as pair positioning tables using
    class-based pair adjustments. This routine builds format 2 PairPos
    subtables.

    Note that if you are implementing a layout compiler, you may find it more
    flexible to use
    :py:class:`fontTools.otlLib.lookupBuilders.ClassPairPosSubtableBuilder`
    instead, as this takes care of ensuring that the supplied pairs can be
    formed into non-overlapping classes and emitting individual subtables
    whenever the non-overlapping requirement means that a new subtable is
    required.

    Example::

        pairs = {}

        pairs[(
            [ "K", "X" ],
            [ "W", "V" ]
        )] = ( buildValue(xAdvance=+5), buildValue() )
        # pairs[(... , ...)] = (..., ...)

        pairpos = buildPairPosClassesSubtable(pairs, font.getReverseGlyphMap())

    Args:
        pairs (dict): Pair positioning data; the keys being a two-element
            tuple of lists of glyphnames, and the values being a two-element
            tuple of ``otTables.ValueRecord`` objects.
        glyphMap: a glyph name to ID map, typically returned from
            ``font.getReverseGlyphMap()``.
        valueFormat1: Force the "left" value records to the given format.
        valueFormat2: Force the "right" value records to the given format.

    Returns:
        A ``otTables.PairPos`` object.
    """
    coverage = set()
    classDef1 = ClassDefBuilder(useClass0=True)
    classDef2 = ClassDefBuilder(useClass0=False)
    for gc1, gc2 in sorted(pairs):
        coverage.update(gc1)
        classDef1.add(gc1)
        classDef2.add(gc2)
    self = ot.PairPos()
    self.Format = 2
    valueFormat1 = self.ValueFormat1 = _getValueFormat(valueFormat1, pairs.values(), 0)
    valueFormat2 = self.ValueFormat2 = _getValueFormat(valueFormat2, pairs.values(), 1)
    self.Coverage = buildCoverage(coverage, glyphMap)
    self.ClassDef1 = classDef1.build()
    self.ClassDef2 = classDef2.build()
    classes1 = classDef1.classes()
    classes2 = classDef2.classes()
    self.Class1Record = []
    for c1 in classes1:
        rec1 = ot.Class1Record()
        rec1.Class2Record = []
        self.Class1Record.append(rec1)
        for c2 in classes2:
            rec2 = ot.Class2Record()
            val1, val2 = pairs.get((c1, c2), (None, None))
            rec2.Value1 = ValueRecord(src=val1, valueFormat=valueFormat1) if valueFormat1 else None
            rec2.Value2 = ValueRecord(src=val2, valueFormat=valueFormat2) if valueFormat2 else None
            rec1.Class2Record.append(rec2)
    self.Class1Count = len(self.Class1Record)
    self.Class2Count = len(classes2)
    return self


def buildPairPosGlyphs(pairs, glyphMap):
    """Builds a list of glyph-based pair adjustment (GPOS2 format 1) subtables.

    This organises a list of pair positioning adjustments into subtables based
    on common value record formats.

    Note that if you are implementing a layout compiler, you may find it more
    flexible to use
    :py:class:`fontTools.otlLib.lookupBuilders.PairPosBuilder`
    instead.

    Example::

        pairs = {
            ("K", "W"): ( buildValue(xAdvance=+5), buildValue() ),
            ("K", "V"): ( buildValue(xAdvance=+5), buildValue() ),
            # ...
        }

        subtables = buildPairPosGlyphs(pairs, font.getReverseGlyphMap())

    Args:
        pairs (dict): Pair positioning data; the keys being a two-element
            tuple of glyphnames, and the values being a two-element
            tuple of ``otTables.ValueRecord`` objects.
        glyphMap: a glyph name to ID map, typically returned from
            ``font.getReverseGlyphMap()``.

    Returns:
        A list of ``otTables.PairPos`` objects.
    """

    p = {}  # (formatA, formatB) --> {(glyphA, glyphB): (valA, valB)}
    for (glyphA, glyphB), (valA, valB) in pairs.items():
        formatA = valA.getFormat() if valA is not None else 0
        formatB = valB.getFormat() if valB is not None else 0
        pos = p.setdefault((formatA, formatB), {})
        pos[(glyphA, glyphB)] = (valA, valB)
    return [
        buildPairPosGlyphsSubtable(pos, glyphMap, formatA, formatB)
        for ((formatA, formatB), pos) in sorted(p.items())
    ]


def buildPairPosGlyphsSubtable(pairs, glyphMap, valueFormat1=None, valueFormat2=None):
    """Builds a single glyph-based pair adjustment (GPOS2 format 1) subtable.

    This builds a PairPos subtable from a dictionary of glyph pairs and
    their positioning adjustments. See also :func:`buildPairPosGlyphs`.

    Note that if you are implementing a layout compiler, you may find it more
    flexible to use
    :py:class:`fontTools.otlLib.lookupBuilders.PairPosBuilder` instead.

    Example::

        pairs = {
            ("K", "W"): ( buildValue(xAdvance=+5), buildValue() ),
            ("K", "V"): ( buildValue(xAdvance=+5), buildValue() ),
            # ...
        }

        pairpos = buildPairPosGlyphsSubtable(pairs, font.getReverseGlyphMap())

    Args:
        pairs (dict): Pair positioning data; the keys being a two-element
            tuple of glyphnames, and the values being a two-element
            tuple of ``otTables.ValueRecord`` objects.
        glyphMap: a glyph name to ID map, typically returned from
            ``font.getReverseGlyphMap()``.
        valueFormat1: Force the "left" value records to the given format.
        valueFormat2: Force the "right" value records to the given format.

    Returns:
        A ``otTables.PairPos`` object.
    """
    self = ot.PairPos()
    self.Format = 1
    valueFormat1 = self.ValueFormat1 = _getValueFormat(valueFormat1, pairs.values(), 0)
    valueFormat2 = self.ValueFormat2 = _getValueFormat(valueFormat2, pairs.values(), 1)
    p = {}
    for (glyphA, glyphB), (valA, valB) in pairs.items():
        p.setdefault(glyphA, []).append((glyphB, valA, valB))
    self.Coverage = buildCoverage({g for g, _ in pairs.keys()}, glyphMap)
    self.PairSet = []
    for glyph in self.Coverage.glyphs:
        ps = ot.PairSet()
        ps.PairValueRecord = []
        self.PairSet.append(ps)
        for glyph2, val1, val2 in sorted(p[glyph], key=lambda x: glyphMap[x[0]]):
            pvr = ot.PairValueRecord()
            pvr.SecondGlyph = glyph2
            pvr.Value1 = ValueRecord(src=val1, valueFormat=valueFormat1) if valueFormat1 else None
            pvr.Value2 = ValueRecord(src=val2, valueFormat=valueFormat2) if valueFormat2 else None
            ps.PairValueRecord.append(pvr)
        ps.PairValueCount = len(ps.PairValueRecord)
    self.PairSetCount = len(self.PairSet)
    return self


def buildSinglePos(mapping, glyphMap):
    """Builds a list of single adjustment (GPOS1) subtables.

    This builds a list of SinglePos subtables from a dictionary of glyph
    names and their positioning adjustments. The format of the subtables are
    determined to optimize the size of the resulting subtables.
    See also :func:`buildSinglePosSubtable`.

    Note that if you are implementing a layout compiler, you may find it more
    flexible to use
    :py:class:`fontTools.otlLib.lookupBuilders.SinglePosBuilder` instead.

    Example::

        mapping = {
            "V": buildValue({ "xAdvance" : +5 }),
            # ...
        }

        subtables = buildSinglePos(pairs, font.getReverseGlyphMap())

    Args:
        mapping (dict): A mapping between glyphnames and
            ``otTables.ValueRecord`` objects.
        glyphMap: a glyph name to ID map, typically returned from
            ``font.getReverseGlyphMap()``.

    Returns:
        A list of ``otTables.SinglePos`` objects.
    """
    result, handled = [], set()
    # In SinglePos format 1, the covered glyphs all share the same ValueRecord.
    # In format 2, each glyph has its own ValueRecord, but these records
    # all have the same properties (eg., all have an X but no Y placement).
    coverages, masks, values = {}, {}, {}
    for glyph, value in mapping.items():
        key = _getSinglePosValueKey(value)
        coverages.setdefault(key, []).append(glyph)
        masks.setdefault(key[0], []).append(key)
        values[key] = value

    # If a ValueRecord is shared between multiple glyphs, we generate
    # a SinglePos format 1 subtable; that is the most compact form.
    for key, glyphs in coverages.items():
        # 5 ushorts is the length of introducing another sublookup
        if len(glyphs) * _getSinglePosValueSize(key) > 5:
            format1Mapping = {g: values[key] for g in glyphs}
            result.append(buildSinglePosSubtable(format1Mapping, glyphMap))
            handled.add(key)

    # In the remaining ValueRecords, look for those whose valueFormat
    # (the set of used properties) is shared between multiple records.
    # These will get encoded in format 2.
    for valueFormat, keys in masks.items():
        f2 = [k for k in keys if k not in handled]
        if len(f2) > 1:
            format2Mapping = {}
            for k in f2:
                format2Mapping.update((g, values[k]) for g in coverages[k])
            result.append(buildSinglePosSubtable(format2Mapping, glyphMap))
            handled.update(f2)

    # The remaining ValueRecords are only used by a few glyphs, normally
    # one. We encode these in format 1 again.
    for key, glyphs in coverages.items():
        if key not in handled:
            for g in glyphs:
                st = buildSinglePosSubtable({g: values[key]}, glyphMap)
            result.append(st)

    # When the OpenType layout engine traverses the subtables, it will
    # stop after the first matching subtable.  Therefore, we sort the
    # resulting subtables by decreasing coverage size; this increases
    # the chance that the layout engine can do an early exit. (Of course,
    # this would only be true if all glyphs were equally frequent, which
    # is not really the case; but we do not know their distribution).
    # If two subtables cover the same number of glyphs, we sort them
    # by glyph ID so that our output is deterministic.
    result.sort(key=lambda t: _getSinglePosTableKey(t, glyphMap))
    return result


def buildSinglePosSubtable(values, glyphMap):
    """Builds a single adjustment (GPOS1) subtable.

    This builds a list of SinglePos subtables from a dictionary of glyph
    names and their positioning adjustments. The format of the subtable is
    determined to optimize the size of the output.
    See also :func:`buildSinglePos`.

    Note that if you are implementing a layout compiler, you may find it more
    flexible to use
    :py:class:`fontTools.otlLib.lookupBuilders.SinglePosBuilder` instead.

    Example::

        mapping = {
            "V": buildValue({ "xAdvance" : +5 }),
            # ...
        }

        subtable = buildSinglePos(pairs, font.getReverseGlyphMap())

    Args:
        mapping (dict): A mapping between glyphnames and
            ``otTables.ValueRecord`` objects.
        glyphMap: a glyph name to ID map, typically returned from
            ``font.getReverseGlyphMap()``.

    Returns:
        A ``otTables.SinglePos`` object.
    """
    self = ot.SinglePos()
    self.Coverage = buildCoverage(values.keys(), glyphMap)
    valueFormat = self.ValueFormat = reduce(int.__or__, [v.getFormat() for v in values.values()], 0)
    valueRecords = [ValueRecord(src=values[g], valueFormat=valueFormat) for g in self.Coverage.glyphs]
    if all(v == valueRecords[0] for v in valueRecords):
        self.Format = 1
        if self.ValueFormat != 0:
            self.Value = valueRecords[0]
        else:
            self.Value = None
    else:
        self.Format = 2
        self.Value = valueRecords
        self.ValueCount = len(self.Value)
    return self


def _getSinglePosTableKey(subtable, glyphMap):
    assert isinstance(subtable, ot.SinglePos), subtable
    glyphs = subtable.Coverage.glyphs
    return (-len(glyphs), glyphMap[glyphs[0]])


def _getSinglePosValueKey(valueRecord):
    # otBase.ValueRecord --> (2, ("YPlacement": 12))
    assert isinstance(valueRecord, ValueRecord), valueRecord
    valueFormat, result = 0, []
    for name, value in valueRecord.__dict__.items():
        if isinstance(value, ot.Device):
            result.append((name, _makeDeviceTuple(value)))
        else:
            result.append((name, value))
        valueFormat |= valueRecordFormatDict[name][0]
    result.sort()
    result.insert(0, valueFormat)
    return tuple(result)


_DeviceTuple = namedtuple("_DeviceTuple", "DeltaFormat StartSize EndSize DeltaValue")


def _makeDeviceTuple(device):
    # otTables.Device --> tuple, for making device tables unique
    return _DeviceTuple(
        device.DeltaFormat,
        device.StartSize,
        device.EndSize,
        () if device.DeltaFormat & 0x8000 else tuple(device.DeltaValue),
    )


def _getSinglePosValueSize(valueKey):
    # Returns how many ushorts this valueKey (short form of ValueRecord) takes up
    count = 0
    for _, v in valueKey[1:]:
        if isinstance(v, _DeviceTuple):
            count += len(v.DeltaValue) + 3
        else:
            count += 1
    return count


def buildValue(value):
    """Builds a positioning value record.

    Value records are used to specify coordinates and adjustments for
    positioning and attaching glyphs. Many of the positioning functions
    in this library take ``otTables.ValueRecord`` objects as arguments.
    This function builds value records from dictionaries.

    Args:
        value (dict): A dictionary with zero or more of the following keys:
            - ``xPlacement``
            - ``yPlacement``
            - ``xAdvance``
            - ``yAdvance``
            - ``xPlaDevice``
            - ``yPlaDevice``
            - ``xAdvDevice``
            - ``yAdvDevice``

    Returns:
        An ``otTables.ValueRecord`` object.
    """
    self = ValueRecord()
    for k, v in value.items():
        setattr(self, k, v)
    return self


# GDEF


def buildAttachList(attachPoints, glyphMap):
    """Builds an AttachList subtable.

    A GDEF table may contain an Attachment Point List table (AttachList)
    which stores the contour indices of attachment points for glyphs with
    attachment points. This routine builds AttachList subtables.

    Args:
        attachPoints (dict): A mapping between glyph names and a list of
            contour indices.

    Returns:
        An ``otTables.AttachList`` object if attachment points are supplied,
            or ``None`` otherwise.
    """
    if not attachPoints:
        return None
    self = ot.AttachList()
    self.Coverage = buildCoverage(attachPoints.keys(), glyphMap)
    self.AttachPoint = [buildAttachPoint(attachPoints[g]) for g in self.Coverage.glyphs]
    self.GlyphCount = len(self.AttachPoint)
    return self


def buildAttachPoint(points):
    # [4, 23, 41] --> otTables.AttachPoint
    # Only used by above.
    if not points:
        return None
    self = ot.AttachPoint()
    self.PointIndex = sorted(set(points))
    self.PointCount = len(self.PointIndex)
    return self


def buildCaretValueForCoord(coord):
    # 500 --> otTables.CaretValue, format 1
    self = ot.CaretValue()
    self.Format = 1
    self.Coordinate = coord
    return self


def buildCaretValueForPoint(point):
    # 4 --> otTables.CaretValue, format 2
    self = ot.CaretValue()
    self.Format = 2
    self.CaretValuePoint = point
    return self


def buildLigCaretList(coords, points, glyphMap):
    """Builds a ligature caret list table.

    Ligatures appear as a single glyph representing multiple characters; however
    when, for example, editing text containing a ``f_i`` ligature, the user may
    want to place the cursor between the ``f`` and the ``i``. The ligature caret
    list in the GDEF table specifies the position to display the "caret" (the
    character insertion indicator, typically a flashing vertical bar) "inside"
    the ligature to represent an insertion point. The insertion positions may
    be specified either by coordinate or by contour point.

    Example::

        coords = {
            "f_f_i": [300, 600] # f|fi cursor at 300 units, ff|i cursor at 600.
        }
        points = {
            "c_t": [28] # c|t cursor appears at coordinate of contour point 28.
        }
        ligcaretlist = buildLigCaretList(coords, points, font.getReverseGlyphMap())

    Args:
        coords: A mapping between glyph names and a list of coordinates for
            the insertion point of each ligature component after the first one.
        points: A mapping between glyph names and a list of contour points for
            the insertion point of each ligature component after the first one.
        glyphMap: a glyph name to ID map, typically returned from
            ``font.getReverseGlyphMap()``.

    Returns:
        A ``otTables.LigCaretList`` object if any carets are present, or
            ``None`` otherwise."""
    glyphs = set(coords.keys()) if coords else set()
    if points:
        glyphs.update(points.keys())
    carets = {g: buildLigGlyph(coords.get(g), points.get(g)) for g in glyphs}
    carets = {g: c for g, c in carets.items() if c is not None}
    if not carets:
        return None
    self = ot.LigCaretList()
    self.Coverage = buildCoverage(carets.keys(), glyphMap)
    self.LigGlyph = [carets[g] for g in self.Coverage.glyphs]
    self.LigGlyphCount = len(self.LigGlyph)
    return self


def buildLigGlyph(coords, points):
    # ([500], [4]) --> otTables.LigGlyph; None for empty coords/points
    carets = []
    if coords:
        carets.extend([buildCaretValueForCoord(c) for c in sorted(coords)])
    if points:
        carets.extend([buildCaretValueForPoint(p) for p in sorted(points)])
    if not carets:
        return None
    self = ot.LigGlyph()
    self.CaretValue = carets
    self.CaretCount = len(self.CaretValue)
    return self


def buildMarkGlyphSetsDef(markSets, glyphMap):
    """Builds a mark glyph sets definition table.

    OpenType Layout lookups may choose to use mark filtering sets to consider
    or ignore particular combinations of marks. These sets are specified by
    setting a flag on the lookup, but the mark filtering sets are defined in
    the ``GDEF`` table. This routine builds the subtable containing the mark
    glyph set definitions.

    Example::

        set0 = set("acute", "grave")
        set1 = set("caron", "grave")

        markglyphsets = buildMarkGlyphSetsDef([set0, set1], font.getReverseGlyphMap())

    Args:

        markSets: A list of sets of glyphnames.
        glyphMap: a glyph name to ID map, typically returned from
            ``font.getReverseGlyphMap()``.

    Returns
        An ``otTables.MarkGlyphSetsDef`` object.
    """
    if not markSets:
        return None
    self = ot.MarkGlyphSetsDef()
    self.MarkSetTableFormat = 1
    self.Coverage = [buildCoverage(m, glyphMap) for m in markSets]
    self.MarkSetCount = len(self.Coverage)
    return self


class ClassDefBuilder(object):
    """Helper for building ClassDef tables."""

    def __init__(self, useClass0):
        self.classes_ = set()
        self.glyphs_ = {}
        self.useClass0_ = useClass0

    def canAdd(self, glyphs):
        if isinstance(glyphs, (set, frozenset)):
            glyphs = sorted(glyphs)
        glyphs = tuple(glyphs)
        if glyphs in self.classes_:
            return True
        for glyph in glyphs:
            if glyph in self.glyphs_:
                return False
        return True

    def add(self, glyphs):
        if isinstance(glyphs, (set, frozenset)):
            glyphs = sorted(glyphs)
        glyphs = tuple(glyphs)
        if glyphs in self.classes_:
            return
        self.classes_.add(glyphs)
        for glyph in glyphs:
            if glyph in self.glyphs_:
                raise OpenTypeLibError(
                    f"Glyph {glyph} is already present in class.", None
                )
            self.glyphs_[glyph] = glyphs

    def classes(self):
        # In ClassDef1 tables, class id #0 does not need to be encoded
        # because zero is the default. Therefore, we use id #0 for the
        # glyph class that has the largest number of members. However,
        # in other tables than ClassDef1, 0 means "every other glyph"
        # so we should not use that ID for any real glyph classes;
        # we implement this by inserting an empty set at position 0.
        #
        # TODO: Instead of counting the number of glyphs in each class,
        # we should determine the encoded size. If the glyphs in a large
        # class form a contiguous range, the encoding is actually quite
        # compact, whereas a non-contiguous set might need a lot of bytes
        # in the output file. We don't get this right with the key below.
        result = sorted(self.classes_, key=lambda s: (len(s), s), reverse=True)
        if not self.useClass0_:
            result.insert(0, frozenset())
        return result

    def build(self):
        glyphClasses = {}
        for classID, glyphs in enumerate(self.classes()):
            if classID == 0:
                continue
            for glyph in glyphs:
                glyphClasses[glyph] = classID
        classDef = ot.ClassDef()
        classDef.classDefs = glyphClasses
        return classDef


AXIS_VALUE_NEGATIVE_INFINITY = fixedToFloat(-0x80000000, 16)
AXIS_VALUE_POSITIVE_INFINITY = fixedToFloat(0x7FFFFFFF, 16)


def buildStatTable(ttFont, axes, locations=None, elidedFallbackName=2):
    """Add a 'STAT' table to 'ttFont'.

    'axes' is a list of dictionaries describing axes and their
    values.

    Example::

        axes = [
            dict(
                tag="wght",
                name="Weight",
                ordering=0,  # optional
                values=[
                    dict(value=100, name='Thin'),
                    dict(value=300, name='Light'),
                    dict(value=400, name='Regular', flags=0x2),
                    dict(value=900, name='Black'),
                ],
            )
        ]

    Each axis dict must have 'tag' and 'name' items. 'tag' maps
    to the 'AxisTag' field. 'name' can be a name ID (int), a string,
    or a dictionary containing multilingual names (see the
    addMultilingualName() name table method), and will translate to
    the AxisNameID field.

    An axis dict may contain an 'ordering' item that maps to the
    AxisOrdering field. If omitted, the order of the axes list is
    used to calculate AxisOrdering fields.

    The axis dict may contain a 'values' item, which is a list of
    dictionaries describing AxisValue records belonging to this axis.

    Each value dict must have a 'name' item, which can be a name ID
    (int), a string, or a dictionary containing multilingual names,
    like the axis name. It translates to the ValueNameID field.

    Optionally the value dict can contain a 'flags' item. It maps to
    the AxisValue Flags field, and will be 0 when omitted.

    The format of the AxisValue is determined by the remaining contents
    of the value dictionary:

    If the value dict contains a 'value' item, an AxisValue record
    Format 1 is created. If in addition to the 'value' item it contains
    a 'linkedValue' item, an AxisValue record Format 3 is built.

    If the value dict contains a 'nominalValue' item, an AxisValue
    record Format 2 is built. Optionally it may contain 'rangeMinValue'
    and 'rangeMaxValue' items. These map to -Infinity and +Infinity
    respectively if omitted.

    You cannot specify Format 4 AxisValue tables this way, as they are
    not tied to a single axis, and specify a name for a location that
    is defined by multiple axes values. Instead, you need to supply the
    'locations' argument.

    The optional 'locations' argument specifies AxisValue Format 4
    tables. It should be a list of dicts, where each dict has a 'name'
    item, which works just like the value dicts above, an optional
    'flags' item (defaulting to 0x0), and a 'location' dict. A
    location dict key is an axis tag, and the associated value is the
    location on the specified axis. They map to the AxisIndex and Value
    fields of the AxisValueRecord.

    Example::

        locations = [
            dict(name='Regular ABCD', location=dict(wght=300, ABCD=100)),
            dict(name='Bold ABCD XYZ', location=dict(wght=600, ABCD=200)),
        ]

    The optional 'elidedFallbackName' argument can be a name ID (int),
    a string, a dictionary containing multilingual names, or a list of
    STATNameStatements. It translates to the ElidedFallbackNameID field.

    The 'ttFont' argument must be a TTFont instance that already has a
    'name' table. If a 'STAT' table already exists, it will be
    overwritten by the newly created one.
    """
    ttFont["STAT"] = ttLib.newTable("STAT")
    statTable = ttFont["STAT"].table = ot.STAT()
    nameTable = ttFont["name"]
    statTable.ElidedFallbackNameID = _addName(nameTable, elidedFallbackName)

    # 'locations' contains data for AxisValue Format 4
    axisRecords, axisValues = _buildAxisRecords(axes, nameTable)
    if not locations:
        statTable.Version = 0x00010001
    else:
        # We'll be adding Format 4 AxisValue records, which
        # requires a higher table version
        statTable.Version = 0x00010002
        multiAxisValues = _buildAxisValuesFormat4(locations, axes, nameTable)
        axisValues = multiAxisValues + axisValues

    # Store AxisRecords
    axisRecordArray = ot.AxisRecordArray()
    axisRecordArray.Axis = axisRecords
    # XXX these should not be hard-coded but computed automatically
    statTable.DesignAxisRecordSize = 8
    statTable.DesignAxisRecord = axisRecordArray
    statTable.DesignAxisCount = len(axisRecords)

    if axisValues:
        # Store AxisValueRecords
        axisValueArray = ot.AxisValueArray()
        axisValueArray.AxisValue = axisValues
        statTable.AxisValueArray = axisValueArray
        statTable.AxisValueCount = len(axisValues)


def _buildAxisRecords(axes, nameTable):
    axisRecords = []
    axisValues = []
    for axisRecordIndex, axisDict in enumerate(axes):
        axis = ot.AxisRecord()
        axis.AxisTag = axisDict["tag"]
        axis.AxisNameID = _addName(nameTable, axisDict["name"], 256)
        axis.AxisOrdering = axisDict.get("ordering", axisRecordIndex)
        axisRecords.append(axis)

        for axisVal in axisDict.get("values", ()):
            axisValRec = ot.AxisValue()
            axisValRec.AxisIndex = axisRecordIndex
            axisValRec.Flags = axisVal.get("flags", 0)
            axisValRec.ValueNameID = _addName(nameTable, axisVal["name"])

            if "value" in axisVal:
                axisValRec.Value = axisVal["value"]
                if "linkedValue" in axisVal:
                    axisValRec.Format = 3
                    axisValRec.LinkedValue = axisVal["linkedValue"]
                else:
                    axisValRec.Format = 1
            elif "nominalValue" in axisVal:
                axisValRec.Format = 2
                axisValRec.NominalValue = axisVal["nominalValue"]
                axisValRec.RangeMinValue = axisVal.get(
                    "rangeMinValue", AXIS_VALUE_NEGATIVE_INFINITY
                )
                axisValRec.RangeMaxValue = axisVal.get(
                    "rangeMaxValue", AXIS_VALUE_POSITIVE_INFINITY
                )
            else:
                raise ValueError("Can't determine format for AxisValue")

            axisValues.append(axisValRec)
    return axisRecords, axisValues


def _buildAxisValuesFormat4(locations, axes, nameTable):
    axisTagToIndex = {}
    for axisRecordIndex, axisDict in enumerate(axes):
        axisTagToIndex[axisDict["tag"]] = axisRecordIndex

    axisValues = []
    for axisLocationDict in locations:
        axisValRec = ot.AxisValue()
        axisValRec.Format = 4
        axisValRec.ValueNameID = _addName(nameTable, axisLocationDict["name"])
        axisValRec.Flags = axisLocationDict.get("flags", 0)
        axisValueRecords = []
        for tag, value in axisLocationDict["location"].items():
            avr = ot.AxisValueRecord()
            avr.AxisIndex = axisTagToIndex[tag]
            avr.Value = value
            axisValueRecords.append(avr)
        axisValueRecords.sort(key=lambda avr: avr.AxisIndex)
        axisValRec.AxisCount = len(axisValueRecords)
        axisValRec.AxisValueRecord = axisValueRecords
        axisValues.append(axisValRec)
    return axisValues


def _addName(nameTable, value, minNameID=0):
    if isinstance(value, int):
        # Already a nameID
        return value
    if isinstance(value, str):
        names = dict(en=value)
    elif isinstance(value, dict):
        names = value
    elif isinstance(value, list):
        nameID = nameTable._findUnusedNameID()
        for nameRecord in value:
            if isinstance(nameRecord, STATNameStatement):
                nameTable.setName(
                    nameRecord.string,
                    nameID,
                    nameRecord.platformID,
                    nameRecord.platEncID,
                    nameRecord.langID,
                )
            else:
                raise TypeError("value must be a list of STATNameStatements")
        return nameID
    else:
        raise TypeError("value must be int, str, dict or list")
    return nameTable.addMultilingualName(names, minNameID=minNameID)
