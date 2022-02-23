import textwrap


class VarLibError(Exception):
    """Base exception for the varLib module."""


class VarLibValidationError(VarLibError):
    """Raised when input data is invalid from varLib's point of view."""


class VarLibMergeError(VarLibError):
    """Raised when input data cannot be merged into a variable font."""

    def __init__(self, merger=None, **kwargs):
        self.merger = merger
        if not kwargs:
            kwargs = {}
        if "stack" in kwargs:
            self.stack = kwargs["stack"]
            del kwargs["stack"]
        else:
            self.stack = []
        self.cause = kwargs

    @property
    def reason(self):
        return self.__doc__

    def _master_name(self, ix):
        if self.merger is not None:
            ttf = self.merger.ttfs[ix]
            if (
                "name" in ttf
                and ttf["name"].getDebugName(1)
                and ttf["name"].getDebugName(2)
            ):
                return ttf["name"].getDebugName(1) + " " + ttf["name"].getDebugName(2)
            elif hasattr(ttf.reader, "file") and hasattr(ttf.reader.file, "name"):
                return ttf.reader.file.name
        return f"master number {ix}"

    @property
    def offender(self):
        if "expected" in self.cause and "got" in self.cause:
            index = [x == self.cause["expected"] for x in self.cause["got"]].index(
                False
            )
            return index, self._master_name(index)
        return None, None

    @property
    def details(self):
        if "expected" in self.cause and "got" in self.cause:
            offender_index, offender = self.offender
            got = self.cause["got"][offender_index]
            return f"Expected to see {self.stack[0]}=={self.cause['expected']}, instead saw {got}\n"
        return ""

    def __str__(self):
        offender_index, offender = self.offender
        location = ""
        if offender:
            location = f"\n\nThe problem is likely to be in {offender}:\n"
        context = "".join(reversed(self.stack))
        basic = textwrap.fill(
            f"Couldn't merge the fonts, because {self.reason}. "
            f"This happened while performing the following operation: {context}",
            width=78,
        )
        return "\n\n" + basic + location + self.details


class ShouldBeConstant(VarLibMergeError):
    """some values were different, but should have been the same"""

    @property
    def details(self):
        if self.stack[0] != ".FeatureCount" or self.merger is None:
            return super().details
        offender_index, offender = self.offender
        bad_ttf = self.merger.ttfs[offender_index]
        good_ttf = self.merger.ttfs[offender_index - 1]

        good_features = [
            x.FeatureTag
            for x in good_ttf[self.stack[-1]].table.FeatureList.FeatureRecord
        ]
        bad_features = [
            x.FeatureTag
            for x in bad_ttf[self.stack[-1]].table.FeatureList.FeatureRecord
        ]
        return (
            "\nIncompatible features between masters.\n"
            f"Expected: {', '.join(good_features)}.\n"
            f"Got: {', '.join(bad_features)}.\n"
        )


class FoundANone(VarLibMergeError):
    """one of the values in a list was empty when it shouldn't have been"""

    @property
    def offender(self):
        cause = self.argv[0]
        index = [x is None for x in cause["got"]].index(True)
        return index, self._master_name(index)

    @property
    def details(self):
        cause, stack = self.args[0], self.args[1:]
        return f"{stack[0]}=={cause['got']}\n"


class MismatchedTypes(VarLibMergeError):
    """data had inconsistent types"""


class LengthsDiffer(VarLibMergeError):
    """a list of objects had inconsistent lengths"""


class KeysDiffer(VarLibMergeError):
    """a list of objects had different keys"""


class InconsistentGlyphOrder(VarLibMergeError):
    """the glyph order was inconsistent between masters"""


class InconsistentExtensions(VarLibMergeError):
    """the masters use extension lookups in inconsistent ways"""


class UnsupportedFormat(VarLibMergeError):
    """an OpenType subtable (%s) had a format I didn't expect"""

    @property
    def reason(self):
        cause, stack = self.args[0], self.args[1:]
        return self.__doc__ % cause["subtable"]


class UnsupportedFormat(UnsupportedFormat):
    """an OpenType subtable (%s) had inconsistent formats between masters"""


class VarLibCFFMergeError(VarLibError):
    pass


class VarLibCFFDictMergeError(VarLibCFFMergeError):
    """Raised when a CFF PrivateDict cannot be merged."""

    def __init__(self, key, value, values):
        error_msg = (
            f"For the Private Dict key '{key}', the default font value list:"
            f"\n\t{value}\nhad a different number of values than a region font:"
        )
        for region_value in values:
            error_msg += f"\n\t{region_value}"
        self.args = (error_msg,)


class VarLibCFFPointTypeMergeError(VarLibCFFMergeError):
    """Raised when a CFF glyph cannot be merged because of point type differences."""

    def __init__(self, point_type, pt_index, m_index, default_type, glyph_name):
        error_msg = (
            f"Glyph '{glyph_name}': '{point_type}' at point index {pt_index} in "
            f"master index {m_index} differs from the default font point type "
            f"'{default_type}'"
        )
        self.args = (error_msg,)


class VarLibCFFHintTypeMergeError(VarLibCFFMergeError):
    """Raised when a CFF glyph cannot be merged because of hint type differences."""

    def __init__(self, hint_type, cmd_index, m_index, default_type, glyph_name):
        error_msg = (
            f"Glyph '{glyph_name}': '{hint_type}' at index {cmd_index} in "
            f"master index {m_index} differs from the default font hint type "
            f"'{default_type}'"
        )
        self.args = (error_msg,)


class VariationModelError(VarLibError):
    """Raised when a variation model is faulty."""
