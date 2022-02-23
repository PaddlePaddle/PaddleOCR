from fontTools.misc import sstruct
from fontTools.misc.fixedTools import (
    fixedToFloat as fi2fl,
    floatToFixed as fl2fi,
    floatToFixedToStr as fl2str,
    strToFixedToFloat as str2fl,
)
from fontTools.misc.textTools import bytesjoin
from fontTools.ttLib import TTLibError
from . import DefaultTable
import struct
import logging


log = logging.getLogger(__name__)

# Apple's documentation of 'avar':
# https://developer.apple.com/fonts/TrueType-Reference-Manual/RM06/Chap6avar.html

AVAR_HEADER_FORMAT = """
    > # big endian
    majorVersion:  H
    minorVersion:  H
    reserved:      H
    axisCount:     H
"""
assert sstruct.calcsize(AVAR_HEADER_FORMAT) == 8, sstruct.calcsize(AVAR_HEADER_FORMAT)


class table__a_v_a_r(DefaultTable.DefaultTable):
    """Axis Variations Table

    This class represents the ``avar`` table of a variable font. The object has one
    substantive attribute, ``segments``, which maps axis tags to a segments dictionary::

        >>> font["avar"].segments   # doctest: +SKIP
        {'wght': {-1.0: -1.0,
          0.0: 0.0,
          0.125: 0.11444091796875,
          0.25: 0.23492431640625,
          0.5: 0.35540771484375,
          0.625: 0.5,
          0.75: 0.6566162109375,
          0.875: 0.81927490234375,
          1.0: 1.0},
         'ital': {-1.0: -1.0, 0.0: 0.0, 1.0: 1.0}}

    Notice that the segments dictionary is made up of normalized values. A valid
    ``avar`` segment mapping must contain the entries ``-1.0: -1.0, 0.0: 0.0, 1.0: 1.0``.
    fontTools does not enforce this, so it is your responsibility to ensure that
    mappings are valid.
    """

    dependencies = ["fvar"]

    def __init__(self, tag=None):
        DefaultTable.DefaultTable.__init__(self, tag)
        self.segments = {}

    def compile(self, ttFont):
        axisTags = [axis.axisTag for axis in ttFont["fvar"].axes]
        header = {
            "majorVersion": 1,
            "minorVersion": 0,
            "reserved": 0,
            "axisCount": len(axisTags)
        }
        result = [sstruct.pack(AVAR_HEADER_FORMAT, header)]
        for axis in axisTags:
            mappings = sorted(self.segments[axis].items())
            result.append(struct.pack(">H", len(mappings)))
            for key, value in mappings:
                fixedKey = fl2fi(key, 14)
                fixedValue = fl2fi(value, 14)
                result.append(struct.pack(">hh", fixedKey, fixedValue))
        return bytesjoin(result)

    def decompile(self, data, ttFont):
        axisTags = [axis.axisTag for axis in ttFont["fvar"].axes]
        header = {}
        headerSize = sstruct.calcsize(AVAR_HEADER_FORMAT)
        header = sstruct.unpack(AVAR_HEADER_FORMAT, data[0:headerSize])
        majorVersion = header["majorVersion"]
        if majorVersion != 1:
            raise TTLibError("unsupported 'avar' version %d" % majorVersion)
        pos = headerSize
        for axis in axisTags:
            segments = self.segments[axis] = {}
            numPairs = struct.unpack(">H", data[pos:pos+2])[0]
            pos = pos + 2
            for _ in range(numPairs):
                fromValue, toValue = struct.unpack(">hh", data[pos:pos+4])
                segments[fi2fl(fromValue, 14)] = fi2fl(toValue, 14)
                pos = pos + 4

    def toXML(self, writer, ttFont):
        axisTags = [axis.axisTag for axis in ttFont["fvar"].axes]
        for axis in axisTags:
            writer.begintag("segment", axis=axis)
            writer.newline()
            for key, value in sorted(self.segments[axis].items()):
                key = fl2str(key, 14)
                value = fl2str(value, 14)
                writer.simpletag("mapping", **{"from": key, "to": value})
                writer.newline()
            writer.endtag("segment")
            writer.newline()

    def fromXML(self, name, attrs, content, ttFont):
        if name == "segment":
            axis = attrs["axis"]
            segment = self.segments[axis] = {}
            for element in content:
                if isinstance(element, tuple):
                    elementName, elementAttrs, _ = element
                    if elementName == "mapping":
                        fromValue = str2fl(elementAttrs["from"], 14)
                        toValue = str2fl(elementAttrs["to"], 14)
                        if fromValue in segment:
                            log.warning("duplicate entry for %s in axis '%s'",
                                        fromValue, axis)
                        segment[fromValue] = toValue
