from fontTools.misc import sstruct
from fontTools.misc.textTools import safeEval
from . import DefaultTable
import array
import itertools
import logging
import struct
import sys
import fontTools.ttLib.tables.TupleVariation as tv


log = logging.getLogger(__name__)
TupleVariation = tv.TupleVariation


# https://www.microsoft.com/typography/otspec/gvar.htm
# https://www.microsoft.com/typography/otspec/otvarcommonformats.htm
#
# Apple's documentation of 'gvar':
# https://developer.apple.com/fonts/TrueType-Reference-Manual/RM06/Chap6gvar.html
#
# FreeType2 source code for parsing 'gvar':
# http://git.savannah.gnu.org/cgit/freetype/freetype2.git/tree/src/truetype/ttgxvar.c

GVAR_HEADER_FORMAT = """
	> # big endian
	version:			H
	reserved:			H
	axisCount:			H
	sharedTupleCount:		H
	offsetToSharedTuples:		I
	glyphCount:			H
	flags:				H
	offsetToGlyphVariationData:	I
"""

GVAR_HEADER_SIZE = sstruct.calcsize(GVAR_HEADER_FORMAT)


class table__g_v_a_r(DefaultTable.DefaultTable):
	dependencies = ["fvar", "glyf"]

	def __init__(self, tag=None):
		DefaultTable.DefaultTable.__init__(self, tag)
		self.version, self.reserved = 1, 0
		self.variations = {}

	def compile(self, ttFont):
		axisTags = [axis.axisTag for axis in ttFont["fvar"].axes]
		sharedTuples =  tv.compileSharedTuples(
			axisTags, itertools.chain(*self.variations.values()))
		sharedTupleIndices = {coord:i for i, coord in enumerate(sharedTuples)}
		sharedTupleSize = sum([len(c) for c in sharedTuples])
		compiledGlyphs = self.compileGlyphs_(
			ttFont, axisTags, sharedTupleIndices)
		offset = 0
		offsets = []
		for glyph in compiledGlyphs:
			offsets.append(offset)
			offset += len(glyph)
		offsets.append(offset)
		compiledOffsets, tableFormat = self.compileOffsets_(offsets)

		header = {}
		header["version"] = self.version
		header["reserved"] = self.reserved
		header["axisCount"] = len(axisTags)
		header["sharedTupleCount"] = len(sharedTuples)
		header["offsetToSharedTuples"] = GVAR_HEADER_SIZE + len(compiledOffsets)
		header["glyphCount"] = len(compiledGlyphs)
		header["flags"] = tableFormat
		header["offsetToGlyphVariationData"] = header["offsetToSharedTuples"] + sharedTupleSize
		compiledHeader = sstruct.pack(GVAR_HEADER_FORMAT, header)

		result = [compiledHeader, compiledOffsets]
		result.extend(sharedTuples)
		result.extend(compiledGlyphs)
		return b''.join(result)

	def compileGlyphs_(self, ttFont, axisTags, sharedCoordIndices):
		result = []
		glyf = ttFont['glyf']
		for glyphName in ttFont.getGlyphOrder():
			glyph = glyf[glyphName]
			pointCount = self.getNumPoints_(glyph)
			variations = self.variations.get(glyphName, [])
			result.append(compileGlyph_(variations, pointCount,
			                            axisTags, sharedCoordIndices))
		return result

	def decompile(self, data, ttFont):
		axisTags = [axis.axisTag for axis in ttFont["fvar"].axes]
		glyphs = ttFont.getGlyphOrder()
		sstruct.unpack(GVAR_HEADER_FORMAT, data[0:GVAR_HEADER_SIZE], self)
		assert len(glyphs) == self.glyphCount
		assert len(axisTags) == self.axisCount
		offsets = self.decompileOffsets_(data[GVAR_HEADER_SIZE:], tableFormat=(self.flags & 1), glyphCount=self.glyphCount)
		sharedCoords = tv.decompileSharedTuples(
			axisTags, self.sharedTupleCount, data, self.offsetToSharedTuples)
		self.variations = {}
		offsetToData = self.offsetToGlyphVariationData
		glyf = ttFont['glyf']
		for i in range(self.glyphCount):
			glyphName = glyphs[i]
			glyph = glyf[glyphName]
			numPointsInGlyph = self.getNumPoints_(glyph)
			gvarData = data[offsetToData + offsets[i] : offsetToData + offsets[i + 1]]
			try:
				self.variations[glyphName] = decompileGlyph_(
					numPointsInGlyph, sharedCoords, axisTags, gvarData)
			except Exception:
				log.error(
					"Failed to decompile deltas for glyph '%s' (%d points)",
					glyphName, numPointsInGlyph,
				)
				raise

	@staticmethod
	def decompileOffsets_(data, tableFormat, glyphCount):
		if tableFormat == 0:
			# Short format: array of UInt16
			offsets = array.array("H")
			offsetsSize = (glyphCount + 1) * 2
		else:
			# Long format: array of UInt32
			offsets = array.array("I")
			offsetsSize = (glyphCount + 1) * 4
		offsets.frombytes(data[0 : offsetsSize])
		if sys.byteorder != "big": offsets.byteswap()

		# In the short format, offsets need to be multiplied by 2.
		# This is not documented in Apple's TrueType specification,
		# but can be inferred from the FreeType implementation, and
		# we could verify it with two sample GX fonts.
		if tableFormat == 0:
			offsets = [off * 2 for off in offsets]

		return offsets

	@staticmethod
	def compileOffsets_(offsets):
		"""Packs a list of offsets into a 'gvar' offset table.

		Returns a pair (bytestring, tableFormat). Bytestring is the
		packed offset table. Format indicates whether the table
		uses short (tableFormat=0) or long (tableFormat=1) integers.
		The returned tableFormat should get packed into the flags field
		of the 'gvar' header.
		"""
		assert len(offsets) >= 2
		for i in range(1, len(offsets)):
			assert offsets[i - 1] <= offsets[i]
		if max(offsets) <= 0xffff * 2:
			packed = array.array("H", [n >> 1 for n in offsets])
			tableFormat = 0
		else:
			packed = array.array("I", offsets)
			tableFormat = 1
		if sys.byteorder != "big": packed.byteswap()
		return (packed.tobytes(), tableFormat)

	def toXML(self, writer, ttFont):
		writer.simpletag("version", value=self.version)
		writer.newline()
		writer.simpletag("reserved", value=self.reserved)
		writer.newline()
		axisTags = [axis.axisTag for axis in ttFont["fvar"].axes]
		for glyphName in ttFont.getGlyphNames():
			variations = self.variations.get(glyphName)
			if not variations:
				continue
			writer.begintag("glyphVariations", glyph=glyphName)
			writer.newline()
			for gvar in variations:
				gvar.toXML(writer, axisTags)
			writer.endtag("glyphVariations")
			writer.newline()

	def fromXML(self, name, attrs, content, ttFont):
		if name == "version":
			self.version = safeEval(attrs["value"])
		elif name == "reserved":
			self.reserved = safeEval(attrs["value"])
		elif name == "glyphVariations":
			if not hasattr(self, "variations"):
				self.variations = {}
			glyphName = attrs["glyph"]
			glyph = ttFont["glyf"][glyphName]
			numPointsInGlyph = self.getNumPoints_(glyph)
			glyphVariations = []
			for element in content:
				if isinstance(element, tuple):
					name, attrs, content = element
					if name == "tuple":
						gvar = TupleVariation({}, [None] * numPointsInGlyph)
						glyphVariations.append(gvar)
						for tupleElement in content:
							if isinstance(tupleElement, tuple):
								tupleName, tupleAttrs, tupleContent = tupleElement
								gvar.fromXML(tupleName, tupleAttrs, tupleContent)
			self.variations[glyphName] = glyphVariations

	@staticmethod
	def getNumPoints_(glyph):
		NUM_PHANTOM_POINTS = 4
		if glyph.isComposite():
			return len(glyph.components) + NUM_PHANTOM_POINTS
		else:
			# Empty glyphs (eg. space, nonmarkingreturn) have no "coordinates" attribute.
			return len(getattr(glyph, "coordinates", [])) + NUM_PHANTOM_POINTS


def compileGlyph_(variations, pointCount, axisTags, sharedCoordIndices):
	tupleVariationCount, tuples, data = tv.compileTupleVariationStore(
		variations, pointCount, axisTags, sharedCoordIndices)
	if tupleVariationCount == 0:
		return b""
	result = [
		struct.pack(">HH", tupleVariationCount, 4 + len(tuples)),
		tuples,
		data
	]
	if (len(tuples) + len(data)) % 2 != 0:
		result.append(b"\0")  # padding
	return b''.join(result)


def decompileGlyph_(pointCount, sharedTuples, axisTags, data):
	if len(data) < 4:
		return []
	tupleVariationCount, offsetToData = struct.unpack(">HH", data[:4])
	dataPos = offsetToData
	return tv.decompileTupleVariationStore(
		"gvar", axisTags,
		tupleVariationCount, pointCount,
		sharedTuples, data, 4, offsetToData
	)
