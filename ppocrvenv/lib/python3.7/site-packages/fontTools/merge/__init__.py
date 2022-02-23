# Copyright 2013 Google, Inc. All Rights Reserved.
#
# Google Author(s): Behdad Esfahbod, Roozbeh Pournader

from fontTools import ttLib
import fontTools.merge.base
from fontTools.merge.cmap import computeMegaGlyphOrder, computeMegaCmap, renameCFFCharStrings
from fontTools.merge.layout import layoutPreMerge, layoutPostMerge
from fontTools.merge.options import Options
import fontTools.merge.tables
from fontTools.misc.loggingTools import Timer
from functools import reduce
import sys
import logging


log = logging.getLogger("fontTools.merge")
timer = Timer(logger=logging.getLogger(__name__+".timer"), level=logging.INFO)


class Merger(object):
	"""Font merger.

	This class merges multiple files into a single OpenType font, taking into
	account complexities such as OpenType layout (``GSUB``/``GPOS``) tables and
	cross-font metrics (e.g. ``hhea.ascent`` is set to the maximum value across
	all the fonts).

	If multiple glyphs map to the same Unicode value, and the glyphs are considered
	sufficiently different (that is, they differ in any of paths, widths, or
	height), then subsequent glyphs are renamed and a lookup in the ``locl``
	feature will be created to disambiguate them. For example, if the arguments
	are an Arabic font and a Latin font and both contain a set of parentheses,
	the Latin glyphs will be renamed to ``parenleft#1`` and ``parenright#1``,
	and a lookup will be inserted into the to ``locl`` feature (creating it if
	necessary) under the ``latn`` script to substitute ``parenleft`` with
	``parenleft#1`` etc.

	Restrictions:

	- All fonts must have the same units per em.
	- If duplicate glyph disambiguation takes place as described above then the
		fonts must have a ``GSUB`` table.

	Attributes:
		options: Currently unused.
	"""

	def __init__(self, options=None):

		if not options:
			options = Options()

		self.options = options

	def _openFonts(self, fontfiles):
		fonts = [ttLib.TTFont(fontfile) for fontfile in fontfiles]
		for font,fontfile in zip(fonts, fontfiles):
			font._merger__fontfile = fontfile
			font._merger__name = font['name'].getDebugName(4)
		return fonts

	def merge(self, fontfiles):
		"""Merges fonts together.

		Args:
			fontfiles: A list of file names to be merged

		Returns:
			A :class:`fontTools.ttLib.TTFont` object. Call the ``save`` method on
			this to write it out to an OTF file.
		"""
		#
		# Settle on a mega glyph order.
		#
		fonts = self._openFonts(fontfiles)
		glyphOrders = [list(font.getGlyphOrder()) for font in fonts]
		computeMegaGlyphOrder(self, glyphOrders)

		# Take first input file sfntVersion
		sfntVersion = fonts[0].sfntVersion

		# Reload fonts and set new glyph names on them.
		fonts = self._openFonts(fontfiles)
		for font,glyphOrder in zip(fonts, glyphOrders):
			font.setGlyphOrder(glyphOrder)
			if 'CFF ' in font:
				renameCFFCharStrings(self, glyphOrder, font['CFF '])

		cmaps = [font['cmap'] for font in fonts]
		self.duplicateGlyphsPerFont = [{} for _ in fonts]
		computeMegaCmap(self, cmaps)

		mega = ttLib.TTFont(sfntVersion=sfntVersion)
		mega.setGlyphOrder(self.glyphOrder)

		for font in fonts:
			self._preMerge(font)

		self.fonts = fonts

		allTags = reduce(set.union, (list(font.keys()) for font in fonts), set())
		allTags.remove('GlyphOrder')

		for tag in allTags:
			if tag in self.options.drop_tables:
				continue

			with timer("merge '%s'" % tag):
				tables = [font.get(tag, NotImplemented) for font in fonts]

				log.info("Merging '%s'.", tag)
				clazz = ttLib.getTableClass(tag)
				table = clazz(tag).merge(self, tables)
				# XXX Clean this up and use:  table = mergeObjects(tables)

				if table is not NotImplemented and table is not False:
					mega[tag] = table
					log.info("Merged '%s'.", tag)
				else:
					log.info("Dropped '%s'.", tag)

		del self.duplicateGlyphsPerFont
		del self.fonts

		self._postMerge(mega)

		return mega

	def mergeObjects(self, returnTable, logic, tables):
		# Right now we don't use self at all.  Will use in the future
		# for options and logging.

		allKeys = set.union(set(), *(vars(table).keys() for table in tables if table is not NotImplemented))
		for key in allKeys:
			try:
				mergeLogic = logic[key]
			except KeyError:
				try:
					mergeLogic = logic['*']
				except KeyError:
					raise Exception("Don't know how to merge key %s of class %s" %
							(key, returnTable.__class__.__name__))
			if mergeLogic is NotImplemented:
				continue
			value = mergeLogic(getattr(table, key, NotImplemented) for table in tables)
			if value is not NotImplemented:
				setattr(returnTable, key, value)

		return returnTable

	def _preMerge(self, font):
		layoutPreMerge(font)

	def _postMerge(self, font):
		layoutPostMerge(font)


__all__ = [
	'Options',
	'Merger',
	'main'
]

@timer("make one with everything (TOTAL TIME)")
def main(args=None):
	"""Merge multiple fonts into one"""
	from fontTools import configLogger

	if args is None:
		args = sys.argv[1:]

	options = Options()
	args = options.parse_opts(args, ignore_unknown=['output-file'])
	outfile = 'merged.ttf'
	fontfiles = []
	for g in args:
		if g.startswith('--output-file='):
			outfile = g[14:]
			continue
		fontfiles.append(g)

	if len(args) < 1:
		print("usage: pyftmerge font...", file=sys.stderr)
		return 1

	configLogger(level=logging.INFO if options.verbose else logging.WARNING)
	if options.timing:
		timer.logger.setLevel(logging.DEBUG)
	else:
		timer.logger.disabled = True

	merger = Merger(options=options)
	font = merger.merge(fontfiles)
	with timer("compile and save font"):
		font.save(outfile)


if __name__ == "__main__":
	sys.exit(main())
