from fontTools.misc import psCharStrings
from fontTools import ttLib
from fontTools.pens.basePen import NullPen
from fontTools.misc.roundTools import otRound
from fontTools.misc.loggingTools import deprecateFunction
from fontTools.varLib.varStore import VarStoreInstancer
from fontTools.subset.util import _add_method, _uniq_sort


class _ClosureGlyphsT2Decompiler(psCharStrings.SimpleT2Decompiler):

	def __init__(self, components, localSubrs, globalSubrs):
		psCharStrings.SimpleT2Decompiler.__init__(self,
							  localSubrs,
							  globalSubrs)
		self.components = components

	def op_endchar(self, index):
		args = self.popall()
		if len(args) >= 4:
			from fontTools.encodings.StandardEncoding import StandardEncoding
			# endchar can do seac accent bulding; The T2 spec says it's deprecated,
			# but recent software that shall remain nameless does output it.
			adx, ady, bchar, achar = args[-4:]
			baseGlyph = StandardEncoding[bchar]
			accentGlyph = StandardEncoding[achar]
			self.components.add(baseGlyph)
			self.components.add(accentGlyph)

@_add_method(ttLib.getTableClass('CFF '))
def closure_glyphs(self, s):
	cff = self.cff
	assert len(cff) == 1
	font = cff[cff.keys()[0]]
	glyphSet = font.CharStrings

	decompose = s.glyphs
	while decompose:
		components = set()
		for g in decompose:
			if g not in glyphSet:
				continue
			gl = glyphSet[g]

			subrs = getattr(gl.private, "Subrs", [])
			decompiler = _ClosureGlyphsT2Decompiler(components, subrs, gl.globalSubrs)
			decompiler.execute(gl)
		components -= s.glyphs
		s.glyphs.update(components)
		decompose = components

def _empty_charstring(font, glyphName, isCFF2, ignoreWidth=False):
	c, fdSelectIndex = font.CharStrings.getItemAndSelector(glyphName)
	if isCFF2 or ignoreWidth:
		# CFF2 charstrings have no widths nor 'endchar' operators
		c.setProgram([] if isCFF2 else ['endchar'])
	else:
		if hasattr(font, 'FDArray') and font.FDArray is not None:
			private = font.FDArray[fdSelectIndex].Private
		else:
			private = font.Private
		dfltWdX = private.defaultWidthX
		nmnlWdX = private.nominalWidthX
		pen = NullPen()
		c.draw(pen)  # this will set the charstring's width
		if c.width != dfltWdX:
			c.program = [c.width - nmnlWdX, 'endchar']
		else:
			c.program = ['endchar']

@_add_method(ttLib.getTableClass('CFF '))
def prune_pre_subset(self, font, options):
	cff = self.cff
	# CFF table must have one font only
	cff.fontNames = cff.fontNames[:1]

	if options.notdef_glyph and not options.notdef_outline:
		isCFF2 = cff.major > 1
		for fontname in cff.keys():
			font = cff[fontname]
			_empty_charstring(font, ".notdef", isCFF2=isCFF2)

	# Clear useless Encoding
	for fontname in cff.keys():
		font = cff[fontname]
		# https://github.com/fonttools/fonttools/issues/620
		font.Encoding = "StandardEncoding"

	return True # bool(cff.fontNames)

@_add_method(ttLib.getTableClass('CFF '))
def subset_glyphs(self, s):
	cff = self.cff
	for fontname in cff.keys():
		font = cff[fontname]
		cs = font.CharStrings

		glyphs = s.glyphs.union(s.glyphs_emptied)

		# Load all glyphs
		for g in font.charset:
			if g not in glyphs: continue
			c, _ = cs.getItemAndSelector(g)

		if cs.charStringsAreIndexed:
			indices = [i for i,g in enumerate(font.charset) if g in glyphs]
			csi = cs.charStringsIndex
			csi.items = [csi.items[i] for i in indices]
			del csi.file, csi.offsets
			if hasattr(font, "FDSelect"):
				sel = font.FDSelect
				# XXX We want to set sel.format to None, such that the
				# most compact format is selected. However, OTS was
				# broken and couldn't parse a FDSelect format 0 that
				# happened before CharStrings. As such, always force
				# format 3 until we fix cffLib to always generate
				# FDSelect after CharStrings.
				# https://github.com/khaledhosny/ots/pull/31
				#sel.format = None
				sel.format = 3
				sel.gidArray = [sel.gidArray[i] for i in indices]
			newCharStrings = {}
			for indicesIdx, charsetIdx in enumerate(indices):
				g = font.charset[charsetIdx]
				if g in cs.charStrings:
					newCharStrings[g] = indicesIdx
			cs.charStrings = newCharStrings
		else:
			cs.charStrings = {g:v
					  for g,v in cs.charStrings.items()
					  if g in glyphs}
		font.charset = [g for g in font.charset if g in glyphs]
		font.numGlyphs = len(font.charset)


		if s.options.retain_gids:
			isCFF2 = cff.major > 1
			for g in s.glyphs_emptied:
				_empty_charstring(font, g, isCFF2=isCFF2, ignoreWidth=True)


	return True # any(cff[fontname].numGlyphs for fontname in cff.keys())

@_add_method(psCharStrings.T2CharString)
def subset_subroutines(self, subrs, gsubrs):
	p = self.program
	for i in range(1, len(p)):
		if p[i] == 'callsubr':
			assert isinstance(p[i-1], int)
			p[i-1] = subrs._used.index(p[i-1] + subrs._old_bias) - subrs._new_bias
		elif p[i] == 'callgsubr':
			assert isinstance(p[i-1], int)
			p[i-1] = gsubrs._used.index(p[i-1] + gsubrs._old_bias) - gsubrs._new_bias

@_add_method(psCharStrings.T2CharString)
def drop_hints(self):
	hints = self._hints

	if hints.deletions:
		p = self.program
		for idx in reversed(hints.deletions):
			del p[idx-2:idx]

	if hints.has_hint:
		assert not hints.deletions or hints.last_hint <= hints.deletions[0]
		self.program = self.program[hints.last_hint:]
		if not self.program:
			# TODO CFF2 no need for endchar.
			self.program.append('endchar')
		if hasattr(self, 'width'):
			# Insert width back if needed
			if self.width != self.private.defaultWidthX:
				# For CFF2 charstrings, this should never happen
				assert self.private.defaultWidthX is not None, "CFF2 CharStrings must not have an initial width value"
				self.program.insert(0, self.width - self.private.nominalWidthX)

	if hints.has_hintmask:
		i = 0
		p = self.program
		while i < len(p):
			if p[i] in ['hintmask', 'cntrmask']:
				assert i + 1 <= len(p)
				del p[i:i+2]
				continue
			i += 1

	assert len(self.program)

	del self._hints

class _MarkingT2Decompiler(psCharStrings.SimpleT2Decompiler):

	def __init__(self, localSubrs, globalSubrs, private):
		psCharStrings.SimpleT2Decompiler.__init__(self,
							  localSubrs,
							  globalSubrs,
							  private)
		for subrs in [localSubrs, globalSubrs]:
			if subrs and not hasattr(subrs, "_used"):
				subrs._used = set()

	def op_callsubr(self, index):
		self.localSubrs._used.add(self.operandStack[-1]+self.localBias)
		psCharStrings.SimpleT2Decompiler.op_callsubr(self, index)

	def op_callgsubr(self, index):
		self.globalSubrs._used.add(self.operandStack[-1]+self.globalBias)
		psCharStrings.SimpleT2Decompiler.op_callgsubr(self, index)

class _DehintingT2Decompiler(psCharStrings.T2WidthExtractor):

	class Hints(object):
		def __init__(self):
			# Whether calling this charstring produces any hint stems
			# Note that if a charstring starts with hintmask, it will
			# have has_hint set to True, because it *might* produce an
			# implicit vstem if called under certain conditions.
			self.has_hint = False
			# Index to start at to drop all hints
			self.last_hint = 0
			# Index up to which we know more hints are possible.
			# Only relevant if status is 0 or 1.
			self.last_checked = 0
			# The status means:
			# 0: after dropping hints, this charstring is empty
			# 1: after dropping hints, there may be more hints
			#	continuing after this, or there might be
			#	other things.  Not clear yet.
			# 2: no more hints possible after this charstring
			self.status = 0
			# Has hintmask instructions; not recursive
			self.has_hintmask = False
			# List of indices of calls to empty subroutines to remove.
			self.deletions = []
		pass

	def __init__(self, css, localSubrs, globalSubrs, nominalWidthX, defaultWidthX, private=None):
		self._css = css
		psCharStrings.T2WidthExtractor.__init__(
			self, localSubrs, globalSubrs, nominalWidthX, defaultWidthX)
		self.private = private

	def execute(self, charString):
		old_hints = charString._hints if hasattr(charString, '_hints') else None
		charString._hints = self.Hints()

		psCharStrings.T2WidthExtractor.execute(self, charString)

		hints = charString._hints

		if hints.has_hint or hints.has_hintmask:
			self._css.add(charString)

		if hints.status != 2:
			# Check from last_check, make sure we didn't have any operators.
			for i in range(hints.last_checked, len(charString.program) - 1):
				if isinstance(charString.program[i], str):
					hints.status = 2
					break
				else:
					hints.status = 1 # There's *something* here
			hints.last_checked = len(charString.program)

		if old_hints:
			assert hints.__dict__ == old_hints.__dict__

	def op_callsubr(self, index):
		subr = self.localSubrs[self.operandStack[-1]+self.localBias]
		psCharStrings.T2WidthExtractor.op_callsubr(self, index)
		self.processSubr(index, subr)

	def op_callgsubr(self, index):
		subr = self.globalSubrs[self.operandStack[-1]+self.globalBias]
		psCharStrings.T2WidthExtractor.op_callgsubr(self, index)
		self.processSubr(index, subr)

	def op_hstem(self, index):
		psCharStrings.T2WidthExtractor.op_hstem(self, index)
		self.processHint(index)
	def op_vstem(self, index):
		psCharStrings.T2WidthExtractor.op_vstem(self, index)
		self.processHint(index)
	def op_hstemhm(self, index):
		psCharStrings.T2WidthExtractor.op_hstemhm(self, index)
		self.processHint(index)
	def op_vstemhm(self, index):
		psCharStrings.T2WidthExtractor.op_vstemhm(self, index)
		self.processHint(index)
	def op_hintmask(self, index):
		rv = psCharStrings.T2WidthExtractor.op_hintmask(self, index)
		self.processHintmask(index)
		return rv
	def op_cntrmask(self, index):
		rv = psCharStrings.T2WidthExtractor.op_cntrmask(self, index)
		self.processHintmask(index)
		return rv

	def processHintmask(self, index):
		cs = self.callingStack[-1]
		hints = cs._hints
		hints.has_hintmask = True
		if hints.status != 2:
			# Check from last_check, see if we may be an implicit vstem
			for i in range(hints.last_checked, index - 1):
				if isinstance(cs.program[i], str):
					hints.status = 2
					break
			else:
				# We are an implicit vstem
				hints.has_hint = True
				hints.last_hint = index + 1
				hints.status = 0
		hints.last_checked = index + 1

	def processHint(self, index):
		cs = self.callingStack[-1]
		hints = cs._hints
		hints.has_hint = True
		hints.last_hint = index
		hints.last_checked = index

	def processSubr(self, index, subr):
		cs = self.callingStack[-1]
		hints = cs._hints
		subr_hints = subr._hints

		# Check from last_check, make sure we didn't have
		# any operators.
		if hints.status != 2:
			for i in range(hints.last_checked, index - 1):
				if isinstance(cs.program[i], str):
					hints.status = 2
					break
			hints.last_checked = index

		if hints.status != 2:
			if subr_hints.has_hint:
				hints.has_hint = True

				# Decide where to chop off from
				if subr_hints.status == 0:
					hints.last_hint = index
				else:
					hints.last_hint = index - 2  # Leave the subr call in

		elif subr_hints.status == 0:
			hints.deletions.append(index)

		hints.status = max(hints.status, subr_hints.status)


@_add_method(ttLib.getTableClass('CFF '))
def prune_post_subset(self, ttfFont, options):
	cff = self.cff
	for fontname in cff.keys():
		font = cff[fontname]
		cs = font.CharStrings

		# Drop unused FontDictionaries
		if hasattr(font, "FDSelect"):
			sel = font.FDSelect
			indices = _uniq_sort(sel.gidArray)
			sel.gidArray = [indices.index (ss) for ss in sel.gidArray]
			arr = font.FDArray
			arr.items = [arr[i] for i in indices]
			del arr.file, arr.offsets

	# Desubroutinize if asked for
	if options.desubroutinize:
		cff.desubroutinize()

	# Drop hints if not needed
	if not options.hinting:
		self.remove_hints()
	elif not options.desubroutinize:
		self.remove_unused_subroutines()
	return True


def _delete_empty_subrs(private_dict):
	if hasattr(private_dict, 'Subrs') and not private_dict.Subrs:
		if 'Subrs' in private_dict.rawDict:
			del private_dict.rawDict['Subrs']
		del private_dict.Subrs


@deprecateFunction("use 'CFFFontSet.desubroutinize()' instead", category=DeprecationWarning)
@_add_method(ttLib.getTableClass('CFF '))
def desubroutinize(self):
	self.cff.desubroutinize()


@_add_method(ttLib.getTableClass('CFF '))
def remove_hints(self):
	cff = self.cff
	for fontname in cff.keys():
		font = cff[fontname]
		cs = font.CharStrings
		# This can be tricky, but doesn't have to. What we do is:
		#
		# - Run all used glyph charstrings and recurse into subroutines,
		# - For each charstring (including subroutines), if it has any
		#   of the hint stem operators, we mark it as such.
		#   Upon returning, for each charstring we note all the
		#   subroutine calls it makes that (recursively) contain a stem,
		# - Dropping hinting then consists of the following two ops:
		#   * Drop the piece of the program in each charstring before the
		#     last call to a stem op or a stem-calling subroutine,
		#   * Drop all hintmask operations.
		# - It's trickier... A hintmask right after hints and a few numbers
		#    will act as an implicit vstemhm. As such, we track whether
		#    we have seen any non-hint operators so far and do the right
		#    thing, recursively... Good luck understanding that :(
		css = set()
		for g in font.charset:
			c, _ = cs.getItemAndSelector(g)
			c.decompile()
			subrs = getattr(c.private, "Subrs", [])
			decompiler = _DehintingT2Decompiler(css, subrs, c.globalSubrs,
								c.private.nominalWidthX,
								c.private.defaultWidthX,
								c.private)
			decompiler.execute(c)
			c.width = decompiler.width
		for charstring in css:
			charstring.drop_hints()
		del css

		# Drop font-wide hinting values
		all_privs = []
		if hasattr(font, 'FDArray'):
			all_privs.extend(fd.Private for fd in font.FDArray)
		else:
			all_privs.append(font.Private)
		for priv in all_privs:
			for k in ['BlueValues', 'OtherBlues',
				  'FamilyBlues', 'FamilyOtherBlues',
				  'BlueScale', 'BlueShift', 'BlueFuzz',
				  'StemSnapH', 'StemSnapV', 'StdHW', 'StdVW',
				  'ForceBold', 'LanguageGroup', 'ExpansionFactor']:
				if hasattr(priv, k):
					setattr(priv, k, None)
	self.remove_unused_subroutines()


@_add_method(ttLib.getTableClass('CFF '))
def remove_unused_subroutines(self):
	cff = self.cff
	for fontname in cff.keys():
		font = cff[fontname]
		cs = font.CharStrings
		# Renumber subroutines to remove unused ones

		# Mark all used subroutines
		for g in font.charset:
			c, _ = cs.getItemAndSelector(g)
			subrs = getattr(c.private, "Subrs", [])
			decompiler = _MarkingT2Decompiler(subrs, c.globalSubrs, c.private)
			decompiler.execute(c)

		all_subrs = [font.GlobalSubrs]
		if hasattr(font, 'FDArray'):
			all_subrs.extend(fd.Private.Subrs for fd in font.FDArray if hasattr(fd.Private, 'Subrs') and fd.Private.Subrs)
		elif hasattr(font.Private, 'Subrs') and font.Private.Subrs:
			all_subrs.append(font.Private.Subrs)

		subrs = set(subrs) # Remove duplicates

		# Prepare
		for subrs in all_subrs:
			if not hasattr(subrs, '_used'):
				subrs._used = set()
			subrs._used = _uniq_sort(subrs._used)
			subrs._old_bias = psCharStrings.calcSubrBias(subrs)
			subrs._new_bias = psCharStrings.calcSubrBias(subrs._used)

		# Renumber glyph charstrings
		for g in font.charset:
			c, _ = cs.getItemAndSelector(g)
			subrs = getattr(c.private, "Subrs", [])
			c.subset_subroutines (subrs, font.GlobalSubrs)

		# Renumber subroutines themselves
		for subrs in all_subrs:
			if subrs == font.GlobalSubrs:
				if not hasattr(font, 'FDArray') and hasattr(font.Private, 'Subrs'):
					local_subrs = font.Private.Subrs
				else:
					local_subrs = []
			else:
				local_subrs = subrs

			subrs.items = [subrs.items[i] for i in subrs._used]
			if hasattr(subrs, 'file'):
				del subrs.file
			if hasattr(subrs, 'offsets'):
				del subrs.offsets

			for subr in subrs.items:
				subr.subset_subroutines (local_subrs, font.GlobalSubrs)

		# Delete local SubrsIndex if empty
		if hasattr(font, 'FDArray'):
			for fd in font.FDArray:
				_delete_empty_subrs(fd.Private)
		else:
			_delete_empty_subrs(font.Private)

		# Cleanup
		for subrs in all_subrs:
			del subrs._used, subrs._old_bias, subrs._new_bias
