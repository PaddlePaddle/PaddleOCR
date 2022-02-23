"""
Merge OpenType Layout tables (GDEF / GPOS / GSUB).
"""
import os
import copy
from operator import ior
import logging
from fontTools.misc import classifyTools
from fontTools.misc.roundTools import otRound
from fontTools.ttLib.tables import otTables as ot
from fontTools.ttLib.tables import otBase as otBase
from fontTools.ttLib.tables.DefaultTable import DefaultTable
from fontTools.varLib import builder, models, varStore
from fontTools.varLib.models import nonNone, allNone, allEqual, allEqualTo
from fontTools.varLib.varStore import VarStoreInstancer
from functools import reduce
from fontTools.otlLib.builder import buildSinglePos
from fontTools.otlLib.optimize.gpos import (
    compact_pair_pos,
    GPOS_COMPACT_MODE_DEFAULT,
    GPOS_COMPACT_MODE_ENV_KEY,
)

log = logging.getLogger("fontTools.varLib.merger")

from .errors import (
    ShouldBeConstant,
    FoundANone,
    MismatchedTypes,
    LengthsDiffer,
    KeysDiffer,
    InconsistentGlyphOrder,
    InconsistentExtensions,
    UnsupportedFormat,
    UnsupportedFormat,
    VarLibMergeError,
)

class Merger(object):

	def __init__(self, font=None):
		self.font = font

	@classmethod
	def merger(celf, clazzes, attrs=(None,)):
		assert celf != Merger, 'Subclass Merger instead.'
		if 'mergers' not in celf.__dict__:
			celf.mergers = {}
		if type(clazzes) == type:
			clazzes = (clazzes,)
		if type(attrs) == str:
			attrs = (attrs,)
		def wrapper(method):
			assert method.__name__ == 'merge'
			done = []
			for clazz in clazzes:
				if clazz in done: continue # Support multiple names of a clazz
				done.append(clazz)
				mergers = celf.mergers.setdefault(clazz, {})
				for attr in attrs:
					assert attr not in mergers, \
						"Oops, class '%s' has merge function for '%s' defined already." % (clazz.__name__, attr)
					mergers[attr] = method
			return None
		return wrapper

	@classmethod
	def mergersFor(celf, thing, _default={}):
		typ = type(thing)

		for celf in celf.mro():

			mergers = getattr(celf, 'mergers', None)
			if mergers is None:
				break;

			m = celf.mergers.get(typ, None)
			if m is not None:
				return m

		return _default

	def mergeObjects(self, out, lst, exclude=()):
		if hasattr(out, "ensureDecompiled"):
			out.ensureDecompiled()
		for item in lst:
			if hasattr(item, "ensureDecompiled"):
				item.ensureDecompiled()
		keys = sorted(vars(out).keys())
		if not all(keys == sorted(vars(v).keys()) for v in lst):
			raise KeysDiffer(self, expected=keys,
				got=[sorted(vars(v).keys()) for v in lst]
			)
		mergers = self.mergersFor(out)
		defaultMerger = mergers.get('*', self.__class__.mergeThings)
		try:
			for key in keys:
				if key in exclude: continue
				value = getattr(out, key)
				values = [getattr(table, key) for table in lst]
				mergerFunc = mergers.get(key, defaultMerger)
				mergerFunc(self, value, values)
		except VarLibMergeError as e:
			e.stack.append('.'+key)
			raise

	def mergeLists(self, out, lst):
		if not allEqualTo(out, lst, len):
			raise LengthsDiffer(self, expected=len(out), got=[len(x) for x in lst])
		for i,(value,values) in enumerate(zip(out, zip(*lst))):
			try:
				self.mergeThings(value, values)
			except VarLibMergeError as e:
				e.stack.append('[%d]' % i)
				raise

	def mergeThings(self, out, lst):
		if not allEqualTo(out, lst, type):
			raise MismatchedTypes(self,
					expected=type(out).__name__,
					got=[type(x).__name__ for x in lst]
			)
		mergerFunc = self.mergersFor(out).get(None, None)
		if mergerFunc is not None:
			mergerFunc(self, out, lst)
		elif hasattr(out, '__dict__'):
			self.mergeObjects(out, lst)
		elif isinstance(out, list):
			self.mergeLists(out, lst)
		else:
			if not allEqualTo(out, lst):
				raise ShouldBeConstant(self, expected=out, got=lst)

	def mergeTables(self, font, master_ttfs, tableTags):
		for tag in tableTags:
			if tag not in font: continue
			try:
				self.ttfs = [m for m in master_ttfs if tag in m]
				self.mergeThings(font[tag], [m[tag] if tag in m else None
							     for m in master_ttfs])
			except VarLibMergeError as e:
				e.stack.append(tag)
				raise

#
# Aligning merger
#
class AligningMerger(Merger):
	pass

@AligningMerger.merger(ot.GDEF, "GlyphClassDef")
def merge(merger, self, lst):
	if self is None:
		if not allNone(lst):
			raise NotANone(merger, expected=None, got=lst)
		return

	lst = [l.classDefs for l in lst]
	self.classDefs = {}
	# We only care about the .classDefs
	self = self.classDefs

	allKeys = set()
	allKeys.update(*[l.keys() for l in lst])
	for k in allKeys:
		allValues = nonNone(l.get(k) for l in lst)
		if not allEqual(allValues):
			raise ShouldBeConstant(merger, expected=allValues[0], got=lst, stack=["." + k])
		if not allValues:
			self[k] = None
		else:
			self[k] = allValues[0]

def _SinglePosUpgradeToFormat2(self):
	if self.Format == 2: return self

	ret = ot.SinglePos()
	ret.Format = 2
	ret.Coverage = self.Coverage
	ret.ValueFormat = self.ValueFormat
	ret.Value = [self.Value for _ in ret.Coverage.glyphs]
	ret.ValueCount = len(ret.Value)

	return ret

def _merge_GlyphOrders(font, lst, values_lst=None, default=None):
	"""Takes font and list of glyph lists (must be sorted by glyph id), and returns
	two things:
	- Combined glyph list,
	- If values_lst is None, return input glyph lists, but padded with None when a glyph
	  was missing in a list.  Otherwise, return values_lst list-of-list, padded with None
	  to match combined glyph lists.
	"""
	if values_lst is None:
		dict_sets = [set(l) for l in lst]
	else:
		dict_sets = [{g:v for g,v in zip(l,vs)} for l,vs in zip(lst,values_lst)]
	combined = set()
	combined.update(*dict_sets)

	sortKey = font.getReverseGlyphMap().__getitem__
	order = sorted(combined, key=sortKey)
	# Make sure all input glyphsets were in proper order
	if not all(sorted(vs, key=sortKey) == vs for vs in lst):
		raise InconsistentGlyphOrder()
	del combined

	paddedValues = None
	if values_lst is None:
		padded = [[glyph if glyph in dict_set else default
			   for glyph in order]
			  for dict_set in dict_sets]
	else:
		assert len(lst) == len(values_lst)
		padded = [[dict_set[glyph] if glyph in dict_set else default
			   for glyph in order]
			  for dict_set in dict_sets]
	return order, padded

def _Lookup_SinglePos_get_effective_value(merger, subtables, glyph):
	for self in subtables:
		if self is None or \
		   type(self) != ot.SinglePos or \
		   self.Coverage is None or \
		   glyph not in self.Coverage.glyphs:
			continue
		if self.Format == 1:
			return self.Value
		elif self.Format == 2:
			return self.Value[self.Coverage.glyphs.index(glyph)]
		else:
			raise UnsupportedFormat(merger, subtable="single positioning lookup")
	return None

def _Lookup_PairPos_get_effective_value_pair(merger, subtables, firstGlyph, secondGlyph):
	for self in subtables:
		if self is None or \
		   type(self) != ot.PairPos or \
		   self.Coverage is None or \
		   firstGlyph not in self.Coverage.glyphs:
			continue
		if self.Format == 1:
			ps = self.PairSet[self.Coverage.glyphs.index(firstGlyph)]
			pvr = ps.PairValueRecord
			for rec in pvr: # TODO Speed up
				if rec.SecondGlyph == secondGlyph:
					return rec
			continue
		elif self.Format == 2:
			klass1 = self.ClassDef1.classDefs.get(firstGlyph, 0)
			klass2 = self.ClassDef2.classDefs.get(secondGlyph, 0)
			return self.Class1Record[klass1].Class2Record[klass2]
		else:
			raise UnsupportedFormat(merger, subtable="pair positioning lookup")
	return None

@AligningMerger.merger(ot.SinglePos)
def merge(merger, self, lst):
	self.ValueFormat = valueFormat = reduce(int.__or__, [l.ValueFormat for l in lst], 0)
	if not (len(lst) == 1 or (valueFormat & ~0xF == 0)):
		raise UnsupportedFormat(merger, subtable="single positioning lookup")

	# If all have same coverage table and all are format 1,
	coverageGlyphs = self.Coverage.glyphs
	if all(v.Format == 1 for v in lst) and all(coverageGlyphs == v.Coverage.glyphs for v in lst):
		self.Value = otBase.ValueRecord(valueFormat)
		merger.mergeThings(self.Value, [v.Value for v in lst])
		self.ValueFormat = self.Value.getFormat()
		return

	# Upgrade everything to Format=2
	self.Format = 2
	lst = [_SinglePosUpgradeToFormat2(v) for v in lst]

	# Align them
	glyphs, padded = _merge_GlyphOrders(merger.font,
					    [v.Coverage.glyphs for v in lst],
					    [v.Value for v in lst])

	self.Coverage.glyphs = glyphs
	self.Value = [otBase.ValueRecord(valueFormat) for _ in glyphs]
	self.ValueCount = len(self.Value)

	for i,values in enumerate(padded):
		for j,glyph in enumerate(glyphs):
			if values[j] is not None: continue
			# Fill in value from other subtables
			# Note!!! This *might* result in behavior change if ValueFormat2-zeroedness
			# is different between used subtable and current subtable!
			# TODO(behdad) Check and warn if that happens?
			v = _Lookup_SinglePos_get_effective_value(merger, merger.lookup_subtables[i], glyph)
			if v is None:
				v = otBase.ValueRecord(valueFormat)
			values[j] = v

	merger.mergeLists(self.Value, padded)

	# Merge everything else; though, there shouldn't be anything else. :)
	merger.mergeObjects(self, lst,
			    exclude=('Format', 'Coverage', 'Value', 'ValueCount'))
	self.ValueFormat = reduce(int.__or__, [v.getEffectiveFormat() for v in self.Value], 0)

@AligningMerger.merger(ot.PairSet)
def merge(merger, self, lst):
	# Align them
	glyphs, padded = _merge_GlyphOrders(merger.font,
				[[v.SecondGlyph for v in vs.PairValueRecord] for vs in lst],
				[vs.PairValueRecord for vs in lst])

	self.PairValueRecord = pvrs = []
	for glyph in glyphs:
		pvr = ot.PairValueRecord()
		pvr.SecondGlyph = glyph
		pvr.Value1 = otBase.ValueRecord(merger.valueFormat1) if merger.valueFormat1 else None
		pvr.Value2 = otBase.ValueRecord(merger.valueFormat2) if merger.valueFormat2 else None
		pvrs.append(pvr)
	self.PairValueCount = len(self.PairValueRecord)

	for i,values in enumerate(padded):
		for j,glyph in enumerate(glyphs):
			# Fill in value from other subtables
			v = ot.PairValueRecord()
			v.SecondGlyph = glyph
			if values[j] is not None:
				vpair = values[j]
			else:
				vpair = _Lookup_PairPos_get_effective_value_pair(
					merger, merger.lookup_subtables[i], self._firstGlyph, glyph
				)
			if vpair is None:
				v1, v2 = None, None
			else:
				v1 = getattr(vpair, "Value1", None)
				v2 = getattr(vpair, "Value2", None)
			v.Value1 = otBase.ValueRecord(merger.valueFormat1, src=v1) if merger.valueFormat1 else None
			v.Value2 = otBase.ValueRecord(merger.valueFormat2, src=v2) if merger.valueFormat2 else None
			values[j] = v
	del self._firstGlyph

	merger.mergeLists(self.PairValueRecord, padded)

def _PairPosFormat1_merge(self, lst, merger):
	assert allEqual([l.ValueFormat2 == 0 for l in lst if l.PairSet]), "Report bug against fonttools."

	# Merge everything else; makes sure Format is the same.
	merger.mergeObjects(self, lst,
			    exclude=('Coverage',
				     'PairSet', 'PairSetCount',
				     'ValueFormat1', 'ValueFormat2'))

	empty = ot.PairSet()
	empty.PairValueRecord = []
	empty.PairValueCount = 0

	# Align them
	glyphs, padded = _merge_GlyphOrders(merger.font,
					    [v.Coverage.glyphs for v in lst],
					    [v.PairSet for v in lst],
					    default=empty)

	self.Coverage.glyphs = glyphs
	self.PairSet = [ot.PairSet() for _ in glyphs]
	self.PairSetCount = len(self.PairSet)
	for glyph, ps in zip(glyphs, self.PairSet):
		ps._firstGlyph = glyph

	merger.mergeLists(self.PairSet, padded)

def _ClassDef_invert(self, allGlyphs=None):

	if isinstance(self, dict):
		classDefs = self
	else:
		classDefs = self.classDefs if self and self.classDefs else {}
	m = max(classDefs.values()) if classDefs else 0

	ret = []
	for _ in range(m + 1):
		ret.append(set())

	for k,v in classDefs.items():
		ret[v].add(k)

	# Class-0 is special.  It's "everything else".
	if allGlyphs is None:
		ret[0] = None
	else:
		# Limit all classes to glyphs in allGlyphs.
		# Collect anything without a non-zero class into class=zero.
		ret[0] = class0 = set(allGlyphs)
		for s in ret[1:]:
			s.intersection_update(class0)
			class0.difference_update(s)

	return ret

def _ClassDef_merge_classify(lst, allGlyphses=None):
	self = ot.ClassDef()
	self.classDefs = classDefs = {}
	allGlyphsesWasNone = allGlyphses is None
	if allGlyphsesWasNone:
		allGlyphses = [None] * len(lst)

	classifier = classifyTools.Classifier()
	for classDef,allGlyphs in zip(lst, allGlyphses):
		sets = _ClassDef_invert(classDef, allGlyphs)
		if allGlyphs is None:
			sets = sets[1:]
		classifier.update(sets)
	classes = classifier.getClasses()

	if allGlyphsesWasNone:
		classes.insert(0, set())

	for i,classSet in enumerate(classes):
		if i == 0:
			continue
		for g in classSet:
			classDefs[g] = i

	return self, classes

def _PairPosFormat2_align_matrices(self, lst, font, transparent=False):

	matrices = [l.Class1Record for l in lst]

	# Align first classes
	self.ClassDef1, classes = _ClassDef_merge_classify([l.ClassDef1 for l in lst], [l.Coverage.glyphs for l in lst])
	self.Class1Count = len(classes)
	new_matrices = []
	for l,matrix in zip(lst, matrices):
		nullRow = None
		coverage = set(l.Coverage.glyphs)
		classDef1 = l.ClassDef1.classDefs
		class1Records = []
		for classSet in classes:
			exemplarGlyph = next(iter(classSet))
			if exemplarGlyph not in coverage:
				# Follow-up to e6125b353e1f54a0280ded5434b8e40d042de69f,
				# Fixes https://github.com/googlei18n/fontmake/issues/470
				# Again, revert 8d441779e5afc664960d848f62c7acdbfc71d7b9
				# when merger becomes selfless.
				nullRow = None
				if nullRow is None:
					nullRow = ot.Class1Record()
					class2records = nullRow.Class2Record = []
					# TODO: When merger becomes selfless, revert e6125b353e1f54a0280ded5434b8e40d042de69f
					for _ in range(l.Class2Count):
						if transparent:
							rec2 = None
						else:
							rec2 = ot.Class2Record()
							rec2.Value1 = otBase.ValueRecord(self.ValueFormat1) if self.ValueFormat1 else None
							rec2.Value2 = otBase.ValueRecord(self.ValueFormat2) if self.ValueFormat2 else None
						class2records.append(rec2)
				rec1 = nullRow
			else:
				klass = classDef1.get(exemplarGlyph, 0)
				rec1 = matrix[klass] # TODO handle out-of-range?
			class1Records.append(rec1)
		new_matrices.append(class1Records)
	matrices = new_matrices
	del new_matrices

	# Align second classes
	self.ClassDef2, classes = _ClassDef_merge_classify([l.ClassDef2 for l in lst])
	self.Class2Count = len(classes)
	new_matrices = []
	for l,matrix in zip(lst, matrices):
		classDef2 = l.ClassDef2.classDefs
		class1Records = []
		for rec1old in matrix:
			oldClass2Records = rec1old.Class2Record
			rec1new = ot.Class1Record()
			class2Records = rec1new.Class2Record = []
			for classSet in classes:
				if not classSet: # class=0
					rec2 = oldClass2Records[0]
				else:
					exemplarGlyph = next(iter(classSet))
					klass = classDef2.get(exemplarGlyph, 0)
					rec2 = oldClass2Records[klass]
				class2Records.append(copy.deepcopy(rec2))
			class1Records.append(rec1new)
		new_matrices.append(class1Records)
	matrices = new_matrices
	del new_matrices

	return matrices

def _PairPosFormat2_merge(self, lst, merger):
	assert allEqual([l.ValueFormat2 == 0 for l in lst if l.Class1Record]), "Report bug against fonttools."

	merger.mergeObjects(self, lst,
			    exclude=('Coverage',
				     'ClassDef1', 'Class1Count',
				     'ClassDef2', 'Class2Count',
				     'Class1Record',
				     'ValueFormat1', 'ValueFormat2'))

	# Align coverages
	glyphs, _ = _merge_GlyphOrders(merger.font,
				       [v.Coverage.glyphs for v in lst])
	self.Coverage.glyphs = glyphs

	# Currently, if the coverage of PairPosFormat2 subtables are different,
	# we do NOT bother walking down the subtable list when filling in new
	# rows for alignment.  As such, this is only correct if current subtable
	# is the last subtable in the lookup.  Ensure that.
	#
	# Note that our canonicalization process merges trailing PairPosFormat2's,
	# so in reality this is rare.
	for l,subtables in zip(lst,merger.lookup_subtables):
		if l.Coverage.glyphs != glyphs:
			assert l == subtables[-1]

	matrices = _PairPosFormat2_align_matrices(self, lst, merger.font)

	self.Class1Record = list(matrices[0]) # TODO move merger to be selfless
	merger.mergeLists(self.Class1Record, matrices)

@AligningMerger.merger(ot.PairPos)
def merge(merger, self, lst):
	merger.valueFormat1 = self.ValueFormat1 = reduce(int.__or__, [l.ValueFormat1 for l in lst], 0)
	merger.valueFormat2 = self.ValueFormat2 = reduce(int.__or__, [l.ValueFormat2 for l in lst], 0)

	if self.Format == 1:
		_PairPosFormat1_merge(self, lst, merger)
	elif self.Format == 2:
		_PairPosFormat2_merge(self, lst, merger)
	else:
		raise UnsupportedFormat(merger, subtable="pair positioning lookup")

	del merger.valueFormat1, merger.valueFormat2

	# Now examine the list of value records, and update to the union of format values,
	# as merge might have created new values.
	vf1 = 0
	vf2 = 0
	if self.Format == 1:
		for pairSet in self.PairSet:
			for pairValueRecord in pairSet.PairValueRecord:
				pv1 = getattr(pairValueRecord, "Value1", None)
				if pv1 is not None:
					vf1 |= pv1.getFormat()
				pv2 = getattr(pairValueRecord, "Value2", None)
				if pv2 is not None:
					vf2 |= pv2.getFormat()
	elif self.Format == 2:
		for class1Record in self.Class1Record:
			for class2Record in class1Record.Class2Record:
				pv1 = getattr(class2Record, "Value1", None)
				if pv1 is not None:
					vf1 |= pv1.getFormat()
				pv2 = getattr(class2Record, "Value2", None)
				if pv2 is not None:
					vf2 |= pv2.getFormat()
	self.ValueFormat1 = vf1
	self.ValueFormat2 = vf2

def _MarkBasePosFormat1_merge(self, lst, merger, Mark='Mark', Base='Base'):
	self.ClassCount = max(l.ClassCount for l in lst)

	MarkCoverageGlyphs, MarkRecords = \
		_merge_GlyphOrders(merger.font,
				   [getattr(l, Mark+'Coverage').glyphs for l in lst],
				   [getattr(l, Mark+'Array').MarkRecord for l in lst])
	getattr(self, Mark+'Coverage').glyphs = MarkCoverageGlyphs

	BaseCoverageGlyphs, BaseRecords = \
		_merge_GlyphOrders(merger.font,
				   [getattr(l, Base+'Coverage').glyphs for l in lst],
				   [getattr(getattr(l, Base+'Array'), Base+'Record') for l in lst])
	getattr(self, Base+'Coverage').glyphs = BaseCoverageGlyphs

	# MarkArray
	records = []
	for g,glyphRecords in zip(MarkCoverageGlyphs, zip(*MarkRecords)):
		allClasses = [r.Class for r in glyphRecords if r is not None]

		# TODO Right now we require that all marks have same class in
		# all masters that cover them.  This is not required.
		#
		# We can relax that by just requiring that all marks that have
		# the same class in a master, have the same class in every other
		# master.  Indeed, if, say, a sparse master only covers one mark,
		# that mark probably will get class 0, which would possibly be
		# different from its class in other masters.
		#
		# We can even go further and reclassify marks to support any
		# input.  But, since, it's unlikely that two marks being both,
		# say, "top" in one master, and one being "top" and other being
		# "top-right" in another master, we shouldn't do that, as any
		# failures in that case will probably signify mistakes in the
		# input masters.

		if not allEqual(allClasses):
			raise ShouldBeConstant(merger, expected=allClasses[0], got=allClasses)
		else:
			rec = ot.MarkRecord()
			rec.Class = allClasses[0]
			allAnchors = [None if r is None else r.MarkAnchor for r in glyphRecords]
			if allNone(allAnchors):
				anchor = None
			else:
				anchor = ot.Anchor()
				anchor.Format = 1
				merger.mergeThings(anchor, allAnchors)
			rec.MarkAnchor = anchor
		records.append(rec)
	array = ot.MarkArray()
	array.MarkRecord = records
	array.MarkCount = len(records)
	setattr(self, Mark+"Array", array)

	# BaseArray
	records = []
	for g,glyphRecords in zip(BaseCoverageGlyphs, zip(*BaseRecords)):
		if allNone(glyphRecords):
			rec = None
		else:
			rec = getattr(ot, Base+'Record')()
			anchors = []
			setattr(rec, Base+'Anchor', anchors)
			glyphAnchors = [[] if r is None else getattr(r, Base+'Anchor')
					for r in glyphRecords]
			for l in glyphAnchors:
				l.extend([None] * (self.ClassCount - len(l)))
			for allAnchors in zip(*glyphAnchors):
				if allNone(allAnchors):
					anchor = None
				else:
					anchor = ot.Anchor()
					anchor.Format = 1
					merger.mergeThings(anchor, allAnchors)
				anchors.append(anchor)
		records.append(rec)
	array = getattr(ot, Base+'Array')()
	setattr(array, Base+'Record', records)
	setattr(array, Base+'Count', len(records))
	setattr(self, Base+'Array', array)

@AligningMerger.merger(ot.MarkBasePos)
def merge(merger, self, lst):
	if not allEqualTo(self.Format, (l.Format for l in lst)):
		raise InconsistentFormats(
			merger,
			subtable="mark-to-base positioning lookup",
			expected=self.Format,
			got=[l.Format for l in lst]
		)
	if self.Format == 1:
		_MarkBasePosFormat1_merge(self, lst, merger)
	else:
		raise UnsupportedFormat(merger, subtable="mark-to-base positioning lookup")

@AligningMerger.merger(ot.MarkMarkPos)
def merge(merger, self, lst):
	if not allEqualTo(self.Format, (l.Format for l in lst)):
		raise InconsistentFormats(
			merger,
			subtable="mark-to-mark positioning lookup",
			expected=self.Format,
			got=[l.Format for l in lst]
		)
	if self.Format == 1:
		_MarkBasePosFormat1_merge(self, lst, merger, 'Mark1', 'Mark2')
	else:
		raise UnsupportedFormat(merger, subtable="mark-to-mark positioning lookup")

def _PairSet_flatten(lst, font):
	self = ot.PairSet()
	self.Coverage = ot.Coverage()

	# Align them
	glyphs, padded = _merge_GlyphOrders(font,
				[[v.SecondGlyph for v in vs.PairValueRecord] for vs in lst],
				[vs.PairValueRecord for vs in lst])

	self.Coverage.glyphs = glyphs
	self.PairValueRecord = pvrs = []
	for values in zip(*padded):
		for v in values:
			if v is not None:
				pvrs.append(v)
				break
		else:
			assert False
	self.PairValueCount = len(self.PairValueRecord)

	return self

def _Lookup_PairPosFormat1_subtables_flatten(lst, font):
	assert allEqual([l.ValueFormat2 == 0 for l in lst if l.PairSet]), "Report bug against fonttools."

	self = ot.PairPos()
	self.Format = 1
	self.Coverage = ot.Coverage()
	self.ValueFormat1 = reduce(int.__or__, [l.ValueFormat1 for l in lst], 0)
	self.ValueFormat2 = reduce(int.__or__, [l.ValueFormat2 for l in lst], 0)

	# Align them
	glyphs, padded = _merge_GlyphOrders(font,
					    [v.Coverage.glyphs for v in lst],
					    [v.PairSet for v in lst])

	self.Coverage.glyphs = glyphs
	self.PairSet = [_PairSet_flatten([v for v in values if v is not None], font)
		        for values in zip(*padded)]
	self.PairSetCount = len(self.PairSet)
	return self

def _Lookup_PairPosFormat2_subtables_flatten(lst, font):
	assert allEqual([l.ValueFormat2 == 0 for l in lst if l.Class1Record]), "Report bug against fonttools."

	self = ot.PairPos()
	self.Format = 2
	self.Coverage = ot.Coverage()
	self.ValueFormat1 = reduce(int.__or__, [l.ValueFormat1 for l in lst], 0)
	self.ValueFormat2 = reduce(int.__or__, [l.ValueFormat2 for l in lst], 0)

	# Align them
	glyphs, _ = _merge_GlyphOrders(font,
				       [v.Coverage.glyphs for v in lst])
	self.Coverage.glyphs = glyphs

	matrices = _PairPosFormat2_align_matrices(self, lst, font, transparent=True)

	matrix = self.Class1Record = []
	for rows in zip(*matrices):
		row = ot.Class1Record()
		matrix.append(row)
		row.Class2Record = []
		row = row.Class2Record
		for cols in zip(*list(r.Class2Record for r in rows)):
			col = next(iter(c for c in cols if c is not None))
			row.append(col)

	return self

def _Lookup_PairPos_subtables_canonicalize(lst, font):
	"""Merge multiple Format1 subtables at the beginning of lst,
	and merge multiple consecutive Format2 subtables that have the same
	Class2 (ie. were split because of offset overflows).  Returns new list."""
	lst = list(lst)

	l = len(lst)
	i = 0
	while i < l and lst[i].Format == 1:
		i += 1
	lst[:i] = [_Lookup_PairPosFormat1_subtables_flatten(lst[:i], font)]

	l = len(lst)
	i = l
	while i > 0 and lst[i - 1].Format == 2:
		i -= 1
	lst[i:] = [_Lookup_PairPosFormat2_subtables_flatten(lst[i:], font)]

	return lst

def _Lookup_SinglePos_subtables_flatten(lst, font, min_inclusive_rec_format):
	glyphs, _ = _merge_GlyphOrders(font,
		[v.Coverage.glyphs for v in lst], None)
	num_glyphs = len(glyphs)
	new = ot.SinglePos()
	new.Format = 2
	new.ValueFormat = min_inclusive_rec_format
	new.Coverage = ot.Coverage()
	new.Coverage.glyphs = glyphs
	new.ValueCount = num_glyphs
	new.Value = [None] * num_glyphs
	for singlePos in lst:
		if singlePos.Format == 1:
			val_rec = singlePos.Value
			for gname in singlePos.Coverage.glyphs:
				i = glyphs.index(gname)
				new.Value[i] = copy.deepcopy(val_rec)
		elif singlePos.Format == 2:
			for j, gname in enumerate(singlePos.Coverage.glyphs):
				val_rec = singlePos.Value[j]
				i = glyphs.index(gname)
				new.Value[i] = copy.deepcopy(val_rec)
	return [new]

@AligningMerger.merger(ot.Lookup)
def merge(merger, self, lst):
	subtables = merger.lookup_subtables = [l.SubTable for l in lst]

	# Remove Extension subtables
	for l,sts in list(zip(lst,subtables))+[(self,self.SubTable)]:
		if not sts:
			continue
		if sts[0].__class__.__name__.startswith('Extension'):
			if not allEqual([st.__class__ for st in sts]):
				raise InconsistentExtensions(
					merger,
					expected="Extension",
					got=[st.__class__.__name__ for st in sts]
				)
			if not allEqual([st.ExtensionLookupType for st in sts]):
				raise InconsistentExtensions(merger)
			l.LookupType = sts[0].ExtensionLookupType
			new_sts = [st.ExtSubTable for st in sts]
			del sts[:]
			sts.extend(new_sts)

	isPairPos = self.SubTable and isinstance(self.SubTable[0], ot.PairPos)

	if isPairPos:
		# AFDKO and feaLib sometimes generate two Format1 subtables instead of one.
		# Merge those before continuing.
		# https://github.com/fonttools/fonttools/issues/719
		self.SubTable = _Lookup_PairPos_subtables_canonicalize(self.SubTable, merger.font)
		subtables = merger.lookup_subtables = [_Lookup_PairPos_subtables_canonicalize(st, merger.font) for st in subtables]
	else:
		isSinglePos = self.SubTable and isinstance(self.SubTable[0], ot.SinglePos)
		if isSinglePos:
			numSubtables = [len(st) for st in subtables]
			if not all([nums == numSubtables[0] for nums in numSubtables]):
				# Flatten list of SinglePos subtables to single Format 2 subtable,
				# with all value records set to the rec format type.
				# We use buildSinglePos() to optimize the lookup after merging.
				valueFormatList = [t.ValueFormat for st in subtables for t in st]
				# Find the minimum value record that can accomodate all the singlePos subtables.
				mirf = reduce(ior, valueFormatList)
				self.SubTable = _Lookup_SinglePos_subtables_flatten(self.SubTable, merger.font, mirf)
				subtables = merger.lookup_subtables = [
					_Lookup_SinglePos_subtables_flatten(st, merger.font, mirf) for st in subtables]
				flattened = True
			else:
				flattened = False

	merger.mergeLists(self.SubTable, subtables)
	self.SubTableCount = len(self.SubTable)

	if isPairPos:
		# If format-1 subtable created during canonicalization is empty, remove it.
		assert len(self.SubTable) >= 1 and self.SubTable[0].Format == 1
		if not self.SubTable[0].Coverage.glyphs:
			self.SubTable.pop(0)
			self.SubTableCount -= 1

		# If format-2 subtable created during canonicalization is empty, remove it.
		assert len(self.SubTable) >= 1 and self.SubTable[-1].Format == 2
		if not self.SubTable[-1].Coverage.glyphs:
			self.SubTable.pop(-1)
			self.SubTableCount -= 1

		# Compact the merged subtables
		# This is a good moment to do it because the compaction should create
		# smaller subtables, which may prevent overflows from happening.
		mode = os.environ.get(GPOS_COMPACT_MODE_ENV_KEY, GPOS_COMPACT_MODE_DEFAULT)
		if mode and mode != "0":
			log.info("Compacting GPOS...")
			self.SubTable = compact_pair_pos(merger.font, mode, self.SubTable)
			self.SubTableCount = len(self.SubTable)

	elif isSinglePos and flattened:
		singlePosTable = self.SubTable[0]
		glyphs = singlePosTable.Coverage.glyphs
		# We know that singlePosTable is Format 2, as this is set
		# in _Lookup_SinglePos_subtables_flatten.
		singlePosMapping = {
			gname: valRecord
			for gname, valRecord in zip(glyphs, singlePosTable.Value)
		}
		self.SubTable = buildSinglePos(singlePosMapping, merger.font.getReverseGlyphMap())
	merger.mergeObjects(self, lst, exclude=['SubTable', 'SubTableCount'])

	del merger.lookup_subtables

#
# InstancerMerger
#

class InstancerMerger(AligningMerger):
	"""A merger that takes multiple master fonts, and instantiates
	an instance."""

	def __init__(self, font, model, location):
		Merger.__init__(self, font)
		self.model = model
		self.location = location
		self.scalars = model.getScalars(location)

@InstancerMerger.merger(ot.CaretValue)
def merge(merger, self, lst):
	assert self.Format == 1
	Coords = [a.Coordinate for a in lst]
	model = merger.model
	scalars = merger.scalars
	self.Coordinate = otRound(model.interpolateFromMastersAndScalars(Coords, scalars))

@InstancerMerger.merger(ot.Anchor)
def merge(merger, self, lst):
	assert self.Format == 1
	XCoords = [a.XCoordinate for a in lst]
	YCoords = [a.YCoordinate for a in lst]
	model = merger.model
	scalars = merger.scalars
	self.XCoordinate = otRound(model.interpolateFromMastersAndScalars(XCoords, scalars))
	self.YCoordinate = otRound(model.interpolateFromMastersAndScalars(YCoords, scalars))

@InstancerMerger.merger(otBase.ValueRecord)
def merge(merger, self, lst):
	model = merger.model
	scalars = merger.scalars
	# TODO Handle differing valueformats
	for name, tableName in [('XAdvance','XAdvDevice'),
				('YAdvance','YAdvDevice'),
				('XPlacement','XPlaDevice'),
				('YPlacement','YPlaDevice')]:

		assert not hasattr(self, tableName)

		if hasattr(self, name):
			values = [getattr(a, name, 0) for a in lst]
			value = otRound(model.interpolateFromMastersAndScalars(values, scalars))
			setattr(self, name, value)


#
# MutatorMerger
#

class MutatorMerger(AligningMerger):
	"""A merger that takes a variable font, and instantiates
	an instance.  While there's no "merging" to be done per se,
	the operation can benefit from many operations that the
	aligning merger does."""

	def __init__(self, font, instancer, deleteVariations=True):
		Merger.__init__(self, font)
		self.instancer = instancer
		self.deleteVariations = deleteVariations

@MutatorMerger.merger(ot.CaretValue)
def merge(merger, self, lst):

	# Hack till we become selfless.
	self.__dict__ = lst[0].__dict__.copy()

	if self.Format != 3:
		return

	instancer = merger.instancer
	dev = self.DeviceTable
	if merger.deleteVariations:
		del self.DeviceTable
	if dev:
		assert dev.DeltaFormat == 0x8000
		varidx = (dev.StartSize << 16) + dev.EndSize
		delta = otRound(instancer[varidx])
		self.Coordinate += delta

	if merger.deleteVariations:
		self.Format = 1

@MutatorMerger.merger(ot.Anchor)
def merge(merger, self, lst):

	# Hack till we become selfless.
	self.__dict__ = lst[0].__dict__.copy()

	if self.Format != 3:
		return

	instancer = merger.instancer
	for v in "XY":
		tableName = v+'DeviceTable'
		if not hasattr(self, tableName):
			continue
		dev = getattr(self, tableName)
		if merger.deleteVariations:
			delattr(self, tableName)
		if dev is None:
			continue

		assert dev.DeltaFormat == 0x8000
		varidx = (dev.StartSize << 16) + dev.EndSize
		delta = otRound(instancer[varidx])

		attr = v+'Coordinate'
		setattr(self, attr, getattr(self, attr) + delta)

	if merger.deleteVariations:
		self.Format = 1

@MutatorMerger.merger(otBase.ValueRecord)
def merge(merger, self, lst):

	# Hack till we become selfless.
	self.__dict__ = lst[0].__dict__.copy()

	instancer = merger.instancer
	for name, tableName in [('XAdvance','XAdvDevice'),
				('YAdvance','YAdvDevice'),
				('XPlacement','XPlaDevice'),
				('YPlacement','YPlaDevice')]:

		if not hasattr(self, tableName):
			continue
		dev = getattr(self, tableName)
		if merger.deleteVariations:
			delattr(self, tableName)
		if dev is None:
			continue

		assert dev.DeltaFormat == 0x8000
		varidx = (dev.StartSize << 16) + dev.EndSize
		delta = otRound(instancer[varidx])

		setattr(self, name, getattr(self, name, 0) + delta)


#
# VariationMerger
#

class VariationMerger(AligningMerger):
	"""A merger that takes multiple master fonts, and builds a
	variable font."""

	def __init__(self, model, axisTags, font):
		Merger.__init__(self, font)
		self.store_builder = varStore.OnlineVarStoreBuilder(axisTags)
		self.setModel(model)

	def setModel(self, model):
		self.model = model
		self.store_builder.setModel(model)

	def mergeThings(self, out, lst):
		masterModel = None
		if None in lst:
			if allNone(lst):
				if out is not None:
					raise FoundANone(self, got=lst)
				return
			masterModel = self.model
			model, lst = masterModel.getSubModel(lst)
			self.setModel(model)

		super(VariationMerger, self).mergeThings(out, lst)

		if masterModel:
			self.setModel(masterModel)


def buildVarDevTable(store_builder, master_values):
	if allEqual(master_values):
		return master_values[0], None
	base, varIdx = store_builder.storeMasters(master_values)
	return base, builder.buildVarDevTable(varIdx)

@VariationMerger.merger(ot.BaseCoord)
def merge(merger, self, lst):
	if self.Format != 1:
		raise UnsupportedFormat(merger, subtable="a baseline coordinate")
	self.Coordinate, DeviceTable = buildVarDevTable(merger.store_builder, [a.Coordinate for a in lst])
	if DeviceTable:
		self.Format = 3
		self.DeviceTable = DeviceTable

@VariationMerger.merger(ot.CaretValue)
def merge(merger, self, lst):
	if self.Format != 1:
		raise UnsupportedFormat(merger, subtable="a caret")
	self.Coordinate, DeviceTable = buildVarDevTable(merger.store_builder, [a.Coordinate for a in lst])
	if DeviceTable:
		self.Format = 3
		self.DeviceTable = DeviceTable

@VariationMerger.merger(ot.Anchor)
def merge(merger, self, lst):
	if self.Format != 1:
		raise UnsupportedFormat(merger, subtable="an anchor")
	self.XCoordinate, XDeviceTable = buildVarDevTable(merger.store_builder, [a.XCoordinate for a in lst])
	self.YCoordinate, YDeviceTable = buildVarDevTable(merger.store_builder, [a.YCoordinate for a in lst])
	if XDeviceTable or YDeviceTable:
		self.Format = 3
		self.XDeviceTable = XDeviceTable
		self.YDeviceTable = YDeviceTable

@VariationMerger.merger(otBase.ValueRecord)
def merge(merger, self, lst):
	for name, tableName in [('XAdvance','XAdvDevice'),
				('YAdvance','YAdvDevice'),
				('XPlacement','XPlaDevice'),
				('YPlacement','YPlaDevice')]:

		if hasattr(self, name):
			value, deviceTable = buildVarDevTable(merger.store_builder,
							      [getattr(a, name, 0) for a in lst])
			setattr(self, name, value)
			if deviceTable:
				setattr(self, tableName, deviceTable)
